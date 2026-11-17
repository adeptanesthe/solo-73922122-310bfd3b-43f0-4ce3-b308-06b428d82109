"""局部风险核查：对独立抽检的四料配方，统计随机森林每个叶节点的误放行情况。

流程：
 1. 校验并解析抽检文本（30～300 行，每行 4 个整数克数 + 1 个实测合格标签，
    克数合计 1000）；
 2. 按原分类器的真实输入口径（克数 ÷ 1000 的 float32 比例）计算合格概率，
    以 0.7 门槛筛出模型放行记录；
 3. 让每条放行记录走完 5 棵树，得到其命中的叶节点；同一记录命中的每个叶
    分别计数（每条记录在每棵树上恰好命中一个叶）；
 4. 对全部叶节点做 95% 同时置信核查：每叶用 Bonferroni 校正（α = 0.05 / 叶总数）
    的 Clopper–Pearson 单侧下界估计误放行率；
 5. 仅当某叶的校正后下界严格大于用户填写的风险上限时，才标记“建议暂停自动放行”；
    其余（包括没有任何放行记录的叶）一律标记“证据不足”，不下结论。

同时置信说明：Bonferroni 对任意相关结构均成立，因此即使同一记录被多个叶
重复计数，所有叶的区间同时覆盖真实误放行率的概率仍不低于 95%。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta

from .model import (
    FEATURE_NAMES,
    PROBA_EPS,
    PROBA_THRESHOLD,
    TOTAL_GRAMS,
    ForestSpec,
    Leaf,
    Node,
    cutoff_left,
    extract_forest,
    leaf_index,
)

MIN_RECORDS = 30
MAX_RECORDS = 300
MAX_TEXT_CHARS = 50_000
CONFIDENCE = 0.95
DEFAULT_RISK_LIMIT = 0.05
AUDIT_SAMPLE_SEED = 20241118

DECISION_PAUSE = "pause"
DECISION_INSUFFICIENT = "insufficient"

_TRUE_LABELS = {"合格", "1", "是", "true", "yes", "y", "pass", "通过", "ok"}
_FALSE_LABELS = {
    "不合格", "0", "否", "false", "no", "n", "fail", "不通过", "未通过", "ng",
}
_SEPARATOR = re.compile(r"[\s,，;；\t]+")
_INT_TOKEN = re.compile(r"[+-]?\d+")


@dataclass(frozen=True)
class Record:
    grams: tuple[int, int, int, int]
    qualified: bool
    line_no: int


def parse_records(text) -> tuple[list[Record], list[str]]:
    """解析抽检文本；返回 (记录, 错误列表)。有错误时调用方不应继续核查。"""
    if not isinstance(text, str):
        return [], ["抽检数据必须是文本"]
    if len(text) > MAX_TEXT_CHARS:
        return [], [f"抽检文本过长（超过 {MAX_TEXT_CHARS} 字符）"]

    lines = [(i + 1, ln.strip()) for i, ln in enumerate(text.splitlines()) if ln.strip()]
    errors: list[str] = []
    if len(lines) < MIN_RECORDS or len(lines) > MAX_RECORDS:
        errors.append(
            f"抽检记录须为 {MIN_RECORDS}～{MAX_RECORDS} 行（当前 {len(lines)} 行）"
        )

    records: list[Record] = []
    for line_no, line in lines:
        tokens = [t for t in _SEPARATOR.split(line) if t]
        if len(tokens) != 5:
            errors.append(
                f"第 {line_no} 行：应为 4 个克数 + 1 个标签（共 5 项），实际 {len(tokens)} 项"
            )
            continue
        grams = []
        bad = False
        for t in tokens[:4]:
            if not _INT_TOKEN.fullmatch(t):
                errors.append(f"第 {line_no} 行：克数“{t}”不是整数")
                bad = True
                break
            grams.append(int(t))
        if bad:
            continue
        label = tokens[4].strip()
        low = label.lower()
        if low in _TRUE_LABELS:
            qualified = True
        elif low in _FALSE_LABELS:
            qualified = False
        else:
            errors.append(
                f"第 {line_no} 行：标签“{tokens[4]}”无法识别（可用 合格/不合格、1/0、是/否）"
            )
            continue
        if any(g < 0 for g in grams):
            errors.append(f"第 {line_no} 行：克数不能为负")
            continue
        if sum(grams) != TOTAL_GRAMS:
            errors.append(
                f"第 {line_no} 行：克数合计为 {sum(grams)}，须等于 {TOTAL_GRAMS}"
            )
            continue
        records.append(Record(tuple(grams), qualified, line_no))

    # 错误过多时截断，避免页面被刷屏（同时保留“还有 N 条”的提示）
    if len(errors) > 20:
        extra = len(errors) - 20
        errors = errors[:20] + [f"……另有 {extra} 条错误未显示"]
    return records, errors


def parse_risk_limit(value) -> tuple[float | None, str | None]:
    """风险上限必须是 [0,1] 内的有限数值；字符串仅接受普通十进制写法。"""
    if isinstance(value, bool):
        return None, "风险上限必须是 0～1 之间的数值"
    if isinstance(value, (int, float)):
        x = float(value)
    elif isinstance(value, str):
        s = value.strip()
        try:
            x = float(s)
        except (TypeError, ValueError):
            return None, "风险上限必须是 0～1 之间的数值"
    else:
        return None, "风险上限必须是 0～1 之间的数值"
    if not math.isfinite(x):
        return None, "风险上限必须是 0～1 之间的有限数值"
    if not (0.0 <= x <= 1.0):
        return None, f"风险上限须在 0～1 之间（当前 {x:g}）"
    return x, None


def _leaf_path(tree, leaf_id: int) -> list[tuple[int, str, int]]:
    """回溯根到叶子的路径，返回 [(特征下标, "<="或">=", 克数截断)]。"""
    parent: dict[int, tuple[int, bool]] = {}
    for pid, node in enumerate(tree.nodes):
        if isinstance(node, Node):
            parent[node.left] = (pid, True)
            parent[node.right] = (pid, False)
    path = []
    cur = leaf_id
    while cur in parent:
        pid, went_left = parent[cur]
        node = tree.nodes[pid]
        # 与分类器逐点一致：左分支 g <= k，右分支 g >= k+1
        k = _cutoff(node.threshold)
        if went_left:
            path.append((node.feature, "<=", k))
        else:
            path.append((node.feature, ">=", k + 1))
        cur = pid
    path.reverse()
    return path


def _cutoff(threshold: float) -> int:
    return cutoff_left(threshold)


def _merged_bounds(path) -> list[dict]:
    """把路径条件合并为各原料的闭区间（与路径条件等价）。"""
    lo = [0] * len(FEATURE_NAMES)
    hi = [TOTAL_GRAMS] * len(FEATURE_NAMES)
    for feature, op, grams in path:
        if op == "<=":
            hi[feature] = min(hi[feature], grams)
        else:
            lo[feature] = max(lo[feature], grams)
    bounds = []
    for i, name in enumerate(FEATURE_NAMES):
        if lo[i] == 0 and hi[i] == TOTAL_GRAMS:
            continue
        bounds.append({"feature": i, "name": name, "min": lo[i], "max": hi[i]})
    return bounds


def _path_text(path) -> str:
    return "；".join(
        f"{FEATURE_NAMES[f]} {'≤' if op == '<=' else '≥'} {g} 克"
        for f, op, g in path
    )


def leaves_catalog(forest: ForestSpec) -> dict:
    """全部叶节点的静态说明（树号、叶内合格概率、路径限制、合并后的原料区间）。"""
    leaves = []
    for ti, nid, node in _all_leaves(forest):
        path = _leaf_path(forest.trees[ti], nid)
        leaves.append(
            {
                "tree": ti + 1,
                "leaf_id": nid,
                "leaf_probability": round(float(node.prob1), 6),
                "path": _path_text(path),
                "bounds": _merged_bounds(path),
            }
        )
    return {
        "leaves": leaves,
        "leaves_total": len(leaves),
        "feature_names": FEATURE_NAMES,
        "total_grams": TOTAL_GRAMS,
    }


def clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    """单侧 Clopper–Pearson 下界：Beta^{-1}(alpha; k, n-k+1)；k=0 时为 0。"""
    if n <= 0 or k <= 0:
        return 0.0
    if k > n:
        raise ValueError("k 不能大于 n")
    return float(beta.ppf(alpha, k, n - k + 1))


def _all_leaves(forest: ForestSpec):
    for ti, tree in enumerate(forest.trees):
        for nid, node in enumerate(tree.nodes):
            if isinstance(node, Leaf):
                yield ti, nid, node


def build_audit_sample(clf) -> dict:
    """生成确定性的抽检样例（独立种子 + 含噪质检标签），仅供页面演示使用。

    标签直接由与训练数据同形式的合成质检规则生成（种子独立于训练），
    因此不依赖模型自身的判断，演示数据不会“自证”模型。
    """
    rng = np.random.default_rng(AUDIT_SAMPLE_SEED)
    n = 120
    frac = rng.dirichlet([2.5, 2.0, 1.6, 1.2], size=n)
    score = (
        0.95 * frac[:, 0]
        + 0.55 * frac[:, 1]
        - 0.65 * frac[:, 2]
        + 0.20 * frac[:, 3]
        + 0.35 * frac[:, 0] * frac[:, 1]
        - 0.25 * frac[:, 2] * frac[:, 3]
        + rng.normal(0.0, 0.03, size=n)
    )
    labels = score >= float(np.quantile(score, 0.5))
    grams = np.rint(frac * TOTAL_GRAMS).astype(int)
    grams[:, 3] = TOTAL_GRAMS - grams[:, 0] - grams[:, 1] - grams[:, 2]
    grams = np.maximum(grams, 0)
    # 极端舍入可能破坏总重，逐行修正到 1000 克（改动最小的一列）
    for row in grams:
        diff = TOTAL_GRAMS - int(row.sum())
        if diff:
            row[int(np.argmax(row))] += diff
    text = "\n".join(
        f"{g[0]} {g[1]} {g[2]} {g[3]} {'合格' if q else '不合格'}"
        for g, q in zip(grams.tolist(), labels.tolist())
    )
    records, errors = parse_records(text)
    assert not errors and len(records) == n, errors
    result = analyze(extract_forest(clf), clf, records, DEFAULT_RISK_LIMIT)
    return {
        "text": text,
        "risk_limit": DEFAULT_RISK_LIMIT,
        "records": result["summary"]["records"],
        "released": result["summary"]["released"],
        "released_unqualified": result["summary"]["released_unqualified"],
    }


def analyze(forest: ForestSpec, clf, records: list[Record], risk_limit: float) -> dict:
    """对全部叶节点做局部误放行统计与 95% 同时置信核查。"""
    grams_matrix = np.asarray([r.grams for r in records], dtype=np.float64)
    # 与原分类器输入完全一致：先做 float64 除法，再转 float32
    x = (grams_matrix / TOTAL_GRAMS).astype(np.float32)
    probs = clf.predict_proba(x)[:, 1]
    released = probs >= PROBA_THRESHOLD - PROBA_EPS

    n_records = len(records)
    n_released = int(released.sum())
    n_released_fail = int(
        sum(1 for r, rel in zip(records, released) if rel and not r.qualified)
    )

    # 每条放行记录在每棵树上命中一个叶；同一记录命中的多个叶分别计数
    counts: dict[tuple[int, int], list[int]] = {}
    for r, rel in zip(records, released):
        if not rel:
            continue
        for ti, tree in enumerate(forest.trees):
            nid = leaf_index(tree, list(r.grams))
            c = counts.setdefault((ti, nid), [0, 0])
            c[0] += 1
            if not r.qualified:
                c[1] += 1

    leaves_meta = list(_all_leaves(forest))
    m = len(leaves_meta)
    alpha_corr = (1.0 - CONFIDENCE) / m if m else 0.0

    leaves_out = []
    flagged = 0
    for ti, nid, node in leaves_meta:
        n, k = counts.get((ti, nid), [0, 0])
        lower = clopper_pearson_lower(k, n, alpha_corr)
        decision = DECISION_PAUSE if (n > 0 and lower > risk_limit) else DECISION_INSUFFICIENT
        if decision == DECISION_PAUSE:
            flagged += 1
        path = _leaf_path(forest.trees[ti], nid)
        leaves_out.append(
            {
                "tree": ti + 1,
                "leaf_id": nid,
                "leaf_probability": round(float(node.prob1), 6),
                "path": _path_text(path),
                "bounds": _merged_bounds(path),
                "released": n,
                "released_unqualified": k,
                "failure_rate": (k / n) if n else None,
                "lower_bound": lower,
                "decision": decision,
            }
        )

    # 一致性断言：每条放行记录在每棵树各贡献 1 次叶计数
    assert n_released * len(forest.trees) == sum(v[0] for v in counts.values())

    return {
        "status": "ok",
        "summary": {
            "records": n_records,
            "released": n_released,
            "released_unqualified": n_released_fail,
            "leaves_total": m,
            "leaves_flagged": flagged,
            "risk_limit": risk_limit,
            "confidence": CONFIDENCE,
            "alpha_corrected": alpha_corr,
            "threshold": PROBA_THRESHOLD,
        },
        "leaves": leaves_out,
    }


def run_audit(forest: ForestSpec, clf, text, risk_limit_value) -> dict:
    """API 入口：先校验风险上限与抽检文本，任一不合法都不返回核查结论。"""
    risk_limit, risk_err = parse_risk_limit(risk_limit_value)
    records, errors = parse_records(text)
    if risk_err:
        errors = [risk_err] + errors
    if errors:
        return {"status": "invalid", "reason": "；".join(errors), "errors": errors}
    assert risk_limit is not None
    return analyze(forest, clf, records, risk_limit)

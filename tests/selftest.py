"""关键功能自测（开发环境运行，不进镜像）：

  python -m tests.selftest

覆盖：
 1. 模型规格：5 棵树、深度 ≤ 3、同种子重训结果一致；
 2. 阈值语义：cutoff_left 对全部树阈值 × 全部克数逐一与 float32 比较核对；
    随机配方上纯 Python 走树概率与 sklearn predict_proba 完全一致；
 3. 可行样例：求解 ok、概率复算达标、改动量一致，并用 z3 反证 + sklearn 暴力
    枚举双重核对最优性结论；
 4. 无解样例：z3 判 unsat，并对整个可行域用 sklearn 暴力枚举复核确无合格配方；
 5. 字典序：改 2 种（量大）与改 3 种（量小）并存时，必须返回改 2 种；
 6. 边界：原配方已合格（0 改动）、全锁定合格/不合格、各类非法输入；
 7. 求解失败路径：异常注入返回 solver_failure 而非无解；
 8. API：/、/api/meta、/api/samples、/api/solve 各状态分支；页面含迟到结果丢弃逻辑；
 9. 局部风险核查：贴入文本解析与逐行校验；放行口径与原分类器 float32 一致；
    逐叶计数与独立走树一致（多条命中分别计数）；95% 同时置信（Bonferroni +
    Clopper–Pearson）与“下界严格大于风险上限”判定边界；无放行记录不下结论；
    /audit 与 /api/audit* 各分支。
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import z3

from app.model import (
    MAX_DEPTH,
    N_TREES,
    TOTAL_GRAMS,
    cutoff_left,
    extract_forest,
    forest_proba_rational,
    is_qualified,
    predict_proba_qualified,
    train_model,
)
from app.solver import (
    INFEASIBLE,
    INVALID,
    OK,
    SOLVER_FAILURE,
    SolveInput,
    _build_optimize,
    solve,
)

PASS, FAIL = "PASS", "FAIL"
failures = []


def check(name, cond, detail=""):
    tag = PASS if cond else FAIL
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def _stub_forest_one_leaf(prob1):
    from app.model import ForestSpec, Leaf, TreeSpec

    tree = TreeSpec(nodes=(Leaf(prob1=prob1),))
    return ForestSpec(trees=(tree,), n_trees=1, max_depth=0, seed=0)


class _StubClf:
    """记录输入 dtype，并按固定概率返回（用于核查口径与单叶计数测试）。"""

    def __init__(self, p=1.0):
        self.p = p
        self.dtypes = []

    def predict_proba(self, x):
        self.dtypes.append(x.dtype)
        p = np.full(len(x), float(self.p))
        return np.column_stack([1.0 - p, p])


def run_audit_tests(clf, forest, client):
    from fractions import Fraction

    from app.audit import (
        DEFAULT_RISK_LIMIT,
        analyze,
        build_audit_sample,
        clopper_pearson_lower,
        parse_records,
        parse_risk_limit,
        run_audit,
    )
    from app.model import leaf_index

    # ---- 9.1 解析与校验 ----
    good_line = "400 300 200 100 合格"
    ok_text = "\n".join([good_line] * 30)
    recs, errs = parse_records(ok_text)
    check("解析 30 行合法记录", not errs and len(recs) == 30 and recs[0].grams == (400, 300, 200, 100))
    recs, errs = parse_records("\n".join([good_line] * 300))
    check("解析 300 行合法记录", not errs and len(recs) == 300)
    _, errs29 = parse_records("\n".join([good_line] * 29))
    _, errs301 = parse_records("\n".join([good_line] * 301))
    check("29 行与 301 行均被拒绝", bool(errs29) and bool(errs301))

    recs, errs = parse_records(
        "400,300,200,100,不合格\n"      # 逗号分隔
        "400 300 200 100 1\n"          # 数字标签
        "400 300 200 100 否\n"         # 中文否定标签
        + "\n".join([good_line] * 27)
    )
    check("支持逗号分隔与多种标签写法",
          not errs and len(recs) == 30 and recs[0].qualified is False
          and recs[1].qualified is True and recs[2].qualified is False)
    _, errs = parse_records(good_line + "\n" + "400 300 200 99 合格\n" + "\n".join([good_line] * 28))
    check("合计不等于 1000 克被拒绝", any("合计" in e for e in errs))
    _, errs = parse_records(good_line + "\n" + "400 300 200 100 也许\n" + "\n".join([good_line] * 28))
    check("无法识别的标签被拒绝", any("标签" in e for e in errs))
    _, errs = parse_records(good_line + "\n" + "400.5 300 200 99.5 合格\n" + "\n".join([good_line] * 28))
    check("非整数克数被拒绝", any("整数" in e for e in errs))
    _, errs = parse_records(good_line + "\n" + "-400 700 200 500 合格\n" + "\n".join([good_line] * 28))
    check("负克数被拒绝", any("为负" in e for e in errs))
    _, errs = parse_records(good_line + "\n" + "400 300 200 100\n" + "\n".join([good_line] * 28))
    check("缺少标签的行被拒绝", any("5 项" in e for e in errs))

    check("风险上限：合法值", parse_risk_limit("0.05") == (0.05, None)
          and parse_risk_limit(0.0) == (0.0, None) and parse_risk_limit(1) == (1.0, None))
    bad_limits = ["abc", -0.01, 1.01, float("nan"), float("inf"), True, None, []]
    check("风险上限：非法值均被拒绝", all(parse_risk_limit(v)[0] is None for v in bad_limits))

    # ---- 9.2 放行口径与原分类器一致（float32 输入）----
    sample = build_audit_sample(clf)
    sample2 = build_audit_sample(clf)
    check("示例抽检数据确定性", sample == sample2 and sample["records"] == 120)
    recs, errs = parse_records(sample["text"])
    res = run_audit(forest, clf, sample["text"], sample["risk_limit"])
    check("示例数据核查成功", res["status"] == "ok", str(res)[:200])
    grams = np.array([r.grams for r in recs], dtype=np.float64)
    p_expected = clf.predict_proba((grams / TOTAL_GRAMS).astype(np.float32))[:, 1]
    expect_released = int((p_expected >= 0.7 - 1e-9).sum())
    check("放行记录数与原分类器逐条一致",
          res["summary"]["released"] == expect_released == sample["released"],
          f"{res['summary']['released']} vs {expect_released}")
    stub = _StubClf()
    analyze(_stub_forest_one_leaf(Fraction(1, 2)), stub, recs[:30], 0.05)
    check("核查输入为 float32 比例", stub.dtypes == [np.dtype(np.float32)])

    # ---- 9.3 逐叶计数：每放行记录每树恰命中一叶 ----
    leaves = res["leaves"]
    check("全部 40 个叶均被核查", len(leaves) == 40
          and len({(L["tree"], L["leaf_id"]) for L in leaves}) == 40)
    expected = {}
    for r, p in zip(recs, p_expected):
        if p < 0.7 - 1e-9:
            continue
        for ti, tree in enumerate(forest.trees):
            key = (ti + 1, leaf_index(tree, list(r.grams)))
            e = expected.setdefault(key, [0, 0])
            e[0] += 1
            if not r.qualified:
                e[1] += 1
    got = {(L["tree"], L["leaf_id"]): [L["released"], L["released_unqualified"]]
           for L in leaves if L["released"] > 0}
    check("逐叶放行数/不合格数与独立走树一致", expected == got,
          f"{len(expected)} vs {len(got)}")
    total_hits = sum(v[0] for v in got.values())
    check("每条放行记录在 5 棵树上各计 1 次（多叶分别计数）",
          total_hits == res["summary"]["released"] * 5,
          f"hits={total_hits}, released={res['summary']['released']}")
    multi = sum(1 for v in got.values() if v[0] > 0)
    check("存在同一记录命中多个叶的情形", multi > 1 and multi <= 40, f"{multi} 个叶有放行记录")

    # 叶路径区间与记录一致：命中该叶的每条放行记录都必须落在其列出的原料区间内
    bounds_ok = True
    leaf_by_key = {(L["tree"], L["leaf_id"]): L for L in leaves}
    for r, p in zip(recs, p_expected):
        if p < 0.7 - 1e-9:
            continue
        for ti, tree in enumerate(forest.trees):
            L = leaf_by_key[(ti + 1, leaf_index(tree, list(r.grams)))]
            for b in L["bounds"]:
                if not (b["min"] <= r.grams[b["feature"]] <= b["max"]):
                    bounds_ok = False
    check("叶路径原料限制覆盖其全部命中记录", bounds_ok)

    # ---- 9.4 同时置信与判定门槛 ----
    n, m = 40, 40
    alpha_corr = 0.05 / m
    manual = float(__import__("scipy.stats", fromlist=["beta"]).beta.ppf(alpha_corr, 7, n - 7 + 1))
    lv = {L["tree"]: L for L in leaves}
    check("校正显著性水平 = 0.05 / 叶数",
          abs(res["summary"]["alpha_corrected"] - alpha_corr) < 1e-12)
    check("Clopper–Pearson 下界与 scipy 一致",
          abs(clopper_pearson_lower(7, n, alpha_corr) - manual) < 1e-12)

    # 构造全部放行记录均不合格的抽检数据 → 误放行率应为 1，应触发暂停
    lines = []
    for r in recs:
        p = float(clf.predict_proba((np.array([r.grams], dtype=np.float32) / TOTAL_GRAMS))[0, 1])
        label = "不合格" if p >= 0.7 - 1e-9 else "合格"
        lines.append(f"{r.grams[0]} {r.grams[1]} {r.grams[2]} {r.grams[3]} {label}")
    bad_text = "\n".join(lines)
    res_bad = run_audit(forest, clf, bad_text, 0.05)
    paused = [L for L in res_bad["leaves"] if L["decision"] == "pause"]
    check("放行记录全不合格且样本充足 → 触发暂停", res_bad["status"] == "ok" and len(paused) > 0,
          f"{len(paused)} 个叶")
    check("暂停叶的校正下界严格大于风险上限",
          all(L["lower_bound"] > 0.05 and L["released"] > 0 for L in paused))
    check("未触发叶均为证据不足（无其他结论）",
          all(L["decision"] in ("pause", "insufficient") for L in res_bad["leaves"]))

    # 同一批数据、把风险上限提到 1 → 没有任何叶可判暂停
    res_hi = run_audit(forest, clf, bad_text, 1.0)
    check("风险上限为 1 时不判暂停", res_hi["summary"]["leaves_flagged"] == 0)

    # 精确边界：下界 == 上限 → 证据不足；上限略低于下界 → 暂停
    stub = _StubClf(1.0)   # 桩分类器把每条记录都判为放行
    r30 = [type(recs[0])(r.grams, False, r.line_no) for r in recs[:30]]  # 标签全为不合格
    res_stub = analyze(_stub_forest_one_leaf(Fraction(1, 2)), stub, r30, 0.05)
    L = res_stub["leaves"][0]
    n_rel, k_rel = L["released"], L["released_unqualified"]
    check("单叶桩：放行/不合格计数正确", n_rel == 30 and k_rel == 30)
    exact = L["lower_bound"]
    check("单叶下界与 Clopper–Pearson(30,30,0.05) 一致",
          abs(exact - clopper_pearson_lower(30, 30, 0.05)) < 1e-12)
    at = analyze(_stub_forest_one_leaf(Fraction(1, 2)), stub, r30, exact)["leaves"][0]
    below = analyze(_stub_forest_one_leaf(Fraction(1, 2)), stub, r30, exact - 1e-9)["leaves"][0]
    check("下界等于上限 → 证据不足（须严格大于）", at["decision"] == "insufficient")
    check("上限略低于下界 → 建议暂停", below["decision"] == "pause")

    # 没有放行记录 → 全部证据不足（即使风险上限为 0）
    no_release = "\n".join(
        [f"{r.grams[0]} {r.grams[1]} {r.grams[2]} {r.grams[3]} 合格"
         for r in recs if clf.predict_proba((np.array([r.grams], dtype=np.float32) / TOTAL_GRAMS))[0, 1] < 0.7]
    )
    n_nr = len(no_release.splitlines())
    res_nr = run_audit(forest, clf, no_release, 0.0) if n_nr >= 30 else None
    check("无放行记录时不下结论", res_nr is not None and res_nr["summary"]["released"] == 0
          and all(L["decision"] == "insufficient" and L["lower_bound"] == 0.0
                  for L in res_nr["leaves"]),
          f"n={n_nr}")

    # ---- 9.5 API ----
    r = client.get("/audit")
    check("GET /audit 返回核查网页", r.status_code == 200 and "局部风险核查" in r.text
          and "审核" not in r.text)
    r = client.get("/api/audit/leaves")
    jl = r.json()
    check("GET /api/audit/leaves 列出 40 个叶", r.status_code == 200 and jl["leaves_total"] == 40
          and len(jl["leaves"]) == 40)
    r = client.get("/api/audit/sample")
    check("GET /api/audit/sample 返回示例", r.status_code == 200
          and r.json()["records"] == 120)
    r = client.post("/api/audit", json={"records": sample["text"], "risk_limit": "0.03"})
    j = r.json()
    check("POST /api/audit 正常分支", r.status_code == 200 and j["status"] == "ok"
          and j["summary"]["records"] == 120 and j["summary"]["risk_limit"] == 0.03)
    r = client.post("/api/audit", json={"records": sample["text"], "risk_limit": "abc"})
    check("POST /api/audit 非法风险上限", r.json()["status"] == "invalid")
    r = client.post("/api/audit", json={"records": "1 2 3 4 合格", "risk_limit": 0.05})
    check("POST /api/audit 非法记录", r.json()["status"] == "invalid"
          and any("30" in e for e in r.json()["errors"]))
    r = client.post("/api/audit", json={"records": 42, "risk_limit": 0.05})
    check("POST /api/audit 非文本记录", r.json()["status"] == "invalid")
    html = client.get("/audit").text
    check("核查页含迟到结果丢弃与即时撤回逻辑",
          "my !== epoch" in html and "invalidate(" in html and "abort" in html)
    check("试配页保留", client.get("/").status_code == 200
          and "开始求解" in client.get("/").text)


def main():
    print("== 1. 模型规格 ==")
    clf = train_model()
    forest = extract_forest(clf)
    check("树数量为 5", len(clf.estimators_) == N_TREES == 5)
    depths = [e.tree_.max_depth for e in clf.estimators_]
    check("树深均 ≤ 3", all(d <= MAX_DEPTH for d in depths), f"depths={depths}")
    clf2 = train_model()
    same = np.allclose(
        clf.predict_proba(np.zeros((1, 4), dtype=np.float32) + 0.25),
        clf2.predict_proba(np.zeros((1, 4), dtype=np.float32) + 0.25),
    )
    check("同种子重训结果一致", same)

    print("== 2. 阈值语义 ==")
    # cutoff_left：对森林中全部阈值 × 全部整数克数，逐一与 float32 比较核对
    thr_ok = True
    for tree in forest.trees:
        for node in tree.nodes:
            if node.__class__.__name__ == "Leaf":
                continue
            k = cutoff_left(node.threshold)
            for g in range(TOTAL_GRAMS + 1):
                expect = bool(np.float32(g / TOTAL_GRAMS) <= node.threshold)
                if (g <= k) != expect:
                    thr_ok = False
                    break
            if not thr_ok:
                break
    check("cutoff_left 与 float32 比较逐点一致", thr_ok)

    rng = np.random.default_rng(99)
    proba_ok = True
    for _ in range(500):
        g = rng.multinomial(TOTAL_GRAMS, [0.3, 0.3, 0.2, 0.2]).tolist()
        p_sk = predict_proba_qualified(clf, g)
        p_py = float(forest_proba_rational(forest, g))
        if abs(p_sk - p_py) > 1e-12:
            proba_ok = False
            break
    check("500 个随机配方概率与 sklearn 完全一致", proba_ok)

    print("== 3. 可行样例 ==")
    from app.samples import build_samples

    samples = build_samples(forest, clf)
    fs = samples["feasible"]
    inp = SolveInput(fs["original"], fs["lower"], fs["upper"], fs["locked"])
    res = solve(forest, clf, inp)
    check("可行样例求解为 ok", res.status == OK, res.reason)
    check("建议概率 ≥ 0.7（原分类器复算）", res.probability is not None and is_qualified(res.probability),
          f"p={res.probability:.6f}")
    sug = res.suggestion
    check("建议满足总重/上下限/锁定",
          sum(sug) == TOTAL_GRAMS
          and all(fs["lower"][i] <= sug[i] <= fs["upper"][i] for i in range(4))
          and all(sug[i] == fs["original"][i] for i in range(4) if fs["locked"][i]))
    k, m = res.changed_count, res.total_change
    real_k = sum(1 for i in range(4) if sug[i] != fs["original"][i])
    real_m = sum(abs(sug[i] - fs["original"][i]) for i in range(4))
    check("报告的改动种数/克数与建议一致", (k, m) == (real_k, real_m), f"k={k}, m={m}")

    # 3a. z3 反证最优性
    opt, g, num_changed, total_change = _build_optimize(forest, inp)
    opt.add(num_changed <= k - 1)
    check("z3 反证：更少种数无解", opt.check() == z3.unsat)
    opt2, g2, num_changed2, total_change2 = _build_optimize(forest, inp)
    opt2.add(num_changed2 == k, total_change2 <= m - 1)
    check("z3 反证：同种数更小改动量无解", opt2.check() == z3.unsat)

    # 3b. sklearn 暴力枚举复核最优性（改动 ≤2 种且更优者不存在）
    def brute_best(inp_d, max_kinds):
        orig, lo, hi, lk = inp_d["original"], inp_d["lower"], inp_d["upper"], inp_d["locked"]
        free = [i for i in range(4) if not lk[i]]
        best = None  # (kinds, delta)
        for n_k in range(0, max_kinds + 1):
            for S in itertools.combinations(free, n_k):
                fixed_idx = [i for i in range(4) if i not in S]
                fixed_sum = sum(orig[i] for i in fixed_idx)
                if len(S) == 0:
                    cand = [tuple(orig)]
                elif len(S) == 1:
                    i = S[0]
                    gi = TOTAL_GRAMS - fixed_sum
                    cand = [] if not (lo[i] <= gi <= hi[i]) else [tuple(orig[:i]) + (gi,) + tuple(orig[i+1:])]
                else:  # len(S) == 2：和约束下第二变量被唯一确定
                    i, j = S
                    cand = []
                    for gi in range(lo[i], hi[i] + 1):
                        gj = TOTAL_GRAMS - fixed_sum - gi
                        if lo[j] <= gj <= hi[j]:
                            row = list(orig); row[i], row[j] = gi, gj
                            cand.append(tuple(row))
                if not cand:
                    continue
                arr = np.array(cand, dtype=np.float32) / TOTAL_GRAMS
                probs = clf.predict_proba(arr)[:, 1]
                for row, p in zip(cand, probs):
                    if p >= 0.7 - 1e-9:
                        d = sum(abs(row[t] - orig[t]) for t in range(4))
                        cur = (n_k, d)
                        if best is None or cur < best:
                            best = cur
        return best

    bb = brute_best(fs, 2)
    check("sklearn 暴力枚举复核最优性", bb == (k, m), f"brute={bb}, solver=({k},{m})")

    print("== 4. 无解样例 ==")
    isf = samples["infeasible"]
    res_i = solve(forest, clf, SolveInput(isf["original"], isf["lower"], isf["upper"], isf["locked"]))
    check("无解样例求解为 infeasible", res_i.status == INFEASIBLE, res_i.reason)
    # 对整个可行域用 sklearn 暴力枚举复核“确实无解”
    lo, hi = isf["lower"], isf["upper"]
    combos = []
    for g0 in range(lo[0], hi[0] + 1):
        for g1 in range(lo[1], hi[1] + 1):
            for g2 in range(lo[2], hi[2] + 1):
                g3 = TOTAL_GRAMS - g0 - g1 - g2
                if lo[3] <= g3 <= hi[3]:
                    combos.append((g0, g1, g2, g3))
    found_q = False
    for s in range(0, len(combos), 2_000_000):
        arr = np.array(combos[s:s + 2_000_000], dtype=np.float32) / TOTAL_GRAMS
        if (clf.predict_proba(arr)[:, 1] >= 0.7 - 1e-9).any():
            found_q = True
            break
    check(f"sklearn 全可行域枚举（{len(combos)} 点）确无合格配方", not found_q)

    print("== 5. 字典序优先级 ==")
    lex_orig = [244, 294, 218, 244]  # 改 2 种最少 218 克；改 3 种可 214 克
    res_lex = solve(forest, clf, SolveInput(lex_orig, [0]*4, [1000]*4, [False]*4))
    check("先最少种数再最小克数", res_lex.status == OK and res_lex.changed_count == 2
          and res_lex.total_change == 218,
          f"got kinds={res_lex.changed_count}, delta={res_lex.total_change}")

    print("== 6. 边界与非法输入 ==")
    r_ok0 = solve(forest, clf, SolveInput([700, 100, 100, 100], [0]*4, [1000]*4, [False]*4))
    check("原配方已合格 → 0 改动", r_ok0.status == OK and r_ok0.changed_count == 0
          and r_ok0.total_change == 0 and r_ok0.suggestion == [700, 100, 100, 100])
    r_lock_bad = solve(forest, clf, SolveInput([150, 130, 620, 100], [0]*4, [1000]*4, [True]*4))
    check("全锁定且不合格 → infeasible", r_lock_bad.status == INFEASIBLE)
    r_lock_ok = solve(forest, clf, SolveInput([700, 100, 100, 100], [0]*4, [1000]*4, [True]*4))
    check("全锁定且合格 → ok 0 改动", r_lock_ok.status == OK and r_lock_ok.changed_count == 0)
    bad_cases = [
        SolveInput([250, 250, 350, 149], [0]*4, [1000]*4, [False]*4),   # 总和 999
        SolveInput([250, 250, 350, 150], [0, 0, 500, 0], [1000, 1000, 400, 1000], [False]*4),  # lo>hi
        SolveInput([250, 250, 350, 150], [300, 0, 0, 0], [1000]*4, [True, False, False, False]),  # 锁定值越界
        SolveInput([250, 250, 350, 150], [600, 600, 0, 0], [1000]*4, [False]*4),  # 下限和超 1000
        SolveInput([250, 250, 350, 150], [0]*4, [200, 200, 200, 200], [False]*4),  # 上限和不足 1000
    ]
    check("非法输入均判 invalid", all(solve(forest, clf, c).status == INVALID for c in bad_cases))

    print("== 7. 求解失败路径 ==")
    import app.solver as sv
    orig_build = sv._build_optimize
    def boom(forest_, inp_):
        raise z3.Z3Exception("injected")
    sv._build_optimize = boom
    try:
        r_fail = solve(forest, clf, SolveInput([250, 250, 350, 150], [0]*4, [1000]*4, [False]*4))
    finally:
        sv._build_optimize = orig_build
    check("求解器异常 → solver_failure（非无解）", r_fail.status == SOLVER_FAILURE)

    print("== 8. API ==")
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    r = client.get("/")
    check("GET / 返回网页", r.status_code == 200 and "开始求解" in r.text)
    r = client.get("/api/meta")
    mj = r.json()
    check("GET /api/meta", r.status_code == 200 and mj["n_trees"] == 5 and mj["max_depth"] == 3
          and abs(mj["proba_threshold"] - 0.7) < 1e-12)
    r = client.get("/api/samples")
    sj = r.json()
    check("GET /api/samples 两类样例齐全", r.status_code == 200
          and sj["feasible"]["check"]["status"] == "ok"
          and sj["infeasible"]["check"]["status"] == "infeasible")
    r = client.post("/api/solve", json={"original": fs["original"], "lower": fs["lower"],
                                        "upper": fs["upper"], "locked": fs["locked"]})
    rj = r.json()
    check("POST /api/solve 可行分支", r.status_code == 200 and rj["status"] == "ok"
          and rj["probability"] >= 0.7 and rj["suggestion"] == sug)
    r = client.post("/api/solve", json={"original": isf["original"], "lower": isf["lower"],
                                        "upper": isf["upper"], "locked": isf["locked"]})
    check("POST /api/solve 无解分支", r.json()["status"] == "infeasible")
    r = client.post("/api/solve", json={"original": [1, 2, 3, 4], "lower": [0]*4,
                                        "upper": [1000]*4, "locked": [False]*4})
    check("POST /api/solve 非法输入分支", r.json()["status"] == "invalid")
    html = client.get("/").text
    check("页面含迟到结果丢弃逻辑", "my !== epoch" in html and "abort" in html)

    print("== 9. 局部风险核查 ==")
    run_audit_tests(clf, forest, client)

    print()
    if failures:
        print(f"共 {len(failures)} 项失败：{failures}")
        return 1
    print("全部自测通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

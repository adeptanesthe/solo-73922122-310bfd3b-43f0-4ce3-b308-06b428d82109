// 局部风险核查页面前端测试：DOM 打桩 + 可控 fetch，验证
//  1) 启动后加载示例数据、编辑后立即撤下旧报告
//  2) 核查期间编辑 → 旧报告立即撤下（状态与报告区同时清空）
//  3) 迟到的旧响应即使成功返回也绝不恢复旧报告
//  4) 正常核查渲染：汇总、逐叶行、暂停/证据不足标签
//  5) 输入不合法分支渲染错误列表
//  6) 行数与风险上限提示即时更新
const fs = require('fs');
const path = require('path');
const html = fs.readFileSync(path.join(__dirname, '..', 'app', 'static', 'audit.html'), 'utf8');
const pageJs = html.match(/<script>([\s\S]*)<\/script>/)[1];

// ---- DOM 打桩 ----
const store = {};
function el(id) {
  if (store[id]) return store[id];
  const e = makeEl(id);
  store[id] = e;
  return e;
}
function makeEl(id) {
  const e = {
    id, checked: false, disabled: false,
    style: { display: 'none' }, dataset: {}, colSpan: 1, className: '',
    children: [], _text: '', _value: '',
    listeners: {},
    classList: {
      _set: new Set(),
      add(c) { this._set.add(c); }, remove(c) { this._set.delete(c); },
      toggle(c, on) { if (on) this._set.add(c); else this._set.delete(c); },
      contains(c) { return this._set.has(c); },
    },
    addEventListener(ev, fn) { (this.listeners[ev] ??= []).push(fn); },
    appendChild(c) { this.children.push(c); return c; },
    append(...cs) { this.children.push(...cs); },
    textOf() { return this._text + this.children.map(c => c.textOf ? c.textOf() : String(c)).join(''); },
  };
  Object.defineProperty(e, 'textContent', {
    get() { return this.textOf(); },
    set(v) { this._text = String(v); this.children.length = 0; },
  });
  Object.defineProperty(e, 'value', {   // 真实 DOM 中 input/textarea 的 value 恒为字符串
    get() { return this._value; },
    set(v) { this._value = String(v); },
  });
  Object.defineProperty(e, 'innerHTML', {
    get() { return this.textOf(); },
    set(v) { this._text = String(v); this.children.length = 0; },
  });
  return e;
}
global.document = {
  getElementById: id => {
    if (!store[id]) store[id] = makeEl(id);   // 与真实 DOM 一致：元素在脚本运行前已存在
    return store[id];
  },
  createElement: tag => makeEl('created_' + tag),
};

const META = { feature_names: ['A', 'B', 'C', 'D'], total_grams: 1000, proba_threshold: 0.7,
  n_trees: 5, max_depth: 3, tree_depths: [3, 3, 3, 3, 3], train_seed: 1 };

function sampleText(n) {
  const rows = [];
  for (let i = 0; i < n; i++) rows.push('400 300 200 100 ' + (i % 3 === 0 ? '不合格' : '合格'));
  return rows.join('\n');
}
const SAMPLE = { text: sampleText(120), risk_limit: 0.05, records: 120, released: 49, released_unqualified: 1 };

const LEAVES = [
  { tree: 1, leaf_id: 3, leaf_probability: 0.986, path: '矿物填料 ≤ 122 克',
    bounds: [{ feature: 2, name: '矿物填料', min: 0, max: 122 }],
    released: 12, released_unqualified: 9, failure_rate: 0.75, lower_bound: 0.30, decision: 'pause' },
  { tree: 1, leaf_id: 7, leaf_probability: 0.861, path: '再生PP ≥ 263 克',
    bounds: [{ feature: 0, name: '再生PP', min: 263, max: 1000 }],
    released: 0, released_unqualified: 0, failure_rate: null, lower_bound: 0.0, decision: 'insufficient' },
  { tree: 2, leaf_id: 6, leaf_probability: 0.855, path: '矿物填料 ≤ 191 克',
    bounds: [{ feature: 2, name: '矿物填料', min: 0, max: 191 }],
    released: 5, released_unqualified: 0, failure_rate: 0.0, lower_bound: 0.0, decision: 'insufficient' },
];
function auditResult() {
  return { status: 'ok',
    summary: { records: 120, released: 49, released_unqualified: 1, leaves_total: 40,
      leaves_flagged: 1, risk_limit: 0.05, confidence: 0.95,
      alpha_corrected: 0.00125, threshold: 0.7 },
    leaves: LEAVES };
}
const INVALID_RESULT = { status: 'invalid', reason: 'bad',
  errors: ['抽检记录须为 30～300 行（当前 1 行）', '第 1 行：克数合计为 999，须等于 1000'] };

let fetchImpl = () => Promise.reject(new Error('fetch not stubbed'));
global.fetch = (...a) => fetchImpl(...a);

// boot 在 eval 时同步发起 fetch（meta + sample），须提前打桩
fetchImpl = url => Promise.resolve({ json: async () => url.includes('meta') ? META : SAMPLE });

const driver = `
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const assert = (name, cond) => { console.log((cond ? '[PASS] ' : '[FAIL] ') + name); if (!cond) process.exitCode = 1; };
  const txt = id => document.getElementById(id).textContent;
  const shown = id => document.getElementById(id).style.display === 'block';

  await sleep(20);
  assert('启动时抽检区为空、提示 0 行', document.getElementById('records').value === ''
    && document.getElementById('rowHint').innerHTML.includes('>0<'));
  await loadSample();
  assert('载入示例后填入 120 行数据', document.getElementById('records').value.split('\\n').length === 120);
  assert('载入示例后风险上限为示例值', document.getElementById('riskLimit').value == '0.05');
  assert('载入示例后提示显示 120 行', document.getElementById('rowHint').innerHTML.includes('120'));
  assert('载入示例后状态提示尚未核查', document.getElementById('status').textContent.includes('尚未核查'));

  // --- 场景 1：正常核查渲染 ---
  fetchImpl = () => Promise.resolve({ json: async () => auditResult() });
  await runAudit();
  assert('核查完成后面板显示', shown('panelOk'));
  assert('汇总含放行记录数', txt('summary').includes('49'));
  assert('汇总含建议暂停叶数', txt('summary').includes('建议暂停自动放行的叶'));
  assert('渲染了叶行（含暂停标签）', txt('leafRows').includes('建议暂停自动放行'));
  assert('渲染了证据不足标签', txt('leafRows').includes('证据不足'));
  assert('无放行记录叶显示破折号而非 0%', txt('leafRows').includes('—'));
  assert('叶行按树分组', txt('leafRows').includes('第 1 棵树') && txt('leafRows').includes('第 2 棵树'));
  assert('状态提示核查完成', txt('status').includes('核查完成'));

  // --- 场景 2：编辑抽检数据 → 旧报告立即撤下 ---
  document.getElementById('records').value = '1 2 3 4 合格';
  document.getElementById('records').listeners.input[0]();
  assert('编辑数据后旧报告立即撤下', !shown('panelOk'));
  assert('编辑数据后状态提示撤下', txt('status').includes('撤下'));
  assert('编辑数据后行数提示更新', document.getElementById('rowHint').innerHTML.includes('>1<'));
  assert('行数不合法时 textarea 标红', document.getElementById('records').classList.contains('invalid'));

  // --- 场景 3：核查期间编辑 → 立即撤下；迟到结果不恢复 ---
  document.getElementById('records').value = sampleText(120);
  document.getElementById('records').listeners.input[0]();
  let resolveStale;
  fetchImpl = () => new Promise(res => { resolveStale = () => res({ json: async () => auditResult() }); });
  const p = runAudit();                 // 发起核查（不等待）
  await sleep(5);
  document.getElementById('riskLimit').value = '0.01';
  document.getElementById('riskLimit').listeners.input[0]();   // 用户编辑风险上限
  assert('核查期间编辑后旧报告保持隐藏', !shown('panelOk'));
  resolveStale();                      // 迟到响应“绕过”abort 成功返回（竞态）
  await p;
  await sleep(5);
  assert('迟到结果未恢复旧报告', !shown('panelOk'));
  assert('迟到结果未改写状态', txt('status').includes('撤下'));

  // --- 场景 4：编辑风险上限（不重新核查）后报告仍在撤下状态 ---
  fetchImpl = () => Promise.resolve({ json: async () => auditResult() });
  await runAudit();
  assert('重新核查后报告恢复', shown('panelOk'));
  document.getElementById('riskLimit').value = '0.2';
  document.getElementById('riskLimit').listeners.input[0]();
  assert('仅改风险上限也立即撤下报告', !shown('panelOk'));

  // --- 场景 5：非法输入分支 ---
  fetchImpl = () => Promise.resolve({ json: async () => INVALID_RESULT });
  document.getElementById('records').value = sampleText(30);
  await runAudit();
  assert('非法输入面板显示', shown('panelInvalid'));
  assert('非法输入面板隐藏报告', !shown('panelOk'));
  assert('错误列表逐条渲染', txt('invalidList').includes('第 1 行') && txt('invalidList').includes('30～300'));

  // --- 场景 6：请求异常（非 abort）提示 ---
  fetchImpl = () => Promise.reject(new Error('网络中断'));
  await runAudit();
  assert('请求失败提示可见', txt('status').includes('网络中断'));
})();
`;

eval(pageJs + driver);

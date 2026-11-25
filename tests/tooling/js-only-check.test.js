import assert from 'node:assert/strict';

import { findJsOnlyViolations } from '../../tools/check-js-only.js';

// Fixture files are in memory, so the repository itself never holds a violation.
const rules = (files) => findJsOnlyViolations(files).map((v) => `${v.path}:${v.rule}`);

// Clean JavaScript, WGSL, docs and data pass, including citations of upstream
// Python sources and recorded commands inside non-script files.
assert.deepEqual(rules([
  { path: 'src/a.js', text: "import x from './x.js';\nconst url = 'https://github.com/org/repo/blob/abc/modeling.py';\n" },
  { path: 'src/gpu/kernels/a.wgsl', text: '@compute @workgroup_size(64) fn main() {}\n' },
  { path: 'docs/a.md', text: 'Run `python3 tools/x.py` outside Doppler.\n' },
  { path: 'reports/a/receipt.json', text: '{"command":"python3 x.py"}\n' },
  { path: 'tools/agent-eval.js', text: "const KNOWN = ['node',\n  'python3',\n  'curl'];\n" },
  { path: 'tools/lexer.js', text: "if (word === 'go') return 1;\nconst gopher = 'go fish';\n" },
  { path: 'package.json', text: JSON.stringify({ scripts: { test: 'node tools/run.js' }, description: 'python3 -m x' }) },
  { path: 'benchmarks/x/results/run/source/package.json', text: JSON.stringify({ scripts: { ref: 'python3 tools/x.py' } }) },
]), []);

// Python and Go sources and project files fail by path.
assert.deepEqual(rules([
  { path: 'tools/x.py', text: null },
  { path: 'tools/types.pyi', text: null },
  { path: 'notebooks/a.ipynb', text: null },
  { path: 'cmd/main.go', text: null },
  { path: 'go.mod', text: 'module x\n' },
  { path: 'tools/requirements.txt', text: 'torch\n' },
  { path: 'pyproject.toml', text: '' },
]), [
  'tools/x.py:forbidden-language-file',
  'tools/types.pyi:forbidden-language-file',
  'notebooks/a.ipynb:forbidden-language-file',
  'cmd/main.go:forbidden-language-file',
  'go.mod:forbidden-language-file',
  'tools/requirements.txt:forbidden-language-file',
  'pyproject.toml:forbidden-language-file',
]);

// Extensionless interpreter scripts fail by shebang.
assert.deepEqual(rules([{ path: 'tools/run-reference', text: '#!/usr/bin/env python3\nprint(1)\n' }]),
  ['tools/run-reference:interpreter-shebang']);

// Scripts that run Python or Go fail, in every way the repository used to do it.
const violations = findJsOnlyViolations([
  { path: 'tools/a.js', text: "execFileSync('python3', [path.join(dir, 'gen.py')]);\n" },
  { path: 'tools/b.js', text: "const child = spawn('python', ['-c', source]);\n" },
  { path: 'tools/c.js', text: "await command(context, 'python3', [script, '--out', out]);\n" },
  { path: 'tools/d.js', text: "const python = process.env.PYTHON || 'python3';\nawait runJson(python, args);\n" },
  { path: 'tools/e.js', text: "spawn(config.metrics.sacrebleuPython, ['-c', program]);\n" },
  { path: 'tools/f.js', text: "const env = { PYTHONPATH: overlay };\n" },
  { path: 'tools/g.js', text: "const bin = join(root, '.venv', 'bin', 'python');\nconst p = '/opt/env/bin/python';\n" },
  { path: 'tools/h.js', text: "spawnSync('go', ['run', './cmd']);\n" },
  { path: 'tools/probe.sh', text: '#!/usr/bin/env bash\nCASES=$(python3 -c "import json")\n' },
  { path: 'tools/heredoc.sh', text: "python3 - <<'PY'\nprint(1)\nPY\n" },
  { path: 'tools/build.sh', text: 'go build ./...\nuv run tool\npip install torch\n' },
  { path: 'examples/x/package.json', text: JSON.stringify({ scripts: { build: 'python3 build.py', serve: 'python3 -m http.server' } }) },
  { path: '.github/workflows/ci.yml', text: 'steps:\n  - uses: actions/setup-python@v6\n  - run: python -m unittest discover\n' },
]);
const byPath = new Map();
for (const v of violations) byPath.set(v.path, [...(byPath.get(v.path) ?? []), v.rule]);
for (const file of ['tools/a.js', 'tools/b.js', 'tools/c.js', 'tools/d.js', 'tools/e.js', 'tools/f.js', 'tools/g.js',
  'tools/h.js', 'tools/probe.sh', 'tools/heredoc.sh', 'tools/build.sh', 'examples/x/package.json', '.github/workflows/ci.yml']) {
  assert.ok(byPath.has(file), `${file} must fail the JavaScript-only check`);
}
assert.ok(byPath.get('tools/a.js').includes('spawns-interpreter'));
assert.ok(byPath.get('tools/h.js').includes('spawns-interpreter'));
assert.ok(byPath.get('tools/build.sh').includes('go-command'));
assert.ok(byPath.get('tools/build.sh').includes('uv-command'));
assert.ok(byPath.get('.github/workflows/ci.yml').includes('toolchain-setup'));
assert.equal(violations.find((v) => v.path === 'tools/probe.sh').line, 2);

console.log('js-only-check.test: clean fixtures pass; Python/Go files, shebangs, spawns, commands, env and workflows fail');

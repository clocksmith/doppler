// A step stores only the activations required by the chain rule. The backward
// sweep reuses one layer's scratch across all blocks, including both residual
// branches. Each parameter gradient is assigned once, except for tied wte.
/** @typedef {{value: GPUBufferBinding, gradient: GPUBufferBinding}} Parameter */
/** @typedef {{output: GPUBuffer, backward: (dy: GPUBuffer, dx: GPUBuffer) => void}} LayerOutput */
/** @typedef {import('./tabgpt-ops.js').TabGPTDispatch} Dispatch */
/** @param {import('./tabgpt-model.js').TabGPTConfig} config
 * @param {ReadonlyMap<string, Parameter>} param
 * @param {(elements:number,label:string) => GPUBuffer} alloc
 * @param {import('./tabgpt-ops.js').TabGPTOps['compile']} op
 * @param {import('./tabgpt-matmul.js').TabGPTMatmul} matmul
 */
export function buildTabGPTGraph(config, param, alloc, op, matmul) {
  const { B, T, C, L, NH, V } = config;
  const rows = B * T;
  const hidden = rows * C;
  const queries = B * NH * T;
  const scores = queries * T;
  /** @type {Dispatch[]} */
  const forward = [];
  /** @type {Dispatch[]} */
  const backward = [];
  /** @param {string} name */
  function getWeight(name) {
    const weight = param.get(name);
    if (!weight) throw new Error(`Missing TabGPT parameter: ${name}`);
    return weight;
  }
  const wte = getWeight('wte');
  const ids = alloc(rows, 'input-ids');
  const targets = alloc(rows, 'targets');
  const outputLoss = alloc(rows, 'token-losses');
  let residual = alloc(hidden, 'token-embeddings');
  forward.push(op('embed', [ids, wte.value, residual]));

  // Functions added to this list consume the gradient of the block output and
  // replace it with the gradient of the block input, in reverse layer order.
  const layers = [];
  const dResidual = alloc(hidden, 'd-residual');
  const dMid = alloc(hidden, 'd-mlp-residual');
  const dBranch = alloc(hidden, 'd-residual-branch');
  const dNorm = alloc(hidden, 'd-norm');
  const dContext = alloc(hidden, 'd-attention-output');
  const dPre = alloc(4 * hidden, 'd-mlp-preactivation');
  const dAct = alloc(4 * hidden, 'd-mlp-activation');
  const dQkv = alloc(3 * hidden, 'd-qkv');
  const dScores = alloc(scores, 'd-attention-scores');

  /** @param {GPUBuffer} input @param {Parameter} weight @returns {LayerOutput} */
  function normalize(input, weight) {
    const output = alloc(hidden, 'norm');
    const rstd = alloc(rows, 'norm-inverse-rms');
    forward.push(op('rms', [input, weight.value, output, rstd], { width: C }));
    return {
      output,
      backward(dy, dx) {
        backward.push(op('rmsDx', [dy, input, weight.value, rstd, dx], { width: C }));
        backward.push(op('rmsDw', [dy, input, rstd, weight.gradient], { width: C }));
      },
    };
  }

  /** @param {GPUBuffer} input @param {Parameter} weight
   * @param {number} inputWidth @param {number} outputWidth @param {boolean} [tiedHead] @returns {LayerOutput}
   */
  function project(input, weight, inputWidth, outputWidth, tiedHead = false) {
    const output = alloc(rows * outputWidth, 'projection');
    forward.push(matmul(input, weight.value, output, rows, outputWidth, inputWidth, tiedHead ? 'b' : ''));
    return {
      output,
      backward(dy, dx) {
        backward.push(matmul(dy, weight.value, dx, rows, inputWidth, outputWidth, tiedHead ? '' : 'b'));
        if (tiedHead) {
          backward.push(matmul(dy, input, weight.gradient, outputWidth, inputWidth, rows, 'a'));
        } else {
          backward.push(matmul(input, dy, weight.gradient, inputWidth, outputWidth, rows, 'a'));
        }
      },
    };
  }

  for (let i = 0; i < L; i++) {
    /** @param {string} suffix */
    const weight = (suffix) => getWeight(`h.${i}.${suffix}.weight`);
    const norm1 = normalize(residual, weight('ln1'));
    const qkv = project(norm1.output, weight('attn.qkv'), C, 3 * C);
    const qkRstd = alloc(2 * queries, 'qk-inverse-rms');
    const probs = alloc(scores, 'attention-probabilities');
    const context = alloc(hidden, 'attention-output');
    forward.push(op('position', [qkv.output, qkRstd]));
    forward.push(op('scores', [qkv.output, probs]));
    forward.push(op('context', [probs, qkv.output, context]));
    const attention = project(context, weight('attn.proj'), C, C);
    const middle = alloc(hidden, 'attention-residual');
    forward.push(op('add', [residual, attention.output, middle], { count: hidden }));
    const norm2 = normalize(middle, weight('ln2'));
    const fc = project(norm2.output, weight('mlp.fc'), C, 4 * C);
    const activated = alloc(4 * hidden, 'relu-squared');
    forward.push(op('relu2', [fc.output, activated], { count: 4 * hidden }));
    const mlp = project(activated, weight('mlp.proj'), 4 * C, C);
    residual = alloc(hidden, 'block-output');
    forward.push(op('add', [middle, mlp.output, residual], { count: hidden }));

    layers.push(() => {
      mlp.backward(dResidual, dAct);
      backward.push(op('relu2Grad', [dAct, fc.output, dPre], { count: 4 * hidden }));
      fc.backward(dPre, dNorm);
      norm2.backward(dNorm, dBranch);
      backward.push(op('add', [dResidual, dBranch, dMid], { count: hidden }));
      attention.backward(dMid, dContext);
      backward.push(op('scoresGrad', [dContext, context, probs, qkv.output, dScores]));
      backward.push(op('qkvGrad', [dScores, probs, qkv.output, dContext, dQkv]));
      backward.push(op('positionGrad', [qkv.output, qkRstd, dQkv]));
      qkv.backward(dQkv, dNorm);
      norm1.backward(dNorm, dBranch);
      backward.push(op('add', [dMid, dBranch, dResidual], { count: hidden }));
    });
  }
  const norm = normalize(residual, getWeight('lnf.weight'));
  const logits = project(norm.output, wte, C, V, true);
  const dLogits = alloc(rows * V, 'd-logits');
  forward.push(op('loss', [logits.output, targets, outputLoss, dLogits]));
  logits.backward(dLogits, dNorm);
  norm.backward(dNorm, dResidual);
  for (const layer of layers.reverse()) layer();
  backward.push(op('embedGrad', [ids, dResidual, wte.gradient]));
  return { forward, backward, ids, targets, outputLoss };
}

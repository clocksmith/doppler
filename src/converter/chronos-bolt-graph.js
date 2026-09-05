// Forge-only lowering. Source tensor names and T5 topology are resolved here;
// the runtime receives only immutable slots, uploads and dispatch commands.
export function lowerChronosBoltGraph(source, recipe, kernels) {
  const config = source.config;
  const chronos = config.chronos_config;
  if (source.repository !== recipe.repository || source.revision !== recipe.revision
    || config.architectures?.[0] !== 'ChronosBoltModelForForecasting'
    || config.is_encoder_decoder !== true || config.is_gated_act !== false
    || config.dense_act_fn !== 'relu' || config.feed_forward_proj !== 'relu'
    || chronos.input_patch_size !== chronos.input_patch_stride || chronos.use_reg_token !== true
    || recipe.dtypeLane !== 'source-f32-reference' || recipe.contextLength > chronos.context_length
    || recipe.contextLength % chronos.input_patch_size !== 0) {
    throw new Error('Source is outside the declared Chronos-Bolt numeric lowering contract.');
  }
  const C = recipe.contextLength;
  const P = chronos.input_patch_size;
  const N = C / P;
  const D = config.d_model;
  const F = config.d_ff;
  const S = N + 1;
  const WG = recipe.workgroupSize;
  const TILE = recipe.matmulTileSize;
  if (S > 129 || D !== config.num_heads * config.d_kv) throw new Error('Attention geometry exceeds this lowering.');
  const slots = [];
  const uploads = [];
  const constants = [];
  const steps = [];
  const tensorBindings = [];
  const byName = new Map();
  const shapes = new Map();
  function slot(id, elements, role = 'intermediate', uniform = false) {
    if (!Number.isSafeInteger(elements) || elements < 1 || slots.some(entry => entry.slotId === id)) throw new Error('Invalid/duplicate graph slot: ' + id);
    slots.push({ slotId: id, role, scope: role === 'intermediate' ? 'session' : 'static', owner: 'runtime',
      usage: [uniform ? 'uniform' : 'storage', 'copy-dst', 'copy-src'], size: { op: 'constant', bytes: elements * 4 } });
    return id;
  }
  function tensor(name, expected) {
    const metadata = source.tensors[name];
    if (!metadata || metadata.dtype !== 'F32' || JSON.stringify(metadata.shape) !== JSON.stringify(expected)) {
      throw new Error(`Source tensor ${name} must have shape ${JSON.stringify(expected)} and dtype F32.`);
    }
    if (byName.has(name)) return byName.get(name);
    const id = slot(`weight-${byName.size}`, metadata.sizeBytes / 4, 'weight');
    uploads.push({ slotId: id, artifactId: 'weights', offsetBytes: metadata.offsetBytes, sizeBytes: metadata.sizeBytes });
    tensorBindings.push({ name, slotId: id, ...metadata });
    byName.set(name, id);
    return id;
  }
  function data(id, values, types = values.map(() => 'u32'), uniform = true) {
    const buffer = new ArrayBuffer(values.length * 4);
    const view = new DataView(buffer);
    values.forEach((value, index) => types[index] === 'f32' ? view.setFloat32(index * 4, value, true) : view.setUint32(index * 4, value, true));
    const offsetBytes = constants.reduce((sum, bytes) => sum + bytes.byteLength, 0);
    constants.push(new Uint8Array(buffer));
    slot(id, values.length, 'constant', uniform);
    uploads.push({ slotId: id, artifactId: 'constants', offsetBytes, sizeBytes: buffer.byteLength });
    return id;
  }
  function command(name, operation, bindings, workgroups, overrides) {
    const kernel = kernels[operation];
    if (!kernel) throw new Error('Missing resolved registry kernel: ' + operation);
    steps.push({ id: name, kind: 'dispatch', phase: 'forecast', moduleId: kernel.id, entry: kernel.entry,
      constants: overrides, bindings: bindings.map((slotId, binding) => ({ slotId, binding })), workgroups,
      waitForCompletion: false });
  }
  function matrix(id, rows, columns) { shapes.set(id, [rows, columns]); return slot(id, rows * columns); }
  function linear(id, input, name, rows, inputWidth, outputWidth, bias = false) {
    const weight = tensor(name + '.weight', [outputWidth, inputWidth]);
    const output = matrix(id, rows, outputWidth);
    const u = data(id + '-dims', [rows, outputWidth, inputWidth, 1, 1, 0, 0, 0], ['u32', 'u32', 'u32', 'f32', 'u32', 'u32', 'u32', 'u32']);
    command(id, 'matmul', [u, input, weight, output], [Math.ceil(rows / TILE), Math.ceil(outputWidth / TILE), 1], { TILE_SIZE: TILE });
    if (bias) {
      const biasSlot = tensor(name + '.bias', [outputWidth]);
      const dims = data(id + '-bias-dims', [rows, outputWidth, 0, 0, 1, 0, 0, 0]);
      command(id + '-bias', 'bias', [dims, output, biasSlot], [Math.ceil(outputWidth / WG), rows, 1], { WORKGROUP_SIZE: WG });
    }
    return output;
  }
  function add(id, a, b, rows, width) {
    const output = matrix(id, rows, width);
    const u = data(id + '-dims', [rows * width, 1, 1, 0], ['u32', 'f32', 'u32', 'u32']);
    command(id, 'residual', [u, a, b, output], [Math.ceil(rows * width / WG), 1, 1], { WORKGROUP_SIZE: WG });
    return output;
  }
  function relu(id, input, rows, width) {
    const output = matrix(id, rows, width);
    command(id, 'relu', [data(id + '-dims', [rows * width, 1, 0, 0]), input, output],
      [Math.ceil(rows * width / WG), 1, 1], { WORKGROUP_SIZE: WG });
    return output;
  }
  function residualBlock(id, input, name, rows, inputWidth, outputWidth) {
    const hidden = linear(id + '-hidden', input, name + '.hidden_layer', rows, inputWidth, F, true);
    const activated = relu(id + '-relu', hidden, rows, F);
    const projected = linear(id + '-project', activated, name + '.output_layer', rows, F, outputWidth, true);
    const residual = linear(id + '-skip', input, name + '.residual_layer', rows, inputWidth, outputWidth, true);
    return add(id, projected, residual, rows, outputWidth);
  }
  function norm(id, input, name, rows) {
    const output = matrix(id, rows, D);
    command(id, 'norm', [input, tensor(name + '.weight', [D]), output], [Math.ceil(rows / WG), 1, 1],
      { WORKGROUP_SIZE: WG, ROWS: rows, HIDDEN_SIZE: D, EPSILON: config.layer_norm_epsilon });
    return output;
  }
  function attention(id, input, kvInput, prefix, qRows, kRows, mask, relative, bidirectional) {
    const q = linear(id + '-q', input, prefix + '.q', qRows, D, D);
    const k = linear(id + '-k', kvInput, prefix + '.k', kRows, D, D);
    const v = linear(id + '-v', kvInput, prefix + '.v', kRows, D, D);
    const attended = matrix(id + '-weighted', qRows, D);
    const bias = relative ? tensor(relative, [config.relative_attention_num_buckets, config.num_heads]) : 'zero';
    command(id + '-attention', 'attention', [q, k, v, bias, mask, attended], [Math.ceil(qRows * config.num_heads / WG), 1, 1], {
      WORKGROUP_SIZE: WG, QUERY_LENGTH: qRows, KEY_LENGTH: kRows, NUM_HEADS: config.num_heads,
      HEAD_DIM: config.d_kv, NUM_BUCKETS: config.relative_attention_num_buckets,
      MAX_DISTANCE: config.relative_attention_max_distance, HAS_RELATIVE_BIAS: Number(Boolean(relative)), BIDIRECTIONAL: Number(bidirectional),
    });
    return linear(id + '-o', attended, prefix + '.o', qRows, D, D);
  }
  function ff(id, input, prefix, rows) {
    const normalized = norm(id + '-norm', input, prefix + '.layer_norm', rows);
    const up = linear(id + '-up', normalized, prefix + '.DenseReluDense.wi', rows, D, F);
    const activated = relu(id + '-relu', up, rows, F);
    const down = linear(id + '-down', activated, prefix + '.DenseReluDense.wo', rows, F, D);
    return add(id, input, down, rows, D);
  }
  slot('input', C, 'input'); slot('mask', C, 'input'); slot('moments', 2);
  matrix('patches', N, P * 2); slot('attention-mask', S);
  data('zero', [0], ['f32'], false); data('one', [1], ['f32'], false);
  command('moments', 'stats', ['input', 'mask', 'moments'], [1, 1, 1], { WORKGROUP_SIZE: WG, CONTEXT_LENGTH: C, ZERO_SCALE_EPS: recipe.instanceNormEpsilon });
  command('patches', 'patch', ['input', 'mask', 'moments', 'patches', 'attention-mask'], [Math.ceil(N * P * 2 / WG), 1, 1],
    { WORKGROUP_SIZE: WG, CONTEXT_LENGTH: C, PATCH_SIZE: P, PATCH_COUNT: N });
  const embedded = residualBlock('patch-embedding', 'patches', 'input_patch_embedding', N, P * 2, D);
  const shared = tensor('shared.weight', [config.vocab_size, D]);
  let hidden = matrix('encoder-input', S, D);
  command('append-register', 'append', [embedded, shared, hidden], [Math.ceil(S * D / WG), 1, 1],
    { WORKGROUP_SIZE: WG, INPUT_ROWS: N, HIDDEN_SIZE: D, EMBEDDING_ID: config.reg_token_id });
  for (let i = 0; i < config.num_layers; i++) {
    const prefix = `encoder.block.${i}`;
    const normalized = norm(`encoder-${i}-norm`, hidden, prefix + '.layer.0.layer_norm', S);
    const projected = attention(`encoder-${i}`, normalized, normalized, prefix + '.layer.0.SelfAttention', S, S,
      'attention-mask', 'encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight', true);
    hidden = ff(`encoder-${i}-ff`, add(`encoder-${i}-residual`, hidden, projected, S, D), prefix + '.layer.1', S);
  }
  const encoded = norm('encoder-final', hidden, 'encoder.final_layer_norm', S);
  let decoder = matrix('decoder-input', 1, D);
  const gatherDims = data('decoder-dims', [1, D, config.vocab_size, 0, 0, D, 0, 0, 0, 0, 0, 0]);
  command('decoder-embed', 'gather', [gatherDims, data('decoder-token', [config.decoder_start_token_id], ['u32'], false), shared, decoder],
    [Math.ceil(D / WG), 1, 1], { WORKGROUP_SIZE_MAIN: WG });
  for (let i = 0; i < config.num_decoder_layers; i++) {
    const prefix = `decoder.block.${i}`;
    const normalized = norm(`decoder-${i}-self-norm`, decoder, prefix + '.layer.0.layer_norm', 1);
    const self = attention(`decoder-${i}-self`, normalized, normalized, prefix + '.layer.0.SelfAttention', 1, 1,
      'one', 'decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight', false);
    const residual = add(`decoder-${i}-self-residual`, decoder, self, 1, D);
    const crossInput = norm(`decoder-${i}-cross-norm`, residual, prefix + '.layer.1.layer_norm', 1);
    const cross = attention(`decoder-${i}-cross`, crossInput, encoded, prefix + '.layer.1.EncDecAttention', 1, S, 'attention-mask', null, false);
    decoder = ff(`decoder-${i}-ff`, add(`decoder-${i}-cross-residual`, residual, cross, 1, D), prefix + '.layer.2', 1);
  }
  const decoded = norm('decoder-final', decoder, 'decoder.final_layer_norm', 1);
  const output = residualBlock('output-embedding', decoded, 'output_patch_embedding', 1, D, chronos.prediction_length * chronos.quantiles.length);
  const indices = recipe.outputQuantiles.map(q => chronos.quantiles.indexOf(q));
  if (indices.some(index => index < 0)) throw new Error('Output quantile is absent from source model.');
  const quantileSlot = data('quantile-indices', indices, indices.map(() => 'u32'), false);
  slot('request', 4, 'input', true);
  slot('output', chronos.prediction_length * indices.length, 'output');
  command('forecast-output', 'output', ['request', output, 'moments', quantileSlot, 'output'],
    [Math.ceil(chronos.prediction_length * indices.length / WG), 1, 1],
    { WORKGROUP_SIZE: WG, PREDICTION_LENGTH: chronos.prediction_length, OUTPUT_QUANTILES: indices.length });
  if (byName.size !== Object.keys(source.tensors).length) throw new Error('Unaccounted source tensors: ' + Object.keys(source.tensors).filter(name => !byName.has(name)).join(', '));
  const constantsBytes = new Uint8Array(constants.reduce((size, bytes) => size + bytes.length, 0));
  let offset = 0;
  for (const bytes of constants) { constantsBytes.set(bytes, offset); offset += bytes.length; }
  return { slots, uploads, steps, constantsBytes, tensorBindings, forecast: {
    contextLength: C, predictionLength: chronos.prediction_length, quantiles: recipe.outputQuantiles,
    inputDtype: 'f32', outputDtype: 'f32', outputLayout: 'time-quantile', missingInput: 'left-pad-masked-zero',
    inputSlot: 'input', maskSlot: 'mask', requestSlot: 'request', outputSlot: 'output',
  } };
}

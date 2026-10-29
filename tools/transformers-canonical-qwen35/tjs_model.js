// The public TJS model constructor accepts a named inference-session record.
// This adapter owns only explicit phase routing and port validation; TJS owns
// text embedding, decoder forward, tensor wrapping and its public cache API.

export function cachePorts(config, kvPhysicalDtype) {
  if (!['float16', 'float32'].includes(kvPhysicalDtype)) throw new Error('Physical KV dtype must be explicit');
  const text = config.text_config;
  return text.layer_types.flatMap((kind, layer) => {
    if (kind === 'linear_attention') {
      const width = 2 * text.linear_num_key_heads * text.linear_key_head_dim +
        text.linear_num_value_heads * text.linear_value_head_dim;
      return [
        { input: `past_conv.${layer}`, output: `present_conv.${layer}`, type: 'float32', kind,
          shape: [1, width, text.linear_conv_kernel_dim] },
        { input: `past_recurrent.${layer}`, output: `present_recurrent.${layer}`, type: 'float32', kind,
          shape: [1, text.linear_num_value_heads, text.linear_key_head_dim, text.linear_value_head_dim] },
      ];
    }
    if (kind !== 'full_attention') throw new Error(`Unsupported explicit cache layer kind ${kind}`);
    return ['key', 'value'].map(role => ({ input: `past_key_values.${layer}.${role}`,
      output: `present.${layer}.${role}`, type: kvPhysicalDtype, kind,
      shape: [1, text.num_key_value_heads, null, text.head_dim] }));
  });
}

export class RoutedDecoder {
  constructor(prefill, decode, config, policy, controlFactory) {
    this.prefill = prefill;
    this.decode = decode;
    this.policy = policy;
    this.controls = policy.hostGeometryInputs ?? { prefill: [], decode: [] };
    if (!Array.isArray(this.controls.prefill) || !Array.isArray(this.controls.decode)) {
      throw new Error('Host geometry must declare both prefill and decode port lists');
    }
    this.controlFactory = controlFactory;
    this.ports = cachePorts(config, policy.kvPhysicalDtype);
    this.inputNames = ['inputs_embeds', 'position_ids', 'attention_mask', ...this.ports.map(port => port.input)];
    this.outputNames = ['logits', ...this.ports.map(port => port.output)];
    for (const [phase, session] of [['prefill', prefill], ['decode', decode]]) {
      const expectedInputs = [...this.inputNames, ...this.controls[phase].map(port => port.name)];
      for (const [field, names] of [['inputNames', expectedInputs], ['outputNames', this.outputNames]]) {
        if (session[field].length !== names.length || names.some(name => !session[field].includes(name))) {
          throw new Error(`${phase} ONNX ${field} differ from the complete explicit text/cache contract`);
        }
      }
    }
    if (Object.values(this.controls).some(ports => ports.length) && !controlFactory) {
      throw new Error('Declared host geometry ports require an explicit public tensor factory');
    }
    this.inputMetadata = prefill.inputMetadata.filter(meta => this.inputNames.includes(meta.name));
    this.integerInputTypes = Object.fromEntries(['position_ids', 'attention_mask'].map(name => {
      const type = this.inputMetadata.find(meta => meta.name === name)?.type;
      if (!['int32', 'int64'].includes(type)) throw new Error(`Unsupported declared application index type: ${name}`);
      return [name, type];
    }));
    for (const [name, type] of Object.entries(this.integerInputTypes)) {
      if (decode.inputMetadata.find(meta => meta.name === name)?.type !== type ||
          policy.integerInputs && policy.integerInputs[name] !== type) {
        throw new Error(`Declared graph/application index types disagree: ${name}`);
      }
    }
    this.outputMetadata = prefill.outputMetadata;
    this.config = { dtype: 'fp32', device: 'webgpu' };
    this.work = [];
    this.pending = false;
    this.closed = false;
  }

  async run(feeds, fetches, options) {
    if (this.closed || this.pending) throw new Error('Routed decoder is closed or already executing');
    const full = this.ports.filter(port => port.kind === 'full_attention');
    const past = feeds[full[0].input].dims[2];
    for (const port of this.ports) {
      const tensor = feeds[port.input];
      if (!tensor || tensor.type !== port.type || tensor.dims.length !== port.shape.length ||
          port.shape.some((dimension, axis) => tensor.dims[axis] !== (dimension ?? past))) {
        throw new Error(`Cache port shape/physical dtype mismatch: ${port.input}`);
      }
    }
    const length = feeds.inputs_embeds.dims[1];
    const phase = past === 0 ? 'prefill' : 'decode';
    if (feeds.inputs_embeds.type !== 'float32' || feeds.position_ids.type !== this.integerInputTypes.position_ids ||
        feeds.attention_mask.type !== this.integerInputTypes.attention_mask || feeds.position_ids.dims[0] !== 3 ||
        feeds.inputs_embeds.dims[0] !== this.policy.batchSize || length < 1 ||
        feeds.attention_mask.dims[1] !== past + length || feeds.position_ids.dims[2] !== length) {
      throw new Error('Text input batch, position or mask geometry differs from explicit graph contract');
    }
    if ((phase === 'prefill' && length > this.policy.prefillBound) ||
        (phase === 'decode' && length !== 1) || past + length > this.policy.maxSequenceLength) {
      throw new Error(`Unsupported ${phase} shape: input ${length}, past ${past}`);
    }
    if (this.policy.rotaryLookup) {
      const { minimumPosition, maximumPositionExclusive } = this.policy.rotaryLookup;
      const positions = feeds.position_ids;
      // These are application-supplied position indices, never activation or
      // state readback. Check the declared lookup domain before GPU dispatch.
      if (positions.location !== 'cpu' || positions.data.length !== 3 * length ||
          !positions.data.every(value => Number.isSafeInteger(Number(value)) &&
            Number(value) >= minimumPosition && Number(value) < maximumPositionExclusive)) {
        throw new Error('Rotary lookup requires CPU application position indices inside its declared bounds');
      }
    }
    this.pending = true;
    const geometryTensors = [];
    try {
      const prepared = { ...feeds };
      const geometry = [];
      for (const control of this.controls[phase]) {
        const recipes = control.values ?? [control];
        const values = recipes.map(expression => expression.constant === undefined ?
          feeds[expression.sourceTensor]?.dims[expression.axis] * expression.multiplier + expression.addend : expression.constant);
        if (!values.every(Number.isSafeInteger)) throw new Error(`Invalid declared host geometry: ${control.name}`);
        const tensor = this.controlFactory(control.type, values, control.shape);
        geometryTensors.push(tensor);
        prepared[control.name] = tensor;
        geometry.push({ name: control.name, recipes, values, type: control.type });
      }
      const result = await this[phase].run(prepared, fetches, options);
      for (const port of this.ports) {
        const tensor = result[port.output];
        if (!tensor || tensor.type !== port.type || tensor.location !== 'gpu-buffer' ||
            tensor.dims.length !== port.shape.length ||
            port.shape.some((dimension, axis) => tensor.dims[axis] !== (dimension ?? past + length))) {
          for (const value of Object.values(result)) value.dispose();
          throw new Error(`Actual state output is not ${port.type} GPU resident: ${port.output}`);
        }
      }
      this.work.push({ phase, inputTokens: length, pastTokens: past, totalTokens: past + length,
        nativeChunkSize: this.policy.chunkSize,
        requiredPromptChunks: phase === 'prefill' ? Math.ceil(length / this.policy.chunkSize) : null,
        scheduledChunks: phase === 'prefill' ? this.policy.prefillBound / this.policy.chunkSize : null,
        scheduledPrefillTokens: phase === 'prefill' ? this.policy.prefillBound : null,
        paddedPrefillTokens: phase === 'prefill' ? this.policy.prefillBound - length : null,
        applicationIndexTypes: this.integerInputTypes, hostGeometry: geometry,
        rotaryLookup: this.policy.rotaryLookup ?? null,
        decode: phase === 'decode' ? 'one-token recurrent update' : null });
      return result;
    } finally {
      for (const tensor of geometryTensors) tensor.dispose();
      this.pending = false;
    }
  }

  async release() {
    if (this.pending) throw new Error('Cannot release a routed decoder during execution');
    if (this.closed) return;
    this.closed = true;
    await Promise.all([this.prefill.release(), this.decode.release()]);
  }
}

export function createTjsTextModel(tjs, config, sessions, policy) {
  const decoder = new RoutedDecoder(sessions.prefill, sessions.decode, config,
    { ...policy.textExport, chunkSize: policy.linearExport.chunkSize }, (type, values, shape) => {
      if (!['int32', 'int64'].includes(type)) throw new Error('Host geometry may contain only declared scalar integer dimensions');
      const data = type === 'int32' ? Int32Array.from(values) : BigInt64Array.from(values, BigInt);
      return new tjs.Tensor(type, data, shape).ort_tensor;
    });
  const model = new tjs.Qwen3_5ForCausalLM(new tjs.PretrainedConfig(config), {
    embed_tokens: sessions.embedding,
    decoder_model_merged: decoder,
  }, { generation_config: { eos_token_id: policy.teacher.eosTokenIds } });
  return { model, decoder, label: `Transformers.js 4.2 + explicit canonical prefill/decode session routing adapter; physical KV ${policy.textExport.kvPhysicalDtype}` };
}

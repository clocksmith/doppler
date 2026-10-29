import test from 'node:test';
import assert from 'node:assert/strict';
import { RoutedDecoder, cachePorts } from './tjs_model.js';

const config = { text_config: { layer_types: ['linear_attention', 'full_attention'],
  linear_num_key_heads: 2, linear_key_head_dim: 4, linear_num_value_heads: 2,
  linear_value_head_dim: 4, linear_conv_kernel_dim: 4, num_key_value_heads: 2, head_dim: 4 } };
const policy = { batchSize: 1, prefillBound: 128, maxSequenceLength: 2048, chunkSize: 64, kvPhysicalDtype: 'float16' };
const ports = cachePorts(config, policy.kvPhysicalDtype);
const tensor = (type, dims) => ({ type, dims, location: 'gpu-buffer', dispose() {} });

function feeds(past, length) {
  return { inputs_embeds: tensor('float32', [1, length, 8]), position_ids: tensor('int64', [3, 1, length]),
    attention_mask: tensor('int64', [1, past + length]), ...Object.fromEntries(ports.map(port =>
      [port.input, tensor(port.type, port.shape.map(dimension => dimension ?? past))])) };
}

function session() {
  return { inputNames: ['inputs_embeds', 'position_ids', 'attention_mask', ...ports.map(p => p.input)],
    outputNames: ['logits', ...ports.map(p => p.output)], inputMetadata: [
      { name: 'inputs_embeds', type: 'float32' }, { name: 'position_ids', type: 'int64' },
      { name: 'attention_mask', type: 'int64' }, ...ports.map(port => ({ name: port.input, type: port.type, shape: port.shape }))],
    outputMetadata: [], calls: [],
    async run(input) {
      this.calls.push(input);
      const length = input.inputs_embeds.dims[1] + input['past_key_values.1.key'].dims[2];
      return { logits: tensor('float32', [1, 1, 10]), ...Object.fromEntries(ports.map(port =>
        [port.output, tensor(port.type, port.shape.map(dimension => dimension ?? length))])) };
    }, async release() { this.released = true; } };
}

test('public-session router batches prefill and preserves one-token recurrent decode', async () => {
  const prefill = session(), decode = session(), routed = new RoutedDecoder(prefill, decode, config, policy);
  await routed.run(feeds(0, 28));
  await routed.run(feeds(28, 1));
  assert.equal(prefill.calls.length, 1);
  assert.equal(decode.calls.length, 1);
  assert.equal(routed.work[0].scheduledChunks, 2);
  assert.equal(routed.work[0].paddedPrefillTokens, 100);
  assert.equal(routed.work[1].decode, 'one-token recurrent update');
  await routed.release();
  assert.equal(prefill.released, true);
  assert.equal(decode.released, true);
});

test('router rejects physical cache dtype substitution and unsupported prompt shapes', async () => {
  const prefill = session(), decode = session(), routed = new RoutedDecoder(prefill, decode, config, policy);
  const wrong = feeds(28, 1);
  wrong['past_key_values.1.key'].type = 'float32';
  await assert.rejects(routed.run(wrong), /physical dtype mismatch/);
  await assert.rejects(routed.run(feeds(0, 129)), /Unsupported prefill shape/);
  await assert.rejects(routed.run(feeds(28, 2)), /Unsupported decode shape/);
  assert.equal(prefill.calls.length + decode.calls.length, 0);
});

test('explicit F32-KV lane preserves phase detection and rejects F16 substitution', async () => {
  const prefill = session(), decode = session();
  for (const target of [prefill, decode]) {
    const original = target.run.bind(target);
    target.run = async input => {
      const output = await original(input);
      for (const port of ports.filter(p => p.kind === 'full_attention')) output[port.output].type = 'float32';
      return output;
    };
  }
  const routed = new RoutedDecoder(prefill, decode, config, { ...policy, kvPhysicalDtype: 'float32' });
  const input = feeds(0, 2);
  for (const port of ports.filter(p => p.kind === 'full_attention')) input[port.input].type = 'float32';
  await routed.run(input);
  assert.equal(prefill.calls.length, 1);
  await assert.rejects(routed.run(feeds(2, 1)), /physical dtype mismatch/);
});

test('router rejects an incomplete exported cache contract', () => {
  const incomplete = session();
  incomplete.inputNames.pop();
  assert.throws(() => new RoutedDecoder(incomplete, session(), config, policy), /complete explicit text\/cache contract/);
});

test('declared geometry ports use only tensor dimensions and release their scalar inputs', async () => {
  const prefill = session(), decode = session();
  const descriptor = { name: 'host_length', type: 'int32', shape: [], sourceTensor: 'attention_mask',
    axis: 1, multiplier: 1, addend: 0 };
  const vector = { name: 'host_target', type: 'int64', shape: [4], values: [{ constant: 1 },
    { sourceTensor: 'attention_mask', axis: 1, multiplier: 1, addend: 0 }, { constant: 2 }, { constant: 4 }] };
  for (const target of [prefill, decode]) {
    for (const control of [descriptor, vector]) {
      target.inputNames.push(control.name);
      target.inputMetadata.push({ name: control.name, type: control.type, shape: control.shape });
    }
  }
  let disposed = 0;
  const factory = (type, values, shape) => ({ type, dims: shape, data: Int32Array.from(values), dispose() { disposed++; } });
  const routed = new RoutedDecoder(prefill, decode, config, { ...policy,
    hostGeometryInputs: { prefill: [descriptor, vector], decode: [descriptor, vector] } }, factory);
  await routed.run(feeds(0, 28));
  assert.equal(prefill.calls[0].host_length.data[0], 28);
  assert.deepEqual([...prefill.calls[0].host_target.data], [1, 28, 2, 4]);
  assert.equal(disposed, 2);
  assert.equal(routed.inputNames.includes('host_length'), false);
  assert.equal(routed.inputMetadata.some(meta => meta.name === 'host_length'), false);
  assert.deepEqual(routed.work[0].hostGeometry[0].values, [28]);
});

test('router fails closed on concurrent execution and nonresident state', async () => {
  const prefill = session(), decode = session();
  let finish;
  const run = prefill.run.bind(prefill);
  prefill.run = async input => {
    await new Promise(resolve => { finish = resolve; });
    const output = await run(input);
    output['present_conv.0'].location = 'cpu';
    return output;
  };
  const routed = new RoutedDecoder(prefill, decode, config, policy);
  const pending = routed.run(feeds(0, 2));
  await assert.rejects(routed.run(feeds(0, 2)), /already executing/);
  finish();
  await assert.rejects(pending, /GPU resident/);
});

test('bounded rotary lookup rejects unsupported application positions before GPU dispatch', async () => {
  const prefill = session(), decode = session();
  const routed = new RoutedDecoder(prefill, decode, config, { ...policy,
    rotaryLookup: { minimumPosition: 0, maximumPositionExclusive: 2048 } });
  const input = feeds(0, 2);
  input.position_ids.location = 'cpu';
  input.position_ids.data = BigInt64Array.from([0n, 1n, 2047n, 17n, 12n, 13n]);
  await routed.run(input);
  input.position_ids.data[0] = 2048n;
  await assert.rejects(routed.run(input), /declared bounds/);
  input.position_ids.data[0] = -1n;
  await assert.rejects(routed.run(input), /declared bounds/);
  assert.equal(prefill.calls.length, 1);
});

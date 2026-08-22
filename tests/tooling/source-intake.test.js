import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { inspectSourceModel } from '../../src/tooling/source-intake.js';

const repoRoot = path.resolve(new URL('../..', import.meta.url).pathname);
const policy = JSON.parse(
  await fs.readFile(
    path.join(repoRoot, 'src/config/evidence/source-intake-policy.json'),
    'utf8'
  )
);
const sourceDir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-source-intake-'));

function safetensorsBytes(tensors) {
  const header = {};
  let offset = 0;
  for (const tensor of tensors) {
    const size = tensor.shape.reduce((total, value) => total * value, 1) * 2;
    header[tensor.name] = {
      dtype: 'F16',
      shape: tensor.shape,
      data_offsets: [offset, offset + size],
    };
    offset += size;
  }
  let headerText = JSON.stringify(header);
  while (headerText.length % 8 !== 0) headerText += ' ';
  const headerBytes = new TextEncoder().encode(headerText);
  const output = new Uint8Array(8 + headerBytes.length);
  new DataView(output.buffer).setBigUint64(0, BigInt(headerBytes.length), true);
  output.set(headerBytes, 8);
  return output;
}

try {
  await fs.writeFile(path.join(sourceDir, 'config.json'), JSON.stringify({
    model_type: 'example',
    hidden_size: 16,
    intermediate_size: 32,
    num_hidden_layers: 1,
    num_attention_heads: 4,
    num_key_value_heads: 2,
    vocab_size: 64,
    hidden_act: 'silu',
    rms_norm_eps: 0.000001,
    rope_theta: 1000000,
    tie_word_embeddings: true,
    text_config: {
      novel_attention_mode: 'new-layout',
      recurrent_state_size: 8,
    },
  }));
  await fs.writeFile(
    path.join(sourceDir, 'model.safetensors'),
    safetensorsBytes([
      { name: 'model.embed_tokens.weight', shape: [64, 16] },
      { name: 'model.layers.0.self_attn.q_proj.weight', shape: [16, 16] },
      { name: 'model.layers.0.mlp.down_proj.weight', shape: [16, 32] },
      { name: 'model.layers.0.delta_state.weight', shape: [16, 16] },
    ])
  );

  const first = await inspectSourceModel({ sourceDir, policy });
  const second = await inspectSourceModel({ sourceDir, policy });
  assert.equal(first.report.schema, 'doppler.source-intake/v1');
  assert.equal(first.report.digest, second.report.digest);
  assert.equal(
    first.report.facts.find((fact) => fact.factId === 'architecture.attention.head_dim')
      .confidence,
    'derived'
  );
  assert.equal(
    first.report.facts.find((fact) => fact.factId === 'checkpoint.output_head')
      .proposal,
    'tied-token-embedding'
  );
  assert.equal(
    first.report.facts.find((fact) => fact.factId === 'source.unmapped.text_config.novel_attention_mode')
      .confidence,
    'ambiguous'
  );
  assert.equal(first.report.ok, false);
  assert.equal(first.artifacts.conversion.completeness, 'skeleton');
  assert.ok(first.artifacts.conversion.unresolvedFactIds.includes(
    'source.unmapped.text_config.novel_attention_mode'
  ));
  assert.ok(first.artifacts.conversion.unresolvedFactIds.includes('checkpoint.unmapped_tensors'));

  const cliOutputDir = path.join(sourceDir, 'cli-output');
  const cli = spawnSync(
    process.execPath,
    [
      'src/cli/doppler-cli.js',
      'onboard',
      'inspect',
      '--source',
      sourceDir,
      '--out',
      cliOutputDir,
    ],
    { cwd: repoRoot, encoding: 'utf8' }
  );
  assert.equal(cli.status, 1);
  const cliReport = JSON.parse(
    await fs.readFile(path.join(cliOutputDir, 'source-intake.json'), 'utf8')
  );
  assert.equal(cliReport.schema, 'doppler.source-intake/v1');
} finally {
  await fs.rm(sourceDir, { recursive: true, force: true });
}

console.log('source-intake.test: ok');

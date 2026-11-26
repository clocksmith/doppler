// trapset: the reference publisher's public known-answer traps for one pinned circuit (trap verification v1), signed
// with an Ed25519 publisher key that requesters pin. A public set is defeated by a perturbation that vanishes on
// every published trap; it is kept for comparison tests only. The secret fresh traps of trap-oracle.js replace it.
//
//   node tools/privacy-protected/trapset.js --key publisher.key [--new-key] [--model tiny-model.json] [--out trapset.json]
//        [--per-length 16] [--seed 1] [--tolerance 0.75] [--min-margin 2.5]

import { writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { DEFAULT_MODEL, circuitDigest, loadModel, newPlan } from './plan.js';
import { loadOrCreateSeedKey, signTrapSet } from './traps.js';

export async function main(argv = process.argv.slice(2)) {
  const { values } = parseArgs({
    args: argv,
    options: {
      model: { type: 'string', default: DEFAULT_MODEL.pathname },
      key: { type: 'string' },
      'new-key': { type: 'boolean', default: false },
      out: { type: 'string', default: 'trapset.json' },
      'per-length': { type: 'string', default: '16' },
      seed: { type: 'string', default: '1' },
      tolerance: { type: 'string', default: '0.75' },
      'min-margin': { type: 'string', default: '2.5' },
    },
  });
  if (!values.key) throw new Error('--key is required');
  const privateKey = await loadOrCreateSeedKey(values.key, values['new-key']);
  const { model, id } = loadModel(values.model);
  const pl = newPlan(model);
  const env = signTrapSet({
    pl, modelId: id, circuit: circuitDigest(pl, id), privateKey, perLength: Number(values['per-length']), seed: Number(values.seed),
    tolerance: Number(values.tolerance), minMargin: Number(values['min-margin']),
    publishedAt: new Date().toISOString().replace(/\.\d{3}Z$/, 'Z'),
  });
  writeFileSync(values.out, JSON.stringify(env, null, 1));
  const body = JSON.parse(env.body);
  console.log(`${body.traps.length} traps for circuit ${body.circuit}, publisher key ${env.publicKey}`);
  return env;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => {
    console.error(`error: ${e.message}`);
    process.exit(1);
  });
}

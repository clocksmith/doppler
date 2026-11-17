/**
 * Stand-in provider for the local run. It serves one session over stdin/stdout with a software test chip and a fixed
 * text generator, so it measures the protocol and the client only. It is not a confidential guest and runs no model.
 */
import { createPrivateKey, createPublicKey } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { createAttestedProvider } from './provider.js';
import { loadSessionPolicy } from './session.js';
import { createTestChip } from './test-support.js';
import { streamChannel } from './wire.js';

const config = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const privateKey = createPrivateKey(config.chipPrivateKeyPem);
const chip = createTestChip({ keys: { privateKey, publicKey: createPublicKey(privateKey) }, measurement: Buffer.from(config.measurement, 'hex'),
  chipId: Buffer.from(config.chipId, 'hex'), tcb: config.tcb });

async function* standIn(messages, { signal }) {
  const words = `Stand-in reply to a ${messages.at(-1).content.length}-character request.`.split(' ');
  for (let index = 0; index < config.outputPieces; index += 1) {
    signal.throwIfAborted();
    yield `${words[index % words.length]} `;
  }
}

const provider = createAttestedProvider({ modelIdentity: config.modelIdentity, attest: chip.attest(), generate: standIn, sessionPolicy: loadSessionPolicy() });
const outcome = await provider.serve(streamChannel(process.stdin, process.stdout, 4 * 1024 * 1024), new AbortController().signal);
process.stderr.write(`${outcome}\n`);
process.exit(0);

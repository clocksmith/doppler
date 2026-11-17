import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { randomBytes } from 'node:crypto';
import test from 'node:test';
import { assertExecutionBackendAvailable } from '../../src/inference/execution-backend-contract.js';
import { createAttestedClient } from './client.js';
import { SNP_FORMAT, createInstalledVerifiers, createSnpVerifier, validateTrustPolicy } from './evidence.js';
import { createAttestedProvider } from './provider.js';
import {
  CLIENT_TO_PROVIDER, PROTOCOL, PROVIDER_TO_CLIENT, bindingReportData, createOpener, createSealer, deriveSessionKeys,
  generateSessionKey, loadSessionPolicy,
} from './session.js';
import { endorseVcek, loadAmdRoots } from './snp-endorsement.js';
import { parseSnpReport } from './snp-report.js';
import { TEST_FORMAT, createTestChip } from './test-support.js';
import { createConfigfsAttester } from './attester-configfs.js';
import { memoryChannelPair } from './wire.js';

const fixture = name => readFileSync(new URL(`fixtures/${name}`, import.meta.url));
const digest = character => 'sha256:' + character.repeat(64);
const installed = createInstalledVerifiers();
const sessionPolicy = loadSessionPolicy();
const NOW = new Date('2026-11-17T00:00:00Z');
const rejects = (code, action) => assert.throws(action, error => error.code === code, `expected ${code}`);
const flip = (bytes, offset) => { const copy = Buffer.from(bytes); copy[offset] ^= 1; return copy; };

const milan = { report: fixture('milan.report.bin'), endorsement: { vcek: fixture('milan.vcek.der') } };
const debug = { report: fixture('milan-debug-policy.report.bin'), endorsement: { vcek: fixture('milan-debug-policy.vcek.der') } };
const parsed = parseSnpReport(milan.report);
const deploymentFor = (report, overrides = {}) => ({ id: 'fixture', product: 'Milan', measurement: report.measurement.toString('hex'),
  minimumTcb: { ...report.reportedTcb }, chipIds: [], hostData: null, ...overrides });
const amdVerifier = deployments => createSnpVerifier({ trustPolicy: { ...installed.trustPolicy, deployments },
  endorse: (endorsement, now) => endorseVcek(endorsement.vcek, installed.roots, now) });

test('a genuine Milan report verifies through the installed AMD chain', () => {
  const appraisal = amdVerifier([deploymentFor(parsed)])(milan, parsed.reportData, NOW);
  assert.equal(appraisal.product, 'Milan');
  assert.equal(appraisal.endorsement, 'amd-vcek-chain');
  assert.equal(appraisal.providerIsolationEstablished, true);
  assert.equal(appraisal.measurement, parsed.measurement.toString('hex'));
});

test('the installed policy admits no deployment', () => {
  assert.deepEqual(installed.trustPolicy.deployments, []);
  rejects('evidence-measurement', () => installed.verifiers[SNP_FORMAT](milan, parsed.reportData, NOW));
});

test('a genuine report recorded earlier cannot stand in for a new session', () => {
  const fresh = bindingReportData({ clientNonce: randomBytes(32), providerKey: randomBytes(32), modelIdentity: digest('a'),
    stateNamespace: digest('b'), sessionPolicyId: sessionPolicy.id });
  rejects('evidence-binding', () => amdVerifier([deploymentFor(parsed)])(milan, fresh, NOW));
});

test('genuine evidence is still refused when the guest, firmware, recipient or dates are wrong', () => {
  const debugReport = parseSnpReport(debug.report);
  assert.equal(debugReport.policy.debugAllowed, true);
  rejects('evidence-guest-policy', () => amdVerifier([deploymentFor(debugReport)])(debug, debugReport.reportData, NOW));
  const newer = { ...parsed.reportedTcb, snp: parsed.reportedTcb.snp + 1 };
  rejects('evidence-tcb', () => amdVerifier([deploymentFor(parsed, { minimumTcb: newer })])(milan, parsed.reportData, NOW));
  rejects('evidence-recipient', () => amdVerifier([deploymentFor(parsed, { chipIds: ['00'.repeat(64)] })])(milan, parsed.reportData, NOW));
  rejects('evidence-recipient', () => amdVerifier([deploymentFor(parsed, { hostData: '11'.repeat(32) })])(milan, parsed.reportData, NOW));
  rejects('evidence-measurement', () => amdVerifier([deploymentFor(parsed, { measurement: '00'.repeat(48) })])(milan, parsed.reportData, NOW));
  rejects('endorsement-validity', () => amdVerifier([deploymentFor(parsed)])(milan, parsed.reportData, new Date('2031-01-01T00:00:00Z')));
});

test('altered reports, swapped chips and altered certificates fail', () => {
  const verifier = amdVerifier([deploymentFor(parsed)]);
  for (const offset of [0x08, 0x50, 0x90, 0x180, 0x1a0, 0x29f]) {
    rejects('evidence-signature', () => verifier({ ...milan, report: flip(milan.report, offset) }, parsed.reportData, NOW));
  }
  rejects('evidence-signature', () => verifier({ ...milan, report: flip(milan.report, 0x2a0) }, parsed.reportData, NOW));
  rejects('evidence-signature', () => verifier({ report: milan.report, endorsement: debug.endorsement }, parsed.reportData, NOW));
  rejects('endorsement-chain', () => verifier({ ...milan, endorsement: { vcek: flip(milan.endorsement.vcek, 700) } }, parsed.reportData, NOW));
  rejects('evidence-format', () => verifier({ ...milan, report: milan.report.subarray(1) }, parsed.reportData, NOW));
  rejects('endorsement-format', () => verifier({ ...milan, endorsement: {} }, parsed.reportData, NOW));
});

test('root pins and policy floors are enforced', () => {
  const products = structuredClone(installed.trustPolicy.formats[SNP_FORMAT].products);
  products.Milan.arkSpkiSha256 = '0'.repeat(64);
  rejects('endorsement-roots', () => loadAmdRoots(products, name => readFileSync(new URL(name, import.meta.url), 'utf8')));
  const relaxed = structuredClone(installed.trustPolicy);
  relaxed.formats[SNP_FORMAT].allowDebug = true;
  rejects('trust-policy', () => validateTrustPolicy(relaxed));
});

test('frames authenticate order, direction, the final marker and the transcript', () => {
  const client = generateSessionKey();
  const provider = generateSessionKey();
  const shared = { reportData: randomBytes(64), reportDigest: randomBytes(32), clientKey: client.publicBytes };
  const a = deriveSessionKeys({ privateKey: client.privateKey, peerKey: provider.publicBytes, ...shared });
  const b = deriveSessionKeys({ privateKey: provider.privateKey, peerKey: client.publicBytes, ...shared });
  assert.deepEqual(a, b);
  rejects('session-key', () => deriveSessionKeys({ privateKey: client.privateKey, peerKey: Buffer.alloc(32), ...shared }));
  const seal = createSealer(a.response, a.transcript, PROVIDER_TO_CLIENT, 256);
  const frames = [seal(Buffer.from('one'), false), seal(Buffer.from('two'), false), seal(Buffer.from('{}'), true)];
  const opener = () => createOpener(b.response, b.transcript, PROVIDER_TO_CLIENT, 2048);
  const open = opener();
  assert.deepEqual(frames.map(frame => open(frame).plaintext.toString()), ['one', 'two', '{}']);
  rejects('session-frame', () => open(frames[2]));
  rejects('session-frame', () => seal(Buffer.from('late'), false));
  rejects('session-frame', () => opener()(frames[1]));
  rejects('session-frame', () => { const next = opener(); next(frames[0]); next(frames[0]); });
  rejects('session-frame', () => opener()(Buffer.concat([Buffer.from([1]), frames[0].subarray(1)])));
  rejects('session-frame', () => opener()(flip(frames[0], 20)));
  rejects('session-frame', () => createOpener(b.response, b.transcript, CLIENT_TO_PROVIDER, 2048)(frames[0]));
  rejects('session-frame', () => createOpener(b.request, b.transcript, PROVIDER_TO_CLIENT, 2048)(frames[0]));
  rejects('session-frame', () => createOpener(b.response, randomBytes(32), PROVIDER_TO_CLIENT, 2048)(frames[0]));
  assert.equal(frames[0].length, frames[1].length);
  const base = { clientNonce: randomBytes(32), providerKey: randomBytes(32), modelIdentity: digest('a'), stateNamespace: digest('b'), sessionPolicyId: digest('c') };
  const reference = bindingReportData(base);
  for (const change of [{ clientNonce: randomBytes(32) }, { providerKey: randomBytes(32) }, { modelIdentity: digest('d') },
    { stateNamespace: digest('d') }, { sessionPolicyId: digest('d') }]) {
    assert.notDeepEqual(bindingReportData({ ...base, ...change }), reference);
  }
});

const MODEL = digest('a');
const descriptor = { schema: 'doppler.execution-backend/v1', id: 'doppler-attested-research/v1', version: 1, kind: 'attested',
  modelIdentities: [MODEL], operations: ['generate'], protocol: PROTOCOL, inputRepresentation: 'opaque', outputRepresentation: 'opaque' };
const binding = { schema: 'doppler.execution-state/v1', modelIdentity: MODEL, adapterIdentities: [], backendId: descriptor.id, policyId: digest('b'),
  routeId: digest('c'), tenantId: 'tenant', sessionId: 'thread', keyContext: null };
const PROMPT = 'The secret launch code is tangerine-41.';
const ANSWER = ['Understood: ', 'tangerine-41', ' is noted.'];
const messages = [{ role: 'user', content: PROMPT }];

/** One client and one provider joined by an in-process channel the test can read and rewrite. */
function route({ chip = createTestChip(), deployment = {}, attest = {}, intercept, generate, model = MODEL, verifiers } = {}) {
  const wire = [];
  const [clientEnd, providerEnd] = memoryChannelPair({ tap: (direction, text) => wire.push({ direction, text }), intercept });
  const provider = createAttestedProvider({ modelIdentity: model, sessionPolicy, attest: chip.attest(attest),
    generate: generate ?? (async function* echo() { for (const part of ANSWER) yield part; }) });
  const served = provider.serve(providerEnd, new AbortController().signal);
  const client = createAttestedClient({ descriptor, sessionPolicy, now: () => NOW, connect: async () => clientEnd,
    verifiers: verifiers ?? { [TEST_FORMAT]: createSnpVerifier({ trustPolicy: { ...installed.trustPolicy, deployments: [chip.deployment(deployment)] }, endorse: chip.endorse }) } });
  return { client, served, wire, chip };
}
const types = wire => wire.map(entry => JSON.parse(entry.text).type);
const signal = () => new AbortController().signal;
const fails = (code, promise) => assert.rejects(promise, error => error.code === code, `expected ${code}`);

test('a complete attempt returns the provider text and keeps both texts off the wire', async () => {
  const { client, served, wire } = route();
  const deltas = [];
  const { text, appraisal } = await client.run(binding, messages, signal(), delta => deltas.push(delta));
  assert.equal(text, ANSWER.join(''));
  assert.deepEqual(deltas, ANSWER);
  assert.equal(await served, 'completed');
  assert.equal(appraisal.endorsement, 'software-test');
  assert.equal(appraisal.providerIsolationEstablished, false);
  assert.deepEqual(types(wire), ['hello', 'evidence', 'request', 'frame', 'frame', 'frame', 'frame']);
  const visible = wire.map(entry => entry.text).join('\n');
  for (const secret of [PROMPT, 'tangerine', ...ANSWER, Buffer.from(PROMPT).toString('base64').slice(0, 24)]) assert(!visible.includes(secret));
});

test('the six contract operations compose and refuse to run out of order', async () => {
  const { client, served } = route();
  assert.equal(client.describe().kind, 'attested');
  assert.throws(() => assertExecutionBackendAvailable(descriptor, MODEL), /unavailable/);
  const state = await client.prepareClient(binding, signal());
  await fails('session-order', client.sealInput(state, messages, signal()));
  await client.verifyEvidence(state, await client.requestEvidence(state, signal()), signal());
  const input = await client.sealInput(state, messages, signal());
  await fails('session-order', client.execute(input, { ...binding }, signal()));
  const result = await client.execute(input, binding, signal());
  assert.equal(await client.finalizeClient(state, result, signal()), ANSWER.join(''));
  await client.dispose(state);
  assert.equal(await served, 'completed');
  await fails('session-binding', client.prepareClient({ ...binding, modelIdentity: digest('f') }, signal()));
});

test('request length is hidden inside a padding bucket', async () => {
  const lengths = [];
  for (const content of ['hi', 'x'.repeat(3000)]) {
    const { client, wire } = route();
    await client.run(binding, [{ role: 'user', content }], signal());
    lengths.push(wire.find(entry => JSON.parse(entry.text).type === 'request').text.length);
  }
  assert.equal(lengths[0], lengths[1]);
});

test('wrong provider evidence stops the attempt before any request is sent', async () => {
  const other = createTestChip();
  const cases = {
    'evidence-measurement': { deployment: { measurement: '00'.repeat(48) } },
    'evidence-guest-policy': { attest: { policy: 0xb0000n } },
    'evidence-tcb': { deployment: { minimumTcb: { bootloader: 4, tee: 0, snp: 23, microcode: 213 } } },
    'evidence-recipient': { deployment: { chipIds: ['00'.repeat(64)] } },
    'evidence-signer': { attest: { signer: 4 } },
    'evidence-signature': { intercept: (direction, message) => (message.type === 'evidence' ? { ...message, report: flip(message.report, 0x90) } : message) },
    'evidence-chip': { attest: { chipId: other.chipId } },
    'evidence-format': { verifiers: {} },
  };
  for (const [code, options] of Object.entries(cases)) {
    const { client, wire } = route(options);
    await fails(code, client.run(binding, messages, signal()));
    assert(!types(wire).includes('request'), code);
  }
});

test('a host that substitutes the key, the model or an old report is caught by the binding', async () => {
  const attacker = generateSessionKey();
  const swapKey = (direction, message) => (message.type === 'evidence' ? { ...message, providerKey: attacker.publicBytes } : message);
  let recorded = null;
  const first = route({ intercept: (direction, message) => { if (message.type === 'evidence') recorded = message; return message; } });
  await first.client.run(binding, messages, signal());
  const replay = (direction, message) => (message.type === 'evidence' ? recorded : message);
  const otherModel = (direction, message) => (message.type === 'hello' ? { ...message, modelIdentity: digest('e') } : message);
  for (const options of [{ intercept: swapKey }, { intercept: replay, chip: first.chip }, { intercept: otherModel, model: digest('e') }]) {
    const { client, wire } = route(options);
    await fails('evidence-binding', client.run(binding, messages, signal()));
    assert(!types(wire).includes('request'));
  }
  const refused = route({ model: digest('e') });
  await fails('provider-refused', refused.client.run(binding, messages, signal()));
  assert.equal(await refused.served, 'refused');
});

test('altered, dropped, repeated or missing-final response frames release no text', async () => {
  const frames = n => { let seen = 0; return message => message.type === 'frame' && ++seen === n; };
  const second = frames(2);
  const tamper = (direction, message) => (second(message) ? { ...message, frame: flip(message.frame, 30) } : message);
  const first = frames(1);
  const drop = (direction, message) => (first(message) ? null : message);
  let last = null;
  const repeat = (direction, message) => { if (message.type !== 'frame') return message; const out = last ?? message; last = message; return out; };
  const early = frames(1);
  const forgeFinal = (direction, message) => (early(message) ? { ...message, frame: Buffer.concat([Buffer.from([1]), message.frame.subarray(1)]) } : message);
  for (const intercept of [tamper, drop, repeat, forgeFinal]) {
    const delivered = [];
    const { client } = route({ intercept });
    await fails('session-frame', client.run(binding, messages, signal(), delta => delivered.push(delta)));
    assert(delivered.join('').length < ANSWER.join('').length);
  }
  const dropFinal = (direction, message) => (message.type === 'frame' && message.frame[0] === 1 ? null : message);
  const truncated = route({ intercept: dropFinal });
  await fails('session-truncated', truncated.client.run(binding, messages, signal()));
});

test('the provider rejects an altered request and a request replayed into another session', async () => {
  const tamper = (direction, message) => (message.type === 'request' ? { ...message, frame: flip(message.frame, 40) } : message);
  let ran = false;
  const generate = async function* guarded() { ran = true; yield 'never'; };
  const altered = route({ intercept: tamper, generate });
  await fails('provider-failed', altered.client.run(binding, messages, signal()));
  assert.equal(await altered.served, 'refused');
  let recorded = null;
  const first = route({ intercept: (direction, message) => { if (message.type === 'request') recorded = message; return message; } });
  await first.client.run(binding, messages, signal());
  const replayed = route({ intercept: (direction, message) => (message.type === 'request' ? recorded : message), generate, chip: first.chip });
  await fails('provider-failed', replayed.client.run(binding, messages, signal()));
  assert.equal(await replayed.served, 'refused');
  assert.equal(ran, false);
});

test('cancellation stops the provider and releases nothing', async () => {
  const controller = new AbortController();
  let observed = false;
  const generate = async function* slow(_messages, { signal: stop }) {
    yield 'first ';
    await new Promise(resolve => stop.addEventListener('abort', () => { observed = true; resolve(); }, { once: true }));
    yield 'never';
  };
  const { client, served } = route({ generate });
  const attempt = client.run(binding, messages, controller.signal, () => controller.abort());
  await assert.rejects(attempt, error => error.name === 'AbortError');
  assert.equal(await served, 'cancelled');
  assert.equal(observed, true);
});

test('a generation failure inside the provider is reported without text', async () => {
  const generate = async function* broken() { yield 'partial'; throw new Error(PROMPT); };
  const { client, served, wire } = route({ generate });
  await fails('provider-failed', client.run(binding, messages, signal()));
  assert.equal(await served, 'failed');
  assert(!wire.map(entry => entry.text).join('\n').includes('tangerine'));
});

test('the guest attester asks the kernel for a report over exactly the session binding', async () => {
  const chip = createTestChip();
  const vcek = fixture('milan.vcek.der');
  const table = Buffer.alloc(48);
  Buffer.from('8d75da6364e66445adc5f4b93be8accd', 'hex').copy(table, 0);
  table.writeUInt32LE(48, 16);
  table.writeUInt32LE(vcek.length, 20);
  const files = new Map();
  const removed = [];
  const fs = {
    mkdir: async path => { files.set(`${path}/provider`, Buffer.from('sev_guest\n')); files.set(`${path}/generation`, Buffer.from('0\n')); },
    writeFile: async (path, bytes) => {
      const entry = path.slice(0, path.lastIndexOf('/'));
      files.set(`${entry}/outblob`, (await chip.attest()(Buffer.from(bytes))).report);
      files.set(`${entry}/auxblob`, Buffer.concat([table, vcek]));
      files.set(`${entry}/generation`, Buffer.from('1\n'));
    },
    readFile: async path => files.get(path),
    rmdir: async path => { removed.push(path); },
  };
  const reportData = randomBytes(64);
  const attestation = await createConfigfsAttester({ fs, root: '/tsm' })(reportData);
  assert.equal(attestation.format, SNP_FORMAT);
  assert.deepEqual(parseSnpReport(attestation.report).reportData, reportData);
  assert.deepEqual(attestation.endorsement.vcek, vcek);
  assert.equal(removed.length, 1);
  fs.writeFile = async path => { files.set(`${path.slice(0, path.lastIndexOf('/'))}/generation`, Buffer.from('2\n')); };
  await assert.rejects(createConfigfsAttester({ fs, root: '/tsm' })(reportData), error => error.code === 'attester-race');
  assert.equal(removed.length, 2);
});

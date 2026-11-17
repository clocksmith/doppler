/**
 * Local two-process run: this process is the client, a child process is the stand-in provider. It records what the
 * client actually has to do. The provider's evidence is software test evidence, so no isolation is established.
 */
import { spawn } from 'node:child_process';
import { generateKeyPairSync, randomBytes } from 'node:crypto';
import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createAttestedClient } from './client.js';
import { createInstalledVerifiers, createSnpVerifier } from './evidence.js';
import { PROTOCOL, loadSessionPolicy } from './session.js';
import { endorseVcek } from './snp-endorsement.js';
import { parseSnpReport } from './snp-report.js';
import { TEST_FORMAT, createTestChip } from './test-support.js';
import { streamChannel } from './wire.js';

const args = Object.fromEntries(process.argv.slice(2).map(item => item.replace(/^--/, '').split('=')));
const outputPieces = Number(args['output-pieces'] ?? 256);
const promptCharacters = Number(args['prompt-characters'] ?? 2000);
const digest = character => 'sha256:' + character.repeat(64);
const cpuMs = usage => (usage.user + usage.system) / 1000;
const median = values => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

const installed = createInstalledVerifiers();
const sessionPolicy = loadSessionPolicy();
const keys = generateKeyPairSync('ec', { namedCurve: 'P-384' });
const chip = createTestChip({ keys });
const MODEL = digest('a');
const directory = mkdtempSync(join(tmpdir(), 'doppler-attested-'));
const configPath = join(directory, 'provider.json');
writeFileSync(configPath, JSON.stringify({ chipPrivateKeyPem: keys.privateKey.export({ type: 'pkcs8', format: 'pem' }),
  measurement: chip.measurement.toString('hex'), chipId: chip.chipId.toString('hex'), tcb: chip.tcb, modelIdentity: MODEL, outputPieces }), { mode: 0o600 });
chmodSync(configPath, 0o600);

const descriptor = { schema: 'doppler.execution-backend/v1', id: 'doppler-attested-research/v1', version: 1, kind: 'attested',
  modelIdentities: [MODEL], operations: ['generate'], protocol: PROTOCOL, inputRepresentation: 'opaque', outputRepresentation: 'opaque' };
const binding = { schema: 'doppler.execution-state/v1', modelIdentity: MODEL, adapterIdentities: [], backendId: descriptor.id, policyId: digest('b'),
  routeId: digest('c'), tenantId: 'local-run', sessionId: randomBytes(8).toString('hex'), keyContext: null };

let channel;
let child;
const client = createAttestedClient({ descriptor, sessionPolicy,
  verifiers: { [TEST_FORMAT]: createSnpVerifier({ trustPolicy: { ...installed.trustPolicy, deployments: [chip.deployment()] }, endorse: chip.endorse }) },
  connect: async () => {
    child = spawn(process.execPath, [fileURLToPath(new URL('provider-process.js', import.meta.url)), configPath], { stdio: ['pipe', 'pipe', 'pipe'] });
    channel = streamChannel(child.stdout, child.stdin, 4 * 1024 * 1024);
    return channel;
  } });

const signal = new AbortController().signal;
const phases = {};
const timed = async (name, action) => {
  const cpu = process.cpuUsage();
  const wall = performance.now();
  const result = await action();
  phases[name] = { clientCpuMs: cpuMs(process.cpuUsage(cpu)), wallMs: performance.now() - wall };
  return result;
};
let text;
let state;
try {
  const started = process.cpuUsage();
  const wallStarted = performance.now();
  state = await timed('openChannelAndHello', () => client.prepareClient(binding, signal));
  const evidence = await timed('waitForEvidence', () => client.requestEvidence(state, signal));
  await timed('verifyEvidence', () => client.verifyEvidence(state, evidence, signal));
  const input = await timed('sealRequest', () => client.sealInput(state, [{ role: 'user', content: 'p'.repeat(promptCharacters) }], signal));
  const result = await timed('waitForResponse', () => client.execute(input, binding, signal));
  text = await timed('openResponse', () => client.finalizeClient(state, result, signal));
  phases.total = { clientCpuMs: cpuMs(process.cpuUsage(started)), wallMs: performance.now() - wallStarted };
  phases.responseFrames = result.frames.length;
} finally {
  if (state) await client.dispose(state);
  rmSync(directory, { recursive: true, force: true });
}
const providerOutcome = await new Promise(resolve => { let seen = ''; child.stderr.on('data', chunk => { seen += chunk; }); child.on('close', () => resolve(seen.trim())); });

// The genuine-evidence check, timed on a real Milan report: one certificate signature and one P-384 report signature.
const fixture = name => readFileSync(new URL(`fixtures/${name}`, import.meta.url));
const real = { report: fixture('milan.report.bin'), endorsement: { vcek: fixture('milan.vcek.der') } };
const parsed = parseSnpReport(real.report);
const genuine = createSnpVerifier({ trustPolicy: { ...installed.trustPolicy, deployments: [{ id: 'fixture', product: 'Milan',
  measurement: parsed.measurement.toString('hex'), minimumTcb: { ...parsed.reportedTcb }, chipIds: [], hostData: null }] },
  endorse: (endorsement, now) => endorseVcek(endorsement.vcek, installed.roots, now) });
const samples = [];
for (let index = 0; index < 51; index += 1) {
  const cpu = process.cpuUsage();
  genuine(real, parsed.reportData, new Date('2026-11-17T00:00:00Z'));
  samples.push(cpuMs(process.cpuUsage(cpu)));
}

const report = {
  schema: 'doppler.attested-local-run/v1',
  protocol: PROTOCOL,
  sessionPolicyId: sessionPolicy.id,
  completed: providerOutcome === 'completed' && text.length > 0,
  evidenceFormat: TEST_FORMAT,
  providerIsolationEstablished: false,
  hardwareEvidence: false,
  modelExecuted: false,
  securityQualified: false,
  productionBackendAvailable: false,
  host: 'mac-host',
  node: process.version,
  request: { promptCharacters, outputPieces, outputCharacters: text.length },
  clientPhases: phases,
  wireBytes: { clientToProvider: channel.bytes.sent, providerToClient: channel.bytes.received },
  clientPeakRssBytes: process.resourceUsage().maxRSS * 1024,
  clientHoldsWeights: false,
  clientModelOperations: 0,
  genuineEvidenceCheck: { fixture: 'fixtures/milan.report.bin', samples: samples.length, medianClientCpuMs: median(samples), maximumClientCpuMs: Math.max(...samples) },
  scope: 'One local run, client and stand-in provider as two processes on one host. Software test evidence and a fixed text generator: this measures client work and wire overhead only.',
};
if (args.output) writeFileSync(args.output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
if (!report.completed) process.exit(1);

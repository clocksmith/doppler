// Server side of the switches that guard publication. The browser's flags file
// can only hide a button; this is what refuses the request. The server reads
// its own switches file, which is never served to a browser, and re-judges any
// submitted result against its own copy of the protocol instead of trusting
// the verdict the page computed.

import { readFile } from 'node:fs/promises';
import { judgeRun, validateProtocol } from '../public/protocol.js';

const SWITCH_SCHEMA = 'doppler.observatory.server-switches/v1';
const EVIDENCE_SCHEMA = 'doppler.observatory.device-benchmark/v1';
export const MAX_RESULT_BYTES = 512 * 1024;

export function validateSwitches(config) {
  if (config?.schema !== SWITCH_SCHEMA || !config.switches || typeof config.switches !== 'object') {
    throw new Error(`Server switches file must use schema ${SWITCH_SCHEMA}.`);
  }
  for (const [name, value] of Object.entries(config.switches)) {
    if (typeof value !== 'boolean') throw new Error(`Server switch ${name} must be true or false.`);
  }
  return config.switches;
}

export async function readSwitches(file) {
  return validateSwitches(JSON.parse(await readFile(file, 'utf8')));
}

const refusal = (status, code, message, extra = {}) => ({ status, body: { ok: false, code, message, ...extra } });

// A switch that is missing is off: an unknown action is never allowed.
export function gate(switches, name) {
  return switches?.[name] === true
    ? null
    : refusal(403, `${name.replace(/\W+/g, '_')}_disabled`, `"${name}" is switched off on this server.`);
}

// Decides one result submission. `store` is where an accepted result goes;
// with none configured the server says so rather than pretending to keep it.
export async function submitResult({ switches, protocol, store, evidence }) {
  const closed = gate(switches, 'results.upload');
  if (closed) return closed;
  validateProtocol(protocol);
  if (evidence?.schema !== EVIDENCE_SCHEMA) return refusal(400, 'unknown_result_schema', `A result must use schema ${EVIDENCE_SCHEMA}.`);
  if (evidence.protocol?.id !== protocol.id || evidence.protocol?.version !== protocol.version) {
    return refusal(409, 'protocol_version_mismatch', `This server accepts ${protocol.id} ${protocol.version} only.`);
  }
  let verdict;
  try { verdict = judgeRun(protocol, evidence); } catch (error) { return refusal(400, 'malformed_result', String(error.message)); }
  if (!verdict.eligible) return refusal(422, 'result_not_eligible', 'The result does not meet the protocol.', { reasons: verdict.reasons });
  if (!store) return refusal(503, 'no_result_store', 'Uploads are switched on but this server has nowhere to keep results.');
  const id = await store.append({ receivedAt: new Date().toISOString(), verdict, evidence });
  return { status: 201, body: { ok: true, id } };
}

async function readJson(request, limit) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > limit) return { tooLarge: true };
    chunks.push(chunk);
  }
  try { return { value: JSON.parse(Buffer.concat(chunks).toString('utf8')) }; } catch { return { invalid: true }; }
}

// Returns true when the request was an API request and has been answered.
export async function handleApi(request, response, { switchesFile, protocolFile, store = null }) {
  const { pathname } = new URL(request.url, 'http://localhost');
  if (!pathname.startsWith('/api/')) return false;
  const send = ({ status, body }) => { response.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' }).end(JSON.stringify(body)); return true; };
  // Read on every request so flipping the file takes effect without a restart.
  let switches;
  try { switches = await readSwitches(switchesFile); } catch { return send(refusal(503, 'switches_unavailable', 'The server switches file could not be read, so every guarded action is refused.')); }
  if (pathname === '/api/switches' && request.method === 'GET') return send({ status: 200, body: { ok: true, switches } });
  if (pathname === '/api/results' && request.method === 'POST') {
    const closed = gate(switches, 'results.upload');
    if (closed) return send(closed);
    const body = await readJson(request, MAX_RESULT_BYTES);
    if (body.tooLarge) return send(refusal(413, 'result_too_large', `A result must be under ${MAX_RESULT_BYTES} bytes.`));
    if (body.invalid) return send(refusal(400, 'malformed_result', 'The result is not valid JSON.'));
    const protocol = JSON.parse(await readFile(protocolFile, 'utf8'));
    return send(await submitResult({ switches, protocol, store, evidence: body.value }));
  }
  return send(refusal(404, 'unknown_endpoint', 'No such endpoint.'));
}

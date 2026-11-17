/**
 * Light client for `doppler-attested-session/v1`. Its whole job is: check the provider's hardware evidence, encrypt the
 * request to the key that evidence names, and decrypt the answer. It holds no weights and runs no model operation.
 */
import { createHash, randomBytes } from 'node:crypto';
import { executionStateNamespace, validateExecutionBackend } from '../../src/inference/execution-backend-contract.js';
import {
  CLIENT_TO_PROVIDER, PROTOCOL, PROVIDER_TO_CLIENT, bindingReportData, createOpener, createSealer, deriveSessionKeys,
  generateSessionKey, sessionError,
} from './session.js';
import { evidenceError } from './snp-report.js';

function abortError(signal) {
  return signal.reason instanceof Error ? signal.reason : Object.assign(new Error('Aborted.'), { name: 'AbortError' });
}

export function createAttestedClient({ descriptor, verifiers, sessionPolicy, connect, now = () => new Date() }) {
  const backend = validateExecutionBackend(descriptor);
  if (backend.kind !== 'attested' || backend.protocol !== PROTOCOL) throw sessionError('session-binding', 'Descriptor is not this attested protocol.');
  const sealed = new WeakMap();

  const require = (state, phase) => {
    if (state.phase !== phase) throw sessionError('session-order', `Expected a ${phase} session, found ${state.phase}.`);
  };

  /** Opens the channel and sends the public hello. Nothing private exists yet. */
  async function prepareClient(binding, signal) {
    signal.throwIfAborted();
    if (binding.backendId !== backend.id || !backend.modelIdentities.includes(binding.modelIdentity)) {
      throw sessionError('session-binding', 'Backend does not serve this exact model.');
    }
    const state = {
      phase: 'prepared', binding, stateNamespace: executionStateNamespace(binding), clientNonce: randomBytes(32),
      channel: await connect(signal), keys: null, open: null, appraisal: null,
    };
    state.channel.send({ type: 'hello', protocol: PROTOCOL, clientNonce: state.clientNonce, modelIdentity: binding.modelIdentity,
      stateNamespace: state.stateNamespace, sessionPolicyId: sessionPolicy.id });
    return state;
  }

  async function requestEvidence(state, signal) {
    require(state, 'prepared');
    const message = await state.channel.receive(signal);
    if (message === null) throw sessionError('wire-closed', 'Provider closed before sending evidence.');
    if (message.type === 'error') throw sessionError('provider-refused', `Provider refused: ${String(message.code).slice(0, 64)}.`);
    if (message.type !== 'evidence' || message.protocol !== PROTOCOL || typeof message.format !== 'string'
      || !Buffer.isBuffer(message.providerKey) || !Buffer.isBuffer(message.report)) throw sessionError('wire-format', 'Malformed evidence message.');
    return { format: message.format, providerKey: message.providerKey, report: message.report, endorsement: message.endorsement ?? {} };
  }

  /** Every identity is checked here, before any request byte is encoded: model, key, recipient, attempt and session policy. */
  async function verifyEvidence(state, evidence, signal) {
    signal.throwIfAborted();
    require(state, 'prepared');
    const verifier = Object.hasOwn(verifiers, evidence.format) ? verifiers[evidence.format] : null;
    if (verifier === null) throw evidenceError('evidence-format', 'Evidence format is not installed.');
    const reportData = bindingReportData({ clientNonce: state.clientNonce, providerKey: evidence.providerKey,
      modelIdentity: state.binding.modelIdentity, stateNamespace: state.stateNamespace, sessionPolicyId: sessionPolicy.id });
    state.appraisal = verifier(evidence, reportData, now());
    state.providerKey = Buffer.from(evidence.providerKey);
    state.reportData = reportData;
    state.reportDigest = createHash('sha256').update(evidence.report).digest();
    state.phase = 'verified';
  }

  async function sealInput(state, messages, signal) {
    signal.throwIfAborted();
    require(state, 'verified');
    const plaintext = Buffer.from(JSON.stringify({ messages: messages.map(({ role, content }) => ({ role, content })) }), 'utf8');
    if (plaintext.length > sessionPolicy.maximumRequestBytes) throw sessionError('session-frame', 'Request exceeds the session limit.');
    const key = generateSessionKey();
    state.keys = deriveSessionKeys({ privateKey: key.privateKey, peerKey: state.providerKey, reportData: state.reportData,
      reportDigest: state.reportDigest, clientKey: key.publicBytes });
    const frame = createSealer(state.keys.request, state.keys.transcript, CLIENT_TO_PROVIDER, sessionPolicy.requestPaddingBytes)(plaintext, true);
    plaintext.fill(0);
    state.open = createOpener(state.keys.response, state.keys.transcript, PROVIDER_TO_CLIENT, sessionPolicy.maximumResponseFrameBytes);
    state.phase = 'sealed';
    const input = Object.freeze({ clientKey: key.publicBytes, frame });
    sealed.set(input, state);
    return input;
  }

  async function* frames(state, input, signal) {
    require(state, 'sealed');
    state.phase = 'executing';
    state.channel.send({ type: 'request', clientKey: input.clientKey, frame: input.frame });
    for (let count = 0; ; count += 1) {
      let message;
      try {
        message = await state.channel.receive(signal);
      } catch (error) {
        try { state.channel.send({ type: 'cancel' }); } catch { /* channel already gone */ }
        throw signal.aborted ? abortError(signal) : error;
      }
      if (message === null) throw sessionError('session-truncated', 'Provider closed before the final frame.');
      if (message.type === 'error') throw sessionError('provider-failed', `Provider failed: ${String(message.code).slice(0, 64)}.`);
      if (message.type !== 'frame' || !Buffer.isBuffer(message.frame)) throw sessionError('wire-format', 'Malformed response message.');
      if (count >= sessionPolicy.maximumResponseFrames) throw sessionError('session-frame', 'Response exceeds the frame limit.');
      yield message.frame;
      if (message.frame[0] === 1) return;
    }
  }

  /** Transport only: sends the sealed request and collects opaque response frames. */
  async function execute(input, binding, signal) {
    const state = sealed.get(input);
    if (state === undefined || state.binding !== binding) throw sessionError('session-order', 'Sealed input does not belong to this binding.');
    const collected = [];
    for await (const frame of frames(state, input, signal)) collected.push(frame);
    return Object.freeze({ frames: collected });
  }

  /** Opens frames in order. Returns the complete text at the authenticated final frame, and null before it. */
  function collector(state, onDelta) {
    const decoder = new TextDecoder('utf-8', { fatal: true });
    let text = '';
    return frame => {
      const { plaintext, final } = state.open(frame);
      if (!final) {
        const delta = decoder.decode(plaintext, { stream: true });
        text += delta;
        if (delta.length > 0) onDelta?.(delta);
        return null;
      }
      let status = null;
      try { status = JSON.parse(plaintext.toString('utf8')); } catch { /* reported below */ }
      if (status?.status !== 'completed') throw sessionError('provider-failed', 'Provider reported a failed generation.');
      return text + decoder.decode();
    };
  }

  /** Releases text only when the authenticated final frame is present. */
  async function finalizeClient(state, result, signal) {
    signal.throwIfAborted();
    require(state, 'executing');
    const push = collector(state);
    for (const frame of result.frames) {
      const text = push(frame);
      if (text !== null) { state.phase = 'finalized'; return text; }
    }
    throw sessionError('session-truncated', 'Response ended without an authenticated final frame.');
  }

  async function dispose(state) {
    for (const key of [state.keys?.request, state.keys?.response, state.keys?.transcript, state.clientNonce]) key?.fill(0);
    state.keys = null;
    state.open = null;
    state.phase = 'disposed';
    state.channel.close();
  }

  /**
   * One complete attempt. `onDelta` receives authenticated text as it arrives; a truncated stream is only detected at
   * the end, so a caller that must not act on partial output should use the returned text alone.
   */
  async function run(binding, messages, signal, onDelta) {
    const state = await prepareClient(binding, signal);
    try {
      await verifyEvidence(state, await requestEvidence(state, signal), signal);
      const input = await sealInput(state, messages, signal);
      const push = collector(state, onDelta);
      for await (const frame of frames(state, input, signal)) {
        const text = push(frame);
        if (text !== null) { state.phase = 'finalized'; return { text, appraisal: state.appraisal }; }
      }
      throw sessionError('session-truncated', 'Response ended without an authenticated final frame.');
    } finally {
      await dispose(state);
    }
  }

  return Object.freeze({ describe: () => backend, prepareClient, requestEvidence, verifyEvidence, sealInput, execute, finalizeClient, dispose, run });
}

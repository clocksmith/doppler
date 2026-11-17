/**
 * Provider side of `doppler-attested-session/v1`. All of this runs inside the measured confidential guest; the host
 * and network outside it only relay the channel's bytes.
 */
import { createHash } from 'node:crypto';
import {
  CLIENT_TO_PROVIDER, PROTOCOL, PROVIDER_TO_CLIENT, bindingReportData, createOpener, createSealer, deriveSessionKeys,
  generateSessionKey, sessionError,
} from './session.js';

const HASH = /^sha256:[a-f0-9]{64}$/;

function messagesFrom(plaintext) {
  const body = JSON.parse(plaintext.toString('utf8'));
  if (!Array.isArray(body?.messages) || body.messages.length === 0
    || !body.messages.every(item => typeof item?.role === 'string' && typeof item?.content === 'string')) {
    throw sessionError('request-format', 'Request is not a message list.');
  }
  return body.messages;
}

/**
 * `attest(reportData)` asks the hardware for a report over exactly those 64 bytes. `generate(messages, { signal })` is
 * the ordinary plaintext runtime, yielding text. Error codes sent to the client are fixed strings, never request text.
 */
export function createAttestedProvider({ modelIdentity, attest, generate, sessionPolicy }) {
  async function serve(channel, signal) {
    const refuse = code => { try { channel.send({ type: 'error', code }); } catch { /* peer gone */ } channel.close(); };
    const hello = await channel.receive(signal);
    if (hello === null) return 'closed';
    if (hello.type !== 'hello' || hello.protocol !== PROTOCOL || !Buffer.isBuffer(hello.clientNonce) || hello.clientNonce.length !== 32
      || !HASH.test(hello.stateNamespace)) { refuse('hello-format'); return 'refused'; }
    if (hello.modelIdentity !== modelIdentity) { refuse('model-unavailable'); return 'refused'; }
    if (hello.sessionPolicyId !== sessionPolicy.id) { refuse('session-policy'); return 'refused'; }

    const key = generateSessionKey();
    const reportData = bindingReportData({ clientNonce: hello.clientNonce, providerKey: key.publicBytes, modelIdentity,
      stateNamespace: hello.stateNamespace, sessionPolicyId: sessionPolicy.id });
    const attestation = await attest(reportData);
    channel.send({ type: 'evidence', protocol: PROTOCOL, format: attestation.format, providerKey: key.publicBytes,
      report: attestation.report, endorsement: attestation.endorsement });

    const request = await channel.receive(signal);
    if (request === null || request.type === 'cancel') { channel.close(); return 'closed'; }
    if (request.type !== 'request' || !Buffer.isBuffer(request.clientKey) || !Buffer.isBuffer(request.frame)) { refuse('request-format'); return 'refused'; }

    let keys = null;
    const stop = new AbortController();
    const onAbort = () => stop.abort(signal.reason);
    signal?.addEventListener('abort', onAbort, { once: true });
    try {
      let messages;
      try {
        keys = deriveSessionKeys({ privateKey: key.privateKey, peerKey: request.clientKey, reportData,
          reportDigest: createHash('sha256').update(attestation.report).digest(), clientKey: request.clientKey });
        const open = createOpener(keys.request, keys.transcript, CLIENT_TO_PROVIDER,
          sessionPolicy.maximumRequestBytes + sessionPolicy.requestPaddingBytes + 32);
        const opened = open(request.frame);
        if (!opened.final) throw sessionError('request-format', 'Request must be one final frame.');
        messages = messagesFrom(opened.plaintext);
      } catch { refuse('request-rejected'); return 'refused'; }

      // A cancel message or a closed channel stops generation; nothing else is accepted after the request.
      channel.receive(stop.signal).then(() => stop.abort(), () => {});
      const seal = createSealer(keys.response, keys.transcript, PROVIDER_TO_CLIENT, sessionPolicy.responsePaddingBytes);
      let sent = 0;
      const emit = (bytes, final) => {
        if (++sent > sessionPolicy.maximumResponseFrames) throw sessionError('session-frame', 'Response exceeds the frame limit.');
        channel.send({ type: 'frame', frame: seal(bytes, final) });
      };
      try {
        for await (const delta of generate(messages, { signal: stop.signal })) {
          stop.signal.throwIfAborted();
          const bytes = Buffer.from(delta, 'utf8');
          for (let offset = 0; offset < bytes.length; offset += sessionPolicy.maximumResponseChunkBytes) {
            emit(bytes.subarray(offset, offset + sessionPolicy.maximumResponseChunkBytes), false);
          }
        }
        stop.signal.throwIfAborted();
        emit(Buffer.from(JSON.stringify({ status: 'completed' })), true);
        return 'completed';
      } catch {
        if (stop.signal.aborted) return 'cancelled';
        try { emit(Buffer.from(JSON.stringify({ status: 'failed' })), true); } catch { /* peer gone */ }
        return 'failed';
      }
    } finally {
      signal?.removeEventListener('abort', onAbort);
      stop.abort();
      for (const secret of [keys?.request, keys?.response, keys?.transcript]) secret?.fill(0);
      channel.close();
    }
  }
  return Object.freeze({ serve });
}

/** Message channels. Everything that crosses a channel is visible to the untrusted host and network. */
import { sessionError } from './session.js';

const BINARY = new Set(['clientNonce', 'providerKey', 'report', 'clientKey', 'frame']);
const encode = message => JSON.stringify(message, (key, value) => {
  if (value?.type === 'Buffer' && Array.isArray(value.data)) return Buffer.from(value.data).toString('base64');
  return value;
});

function decode(line) {
  let message;
  try { message = JSON.parse(line); } catch { throw sessionError('wire-format', 'Message is not JSON.'); }
  if (message === null || typeof message !== 'object' || typeof message.type !== 'string') throw sessionError('wire-format', 'Message has no type.');
  const binary = value => {
    if (typeof value !== 'string' || !/^[A-Za-z0-9+/]*={0,2}$/.test(value)) throw sessionError('wire-format', 'Binary field is not base64.');
    return Buffer.from(value, 'base64');
  };
  for (const key of Object.keys(message)) if (BINARY.has(key)) message[key] = binary(message[key]);
  if (message.endorsement !== undefined) {
    if (message.endorsement === null || typeof message.endorsement !== 'object') throw sessionError('wire-format', 'Malformed endorsement.');
    message.endorsement = Object.fromEntries(Object.entries(message.endorsement).map(([key, value]) => [key, binary(value)]));
  }
  return message;
}

function aborted(signal) {
  return signal?.reason instanceof Error ? signal.reason : Object.assign(new Error('Aborted.'), { name: 'AbortError' });
}

/** Newline-delimited JSON over a pair of byte streams, with a bound on any one message. */
export function streamChannel(readable, writable, maximumLineBytes) {
  const lines = [];
  const waiters = [];
  let pending = Buffer.alloc(0);
  let ended = false;
  let failure = null;
  const bytes = { sent: 0, received: 0 };
  const settle = () => { while (waiters.length > 0) waiters.shift()(); };
  readable.on('data', chunk => {
    bytes.received += chunk.length;
    pending = Buffer.concat([pending, chunk]);
    for (let index = pending.indexOf(10); index !== -1; index = pending.indexOf(10)) {
      lines.push(pending.subarray(0, index).toString('utf8'));
      pending = pending.subarray(index + 1);
    }
    if (pending.length > maximumLineBytes) { failure = sessionError('wire-format', 'Message exceeds the frame limit.'); readable.destroy(); }
    settle();
  });
  readable.on('end', () => { ended = true; settle(); });
  readable.on('close', () => { ended = true; settle(); });
  readable.on('error', () => { ended = true; settle(); });
  writable.on('error', () => { ended = true; settle(); });
  return {
    bytes,
    send(message) {
      if (!writable.writable) throw sessionError('wire-closed', 'Channel is closed.');
      const line = Buffer.from(encode(message) + '\n', 'utf8');
      bytes.sent += line.length;
      writable.write(line);
    },
    async receive(signal) {
      for (;;) {
        if (signal?.aborted) throw aborted(signal);
        if (failure) throw failure;
        if (lines.length > 0) return decode(lines.shift());
        if (ended) return null;
        await new Promise(resolve => {
          const wake = () => { signal?.removeEventListener('abort', wake); resolve(); };
          waiters.push(wake);
          signal?.addEventListener('abort', wake, { once: true });
        });
      }
    },
    close() { ended = true; writable.end(); settle(); },
  };
}

/**
 * In-process channel pair. `tap(direction, text)` sees every serialized message; `intercept(direction, message)` may
 * return a replacement, which is how tests play the hostile host.
 */
export function memoryChannelPair({ tap = () => {}, intercept = (direction, message) => message } = {}) {
  const end = () => ({ queue: [], waiters: [], closed: false });
  const sides = { client: end(), provider: end() };
  const wake = side => { while (side.waiters.length > 0) side.waiters.shift()(); };
  const make = (own, peer, direction) => ({
    send(message) {
      if (own.closed || peer.closed) throw sessionError('wire-closed', 'Channel is closed.');
      const text = encode(message);
      tap(direction, text);
      const delivered = intercept(direction, decode(text));
      if (delivered !== null) { peer.queue.push(delivered); wake(peer); }
    },
    async receive(signal) {
      for (;;) {
        if (signal?.aborted) throw aborted(signal);
        if (own.queue.length > 0) return own.queue.shift();
        if (own.closed || peer.closed) return null;
        await new Promise(resolve => {
          const done = () => { signal?.removeEventListener('abort', done); resolve(); };
          own.waiters.push(done);
          signal?.addEventListener('abort', done, { once: true });
        });
      }
    },
    close() { own.closed = true; wake(own); wake(peer); },
  });
  return [make(sides.client, sides.provider, 'client-to-provider'), make(sides.provider, sides.client, 'provider-to-client')];
}

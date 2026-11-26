// In-memory share operators for the cluster tests (no GPU, no socket). Not a test file itself.
export const MAGIC = 0x31535352;
export const plan = { fx: 20, fw: 24, stages: [0, 1].map(index => ({ index, groups: [{ id: `g${index}`, in: 4, tensors: [{ shape: [3, 4] }] }] })) };

export function frame(header, payload = new Uint8Array(0)) {
  const js = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(16 + js.length + payload.length), view = new DataView(out.buffer);
  view.setUint32(0, MAGIC, true); view.setUint32(4, js.length, true); out.set(js, 8);
  view.setUint32(8 + js.length, payload.length, true); out.set(payload, 16 + js.length);
  return out;
}

/**
 * An in-memory operator for stage/share; `stallOn(header)` true makes it never answer that message (or anything after).
 * `compute(header, payload, state)` returns the linear reply payload (default: zero shares of the plan's 3 rows).
 */
export function fakeOperator(stage, share, { stallOn = () => false, compute = null, caps = [] } = {}) {
  let inbox = new Uint8Array(0), outbox = [], waiter = null, stalled = false;
  const seen = [], state = { closed: false, seen };
  const pump = () => {
    if (!waiter) return;
    if (state.closed) { const w = waiter; waiter = null; w.reject(Error('operator closed the connection')); return; }
    const have = outbox.reduce((s, c) => s + c.length, 0);
    if (have < waiter.n) return;
    const all = new Uint8Array(have); let o = 0; for (const c of outbox) { all.set(c, o); o += c.length; }
    const { n, resolve } = waiter; waiter = null;
    outbox = [all.subarray(n)];
    resolve(all.slice(0, n));
  };
  const handle = (header, payload) => {
    seen.push(header.op);
    if (stalled || stallOn(header)) { stalled = true; return; }
    if (header.op === 'hello') return frame({ stage, share, fx: plan.fx, fw: plan.fw, caps });
    if (header.op === 'linear') return frame({ round: header.round, computeNs: 1000 }, compute ? compute(header, payload, state) : new Uint8Array(header.n * 3 * 8));
    if (header.op === 'options') state.options = header;
    return frame({ ok: true });
  };
  const stream = {
    write(bytes) {
      const next = new Uint8Array(inbox.length + bytes.length); next.set(inbox); next.set(bytes, inbox.length); inbox = next;
      for (;;) {
        if (inbox.length < 8) return;
        const view = new DataView(inbox.buffer, inbox.byteOffset, inbox.length), jsLen = view.getUint32(4, true);
        if (inbox.length < 16 + jsLen) return;
        const len = view.getUint32(8 + jsLen, true) + view.getUint32(12 + jsLen, true) * 4294967296;
        if (inbox.length < 16 + jsLen + len) return;
        const header = JSON.parse(new TextDecoder().decode(inbox.subarray(8, 8 + jsLen)));
        const payload = inbox.slice(16 + jsLen, 16 + jsLen + len);
        inbox = inbox.slice(16 + jsLen + len);
        state.payloads ??= [];
        if (header.op === 'linear') state.payloads.push(payload);
        const reply = handle(header, payload);
        if (reply) { outbox.push(reply); queueMicrotask(pump); }
      }
    },
    read: n => new Promise((resolve, reject) => { waiter = { n, resolve, reject }; pump(); }),
    close() { state.closed = true; pump(); }
  };
  return { stream, state };
}


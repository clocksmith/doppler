// The one selection every view of a run shares: which run, branch, token
// position, layer and operation is being looked at, and which run it is
// compared with. A view calls select() with what the visitor picked and
// subscribes to be told; views never talk to each other.

export const SELECTION_FIELDS = ['runId', 'branchId', 'position', 'layer', 'op', 'compareRunId'];
const WHOLE_NUMBER = new Set(['position', 'layer']);

export function createSelection(initial = {}) {
  let state = Object.fromEntries(SELECTION_FIELDS.map((field) => [field, null]));
  const listeners = new Set();

  function check(patch) {
    for (const [field, value] of Object.entries(patch)) {
      if (!SELECTION_FIELDS.includes(field)) throw new Error(`"${field}" is not part of the selection. It holds: ${SELECTION_FIELDS.join(', ')}.`);
      if (value === null) continue;
      if (WHOLE_NUMBER.has(field) ? !(Number.isInteger(value) && value >= 0) : !(typeof value === 'string' && value !== '')) {
        throw new Error(`Selection ${field} must be ${WHOLE_NUMBER.has(field) ? 'a whole number from 0' : 'a non-empty string'} or null.`);
      }
    }
  }

  function select(patch) {
    check(patch);
    const next = { ...state, ...patch };
    // A token position, branch and operation belong to one run. Choosing another
    // run drops them unless the same call names them; the layer is kept, since
    // layers line up across runs of one model.
    if ('runId' in patch && patch.runId !== state.runId) {
      for (const field of ['branchId', 'position', 'op']) if (!(field in patch)) next[field] = null;
    }
    if (next.compareRunId !== null && next.compareRunId === next.runId) next.compareRunId = null;
    const changed = SELECTION_FIELDS.filter((field) => next[field] !== state[field]);
    if (changed.length === 0) return state;
    const previous = state;
    state = Object.freeze(next);
    for (const listener of [...listeners]) listener(state, changed, previous);
    return state;
  }

  check(initial);
  state = Object.freeze({ ...state, ...initial });
  return {
    get: () => state,
    select,
    // Returns the function that ends the subscription.
    subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
  };
}

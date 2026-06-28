export function describeToolLoopStatus(state) {
  if (state.ready === true) {
    return 'ready';
  }
  return 'indexing';
}

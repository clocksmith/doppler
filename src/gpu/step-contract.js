// Step-contract declarations for an observer of one recorder's commands (the
// whole-token decode program). An operation whose recorded commands carry the step's
// input token, or stay valid only within a range of positions or context lengths,
// declares that here; ordinary recording has no observer and the call does nothing.
/** @type {WeakMap<object, (declaration: import('./step-contract.js').StepContractDeclaration) => void>} */
const stepContractObservers = new WeakMap();

/** @type {typeof import('./step-contract.js').observeStepContract} */
export function observeStepContract(recorder, observer) {
  if (!recorder || typeof observer !== 'function') {
    throw new Error('observeStepContract requires a recorder and an observer function.');
  }
  if (stepContractObservers.has(recorder)) {
    throw new Error('observeStepContract: the recorder already has an observer.');
  }
  stepContractObservers.set(recorder, observer);
  return () => {
    if (stepContractObservers.get(recorder) === observer) stepContractObservers.delete(recorder);
  };
}

/** @type {typeof import('./step-contract.js').declareStepContract} */
export function declareStepContract(recorder, declaration) {
  const observer = recorder ? stepContractObservers.get(recorder) : undefined;
  if (observer) observer(declaration);
}

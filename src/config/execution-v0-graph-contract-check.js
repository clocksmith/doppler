import { compileExecutionV0 } from '../inference/pipelines/text/execution-v0.js';

function classifyGraphError(error) {
  const message = error instanceof Error ? error.message : String(error);
  if (message.includes('reads slot "') && message.includes('before it is produced')) {
    return {
      slotGraphOk: false,
      phaseBoundaryOk: true,
      error: message,
    };
  }
  if (message.includes('reads carried slot "') && message.includes('Add explicit cast at phase boundary')) {
    return {
      slotGraphOk: true,
      phaseBoundaryOk: false,
      error: message,
    };
  }
  return {
    slotGraphOk: false,
    phaseBoundaryOk: false,
    error: message,
  };
}

export function buildExecutionV0GraphContractArtifact(options = {}) {
  const modelId = String(options.modelId ?? 'model');
  try {
    const compiled = compileExecutionV0(options);
    if (!compiled) {
      return null;
    }
    return {
      schemaVersion: 1,
      source: 'doppler',
      ok: true,
      checks: [
        { id: `${modelId}.slotGraph`, ok: true },
        { id: `${modelId}.phaseBoundary`, ok: true },
      ],
      errors: [],
      stats: {
        prefillSteps: compiled.resolvedSteps.prefill.length,
        decodeSteps: compiled.resolvedSteps.decode.length,
      },
    };
  } catch (error) {
    const classified = classifyGraphError(error);
    return {
      schemaVersion: 1,
      source: 'doppler',
      ok: false,
      checks: [
        { id: `${modelId}.slotGraph`, ok: classified.slotGraphOk },
        { id: `${modelId}.phaseBoundary`, ok: classified.phaseBoundaryOk },
      ],
      errors: [classified.error],
      stats: {
        prefillSteps: 0,
        decodeSteps: 0,
      },
    };
  }
}

function requireHostBridge(hostBridge) {
  if (!hostBridge || typeof hostBridge.createTextGenerationProgram !== 'function') {
    throw new Error(
      'program bundle host: hostBridge.createTextGenerationProgram(bundle, options) is required.',
    );
  }
  return hostBridge;
}

export function createTextGenerationProgram(hostBridge, bundle, options = {}) {
  return requireHostBridge(hostBridge).createTextGenerationProgram(bundle, options);
}

export function createSequenceProgram(hostBridge, bundle, options = {}) {
  if (!hostBridge || typeof hostBridge.createSequenceProgram !== 'function') {
    throw new Error('program bundle host: hostBridge.createSequenceProgram(bundle, options) is required.');
  }
  return hostBridge.createSequenceProgram(bundle, options);
}

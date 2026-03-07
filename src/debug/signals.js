


export const SIGNALS = {
  
  DONE: '[DOPPLER:DONE]',
  
  RESULT: '[DOPPLER:RESULT]',
  
  ERROR: '[DOPPLER:ERROR]',
  
  PROGRESS: '[DOPPLER:PROGRESS]',
};


export function signalDone(payload) {
  console.log(`${SIGNALS.DONE} ${JSON.stringify(payload)}`);
}


export function signalResult(data) {
  console.log(`${SIGNALS.RESULT} ${JSON.stringify(data)}`);
}


export function signalError(error, details) {
  if (details != null && (typeof details !== 'object' || Array.isArray(details))) {
    throw new Error('signalError details must be an object when provided.');
  }
  if (details && Object.hasOwn(details, 'error')) {
    throw new Error('signalError details.error is reserved. Pass the primary error as the first argument.');
  }
  console.log(`${SIGNALS.ERROR} ${JSON.stringify({ error, ...(details ?? {}) })}`);
}


export function signalProgress(percent, message) {
  console.log(`${SIGNALS.PROGRESS} ${JSON.stringify({ percent, message })}`);
}

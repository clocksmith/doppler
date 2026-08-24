import {
  getInitialRetryDelayMs,
  getMaxRetries,
  getMaxRetryDelayMs,
} from '../download-types.js';

export function createAbortError(message = 'Download aborted') {
  const error = new Error(message);
  error.name = 'AbortError';
  return error;
}

// ============================================================================
// Fetch Operations
// ============================================================================


export async function fetchWithRetry(url, options = {}) {
  
  let lastError;
  const maxRetries = getMaxRetries();
  let delay = getInitialRetryDelayMs();

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, {
        ...options,
        signal: options.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return response;
    } catch (error) {
      lastError =  (error);

      // Don't retry if aborted
      if ( (error).name === 'AbortError') {
        throw error;
      }

      // Don't retry on 4xx errors (except 429)
      if ( (error).message.includes('HTTP 4') && ! (error).message.includes('HTTP 429')) {
        throw error;
      }

      if (attempt < maxRetries) {
        await new Promise(r => setTimeout(r, delay));
        delay = Math.min(delay * 2, getMaxRetryDelayMs());
      }
    }
  }

  throw  (lastError);
}

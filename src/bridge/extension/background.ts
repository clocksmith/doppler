/**
 * DOPPLER Bridge Background Script
 * Phase 3: Native Messaging Bridge
 *
 * Handles:
 * - Connection to native host
 * - Message routing between web page and native host
 * - Binary protocol handling
 * - Backpressure management
 *
 * @module bridge/extension/background
 */

// ============================================================================
// Constants
// ============================================================================

const NATIVE_HOST_NAME = 'dev.reploid.doppler';

// ============================================================================
// Types
// ============================================================================

interface Connection {
  webPort: chrome.runtime.Port;
  nativePort: chrome.runtime.Port;
}

interface BinaryMessage {
  type: 'binary';
  data: number[];
}

interface AckMessage {
  type: 'ack';
  reqId: number;
}

interface ErrorMessage {
  type: 'error';
  message: string;
}

type WebMessage = BinaryMessage | AckMessage;
type NativeMessage = BinaryMessage | ErrorMessage;

// ============================================================================
// Global State
// ============================================================================

/** Active connections */
const connections = new Map<string, Connection>();

/** Message queue for backpressure */
const messageQueues = new Map<string, unknown[]>();

// ============================================================================
// Connection Handling
// ============================================================================

function handleWebConnection(webPort: chrome.runtime.Port): void {
  if (webPort.name !== 'doppler-bridge') {
    console.warn('[DopplerBridge] Unknown connection:', webPort.name);
    return;
  }

  console.log('[DopplerBridge] Web page connected');

  const portId = crypto.randomUUID();
  let nativePort: chrome.runtime.Port;

  // Connect to native host
  try {
    nativePort = chrome.runtime.connectNative(NATIVE_HOST_NAME);
    console.log('[DopplerBridge] Connected to native host');
  } catch (err) {
    console.error('[DopplerBridge] Failed to connect to native host:', err);
    webPort.postMessage({
      type: 'error',
      message: `Failed to connect to native host: ${(err as Error).message}`,
    });
    return;
  }

  // Store connection
  connections.set(portId, { webPort, nativePort });

  // Handle messages from web page
  webPort.onMessage.addListener((message: unknown) => {
    handleWebMessage(portId, message as WebMessage);
  });

  // Handle messages from native host
  nativePort.onMessage.addListener((message: unknown) => {
    handleNativeMessage(portId, message as NativeMessage);
  });

  // Handle web page disconnect
  webPort.onDisconnect.addListener(() => {
    console.log('[DopplerBridge] Web page disconnected');
    cleanupConnection(portId);
  });

  // Handle native host disconnect
  nativePort.onDisconnect.addListener(() => {
    const error = chrome.runtime.lastError;
    console.log('[DopplerBridge] Native host disconnected:', error?.message);

    // Notify web page
    try {
      webPort.postMessage({
        type: 'error',
        message: error?.message || 'Native host disconnected',
      });
    } catch {
      // Port already closed
    }

    cleanupConnection(portId);
  });
}

/**
 * Handle incoming connection from web page (external)
 */
chrome.runtime.onConnectExternal.addListener(handleWebConnection);

/**
 * Handle message from web page
 */
function handleWebMessage(portId: string, message: WebMessage): void {
  const conn = connections.get(portId);
  if (!conn) {
    console.error('[DopplerBridge] No connection for port:', portId);
    return;
  }

  if (message.type === 'binary') {
    // Forward binary message to native host
    // Chrome native messaging uses JSON, so we send as array
    conn.nativePort.postMessage({
      type: 'binary',
      data: message.data,
    });
  } else if (message.type === 'ack') {
    // Forward ACK to native host
    conn.nativePort.postMessage({
      type: 'ack',
      reqId: message.reqId,
    });
  } else {
    console.warn('[DopplerBridge] Unknown message type from web:', (message as { type: string }).type);
  }
}

/**
 * Handle message from native host
 */
function handleNativeMessage(portId: string, message: NativeMessage): void {
  const conn = connections.get(portId);
  if (!conn) {
    console.error('[DopplerBridge] No connection for port:', portId);
    return;
  }

  if (message.type === 'binary') {
    // Forward binary message to web page
    conn.webPort.postMessage({
      type: 'binary',
      data: message.data,
    });
  } else if (message.type === 'error') {
    conn.webPort.postMessage({
      type: 'error',
      message: message.message,
    });
  } else {
    console.warn('[DopplerBridge] Unknown message type from native:', (message as { type: string }).type);
  }
}

/**
 * Cleanup connection
 */
function cleanupConnection(portId: string): void {
  const conn = connections.get(portId);
  if (conn) {
    try {
      conn.nativePort?.disconnect();
    } catch {
      // Already disconnected
    }
    connections.delete(portId);
  }
  messageQueues.delete(portId);
}

/**
 * Handle internal connections (same extension)
 */
chrome.runtime.onConnect.addListener((port: chrome.runtime.Port) => {
  if (port.name === 'doppler-bridge') {
    handleWebConnection(port);
  }
});

console.log('[DopplerBridge] Background script loaded');

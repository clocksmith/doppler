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

import { log } from '../../debug/index.js';

// ============================================================================
// Constants
// ============================================================================

const NATIVE_HOST_NAME = 'dev.reploid.doppler';

// ============================================================================
// Global State
// ============================================================================

/** Active connections */
const connections = new Map();

/** Message queue for backpressure */
const messageQueues = new Map();

// ============================================================================
// Connection Handling
// ============================================================================

function handleWebConnection(webPort) {
  if (webPort.name !== 'doppler-bridge') {
    log.warn('DopplerBridge', `Unknown connection: ${webPort.name}`);
    return;
  }

  log.info('DopplerBridge', 'Web page connected');

  const portId = crypto.randomUUID();
  let nativePort;

  // Connect to native host
  try {
    nativePort = chrome.runtime.connectNative(NATIVE_HOST_NAME);
    log.info('DopplerBridge', 'Connected to native host');
  } catch (err) {
    log.error('DopplerBridge', `Failed to connect to native host: ${err.message}`);
    webPort.postMessage({
      type: 'error',
      message: `Failed to connect to native host: ${err.message}`,
    });
    return;
  }

  // Store connection
  connections.set(portId, { webPort, nativePort });

  // Handle messages from web page
  webPort.onMessage.addListener((message) => {
    handleWebMessage(portId, message);
  });

  // Handle messages from native host
  nativePort.onMessage.addListener((message) => {
    handleNativeMessage(portId, message);
  });

  // Handle web page disconnect
  webPort.onDisconnect.addListener(() => {
    log.info('DopplerBridge', 'Web page disconnected');
    cleanupConnection(portId);
  });

  // Handle native host disconnect
  nativePort.onDisconnect.addListener(() => {
    const error = chrome.runtime.lastError;
    log.warn('DopplerBridge', `Native host disconnected: ${error?.message || 'unknown'}`);

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
function handleWebMessage(portId, message) {
  const conn = connections.get(portId);
  if (!conn) {
    log.error('DopplerBridge', `No connection for port: ${portId}`);
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
    log.warn('DopplerBridge', `Unknown message type from web: ${message.type}`);
  }
}

/**
 * Handle message from native host
 */
function handleNativeMessage(portId, message) {
  const conn = connections.get(portId);
  if (!conn) {
    log.error('DopplerBridge', `No connection for port: ${portId}`);
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
    log.warn('DopplerBridge', `Unknown message type from native: ${message.type}`);
  }
}

/**
 * Cleanup connection
 */
function cleanupConnection(portId) {
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
chrome.runtime.onConnect.addListener((port) => {
  if (port.name === 'doppler-bridge') {
    handleWebConnection(port);
  }
});

log.info('DopplerBridge', 'Background script loaded');

import { ELECTRON_RELEASE_IPC_CHANNEL } from '../../src/client/electron/index.js';

export function exposeDocumentSearchReleaseBridge(contextBridge, ipcRenderer) {
  contextBridge.exposeInMainWorld('dopplerRelease', Object.freeze({
    status: () => ipcRenderer.invoke(ELECTRON_RELEASE_IPC_CHANNEL, { action: 'status' }),
    activate: (decision, customerAuthorizationDigest) => ipcRenderer.invoke(
      ELECTRON_RELEASE_IPC_CHANNEL,
      { action: 'activate', decision, customerAuthorizationDigest }
    ),
    rollback: (customerAuthorizationDigest) => ipcRenderer.invoke(
      ELECTRON_RELEASE_IPC_CHANNEL,
      { action: 'rollback', customerAuthorizationDigest }
    ),
  }));
}

import {
  ELECTRON_RELEASE_IPC_CHANNEL,
  createElectronReleaseIpcHandler,
  createElectronReleaseStateCoordinator,
} from '../../src/client/electron/index.js';

export function registerDocumentSearchReleaseMain(options) {
  const coordinator = createElectronReleaseStateCoordinator({
    stateStore: options.stateStore,
    verifyReleaseDecision: options.verifyReleaseDecision,
    verifyRevocationSnapshot: options.verifyRevocationSnapshot,
    now: options.now,
  });
  options.ipcMain.handle(
    ELECTRON_RELEASE_IPC_CHANNEL,
    createElectronReleaseIpcHandler(coordinator)
  );
  return coordinator;
}

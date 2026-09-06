import { app, BrowserWindow, ipcMain } from 'electron';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const userData = app.commandLine.getSwitchValue('doppler-probe-user-data');
if (!userData) throw new Error('Electron probe requires an isolated user-data directory.');
app.setPath('userData', userData);
// Electron must finish importing its ESM entry before it can emit ready.
app.whenReady().then(async () => {
  const configPath = app.commandLine.getSwitchValue('doppler-release-main');
  const config = configPath ? JSON.parse(await fs.readFile(configPath, 'utf8')) : null;
  const receipt = { evidenceClass: 'internal-electron-main-coordinator', before: null, after: null,
    transactions: [], ipc: [], error: null, externalAdoption: false };
  const save = async () => { if (config) await fs.writeFile(config.receiptPath, JSON.stringify(receipt, null, 2)); };
  try {
    const window = new BrowserWindow({
      show: false,
      webPreferences: { nodeIntegration: false, contextIsolation: true, sandbox: true,
        ...(config ? { preload: config.preloadPath } : {}) },
    });
    if (config) {
      const { registerDocumentSearchReleaseMain } = await import(pathToFileURL(path.join(config.consumerDir, 'main.js')).href);
      const { createDocumentSearchReleaseStore } = await import(pathToFileURL(path.join(config.consumerDir, 'release-storage.js')).href);
      const { createElectronReleaseIpcHandler, verifyProductionReleaseEvidenceSignature } = await import(pathToFileURL(config.electronEntry).href);
      const verify = record => verifyProductionReleaseEvidenceSignature(record, config.trustedSigners);
      const coordinator = registerDocumentSearchReleaseMain({
        ipcMain, stateStore: createDocumentSearchReleaseStore(config.statePath),
        now: () => config.now, verifyReleaseDecision: verify, verifyRevocationSnapshot: verify,
        async authorizeRequest(event, request) {
          const frameUrl = event.senderFrame?.url ?? null;
          const permitted = event.sender === window.webContents && event.senderFrame === window.webContents.mainFrame
            && frameUrl !== null && new URL(frameUrl).origin === config.allowedOrigin
            && config.allowedRendererActions.includes(request.action);
          receipt.ipc.push({ action: request.action, frameUrl, permitted, senderId: event.sender.id });
          await save();
          return permitted;
        },
      });
      receipt.before = await coordinator.load();
      const applicationAuthority = {};
      const initialize = createElectronReleaseIpcHandler(coordinator, { authorizeRequest: event => event === applicationAuthority });
      for (const request of config.actions) {
        const transaction = { action: request.action, accepted: false };
        receipt.transactions.push(transaction);
        try {
          await initialize(applicationAuthority, request);
          transaction.accepted = true;
        } catch (error) { transaction.error = error.message; throw error; }
        finally { receipt.after = await coordinator.load(); await save(); }
      }
      receipt.after = await coordinator.load();
      await save();
    }
    await window.loadURL('about:blank');
  } catch (error) { receipt.error = error.message; await save(); throw error; }
}).catch((error) => {
  console.error(error.stack);
  app.exit(1);
});

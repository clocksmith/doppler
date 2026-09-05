import { app, BrowserWindow } from 'electron';

const userData = app.commandLine.getSwitchValue('doppler-probe-user-data');
if (!userData) throw new Error('Electron probe requires an isolated user-data directory.');
app.setPath('userData', userData);
// Electron must finish importing its ESM entry before it can emit ready.
app.whenReady().then(async () => {
  const window = new BrowserWindow({
    show: false,
    webPreferences: { nodeIntegration: false, contextIsolation: true, sandbox: true },
  });
  await window.loadURL('about:blank');
}).catch((error) => {
  console.error(error.stack);
  app.exit(1);
});

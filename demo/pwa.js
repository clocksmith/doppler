const DEMO_SCOPE = '/demo/';
const DEMO_SW_URL = '/demo/sw.js';
const MAX_LAUNCH_FILE_BYTES = 256 * 1024;

let pwaInitialized = false;
let deferredInstallPrompt = null;
let pendingLaunchPrompt = null;

function $(id) {
  return document.getElementById(id);
}

function setInstallButtonVisible(visible) {
  const button = $('install-btn');
  if (button) {
    button.hidden = !visible;
  }
}

function createDisplayModeQuery(mode) {
  return window.matchMedia(`(display-mode: ${mode})`);
}

function isStandaloneModeActive() {
  return createDisplayModeQuery('standalone').matches
    || createDisplayModeQuery('window-controls-overlay').matches;
}

function isWindowControlsOverlayActive() {
  if (!createDisplayModeQuery('window-controls-overlay').matches) {
    return false;
  }
  const overlay = navigator.windowControlsOverlay ?? null;
  return overlay?.visible ?? true;
}

function bindMediaQueryChange(query, listener) {
  if (typeof query?.addEventListener === 'function') {
    query.addEventListener('change', listener);
    return;
  }
  if (typeof query?.addListener === 'function') {
    query.addListener(listener);
  }
}

function syncTitlebarPhase() {
  const source = $('status-text');
  const target = $('desktop-titlebar-phase');
  if (source && target) {
    target.textContent = source.textContent || 'Ready';
  }
}

function ensureStatusMirror() {
  const source = $('status-text');
  if (!source || source.dataset.pwaStatusObserved === 'true') {
    syncTitlebarPhase();
    return;
  }
  const observer = new MutationObserver(() => {
    syncTitlebarPhase();
  });
  observer.observe(source, {
    childList: true,
    characterData: true,
    subtree: true,
  });
  source.dataset.pwaStatusObserved = 'true';
  syncTitlebarPhase();
}

function updateDesktopDisplayMode() {
  const standalone = isStandaloneModeActive();
  const overlayVisible = isWindowControlsOverlayActive();
  document.body.classList.toggle('pwa-standalone', standalone);
  document.body.classList.toggle('pwa-window-controls', overlayVisible);
  const titlebar = $('desktop-titlebar');
  if (titlebar) {
    titlebar.hidden = !overlayVisible;
  }
  if (standalone) {
    setInstallButtonVisible(false);
  }
  syncTitlebarPhase();
}

async function registerDemoServiceWorker() {
  if (!window.isSecureContext || !('serviceWorker' in navigator)) {
    return;
  }
  try {
    await navigator.serviceWorker.register(DEMO_SW_URL, { scope: DEMO_SCOPE });
  } catch (error) {
    console.warn('[DemoPwa] Service worker registration failed:', error?.message ?? error);
  }
}

function queueLaunchPrompt(text, sourceLabel = null) {
  pendingLaunchPrompt = {
    text,
    sourceLabel,
  };
}

function extractPromptFromJson(text) {
  try {
    const parsed = JSON.parse(text);
    if (typeof parsed === 'string') {
      return parsed.trim();
    }
    if (typeof parsed?.prompt === 'string') {
      return parsed.prompt.trim();
    }
    if (typeof parsed?.input === 'string') {
      return parsed.input.trim();
    }
    return JSON.stringify(parsed, null, 2);
  } catch {
    return text.trim();
  }
}

function normalizeLaunchPrompt(text, fileName = '') {
  const normalizedName = String(fileName || '').trim().toLowerCase();
  if (normalizedName.endsWith('.json')) {
    return extractPromptFromJson(text);
  }
  return String(text ?? '').trim();
}

function applyLaunchPrompt(text, sourceLabel = null) {
  const promptInput = $('prompt-input');
  if (!promptInput) {
    queueLaunchPrompt(text, sourceLabel);
    return false;
  }
  promptInput.value = text;
  promptInput.dispatchEvent(new Event('input', { bubbles: true }));
  promptInput.focus();
  promptInput.selectionStart = promptInput.value.length;
  promptInput.selectionEnd = promptInput.value.length;
  if (sourceLabel) {
    const status = $('status-text');
    if (status && (!status.textContent || status.textContent === 'Ready')) {
      status.textContent = sourceLabel;
      syncTitlebarPhase();
    }
  }
  return true;
}

function applyShortcutUrl(targetUrl) {
  if (!targetUrl) {
    return;
  }
  const url = new URL(targetUrl, window.location.origin);
  const shortcut = url.searchParams.get('shortcut');
  const prompt = url.searchParams.get('prompt');
  if (shortcut === 'new') {
    queueLaunchPrompt('', 'New prompt');
    return;
  }
  if (prompt) {
    const label = shortcut === 'xray' ? 'Kernel X-Ray shortcut' : 'Desktop shortcut';
    queueLaunchPrompt(prompt, label);
    return;
  }
  if (shortcut === 'xray') {
    queueLaunchPrompt(
      'Explain the current decode hotspots.',
      'Kernel X-Ray shortcut'
    );
  }
}

async function consumeLaunchFiles(files) {
  const handles = Array.from(files ?? []);
  if (handles.length < 1) {
    return;
  }
  const file = await handles[0].getFile();
  if (file.size > MAX_LAUNCH_FILE_BYTES) {
    queueLaunchPrompt(
      '',
      `${file.name} is too large for prompt import`
    );
    return;
  }
  const text = await file.text();
  const prompt = normalizeLaunchPrompt(text, file.name);
  queueLaunchPrompt(prompt, `Opened ${file.name}`);
}

function initLaunchQueueConsumer() {
  const launchQueue = window.launchQueue;
  if (!launchQueue?.setConsumer) {
    return;
  }
  launchQueue.setConsumer(async (launchParams) => {
    try {
      applyShortcutUrl(launchParams?.targetURL ?? null);
      if (launchParams?.files) {
        await consumeLaunchFiles(launchParams.files);
      }
      flushPwaLaunchState();
    } catch (error) {
      console.warn('[DemoPwa] Launch handling failed:', error?.message ?? error);
    }
  });
}

function initInstallPrompt() {
  window.addEventListener('beforeinstallprompt', (event) => {
    event.preventDefault();
    deferredInstallPrompt = event;
    if (!document.body.classList.contains('pwa-standalone')) {
      setInstallButtonVisible(true);
    }
  });

  window.addEventListener('appinstalled', () => {
    deferredInstallPrompt = null;
    setInstallButtonVisible(false);
    updateDesktopDisplayMode();
  });

  $('install-btn')?.addEventListener('click', async () => {
    if (!deferredInstallPrompt) {
      return;
    }
    deferredInstallPrompt.prompt();
    try {
      await deferredInstallPrompt.userChoice;
    } finally {
      deferredInstallPrompt = null;
      setInstallButtonVisible(false);
    }
  });
}

export function flushPwaLaunchState() {
  ensureStatusMirror();
  if (!pendingLaunchPrompt) {
    return;
  }
  const { text, sourceLabel } = pendingLaunchPrompt;
  pendingLaunchPrompt = null;
  applyLaunchPrompt(text, sourceLabel);
}

export function initPwa() {
  if (pwaInitialized) {
    updateDesktopDisplayMode();
    return;
  }
  pwaInitialized = true;

  applyShortcutUrl(window.location.href);
  initLaunchQueueConsumer();
  initInstallPrompt();
  ensureStatusMirror();
  updateDesktopDisplayMode();
  registerDemoServiceWorker();

  bindMediaQueryChange(createDisplayModeQuery('standalone'), updateDesktopDisplayMode);
  bindMediaQueryChange(createDisplayModeQuery('window-controls-overlay'), updateDesktopDisplayMode);
  navigator.windowControlsOverlay?.addEventListener?.('geometrychange', updateDesktopDisplayMode);
}

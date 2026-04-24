import { state } from './ui/state.js';

let examples = null;
let shuffleIndex = -1;
let imageData = null;
let onRun = null;

function $(id) { return document.getElementById(id); }

async function loadExamples() {
  try {
    const url = new URL('./examples.json', import.meta.url).toString();
    const res = await fetch(url);
    examples = await res.json();
  } catch {
    examples = { text: ['hello world'], image: [] };
  }
}

function shuffle() {
  if (!examples?.text?.length) return;
  shuffleIndex = (shuffleIndex + 1) % examples.text.length;
  const promptEl = $('prompt-input');
  if (promptEl) promptEl.value = examples.text[shuffleIndex];
}

function setupImageDrop() {
  const zone = $('image-drop');
  if (!zone) return;

  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });

  zone.addEventListener('dragleave', () => {
    zone.classList.remove('drag-over');
  });

  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer?.files?.[0];
    if (file && file.type.startsWith('image/')) {
      readImageFile(file);
    }
  });

  zone.addEventListener('click', () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = () => {
      const file = input.files?.[0];
      if (file) readImageFile(file);
    };
    input.click();
  });
}

function readImageFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    imageData = reader.result;
    const zone = $('image-drop');
    if (zone) {
      zone.innerHTML = '';
      zone.classList.add('has-image');
      const img = document.createElement('img');
      img.src = imageData;
      img.alt = 'Attached image';
      zone.appendChild(img);
    }
  };
  reader.readAsDataURL(file);
}

export function clearImage() {
  imageData = null;
  const zone = $('image-drop');
  if (zone) {
    zone.textContent = 'Drop image here';
    zone.classList.remove('has-image');
  }
}

export function getPrompt() {
  return ($('prompt-input')?.value ?? '').trim();
}

export function setPromptValue(value) {
  const promptEl = $('prompt-input');
  if (!promptEl) {
    return;
  }
  promptEl.value = typeof value === 'string' ? value : String(value ?? '');
  promptEl.focus();
}

export function getImage() {
  return imageData;
}

export function setRunHandler(handler) {
  onRun = handler;
}

export async function initInput() {
  await loadExamples();

  $('shuffle-btn')?.addEventListener('click', shuffle);

  $('run-btn')?.addEventListener('click', () => {
    if (onRun) onRun();
  });

  $('prompt-input')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (onRun) onRun();
    }
  });

  setupImageDrop();

  // Start with a random example
  if (examples?.text?.length) {
    shuffleIndex = Math.floor(Math.random() * examples.text.length) - 1;
    shuffle();
  }
}

export function setRunEnabled(enabled) {
  const btn = $('run-btn');
  if (btn) btn.disabled = !enabled;
}

export function setGenerating(active) {
  const runBtn = $('run-btn');
  const stopBtn = $('stop-btn');
  if (runBtn) runBtn.hidden = active;
  if (stopBtn) stopBtn.hidden = !active;
}

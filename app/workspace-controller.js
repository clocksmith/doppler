import { log } from '../src/debug/index.js';
import { createWorkspaceVfs } from './workspace-vfs.js';
import { createWorkspaceImporter } from './workspace-importer.js';

const MAX_LIST_ENTRIES = 200;

export class WorkspaceController {
  #importButton = null;
  #refreshButton = null;
  #statusEl = null;
  #filesEl = null;

  #workspaceId = 'default';
  #vfs = null;
  #importer = null;

  constructor(options = {}) {
    this.#importButton = options.importButton || null;
    this.#refreshButton = options.refreshButton || null;
    this.#statusEl = options.statusEl || null;
    this.#filesEl = options.filesEl || null;
  }

  async init() {
    if (this.#importButton && typeof window.showDirectoryPicker !== 'function') {
      this.#importButton.disabled = true;
      this.#importButton.title = 'Folder import requires File System Access API';
    }
    await this.#initVfs();
    this.#initImporter();
    this.#bind();
    await this.refresh();
  }

  async refresh() {
    if (!this.#vfs) return;
    const files = await this.#vfs.list('/');
    this.#renderFiles(files);
    this.#setStatus(`${files.length} files`);
  }

  async importFolder() {
    if (!this.#importer) return;
    this.#setStatus('Awaiting permission...');
    const result = await this.#importer.importDirectory({ workspaceId: this.#workspaceId });
    if (!result) {
      this.#setStatus('Import cancelled');
      return;
    }
    if (result.workspaceId && result.workspaceId !== this.#workspaceId) {
      this.#workspaceId = result.workspaceId;
      await this.#initVfs();
      this.#initImporter();
    }
    this.#setStatus(`Imported ${result.total || 0} files`);
    await this.refresh();
  }

  #initImporter() {
    if (this.#importer) {
      this.#importer.terminate();
    }
    this.#importer = createWorkspaceImporter({
      backendType: this.#vfs?.backendType,
      rootDirName: this.#vfs?.rootDirName,
      onProgress: (progress) => {
        if (!progress) return;
        const total = progress.total || 0;
        const completed = progress.completed || 0;
        this.#setStatus(`Importing ${completed}/${total}`);
      },
    });
  }

  async #initVfs() {
    this.#vfs = await createWorkspaceVfs({ workspaceId: this.#workspaceId });
    this.#setStatus(`Backend: ${this.#vfs.backendType}`);
  }

  #bind() {
    if (this.#importButton) {
      this.#importButton.addEventListener('click', () => {
        this.importFolder().catch((error) => {
          log.error('Workspace', 'Import failed', error);
          this.#setStatus(error?.message || 'Import failed');
        });
      });
    }

    if (this.#refreshButton) {
      this.#refreshButton.addEventListener('click', () => {
        this.refresh().catch((error) => {
          log.error('Workspace', 'Refresh failed', error);
          this.#setStatus(error?.message || 'Refresh failed');
        });
      });
    }
  }

  #setStatus(text) {
    if (this.#statusEl) {
      this.#statusEl.textContent = text;
    }
  }

  #renderFiles(files) {
    if (!this.#filesEl) return;
    const list = Array.isArray(files) ? files.slice(0, MAX_LIST_ENTRIES) : [];
    this.#filesEl.innerHTML = '';
    if (list.length === 0) {
      this.#filesEl.textContent = 'No files loaded';
      return;
    }

    for (const path of list) {
      const row = document.createElement('div');
      row.className = 'workspace-file';
      row.textContent = path;
      this.#filesEl.appendChild(row);
    }

    if (files.length > MAX_LIST_ENTRIES) {
      const more = document.createElement('div');
      more.className = 'workspace-file';
      more.textContent = `... ${files.length - MAX_LIST_ENTRIES} more`;
      this.#filesEl.appendChild(more);
    }
  }
}

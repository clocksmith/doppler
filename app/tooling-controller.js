import { log } from '../src/debug/index.js';
import { ToolRunner } from './tool-runner.js';

const DEFAULT_ARGS = '{\n  \n}';

export class ToolingController {
  #toolSelect = null;
  #argsInput = null;
  #runButton = null;
  #refreshButton = null;
  #outputEl = null;
  #statusEl = null;

  #runner = null;
  #vfs = null;

  constructor(options = {}) {
    this.#toolSelect = options.toolSelect || null;
    this.#argsInput = options.argsInput || null;
    this.#runButton = options.runButton || null;
    this.#refreshButton = options.refreshButton || null;
    this.#outputEl = options.outputEl || null;
    this.#statusEl = options.statusEl || null;
  }

  init() {
    if (this.#argsInput && !this.#argsInput.value) {
      this.#argsInput.value = DEFAULT_ARGS;
    }

    if (this.#refreshButton) {
      this.#refreshButton.addEventListener('click', () => {
        this.refresh().catch((error) => {
          log.error('Tools', 'Refresh failed', error);
          this.#setStatus(error?.message || 'Refresh failed');
        });
      });
    }

    if (this.#runButton) {
      this.#runButton.addEventListener('click', () => {
        this.runSelected().catch((error) => {
          log.error('Tools', 'Run failed', error);
          this.#setStatus(error?.message || 'Run failed');
          this.#setOutput(error?.message || 'Run failed');
        });
      });
    }
  }

  async setVfs(vfs) {
    this.#vfs = vfs;
    if (!this.#runner) {
      this.#runner = new ToolRunner({ vfs });
    } else {
      this.#runner.setVfs(vfs);
    }
    await this.refresh();
  }

  async refresh() {
    if (!this.#runner) {
      this.#setStatus('No workspace loaded');
      return;
    }

    this.#setStatus('Loading tools...');
    const result = await this.#runner.refresh({ force: true });
    this.#renderTools(result.tools);

    if (result.errors.length > 0) {
      this.#setStatus(`Loaded ${result.tools.length} tools, ${result.errors.length} errors`);
      this.#setOutput(JSON.stringify(result.errors, null, 2));
    } else {
      this.#setStatus(`Loaded ${result.tools.length} tools`);
    }
  }

  async runSelected() {
    if (!this.#runner) {
      this.#setStatus('No workspace loaded');
      return;
    }

    const toolName = this.#toolSelect?.value;
    if (!toolName) {
      this.#setStatus('Select a tool');
      return;
    }

    const args = this.#parseArgs();
    this.#setStatus(`Running ${toolName}...`);

    const result = await this.#runner.execute(toolName, args);
    this.#setStatus(`Completed ${toolName}`);
    this.#setOutput(this.#formatResult(result));
  }

  #parseArgs() {
    const raw = this.#argsInput?.value?.trim();
    if (!raw) return {};
    try {
      return JSON.parse(raw);
    } catch (error) {
      this.#setStatus('Args JSON invalid');
      throw error;
    }
  }

  #renderTools(tools) {
    if (!this.#toolSelect) return;
    this.#toolSelect.innerHTML = '';

    if (!tools || tools.length === 0) {
      const option = document.createElement('option');
      option.textContent = 'No tools found';
      option.value = '';
      this.#toolSelect.appendChild(option);
      this.#toolSelect.disabled = true;
      if (this.#runButton) this.#runButton.disabled = true;
      return;
    }

    for (const name of tools) {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = name;
      this.#toolSelect.appendChild(option);
    }

    this.#toolSelect.disabled = false;
    if (this.#runButton) this.#runButton.disabled = false;
  }

  #setStatus(text) {
    if (this.#statusEl) {
      this.#statusEl.textContent = text;
    }
  }

  #setOutput(text) {
    if (this.#outputEl) {
      this.#outputEl.textContent = text;
    }
  }

  #formatResult(result) {
    if (typeof result === 'string') return result;
    try {
      return JSON.stringify(result, null, 2);
    } catch (error) {
      return String(result);
    }
  }
}

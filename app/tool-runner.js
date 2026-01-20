import { loadWorkspaceModule } from './module-loader.js';

const DEFAULT_ROOT = '/tools';

function normalizeRoot(root) {
  if (!root) return DEFAULT_ROOT;
  let clean = root.trim().replace(/\/g, '/');
  if (!clean.startsWith('/')) clean = '/' + clean;
  clean = clean.replace(/\/+$/, '');
  return clean || DEFAULT_ROOT;
}

function nameFromPath(path) {
  const parts = path.split('/').filter(Boolean);
  const filename = parts[parts.length - 1] || '';
  return filename.endsWith('.js') ? filename.slice(0, -3) : filename;
}

export class ToolRunner {
  #vfs = null;
  #root = DEFAULT_ROOT;
  #tools = new Map();
  #toolInfo = new Map();

  constructor(options = {}) {
    this.#vfs = options.vfs || null;
    this.#root = normalizeRoot(options.root);
  }

  setVfs(vfs) {
    this.#vfs = vfs;
  }

  list() {
    return Array.from(this.#tools.keys()).sort();
  }

  getToolInfo(name) {
    return this.#toolInfo.get(name) || null;
  }

  async refresh(options = {}) {
    const vfs = this.#vfs;
    if (!vfs || typeof vfs.list !== 'function') {
      throw new Error('VFS not available');
    }

    const force = options.force === true;
    const root = normalizeRoot(this.#root);
    const files = await vfs.list(root);
    const toolFiles = files.filter((path) => path.endsWith('.js'));

    this.#tools.clear();
    this.#toolInfo.clear();

    const loaded = [];
    const errors = [];

    for (const path of toolFiles) {
      try {
        const name = await this.#loadTool(path, { force });
        loaded.push(name);
      } catch (error) {
        errors.push({ path, error: error?.message || String(error) });
      }
    }

    return { tools: loaded, errors };
  }

  async execute(name, args = {}) {
    if (!name) throw new Error('Missing tool name');
    const available = await this.#ensureLoaded(name);
    if (!available) {
      throw new Error(`Tool not found: ${name}`);
    }

    const handler = this.#tools.get(name);
    if (!handler) throw new Error(`Tool not found: ${name}`);

    const deps = {
      vfs: this.#vfs,
      ToolRunner: {
        execute: this.execute.bind(this),
        list: this.list.bind(this),
      },
    };

    return await handler(args, deps);
  }

  async #ensureLoaded(name) {
    if (this.#tools.has(name)) return true;
    const vfs = this.#vfs;
    if (!vfs) return false;
    const root = normalizeRoot(this.#root);
    const path = `${root}/${name}.js`;
    if (typeof vfs.exists === 'function') {
      const exists = await vfs.exists(path);
      if (!exists) return false;
    }
    await this.#loadTool(path, { force: true });
    return this.#tools.has(name);
  }

  async #loadTool(path, options = {}) {
    const vfs = this.#vfs;
    if (!vfs) throw new Error('VFS not available');

    const mod = await loadWorkspaceModule(vfs, path, { force: options.force === true });
    const handler = typeof mod?.default === 'function'
      ? mod.default
      : typeof mod?.tool?.call === 'function'
        ? mod.tool.call
        : null;

    if (!handler) {
      throw new Error(`Tool missing default export: ${path}`);
    }

    const name = mod?.tool?.name || nameFromPath(path);
    this.#tools.set(name, handler);

    if (mod?.tool) {
      this.#toolInfo.set(name, {
        name,
        description: mod.tool.description || null,
        inputSchema: mod.tool.inputSchema || null,
      });
    }

    return name;
  }
}

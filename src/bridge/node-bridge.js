/**
 * Node bridge — re-exports Node WebGPU bootstrap with provider priority documentation.
 *
 * Provider resolution priority (first match wins):
 *
 * 1. **Environment variable** — `DOPPLER_NODE_WEBGPU_MODULE` env var.
 *    Accepts npm package names, relative/absolute file paths, or `file://` URLs.
 *    When set, only this specifier is attempted.
 *
 * 2. **Pre-installed runtime** — if `navigator.gpu` and WebGPU enums
 *    (`GPUBufferUsage`, `GPUShaderStage`) are already available on `globalThis`,
 *    no module is loaded (provider is reported as `'pre-installed'`).
 *
 * 3. **Default candidates** — tried in order:
 *    a. `'webgpu'` (community Dawn bindings)
 *    The first candidate that imports, installs WebGPU globals, and yields a
 *    usable adapter wins.
 *
 * To force a specific provider, set the environment variable:
 *   DOPPLER_NODE_WEBGPU_MODULE=<module-or-path>
 *
 * See `src/tooling/node-webgpu.js` for the full implementation.
 */
export { bootstrapNodeWebGPU, bootstrapNodeWebGPUProvider } from '../tooling/node-webgpu.js';

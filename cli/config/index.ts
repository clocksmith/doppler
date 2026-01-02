/**
 * CLI Config Module
 *
 * Unified config loading, resolution, and composition for DOPPLER CLI.
 *
 * @module cli/config
 */

export { ConfigResolver, resolveConfig, listPresets as listPresetsFromResolver } from './config-resolver.js';
export type { ResolvedConfig, ResolverOptions } from './config-resolver.js';

export { ConfigComposer, composeConfig } from './config-composer.js';
export type { RawConfig, ComposedConfig } from './config-composer.js';

export { ConfigLoader, loadConfig, listPresets, dumpConfig } from './config-loader.js';
export type { LoadedConfig, LoadOptions } from './config-loader.js';

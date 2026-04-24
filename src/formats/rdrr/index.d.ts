/**
 * RDRR (Recursive DOPPLER Runtime Registry) format module.
 *
 * Provides types, validation, and utilities for the RDRR model format.
 *
 * @module formats/rdrr
 */

// Types and constants
export * from './types.js';

// Tensor classification
export * from './classification.js';

// Validation
export * from './validation.js';

// Parsing
export * from './parsing.js';

// Manifest creation and serialization
export * from './manifest.js';

// Group accessors
export * from './groups.js';

// Integrity + slicing
export * from './merkle.js';
export * from './integrity.js';
export * from './lowerings.js';
export * from './slice-compiler.js';

// Distributed plans
export * from './distributed/index.js';

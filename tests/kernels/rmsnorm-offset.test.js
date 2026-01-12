
import { describe, it } from 'node:test'; // Using node:test as per project style (assumed from context, or will double check)
// Actually, project uses a custom test runner or mocha? 
// Checking package.json would be wise, but I recall seeing `npm test -- --filter` usage.
// Let's assume standard mocha/chai or similar. 
// Wait, internal context says `npm test -- --filter matmul`.
// I'll use the existing test pattern. I should read a test file first to match style.
// But for now, I'll assume I can write a script that imports the kernel and asserts.

import assert from 'node:assert';
import { getDevice } from '../../src/gpu/device.js';
import { runRMSNorm } from '../../src/gpu/kernels/rmsnorm.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { acquireBuffer } from '../../src/gpu/buffer-pool.js';
import { compareTensorsArray } from '../../src/gpu/tensor_utils.js'; // Assuming this exists or I'll implement simple check

// I need to import getDevice first to initialize?
// Usually the test runner handles setup. 

import { getTestDevice } from './utils.js'; // Hypothetcial

// Let's peek at an existing test first to be safe.

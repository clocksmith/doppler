
import { describe, it } from 'node:test';

import assert from 'node:assert';
import { getDevice } from '../../src/gpu/device.js';
import { runRMSNorm } from '../../src/gpu/kernels/rmsnorm.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { acquireBuffer } from '../../src/memory/buffer-pool.js';
import { compareTensorsArray } from '../../src/gpu/tensor_utils.js'; // Assuming this exists or I'll implement simple check

// I need to import getDevice first to initialize?
// Usually the test runner handles setup. 

import { getTestDevice } from './utils.js'; // Hypothetcial

// Let's peek at an existing test first to be safe.

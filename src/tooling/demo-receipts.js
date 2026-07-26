export const DEMO_CONTRACT_RECEIPT_SCHEMA = 'doppler.demo-contract-receipt/v1';
export const DEMO_HARDWARE_RECEIPT_SCHEMA = 'doppler.demo-hardware-receipt/v1';

const CONTRACT_EXECUTION_CLASSES = new Set([
  'mocked-contract',
  'software-webgpu',
  'hardware-webgpu',
]);
const HARDWARE_EXECUTION_CLASSES = new Set([
  'software-webgpu',
  'hardware-webgpu',
]);
const DIGEST_PATTERN = /^sha256:[0-9a-f]{64}$/;

function assertObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
}

function assertDigest(value, label) {
  if (!DIGEST_PATTERN.test(String(value ?? ''))) {
    throw new Error(`${label} must be a sha256 digest.`);
  }
}

function assertBooleanFields(value, fields, label) {
  assertObject(value, label);
  for (const field of fields) {
    if (typeof value[field] !== 'boolean') {
      throw new Error(`${label}.${field} must be boolean.`);
    }
  }
}

function assertGeneration(value, label) {
  assertObject(value, label);
  if (typeof value.outputText !== 'string') {
    throw new Error(`${label}.outputText must be a string.`);
  }
  if (!Array.isArray(value.tokenIds) || value.tokenIds.some((id) => !Number.isInteger(id) || id < 0)) {
    throw new Error(`${label}.tokenIds must contain non-negative integers.`);
  }
  assertDigest(value.transcriptHash, `${label}.transcriptHash`);
}

export function validateDemoContractReceipt(receipt) {
  assertObject(receipt, 'demo contract receipt');
  if (receipt.schema !== DEMO_CONTRACT_RECEIPT_SCHEMA) {
    throw new Error(`Demo contract receipt schema must be ${DEMO_CONTRACT_RECEIPT_SCHEMA}.`);
  }
  if (receipt.status !== 'passed' && receipt.status !== 'failed') {
    throw new Error('Demo contract receipt status must be passed or failed.');
  }
  if (!CONTRACT_EXECUTION_CLASSES.has(receipt.executionClass)) {
    throw new Error('Demo contract receipt executionClass is invalid.');
  }
  if (receipt.entrypoint !== '/demo/index.html') {
    throw new Error('Demo contract receipt must exercise /demo/index.html.');
  }
  assertBooleanFields(
    receipt.journey,
    ['catalogRendered', 'modelSelected', 'modelLoaded', 'generationCompleted'],
    'demo contract receipt journey'
  );
  assertDigest(receipt.shellManifestDigest, 'demo contract receipt shellManifestDigest');
  if (!Array.isArray(receipt.fatalConsoleErrors)) {
    throw new Error('Demo contract receipt fatalConsoleErrors must be an array.');
  }
  return receipt;
}

export function validateDemoHardwareReceipt(receipt) {
  assertObject(receipt, 'demo hardware receipt');
  if (receipt.schema !== DEMO_HARDWARE_RECEIPT_SCHEMA) {
    throw new Error(`Demo hardware receipt schema must be ${DEMO_HARDWARE_RECEIPT_SCHEMA}.`);
  }
  if (!['passed', 'failed'].includes(receipt.status)) {
    throw new Error('Demo hardware receipt status is invalid.');
  }
  if (!HARDWARE_EXECUTION_CLASSES.has(receipt.executionClass)) {
    throw new Error('Demo hardware receipt cannot claim a mocked execution class.');
  }
  assertObject(receipt.browser, 'demo hardware receipt browser');
  assertObject(receipt.adapter, 'demo hardware receipt adapter');
  assertObject(receipt.artifact, 'demo hardware receipt artifact');
  assertDigest(receipt.artifact.manifestHash, 'demo hardware receipt artifact.manifestHash');
  assertGeneration(receipt.online, 'demo hardware receipt online');
  assertGeneration(receipt.offline, 'demo hardware receipt offline');
  assertBooleanFields(
    receipt.lifecycle,
    [
      'allPagesClosed',
      'networkDisabled',
      'persistentCacheRestored',
      'upgradeChecked',
      'partialCacheFailureChecked',
    ],
    'demo hardware receipt lifecycle'
  );
  assertObject(receipt.fingerprint, 'demo hardware receipt fingerprint');
  if (receipt.status === 'passed') {
    for (const [field, value] of Object.entries(receipt.lifecycle)) {
      if (value !== true) {
        throw new Error(`Passed demo hardware receipt requires lifecycle.${field}=true.`);
      }
    }
    if (receipt.online.transcriptHash !== receipt.offline.transcriptHash) {
      throw new Error('Passed demo hardware receipt requires identical online/offline transcripts.');
    }
  }
  return receipt;
}

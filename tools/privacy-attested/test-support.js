/**
 * Software stand-in for the hardware, for tests and the local run only. It signs SEV-SNP-shaped reports with a key made
 * in this process, so it shows the protocol works and says nothing about where the provider runs.
 */
import { generateKeyPairSync, randomBytes, sign } from 'node:crypto';
import { REPORT_BYTES, SIGNED_BYTES } from './snp-report.js';

export const TEST_FORMAT = 'software-test/v1';
const tcbBytes = tcb => Buffer.from([tcb.bootloader, tcb.tee, 0, 0, 0, 0, tcb.snp, tcb.microcode]);

export function buildReport({ reportData, measurement, chipId, tcb, policy = 0x30000n, vmpl = 0, version = 2, signer = 0, hostData = Buffer.alloc(32) }, privateKey) {
  const report = Buffer.alloc(REPORT_BYTES);
  report.writeUInt32LE(version, 0);
  report.writeBigUInt64LE(policy, 8);
  report.writeUInt32LE(vmpl, 0x30);
  report.writeUInt32LE(1, 0x34);
  tcbBytes(tcb).copy(report, 0x38);
  report.writeUInt32LE(signer, 0x48);
  reportData.copy(report, 0x50);
  measurement.copy(report, 0x90);
  hostData.copy(report, 0xc0);
  randomBytes(32).copy(report, 0x140);
  tcbBytes(tcb).copy(report, 0x180);
  chipId.copy(report, 0x1a0);
  tcbBytes(tcb).copy(report, 0x1e0);
  tcbBytes(tcb).copy(report, 0x1f0);
  const signature = sign('sha384', report.subarray(0, SIGNED_BYTES), { key: privateKey, dsaEncoding: 'ieee-p1363' });
  Buffer.from(signature.subarray(0, 48)).reverse().copy(report, SIGNED_BYTES);
  Buffer.from(signature.subarray(48)).reverse().copy(report, SIGNED_BYTES + 72);
  return report;
}

export function createTestChip({ keys = generateKeyPairSync('ec', { namedCurve: 'P-384' }), measurement = randomBytes(48), chipId = randomBytes(64),
  tcb = { bootloader: 4, tee: 0, snp: 22, microcode: 213 } } = {}) {
  const endorsement = Object.freeze({ source: 'software-test', product: 'Milan', publicKey: keys.publicKey, hardwareId: chipId, tcb: Object.freeze({ ...tcb }) });
  return {
    measurement, chipId, tcb, keys,
    endorse: () => endorsement,
    deployment: (overrides = {}) => ({ id: 'software-test', product: 'Milan', measurement: measurement.toString('hex'),
      minimumTcb: { ...tcb }, chipIds: [], hostData: null, ...overrides }),
    attest: (overrides = {}) => async reportData => ({ format: TEST_FORMAT, endorsement: {},
      report: buildReport({ reportData, measurement, chipId, tcb, ...overrides }, keys.privateKey) }),
  };
}

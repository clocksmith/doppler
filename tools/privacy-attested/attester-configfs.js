/**
 * Report source for a Linux SEV-SNP guest, through the kernel's configfs TSM interface. Written from the kernel ABI
 * description and exercised only against a fake file system: it has never run on SEV-SNP hardware.
 */
import * as nodeFs from 'node:fs/promises';
import { randomBytes } from 'node:crypto';
import { SNP_FORMAT } from './evidence.js';
import { REPORT_BYTES, evidenceError } from './snp-report.js';

const VCEK_GUIDS = ['8d75da6364e66445adc5f4b93be8accd', '63da758de6644564adc5f4b93be8accd'];

/** GHCB certificate table: 24-byte entries of GUID, offset and length, ended by an all-zero entry. */
function vcekFrom(table) {
  for (let offset = 0; offset + 24 <= table.length; offset += 24) {
    const entry = table.subarray(offset, offset + 24);
    if (entry.every(byte => byte === 0)) break;
    const start = entry.readUInt32LE(16);
    const length = entry.readUInt32LE(20);
    if (VCEK_GUIDS.includes(entry.subarray(0, 16).toString('hex')) && start + length <= table.length) return Buffer.from(table.subarray(start, start + length));
  }
  return null;
}

/** `vcek` supplies the chip certificate when the host has not provisioned one; it is checked by the client either way. */
export function createConfigfsAttester({ fs = nodeFs, root = '/sys/kernel/config/tsm/report', vcek = null } = {}) {
  return async function attest(reportData) {
    if (reportData.length !== 64) throw evidenceError('attester-input', 'Report data must be 64 bytes.');
    const entry = `${root}/doppler-${randomBytes(8).toString('hex')}`;
    await fs.mkdir(entry);
    try {
      const provider = (await fs.readFile(`${entry}/provider`)).toString('utf8').trim();
      if (provider !== 'sev_guest') throw evidenceError('attester-provider', 'Guest is not an SEV-SNP guest.');
      await fs.writeFile(`${entry}/inblob`, reportData);
      const report = await fs.readFile(`${entry}/outblob`);
      const generation = (await fs.readFile(`${entry}/generation`)).toString('utf8').trim();
      if (generation !== '1') throw evidenceError('attester-race', 'Report request was modified concurrently.');
      if (report.length < REPORT_BYTES) throw evidenceError('attester-output', 'Kernel returned a short report.');
      const certificate = vcek ?? vcekFrom(await fs.readFile(`${entry}/auxblob`));
      if (certificate === null) throw evidenceError('attester-endorsement', 'No VCEK certificate is provisioned; supply one from the AMD key distribution service.');
      return { format: SNP_FORMAT, report: Buffer.from(report.subarray(0, REPORT_BYTES)), endorsement: { vcek: certificate } };
    } finally {
      await fs.rmdir(entry);
    }
  };
}

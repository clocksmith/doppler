/** Evidence appraisal: endorsement, report signature, guest policy, deployment and session binding, in that order. */
import { readFileSync } from 'node:fs';
import { timingSafeEqual, verify } from 'node:crypto';
import { evidenceError, parseSnpReport } from './snp-report.js';
import { endorseVcek, loadAmdRoots } from './snp-endorsement.js';

export const SNP_FORMAT = 'amd-sev-snp/v1';
const TCB_COMPONENTS = ['bootloader', 'tee', 'snp', 'microcode'];
const HEX = { measurement: 96, chipId: 128, hostData: 64 };

function check(ok, code, message) {
  if (!ok) throw evidenceError(code, message);
}

const hex = (value, length) => typeof value === 'string' && value.length === length && /^[a-f0-9]+$/.test(value);
const tcb = value => value !== null && typeof value === 'object' && TCB_COMPONENTS.every(name => Number.isInteger(value[name]) && value[name] >= 0 && value[name] <= 255);

/** A deployment names one launch measurement on one product, with the oldest firmware the requester accepts. */
export function validateTrustPolicy(policy) {
  check(policy?.schema === 'doppler.attested.trust-policy/v1', 'trust-policy', 'Versioned trust policy required.');
  const format = policy.formats?.[SNP_FORMAT];
  check(format !== undefined && Object.keys(policy.formats).length === 1, 'trust-policy', 'Only the SEV-SNP format is installed.');
  check(format.maximumVmpl === 0 && format.requireVcek === true && format.allowDebug === false && format.allowMigrationAgent === false,
    'trust-policy', 'Guest policy floors cannot be relaxed.');
  for (const pins of Object.values(format.products)) {
    check(hex(pins.arkSpkiSha256, 64) && hex(pins.askSpkiSha256, 64) && typeof pins.chain === 'string', 'trust-policy', 'Root key pins required.');
  }
  check(Array.isArray(policy.deployments), 'trust-policy', 'Deployment list required.');
  for (const deployment of policy.deployments) {
    check(typeof deployment.id === 'string' && deployment.id.length > 0 && Object.hasOwn(format.products, deployment.product)
      && hex(deployment.measurement, HEX.measurement) && tcb(deployment.minimumTcb)
      && Array.isArray(deployment.chipIds) && deployment.chipIds.every(id => hex(id, HEX.chipId))
      && (deployment.hostData === null || hex(deployment.hostData, HEX.hostData)), 'trust-policy', 'Malformed deployment.');
  }
  return policy;
}

/** `endorse` is the only seam: the installed verifier uses AMD's chain; tests substitute a pinned software key and are labelled so. */
export function createSnpVerifier({ trustPolicy, endorse }) {
  const policy = validateTrustPolicy(trustPolicy);
  const format = policy.formats[SNP_FORMAT];
  return function verifyEvidence(evidence, expectedReportData, now) {
    check(expectedReportData.length === 64, 'evidence-binding', 'Expected report data must be 64 bytes.');
    const report = parseSnpReport(evidence.report);
    const endorsement = endorse(evidence.endorsement, now);
    check(report.signingKey === 0 && !report.chipKeyMasked, 'evidence-signer', 'Report is not signed by a chip endorsement key.');
    let genuine = false;
    try { genuine = verify('sha384', report.signed, { key: endorsement.publicKey, dsaEncoding: 'ieee-p1363' }, report.signature); } catch { genuine = false; }
    check(genuine, 'evidence-signature', 'Report signature does not verify under the endorsed chip key.');
    check(timingSafeEqual(report.chipId, endorsement.hardwareId), 'evidence-chip', 'Report chip differs from the endorsed chip.');
    check(TCB_COMPONENTS.every(name => report.reportedTcb[name] === endorsement.tcb[name]), 'evidence-tcb', 'Reported TCB differs from the endorsed TCB.');
    check(report.policy.reservedOne && !report.policy.debugAllowed && !report.policy.migrationAgentAllowed,
      'evidence-guest-policy', 'Guest policy permits debugging or a migration agent.');
    check(report.vmpl <= format.maximumVmpl, 'evidence-guest-policy', 'Report was requested from a lower privilege level.');
    const measurement = report.measurement.toString('hex');
    const chipId = report.chipId.toString('hex');
    const candidates = policy.deployments.filter(item => item.product === endorsement.product && item.measurement === measurement);
    check(candidates.length > 0, 'evidence-measurement', 'Launch measurement is not an admitted deployment.');
    const deployment = candidates.find(item => (item.chipIds.length === 0 || item.chipIds.includes(chipId))
      && (item.hostData === null || item.hostData === report.hostData.toString('hex')));
    check(deployment !== undefined, 'evidence-recipient', 'Chip or host data is not an admitted recipient.');
    check(TCB_COMPONENTS.every(name => report.reportedTcb[name] >= deployment.minimumTcb[name]
      && report.committedTcb[name] >= deployment.minimumTcb[name]), 'evidence-tcb', 'Platform firmware is older than the deployment minimum.');
    check(timingSafeEqual(report.reportData, expectedReportData), 'evidence-binding', 'Report is not bound to this session.');
    return Object.freeze({
      format: SNP_FORMAT, endorsement: endorsement.source, product: endorsement.product, deployment: deployment.id,
      measurement, chipId, reportedTcb: report.reportedTcb, reportId: report.reportId.toString('hex'),
      providerIsolationEstablished: endorsement.source === 'amd-vcek-chain',
    });
  };
}

/** The installed verifier set. With no deployments in the policy it admits nothing. */
export function createInstalledVerifiers(directory = new URL('.', import.meta.url)) {
  const trustPolicy = JSON.parse(readFileSync(new URL('trust-policy.json', directory), 'utf8'));
  validateTrustPolicy(trustPolicy);
  const roots = loadAmdRoots(trustPolicy.formats[SNP_FORMAT].products, name => readFileSync(new URL(name, directory), 'utf8'));
  return Object.freeze({
    trustPolicy,
    roots,
    verifiers: Object.freeze({ [SNP_FORMAT]: createSnpVerifier({ trustPolicy, endorse: (endorsement, now) => endorseVcek(endorsement?.vcek, roots, now) }) }),
  });
}

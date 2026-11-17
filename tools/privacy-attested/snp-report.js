/** AMD SEV-SNP attestation report layout (ABI report versions 2 and 3, Milan and Genoa TCB encoding). */

export const REPORT_BYTES = 1184;
export const SIGNED_BYTES = 0x2a0;
const SIGNATURE_COMPONENT = 72;

export function evidenceError(code, message) {
  return Object.assign(new Error(message), { code });
}

function check(ok, code, message) {
  if (!ok) throw evidenceError(code, message);
}

/** Milan/Genoa: bootloader, tee, four reserved bytes, snp, microcode. */
export function decodeTcb(bytes) {
  check(bytes.length === 8 && bytes.subarray(2, 6).every(byte => byte === 0), 'evidence-format', 'Unsupported TCB encoding.');
  return Object.freeze({ bootloader: bytes[0], tee: bytes[1], snp: bytes[6], microcode: bytes[7] });
}

export function decodeGuestPolicy(value) {
  return Object.freeze({
    abiMinor: Number(value & 0xffn),
    abiMajor: Number((value >> 8n) & 0xffn),
    smtAllowed: (value & (1n << 16n)) !== 0n,
    reservedOne: (value & (1n << 17n)) !== 0n,
    migrationAgentAllowed: (value & (1n << 18n)) !== 0n,
    debugAllowed: (value & (1n << 19n)) !== 0n,
    singleSocketRequired: (value & (1n << 20n)) !== 0n,
  });
}

/** P-384 signature: r and s are 72-byte little-endian fields; the high 24 bytes of each must be zero. */
function signature(report) {
  const component = offset => {
    const field = Buffer.from(report.subarray(offset, offset + SIGNATURE_COMPONENT)).reverse();
    check(field.subarray(0, 24).every(byte => byte === 0), 'evidence-format', 'Signature component exceeds P-384.');
    return field.subarray(24);
  };
  const tail = report.subarray(SIGNED_BYTES + 2 * SIGNATURE_COMPONENT);
  check(tail.every(byte => byte === 0), 'evidence-format', 'Signature padding is not zero.');
  return Buffer.concat([component(SIGNED_BYTES), component(SIGNED_BYTES + SIGNATURE_COMPONENT)]);
}

/** Structural parse only. Nothing here establishes that the report is genuine. */
export function parseSnpReport(input) {
  check(input instanceof Uint8Array && input.length === REPORT_BYTES, 'evidence-format', 'Attestation report must be 1184 bytes.');
  const report = Buffer.from(input);
  const version = report.readUInt32LE(0);
  check(version === 2 || version === 3, 'evidence-format', `Unsupported report version ${version}.`);
  check(report.readUInt32LE(0x34) === 1, 'evidence-format', 'Report is not signed with ECDSA P-384 SHA-384.');
  const signer = report.readUInt32LE(0x48);
  const slice = (start, end) => Buffer.from(report.subarray(start, end));
  return Object.freeze({
    version,
    guestSvn: report.readUInt32LE(4),
    policy: decodeGuestPolicy(report.readBigUInt64LE(8)),
    vmpl: report.readUInt32LE(0x30),
    platformInfo: report.readBigUInt64LE(0x40),
    authorKeyEnabled: (signer & 1) !== 0,
    chipKeyMasked: (signer & 2) !== 0,
    signingKey: (signer >> 2) & 7,
    reportData: slice(0x50, 0x90),
    measurement: slice(0x90, 0xc0),
    hostData: slice(0xc0, 0xe0),
    idKeyDigest: slice(0xe0, 0x110),
    authorKeyDigest: slice(0x110, 0x140),
    reportId: slice(0x140, 0x160),
    reportedTcb: decodeTcb(report.subarray(0x180, 0x188)),
    chipId: slice(0x1a0, 0x1e0),
    committedTcb: decodeTcb(report.subarray(0x1e0, 0x1e8)),
    launchTcb: decodeTcb(report.subarray(0x1f0, 0x1f8)),
    signed: slice(0, SIGNED_BYTES),
    signature: signature(report),
  });
}

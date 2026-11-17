/** AMD endorsement: the pinned ARK and ASK, and a chip's VCEK certificate with its hardware ID and TCB extensions. */
import { X509Certificate, createHash } from 'node:crypto';
import { evidenceError } from './snp-report.js';

const AMD_OID = '2b060104019c7801';
const HARDWARE_ID = AMD_OID + '04';
const TCB = { bootloader: AMD_OID + '0301', tee: AMD_OID + '0302', snp: AMD_OID + '0303', microcode: AMD_OID + '0308' };

function check(ok, code, message) {
  if (!ok) throw evidenceError(code, message);
}

/** One DER element at offset: definite lengths only, bounded by the enclosing buffer. */
function element(bytes, offset, limit) {
  check(offset + 2 <= limit, 'endorsement-format', 'Truncated certificate.');
  const tag = bytes[offset];
  let length = bytes[offset + 1];
  let start = offset + 2;
  if (length & 0x80) {
    const count = length & 0x7f;
    check(count >= 1 && count <= 3 && start + count <= limit, 'endorsement-format', 'Unsupported certificate length.');
    length = 0;
    for (let index = 0; index < count; index += 1) length = length * 256 + bytes[start + index];
    start += count;
  }
  check(start + length <= limit, 'endorsement-format', 'Certificate element exceeds its parent.');
  return { tag, start, end: start + length };
}

function children(bytes, parent) {
  const found = [];
  for (let offset = parent.start; offset < parent.end;) {
    const child = element(bytes, offset, parent.end);
    found.push(child);
    offset = child.end;
  }
  return found;
}

/** Map of extension OID (hex of its content bytes) to the raw extnValue octets. */
function extensions(der) {
  const certificate = element(der, 0, der.length);
  check(certificate.tag === 0x30, 'endorsement-format', 'Certificate is not a sequence.');
  const tbs = children(der, certificate)[0];
  const wrapper = children(der, tbs).find(child => child.tag === 0xa3);
  check(wrapper !== undefined, 'endorsement-format', 'Certificate has no extensions.');
  const found = new Map();
  for (const extension of children(der, children(der, wrapper)[0])) {
    const fields = children(der, extension);
    const value = fields.at(-1);
    check(fields[0].tag === 0x06 && value.tag === 0x04, 'endorsement-format', 'Malformed extension.');
    const oid = der.subarray(fields[0].start, fields[0].end).toString('hex');
    check(!found.has(oid), 'endorsement-format', 'Duplicate extension.');
    found.set(oid, der.subarray(value.start, value.end));
  }
  return found;
}

function smallInteger(octets) {
  const integer = element(octets, 0, octets.length);
  const length = integer.end - integer.start;
  check(integer.tag === 0x02 && integer.end === octets.length && length >= 1 && length <= 2, 'endorsement-format', 'Malformed TCB extension.');
  const value = octets.readUIntBE(integer.start, length);
  check(value <= 255, 'endorsement-format', 'TCB component exceeds one byte.');
  return value;
}

const spkiDigest = certificate => createHash('sha256').update(certificate.publicKey.export({ type: 'spki', format: 'der' })).digest('hex');
const within = (certificate, now) => new Date(certificate.validFrom) <= now && now <= new Date(certificate.validTo);

/** Installed roots. Every product's ARK and ASK must match the policy's key pins; a served chain is never trusted on its own. */
export function loadAmdRoots(products, readChain) {
  const roots = new Map();
  for (const [product, pins] of Object.entries(products)) {
    const pems = readChain(pins.chain).match(/-----BEGIN CERTIFICATE-----[\s\S]+?-----END CERTIFICATE-----/g) ?? [];
    check(pems.length === 2, 'endorsement-roots', `${product} chain must hold exactly the ASK and the ARK.`);
    const [ask, ark] = pems.map(pem => new X509Certificate(pem));
    check(spkiDigest(ark) === pins.arkSpkiSha256 && spkiDigest(ask) === pins.askSpkiSha256, 'endorsement-roots', `${product} root keys do not match the installed pins.`);
    check(ark.checkIssued(ark) && ark.verify(ark.publicKey) && ask.checkIssued(ark) && ask.verify(ark.publicKey),
      'endorsement-roots', `${product} chain does not verify.`);
    roots.set(product, Object.freeze({ ark, ask }));
  }
  return roots;
}

/** Returns the chip key AMD endorsed, with the hardware ID and TCB the certificate names. */
export function endorseVcek(vcekDer, roots, now) {
  check(vcekDer instanceof Uint8Array && vcekDer.length > 0 && vcekDer.length <= 4096, 'endorsement-format', 'VCEK certificate required.');
  const der = Buffer.from(vcekDer);
  let vcek;
  try { vcek = new X509Certificate(der); } catch { throw evidenceError('endorsement-format', 'VCEK certificate does not parse.'); }
  const issuer = [...roots].find(([, root]) => vcek.checkIssued(root.ask) && vcek.verify(root.ask.publicKey));
  check(issuer !== undefined, 'endorsement-chain', 'VCEK is not signed by an installed AMD signing key.');
  const [product, root] = issuer;
  check(within(vcek, now) && within(root.ask, now) && within(root.ark, now), 'endorsement-validity', 'Endorsement certificate is outside its validity period.');
  check(/(^|\n)CN=SEV-VCEK(\n|$)/.test(vcek.subject), 'endorsement-chain', 'Certificate is not a VCEK.');
  check(vcek.publicKey.asymmetricKeyType === 'ec' && vcek.publicKey.asymmetricKeyDetails.namedCurve === 'secp384r1',
    'endorsement-format', 'VCEK key is not P-384.');
  const found = extensions(der);
  const hardwareId = found.get(HARDWARE_ID);
  check(hardwareId !== undefined && hardwareId.length === 64, 'endorsement-format', 'VCEK hardware ID missing.');
  const tcb = {};
  for (const [name, oid] of Object.entries(TCB)) {
    check(found.has(oid), 'endorsement-format', `VCEK ${name} version missing.`);
    tcb[name] = smallInteger(found.get(oid));
  }
  return Object.freeze({ source: 'amd-vcek-chain', product, publicKey: vcek.publicKey, hardwareId: Buffer.from(hardwareId), tcb: Object.freeze(tcb) });
}

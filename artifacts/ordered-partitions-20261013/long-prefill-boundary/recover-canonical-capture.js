// Read-only observation export; bypass large inspector result serialization.
import { connectNativeInspector } from '/var/tmp/reploid-cold-compile-three-stage-20261013/tests/fixtures/native-partition-endpoint.js';
const [endpoint, output] = process.argv.slice(2);
if (!endpoint || !output.startsWith('/var/tmp/doppler-long-prefill-boundary-20261013/')) throw Error('Explicit owned output required');
const c = await connectNativeInspector(endpoint);
try {
  const fn = output => {
    const fs = process.getBuiltinModule('fs'), zlib = process.getBuiltinModule('zlib');
    const crypto = process.getBuiltinModule('crypto');
    if (stageCaptureError) throw Error(stageCaptureError);
    let byteLength = 0;
    const records = stageCaptureRecords.filter(row => row.capturePolicy === 'full').map((row, ordinal) => {
      const { capture, ...metadata } = row;
      if (capture.error || !Array.isArray(capture.storageComponents)) throw Error('Missing paired canonical snapshot');
      const data = Buffer.from(Float32Array.from(capture.storageComponents).buffer);
      byteLength += data.byteLength;
      if (byteLength > 33554432) throw Error('Snapshot byte budget exceeded');
      const storageDigest = 'sha256:' + crypto.createHash('sha256').update(data).digest('hex');
      if (storageDigest !== capture.storageDigest) throw Error('Snapshot digest mismatch');
      return { ...metadata, ordinal, layer: row.layerIdx, shape: capture.shape, precision: capture.precision,
        components: capture.components, elementStrideBytes: capture.elementStrideBytes,
        byteLength: data.byteLength, storageDigest, data: data.toString('base64') };
    });
    const raw = Buffer.from(JSON.stringify({ scope: 'Readonly recovery of existing completed canonical snapshots', records }));
    fs.writeFileSync(output, zlib.gzipSync(raw), { flag: 'wx' });
    return { output, records: records.length, byteLength, rawSha256: crypto.createHash('sha256').update(raw).digest('hex') };
  };
  const r = await c.send('Runtime.evaluate', { expression: `(${fn.toString()})(${JSON.stringify(output)})`, returnByValue: true });
  if (r.exceptionDetails) throw Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  console.log(JSON.stringify(r.result.value));
} finally { await c.detach(); }

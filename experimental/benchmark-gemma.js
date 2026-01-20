import fs from 'node:fs';
import { execSync } from 'node:child_process';
import path from 'node:path';

const EXP_DIR = 'experimental';
const TARGET_FILE = path.join(EXP_DIR, 'gemma-chunk.bin');

const algos = [
  { name: 'Gzip (Default)', cmd: 'gzip -k -f', ext: '.gz' },
  { name: 'LZ4 (Fast)', cmd: 'lz4 -f', ext: '.lz4' },
  { name: 'Zstd (Default)', cmd: 'zstd -f -k', ext: '.zst' },
  { name: 'Zstd (Ultra)', cmd: 'zstd -f -k --ultra -22', ext: '.zst.ultra' },
  { name: 'XZ (LZMA2)', cmd: 'xz -k -f', ext: '.xz' }
];

console.log('Running Compression Benchmark on Gemma Model Chunk (100MB)...');
console.log('');

if (!fs.existsSync(TARGET_FILE)) {
    console.error(`Missing target file: ${TARGET_FILE}`);
    process.exit(1);
}

const baseSize = fs.statSync(TARGET_FILE).size;
console.log(`Target: ${TARGET_FILE} [${(baseSize / 1024 / 1024).toFixed(2)} MB]`);

for (const algo of algos) {
    const startTime = process.hrtime();
    let outFile = `${TARGET_FILE}${algo.ext}`;
    let command = `${algo.cmd} "${TARGET_FILE}"`;
    
    if (algo.name.includes('Ultra')) {
       command = `zstd -f --ultra -22 "${TARGET_FILE}" -o "${outFile}"`;
    } else if (algo.name.includes('LZ4')) {
        command = `lz4 -f "${TARGET_FILE}" "${outFile}"`;
    }

    try {
      execSync(command);
      const endTime = process.hrtime(startTime);
      const duration = (endTime[0] + endTime[1] / 1e9).toFixed(3);
      
      const compSize = fs.statSync(outFile).size;
      const ratio = (baseSize / compSize).toFixed(3);
      const savings = (((baseSize - compSize) / baseSize) * 100).toFixed(2);
      
      console.log(`  [${algo.name.padEnd(14)}] Size: ${(compSize / 1024 / 1024).toFixed(2)} MB | Ratio: ${ratio}x | Savings: ${savings}% | Time: ${duration}s`);
      
      // Cleanup
      fs.unlinkSync(outFile);
      
    } catch (e) {
      console.error(`  [${algo.name}] Failed: ${e.message}`);
    }
}

import fs from 'node:fs';
import { execSync } from 'node:child_process';
import path from 'node:path';

const EXP_DIR = 'experimental';
const RAW_FILE = path.join(EXP_DIR, 'raw.bin'); // The shard copy
const MORPHED_FILE = path.join(EXP_DIR, 'morphed.bin'); // The tanh-morphed copy

// Ensure source files exist (re-create if needed or just assume from previous steps)
// We assume raw.bin and morphed.bin exist from previous turns.

const algos = [
  { name: 'Gzip (Default)', cmd: 'gzip -k -f', ext: '.gz' },
  { name: 'LZ4 (Fast)', cmd: 'lz4 -f', ext: '.lz4' },
  { name: 'Zstd (Default)', cmd: 'zstd -f -k', ext: '.zst' },
  { name: 'Zstd (Ultra)', cmd: 'zstd -f -k --ultra -22', ext: '.zst.ultra' }, // suffix for unique filename
  { name: 'XZ (LZMA2)', cmd: 'xz -k -f', ext: '.xz' }
];

const targets = [
  { name: 'RAW (AI Weights)', path: RAW_FILE },
  { name: 'MORPHED (Tanh Noise)', path: MORPHED_FILE }
];

console.log('Running Compression Benchmark...\n');

const results = {};

for (const target of targets) {
  if (!fs.existsSync(target.path)) {
    console.error(`Missing target file: ${target.path}`);
    continue;
  }

  const baseSize = fs.statSync(target.path).size;
  console.log(`--- Benchmarking ${target.name} [${(baseSize / 1024 / 1024).toFixed(2)} MB] ---`);
  
  results[target.name] = [];

  for (const algo of algos) {
    const startTime = process.hrtime();
    
    // Construct command
    // Special handling for output naming if needed, but mostly standard
    let outFile = `${target.path}${algo.ext}`;
    let command = `${algo.cmd} "${target.path}"`;
    
    if (algo.name.includes('Ultra')) {
       // zstd -k outputs to .zst automatically, we need to rename or specify output
       // zstd syntax: zstd input -o output
       command = `zstd -f --ultra -22 "${target.path}" -o "${outFile}"`;
    } else if (algo.name.includes('LZ4')) {
        // lz4 syntax: lz4 input output
        command = `lz4 -f "${target.path}" "${outFile}"`;
    }

    try {
      execSync(command);
      const endTime = process.hrtime(startTime);
      const duration = (endTime[0] + endTime[1] / 1e9).toFixed(3);
      
      const compSize = fs.statSync(outFile).size;
      const ratio = (baseSize / compSize).toFixed(3);
      const savings = (((baseSize - compSize) / baseSize) * 100).toFixed(2);
      
      console.log(`  [${algo.name.padEnd(14)}] Size: ${(compSize / 1024 / 1024).toFixed(2)} MB | Ratio: ${ratio}x | Savings: ${savings}% | Time: ${duration}s`);
      
      results[target.name].push({
        algo: algo.name,
        sizeMB: (compSize / 1024 / 1024).toFixed(2),
        ratio: ratio,
        savings: savings,
        time: duration
      });

      // Cleanup
      fs.unlinkSync(outFile);
      
    } catch (e) {
      console.error(`  [${algo.name}] Failed: ${e.message}`);
    }
  }
  console.log('');
}

import fs from 'node:fs';
import { execSync } from 'node:child_process';
import path from 'node:path';
import os from 'node:os';

const MODEL_DIR = 'models/google-gemma-3-1b-it-wf16';
const EXP_DIR = 'experimental';
const REPORT_FILE = path.join(EXP_DIR, 'compression_report.json');

// Get all shard files
const files = fs.readdirSync(MODEL_DIR)
  .filter(f => f.startsWith('shard_') && f.endsWith('.bin'))
  .sort();

console.log(`Found ${files.length} shards in ${MODEL_DIR}`);

const results = [];

for (const file of files) {
  const inputPath = path.join(MODEL_DIR, file);
  // We'll output the .gz to a temp location to avoid cluttering the model dir
  const tempGzPath = path.join(os.tmpdir(), `${file}.gz`);
  
  try {
    // -c writes to stdout, > redirects to file. -1 for fastest speed (sufficient for rough check)
    // or keep default (-6) for fair comparison to previous manual test. Let's use default.
    execSync(`gzip -c "${inputPath}" > "${tempGzPath}"`);
    
    const originalSize = fs.statSync(inputPath).size;
    const compressedSize = fs.statSync(tempGzPath).size;
    const ratio = originalSize / compressedSize;
    const savings = ((originalSize - compressedSize) / originalSize) * 100;

    console.log(`Processed ${file}: ${savings.toFixed(2)}% savings (${ratio.toFixed(2)}x)`);
    
    results.push({
      shard: file,
      originalBytes: originalSize,
      compressedBytes: compressedSize,
      ratio: Number(ratio.toFixed(4)),
      savingsPercent: Number(savings.toFixed(2))
    });

    // Cleanup
    fs.unlinkSync(tempGzPath);

  } catch (err) {
    console.error(`Error processing ${file}:`, err.message);
  }
}

fs.writeFileSync(REPORT_FILE, JSON.stringify(results, null, 2));
console.log(`\nReport saved to ${REPORT_FILE}`);

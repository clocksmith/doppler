import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import path from 'node:path';

const execFileAsync = promisify(execFile);

function extractTopLevelJsonObjects(text) {
  const source = String(text == null ? '' : text);
  const candidates = [];
  let inString = false;
  let escaped = false;
  let depth = 0;
  let start = -1;

  for (let i = 0; i < source.length; i += 1) {
    const char = source[i];

    if (inString) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (char === '\\') {
        escaped = true;
        continue;
      }
      if (char === '"') {
        inString = false;
      }
      continue;
    }

    if (char === '"') {
      inString = true;
      continue;
    }
    if (char === '{') {
      if (depth === 0) start = i;
      depth += 1;
      continue;
    }
    if (char === '}') {
      if (depth > 0) {
        depth -= 1;
        if (depth === 0 && start >= 0) {
          candidates.push(source.slice(start, i + 1));
          start = -1;
        }
      }
    }
  }

  return candidates;
}

function parseJsonFromStdout(stdout, label) {
  const normalized = String(stdout == null ? '' : stdout);
  if (!normalized.trim()) {
    throw new Error(`No output to parse for ${label}`);
  }

  const tryParse = (candidate) => {
    try {
      const parsed = JSON.parse(candidate);
      return parsed;
    } catch {
      return null;
    }
  };

  const direct = tryParse(normalized.trim());
  if (direct !== null) return direct;

  const candidates = extractTopLevelJsonObjects(normalized);
  for (let i = candidates.length - 1; i >= 0; i -= 1) {
    const parsed = tryParse(candidates[i]);
    if (parsed !== null) return parsed;
  }

  throw new Error(`Could not parse JSON payload from ${label}`);
}

// E2E NaN Regression Test for Gemma 3 F16a/F32a Stability
async function runTest() {
    const modelId = process.env.MODEL_ID || 'gemma-3-270m-it-f16-f32a';
    console.log(`[NaN Regression] Testing model: ${modelId}`);

    try {
        const { stdout, stderr } = await execFileAsync(
            'node',
            [
                'tools/doppler-cli.js',
                'debug',
                '--model-id', modelId,
                '--model-url', `/models/converted/${modelId}`,
                '-p', 'Briefly explain why the sky is blue in one sentence.',
                '--chat', 'gemma',
                '--json'
            ],
            {
                cwd: path.resolve(process.cwd()),
                maxBuffer: 50 * 1024 * 1024
            }
        );

        const jsonParsed = parseJsonFromStdout(stdout, `NaN regression test ${modelId}`);

        assert.equal(jsonParsed.ok, true, 'Command should execute successfully');

        const outputText = jsonParsed.result?.output;
        if (!outputText) {
            throw new Error('No output text was generated');
        }

        // The primary test: Ensure the text did not collapse into NaNs
        if (outputText.includes('NaN')) {
            throw new Error(`Output contains NaN! Model collapsed.\nOutput: ${outputText}`);
        }

        console.log('[NaN Regression] Test passed! Output generated without NaNs.');
        console.log(`[NaN Regression] Output: ${outputText.trim()}`);
    } catch (error) {
        console.error('[NaN Regression] Test failed:', error.message);
        if (error.stdout) console.error('STDOUT:', error.stdout);
        if (error.stderr) console.error('STDERR:', error.stderr);
        process.exit(1);
    }
}

runTest();

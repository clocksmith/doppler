import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import path from 'node:path';

const execFileAsync = promisify(execFile);

// E2E NaN Regression Test for Gemma 3 F16a/F32a Stability
async function runTest() {
    const modelId = process.env.MODEL_ID || 'gemma-3-270m-it-f16-f32a-wf16';
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

        const match = stdout.match(/\{[\s\S]*\}$/m) || stdout.match(/\{[\s\S]*\}/);
        let jsonParsed = null;
        if (match) {
            try {
                jsonParsed = JSON.parse(match[0]);
            } catch {
                // failed to parse
            }
        }

        if (!jsonParsed) {
            throw new Error(`Failed to parse JSON output: ${stdout.slice(-1000)}`);
        }

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

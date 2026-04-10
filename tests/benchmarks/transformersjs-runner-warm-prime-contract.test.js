import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const runnerPath = new URL('../../benchmarks/runners/transformersjs-runner.html', import.meta.url);
const runnerSource = await fs.readFile(runnerPath, 'utf8');

assert.match(runnerSource, /const prompt = typeof config\?\.prompt === 'string' \? config\.prompt : '';/);
assert.match(runnerSource, /const useChatTemplate = config\?\.useChatTemplate === true;/);
assert.match(runnerSource, /const primeMaxNewTokens = Number\.isFinite\(Number\(config\?\.maxNewTokens\)\)/);
assert.match(runnerSource, /const LOCAL_CHAT_TEMPLATE_CACHE = new Map\(\);/);
assert.match(runnerSource, /const chatTemplateUrl = `\$\{localModelPath\.replace\(\/\\\/\$\/, ''\)\}\/\$\{modelId\}\/chat_template\.jinja`;/);
assert.match(runnerSource, /LOCAL_CHAT_TEMPLATE_CACHE\.set\(modelId, loadedChatTemplate\);/);
assert.match(runnerSource, /const generationPrompt = await renderGenerationPrompt\(load\.generator, modelId, prompt, useChatTemplate\);/);
assert.match(runnerSource, /await load\.generator\(generationPrompt, \{/);
assert.match(runnerSource, /primeMode: generationPrimed \? 'load-and-generate' : 'load-only'/);

console.log('transformersjs-runner-warm-prime-contract.test: ok');

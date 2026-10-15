import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';

const endpoint = process.argv[2];
const output = process.argv[3];
assert(endpoint && output, 'Usage: node smoke.js <localhost endpoint> <receipt.json>');
assert(['127.0.0.1', 'localhost'].includes(new URL(endpoint).hostname));
const model = 'qwen3.8-27b-pinned-q8';
const results = [];
async function answer(messages) {
  const start = performance.now();
  const response = await fetch(`${endpoint}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, messages, temperature: 0, seed: 42, max_tokens: 128,
      chat_template_kwargs: { enable_thinking: false } }),
    signal: AbortSignal.timeout(180_000),
  });
  assert.equal(response.status, 200, await response.clone().text());
  const body = await response.json();
  const content = body.choices[0].message.content;
  const result = { messages: structuredClone(messages), content, finishReason: body.choices[0].finish_reason,
    usage: body.usage, timings: body.timings, elapsedMs: performance.now() - start };
  results.push(result);
  await writeFile(output, JSON.stringify({ model, complete: false, results }, null, 2));
  assert.equal(body.choices[0].finish_reason, 'stop', 'Baseline answer must complete');
  assert(!content.includes('\uFFFD'), 'Unexpected replacement character');
  return content;
}
const messages = [{ role: 'user', content: 'Use only this fictional catalog. Cedar: English, 100 minutes, gentle comedy, no violence. Storm: English, 130 minutes, violent thriller. Lumiere: French, 95 minutes, gentle comedy. We require English, no violence, and at most 110 minutes. Pick exactly one title and briefly explain which requirements it meets.' }];
const first = await answer(messages);
assert(first.includes('Cedar'), 'Must choose the only eligible movie');
messages.push({ role: 'assistant', content: first }, { role: 'user', content: 'What is the runtime of the movie you picked? Answer in one sentence.' });
const followup = await answer(messages);
assert(/100/.test(followup), 'Follow-up must retain supplied runtime');
const arithmetic = await answer([{ role: 'user', content: 'What is 17 + 25? Answer with the number only.' }]);
assert.equal(arithmetic.trim(), '42');
await writeFile(output, JSON.stringify({ model, complete: true, scope: 'Single-machine functional smoke; not discovery quality or matched performance qualification', results }, null, 2));
console.log(JSON.stringify({ passed: results.length, answers: results.map(x => x.content), elapsedMs: results.map(x => x.elapsedMs) }, null, 2));

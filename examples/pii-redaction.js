/**
 * In-browser PII redaction example for doppler-gpu.
 *
 * Loads a small instruction-tuned model and uses structured prompting
 * to detect and redact personally identifiable information from text.
 * All inference runs locally — no data leaves the browser.
 */

import { DopplerProvider } from 'doppler-gpu/provider';

const MODEL_ID = 'gemma-3-270m-it-wq4k-ef16';
const MODEL_URL =
  'https://huggingface.co/Clocksmith/rdrr/resolve/HEAD/models/gemma-3-270m-it-wq4k-ef16';

const REDACTION_PROMPT = `You are a PII redaction assistant. Given the following text, identify all personally identifiable information (names, emails, phone numbers, addresses, SSNs, dates of birth, account numbers) and return the text with each PII entity replaced by its category in square brackets.

Example:
Input: "John Smith called from 555-0123 about his account 4532-1234."
Output: "[NAME] called from [PHONE] about his account [ACCOUNT_NUMBER]."

Now redact the following text:

`;

async function redact(text) {
  const pipeline = DopplerProvider.getPipeline();
  const prompt = REDACTION_PROMPT + text;

  let result = '';
  for await (const token of pipeline.generate(prompt, {
    maxTokens: 512,
    temperature: 0.1,
    topP: 0.9,
  })) {
    result += token;
  }
  return result.trim();
}

async function main() {
  console.log('Initializing Doppler...');
  await DopplerProvider.init();

  console.log('Loading model (cached after first download)...');
  await DopplerProvider.loadModel(MODEL_ID, MODEL_URL, (progress) => {
    if (progress.percent != null) {
      console.log(`  ${progress.stage}: ${Math.round(progress.percent)}%`);
    }
  });

  // Example clinical note with PII.
  const note = `
    Patient Jane Doe (DOB: 03/15/1987, MRN: 847291) presented on 2026-03-01
    with complaints of recurring headaches. Contact: jane.doe@email.com,
    (555) 867-5309. Referred by Dr. Robert Chen at Springfield Medical Group.
    Insurance ID: BC-9928-4471.
  `.trim();

  console.log('\n--- Original ---');
  console.log(note);

  console.log('\n--- Redacted ---');
  const redacted = await redact(note);
  console.log(redacted);

  await DopplerProvider.destroy();
}

main().catch(console.error);

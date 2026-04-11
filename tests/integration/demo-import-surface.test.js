import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const downloadsModule = readFileSync(new URL('../../demo/ui/downloads/index.js', import.meta.url), 'utf8');
const minimalChat = readFileSync(new URL('../../examples/minimal-chat.js', import.meta.url), 'utf8');
const piiRedaction = readFileSync(new URL('../../examples/pii-redaction.js', import.meta.url), 'utf8');
const browserChat = readFileSync(new URL('../../examples/browser-chat.html', import.meta.url), 'utf8');

assert.doesNotMatch(downloadsModule, /from '@doppler\/core'/);
assert.match(downloadsModule, /from '@simulatte\/doppler'/);
assert.match(minimalChat, /from '@simulatte\/doppler'/);
assert.match(piiRedaction, /from '@simulatte\/doppler'/);
assert.match(browserChat, /"@simulatte\/doppler": "https:\/\/d4da\.com\/src\/index-browser\.js"/);
assert.match(browserChat, /import \{ doppler \} from '@simulatte\/doppler'/);

console.log('demo-import-surface.test: ok');

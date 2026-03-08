import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const downloadsModule = readFileSync(new URL('../../demo/ui/downloads/index.js', import.meta.url), 'utf8');
const minimalChat = readFileSync(new URL('../../examples/minimal-chat.js', import.meta.url), 'utf8');
const piiRedaction = readFileSync(new URL('../../examples/pii-redaction.js', import.meta.url), 'utf8');
const browserChat = readFileSync(new URL('../../examples/browser-chat.html', import.meta.url), 'utf8');

assert.doesNotMatch(downloadsModule, /from '@doppler\/core'/);
assert.match(downloadsModule, /from '@simulatte\/doppler'/);
assert.match(minimalChat, /from '@simulatte\/doppler\/provider'/);
assert.match(piiRedaction, /from '@simulatte\/doppler\/provider'/);
assert.match(browserChat, /"@simulatte\/doppler\/provider": "https:\/\/d4da\.com\/src\/client\/doppler-provider\.js"/);
assert.match(browserChat, /import \{ DopplerProvider \} from '@simulatte\/doppler\/provider'/);

console.log('demo-import-surface.test: ok');

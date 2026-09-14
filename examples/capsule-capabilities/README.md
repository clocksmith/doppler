# Installed Capsule capabilities

Install the exact accepted archive in an ordinary JavaScript application:

```sh
npm install ./doppler-gpu-0.6.1.tgz
```

Copy `app.js` and explicitly adopt a versioned model descriptor from the release
acceptance bundle. The descriptor includes its signed Capsule URL, trust roots,
accepted plan, application binding, request and limits. It declares one operation;
generation, embedding and reranking use their corresponding qualified models.

```js
import { runCapability } from './app.js';

const descriptor = await fetch('./model.json').then(response => response.json());
const controller = new AbortController();
document.querySelector('#stop').onclick = () => controller.abort();
const completed = await runCapability(descriptor, {
  signal: controller.signal,
  persistReleaseCheckpoint: checkpoint => {
    localStorage.setItem(`release-checkpoint:${descriptor.capsuleUrl}`, JSON.stringify(checkpoint));
  },
  onProgress: progress => console.log(progress),
  onEvent: event => {
    if (event.status === 'partial') console.log(event.output);
  },
});
console.log(completed.output, completed.receipt);
```

The application uses `doppler-gpu/host` and `openCapsule()`. A browser bundler
resolves the package's browser export. Cleanup awaits the session's close on
success, cancellation or failure. Stream callbacks provide backpressure and do
not reload the page. A version 1 partial event is a complete snapshot; completion
is required before treating output as accepted.
Capsule v3 requires the application to persist release checkpoints; persistence
errors must propagate. On subsequent opens, supply the saved checkpoint through
the descriptor's release policy rather than discarding its accepted history.

`node tools/check-packed-package.js --retain /absolute/new-bundle` builds one
candidate and copies standalone contract consumers plus signed fixture data into
that bundle. Reploid consumes the same installed bytes with
`DOPPLER_TEST_CONSUMER=/absolute/new-bundle/consumer npm run test:doppler-consumer`.
Those injected contract fixtures test API wiring; physical model receipts are
recorded separately in the release acceptance bundle.

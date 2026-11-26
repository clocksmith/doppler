# Installed Capsule capabilities

`runCapability()` intentionally opens a session for one operation and closes it
afterward. For repeated queries with resident models and a document index, see the
[document-search application](../document-search/README.md); this example stays
one-shot and must not become a second search engine.

`app.js` imports `doppler-gpu/host`, so it needs `doppler-gpu`, the package's
former name. Install the published `doppler-gpu@0.6.1` from npm in an ordinary
JavaScript application:

```sh
npm install doppler-gpu@0.6.1
```

That registry tarball has SHA-256
`96d2699a3e677815890f804459c023bea4fb2e2a1a59d157b5b6a04c9e573d5f`, the
"Published npm 0.6.1" archive in the
[installed-consumer baseline](../../artifacts/installed-consumers-2026-09-12/README.md),
where the standalone and Reploid contract consumers passed. It exports `./host`
with `openCapsule()`. The same baseline tested a separately built archive that is
not interchangeable with the npm one. The current package is `dopplerx`
([getting started](../../docs/getting-started.md)); using it requires changing the
import in `app.js` to `dopplerx/host`, which has not been checked for this example.

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

To run the standalone contract consumers against a candidate build, retain an
installed-package bundle as described in
[document-search ENGINEERING.md](../document-search/ENGINEERING.md#retain-an-installed-package-bundle).
Reploid consumes the same installed bytes with
`DOPPLER_TEST_CONSUMER=/absolute/new-bundle/consumer npm run test:doppler-consumer`.
Those injected contract fixtures test API wiring; physical model receipts are
recorded separately in the release acceptance bundle.

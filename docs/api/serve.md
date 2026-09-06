# Pack HTTP adapter

## Purpose

Serve an application-opened Pack session without another model loader or inference
implementation. `POST /v1/operations` accepts the same versioned request as
`session.executeOperation()` and streams its unchanged events as newline-delimited
JSON. `GET /v1/model` reports the pinned Pack, selected TargetPlan, and operations
qualified on that session's actual surface.

## Import path and audience

`doppler-gpu/serve` is for Node application authors exposing a local inference
session over HTTP. It imports no model host, converter, discovery, GPU provider,
or network coordinator. Browser applications call their local session directly;
this server adapter requires Node HTTP interfaces.

## Stability

Experimental transport adapter. Connected tests cover generation, text embeddings,
reranking, and sequence encoding with synthetic programs. An available operation
adapter does not qualify a model or a physical device. Forecasting and remote
delegation are not admitted by this HTTP contract.

The existing `doppler-serve` executable remains the **compatibility** chat server.
Its `/v1/chat/completions` contract is not this Pack endpoint. This change does not
claim OpenAI compatibility or finish migration of the compatibility CLI.

## Primary exports and minimal example

```js
import http from 'node:http';
import { randomBytes } from 'node:crypto';
import { openPack } from 'doppler-gpu/host';
import { createPackServeHandler } from 'doppler-gpu/serve';

// Application inputs: exact Pack location, signer/adoption policy, and JSON
// serving policy. Pack v3 still needs durable release-checkpoint persistence.
const session = await openPack(packLocation, applicationTrustOptions);
const token = randomBytes(32).toString('hex');
const handler = createPackServeHandler({ session, policy: servingPolicy, token });
const server = http.createServer(handler);
server.listen(8080, '127.0.0.1');

async function stop() {
  await handler.close();
  await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  await session.close();
}
```

Choose resource limits for the actual workload. Every field is required; no
sampling, precision, model, signing, or memory defaults are introduced here:

```json
{
  "schema": "doppler.pack-serve/v1",
  "maxRequestBytes": 65536,
  "maxResponseBytes": 4194304,
  "maxOutputBytes": 1048576,
  "maxDurationMs": 60000,
  "allowedOrigins": []
}
```

Provide the bearer token to the authorized client through application-owned
custody, not a URL or public log. All non-preflight requests, including model
inspection, require `Authorization: Bearer <token>`. Browser origins must match
an explicit HTTP(S) origin; wildcards and opaque origins are rejected. An empty
list rejects requests carrying `Origin`, while authenticated non-browser clients
remain usable. Preflight grants no execution authority.

Bind loopback for local applications. Authentication and origin filtering do not
provide TLS, multi-user isolation, admission by memory, or an internet deployment
security policy. Do not expose the listener publicly without those application
controls. This module never creates a listener by itself.

Send an unchanged operation request, for example a rerank request:

```js
const request = {
  schema: 'doppler.pack-operation-request/v1',
  operation: { name: 'rerank', version: 1 },
  input: { application: acceptedRelease.application, query, documents },
  options: {},
  assignment: null,
  limits: { maxInputBytes: 65536, maxOutputBytes: 1048576,
    deadlineAt: Date.now() + 30000 },
};
const response = await fetch('http://127.0.0.1:8080/v1/operations', {
  method: 'POST', signal: controller.signal,
  headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
  body: JSON.stringify(request),
});
```

Clients must frame response bytes by newlines (network chunks are not event
boundaries), require a completed event, and preserve each event's request and
output identity. Partial output, a successful HTTP status, signatures, or schema
validity do not establish requester acceptance or model correctness. The endpoint
does not invent chat formatting or change operation-specific output shapes.

## Limits, failure, and ownership

- The handler snapshots its serving policy. Oversized bodies, unknown operations,
  unsupported qualifications, delegated assignments, and excess caller limits
  fail before model invocation. Signed application binding remains checked by
  the session; the server never manufactures that identity for a caller.
- One HTTP operation occupies the session from body intake through iterator
  cleanup. Overlap returns `409 SESSION_BUSY`; there is no hidden queue. Dedicate
  one session to one handler; do not concurrently invoke its direct methods.
- Body intake, execution, and blocked output writes share a serving deadline.
  Client limits are validated, not silently clamped. Caller cancellation and
  disconnection abort future work; already-submitted GPU work is not preempted.
- `maxOutputBytes` bounds each cumulative operation output; `maxResponseBytes`
  also bounds repeated partial events and transport diagnostics. No complete
  transcript is buffered by the HTTP layer. Backpressure pauses event consumption.
- Pre-stream errors use an HTTP status plus `{ error: { code, message } }`.
  Errors after streaming terminate with that error line when it fits the remaining
  response budget. Otherwise the stream ends without completion. Runtime error
  codes are retained when supplied; no success receipt is created on failure.
- `handler.close()` rejects new work, aborts active work, and awaits its cleanup.
  It does **not** close the borrowed session or the HTTP server. The application
  closes those resources and decides whether to activate a replacement Pack.

## Code pointers and related surfaces

- [HTTP adapter](../../src/cli/serve/pack-handler.js)
- [Policy schema](../../src/config/pack-serve.schema.json)
- [Connected HTTP tests](../../tests/integration/pack-serve.test.js)
- [Pack runtime and operations](root.md)
- [Compatibility facade](compat.md)

# Authenticated research evaluator interface

The all-linear client can use an explicitly configured OpenSSH endpoint instead of spawning its evaluator locally. This interface installs nothing and transfers no repository source. It does not authorize any connection or deployment to Jupiter, and the prior Jupiter source-transfer denial remains unresolved.

The client requires a locally supplied route manifest, an explicit private SSH identity file and a requester operator allowlist. The manifest pins one Ed25519 host key, host/port/account, exact base model/plan/protocol and pre-provisioned interpreter/provider/cache/manifest paths. Its immutable identity includes the SSH policy and is bound into every encrypted frame. Endpoint, key, participant or executable-path changes produce a different route; no local fallback or retry is selected after failure.

Ambient SSH configuration, known-host files, agents, forwarding and password/interactive authentication are disabled. The public host pin lives in an owned temporary directory until SSH exits. A host key must be obtained and authenticated independently; this interface does not fetch a key from the endpoint and trust it on first use. Private keys are not created or copied by the deployment interface.

On an independently authorized, pre-provisioned endpoint, place the same route manifest at its declared `remoteManifest` path and the same validated public model cache at `remoteCache`. The SSH command starts the installed provider with the manifest identity. Both ends must currently match the exact implementation versions recorded in the model plan; no cross-platform dependency variant is silently substituted.

The client invocation adds:

```sh
--ssh-deployment /client/trusted-route.json \
--ssh-identity-file /client/private-identity \
--allowed-operator research-provider
```

`ssh-policy.json` and `SSHRoute` own the manifest contract. The manifest fields are `schema`, `operatorId`, `host`, `port`, `user`, `hostKey` (`algorithm`, `base64`), `basePlanId`, `modelId`, `backendProtocol`, `remotePython`, `remoteProvider`, `remoteCache` and `remoteManifest`. Unknown fields, duplicate JSON keys, unsupported keys/protocols and malformed paths fail before evaluator startup/key export. No attestation flag is accepted.

The isolated transport workflow exercises a real loopback SSH daemon with ephemeral fixture keys, opaque framing, wrong-host-key rejection and wrong-client-key rejection. It does not establish separate-machine model execution, endpoint software appraisal, hardware attestation or production cryptographic privacy. SSH authenticates an endpoint/account; it does not attest its application, accelerators or complete downstream execution. The existing pre-decryption replay remains required. Production admission stays closed.

Direct transport launch rechecks the requester allowlist, base plan/model binding and private identity file; callers cannot bypass admission by skipping bridge construction. Client evidence carries the bound execution-route identity and actual frame protocol. The constructed public-fixture artifact may include only the route digest and a closed transport-kind label, never the endpoint manifest or credential path.

## Explicit version-two research profiles

Version-one manifests still accept only their original stock full-Qwen protocol.
They cannot acquire another backend by adding a field or changing a protocol. A
separate `doppler.encrypted-ssh-route/v2` manifest requires `backendProfile` as well
as `backendProtocol`; both must identify the same installed Qwen research route.
The supported profiles are those registered by `plan_backends.qwen_backend`,
including the separate NTT, higher-storage and twelve-bit candidates. Flash and
advertised/unknown backends fail before any process or evaluation-key export.

The immutable v2 route binds the explicit profile, exact base plan/model, endpoint,
Ed25519 key, participant and executable paths. Every frame has a distinct v2
transport schema and route identity. The remote command includes the admitted
`--profile`; CPU limits come from that installed backend's policy. Client selection
must use the same `--backend-profile`. A mismatch is rejected rather than retried
with the stock implementation.

Version-two support adds no provisioning or connection authority. Its isolated
loopback tests use an authenticated echo service and opaque public fixture bytes;
they cannot qualify remote model execution, native ciphertext processing or
cryptographic security. All candidates retain full trusted-client replay and
production admission remains closed. A genuine separate-machine execution still
requires an authorized, independently provisioned endpoint and matching complete
source/binary/model identities.

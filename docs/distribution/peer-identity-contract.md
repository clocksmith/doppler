# Peer Identity Contract

Session-scoped peer identity for distributed inference.

## Goal

Every cross-peer tensor envelope must be attributable to one session peer
without creating durable machine identity.

## Contract

- key type: `ed25519`
- scope: one distributed session
- durability: non-durable
- signer metadata includes peer id + session id

## Verification

Envelope verification must reject:
- forged signer
- wrong session
- expired session


/**
 * Pinned development authority for the Pack-Native Runtime v0 vertical slice.
 *
 * This public repository key is for reproducible development artifacts only.
 * Production publishers must provide and pin their own signing authority.
 */

export const PACK_V0_DEVELOPMENT_AUTHORITY = 'doppler-pack-v0-development';

export const PACK_V0_TRUSTED_SIGNERS = Object.freeze({
  [PACK_V0_DEVELOPMENT_AUTHORITY]: Object.freeze({
    crv: 'Ed25519',
    x: 'FLU5-eSyW8ORkAf8HupzJn8juiJ2TrGSw2rgMNqGPfc',
    kty: 'OKP',
  }),
});

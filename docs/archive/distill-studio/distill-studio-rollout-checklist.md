# Distill Studio Rollout Checklist

## Stage 0: Internal

1. Training contract drift check passes.
2. Provenance report checks pass on sampled artifacts.
3. Distill Studio MVP commands produce deterministic outputs on tiny workload.

## Stage 1: Limited exposure

1. Enable only replay + branch-compare.
2. Require operator diagnostics run for each release candidate.
3. Record report IDs and artifact hashes for every shared demo.

## Stage 2: Wider usage

1. Enable mini-eval pulse mode with holdout fixtures.
2. Add reliability dashboard ingestion for MVP output JSON.
3. Require incident playbook acknowledgment for oncall owners.

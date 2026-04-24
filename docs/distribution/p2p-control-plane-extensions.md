# P2P Control-Plane Extensions

Additive control-plane protocol for distributed inference planning and session
execution.

## Session states

- `join`
- `ready`
- `run`
- `complete`
- `abort`

## Responsibilities

- coordinator assignment protocol
- peer introspection payload collection
- distributed plan selection over `plans[]`
- ingress routing to stage-0 owner
- egress routing from final-stage owner


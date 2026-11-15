export const TOOLING_COMMAND_SET = ['convert', 'refresh-integrity', 'release', 'debug', 'bench', 'verify', 'lora', 'distill', 'diagnose'];
export const TOOLING_SURFACE_SET = ['browser', 'node'];
export const NODE_ONLY_TOOLING_COMMANDS = Object.freeze(['lora', 'distill', 'diagnose', 'convert', 'refresh-integrity', 'release']);
export const TOOLING_WORKLOAD_SET = Object.freeze(['kernels', 'inference', 'embedding', 'rerank', 'training', 'diffusion', 'energy']);
export const TOOLING_INTENT_SET = ['verify', 'investigate', 'calibrate'];
// Compatibility name: verification consumes the canonical workload set.
export const VERIFY_WORKLOADS = TOOLING_WORKLOAD_SET;
export const TRAINING_STAGE_SET = ['stage1_joint', 'stage2_base', 'stage_a', 'stage_b'];
export const DISTILL_ACTION_SET = ['run', 'stage-a', 'stage-b', 'eval', 'watch', 'compare', 'quality-gate', 'subsets'];
export const LORA_ACTION_SET = ['run', 'eval', 'watch', 'export', 'compare', 'quality-gate', 'activate'];
export const TRAINING_COMMAND_SCHEMA_VERSION = 1;

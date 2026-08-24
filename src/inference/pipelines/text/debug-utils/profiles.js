export const DEBUG_PROFILES = {
  quick: { embed: true, logits: true, sample: true },
  layers: { layer: true },
  attention: { attn: true, kv: true },
  full: { all: true },
  perf: { perf: true },
  kernelStep: { kernel: true },
};

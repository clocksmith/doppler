

export function parseArgs(argv) {
  const opts = { config: null, help: false };
  let i = 0;
  while (i < argv.length) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      opts.help = true;
      i++;
      continue;
    }
    if (arg === '--config' || arg === '-c') {
      opts.config = argv[i + 1] || null;
      i += 2;
      continue;
    }
    if (!arg.startsWith('-') && !opts.config) {
      opts.config = arg;
      i++;
      continue;
    }
    console.error(`Unknown argument: ${arg}`);
    opts.help = true;
    break;
  }

  return opts;
}


export function printHelp() {
  console.log(`
Node.js Model Converter - Convert HuggingFace/GGUF models to RDRR format.

Usage:
  doppler --config <path|json>

Config requirements:
  converter.paths.input (string, required unless converter.test=true)
  converter.paths.output (string, required)
  converter.test (boolean, optional)
  converter.verbose (boolean, optional)

Optional converter settings (match converter schema):
  converter.quantization
  converter.sharding
  converter.weightLayout
  converter.manifest
  converter.output
  converter.presets

Naming Convention:
  Model IDs use format: {name}-w{weights}[-e{embeddings}][-h{head}][-v{vision}]...

  Examples:
    gemma-2b-wq4k           Weights Q4K, embeddings default to weights
    gemma-2b-wq4k-ef16      Weights Q4K, embeddings F16
    llama-8b-wq4k-ef16-hf16 With separate head quantization
    qwen2-vl-7b-wq4k-vf16   Multimodal with vision encoder

  Quantization tokens:
    q4k = Q4_K_M block quant    f16 = Float16       bf16 = BFloat16
    q6k = Q6_K block quant      f32 = Float32       fp8e4/fp8e5 = Float8
    q8_0 = Q8_0 quant           i4/i8 = Integer

Examples:
  doppler --config ./tmp-convert.json
`);
}

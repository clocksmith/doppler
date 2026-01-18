

/**
 * CLI help text.
 */

/**
 * Print CLI help message.
 */
export function printHelp() {
  console.log(`
DOPPLER CLI - Test, Benchmark, Debug, Simulation

Four commands, four purposes:

  doppler test   ->  Correctness (does it work?)
  doppler bench  ->  Performance (how fast?)
  doppler debug  ->  Debugging (why is it broken?)
  doppler simulate -> Simulation (pod emulation)

===============================================================

TEST - Correctness Tests
  doppler test                        All kernel tests (default)
  doppler test --quick                Quick subset (matmul, rmsnorm, softmax, gather)
  doppler test --inference            Model loads + generates (smoke test)
  doppler test simulation             Simulation context init (pod emulation)
  doppler test --training             Training kernel + loop checks
  doppler test --filter matmul        Filter to specific kernel

BENCH - Performance Benchmarks
  doppler bench                       Full inference benchmark (tok/s)
  doppler bench --kernels             Kernel microbenchmarks only
  doppler bench --compare base.json   Compare against baseline

DEBUG - Interactive Debugging
  doppler debug                       Debug mode (config-driven)

===============================================================

Common Options:
  --model, -m <name>     Model (default: gemma-2-2b-it-wf16)
  --config <ref>         Load config (preset name, path, URL, or inline JSON)
  --mode <name>          Shortcut for runtime preset (e.g., bench, debug)
  --dump-config          Print resolved config and exit
  --list-presets         List available config presets
  --headed               Show browser window (default: headless with real GPU)
  --no-reuse-browser     Always launch new browser (don't try CDP)
  --cdp-endpoint <url>   CDP endpoint (default: http://localhost:9222)
  --timeout <ms>         Timeout (default: 300000)
  --output, -o <file>    Save JSON results
  --help, -h             Show this help

Config System:
  --config debug               Use built-in 'debug' preset
  --config ./my-config.json    Load from file path
  --config '{"runtime":...}'   Inline JSON config
  --mode bench                 Shortcut for --config bench

  Built-in presets live under src/config/presets/runtime/* (use --list-presets)
  User presets: ~/.doppler/presets/*.json
  Project presets: ./.doppler/*.json

Warm Mode (preserve model in GPU RAM):
  --warm                 Keep browser open with model loaded for reuse
  --skip-load            Skip model loading (use existing window.pipeline)

  Usage:
    1. First run: doppler debug --warm  (loads model, keeps browser open)
    2. Next runs: doppler debug --skip-load  (reuses loaded model via CDP)

  Start Chrome with CDP first for best results:
    /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222

Headless Mode (default):
  Uses --headless=new with real GPU acceleration (not SwiftShader).
  No browser window, no focus stealing, full GPU compute support.

Examples:
  doppler test                    # Quick correctness check
  doppler bench --config bench    # Benchmark preset
  doppler debug --config debug    # Debug preset
  doppler simulate --config simulation  # Simulation preset

Notes:
  - Headless mode by default (real GPU via --headless=new)
  - Use --headed for visible browser window (debugging)
  - Dev server auto-starts at localhost:8080
  - Exit code: 0=pass, 1=fail
`);
}

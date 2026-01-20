


export function printHelp() {
  console.log(`
DOPPLER Command Interface (Config-Only)

Usage:
  doppler --config <ref>
  doppler --help

Commands:
  run     Serve demo UI (dev server)
  test    Correctness validation
  bench   Performance benchmarks
  debug   Interactive/debug runs
  convert Model conversion (RDRR)
  tool    Tooling utilities

Test suites:
  kernels, inference, demo, converter, simulation, training, quick, all

Bench suites:
  kernels, inference, loading, system, all

Config requirements:
  - config.model (required for run/test/bench/debug)
  - config.cli (required) with fields:
      command, suite (suite required for test/bench; null for run/debug/convert/tool),
      tool (required for tool), baseUrl/noServer/headless/minimized/reuseBrowser/
      cdpEndpoint/timeout/retries/profileDir/output/html/compare/filter
      (required for run/test/bench/debug)
  - config.converter (required for convert)
  - config.tools.<toolKey> (required for tool-specific settings)
  - runtime.shared.tooling.intent (required for test/bench/debug):
      verify | investigate | calibrate
      test -> verify, debug -> investigate, bench -> calibrate or investigate

Config sources:
  --config debug               Use built-in 'debug' preset
  --config ./my-config.json    Load from file path
  --config '{"model":"...","cli":{...},"runtime":{...}}'

Notes:
  - No runtime flags; config is the source of truth.
  - Command + suite are read only from config.
  - Dev server auto-starts at cli.baseUrl when noServer=false.
`);
}

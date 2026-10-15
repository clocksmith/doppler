#!/usr/bin/env node
// Compatibility entrypoint; one generator owns all README evidence charts.
import { main } from '../benchmarks/vendors/generate-readme-evidence-svg.js';
try {
  main();
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}

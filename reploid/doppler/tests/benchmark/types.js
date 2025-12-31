const DEFAULT_BENCHMARK_CONFIG = {
  promptName: "medium",
  maxNewTokens: 128,
  runType: "warm",
  warmupRuns: 2,
  timedRuns: 3,
  sampling: {
    temperature: 0,
    topK: 1,
    topP: 1
  },
  debug: false,
  useChatTemplate: false
};
export {
  DEFAULT_BENCHMARK_CONFIG
};

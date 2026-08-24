export function normalizeRuntimeConfigInput(input: any): Promise<{
    runtimeConfig?: undefined;
    runtimeConfigUrl?: undefined;
} | {
    runtimeConfig: any;
    runtimeConfigUrl?: undefined;
} | {
    runtimeConfigUrl: any;
    runtimeConfig?: undefined;
}>;
export function withReferenceTranscriptRuntimeConfig(runtimeInput: any): any;
export function assertLocalModelArtifactsReadable({ modelUrl, manifest }: {
    modelUrl: any;
    manifest: any;
}): Promise<void>;
export function normalizeModelUrl(value: any, modelDir: any): any;
export function runReferenceCapture({ manifest, modelId, modelUrl, surface, prompt, maxTokens, runtimeConfig, repoRoot, browserTimeoutMs, }: {
    manifest: any;
    modelId: any;
    modelUrl: any;
    surface: any;
    prompt: any;
    maxTokens: any;
    runtimeConfig: any;
    repoRoot: any;
    browserTimeoutMs: any;
}): Promise<import("./node-command-runner.js").NodeCommandRunResult | import("./browser-command-runner.js").BrowserCommandRunResult>;
export function extractReferenceReport(response: any): any;
export function extractReferenceTranscriptSeed(report: any): any;
export function writeReferenceReport(report: any, reportPath: any): Promise<any>;
export function writeReferenceTranscript(transcript: any, transcriptPath: any): Promise<any>;
export const PROGRAM_BUNDLE_REFERENCE_TRANSCRIPT_SCHEMA_ID: "doppler.reference-transcript/v1";

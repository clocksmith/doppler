export interface DistillStudentFixture {
  config: Record<string, unknown>;
  model: {
    forward: (input: unknown, tape: unknown) => Promise<unknown>;
    forwardDistill?: (
      batch: unknown,
      tape: unknown,
      options?: Record<string, unknown>
    ) => Promise<{ logits: unknown }>;
    cleanupDistillStep?: () => void;
    loraParams?: () => unknown[];
    paramGroups?: () => Record<string, unknown[]>;
  };
  outputDim?: number;
  embeddingDim?: number;
  cleanup(): void;
}

export declare function createDistillStudentRuntimeModelFixture(
  overrides?: Record<string, unknown>,
  options?: Record<string, unknown>
): Promise<DistillStudentFixture>;

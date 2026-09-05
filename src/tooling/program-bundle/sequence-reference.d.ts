import type { SequenceReferenceTranscript } from '../../config/sequence-reference.js';
import type { ProgramBundleArtifact } from '../../config/schema/program-bundle.schema.js';
export function buildSequenceReferenceTranscript(
  report: Record<string, unknown>, artifact: ProgramBundleArtifact, executionGraphHash: string
): { artifact: ProgramBundleArtifact; transcript: SequenceReferenceTranscript; adapter: Record<string, unknown> };

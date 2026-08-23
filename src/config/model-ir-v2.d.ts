export const MODEL_IR_V2_SCHEMA_ID: 'doppler.model-ir/v2';
export const MODEL_IR_V2_SCHEMA_VERSION: 2;

export interface ModelIRV2Fact {
  id: string;
  subject: string;
  predicate: string;
  value: unknown;
  confidence: 'direct' | 'derived';
  disposition: 'accepted';
  evidence: Array<{
    kind: 'json-pointer' | 'tensor-header';
    artifactId: string;
    file: string;
    pointer?: string;
    tensorName?: string;
    dtype?: string;
    shape?: number[];
  }>;
  authorship: { kind: 'human' | 'ai' | 'tool'; actor: string; proposalId?: string };
  validation: { status: 'passed'; validator: string; receipt: `sha256:${string}` };
}

export interface ModelIRV2Node {
  id: string;
  factRefs: string[];
  [key: string]: unknown;
}

export interface ModelIRV2 {
  schema: 'doppler.model-ir/v2';
  schemaVersion: 2;
  modelId: string;
  sourceIdentity: {
    checkpointId: string;
    repository: string;
    revision: string;
    artifacts: Array<{ artifactId: string; path: string; role: string; hash: `sha256:${string}` }>;
  };
  provenance: { forgeVersion: string; intakeDigest: `sha256:${string}`; facts: ModelIRV2Fact[] };
  components: ModelIRV2Node[];
  blockClasses: ModelIRV2Node[];
  blockSchedules: ModelIRV2Node[];
  stateSpaces: ModelIRV2Node[];
  tensorRoleBindings: ModelIRV2Node[];
  entryPoints: ModelIRV2Node[];
  outputHeads: ModelIRV2Node[];
  supportScope: {
    sourceTopology: 'complete';
    loweredEntryPoints: string[];
    qualifiedEntryPoints: string[];
    unloweredEntryPoints: string[];
  };
}

export declare function validateModelIRV2(ir: unknown): { ok: boolean; errors: string[] };
export declare function createModelIRV2(params: Omit<ModelIRV2, 'schema' | 'schemaVersion'>): ModelIRV2;

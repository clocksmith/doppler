export function evaluateTranslationArtifactCompetition(options?: {}): {
    receiptHash: string;
    schema: string;
    contractId: unknown;
    policyState: unknown;
    gammaSelectionAuthority: any;
    observedSource: {
        bridgeId: string;
        artifactRole: "selected_candidate" | "diagnostic_baseline" | "diagnostic_candidate";
        checkpointSha256: string;
        selectionReceipt: string | null;
        handoffSha256: any;
        identityVerificationReceiptHash: {} | null;
    } | null;
    lanes: any;
    admission: {
        artifactGenerationAllowed: boolean;
        artifactComparisonAllowed: any;
        promotionSubmissionAllowed: any;
    };
    decision: string;
    blockers: any[];
};
export const TRANSLATION_ARTIFACT_COMPETITION_SCHEMA_ID: "doppler.translation-artifact-competition-readiness/v1";

export const PROGRAM_BUNDLE_JSON_SCHEMA_ID: "urn:doppler:program-bundle-schema:v1";
export const PROGRAM_BUNDLE_JSON_SCHEMA: Readonly<{
    $schema: "https://json-schema.org/draft/2020-12/schema";
    $id: "urn:doppler:program-bundle-schema:v1";
    title: "Doppler Program Bundle v1";
    description: "Portable Doppler program contract with packaged WGSL and constrained host-JS sources.";
    type: "object";
    additionalProperties: false;
    required: string[];
    properties: {
        schema: {
            const: "doppler.program-bundle/v1";
        };
        schemaVersion: {
            const: 1;
        };
        bundleId: {
            type: string;
            minLength: number;
        };
        modelId: {
            type: string;
            minLength: number;
        };
        createdAtUtc: {
            type: string;
            minLength: number;
        };
        package: {
            $ref: string;
        };
        sources: {
            $ref: string;
        };
        host: {
            $ref: string;
        };
        wgslModules: {
            type: string;
            minItems: number;
            items: {
                $ref: string;
            };
        };
        execution: {
            $ref: string;
        };
        captureProfile: {
            $ref: string;
        };
        artifacts: {
            type: string;
            minItems: number;
            items: {
                $ref: string;
            };
        };
        referenceTranscript: {
            $ref: string;
        };
    };
    $defs: {
        digest: Readonly<{
            type: "string";
            pattern: "^sha256:[0-9a-f]{64}$";
        }>;
        bundlePath: Readonly<{
            type: "string";
            minLength: 1;
            pattern: "^(?!/)(?![A-Za-z][A-Za-z0-9+.-]*:)(?!.*(?:^|/)\\.{1,2}(?:/|$))(?!.*\\\\).+$";
        }>;
        artifact: {
            type: string;
            additionalProperties: boolean;
            required: string[];
            properties: {
                role: {
                    enum: string[];
                };
                path: {
                    type: string;
                    minLength: number;
                };
                hash: Readonly<{
                    type: "string";
                    pattern: "^sha256:[0-9a-f]{64}$";
                }>;
                sizeBytes: Readonly<{
                    type: string[];
                    minimum: 0;
                }>;
            };
        };
        packageFile: {
            type: string;
            additionalProperties: boolean;
            required: string[];
            properties: {
                role: {
                    enum: string[];
                };
                path: Readonly<{
                    type: "string";
                    minLength: 1;
                    pattern: "^(?!/)(?![A-Za-z][A-Za-z0-9+.-]*:)(?!.*(?:^|/)\\.{1,2}(?:/|$))(?!.*\\\\).+$";
                }>;
                hash: Readonly<{
                    type: "string";
                    pattern: "^sha256:[0-9a-f]{64}$";
                }>;
                sizeBytes: {
                    type: string;
                    minimum: number;
                };
            };
        };
        package: {
            type: string;
            additionalProperties: boolean;
            required: string[];
            properties: {
                schema: {
                    const: "doppler.program-bundle-package/v1";
                };
                root: {
                    const: string;
                };
                files: {
                    type: string;
                    minItems: number;
                    items: {
                        $ref: string;
                    };
                };
                fileSetHash: Readonly<{
                    type: "string";
                    pattern: "^sha256:[0-9a-f]{64}$";
                }>;
            };
        };
        sources: {
            type: string;
            additionalProperties: boolean;
            required: string[];
            properties: {
                manifest: {
                    $ref: string;
                };
                conversionConfig: {
                    anyOf: ({
                        $ref: string;
                        type?: undefined;
                    } | {
                        type: string;
                        $ref?: undefined;
                    })[];
                };
                executionGraph: {
                    type: string;
                    additionalProperties: boolean;
                    required: string[];
                    properties: {
                        schema: {
                            type: string[];
                        };
                        hash: Readonly<{
                            type: "string";
                            pattern: "^sha256:[0-9a-f]{64}$";
                        }>;
                        expandedStepHash: Readonly<{
                            type: "string";
                            pattern: "^sha256:[0-9a-f]{64}$";
                        }>;
                    };
                };
                weightSetHash: Readonly<{
                    type: "string";
                    pattern: "^sha256:[0-9a-f]{64}$";
                }>;
                artifactSetHash: Readonly<{
                    type: "string";
                    pattern: "^sha256:[0-9a-f]{64}$";
                }>;
            };
        };
        sourceRef: {
            type: string;
            additionalProperties: boolean;
            required: string[];
            properties: {
                path: {
                    type: string;
                    minLength: number;
                };
                hash: Readonly<{
                    type: "string";
                    pattern: "^sha256:[0-9a-f]{64}$";
                }>;
            };
        };
        host: {
            type: string;
            additionalProperties: boolean;
            required: string[];
            properties: {
                schema: {
                    const: "doppler.host-js/v1";
                };
                jsSubset: {
                    const: "doppler-webgpu-host/v1";
                };
                entrypoints: {
                    type: string;
                    minItems: number;
                    items: {
                        type: string;
                        additionalProperties: boolean;
                        required: string[];
                        properties: {
                            id: {
                                type: string;
                                minLength: number;
                            };
                            module: Readonly<{
                                type: "string";
                                minLength: 1;
                                pattern: "^(?!/)(?![A-Za-z][A-Za-z0-9+.-]*:)(?!.*(?:^|/)\\.{1,2}(?:/|$))(?!.*\\\\).+$";
                            }>;
                            export: {
                                type: string;
                                minLength: number;
                            };
                            role: {
                                type: string;
                                minLength: number;
                            };
                            sourceHash: Readonly<{
                                type: "string";
                                pattern: "^sha256:[0-9a-f]{64}$";
                            }>;
                            validation: {
                                type: string;
                                additionalProperties: boolean;
                                required: string[];
                                properties: {
                                    dynamicImport: {
                                        const: string;
                                    };
                                    staticImport: {
                                        const: string;
                                    };
                                    dom: {
                                        const: string;
                                    };
                                    runtimeGlobals: {
                                        const: string;
                                    };
                                    network: {
                                        const: string;
                                    };
                                    dynamicCode: {
                                        const: string;
                                    };
                                };
                            };
                        };
                    };
                };
                constraints: {
                    type: string;
                    additionalProperties: boolean;
                    required: string[];
                    properties: {
                        dynamicImport: {
                            const: string;
                        };
                        staticImport: {
                            const: string;
                        };
                        dom: {
                            const: string;
                        };
                        runtimeGlobals: {
                            const: string;
                        };
                        dynamicCode: {
                            const: string;
                        };
                        filesystem: {
                            const: string;
                        };
                        network: {
                            const: string;
                        };
                    };
                };
            };
        };
        binding: {
            type: string;
            additionalProperties: boolean;
            required: string[];
            properties: {
                group: {
                    type: string;
                    minimum: number;
                };
                binding: {
                    type: string;
                    minimum: number;
                };
                name: {
                    type: string;
                    minLength: number;
                };
                addressSpace: {
                    type: string[];
                };
                access: {
                    type: string[];
                };
            };
        };
        wgslModule: {
            type: string;
            additionalProperties: boolean;
            required: string[];
            properties: {
                id: {
                    type: string;
                    minLength: number;
                };
                file: {
                    type: string;
                    minLength: number;
                };
                entry: {
                    type: string;
                    minLength: number;
                };
                digest: Readonly<{
                    type: "string";
                    pattern: "^sha256:[0-9a-f]{64}$";
                }>;
                sourcePath: Readonly<{
                    type: "string";
                    minLength: 1;
                    pattern: "^(?!/)(?![A-Za-z][A-Za-z0-9+.-]*:)(?!.*(?:^|/)\\.{1,2}(?:/|$))(?!.*\\\\).+$";
                }>;
                sourceHash: Readonly<{
                    type: "string";
                    pattern: "^sha256:[0-9a-f]{64}$";
                }>;
                reachable: {
                    const: boolean;
                };
                metadata: {
                    type: string;
                    additionalProperties: boolean;
                    required: string[];
                    properties: {
                        entry: {
                            type: string;
                            minLength: number;
                        };
                        sourceMetadataHash: Readonly<{
                            type: "string";
                            pattern: "^sha256:[0-9a-f]{64}$";
                        }>;
                        bindings: {
                            type: string;
                            items: {
                                $ref: string;
                            };
                        };
                        overrides: {
                            type: string;
                            items: {
                                type: string;
                            };
                        };
                        workgroupSize: {
                            type: string;
                            items: {
                                type: string;
                            };
                        };
                        requiresSubgroups: {
                            type: string;
                        };
                    };
                };
            };
        };
        execution: {
            type: string;
            required: string[];
            properties: {
                graphHash: Readonly<{
                    type: "string";
                    pattern: "^sha256:[0-9a-f]{64}$";
                }>;
                stepMetadataHash: Readonly<{
                    type: "string";
                    pattern: "^sha256:[0-9a-f]{64}$";
                }>;
                kernelClosure: {
                    type: string;
                };
                steps: {
                    type: string;
                    minItems: number;
                    items: {
                        type: string;
                    };
                };
            };
        };
        captureProfile: {
            type: string;
            required: string[];
            properties: {
                schema: {
                    const: "doppler.capture-profile/v1";
                };
                deterministic: {
                    const: boolean;
                };
                phases: {
                    type: string;
                    minItems: number;
                    items: {
                        type: string;
                    };
                };
                surfaces: {
                    type: string;
                    minItems: number;
                    items: {
                        type: string;
                    };
                };
                adapter: {
                    type: string;
                };
                hashPolicy: {
                    type: string;
                };
                captureHash: Readonly<{
                    type: "string";
                    pattern: "^sha256:[0-9a-f]{64}$";
                }>;
            };
        };
        referenceTranscript: {
            type: string;
            required: string[];
            properties: {
                schema: {
                    const: "doppler.reference-transcript/v1";
                };
                source: {
                    type: string;
                };
                executionGraphHash: Readonly<{
                    type: "string";
                    pattern: "^sha256:[0-9a-f]{64}$";
                }>;
                surface: {
                    type: string[];
                };
                generationConfig: {
                    type: string;
                };
                sourceParity: {
                    type: string[];
                };
                prompt: {
                    type: string;
                };
                output: {
                    type: string;
                };
                tokens: {
                    type: string;
                };
                phase: {
                    type: string;
                };
                kvCache: {
                    type: string;
                };
                logits: {
                    type: string;
                };
                tolerance: {
                    type: string;
                };
            };
        };
    };
    'x-doppler-semantic-validator': "src/config/schema/program-bundle.schema.js#validateProgramBundle";
}>;

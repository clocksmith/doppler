// Real Qwen 0.8B receipt; fingerprint and token IDs match assets/activity-example.json.gz.
export const SAMPLE_INSPECTION_RECEIPT = {
  "schema": "doppler.model-inspection-receipt/v1",
  "policy": {
    "id": "demo/model-activity",
    "label": "Model activity capture",
    "modifiesExecution": true,
    "performanceRepresentative": false,
    "requiredCaptures": [
      "artifact-identity",
      "prompt-token-ids",
      "generated-token-ids",
      "selected-token-probabilities",
      "top-candidates",
      "word-quality",
      "layer-activations"
    ],
    "allowedClaimTypes": [
      "diagnostic"
    ],
    "gpuTimestampQueries": false,
    "perplexity": {
      "wordSegmentation": "doppler.word-segmentation/unicode-whitespace-v1",
      "aggregation": "doppler.perplexity/summed-word-surprisal-v1",
      "rollingWindow": {
        "unit": "words",
        "size": 8
      }
    },
    "activationTrace": {
      "maxBytes": 33554432,
      "maxFrameBytes": 8388608,
      "maxFrames": 8192,
      "maxGridValues": 4194304,
      "maxInFlight": 32,
      "activityMaxBytes": 33554432,
      "activityMaxFrameValues": 1048576,
      "activityMaxStateValues": 16384,
      "weightBlockValues": 16384,
      "weightReadValues": 262144,
      "weightRowGroupSize": 128,
      "weightRawRows": 4,
      "weightMaxBytes": 16777216,
      "activityTokenWindow": 4,
      "activitySummaryBlock": 256
    }
  },
  "fingerprint": {
    "schema": "doppler.comparison-fingerprint/v1",
    "identity": {
      "artifact": {
        "modelId": "http://127.0.0.1:51762/.cache/qwen08-sha256-repair/",
        "manifestHash": "sha256:c5fcf38e945f1c048faca9ccda8991f1494563317f46a6dd79e0714ad001c569"
      },
      "tokenizer": {
        "contract": {
          "type": "bundled",
          "vocabSize": 248044,
          "file": "tokenizer.json",
          "addBosToken": false
        },
        "digest": "sha256:7dc87d171ff9c123d3fcf3e3037a403febb2e74fa7b6b94c67eea4d565fe78da"
      },
      "promptTokenIds": [
        44240,
        23066,
        303,
        799,
        3299,
        13
      ],
      "sampling": {
        "temperature": 0,
        "topP": 1,
        "topK": 1,
        "repetitionPenalty": 1.1,
        "repetitionPenaltyWindow": 100,
        "presencePenalty": 0,
        "suppressTokenIds": [],
        "greedyThreshold": 0.01,
        "suppressSpecialTokens": false,
        "suppressSpecialLikeTokens": false,
        "maxTokens": 3,
        "stopSequences": [],
        "useChatTemplate": true,
        "useSpeculative": null,
        "speculation": null,
        "seed": null
      },
      "observationPolicy": {
        "id": "demo/model-activity",
        "modifiesExecution": true,
        "performanceRepresentative": false,
        "requiredCaptures": [
          "artifact-identity",
          "prompt-token-ids",
          "generated-token-ids",
          "selected-token-probabilities",
          "top-candidates",
          "word-quality",
          "layer-activations"
        ],
        "allowedClaimTypes": [
          "diagnostic"
        ]
      },
      "perplexity": {
        "wordSegmentation": "doppler.word-segmentation/unicode-whitespace-v1",
        "aggregation": "doppler.perplexity/summed-word-surprisal-v1",
        "rollingWindow": {
          "unit": "words",
          "size": 8
        }
      },
      "execution": {
        "backend": "webgpu",
        "executionPlanId": "primary",
        "kernelPathId": "qwen-3-5-0-8b-q4k-ehaf16-execution-inline",
        "kernelPathSource": "execution-v1",
        "activationDtype": "f32",
        "hasF16": true,
        "hasSubgroups": true
      },
      "browser": {
        "userAgent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/145.0.7632.6 Safari/537.36",
        "platform": "MacIntel",
        "language": "en-US"
      },
      "adapter": {
        "vendor": "apple",
        "architecture": "metal-3",
        "device": "unknown",
        "description": null
      }
    },
    "fullDigest": "sha256:88590bd1f8054282d365840a6ddc886a8fac918b2f4bdb7b9ed86e217cd15d27",
    "qualityDigest": "sha256:3d84fd9202a09d7a9b14caff524c0ab20920751c004cda522ec805df2db27aa4",
    "performanceDigest": "sha256:bcec59d68b368786c593de53bf9ddd87df57ab89c4f8b8ec3d0c91b1ad5d0a42"
  },
  "outputText": "Hello!",
  "inputTokenIds": [
    248045,
    846,
    198,
    44240,
    23066,
    303,
    799,
    3299,
    13,
    248046,
    198,
    248045,
    74455,
    198,
    248068,
    271,
    248069,
    271
  ],
  "generatedTokenIds": [
    9419,
    0,
    248046
  ],
  "wallTimingMs": 18062.5,
  "performanceRepresentative": false,
  "tokens": [
    {
      "index": 0,
      "tokenId": 9419,
      "text": "Hello",
      "probability": 0.7563575609257694,
      "surprisal": 0.27924105042738556,
      "topCandidates": [
        {
          "tokenId": 9419,
          "logit": 21.501251220703125,
          "text": "Hello",
          "probability": 0.7563575609257694
        },
        {
          "tokenId": 14556,
          "logit": 19.25992774963379,
          "text": "hello",
          "probability": 0.080414198390108
        },
        {
          "tokenId": 12675,
          "logit": 18.827577590942383,
          "text": "Hi",
          "probability": 0.0521873745509477
        },
        {
          "tokenId": 38,
          "logit": 18.04969024658203,
          "text": "G",
          "probability": 0.02397360070793431
        },
        {
          "tokenId": 14773,
          "logit": 17.972660064697266,
          "text": "Good",
          "probability": 0.022196243715309934
        }
      ]
    },
    {
      "index": 1,
      "tokenId": 0,
      "text": "!",
      "probability": 0.9060771044207434,
      "surprisal": 0.09863087234316008,
      "topCandidates": [
        {
          "tokenId": 0,
          "logit": 20.205562591552734,
          "text": "!",
          "probability": 0.9060771044207434
        },
        {
          "tokenId": 13,
          "logit": 16.9827823638916,
          "text": ".",
          "probability": 0.0361018524698963
        },
        {
          "tokenId": 248046,
          "logit": 16.643892288208008,
          "text": "",
          "probability": 0.025724763886078257
        },
        {
          "tokenId": 11,
          "logit": 15.92564582824707,
          "text": ",",
          "probability": 0.012543563222078079
        },
        {
          "tokenId": 1017,
          "logit": 15.658685684204102,
          "text": " there",
          "probability": 0.00960465137431272
        }
      ]
    },
    {
      "index": 2,
      "tokenId": 248046,
      "text": "",
      "probability": 0.4246223369696148,
      "surprisal": 0.8565551240085678,
      "topCandidates": [
        {
          "tokenId": 248046,
          "logit": 17.757047653198242,
          "text": "",
          "probability": 0.4246223369696148
        },
        {
          "tokenId": 2500,
          "logit": 16.76971435546875,
          "text": " How",
          "probability": 0.15820107606590447
        },
        {
          "tokenId": 59720,
          "logit": 16.03378677368164,
          "text": " �",
          "probability": 0.07578794745520134
        },
        {
          "tokenId": 10838,
          "logit": 15.553412437438965,
          "text": " �",
          "probability": 0.04687877147626412
        },
        {
          "tokenId": 353,
          "logit": 15.346002578735352,
          "text": " I",
          "probability": 0.03809774448496842
        }
      ]
    }
  ],
  "quality": {
    "wordSegmentation": "doppler.word-segmentation/unicode-whitespace-v1",
    "aggregation": "doppler.perplexity/summed-word-surprisal-v1",
    "rollingWindow": {
      "unit": "words",
      "size": 8
    },
    "words": [
      {
        "text": "Hello!",
        "tokenIndexes": [
          0,
          1,
          2
        ],
        "tokenCount": 3,
        "summedSurprisal": 1.2344270467791134,
        "probabilityAvailable": true,
        "wordIndex": 0,
        "rollingPerplexity": 1.5090430108462098,
        "cumulativePerplexity": 1.5090430108462098,
        "rollingWindow": {
          "unit": "words",
          "size": 8,
          "tokenCount": 3
        }
      }
    ]
  },
  "generationEvidence": {
    "schema": "doppler_generation_evidence/v1",
    "outputText": "Hello!",
    "tokenIds": [
      9419,
      0,
      248046
    ],
    "transcript": {
      "schema": "doppler_generation_transcript/v1",
      "outputText": "Hello!",
      "tokenIds": [
        9419,
        0,
        248046
      ]
    },
    "transcriptHash": "sha256:e2492891ec17154aac133d04732a512f5f1b82c12e226102de8b0bb8b0fb20df",
    "generationConfig": {
      "temperature": 0,
      "topP": 1,
      "topK": 1,
      "repetitionPenalty": 1.1,
      "repetitionPenaltyWindow": 100,
      "presencePenalty": 0,
      "suppressTokenIds": [],
      "greedyThreshold": 0.01,
      "suppressSpecialTokens": false,
      "suppressSpecialLikeTokens": false,
      "maxTokens": 3,
      "stopSequences": [],
      "useChatTemplate": true,
      "useSpeculative": null,
      "speculation": null,
      "seed": null
    },
    "generationConfigHash": "sha256:84f71f51a24eb61f5d8670ebf003da37853424c2f2ccfbf117bdbe18ba688192",
    "resolution": {
      "schema": "doppler.resolution-identity/v1",
      "logicalModelId": "qwen-3-5-0-8b-q4k-ehaf16",
      "resolvedArtifactVariantId": "sha256:c5fcf38e945f1c048faca9ccda8991f1494563317f46a6dd79e0714ad001c569",
      "resolvedExecutionId": "sha256:af98a5f4d91ae6519fb1c5d157c0bbe9824e079952896f00a3953ec11eccb059"
    },
    "executionIdentity": {
      "schema": "doppler.resolved-execution-identity/v1",
      "runtime": {
        "package": "doppler-gpu",
        "version": "0.6.33",
        "surface": "browser"
      },
      "resolvedRuntimeSessionId": "sha256:5f40c4370bc83412f8f5b1179ec5fee9e417ad6e303dc0bb6aa9a11bfc6b2260",
      "activeAdapter": null,
      "activeAdapterId": null,
      "activeAdapterDigest": null,
      "backendIdentity": {
        "backend": "webgpu",
        "adapter": {
          "vendor": "apple",
          "architecture": "metal-3",
          "device": "unknown",
          "description": null
        },
        "hasF16": true,
        "hasSubgroups": true,
        "maxBufferSize": 4294967292,
        "deviceEpoch": 0,
        "kernelPathId": "qwen-3-5-0-8b-q4k-ehaf16-execution-inline",
        "kernelPathSource": "execution-v1",
        "executionPlanId": "primary",
        "activationDtype": "f32"
      }
    },
    "runtimeProfile": {
      "schema": "doppler_runtime_profile/v1",
      "runtime": {
        "package": "doppler-gpu",
        "version": "0.6.33",
        "surface": "browser"
      },
      "model": {
        "modelId": "http://127.0.0.1:51762/.cache/qwen08-sha256-repair/",
        "manifestHash": "sha256:c5fcf38e945f1c048faca9ccda8991f1494563317f46a6dd79e0714ad001c569",
        "activeAdapter": null,
        "activeAdapterId": null,
        "activeAdapterDigest": null
      },
      "resolvedRuntimeSessionId": "sha256:5f40c4370bc83412f8f5b1179ec5fee9e417ad6e303dc0bb6aa9a11bfc6b2260",
      "backendIdentity": {
        "backend": "webgpu",
        "adapter": {
          "vendor": "apple",
          "architecture": "metal-3",
          "device": "unknown",
          "description": null
        },
        "hasF16": true,
        "hasSubgroups": true,
        "maxBufferSize": 4294967292,
        "deviceEpoch": 0,
        "kernelPathId": "qwen-3-5-0-8b-q4k-ehaf16-execution-inline",
        "kernelPathSource": "execution-v1",
        "executionPlanId": "primary",
        "activationDtype": "f32"
      }
    },
    "runtimeProfileHash": "sha256:8ea7b6f0e96eb7ea210639364de352593cb9933380b5317d8e95fdd2eac570cc",
    "backendIdentity": {
      "backend": "webgpu",
      "adapter": {
        "vendor": "apple",
        "architecture": "metal-3",
        "device": "unknown",
        "description": null
      },
      "hasF16": true,
      "hasSubgroups": true,
      "maxBufferSize": 4294967292,
      "deviceEpoch": 0,
      "kernelPathId": "qwen-3-5-0-8b-q4k-ehaf16-execution-inline",
      "kernelPathSource": "execution-v1",
      "executionPlanId": "primary",
      "activationDtype": "f32"
    },
    "backendIdentityHash": "sha256:69cab6e8edee304387cde422555798703b6ed09086081d7b20767c8398ccf9b7",
    "stats": {
      "prefillTimeMs": 403.7000000476837,
      "decodeTimeMs": 235.5,
      "ttftMs": 17825.400000095367,
      "loadTiming": {
        "schemaVersion": 1,
        "source": "doppler-loader",
        "modelId": "qwen-3-5-0-8b-q4k-ehaf16",
        "status": "complete",
        "customShardLoader": true,
        "byteAccountingMode": "custom-loader-read-progress",
        "totalBytes": 1146048704,
        "totalShards": 18,
        "bytesLoaded": 933399232,
        "shardsLoaded": 16,
        "bytesPerSecond": 1624150395,
        "phasesMs": {
          "preflight": 0.7,
          "tensorLocations": 6,
          "embeddings": 302.1,
          "layers": 179,
          "finalWeights": 86.4,
          "expertPreparation": null,
          "cleanup": 0
        },
        "expertPreparation": null,
        "layers": {
          "count": 24,
          "totalMs": 174.7,
          "meanMs": 7.279,
          "maxMs": 19.6,
          "maxLayer": 23
        },
        "totalMs": 574.7,
        "failedPhase": null,
        "error": null
      },
      "pipelineLoadTiming": {
        "schemaVersion": 1,
        "source": "doppler-pipeline",
        "modelId": "qwen-3-5-0-8b-q4k-ehaf16",
        "status": "complete",
        "phasesMs": {
          "reset": 0.1,
          "configResolution": 4.4,
          "kernelWarmup": 0,
          "tokenizer": 134.4,
          "executionSetup": 4.1,
          "loadWeights": 614.7,
          "rope": 5.3,
          "convStates": 0.3,
          "warmGeneration": null
        },
        "details": {
          "tokenizer": {
            "schemaVersion": 1,
            "source": "doppler-tokenizer",
            "modelId": "qwen-3-5-0-8b-q4k-ehaf16",
            "status": "complete",
            "tokenizerType": "bundled",
            "tokenizerFile": "tokenizer.json",
            "backend": "bundled",
            "assetSource": "custom-loader",
            "cacheHit": false,
            "phasesMs": {
              "configResolution": 0,
              "cacheLookup": 0,
              "backendCreate": 0.1,
              "assetLoad": 56.5,
              "assetParse": 0,
              "backendLoad": 77.3,
              "cacheStore": 0.1
            },
            "totalMs": 134.3,
            "error": null
          },
          "warmGeneration": null
        },
        "totalMs": 763.4
      },
      "prefillTokens": 18,
      "decodeTokens": 2,
      "memoryUsageBytes": 0,
      "tokensGenerated": 3,
      "totalTimeMs": 18061,
      "decodeRecordMs": 0,
      "decodeRecordOps": 0,
      "decodeRecordPasses": 0,
      "decodeRecordOpLabels": {},
      "decodeSubmitWaitMs": 0,
      "decodeReadbackWaitMs": 0,
      "decodeReadbackMapWaitMs": 0,
      "decodeReadbackCleanupMs": 0,
      "decodeReadbackCopyMs": 0,
      "prefillRecordMs": 357.69999957084656,
      "prefillRecordOps": 0,
      "prefillRecordPasses": 0,
      "prefillRecordOpLabels": {},
      "prefillSubmitWaitMs": 0,
      "prefillProfileSteps": [],
      "decodeProfileSteps": [],
      "executionPlan": {
        "primary": {
          "id": "primary",
          "kernelPathId": "qwen-3-5-0-8b-q4k-ehaf16-execution-inline",
          "kernelPathSource": "execution-v1",
          "activationDtype": "f32",
          "readbackInterval": 8,
          "readbackMode": "sequential",
          "maxBatchDecodeTokens": null,
          "batchSize": 4,
          "stopCheckMode": "batch",
          "disableCommandBatching": false,
          "ringTokens": 1,
          "ringStop": 1,
          "ringStaging": 1
        },
        "fallback": null,
        "activePlanIdAtStart": "primary",
        "finalActivePlanId": "primary",
        "transitions": []
      },
      "kernelPathId": "qwen-3-5-0-8b-q4k-ehaf16-execution-inline",
      "kernelPathSource": "execution-v1",
      "operatorDiagnostics": null,
      "attentionInputs": [
        {
          "phase": "prefill",
          "layerIdx": 3,
          "numTokens": 18,
          "kvLen": 18,
          "numHeads": 8,
          "numKVHeads": 2,
          "headDim": 256,
          "activationDtype": "f32",
          "inputDtype": "f32",
          "normedDtype": "f32",
          "useF16Activations": false,
          "matmulOutputDtype": "f32",
          "kvCacheDtype": "f16",
          "cachedKDtype": "f16",
          "cachedVDtype": "f16",
          "qDtype": "f32",
          "kDtype": "f32",
          "vDtype": "f32",
          "useFusedQKV": false,
          "kvStart": 0,
          "kvLayout": "contiguous",
          "kvPageSize": 256,
          "hotLen": null,
          "coldLen": null,
          "hotWindow": null,
          "hotStart": null,
          "coldQuantMode": null
        },
        {
          "phase": "prefill",
          "layerIdx": 7,
          "numTokens": 18,
          "kvLen": 18,
          "numHeads": 8,
          "numKVHeads": 2,
          "headDim": 256,
          "activationDtype": "f32",
          "inputDtype": "f32",
          "normedDtype": "f32",
          "useF16Activations": false,
          "matmulOutputDtype": "f32",
          "kvCacheDtype": "f16",
          "cachedKDtype": "f16",
          "cachedVDtype": "f16",
          "qDtype": "f32",
          "kDtype": "f32",
          "vDtype": "f32",
          "useFusedQKV": false,
          "kvStart": 0,
          "kvLayout": "contiguous",
          "kvPageSize": 256,
          "hotLen": null,
          "coldLen": null,
          "hotWindow": null,
          "hotStart": null,
          "coldQuantMode": null
        },
        {
          "phase": "prefill",
          "layerIdx": 11,
          "numTokens": 18,
          "kvLen": 18,
          "numHeads": 8,
          "numKVHeads": 2,
          "headDim": 256,
          "activationDtype": "f32",
          "inputDtype": "f32",
          "normedDtype": "f32",
          "useF16Activations": false,
          "matmulOutputDtype": "f32",
          "kvCacheDtype": "f16",
          "cachedKDtype": "f16",
          "cachedVDtype": "f16",
          "qDtype": "f32",
          "kDtype": "f32",
          "vDtype": "f32",
          "useFusedQKV": false,
          "kvStart": 0,
          "kvLayout": "contiguous",
          "kvPageSize": 256,
          "hotLen": null,
          "coldLen": null,
          "hotWindow": null,
          "hotStart": null,
          "coldQuantMode": null
        },
        {
          "phase": "prefill",
          "layerIdx": 15,
          "numTokens": 18,
          "kvLen": 18,
          "numHeads": 8,
          "numKVHeads": 2,
          "headDim": 256,
          "activationDtype": "f32",
          "inputDtype": "f32",
          "normedDtype": "f32",
          "useF16Activations": false,
          "matmulOutputDtype": "f32",
          "kvCacheDtype": "f16",
          "cachedKDtype": "f16",
          "cachedVDtype": "f16",
          "qDtype": "f32",
          "kDtype": "f32",
          "vDtype": "f32",
          "useFusedQKV": false,
          "kvStart": 0,
          "kvLayout": "contiguous",
          "kvPageSize": 256,
          "hotLen": null,
          "coldLen": null,
          "hotWindow": null,
          "hotStart": null,
          "coldQuantMode": null
        },
        {
          "phase": "prefill",
          "layerIdx": 19,
          "numTokens": 18,
          "kvLen": 18,
          "numHeads": 8,
          "numKVHeads": 2,
          "headDim": 256,
          "activationDtype": "f32",
          "inputDtype": "f32",
          "normedDtype": "f32",
          "useF16Activations": false,
          "matmulOutputDtype": "f32",
          "kvCacheDtype": "f16",
          "cachedKDtype": "f16",
          "cachedVDtype": "f16",
          "qDtype": "f32",
          "kDtype": "f32",
          "vDtype": "f32",
          "useFusedQKV": false,
          "kvStart": 0,
          "kvLayout": "contiguous",
          "kvPageSize": 256,
          "hotLen": null,
          "coldLen": null,
          "hotWindow": null,
          "hotStart": null,
          "coldQuantMode": null
        },
        {
          "phase": "prefill",
          "layerIdx": 23,
          "numTokens": 18,
          "kvLen": 18,
          "numHeads": 8,
          "numKVHeads": 2,
          "headDim": 256,
          "activationDtype": "f32",
          "inputDtype": "f32",
          "normedDtype": "f32",
          "useF16Activations": false,
          "matmulOutputDtype": "f32",
          "kvCacheDtype": "f16",
          "cachedKDtype": "f16",
          "cachedVDtype": "f16",
          "qDtype": "f32",
          "kDtype": "f32",
          "vDtype": "f32",
          "useFusedQKV": false,
          "kvStart": 0,
          "kvLayout": "contiguous",
          "kvPageSize": 256,
          "hotLen": null,
          "coldLen": null,
          "hotWindow": null,
          "hotStart": null,
          "coldQuantMode": null
        },
        {
          "phase": "decode",
          "layerIdx": 3,
          "numTokens": 1,
          "kvLen": 19,
          "numHeads": 8,
          "numKVHeads": 2,
          "headDim": 256,
          "activationDtype": "f32",
          "inputDtype": "f32",
          "normedDtype": "f32",
          "useF16Activations": false,
          "matmulOutputDtype": "f32",
          "kvCacheDtype": "f16",
          "cachedKDtype": "f16",
          "cachedVDtype": "f16",
          "qDtype": "f32",
          "kDtype": "f32",
          "vDtype": "f32",
          "useFusedQKV": false,
          "kvStart": 0,
          "kvLayout": "contiguous",
          "kvPageSize": 256,
          "hotLen": null,
          "coldLen": null,
          "hotWindow": null,
          "hotStart": null,
          "coldQuantMode": null
        },
        {
          "phase": "decode",
          "layerIdx": 7,
          "numTokens": 1,
          "kvLen": 19,
          "numHeads": 8,
          "numKVHeads": 2,
          "headDim": 256,
          "activationDtype": "f32",
          "inputDtype": "f32",
          "normedDtype": "f32",
          "useF16Activations": false,
          "matmulOutputDtype": "f32",
          "kvCacheDtype": "f16",
          "cachedKDtype": "f16",
          "cachedVDtype": "f16",
          "qDtype": "f32",
          "kDtype": "f32",
          "vDtype": "f32",
          "useFusedQKV": false,
          "kvStart": 0,
          "kvLayout": "contiguous",
          "kvPageSize": 256,
          "hotLen": null,
          "coldLen": null,
          "hotWindow": null,
          "hotStart": null,
          "coldQuantMode": null
        },
        {
          "phase": "decode",
          "layerIdx": 11,
          "numTokens": 1,
          "kvLen": 19,
          "numHeads": 8,
          "numKVHeads": 2,
          "headDim": 256,
          "activationDtype": "f32",
          "inputDtype": "f32",
          "normedDtype": "f32",
          "useF16Activations": false,
          "matmulOutputDtype": "f32",
          "kvCacheDtype": "f16",
          "cachedKDtype": "f16",
          "cachedVDtype": "f16",
          "qDtype": "f32",
          "kDtype": "f32",
          "vDtype": "f32",
          "useFusedQKV": false,
          "kvStart": 0,
          "kvLayout": "contiguous",
          "kvPageSize": 256,
          "hotLen": null,
          "coldLen": null,
          "hotWindow": null,
          "hotStart": null,
          "coldQuantMode": null
        },
        {
          "phase": "decode",
          "layerIdx": 15,
          "numTokens": 1,
          "kvLen": 19,
          "numHeads": 8,
          "numKVHeads": 2,
          "headDim": 256,
          "activationDtype": "f32",
          "inputDtype": "f32",
          "normedDtype": "f32",
          "useF16Activations": false,
          "matmulOutputDtype": "f32",
          "kvCacheDtype": "f16",
          "cachedKDtype": "f16",
          "cachedVDtype": "f16",
          "qDtype": "f32",
          "kDtype": "f32",
          "vDtype": "f32",
          "useFusedQKV": false,
          "kvStart": 0,
          "kvLayout": "contiguous",
          "kvPageSize": 256,
          "hotLen": null,
          "coldLen": null,
          "hotWindow": null,
          "hotStart": null,
          "coldQuantMode": null
        },
        {
          "phase": "decode",
          "layerIdx": 19,
          "numTokens": 1,
          "kvLen": 19,
          "numHeads": 8,
          "numKVHeads": 2,
          "headDim": 256,
          "activationDtype": "f32",
          "inputDtype": "f32",
          "normedDtype": "f32",
          "useF16Activations": false,
          "matmulOutputDtype": "f32",
          "kvCacheDtype": "f16",
          "cachedKDtype": "f16",
          "cachedVDtype": "f16",
          "qDtype": "f32",
          "kDtype": "f32",
          "vDtype": "f32",
          "useFusedQKV": false,
          "kvStart": 0,
          "kvLayout": "contiguous",
          "kvPageSize": 256,
          "hotLen": null,
          "coldLen": null,
          "hotWindow": null,
          "hotStart": null,
          "coldQuantMode": null
        },
        {
          "phase": "decode",
          "layerIdx": 23,
          "numTokens": 1,
          "kvLen": 19,
          "numHeads": 8,
          "numKVHeads": 2,
          "headDim": 256,
          "activationDtype": "f32",
          "inputDtype": "f32",
          "normedDtype": "f32",
          "useF16Activations": false,
          "matmulOutputDtype": "f32",
          "kvCacheDtype": "f16",
          "cachedKDtype": "f16",
          "cachedVDtype": "f16",
          "qDtype": "f32",
          "kDtype": "f32",
          "vDtype": "f32",
          "useFusedQKV": false,
          "kvStart": 0,
          "kvLayout": "contiguous",
          "kvPageSize": 256,
          "hotLen": null,
          "coldLen": null,
          "hotWindow": null,
          "hotStart": null,
          "coldQuantMode": null
        }
      ],
      "decodeMode": "single_token",
      "batchGuardReason": "command_batching_disabled",
      "singleTokenSubmitWaitMs": 0,
      "singleTokenReadbackWaitMs": 23.50000023841858,
      "singleTokenReadbackMapWaitMs": 0,
      "singleTokenReadbackCleanupMs": 0,
      "singleTokenReadbackCopyMs": 0,
      "singleTokenOrchestrationMs": 211.2999997138977,
      "plePreparedTokenCacheHits": 0,
      "plePreparedTokenCacheMisses": 0,
      "plePreparedTokenCacheEntries": 0,
      "plePreparedTokenCacheBytes": 0,
      "pleHotVocabularyHits": 0,
      "pleHotVocabularyMisses": 0,
      "modelLoadMs": 763.4000000953674,
      "wholeTokenProgramWallMs": 0,
      "speculationAttempts": 0,
      "speculationAccepted": 0,
      "gpuTimePrefillMs": null,
      "gpuTimeDecodeMs": null,
      "stopReason": "stop-token",
      "stopTokenId": 248046,
      "batching": {
        "batchedForwardCalls": 0,
        "unbatchedForwardCalls": 0,
        "totalBatchedTimeMs": 0,
        "totalUnbatchedTimeMs": 0,
        "gpuSubmissions": 0,
        "requestedBatchTokens": 0,
        "effectiveBatchTokens": 0,
        "executedBatchTokens": 0,
        "resolvedBatchTokens": 0,
        "maxBatchTokenCap": null,
        "batchClampCount": 0
      }
    }
  },
  "prompt": "Say hello in one word.",
  "recording": {
    "publishedAt": "2026-11-08",
    "receiptSha256": "976a3797eeb9d6e13ab83f80efdd821120d70a71c0daa6bc8672737a244b51cc",
    "modelId": "qwen-3-5-0-8b-q4k-ehaf16"
  }
};

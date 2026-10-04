// Recorded through the public model.inspect.generate API. See recording for provenance.
export const SAMPLE_INSPECTION_RECEIPT = {
  "schema": "doppler.model-inspection-receipt/v1",
  "policy": {
    "id": "demo/guided-quality",
    "label": "Guided quality inspection",
    "modifiesExecution": true,
    "performanceRepresentative": false,
    "requiredCaptures": [
      "artifact-identity",
      "prompt-token-ids",
      "generated-token-ids",
      "selected-token-probabilities",
      "top-candidates",
      "word-quality"
    ],
    "allowedClaimTypes": [
      "quality-inspection"
    ],
    "gpuTimestampQueries": false,
    "perplexity": {
      "wordSegmentation": "doppler.word-segmentation/unicode-whitespace-v1",
      "aggregation": "doppler.perplexity/summed-word-surprisal-v1",
      "rollingWindow": {
        "unit": "words",
        "size": 8
      }
    }
  },
  "fingerprint": {
    "schema": "doppler.comparison-fingerprint/v1",
    "identity": {
      "artifact": {
        "modelId": "gemma-3-270m-it-q4k-ehf16-af32",
        "manifestHash": "sha256:230104df762ff394095326d8e9fa4dc144d431bbd61ad5139e1030e66836ab78"
      },
      "tokenizer": {
        "contract": {
          "type": "bundled",
          "vocabSize": 262144,
          "file": "tokenizer.json",
          "addBosToken": true
        },
        "digest": "sha256:58b3e0e0de7db802decbf097dd1666a0862f6ff96727ecff82bf49787279bede"
      },
      "promptTokenIds": [
        2,
        155122,
        3217,
        506,
        7217,
        563,
        3730,
        528,
        886,
        13315,
        236761
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
        "maxTokens": 64,
        "stopSequences": [],
        "useChatTemplate": true,
        "useSpeculative": null,
        "seed": null
      },
      "observationPolicy": {
        "id": "demo/guided-quality",
        "modifiesExecution": true,
        "performanceRepresentative": false,
        "requiredCaptures": [
          "artifact-identity",
          "prompt-token-ids",
          "generated-token-ids",
          "selected-token-probabilities",
          "top-candidates",
          "word-quality"
        ],
        "allowedClaimTypes": [
          "quality-inspection"
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
        "executionPlanId": null,
        "kernelPathId": null,
        "kernelPathSource": "none",
        "activationDtype": null,
        "hasF16": true,
        "hasSubgroups": true
      },
      "browser": {
        "userAgent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/154.0.0.0 Safari/537.36",
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
    "fullDigest": "sha256:5ec9bae6c9cb2d327c480f2dbeafb005735d84586c5a751e2b68888acb523b44",
    "qualityDigest": "sha256:565ba95caba400678956248757dbd281a9a0a93a165ffb099d39bea5016f1beb",
    "performanceDigest": "sha256:556f0ec9272c2181592e47c1a8cb460afd844e5b6137e6e7485588734c660c1d"
  },
  "outputText": "The sky is blue because of a phenomenon called **Rayleigh-induced scattering of sunlight.**\n",
  "generatedTokenIds": [
    818,
    7217,
    563,
    3730,
    1547,
    529,
    496,
    20284,
    2760,
    5213,
    30958,
    53700,
    236772,
    21681,
    19389,
    529,
    26808,
    99382,
    107,
    106
  ],
  "wallTimingMs": 724.6999999284744,
  "performanceRepresentative": false,
  "tokens": [
    {
      "index": 0,
      "tokenId": 818,
      "text": "The",
      "probability": 0.9977418896456159,
      "surprisal": 0.002260663730164067,
      "topCandidates": [
        {
          "tokenId": 818,
          "logit": 35.02208709716797,
          "text": "The",
          "probability": 0.9977418896456159
        },
        {
          "tokenId": 1509,
          "logit": 28.378780364990234,
          "text": "It",
          "probability": 0.0012997707244093405
        },
        {
          "tokenId": 236776,
          "logit": 26.966976165771484,
          "text": "A",
          "probability": 0.0003167582811050711
        },
        {
          "tokenId": 3810,
          "logit": 26.434093475341797,
          "text": "There",
          "probability": 0.00018590880771945538
        },
        {
          "tokenId": 4339,
          "logit": 26.241464614868164,
          "text": "True",
          "probability": 0.0001533353590516697
        }
      ]
    },
    {
      "index": 1,
      "tokenId": 7217,
      "text": " sky",
      "probability": 0.5253852341342858,
      "surprisal": 0.6436235061715204,
      "topCandidates": [
        {
          "tokenId": 7217,
          "logit": 28.53620147705078,
          "text": " sky",
          "probability": 0.5253852341342858
        },
        {
          "tokenId": 2258,
          "logit": 27.619169235229492,
          "text": " color",
          "probability": 0.20999832006380992
        },
        {
          "tokenId": 3768,
          "logit": 27.478052139282227,
          "text": " sun",
          "probability": 0.18235994158952168
        },
        {
          "tokenId": 27801,
          "logit": 26.14948272705078,
          "text": " longest",
          "probability": 0.048299104630245074
        },
        {
          "tokenId": 3730,
          "logit": 24.89862823486328,
          "text": " blue",
          "probability": 0.013826105812166095
        }
      ]
    },
    {
      "index": 2,
      "tokenId": 563,
      "text": " is",
      "probability": 0.9453465637224017,
      "surprisal": 0.05620368462112735,
      "topCandidates": [
        {
          "tokenId": 563,
          "logit": 34.62862014770508,
          "text": " is",
          "probability": 0.9453465637224017
        },
        {
          "tokenId": 236789,
          "logit": 31.560392379760742,
          "text": "'",
          "probability": 0.043961920956869495
        },
        {
          "tokenId": 7412,
          "logit": 29.051271438598633,
          "text": " appears",
          "probability": 0.0035758499110750213
        },
        {
          "tokenId": 528,
          "logit": 28.561614990234375,
          "text": " in",
          "probability": 0.0021914127712997964
        },
        {
          "tokenId": 236764,
          "logit": 28.4368896484375,
          "text": ",",
          "probability": 0.0019344462215217855
        }
      ]
    },
    {
      "index": 3,
      "tokenId": 3730,
      "text": " blue",
      "probability": 0.9941460142663466,
      "surprisal": 0.0058711874734833375,
      "topCandidates": [
        {
          "tokenId": 3730,
          "logit": 36.33572006225586,
          "text": " blue",
          "probability": 0.9941460142663466
        },
        {
          "tokenId": 236743,
          "logit": 29.494611740112305,
          "text": " ",
          "probability": 0.0010626606064077823
        },
        {
          "tokenId": 784,
          "logit": 29.17104148864746,
          "text": " all",
          "probability": 0.0007688999037677555
        },
        {
          "tokenId": 496,
          "logit": 28.92782211303711,
          "text": " a",
          "probability": 0.0006028940167154507
        },
        {
          "tokenId": 6816,
          "logit": 28.853656768798828,
          "text": " generally",
          "probability": 0.0005597980406767861
        }
      ]
    },
    {
      "index": 4,
      "tokenId": 1547,
      "text": " because",
      "probability": 0.9213166940220324,
      "surprisal": 0.0819514429597441,
      "topCandidates": [
        {
          "tokenId": 1547,
          "logit": 31.165058135986328,
          "text": " because",
          "probability": 0.9213166940220324
        },
        {
          "tokenId": 573,
          "logit": 28.225841522216797,
          "text": " for",
          "probability": 0.04874424904868047
        },
        {
          "tokenId": 2779,
          "logit": 27.430809020996094,
          "text": " due",
          "probability": 0.022011272775917374
        },
        {
          "tokenId": 13336,
          "logit": 25.966144561767578,
          "text": " primarily",
          "probability": 0.005088027657251671
        },
        {
          "tokenId": 618,
          "logit": 24.803483963012695,
          "text": " as",
          "probability": 0.0015907882737749817
        }
      ]
    },
    {
      "index": 5,
      "tokenId": 529,
      "text": " of",
      "probability": 0.9916955365853845,
      "surprisal": 0.008339137571201126,
      "topCandidates": [
        {
          "tokenId": 529,
          "logit": 40.76757049560547,
          "text": " of",
          "probability": 0.9916955365853845
        },
        {
          "tokenId": 993,
          "logit": 35.480064392089844,
          "text": " there",
          "probability": 0.005012375918414709
        },
        {
          "tokenId": 26808,
          "logit": 34.289066314697266,
          "text": " sunlight",
          "probability": 0.0015233501575228428
        },
        {
          "tokenId": 496,
          "logit": 33.17702865600586,
          "text": " a",
          "probability": 0.0005010117634721089
        },
        {
          "tokenId": 112242,
          "logit": 33.04151153564453,
          "text": " chlorophyll",
          "probability": 0.00043751564293223103
        }
      ]
    },
    {
      "index": 6,
      "tokenId": 496,
      "text": " a",
      "probability": 0.849430390877782,
      "surprisal": 0.1631892825132395,
      "topCandidates": [
        {
          "tokenId": 496,
          "logit": 22.49373435974121,
          "text": " a",
          "probability": 0.849430390877782
        },
        {
          "tokenId": 26808,
          "logit": 19.91317367553711,
          "text": " sunlight",
          "probability": 0.06432866367661848
        },
        {
          "tokenId": 614,
          "logit": 19.416833877563477,
          "text": " an",
          "probability": 0.03916037972182691
        },
        {
          "tokenId": 121707,
          "logit": 18.692604064941406,
          "text": " Rayleigh",
          "probability": 0.018980947286727792
        },
        {
          "tokenId": 506,
          "logit": 18.032981872558594,
          "text": " the",
          "probability": 0.00981403505192723
        }
      ]
    },
    {
      "index": 7,
      "tokenId": 20284,
      "text": " phenomenon",
      "probability": 0.9165863072519459,
      "surprisal": 0.08709904564847336,
      "topCandidates": [
        {
          "tokenId": 20284,
          "logit": 14.773636817932129,
          "text": " phenomenon",
          "probability": 0.9165863072519459
        },
        {
          "tokenId": 8376,
          "logit": 11.975756645202637,
          "text": " combination",
          "probability": 0.05585595030167085
        },
        {
          "tokenId": 3530,
          "logit": 10.117792129516602,
          "text": " specific",
          "probability": 0.008712959760574799
        },
        {
          "tokenId": 6495,
          "logit": 9.232254981994629,
          "text": " slight",
          "probability": 0.003594030941004244
        },
        {
          "tokenId": 38161,
          "logit": 8.401634216308594,
          "text": " pigment",
          "probability": 0.0015662020805029778
        }
      ]
    },
    {
      "index": 8,
      "tokenId": 2760,
      "text": " called",
      "probability": 0.9812109738585915,
      "surprisal": 0.018967782540296248,
      "topCandidates": [
        {
          "tokenId": 2760,
          "logit": 28.03939437866211,
          "text": " called",
          "probability": 0.9812109738585915
        },
        {
          "tokenId": 3224,
          "logit": 23.797853469848633,
          "text": " known",
          "probability": 0.014115120343914917
        },
        {
          "tokenId": 528,
          "logit": 21.872406005859375,
          "text": " in",
          "probability": 0.002058132716934839
        },
        {
          "tokenId": 529,
          "logit": 21.605365753173828,
          "text": " of",
          "probability": 0.001575793367667328
        },
        {
          "tokenId": 1298,
          "logit": 19.856693267822266,
          "text": " where",
          "probability": 0.000274195584717962
        }
      ]
    },
    {
      "index": 9,
      "tokenId": 5213,
      "text": " **",
      "probability": 0.8681574215363301,
      "surprisal": 0.14138221954877403,
      "topCandidates": [
        {
          "tokenId": 5213,
          "logit": 15.875510215759277,
          "text": " **",
          "probability": 0.8681574215363301
        },
        {
          "tokenId": 121707,
          "logit": 13.981234550476074,
          "text": " Rayleigh",
          "probability": 0.1305945380706196
        },
        {
          "tokenId": 808,
          "logit": 8.76159381866455,
          "text": " *",
          "probability": 0.000706421398614817
        },
        {
          "tokenId": 12705,
          "logit": 6.683739185333252,
          "text": " quantum",
          "probability": 8.844291432291639e-05
        },
        {
          "tokenId": 496,
          "logit": 6.2877068519592285,
          "text": " a",
          "probability": 5.95207490097862e-05
        }
      ]
    },
    {
      "index": 10,
      "tokenId": 30958,
      "text": "Ray",
      "probability": 0.5809856604058035,
      "surprisal": 0.5430292033198237,
      "topCandidates": [
        {
          "tokenId": 30958,
          "logit": 15.334250450134277,
          "text": "Ray",
          "probability": 0.5809856604058035
        },
        {
          "tokenId": 154030,
          "logit": 13.457769393920898,
          "text": "scattering",
          "probability": 0.08896517663930519
        },
        {
          "tokenId": 236755,
          "logit": 13.196212768554688,
          "text": "c",
          "probability": 0.0684900441673598
        },
        {
          "tokenId": 29188,
          "logit": 12.450703620910645,
          "text": "isot",
          "probability": 0.0324980226660011
        },
        {
          "tokenId": 147505,
          "logit": 12.263327598571777,
          "text": "Einstein",
          "probability": 0.026945147462691756
        }
      ]
    },
    {
      "index": 11,
      "tokenId": 53700,
      "text": "leigh",
      "probability": 0.9997981530735081,
      "surprisal": 0.00020186730032445446,
      "topCandidates": [
        {
          "tokenId": 53700,
          "logit": 43.2318229675293,
          "text": "leigh",
          "probability": 0.9997981530735081
        },
        {
          "tokenId": 1733,
          "logit": 34.18557357788086,
          "text": " Tr",
          "probability": 0.00011780836760737865
        },
        {
          "tokenId": 1700,
          "logit": 33.29893493652344,
          "text": " Con",
          "probability": 4.8541575602888075e-05
        },
        {
          "tokenId": 529,
          "logit": 32.3482780456543,
          "text": " of",
          "probability": 1.8760690853676278e-05
        },
        {
          "tokenId": 11379,
          "logit": 30.9737606048584,
          "text": " Cond",
          "probability": 4.745735038119092e-06
        }
      ]
    },
    {
      "index": 12,
      "tokenId": 236772,
      "text": "-",
      "probability": 0.8159920044555004,
      "surprisal": 0.20335072252743192,
      "topCandidates": [
        {
          "tokenId": 236772,
          "logit": 30.254804611206055,
          "text": "-",
          "probability": 0.8159920044555004
        },
        {
          "tokenId": 1788,
          "logit": 28.456811904907227,
          "text": "ness",
          "probability": 0.13515359198785948
        },
        {
          "tokenId": 4314,
          "logit": 25.861818313598633,
          "text": " gas",
          "probability": 0.010088722933845377
        },
        {
          "tokenId": 4918,
          "logit": 25.586467742919922,
          "text": " band",
          "probability": 0.007660426707711755
        },
        {
          "tokenId": 60596,
          "logit": 25.30961799621582,
          "text": " optics",
          "probability": 0.005807893505151346
        }
      ]
    },
    {
      "index": 13,
      "tokenId": 21681,
      "text": "induced",
      "probability": 0.30549026748678604,
      "surprisal": 1.185837358531681,
      "topCandidates": [
        {
          "tokenId": 21681,
          "logit": 36.87627029418945,
          "text": "induced",
          "probability": 0.30549026748678604
        },
        {
          "tokenId": 10619,
          "logit": 36.17497253417969,
          "text": "related",
          "probability": 0.15150523221727497
        },
        {
          "tokenId": 87938,
          "logit": 36.110740661621094,
          "text": "dominated",
          "probability": 0.1420797170676024
        },
        {
          "tokenId": 154030,
          "logit": 35.70878601074219,
          "text": "scattering",
          "probability": 0.09505290553934304
        },
        {
          "tokenId": 18318,
          "logit": 34.76083755493164,
          "text": "Sun",
          "probability": 0.0368363519082746
        }
      ]
    },
    {
      "index": 14,
      "tokenId": 19389,
      "text": " scattering",
      "probability": 0.9247566007021066,
      "surprisal": 0.07822471047173706,
      "topCandidates": [
        {
          "tokenId": 19389,
          "logit": 25.930952072143555,
          "text": " scattering",
          "probability": 0.9247566007021066
        },
        {
          "tokenId": 1700,
          "logit": 22.150737762451172,
          "text": " Con",
          "probability": 0.02110091192659536
        },
        {
          "tokenId": 178868,
          "logit": 21.338760375976562,
          "text": " Scattering",
          "probability": 0.009368367664192221
        },
        {
          "tokenId": 51585,
          "logit": 21.078083038330078,
          "text": " condensation",
          "probability": 0.0072186036555831725
        },
        {
          "tokenId": 16204,
          "logit": 20.91907501220703,
          "text": " cooling",
          "probability": 0.006157393214859792
        }
      ]
    },
    {
      "index": 15,
      "tokenId": 529,
      "text": " of",
      "probability": 0.45442117266540943,
      "surprisal": 0.7887308178864697,
      "topCandidates": [
        {
          "tokenId": 529,
          "logit": 27.94728660583496,
          "text": " of",
          "probability": 0.45442117266540943
        },
        {
          "tokenId": 99382,
          "logit": 27.285871505737305,
          "text": ".**",
          "probability": 0.23453606256734377
        },
        {
          "tokenId": 84750,
          "logit": 27.099016189575195,
          "text": "**.",
          "probability": 0.19456261608265182
        },
        {
          "tokenId": 1018,
          "logit": 26.490188598632812,
          "text": "**",
          "probability": 0.10583978134397475
        },
        {
          "tokenId": 532,
          "logit": 23.89683723449707,
          "text": " and",
          "probability": 0.007913538997559588
        }
      ]
    },
    {
      "index": 16,
      "tokenId": 26808,
      "text": " sunlight",
      "probability": 0.8494666755091058,
      "surprisal": 0.16314656699851268,
      "topCandidates": [
        {
          "tokenId": 26808,
          "logit": 28.187301635742188,
          "text": " sunlight",
          "probability": 0.8494666755091058
        },
        {
          "tokenId": 2214,
          "logit": 25.870746612548828,
          "text": " light",
          "probability": 0.08376821912104854
        },
        {
          "tokenId": 35085,
          "logit": 24.282480239868164,
          "text": " electromagnetic",
          "probability": 0.017112125676627846
        },
        {
          "tokenId": 52243,
          "logit": 24.199851989746094,
          "text": " photons",
          "probability": 0.015755020336225126
        },
        {
          "tokenId": 3826,
          "logit": 24.19240379333496,
          "text": " green",
          "probability": 0.015638109777176102
        }
      ]
    },
    {
      "index": 17,
      "tokenId": 99382,
      "text": ".**",
      "probability": 0.41588240072276167,
      "surprisal": 0.8773527492556694,
      "topCandidates": [
        {
          "tokenId": 99382,
          "logit": 34.51542663574219,
          "text": ".**",
          "probability": 0.41588240072276167
        },
        {
          "tokenId": 84750,
          "logit": 34.515079498291016,
          "text": "**.",
          "probability": 0.41573805742111997
        },
        {
          "tokenId": 532,
          "logit": 33.34981918334961,
          "text": " and",
          "probability": 0.12964436881982094
        },
        {
          "tokenId": 1018,
          "logit": 30.56464195251465,
          "text": "**",
          "probability": 0.008001410562893605
        },
        {
          "tokenId": 236764,
          "logit": 30.441553115844727,
          "text": ",",
          "probability": 0.007074728086795765
        }
      ]
    },
    {
      "index": 18,
      "tokenId": 107,
      "text": "\n",
      "probability": 0.9775577555253775,
      "surprisal": 0.02269790393161476,
      "topCandidates": [
        {
          "tokenId": 107,
          "logit": 16.116565704345703,
          "text": "\n",
          "probability": 0.9775577555253775
        },
        {
          "tokenId": 146430,
          "logit": 11.514627456665039,
          "text": " Sunlight",
          "probability": 0.009807222728910717
        },
        {
          "tokenId": 1174,
          "logit": 10.389909744262695,
          "text": " This",
          "probability": 0.0031848379702512806
        },
        {
          "tokenId": 106,
          "logit": 10.36296558380127,
          "text": "",
          "probability": 0.0031001709480597156
        },
        {
          "tokenId": 10847,
          "logit": 10.152122497558594,
          "text": " Light",
          "probability": 0.0025108319898098845
        }
      ]
    },
    {
      "index": 19,
      "tokenId": 106,
      "text": "",
      "probability": 0.9999525219914334,
      "surprisal": 4.7479135682961446e-05,
      "topCandidates": [
        {
          "tokenId": 106,
          "logit": 9.802950859069824,
          "text": "",
          "probability": 0.9999525219914334
        },
        {
          "tokenId": 2717,
          "logit": -0.5390890836715698,
          "text": "```",
          "probability": 3.224693938950387e-05
        },
        {
          "tokenId": 8291,
          "logit": -3.935451030731201,
          "text": "Here",
          "probability": 1.0801081651389171e-06
        },
        {
          "tokenId": 9474,
          "logit": -4.377915859222412,
          "text": "More",
          "probability": 6.939165022574772e-07
        },
        {
          "tokenId": 236776,
          "logit": -5.132486343383789,
          "text": "A",
          "probability": 3.2628823767059006e-07
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
        "text": "The",
        "tokenIndexes": [
          0
        ],
        "tokenCount": 1,
        "summedSurprisal": 0.002260663730164067,
        "probabilityAvailable": true,
        "wordIndex": 0,
        "rollingPerplexity": 1.0022632209570614,
        "cumulativePerplexity": 1.0022632209570614,
        "rollingWindow": {
          "unit": "words",
          "size": 8,
          "tokenCount": 1
        }
      },
      {
        "text": "sky",
        "tokenIndexes": [
          1
        ],
        "tokenCount": 1,
        "summedSurprisal": 0.6436235061715204,
        "probabilityAvailable": true,
        "wordIndex": 1,
        "rollingPerplexity": 1.3811853571713752,
        "cumulativePerplexity": 1.3811853571713752,
        "rollingWindow": {
          "unit": "words",
          "size": 8,
          "tokenCount": 2
        }
      },
      {
        "text": "is",
        "tokenIndexes": [
          2
        ],
        "tokenCount": 1,
        "summedSurprisal": 0.05620368462112735,
        "probabilityAvailable": true,
        "wordIndex": 2,
        "rollingPerplexity": 1.2636814983775906,
        "cumulativePerplexity": 1.2636814983775906,
        "rollingWindow": {
          "unit": "words",
          "size": 8,
          "tokenCount": 3
        }
      },
      {
        "text": "blue",
        "tokenIndexes": [
          3
        ],
        "tokenCount": 1,
        "summedSurprisal": 0.0058711874734833375,
        "probabilityAvailable": true,
        "wordIndex": 3,
        "rollingPerplexity": 1.1936188710030295,
        "cumulativePerplexity": 1.1936188710030295,
        "rollingWindow": {
          "unit": "words",
          "size": 8,
          "tokenCount": 4
        }
      },
      {
        "text": "because",
        "tokenIndexes": [
          4
        ],
        "tokenCount": 1,
        "summedSurprisal": 0.0819514429597441,
        "probabilityAvailable": true,
        "wordIndex": 4,
        "rollingPerplexity": 1.1711452274966772,
        "cumulativePerplexity": 1.1711452274966772,
        "rollingWindow": {
          "unit": "words",
          "size": 8,
          "tokenCount": 5
        }
      },
      {
        "text": "of",
        "tokenIndexes": [
          5
        ],
        "tokenCount": 1,
        "summedSurprisal": 0.008339137571201126,
        "probabilityAvailable": true,
        "wordIndex": 5,
        "rollingPerplexity": 1.1422975212080064,
        "cumulativePerplexity": 1.1422975212080064,
        "rollingWindow": {
          "unit": "words",
          "size": 8,
          "tokenCount": 6
        }
      },
      {
        "text": "a",
        "tokenIndexes": [
          6
        ],
        "tokenCount": 1,
        "summedSurprisal": 0.1631892825132395,
        "probabilityAvailable": true,
        "wordIndex": 6,
        "rollingPerplexity": 1.1472277902729457,
        "cumulativePerplexity": 1.1472277902729457,
        "rollingWindow": {
          "unit": "words",
          "size": 8,
          "tokenCount": 7
        }
      },
      {
        "text": "phenomenon",
        "tokenIndexes": [
          7
        ],
        "tokenCount": 1,
        "summedSurprisal": 0.08709904564847336,
        "probabilityAvailable": true,
        "wordIndex": 7,
        "rollingPerplexity": 1.140044439696031,
        "cumulativePerplexity": 1.140044439696031,
        "rollingWindow": {
          "unit": "words",
          "size": 8,
          "tokenCount": 8
        }
      },
      {
        "text": "called",
        "tokenIndexes": [
          8
        ],
        "tokenCount": 1,
        "summedSurprisal": 0.018967782540296248,
        "probabilityAvailable": true,
        "wordIndex": 8,
        "rollingPerplexity": 1.1424277847444695,
        "cumulativePerplexity": 1.1259326879519196,
        "rollingWindow": {
          "unit": "words",
          "size": 8,
          "tokenCount": 8
        }
      },
      {
        "text": "**Rayleigh-induced",
        "tokenIndexes": [
          9,
          10,
          11,
          12,
          13
        ],
        "tokenCount": 5,
        "summedSurprisal": 2.0738013712280354,
        "probabilityAvailable": true,
        "wordIndex": 9,
        "rollingPerplexity": 1.2311539634245514,
        "cumulativePerplexity": 1.2515453639427914,
        "rollingWindow": {
          "unit": "words",
          "size": 8,
          "tokenCount": 12
        }
      },
      {
        "text": "scattering",
        "tokenIndexes": [
          14
        ],
        "tokenCount": 1,
        "summedSurprisal": 0.07822471047173706,
        "probabilityAvailable": true,
        "wordIndex": 10,
        "rollingPerplexity": 1.2334153104437233,
        "cumulativePerplexity": 1.2394099930532032,
        "rollingWindow": {
          "unit": "words",
          "size": 8,
          "tokenCount": 12
        }
      },
      {
        "text": "of",
        "tokenIndexes": [
          15
        ],
        "tokenCount": 1,
        "summedSurprisal": 0.7887308178864697,
        "probabilityAvailable": true,
        "wordIndex": 11,
        "rollingPerplexity": 1.3165639825077888,
        "cumulativePerplexity": 1.2846886773407182,
        "rollingWindow": {
          "unit": "words",
          "size": 8,
          "tokenCount": 12
        }
      },
      {
        "text": "sunlight.**",
        "tokenIndexes": [
          16,
          17
        ],
        "tokenCount": 2,
        "summedSurprisal": 1.040499316254182,
        "probabilityAvailable": true,
        "wordIndex": 12,
        "rollingPerplexity": 1.3876393010858332,
        "cumulativePerplexity": 1.3237740355665957,
        "rollingWindow": {
          "unit": "words",
          "size": 8,
          "tokenCount": 13
        }
      },
      {
        "text": "",
        "tokenIndexes": [
          19
        ],
        "tokenCount": 1,
        "summedSurprisal": 4.7479135682961446e-05,
        "probabilityAvailable": true,
        "wordIndex": 13,
        "rollingPerplexity": 1.386754519349495,
        "cumulativePerplexity": 1.304378669422667,
        "rollingWindow": {
          "unit": "words",
          "size": 8,
          "tokenCount": 13
        }
      }
    ]
  },
  "generationEvidence": {
    "schema": "doppler_generation_evidence/v1",
    "outputText": "The sky is blue because of a phenomenon called **Rayleigh-induced scattering of sunlight.**\n",
    "tokenIds": [
      818,
      7217,
      563,
      3730,
      1547,
      529,
      496,
      20284,
      2760,
      5213,
      30958,
      53700,
      236772,
      21681,
      19389,
      529,
      26808,
      99382,
      107,
      106
    ],
    "transcript": {
      "schema": "doppler_generation_transcript/v1",
      "outputText": "The sky is blue because of a phenomenon called **Rayleigh-induced scattering of sunlight.**\n",
      "tokenIds": [
        818,
        7217,
        563,
        3730,
        1547,
        529,
        496,
        20284,
        2760,
        5213,
        30958,
        53700,
        236772,
        21681,
        19389,
        529,
        26808,
        99382,
        107,
        106
      ]
    },
    "transcriptHash": "sha256:e3944a0187f7e1b21d1da12a25ae398040ef8ae4b493fa14090b9e1549ee6a5e",
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
      "maxTokens": 64,
      "stopSequences": [],
      "useChatTemplate": true,
      "useSpeculative": null,
      "seed": null
    },
    "generationConfigHash": "sha256:2c2ba6a18436a5bad36ef3750312b8f513609cf8224aee491704e8bd73aeb7bc",
    "resolution": {
      "schema": "doppler.resolution-identity/v1",
      "logicalModelId": "gemma-3-270m-it-q4k-ehf16-af32",
      "resolvedArtifactVariantId": "sha256:230104df762ff394095326d8e9fa4dc144d431bbd61ad5139e1030e66836ab78",
      "resolvedExecutionId": "sha256:09644f0973c9e0be10b6e375aae8bbe1d5bd27d0cbb0290bf53b408ff050e6dc"
    },
    "executionIdentity": {
      "schema": "doppler.resolved-execution-identity/v1",
      "runtime": {
        "package": "doppler-gpu",
        "version": "0.6.3-dev.split.10",
        "surface": "browser"
      },
      "resolvedRuntimeSessionId": "sha256:3d238a7a9b3934f7b677123cee5289594877d5a794eef1c1359ca579e9e7a676",
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
        "kernelPathId": null,
        "kernelPathSource": "none",
        "executionPlanId": null,
        "activationDtype": null
      }
    },
    "runtimeProfile": {
      "schema": "doppler_runtime_profile/v1",
      "runtime": {
        "package": "doppler-gpu",
        "version": "0.6.3-dev.split.10",
        "surface": "browser"
      },
      "model": {
        "modelId": "gemma-3-270m-it-q4k-ehf16-af32",
        "manifestHash": "sha256:230104df762ff394095326d8e9fa4dc144d431bbd61ad5139e1030e66836ab78",
        "activeAdapter": null,
        "activeAdapterId": null,
        "activeAdapterDigest": null
      },
      "resolvedRuntimeSessionId": "sha256:3d238a7a9b3934f7b677123cee5289594877d5a794eef1c1359ca579e9e7a676",
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
        "kernelPathId": null,
        "kernelPathSource": "none",
        "executionPlanId": null,
        "activationDtype": null
      }
    },
    "runtimeProfileHash": "sha256:6db4ef5de8d71697ae18bb7d63881aec3dc36a54b1bef5bec60df0f2b174a44d",
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
      "kernelPathId": null,
      "kernelPathSource": "none",
      "executionPlanId": null,
      "activationDtype": null
    },
    "backendIdentityHash": "sha256:f637c41d7754cd1075dd8c1a4c8d76cd87d854945ea9bd722c06c38cb3429039",
    "stats": {
      "prefillTimeMs": 375.10000002384186,
      "decodeTimeMs": 342.1999999284744,
      "ttftMs": 379,
      "loadTiming": {
        "schemaVersion": 1,
        "source": "doppler-loader",
        "modelId": "gemma-3-270m-it-q4k-ehf16-af32",
        "status": "complete",
        "customShardLoader": true,
        "byteAccountingMode": "custom-loader-read-progress",
        "totalBytes": 399357184,
        "totalShards": 6,
        "bytesLoaded": 399357184,
        "shardsLoaded": 6,
        "bytesPerSecond": 1281222919,
        "phasesMs": {
          "preflight": 0.5,
          "tensorLocations": 0.7,
          "embeddings": 141.8,
          "layers": 167.1,
          "finalWeights": 1.2,
          "cleanup": 0
        },
        "layers": {
          "count": 18,
          "totalMs": 158.4,
          "meanMs": 8.8,
          "maxMs": 72.2,
          "maxLayer": 0
        },
        "totalMs": 311.7,
        "failedPhase": null,
        "error": null
      },
      "pipelineLoadTiming": {
        "schemaVersion": 1,
        "source": "doppler-pipeline",
        "modelId": "gemma-3-270m-it-q4k-ehf16-af32",
        "status": "complete",
        "phasesMs": {
          "reset": 0,
          "configResolution": 3.5,
          "kernelWarmup": 0,
          "tokenizer": 291.4,
          "executionSetup": 3.9,
          "loadWeights": 358.5,
          "rope": 4.9,
          "convStates": 0.3
        },
        "details": {
          "tokenizer": {
            "schemaVersion": 1,
            "source": "doppler-tokenizer",
            "modelId": "gemma-3-270m-it-q4k-ehf16-af32",
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
              "assetLoad": 129.9,
              "assetParse": 0,
              "backendLoad": 161,
              "cacheStore": 0
            },
            "totalMs": 291.2,
            "error": null
          }
        },
        "totalMs": 662.6
      },
      "prefillTokens": 19,
      "decodeTokens": 20,
      "memoryUsageBytes": 0,
      "tokensGenerated": 20,
      "totalTimeMs": 721.5,
      "decodeRecordMs": 0,
      "decodeRecordOps": 0,
      "decodeRecordPasses": 0,
      "decodeRecordOpLabels": {},
      "decodeSubmitWaitMs": 0,
      "decodeReadbackWaitMs": 0,
      "decodeReadbackMapWaitMs": 0,
      "decodeReadbackCleanupMs": 0,
      "decodeReadbackCopyMs": 0,
      "prefillRecordMs": 284.90000009536743,
      "prefillRecordOps": 0,
      "prefillRecordPasses": 0,
      "prefillRecordOpLabels": {},
      "prefillSubmitWaitMs": 0,
      "prefillProfileSteps": [],
      "decodeProfileSteps": [],
      "executionPlan": null,
      "kernelPathId": null,
      "kernelPathSource": "none",
      "operatorDiagnostics": null,
      "attentionInputs": [
        {
          "phase": "prefill",
          "layerIdx": 0,
          "numTokens": 19,
          "kvLen": 19,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 1,
          "numTokens": 19,
          "kvLen": 19,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 2,
          "numTokens": 19,
          "kvLen": 19,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 3,
          "numTokens": 19,
          "kvLen": 19,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 4,
          "numTokens": 19,
          "kvLen": 19,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 5,
          "numTokens": 19,
          "kvLen": 19,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 6,
          "numTokens": 19,
          "kvLen": 19,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "numTokens": 19,
          "kvLen": 19,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 8,
          "numTokens": 19,
          "kvLen": 19,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 9,
          "numTokens": 19,
          "kvLen": 19,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 10,
          "numTokens": 19,
          "kvLen": 19,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "numTokens": 19,
          "kvLen": 19,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 12,
          "numTokens": 19,
          "kvLen": 19,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 13,
          "numTokens": 19,
          "kvLen": 19,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 14,
          "numTokens": 19,
          "kvLen": 19,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "numTokens": 19,
          "kvLen": 19,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 16,
          "numTokens": 19,
          "kvLen": 19,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 17,
          "numTokens": 19,
          "kvLen": 19,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 0,
          "numTokens": 1,
          "kvLen": 20,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 1,
          "numTokens": 1,
          "kvLen": 20,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 2,
          "numTokens": 1,
          "kvLen": 20,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "kvLen": 20,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 4,
          "numTokens": 1,
          "kvLen": 20,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 5,
          "numTokens": 1,
          "kvLen": 20,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 6,
          "numTokens": 1,
          "kvLen": 20,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "kvLen": 20,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 8,
          "numTokens": 1,
          "kvLen": 20,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 9,
          "numTokens": 1,
          "kvLen": 20,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 10,
          "numTokens": 1,
          "kvLen": 20,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "kvLen": 20,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 12,
          "numTokens": 1,
          "kvLen": 20,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 13,
          "numTokens": 1,
          "kvLen": 20,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 14,
          "numTokens": 1,
          "kvLen": 20,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "kvLen": 20,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 16,
          "numTokens": 1,
          "kvLen": 20,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
          "layerIdx": 17,
          "numTokens": 1,
          "kvLen": 20,
          "numHeads": 4,
          "numKVHeads": 1,
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
          "useFusedQKV": true,
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
      "singleTokenReadbackWaitMs": 0,
      "singleTokenReadbackMapWaitMs": 0,
      "singleTokenReadbackCleanupMs": 0,
      "singleTokenReadbackCopyMs": 0,
      "singleTokenOrchestrationMs": 0,
      "plePreparedTokenCacheHits": 0,
      "plePreparedTokenCacheMisses": 0,
      "plePreparedTokenCacheEntries": 0,
      "plePreparedTokenCacheBytes": 0,
      "pleHotVocabularyHits": 0,
      "pleHotVocabularyMisses": 0,
      "modelLoadMs": 662.6000000238419,
      "gpuTimePrefillMs": null,
      "gpuTimeDecodeMs": null,
      "stopReason": "stop-token",
      "stopTokenId": 106,
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
  "recording": {
    "kind": "precomputed-run",
    "recordedAt": "2026-10-04",
    "sourceCommit": "4015be029a5742e06b9191073f04b741804de73e",
    "modelId": "gemma-3-270m-it-q4k-ehf16-af32",
    "surface": "Chrome WebGPU on macOS",
    "rawReceiptSha256": "ccc820af423446295efda5040da6ff1e606125650d3d935afbcca32fd22aac34"
  }
};

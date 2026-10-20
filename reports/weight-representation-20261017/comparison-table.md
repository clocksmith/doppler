# Frozen baseline versus candidate

| Host / model / engine / order | Open ms | Warm TTFT ms | Warm decode tok/s | Pool model MiB | Peak pool MiB | Peak summed RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| metal / baseline-2b-first | 5180.8 | 124.7 | 42.65 | 1982.2 | 2612.0 | 2688.3 |
| metal / candidate-2b-first | 3840.2 | 113.0 | 43.64 | 1982.2 | 2612.0 | 3371.7 |
| metal / candidate-2b-second | 4184.6 | 111.9 | 43.24 | 1982.2 | 2612.0 | 3163.8 |
| metal / baseline-2b-second | 3794.5 | 116.3 | 43.48 | 1982.2 | 2612.0 | 3361.5 |
| vulkan / baseline-2b-first | 5858.1 | 70.1 | 80.98 | 1982.2 | 2612.0 | 2015.5 |
| vulkan / candidate-2b-first | 5892.2 | 70.8 | 79.32 | 1982.2 | 2612.0 | 2021.0 |
| vulkan / candidate-2b-second | 5971.0 | 71.3 | 81.36 | 1982.2 | 2612.0 | 2085.4 |
| vulkan / baseline-2b-second | 5898.7 | 65.1 | 81.34 | 1982.2 | 2612.0 | 2034.3 |
| vulkan / baseline-27b-first | 53093.1 | 283.0 | 8.86 | 16214.1 | 17539.2 | 1611.2 |
| vulkan / candidate-27b-first | 53392.1 | 271.8 | 8.79 | 16214.1 | 17539.2 | 1698.4 |
| vulkan / candidate-27b-second | 53783.3 | 274.7 | 8.84 | 16214.1 | 17539.2 | 1606.0 |
| vulkan / baseline-27b-second | 52586.1 | 286.3 | 8.82 | 16214.1 | 17539.2 | 1610.6 |

See `summary.json` for long prefill, first request, recreated-device opening, failure and identity details. These are scoped measurements, not a general speedup claim.

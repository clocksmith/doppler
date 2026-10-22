| platform | lane | runtime | test acc | macro-F1 | ECE | Brier | cold ms (median) | warm median/p90 ms | peak RSS GiB | GPU mem GiB | MB fetched |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chrome | doppler-capsule-q4k | capsule | 0.573 | 0.523 | 0.218 | 0.621 | 11187 | 542/781 | 2.42 | 5.99 | 945 |
| chrome | doppler-f16 | compat f16 | 0.587 | 0.528 | 0.180 | 0.563 | 4629 | 578/829 | 1.48 | 7.07 | 1208 |
| chrome | open-jev-deberta | webgpu fp16 | 0.580 | 0.480 | 0.136 | 0.587 | 3553 | 81/86 | 3.43 | 1.98 | 884 |
| chrome | open-jev-kev-0.6b | webgpu q4f16 | 0.567 | 0.518 | 0.275 | 0.660 | 3002 | 83/118 | 2.23 | 0.94 | 346 |
| chrome | open-jev-kev-4b | webgpu q4f16 | 0.720 | 0.697 | 0.187 | 0.469 | 7742 | 356/553 | 5.32 | 4.94 | 2277 |
| chrome | transformersjs-fp16 | webgpu fp16 | 0.613 | 0.563 | 0.160 | 0.557 | 6889 | 255/367 | 7.80 | 3.40 | 1515 |
| firefox-nightly | doppler-capsule-q4k | failed: rep0: Error: TargetSelector: Device does not satisfy capabil | – | – | – | – | – | –/– | – | – | 0 |
| firefox-nightly | doppler-f16 | failed: rep0: Error: Shader compilation failed for rmsnorm_default | – | – | – | – | – | –/– | – | – | 0 |
| firefox-nightly | open-jev-deberta | failed: rep0: Error: failed to call OrtRun(). ERROR_CODE: 1, ERROR_M | – | – | – | – | – | –/– | – | – | 0 |
| firefox-nightly | open-jev-kev-0.6b | webgpu q4f16 | 0.580 | 0.530 | 0.294 | 0.661 | 4068 | 699/703 | 2.60 | 1.68 | 346 |
| firefox-nightly | open-jev-kev-4b | webgpu q4f16 | 0.720 | 0.697 | 0.193 | 0.472 | 8945 | 1115/1529 | 7.48 | 5.65 | 2277 |
| firefox-nightly | transformersjs-fp16 | webgpu fp16 | 0.613 | 0.563 | 0.160 | 0.557 | 9404 | 501/660 | 8.20 | 3.95 | 1515 |
| node | doppler-f16 | node compat f16 | 0.587 | 0.528 | 0.180 | 0.563 | 2152 | 641/963 | 0.59 | 6.50 | 0 |
| node | doppler-q4k | node compat q4k | 0.573 | 0.523 | 0.218 | 0.621 | 2034 | 589/920 | 0.57 | 5.37 | 0 |
| node | open-jev-deberta | cpu q4 | 0.607 | 0.528 | 0.140 | 0.578 | 1712 | 494/613 | 1.20 | 0.00 | 0 |
| node | open-jev-kev-0.6b | cpu q4 | 0.587 | 0.536 | 0.289 | 0.662 | 1990 | 463/854 | 0.75 | 0.00 | 0 |
| node | open-jev-kev-4b | cpu q4 | 0.720 | 0.697 | 0.193 | 0.471 | 5617 | 1899/3184 | 3.53 | 0.00 | 0 |
| node | transformersjs-fp16 | failed: rep0: TypeError: Invalid argument: "present.0.key" is not a  | – | – | – | – | – | –/– | – | – | 0 |
| node | transformersjs-fp16-cpu | cpu fp16 | 0.587 | 0.528 | 0.180 | 0.563 | 6287 | 1756/3193 | 3.59 | 0.00 | 0 |

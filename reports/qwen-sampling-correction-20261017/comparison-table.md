Cold rows below summarize resident requests after initialization; actual cold first requests are in the opening table. D = Doppler, T = Transformers.js.

| GPU | Model | Mode / input | Order | Decode tok/s D / T | TTFT ms D / T | Complete ms D / T | Delivery p95 ms D / T | Tokens |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| metal | 0-8b | cold / 64 | doppler-first | 55.3 / 41.9 | 220.4 / 166.8 | 2589.5 / 3280.1 | 281.2 / 31.4 | match |
| metal | 0-8b | cold / 64 | transformersjs-first | 43.8 / 39.5 | 280.1 / 192.4 | 3127.9 / 3383.1 | 314.0 / 32.4 | match |
| metal | 2b | cold / 64 | doppler-first | 20.8 / 26.4 | 666.2 / 330.8 | 6787.0 / 5276.2 | 713.8 / 57.6 | match |
| metal | 2b | cold / 64 | transformersjs-first | 29.4 / 41.0 | 457.5 / 197.6 | 4774.5 / 3414.5 | 463.5 / 31.0 | match |
| metal | 0-8b | warm / 1024 | doppler-first | 48.5 / 46.1 | 3425.0 / 2249.5 | 6045.8 / 5003.4 | 302.6 / 23.3 | match |
| metal | 0-8b | warm / 1024 | transformersjs-first | 54.1 / 49.8 | 2881.5 / 1952.0 | 5248.6 / 4545.9 | 290.1 / 23.0 | match |
| metal | 2b | warm / 1024 | doppler-first | 23.9 / 27.8 | 8631.5 / 4484.6 | 13956.8 / 9046.1 | 592.6 / 54.2 | match |
| metal | 2b | warm / 1024 | transformersjs-first | 26.6 / 29.2 | 7614.1 / 4243.5 | 12342.4 / 8532.0 | 529.7 / 50.0 | match |
| metal | 0-8b | warm / 64 | doppler-first | 57.3 / 48.2 | 175.2 / 131.4 | 2393.5 / 2767.2 | 276.8 / 24.9 | match |
| metal | 0-8b | warm / 64 | transformersjs-first | 56.4 / 48.9 | 179.9 / 130.1 | 2433.7 / 2728.1 | 279.7 / 25.4 | match |
| metal | 2b | warm / 64 | doppler-first | 35.6 / 32.8 | 508.4 / 222.4 | 4081.5 / 4126.0 | 429.4 / 33.7 | match |
| metal | 2b | warm / 64 | transformersjs-first | 25.6 / 31.5 | 533.0 / 279.5 | 5493.7 / 4344.6 | 590.4 / 38.4 | match |
| vulkan | 0-8b | cold / 64 | doppler-first | 108.6 / 42.3 | 126.6 / 83.8 | 1289.7 / 3194.2 | 139.5 / 26.7 | match |
| vulkan | 0-8b | cold / 64 | transformersjs-first | 111.6 / 41.6 | 126.5 / 95.3 | 1264.7 / 3262.0 | 137.8 / 29.8 | match |
| vulkan | 2b | cold / 64 | doppler-first | 79.3 / 41.8 | 162.9 / 254.2 | 1765.2 / 3328.2 | 193.8 / 28.2 | match |
| vulkan | 2b | cold / 64 | transformersjs-first | 80.5 / 42.0 | 166.0 / 256.1 | 1735.2 / 3298.7 | 193.9 / 27.5 | match |
| vulkan | 0-8b | warm / 1024 | doppler-first | 102.9 / 40.9 | 1069.1 / 816.6 | 2298.8 / 3921.9 | 153.7 / 28.4 | match |
| vulkan | 0-8b | warm / 1024 | transformersjs-first | 101.0 / 41.3 | 1080.5 / 790.6 | 2341.5 / 3867.8 | 152.5 / 29.9 | match |
| vulkan | 2b | warm / 1024 | doppler-first | 76.7 / 41.2 | 1586.9 / 1988.2 | 3252.9 / 5266.0 | 205.2 / 28.9 | match |
| vulkan | 2b | warm / 1024 | transformersjs-first | 76.5 / 41.9 | 1586.0 / 1934.9 | 3244.0 / 4963.8 | 205.5 / 27.1 | match |
| vulkan | 0-8b | warm / 64 | doppler-first | 113.2 / 40.9 | 117.8 / 82.0 | 1233.3 / 3192.2 | 136.7 / 28.6 | match |
| vulkan | 0-8b | warm / 64 | transformersjs-first | 113.3 / 43.1 | 117.3 / 85.3 | 1236.2 / 3041.8 | 137.7 / 25.9 | match |
| vulkan | 2b | warm / 64 | doppler-first | 81.8 / 42.0 | 156.9 / 241.0 | 1710.0 / 3268.3 | 190.7 / 27.0 | match |
| vulkan | 2b | warm / 64 | transformersjs-first | 82.8 / 41.3 | 154.9 / 225.6 | 1685.8 / 3297.8 | 189.9 / 28.9 | match |

| GPU | Model | Order | Load ms D / T | Cold first request TTFT ms D / T | Opening to first output ms D / T | Opening to complete ms D / T |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| metal | 0-8b | doppler-first | 1060.4 / 2355.9 | 294.1 / 274.5 | 1354.5 / 2630.4 | 3649.9 / 5636.0 |
| metal | 0-8b | transformersjs-first | 1397.4 / 1798.9 | 321.7 / 345.2 | 1719.1 / 2144.1 | 4877.2 / 5366.6 |
| metal | 2b | doppler-first | 2514.6 / 4409.5 | 581.8 / 457.3 | 3096.4 / 4866.7 | 9131.3 / 9685.7 |
| metal | 2b | transformersjs-first | 2234.9 / 5407.9 | 457.5 / 411.2 | 2692.4 / 5819.0 | 7009.4 / 8822.4 |
| vulkan | 0-8b | doppler-first | 2016.5 / 2327.0 | 439.5 / 291.5 | 2456.0 / 2618.5 | 3799.0 / 5613.5 |
| vulkan | 0-8b | transformersjs-first | 1883.4 / 2350.2 | 419.5 / 326.3 | 2302.9 / 2676.5 | 3619.2 / 5684.1 |
| vulkan | 2b | doppler-first | 3492.8 / 3469.9 | 526.5 / 523.8 | 4019.3 / 3993.7 | 5703.2 / 7009.4 |
| vulkan | 2b | transformersjs-first | 3645.9 / 3393.6 | 538.1 / 522.7 | 4184.0 / 3916.3 | 5885.8 / 6878.6 |

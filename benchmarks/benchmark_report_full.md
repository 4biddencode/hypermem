# HyperMEM — Benchmark Report

Model(s): qwen2.5:7b, gemma3:12b  |  scales: [100, 1000, 5000, 10000, 25000, 50000]  |  seeds: 5  |  platform: win32

## recall_scaling

| metric | qwen2.5:7b |
|--------|--------|
| recall@100 | 0.567 ± 0.07 |
| recall@1000 | 0.65 ± 0.124 |
| recall@10000 | 0.617 ± 0.139 |
| recall@25000 | 0.567 ± 0.109 |
| recall@5000 | 0.6 ± 0.16 |
| recall@50000 | 0.667 ± 0.132 |
| recall_latency_ms@100 | 322.6 ± 17.051 |
| recall_latency_ms@1000 | 309.96 ± 18.205 |
| recall_latency_ms@10000 | 312.18 ± 14.57 |
| recall_latency_ms@25000 | 312.22 ± 7.791 |
| recall_latency_ms@5000 | 308.8 ± 9.808 |
| recall_latency_ms@50000 | 323.14 ± 24.003 |

## answer_scaling

| metric | qwen2.5:7b |
|--------|--------|
| answer_hypermem@100 | 0.555 ± 0.048 |
| answer_hypermem@1000 | 0.639 ± 0.048 |
| answer_hypermem@10000 | 0.472 ± 0.096 |
| answer_hypermem@25000 | 0.5 ± 0.144 |
| answer_hypermem@5000 | 0.556 ± 0.096 |
| answer_hypermem@50000 | 0.472 ± 0.048 |
| answer_hypermem_ctx_tokens@100 | 61.867 ± 8.406 |
| answer_hypermem_ctx_tokens@1000 | 62.233 ± 6.475 |
| answer_hypermem_ctx_tokens@10000 | 58.233 ± 3.308 |
| answer_hypermem_ctx_tokens@25000 | 59.167 ± 9.259 |
| answer_hypermem_ctx_tokens@5000 | 56.333 ± 3.009 |
| answer_hypermem_ctx_tokens@50000 | 56.667 ± 1.724 |
| answer_hypermem_ida@100 | 0.556 ± 0.127 |
| answer_hypermem_ida@1000 | 0.583 ± 0.167 |
| answer_hypermem_ida@10000 | 0.528 ± 0.174 |
| answer_hypermem_ida@25000 | 0.611 ± 0.048 |
| answer_hypermem_ida@5000 | 0.611 ± 0.048 |
| answer_hypermem_ida@50000 | 0.445 ± 0.048 |
| answer_hypermem_ida_ctx_tokens@100 | 89.333 ± 5.689 |
| answer_hypermem_ida_ctx_tokens@1000 | 95.733 ± 3.288 |
| answer_hypermem_ida_ctx_tokens@10000 | 85.567 ± 4.389 |
| answer_hypermem_ida_ctx_tokens@25000 | 92.9 ± 6.275 |
| answer_hypermem_ida_ctx_tokens@5000 | 115.4 ± 15.162 |
| answer_hypermem_ida_ctx_tokens@50000 | 94.433 ± 18.011 |
| answer_hypermem_ida_latency_ms@100 | 330.167 ± 13.759 |
| answer_hypermem_ida_latency_ms@1000 | 321.767 ± 25.262 |
| answer_hypermem_ida_latency_ms@10000 | 316.9 ± 14.545 |
| answer_hypermem_ida_latency_ms@25000 | 328.3 ± 12.413 |
| answer_hypermem_ida_latency_ms@5000 | 329.567 ± 6.519 |
| answer_hypermem_ida_latency_ms@50000 | 317.9 ± 12.758 |
| answer_hypermem_latency_ms@100 | 327.2 ± 2.8 |
| answer_hypermem_latency_ms@1000 | 323.133 ± 21.955 |
| answer_hypermem_latency_ms@10000 | 329.667 ± 15.684 |
| answer_hypermem_latency_ms@25000 | 319.267 ± 11.35 |
| answer_hypermem_latency_ms@5000 | 317.533 ± 10.149 |
| answer_hypermem_latency_ms@50000 | 319.533 ± 5.16 |
| answer_normal@100 | 0.0 ± 0.0 |
| answer_normal@1000 | 0.0 ± 0.0 |
| answer_normal@10000 | 0.0 ± 0.0 |
| answer_normal@25000 | 0.0 ± 0.0 |
| answer_normal@5000 | 0.0 ± 0.0 |
| answer_normal@50000 | 0.0 ± 0.0 |
| answer_normal_ctx_tokens@100 | 28 ± 0.0 |
| answer_normal_ctx_tokens@1000 | 28 ± 0.0 |
| answer_normal_ctx_tokens@10000 | 28 ± 0.0 |
| answer_normal_ctx_tokens@25000 | 28 ± 0.0 |
| answer_normal_ctx_tokens@5000 | 28 ± 0.0 |
| answer_normal_ctx_tokens@50000 | 28 ± 0.0 |
| answer_normal_latency_ms@100 | 297.833 ± 2.757 |
| answer_normal_latency_ms@1000 | 320.0 ± 7.499 |
| answer_normal_latency_ms@10000 | 322.167 ± 18.548 |
| answer_normal_latency_ms@25000 | 316.9 ± 13.707 |
| answer_normal_latency_ms@5000 | 318.667 ± 14.351 |
| answer_normal_latency_ms@50000 | 325.933 ± 2.47 |

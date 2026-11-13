# Official short-budget pilot: official_pilot_v1

Fixed final epoch; 20 epochs x 40 steps, trial 1 / seed 1 only. No reranking, flip ensemble, test-selected checkpoint or full-convergence claim.

Metrics below are percentages. SYSU means use 10 single-shot gallery samples; these are not 10 independent training seeds.

| Variant | RegDB V-to-I R1 / mAP | RegDB I-to-V R1 / mAP | RegDB mean R1 / mAP | SYSU all R1 / mAP | SYSU indoor R1 / mAP |
|---|---:|---:|---:|---:|---:|
| agw | 46.02 / 39.40 | 43.40 / 37.06 | 44.71 / 38.23 | 5.08 / 6.36 | 7.15 / 13.78 |
| center | 43.59 / 38.51 | 44.08 / 38.26 | 43.83 / 38.38 | 6.87 / 7.96 | 9.09 / 16.21 |
| unbounded | 41.70 / 37.62 | 44.61 / 38.70 | 43.16 / 38.16 | 6.48 / 7.80 | 8.74 / 15.76 |
| bcrt | 49.22 / 42.56 | 49.42 / 42.25 | 49.32 / 42.41 | 6.67 / 8.09 | 7.02 / 14.11 |

Matched finite training histories; zero skipped updates; disjoint train/test identities; current source hashes match all archived configurations.
Single-seed short pilots do not establish significance, novelty or a complete AGW reproduction.
Boundary diagnostics certify fixed batch Euclidean prototypes only, not retrieval generalization.

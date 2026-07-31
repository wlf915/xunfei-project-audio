| Round | Learning rate | Scheduler | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Epoch 6 | Epoch 7 | Epoch 8 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 epoch | 1e-6 | constant | 13.4189 | 11.7624 | 10.4742 | 9.6638 | 9.1367 | - | - | - |
| 5 epoch | 2e-6 | constant | 11.4952 | 8.7640 | 7.9158 | 7.5772 | 7.3460 | - | - | - |
| 5 epoch | 5e-6 | constant | 8.8137 | 7.0203 | 6.6870 | 6.4853 | 6.3477 | - | - | - |
| 8 epoch | 5e-7 | cosine | 14.4088 | 13.8552 | 13.4439 | 13.2178 | 13.0698 | 13.0226 | 13.0052 | 12.9967 |
| 8 epoch | 1e-6 | cosine | 14.0495 | 12.2163 | 11.0110 | 10.4390 | 10.1295 | 10.0324 | 9.9962 | 9.9881 |
| 8 epoch | 1.5e-6 | cosine | 13.6796 | 11.0872 | 9.5738 | 8.9260 | 8.6448 | 8.5593 | 8.5257 | 8.5235 |

> Each value is the arithmetic mean of 90 micro-batch losses in that epoch. It describes optimization only and is not an audio-quality, completeness, or speaker-similarity score.

# TOFU sensitivity and scorer-transfer ablations

All runs use TOFU forget10 and Llama-3.2-1B-Instruct. The selected operating
point is ρ = 0.2 and β = 7; the selected ρ closely matches the labelled
forget-token ratio of 0.22.

## ρ sensitivity (β = 7)

| ρ | Forget ES ↓ | Retain ES ↑ | MMLU ↑ | Rep. ↑ | Selected % |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.0532 | 0.7524 | 45.1 | 575 | 4.6 |
| 0.2 | 0.0353 | 0.7527 | 45.0 | 575 | 17.4 |
| 0.3 | 0.0342 | 0.7449 | 45.2 | 575 | 30.3 |
| 0.4 | 0.0352 | 0.7376 | 45.1 | 578 | 39.2 |
| 0.5 | 0.0373 | 0.7331 | 45.1 | 577 | 46.8 |
| 0.6 | 0.0384 | 0.7370 | 45.1 | 570 | 65.7 |
| 0.7 | 0.0404 | 0.7370 | 45.2 | 567 | 70.4 |
| 0.8 | 0.0438 | 0.7429 | 45.1 | 566 | 86.4 |
| 0.9 | 0.0474 | 0.7389 | 44.9 | 562 | 96.9 |
| 1.0 | 0.0547 | 0.7357 | 45.1 | 566 | 100.0 |

## β sensitivity (ρ = 0.2)

| β | Forget ES ↓ | Retain ES ↑ | MMLU ↑ | Rep. ↑ | Selected % |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0325 | 0.0348 | 41.4 | 0 | 100.0 |
| 1 | 0.0325 | 0.3856 | 45.2 | 574 | 17.3 |
| 2 | 0.0325 | 0.5862 | 45.3 | 570 | 13.5 |
| 3 | 0.0325 | 0.6450 | 45.4 | 572 | 15.6 |
| 4 | 0.0325 | 0.6980 | 45.0 | 570 | 16.8 |
| 5 | 0.0329 | 0.7168 | 45.0 | 576 | 16.1 |
| 6 | 0.0337 | 0.7439 | 45.1 | 576 | 17.0 |
| 7 | 0.0353 | 0.7527 | 45.0 | 575 | 17.4 |
| 8 | 0.0368 | 0.7526 | 45.1 | 575 | 18.7 |
| 9 | 0.0388 | 0.7475 | 45.0 | 577 | 16.8 |
| 10 | 0.0430 | 0.7438 | 45.0 | 574 | 15.3 |

Utility and forgetting are stable around the selected operating point. For β,
there is a broad plateau from approximately 6–10; disabling saturation at β = 0
destroys retention. For ρ, approximately 0.2–0.5 gives similar forgetting, while
both very small and all-token budgets degrade the result.

## Five-fold scorer transfer

| Run | Scheme | Seen AUROC ↑ | Unseen AUROC ↑ | Gap |
|---|---|---:|---:|---:|
| A1 | author | 0.7463 | 0.7601 | -0.0138 |
| A2 | author | 0.7276 | 0.7190 | +0.0086 |
| A3 | author | 0.7255 | 0.7410 | -0.0155 |
| A4 | author | 0.7391 | 0.7213 | +0.0179 |
| A5 | author | 0.7718 | 0.7469 | +0.0249 |
| F1 | fact | 0.7663 | 0.7685 | -0.0022 |
| F2 | fact | 0.7523 | 0.7203 | +0.0320 |
| F3 | fact | 0.7440 | 0.7411 | +0.0029 |
| F4 | fact | 0.7281 | 0.7174 | +0.0107 |
| F5 | fact | 0.7361 | 0.6938 | +0.0423 |
| **author mean** | **author** | **0.7421** | **0.7376** | **+0.0044** |
| **fact mean** | **fact** | **0.7454** | **0.7282** | **+0.0171** |

The small seen–unseen gaps show that the learned scorer transfers both to unseen
authors and to unseen facts about authors observed during training. The JSON
companion contains all sweep points, fold-level results, aggregate statistics,
and end-to-end fold metrics.

# Experiment comparison

This file is generated from the saved test metrics after each experiment. Missing runs are shown as pending, not estimated.

| Metric | Experiment A (class weights) | Experiment B (oversampling) |
|---|---:|---:|
| Test accuracy | 0.6026 | 0.6026 |
| Test macro-F1 | 0.5532 | 0.5310 |
| Test micro-F1 | 0.6026 | 0.6026 |
| Best val macro-F1 | 0.5822 | 0.5745 |
| Best epoch | 11 | 18 |

## Per-class test F1

| Cause | Experiment A | Experiment B |
|---|---:|---:|
| EQUIPMENT FAILURE | 0.7000 | 0.6967 |
| CORROSION FAILURE | 0.6766 | 0.6604 |
| MATERIAL FAILURE OF PIPE OR WELD | 0.4626 | 0.4677 |
| EXCAVATION DAMAGE | 0.7736 | 0.7679 |
| NATURAL FORCE DAMAGE | 0.5867 | 0.5679 |
| OTHER OUTSIDE FORCE DAMAGE | 0.5091 | 0.5000 |
| OTHER INCIDENT CAUSE | 0.2807 | 0.2128 |
| INCORRECT OPERATION | 0.4364 | 0.3750 |

## Summary

Experiment A (class weights) has the higher test macro-F1 (0.5532 vs. 0.5310). Compared with class weighting, oversampling had higher test F1 for: MATERIAL FAILURE OF PIPE OR WELD (+0.0052). It had lower test F1 for: OTHER INCIDENT CAUSE (-0.0679), INCORRECT OPERATION (-0.0614), NATURAL FORCE DAMAGE (-0.0188). These are measured test-set differences, not evidence that oversampling caused the change.

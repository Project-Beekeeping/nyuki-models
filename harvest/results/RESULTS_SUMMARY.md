# Smart Nyuki v2: executed synthetic results

Run date: 6 November 2026. All results below use generated scenarios; no measured Kenyan field performance is established.

Selected readiness estimator: **logistic**. Selected yield estimator: **xgboost**.

The fixed validation objective selects a feasible precision policy first, then the lowest readiness Brier score. Yield selection minimizes validation MAE. These are winners of this experiment, not universally best algorithms.

| Test | Inspected labels | Model-ready predictions | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| future_known | 269 | 26 | 96.2% | 20.8% |
| future_new_hives | 171 | 21 | 95.2% | 26.0% |
| future_new_apiaries | 225 | 21 | 90.5% | 18.6% |

Model-ready predictions above are thresholded classifier scores before the operational checks. Conservative thresholds miss many ready cases.

| Yield test | Weighed extraction labels | MAE (kg) | Observed interval coverage |
| --- | ---: | ---: | ---: |
| future_known | 11 | 2.25 | 81.8% |
| future_new_hives | 9 | 1.64 | 100.0% |
| future_new_apiaries | 16 | 2.06 | 81.2% |

Intervals have nominal 90% coverage under the calibration assumptions. Some observed test coverage is lower, and sample counts are small. Do not claim reliable 90% coverage for Kenyan apiaries.

Pooled readiness precision check: **True**.
Complete readiness release check, including hive-type sample counts and precision: **False**.
Hive-type checks: `{"Langstroth": false, "Kenya_Top_Bar": false}`.

The held-out Langstroth subgroup at new apiaries has weaker precision than the pooled result, and several top-bar subgroups have too few model-ready predictions. The complete check therefore blocks candidate-ready advisories. See policy_metrics.csv for actual advisory counts.

Verification: 29 behavioural tests passed; all 10 notebook code cells executed without errors; portable bootstrap checked; source and model-card checksums agree.

Original notebook preserved in backups/. Current notebooks and exported artifacts carry synthetic provenance. No trained acoustic fanning or absconding classifier has been fabricated.

Current source SHA256: `e1ac54bcc3f69a024d000b2eab9c4f4cf21968472f0a6a3f1f31671318af6acb`.

# Smart Nyuki harvest v2

The updated `smart_nyuki_harvest_synthetic.ipynb` is the presentation notebook. `harvest_readiness.py` contains the reusable implementation. The original notebook is preserved in `backups/smart_nyuki_harvest_synthetic_v1.ipynb`.

This implementation demonstrates model selection and an inspection advisory. It has no measured Kenyan field training data. Public harvest calendars are research context, not readiness labels. Read `KENYA_BEE_BEHAVIOUR_AND_MODEL_DESIGN.md` for the Kenya-specific evidence and the limitations of the simulator.

## Run locally

The project already has an isolated `.venv` with the tested dependencies.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python harvest_readiness.py --output outputs/harvest_v2
```

The second command regenerates all simulated data. In the notebook, cached synthetic CSVs are reused when present; training and evaluation run again. `requirements-harvest-lock.txt` records exact tested package versions. `requirements-harvest.txt` gives the supported installation ranges. No GPU is required.

Open the notebook in Jupyter or Colab and run all cells. It includes a copy of the Python module so it can bootstrap a standalone Colab session. An existing module with different source fails explicitly to prevent an old embedded copy from overwriting your edits. After changing the Python module, run `build_harvest_notebook.py` to regenerate the notebook. Regeneration clears notebook outputs; rerun it to refresh the results.

## What changed

- Hourly observations support a true 24-hour weight difference, daytime gain and the most recent completed 21:00–05:00 nighttime loss.
- Synthetic climate readings use a DHT11-like rounded ±2 C / ±5% RH error scenario; smoothing does not turn those readings into precise comb-moisture measurements.
- Sensor features use current and earlier observations only. Missing sensors are not reconstructed from future readings.
- Corrections distinguish equipment changes, total harvested mass removed, and feeding. Unknown events and absent event logs trigger inspection.
- Readiness labels exist only at independently inspected times. A configurable policy combines measured moisture and capping. Unsampled times remain unknown.
- Calibration and model/threshold selection use separate chronological periods. Tests cover future observations from known hives, unseen hives of both supported types, and entirely held-out apiaries.
- XGBoost, CatBoost and LightGBM compete with simpler baselines. The exported yield estimator is selected from the validation leaderboard.
- Only actual weighed extractions supply yield labels. A separate calibration set supplies finite-sample split-conformal intervals; coverage is checked in each test group.
- The recommendation threshold must meet an empirical development precision target. Failure on held-out groups blocks candidate-ready advisories. Synthetic/field origins cannot be mixed.
- Release checks cover pooled and separate hive-type results. A subgroup with too few recommendations fails the evidence requirement even when its observed precision is 100%.
- Health, reserves, missingness, disturbance, recent feeding, feature ranges and interval availability can independently request inspection.

## Decisions and time

Each daily decision occurs at **18:00 Africa/Nairobi before manipulation**. Features use only observations up to that timestamp. Previous health/reserve records must already be available and no older than the configured age. A contemporaneous inspection label is used for training/evaluation, not as an input feature.

This is an explicit demonstration schedule, not a requirement that users inspect at 18:00. Actual deployments need the same well-defined time alignment. Do not shift morning or evening inspection findings to a different time and assume honey state is unchanged.

The readiness target defaults to moisture ≤18% **and** capped fraction ≥0.8 for the inspected storage area. These are conservative project choices. The classifier does not output an exact moisture percentage. Colony reserve sufficiency is a separate verified inspection finding, not total hive weight minus an arbitrary constant.

The model returns `inspect_hive`, `wait_and_monitor`, or `candidate_ready_verify_frames`. Every output includes provenance, field-validation status and `harvest_authorized=false`. Candidate readiness is an advisory requiring representative frame checks, not an instruction to extract honey.

A current representative measurement that fails the moisture/capping policy overrides a high model score. Predictive policy evaluation supplies only earlier inspections to the advisory; it never uses the inspection providing the current test target to improve its apparent recommendation precision.

## Field data contract

Blank CSV templates and an example split declaration are in `data_templates/`. Blank rows are not training data. Use explicit `+03:00` or UTC offsets in timestamps. The code rejects naive times, duplicate hive timestamps and undeclared hive types/sensor locations.

Hourly telemetry:

| Field | Meaning |
| --- | --- |
| `hive_id`, `apiary_id` | Stable identifiers; a relocated hive needs a new explicitly defined episode |
| `timestamp` | Exact hourly observation time, with timezone |
| `weight_kg` | Observed total hive mass; leave a missing measurement blank |
| `int_temp`, `int_hum`, `ext_temp`, `ext_hum` | Degrees C and percent RH; internal sensor in the documented honey-storage zone |
| `rain_mm`, `wind_ms` | Hourly rain and measured/local estimated wind, with source recorded separately |
| `equipment_delta_kg` | Signed, known mass change from boxes/bars/frames or equipment, counted once |
| `harvest_mass_removed_kg` | Total biological/comb mass removed from the hive, positive; **not necessarily bottled liquid-honey weight** |
| `feed_kg` | Known mass added as feed; zero means known none, not unknown |
| `inspection_event` | Whether a disturbance took place during this hour |
| `tare_kg`, `frame_count` | Recorded hive/configuration metadata; frame count means bar count for top-bar hives |
| `shade_fraction` | Recorded shade estimate from 0 to 1, using a consistent protocol |
| `hive_type` | `Langstroth` or `Kenya_Top_Bar` |
| `sensor_location` | `honey_storage`; keep detailed position and sensor/calibration metadata separately |
| `data_origin` | `field` for actual observed data; never relabel simulated rows |

An event field may be zero only when the log is known complete. A missing complete hourly record is reindexed and flagged as an unknown event log. For partial or unreliable intervention history, request inspection rather than supply invented zeroes. If a top-bar comb is removed and processed, log both total mass leaving the hive and net liquid honey separately. The simulator does not fully represent wax rebuilding costs.

Inspection and extraction records:

| Field | Meaning |
| --- | --- |
| `hive_id`, `apiary_id`, `timestamp` | The hive and corresponding decision-time observation |
| `readiness_observed_at` | When that inspection record was actually available to the model |
| `moisture_pct`, `capping_fraction` | Refractometer measurement and capping estimate for the same representative harvest area |
| `representative_sample` | Whether the documented sampling procedure covers the intended harvest area |
| `queenright`, `disease_flag`, `swarm_flag` | Independently observed health checks; unknown values must not become false |
| `reserve_verified` | Beekeeper has verified adequate stores/brood protection under an agreed local policy |
| `recoverable_yield_kg` | Actual net extracted liquid honey from the defined area; blank if no extraction occurred |
| `yield_observed_at` | When weighed yield was available; never earlier than the prediction |
| `data_origin` | `field` |

Keep county, GPS/altitude, forage species/flowering, water availability, colony occupancy, brood estimate, colony lineage when known, comb/frame selection, extraction method, instrument calibration and inspector identity in accompanying field records. These variables need consistent measurement before they become model inputs. The current model does not infer them from a county name.

Field training uses a deliberately declared split plan rather than the synthetic apiary IDs:

```bash
.venv/bin/python harvest_readiness.py \
  --telemetry your_hourly_telemetry.csv \
  --inspections your_inspections.csv \
  --split-config your_split_plan.json \
  --output outputs/field_experiment
```

Declare every apiary as development or held-out, include both hive types in held-out hives where possible, and choose chronological boundaries that provide enough independently observed labels. Label availability must precede each training/calibration/validation cutoff. Insufficient data raises an explicit error; the pipeline does not fabricate labels or relax requirements automatically.

## Results and presentation

`outputs/harvest_v2/` contains the model, model card, validation leaderboards, final test metrics, policy metrics, ablations, predictions, figures and demonstration CSVs. Read both `test_metrics.csv` and `policy_metrics.csv`: the first evaluates the selected estimator, and the second applies the operational checks. A model can rank first among candidates and still fail the release criterion.

Report the test sample counts with every score. There are only two entirely held-out synthetic apiaries; the daily rows are correlated. Nominal 90% yield intervals do not guarantee 90% coverage in another climate or apiary. The source model, synthetic generator and all metrics remain unvalidated for real Kenyan harvest decisions.

The previous deck's 1.7 kg / 8 days / 79% numbers have not been adopted. The revised task predicts present inspection-defined readiness and event yield. It does not advertise precise days-to-harvest or a trained 14-day dearth classifier. Recent observed weight trends remain visible as context. Reintroduce forecasting tasks only with defined outcomes, censoring and purged validation.

Suggested presentation wording:

> We implemented and executed a harvest-advisory research pipeline for Kenyan Langstroth and top-bar scenarios. It compares calibrated models using separate development and future held-out hive/apiary data, and blocks recommendations when the evidence is insufficient. All current performance results are from synthetic scenarios. Regional harvest calendars inform the research, while local moisture, inspection and weighed-yield records will establish field performance.

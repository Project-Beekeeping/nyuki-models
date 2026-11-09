# Research artifacts

Harvest v2 source, dependency locks, fitted synthetic artifact, model card and example advisory are under `harvest/`. The four supplied notebooks are under `notebooks/`.

The harvest model is synthetic and its full release gate failed. No field probability, yield forecast or harvest authorization is enabled.

The other three notebooks remain experiments. Microphone waveforms and real labelled trap photos are required before they can serve field inputs. A scalar sound-level sensor is not an audio waveform.

To reproduce harvest inference, use the locked research environment and import `harvest_readiness.py` from this directory so the trusted joblib resolves its original module name. These dependencies belong in a research/worker environment, not the web image. The original capstone notebook build and executed outputs remain preserved in the parent workspace.

The pure, dependency-free canonical export adapter is `harvest/export_adapter.py`. It preserves missing inputs and provenance and always disables field predictions. Its schema, model inventory, label timing limits, field collection and shadow/promotion handoff are documented in [`docs/MODEL_HANDOFF.md`](../docs/MODEL_HANDOFF.md). Run its isolated contract tests with `python3 -m unittest discover -s research/harvest/tests -p test_export_adapter.py -v`. This adapter is not yet connected to a backend exporter or inference worker.

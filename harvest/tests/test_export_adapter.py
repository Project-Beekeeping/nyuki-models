"""Contract tests: no trained artifact import or third-party dependencies."""
import copy
from datetime import datetime, timedelta
import importlib.util
from pathlib import Path
import sys
import unittest

MODULE = Path(__file__).resolve().parents[1] / "export_adapter.py"
spec = importlib.util.spec_from_file_location("export_adapter", MODULE)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def fixture(days=0):
    end = datetime.fromisoformat("2026-11-08T18:00:00+03:00")
    start = end - timedelta(days=days)
    return {
        "schema_version": "smartnyuki.canonical-research.v1", "export_id": "test-export",
        "data_origin": "field", "start_at": start.isoformat(), "end_at": end.isoformat(),
        "hive": {"id": "h1", "apiary_id": "a1", "county": "Kiambu", "hive_type": "Langstroth",
                 "sensor_location": "honey_storage", "tare_kg": 20, "frame_count": 10,
                 "shade_fraction": 0.5, "metadata_observed_at": start.isoformat(),
                 "metadata_recorded_at": start.isoformat()},
        "observations": [], "interventions": [], "visits": [], "samples": [], "harvests": [],
        "event_coverage": [{"id": "coverage1", "hive_id": "h1", "data_origin": "field",
                            "start_at": (start-timedelta(hours=1)).isoformat(), "end_at": end.isoformat(),
                            "recorded_at": end.isoformat(), "complete": True}],
    }


def point(identity="p1", **changes):
    result = {"id": identity, "hive_id": "h1", "data_origin": "field", "kind": "weight",
              "value": 42, "unit": "kg", "quality": "valid", "source_id": "scale1",
              "observed_at": "2026-11-08T17:50:00+03:00", "recorded_at": "2026-11-08T17:51:00+03:00"}
    result.update(changes)
    return result


class AdapterTests(unittest.TestCase):
    def run_adapter(self, data, as_of="2026-11-08T18:00:00+03:00"):
        return adapter.adapt_export(data, as_of=as_of)

    def test_eat_alignment_latest_value_and_source_provenance(self):
        data = fixture()
        data["observations"] = [point("old", value=40, observed_at="2026-11-08T14:10:00Z"),
                                point(observed_at="2026-11-08T14:50:00Z")]
        original = copy.deepcopy(data)
        result = self.run_adapter(data)
        self.assertEqual(result["hourly_rows"][0]["timestamp"], "2026-11-08T18:00:00+03:00")
        self.assertEqual(result["hourly_rows"][0]["weight_kg"], 42)
        self.assertEqual(result["provenance"][0]["observations"]["weight_kg"]["id"], "p1")
        self.assertEqual(data, original)

    def test_future_event_and_late_available_record_cannot_leak(self):
        data = fixture()
        data["observations"] = [point(), point("late", value=99, observed_at="2026-11-08T17:55:00+03:00",
                                              recorded_at="2026-11-08T19:00:00+03:00"),
                                point("future", value=100, observed_at="2026-11-08T18:01:00+03:00",
                                      recorded_at="2026-11-08T18:02:00+03:00")]
        self.assertEqual(self.run_adapter(data)["hourly_rows"][0]["weight_kg"], 42)
        self.assertEqual(self.run_adapter(data, "2026-11-08T19:00:00+03:00")["hourly_rows"][0]["weight_kg"], 99)

    def test_gaps_and_event_unknowns_are_never_zero_filled(self):
        data = fixture(1)
        data["event_coverage"] = []
        data["observations"] = [point()]
        result = self.run_adapter(data)
        self.assertEqual(len(result["hourly_rows"]), 25)
        self.assertIsNone(result["hourly_rows"][0]["weight_kg"])
        self.assertIsNone(result["hourly_rows"][-1]["feed_kg"])
        self.assertIn("event_log_coverage_unknown", result["input_issues"])
        self.assertFalse(result["research_input_eligible"])

    def test_wrong_units_invalid_quality_and_air_humidity_do_not_become_honey(self):
        data = fixture()
        data["observations"] = [point(unit="g"), point("humidity", kind="internal_air_humidity", value=60, unit="%RH"),
                                point("temperature", kind="internal_temperature", value=34, unit="degC", quality="uncalibrated")]
        row = self.run_adapter(data)["hourly_rows"][0]
        self.assertIsNone(row["weight_kg"])
        self.assertIsNone(row["int_temp"])
        self.assertEqual(row["int_hum"], 60)
        self.assertNotIn("moisture_pct", row)

    def test_mass_removed_is_independent_of_net_honey_and_signed_equipment(self):
        data = fixture()
        base = {"hive_id": "h1", "data_origin": "field", "event_at": "2026-11-08T17:00:00+03:00",
                "recorded_at": "2026-11-08T17:05:00+03:00", "unit": "kg"}
        # Event exactly at 17:00 belongs to 17:00, not the later 18:00 bin.
        base["event_at"] = "2026-11-08T17:30:00+03:00"
        base["recorded_at"] = "2026-11-08T17:35:00+03:00"
        data["interventions"] = [dict(base, id="harvest", kind="harvest_removal", mass_kg=5.4, net_honey_kg=3.4),
                                 dict(base, id="equipment", kind="equipment_change", mass_kg=-2),
                                 dict(base, id="feed", kind="feeding", mass_kg=1.2)]
        row = self.run_adapter(data)["hourly_rows"][0]
        self.assertEqual(row["harvest_mass_removed_kg"], 5.4)
        self.assertEqual(row["equipment_delta_kg"], -2)
        self.assertEqual(row["feed_kg"], 1.2)
        data["interventions"][0].pop("mass_kg")
        result = self.run_adapter(data)
        self.assertIsNone(result["hourly_rows"][0]["harvest_mass_removed_kg"])
        self.assertIn("unknown_intervention_mass:harvest", result["input_issues"])

    def test_replays_deduplicate_and_conflicts_are_rejected(self):
        data = fixture()
        data["observations"] = [point(), point()]
        self.assertEqual(self.run_adapter(data)["hourly_rows"][0]["weight_kg"], 42)
        data["observations"][1]["value"] = 50
        with self.assertRaisesRegex(ValueError, "Conflicting replay"):
            self.run_adapter(data)
        data["observations"][1]["id"] = "different-id"
        with self.assertRaisesRegex(ValueError, "Conflicting channel"):
            self.run_adapter(data)

    def test_naive_time_mixed_hive_and_origin_are_rejected(self):
        for changes, message in [({"observed_at": "2026-11-08T17:00:00"}, "Naive"),
                                 ({"hive_id": "h2"}, "Mixed hive"), ({"data_origin": "synthetic"}, "Mixed hive")]:
            with self.subTest(changes=changes):
                data = fixture()
                data["observations"] = [point(**changes)]
                with self.assertRaisesRegex(ValueError, message):
                    self.run_adapter(data)

    def test_rain_requires_whole_hour_accumulation(self):
        data = fixture()
        data["observations"] = [point(kind="rain_accumulation", value=2, unit="mm",
                                      interval_start_at="2026-11-08T17:00:00+03:00")]
        self.assertIsNone(self.run_adapter(data)["hourly_rows"][0]["rain_mm"])
        data["observations"][0].update(observed_at="2026-11-08T18:00:00+03:00", recorded_at="2026-11-08T18:00:00+03:00")
        self.assertEqual(self.run_adapter(data)["hourly_rows"][0]["rain_mm"], 2)

    def test_outcomes_preserve_unknowns_and_late_labels_without_inventing_targets(self):
        data = fixture()
        data["samples"] = [dict(id="sample", hive_id="h1", data_origin="field",
                               event_at="2026-11-08T17:00:00+03:00", recorded_at="2026-11-08T17:10:00+03:00",
                               moisture_pct=18, capping_fraction=None, representative_sample=None)]
        result = self.run_adapter(data)
        self.assertIsNone(result["evidence"]["samples"][0]["capping_fraction"])
        self.assertNotIn("ready", result["evidence"]["samples"][0])
        data["samples"][0]["recorded_at"] = "2026-11-09T17:10:00+03:00"
        self.assertEqual(self.run_adapter(data)["evidence"]["samples"], [])

    def test_complete_research_input_still_never_enables_serving(self):
        data = fixture(31)
        start = datetime.fromisoformat(data["start_at"])
        for i in range(31*24+1):
            when = start + timedelta(hours=i)
            for kind, (_, unit, _, _) in adapter.CHANNELS.items():
                value = 45 if "humidity" in kind else 1 if kind in {"rain_accumulation", "wind_speed"} else 25
                data["observations"].append(point(f"{i}-{kind}", kind=kind, value=value, unit=unit,
                    observed_at=when.isoformat(), recorded_at=when.isoformat(),
                    interval_start_at=(when-timedelta(hours=1)).isoformat()))
        result = self.run_adapter(data)
        self.assertTrue(result["research_input_eligible"], result["input_issues"])
        self.assertFalse(result["field_prediction_enabled"])
        self.assertFalse(result["harvest_authorized"])
        self.assertEqual(len(result["source_sha256"]), 64)
        self.assertNotIn("joblib", sys.modules)

    def test_late_intervention_uses_event_hour_and_preserves_correction_identity(self):
        data = fixture(1)
        record = dict(id="late-harvest", hive_id="h1", data_origin="field", kind="harvest_removal",
                      event_at="2026-11-08T15:30:00+03:00", recorded_at="2026-11-08T19:00:00+03:00",
                      mass_kg=5, unit="kg", revision=2, supersedes_id="original")
        original = dict(record, id="original", mass_kg=3, revision=1, recorded_at="2026-11-08T16:00:00+03:00")
        original.pop("supersedes_id")
        data["interventions"] = [original, record]
        before = self.run_adapter(data)
        after = self.run_adapter(data, "2026-11-08T19:00:00+03:00")
        row = next(r for r in after["hourly_rows"] if r["timestamp"] == "2026-11-08T16:00:00+03:00")
        self.assertEqual(row["harvest_mass_removed_kg"], 5)
        self.assertEqual(sum(r["harvest_mass_removed_kg"] for r in before["hourly_rows"]), 3)
        self.assertEqual(sum(r["harvest_mass_removed_kg"] for r in after["hourly_rows"]), 5)
        self.assertEqual(len(after["intervention_revisions"]), 2)
        event = next(r for r in after["provenance"] if r["event_ids"])
        self.assertEqual(event["event_ids"], ["late-harvest"])
        self.assertNotEqual(before["source_sha256"], self.run_adapter(fixture(1))["source_sha256"])

    def test_draft_does_not_become_completed_inspection(self):
        data = fixture()
        base = dict(hive_id="h1", data_origin="field", kind="inspection",
                    event_at="2026-11-08T17:30:00+03:00", recorded_at="2026-11-08T17:35:00+03:00")
        data["interventions"] = [dict(base, id="draft", status="draft")]
        self.assertEqual(self.run_adapter(data)["hourly_rows"][0]["inspection_event"], 0)
        data["interventions"] = [dict(base, id="done1", status="completed"), dict(base, id="done2", status="reviewed")]
        self.assertEqual(self.run_adapter(data)["hourly_rows"][0]["inspection_event"], 1)
        data["interventions"] = [dict(base, id="unknown", status=None)]
        result = self.run_adapter(data)
        self.assertIsNone(result["hourly_rows"][0]["inspection_event"])
        self.assertIn("unknown_inspection_status:unknown", result["input_issues"])

    def test_unsupported_domain_and_metadata_remain_fail_closed(self):
        data = fixture()
        data["hive"].update(hive_type="Log", sensor_location="brood", frame_count=10.5)
        result = self.run_adapter(data)
        self.assertIn("unsupported_hive_type", result["input_issues"])
        self.assertIn("unsupported_sensor_location", result["input_issues"])
        self.assertIsNone(result["hourly_rows"][0]["frame_count"])


if __name__ == "__main__":
    unittest.main()

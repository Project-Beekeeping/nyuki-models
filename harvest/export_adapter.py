"""Pure, fail-closed canonical-export -> harvest-v2 hourly research adapter.

No database, model loader, pandas, training, inference or network side effects.
See docs/MODEL_HANDOFF.md for the explicit version-1 input contract.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import json
import math
from zoneinfo import ZoneInfo

VERSION = "1.0.0"
TZ = ZoneInfo("Africa/Nairobi")
CHANNELS = {
    "weight": ("weight_kg", "kg", 0, None),
    "internal_temperature": ("int_temp", "degC", -10, 65),
    "internal_air_humidity": ("int_hum", "%RH", 0, 100),
    "external_temperature": ("ext_temp", "degC", -10, 65),
    "external_air_humidity": ("ext_hum", "%RH", 0, 100),
    "rain_accumulation": ("rain_mm", "mm", 0, None),
    "wind_speed": ("wind_ms", "m/s", 0, None),
}
EVENT_COLUMNS = {
    "feeding": "feed_kg",
    "equipment_change": "equipment_delta_kg",
    "harvest_removal": "harvest_mass_removed_kg",
    "inspection": "inspection_event",
}


def _time(value):
    if not isinstance(value, str):
        raise ValueError("Explicit ISO timestamp required")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Naive timestamps are not accepted")
    return result.astimezone(TZ)


def _hour(value):
    floor = value.replace(minute=0, second=0, microsecond=0)
    return floor if value == floor else floor + timedelta(hours=1)


def _number(value, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or (low is not None and value < low):
        return None
    if high is not None and value > high:
        return None
    return float(value)


def _unique(records):
    seen = {}
    for record in records:
        identity = record.get("id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("Every exported record needs a stable string id")
        if identity in seen and seen[identity] != record:
            raise ValueError(f"Conflicting replay for record {identity}")
        seen[identity] = record
    return list(seen.values())


def adapt_export(payload, *, as_of):
    """Return exact hourly columns plus provenance, evidence and eligibility.

    `as_of` limits both event time and when a record became available. Unknown
    event logs are None, never fabricated zeroes. Result always disables serving.
    Invalid values remain missing and documented; malformed identity/time fails.
    """
    if payload.get("schema_version") != "smartnyuki.canonical-research.v1":
        raise ValueError("Unsupported canonical export schema")
    if not isinstance(payload.get("export_id"), str) or not payload["export_id"]:
        raise ValueError("A stable export_id is required")
    cutoff = _time(as_of)
    start, end = _time(payload["start_at"]), _time(payload["end_at"])
    if start != _hour(start) or end != _hour(end) or end < start or end > cutoff:
        raise ValueError("Require ordered complete hourly bounds no later than as_of")
    if end - start > timedelta(days=366):
        raise ValueError("Export at most 366 days per episode")
    origin = payload.get("data_origin")
    if origin not in {"field", "synthetic"}:
        raise ValueError("Explicit field or synthetic origin required")
    hive = payload["hive"]
    for key in ("id", "apiary_id"):
        if not isinstance(hive.get(key), str) or not hive[key]:
            raise ValueError(f"Hive {key} must be a stable nonempty string")
    issues = set()
    if hive.get("hive_type") not in {"Langstroth", "Kenya_Top_Bar"}:
        issues.add("unsupported_hive_type")
    if hive.get("sensor_location") != "honey_storage":
        issues.add("unsupported_sensor_location")
    if _time(hive["metadata_recorded_at"]) < _time(hive["metadata_observed_at"]):
        raise ValueError("Metadata availability cannot precede observation")
    metadata_known = (
        _time(hive["metadata_observed_at"]) <= start
        and _time(hive["metadata_recorded_at"]) <= cutoff
    )
    if not metadata_known:
        issues.add("metadata_unavailable_at_window_start")
    metadata = {
        "tare_kg": _number(hive.get("tare_kg"), 0),
        "frame_count": _number(hive.get("frame_count"), 1),
        "shade_fraction": _number(hive.get("shade_fraction"), 0, 1),
    }
    if metadata["frame_count"] is not None and not metadata["frame_count"].is_integer():
        metadata["frame_count"] = None
    if not metadata_known:
        metadata = {key: None for key in metadata}
    for key, value in metadata.items():
        if value is None:
            issues.add(f"missing_metadata:{key}")
    if end - start < timedelta(days=31):
        issues.add("insufficient_31_day_history")
    if end.hour != 18:
        issues.add("not_18_eat_decision")

    observations = defaultdict(dict)
    time_values = {}
    for record in _unique(payload.get("observations", [])):
        if record.get("data_origin") != origin or record.get("hive_id") != hive["id"]:
            raise ValueError("Mixed hive or data origin in export")
        observed, recorded = _time(record["observed_at"]), _time(record["recorded_at"])
        if recorded < observed:
            raise ValueError("Record availability cannot precede observation")
        if observed > end or recorded > cutoff or _hour(observed) < start:
            continue
        kind = record.get("kind")
        if kind not in CHANNELS:
            issues.add(f"unsupported_channel:{kind}")
            continue
        column, unit, low, high = CHANNELS[kind]
        value = _number(record.get("value"), low, high)
        if record.get("unit") != unit or record.get("quality") != "valid":
            value = None
        # Rain increments need an explicit whole-hour accumulation interval.
        if not record.get("source_id"):
            issues.add(f"missing_source_id:{record['id']}")
        if kind == "rain_accumulation":
            if not record.get("interval_start_at") or observed != _hour(observed) or _time(record["interval_start_at"]) != observed - timedelta(hours=1):
                value = None
        if value is None:
            issues.add(f"invalid_observation:{record['id']}")
        signature = (kind, observed)
        if signature in time_values and time_values[signature] != value:
            raise ValueError("Conflicting channel values at the same observation time")
        time_values[signature] = value
        hour = _hour(observed)
        previous = observations[hour].get(column)
        # Latest observation in (hour-1h, hour], no forward fill or averaging.
        if previous is None or observed > previous[0]:
            observations[hour][column] = (observed, value, record)

    events = defaultdict(list)
    available_events = []
    for record in _unique(payload.get("interventions", [])):
        if record.get("data_origin") != origin or record.get("hive_id") != hive["id"]:
            raise ValueError("Mixed hive or data origin in intervention export")
        observed, recorded = _time(record["event_at"]), _time(record["recorded_at"])
        if recorded < observed:
            raise ValueError("Intervention availability cannot precede event time")
        if observed <= end and recorded <= cutoff and _hour(observed) >= start:
            available_events.append(record)
    superseded = {r["supersedes_id"] for r in available_events if r.get("supersedes_id")}
    replacements = [r["supersedes_id"] for r in available_events if r.get("supersedes_id")]
    if len(replacements) != len(set(replacements)):
        raise ValueError("Ambiguous concurrent intervention corrections")
    links = {r["id"]: r.get("supersedes_id") for r in available_events}
    for identity in links:
        visited = set()
        current = identity
        while current in links and current is not None:
            if current in visited:
                raise ValueError("Cyclic intervention correction trail")
            visited.add(current)
            current = links[current]
    for record in available_events:
        if record["id"] not in superseded:
            events[_hour(_time(record["event_at"]))].append(record)
    coverages = []
    for record in _unique(payload.get("event_coverage", [])):
        if record.get("hive_id") != hive["id"] or record.get("data_origin") != origin:
            raise ValueError("Mixed hive or origin in event coverage")
        a, b = _time(record["start_at"]), _time(record["end_at"])
        if a >= b or _time(record["recorded_at"]) < b:
            raise ValueError("Invalid event coverage interval or premature confirmation")
        if record.get("complete") is True and _time(record["recorded_at"]) <= cutoff:
            coverages.append((a, b, record["id"]))

    rows, provenance = [], []
    cursor = start
    while cursor <= end:
        covered_by = [identity for a, b, identity in coverages if a <= cursor - timedelta(hours=1) and b >= cursor]
        row = dict(hive_id=hive["id"], apiary_id=hive["apiary_id"], timestamp=cursor.isoformat(),
                   county=hive.get("county"), hive_type=hive.get("hive_type"),
                   sensor_location=hive.get("sensor_location"), data_origin=origin, **metadata)
        sources = {}
        for column, *_ in CHANNELS.values():
            point = observations[cursor].get(column)
            row[column] = point[1] if point else None
            if point:
                sources[column] = {key: point[2].get(key) for key in
                                   ("id", "observed_at", "recorded_at", "source_id", "quality", "unit")}
            if row[column] is None:
                issues.add(f"missing_hourly_input:{column}")
        for column in EVENT_COLUMNS.values():
            row[column] = 0 if covered_by else None
        for record in events[cursor]:
            kind = record.get("kind")
            column = EVENT_COLUMNS.get(kind)
            if column is None:
                issues.add(f"unsupported_intervention:{record['id']}")
                continue
            if kind == "inspection":
                status = record.get("status")
                value = 1 if status in {"completed", "reviewed"} else 0 if status == "draft" else None
            else:
                # Net extracted honey is never substituted for total mass removed.
                value = _number(record.get("mass_kg"), None if kind == "equipment_change" else 0)
                if record.get("unit") != "kg":
                    value = None
            if value is None:
                row[column] = None
                issue = "unknown_inspection_status" if kind == "inspection" else "unknown_intervention_mass"
                issues.add(f"{issue}:{record['id']}")
            elif row[column] is not None:
                row[column] = max(row[column], value) if kind == "inspection" else row[column] + value
        if not covered_by:
            issues.add("event_log_coverage_unknown")
        rows.append(row)
        provenance.append(dict(timestamp=cursor.isoformat(), observations=sources,
                               event_ids=[r["id"] for r in events[cursor]], coverage_ids=covered_by))
        cursor += timedelta(hours=1)

    evidence = {}
    for collection in ("visits", "samples", "harvests"):
        evidence[collection] = []
        for record in _unique(payload.get(collection, [])):
            if record.get("data_origin") != origin or record.get("hive_id") != hive["id"]:
                raise ValueError("Mixed hive or origin in outcome evidence")
            observed, recorded = _time(record["event_at"]), _time(record["recorded_at"])
            if recorded < observed:
                raise ValueError("Outcome availability cannot precede event time")
            if observed <= end and recorded <= cutoff:
                evidence[collection].append(dict(record))
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    return {
        "adapter_version": VERSION, "export_id": payload.get("export_id"), "source_sha256": digest,
        "as_of": cutoff.isoformat(), "timezone": "Africa/Nairobi", "hourly_rows": rows,
        "provenance": provenance, "evidence": evidence,
        "intervention_revisions": [dict(r) for r in available_events],
        "research_input_eligible": not issues, "input_issues": sorted(issues),
        "field_prediction_enabled": False, "harvest_authorized": False,
        "serving_blockers": ["field_prediction_disabled", "field_validation_and_promotion_required"],
        "fallback": "Record an inspection and representative honey samples; review observed measurements.",
    }

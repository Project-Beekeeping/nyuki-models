"""Smart Nyuki v2: causal, inspection-labelled harvest advisory experiments.

Synthetic results are pipeline evidence only. This module does not certify honey
moisture or authorize harvesting. See HARVEST_V2_GUIDE.md for data contracts.
"""
from __future__ import annotations

import utils.transport
import utils.session
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import math
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             mean_absolute_error, precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier, XGBRegressor

VERSION = "2.0.0"
TZ = "Africa/Nairobi"
SEED = 42
DECISION_HOUR = 18


@dataclass(frozen=True)
class Policy:
    moisture_limit_pct: float = 18.0
    capping_min: float = 0.80
    precision_target: float = 0.90
    min_validation_recommendations: int = 10
    min_weight_coverage: float = 0.80
    max_weight_age_h: float = 2.0
    max_inspection_age_days: int = 14
    min_harvest_kg: float = 3.0
    jump_threshold_kg: float = 2.0
    feeding_review_days: int = 30
    interval_alpha: float = 0.10
    min_interval_calibration: int = 10

    def __post_init__(self):
        if not 0<self.moisture_limit_pct<100 or not 0<=self.capping_min<=1:
            raise ValueError("Invalid moisture/capping policy")
        if not 0<self.precision_target<=1 or not 0<self.interval_alpha<1:
            raise ValueError("Invalid precision/interval probability")
        if self.min_validation_recommendations<1 or self.min_interval_calibration<1:
            raise ValueError("Calibration/recommendation counts must be positive")


@dataclass(frozen=True)
class SplitPlan:
    train_end: str = "2025-01-01"
    calibration_end: str = "2025-04-01"
    validation_end: str = "2025-07-01"
    # Must be declared before training; names do not encode the target.
    development_apiaries: tuple = ("apiary_0", "apiary_1", "apiary_2", "apiary_3")
    heldout_apiaries: tuple = ("apiary_4", "apiary_5")
    heldout_hives: tuple = tuple(f"apiary_{a}_h{h:02d}" for a in range(4) for h in (3,4))


def local_times(values):
    """Require explicit time zones; naive field timestamps must be fixed by owner."""
    s = pd.Series(values)
    if len(s) == 0 or not s.notna().any():
        return pd.to_datetime(s, utc=True).dt.tz_convert(TZ)
    parsed = pd.to_datetime(s)
    if parsed.dt.tz is None:
        raise ValueError("Timestamps must include a timezone, e.g. +03:00")
    return parsed.dt.tz_convert(TZ)


def vpd_kpa(temp_c, rh_pct):
    """Air VPD, not comb moisture. Invalid sensor values remain missing."""
    t, rh = np.asarray(temp_c, float), np.asarray(rh_pct, float)
    valid = (t >= -10) & (t <= 65) & (rh >= 0) & (rh <= 100)
    return np.where(valid, 0.6108 * np.exp(17.27 * t / (t + 237.3)) * (1 - rh / 100), np.nan)


# %% Synthetic hourly telemetry and independent inspection records
def simulate(seed=SEED, days=731, hives_per_apiary=5):
    """Illustrative sugar/water mass balance; NOT a fitted Kenyan bee model.

    Weather/forage profiles vary by apiary and year, flowering responds with
    variable lags, and drought can interrupt flow. No universal Kenyan harvest
    month or universal flight cutoff is imposed. Inspections precede extraction.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days, tz=TZ)
    telemetry, inspections = [], []
    # Illustrative highland/dryland scenarios, not county climate estimates.
    profiles = [(18, 1.1), (25, .65), (19, .9), (29, .7), (19, 1.2), (28, .55)]
    counties = ["Kiambu", "Kajiado", "Nakuru", "Baringo", "Kiambu", "Baringo"]
    for a, (base_t, rain_scale) in enumerate(profiles):
        apiary = f"apiary_{a}"
        doy = dates.strftime("%j").astype(int).to_numpy()
        phase = rng.normal(0, 20)
        peaks = (np.exp(-.5 * ((doy - (110 + phase)) / 32)**2)
                 + .7 * np.exp(-.5 * ((doy - (310 + phase)) / 28)**2))
        wet_year = {y: rng.uniform(.5, 1.4) for y in dates.year.unique()}
        rain = rng.gamma(.65, 7 * peaks * rain_scale + .25, days)
        rain *= np.array([wet_year[y] for y in dates.year])
        lag = int(rng.integers(12, 36))
        past_rain = pd.Series(rain).rolling(21, min_periods=1).mean().shift(lag).fillna(0)
        flowering = np.clip(past_rain.to_numpy() / 4, 0, 1.3)
        # A local flowering pulse and dry interruption deliberately break a
        # fixed national two-season calendar. Parameters are scenario choices.
        flowering += .3 * np.exp(-.5 * ((doy - (40 + 13*a)) / 25)**2)
        drought_start = 170 + 11 * a
        drought = (np.arange(days) % 365 > drought_start) & (np.arange(days) % 365 < drought_start + 35)
        flowering[drought] *= .1
        for h in range(hives_per_apiary):
            hive = f"{apiary}_h{h:02d}"
            hive_type = "Langstroth" if h % 2 == 0 else "Kenya_Top_Bar"
            frame_count = 10 if hive_type == "Langstroth" else 24
            tare, reserve = rng.uniform(17, 27), rng.uniform(4, 8)
            strength, ventilation = rng.uniform(.6, 1.2), rng.uniform(.65, 1.4)
            sugar, water = rng.uniform(4, 7), rng.uniform(.8, 1.5)
            capped, equipment = .5, 0.0
            last_harvest = -100
            inspection_offset = int(rng.integers(0, 7))
            queen_loss_start = int(rng.integers(100, max(101, days-50)))
            shade = rng.uniform(.1, .9)
            for d, day in enumerate(dates):
                queenright = not (queen_loss_start <= d < queen_loss_start+28)
                current_strength = strength * (1 if queenright else .65)
                mean_t = base_t + 2*np.cos(2*np.pi*(doy[d]-50)/365) + rng.normal(0, 1)
                ext_hum_base = np.clip(45 + 25*min(rain[d]/10, 1) + rng.normal(0, 8), 20, 90)
                nectar_today = max(0, 1.9 * flowering[d] * current_strength * rng.uniform(.7, 1.3))
                nectar_today *= np.exp(-rain[d]/22) * np.exp(-max(mean_t-32, 0)/14)
                sugar_fraction = rng.uniform(.25, .45)
                harvest_due = 0.0
                feeding = .7 if sugar+water < reserve*.6 and rng.random() < .25 else 0
                is_inspection = d % 7 == inspection_offset
                for hour in range(24):
                    timestamp = day + pd.Timedelta(hours=hour)
                    daylight = max(0, np.sin(np.pi*(hour-6)/12))
                    ext_t = mean_t + 5*np.sin(2*np.pi*(hour-8)/24)
                    ext_rh = np.clip(ext_hum_base - 12*np.sin(2*np.pi*(hour-8)/24), 15, 98)
                    int_t = ext_t + (33-ext_t)*(.35+.25*current_strength) - 1.5*shade
                    int_rh = np.clip(ext_rh*.55 + 29 + 5*flowering[d], 30, 95)
                    incoming = nectar_today * daylight / 7.595754
                    sugar += incoming * sugar_fraction
                    water += incoming * (1-sugar_fraction)
                    equilibrium_water = sugar * .17/.83
                    evaporated = max(water-equilibrium_water, 0) * .075 * ventilation * min(float(vpd_kpa(int_t, int_rh))/1.2, 1.8)
                    water = max(water-evaporated, 0)
                    consumed = min(sugar+water, (.09+.07*current_strength)/24)
                    fraction = sugar/max(sugar+water, 1e-9)
                    sugar = max(0, sugar-consumed*fraction)
                    water = max(0, water-consumed*(1-fraction))
                    moisture = 100*water/max(sugar+water, 1e-9)
                    desired_capping = np.clip((21-moisture)/3.0, 0, 1)
                    capped += .04 * (desired_capping-capped)
                    # Inspector samples the designated storage area at 18:00 before manipulation.
                    if hour == DECISION_HOUR and is_inspection:
                        surplus = max(sugar+water-reserve, 0)
                        measured_moisture = np.clip(moisture+rng.normal(0, .25), 10, 70)
                        measured_capping = np.clip(capped+rng.normal(0, .025), 0, 1)
                        harvest_due = (surplus*rng.uniform(.82, .94) if
                            moisture < 18.3 and capped > .75 and surplus > 3
                            and queenright and d-last_harvest > 28 and rng.random() < .75 else 0)
                        inspections.append(dict(hive_id=hive, apiary_id=apiary, timestamp=timestamp,
                            readiness_observed_at=timestamp, yield_observed_at=timestamp+pd.Timedelta(hours=1) if harvest_due else pd.NaT,
                            moisture_pct=measured_moisture, capping_fraction=measured_capping,
                            representative_sample=sugar+water>.5, queenright=queenright, disease_flag=False,
                            swarm_flag=False, reserve_verified=surplus > 3,
                            recoverable_yield_kg=round(harvest_due, 2) if harvest_due else np.nan,
                            data_origin="synthetic"))
                    removed, equip_delta, feed = 0., 0., 0.
                    if hour == DECISION_HOUR+1 and harvest_due > 0:
                        removed = min(harvest_due, sugar+water)
                        f = removed/max(sugar+water, 1e-9)
                        sugar *= 1-f; water *= 1-f
                        last_harvest = d
                        harvest_due = 0
                    if hour == 12 and feeding:
                        feed = feeding
                        sugar += feed*.6; water += feed*.4
                    if hour == 10 and rng.random() < 1/(365*3):
                        equip_delta = 3.0 if equipment == 0 else -equipment
                        equipment += equip_delta
                    bee_mass = 2.5*current_strength - .35*current_strength*daylight
                    weight = tare + equipment + sugar+water+bee_mass + rng.normal(0, .035)
                    if hour == DECISION_HOUR+1 and is_inspection:
                        weight -= rng.uniform(1.5, 3.0)  # temporary disturbance, not a permanent correction
                    row = dict(hive_id=hive, apiary_id=apiary, timestamp=timestamp,
                        weight_kg=weight,
                        int_temp=np.clip(np.round(int_t+rng.uniform(-2,2)),0,50),
                        int_hum=np.clip(np.round(int_rh+rng.uniform(-5,5)),20,90),
                        ext_temp=np.clip(np.round(ext_t+rng.uniform(-2,2)),0,50),
                        ext_hum=np.clip(np.round(ext_rh+rng.uniform(-5,5)),20,90),
                        rain_mm=rain[d]/24, wind_ms=max(0, rng.normal(2.2, 1)),
                        equipment_delta_kg=equip_delta, harvest_mass_removed_kg=removed, feed_kg=feed,
                        inspection_event=is_inspection and hour in (19, 20, 21),
                        tare_kg=tare, frame_count=frame_count, shade_fraction=shade, county=counties[a],
                        sensor_location="honey_storage", hive_type=hive_type, data_origin="synthetic")
                    for col in ("weight_kg", "int_temp", "int_hum", "ext_temp", "ext_hum"):
                        if rng.random() < .02: row[col] = np.nan
                    telemetry.append(row)
    return pd.DataFrame(telemetry), pd.DataFrame(inspections)


# %% Causal telemetry processing; decision at 18:00 EAT, before manipulation
SENSORS = ["weight_kg", "int_temp", "int_hum", "ext_temp", "ext_hum"]
EVENTS = ["equipment_delta_kg", "harvest_mass_removed_kg", "feed_kg", "inspection_event"]
BASE_FEATURES = ["weight_kg", "dw_24h", "night_loss_kg", "day_gain_kg", "int_vpd", "ext_vpd",
                 "vapor_pressure_gradient", "int_temp_mean", "int_temp_range", "int_hum_mean",
                 "ext_temp_mean", "ext_hum_mean", "rain_24h", "wind_mean", "weight_coverage",
                 "climate_coverage", "weight_age_h", "tare_kg", "frame_count", "shade_fraction", "hive_type_ktbh"]
FEATURES = BASE_FEATURES + [f"{c}_{w}d" for c in ("dw_24h", "night_loss_kg", "int_vpd", "rain_24h") for w in (3, 7, 14, 30)]
FEATURES += ["night_loss_decay_7d", "weight_slope_7d", "doy_sin", "doy_cos"]


def validate_telemetry(data):
    required = ["hive_id", "apiary_id", "timestamp", *SENSORS, *EVENTS,
                "rain_mm", "wind_ms", "tare_kg", "frame_count", "shade_fraction",
                "hive_type", "sensor_location", "data_origin"]
    missing = set(required)-set(data)
    if missing: raise ValueError(f"Telemetry columns missing: {sorted(missing)}")
    t = data.copy()
    t["timestamp"] = local_times(t.timestamp).array
    if t[["hive_id", "apiary_id", "timestamp", "data_origin"]].isna().any().any():
        raise ValueError("Hive/apiary/time/origin cannot be missing")
    if t.duplicated(["hive_id", "timestamp"]).any(): raise ValueError("Duplicate hive timestamps")
    if t.data_origin.nunique() != 1 or not set(t.data_origin) <= {"synthetic", "field"}:
        raise ValueError("Use exactly one explicit data origin per experiment")
    if t.groupby("hive_id").apiary_id.nunique().max() != 1:
        raise ValueError("Hive relocation requires explicit episode IDs")
    # This first fitted domain is deliberately explicit; other hive designs need
    # their own metadata, placement contract and held-out validation.
    if not t.hive_type.isin(["Langstroth", "Kenya_Top_Bar"]).all() or not t.sensor_location.eq("honey_storage").all():
        raise ValueError("Contract: Langstroth/Kenya_Top_Bar with sensor in designated honey_storage zone")
    if not t.timestamp.eq(t.timestamp.dt.floor("h")).all():
        raise ValueError("Provide hourly observations on the hour; aggregate higher-rate input causally first")
    for col in SENSORS + ["rain_mm", "wind_ms", "tare_kg", "frame_count", "shade_fraction", *EVENTS]:
        t[col] = pd.to_numeric(t[col], errors="raise")
        if np.isinf(t[col].to_numpy(float)).any(): raise ValueError(f"Infinite values in {col}")
    for col in ("int_hum", "ext_hum"):
        t.loc[~t[col].between(0, 100), col] = np.nan
    for col in ("int_temp", "ext_temp"):
        t.loc[~t[col].between(-10, 65), col] = np.nan
    t.loc[t.weight_kg < 0, "weight_kg"] = np.nan
    if t[EVENTS].isna().any().any(): raise ValueError("Event logs must be explicit, not unknown/blank")
    return t.sort_values(["hive_id", "timestamp"]).reset_index(drop=True)


def build_features(telemetry, policy=Policy()):
    t = validate_telemetry(telemetry)
    all_days = []
    for hive, g in t.groupby("hive_id", sort=True):
        g = g.set_index("timestamp").sort_index()
        original_end = g.index.max()
        grid = pd.date_range(g.index.min(), original_end, freq="h", tz=TZ)
        g = g.reindex(grid)
        # Metadata carry-forward is causal; missing complete telemetry hours do
        # not count as sensor readings. Do not silently invent event log entries.
        g["event_log_missing"] = g[EVENTS].isna().any(axis=1)
        for c in ("apiary_id", "tare_kg", "frame_count", "shade_fraction", "data_origin", "hive_type"):
            g[c] = g[c].ffill()
        corrections = (g.equipment_delta_kg.fillna(0) - g.harvest_mass_removed_kg.fillna(0) + g.feed_kg.fillna(0)).cumsum()
        g["adjusted_weight"] = g.weight_kg-corrections
        previous = g.weight_kg.ffill().shift(1)
        explained = (g[EVENTS].fillna(0).ne(0).any(axis=1))
        g["unexplained_jump"] = (g.weight_kg-previous).abs().gt(policy.jump_threshold_kg) & ~explained
        g["known_event"] = explained
        g["feeding_event"] = g.feed_kg.fillna(0).gt(0)
        g["int_vpd"] = vpd_kpa(g.int_temp, g.int_hum)
        g["ext_vpd"] = vpd_kpa(g.ext_temp, g.ext_hum)
        g["vapor_pressure_gradient"] = (vpd_kpa(g.int_temp, 100-g.int_hum)
                                           - vpd_kpa(g.ext_temp, 100-g.ext_hum))
        seen = pd.Series(g.index.where(g.weight_kg.notna()), index=g.index).ffill()
        g["weight_age_h"] = (pd.Series(g.index, index=g.index)-seen).dt.total_seconds()/3600
        g["decision_time"] = (g.index-pd.Timedelta(hours=DECISION_HOUR)).ceil("D") + pd.Timedelta(hours=DECISION_HOUR)
        # Aggregate all days at once; all groups end at their decision time.
        g['weight_valid'] = g.weight_kg.notna()
        g['climate_valid'] = g[['int_temp','int_hum','ext_temp','ext_hum']].notna().all(axis=1)
        gr = g.groupby('decision_time', sort=True)
        b = gr.agg(apiary_id=('apiary_id','last'), data_origin=('data_origin','last'),
            int_vpd=('int_vpd','mean'),ext_vpd=('ext_vpd','mean'),
            vapor_pressure_gradient=('vapor_pressure_gradient','mean'),
            int_temp_mean=('int_temp','mean'), int_temp_min=('int_temp','min'),int_temp_max=('int_temp','max'),
            int_hum_mean=('int_hum','mean'),ext_temp_mean=('ext_temp','mean'),ext_hum_mean=('ext_hum','mean'),
            wind_mean=('wind_ms','mean'),weight_count=('weight_valid','sum'),climate_count=('climate_valid','sum'),
            event_recent=('known_event','any'),anomaly_recent=('unexplained_jump','any'),
            feeding_recent=('feeding_event','any'),
            event_log_missing=('event_log_missing','any'),
            tare_kg=('tare_kg','last'),frame_count=('frame_count','last'),shade_fraction=('shade_fraction','last'),
            hive_type=('hive_type','last'))
        b = b.loc[b.index <= original_end].copy()
        idx = b.index
        def weights(index, column='adjusted_weight'):
            return g[column].reindex(index).to_numpy()
        b['hive_id'] = hive
        b['timestamp'] = idx
        b['weight_kg'] = weights(idx, 'weight_kg')
        b['adjusted_weight'] = weights(idx)
        b['dw_24h'] = weights(idx)-weights(idx-pd.Timedelta(days=1))
        night_end = (idx-pd.Timedelta(hours=5)).floor('D')+pd.Timedelta(hours=5)
        b['night_loss_kg'] = weights(night_end-pd.Timedelta(hours=8))-weights(night_end)
        day_end = (idx-pd.Timedelta(hours=18)).floor('D')+pd.Timedelta(hours=18)
        b['day_gain_kg'] = weights(day_end)-weights(day_end-pd.Timedelta(hours=12))
        b['int_temp_range'] = b.pop('int_temp_max')-b.pop('int_temp_min')
        b['weight_coverage'] = b.pop('weight_count')/24
        b['climate_coverage'] = b.pop('climate_count')/24
        b['weight_age_h'] = weights(idx, 'weight_age_h')
        b['rain_24h'] = gr.rain_mm.sum(min_count=20).reindex(idx)
        b['hive_type_ktbh'] = b.pop('hive_type').eq('Kenya_Top_Bar').astype(int)
        all_days.extend(b.reset_index(drop=True).to_dict('records'))
    daily = pd.DataFrame(all_days).sort_values(["hive_id", "timestamp"]).reset_index(drop=True)
    result=[]
    for _, g in daily.groupby("hive_id", sort=True):
        g=g.copy()
        for c in ("dw_24h", "night_loss_kg", "int_vpd", "rain_24h"):
            for w in (3, 7, 14, 30):
                g[f"{c}_{w}d"] = g[c].rolling(w, min_periods=max(2, w//2)).mean()
        g["night_loss_decay_7d"] = g.night_loss_kg.rolling(3, min_periods=2).mean()-g.night_loss_kg.shift(4).rolling(3, min_periods=2).mean()
        # Endpoint slope; name explicitly denotes average kg/day, not OLS.
        g["weight_slope_7d"] = (g.adjusted_weight-g.adjusted_weight.shift(7))/7
        g["history_days"] = np.arange(len(g))+1
        g["event_recent"] = g.event_recent.rolling(2, min_periods=1).max().astype(bool)
        g["anomaly_recent"] = g.anomaly_recent.rolling(2, min_periods=1).max().astype(bool)
        g["event_log_missing"] = g.event_log_missing.rolling(2, min_periods=1).max().astype(bool)
        g["feeding_recent"] = g.feeding_recent.rolling(policy.feeding_review_days, min_periods=1).max().astype(bool)
        doy=g.timestamp.dt.strftime("%j").astype(int)
        g["doy_sin"],g["doy_cos"] = np.sin(2*np.pi*doy/365.25),np.cos(2*np.pi*doy/365.25)
        result.append(g)
    return pd.concat(result, ignore_index=True)


def join_inspections(daily, inspections, policy=Policy()):
    """Labels join only the inspected decision time. Never backfill targets."""
    i=inspections.copy()
    required={"hive_id","apiary_id","timestamp","readiness_observed_at","yield_observed_at",
        "moisture_pct","capping_fraction","representative_sample","queenright","disease_flag",
        "swarm_flag","reserve_verified","recoverable_yield_kg","data_origin"}
    if required-set(i): raise ValueError(f"Inspection columns missing: {sorted(required-set(i))}")
    for c in ("timestamp","readiness_observed_at","yield_observed_at"):
        i[c]=local_times(i[c]).array
    if i.duplicated(["hive_id","timestamp"]).any(): raise ValueError("Duplicate inspections")
    if set(i.data_origin) != set(daily.data_origin): raise ValueError("Inspection/telemetry origin mismatch")
    if not i.timestamp.eq(i.timestamp.dt.normalize()+pd.Timedelta(hours=DECISION_HOUR)).all():
        raise ValueError("Inspect at 18:00 decision time for this experiment; do not relabel other hours silently")
    for c in ("representative_sample","queenright","disease_flag","swarm_flag","reserve_verified"):
        if not i[c].dropna().isin([True,False,0,1]).all(): raise ValueError(f"{c} must be boolean or missing")
    if i.readiness_observed_at.lt(i.timestamp).any(): raise ValueError("Label observed before decision")
    if (i.recoverable_yield_kg.notna() & (i.yield_observed_at.isna() | i.yield_observed_at.lt(i.timestamp))).any():
        raise ValueError("Harvest yield requires a valid later observation timestamp")
    if (i.recoverable_yield_kg.dropna()<0).any(): raise ValueError("Yield cannot be negative")
    i["ready"] = np.nan
    measured = i.moisture_pct.between(0,100) & i.capping_fraction.between(0,1) & i.representative_sample.eq(True)
    i.loc[measured,"ready"] = ((i.loc[measured,"moisture_pct"]<=policy.moisture_limit_pct)
        & (i.loc[measured,"capping_fraction"]>=policy.capping_min)).astype(int)
    merged=daily.merge(i.drop(columns="data_origin"), on=["hive_id","apiary_id","timestamp"], how="left", validate="one_to_one")
    # Availability, not just measurement date, governs label eligibility.
    merged["readiness_label_time"] = merged.readiness_observed_at
    return merged


def split_masks(data, plan=SplitPlan(), label_time="readiness_label_time"):
    t=data.timestamp
    a,b,c=(pd.Timestamp(v,tz=TZ) for v in (plan.train_end,plan.calibration_end,plan.validation_end))
    if not a<b<c: raise ValueError("Split boundaries must be increasing")
    if set(plan.development_apiaries)&set(plan.heldout_apiaries): raise ValueError("Overlapping apiary groups")
    if not set(data.apiary_id) <= set(plan.development_apiaries)|set(plan.heldout_apiaries):
        raise ValueError("SplitPlan must assign every apiary to development or holdout")
    dev=data.apiary_id.isin(plan.development_apiaries) & ~data.hive_id.isin(plan.heldout_hives)
    observed=data[label_time]
    return {
        "train": dev & (t<a) & (observed<a),
        "calibration": dev & (t>=a) & (t<b) & (observed<b),
        "validation": dev & (t>=b) & (t<c) & (observed<c),
        "future_known":dev & (t>=c) & observed.notna(),
        "future_new_hives":data.apiary_id.isin(plan.development_apiaries) & data.hive_id.isin(plan.heldout_hives) & (t>=c) & observed.notna(),
        "future_new_apiaries":data.apiary_id.isin(plan.heldout_apiaries) & (t>=c) & observed.notna(),
    }


# %% Model candidates, independent probability calibration and policy selection
class PlattModel:
    """Calibration learns on a distinct chronological calibration partition."""
    def __init__(self, estimator, columns):
        self.estimator, self.columns = estimator, list(columns)

    @staticmethod
    def logits(p):
        p=np.clip(p, 1e-5, 1-1e-5)
        return np.log(p/(1-p)).reshape(-1,1)

    def fit(self, train, calibration, target="ready"):
        if train[target].nunique()<2 or calibration[target].nunique()<2:
            raise ValueError("Both classes are required in training and calibration")
        self.estimator.fit(train[self.columns], train[target].astype(int))
        p=self.estimator.predict_proba(calibration[self.columns])[:,1]
        self.calibrator=LogisticRegression(C=1.0, random_state=SEED).fit(self.logits(p), calibration[target].astype(int))
        return self

    def predict_proba(self, frame):
        p=self.estimator.predict_proba(frame[self.columns])[:,1]
        return self.calibrator.predict_proba(self.logits(p))[:,1]


def classifier_candidates():
    from sklearn.preprocessing import StandardScaler
    candidates={
        "prior_baseline":DummyClassifier(strategy="prior"),
        "logistic":make_pipeline(SimpleImputer(strategy="median",add_indicator=True), StandardScaler(),
                                 LogisticRegression(C=.5,max_iter=1500,random_state=SEED)),
        "xgboost":XGBClassifier(n_estimators=250,max_depth=3,learning_rate=.04,min_child_weight=5,
            subsample=.85,colsample_bytree=.85,reg_lambda=5,n_jobs=2,random_state=SEED,eval_metric="logloss"),
    }
    try:
        from catboost import CatBoostClassifier
        candidates["catboost"]=CatBoostClassifier(iterations=250,depth=4,learning_rate=.04,
            l2_leaf_reg=5,verbose=False,thread_count=2,random_seed=SEED,allow_writing_files=False)
    except ImportError:
        warnings.warn("CatBoost unavailable: benchmark will explicitly omit it")
    try:
        from lightgbm import LGBMClassifier
        candidates["lightgbm"]=LGBMClassifier(n_estimators=250,max_depth=4,num_leaves=12,
            min_child_samples=20,learning_rate=.04,reg_lambda=5,verbosity=-1,n_jobs=2,random_state=SEED)
    except ImportError:
        warnings.warn("LightGBM unavailable: benchmark will explicitly omit it")
    return candidates


def choose_threshold(y, p, policy=Policy()):
    """Validation policy only. No feasible threshold means no ready advisory."""
    y=np.asarray(y,int); p=np.asarray(p,float)
    feasible=[]
    for threshold in np.unique(np.r_[np.linspace(.4,.99,80),p]):
        selected=p>=threshold
        n=int(selected.sum())
        if n<policy.min_validation_recommendations: continue
        precision=float(y[selected].mean())
        if precision>=policy.precision_target:
            feasible.append((int(y[selected].sum()),precision,float(threshold),n))
    if not feasible:
        return {"threshold":1.01,"enabled":False,"validation_recommendations":0,
                "validation_precision":None,"reason":"No validation threshold meets precision and sample requirements"}
    tp,precision,threshold,n=max(feasible,key=lambda x:(x[0],x[1],x[2]))
    return {"threshold":threshold,"enabled":True,"validation_recommendations":n,
            "validation_precision":precision,"reason":"Empirical validation criterion; not a field guarantee"}


def binary_metrics(y,p,threshold):
    y=np.asarray(y,int); p=np.asarray(p,float); pred=p>=threshold
    n_ready=int(pred.sum()); negatives=int((y==0).sum())
    return dict(n=len(y),positive_rate=float(y.mean()),brier=float(brier_score_loss(y,p)),
        pr_auc=float(average_precision_score(y,p)) if y.sum() else None,
        recommendations=n_ready,precision=float(precision_score(y,pred,zero_division=0)) if n_ready else None,
        recall=float(recall_score(y,pred,zero_division=0)),
        false_positive_rate=float(((y==0)&pred).sum()/negatives) if negatives else None)


def quality_mask(data, policy=Policy()):
    return (data.weight_coverage.ge(policy.min_weight_coverage)
        & data.climate_coverage.ge(policy.min_weight_coverage)
        & data.weight_age_h.le(policy.max_weight_age_h) & data.weight_kg.notna()
        & data.history_days.ge(30) & ~data.anomaly_recent & ~data.event_recent & ~data.event_log_missing)


def release_check(test_rows,policy=Policy()):
    """Pooled success must not hide weak or undersampled hive-type groups."""
    def meets(group):
        rows=[r for r in test_rows if r["task"]=="readiness" and r["group"]==group]
        return len(rows)==3 and all(r["recommendations"]>=policy.min_validation_recommendations
            and r["precision"] is not None and r["precision"]>=policy.precision_target for r in rows)
    pooled=meets("all")
    by_type={name:meets(name) for name in ("Langstroth","Kenya_Top_Bar")}
    return dict(pooled_precision_passed=pooled,hive_type_release_passed=by_type,
                holdout_precision_passed=pooled and all(by_type.values()))


def conformal_radius(residuals, alpha=.1):
    scores=np.asarray(residuals,float)
    scores=scores[np.isfinite(scores)]
    if len(scores)==0: raise ValueError("No calibration residuals")
    rank=math.ceil((len(scores)+1)*(1-alpha))
    if rank>len(scores): return float("inf")
    return float(np.sort(scores)[rank-1])


def predict_yield(bundle, data):
    centre=np.maximum(0,bundle["yield_model"].predict(data[bundle["features"]]))
    r=bundle["yield_radius_kg"]
    return centre,np.maximum(0,centre-r),centre+r


def train_experiment(daily, inspections, plan=SplitPlan(), policy=Policy()):
    data=join_inspections(daily,inspections,policy)
    masks=split_masks(data,plan)
    valid=quality_mask(data,policy) & data.ready.notna()
    parts={k:data.loc[m&valid].copy() for k,m in masks.items()}
    for name in ("train","calibration","validation"):
        if len(parts[name])<30 or parts[name].ready.nunique()!=2:
            raise ValueError(f"Insufficient independently observed readiness labels in {name}: {len(parts[name])}")
    fitted, rows={},[]
    for name,est in classifier_candidates().items():
        model=PlattModel(est,FEATURES).fit(parts["train"],parts["calibration"])
        p=model.predict_proba(parts["validation"])
        selection=choose_threshold(parts["validation"].ready,p,policy)
        metrics=binary_metrics(parts["validation"].ready,p,selection["threshold"])
        rows.append(dict(model=name,policy_enabled=selection["enabled"],**metrics,threshold=selection["threshold"]))
        fitted[name]=(model,selection)
    leaderboard=pd.DataFrame(rows).sort_values(["policy_enabled","brier"],ascending=[False,True]).reset_index(drop=True)
    champion=leaderboard.iloc[0].model
    readiness,selected_policy=fitted[champion]
    # This ablation is development evidence, not extra tuning on the final test.
    ablations=[]
    for label,columns in {
        "full":FEATURES,
        "without_calendar":[c for c in FEATURES if not c.startswith("doy_")],
        "without_vapor_features":[c for c in FEATURES if "vpd" not in c and "vapor" not in c],
        "without_weather":[c for c in FEATURES if not any(x in c for x in ("ext_","rain","wind","vapor"))],
    }.items():
        model=PlattModel(classifier_candidates()["xgboost"],columns).fit(parts["train"],parts["calibration"])
        p=model.predict_proba(parts["validation"])
        ablations.append(dict(features=label,**binary_metrics(parts["validation"].ready,p,.5)))
    # Yield labels exist only after a weighed extraction. Do not create zero
    # targets on unharvested days, or infer yields from readiness labels.
    ymasks=split_masks(data,plan,label_time="yield_observed_at")
    yvalid=quality_mask(data,policy)&data.recoverable_yield_kg.notna()
    yp={k:data.loc[m&yvalid].copy() for k,m in ymasks.items()}
    for k in ("train","validation"):
        if len(yp[k])<5: raise ValueError(f"Too few measured yield events in {k}: {len(yp[k])}")
    yield_candidates={
        "median_baseline":DummyRegressor(strategy="median"),
        "random_forest":make_pipeline(SimpleImputer(strategy="median",add_indicator=True),
            RandomForestRegressor(n_estimators=250,min_samples_leaf=3,n_jobs=2,random_state=SEED)),
        "xgboost":XGBRegressor(n_estimators=250,max_depth=2,min_child_weight=4,learning_rate=.04,
            reg_lambda=10,subsample=.85,n_jobs=2,random_state=SEED),
    }
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    yield_candidates["catboost"]=CatBoostRegressor(iterations=250,depth=3,learning_rate=.04,
        loss_function="MAE",verbose=False,thread_count=2,random_seed=SEED,allow_writing_files=False)
    yield_candidates["lightgbm"]=LGBMRegressor(n_estimators=200,max_depth=3,num_leaves=7,
        min_child_samples=8,reg_lambda=5,learning_rate=.04,verbosity=-1,n_jobs=2,random_state=SEED)
    yield_rows=[]
    for name,m in yield_candidates.items():
        m.fit(yp["train"][FEATURES],yp["train"].recoverable_yield_kg)
        p=np.maximum(0,m.predict(yp["validation"][FEATURES]))
        yield_rows.append(dict(model=name,n=len(p),mae_kg=float(mean_absolute_error(yp["validation"].recoverable_yield_kg,p))))
    yield_board=pd.DataFrame(yield_rows).sort_values("mae_kg").reset_index(drop=True)
    yield_name=yield_board.iloc[0].model
    yield_model=yield_candidates[yield_name]
    calibration=yp["calibration"]
    interval_available=len(calibration)>=policy.min_interval_calibration
    radius=(conformal_radius(np.abs(calibration.recoverable_yield_kg-
        np.maximum(0,yield_model.predict(calibration[FEATURES]))),policy.interval_alpha)
        if interval_available else float("inf"))
    bundle=dict(version=VERSION,features=list(FEATURES),readiness_model=readiness,
        readiness_champion=champion,selected_policy=selected_policy,yield_model=yield_model,
        yield_champion=yield_name,yield_radius_kg=radius,interval_available=interval_available,
        policy=asdict(policy),split_plan=asdict(plan),data_origin=data.data_origin.iloc[0],
        field_validated=False,decision_hour_local=DECISION_HOUR,timezone=TZ,
        training_domains={"hive_types":["Langstroth","Kenya_Top_Bar"],"sensor_location":"honey_storage"},
        training_feature_bounds={c:(float(parts["train"][c].min()),float(parts["train"][c].max())) for c in FEATURES})
    # Freeze candidate and threshold before evaluating any final test partition.
    tests=[]; predictions=[]
    for split in ("future_known","future_new_hives","future_new_apiaries"):
        pset=parts[split]
        if len(pset):
            prob=readiness.predict_proba(pset)
            for group,sub in [("all",pset),("Langstroth",pset[pset.hive_type_ktbh==0]),("Kenya_Top_Bar",pset[pset.hive_type_ktbh==1])]:
                if len(sub):
                    tests.append(dict(task="readiness",split=split,group=group,apiaries=int(sub.apiary_id.nunique()),
                        hives=int(sub.hive_id.nunique()),**binary_metrics(sub.ready,readiness.predict_proba(sub),selected_policy["threshold"])))
            pp=pset[["hive_id","apiary_id","timestamp","ready","hive_type_ktbh"]].copy()
            pp["probability"]=prob; pp["split"]=split
            predictions.append(pp)
        ys=yp[split]
        if len(ys):
            p,lo,hi=predict_yield(bundle,ys)
            tests.append(dict(task="yield",split=split,group="all",n=len(ys),apiaries=int(ys.apiary_id.nunique()),
                hives=int(ys.hive_id.nunique()),mae_kg=float(mean_absolute_error(ys.recoverable_yield_kg,p)),
                bias_kg=float(np.mean(p-ys.recoverable_yield_kg)),interval_coverage=float(np.mean((ys.recoverable_yield_kg>=lo)&(ys.recoverable_yield_kg<=hi))) if interval_available else None,
                interval_mean_width_kg=float(np.mean(hi-lo)) if interval_available else None))
    counts={k:{"readiness_labels":len(parts[k]),"yield_events":len(yp[k])} for k in parts}
    release=release_check(tests,policy)
    bundle.update(release)
    policy_tests=[]
    historical_inspections=inspections.copy()
    historical_inspections["timestamp"]=local_times(historical_inspections.timestamp).array
    for split in ("future_known","future_new_hives","future_new_apiaries"):
        ps=parts[split]
        if ps.empty: continue
        # Predictive policy evaluation must not use the same inspection that
        # supplies the test label, even as an authoritative measurement veto.
        actions=[advise(bundle,r,historical_inspections[historical_inspections.timestamp<r.timestamp],
            allow_synthetic_demo=bundle["data_origin"]=="synthetic")["action"] for _,r in ps.iterrows()]
        selected=np.asarray(actions)=="candidate_ready_verify_frames"
        policy_tests.append(dict(split=split,n=len(ps),candidate_count=int(selected.sum()),
            candidate_precision=float(ps.loc[selected,"ready"].mean()) if selected.any() else None,
            inspection_fraction=float(np.mean(np.asarray(actions)=="inspect_hive")),
            monitor_fraction=float(np.mean(np.asarray(actions)=="wait_and_monitor"))))
    metadata=dict(version=VERSION,data_origin=bundle["data_origin"],field_validated=False,
        readiness_champion=champion,yield_champion=yield_name,selected_policy=selected_policy,
        holdout_precision_passed=bundle["holdout_precision_passed"],
        pooled_precision_passed=bundle["pooled_precision_passed"],
        hive_type_release_passed=bundle["hive_type_release_passed"],
        policy=asdict(policy),split_plan=asdict(plan),sample_counts=counts,
        interval_calibration_events=len(calibration),interval_available=interval_available,
        interval_radius_kg=radius if np.isfinite(radius) else None,
        decision_time="18:00 Africa/Nairobi, pre-extraction",features=FEATURES,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        simulator_seed=SEED if bundle["data_origin"]=="synthetic" else None,
        simulated_sensor_profile="DHT11-like: uniform +/-2 C, +/-5 percent RH, rounded; not a calibration certificate" if bundle["data_origin"]=="synthetic" else None,
        libraries={p:importlib.metadata.version(p) for p in ("numpy","pandas","scikit-learn","xgboost","catboost","lightgbm")},
        limitations=["Synthetic mechanism, no field accuracy established" if bundle["data_origin"]=="synthetic" else "Local field performance still requires independent validation",
          "Field validation requires representative local moisture, inspection and weighed extraction records",
          "Intervals rely on calibration residual transfer; check coverage under shift",
          "No trained fanning/absconding detector; colony checks use logged inspections",
          "Six apiaries and correlated hive-days are not independent field replication",
          "Readiness label is the configured conservative moisture-plus-capping policy",
          "Do not use this model to authorize a real harvest"])
    return bundle,dict(readiness_leaderboard=leaderboard,yield_leaderboard=yield_board,
        ablations=pd.DataFrame(ablations),test_metrics=pd.DataFrame(tests),metadata=metadata,
        policy_metrics=pd.DataFrame(policy_tests),
        test_predictions=pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame())


# %% Advisory: uncertainty, quality, health and reserves can suppress readiness
def advise(bundle, row, inspections, allow_synthetic_demo=False):
    if bundle["data_origin"]=="synthetic" and not allow_synthetic_demo:
        raise ValueError("Synthetic model is demonstration-only; explicitly opt into demo mode")
    if row["data_origin"]!=bundle["data_origin"]:
        raise ValueError("Cannot apply a synthetic model to field telemetry, or mix data origins")
    policy=Policy(**bundle["policy"])
    frame=pd.DataFrame([row])
    p=float(bundle["readiness_model"].predict_proba(frame)[0])
    centre,lo,hi=predict_yield(bundle,frame)
    reasons=[]
    if not bundle.get("holdout_precision_passed",False):
        reasons.append("Held-out tests lack required precision or sample counts overall or within a hive type")
    if not quality_mask(frame,policy).iloc[0]: reasons.append("Telemetry incomplete, stale, disturbed, or history shorter than 30 days")
    if row.get("feeding_recent",True): reasons.append("Recent or unknown feeding history: inspect crop provenance before extraction")
    i=inspections[inspections.hive_id==row["hive_id"]].copy()
    if len(i):
        i["timestamp"]=local_times(i.timestamp).array
        i["readiness_observed_at"]=local_times(i.readiness_observed_at).array
        # Health/reserve findings are available when the inspection record is
        # available; do not use a later-entered record to decide in the past.
        i=i[(i.timestamp<=row["timestamp"]) & (i.readiness_observed_at<=row["timestamp"])].sort_values("timestamp")
    if i.empty:
        reasons.append("No available colony-health/reserve inspection")
    else:
        latest=i.iloc[-1]
        age=(row["timestamp"]-latest.timestamp).total_seconds()/86400
        if age>policy.max_inspection_age_days: reasons.append("Colony-health/reserve inspection is stale")
        if pd.isna(latest.queenright) or latest.queenright!=True:
            reasons.append("Queen status unknown or queenless")
        for col in ("disease_flag","swarm_flag"):
            if pd.isna(latest[col]) or latest[col]!=False: reasons.append(f"Health check requires attention: {col}")
        if pd.isna(latest.reserve_verified) or latest.reserve_verified!=True: reasons.append("Adequate colony reserves not verified")
        representative=latest.get("representative_sample",False)
        if latest.timestamp==row["timestamp"] and pd.notna(representative) and representative==True:
            if pd.notna(latest.get("moisture_pct")) and latest.moisture_pct>policy.moisture_limit_pct:
                reasons.append("Current measured moisture exceeds project policy; model score cannot override it")
            if pd.notna(latest.get("capping_fraction")) and latest.capping_fraction<policy.capping_min:
                reasons.append("Current measured capping is below project policy; model score cannot override it")
    if not bundle["interval_available"]: reasons.append("Too few independent yield-calibration events")
    # Simple conservative range monitor. It is not a proven distribution-shift detector.
    out_of_range=[]
    for c in ("weight_kg","int_temp_mean","int_hum_mean","ext_temp_mean","ext_hum_mean","tare_kg","frame_count"):
        low,high=bundle["training_feature_bounds"][c]
        value=row[c]
        tolerance=max((high-low)*.15,.5)
        if pd.notna(value) and (value<low-tolerance or value>high+tolerance): out_of_range.append(c)
    if out_of_range: reasons.append("Outside fitted feature ranges: "+", ".join(out_of_range))
    if reasons:
        action="inspect_hive"
    elif not bundle["selected_policy"]["enabled"]:
        action="inspect_hive"; reasons.append("Validation did not support a recommendation threshold")
    elif p>=bundle["selected_policy"]["threshold"]:
        if lo[0]>=policy.min_harvest_kg:
            action="candidate_ready_verify_frames"
        else:
            action="inspect_hive"; reasons.append("Yield lower bound does not support the minimum harvest quantity")
    elif p<.35:
        action="wait_and_monitor"
    else:
        action="inspect_hive"; reasons.append("Readiness probability is inconclusive")
    return dict(hive_id=row["hive_id"],timestamp=str(row["timestamp"]),action=action,
        probability_ready=p,yield_kg=float(centre[0]),yield_interval_kg=[float(lo[0]),float(hi[0])] if bundle["interval_available"] else None,
        reasons=reasons,data_origin=bundle["data_origin"],field_validated=bundle["field_validated"],
        demonstration_only=bundle["data_origin"]=="synthetic",harvest_authorized=False,
        holdout_precision_passed=bundle.get("holdout_precision_passed",False))


def save_results(bundle,results,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,output/"harvest_readiness_v2.joblib")
    for name,value in results.items():
        if isinstance(value,pd.DataFrame): value.to_csv(output/f"{name}.csv",index=False)
    (output/"model_card.json").write_text(json.dumps(results["metadata"],indent=2,allow_nan=False)+"\n")
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="outputs/harvest_v2")
    parser.add_argument("--telemetry",help="Field hourly CSV; requires --inspections")
    parser.add_argument("--inspections",help="Actual field labels CSV")
    parser.add_argument("--split-config",help="JSON SplitPlan using actual apiary IDs, held-out hives and date boundaries")
    args=parser.parse_args()
    if bool(args.telemetry)!=bool(args.inspections): parser.error("Supply both --telemetry and --inspections")
    plan=SplitPlan()
    if args.telemetry:
        telemetry,inspections=pd.read_csv(args.telemetry),pd.read_csv(args.inspections)
        if not telemetry.data_origin.eq("field").all(): raise ValueError("Field files must declare data_origin=field")
        if not args.split_config: parser.error("Field training requires --split-config; no synthetic apiary/date defaults")
        config=json.loads(Path(args.split_config).read_text())
        for k in ("development_apiaries","heldout_apiaries","heldout_hives"):
            if k not in config: parser.error(f"Split configuration must declare {k}")
            config[k]=tuple(config[k])
        plan=SplitPlan(**config)
    else:
        print("DEMONSTRATION ONLY: generating synthetic telemetry",flush=True)
        telemetry,inspections=simulate()
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    origin=telemetry.data_origin.iloc[0]
    telemetry.to_csv(output/f"{origin}_hourly_telemetry.csv.gz",index=False,compression="gzip")
    inspections.to_csv(output/f"{origin}_inspections.csv",index=False)
    print(f"{len(telemetry):,} hourly rows, {len(inspections):,} inspections",flush=True)
    daily=build_features(telemetry)
    daily.to_csv(output/"daily_features.csv.gz",index=False,compression="gzip")
    print("Training development candidates; final tests remain separate",flush=True)
    bundle,results=train_experiment(daily,inspections,plan=plan)
    save_results(bundle,results,output)
    sample_hive=daily.hive_id.iloc[-1]
    row=daily[daily.hive_id==sample_hive].iloc[-1]
    advisory=advise(bundle,row,inspections,allow_synthetic_demo=origin=="synthetic")
    (output/"example_advisory.json").write_text(json.dumps(advisory,indent=2,allow_nan=False)+"\n")
    print(results["readiness_leaderboard"].to_string(index=False))
    print(results["test_metrics"].to_string(index=False))
    print(json.dumps(advisory,indent=2))


if __name__=="__main__":
    main()

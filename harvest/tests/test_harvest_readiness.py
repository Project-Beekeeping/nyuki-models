"""Behavioral checks for time leakage and unsafe advisories."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import harvest_readiness as h


def telemetry(days=45):
    ts=pd.date_range("2024-01-01",periods=days*24,freq="h",tz=h.TZ)
    return pd.DataFrame(dict(timestamp=ts,hive_id="test",apiary_id="apiary_0",
        weight_kg=60-np.arange(len(ts))*.01,int_temp=30.,int_hum=60.,ext_temp=24.,ext_hum=50.,
        rain_mm=0.,wind_ms=2.,equipment_delta_kg=0.,harvest_mass_removed_kg=0.,feed_kg=0.,
        inspection_event=False,tare_kg=22.,frame_count=10,shade_fraction=.5,
        hive_type="Langstroth",sensor_location="honey_storage",data_origin="synthetic"))


def inspection(timestamp,**updates):
    row=dict(hive_id="test",apiary_id="apiary_0",timestamp=timestamp,
        readiness_observed_at=timestamp,yield_observed_at=pd.NaT,moisture_pct=17.5,
        capping_fraction=.9,representative_sample=True,queenright=True,disease_flag=False,
        swarm_flag=False,reserve_verified=True,recoverable_yield_kg=np.nan,data_origin="synthetic")
    row.update(updates)
    return pd.DataFrame([row])


def test_future_changes_cannot_change_existing_features():
    data=telemetry();cut=pd.Timestamp("2024-02-05 18:00",tz=h.TZ)
    base=h.build_features(data[data.timestamp<=cut])
    modified=data.copy()
    modified.loc[modified.timestamp>cut,h.SENSORS]=999
    after=h.build_features(modified)
    assert_frame_equal(base,after[after.timestamp<=cut].reset_index(drop=True))


def test_true_24h_and_cross_midnight_night_loss():
    daily=h.build_features(telemetry(4))
    row=daily.iloc[-1]
    assert row.dw_24h==pytest.approx(-.24)
    assert row.night_loss_kg==pytest.approx(.08)
    assert row.day_gain_kg==pytest.approx(-.12)


def test_six_day_gap_is_not_interpolated_and_has_no_good_coverage():
    data=telemetry();a=pd.Timestamp("2024-01-12",tz=h.TZ);b=a+pd.Timedelta(days=6)
    data.loc[data.timestamp.between(a,b),"weight_kg"]=np.nan
    daily=h.build_features(data)
    gap=daily[daily.timestamp.between(a+pd.Timedelta(days=1),b-pd.Timedelta(days=1))]
    assert gap.weight_kg.isna().all()
    assert gap.weight_coverage.eq(0).all()
    assert not h.quality_mask(gap).any()


def test_known_equipment_change_is_corrected_unlogged_jump_is_flagged():
    base=telemetry();changed=base.copy();event=80
    changed.loc[event:,"weight_kg"]+=3
    changed.loc[event,"equipment_delta_kg"]=3
    daily=h.build_features(changed)
    reference=h.build_features(base)
    np.testing.assert_allclose(daily.dw_24h,reference.dw_24h,equal_nan=True,atol=1e-10)
    changed.loc[event,"equipment_delta_kg"]=0
    unexplained=h.build_features(changed)
    assert unexplained.anomaly_recent.any()
    assert unexplained.weight_kg.max()>reference.weight_kg.max()-2


def test_missing_hour_marks_unknown_event_log_and_reduces_coverage():
    data=telemetry().drop(index=75)
    daily=h.build_features(data)
    assert daily.event_log_missing.any()
    assert daily.loc[daily.event_log_missing,"weight_coverage"].min()<1


def test_no_inspection_means_unknown_not_negative():
    daily=h.build_features(telemetry())
    measured=daily.timestamp.iloc[35]
    joined=h.join_inspections(daily,inspection(measured))
    assert joined.ready.notna().sum()==1
    assert joined.loc[joined.timestamp==measured,"ready"].item()==1
    assert not set(["moisture_pct","capping_fraction","ready","recoverable_yield_kg"])&set(h.FEATURES)


def test_label_availability_and_group_holdouts_are_enforced():
    ts=pd.Timestamp("2024-12-31 18:00",tz=h.TZ)
    rows=pd.DataFrame(dict(timestamp=[ts,ts,ts,ts],readiness_label_time=[ts,ts+pd.Timedelta(days=2),ts,ts],
        hive_id=["apiary_0_h00","apiary_0_h01","apiary_0_h04","apiary_4_h00"],
        apiary_id=["apiary_0","apiary_0","apiary_0","apiary_4"]))
    masks=h.split_masks(rows)
    assert masks["train"].tolist()==[True,False,False,False]


def test_infeasible_precision_policy_disables_ready_recommendations():
    policy=replace(h.Policy(),min_validation_recommendations=3)
    selected=h.choose_threshold([0,0,0,1],[.9,.9,.9,.9],policy)
    assert not selected["enabled"] and selected["threshold"]>1


def test_conformal_radius_uses_finite_sample_rank():
    assert h.conformal_radius(np.arange(1,11),.1)==10
    assert np.isinf(h.conformal_radius([1],.1))


def test_pooled_precision_cannot_hide_a_weak_or_sparse_hive_type():
    rows=[dict(task="readiness",group=group,recommendations=20,precision=.95)
        for group in ("all","Langstroth","Kenya_Top_Bar") for _ in range(3)]
    assert h.release_check(rows)["holdout_precision_passed"]
    rows[-1]["recommendations"]=3
    checked=h.release_check(rows)
    assert checked["pooled_precision_passed"] and not checked["holdout_precision_passed"]
    rows[-1]["recommendations"]=20;rows[-1]["precision"]=.8
    assert not h.release_check(rows)["holdout_precision_passed"]


class ConstantReady:
    def predict_proba(self,data): return np.full(len(data),.99)


class ConstantYield:
    def predict(self,data): return np.full(len(data),10.)


def bundle():
    return dict(data_origin="synthetic",field_validated=False,holdout_precision_passed=True,features=h.FEATURES,
        readiness_model=ConstantReady(),yield_model=ConstantYield(),yield_radius_kg=2.,interval_available=True,
        selected_policy=dict(enabled=True,threshold=.9),policy=h.asdict(h.Policy()),
        training_feature_bounds={c:(-1000,1000) for c in h.FEATURES})


def test_synthetic_model_requires_demo_opt_in_and_rejects_field_data():
    row=h.build_features(telemetry()).iloc[-1]
    with pytest.raises(ValueError,match="demonstration-only"):
        h.advise(bundle(),row,inspection(row.timestamp))
    row=row.copy();row["data_origin"]="field"
    with pytest.raises(ValueError,match="data origins"):
        h.advise(bundle(),row,inspection(row.timestamp),allow_synthetic_demo=True)


@pytest.mark.parametrize("change",[
    {"queenright":False},{"queenright":np.nan},{"reserve_verified":False},
    {"queenright":"False"},{"reserve_verified":"False"},{"disease_flag":"unknown"},
    {"disease_flag":True},{"swarm_flag":True},
])
def test_high_model_score_cannot_override_colony_checks(change):
    row=h.build_features(telemetry()).iloc[-1]
    result=h.advise(bundle(),row,inspection(row.timestamp,**change),allow_synthetic_demo=True)
    assert result["action"]=="inspect_hive"
    assert not result["harvest_authorized"]


def test_later_record_cannot_supply_a_past_health_check():
    row=h.build_features(telemetry()).iloc[-1]
    i=inspection(row.timestamp,readiness_observed_at=row.timestamp+pd.Timedelta(days=1))
    result=h.advise(bundle(),row,i,allow_synthetic_demo=True)
    assert result["action"]=="inspect_hive"


def test_no_health_record_is_an_explicit_unknown():
    row=h.build_features(telemetry()).iloc[-1]
    empty=inspection(row.timestamp).iloc[:0]
    result=h.advise(bundle(),row,empty,allow_synthetic_demo=True)
    assert result["action"]=="inspect_hive"


def test_recent_feeding_and_failed_holdout_check_block_advisories():
    row=h.build_features(telemetry()).iloc[-1].copy()
    row["feeding_recent"]=True
    result=h.advise(bundle(),row,inspection(row.timestamp),allow_synthetic_demo=True)
    assert result["action"]=="inspect_hive"


@pytest.mark.parametrize("reading",[{"moisture_pct":22.},{"capping_fraction":.4}])
def test_actual_current_measurements_override_high_model_probability(reading):
    row=h.build_features(telemetry()).iloc[-1]
    result=h.advise(bundle(),row,inspection(row.timestamp,**reading),allow_synthetic_demo=True)
    assert result["action"]=="inspect_hive"
    row["feeding_recent"]=False
    blocked=bundle();blocked["holdout_precision_passed"]=False
    result=h.advise(blocked,row,inspection(row.timestamp),allow_synthetic_demo=True)
    assert result["action"]=="inspect_hive"


def test_stale_inspection_and_small_yield_lower_bound_block_advisories():
    row=h.build_features(telemetry()).iloc[-1]
    old=inspection(row.timestamp-pd.Timedelta(days=20))
    assert h.advise(bundle(),row,old,allow_synthetic_demo=True)["action"]=="inspect_hive"
    uncertain=bundle();uncertain["yield_radius_kg"]=9
    assert h.advise(uncertain,row,inspection(row.timestamp),allow_synthetic_demo=True)["action"]=="inspect_hive"


def test_valid_advisory_is_still_a_demonstration_without_harvest_authorization():
    row=h.build_features(telemetry()).iloc[-1]
    result=h.advise(bundle(),row,inspection(row.timestamp),allow_synthetic_demo=True)
    assert result["action"]=="candidate_ready_verify_frames"
    assert result["demonstration_only"] and not result["harvest_authorized"]


@pytest.mark.parametrize("bad",["duplicate","naive","hive_type"])
def test_invalid_telemetry_fails_explicitly(bad):
    data=telemetry(4)
    if bad=="duplicate":data=pd.concat([data,data.iloc[:1]])
    if bad=="naive":data["timestamp"]=data.timestamp.dt.tz_localize(None)
    if bad=="hive_type":data["hive_type"]="log_hive"
    with pytest.raises(ValueError):h.build_features(data)

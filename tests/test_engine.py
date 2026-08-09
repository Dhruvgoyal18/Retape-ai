from __future__ import annotations

import json
from datetime import date

import pytest

from feasibility.engine import (
    build_staircase,
    cadence_dates_through_horizon,
    evaluate_offer,
    position_floor,
    _balance_timeline,
    _find_minimum,
)
from feasibility.models import (
    Client,
    CreditorRules,
    LedgerEntry,
    Offer,
    load_case,
    load_offer,
    monthly_payment_dates,
    round_half_up,
)

def test_round_half_up_rounds_positive_half_away_from_zero():
    assert round_half_up(2.5) == 3


def test_round_half_up_rounds_negative_half_away_from_zero():
    assert round_half_up(-2.5) == -3


def test_round_half_up_avoids_binary_float_drift():
    assert round_half_up(2.675) == 3

def test_load_offer_accepts_new_field_name(tmp_path):
    p = tmp_path / "offer.json"
    p.write_text(json.dumps({
        "creditor": "X", "creditor_balance_cents": 100000,
        "original_balance_cents": 100000, "settlement_pct": 0.5,
    }))
    assert load_offer(p).creditor_balance_cents == 100000


def test_load_offer_accepts_legacy_field_name(tmp_path):
    p = tmp_path / "offer.json"
    p.write_text(json.dumps({
        "creditor": "X", "current_balance_cents": 100000,
        "original_balance_cents": 100000, "settlement_pct": 0.5,
    }))
    assert load_offer(p).creditor_balance_cents == 100000

def test_cadence_date_exactly_on_horizon_is_included():
    dates = cadence_dates_through_horizon(date(2026, 1, 31), date(2026, 3, 31))
    assert dates == [date(2026, 1, 31), date(2026, 2, 28), date(2026, 3, 31)]


def test_cadence_date_one_day_past_horizon_is_excluded():
    dates = cadence_dates_through_horizon(date(2026, 1, 31), date(2026, 3, 30))
    assert dates == [date(2026, 1, 31), date(2026, 2, 28)]

def test_staircase_rejects_k_when_tiers_exceed_max_segments():
    rules = CreditorRules(
        max_terms=5, max_payments=5, min_payment_cents=1000, max_token_pays=5,
        min_payment_tiers=[(3, 5000)], even_pays=False, is_ballooning_allowed=False,
        max_segments=1, bank_fee_cents=0, program_fee_pct=0.0,
    )
    # k=3 spans two tier groups (positions 1-2 at the base floor, position 3+
    # at the $50 tier) -- two distinct levels, but max_segments only allows 1.
    assert build_staircase(3, 6000, rules) is None
    # k=2 never touches the tier, so it stays a single level and succeeds.
    assert build_staircase(2, 6000, rules) == [3000, 3000]


def test_evaluate_offer_respects_max_segments_cap_end_to_end():
    client = Client(
        draft_amount_cents=100000, draft_day=1,
        first_draft_date=date(2026, 1, 1), last_draft_date=date(2026, 5, 1),
        as_of_date=date(2025, 12, 31), current_balance_cents=0,
        ledger=[LedgerEntry(date(2026, m, 1), 100000, "credit") for m in range(1, 6)],
    )
    offer = Offer(creditor="X", creditor_balance_cents=12000, original_balance_cents=12000,
                  settlement_pct=0.5, first_payment_date=date(2026, 1, 31))
    rules = CreditorRules(
        max_terms=5, max_payments=5, min_payment_cents=1000, max_token_pays=5,
        min_payment_tiers=[(3, 5000)], even_pays=False, is_ballooning_allowed=False,
        max_segments=1, bank_fee_cents=0, program_fee_pct=0.0,
    )
    result = evaluate_offer(client, offer, rules)
    assert result.feasible is True
    assert result.pay_shape_used == "staircase"
    payments = [r.creditor_payment_cents for r in result.schedule if r.creditor_payment_cents > 0]
    assert len(set(payments)) <= rules.max_segments

def test_fee_spills_onto_fee_only_date_after_last_creditor_payment():
    client = Client(
        draft_amount_cents=10000, draft_day=1,
        first_draft_date=date(2026, 1, 1), last_draft_date=date(2026, 6, 1),
        as_of_date=date(2025, 12, 31), current_balance_cents=0,
        ledger=[LedgerEntry(date(2026, m, 1), 10000, "credit") for m in range(1, 7)],
    )
    offer = Offer(creditor="X", creditor_balance_cents=10000, original_balance_cents=100000,
                  settlement_pct=0.5, first_payment_date=date(2026, 1, 31))
    rules = CreditorRules(
        max_terms=2, max_payments=2, min_payment_cents=2500, max_token_pays=2,
        min_payment_tiers=[], even_pays=False, is_ballooning_allowed=False,
        max_segments=1, bank_fee_cents=0, program_fee_pct=0.25,
    )
    result = evaluate_offer(client, offer, rules)
    assert result.feasible is True
    fee_only_rows = [r for r in result.schedule if r.creditor_payment_cents == 0 and r.program_fee_cents > 0]
    assert len(fee_only_rows) >= 1
    assert all(r.balance_cents >= 0 for r in result.schedule)

def test_structural_infeasibility_reported_independent_of_funding():
    client = Client(
        draft_amount_cents=10000, draft_day=1,
        first_draft_date=date(2026, 1, 1), last_draft_date=date(2026, 3, 1),
        as_of_date=date(2025, 12, 31), current_balance_cents=0,
        ledger=[LedgerEntry(date(2026, m, 1), 10000, "credit") for m in range(1, 4)],
    )
    offer = Offer(creditor="X", creditor_balance_cents=10000, original_balance_cents=10000,
                  settlement_pct=0.5, first_payment_date=date(2026, 1, 31))
    rules = CreditorRules(
        max_terms=3, max_payments=3, min_payment_cents=10_000_000, max_token_pays=3,
        min_payment_tiers=[], even_pays=False, is_ballooning_allowed=False,
        max_segments=1, bank_fee_cents=0, program_fee_pct=0.0,
    )
    result = evaluate_offer(client, offer, rules)
    assert result.feasible is False
    af = result.additional_funds
    assert af.lump_sum.within_guardrail is False
    assert af.monthly_increment.within_guardrail is False
    assert "independent" in af.lump_sum.reason.lower() or "cannot fix" in af.lump_sum.reason.lower()

def test_guardrails_fail_when_required_funding_is_too_large():
    client = Client(
        draft_amount_cents=100, draft_day=1,
        first_draft_date=date(2026, 1, 1), last_draft_date=date(2026, 3, 1),
        as_of_date=date(2025, 12, 31), current_balance_cents=0,
        ledger=[LedgerEntry(date(2026, m, 1), 100, "credit") for m in range(1, 4)],
    )
    offer = Offer(creditor="X", creditor_balance_cents=200000, original_balance_cents=200000,
                  settlement_pct=0.5, first_payment_date=date(2026, 1, 31))
    rules = CreditorRules(
        max_terms=1, max_payments=1, min_payment_cents=100, max_token_pays=1,
        min_payment_tiers=[], even_pays=False, is_ballooning_allowed=False,
        max_segments=1, bank_fee_cents=0, program_fee_pct=0.0,
    )
    result = evaluate_offer(client, offer, rules)
    assert result.feasible is False
    af = result.additional_funds
    assert af.lump_sum.within_guardrail is False
    assert af.lump_sum.reason != ""
    assert af.monthly_increment.within_guardrail is False
    assert af.monthly_increment.reason != ""

def test_same_day_ordering_applies_credit_before_debit():
    d = date(2026, 1, 15)
    timeline = _balance_timeline(0, [(d, 50, False), (d, 100, True)])
    assert timeline == [(d, 50)]

def test_no_fee_before_first_creditor_payment_date():
    client, offer, rules = load_case("cases/case1_feasible_even")
    result = evaluate_offer(client, offer, rules)
    assert result.feasible is True
    assert result.schedule[0].date == offer.first_payment_date
    assert all(row.date >= offer.first_payment_date for row in result.schedule)

def test_monthly_payment_dates_eom_includes_feb_29_in_leap_year():
    dates = monthly_payment_dates(date(2028, 1, 31), 3)
    assert dates == [date(2028, 1, 31), date(2028, 2, 29), date(2028, 3, 31)]


def test_monthly_payment_dates_mid_month_clamps_to_feb_29_in_leap_year():
    dates = monthly_payment_dates(date(2028, 1, 30), 3)
    assert dates == [date(2028, 1, 30), date(2028, 2, 29), date(2028, 3, 30)]


def test_monthly_payment_dates_mid_month_clamps_to_feb_28_in_non_leap_year():
    dates = monthly_payment_dates(date(2029, 1, 30), 3)
    assert dates == [date(2029, 1, 30), date(2029, 2, 28), date(2029, 3, 30)]


def test_cadence_dates_through_horizon_leap_year():
    dates = cadence_dates_through_horizon(date(2028, 1, 31), date(2028, 4, 1))
    assert dates == [date(2028, 1, 31), date(2028, 2, 29), date(2028, 3, 31)]


def test_evaluate_offer_leap_year_cadence_includes_feb_29():
    client = Client(
        draft_amount_cents=10000, draft_day=1,
        first_draft_date=date(2028, 1, 1), last_draft_date=date(2028, 4, 1),
        as_of_date=date(2027, 12, 31), current_balance_cents=0,
        ledger=[LedgerEntry(date(2028, m, 1), 10000, "credit") for m in range(1, 5)],
    )
    offer = Offer(creditor="LeapCo", creditor_balance_cents=45000, original_balance_cents=15000,
                  settlement_pct=0.5, first_payment_date=date(2028, 1, 31))
    rules = CreditorRules(
        max_terms=3, max_payments=3, min_payment_cents=2500, max_token_pays=3,
        min_payment_tiers=[], even_pays=True, is_ballooning_allowed=False,
        max_segments=1, bank_fee_cents=0, program_fee_pct=0.1,
    )
    result = evaluate_offer(client, offer, rules)
    assert result.feasible is True

    dates = [row.date for row in result.schedule]
    assert date(2028, 2, 29) in dates
    assert date(2028, 2, 28) not in dates

    payments = [row.creditor_payment_cents for row in result.schedule if row.creditor_payment_cents > 0]
    assert payments == [7500, 7500, 7500]
    assert all(row.balance_cents >= 0 for row in result.schedule)

def test_position_floor_token_pay_boundary_with_no_tiers_present():
    rules = CreditorRules(
        max_terms=5, max_payments=5, min_payment_cents=2500, max_token_pays=3,
        min_payment_tiers=[], even_pays=False, is_ballooning_allowed=False,
        max_segments=2, bank_fee_cents=0, program_fee_pct=0.0,
    )
    assert [position_floor(i, rules) for i in range(1, 6)] == [2500, 2500, 2500, 2501, 2501]

def test_staircase_room_for_new_level_uses_each_positions_own_floor():
    rules = CreditorRules(
        max_terms=4, max_payments=4, min_payment_cents=2500, max_token_pays=3,
        min_payment_tiers=[], even_pays=False, is_ballooning_allowed=False,
        max_segments=2, bank_fee_cents=0, program_fee_pct=0.0,
    )
    assert build_staircase(4, 20000, rules) == [2500, 2500, 2500, 12500]


def test_staircase_does_not_falsely_reject_a_tight_but_valid_schedule():
    rules = CreditorRules(
        max_terms=4, max_payments=4, min_payment_cents=2500, max_token_pays=3,
        min_payment_tiers=[], even_pays=False, is_ballooning_allowed=False,
        max_segments=2, bank_fee_cents=0, program_fee_pct=0.0,
    )
    assert build_staircase(4, 10002, rules) == [2500, 2500, 2500, 2502]


def test_zero_offer_total_is_structurally_infeasible():
    client = Client(
        draft_amount_cents=10000, draft_day=1,
        first_draft_date=date(2026, 1, 1), last_draft_date=date(2026, 3, 1),
        as_of_date=date(2025, 12, 31), current_balance_cents=0,
        ledger=[LedgerEntry(date(2026, m, 1), 10000, "credit") for m in range(1, 4)],
    )
    offer = Offer(creditor="ForgivenCo", creditor_balance_cents=50000, original_balance_cents=50000,
                  settlement_pct=0.0, first_payment_date=date(2026, 1, 31))
    rules = CreditorRules(
        max_terms=3, max_payments=3, min_payment_cents=2500, max_token_pays=3,
        min_payment_tiers=[], even_pays=False, is_ballooning_allowed=False,
        max_segments=1, bank_fee_cents=0, program_fee_pct=0.1,
    )
    result = evaluate_offer(client, offer, rules)
    assert result.feasible is False
    af = result.additional_funds
    assert af.lump_sum.within_guardrail is False
    assert af.monthly_increment.within_guardrail is False

def test_fixed_debit_on_same_date_as_creditor_payment():
    client = Client(
        draft_amount_cents=20000, draft_day=1,
        first_draft_date=date(2026, 1, 1), last_draft_date=date(2026, 4, 1),
        as_of_date=date(2025, 12, 31), current_balance_cents=0,
        ledger=[LedgerEntry(date(2026, m, 1), 20000, "credit") for m in range(1, 5)]
        + [LedgerEntry(date(2026, 1, 31), 5000, "debit")],
    )
    offer = Offer(creditor="X", creditor_balance_cents=20000, original_balance_cents=20000,
                  settlement_pct=0.5, first_payment_date=date(2026, 1, 31))
    rules = CreditorRules(
        max_terms=1, max_payments=1, min_payment_cents=2500, max_token_pays=1,
        min_payment_tiers=[], even_pays=False, is_ballooning_allowed=False,
        max_segments=1, bank_fee_cents=1000, program_fee_pct=0.0,
    )
    result = evaluate_offer(client, offer, rules)
    assert result.feasible is True
    row = result.schedule[0]
    assert row.balance_cents == 4000

def test_zero_dollar_payment_position_still_produces_a_row_for_its_bank_fee():
    client = Client(
        draft_amount_cents=10000, draft_day=1,
        first_draft_date=date(2026, 1, 1), last_draft_date=date(2026, 2, 1),
        as_of_date=date(2025, 12, 31), current_balance_cents=0,
        ledger=[LedgerEntry(date(2026, m, 1), 10000, "credit") for m in range(1, 3)],
    )
    offer = Offer(creditor="X", creditor_balance_cents=10000, original_balance_cents=10000,
                  settlement_pct=0.0, first_payment_date=date(2026, 1, 31))
    rules = CreditorRules(
        max_terms=1, max_payments=1, min_payment_cents=0, max_token_pays=1,
        min_payment_tiers=[], even_pays=False, is_ballooning_allowed=False,
        max_segments=1, bank_fee_cents=1000, program_fee_pct=0.0,
    )
    result = evaluate_offer(client, offer, rules)
    assert result.feasible is True
    assert len(result.schedule) == 1
    row = result.schedule[0]
    assert row.creditor_payment_cents == 0
    assert row.bank_fee_cents == 1000
    assert row.balance_cents == 9000


def test_find_minimum_succeeds_right_at_a_high_but_proven_feasible_value():
    threshold = 10**14
    assert _find_minimum(lambda x: x >= threshold) == threshold

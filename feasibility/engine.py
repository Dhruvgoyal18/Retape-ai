"""Settlement feasibility & fee engine.

See ASSIGNMENT.md for the full specification and README.md for the design
rationale (approach, shape interpretation, and documented assumptions). This
module is organized as a sequence of small pure functions plus one
orchestrator (``solve``), in roughly the order they're used:

  round_half_up / pct_of        -- money rounding (imported from models)
  cadence_dates_through_horizon -- the monthly cadence dates available to us
  position_floor / tier_floor   -- per-position minimum payment
  build_even / build_balloon / build_staircase
                                 -- construct a candidate payment shape for a
                                    given k (payment count)
  validate_schedule              -- generic safety-net checker for any shape
  _balance_timeline / _balance_at
                                 -- shared ledger-simulation primitives
  fee_timing                     -- front-loaded fee collection (safe frontier)
  simulate_full                  -- final correctness oracle + ScheduleRow build
  solve                          -- brute-forces k, picks the best schedule
  compute_additional_funds       -- Part 2: minimum lump sum / monthly increment
  evaluate_offer                 -- the required entry point
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Literal

from feasibility.models import (
    Client,
    CreditorRules,
    LedgerEntry,
    Offer,
    default_first_payment_date,
    monthly_payment_dates,
    offer_total_cents,
    pct_of,
    program_fee_cents,
    round_half_up,
)

PayShape = Literal["even", "staircase", "balloon"]


@dataclass
class ScheduleRow:
    date: date
    creditor_payment_cents: int
    program_fee_cents: int
    bank_fee_cents: int
    balance_cents: int


@dataclass
class FundsOption:
    amount_cents: int
    within_guardrail: bool
    reason: str
    # lump-sum only:
    date: date | None = None
    # monthly-increment only:
    num_drafts: int | None = None


@dataclass
class AdditionalFunds:
    lump_sum: FundsOption
    monthly_increment: FundsOption


@dataclass
class Result:
    feasible: bool
    # One of "even", "staircase", or "balloon" — the shape your solution produced
    # (driven by the creditor flags). None when infeasible.
    pay_shape_used: str | None = None
    schedule: list[ScheduleRow] | None = None
    additional_funds: AdditionalFunds | None = None

    def to_dict(self) -> dict:
        out: dict = {"feasible": self.feasible, "pay_shape_used": self.pay_shape_used}
        out["schedule"] = (
            [
                {
                    "date": r.date.isoformat(),
                    "creditor_payment_cents": r.creditor_payment_cents,
                    "program_fee_cents": r.program_fee_cents,
                    "bank_fee_cents": r.bank_fee_cents,
                    "balance_cents": r.balance_cents,
                }
                for r in self.schedule
            ]
            if self.schedule is not None
            else None
        )
        if self.additional_funds is None:
            out["additional_funds"] = None
        else:
            def opt(o: FundsOption) -> dict:
                d = {
                    "amount_cents": o.amount_cents,
                    "within_guardrail": o.within_guardrail,
                    "reason": o.reason,
                }
                if o.date is not None:
                    d["date"] = o.date.isoformat()
                if o.num_drafts is not None:
                    d["num_drafts"] = o.num_drafts
                return d

            out["additional_funds"] = {
                "lump_sum": opt(self.additional_funds.lump_sum),
                "monthly_increment": opt(self.additional_funds.monthly_increment),
            }
        return out


# ---------------------------------------------------------------------------
# Stage 1: cadence dates
# ---------------------------------------------------------------------------

def cadence_dates_through_horizon(fp_date: date, horizon: date) -> list[date]:
    """All monthly cadence dates from ``fp_date`` through ``horizon`` inclusive.

    This is independent of ``max_payments``/``max_terms`` — it's simply how many
    monthly cadence slots exist before the horizon. Creditor payments occupy
    only the first ``k`` of these; any remaining slots are legal "fee-only"
    dates (ASSIGNMENT.md glossary).
    """
    if fp_date > horizon:
        return []
    months_between = (horizon.year - fp_date.year) * 12 + (horizon.month - fp_date.month)
    count = months_between + 3  # generous buffer against day-of-month clamping
    return [d for d in monthly_payment_dates(fp_date, count) if d <= horizon]


# ---------------------------------------------------------------------------
# Stage 2: per-position floors and shape construction
# ---------------------------------------------------------------------------

def tier_floor(position: int, tiers: list[tuple[int, int]]) -> int:
    applicable = [min_cents for (frm, min_cents) in tiers if frm <= position]
    return max(applicable) if applicable else 0


def position_floor(position: int, rules: CreditorRules) -> int:
    """Floor for a 1-based payment position: max of base/token-rule/tier.

    Non-decreasing in ``position`` by construction (both the token-pay step and
    every tier step only ever raise the floor). Because payments must also be
    non-decreasing, any payment sitting exactly at ``min_payment_cents`` is
    necessarily a prefix of the sequence — so "position <= max_token_pays may
    sit at the base minimum, position > max_token_pays must strictly exceed it"
    is both necessary and sufficient; it never needs to be treated as a forced
    payment-level boundary on its own (see ``build_staircase``).
    """
    base = rules.min_payment_cents
    if position > rules.max_token_pays:
        base += 1
    return max(base, tier_floor(position, rules.min_payment_tiers))


def build_even(k: int, offer_total: int) -> list[int]:
    base, rem = divmod(offer_total, k)
    payments = [base] * k
    for i in range(k - rem, k):
        payments[i] += 1
    return payments


def build_balloon(k: int, offer_total: int, rules: CreditorRules) -> list[int] | None:
    """Positions 1..k-1 at their own floor; the final payment absorbs the rest.

    This is provably optimal for the front-loading objective: the zero-fee
    balance at the *last* cadence position is invariant to how 1..k-1 are
    shaped (it only depends on total credits through that point minus the
    fixed offer_total), so minimizing every earlier position down to its floor
    strictly maximizes fee-collection headroom at every earlier date without
    costing anything at the last date.
    """
    floors = [position_floor(i, rules) for i in range(1, k)]
    total_floor = sum(floors)
    if total_floor > offer_total:
        return None
    return floors + [offer_total - total_floor]


def _tier_groups(k: int, tiers: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Contiguous 1-based (start, end) position ranges sharing one tier floor."""
    vals = [tier_floor(i, tiers) for i in range(1, k + 1)]
    groups: list[tuple[int, int]] = []
    start = 1
    for i in range(2, k + 1):
        if vals[i - 1] != vals[i - 2]:
            groups.append((start, i - 1))
            start = i
    groups.append((start, k))
    return groups


def build_staircase(k: int, offer_total: int, rules: CreditorRules) -> list[int] | None:
    """Group positions by tier boundary; front-load within the max_segments cap.

    Non-last groups are each pinned to one uniform level — the strictest
    per-position floor within that group (which, since floors are
    non-decreasing, is just the floor at the group's last position). This
    deliberately treats a token-pay boundary that happens to fall *inside* a
    tier group as merged into that group's single level rather than as its own
    extra segment: it costs a little front-loading in that specific case, but
    guarantees the distinct-level count never silently exceeds max_segments
    (documented as a known simplification in the README).

    The last group either gets one new top-of-stack level (if segment budget
    allows — the whole remainder dumped onto the final position alone, which
    is optimal per the same argument as ``build_balloon``), or, if the budget
    is exhausted, must itself become a single uniform level absorbing the
    remainder — and if the remainder doesn't divide evenly across that group,
    this k has no valid staircase construction and is rejected outright rather
    than forcing an uneven split that would exceed max_segments.
    """
    groups = _tier_groups(k, rules.min_payment_tiers)
    m = len(groups)
    if m > rules.max_segments:
        return None

    payments = [0] * k
    for (s, e) in groups[:-1]:
        level = position_floor(e, rules)
        for pos in range(s, e + 1):
            payments[pos - 1] = level

    last_s, last_e = groups[-1]
    prior_sum = sum(payments[: last_s - 1])
    remaining = offer_total - prior_sum
    group_size = last_e - last_s + 1

    if m < rules.max_segments:
        # Room for one more level: positions last_s..last_e-1 are pinned to
        # their own strictest floor -- position_floor(last_e - 1), the floor
        # at the last of *those* positions (floors are non-decreasing, so
        # that's the max within the sub-range) -- and the final position
        # (last_e == k) is peeled off as its own brand-new top level,
        # absorbing everything else. Using position_floor(last_e) here
        # instead (position k's own floor) would needlessly over-pay every
        # position before it whenever k's floor is stricter than k-1's, and
        # can even wrongly reject an otherwise-valid k (a real bug this fixes:
        # e.g. min_payment_cents=2500, max_token_pays=3, k=4, offer_total=10002
        # is legitimately satisfiable as [2500, 2500, 2500, 2502] but the old
        # code rejected it by testing against position 4's floor of 2501).
        sub_floor = position_floor(last_e - 1, rules) if group_size > 1 else 0
        final_floor = position_floor(last_e, rules)
        final_payment = remaining - sub_floor * (group_size - 1)
        if final_payment < final_floor:
            return None
        for pos in range(last_s, last_e):
            payments[pos - 1] = sub_floor
        payments[last_e - 1] = final_payment
    else:
        # No room for a new level: the whole group must be one uniform value.
        last_floor = position_floor(last_e, rules)
        if remaining < last_floor * group_size or remaining % group_size != 0:
            return None
        level = remaining // group_size
        if level < last_floor:
            return None
        for pos in range(last_s, last_e + 1):
            payments[pos - 1] = level

    return payments


def validate_schedule(payments: list[int], rules: CreditorRules, offer_total: int, shape: PayShape) -> bool:
    """Generic safety-net check, independent of how ``payments`` was built."""
    k = len(payments)
    if k == 0 or sum(payments) != offer_total:
        return False
    for i in range(1, k):
        if payments[i] < payments[i - 1]:
            return False
    for idx, p in enumerate(payments, start=1):
        if p < position_floor(idx, rules):
            return False
    if shape == "staircase" and len(set(payments)) > rules.max_segments:
        return False
    return True


# ---------------------------------------------------------------------------
# Shared simulation primitives
# ---------------------------------------------------------------------------

# An event is (date, magnitude_cents, is_credit). Same-day ordering applies
# all credits before all debits; ordering among same-side events on the same
# date is irrelevant to the resulting balance.
_Event = tuple[date, int, bool]


def _sorted_events(events: list[_Event]) -> list[_Event]:
    return sorted(events, key=lambda e: (e[0], 0 if e[2] else 1))


def _balance_timeline(start_balance: int, events: list[_Event]) -> list[tuple[date, int]]:
    """Balance after each distinct event date, in chronological order."""
    balance = start_balance
    timeline: list[tuple[date, int]] = []
    for d, amount, is_credit in _sorted_events(events):
        balance += amount if is_credit else -amount
        if timeline and timeline[-1][0] == d:
            timeline[-1] = (d, balance)
        else:
            timeline.append((d, balance))
    return timeline


def _balance_at(timeline: list[tuple[date, int]], start_balance: int, query_date: date) -> int:
    """Balance as of ``query_date`` (carrying forward from the latest prior event)."""
    dates = [d for d, _ in timeline]
    idx = bisect.bisect_right(dates, query_date) - 1
    return timeline[idx][1] if idx >= 0 else start_balance


def _future_ledger_events(client: Client) -> list[_Event]:
    return [(e.date, e.amount_cents, e.type == "credit") for e in client.ledger if e.date > client.as_of_date]


# ---------------------------------------------------------------------------
# Stage 3: fee timing (front-loaded, safe-frontier)
# ---------------------------------------------------------------------------

def fee_timing(
    client: Client,
    all_dates: list[date],
    k: int,
    payments: list[int],
    bank_fee_cents: int,
    total_fee: int,
) -> tuple[dict[date, int], list[int]] | None:
    """Front-load program-fee collection without ever risking a future date.

    ``safe_frontier(t)`` — the most fee collectible by date t without
    endangering any later date — is the suffix-min of the zero-fee balance
    over *every* future event date (not just cadence dates): a committed
    ledger debit landing between two cadence dates, or even after the
    horizon, still has to be protected, since fee collected early lowers the
    balance permanently from then on. Returns ``None`` if the fee can't be
    fully collected by the horizon.
    """
    zero_fee_events = _future_ledger_events(client)
    for i in range(k):
        debit = payments[i] + bank_fee_cents
        if debit:
            zero_fee_events.append((all_dates[i], debit, False))

    timeline = _balance_timeline(client.current_balance_cents, zero_fee_events)
    checkpoints = sorted(set(all_dates) | {d for d, _ in timeline})
    slack_at = {d: _balance_at(timeline, client.current_balance_cents, d) for d in checkpoints}

    suffix_min: dict[date, int] = {}
    running_min: int | None = None
    for d in reversed(checkpoints):
        v = slack_at[d]
        running_min = v if running_min is None else min(running_min, v)
        suffix_min[d] = running_min

    per_date_fee: dict[date, int] = {}
    cumulative_vector: list[int] = []
    collected = 0
    for t in all_dates:
        available = max(0, suffix_min[t] - collected)
        take = max(0, min(total_fee - collected, available))
        if take:
            per_date_fee[t] = take
        collected += take
        cumulative_vector.append(collected)

    if collected < total_fee:
        return None
    return per_date_fee, cumulative_vector


# ---------------------------------------------------------------------------
# Stage 4: full simulation (correctness oracle + ScheduleRow construction)
# ---------------------------------------------------------------------------

def simulate_full(
    client: Client,
    all_dates: list[date],
    k: int,
    payments: list[int],
    per_date_fee: dict[date, int],
    bank_fee_cents: int,
) -> list[ScheduleRow] | None:
    events = _future_ledger_events(client)
    row_specs: list[tuple[date, int, int, int]] = []
    for i, d in enumerate(all_dates):
        cp = payments[i] if i < k else 0
        bf = bank_fee_cents if i < k else 0
        fee = per_date_fee.get(d, 0)
        # Every designated creditor-payment position (i < k) gets a row, even
        # in the degenerate case where min_payment_cents=0 lets a position's
        # payment itself be $0 -- a bank fee can still be debited that day
        # (constraint 5: bank_fee applies to every date carrying a creditor
        # payment, regardless of that payment's size), and dropping the row
        # would silently hide a real balance change from the output.
        if i < k or fee:
            row_specs.append((d, cp, fee, bf))
        debit = cp + fee + bf
        if debit:
            events.append((d, debit, False))

    timeline = _balance_timeline(client.current_balance_cents, events)
    if any(bal < 0 for _, bal in timeline):
        return None

    return [
        ScheduleRow(
            date=d,
            creditor_payment_cents=cp,
            program_fee_cents=fee,
            bank_fee_cents=bf,
            balance_cents=_balance_at(timeline, client.current_balance_cents, d),
        )
        for (d, cp, fee, bf) in row_specs
    ]


# ---------------------------------------------------------------------------
# Stage 1-4 orchestrator
# ---------------------------------------------------------------------------

def solve(client: Client, offer: Offer, rules: CreditorRules) -> tuple[list[ScheduleRow], PayShape] | None:
    offer_total = offer_total_cents(offer)
    total_fee = program_fee_cents(offer, rules)
    horizon = client.last_draft_date
    fp_date = offer.first_payment_date or default_first_payment_date(client)
    all_dates = cadence_dates_through_horizon(fp_date, horizon)
    if not all_dates:
        return None

    k_max = min(rules.max_payments, rules.max_terms, len(all_dates))

    if rules.even_pays:
        shape_name: PayShape = "even"
    elif rules.is_ballooning_allowed:
        shape_name = "balloon"
    else:
        shape_name = "staircase"

    best: tuple[list[int], int, list[ScheduleRow]] | None = None
    best_key: tuple[tuple[int, ...], int] | None = None

    for k in range(1, k_max + 1):
        if shape_name == "even":
            payments = build_even(k, offer_total)
        elif shape_name == "balloon":
            payments = build_balloon(k, offer_total, rules)
        else:
            payments = build_staircase(k, offer_total, rules)

        if payments is None or not validate_schedule(payments, rules, offer_total, shape_name):
            continue

        fee_result = fee_timing(client, all_dates, k, payments, rules.bank_fee_cents, total_fee)
        if fee_result is None:
            continue
        per_date_fee, cumulative_vector = fee_result

        rows = simulate_full(client, all_dates, k, payments, per_date_fee, rules.bank_fee_cents)
        if rows is None:
            continue

        # Prefer the most front-loaded cumulative-fee vector; tie-break on
        # smallest k (fewer bank fees) via -k.
        key = (tuple(cumulative_vector), -k)
        if best_key is None or key > best_key:
            best_key = key
            best = (payments, k, rows)

    if best is None:
        return None
    _, _, rows = best
    return rows, shape_name


# ---------------------------------------------------------------------------
# Stage 5: Part 2 — minimum additional funds
# ---------------------------------------------------------------------------

_UNBOUNDED_CASH_CENTS = 10**15
# Deliberately far above _UNBOUNDED_CASH_CENTS (not equal to it): the
# structural precheck below proves a feasible amount exists at or below
# _UNBOUNDED_CASH_CENTS, so the exponential-doubling search in
# _find_minimum must be able to overshoot past that proven-feasible value
# and still land on a hi it can test at or beyond it. If the two constants
# were equal, a case needing an amount just under _UNBOUNDED_CASH_CENTS
# could have the doubling step jump past the cap and abort with None
# without ever re-testing the value already known to work.
_SEARCH_CAP_CENTS = 10**18


def _with_lump(client: Client, amount: int, on_date: date) -> Client:
    entry = LedgerEntry(date=on_date, amount_cents=amount, type="credit")
    return replace(client, ledger=[*client.ledger, entry])


def _with_increment(client: Client, amount: int) -> Client:
    new_ledger = [
        replace(e, amount_cents=e.amount_cents + amount) if e.type == "credit" and e.date > client.as_of_date else e
        for e in client.ledger
    ]
    return replace(client, ledger=new_ledger)


def _find_minimum(oracle) -> int | None:
    """Minimal non-negative integer x with oracle(x) True, given oracle is
    monotonic non-decreasing in x. None if not found even at the search cap.
    """
    if oracle(0):
        return 0
    lo, hi = 0, 1
    while not oracle(hi):
        lo = hi
        hi *= 2
        if hi > _SEARCH_CAP_CENTS:
            return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if oracle(mid):
            hi = mid
        else:
            lo = mid
    return hi


def compute_additional_funds(client: Client, offer: Offer, rules: CreditorRules) -> AdditionalFunds:
    lump_date = client.as_of_date + timedelta(days=1)
    offer_total = offer_total_cents(offer)

    structurally_ok = solve(_with_lump(client, _UNBOUNDED_CASH_CENTS, lump_date), offer, rules) is not None
    if not structurally_ok:
        reason = (
            "No valid creditor-payment schedule exists for this offer under the "
            "creditor's rules, independent of any additional funding (e.g. the "
            "minimum required payment already exceeds the offer total, or no "
            "cadence dates fall before the horizon) — adding money cannot fix this."
        )
        return AdditionalFunds(
            lump_sum=FundsOption(amount_cents=0, within_guardrail=False, reason=reason, date=None),
            monthly_increment=FundsOption(amount_cents=0, within_guardrail=False, reason=reason, num_drafts=0),
        )

    def lump_oracle(amount: int) -> bool:
        return solve(_with_lump(client, amount, lump_date), offer, rules) is not None

    def increment_oracle(amount: int) -> bool:
        return solve(_with_increment(client, amount), offer, rules) is not None

    lump_min = _find_minimum(lump_oracle) or 0
    increment_min = _find_minimum(increment_oracle) or 0

    n_future_drafts = sum(1 for e in client.ledger if e.type == "credit" and e.date > client.as_of_date)

    lump_cap = round_half_up(pct_of(0.65, offer_total))
    lump_within = lump_min <= lump_cap
    lump_reason = (
        ""
        if lump_within
        else f"Lump sum of {lump_min} cents exceeds the guardrail cap of 65% of the offer total ({lump_cap} cents)."
    )

    increment_cap = max(10_000, round_half_up(pct_of(0.40, client.draft_amount_cents)))
    increment_within = increment_min <= increment_cap
    increment_reason = (
        ""
        if increment_within
        else (
            f"Monthly increment of {increment_min} cents exceeds the guardrail cap of "
            f"max($100.00, 40% of the draft amount) = {increment_cap} cents."
        )
    )

    return AdditionalFunds(
        lump_sum=FundsOption(amount_cents=lump_min, within_guardrail=lump_within, reason=lump_reason, date=lump_date),
        monthly_increment=FundsOption(
            amount_cents=increment_min,
            within_guardrail=increment_within,
            reason=increment_reason,
            num_drafts=n_future_drafts,
        ),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def evaluate_offer(client: Client, offer: Offer, rules: CreditorRules) -> Result:
    """Evaluate a single offer. See ASSIGNMENT.md for the full specification.

    Return a Result with feasible=True and a schedule when the offer fits, or
    feasible=False with additional_funds (minimum lump sum AND minimum monthly
    increment) when it does not.
    """
    solved = solve(client, offer, rules)
    if solved is not None:
        rows, shape_name = solved
        return Result(feasible=True, pay_shape_used=shape_name, schedule=rows, additional_funds=None)

    additional_funds = compute_additional_funds(client, offer, rules)
    return Result(feasible=False, pay_shape_used=None, schedule=None, additional_funds=additional_funds)

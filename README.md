# Settlement Feasibility & Fee Engine — Take-home

Welcome, and thanks for taking the time. The full problem is in
[`ASSIGNMENT.md`](./ASSIGNMENT.md). This README is just orientation.

## The task in one line

Given a client's escrow account, a settlement offer, and a creditor's rules,
decide whether the offer is affordable (and schedule it, collecting our fee as
early as allowed) or — if not — compute the minimum extra funding needed.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Layout

```
hiring_takehome/
├── ASSIGNMENT.md            # full specification — read this
├── feasibility/
│   ├── models.py            # data models, JSON loaders, date/EOM helpers (provided)
│   └── engine.py            # >>> implement evaluate_offer here <<< (+ Result shape)
├── cases/                   # four example cases (client.json / offer.json / creditor_rules.json)
│   ├── case1_feasible_even
│   ├── case2_infeasible_minima
│   ├── case3_balloon
│   └── case4_tiers
├── tests/
│   ├── test_smoke.py        # scaffolding sanity tests (pass out of the box)
│   └── test_cases.py        # example expectations — make these pass, then add your own
├── run.py                   # python run.py cases/<case>
└── requirements.txt
```

## Run

```bash
# evaluate a single case (prints the Result as JSON)
python run.py cases/case1_feasible_even

# tests
pytest -q
```

Out of the box, `tests/test_smoke.py` passes and `tests/test_cases.py` fails —
the latter is your target. Go beyond those four cases with your own tests.

## What to submit

Your implementation, your tests, and a short README section describing:
- your approach and the alternatives you considered,
- **your interpretation of the payment shapes** (even / staircase / balloon — we
  left these loosely defined on purpose),
- assumptions you made, and known edge cases / limitations.

Budget ~5–6 hours. Prefer a correct, well-tested core over breadth. When in
doubt, write down your assumption and keep going.

---

## Implementation notes

### Approach

`evaluate_offer` is a brute force over the one small, bounded unknown —
the creditor-payment count `k` (`1..min(max_payments, max_terms, available
cadence dates)`) — combined with a closed-form construction of the payment
*shape* for each `k` and a greedy, provably-safe fee-timing pass. There's no
need for `ortools`/`numpy`; `k` never realistically exceeds a few dozen, so
brute-forcing it and re-simulating the whole ledger for each is fast and easy
to reason about.

For a fixed `k`, `feasibility/engine.py` does, in order:

1. **Shape construction.** `build_even` / `build_balloon` / `build_staircase`
   each produce a candidate list of `k` payments summing exactly to
   `offer_total`, chosen by the creditor's flags (not brute-forced). All three
   share one idea: **push every early payment down to its floor and let the
   back of the schedule absorb the rest.** This is provably optimal for the
   "collect the fee as early as possible" objective — for a fixed `k`, the
   zero-fee balance at the *last* cadence position depends only on total
   credits through that point minus the fixed `offer_total`, which is
   invariant to how the earlier positions are shaped. So minimizing every
   earlier position simultaneously maximizes fee-collection headroom at every
   earlier date, at zero cost to the last date.
2. **Validation.** A generic `validate_schedule` re-checks sum/non-decreasing/
   floor/segment-count regardless of which builder produced the candidate —
   a cheap safety net rather than trusting each builder's algebra blindly.
3. **Fee timing (`fee_timing`).** The program fee is collected greedily using
   a right-to-left suffix-min ("safe frontier"): the most fee collectible by
   date `t` without ever risking a *later* date is `min(zero-fee balance at
   every date from t onward)`. This is computed over every real future event
   date (every committed ledger entry, not just cadence dates, and not capped
   at the horizon — a large committed debit sitting between two cadence
   dates, or even after the horizon, still has to be protected, since fee
   collected early lowers the balance permanently from then on).
4. **Final simulation (`simulate_full`).** A literal replay of the whole
   future ledger plus the new rows, checked for `balance >= 0` everywhere.
   This is a real `if/else` check, not an `assert` — if steps 1–3 ever
   produced something unsafe, this rejects that `k` rather than crashing.

Across all `k` that produce a fully fee-collected, feasible schedule, the
winner is chosen by comparing **cumulative-fee-collected vectors
lexicographically** (prefer whichever collects more, sooner), tie-broken by
smallest `k`. An earlier "pick whichever k finishes collecting fee first"
metric turns out to be too coarse — two `k`s can finish on the same date while
differing in how front-loaded they are along the way — so the full vector
comparison is what's actually implemented.

Part 2 (infeasible case) reuses the exact same `solve()` as its feasibility
oracle: extra cash (a lump sum or a per-draft increment) only relaxes the
non-negativity constraint, never the schedule-structure constraints, so
feasibility is monotonic in the amount of extra money — which is what makes
binary search valid. Before searching, one probe with an effectively
unlimited injected credit checks whether the offer is *structurally*
infeasible (no valid `(k, shape)` schedule exists at all, independent of
cash) — see "Known edge cases" below.

### Payment shape interpretation

- **`even_pays`**: all `k` payments equal, remainder cents (when `offer_total`
  isn't evenly divisible by `k`) placed on the *latest* payments so the
  sequence stays non-decreasing. `k` itself is chosen by the same brute-force
  search as every other shape — "equal" only fixes the shape, not the count.
- **`is_ballooning_allowed`**: positions `1..k-1` sit exactly at their own
  floor; the final position absorbs `offer_total` minus everything already
  paid. See the optimality argument above for why this is the right choice,
  not just a convenient one.
- **staircase (neither flag set)**: positions are grouped by **tier
  boundary** (`min_payment_tiers`), not by the token-pay cutoff — because
  payments and floors are both non-decreasing, any payment sitting exactly at
  `min_payment_cents` is necessarily a contiguous prefix of the sequence, so
  the token-pay rule never needs to force its own segment boundary; it just
  raises the floor within whatever tier-group it falls in. Each non-last
  tier-group is pinned to one uniform level (that group's strictest
  per-position floor). The last group either gets one brand-new top-of-stack
  level holding the entire remainder (if `max_segments` allows one more
  level — this is the balloon-style optimal move again), or, if the segment
  budget is exhausted, must become one uniform level itself; if the remaining
  amount doesn't divide evenly across that group's positions, that `k` has no
  valid staircase construction and is skipped rather than forced.

### Assumptions

1. **Offer balance field rename.** ASSIGNMENT.md §3 renames the offer's
   balance field to `creditor_balance_cents`, but the shipped `Offer`
   dataclass and all four `cases/*/offer.json` fixtures still use
   `current_balance_cents`. The dataclass field is renamed to match the spec,
   but `load_offer` accepts **either** JSON key, so the shipped fixtures work
   untouched.
2. **Rounding.** All `round(...)` computations (`offer_total_cents`,
   `program_fee_cents`, and both Part-2 guardrail thresholds) use an explicit
   round-half-up implementation built on `Decimal`, converting each operand
   via `str()` *before* multiplying (not `float * int` followed by rounding),
   so a genuine `.5`-cent boundary can't be misrounded by binary-float drift.
3. **Fee-eligible dates.** Program fee may be collected on *any* monthly
   cadence date up to the horizon, not just the `k` dates that carry a
   creditor payment — when `max_payments`/`max_terms` caps `k` below the
   number of available cadence dates, the remaining dates are legal
   "fee-only" dates (see the glossary's own definition of that term).
4. **Lump-sum placement.** The earliest usable date for the Part-2 lump sum
   is `as_of_date + 1 day` — the start of the modifiable window.
5. **Schedule rows.** Output includes one row per cadence date carrying a
   nonzero creditor payment and/or nonzero program fee; cadence dates with
   neither are omitted (this matches `tests/test_cases.py`'s own filtering on
   `creditor_payment_cents > 0`, which implies rows can otherwise carry a
   zero payment).
6. **k tie-break.** Among `k`s tied on the full cumulative-fee vector,
   smallest `k` wins (fewest bank fees charged overall).
7. **`offer_total == 0`.** Constraint 1 (`k >= 1`) and constraint 4 (every
   payment clears a normally-positive floor) directly conflict when
   `settlement_pct` is `0` — a required payment can never legally be `$0`.
   Reported as `feasible: false`, specifically as *structurally* infeasible
   (constraint 2 pins the sum to exactly `offer_total`, so no amount of extra
   cash changes what's required) rather than something a lump sum or
   increment could fix. A reasonable alternative reading would treat a fully
   forgiven debt as trivially feasible with an empty schedule; this wasn't
   implemented since `pay_shape_used` has no natural value for "no payments."

### Known edge cases / limitations

- **Structural infeasibility.** If no `(k, shape)` schedule can ever satisfy
  the hard constraints (count/sum/floor/non-decreasing/segments) — e.g. a
  single base `min_payment_cents` already exceeds `offer_total` — no amount
  of extra funding can fix it, since money only relaxes non-negativity, not
  schedule structure. This is detected with one probe (an effectively
  unlimited injected credit) before the Part-2 binary searches run, and
  reported as `within_guardrail: false` on both `FundsOption`s with a reason
  explaining it's independent of funding, rather than the search silently
  running to its cap.
- **Staircase last-segment divisibility.** When the top tier-segment's
  budget is fully used (`max_segments` groups already touched) and the
  remaining amount doesn't divide evenly across that segment's positions,
  that specific `k` is rejected rather than forcing an uneven split (which
  would silently exceed `max_segments`). The outer `k` search tries other
  values instead — different `k` changes the segment's size and can restore
  divisibility — but a pathological rule set could in principle reject every
  `k` for this reason alone even though a cleverer (non-uniform-within-tier)
  construction might have found one. Not solved here; documented as a
  deliberate simplification given the assignment's explicit invitation to
  reason about the shape rather than chase a hidden formula.
- **Token-pay boundary inside a tier group.** A token-pay boundary that falls
  strictly inside a tier group is merged into that group's single uniform
  level rather than treated as its own segment, which can sacrifice a little
  front-loading in that specific case (positions before the boundary get
  bumped up to the group's stricter shared level) in exchange for guaranteed
  `max_segments` compliance.

"""#188: the pre-registered bar for the retrained A2 massing source, and the verdict that reads it.

[#188](https://github.com/danvisai/SDFusion/issues/188) requires a bar stated **before** the run,
citing [#183](https://github.com/danvisai/SDFusion/issues/183) as the precedent to respect: that
ladder's arms 2 and 3 both failed #178's BuildingWorld bar, and the lesson recorded there is to
state the bar in advance rather than discover it afterwards.

## What this bar is, and what it deliberately is not

#188 is a **substitution**, not a new quality claim. #115's amendment says so in as many words:
"#115's *method* ... carries over unchanged to whatever checkpoint replaces it. Only the identity of
the frozen checkpoint changes." The sealed source `vecset_v4_surf` @240k was disqualified for a
**structural** reason -- its `nn.Embedding(3, 512)` raises `IndexError` on every BuildingWorld row
-- and not for a measured quality failure.

So the bar asks: *is the replacement as good a source as the one it replaces?* It does **not** ask
the replacement to beat the footprint envelope, and that restraint is deliberate and disclosed,
because **the frozen source does not beat the envelope either**. Measured on its own legacy
population (`criterion2_full714.json`, n=714, strength 0.5):

| arm | fp_iou | missing | extra | vol_iou |
|---|---|---|---|---|
| blockout (doing nothing) | 1.0000 | 0.0000 | 0.0714 | 0.9334 |
| frozen A2 @240k | 0.9589 | 0.0026 | 0.0876 | 0.8625 |

It wins on `extra` for 11.3% of rows and on `vol_iou` for **0.56%**. A bar demanding the retrain
beat the envelope would be one the sealed source itself fails by a wide margin; imposing it here
would quietly convert #188 from "replace a structurally unusable checkpoint" into "solve the open
quality problem map #1 records as unsolved". Those are different tickets.

## Non-inferiority, normalised by population

The retrained source is scored on BuildingWorld held-out rows and the frozen one was scored on
legacy rows, so raw numbers are not comparable -- the populations differ in difficulty. The bar is
therefore stated as a **ratio to each population's own blockout**: relative to doing nothing on its
own population, the replacement must be at least as good as the frozen source was relative to
doing nothing on its population. That normalisation is what makes a cross-population
non-inferiority claim honest.

## Clause kinds: PASS / GUARD / KILL, which is CONTEXT.md's own vocabulary

`CONTEXT.md` ("The bar itself") already defines the structure, so this uses it rather than
inventing a parallel one: "**PASS** is what the arm must achieve. **GUARD** is what it may not
break on the way (collapse and `vs_input`). **KILL** is a pre-registered clause that answers the
ticket 'no' -- it exists so a disappointing result cannot be re-narrated as partial success."

  * **KILL** -- the structural capability #188 exists to restore. A KILL is not outvoted by the
    clauses that passed (#183's rule, per #172(b)).
  * **GUARD** -- `CONTEXT.md`'s two documented guards, **at their documented values**:
    `vs_input < 0.98` and `collapse_rate <= 0.1582`. Neither is #188's to loosen.
  * **PASS** -- the non-inferiority clauses above, plus footprint adherence.
  * **INFORMATIONAL** -- reported, never scored.

⚠️ **`vs_input` is the clause that makes this bar falsifiable against an INERT model**, and it was
missing from the first version of this registration (caught in review, before any training run --
see the revision note in `188-a2-retrain.md`). Every other clause here is satisfied by *doing
nothing*: the blockout arm scores `fp_iou` 1.0, `collapse_rate` 0.0, and ratios of exactly 1.0
against itself. `CONTEXT.md` states the point directly -- "**1.0 means it did nothing**. An arm at
0.99 has not been measured as a generator however good its other numbers look (#75: a model scored
3D IoU 0.857 while being 99.9% its own input)". A bar that a no-op passes is not a bar.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

REGISTRATION_PATH = REPO / "execution/artifacts/188_preregistration.json"

# The frozen source's own ratios against its population's blockout, from
# `execution/artifacts/criterion2_full714.json` (n=714, strength 0.5). These are the numbers the
# non-inferiority clauses are set from, and they are a property of a checkpoint that already
# exists -- nothing here is derived from the retrained model.
FROZEN_RATIOS = {"vol_iou": 0.8625 / 0.9334, "extra": 0.0876 / 0.0714}
FROZEN_LEGACY = {"fp_iou": 0.9589, "missing": 0.0026, "extra": 0.0876, "vol_iou": 0.8625}

# The operating point the frozen source was scored at, and the one #119 will seal alongside the
# replacement. #115's method forbids sweeping, so this is fixed in advance and not tuned.
OPERATING_POINT = {"strength": 0.5, "steps": 20, "guidance": 1.0}


def clause_value(clause: dict, summary: dict, extras: dict) -> float:
    """The measured number a clause is judged on.

    Raises rather than defaulting. A clause that silently evaluated to 0 would sail through any
    `<=` threshold it was never actually measured against -- the failure mode #84's no-op weighting
    flags already cost this project two training runs.
    """
    source = clause.get("source", "summary")
    if source == "extra":
        return extras[clause["metric"]]
    arm = summary[clause["arm"]]
    if not arm:
        raise KeyError(f"arm {clause['arm']!r} produced no rows")
    value = arm[clause["metric"]]
    if source == "summary":
        return value
    if source == "ratio":
        reference = summary[clause["relative_to"]][clause["metric"]]
        if reference == 0:
            raise ZeroDivisionError(
                f"reference arm {clause['relative_to']!r} measured {clause['metric']}=0, so the "
                f"ratio clause {clause.get('id', '?')} is undefined on this population")
        return value / reference
    raise ValueError(f"unknown clause source {source!r}")


def _holds(rule: str, value: float, threshold: float) -> bool:
    ops = {">=": lambda v, t: v >= t, "<=": lambda v, t: v <= t, "==": lambda v, t: v == t,
           "<": lambda v, t: v < t, ">": lambda v, t: v > t}
    if rule not in ops:
        raise ValueError(f"unknown clause rule {rule!r}")
    return ops[rule](value, threshold)


def build_registration(reference: dict, frozen_ratios: dict, operating_point: dict,
                       gate: dict) -> dict:
    """The clauses, with every threshold and the reason it was chosen.

    `reference` holds the blockout and codec-ceiling summaries measured on the gate population
    **before any retrained checkpoint existed** -- they are properties of the data, so measuring
    them first is not peeking.
    """
    arm = "{arm}"          # filled in at scoring time by `score_registration`
    clauses = [
        {"id": "K1", "kind": "KILL", "source": "extra", "metric": "generation_failures",
         "rule": "==", "threshold": 0,
         "rationale": "The whole reason #188 exists: the frozen source raises IndexError on every "
                      "BuildingWorld row. A replacement that fails to generate for any scored row "
                      "has not fixed the structural block."},
        {"id": "K2", "kind": "KILL", "source": "extra", "metric": "buckets_scored",
         "rule": "==", "threshold": int(gate["buckets"]),
         "rationale": "All six of #171's BuildingWorld style buckets must generate. A checkpoint "
                      "that silently covers five has the same class of defect in a smaller place."},
        {"id": "G1", "kind": "GUARD", "source": "summary", "arm": arm, "metric": "vs_input",
         "rule": "<", "threshold": 0.98,
         "rationale": "CONTEXT.md's documented guard, at its documented value: 'IoU against the "
                      "blockout the arm started from. 1.0 means it did nothing. An arm at 0.99 has "
                      "not been measured as a generator however good its other numbers look (#75). "
                      "The guard is < 0.98.' This is the ONLY clause an inert model fails -- every "
                      "other one is satisfied by returning the envelope untouched. The frozen "
                      "source measures 0.9616 (n=12), so it clears this."},
        {"id": "G2", "kind": "GUARD", "source": "summary", "arm": arm, "metric": "collapse_rate",
         "rule": "<=", "threshold": 0.1582,
         "rationale": "CONTEXT.md's documented bar, at its documented value: 1-NN retrieval's "
                      "0.1582, 'a generator that destroys more buildings than naive retrieval is "
                      "not servable whatever else it scores'. The frozen source measures 0.098 on "
                      "the pinned 714, so it clears this comfortably; #92's aligned retrain reached "
                      "0.4636 and was recorded as a failure. Not #188's bar to loosen."},
        {"id": "P0", "kind": "PASS", "source": "summary", "arm": arm, "metric": "fp_iou",
         "rule": ">=", "threshold": 0.90,
         "rationale": "Footprint adherence is the hard criterion of the eval harness. The frozen "
                      "source's own legacy median is 0.9589 with p10 0.8803; 0.90 sits well below "
                      "its median, so this fails only on a source that has genuinely come off its "
                      "own conditioning footprint."},
        {"id": "B1", "kind": "PASS", "source": "ratio", "arm": arm, "metric": "vol_iou",
         "relative_to": "blockout", "rule": ">=", "threshold": round(frozen_ratios["vol_iou"], 4),
         "rationale": "Non-inferiority, normalised by population: relative to doing nothing on its "
                      "own rows, the replacement must be at least as good as the frozen source was "
                      "relative to doing nothing on its rows (0.8625/0.9334). ⚠️ This threshold is "
                      "below 1.0 BECAUSE the frozen source loses to its own envelope; #188 is a "
                      "substitution, not the open quality problem."},
        {"id": "B2", "kind": "PASS", "source": "ratio", "arm": arm, "metric": "extra",
         "relative_to": "blockout", "rule": "<=", "threshold": round(frozen_ratios["extra"], 4),
         "rationale": "The same non-inferiority test on the over-fill the envelope is supposed to "
                      "be carved out of (0.0876/0.0714 for the frozen source)."},
        {"id": "B3", "kind": "PASS", "source": "summary", "arm": arm, "metric": "missing",
         "rule": "<=", "threshold": 0.05,
         "rationale": "Absolute, not a ratio: the blockout's `missing` is 0 by construction, so a "
                      "ratio is undefined. `missing` is the dangerous direction -- it means the "
                      "generator ate the building rather than over-filled it -- and the frozen "
                      "source sits at 0.0026."},
        {"id": "I1", "kind": "INFORMATIONAL", "source": "summary", "arm": arm,
         "metric": "beats_envelope_rate", "rule": ">=", "threshold": 0.0056,
         "rationale": "Reported, never scored. The paired rate at which the arm beats the footprint "
                      "envelope on vol_iou; the frozen source manages 0.0056 on its own population. "
                      "It is not a bar for the reason set out in this module's docstring -- #188 is "
                      "a substitution, and requiring the replacement to beat an envelope the sealed "
                      "source loses to would make this a different ticket."},
    ]
    return {
        "meta": {
            "ticket": 188,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_rev": _git_rev(),
            "preregistered_before": "any training run on the new corpus",
            "frozen_source": "logs_building/vecset_v4_surf/vecset_denoiser_step240000.pth",
            "frozen_source_sha256": "643aed0896e2edc36ab3ecb073da63847881dc2ba95459eed896a19b65fed04d",
            "frozen_source_legacy_714": FROZEN_LEGACY,
        },
        "operating_point": dict(operating_point),
        "gate_population": dict(gate),
        "reference": reference,
        "clauses": clauses,
    }


def observed_extras(artifact: dict, registration: dict, arm: str, cohort: dict | None = None) -> dict:
    """The non-summary facts the KILL clauses need, READ OFF the run rather than asserted.

    ⚠️ `generation_failures` used to be an operator-supplied flag defaulting to 0, which meant the
    single KILL clause #188 exists for passed by default unless somebody remembered to contradict
    it. It is derived here instead: the eval harness scores exactly the rows that survived, so a row
    the candidate could not generate for is one the arm's `n` is short by.

    `buckets_scored` counts the distinct region buckets actually represented among the scored ids,
    which needs the cohort manifest's own region quota to be meaningful; without it the caller must
    supply the count.
    """
    expected = int(registration["gate_population"]["n"])
    scored = int(artifact["summary"][arm]["n"])
    extras = {"generation_failures": max(expected - scored, 0)}
    if cohort is not None:
        extras["buckets_scored"] = len(cohort["gate"]["quota"])
    return extras


def score_registration(summary: dict, registration: dict, arm: str, extras: dict) -> dict:
    """Judge one measured run against the registration. Pure: no I/O, no thresholds of its own."""
    if arm not in summary:
        raise KeyError(f"the run produced no arm {arm!r} (has {sorted(summary)})")
    scored, kill_clear, guards_held, bars_met = [], True, True, True
    for clause in registration["clauses"]:
        resolved = dict(clause)
        if resolved.get("arm") == "{arm}":
            resolved["arm"] = arm
        try:
            value = clause_value(resolved, summary, extras)
        except (KeyError, ZeroDivisionError):
            # An INFORMATIONAL clause decides nothing, so a metric this run happened not to emit
            # must not stop the verdict -- but it is recorded as unmeasured, never as a pass. A
            # KILL or BAR clause gets no such leniency: an unmeasured clause that decides the
            # verdict is exactly the silent-pass failure `clause_value` refuses.
            if resolved["kind"] != "INFORMATIONAL":
                raise
            scored.append({**resolved, "value": None, "pass": False, "measured": False})
            continue
        ok = _holds(resolved["rule"], value, resolved["threshold"])
        if resolved["kind"] == "KILL" and not ok:
            kill_clear = False
        if resolved["kind"] == "GUARD" and not ok:
            guards_held = False
        if resolved["kind"] == "PASS" and not ok:
            bars_met = False
        scored.append({**resolved, "value": value, "pass": bool(ok), "measured": True})

    # CONTEXT.md's three-part structure: a KILL answers the ticket "no"; a broken GUARD means the
    # arm is not servable whatever else it scored; only then does the PASS set decide.
    if not kill_clear:
        verdict = "KILL"
    elif not guards_held:
        verdict = "GUARD BROKEN"
    else:
        verdict = "PASS" if bars_met else "NOT MET"
    return {"arm": arm, "verdict": verdict, "clear_kill": kill_clear, "guards_held": guards_held,
            "numeric_pass": bars_met, "clauses": scored}


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="#188: write or evaluate the pre-registered bar")
    sub = ap.add_subparsers(dest="cmd", required=True)

    reg = sub.add_parser("register", help="write the pre-registration from a REFERENCE eval "
                                          "artifact (gt/blockout/codec_ceiling, no A2 arm)")
    reg.add_argument("--reference", required=True,
                     help="massing_arms_eval_*.json measured on the gate population with no "
                          "candidate arm, so nothing here can be tuned to a result")
    reg.add_argument("--cohort", default=str(REPO / "execution/artifacts/188_cohort.json"))
    reg.add_argument("--out", default=str(REGISTRATION_PATH))

    sc = sub.add_parser("score", help="verdict a candidate eval artifact against the registration")
    sc.add_argument("--eval", required=True, help="massing_arms_eval_*.json WITH the a2 arm")
    sc.add_argument("--arm", default="a2_s0.5")
    sc.add_argument("--registration", default=str(REGISTRATION_PATH))
    sc.add_argument("--cohort", default=str(REPO / "execution/artifacts/188_cohort.json"),
                    help="the cohort manifest, so the KILL clauses are read off the run and the "
                         "manifest instead of being asserted on the command line")
    sc.add_argument("--buckets_scored", type=int, default=None,
                    help="override the bucket count derived from --cohort (rarely needed)")
    sc.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.cmd == "register":
        artifact = json.loads(Path(args.reference).read_text())
        summary = artifact["summary"]
        if any(a.startswith("a2") for a in summary):
            raise SystemExit(
                "[188] the reference artifact contains an A2 arm. The bar must be set from the "
                "population alone, before any candidate has been scored on it.")
        cohort = json.loads(Path(args.cohort).read_text())
        registration = build_registration(
            reference={k: summary[k] for k in ("blockout", "codec_ceiling") if k in summary},
            frozen_ratios=FROZEN_RATIOS,
            operating_point=OPERATING_POINT,
            gate={"n": cohort["gate"]["n"], "digest": cohort["gate"]["digest"],
                  "buckets": len(cohort["gate"]["quota"]),
                  "source": "188_cohort.json role=gate"})
        Path(args.out).write_text(json.dumps(registration, indent=1))
        print(f"[188] pre-registration -> {args.out}")
        for clause in registration["clauses"]:
            print(f"  {clause['id']:>3} {clause['kind']:<13} {clause['metric']:<18} "
                  f"{clause['rule']} {clause['threshold']}")
        return

    artifact = json.loads(Path(args.eval).read_text())
    registration = json.loads(Path(args.registration).read_text())
    cohort = json.loads(Path(args.cohort).read_text())
    extras = observed_extras(artifact, registration, args.arm, cohort)
    if args.buckets_scored is not None:
        extras["buckets_scored"] = args.buckets_scored
    verdict = score_registration(artifact["summary"], registration, arm=args.arm, extras=extras)
    verdict["observed"] = extras
    print(f"=== #188 bar: {verdict['verdict']} (clear_kill={verdict['clear_kill']}, "
          f"guards_held={verdict['guards_held']}, numeric_pass={verdict['numeric_pass']}) ===")
    print(f"    generation failures: {extras['generation_failures']} "
          f"(expected {registration['gate_population']['n']} rows, "
          f"scored {artifact['summary'][args.arm]['n']})")
    for clause in verdict["clauses"]:
        mark = "PASS" if clause["pass"] else ("FAIL" if clause.get("measured", True)
                                              else "NOT MEASURED")
        shown = "        -" if clause["value"] is None else f"{clause['value']:>9.4f}"
        print(f"  {clause['id']:>3} {clause['kind']:<13} {clause['metric']:<18} "
              f"{shown} {clause['rule']} {clause['threshold']:<8} {mark}")
    if args.out:
        Path(args.out).write_text(json.dumps(verdict, indent=1))
        print(f"[188] verdict -> {args.out}")


if __name__ == "__main__":
    main()

# #140 — Do the assignment and type fixes compound, or do they trade against each other?

*Effort: solid-first semantic architectural carving. Opened 2026-09-02, discovered rather than
planned: `heightmap_program_assign_tau05`, trained while writing up [#139](139-assignment-temperature.md),
was found to already be this arm — #138's `type_prior` was wired unconditionally into every
`--objective program` run with no flag to disable it, so #139's first training run combined both
fixes without anyone asking it to. Written up on its own terms rather than discarded, because it is
a real, informative arm: [#138](138-type-head-imbalance.md)'s and [#139](139-assignment-temperature.md)'s
own closing sections both asked exactly this question next.*

> Does a better-calibrated assignment head (#139: `ASSIGN_TEMPERATURE` 1.0 → 0.5) hand #138's
> type-head fix cleaner regions to be right about, so the two corrections compound? Or does each
> fix's own surplus cost simply add to the other's?

Code: no new code. This arm is `heightmap_program_assign_tau05` exactly as trained for #139,
before the isolation bug was found. Artifacts `execution/artifacts/height_map_generator_assign_tau05_train.json`
and `..._assign_tau05_714_diagnostics.json` — the same files #139's first draft cited, now correctly
attributed. Montage/maps: `outputs/height_map_generator/{best,worst,representative}*.png`,
`maps_best.png` (rendered post-hoc from the same checkpoint, not re-trained).


## 🔑🔑 The full 2×2, on the same 411 carve-needing buildings

Four checkpoints, two binary factors (`ASSIGN_TEMPERATURE` 1.0 vs 0.5, `type_prior` off vs on),
every number from this map's own scoring harness:

| arm | assign τ | type fix | `missing` | `extra` | `vs_input` | collapse | `planar` | slots used | *(3D IoU)* |
|---|---|---|---|---|---|---|---|---|---|
| `heightmap_program_adj` *(#132)* | 1.0 | off | 0.0659 | 0.0832 | 0.8470 | 0.2579 | 0.12 | 2.03 | *0.8080* |
| `heightmap_program_typeadj` *(#138)* | 1.0 | on | 0.0835 | 0.0938 | 0.8163 | 0.2774 | 0.50 | 2.00 | *0.7960* |
| `heightmap_program_assign_tau05_only` *(#139)* | 0.5 | off | 0.0926 | **0.0772** | 0.8039 | 0.2603 | **0.67** | 1.77 | 0.8131 |
| **`heightmap_program_assign_tau05` *(#140, this arm)*** | 0.5 | on | **0.0592** | 0.1064 | 0.8432 | **0.1727** | 0.33 | **2.28** | **0.8189** |

🔑 **No metric is monotonic in "more fixes applied."** Reading down each column tells a different,
sometimes contradictory story:

* **`missing`**: worst under type-alone (0.0835) and assign-alone (0.0926) each pull it up from
  #132's 0.0659 — but COMBINED it falls to the best value in the table, 0.0592. Assign-alone hurts
  `missing`, type-alone hurts `missing`, and together they fix it. A genuine positive interaction.
* **`extra`**: assign-alone is the best in the table (0.0772, and the only arm to beat 1-NN's
  0.1031). Adding the type fix on top makes it the WORST (0.1064). A genuine negative interaction —
  the combination is worse than either ingredient alone, not just worse than the best one.
* **collapse**: type-alone is worst (0.2774), assign-alone is flat against baseline (0.2603 vs
  0.2579), and COMBINED is by far the best (0.1727) — the closest any program-route arm has come to
  the 1-NN guard (0.1582). Another positive interaction, and the biggest single effect in the table.
* **`planar_fraction`**: assign-alone is the highest measured on this whole map (0.67, above the
  real building's own 0.50). Type-alone reaches 0.50. COMBINED falls to 0.33 — LOWER than either
  ingredient alone. Stacking the fixes actively hurts the metric each one improves on its own.

None of the four cells is uniformly better or worse than another; this is a real factorial result,
not a monotone dial.


## 🔑 Why `planar_fraction` falls when the fixes are combined

Assign-alone's high `planar_fraction` (0.67) comes from using FEWER slots (1.77, down from #132's
2.03) — it makes the head more confident about answering "slot 0 alone," which the type head then
types correctly more often because slot 0 is the least-imbalanced slot to begin with. Adding the
type fix pulls the opposite lever: it exists specifically to make the head answer slot 2/3 (the
rare, hard-to-type slots) MORE often. Combined, slots used rises to 2.28 — more than either arm
alone, and closer to the label's 3.06 — but each additional slot beyond slot 0 is one the type head
is least reliable on, so the fraction of the (now larger) slot set that is typed correctly falls.

⚠️ This means assign-alone's 0.67 was flagged correctly in #139 as a low-complexity artefact, not a
free lunch — and this table is the direct evidence for that flag: the arm that actually uses more of
the label's real structure (2.28 slots, closer to 3.06) has the LOWER planar fraction, because using
more structure means using more of the slots the type head still gets wrong.


## 🔑 Why collapse improves so much when combined, and not from either fix alone

Collapse is the rate of buildings the arm effectively destroys (`missing` above threshold). Neither
fix alone moves it off #132's 0.258 baseline by more than noise (type-alone: 0.277, slightly worse;
assign-alone: 0.260, flat). Combined: 0.173, a 33% relative drop. The mechanism from #139 —
dominant-slot accuracy recovering from 0.27 to ~0.76-0.81 under the lower tau — only pays off against
collapse once the type head ALSO has a reason to draw a second region correctly instead of forcing
everything flat; a confidently-correct single flat slot (assign-alone's regime) does not carve
enough to destroy a building, but it also does not carve enough to match it. The combination is
where a correctly-identified SECOND region, correctly typed, first shows up often enough to matter.


## Against `PROGRAM_BAR`

    PASS   ops <= 3.0            1.0     ✔
           planar >= 0.40        0.33    ✘
           extra < 0.0603      0.1064    ✘
    GUARD  collapse <= 0.1582  0.1727    ✘  (closest measured -- 0.0145 over)
           vs_input < 0.98     0.8432    ✔
    KILL   planar <= 0.20        0.33    -> not fired

**Verdict: NOT MET**, the fourth arm in this chain to fail it, and the fourth different way of
failing it: #138 tripped the KILL clause outright; #139 (isolated) cleared `extra` but failed
collapse by a wide margin; this arm clears neither PASS clause but comes closer to the collapse
GUARD than anything else measured, on this map or its predecessors on the program route.


## What this settles, and what it does not

**Settles:**
* 🔑🔑 **The two fixes interact, and the interaction is metric-dependent, not uniformly good or
  bad.** `missing` and collapse compound favourably; `extra` and `planar_fraction` compound
  unfavourably. A single "do both fixes help" question does not have one answer.
* 🔑 **Assign-alone's headline `planar_fraction` (0.67, #139) is confirmed as a low-complexity
  artefact by this table**, not merely flagged as a possibility: the arm using more real structure
  (this one) scores lower on the same metric.
* **Collapse (0.1727) is the best this program route has measured**, and it required both fixes
  together — neither alone gets close.

**Does not settle:**
* ⚠️ **Still NOT MET.** Best collapse and best `missing` on this route is not a passing arm; `extra`
  and `planar_fraction` are worse here than in at least one of the two single-fix arms.
* **Whether a THIRD factor (e.g. a smaller type temperature, or a different assign tau) would
  recover the planar loss without giving back the collapse gain is untested** and would be a new,
  separately pre-registered arm, not a sweep over this one.
* ⚠️ **This arm was not planned, and its numbers were not pre-registered as a combined-arm
  hypothesis before running** — #139's own docstring predicted isolation, not this. It is reported
  in full because the numbers are real and the checkpoint exists, not because the experiment was
  designed to produce them. Read the 2×2 table as a factorial result found after the fact, not as a
  confirmed pre-registered prediction the way #138/#139's own single-factor arms are.


## What follows

- **No arm in this 2×2 passes `PROGRAM_BAR`.** The honest headline is that four runs on this route
  now span every combination of the two diagnosed fixes, and the closest any comes is this one, on
  collapse alone.
- 🔑 **`planar_fraction` needs a complexity-normalised reading before it is trusted again on this
  map.** Comparing arms that use very different numbers of slots on the same fraction metric is
  exactly the trap #126 named for `extra`/`missing` in isolation — the same caution now applies here.
- **A properly pre-registered combined arm — same two changes, run on purpose with a stated
  prediction — has not happened yet.** This ticket answers "what does the accidental combination
  look like," not "what is the right way to combine them."

See [138-type-head-imbalance.md](138-type-head-imbalance.md),
[139-assignment-temperature.md](139-assignment-temperature.md),
[132-overcarve-and-assignment.md](132-overcarve-and-assignment.md),
[126-massing-scoring.md](126-massing-scoring.md).

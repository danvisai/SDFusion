"""#148/#7: the human-evaluation rubric, formalized.

#127 established the precedent this formalizes: a human looks at a rendered visual and gives a
plain yes/no verdict against a stated scope sentence -- there, a montage; here, #147's own visual
carving trace. ⚠️ "This is recorded as their judgement, and it is not the same as mine" (#127) is
the exact posture this module keeps: a `Verdict` here is a human's answer, never derived from and
never overwriting any automated metric already computed for the same building
(`missing`/`extra`/`vol_iou`/`collapse_rate` from #10/#126/#127, `finalize_problems`/
`containment_problems` from #145/#146) -- both are kept, side by side, on purpose.

🔑 The three questions are fixed here, once, as the single source of truth: a future ticket
imports `RUBRIC` to show them, and calls `record_verdict` by its named keyword arguments -- which
Python itself enforces -- rather than re-typing or re-inventing the rubric ad hoc.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class RubricQuestion:
    key: str
    text: str


# The fixed rubric, verbatim from #148's own acceptance criteria. Question 2's polarity is
# deliberately NOT flipped to match the other two ("yes" is the good outcome for 1 and 3, the bad
# outcome for 2) -- it is kept exactly as #148 states it, rather than paraphrased into a false
# consistency; `record_verdict`'s own keyword name says so too.
RUBRIC: List[RubricQuestion] = [
    RubricQuestion("looks_like_a_building", "Does it look like a building?"),
    RubricQuestion("has_visible_artifacts", "Are there visible geometric artifacts?"),
    RubricQuestion("edit_matches_request", "Does the edit match what was requested?"),
]

_RUBRIC_KEYS = tuple(q.key for q in RUBRIC)


@dataclass(frozen=True)
class Verdict:
    """One rater's answers to `RUBRIC`, for one building's #147 visual carving trace.

    A human judgement, not a metric -- see the module docstring's #127 quote. `answers` is keyed
    exactly by `RUBRIC`'s own question keys; `record_verdict` is what enforces that in practice.
    """
    building_id: str
    rater: str
    answers: dict = field(default_factory=dict)          # {question.key: bool}
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Verdict":
        return Verdict(**{k: v for k, v in d.items() if k in Verdict.__dataclass_fields__})


def _verdict_path(trace_dir: Path) -> Path:
    return Path(trace_dir) / "verdict.json"


def record_verdict(trace_dir: Path, building_id: str, rater: str, *, looks_like_a_building: bool,
                   has_visible_artifacts: bool, edit_matches_request: bool,
                   notes: str = "") -> Path:
    """Record one verdict against `trace_dir` -- the same directory `carving_trace.save_carving_
    trace` wrote a building's frames to. ⚠️ REFUSED if `trace_dir` holds no such frames yet: a
    verdict answers questions about a visual carving trace, so the trace must genuinely exist
    first -- the evidence and the verdict travel together because a verdict cannot be recorded
    without it, not merely by convention. The three keyword-only questions are `RUBRIC`'s own,
    spelled out as real parameters rather than a free-form dict, so a caller cannot answer the
    wrong question, skip one, or invent a fourth -- the rubric stays fixed by construction.

    A second call for the same `trace_dir` overwrites the first -- one verdict per trace, the
    latest one standing, matching #127's own single-verdict precedent rather than aggregating
    several raters (not asked for here).
    """
    trace_dir = Path(trace_dir)
    if not any(trace_dir.glob("step*_view*.png")):
        raise ValueError(f"{trace_dir} holds no visual carving trace (no step*_view*.png) -- "
                         f"render one with carving_trace.save_carving_trace first")
    answers = dict(looks_like_a_building=looks_like_a_building,
                   has_visible_artifacts=has_visible_artifacts,
                   edit_matches_request=edit_matches_request)
    assert tuple(answers) == _RUBRIC_KEYS, "record_verdict's keywords must track RUBRIC exactly"
    verdict = Verdict(building_id=str(building_id), rater=rater, answers=answers, notes=notes)
    path = _verdict_path(trace_dir)                      # trace_dir already exists (checked above)
    path.write_text(json.dumps(verdict.to_dict(), indent=2, sort_keys=True))
    return path


def load_verdict(trace_dir: Path) -> Optional[Verdict]:
    """Read back the verdict recorded against one building's trace, or None if none exists yet."""
    path = _verdict_path(trace_dir)
    if not path.exists():
        return None
    return Verdict.from_dict(json.loads(path.read_text()))

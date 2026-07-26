# Run the Equal-Data Detail Scaling Curve

Type: task
Status: out of scope (2026-07-15)
Blocked by: 13

## Question

If the full-data kill-gate supports continuing, build and run matched 25% and 50% monolith/library
arms under the proven contracts, then estimate the two 25/50/100 scaling trends and the uncertainty
on any claim that retrieval degrades more gracefully than synthesis.

## Out of scope

This ticket's own precondition ("if the full-data kill-gate supports continuing") is false: [Decide
the Full-Data C2 Kill-Gate](13-decide-c2-kill-gate.md) resolved **FAIL** on 2026-07-15 --
decomposition does not win detail fidelity despite winning massing fidelity. Per the PRD's own
preregistered rule, a failed 100% kill-gate stops the scaling curve rather than running the lower
fractions anyway. The project owner confirmed proceeding straight to the evidence package (#18)
rather than pursuing remediation, so this ticket is ruled out of scope rather than left open.

## Comments

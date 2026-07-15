# Test LoD3 Element Enrichment

Type: task
Status: out of scope (2026-07-15)
Blocked by: 04, 13

## Question

Add the real `lod3_tum` facade components as a clearly separated enrichment ablation, rerun the
supported decomposition evaluation, and measure whether additional real element coverage improves
detail fidelity without contaminating the BuildingNet equal-data headline.

## Out of scope

This ticket presupposes a "**supported** decomposition evaluation" to rerun with enrichment.
[Decide the Full-Data C2 Kill-Gate](13-decide-c2-kill-gate.md) resolved **FAIL** on 2026-07-15 --
there is no supported decomposition result to enrich. Also worth noting: ticket 12's own follow-up
already found `data/lod3_tum` dead-ends for this purpose (CityGML LoD3's semantic vocabulary is
window/door/roof only, no discrete tower/dome/balcony objects to extract), so this ablation would
likely have been a dead end even independent of the kill-gate result. The project owner confirmed
proceeding straight to the evidence package (#18) rather than pursuing remediation, so this ticket
is ruled out of scope rather than left open.

## Comments

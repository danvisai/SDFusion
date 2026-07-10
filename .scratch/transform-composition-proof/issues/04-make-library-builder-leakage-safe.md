# Make the Element-Library Builder Leakage-Safe

Type: task
Status: open
Blocked by: 03

## Question

Extend `scripts/foundations/build_element_library.py` with explicit include/exclude id inputs and
configurable output locations, then prove from emitted metadata that excluded test buildings never
contribute elements and that repeated builds are deterministic enough for fraction comparisons.

## Comments

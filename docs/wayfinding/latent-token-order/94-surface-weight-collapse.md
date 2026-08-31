<!-- Mirrored from the tracker, 2026-08-14. -->

> **Open ticket, mirrored locally** so this effort can be read without the
> tracker. Nothing was lost for this one — it had no committed asset.


# #94 — Why does weighting up the surface term collapse the model, if the term is the cleaner signal?

*State: open · opened 2026-08-09*


## Ticket

Part of #87

## Question

[#84](https://github.com/danvisai/SDFusion/issues/84) explained the collapse under increased
decoded-surface pressure as the surface term overshooting: solidity fell **89.9% → 57.6% → 19.2%** as
the term pressed harder, and the reading was "the pressure is the mechanism".

[#89](https://github.com/danvisai/SDFusion/issues/89) complicates that. It measured the surface term as
**order-insensitive at every t** (varies 0.0001–0.0005% against the epsilon loss's 0.31–1.53%, a
1,900–5,300× gap), which supports reading it as the *cleaner* of the two signals. A cleaner signal
collapsing the model when weighted up is not explained by "too much pressure on a good signal".

So: **what actually collapses the model when the surface term is weighted up?**

## Candidates, none tested

- The term is clean but its *gradient path* is not — it reaches the latent through a frozen 191.6M
  decoder, and #73 measured that decoder as intolerant to leaving its manifold. A clean objective can
  still push along a direction the decoder cannot absorb.
- The collapse is not about the surface term at all but about what it *displaces*: weighting it up
  reduces the relative weight of the epsilon term, which is what keeps the latent on-manifold.
- The two terms disagree, and the disagreement grows with weight.

## ⚠️ Why this is not urgent

This is a *diagnosis of a rejected branch* — #84 already ruled the band-fix family out, and the shipped
arm is `vecset_v4_surf@240k`. It matters because the map's central problem is "how do we move the latent
without leaving the manifold", and this is a measured instance of exactly that going wrong.

Blocked in spirit by [#92](https://github.com/danvisai/SDFusion/issues/92): once the epsilon target is
aligned, re-run the surface-weight sweep. If the collapse disappears, it was the corrupted bridge all
along; if it persists, it is decoder-side and #73 owns it.

## Judged on

A written answer that picks between the candidates with a measurement, or records that it cannot and
says what would.

# #64 — is 35,776 buildings enough to train a vecset autoencoder to crisp?

**Date:** 2026-07-27 · **Verdict: NOT from scratch — but the question turns out to be the wrong one.**
The literature says a from-scratch vecset AE at our scale underperforms on sharp detail; the same
literature shows the standard remedy, and it reframes what this effort should build.

---

## 1. What the exemplars actually trained on

Verified against paper text, not summaries:

| model | training shapes | note |
|---|---|---|
| **3DShape2VecSet** (SIGGRAPH 2023) | **~51K** ShapeNet-v2, 55 categories, trained **jointly** | AE reaches **IoU 0.965**. Explicitly *not* per-category — they criticize NeuralWavelet for training categories separately |
| **Dora** (CVPR 2025) | **~400,000** Objaverse meshes | *"Our training data consists of approximately 400,000 3D meshes carefully filtered from Objaverse."* |
| **TRELLIS** (CVPR 2025) | **TRELLIS-500K** | Objaverse XL, ABO, 3D-FUTURE, HSSD, Toys4k |
| **Hunyuan3D-2** | Objaverse / Objaverse-XL scale | **exact count not published**; the paper does not quantify it |

**Our corpus: 35,776.** Same order as 3DShape2VecSet's training set; **~11× smaller** than Dora's.

## 2. Nobody publishes a scaling curve — so this cannot be settled by citation

The ticket asked for an ablation or scaling study on training-set **size**. There isn't one in this
family. Dora ablates only its architectural contributions (Sharp Edge Sampling, Dual Cross Attention);
3DShape2VecSet ablates latent size and query type. **No vecset paper reports reconstruction quality as a
function of dataset size.** Everything below is therefore indirect evidence, and is labelled as such.

## 3. The sobering result — and it lands almost exactly on our number

Dora attributes 3DShape2VecSet's weakness **directly to data**:

> *"3DShape2VecSet consistently underperforms across all detail levels, primarily due to its **limited
> training data** affecting generalization capability."*

3DShape2VecSet trained on **~51K shapes (~40K train)** — essentially **our corpus size**. So the one
data point closest to our scale is a published *underperformance on sharp detail*, which is precisely
the quantity this whole effort exists to fix. **A from-scratch vecset AE at 35,776 shapes is evidenced
to be the weak configuration, not a safe one.**

## 4. The mitigating result, from the same paper

Fine-tuning a **pretrained** vecset VAE onto a new corpus is standard, published practice:

> *"we fine-tune Craftsman-VAE on our dataset (denoted as Craftsman-VAE†) since both Craftsman-VAE and
> 3DShape2VecSet were originally trained on smaller datasets."*

Dora hit our exact problem and solved it by fine-tuning rather than retraining. That is the remedy, and
it is not a hack — it is what the state of the art does.

## 5. Our own counter-evidence: narrow domain compensates, on the AE half

Dora's data-hunger finding concerns reconstructing **55 diverse Objaverse categories**. We need **one
narrow category**. And we already have the in-house measurement:
[#56](https://github.com/danvisai/SDFusion/issues/56) showed our **own** VQVAE — trained on our **own**
35,776 buildings — round-trips at **0.0044 against a GT floor of 0.0041**.

**At our scale, in our domain, an autoencoder already reconstructs crisply.** That is direct, local
evidence that narrow domain compensates for raw count on the autoencoder half. It is the single
strongest reason to think 35,776 is workable, and it is our own data rather than an inference.

## 6. Which half is data-hungry — and why the answer changes the plan

The ticket asked this, and it is the pivotal question. Combining the evidence:

- **The AE half is the one the literature flags as data-hungry** — but only for *open-domain* coverage.
  Narrowed to buildings, our own codec already clears it (§5).
- **The diffusion half has no published scaling curve at all**, and it is where CLAY / Hunyuan3D-2 /
  TRELLIS actually spend their hundreds of thousands of shapes: learning a *distribution* over
  open-domain geometry.
- **Crucially, our problem was never the AE.** #56 exonerated our codec; #54/#59/#60 indicted the
  diffusion. So a from-scratch vecset AE would be rebuilding the half that already works, at a scale the
  literature says is the weak one.

**There is an architectural asymmetry that matters more than either count.** In our dense-grid stack,
diffusion error becomes **surface error** — a slightly-wrong latent decodes to a wavy wall, which is
exactly what #59 measured (latent L1 fell 4×, decoded roughness barely moved). In a vecset stack the
decoder is trained to map *any* token set to a crisp surface, so diffusion error becomes **shape error,
not surface error** — a wrong-but-crisp building rather than a right-but-wavy one. That asymmetry is the
real reason the representation change should work, and it does **not** depend on us matching Dora's data
scale. *(This is an argument from the mechanism our own results established, not a cited result — flagged
as such.)*

## 7. Answer

**No, 35,776 is not enough to train a competitive vecset AE from scratch** — §3 is close to a direct
measurement of that. **But the effort does not need to.** The evidenced configuration is:

> **Take a pretrained vecset autoencoder (frozen, or fine-tuned on our 35,776 buildings per §4), and
> train OUR OWN footprint-conditioned diffusion on its latent space.**

This is the standard division of labour in latent generative modelling — nobody trains a Stable Diffusion
VAE from scratch to build a new conditional image model — and it maps cleanly onto our situation:

- the **decoder**, which needs 400K+ shapes to learn crisp reconstruction, is inherited;
- the **generative model**, which is the actual research contribution (footprint conditioning, the C1/C2
  thesis), is **ours**, trained on our corpus at a scale that only has to learn *building distribution*,
  not *open-domain geometry*;
- and #63 already measured that this decoder family reaches **0.00328**, at the GT floor.

## 8. What this does to the map

**It splits option A in two, and the split is the important output of this ticket.**

- **A1 — vecset AE from scratch on 35,776.** Now *evidenced-risky* by §3. Should be dropped unless #62
  recovers meshes *and* someone accepts the 3DShape2VecSet-scale result as tolerable.
- **A2 — pretrained/fine-tuned vecset AE + our own footprint-conditioned diffusion.** Evidenced by §4,
  §5, §6 and #63. **This is the recommended shape of the rebuild.**

**A2 also softens the constraint question in [#65](https://github.com/danvisai/SDFusion/issues/65)**,
which was framed as build-ours vs adopt-frontier. A2 is neither: it adopts a **component** (an
autoencoder) while the **generative model stays ours**. That is a materially easier position to defend
to a supervisor than adopting a whole generative pipeline — the contribution remains the
footprint-conditioned diffusion and the editable-SDF downstream. It does **not** dissolve the licensing
question, which still applies to whichever pretrained AE is chosen.

**Contingency on [#62](https://github.com/danvisai/SDFusion/issues/62):** A2 still needs surface samples
to fine-tune the AE and to train the diffusion's target — so mesh recovery remains a prerequisite. If #62
comes back negative, a **frozen** pretrained AE (no fine-tune) becomes the fallback, since only the
diffusion would then need our data, and the diffusion can be supervised through the frozen encoder.

## Sources

- Dora, *Sampling and Benchmarking for 3D Shape Variational Auto-Encoders* — [arXiv 2412.17808](https://arxiv.org/html/2412.17808v1)
- 3DShape2VecSet — [arXiv 2301.11445](https://arxiv.org/abs/2301.11445)
- TRELLIS — [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS)
- Hunyuan3D 2.0 — [arXiv 2501.12202](https://arxiv.org/html/2501.12202v1)
- COD-VAE, *Representing 3D Shapes with 64 Latent Vectors* — [arXiv 2503.08737](https://arxiv.org/html/2503.08737)

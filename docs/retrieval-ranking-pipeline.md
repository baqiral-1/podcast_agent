# Building a Retrieval + Ranking Pipeline for Cross-Book Synthesis

When teams talk about retrieval quality, the conversation usually starts with embedding models and vector search settings.

That is necessary, but not sufficient.

In our pipeline, the real challenge was not finding semantically similar text. The challenge was reliably selecting evidence that could support cross-book reasoning, survive strict grounding checks, and still read naturally when transformed into spoken narrative.

This is the story of the hardest problems we hit in production, and the architectural choices we made to address them.

## Challenge 1: Similarity Search Is Not Synthesis-Ready Retrieval

Our first retrieval iterations were good at finding “locally relevant” chunks. They were not good at building a synthesis corpus.

The failure mode was subtle: vector similarity gave us passages that matched axis language, but many of those passages were poor building blocks for comparative storytelling. Some were repetitive. Some were conceptually narrow. Some were technically relevant but not quotable. Many did not create useful tension between books.

That mismatch forced a key design decision: retrieval had to be treated as a staged policy, not a single rank.

We moved to a hybrid model:

- A deterministic retrieval layer to maximize recall.
- A judgment layer to score usefulness for downstream narrative work.

This split improved control immediately. It let us keep recall broad while making relevance decisions in a context that understands thematic intent, spoken suitability, and cross-book relationships.

## Challenge 2: Book Imbalance Quietly Distorts the Final Narrative

In cross-book projects, small ranking biases compound.

Without constraints, one or two books can dominate a thematic axis simply because their language aligns better with the query representation. If that happens early, later stages inherit a skewed corpus and produce “synthesis” that is actually single-source amplification.

We addressed this by making representation a first-class concern in candidate admission.

The retrieval stage now explicitly enforces cross-book participation before reranking. Instead of letting global similarity decide everything, we preserve a meaningful stream of candidates from each source and only then allow quality-sensitive selection. That changed the behavior of the whole pipeline:

- Axis coverage became more stable across runs.
- Minority-book perspectives stopped disappearing.
- Cross-book comparisons became easier to surface in synthesis mapping.

The key insight was operational: fairness at retrieval time is cheaper and more reliable than trying to “rebalance” after synthesis artifacts are already formed.

## Challenge 3: Chapter Clustering Reduces Narrative Breadth

Even when book-level representation improved, we still saw concentration inside individual books.

High-confidence retrieval often clustered around the same chapter neighborhoods. That created a hidden bottleneck: we appeared to have broad passage volume, but it came from narrow local contexts. Downstream writing then felt repetitive because many citations traced back to adjacent source material.

We solved this by introducing diversity pressure during candidate selection, not after.

Selection now accounts for structural spread so that no single chapter can consume disproportionate attention when alternatives exist. This does not force artificial uniformity; strong chapters can still contribute heavily. But it prevents early over-concentration from collapsing thematic range.

The practical outcome was significant:

- Better variation in argument types and examples.
- Improved narrative pacing in episode planning.
- Fewer repair loops triggered by brittle evidence concentration.

## Challenge 4: LLM Rerankers Are Powerful but Operationally Fragile

Moving ranking judgment into an LLM gave us better quality, but introduced a different class of risk: contract reliability.

In production, an LLM reranker that occasionally drops IDs, duplicates records, or returns partial outputs can silently poison downstream artifacts. The impact is amplified because later stages assume retrieval outputs are complete and stable.

We treated this as a systems reliability problem, not just a prompt problem.

Our reranking stage runs with strict schema contracts, output integrity checks, and explicit retry behavior when coverage drops below quality thresholds. Invalid pairings are filtered before they can enter synthesis. We fail loudly when quality guarantees are not met.

That discipline changed the team’s debugging posture. Instead of arguing about whether later-stage behavior “looks wrong,” we can point to concrete stage-level integrity events and fix root causes quickly.

## Challenge 5: Fixed Retention Rules Overfit Either Dense or Sparse Axes

A single retention size does not work across all thematic axes.

Some axes are rich with cross-book evidence and deserve broader retention. Others are sparse and should remain compact to avoid diluting signal with marginal passages. A fixed cutoff over-prunes strong axes and over-inflates weak ones.

We moved to adaptive retention based on the observed quality and richness of each axis candidate set.

The important behavior is not the exact policy math. The important behavior is that retention is responsive to axis reality while staying bounded for predictability and cost control.

Once we made retention adaptive, downstream synthesis quality became more consistent:

- Rich axes carried enough evidence to support non-trivial insight formation.
- Sparse axes stopped introducing low-value noise.
- End-to-end token spend aligned better with actual informational density.

## Challenge 6: Retrieval Systems Fail Slowly Without Deep Observability

Retrieval failures are rarely dramatic. They degrade quality over multiple stages before becoming visible in final output.

We needed observability that made retrieval behavior inspectable at the same granularity as generation behavior.

So we made retrieval artifacts first-class outputs of every run:

- Axis-level candidate logs with selection decisions.
- Per-axis and per-book summary metrics.
- Cross-book relationship validation stats.

This has been critical for production iteration. When quality shifts, we can determine whether the issue came from recall breadth, admission distribution, rerank quality, or pair validation. That cuts tuning cycles from guesswork to diagnosis.

## What Changed in Practice

The biggest lesson from this work is that retrieval quality came from policy composition, not any single model upgrade.

The system improved when we combined:

- deterministic recall,
- structured admission control,
- diversity-aware selection,
- contract-safe reranking,
- adaptive retention,
- and strong retrieval telemetry.

Each component addressed a different failure mode. Together, they turned retrieval from a best-effort heuristic into a dependable substrate for synthesis and generation.

## The Broader Takeaway

If your goal is cross-document synthesis, retrieval is not a “front-end” to generation. It is part of generation quality itself.

Treating retrieval as a controllable, observable, contract-driven subsystem gives you leverage that pure prompt tuning cannot.

That mindset is what made this pipeline production-usable for us: we stopped asking “did we retrieve similar text?” and started asking “did we retrieve the right evidence to support robust, grounded synthesis?”

Once you optimize for that question, architecture decisions become much clearer.

# G-cand-004 — layout provenance (§43)

2026-09-18; analysis only, no named-axis syntax.

Observed failures: M2 MHA merges [B,H,T,DH] directly into [B,T,D]; M4 ViT flattens patch grids without moving channel/patch axes. Both compile and produce wrong numeric layouts. DenseNet reordered concatenations illustrate feature ordering, but axis labels alone would not detect those.

Common underlying concept: preserving meaningful axis grouping/order through reshape, transpose and flatten.

Why existing constructs are insufficient: dimensions prove cardinalities, not permutations. Tensor kinds add semantic hints, not layout identity. Explicit transposes plus metamorphic checks work today; this prevents calling the gap an automatic language-change mandate.

Proposed semantics: investigate optional layout provenance attached to reshapes/grouped axes, with an explicit escape for intentionally changed interpretation. This could begin as inspect/tooling contracts rather than syntax. Do not infer axis names from B/T/H spelling.

Static knowledge gained: whether a reshape merges the intended groups in the intended order; warnings at lost provenance, not false errors for equal extents.

New ambiguity introduced: flatten erases structure; equal-size axes may intentionally swap; concat order is feature identity, not merely axis identity. What gets inferred and where provenance can be dropped is unresolved.

Interaction with existing constructs: block templates, catalog attention, custom unknown shapes, broadcast, concat, transpose and scan. Unknown remains unknown; never invent named layout certainty.

Programs simplified: manual attention and ViT patchify checks; potential windowed attention/space-to-depth witnesses later.

Programs made harder: free-form tensor mathematics if every reshape needs ceremonial labels.

Alternative rejected: require names on every axis; declare shape-compatible reshapes wrong; silently insert transposes; pretend DenseNet channel counts prove feature order.

Evidence bars: two distinct concrete numerical witnesses meet recurrence. Optional provenance's benefit and ambiguity bars are not settled. Keep G-candidate status, existing semantics unchanged; retain numeric mutant tests.

# Challenge record — tabular-pipeline (Milestone 3, §19 data contracts)

```text
Challenge:                  tabular-pipeline (+ twins impute-fit-all, impute-no-fit,
                            misspelled-fit, unknown-category-unhandled)
Source/paper:               the "adult" census recipe: numeric impute → standardize,
                            categorical vocab → encode, all statistics fitted on train
TENSA features stressed:     fitted data ops with `fit:`, `vocab(unknown:)`, Tokens →
                            embedding inside a model, concat of a numeric and an
                            embedded column, split declaration
TENSA implementation:
    field age:   Tensor[1] = select(column: "age") |> to_float
                             |> impute(median, fit: train) |> standardize(fit: train)
    field job:   Tokens[1]  = select(column: "job")
                             |> vocab(fit: train, unknown: "<unk>") |> encode
    field label: Class      = select(column: "income") |> as_class
    model Tabular(age: Tensor[B, 1], job: Tokens[B, 1]) -> Logits[B, K] {
      let e = job |> embedding(NC, 4) |> reshape([B, 4])
      let h = concat(age, e, axis: -1)
      return h |> linear(16) |> relu |> linear(K)
    }
Check result:               ok, no warnings. `mod.data[0].fitted` lists exactly three
                            statistics (median, standardize, vocabulary), all on
                            `train`. `impute(median, …)` records the strategy actually
                            chosen, not the catalog's "median|mean" alternative.
Inspect result:             data table shows the three fitted statistics and the split.
IR result:                  pipelines: age 4 ops, job 3 ops, label 2 ops.
Runtime result:             4 steps, output `[4, 3]`, every table updated.
Reference result:           emitted plan `py_compile`s and trains under torch 2.12.
Twins:                      impute-fit-all / impute-no-fit → AXS0620 at the op (F-020);
                            misspelled-fit (`fitt:`) → AXS0620 as well — a silently
                            ignored argument must not read as a fitted-on-train claim;
                            unknown-category-unhandled (`vocab(fit: train)` with no
                            `unknown:`) → new AXS0622: a vocabulary fitted on train WILL
                            meet unseen categories at eval time, and what happens then
                            is part of the learning problem, not a runtime accident.
Finding classification:     F-020 (location), plus AXS0622 as a new contract (recorded
                            here; no separate F — the check did not exist to be wrong)
Severity:                   minor
Proposed action:            DONE
Language change required?:  no
Regression test added?:     yes — `challenge:tabular-pipeline` (4 twins)
```

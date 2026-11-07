# Instagram track: findings

Group "Penyayang Claude", Teknologi Big Data, SV IPB University.
Pipeline: `src/instagram_pipeline.py`, with shared helpers in `src/profile_tracks_utils.py`.
Numbers: `outputs/metrics/instagram.json`. Per-account scores: `outputs/predictions/instagram_scored.csv`.
Both files are generated locally by the pipeline and are **not committed**: the metrics file contains
file-row handles and the scores are per account (docs/adr/0007).

## In one paragraph

On 696 hand-labelled Instagram accounts, a Random Forest separates "fake/spammer" from "genuine"
profiles with **about 92–93% accuracy and a ROC-AUC of about 0.98**. The contract's single seed-42
split reports **95.3%**, but that split is a lucky draw. It sits 1.4 standard deviations above the
cross-validated mean, and 20 other seeds of the same procedure give anything from **87.0% to
96.1%**. The data cannot tell the Random Forest from the Logistic Regression baseline (McNemar
p = 1.0). The strongest signals are low follower count, digits in the username, no profile
picture and few posts. These are plausible behavioural signals, not artefacts of the file, so we
found no leakage. The real limit of this dataset is not its accuracy but its narrowness. It is
one person's labels on accounts crawled over five days in March 2019, in a sample where a
missing profile picture almost always means fake (199 of the 201 picture-less accounts) and no
fake account has an external link.

## 1. What this is for

The target user is a **trust & safety moderator triaging accounts for human review**. The model
ranks accounts; a person decides. A score is not evidence of intent, and it is never grounds for
automated action. A false positive here is a real person wrongly flagged. Section 8 shows who
those people are in this dataset.

## 2. Dataset and its real provenance

| | |
|---|---|
| Source | Kaggle `free4ever1/instagram-fake-spammer-genuine-accounts`, license CC BY 3.0 |
| Files | `train.csv` (576 rows) + `test.csv` (120 rows) = **696 accounts** |
| Collection | the author's own crawler, **15–19 March 2019** (per the dataset card) |
| Labels | hand-assigned by the author for a final-year project. The card says "fake" and "spammer" are used interchangeably and that "a couple" of accounts may be misidentified |
| Target | `fake` → our operational `1 = Inauthentic`, `0 = Authentic` |

"Fake/spammer" is the author's judgement, not a platform ground truth. It is also a different
concept from Facebook "spam" and X "bot". The shared `0/1` target is an operational convention,
and the comparison across platforms has to be read with that in mind.

## 3. Class balance and file structure

- **348 genuine / 348 fake**: exactly balanced overall, and each vendor file is 50/50 (288/288
  and 60/60). The majority-class baseline is therefore about 50%. Every number below should be
  read against that, not against 0%.
- **No missing values** (0 of 7,656 cells). No imputation was needed, and zeros in this data are
  real values (an empty bio, zero posts).
- **Both vendor files are sorted by class**: all genuine rows first, then all fake rows. This does
  not affect our random stratified split. It would, however, ruin any un-shuffled k-fold or
  "first 80% / last 20%" split of these files, so we record it as a hazard for anyone reusing them.
- **4 exact duplicate pairs** (8 rows), each pair inside a single vendor file and sharing one
  label (two pairs in `test.csv`, two in `train.csv`; row handles are in the locally generated
  `outputs/metrics/instagram.json`, not in this public copy).
  There is no ID column, so we cannot tell whether a pair is one account crawled twice or two
  accounts with identical metadata. We kept them. **None straddles the headline train/test
  split**, so they cannot inflate the test score.

## 4. Features

All 11 profile columns are used. There is no identifier column to drop (contract rule 1). The
`account_key` in the output CSV is a provenance handle (vendor file + row number) and is never a
feature.

| Feature | Meaning |
|---|---|
| `profile pic` | has a profile picture (0/1) |
| `nums/length username` | share of the username that is digits |
| `fullname words` | words in the display name |
| `nums/length fullname` | share of the display name that is digits |
| `name==username` | display name identical to username |
| `description length` | bio length in characters |
| `external URL` | bio links out (0/1) |
| `private` | private account (0/1) |
| `log1p(#posts)`, `log1p(#followers)`, `log1p(#follows)` | counts, log-transformed |

**Engineered:** `log1p` of the three count columns. `#followers` runs from 0 to 15.3 million. The
transform is monotone, so it changes nothing for the Random Forest; it only stops a handful of
celebrity-sized accounts from dominating the Logistic Regression.

## 5. Split and models

- **Headline:** both vendor files combined, then `common.stratified_split` (stratified 80:20,
  seed 42). Because Spark's `sampleBy` draws each row independently, the split came out
  **589 train / 107 test** (84.6 / 15.4), with 51 fakes and 56 genuine in test.
- **Random Forest** (main): 200 trees, maxDepth 10, featureSubsetStrategy `sqrt`, Gini, maxBins 32,
  seed 42.
- **Logistic Regression** (baseline): StandardScaler, then binomial LR with ridge regParam 0.01
  and maxIter 200.
- **No tuning.** The hyperparameters were fixed before seeing any result. No class weighting was
  used, since the classes are balanced. The test split is used once, for the final scores.

## 6. Headline results (seed-42 test split, n = 107)

| Metric | Random Forest | Logistic Regression | Majority baseline |
|---|---|---|---|
| Accuracy | **0.953** | **0.953** | 0.523 |
| Precision (fake) | 0.979 | 0.960 | – |
| Recall (fake) | 0.922 | 0.941 | – |
| F1 (fake) | 0.950 | 0.951 | – |
| Specificity | 0.982 | 0.964 | – |
| Balanced accuracy | 0.952 | 0.953 | 0.5 |
| MCC | 0.908 | 0.906 | 0 |
| ROC-AUC | 0.993 | 0.996 | 0.5 |
| PR-AUC | 0.992 | 0.996 | 0.48 |
| Confusion (TN, FP, FN, TP) | 55, 1, 4, 47 | 54, 2, 3, 48 | |

Both models beat the 52% majority baseline by over 40 percentage points, so they have clearly
learned something. See [`outputs/figures/instagram_confusion_matrix.png`](../outputs/figures/instagram_confusion_matrix.png).

## 7. How much to trust that number

This is the core of this track. 696 rows make a single split high-variance, and five
independent measurements show how much.

**a. A single test account is worth 0.93 percentage points.** With 107 test accounts, five errors
versus four moves accuracy from 95.3% to 94.4%. The 95% Wilson interval on the headline accuracy
is **0.895 – 0.980**, 8.5 points wide. Bootstrapping the test accounts gives the same picture:
RF recall 0.84 – 0.98, F1 0.90 – 0.99, MCC 0.82 – 0.98.

**b. Repeated stratified 5-fold × 10 cross-validation inside the training split** (50 fits per
model; the test split is never touched):

| | RF mean ± SD | RF range over 50 folds | LR mean ± SD |
|---|---|---|---|
| Accuracy | **0.924 ± 0.020** | 0.889 – 0.975 | 0.918 ± 0.023 |
| F1 (fake) | 0.923 ± 0.021 | 0.885 – 0.974 | 0.917 ± 0.024 |
| Recall (fake) | 0.911 ± 0.031 | 0.833 – 0.967 | 0.902 ± 0.036 |
| Precision (fake) | 0.937 ± 0.025 | 0.883 – 1.000 | 0.933 ± 0.025 |
| MCC | 0.849 ± 0.040 | 0.778 – 0.950 | 0.837 ± 0.045 |
| ROC-AUC | 0.979 ± 0.009 | 0.959 – 0.998 | 0.976 ± 0.011 |

The headline 0.953 is **+1.45 SD above the CV mean** (ROC-AUC +1.49 SD), so seed 42 happened to
produce an easier-than-average test set. See [`outputs/figures/instagram_cv_variability.png`](../outputs/figures/instagram_cv_variability.png):
the orange diamond sits near the top of the blue cloud for every metric.

**c. The "split lottery".** We re-ran exactly the contract's procedure with 20 other seeds. This
is a robustness check only: nothing was chosen from it. RF accuracy came out **0.925 ± 0.023,
anywhere from 0.870 to 0.961**, and LR 0.923 ± 0.023 (0.888 – 0.977). Had the group used seed 7
instead of 42, the slide could honestly have said 87% or 96%.

**d. The vendor's own partition** (train on `train.csv`, test on `test.csv`) gives
**RF 0.908** (95% CI 0.843 – 0.948; F1 0.908, MCC 0.817, ROC-AUC 0.982) and **LR 0.900**. That
is 4.5 points below the headline, which looks like a disagreement worth a finding. Measured
against the CV spread, though, it is only **−0.77 SD**: well within ordinary split-to-split
noise. **The finding is that the two evaluations disagree by about as much as chance alone
produces.** There is no sign of a distribution shift between the vendor's files, but there is
clear evidence that any single split on this dataset is a ±2–3-point measurement.

**e. RF versus LR cannot be separated.** On the test split the two disagree on just 4 accounts,
2 each way (exact McNemar p = 1.0). Across the 50 CV folds the RF is ahead by +0.6 ± 2.2
accuracy points and wins only 52% of folds (16% are ties). **The group should not claim the
Random Forest beats the baseline on Instagram.** On this data they are the same.

**f. Learning curve.** With training size varied inside CV (`instagram_learning_curve.png`):

| Training accounts | 47 | 94 | 164 | 235 | 353 | 471 |
|---|---|---|---|---|---|---|
| RF accuracy | 0.899 | 0.909 | 0.913 | 0.915 | 0.924 | 0.930 |
| RF ROC-AUC | 0.962 | 0.967 | 0.970 | 0.975 | 0.977 | 0.979 |

Ten times more training data buys about 3 accuracy points, and the curve is flattening. **More
rows of the same kind would mainly sharpen the estimate, not the model.** "Too small to
generalise" is therefore not mainly a sample-size problem the model would grow out of. It is a
representativeness problem: one labeller, one five-day crawl in 2019, hand-picked accounts.
Resampling cannot measure that, so it stays a stated limitation (section 10).

**Best single summary to report:** accuracy about 0.92 (±0.02), ROC-AUC about 0.98, with RF and
LR equivalent. The 0.953 headline is real but optimistic.

## 8. What the models look at, and who gets wrongly flagged

[`outputs/figures/instagram_feature_importance.png`](../outputs/figures/instagram_feature_importance.png) compares three views.

| Rank | RF impurity (train) | RF permutation, test ROC-AUC drop | LR coefficient (per SD) |
|---|---|---|---|
| 1 | log #followers 0.285 | log #followers 0.054 ± 0.015 | log #followers −1.65 |
| 2 | log #posts 0.249 | username digits 0.029 ± 0.008 | profile pic −1.05 |
| 3 | username digits 0.122 | profile pic 0.011 | username digits +0.99 |
| 4 | description length 0.110 | log #posts 0.008 | log #posts −0.81 |
| 5 | profile pic 0.098 | fullname words 0.002 | private −0.53 |

Impurity and permutation importance agree (Spearman ρ = 0.84, 4 of the top 5 shared), so the
ranking is not an artefact of the impurity measure. Description length ranks higher under
impurity than under permutation because it overlaps with #posts and profile pic; shuffling it
alone costs almost nothing.

**Behavioural reading.** The fake accounts here look *empty and machine-named*: few followers,
few or no posts, no bio, no picture, and a username padded with digits (`user84721`-style).
Genuine accounts have an audience, a posting history, a bio and a face. Following many accounts
(`#follows`) pushes slightly toward fake in the LR, which fits follow-for-follow behaviour.

**Two strong signals are partly selection artefacts.** In this file, **199 of 348 fakes have no
profile picture but only 2 of 348 genuine accounts lack one**, and **no fake account has an
external URL** (22.7% of genuine ones do). On real Instagram, plenty of genuine people have no
picture and no link. The model has learned that "no picture means fake" from a sample in which
that is almost always true.

**Who gets wrongly flagged.** We scored every account once out-of-sample: test accounts by the
final model, training accounts by the CV model that did not see them. The RF makes **51 errors
in 696** (18 false positives, 33 false negatives). **All 18 false positives are genuine
accounts with a profile picture and only 0–28 posts. 14 of them have an empty bio and 17 have
fewer than 500 followers**: quiet, new or lurking real users. These are the people a deployed model would send
to a moderator by mistake, and the review step exists for them. The false negatives are fakes
that look populated (a picture, some posts, often following thousands of accounts).

## 9. Leakage audit: what we checked and what we found

| Check | Outcome |
|---|---|
| Identifier columns | none exist in the data |
| Row order vs label | both files sorted by class; irrelevant to our shuffled split, recorded as a hazard |
| Exact duplicates | 4 pairs, all same-label, **none straddle the train/test split** |
| Same features, conflicting labels | 0 |
| Impurity vs permutation importance | agree (ρ = 0.84) |
| Single split vs CV / other seeds / vendor partition | headline optimistic by ~1.4 SD; vendor result within noise |
| Near the 99% audit threshold? | No. ~92–95%, driven by plausible signals |

**Outcome: no leakage found.** The score is not inflated by IDs, duplicates or file order. It is
inflated a little by a lucky split, which we quantified, and possibly by an easy, hand-selected
sample, which we cannot quantify.

## 10. Limitations a reader must know

1. **One person's labels**, with "fake" and "spammer" merged and an admitted handful of mistakes.
   `1 = Inauthentic` is an operational convention, not a finding about intent.
2. **One crawl, March 2019.** Instagram and fake-account tactics have changed since. Nothing here
   shows the model would transfer to 2026 accounts.
3. **Small.** The test split has 107 accounts, so one account moves accuracy by 0.93 points.
   Quote the CV mean ± SD, not the single split.
4. **Balanced 50/50 by construction.** Real prevalence of fakes is unknown and far from 50%, so
   precision in deployment would be lower than measured here. The same model applied where 1
   in 20 accounts is fake would flag many more real people for every fake it finds.
5. **Selected, clear-cut cases.** "No picture → fake" and "has a link → genuine" hold almost
   perfectly in this sample and much less in reality. Borderline genuine users (new, quiet, no
   picture) are under-represented, and they are exactly who a deployed model would wrongly flag.
6. **Profile metadata only.** There is no behaviour over time, no content and no network. The
   strongest signals (picture, digits in the username, follower count) are cheap for an
   adversary to change.
7. **Main model and baseline are statistically indistinguishable** on this data.

## Reproduce

```bash
# dari root repo; Java 17 di PATH atau JAVA_HOME di-set (lihat README)
python src/instagram_pipeline.py      # ~5 min: 50-fold CV, 20-seed lottery, learning curve
```

Figures: [`outputs/figures/instagram_confusion_matrix.png`](../outputs/figures/instagram_confusion_matrix.png), `instagram_feature_importance.png`,
`instagram_cv_variability.png`, `instagram_learning_curve.png`.

# Facebook track: findings

Group "Penyayang Claude", Teknologi Big Data, SV IPB University.
Pipeline: `src/facebook_pipeline.py`, with shared helpers in `src/profile_tracks_utils.py`.
Numbers: `outputs/metrics/facebook.json`. Per-account scores: `outputs/predictions/facebook_scored.csv`.
Both files are generated locally by the pipeline and are **not committed**: the metrics file contains
row-level profile ids and the scores are per account (docs/adr/0007).

## Verdict first

The Random Forest scores **about 97% accuracy** on this dataset (96.7% ± 1.8 over cross-validation;
94.7% on the contract's seed-42 split). On the metrics that matter for a 17%-positive problem it
scores **PR-AUC ≈ 0.89, recall ≈ 0.89, F1 ≈ 0.90**. The number is reproducible, and it is **not**
caused by duplicate rows or by the profile id, which we dropped.

**It should still not be believed as a measure of how well spam can be detected on Facebook.**

1. **Most of it is free.** Always answering "legit" already scores 82.8%.
2. **A single threshold on one column does as well.** `#friends ≤ 126 → spam`, learned on the
   training data, reproduces the Random Forest's test confusion matrix *exactly*. So does an RF
   given only the six weakest columns.
3. **The file is two blocks, and the model learns the blocks.** Legit rows come first and spam rows
   last. The same Random Forest predicts *which half of the file a row sits in* (test accuracy
   0.990, PR-AUC 0.991) better than it predicts *the spam label* (0.947, 0.915).
4. **Its errors are the rows that disagree with their block.** 19 of the RF's 24 out-of-sample
   errors fall on the 23 rows whose label disagrees with their position in the file. Those rows
   carry the profile of their block, not of their label. On the other 577 rows it makes 5
   errors. Its accuracy is, almost exactly, the share of rows whose label agrees with their
   block (577/600 = 96.2%).
5. **The values look constructed rather than measured.** The "fractions" exceed 1 in 27 rows,
   the "averages" are always whole numbers, round numbers are over-represented, the first rows
   look hand-typed, and the class counts do not match the dataset card.

What the 97% measures is how different two separately produced groups of rows are, not how well
profile metadata detects Facebook spam. **Report it as an upper bound on an easy benchmark.** Do
not put it side by side with X or TikTok as if Facebook spam were the easiest to catch.

## 1. What this is for

The target user is a **trust & safety moderator triaging accounts for human review**. Scores
rank accounts for a person to look at; they are never grounds for automated action. A false
positive is a real person wrongly flagged. Section 10 shows who that would be here.

## 2. Dataset and its real provenance

| | |
|---|---|
| Source | Kaggle `khajahussainsk/facebook-spam-dataset`, license **unknown** |
| File | `Facebook Spam Dataset.csv`, **600 rows**, 15 columns |
| Claimed collection | "Facebook API and Facebook Graph API … from public profiles" (dataset card). No date, no sampling method, no labelling procedure |
| Claimed size | card: 500 legit + 100 spam. **File: 497 legit + 103 spam** |
| Target | `Label` (0 = legit, 1 = spam) → our operational `0 = Authentic`, `1 = Inauthentic` |

"Spam" is the dataset author's label for a profile. It is not the same concept as Instagram
"fake" or X "bot", and the shared `0/1` target is an operational convention.

## 3. Schema correction for the concept document

Worksheet 3 marked this schema as unverified. It is now verified, and four names in the group's
concept document are wrong:

| Concept doc guessed | Actual column | Card description |
|---|---|---|
| `avg_comments` | `avgcomment/post` | average number of comments per post |
| `avg_likes` | `likes/post` | average number of likes per post |
| `avg_tags` | `tags/post` | average number of tags in a post (rate of tagging) |
| `avg_hashtags` | `#tags/post` | average number of hashtags present in a post |

Full verified schema, in file order: `profile id, #friends, #following, #community, age,
#postshared, #urlshared, #photos/videos, fpurls, fpphotos/videos, avgcomment/post, likes/post,
tags/post, #tags/post, Label`. `age` is the account age in days. `fpurls` and `fpphotos/videos`
are the fractions of posts containing URLs and photos/videos.

## 4. Class balance, and why accuracy is the wrong headline

**497 legit / 103 spam (17.2% positive).** A classifier that never flags anyone scores 82.8%
accuracy and catches zero spam. So this track leads with **PR-AUC** (the no-skill value equals
the prevalence, 0.21 on the test split), **recall** (share of spam caught) and **F1**. Accuracy
is reported but never used as the verdict.

## 5. Features

| Decision | Detail |
|---|---|
| **Dropped: `profile id`** | an identifier (contract rule 1), and also a label leak; see 8.1 |
| **Used: 13 columns** | every other column |
| **Engineered: `log1p`** | on `#friends, #following, #community, #postshared, #urlshared, #photos/videos, avgcomment/post`. Monotone, so no effect on the RF; it stops LR being driven by outliers (e.g. one account with 665 comments per post, where the next highest is 1.06) |
| **Missing: `fpurls`, 2 rows** | reconstructed as `#urlshared / #postshared`. That is the column's own definition, and it matches the stored value on 594 of the 598 non-missing rows. **Not zero, not a median** (contract rule 5) |
| **Class weighting** | weight = n / (2 × n_class) on training rows (legit ≈ 0.6, spam ≈ 3.0) |

## 6. Split and models

- `common.stratified_split`, seed 42. Spark's `sampleBy` came out at **505 train (83 spam) /
  95 test (20 spam)**, roughly 84:16 rather than exactly 80:20.
- **Random Forest** (main): 200 trees, maxDepth 10, featureSubsetStrategy `sqrt`, Gini, maxBins 32,
  seed 42, class-weighted.
- **Logistic Regression** (baseline): StandardScaler, then binomial LR, ridge regParam 0.01,
  maxIter 200, class-weighted.
- **No tuning.** The hyperparameters were fixed in advance, the decision threshold is 0.5, and
  the test split is used once.

## 7. Headline results

### Seed-42 test split (n = 95, 20 spam)

| Metric | Random Forest | Logistic Regression | No-skill / majority |
|---|---|---|---|
| **PR-AUC** | **0.915** | 0.913 | 0.211 |
| **Recall (spam)** | **0.900** (18 / 20) | 0.900 | 0 |
| **F1 (spam)** | **0.878** | 0.857 | 0 |
| Precision (spam) | 0.857 | 0.818 | – |
| MCC | 0.845 | 0.818 | 0 |
| Balanced accuracy | 0.930 | 0.923 | 0.5 |
| ROC-AUC | 0.973 | 0.980 | 0.5 |
| Accuracy | 0.947 | 0.937 | 0.789 |
| Confusion (TN, FP, FN, TP) | 72, 3, 2, 18 | 71, 4, 2, 18 | |

### How much that moves

With 20 spam accounts in test, **one missed spam account moves recall by 5 points**. The 95%
Wilson interval on recall 18/20 is **0.70 – 0.97**. Bootstrapping the test accounts gives RF
precision 0.67 – 1.00, F1 0.75 – 0.97 and average precision 0.80 – 0.99.

**Repeated stratified 5-fold × 10 CV inside the training split** (50 fits per model):

| | RF mean ± SD | RF range | LR mean ± SD |
|---|---|---|---|
| **PR-AUC** | **0.887 ± 0.074** | 0.727 – 1.000 | 0.913 ± 0.062 |
| **Recall** | **0.894 ± 0.076** | 0.688 – 1.000 | 0.912 ± 0.060 |
| **F1** | **0.900 ± 0.056** | 0.786 – 1.000 | 0.906 ± 0.050 |
| Precision | 0.909 ± 0.061 | 0.778 – 1.000 | 0.902 ± 0.061 |
| MCC | 0.882 ± 0.066 | 0.747 – 1.000 | 0.888 ± 0.060 |
| ROC-AUC | 0.973 ± 0.025 | 0.876 – 1.000 | 0.979 ± 0.015 |
| Accuracy | 0.967 ± 0.018 | 0.931 – 1.000 | 0.969 ± 0.017 |

**Split lottery** (the same 80:20 procedure with 20 other seeds; nothing chosen from it): RF
accuracy 0.948 – 0.983, recall 0.80 – 1.00, PR-AUC 0.770 – 0.995. Seed 42 sits slightly
*below* average on accuracy (−1.1 SD) and on the average for recall and PR-AUC.

**RF vs LR:** the RF is not better. On the test split they disagree on one account (McNemar
p = 1.0). Across CV folds the LR is ahead on PR-AUC by 0.026 ± 0.057 and the RF wins only 16% of
folds on accuracy (56% are ties). See `facebook_cv_variability.png` and `facebook_pr_curve.png`.

## 8. The audit: should the high score be believed?

Bu Gema's rule is to distrust anything near 99%. The brief predicted 95–99% accuracy here, and
cross-validated accuracy is 96.7%. We ran seven checks.

### 8.1 `profile id` is a label leak: the file is sorted by class (removed)

`profile id` correlates **r = +0.58** with `Label`. An identifier cannot cause spam, so we checked
the row order (`facebook_id_ordering_leak.png`):

- `profile id` runs 1 → 601 and is strictly increasing in file order. Only id 196 is missing.
- Spam per block of 50 rows: **4, 0, 0, 0, 0, 0, 0, 1, 0, 9, 45, 44**.
- The boundary is at **profile id 496**: 93 of the 106 rows at or above it are spam, against 10
  of the 494 rows below it. 4 of those 10 are in the first 8 rows, which look like hand-typed
  examples (8.4).
- The rule `profile id ≥ 496` alone scores **96.2% accuracy** on the full file, against 82.8%
  for the majority class. Learned on train and scored on test, it reaches **balanced accuracy
  0.955 and recall 0.95**, which is *better than the Random Forest*. An RF given only the id gets
  PR-AUC 0.913.

The id was dropped per contract rule 1. The ordering it reveals turns out to be central to the
verdict (8.7).

### 8.2 Duplicates and near-duplicates: none

- 0 exact duplicate rows and 0 identical feature vectors (excluding the id).
- In standardised feature space, the median distance from a test account to its nearest training
  account is **0.90**, against **0.92** between training accounts. No test account lies within
  0.25 of a training account. Test rows are no closer to training rows than training rows are
  to each other.
- A 1-nearest-neighbour "memoriser" scores only balanced accuracy 0.848 on test, well below the
  RF. **The score is not explained by duplication.**

### 8.3 Is one threshold on one column enough? Yes

Each threshold below was chosen on train (maximising balanced accuracy) and scored on test
(`facebook_single_feature_separability.png`):

| One-line rule | Test balanced acc | Test recall | Test precision |
|---|---|---|---|
| `#friends ≤ 126` | **0.930** | 0.90 | 0.857 |
| `#following ≤ 129` | 0.930 | 0.90 | 0.857 |
| `#community ≥ 136` | 0.930 | 0.90 | 0.857 |
| `#postshared ≥ 1628` | 0.930 | 0.90 | 0.857 |
| `#photos/videos ≥ 1585` | 0.930 | 0.90 | 0.857 |
| `#urlshared ≥ 152` | 0.923 | 0.90 | 0.818 |
| `fpurls ≥ 0.28` | 0.923 | 0.90 | 0.818 |
| **Random Forest, all 13 columns** | **0.930** | 0.90 | 0.857 |

Five different columns each reproduce the Random Forest's result exactly, misclassifying the
same accounts. A depth-2 decision tree scores 0.923. **On this file the Random Forest adds
nothing over a one-line rule.**

### 8.4 Do the values look measured or generated? Partly generated

| Observation | Why it matters |
|---|---|
| `fpphotos/videos` > 1 in **25 rows** (all legit; max 2.74) and `fpurls` > 1 in **2 rows** (max 1.09) | the card defines these as *fractions of posts*, which cannot exceed 1. These rows have more photos (or URLs) than posts |
| `tags/post` and `#tags/post` are whole numbers in **100%** of rows | they are described as *averages per post*, which are almost never whole numbers. `tags/post` also runs from 10 to 99 tags per post on average, which is implausible |
| The last digit of count fields is far from uniform (χ² p ≈ 0 for both classes). "0" makes up 14% of legit and 21% of spam values, against 10% expected; `#photos/videos` ends in 0 in 43% of spam rows | the classic signature of rounded or hand-entered numbers (`facebook_digit_heaping.png`) |
| The first 8 rows are 65% round numbers vs 13% for the rest, and their `fpurls` values do not match their own counts (row 1: 850 URLs / 1000 posts, but `fpurls` = 0.49) | they look like hand-typed examples placed at the top of the file |
| Count distributions are right-skewed for legit (`#community` skew 7.8, `#urlshared` 7.3), as real counts are, but **symmetric or left-skewed for spam** (−0.2, −1.2) | measured social-media counts are heavy-tailed. The spam block looks drawn from bounded ranges |
| Card: 500 / 100. File: 497 / 103 | the file is not what the card describes |
| `#postshared` vs `#photos/videos`: r = 0.994 | almost every post is a photo/video in both classes, so the column adds nothing independent |

None of this proves fabrication. Together it says the rows were at least partly constructed or
edited by hand, so whatever the model learns may describe how the file was made more than how
spammers behave.

### 8.5 Impurity importance vs permutation importance

| Rank | RF impurity (train) | RF permutation, test ROC-AUC drop | LR coefficient (per SD) |
|---|---|---|---|
| 1 | log #urlshared 0.204 | log #urlshared 0.019 ± 0.012 | log #urlshared +0.86 |
| 2 | log #community 0.193 | log #friends 0.014 | log #community +0.81 |
| 3 | log #friends 0.184 | fpurls 0.014 | fpurls +0.51 |
| 4 | fpurls 0.122 | log #community 0.012 | log avgcomment/post +0.51 |
| 5 | log #following 0.093 | log #following 0.012 | age +0.44 |

The two RF measures **agree on the ranking**: Spearman ρ = 0.71 and all 5 of the top 5 shared.
So the impurity ranking is not an artefact of the impurity measure. But permutation shows
something impurity hides: **no single column matters**. Shuffling the "most important" one costs
only 0.019 ROC-AUC, because every other column carries the same signal. Shuffling the five
flagged high-correlation columns *together* costs 0.197 ROC-AUC (average precision −0.43). The
signal is real for this file but spread across nearly every column at once.
(`facebook_feature_importance.png`)

### 8.6 Ablation: retrain without the suspicious columns

| RF trained on | PR-AUC | F1 | Balanced acc |
|---|---|---|---|
| all 13 columns (headline) | 0.915 | 0.878 | 0.930 |
| without the 5 flagged columns (`#urlshared, fpurls, #community, #photos/videos, #postshared`) | 0.882 | 0.878 | 0.930 |
| without those **and** `#friends, #following` (only `age, fpphotos/videos, avgcomment/post, likes/post, tags/post, #tags/post` remain) | 0.882 | 0.878 | 0.930 |
| only `#friends` | 0.862 | 0.857 | 0.923 |
| only `profile id` (leak demonstration) | 0.913 | 0.769 | 0.920 |

Removing the seven strongest columns leaves the confusion matrix unchanged. **The classes differ
on almost every column**, which is what you would expect if the two classes came from two
different processes rather than from one population of accounts.

### 8.7 The decisive check: does the model learn spam, or the file's two blocks?

Split the file at profile id 496 into a "legit block" and a "spam block". 577 of 600 rows have a
label that agrees with their block; 23 do not. These are 10 spam-labelled rows in the legit block
(four of them in the first eight rows of the file) and 13 legit-labelled rows in the spam block. Row-level
profile ids are not reproduced in this public copy (docs/adr/0007); the local run writes them to
`outputs/metrics/facebook.json`.

| Median profile | #friends | #community | #urlshared | #postshared |
|---|---|---|---|---|
| all legit rows (n = 497) | 1245 | 51 | 46 | 765 |
| all spam rows (n = 103) | 47 | 841 | 1897 | 2751 |
| **legit-labelled rows in the spam block** (n = 13) | **64** | **789** | **1423** | **2165** |
| **spam-labelled rows in the legit block**, excl. rows 1–8 (n = 6) | **1326** | **58.5** | **57** | **625** |

The rows that disagree with their block **look like their block, not like their label**
(`facebook_block_conflicts.png`). There are only four exceptions: two of the
hand-typed spam examples at the top of the file, which do look like spam; and two rows right at
the boundary, which look legit. The consequences:

- Every account was scored out-of-sample once (test accounts by the final model, training
  accounts by the CV model that had not seen them). The RF makes **24 errors, and 19 of them are
  on the 23 out-of-block rows** (83% error rate there) against **5 on the 577 in-block rows**
  (0.9%). Three of those five sit right at the boundary, with legit-looking
  profiles.
- The RF's out-of-sample predictions agree with the **block** 98.5% of the time and with the
  **label** 96.0%.
- Trained to predict the **block** instead of the label (same RF, same split), the model scores
  **test accuracy 0.990, PR-AUC 0.991, MCC 0.970**. Predicting the label, it scores 0.947,
  0.915 and 0.845.

The 13 features encode *which part of the file a row came from* more faithfully than they encode
*spam*. The model's accuracy ceiling is the share of rows whose label agrees with their block.
The 23 conflicting rows are either mislabelled or deliberately mixed in, and nothing in the data
can say which.

### 8.8 Verdict

| Question | Answer |
|---|---|
| Is the number computed correctly and reproducible? | **Yes.** 96.7% ± 1.8 accuracy, PR-AUC 0.89 ± 0.07 across CV; stable across seeds |
| Is it inflated by the profile id? | No. The id is dropped; it would have given the same score on its own |
| Is it inflated by duplicates? | No. 0 duplicates and no near-duplicates |
| Is it one leaky column? | No. Removing the seven strongest columns changes nothing |
| Does it measure Facebook spam detection? | **No.** One threshold on one column does as well; the features predict the file block better than the label; the errors are exactly the block conflicts; the values look partly constructed |

**Believe the 97% as a description of this file: two separately produced groups of rows that
differ on every column. Do not believe it as evidence that profile metadata detects Facebook
spam at 97%.** In the cross-platform comparison, Facebook should be reported as an easy, possibly
synthetic benchmark, an upper bound, and not ranked against X or TikTok on raw scores.

## 9. What the features say behaviourally, if taken at face value

In this file, spam profiles have **few friends and followings** (median 47 / 64 vs 1245 / 1235),
belong to **many communities** (841 vs 51), and share **far more posts and links** (2751 posts and
1897 URLs vs 765 and 46). They are also **older** accounts (1840 vs 1102 days) with fewer likes per
post. That reads like a broadcast account pushing links into groups rather than talking to
friends: a plausible spam pattern. It is also exactly what two hand-built, non-overlapping
groups would look like, so it should not be quoted as a finding about real Facebook spammers.

## 10. Who would be wrongly flagged

Sorted by `rf_p_inauthentic`, the review queue (`facebook_scored.csv`, `review_priority`)
flags **11 accounts labelled legit with scores of 0.75–0.98**. All 11 are legit-labelled rows in the spam block. If their labels are right, these are
real people with unusual profiles: few friends, many groups, lots of shared links. They are
exactly the false positives the human-review step exists to catch. If their labels are wrong,
the model is right and the dataset is wrong. A moderator, not the model, has to make that call.

## 11. Limitations

1. **Thin, inconsistent provenance.** There is no collection date, sampling method or labelling
   procedure, the license is unknown, and the class counts do not match the card.
2. **Values look partly constructed** (8.4): impossible fractions, whole-number averages, digit
   heaping, and hand-typed first rows.
3. **The file is two blocks, and the model learns the blocks** (8.7). 23 rows conflict with
   their block, and we cannot tell whether they are label noise or real exceptions.
4. **Small and imbalanced.** 20 spam accounts in test means 5 recall points per account. Quote CV
   mean ± SD, not the single split.
5. **Unknown real prevalence.** Precision measured at 17% spam would fall wherever spam is rarer.
6. **"Spam" ≠ "fake" ≠ "bot".** `1 = Inauthentic` is an operational convention across platforms,
   not a claim about intent.
7. **Main model and baseline are indistinguishable** (McNemar p = 1.0; LR slightly ahead in CV).

## Reproduce

```bash
# dari root repo; Java 17 di PATH atau JAVA_HOME di-set (lihat README)
python src/facebook_pipeline.py      # ~3 min: audits, 50-fold CV, 20-seed lottery
```

Figures: [`outputs/figures/facebook_confusion_matrix.png`](../outputs/figures/facebook_confusion_matrix.png), `facebook_feature_importance.png`,
`facebook_cv_variability.png`, `facebook_pr_curve.png`, `facebook_id_ordering_leak.png`,
`facebook_single_feature_separability.png`, `facebook_digit_heaping.png`,
`facebook_block_conflicts.png`.

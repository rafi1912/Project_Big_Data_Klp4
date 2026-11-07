# X / Twitter track: findings

Group "Penyayang Claude", Teknologi Big Data, SV IPB University.
Pipeline: `src/twitter_pipeline.py` (Spark MLlib, uses `src/common.py`). Optional provenance check: `src/twitter_faker_check.py`.
Numbers: [`outputs/metrics/twitter_synthetic.json`](../outputs/metrics/twitter_synthetic.json), [`outputs/metrics/twitter_real.json`](../outputs/metrics/twitter_real.json).
Per-account scores: `outputs/predictions/twitter_synthetic_scored.csv`, `outputs/predictions/twitter_real_scored.csv`
(generated locally, not committed: per-account output, docs/adr/0007).
Figures: `outputs/figures/twitter_*.png`.

## Verdict first

This track ran **two datasets through the same Spark pipeline**: the same split, the same
cross-validation, the same two models and the same audit.

| Held-out test set, seed 42 | Synthetic: `goyaladi` (Worksheets 2–3) | Real: `danieltreiman` (adopted) |
|---|---|---|
| What a row is | one tweet, by a user who appears once | one real account |
| Rows / positive rate | 50,000 / 50.0% | 37,438 / 33.2% |
| Majority-class baseline accuracy | 0.501 | 0.666 |
| **Random Forest accuracy** (95% CI) | **0.494** (0.485–0.504) | **0.854** (0.846–0.862) |
| RF balanced accuracy | 0.494 | 0.841 |
| RF precision / recall (inauthentic) | 0.494 / 0.491 | 0.769 / 0.802 |
| RF F1 (inauthentic) | 0.492 | 0.785 |
| RF MCC (0 = no signal) | −0.011 | 0.675 |
| RF ROC-AUC (no-signal band) | 0.492 (0.489–0.511) | 0.923 |
| RF PR-AUC (chance = positive rate) | 0.490 (0.499) | 0.880 (0.334) |
| Logistic Regression accuracy | 0.499 | 0.788 |
| Share of bots among the 5% highest-scored accounts | 0.453 (worse than the 0.499 base rate) | 0.995 |

1. **The original dataset is generated noise, so it was set aside.** Every numeric column is
   uniform and independent of the label. The tweet and hashtag text is **exactly the 971-word
   `lorem` list of the Faker library**. Every location is a Faker city name, and each "user"
   has exactly one tweet. Both models, with TF-IDF text features included, land on the
   majority baseline. This score is suspiciously **low**, and the explanation lies in the data,
   not the model. It is kept as a documented **negative control**.
2. **The replacement dataset gives a real, honest result.** On real accounts, Random Forest
   reaches 85.4% accuracy against a 66.6% baseline, MCC 0.675 and ROC-AUC 0.923. The vendor's
   own split gives the same answer (85.5%). Nothing was tuned toward 80%. The result is not
   near 99%, and the leakage audit (section 4.9) found and removed one real leakage path.
3. **What separates bots from humans here** is engagement and reach: few likes given, few
   followers, few accounts followed. Also a high posting rate for such an account, no
   verification and geolocation switched off. Section 4.8 shows that impurity importance and
   permutation importance disagree on the order. It also shows why we trust permutation more.
4. **These scores are a review queue for trust-and-safety moderators. They are not proof of
   intent and never grounds for an automated sanction.** At the default threshold, 600 of the
   4,981 real people in the test set (12.0%) would be flagged. About one flag in four (23%)
   points at a human.

## Ringkasan (Bahasa Indonesia)

Jalur X/Twitter memakai dua dataset dengan pipeline Spark yang sama persis. **(1) Dataset asli
Worksheet 2–3** (`goyaladi`, 50.000 baris) terbukti **sintetis**. Semua kolom angka berdistribusi
seragam dan tidak berhubungan dengan label (|r| maksimum 0,007). Teks tweet dan hashtag persis
sama dengan daftar kata `lorem` milik pustaka Faker (971 kata). Setiap lokasi berpola nama kota
Faker, dan setiap User ID hanya punya satu tweet. Random Forest dengan fitur TF-IDF mendapat
akurasi 0,494 dan ROC-AUC 0,492, sama dengan menebak acak (baseline mayoritas 0,501). Skor yang
"terlalu rendah" ini bukan kegagalan model. Skor ini bukti bahwa datanya tidak mengandung sinyal,
sehingga dataset ini disimpan sebagai **kontrol negatif** dan tidak dipakai untuk hasil X.
**(2) Dataset pengganti** (`danieltreiman`, 37.438 akun nyata, 33% bot) memberi akurasi Random
Forest **0,854** (baseline 0,666), balanced accuracy 0,841, F1 0,785, MCC 0,675, ROC-AUC 0,923 dan
PR-AUC 0,880. Logistic Regression mendapat 0,788. Hasil ini tidak di-tuning ke arah 80%, tidak
mendekati 99%, dan tetap diaudit kebocorannya. Skor model adalah **antrean tinjauan untuk
moderator**, bukan bukti niat dan bukan dasar sanksi otomatis.

---

## 1. What this is for

The target user is a **trust & safety moderator triaging accounts for human review**. The model
puts accounts in order for a person to look at, and the person decides. A false positive is a
real person wrongly put in the queue. The output must therefore be read as "worth a look",
never as "is a bot". Bot detection labels are themselves contested (section 4.10). That is one
more reason a human decides.

## 2. Why this track has two datasets

Worksheets 2 and 3 named `goyaladi/twitter-bot-detection-dataset`. When we profiled it, it
showed no signal at all. We could have dropped it quietly. Instead we kept it and ran it through
the full pipeline, for two reasons:

* Bu Gema's rule is to **distrust a suspicious score and audit it**. People usually apply that
  rule to scores near 99%. It applies just as much to a "bot detection benchmark" on which a
  Random Forest with text features scores exactly what a coin would. The audit is the same in
  both directions: look at the data before you blame or praise the model.
* Running the identical pipeline on both datasets is a **controlled comparison**. The only thing
  that changes between the two columns of the table above is the data. So the difference in
  results cannot come from a modelling choice.

We adopted the replacement dataset, `danieltreiman/twitter-human-bots-dataset`, because it
contains real account metadata with real behavioural variation. Section 5 sets out the
side-by-side case.

---

## 3. Dataset 1: synthetic (negative control)

### 3.1 What the file claims to be

`data/landing/twitter_bot_detection/bot_detection_data.csv`, Kaggle `goyaladi/twitter-bot-detection-dataset`.
It has 50,000 rows and the columns `User ID, Username, Tweet, Retweet Count, Mention Count,
Follower Count, Verified, Bot Label, Location, Created At, Hashtags`. The dataset README
describes "user profiles and associated tweet data" with a bot label. It says nothing about how
the data were collected or how the labels were assigned.

### 3.2 Evidence that it is generated, not collected

All numbers come from `dataset_validation` in `twitter_synthetic.json`. The pipeline recomputes
them on every run, except the Faker comparison, which is a separate optional script.

| Check | What real Twitter data would show | What this file shows |
|---|---|---|
| Rows per user | many tweets per active user | **50,000 distinct User IDs for 50,000 rows**, one tweet each |
| Usernames | one username, one person | 6,035 usernames are shared by several User IDs, and 60% of those shared names carry *both* labels |
| Tweet text | open vocabulary, URLs, emoji, mentions | **971 distinct words in total, exactly Faker's `en_US` `lorem` word list** (971/971 both ways, Faker 40.44.0). Every tweet has 6–14 words |
| Hashtags | topical tags | the **same 971 words** again |
| Location | real places, blanks, jokes | 25,199 invented town names. **100% carry a Faker city prefix (North/Lake/Port…) or suffix (-ville/-furt/-mouth…)** |
| Retweet / Mention / Follower counts | heavy-tailed: most near zero, a few huge | **uniform** over 0–100, 0–5 and 0–10,000 (χ² uniformity p = 0.44, 0.42, 0.31) |
| Created At | daily and weekly rhythms | uniform over Jan 2020 – May 2023. Hour of day independent of label (p = 0.54) |
| Label vs. each numeric column | some association | max \|r\| = **0.007**. KS tests bot vs. human p = 0.81 / 0.32 / 0.90 |
| Label vs. Verified | bots are rarely verified | 12,456 / 12,540 / 12,526 / 12,478 across the 2×2 table (χ² p = 0.56) |
| Label vs. individual words | some words typical of bots | 36 of 971 words at p < 0.05 where 48.6 are expected by chance. **None** survive a Bonferroni correction |
| Best single feature, train AUC | > 0.6 for any useful signal | **0.504** |
| Label balance | real collections are rarely 50/50 | 25,018 / 24,982 (binomial p = 0.88 against exactly 50/50) |

We verified these findings independently of the group's earlier profiling, and they agree with
it. The Faker comparison is reproducible with `python src/twitter_faker_check.py <dir with
faker>`. Faker is installed outside the project venv, because modelling does not need it.

### 3.3 The full pipeline anyway

We treated the file exactly like a real dataset:

* **Dropped**: `User ID` and `Username` (identifiers, contract rule 1), and `Location`
  (25,199 free-text values, median 2 rows each; its Faker prefix and suffix are unrelated to the
  label, p = 0.97 and 0.72). `Created At` was decomposed into hour, day of week and days since
  2020-01-01.
* **Engineered**: tweet length in characters and words, `has_hashtags` and `hashtag_count`. A
  NULL `Hashtags` means the tweet had no hashtags, so 0 is the true value here, not an
  imputation. `Verified` became 0/1.
* **Text, as Worksheet 3 specified**: TF-IDF (`RegexTokenizer` → `CountVectorizer`(vocab
  1,000, minDF 5) → `IDF`) on both `Tweet` and `Hashtags`. Fitted inside each training fold
  only. Together with the 11 numeric features this gives **1,949 dimensions**. This answers the
  objection "you just didn't use the text".
* **Split**: stratified 80:20, seed 42, giving 40,128 train and 9,872 test. Class weights came
  out at about 1.0, because the classes are balanced.
* **Tuning**: 3-fold CV inside train. Random Forest `maxDepth` ∈ {8, 12, 16, 20} gave CV PR-AUC
  0.501 / 0.500 / 0.500 / 0.499 (SD ≈ 0.007). LR `regParam` ∈ {0.001, 0.01, 0.1} gave 0.498 for
  every value. **No setting finds anything**, and the CV numbers equal the positive rate.

Test results:

| | Accuracy (95% CI) | Balanced acc. | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| Majority baseline | 0.501 | 0.5 | – | 0 | – | 0 | 0.5 | 0.499 |
| Random Forest | 0.494 (0.485–0.504) | 0.494 | 0.494 | 0.491 | 0.492 | −0.011 | 0.492 | 0.490 |
| Logistic Regression | 0.499 (0.489–0.509) | 0.499 | 0.498 | 0.497 | 0.498 | −0.002 | 0.495 | 0.494 |

Both ROC-AUCs sit inside the **95% band expected from a model with no signal at all
(0.489–0.511)**. Both accuracy intervals contain the baseline.

**On "below 50% is unacceptable".** 0.494 is below 50%. That does not mean the model is broken
or inverted. With no signal, test accuracy scatters around 50% by about ±1%. We report the
number as it came out rather than re-running seeds until it crossed 0.5. The right reading is
"indistinguishable from chance". It is not "slightly worse than chance".

### 3.4 What the importance plots would have told an unwary reader

See [`outputs/figures/twitter_synthetic_feature_importance.png`](../outputs/figures/twitter_synthetic_feature_importance.png). Random Forest **impurity
importance gives the TF-IDF terms 90% of the forest** (tweet terms 0.569, hashtag terms 0.335).
Read naively, this says "the text is what identifies bots". **Permutation importance on the test
set says nothing matters.** The drop in ROC-AUC when the tweet text is shuffled is
−0.006 ± 0.004 (shuffling made the model marginally *better*). For the hashtags it is
+0.002 ± 0.004. Both are noise. Logistic Regression shows the same thing: most of its ten
largest coefficients sit on random words such as `hashtags:might`, `hashtags:fear` and
`tweet:finish`.

This is the bias Worksheet 3 warned about, shown on data where the right answer is known.
Impurity importance rewards features that offer many possible split points: 1,938 sparse text
columns, or a continuous count. It does so whether or not the splits generalise. This is why
the real dataset's rankings below are checked with permutation importance.

### 3.5 Conclusion on the synthetic dataset

No leakage was found, and there was nothing to leak. The dataset is unusable for bot detection.
The reason is that its labels are independent of everything in it. The cause is not a column
we failed to engineer. It stays in the repository as a negative control and is **not** used
for the X/Twitter result or the cross-platform comparison.

---

## 4. Dataset 2: real accounts (the X/Twitter result)

### 4.1 Provenance

`data/landing/twitter_human_bots/twitter_human_bots_dataset.csv`, Kaggle
`danieltreiman/twitter-human-bots-dataset`. It has 37,438 accounts with fields in the shape of
Twitter API v1.1 user objects. The `profile_images/` folders were ignored. `created_at +
account_age_days` falls between **2020-08-18 10:55 and 2020-08-19 10:58 for every row**, so the
metadata is a single snapshot taken over about 24 hours. The `bot`/`human` labels come from the
dataset author's sources, not from Twitter. We could not check how they were annotated from the
files on the box (see limitations). Spark's multi-line CSV parse is cross-checked against an
independent pandas read on every run: same row count, same number of bots.

### 4.2 Class balance

Of the 37,438 accounts, 25,013 are human (0 = Authentic) and 12,425 are bots
(1 = Inauthentic), a positive rate of 33.2%. The imbalance is real but moderate. The majority
baseline is 66.6%, so **accuracy alone would flatter any model**. The headline therefore leans
on balanced accuracy, F1, MCC and PR-AUC. Class weights were computed on train with
n / (2·n_class): 0.748 for humans and 1.509 for bots.

### 4.3 What we dropped and why

| Column | Reason |
|---|---|
| `id`, `screen_name`, unnamed row index | identifiers (contract rule 1) |
| `profile_image_path`, `profile_background_image_path` | local download paths, NULL for 35% of rows. **The NULLs depend on the label: 45.7% of humans but 12.2% of bots.** That is a download-completeness artefact that would leak the label (single-flag AUC 0.67) |
| `profile_image_url` | URL string, no behaviour (1 NULL) |
| `profile_background_image_url` | URL string. Its NULLs occur only for default-profile accounts created from 2015 on (0% before 2015, 80% in 2016–17), so they encode era plus `default_profile`, which are already present |
| `split` | the vendor's own train/val/test partition, not an account property |
| `statuses_count` | = `average_tweets_per_day × account_age_days` by construction (section 4.5) |
| `created_at` | = collection date − `account_age_days` by construction (r = −1.0000) |
| raw `description`, `location`, `lang` | replaced by the engineered features below |

### 4.4 What we engineered, and how missing values were handled

| Feature | Definition |
|---|---|
| `log_followers_count`, `log_friends_count`, `log_favourites_count`, `log_average_tweets_per_day` | ln(1 + x). The counts span 0 to 10⁸, so the logs keep LR well-behaved. RF splits are unaffected by monotone transforms |
| `follower_share` | (followers + 1) / (followers + friends + 2). This is the **followers/friends ratio** in bounded form, r/(1+r). The +1/+2 smoothing guards against division by zero: an account with no followers and no friends gets 0.5 |
| `account_age_days` | as given |
| `has_description`, `description_length` | An empty bio arrives as NULL. Length 0 is the true length of an empty bio, so this is not a zero-fill of unknown data |
| `has_location` | 1 if set and not the literal `unknown`, which is the vendor's placeholder for an empty field and covers 34.8% of rows |
| `lang` (one-hot) | **21.3% NULL, kept as its own `__missing__` level**, never zero-filled or dropped. Levels with fewer than 30 training accounts become `__other__`. The level list (31) is learned from train only |
| `default_profile`, `default_profile_image`, `geo_enabled`, `verified` | 'True'/'False' → 1/0 |

After these rules, every model input is asserted NULL-free. Nothing is silently filled. The
missingness is itself informative. A missing `lang` affects 13.8% of humans but 36.2% of bots,
and an empty bio affects 11.6% against 35.0%. That is why missingness is encoded rather than
imputed away.

### 4.5 The construction trap: three columns that are one

`statuses_count`, `average_tweets_per_day` and `account_age_days` are tied by definition:

* `statuses_count = rate × age` holds **within rounding for 100% of training rows** (median
  relative error 0.013%).
* In log space, ln(statuses) = 1.0003·ln(rate) + 1.0000·ln(age), with **R² = 0.99993**. The
  variance inflation factor (VIF) of ln(statuses) is therefore about 14,000.

This is collinearity, not label leakage: no column encodes the label. It still does damage, and
we measured the damage rather than asserting it:

* **Logistic Regression.** Over 5 bootstrap resamples of train with all three columns kept,
  `statuses_count` gets a *negative* coefficient (−0.086 ± 0.023). Meanwhile the rate
  coefficient inflates (0.652 against 0.589) and its bootstrap SD rises by 42% (0.039 against
  0.027). The two partly cancel each other, which is the textbook symptom.
* **Random Forest impurity importance.** The redundant column takes 0.070 of the importance
  (rank 5) and pushes the rate down from 0.092 to 0.074. The trio's combined importance grows
  from 0.151 to 0.199 without adding any information. That is credit being double-counted.
* **Handling.** We dropped `statuses_count` and kept the rate (activity intensity) and the age
  (account maturity). These two are almost unrelated, with VIF 1.67 and 1.33 in the model.
  Checked on a validation split *inside* train, so the test set played no part, the drop costs
  RF 0.0025 ROC-AUC (0.9241 against 0.9266). Trees cannot multiply, so the product helps them
  slightly. We accept that cost to keep LR coefficients interpretable and importances honest.

**A second instance, caused by our own code and caught by the same check.** Our first version
of the ratio feature was ln((followers + 1)/(friends + 1)). That is *exactly*
`log_followers − log_friends`, and the VIF check reported it as infinite. The fix was the
bounded `follower_share` above, which keeps the same ordering, so Random Forest splits are
unchanged. All VIFs in the final feature set are below 6. We recorded this because it is the
same trap the worksheet warned about, and we were not immune to it.

### 4.6 Split and tuning

The split is stratified 80:20 with seed 42 via `common.stratified_split`: 29,964 train and 7,474
test. Tuning used 3-fold cross-validation **inside train only**, scored on PR-AUC.

* **Random Forest** (numTrees 150, `sqrt` features, gini, maxBins 32, minInstancesPerNode 5,
  class weights). `maxDepth` 8 / 12 / 16 / 20 gave CV PR-AUC 0.840 / 0.870 / 0.878 / 0.879.
  The curve has flattened, and the last step (+0.0015) is smaller than the fold SD (0.004). We
  selected 20.
* **Logistic Regression** (L2, standardised inputs, class weights). `regParam` 0.001 / 0.01 / 0.1
  gave 0.721 / 0.720 / 0.708. We selected 0.001.

The test set was scored once, after the refit on all of train.

### 4.7 Headline results

| | Accuracy (95% CI) | Balanced acc. | Precision | Recall | Specificity | F1 | MCC | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|
| Majority baseline | 0.666 | 0.5 | – | 0 | 1 | – | 0 | 0.5 | 0.334 |
| **Random Forest** | **0.854** (0.846–0.862) | 0.841 | 0.769 | 0.802 | 0.880 | 0.785 | 0.675 | 0.923 | 0.880 |
| Logistic Regression | 0.788 (0.778–0.797) | 0.790 | 0.648 | 0.797 | 0.783 | 0.715 | 0.556 | 0.846 | 0.711 |

Both models clearly beat the baseline, and the Random Forest clearly beats LR: the confidence
intervals do not overlap. The gap means the signal is non-linear. A threshold on follower
counts means something different for a verified account than for a default profile, and trees
can express that.

**The confusion matrix, from a moderator's point of view** (RF, test set, threshold 0.5, see
[`outputs/figures/twitter_real_confusion_matrix.png`](../outputs/figures/twitter_real_confusion_matrix.png)):

* 2,000 of 2,493 bots are flagged (recall 80%). 493 bots are missed.
* **600 of 4,981 real people are flagged (12.0%).** Of the 2,600 flagged accounts, 600 (23%)
  are humans. LR flags 1,080 humans (21.7%). A false positive is a person whose account would
  be put in front of a moderator for no reason. This is why nothing is ever actioned
  automatically.

**Used as a ranked queue** instead of a yes/no threshold, the scores are much cleaner at the top:

| Moderators review the top… | Accounts | Bots among them (RF) | Share of all bots found | LR, same budget |
|---|---|---|---|---|
| 5% | 374 | **99.5%** (2 humans) | 14.9% | 81.3% |
| 10% | 747 | **98.3%** (13 humans) | 29.4% | 76.3% |
| 20% | 1,495 | **92.0%** (119 humans) | 55.2% | 74.8% |

The base rate is 33.4%. For a team with limited review capacity, this ranking is the useful
output. Even so, the 2–119 humans at the top of the queue deserve a careful look, not a
presumption.

**Sanity check on the vendor's own split** (train on `split = 0`, evaluate on `split ∈ {1, 2}`,
same hyperparameters, no re-tuning): RF accuracy 0.855, MCC 0.677, ROC-AUC 0.925. LR accuracy
0.795. These agree with the stratified split to within 0.01.

### 4.8 What the model leans on, and why we use two importance measures

See [`outputs/figures/twitter_real_feature_importance.png`](../outputs/figures/twitter_real_feature_importance.png).

| Rank | RF impurity importance (train) | RF permutation importance (test ROC-AUC drop) | LR coefficient (per 1 SD) |
|---|---|---|---|
| 1 | log_followers_count 0.221 | **log_favourites_count 0.056** | log_followers_count −0.89 |
| 2 | log_favourites_count 0.167 | log_followers_count 0.047 | log_average_tweets_per_day **+0.57** |
| 3 | log_friends_count 0.144 | log_friends_count 0.027 | log_favourites_count −0.49 |
| 4 | log_average_tweets_per_day 0.092 | log_average_tweets_per_day 0.026 | log_friends_count −0.43 |
| 5 | verified 0.064 | account_age_days 0.017 | geo_enabled −0.41 |
| 6 | follower_share 0.060 | verified 0.015 | has_description −0.36 |
| 7 | account_age_days 0.058 | follower_share 0.010 | verified −0.35 |
| 8 | geo_enabled 0.050 | lang (all levels) 0.009 | lang=ru −0.24 |
| 9 | description_length 0.041 | geo_enabled 0.008 | lang=ar +0.18 |

**Where the two RF views disagree, we trust permutation.** Impurity importance is measured on
training data and favours continuous, many-valued columns. Section 3.4 showed it giving 90% of
the credit to pure noise. Permutation importance asks a direct question on held-out data: how
much worse do the predictions get if this column is scrambled? Two shifts matter here:

* `log_favourites_count` (likes the account has given) moves from #2 to **#1**.
* `description_length` falls from #9 (0.041) to #12 (0.002). Its impurity score was mostly
  split-point opportunity: a continuous length has many possible cut points.

Permutation importance has its own blind spot. It understates features that share information
with others. Scrambling `has_description` costs nothing (−0.0001), because `description_length`
= 0 carries the same fact. LR, which cannot exploit that redundancy in the same way, gives
`has_description` a clear coefficient (−0.36). Shared information also splits the credit
among followers, friends and `follower_share`.

**What the signals mean behaviourally.** These are medians in this sample, bots against humans:

* **Engagement:** bots have given far fewer likes (1,131 against 3,649). A human account is
  used, so it likes things.
* **Reach:** bots have few followers (25 against 1,046) and follow few accounts (31 against
  474). Many bots in this collection are quiet or dormant accounts, not "follow-everyone"
  spammers.
* **Activity:** marginally, bots in this sample post *less* per day (0.8 against 2.4). But
  conditional on the other features, LR gives the posting rate a **positive** coefficient. The
  bot pattern is a high posting rate *for an account nobody follows and that likes nothing*.
* **Trust markers:** only 4.2% of verified accounts are bots. Accounts with geolocation
  enabled are bots 16% of the time, against 48% for those without. Bots are younger (median
  2,882 days against 3,395 days) and more often leave the language, bio and location empty.

### 4.9 Leakage audit

The headline result is not near 99%, so the contract did not require a ceiling audit. We ran
one anyway:

| Check | Outcome |
|---|---|
| Identifiers out of the feature set | `id`, `screen_name` and the row index dropped |
| Is the file sorted by label? | No. Row order against label gives r = −0.001. Adjacent-row label agreement is 0.562 against 0.557 expected if shuffled |
| Does an ID track the label? | `id` has Spearman 0.23 with the label. This is **explained, not leakage**: Twitter ids are issued in time order (Spearman 0.9999 with `created_at`) and bots here are younger. The information is already in `account_age_days`. `id` is excluded |
| Duplicates | 0 duplicate rows. 1 test account has a feature vector identical to a train account |
| Any single feature give the answer away? | Best single-feature train AUC is 0.810 (followers). Nothing approaches 0.95 |
| Label-dependent missingness in artefact columns | **Found:** image-download paths are missing for 45.7% of humans against 12.2% of bots. Dropped (section 4.3) |
| Construction redundancy | statuses = rate × age, and created_at = collection date − age. Handled (section 4.5) |
| Encoders and vocabularies fitted only on train | `lang` levels, scaler and one-hot all fitted on train |
| A different partition | The vendor split gives the same result (0.855) |

The only real leakage path in the file was the profile-image download columns. It was removed
before any model was fitted.

### 4.10 Limitations a reader must know before trusting the numbers

* **"Bot" is the dataset author's label, not a platform verdict.** We could not check the
  original annotation procedure from the files we have. It is also **not the same concept** as
  Facebook "spam" or Instagram "fake". The shared `0 = Authentic / 1 = Inauthentic` target is
  an operational convention, so cross-platform comparisons must be read with that in mind.
* **A 2020 snapshot of the old Twitter.** The metadata was collected on 18–19 August 2020 via
  the v1.1 API. The platform (now X), its verification system and bot operators have all changed
  since. Expect lower performance on today's accounts.
* **These bots are a particular kind of bot:** young, unfollowed, quiet, with empty profiles.
  An operator who fills in a profile and buys followers would not look like them. The model has
  learned *this collection's* bots, not bots in general.
* **Profile metadata only.** There is no tweet content, timing or network structure. Those are
  the signals that catch coordinated or sophisticated automation.
* **Missing `lang` is informative** (36% of bots against 14% of humans). Part of that may
  reflect how the author collected the data rather than how accounts behave.
* **The threshold is a policy choice.** Class weighting at 0.5 favours recall (80%) over
  precision (77%). A moderation team should choose its own threshold or review budget. The
  ranked queue in section 4.7 is the recommended way to use the scores.
* **Scores are a review queue, not proof of intent, and never grounds for automated action.**

---

## 5. Side by side: why the original had to be set aside

| | Synthetic (`goyaladi`) | Real (`danieltreiman`) |
|---|---|---|
| Unit of observation | a tweet, each with its own one-off user | an account |
| Text | Faker `lorem` words (exactly 971, a closed list) | not used (profile metadata) |
| Numeric distributions | uniform, identical for both classes (`twitter_distributions.png`, top row) | heavy-tailed, and the classes separate (bottom row) |
| Best single feature, train AUC | 0.504 | 0.810 |
| Cross-validated PR-AUC, best RF | 0.501 (= positive rate) | 0.879 (positive rate 0.334) |
| RF test MCC | −0.011 | 0.675 |
| Bots among the top-5% scores | 45.3% (base 49.9%) | 99.5% (base 33.4%) |
| Impurity vs. permutation importance | impurity: 90% text; permutation: nothing | broadly agree, with an informative reorder |
| Collection date / method | none stated; uniform 2020–2023 timestamps | API snapshot, 18–19 Aug 2020 |
| Usable for the study? | **No** (negative control) | **Yes** (the X/Twitter result) |

[`outputs/figures/twitter_side_by_side.png`](../outputs/figures/twitter_side_by_side.png) shows the same comparison as bars, with what
no-signal guessing would score marked on each metric.

The decision does not rest on the synthetic dataset scoring lower. It rests on the evidence in
section 3.2 that the file **contains no relationship between its features and its label at
all**. No modelling choice could change that, and a better score on it would have been a sign of
a bug. Keeping it in the report shows that the group checked its data before trusting a number.
It also shows that the checks distinguish "the model is weak" from "there is nothing to learn".

---

## 6. Reproduce

```bash
# dari root repo; Java 17 di PATH atau JAVA_HOME di-set (lihat README)
python src/twitter_pipeline.py                 # both datasets, about 19 min on this box
python src/twitter_pipeline.py --figures-only  # redraw figures from the JSON, no Spark
```

Spark runs in local mode with 3 cores and 2 GB of driver memory, per the contract. The seed is
42 everywhere. The final run was launched from an emptied environment (`env -i`, keeping only
`HOME` and a basic `PATH`). It ran the first three lines above plus the first `python` command,
and it completed without error.

## 7. Files

| Path | Content |
|---|---|
| `src/twitter_pipeline.py` | end-to-end pipeline for both datasets |
| `src/twitter_faker_check.py` | optional Faker provenance check → [`outputs/metrics/twitter_synthetic_faker_check.json`](../outputs/metrics/twitter_synthetic_faker_check.json) |
| [`outputs/metrics/twitter_real.json`](../outputs/metrics/twitter_real.json), `twitter_synthetic.json` | contract-shaped metrics, plus `dataset_validation`, `vendor_split_check`, permutation importance, CV tables and the collinearity diagnostics |
| `outputs/predictions/twitter_real_scored.csv` | 7,474 held-out accounts: `id`, `screen_name`, label, RF and LR probability and prediction, sorted by RF score |
| `outputs/predictions/twitter_synthetic_scored.csv` | the same for the 9,872 held-out synthetic rows |
| `outputs/figures/twitter_{real,synthetic}_confusion_matrix.png` | confusion matrices, RF and LR |
| `outputs/figures/twitter_{real,synthetic}_feature_importance.png` | impurity, permutation and LR coefficients, on a shared permutation scale |
| [`outputs/figures/twitter_side_by_side.png`](../outputs/figures/twitter_side_by_side.png), `twitter_distributions.png` | the two-dataset comparison |

The scored CSVs contain **held-out test rows only**, so every score is out-of-sample. Scoring
the training rows would give a misleadingly confident queue.

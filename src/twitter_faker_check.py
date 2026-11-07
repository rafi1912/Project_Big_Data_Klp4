"""Optional provenance check for the synthetic X/Twitter dataset (negative control).

Compares the vocabulary of `Tweet` / `Hashtags` and the shape of `Location`
in data/landing/twitter_bot_detection/bot_detection_data.csv with the word and place-name lists
shipped in the open-source `Faker` library (en_US locale). Writes
outputs/metrics/twitter_synthetic_faker_check.json, which twitter_pipeline.py
embeds in the synthetic dataset's `dataset_validation` block when present.

Faker is listed in requirements.txt. Run from the repo root:

    python src/twitter_faker_check.py

An optional first argument adds a directory to sys.path (e.g. a Faker
installed with pip --target) for environments without Faker.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import pandas as pd  # noqa: E402


def compare(d: pd.DataFrame, word_list: set, prefixes: set, suffixes: set) -> dict:
    """Compare one dataset frame with Faker's lists. Pure: no Faker import, no file I/O."""
    tweet_vocab = {w for t in d["Tweet"].str.lower() for w in re.findall(r"[a-z]+", t)}
    tag_vocab = {w for t in d["Hashtags"].fillna("").str.lower() for w in re.findall(r"[a-z]+", t)}
    loc = d["Location"]
    first = loc.str.split().str[0]
    has_prefix = first.isin(prefixes) & (loc.str.split().str.len() > 1)
    has_suffix = loc.apply(lambda s: any(s.endswith(x) for x in suffixes))
    return {
        "faker_en_US_lorem_word_list_size": len(word_list),
        "tweet_vocabulary_size": len(tweet_vocab),
        "tweet_words_in_faker_list": len(tweet_vocab & word_list),
        "tweet_words_not_in_faker_list": sorted(tweet_vocab - word_list),
        "faker_words_absent_from_tweets": len(word_list - tweet_vocab),
        "hashtag_vocabulary_size": len(tag_vocab),
        "hashtag_words_in_faker_list": len(tag_vocab & word_list),
        "location_share_with_faker_city_prefix": round(float(has_prefix.mean()), 4),
        "location_share_with_faker_city_suffix": round(float(has_suffix.mean()), 4),
        "location_share_with_prefix_or_suffix": round(float((has_prefix | has_suffix).mean()), 4),
        "verdict": ("Tweet and Hashtags vocabularies are exactly Faker's en_US lorem word list"
                    if tweet_vocab == word_list and tag_vocab <= word_list
                    else "vocabulary differs from Faker's list"),
    }


def main() -> None:
    if len(sys.argv) > 1:
        sys.path.insert(0, sys.argv[1])
    import faker
    from faker.providers.address.en_US import Provider as Address
    from faker.providers.lorem.en_US import Provider as Lorem

    path = common.landing_path("twitter_bot_detection") / "bot_detection_data.csv"
    d = pd.read_csv(path, keep_default_na=False, na_values=[""])
    word_list = {w.lower() for w in Lorem.word_list}
    out = {"faker_version": faker.VERSION,
           **compare(d, word_list, set(Address.city_prefixes), set(Address.city_suffixes))}
    dest = common.OUT_METRICS / "twitter_synthetic_faker_check.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()

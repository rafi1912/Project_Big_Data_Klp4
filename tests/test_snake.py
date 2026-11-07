"""Regresi bug tabrakan nama kolom: '#tags/post' dan 'tags/post' harus tetap berbeda."""
from common import snake, snake_all

FACEBOOK_HEADER = ["profile id", "#friends", "#following", "#community", "age", "#postshared",
                   "#urlshared", "#photos/videos", "fpurls", "fpphotos/videos", "avgcomment/post",
                   "likes/post", "tags/post", "#tags/post", "Label"]


def test_hash_prefix_becomes_n():
    assert snake("#friends") == "n_friends"
    assert snake("#photos/videos") == "n_photos_videos"


def test_hash_and_plain_column_stay_distinct():
    assert snake("tags/post") == "tags_post"
    assert snake("#tags/post") == "n_tags_post"


def test_existing_names_unchanged():
    # nama tanpa '#' di awal harus tetap sama seperti sebelum perbaikan
    assert snake("Bot Label") == "bot_label"
    assert snake("User ID") == "user_id"
    assert snake("followersCount") == "followers_count"
    assert snake("profile id") == "profile_id"


def test_facebook_header_has_no_collisions():
    names = snake_all(FACEBOOK_HEADER)
    assert len(set(names)) == len(names)
    assert {"tags_post", "n_tags_post", "n_friends", "label"} <= set(names)


def test_snake_all_deduplicates():
    assert snake_all(["a b", "a-b", "a_b"]) == ["a_b", "a_b_1", "a_b_2"]
    # akhiran tidak boleh menabrak nama yang sudah ada
    assert snake_all(["x", "x", "x_1"]) == ["x", "x_1", "x_1_1"]
    assert snake_all(["", "!!"]) == ["col", "col_1"]


def test_snake_all_keeps_order_and_length():
    names = snake_all(FACEBOOK_HEADER)
    assert len(names) == len(FACEBOOK_HEADER)
    assert names[0] == "profile_id" and names[-1] == "label"


def test_standardize_on_facebook_like_header(spark):
    """Sebelum perbaikan: standardize() gagal dengan AMBIGUOUS_REFERENCE."""
    from common import standardize
    df = spark.createDataFrame([(1, 3.0, 4.0, 0), (2, 5.0, 6.0, 1)],
                               ["profile id", "tags/post", "#tags/post", "Label"])
    ds = {"label_column": "Label", "positive_values": [1], "id_columns": []}
    out = standardize(df, ds)
    assert {"tags_post", "n_tags_post", "label"} <= set(out.columns)
    assert out.select("tags_post", "n_tags_post").collect()[0].asDict() == {"tags_post": 3.0, "n_tags_post": 4.0}

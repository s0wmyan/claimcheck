from claimcheck.fusion import reciprocal_rank_fusion


def test_orders_by_sum_of_reciprocal_ranks():
    # with k=60:  a = 1/61 + 1/63,  b = 1/62 + 1/61,  c = 1/63 + 1/62
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]])
    assert fused == ["b", "a", "c"]


def test_document_in_only_one_list_is_kept():
    fused = reciprocal_rank_fusion([["a", "b"], ["c"]])
    assert set(fused) == {"a", "b", "c"}
    assert fused[-1] == "b"


def test_document_in_both_lists_beats_top_of_one_list():
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]])
    assert fused[0] == "b"

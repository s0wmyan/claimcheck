from claimcheck.data import label_from_evidence


def test_claim_without_evidence_is_not_enough_info():
    assert label_from_evidence({}) == "NOT_ENOUGH_INFO"


def test_supporting_evidence_means_supported():
    evidence = {"14717500": [{"sentences": [2, 5], "label": "SUPPORT"}]}
    assert label_from_evidence(evidence) == "SUPPORTED"


def test_contradicting_evidence_means_refuted():
    evidence = {"13734012": [{"sentences": [4], "label": "CONTRADICT"}]}
    assert label_from_evidence(evidence) == "REFUTED"

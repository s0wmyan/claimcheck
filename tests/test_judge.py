from claimcheck.judge import build_prompt, parse_verdict

ABSTRACTS = [
    {"title": "Aspirin trial", "text": "Fewer strokes occurred with aspirin."},
    {"title": "Statin cohort", "text": "Statins lowered cholesterol."},
]


def test_parse_reads_verdict_and_reason():
    raw = '{"reason": "The trial found fewer strokes.", "verdict": "SUPPORTED"}'
    assert parse_verdict(raw) == ("SUPPORTED", "The trial found fewer strokes.")


def test_parse_treats_invalid_json_as_not_enough_info():
    assert parse_verdict("I think it is true")[0] == "NOT_ENOUGH_INFO"


def test_parse_treats_unknown_verdict_as_not_enough_info():
    raw = '{"reason": "Unclear.", "verdict": "MAYBE"}'
    assert parse_verdict(raw)[0] == "NOT_ENOUGH_INFO"


def test_prompt_contains_the_claim_and_every_abstract():
    prompt = build_prompt("Aspirin prevents stroke.", ABSTRACTS)
    assert "Aspirin prevents stroke." in prompt
    for abstract in ABSTRACTS:
        assert abstract["title"] in prompt
        assert abstract["text"] in prompt


def test_closed_book_prompt_contains_the_claim():
    assert "Aspirin prevents stroke." in build_prompt("Aspirin prevents stroke.", [])


def test_prompt_cuts_long_abstracts_to_the_word_limit():
    long_text = " ".join(f"w{i}" for i in range(400))
    prompt = build_prompt("Claim.", [{"title": "Long", "text": long_text}], max_words=300)
    assert "w299" in prompt
    assert "w300" not in prompt

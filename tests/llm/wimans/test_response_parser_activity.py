import pytest

from src.llm.wimans.response_parser_activity import (
    parse_response_activity,
)


def test_valid_activity_response():
    result = parse_response_activity(
        '{"activity":"walk","confidence":0.85,'
        '"reason":"Sustained temporal changes."}'
    )

    assert result.status == "ok"
    assert result.activity == "walk"
    assert result.confidence == 0.85
    assert result.reason == "Sustained temporal changes."


def test_nothing_is_valid():
    result = parse_response_activity(
        '{"activity":"nothing","confidence":0.9,'
        '"reason":"The person appears inactive."}'
    )

    assert result.status == "ok"
    assert result.activity == "nothing"


def test_invalid_activity_rejected():
    result = parse_response_activity(
        '{"activity":"running","confidence":0.8,'
        '"reason":"Movement detected."}'
    )

    assert result.status == "schema_error"


def test_extra_presence_key_rejected():
    result = parse_response_activity(
        '{"present":true,"activity":"walk","confidence":0.8,'
        '"reason":"Movement detected."}'
    )

    assert result.status == "schema_error"


def test_extra_activity_counts_key_rejected():
    result = parse_response_activity(
        '{"activity":"walk","confidence":0.8,'
        '"reason":"Movement detected.",'
        '"predicted_activity_counts":{}}'
    )

    assert result.status == "schema_error"


@pytest.mark.parametrize("confidence", ["true", "-0.1", "1.1", "null"])
def test_invalid_confidence_rejected(confidence):
    result = parse_response_activity(
        '{"activity":"walk","confidence":'
        + confidence
        + ',"reason":"Movement detected."}'
    )

    assert result.status in {"schema_error", "value_error"}


def test_empty_reason_rejected():
    result = parse_response_activity(
        '{"activity":"walk","confidence":0.8,"reason":"  "}'
    )

    assert result.status == "schema_error"


def test_empty_response_rejected():
    result = parse_response_activity("")

    assert result.status == "empty"


def test_invalid_json_rejected():
    result = parse_response_activity('{"activity":')

    assert result.status == "json_error"


def test_markdown_fenced_json():
    result = parse_response_activity(
        '```json\n{"activity":"jump","confidence":0.7,'
        '"reason":"Temporal changes suggest movement."}\n```'
    )

    assert result.status == "ok"
    assert result.activity == "jump"

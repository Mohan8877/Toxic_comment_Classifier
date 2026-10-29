from frontend.app import get_label_order, normalize_categories, validate_prediction_payload


def test_get_label_order_returns_expected_sequence():
    categories = {
        "toxic": 0.9,
        "severe_toxic": 0.1,
        "obscene": 0.2,
        "threat": 0.3,
        "insult": 0.4,
        "identity_hate": 0.5,
    }
    assert get_label_order(categories) == [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]


def test_normalize_categories_validates_range_and_type():
    payload = {
        "toxic": 0.1,
        "severe_toxic": 0.2,
        "obscene": 0.3,
        "threat": 0.4,
        "insult": 0.5,
        "identity_hate": 0.6,
    }
    normalized = normalize_categories(payload)
    assert normalized["toxic"] == 0.1
    assert normalized["identity_hate"] == 0.6


def test_validate_prediction_payload_accepts_valid_contract():
    payload = {
        "is_toxic": True,
        "toxicity_score": 0.72,
        "moderation_action": "flag",
        "model_version": "distilbert_tuned_experimental",
        "categories": {
            "toxic": 0.8,
            "severe_toxic": 0.1,
            "obscene": 0.2,
            "threat": 0.3,
            "insult": 0.4,
            "identity_hate": 0.5,
        },
    }
    validated = validate_prediction_payload(payload)
    assert validated["is_toxic"] is True
    assert validated["categories"]["toxic"] == 0.8
    assert validated["model_version"] == "distilbert_tuned_experimental"


def test_validate_prediction_payload_rejects_missing_or_invalid_fields():
    try:
        validate_prediction_payload({"categories": {"toxic": 1.0}})
        assert False, "Expected ValueError for missing fields"
    except ValueError:
        pass

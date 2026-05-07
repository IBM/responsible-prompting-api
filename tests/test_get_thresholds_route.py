import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import app as app_module


def test_get_thresholds_requires_prompts():
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    response = client.get("/get_thresholds")

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Missing required query parameter: prompts"
    }


def test_get_thresholds_uses_prompt_samples_and_embedding_function(monkeypatch):
    captured = {}

    def fake_populate_json():
        return {"positive_values": [], "negative_values": []}

    def fake_get_embedding_func(inference, model_id):
        captured["inference"] = inference
        captured["model_id"] = model_id
        return "embedding-fn"

    def fake_get_thresholds(prompts, prompt_json, embedding_fn):
        captured["prompts"] = prompts
        captured["prompt_json"] = prompt_json
        captured["embedding_fn"] = embedding_fn
        return {
            "add_lower_threshold": 0.3,
            "add_higher_threshold": 0.5,
            "remove_lower_threshold": 0.1,
            "remove_higher_threshold": 0.5,
        }

    monkeypatch.setattr(
        app_module.recommendation_handler,
        "populate_json",
        fake_populate_json,
    )
    monkeypatch.setattr(
        app_module.recommendation_handler,
        "get_embedding_func",
        fake_get_embedding_func,
    )
    monkeypatch.setattr(
        app_module.recommendation_handler,
        "get_thresholds",
        fake_get_thresholds,
    )

    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    response = client.get(
        "/get_thresholds",
        query_string=[
            ("prompts", "first prompt"),
            ("prompts", "  "),
            ("prompts", "second prompt"),
        ],
        headers={"model_id": "custom-model"},
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "add_lower_threshold": 0.3,
        "add_higher_threshold": 0.5,
        "remove_lower_threshold": 0.1,
        "remove_higher_threshold": 0.5,
    }
    assert captured == {
        "inference": "local",
        "model_id": "custom-model",
        "prompts": ["first prompt", "second prompt"],
        "prompt_json": {"positive_values": [], "negative_values": []},
        "embedding_fn": "embedding-fn",
    }


def test_get_thresholds_defaults_to_local_model(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        app_module.recommendation_handler,
        "populate_json",
        lambda: {},
    )

    def fake_get_embedding_func(inference, model_id):
        captured["model_id"] = model_id
        return "embedding-fn"

    monkeypatch.setattr(
        app_module.recommendation_handler,
        "get_embedding_func",
        fake_get_embedding_func,
    )
    monkeypatch.setattr(
        app_module.recommendation_handler,
        "get_thresholds",
        lambda prompts, prompt_json, embedding_fn: {
            "add_lower_threshold": 0.3,
            "add_higher_threshold": 0.5,
            "remove_lower_threshold": 0.1,
            "remove_higher_threshold": 0.5,
        },
    )

    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    response = client.get(
        "/get_thresholds",
        query_string={"prompts": "prompt sample"},
    )

    assert response.status_code == 200
    assert captured["model_id"] == "./models/all-MiniLM-L6-v2/"

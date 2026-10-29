import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Health and Basic Endpoint Tests
# ---------------------------------------------------------------------------
def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["engine"] in {"rules+model", "rules only"}
    assert "model_loaded" in data


def test_index_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Grammar Autocorrector" in response.text


def test_form_submission():
    response = client.post("/correct", data={"text": "She go to college."})
    assert response.status_code == 200
    assert "She goes to college." in response.text
    assert "subject_verb_agreement" in response.text


def test_form_submission_empty():
    response = client.post("/correct", data={"text": ""})
    assert response.status_code == 400
    assert "Please enter some text to correct" in response.text


# ---------------------------------------------------------------------------
# 2. API Input Validation Tests
# ---------------------------------------------------------------------------
def test_api_empty_text():
    response = client.post("/api/correct", json={"text": "   "})
    assert response.status_code == 400
    assert "Input text cannot be empty" in response.json()["detail"]


def test_api_text_too_long():
    long_text = "word " * 500  # 2500 characters
    response = client.post("/api/correct", json={"text": long_text})
    assert response.status_code == 400
    assert "exceeds the maximum limit of 2000 characters" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 3. 8 Report Test Cases via POST /api/correct
# ---------------------------------------------------------------------------
REPORT_CASES = [
    ("She go to college every day.", "She goes to college every day.", "subject_verb_agreement"),
    ("He eat an apple.", "He eats an apple.", "subject_verb_agreement"),
    ("They goes to school.", "They go to school.", "subject_verb_agreement"),
    ("She are my friend.", "She is my friend.", "is_are"),
    ("They is playing football.", "They are playing football.", "is_are"),
    ("He have a car.", "He has a car.", "has_have"),
    ("I has a book.", "I have a book.", "has_have"),
    ("She is a engineer.", "She is an engineer.", "a_an"),
]


@pytest.mark.parametrize("input_text, expected_corrected, expected_rule", REPORT_CASES)
def test_api_report_cases(input_text, expected_corrected, expected_rule):
    response = client.post("/api/correct", json={"text": input_text})
    assert response.status_code == 200
    data = response.json()

    assert data["original"] == input_text
    assert data["corrected"] == expected_corrected
    assert data["error_count"] >= 1
    assert data["engine"] in {"rules+model", "rules only"}
    assert any(e["rule"] == expected_rule for e in data["edits"])

    # Verify structured fields
    assert isinstance(data["tokens"], list)
    assert isinstance(data["pos_tags"], list)
    for edit in data["edits"]:
        assert "start" in edit
        assert "end" in edit
        assert "old" in edit
        assert "new" in edit
        assert "rule" in edit


# ---------------------------------------------------------------------------
# 4. Phase 2 Grammar Cases via POST /api/correct
# ---------------------------------------------------------------------------
ADDITIONAL_API_CASES = [
    # does/do mismatches
    ("He do not understand.", "He does not understand."),
    ("They does not like spicy food.", "They do not like spicy food."),

    # a/an phonetic cases
    ("Give me a apple.", "Give me an apple."),
    ("He waited for a hour.", "He waited for an hour."),
    ("She attends an university.", "She attends a university."),
    ("He wears an uniform.", "He wears a uniform."),

    # Tricky cases: must NOT change
    ("She goes home.", "She goes home."),
    ("I read books.", "I read books."),
    ("They play football.", "They play football."),
    ("It takes an hour.", "It takes an hour."),
    ("He can go home.", "He can go home."),
    ("She will eat lunch.", "She will eat lunch."),
    ("She went home yesterday.", "She went home yesterday."),

    # Quotes and proper nouns preserved
    ('She said, "they goes to school."', 'She said, "they goes to school."'),
    ("Microsoft is a large company.", "Microsoft is a large company."),
]


@pytest.mark.parametrize("input_text, expected_corrected", ADDITIONAL_API_CASES)
def test_api_additional_cases(input_text, expected_corrected):
    response = client.post("/api/correct", json={"text": input_text})
    assert response.status_code == 200
    data = response.json()
    assert data["corrected"] == expected_corrected

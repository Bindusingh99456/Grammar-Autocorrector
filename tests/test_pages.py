import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_static_assets():
    """Verify that style.css and app.js are served correctly."""
    css_resp = client.get("/static/style.css")
    assert css_resp.status_code == 200
    assert "Source Serif 4" in css_resp.text
    assert "--navy-primary: #0e2240;" in css_resp.text

    js_resp = client.get("/static/app.js")
    assert js_resp.status_code == 200
    assert "copy-btn" in js_resp.text
    assert "char-count" in js_resp.text
    # Verify app.js is strictly under 40 lines
    lines = [line for line in js_resp.text.splitlines() if line.strip()]
    assert len(lines) < 40


def test_index_page_initial():
    """Verify home page editor loads with empty state and student nav."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Grammar Autocorrector" in response.text
    assert "No input has been checked yet." in response.text
    assert "Check Grammar" in response.text
    assert 'href="/how-it-works"' in response.text
    assert 'href="/test-cases"' in response.text
    assert 'href="/about"' in response.text


def test_form_correction_with_diff_and_tags():
    """Verify form submission generates inline diff, edit badges, and POS details."""
    response = client.post("/correct", data={"text": "She go to college every day."})
    assert response.status_code == 200
    # Check corrected sentence
    assert "She goes to college every day." in response.text
    # Check inline diff markup
    assert '<del class="diff-old">go</del>' in response.text
    assert '<ins class="diff-new">goes</ins>' in response.text
    # Check rule badge
    assert "subject_verb_agreement" in response.text
    # Check POS tag details
    assert "Tokens &amp; POS Tags" in response.text
    assert "PRP" in response.text


def test_how_it_works_page():
    """Verify how_it_works page has 8-step pipeline and worked example."""
    response = client.get("/how-it-works")
    assert response.status_code == 200
    assert "The 8-Step Processing Pipeline" in response.text
    assert "1. Input" in response.text
    assert "2. Preprocessing" in response.text or "2. Clean" in response.text
    assert "8. Output" in response.text
    assert "Worked Example" in response.text
    assert "She go to college every day." in response.text
    assert "She goes to college every day." in response.text


def test_test_cases_page():
    """Verify test_cases page contains all 8 report cases and 'Run it' buttons."""
    response = client.get("/test-cases")
    assert response.status_code == 200
    assert "Report Test Cases" in response.text
    # Check for presence of all 8 cases
    report_sentences = [
        "She go to college every day.",
        "He eat an apple.",
        "They goes to school.",
        "She are my friend.",
        "They is playing football.",
        "He have a car.",
        "I has a book.",
        "She is a engineer.",
    ]
    for sent in report_sentences:
        assert sent in response.text
    # Check that Run it form buttons exist
    assert "Run it" in response.text
    assert '<button type="submit" class="btn btn-secondary"' in response.text


def test_about_page():
    """Verify about page contains author details, university, tools, and datasets."""
    response = client.get("/about")
    assert response.status_code == 200
    assert "Bindu Singh M" in response.text
    assert "R24EJ024" in response.text
    assert "REVA University" in response.text
    assert "Python" in response.text
    assert "NLTK" in response.text
    assert "FastAPI" in response.text
    assert "Transformers" in response.text
    assert "JFLEG" in response.text
    assert "C4_200M" in response.text


def test_forbidden_words():
    """Verify that forbidden AI marketing buzzwords are not present in any template."""
    forbidden = ["ai-powered", "magic", "seamless", "cutting-edge", "unleash"]
    for path in ["/", "/how-it-works", "/test-cases", "/about"]:
        resp = client.get(path)
        content_lower = resp.text.lower()
        for word in forbidden:
            assert word not in content_lower, f"Forbidden word '{word}' found in {path}"

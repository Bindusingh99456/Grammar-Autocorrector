import sys
import time
import subprocess
from pathlib import Path
import httpx

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def verify_pages():
    base_url = "http://127.0.0.1:8000"
    client = httpx.Client(base_url=base_url, timeout=30.0)

    checks = [
        {
            "name": "Home Editor Page (GET /)",
            "method": "GET",
            "url": "/",
            "data": None,
            "json": None,
            "expected_status": 200,
            "expected_content": ["Grammar Autocorrector", "Input Text", "Check Grammar"],
        },
        {
            "name": "Form Submission - Valid (POST /correct)",
            "method": "POST",
            "url": "/correct",
            "data": {"text": "She go to college every day."},
            "json": None,
            "expected_status": 200,
            "expected_content": ["She goes to college every day.", 'class="diff-old">go</del>', 'class="diff-new">goes</ins>', "subject_verb_agreement"],
        },
        {
            "name": "Form Submission - Empty (POST /correct)",
            "method": "POST",
            "url": "/correct",
            "data": {"text": ""},
            "json": None,
            "expected_status": 400,
            "expected_content": ["Please enter some text to correct."],
        },
        {
            "name": "Form Submission - Over 2000 chars (POST /correct)",
            "method": "POST",
            "url": "/correct",
            "data": {"text": "word " * 500},
            "json": None,
            "expected_status": 400,
            "expected_content": ["exceeds the maximum limit"],
        },
        {
            "name": "Test Cases 'Run it' Action (POST /correct)",
            "method": "POST",
            "url": "/correct",
            "data": {"text": "He have a car."},
            "json": None,
            "expected_status": 200,
            "expected_content": ["He has a car.", 'class="diff-old">have</del>', 'class="diff-new">has</ins>', "has_have"],
        },
        {
            "name": "How It Works Page (GET /how-it-works)",
            "method": "GET",
            "url": "/how-it-works",
            "data": None,
            "json": None,
            "expected_status": 200,
            "expected_content": ["The 8-Step Processing Pipeline", "1. Input", "Worked Example", "She go to college every day."],
        },
        {
            "name": "Test Cases Page (GET /test-cases)",
            "method": "GET",
            "url": "/test-cases",
            "data": None,
            "json": None,
            "expected_status": 200,
            "expected_content": ["Report Test Cases", "She go to college every day.", "He have a car.", "Run it"],
        },
        {
            "name": "About Page (GET /about)",
            "method": "GET",
            "url": "/about",
            "data": None,
            "json": None,
            "expected_status": 200,
            "expected_content": ["Bindu Singh M", "R24EJ024", "REVA University", "Python", "NLTK", "FastAPI", "Transformers", "JFLEG", "C4_200M"],
        },
        {
            "name": "API Health Endpoint (GET /api/health)",
            "method": "GET",
            "url": "/api/health",
            "data": None,
            "json": None,
            "expected_status": 200,
            "expected_content": ['"status":"ok"', '"engine"'],
        },
        {
            "name": "API Grammar Correction (POST /api/correct)",
            "method": "POST",
            "url": "/api/correct",
            "data": None,
            "json": {"text": "They is playing football."},
            "expected_status": 200,
            "expected_content": ['"corrected":"They are playing football."', '"error_count":1'],
        },
        {
            "name": "Static CSS (GET /static/style.css)",
            "method": "GET",
            "url": "/static/style.css",
            "data": None,
            "json": None,
            "expected_status": 200,
            "expected_content": ["--navy-primary: #0e2240;", "Source Serif 4"],
        },
        {
            "name": "Static JavaScript (GET /static/app.js)",
            "method": "GET",
            "url": "/static/app.js",
            "data": None,
            "json": None,
            "expected_status": 200,
            "expected_content": ["copy-btn", "char-count", "keydown"],
        },
    ]

    print("\n" + "=" * 80)
    print(f"{'LIVE SERVER ENDPOINT & FORM VERIFICATION':^80}")
    print("=" * 80)
    print(f"{'Endpoint / Test Case':<42} | {'Method':<6} | {'Status':<6} | {'Result':<8}")
    print("-" * 80)

    all_passed = True
    for c in checks:
        name = c["name"]
        method = c["method"]
        url = c["url"]

        try:
            if method == "GET":
                resp = client.get(url)
            elif method == "POST":
                if c["json"] is not None:
                    resp = client.post(url, json=c["json"])
                else:
                    resp = client.post(url, data=c["data"])

            status_ok = (resp.status_code == c["expected_status"])
            body = resp.text

            content_ok = all(snippet in body for snippet in c["expected_content"])
            passed = status_ok and content_ok

            if not passed:
                all_passed = False
                result_str = "FAIL"
            else:
                result_str = "PASS"

            print(f"{name:<42} | {method:<6} | {resp.status_code:<6} | {result_str:<8}")

            if not passed:
                print(f"   [!] Failed checks for {name}:")
                if not status_ok:
                    print(f"       Expected status: {c['expected_status']}, Got: {resp.status_code}")
                if not content_ok:
                    missing = [s for s in c["expected_content"] if s not in body]
                    print(f"       Missing content snippets: {missing}")

        except Exception as e:
            all_passed = False
            print(f"{name:<42} | {method:<6} | ERROR  | FAIL")
            print(f"   [!] Connection / request exception: {e}")

    print("=" * 80)
    if all_passed:
        print("ALL 12 ENDPOINTS & FORMS RETURNED EXPECTED STATUS AND CONTENT!")
    else:
        print("SOME ENDPOINT VERIFICATIONS FAILED. SEE DETAILS ABOVE.")
    print("=" * 80)

    return 0 if all_passed else 1


def main():
    # Start server as a subprocess
    python_exe = sys.executable
    cmd = [python_exe, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"]
    print(f"Starting server with: {' '.join(cmd)}...", flush=True)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Wait for server to be responsive
    print("Waiting for server startup...", flush=True)
    server_ready = False
    for _ in range(30):
        try:
            resp = httpx.get("http://127.0.0.1:8000/api/health", timeout=1.0)
            if resp.status_code == 200:
                server_ready = True
                print("Server is up and responsive!\n", flush=True)
                break
        except Exception:
            time.sleep(1.0)

    if not server_ready:
        print("Error: Server failed to start within 30 seconds.")
        proc.kill()
        sys.exit(1)

    try:
        ret = verify_pages()
    finally:
        print("Shutting down live server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    sys.exit(ret)


if __name__ == "__main__":
    main()

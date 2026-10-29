import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app

def main():
    cases_path = Path("tests/cases.json")
    if not cases_path.exists():
        print(f"Error: {cases_path} does not exist.")
        sys.exit(1)

    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    client = TestClient(app)

    print("\n" + "=" * 80)
    print(f"{'VERIFICATION OF 110 HAND-CHECKED CASES VIA TESTCLIENT':^80}")
    print("=" * 80)

    results = []
    category_stats = {}
    failing_cases = []

    for case in cases:
        cid = case["id"]
        inp = case["input"]
        exp = case["expected"]
        cat = case["category"]
        ctype = case.get("type", "unknown")

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "passed": 0, "failed": 0}
        category_stats[cat]["total"] += 1

        try:
            resp = client.post("/api/correct", json={"text": inp})
            if resp.status_code != 200:
                passed = False
                actual = f"HTTP {resp.status_code}"
            else:
                data = resp.json()
                actual = data.get("corrected", "").strip()
                passed = (actual == exp.strip())
        except Exception as e:
            passed = False
            actual = f"Exception: {e}"

        if passed:
            category_stats[cat]["passed"] += 1
        else:
            category_stats[cat]["failed"] += 1
            failing_cases.append({
                "id": cid,
                "category": cat,
                "input": inp,
                "expected": exp,
                "actual": actual,
            })

        results.append({
            "id": cid,
            "category": cat,
            "type": ctype,
            "passed": passed,
            "input": inp,
            "expected": exp,
            "actual": actual,
        })

    total_cases = len(cases)
    total_passed = sum(1 for r in results if r["passed"])
    pass_rate = (total_passed / total_cases) * 100

    # Print Category Breakdown Table
    print(f"\n{'Category':<26} | {'Total':<6} | {'Passed':<6} | {'Failed':<6} | {'Pass Rate':<10}")
    print("-" * 65)
    for cat, stats in sorted(category_stats.items()):
        cat_rate = (stats["passed"] / stats["total"]) * 100
        print(f"{cat:<26} | {stats['total']:<6} | {stats['passed']:<6} | {stats['failed']:<6} | {cat_rate:<9.1f}%")
    print("-" * 65)
    print(f"{'TOTAL':<26} | {total_cases:<6} | {total_passed:<6} | {len(failing_cases):<6} | {pass_rate:<9.1f}%\n")

    if failing_cases:
        print("FAILING CASES BREAKDOWN:")
        print("-" * 80)
        for fc in failing_cases:
            print(f"[{fc['id']}] Category: {fc['category']}")
            print(f"     Input:    '{fc['input']}'")
            print(f"     Expected: '{fc['expected']}'")
            print(f"     Actual:   '{fc['actual']}'")
            print("-" * 80)
    else:
        print(f"All {total_passed} of {total_cases} regression cases passed on targeted rule patterns.")

    print("=" * 80)

    # Return exit code 0 if all pass, 1 if any fail
    return 0 if len(failing_cases) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

"""Benchmark execution timing for a 5-sentence paragraph through the Phase M5 pipeline."""
import time
from app.engine import get_engine, correct_text

# Initialize engine and ensure warmup
eng = get_engine()
print(f"Grammar Autocorrector Engine: {eng.engine_name}")

paragraph = (
    "She go to college every day. "
    "She is definately working on her final semester project. "
    "They has completed all their lab assignments. "
    "He received a honorable mention in the contest. "
    "You can visit https://reva.edu.in or email info@reva.edu.in to view 120 project listings."
)

print("\n" + "=" * 65)
print("BENCHMARK INPUT PARAGRAPH (5 SENTENCES)")
print("=" * 65)
print(paragraph)
print("=" * 65)

# 1. Cold Run Execution (No cache)
t0 = time.perf_counter()
result_cold = correct_text(paragraph)
cold_elapsed = time.perf_counter() - t0

print("\n" + "=" * 65)
print(f"COLD INFERENCE TIMING: {cold_elapsed:.4f} seconds ({cold_elapsed * 1000 / 5:.1f} ms/sentence)")
print("=" * 65)
print("Corrected Output:")
print(result_cold["corrected"])
print(f"\nEdits Detected ({result_cold['error_count']}):")
for idx, edit in enumerate(result_cold["edits"], 1):
    print(f"  {idx}. [{edit['type'].upper():8s}] \"{edit['old']}\" -> \"{edit['new']}\" (Rule: {edit['rule']}, Offset: {edit['start']}-{edit['end']})")

print(f"\nTokens Analyzed: {len(result_cold['tokens'])}")
print(f"POS Tags Sample: {result_cold['pos_tags'][:8]}")

# 2. In-Memory Cache Run
t1 = time.perf_counter()
result_cached = correct_text(paragraph)
cached_elapsed = time.perf_counter() - t1

print("\n" + "=" * 65)
print(f"CACHED LOOKUP TIMING: {cached_elapsed * 1000:.3f} ms ({cached_elapsed:.6f} seconds)")
print("=" * 65)

# Verify correctness and equality
assert result_cold["corrected"] == result_cached["corrected"]
assert result_cold["error_count"] == result_cached["error_count"]
print("Cache verification: 100% exact match.")

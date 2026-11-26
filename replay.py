"""Recompute the published tables from compact records. Python standard library only."""
from collections import Counter
import json
from pathlib import Path

from scoring import score

EVIDENCE = Path(__file__).parent / "evidence"


def replay():
    repair = json.loads((EVIDENCE / "repair.json").read_text(encoding="utf-8"))
    rows = repair["rows"]
    for row in rows:
        if row["n"] != 16 or row["valid"] != (row["triggered"] >= 12 and row["control"] <= 1):
            raise ValueError("Repair validity mismatch")
    valid = [r for r in rows if r["valid"]]
    summary = dict(attempts=len(rows), valid=len(valid),
                   removed=sum(r["alarm"] and r["after"] == 0 for r in valid),
                   random_control_preserved=sum(len(r["random"]) == 3 and sum(r["random"]) >= 36 for r in valid),
                   no_added_clean_drift=sum(b <= a + 1e-5 for a, b in (r["kl_before_after"] for r in valid)),
                   rotated_reported=repair["rotated"])
    result = json.loads((EVIDENCE / "detection.json").read_text(encoding="utf-8"))["rows"]
    counts, clean, groups, gains = Counter(), Counter(), {}, Counter()
    for r in result:
        if "clean" in r:
            clean["scheduled"] += 1
            if r["status"] != "scanned":
                clean["missing"] += 1
                continue
        else:
            counts["attempts"] += 1
            if not r["valid"]:
                continue
            if r["status"] != "scanned":
                raise ValueError("Missing scan on a valid positive")
        scored = score(r["v4"], r["steps"])
        if scored["baseline"] != r["recorded_baseline"]:
            raise ValueError("Replayed baseline differs from original run")
        if "clean" in r:
            clean["complete"] += 1
            clean["baseline_alarms"] += scored["baseline"]
            clean["combined_alarms"] += scored["combined"]
            continue
        for counter in (counts, groups.setdefault(r["base"], Counter())):
            counter["valid"] += 1
            counter["baseline"] += scored["baseline"]
            counter["with_peak"] += scored["combined"]
        if not scored["baseline"] and scored["peak_alarm"]:
            supports = sum(score(False, [steps])["peak_alarm"] for steps in r["steps"])
            gains["new"] += 1
            gains["still_above_11_without_best_start"] += supports >= 2
    return dict(repair=summary, detection=counts, by_base=groups, clean=clean, peak_gain_stress_test=gains)


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2))

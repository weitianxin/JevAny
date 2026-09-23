# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""Fit one temperature on a checkpoint's calibration rows and write it into head.pt, so every loader
(jevany.serve, jevany.benchmark and third-party clients) serves calibrated probabilities by default.

    uv run python scripts/calibrate_checkpoint.py --run runs/rlcr --rows runs/rlcr/training_eval/step-000384/calibration/rows.json

Fit on the calibration partition only. Never fit on development, transfer, or test rows. The optional --transfer rows are
reported but not fitted.
Argmax never changes; accuracy is identical before and after. JEVANY_TEMPERATURE=1.0 restores raw logits at load time.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevany.metrics import fit_temperature, metrics  # noqa: E402
from jevany.checkpoint import read_meta, write_meta  # noqa: E402
from jevany.suite import read_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--rows", required=True, help="in-distribution calibration rows.json (fit set)")
    ap.add_argument("--transfer", help="out-of-domain rows.json, reported before/after (never fitted)")
    ap.add_argument("--temperature", type=float, help="skip fitting and write this value")
    a = ap.parse_args()
    calibration_rows = [r for r in read_json(a.rows)
                        if r["variant"] == "clean" and r["source"] != "unknowable"]
    T = a.temperature or fit_temperature(calibration_rows, aggregation="micro")
    for name, rows in (("calibration", calibration_rows), *((("transfer", [r for r in read_json(a.transfer) if r["variant"] == "clean"]),) if a.transfer else ())):
        raw, cal = metrics(rows), metrics(rows, T)
        print(f"{name:12} T={T:.2f}  acc {raw['acc']:.3f} -> {cal['acc']:.3f} | brier {raw['brier']:.3f} -> {cal['brier']:.3f} | ece {raw['ece']:.3f} -> {cal['ece']:.3f} | conf-err {raw['confident_error_rate']:.3f} -> {cal['confident_error_rate']:.3f} | cov@5% {raw['coverage_at_5pct_error']:.2f} -> {cal['coverage_at_5pct_error']:.2f}")
    meta = read_meta(a.run)
    meta.temperature = T; meta.extra["temperature_fit"] = {
        "rows": a.rows, "n": len(calibration_rows), "method": "micro NLL via jevany.metrics.fit_temperature"}
    write_meta(a.run, meta); print(f"wrote temperature {T:.2f} to {a.run}/head.pt")


if __name__ == "__main__":
    main()

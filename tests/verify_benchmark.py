"""Exercise the installed CLI with the official telemetry and a saved model."""

import csv
from io import BytesIO, StringIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile


BENCHMARK_HORIZON = 30
# Reference capture on the pinned FD001 archive at the benchmark horizon.
MINIMUM_DUE_CAPTURE = {10: 10, 25: 22, 50: 25}


def assess_from_disk(model: Path, telemetry: Path, *options: str) -> list[dict]:
    result = subprocess.run(
        ["turbofan-rul", "assess", str(model), str(telemetry), *options],
        check=True, capture_output=True, text=True, timeout=30,
    )
    return list(csv.DictReader(StringIO(result.stdout)))


def verify_mixed_regimes(bundle: ZipFile, model: Path, telemetry: Path) -> None:
    compatible = [
        line for line in bundle.read("test_FD001.txt").splitlines()
        if line.strip() and line.split(maxsplit=1)[0] == b"1"
    ]
    incompatible = [
        b"101 " + line.split(maxsplit=1)[1] for line in bundle.read("test_FD002.txt").splitlines()
        if line.strip() and line.split(maxsplit=1)[0] == b"1"
    ]
    telemetry.write_bytes(b"\n".join(compatible + incompatible) + b"\n")
    results = assess_from_disk(model, telemetry, "--horizon", "125", "--review-limit", "1")
    statuses = [(row["unit"], row["review_status"]) for row in results]
    if statuses != [("1", "review"), ("101", "unsupported_conditions")]:
        raise RuntimeError(f"unexpected mixed-regime plan: {statuses}")
    if any(results[1][field] for field in ("estimate", "lower", "upper")):
        raise RuntimeError("unsupported operating conditions must have no RUL forecast")


def verify_review_outcomes(report: dict, fleet: list[dict], actual_by_unit: dict[int, int]) -> None:
    review = report["review"]
    horizon = review["horizon_cycles"]
    if horizon != BENCHMARK_HORIZON:
        raise RuntimeError(f"expected a {BENCHMARK_HORIZON}-cycle benchmark horizon, got {horizon}")
    units = [int(row["unit"]) for row in fleet]
    if len(units) != len(actual_by_unit) or set(units) != set(actual_by_unit):
        raise RuntimeError("fleet must contain each official unit exactly once")
    if any(int(row["actual_rul"]) != actual_by_unit[int(row["unit"])] for row in fleet):
        raise RuntimeError("fleet labels must match the official archive")

    due = {unit for unit, actual in actual_by_unit.items() if actual <= horizon}
    eligible = [int(row["unit"]) for row in fleet if row["review_status"] in {"review", "deferred"}]
    scheduled = [int(row["unit"]) for row in fleet if row["review_status"] == "review"]
    if scheduled != eligible[:review["limit"]]:
        raise RuntimeError("scheduled engines must be the first eligible engines within capacity")

    scenarios = report["review_capacity_scenarios"]
    if [plan["limit"] for plan in scenarios] != list(MINIMUM_DUE_CAPTURE):
        raise RuntimeError("capacity scenarios must match the benchmark contract")
    for plan in [review, *scenarios]:
        selected = set(eligible[:plan["limit"]])
        captured = len(selected & due)
        expected = dict(
            scheduled=len(selected),
            due_scheduled=captured,
            missed_due=len(due) - captured,
            scheduled_outside_horizon=len(selected - due),
        )
        if any(plan[key] != count for key, count in expected.items()):
            raise RuntimeError(f"review counts do not match the fleet for capacity {plan['limit']}")
        if plan["limit"] in MINIMUM_DUE_CAPTURE:
            minimum = MINIMUM_DUE_CAPTURE[plan["limit"]]
            if captured < minimum:
                raise RuntimeError(
                    f"FD001 capacity {plan['limit']} captured {captured} due engines; expected at least {minimum}"
                )


def verify(archive: Path, output: Path) -> dict:
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    model = json.loads((output / "model.json").read_text(encoding="utf-8"))
    if model["training_archive_sha256"] != report["archive_sha256"]:
        raise RuntimeError("model and report must reference the same dataset")
    fit_ids = set(report["fit_units"])
    calibration_ids = {int(unit) for unit in report["calibration_cutoffs"]}
    if len(fit_ids) != 75 or len(calibration_ids) != 25 or fit_ids & calibration_ids:
        raise RuntimeError("unexpected engine split")
    if report["engines"] != 100:
        raise RuntimeError("expected predictions for each official test engine")
    if report["sensor_model"]["mae_cycles"] >= report["age_baseline"]["mae_cycles"]:
        raise RuntimeError("sensor model must improve on the age-only MAE")

    review = report["review"]
    options = ["--horizon", str(review["horizon_cycles"])]
    if review["limit"] is not None:
        options.extend(("--review-limit", str(review["limit"])))

    with ZipFile(archive) as outer:
        inner = outer.read("6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip")
    with ZipFile(BytesIO(inner)) as bundle, TemporaryDirectory() as directory:
        targets = bundle.read("RUL_FD001.txt").split()
        actual_by_unit = {unit: int(target) for unit, target in enumerate(targets, start=1)}
        telemetry = Path(directory) / "telemetry.txt"
        telemetry.write_bytes(bundle.read("test_FD001.txt"))
        model_path = output / "model.json"
        from_saved_model = assess_from_disk(model_path, telemetry, *options)
        verify_mixed_regimes(bundle, model_path, telemetry)
    if len(from_saved_model) != 100:
        raise RuntimeError("saved model must score all 100 test engines")
    statuses = [row["review_status"] for row in from_saved_model]
    observed_counts = (statuses.count("review"), statuses.count("deferred"))
    if observed_counts != (review["scheduled"], review["deferred"]):
        raise RuntimeError("saved model must reproduce the review and deferral counts")
    if review["limit"] is not None and review["scheduled"] != min(review["limit"], review["flagged"]):
        raise RuntimeError("scheduled reviews must fill at most the configured capacity")
    with (output / "fleet.csv").open(encoding="utf-8", newline="") as stream:
        fleet = list(csv.DictReader(stream))
    for actual, expected in zip(from_saved_model, fleet, strict=True):
        if actual != {key: expected[key] for key in actual}:
            raise RuntimeError(f"saved model changed forecast for unit {actual['unit']}")
    verify_review_outcomes(report, fleet, actual_by_unit)
    return report


if __name__ == "__main__":
    results = verify(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps({key: results[key] for key in (
        "engines", "archive_sha256", "fit_operating_settings", "sensor_model", "age_baseline", "interval", "review",
        "review_capacity_scenarios",
    )}, indent=2))

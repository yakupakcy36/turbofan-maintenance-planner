"""Exercise the installed CLI with the official telemetry and a saved model."""

import csv
from io import BytesIO, StringIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile


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
        telemetry = Path(directory) / "telemetry.txt"
        telemetry.write_bytes(bundle.read("test_FD001.txt"))
        result = subprocess.run(
            ["turbofan-rul", "assess", str(output / "model.json"), str(telemetry), *options],
            check=True, capture_output=True, text=True, timeout=30,
        )
    from_saved_model = list(csv.DictReader(StringIO(result.stdout)))
    if len(from_saved_model) != 100:
        raise RuntimeError("saved model must score all 100 test engines")
    statuses = [row["review_status"] for row in from_saved_model]
    observed_counts = (statuses.count("review"), statuses.count("deferred"))
    if observed_counts != (review["scheduled"], review["deferred"]):
        raise RuntimeError("saved model must reproduce the review and deferral counts")
    if review["limit"] is not None and review["scheduled"] != min(review["limit"], review["flagged"]):
        raise RuntimeError("scheduled reviews must fill at most the configured capacity")
    with (output / "fleet.csv").open(encoding="utf-8", newline="") as stream:
        for actual, expected in zip(from_saved_model, csv.DictReader(stream), strict=True):
            if actual != {key: expected[key] for key in actual}:
                raise RuntimeError(f"saved model changed forecast for unit {actual['unit']}")
    return report


if __name__ == "__main__":
    results = verify(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps({key: results[key] for key in (
        "engines", "archive_sha256", "sensor_model", "age_baseline", "interval", "review",
        "review_capacity_scenarios",
    )}, indent=2))

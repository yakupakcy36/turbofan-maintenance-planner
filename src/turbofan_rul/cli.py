"""Download FD001, produce a benchmark, or assess saved engine histories."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from .dataset import (
    ARCHIVE_LIMIT,
    NASA_ARCHIVE_URL,
    TEXT_LIMIT,
    load_fd001,
    parse_trajectories,
    read_bounded,
    verify_archive,
)
from .evaluation import assess, score
from .model import CALIBRATION_COVERAGE, LifeModel, fit


DEFAULT_HORIZON = 30
MODEL_LIMIT = 64 * 1024
FORECAST_COLUMNS = ("unit", "observed_cycles", "review_status", "estimate", "lower", "upper")


def fetch(path: Path) -> None:
    with urlopen(NASA_ARCHIVE_URL, timeout=45) as response:
        if response.geturl() != NASA_ARCHIVE_URL:
            raise ValueError("the NASA download unexpectedly redirected")
        content = response.read(ARCHIVE_LIMIT + 1)
    if not content.startswith(b"PK\x03\x04") or len(content) > ARCHIVE_LIMIT:
        raise ValueError("NASA response was not a bounded ZIP archive")
    digest = verify_archive(content)

    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as scratch:
        staged = Path(scratch) / "cmapss.zip"
        staged.write_bytes(content)
        staged.replace(path)
    print(f"SHA-256 {digest}")


def _json(value: dict) -> str:
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def _forecast_rows(forecasts):
    for item in forecasts:
        row = asdict(item)
        for key in ("estimate", "lower", "upper"):
            if row[key] is not None:
                row[key] = f"{row[key]:.6f}"
        yield row


def benchmark(archive: Path, output: Path, horizon: int) -> dict:
    if output.exists():
        raise ValueError("benchmark output directory already exists")
    dataset = load_fd001(archive)
    fitted = fit(dataset.training)
    model = fitted.model
    forecasts = assess(model, dataset.test, horizon)
    actual = {engine.unit: int(target) for engine, target in
              zip(dataset.test, dataset.test_rul, strict=True)}
    report = score(model, forecasts, actual, horizon)
    report.update(
        dataset="NASA C-MAPSS FD001",
        archive_sha256=dataset.archive_sha256,
        fit_units=fitted.fit_units,
        calibration_cutoffs=fitted.calibration_cutoffs,
        calibration_target_coverage=CALIBRATION_COVERAGE,
        interval_radius_cycles=model.residual_radius,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as scratch:
        staging = Path(scratch) / "benchmark"
        staging.mkdir()
        model_document = {**model.document(), "training_archive_sha256": dataset.archive_sha256}
        (staging / "model.json").write_text(_json(model_document), encoding="utf-8")
        (staging / "report.json").write_text(_json(report), encoding="utf-8")
        columns = (*FORECAST_COLUMNS, "actual_rul", "age_baseline_rul")
        with (staging / "fleet.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in _forecast_rows(forecasts):
                row.update(
                    actual_rul=actual[row["unit"]],
                    age_baseline_rul=model.age_baseline(row["observed_cycles"]),
                )
                writer.writerow(row)
        staging.rename(output)
    return report


def _horizon(value: str) -> int:
    cycles = int(value)
    if not 1 <= cycles <= 125:
        raise argparse.ArgumentTypeError("review horizon must be between 1 and 125 cycles")
    return cycles


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("fetch", help="download the official NASA archive")
    download.add_argument("archive", type=Path)
    evaluate = commands.add_parser("benchmark", help="fit and evaluate FD001")
    evaluate.add_argument("archive", type=Path)
    evaluate.add_argument("--out", type=Path, required=True, help="new output directory")
    inference = commands.add_parser("assess", help="score engine histories from a JSON model")
    inference.add_argument("model", type=Path)
    inference.add_argument("telemetry", type=Path, help="whitespace-delimited FD001-style rows")
    for command in (evaluate, inference):
        command.add_argument("--horizon", type=_horizon, default=DEFAULT_HORIZON)
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            fetch(args.archive)
        elif args.command == "benchmark":
            print(_json(benchmark(args.archive, args.out, args.horizon)), end="")
        else:
            model = LifeModel.from_document(json.loads(read_bounded(args.model, MODEL_LIMIT)))
            raw = read_bounded(args.telemetry, TEXT_LIMIT)
            engines = parse_trajectories(raw, str(args.telemetry))
            writer = csv.DictWriter(sys.stdout, fieldnames=FORECAST_COLUMNS, lineterminator="\n")
            rows = list(_forecast_rows(assess(model, engines, args.horizon)))
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

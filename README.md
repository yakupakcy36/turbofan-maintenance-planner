# Turbofan Maintenance Planner

Estimate the **remaining useful life (RUL)** of a simulated turbofan engine from its observed sensor history. Rank a fleet for maintenance review and see how the sensor model compares with an age-only rule.

## FD001 results

| Method | MAE (cycles) | RMSE (cycles) |
| --- | ---: | ---: |
| Sensor history | **15.53** | **19.65** |
| Age-only baseline | 28.39 | 37.96 |

The 30-cycle review rule flagged all 25 test engines due within the horizon and 43 more outside it. The calibrated intervals covered 100 of 100 observed outcomes and averaged 120.47 cycles wide. The wide intervals make the review rule conservative and increase the review workload. [Recorded benchmark](https://github.com/yakupakcy36/turbofan-maintenance-planner/actions/runs/45999340143)

## Reproduce the benchmark

Use Python 3.12 and install the project from the repository directory.

```sh
python -m pip install .
turbofan-rul fetch data/cmapss.zip
turbofan-rul benchmark data/cmapss.zip --out reports/fd001
```

The benchmark writes a new directory with three files.

| File | Contents |
| --- | --- |
| `report.json` | Dataset SHA-256, engine splits, error metrics, interval coverage and review outcomes at capacities of 10, 25 and 50 |
| `fleet.csv` | One row per test engine with the estimate, prediction interval, review status, age baseline and actual RUL |
| `model.json` | Versioned numeric coefficients and calibration parameters for offline scoring |

You can also download these files from the **fd001-maintenance-report** artifact in a successful [GitHub Actions run](https://github.com/yakupakcy36/turbofan-maintenance-planner/actions/workflows/verify.yml). The dataset stays outside the repository. Fetch and evaluation check its SHA-256 against the verified NASA archive.

## Assess an engine history offline

```sh
turbofan-rul assess reports/fd001/model.json telemetry.txt --horizon 30 > assessment.csv
turbofan-rul assess reports/fd001/model.json telemetry.txt --horizon 30 --review-limit 25 > queue.csv
```

The input uses FD001's 26 whitespace-delimited fields. Each row contains the unit ID, cycle, three operating settings and 21 sensor readings. Sort by unit and supply consecutive observed cycles starting at one. The last observed cycle determines each forecast. Histories shorter than 20 cycles are reported as `insufficient_history`. `test_FD001.txt`, inside the NASA bundle's inner ZIP, can be used as an example input.

Engines are ranked by the lower bound of their interval, then their point estimate and unit ID. The default 30-cycle rule marks an engine `review` if its lower bound is within the horizon. An optional `--review-limit` keeps only the highest-ranked candidates in `review` and marks the others `deferred`. A limit of zero defers every eligible engine.

Pass the same limit to `benchmark` to evaluate a planned workload. In `report.json`, `flagged` counts every eligible engine and `scheduled` counts those assigned a review slot. `missed_due` counts engines due within the horizon without a review slot, including any due engines marked `deferred` or `monitor`.

## Evaluation design

- **Source.** NASA C-MAPSS FD001 contains 100 simulated run-to-failure training engines and 100 official test engines recorded up to an earlier cutoff. It covers one operating condition and one high-pressure compressor degradation mode. [NASA dataset catalog](https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data)
- **Fitting.** Features use the current cycle and the sensor levels and trends over the last 20 observed cycles. Ridge regression is fitted on 75 complete training engines with equal aggregate weight per engine. Training targets are capped at 125 cycles. The baseline subtracts current age from the fit engines' median lifetime and uses the same prediction bounds.
- **Calibration.** The other 25 training engines each contribute one reproducibly sampled pre-failure cutoff. A 90% split-conformal interval is calibrated from their absolute residuals. Its measured coverage on the official cutoffs is reported separately because the cutoff mechanism can differ.
- **Holdout.** Official labels enter only the evaluation. Both methods are scored against the original uncapped RUL at one cutoff per test engine. The CI comparison and saved-model round-trip run on GitHub Actions.

These results describe the FD001 simulation. Field validation and aircraft maintenance use are outside the project scope.

The dataset is credited to A. Saxena and K. Goebel (2008), *Turbofan Engine Degradation Simulation Data Set*, NASA Ames Prognostics Data Repository. [NASA repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

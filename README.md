# FuExoMLP: Lightweight State-Conditioned Future Mapping for Time Series Forecasting with Known Future Exogenous Variables

This repository contains the anonymous code package for the paper above. It includes the model implementation, benchmark runner, datasets, and experiment scripts needed to reproduce the main forecasting experiments.

## Environment

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Data

The datasets used by the main experiments are included under:

```text
dataset/forecasting/
```

## Model

The FuExoMLP implementation is located in:

```text
ts_benchmark/baselines/fuexomlp/
```

The code follows the paper terminology:

- `FuExoMLP`: the full forecasting model.
- `EndogenousTrajectoryProjector`: endogenous trajectory modeling.
- `StateConditionedFutureMapper`: future exogenous mapping conditioned on historical state.
- `SelectiveFutureTimeAdapter`: future-time adaptation used by the state-conditioned mapper.

## Main Experiment

Run the main FuExoMLP experiments with:

```bash
sh ./scripts/covariate_forecasting/FuExoMLP.sh
```

## License

Some benchmark components are adapted from public open-source forecasting libraries. Their original licenses are preserved under `THIRD_PARTY_LICENSES/`.

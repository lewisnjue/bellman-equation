# Bellman Equation

A small reinforcement learning project demonstrating Bellman value iteration and Q-iteration on Gymnasium's `FrozenLake-v1` environment.

- Implements a model-based agent that learns state and action values from observed transitions.
- Includes a reusable Python package, CLI entrypoint, and experiment runner.
- Provides GitHub Actions CI to run tests on push and pull requests.

## Features

- `bellman_equation.agent.Agent` — core agent logic
- `bellman_equation.runner.run` — training loop with exploration and evaluation
- `bellman_equation.cli.main` — command-line interface
- `tests/` — unit tests covering package functionality

## Requirements

- Python 3.12+
- `gymnasium`
- `tensorboard`
- `torch`
- `uv` (for the `uv run` command)

## Installation

Create and activate a virtual environment, then install the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

## Usage

Run either algorithm with the CLI:

```bash
bellman --algorithm v
bellman --algorithm q
```

Change exploration and evaluation settings:

```bash
bellman --algorithm q --random-steps 200 --test-episodes 30 --target-reward 0.85
```

Run the package module directly:

```bash
python -m bellman_equation
```

Use `uv run` if you prefer:

```bash
uv run python -m pytest -q
```

## Project structure

- `bellman_equation/`
  - `__init__.py` — package exports
  - `__main__.py` — module entrypoint
  - `agent.py` — Bellman agent implementation
  - `runner.py` — training and evaluation loop
  - `cli.py` — command-line interface
- `01_frozenlake_v_iteration.py` — original value iteration script
- `02_frozenlake_q_iteration.py` — original Q-iteration script
- `tests/` — unit tests for the package
- `.github/workflows/python-tests.yml` — CI workflow for automated testing
- `pyproject.toml` — project metadata and package configuration

## Tests

Run the test suite with:

```bash
uv run python -m pytest -q
```

Or if you do not use `uv`:

```bash
python -m pytest -q
```

## Continuous integration

This repository includes a GitHub Actions workflow at `.github/workflows/python-tests.yml` that runs tests on every push and pull request targeting `main`.

## Notes

- The package currently targets the FrozenLake environment and demonstrates Bellman backups with simple tabular estimates.
- TensorBoard logging is available through the runner, which writes metrics during training.

## License

This repository is provided as-is. Add a license file if you want to share or publish this project publicly.

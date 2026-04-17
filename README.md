# Neural Evolution Ecosystem

Interactive Flask app for running and visualizing a neural evolution simulation.

## Requirements

- Python 3.10 or newer
- Internet access the first time dependencies are installed
- Internet access for the optional 3D arena page, which loads Three.js from a CDN

## Quick Start on Windows

Double-click `Neural Evolution.bat`.

The launcher creates a local `.venv`, installs dependencies from `requirements.txt`, starts the Flask server, and opens:

```text
http://127.0.0.1:5000
```

## Quick Start on macOS or Linux

Run:

```bash
bash run.sh
```

Or run the steps manually:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

## Manual Windows Run

```bat
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

## Configuration

By default, the app binds to `127.0.0.1` on port `5000`.

Optional environment variables:

```text
NEURAL_EVOLUTION_HOST=127.0.0.1
NEURAL_EVOLUTION_PORT=5000
NEURAL_EVOLUTION_DEBUG=0
```

Use `NEURAL_EVOLUTION_HOST=0.0.0.0` only if you intentionally want other devices on your network to reach the app.

## Generated Files

The app may create these local files while running:

- `.venv/`
- `best_population.pkl`
- `__pycache__/`

They are intentionally ignored by Git.

## Publishing Notes

Before publishing this repository, choose and add a license file such as MIT, Apache-2.0, or GPL-3.0. Without a license, other people can view the code but do not automatically have clear rights to reuse it.

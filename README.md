# RawAccel Studio

**Record your mouse data while gaming → ML predicts your optimal [RawAccel](https://github.com/RawAccelOfficial/rawaccel) settings.**

Outputs a `settings.json` you can import directly into RawAccel.

## How It Works

```
1. Record mouse data while playing (desktop logger)
2. ML model analyzes your movement patterns
3. Predicts the best acceleration curve type + parameters
4. Export settings.json → import into RawAccel
```

## Project Structure

```
rawaccel/         RawAccel integration
  curves.py       Curve math (Linear, Classic, Natural, Power, Synchronous, Jump)
  config.py       settings.json builder

collector/        Data collection
  logger.py       Desktop mouse logger (run while gaming)

ml/               ML pipeline
  features.py     Feature extraction from mouse data
  dataset.py      Synthetic data generation + PyTorch dataset
  model.py        Neural network (features → curve params)
  train.py        Training loop
  predict.py      Inference (mouse data → RawAccel settings)

app/              Web application (coming soon)
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. Record mouse data (10 seconds)
python -m collector.logger --duration 10 --dpi 800

# 2. Train the model
python -m ml.train --epochs 30

# 3. Predict your curve
python -c "
from ml.predict import AccelCurvePredictor
p = AccelCurvePredictor('checkpoints/best_model.pt')
print(p.predict_and_export('data/raw/session_XXX.json'))
"
```

## Supported RawAccel Curves

| Style | Description |
|-------|-------------|
| Linear | Constant rate sensitivity increase |
| Classic | Quake 3 style (rate × speed^exponent) |
| Natural | Concave curve approaching a maximum |
| Power | CS:GO / Source Engine style |
| Synchronous | Log-symmetrical change around anchor speed |
| Jump | Step function with optional smoothing |

## Tech Stack

- **ML**: PyTorch (ResBlock MLP with dual classification + regression heads)
- **Data**: Synthetic generation with realistic mouse behavior simulation
- **Backend**: FastAPI
- **Frontend**: Vanilla HTML/CSS/JS with Chart.js

# Data Directory

This directory contains generated datasets and logs. **All contents are gitignored.**

## Structure

```
data/
├── __init__.py           # Package init
├── utils.py              # Shared data loading utilities
├── README.md             # This file
├── sim_inverse/          # P2: Inverse model training data
│   ├── train.npz         # Training split (sequences + params)
│   └── val.npz           # Validation split
├── aim_anomaly/          # P4: Anomaly detection data (if pre-generated)
└── real_logs/            # Real mouse/view logs (future)
```

## File Formats

### sim_inverse/{train,val}.npz
- `sequences`: array of variable-length (T, 3) arrays — `[mouse_dx, dt, view_delta]`
- `params`: (N, 5) float32 array — `[k1, a, k2, b, v0]`

### aim_anomaly/
- Generated on-the-fly during training (not stored by default)

## Regenerating Data

```bash
python -m experiments.inverse_model.gen_synthetic_data --num-curves 2000
```

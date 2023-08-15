# Geometric-peg-in-hole
This is the official PyTorch/PyBullet implementation of Imitation Learning for Spatio-Geometry Driven Assembly Task with Dual-arm Robot Manipulator

## Environment Setup
```
conda env create --file environment.yml
conda activate cap
pip install --no-deps -e .
```

## Generate Data
```
bash scripts/data/generate_data.sh
```

## Train Models
```
bash scripts/train/train_all.sh
```

## Test Models
```
bash scripts/test/test_each.sh
```
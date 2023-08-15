for TASK in ypzr zpzr yzr yzpyzr yp zp yr zr
do
    python scripts/data/generate_data.py start_idx=0 n_samples=100 env/variant=$TASK &
    python scripts/data/generate_data.py start_idx=100 n_samples=100 env/variant=$TASK &
    python scripts/data/generate_data.py start_idx=200 n_samples=100 env/variant=$TASK &
    python scripts/data/generate_data.py start_idx=300 n_samples=100 env/variant=$TASK &
    python scripts/data/generate_data.py start_idx=400 n_samples=100 env/variant=$TASK &
    python scripts/data/generate_data.py start_idx=500 n_samples=100 env/variant=$TASK &
    python scripts/data/generate_data.py start_idx=600 n_samples=100 env/variant=$TASK &
    python scripts/data/generate_data.py start_idx=700 n_samples=100 env/variant=$TASK &
    python scripts/data/generate_data.py start_idx=800 n_samples=100 env/variant=$TASK &
    python scripts/data/generate_data.py start_idx=900 n_samples=100 env/variant=$TASK
done


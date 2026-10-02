# DecoupledDGNN Environment Setup

This setup reuses the existing benchmark Python stack where possible and keeps
DecoupledDGNN-specific build dependencies isolated in a dedicated conda
environment.

## Environment Choice

The closest reusable environment is `NTP`:

- Python 3.10
- PyTorch 1.11.0
- PyG 2.1.0
- OGB 1.3.6
- numpy/pandas/sklearn already available

`NTP` was cloned instead of modified in place so other benchmark runs keep their
current dependency surface unchanged.

## Commands

Run from any directory unless a command explicitly changes directory.

```bash
conda create -y -n decoupleddgnn-py310 --clone NTP
conda install -y -n decoupleddgnn-py310 cython=0.29.36 eigen=3.4.0
/home/wzq/miniconda3/envs/decoupleddgnn-py310/bin/python -m pip install eigency==1.77

cd /home/wzq/TGN-benchmark/systems/DecoupledDGNN
/home/wzq/miniconda3/envs/decoupleddgnn-py310/bin/python setup.py build_ext --inplace
/home/wzq/miniconda3/envs/decoupleddgnn-py310/bin/python scripts/verify_env.py --data-root /home/phc/data --dataset lastfm --edge-count 8 --feature-dim 8
```

## Version Notes

- `cython=0.29.36` is used to stay on the Cython 0.29 line required by the
  original extension code while supporting Python 3.10.
- `eigency==1.77` is installed with pip because the conda-forge eigency builds
  conflict with the cloned environment's defaults toolchain pins.
- `eigen=3.4.0` is used because `eigen=3.3.9` is not available from the active
  defaults channel for this environment. The extension includes eigency headers
  during compilation and has been validated after build.

## Validation Scope

`scripts/verify_env.py` is intentionally read-only for `/home/phc/data`. It
checks Python dependencies, imports the compiled `propagation` module, reads the
requested dataset's node features and edge table, then runs `InstantGNN` on a
temporary remapped mini graph.

This validates the runtime and compiled propagation layer. Full DecoupledDGNN
training still requires its native generated files under `data/<dataset>/`
(`ml_<dataset>.csv`, `*_time_edge_map*.pkl`, `*_nodes_seq_lst*.pkl`). Those are
not the same as the benchmark-preprocessed `/home/phc/data/<dataset>/` layout, so
a separate data conversion/integration step is needed before benchmark training
can run end to end.

Generated Cython build outputs are ignored by git:

- `build/`
- `propagation.cpp`
- `propagation*.so`

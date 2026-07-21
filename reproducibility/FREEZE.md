# Reproducibility FREEZE

- **Frozen timestamp:** `2026-07-21T20:00:00Z` (source: supplied)
- **Schema:** `pgc_freeze/1`
- **Repo root:** `/home/ec2-user/predictive-grid-cell-analysis`

## Git

- **Commit:** `80453740f152865b98cd314b68495f23292e53d7`
- **Branch:** `main`
- **Dirty:** `True` (56 changed files)

<details><summary>Changed files</summary>

- `M` code/main.py
- `??` "Seed 0(trained advanced)/analysis_outputs/torus/"
- `??` "Seed 0(trained advanced)/analysis_outputs/torus_intersection_random_walk_classes/"
- `??` "Seed 0(trained advanced)/analysis_outputs/torus_saved_random_walk_classes/"
- `??` "analysis_outputs/Seed 1/torus/"
- `??` "analysis_outputs/Seed 2/torus/"
- `??` "analysis_outputs/Seed 3/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 0 weight decay 1e-06/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 0/pgc_rigor_test/"
- `??` "analysis_outputs/Single agent path integration/Seed 0/spatial_shift/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 0/spatial_shift_allunits/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 1 weight decay 1e-06/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 1/spatial_shift/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 1/spatial_shift_allunits/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 2 weight decay 1e-06/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 2/spatial_shift/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 2/spatial_shift_allunits/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 3 weight decay 1e-06/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 3/spatial_shift/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 3/spatial_shift_allunits/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 4 weight decay 1e-06/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 4/spatial_shift/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 4/spatial_shift_allunits/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 5/spatial_shift/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 5/spatial_shift_allunits/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 6/spatial_shift/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 6/spatial_shift_allunits/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 7/spatial_shift/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 7/spatial_shift_allunits/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 8/spatial_shift/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 8/spatial_shift_allunits/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 9/spatial_shift/torus/"
- `??` "analysis_outputs/Single agent path integration/Seed 9/spatial_shift_allunits/torus/"
- `??` "analysis_outputs/Single agent path integration/summary/aarav_activity_space_ablation/toroidal_embedding.png"
- `??` "analysis_outputs/Single agent path integration/summary/torus_distance_single/"
- `??` "analysis_outputs/Single agent path integration/summary/torus_trajectory_variation/"
- `??` analysis_outputs/canonical_cohort_v1/
- `??` code/generate_fig1b_torus.py
- `??` code/pgc_aggregate_seeds.py
- `??` code/pgc_classifier.py
- `??` code/pgc_common.py
- `??` code/pgc_covariates.py
- `??` code/pgc_fastscore.py
- `??` code/pgc_freeze.py
- `??` code/pgc_intervention.py
- `??` code/pgc_matched_ablation.py
- `??` code/pgc_pathdep_allseeds.py
- `??` code/pgc_torus_topology.py
- `??` code/single_seed_torus_distance.py
- `??` code/torus_trajectory_variation_metric.py
- `??` code/train_cohort_worker.sh
- `??` docs/PGC_RIGOR_UPGRADE.md
- `??` reproducibility/
- `??` run_pgc_classify_cohort.sh
- `??` run_pgc_rigor_pipeline.sh
- `??` run_train_cohort.sh

</details>

## Environment

- **Python:** `3.11.15` (CPython)
- **Platform:** `Linux-6.12.88-119.157.amzn2023.x86_64-x86_64-with-glibc2.34`
- **venv python:** `/home/ec2-user/predictive-grid-cell-analysis/.venv/bin/python` (`Python 3.11.15`)
- **Packages (pip freeze):** 60

<details><summary>pip freeze</summary>

```
contourpy==1.3.3
cuda-bindings==13.3.1
cuda-pathfinder==1.5.5
cuda-toolkit==13.0.2
cycler==0.12.1
Cython==3.2.5
Deprecated==1.3.1
filelock==3.29.4
fonttools==4.63.0
fsspec==2026.6.0
hopcroftkarp==1.2.5
ImageIO==2.37.3
imageio-ffmpeg==0.6.0
Jinja2==3.1.6
joblib==1.5.3
kiwisolver==1.5.0
llvmlite==0.47.0
MarkupSafe==3.0.3
matplotlib==3.11.0
mpmath==1.3.0
narwhals==2.22.1
networkx==3.6.1
numba==0.65.1
numpy==2.4.6
nvidia-cublas==13.1.1.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.20.0.48
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu13==2.29.7
nvidia-nvjitlink==13.0.88
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
opencv-python-headless==4.13.0.92
packaging==26.2
pandas==3.0.3
persim==0.3.8
pillow==12.2.0
pynndescent==0.6.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
ripser==0.6.15
scikit-learn==1.9.0
scipy==1.17.1
seaborn==0.13.2
six==1.17.0
sympy==1.14.0
threadpoolctl==3.6.0
torch==2.12.1
tqdm==4.68.3
triton==3.7.1
typing_extensions==4.15.0
umap-learn==0.5.12
wrapt==2.2.2
```

</details>

## Checkpoints

### `Models/Single agent path integration/Seed 0/most_recent_model.pth`

- **sha256:** `bb5ecb599e276bbd953004587c15386f03457b3d66ba10dae58783059a38a44b`
- **size (bytes):** `83954107`
- **run_ID:** `None`

### `Models/Single agent path integration/Seed 1/most_recent_model.pth`

- **sha256:** `e9dc4cd59aec3d898a1952bd64d555700c2ec9435c459862faa4719c3c6d9451`
- **size (bytes):** `83954107`
- **run_ID:** `None`

### `Models/Single agent path integration/Seed 2/most_recent_model.pth`

- **sha256:** `92de7abfccccfc3694fa4ff97c6fe4ce92570291aba7a238c386a373fad48dfc`
- **size (bytes):** `83954107`
- **run_ID:** `None`

### `Models/Single agent path integration/Seed 3/most_recent_model.pth`

- **sha256:** `d224de0a0f46b2bcc12b01c29fc1ee313e13d1f4c3d898fc401ca609cd32e207`
- **size (bytes):** `83954107`
- **run_ID:** `None`

### `Models/Single agent path integration/Seed 4/most_recent_model.pth`

- **sha256:** `c44fdb145367b2a130c6dc24c7e68f3eb0c9dcc9d6226559a379215f746dbd77`
- **size (bytes):** `83954107`
- **run_ID:** `None`

### `Models/Single agent path integration/Seed 5/most_recent_model.pth`

- **sha256:** `ef7006439c752dfbd87f91bf0f18b9b007da5f4cbc02d732597db1cf4e056e07`
- **size (bytes):** `83954107`
- **run_ID:** `None`

### `Models/Single agent path integration/Seed 6/most_recent_model.pth`

- **sha256:** `009835a702f6190bf9488a703eeeea8df0ff91569ef04df47bdcb40006010fe6`
- **size (bytes):** `83954107`
- **run_ID:** `None`

### `Models/Single agent path integration/Seed 7/most_recent_model.pth`

- **sha256:** `aca8d47f0009697fdf00930e4f0edb3ce3c121b4854f4bb98b7db092b798d1cd`
- **size (bytes):** `83954107`
- **run_ID:** `None`

### `Models/Single agent path integration/Seed 8/most_recent_model.pth`

- **sha256:** `f8e92c766d453f0e3a0eb97dbc3aece3ded7490cd89728dbb10ea817baa46450`
- **size (bytes):** `83954107`
- **run_ID:** `None`

### `Models/Single agent path integration/Seed 9/most_recent_model.pth`

- **sha256:** `331d700afeabefa76aa999c8bac836577363684082df1342756a21f0885e4134`
- **size (bytes):** `83954107`
- **run_ID:** `None`

### `Models/canonical_cohort_v1/seed_0/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/most_recent_model.pth`

- **sha256:** `e9d4ff24a76fb819d1aff3bc60ec7b3ed018400603778c9d4802cafefa0af5e6`
- **size (bytes):** `83954107`
- **run_ID:** `steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06`
- **training config (parsed from run_ID):**

| key | value |
| --- | --- |
| `sequence_length` | `20` |
| `batch_size` | `200` |
| `place_cell_rf` | `012` |
| `DoG` | `True` |
| `periodic` | `False` |
| `learning_rate` | `00001` |
| `weight_decay` | `1e-06` |
| `activation` | `relu` |
| `Ng` | `4096` |
| `RNN_type` | `RNN` |

### `Models/canonical_cohort_v1/seed_1/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/most_recent_model.pth`

- **sha256:** `5f32f1ce1f5135dd1b95f591a364695a46f767c057a5f75170354c3fd568884b`
- **size (bytes):** `83954107`
- **run_ID:** `steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06`
- **training config (parsed from run_ID):**

| key | value |
| --- | --- |
| `sequence_length` | `20` |
| `batch_size` | `200` |
| `place_cell_rf` | `012` |
| `DoG` | `True` |
| `periodic` | `False` |
| `learning_rate` | `00001` |
| `weight_decay` | `1e-06` |
| `activation` | `relu` |
| `Ng` | `4096` |
| `RNN_type` | `RNN` |

### `Models/canonical_cohort_v1/seed_15/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/most_recent_model.pth`

- **sha256:** `b44c00f50765ec3e5677b4af97784c8136f1d2e56de75cc88d1a2d321fa373ff`
- **size (bytes):** `83954107`
- **run_ID:** `steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06`
- **training config (parsed from run_ID):**

| key | value |
| --- | --- |
| `sequence_length` | `20` |
| `batch_size` | `200` |
| `place_cell_rf` | `012` |
| `DoG` | `True` |
| `periodic` | `False` |
| `learning_rate` | `00001` |
| `weight_decay` | `1e-06` |
| `activation` | `relu` |
| `Ng` | `4096` |
| `RNN_type` | `RNN` |

### `Models/canonical_cohort_v1/seed_16/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/most_recent_model.pth`

- **sha256:** `5b75708e1be33349bbd93a4f86ade0006513fe51c78347b85ddbea449ec1782e`
- **size (bytes):** `83954107`
- **run_ID:** `steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06`
- **training config (parsed from run_ID):**

| key | value |
| --- | --- |
| `sequence_length` | `20` |
| `batch_size` | `200` |
| `place_cell_rf` | `012` |
| `DoG` | `True` |
| `periodic` | `False` |
| `learning_rate` | `00001` |
| `weight_decay` | `1e-06` |
| `activation` | `relu` |
| `Ng` | `4096` |
| `RNN_type` | `RNN` |

### `Models/canonical_cohort_v1/seed_17/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/most_recent_model.pth`

- **sha256:** `cac33a36ed8ff5ec7024eb3146e60b64a9de4768c04fa58d5f63631d22905b79`
- **size (bytes):** `83954107`
- **run_ID:** `steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06`
- **training config (parsed from run_ID):**

| key | value |
| --- | --- |
| `sequence_length` | `20` |
| `batch_size` | `200` |
| `place_cell_rf` | `012` |
| `DoG` | `True` |
| `periodic` | `False` |
| `learning_rate` | `00001` |
| `weight_decay` | `1e-06` |
| `activation` | `relu` |
| `Ng` | `4096` |
| `RNN_type` | `RNN` |

### `Models/canonical_cohort_v1/seed_18/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/most_recent_model.pth`

- **sha256:** `b744a09ce285a1ab6ba0332d427db9b4285f8c6273aef911b83246db59c7af76`
- **size (bytes):** `83954107`
- **run_ID:** `steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06`
- **training config (parsed from run_ID):**

| key | value |
| --- | --- |
| `sequence_length` | `20` |
| `batch_size` | `200` |
| `place_cell_rf` | `012` |
| `DoG` | `True` |
| `periodic` | `False` |
| `learning_rate` | `00001` |
| `weight_decay` | `1e-06` |
| `activation` | `relu` |
| `Ng` | `4096` |
| `RNN_type` | `RNN` |

### `Models/canonical_cohort_v1/seed_19/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/most_recent_model.pth`

- **sha256:** `05845b014ec2cc7e78a5cbf761e32d0018b6bca4647798c72b9cefff60afcc4d`
- **size (bytes):** `83954107`
- **run_ID:** `steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06`
- **training config (parsed from run_ID):**

| key | value |
| --- | --- |
| `sequence_length` | `20` |
| `batch_size` | `200` |
| `place_cell_rf` | `012` |
| `DoG` | `True` |
| `periodic` | `False` |
| `learning_rate` | `00001` |
| `weight_decay` | `1e-06` |
| `activation` | `relu` |
| `Ng` | `4096` |
| `RNN_type` | `RNN` |

### `Models/canonical_cohort_v1/seed_2/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/most_recent_model.pth`

- **sha256:** `e11a15a676c6026c5a5875bc152d558b04522e56c5e754aef290e0822ed50596`
- **size (bytes):** `83954107`
- **run_ID:** `steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06`
- **training config (parsed from run_ID):**

| key | value |
| --- | --- |
| `sequence_length` | `20` |
| `batch_size` | `200` |
| `place_cell_rf` | `012` |
| `DoG` | `True` |
| `periodic` | `False` |
| `learning_rate` | `00001` |
| `weight_decay` | `1e-06` |
| `activation` | `relu` |
| `Ng` | `4096` |
| `RNN_type` | `RNN` |

### `Models/canonical_cohort_v1/seed_20/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/most_recent_model.pth`

- **sha256:** `b6dd7a50409191f03a9c02d563b66aae44d8c53c79018b021d3585c01411a29d`
- **size (bytes):** `83954107`
- **run_ID:** `steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06`
- **training config (parsed from run_ID):**

| key | value |
| --- | --- |
| `sequence_length` | `20` |
| `batch_size` | `200` |
| `place_cell_rf` | `012` |
| `DoG` | `True` |
| `periodic` | `False` |
| `learning_rate` | `00001` |
| `weight_decay` | `1e-06` |
| `activation` | `relu` |
| `Ng` | `4096` |
| `RNN_type` | `RNN` |

### `Models/canonical_cohort_v1/seed_21/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/most_recent_model.pth`

- **sha256:** `e10d40dea7ff9f1480ae59c5617596a26a9022ced3fe2c3bd7d30fc2192950f7`
- **size (bytes):** `83954107`
- **run_ID:** `steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06`
- **training config (parsed from run_ID):**

| key | value |
| --- | --- |
| `sequence_length` | `20` |
| `batch_size` | `200` |
| `place_cell_rf` | `012` |
| `DoG` | `True` |
| `periodic` | `False` |
| `learning_rate` | `00001` |
| `weight_decay` | `1e-06` |
| `activation` | `relu` |
| `Ng` | `4096` |
| `RNN_type` | `RNN` |

### `Models/canonical_cohort_v1/seed_22/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/most_recent_model.pth`

- **sha256:** `3d06a225abb85ed8ba5b8403ba95ce6fdb9e1c311decf095e04e77bbefbcc335`
- **size (bytes):** `83954107`
- **run_ID:** `steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06`
- **training config (parsed from run_ID):**

| key | value |
| --- | --- |
| `sequence_length` | `20` |
| `batch_size` | `200` |
| `place_cell_rf` | `012` |
| `DoG` | `True` |
| `periodic` | `False` |
| `learning_rate` | `00001` |
| `weight_decay` | `1e-06` |
| `activation` | `relu` |
| `Ng` | `4096` |
| `RNN_type` | `RNN` |

### `Models/canonical_cohort_v1/seed_3/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/most_recent_model.pth`

- **sha256:** `2ec5caa70feea63b13d929572d0b3ed698039cb9e4a98befe023a7a54dd7a083`
- **size (bytes):** `83954107`
- **run_ID:** `steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06`
- **training config (parsed from run_ID):**

| key | value |
| --- | --- |
| `sequence_length` | `20` |
| `batch_size` | `200` |
| `place_cell_rf` | `012` |
| `DoG` | `True` |
| `periodic` | `False` |
| `learning_rate` | `00001` |
| `weight_decay` | `1e-06` |
| `activation` | `relu` |
| `Ng` | `4096` |
| `RNN_type` | `RNN` |

### `Models/canonical_cohort_v1/seed_4/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/most_recent_model.pth`

- **sha256:** `3d1b5658053c719ee69aebbf66bab7c03af93fbca2e4f2dd545703e94c750f3b`
- **size (bytes):** `83954107`
- **run_ID:** `steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06`
- **training config (parsed from run_ID):**

| key | value |
| --- | --- |
| `sequence_length` | `20` |
| `batch_size` | `200` |
| `place_cell_rf` | `012` |
| `DoG` | `True` |
| `periodic` | `False` |
| `learning_rate` | `00001` |
| `weight_decay` | `1e-06` |
| `activation` | `relu` |
| `Ng` | `4096` |
| `RNN_type` | `RNN` |

### `Models/canonical_cohort_v1/seed_5/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/most_recent_model.pth`

- **sha256:** `1fe3ca2721a627462b889b0c927bee4fcc51b4fd68eaeba96dd96f9f44d47a37`
- **size (bytes):** `83954107`
- **run_ID:** `steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06`
- **training config (parsed from run_ID):**

| key | value |
| --- | --- |
| `sequence_length` | `20` |
| `batch_size` | `200` |
| `place_cell_rf` | `012` |
| `DoG` | `True` |
| `periodic` | `False` |
| `learning_rate` | `00001` |
| `weight_decay` | `1e-06` |
| `activation` | `relu` |
| `Ng` | `4096` |
| `RNN_type` | `RNN` |

### `Models/canonical_cohort_v1/seed_6/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/most_recent_model.pth`

- **sha256:** `e983ace1752e84e6457406bbf79b0d85dc445f8cdcabff5a935bbb04b552a6aa`
- **size (bytes):** `83954107`
- **run_ID:** `steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06`
- **training config (parsed from run_ID):**

| key | value |
| --- | --- |
| `sequence_length` | `20` |
| `batch_size` | `200` |
| `place_cell_rf` | `012` |
| `DoG` | `True` |
| `periodic` | `False` |
| `learning_rate` | `00001` |
| `weight_decay` | `1e-06` |
| `activation` | `relu` |
| `Ng` | `4096` |
| `RNN_type` | `RNN` |

### `Models/canonical_cohort_v1/seed_7/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/most_recent_model.pth`

- **sha256:** `77e35d3ae2448f2f7782c4e0e00e88a8b4b0145561e9e94149b3281a7e27711e`
- **size (bytes):** `83954107`
- **run_ID:** `steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06`
- **training config (parsed from run_ID):**

| key | value |
| --- | --- |
| `sequence_length` | `20` |
| `batch_size` | `200` |
| `place_cell_rf` | `012` |
| `DoG` | `True` |
| `periodic` | `False` |
| `learning_rate` | `00001` |
| `weight_decay` | `1e-06` |
| `activation` | `relu` |
| `Ng` | `4096` |
| `RNN_type` | `RNN` |

### `Models/straight/steps_40/Seed 0/most_recent_model.pth`

- **sha256:** `1bb32c1403a891248cf8207836fcafca3d30d7451c4cb0c2e34c7613e3e0d618`
- **size (bytes):** `83954107`
- **run_ID:** `steps_40`
- **training config (parsed from run_ID):**

| key | value |
| --- | --- |
| `sequence_length` | `40` |

### `Models/straight/steps_40/Seed 1/most_recent_model.pth`

- **sha256:** `27a33f1a348edb2430c9a56095013a65399d7ee00ed10d180bd277b4748beb88`
- **size (bytes):** `83954107`
- **run_ID:** `steps_40`
- **training config (parsed from run_ID):**

| key | value |
| --- | --- |
| `sequence_length` | `40` |

## Analysis config

### `pgc_classifier.ClassifierConfig`

| key | value |
| --- | --- |
| `shift_mode` | `time` |
| `max_lag` | `25` |
| `lag_step` | `1` |
| `max_shift_cm` | `50.0` |
| `shift_step_cm` | `2.0` |
| `space_projection` | `path` |
| `min_shift_cm` | `5.0` |
| `gridness_floor` | `0.1` |
| `alpha` | `0.05` |
| `n_shuffles` | `100` |
| `null_block` | `False` |
| `res` | `20` |
| `n_batches` | `20` |
| `Ng_use` | `512` |
| `classify_seed` | `1234` |
| `confirm_seed` | `9876` |
| `heldout_confirm` | `True` |
| `n_workers` | `0` |

### `pgc_covariates.assemble_covariates_defaults`

| key | value |
| --- | --- |
| `Ng_use` | `512` |
| `n_batches` | `20` |
| `res` | `20` |
| `collection_seed` | `4321` |
| `grid_floor` | `0.2` |
| `n_workers` | `0` |


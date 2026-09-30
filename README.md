# Radeon Object Detection Benchmark

A local, reproducible benchmark harness for object detection with Ultralytics YOLO and PyTorch. It is intentionally an experiment runner and hardware validation project, not a web service. It writes configuration, software/hardware discovery, training logs, COCO-compatible accuracy, timing, memory, and export artifacts locally.

> **Important RX 6800 compatibility finding (checked 2026-09-30):** The exact Radeon RX 6800 is not listed in AMD's current ROCm Radeon Linux GPU support matrix (ROCm 7.2.1); it is not a currently AMD-certified ROCm/PyTorch target. AMD's current generic ROCm architecture docs list gfx1030 support in some tables, but that must not be read as a Radeon RX 6800 product certification. Earlier ROCm 6.2.x generic compatibility documentation lists gfx1030 and PyTorch 2.3, and AMD GPU specs identify RX 6800 as RDNA2/gfx1030/16 GiB. **ROCm 6.2.x + its tested PyTorch 2.3 generation is only a legacy experimental candidate, not a guaranteed or current officially supported RX 6800 stack.** The supported OS combinations in those docs are Ubuntu/RHEL/SLES etc.; Manjaro is not listed. Install only after reviewing the host-specific AMD/PyTorch instructions, and treat results as experimental. Newer library support does not guarantee operation on this card.

## Why this framework?

Ultralytics YOLO is the primary baseline: maintained detection models and an integrated train/val/predict/export workflow; COCO-compatible validation; direct PyTorch execution; easy local artifact handling and future inference adapter. AMD documents PyTorch ROCm usage through the regular PyTorch GPU interfaces and Ultralytics documents `device=0` for AMD. Detectron2 and MMDetection are capable research frameworks, but their custom operators/build matrix and extra setup add friction for this single GPU benchmark. torchvision detection models have a stable PyTorch foundation but less turnkey training, COCO evaluation, and benchmark tooling. The lightweight `Detector` interface isolates inference callers from Ultralytics-specific calls so another runtime can be introduced later.

## Hardware and compatibility

- Linux with AMDGPU/KFD support and access to `/dev/kfd` and `/dev/dri`.
- Radeon RX 6800: RDNA2, gfx1030, 16 GiB VRAM. The card's official support status is described above; device visibility is not the same as vendor certification.
- For an experimental ROCm attempt, use a PyTorch ROCm wheel/image matching the installed ROCm runtime and target architecture. ROCm 6.2.x / PyTorch 2.3 is the older documented path worth testing, with a compatible Python (prefer Python 3.10/3.11). Do not install a CUDA wheel and expect ROCm to work. Do not overlay unrelated ROCm user-space versions.
- Current AMD Radeon-specific support information and installer: [ROCm Radeon Linux matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html), [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html), [PyTorch install selector](https://pytorch.org/get-started/locally/), [PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html), [Ultralytics AMD guide](https://docs.ultralytics.com/integrations/amd/).
- PyTorch ROCm deliberately exposes HIP devices through `torch.cuda.*`; `torch.cuda.is_available()` is expected to be `True` when operational. Ultralytics device syntax is `0` or `cuda:0`. Never specify `rocm:0`.

This workspace's host has Manjaro Linux and no `rocminfo`, `rocm-smi`, or `nvidia-smi` command on PATH at inspection time; no PyTorch/Ultralytics packages are installed in the selected Python environment. Do not infer the actual graphics hardware from the requested target model.

## Installation

1. Install the ROCm driver/runtime using AMD's current directions for a supported host, if available. Do not install ROCm from an unverified third-party script. On this RX 6800, verify exact GPU support before making a system-level change.
2. Use the supported Python matching the selected ROCm/PyTorch release (Python 3.10/3.11 recommended for the legacy experimental route).
3. Install PyTorch from the official PyTorch selector choosing Linux, pip, and the matching ROCm build. Check the ROCm-specific wheel URL/version. Install PyTorch first, then this project:

   ```sh
   python -m venv .venv
   . .venv/bin/activate
   python -m pip install --upgrade pip
   # Install the matching ROCm PyTorch build from the official selector before the next command.
   python -m pip install -e '.[benchmark,test]'
   ```

   The project deliberately does not pin/install a CUDA or ROCm PyTorch wheel: AMD support depends on the machine's driver, OS, ROCm and GPU combination. The user selects a coherent stack explicitly.

## GPU verification

Run `python scripts/check_environment.py`. The command reports OS/kernel/CPU/memory, PyTorch/HIP, detected devices and relevant environment. It prints `GPU detected` or `GPU NOT detected` and exits nonzero if the GPU is unavailable or the visible PyTorch build lacks HIP. Confirm manually with `rocminfo` and `groups` if installed. Do not begin a GPU benchmark until the diagnostic succeeds and the reported name is the RX 6800.

## Dataset setup

No dataset is downloaded implicitly by these project commands. COCO 2017 needs substantial storage (train images alone are about 19 GB; train + validation images around 20 GB plus extracted labels). Set `COCO_DATA_DIR` to the target directory or use `--root` explicitly.

```sh
export COCO_DATA_DIR=/data/coco
python scripts/download_dataset.py --root "$COCO_DATA_DIR" --splits val
# Training download is large; opt in explicitly:
python scripts/download_dataset.py --root "$COCO_DATA_DIR" --splits train val --confirm-large-download
python scripts/prepare_dataset.py --root "$COCO_DATA_DIR"
```

The download utility fetches the requested COCO image zips and Ultralytics' COCO2017 label archive. Ultralytics training consumes YOLO-format labels; this project avoids implicit conversion/downloads at training time. Review COCO's terms and cite the dataset when publishing results. For a genuine COCO workload, include both `train` and `val` splits.

### Config-driven dataset preparation

Dataset preparation has no dataset-specific switches or hardcoded class list. Pass a YAML config containing source image and YOLO label directories, the output path, class names, validation fraction, seed, and whether to symlink or copy files:

```sh
python scripts/prepare_dataset.py --config configs/dataset_mbdd2025.yaml
```

The checked-in config is an example for MBDD2025: its local README describes 14,471 UAV images and five defect classes, and the supplied label `.txt` files are already YOLO detection labels. This config sets the class names and source paths as user data rather than code constants. Other datasets use their own preparation YAML and training config. Training requires images plus matching YOLO detection text labels (normalized `class_id x_center y_center width height`) and a names list whose order matches the label class IDs. Raw VOC XML, COCO JSON, segmentation labels, or classification-only folder layouts are not auto-converted; convert them to the configured YOLO detection format first or add an explicit format adapter.

Training is likewise selected by a YAML argument, e.g. `python scripts/train.py --config configs/mbdd2025.yaml`. Common choices (model, dataset YAML, image size, batch, epochs, optimizer, learning rate, seed, workers, AMP and fraction) are top-level settings. Additional Ultralytics preprocessing/augmentation settings such as color, flips, mosaic, mixup, crop/geometry and multi-scale can be passed under `train_options`; they are forwarded as Ultralytics `model.train()` options and saved with the run config. Top-level controlled settings cannot be overridden there. No class-specific tuning is inferred automatically. Any options omitted from the YAML use the installed Ultralytics version's defaults, so pin/record that version when comparing runs.

For a new dataset, use the generic starter at `configs/detection.yaml`, set `dataset_yaml` and `data_root` to the prepared dataset, then pass it as the script argument: `python scripts/train.py --config configs/detection.yaml`. The script and core dataset utilities contain no dataset name/class assumptions; dataset class names and source layout belong in the respective dataset YAML files. The data adapter currently targets object-detection datasets already represented by YOLO `.txt` boxes; it does not infer labels or convert arbitrary annotation formats automatically.

Preparation creates a deterministic image-level train/validation split using symlinks by default (or copies if `link_files: false`), plus `dataset.yaml` and `split_manifest.json`. The split config is idempotent: repeating the same config validates and reuses a matching output; changed settings refuse to overwrite it. The training config at `configs/mbdd2025.yaml` uses pretrained YOLO26n, 640px images, batch 4, five epochs, and requires the configured GPU. First confirm the dataset validates, then run:

```sh
python scripts/train.py --config configs/mbdd2025.yaml
```

This short run checks end-to-end loading/training/validation; it is not enough to claim converged accuracy. Then increase epochs and benchmark only selected settings using `scripts/benchmark_training.py --config configs/mbdd2025.yaml --models yolo26n.pt yolo26s.pt --imgsz 640 --batches 1 4 8`. The MBDD2025 dataset README cites CC BY 4.0; preserve attribution and follow its citation. Its prepared split is image-random because no official split is present in the local README. UAV imagery may contain adjacent/near-duplicate frames, so use acquisition/building-level grouping if metadata permits before treating validation accuracy as a reliable generalization estimate.

A deterministic tiny fixture (4 train / 2 validation images) is generated locally and needs Pillow, not COCO:

```sh
python scripts/prepare_dataset.py --smoke --root datasets/smoke
```

## Smoke test and training

The smoke configuration is intentionally CPU-enabled, uses a small one-class generated dataset, `yolo26n.yaml`, one epoch, `imgsz=64`, and no pretrained checkpoint. It may be slow on CPU, but needs no AMD-specific stack. It still requires Ultralytics and its compatible PyTorch dependency.

```sh
python scripts/prepare_dataset.py --smoke --root datasets/smoke
python scripts/train.py --config configs/smoke.yaml
```

For an explicit CPU test with another config, pass `--allow-cpu`; benchmark config defaults to no CPU fallback. For actual GPU use, `device: 0` is the default. `allow_cpu: false` prevents accidental fallback. Training records run settings, full software environment, git commit, weights, Ultralytics `results.csv`, accuracy and a summary. Configure optimizer, learning rate, AMP, random seed, image size, batch and epochs explicitly; deterministic mode is requested. Ultralytics' deterministic operation and GPU kernel behavior can still vary across driver/library builds.

## Evaluation

Training's standard validation runs automatically and returns precision, recall, mAP50 and mAP50-95 in `results.json`/`benchmark_summary.md`. Evaluate an existing checkpoint independently with:

```sh
python scripts/evaluate.py --run runs/<run-directory>
```

For validation on CPU only, add `--allow-cpu`. The wrapper calls Ultralytics validation, which supplies COCO-style detection metrics when the dataset is COCO.

## Benchmarks

Benchmark scripts never enumerate the full hardware matrix unless you explicitly ask. The configured run is just one setting; pass repeated model and image-size choices and a batch list to select a subset.

```sh
python scripts/benchmark_training.py --config configs/benchmark.yaml
python scripts/benchmark_training.py --config configs/benchmark.yaml --models yolo26n.pt yolo26s.pt yolo26m.pt --imgsz 640 960 --batches 1 4 8 16
python scripts/benchmark_inference.py --config configs/benchmark.yaml
python scripts/benchmark_inference.py --config configs/benchmark.yaml --model yolo26n.pt --model yolo26s.pt --model yolo26m.pt --imgsz 640 960 --batches 1 4 8 16
```

Edit `configs/benchmark.yaml` for one run's model, image size, batch, epochs, optimizer, `lr0`, AMP, seed, device, dataset path, and warm-up/measured inference iterations. The default matrix is not run automatically. Start with `yolo26n.pt`; medium/large models and larger images/batches may exhaust the 16 GiB card. Every attempted setting is recorded. Out-of-memory exits are clearly marked with model/batch/resolution; parameters are never silently reduced. Ultralytics may perform an internal first-epoch OOM batch retry in some versions; this project passes a fixed positive batch and records the requested value—inspect framework logs and version-specific behavior before treating such a retry as comparable.

Inference does GPU synchronization around each measured call, separates configured warm-ups, and reports preprocessing/inference/postprocessing from Ultralytics result timing plus synchronized total wall latency. It summarizes mean, median, p50, p95, min and max; images/sec uses synchronized wall time. Memory reporting uses PyTorch allocated/reserved/peak stats and is unavailable when running CPU. Results can be affected by shared GPU use, thermal/power state, data-loader/storage throughput, and first-use kernel compilation; warm up before comparing.

## Results and reproducibility

Each run folder under `runs/` contains a copied source YAML, normalized `config.json`, `environment.json`, `results.json`, and `benchmark_summary.md`; training adds `training_metrics.csv`, Ultralytics plots and `weights/`. Benchmark matrices store one record per case in `benchmark.json`. Values include commit (if available), timestamp, OS/kernel, Python, PyTorch/HIP, Ultralytics, GPU, model/checkpoint, dataset path/fraction, image size, batch, optimizer, learning rate, epochs, AMP and seed. The COCO dataset directory and source YAML are recorded; data itself is not copied into experiment artifacts.

## Inference abstraction and export

`detector_benchmark.inference.Detector` defines `load()`, `predict()`, and `batch_predict()`; `UltralyticsDetector` implements it. New API/service layers can depend on this interface without directly importing Ultralytics. No API server, database, distributed queue, or SaaS tracking service is included.

```sh
python scripts/export_model.py --run runs/<run-directory> --format all
```

The native PyTorch checkpoint remains the source artifact; ONNX export is optional and saved separately under `exports/`. ONNX is a model interchange format, not an accelerator/runtime. AMD acceleration requires a separately compatible runtime. AMD documentation identifies ONNX Runtime/MIGraphX combinations for supported hardware, but that does not certify RX 6800 or imply Ultralytics directly executes MIGraphX. The current Ultralytics AMD guide describes native PyTorch ROCm support; MIGraphX integration availability must be rechecked before use. No AMD-specific runtime is installed by this project.

## Framework/runtime choices and containerization

Ultralytics is the detector baseline; infrastructure modules avoid tight coupling. Detectron2/MMDetection are viable research options but add operator/build compatibility burdens on an already legacy/uncertified card. torchvision models are a lower-dependency alternative but require more custom benchmark/evaluation implementation.

The primary workflow is native Linux because this RX 6800 lacks a current official Radeon ROCm stack listing and the current host distribution is outside the published Linux matrix. ROCm containers can expose `/dev/kfd` and `/dev/dri` with a compatible host driver and appropriate video/render permissions, but do not fix unsupported host/GPU combinations. This repository intentionally does not supply a purportedly known-good Dockerfile/image tag for an uncertified legacy GPU; use only AMD-published tags after verifying their support matrix. Generic CUDA containers are not suitable.

## Troubleshooting

- **`torch.cuda.is_available() == False`:** verify the wheel is a ROCm build, GPU is visible to `rocminfo`, the user can access `/dev/kfd` and `/dev/dri`, and no visibility env var hides the device. Re-run environment check.
- **Incorrect PyTorch build:** uninstall conflicting torch/torchvision packages from the active venv, then install the exact ROCm wheel selected for the host. The ordinary PyPI wheel may be CPU-only; CUDA wheels are not ROCm wheels.
- **ROCm mismatch:** PyTorch ROCm wheels package user-space runtime components, but still depend on a supported host driver/kernel. Use AMD's documented matching tuple; don't mix system ROCm libraries from unrelated releases.
- **Unsupported architecture (`hipErrorNoBinaryForGPU`):** check `rocminfo` gfx target and whether that exact product is listed in the matching release matrix. `HSA_OVERRIDE_GFX_VERSION` is an unsupported workaround and is not set by the project; it can conceal incompatibilities and invalidate benchmarks.
- **GPU out of memory:** lower batch or resolution explicitly and rerun; use `yolo26n` first; close other GPU jobs. Failed settings remain marked; no automatic parameter edits occur.
- **AMP failures/NaN:** rerun with `amp: false`, preserve that setting in artifacts, and compare accuracy. ROCm AMP support is stack/model dependent.
- **Slow training:** separate CPU/data-loader bottlenecks from GPU; inspect utilization/temperature/clock/storage, try workers deliberately, allow initial MIOpen kernel compilation, and compare after warm-up. Keep settings constant.
- **CPU fallback:** GPU training/benchmark configs forbid it. CPU is only used with explicit `allow_cpu: true` or `--allow-cpu`; CPU results are not GPU performance.
- **Container cannot access GPU:** host driver must work first. Pass `/dev/kfd`, `/dev/dri`, `--group-add video` and commonly `--group-add render`; ensure host user/container permissions. Follow AMD's current container documentation. A container cannot add GPU support that the host stack lacks.
- **First-run weight fetch:** a `.pt` model name may cause Ultralytics to fetch pretrained weights. Dataset downloads are separate and explicit. For offline training, use a local model YAML/checkpoint.

## Tests

Lightweight tests do not require PyTorch or a GPU:

```sh
python -m pip install -e '.[test]'
pytest
```

GPU behavior has mocked hardware unit coverage; no tests silently require a GPU. The project smoke training command itself requires the benchmark extra and PyTorch, and is not part of the unit suite.

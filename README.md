# YOLO Object Detection

This repository provides three workflows: inspect the machine and accelerator, train an Ultralytics object-detection model, and run predictions on new images. Dataset preparation is part of training. The project does not download datasets automatically.

## Installation

Install a PyTorch build that matches the host GPU runtime first. For AMD GPUs, follow the matching ROCm and PyTorch installation instructions for the operating system and GPU. PyTorch ROCm exposes devices through `torch.cuda.*`; GPU visibility does not by itself guarantee that a hardware/runtime combination is officially supported.

Then install the project and test dependency:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
# Install the matching PyTorch build before installing this project.
python -m pip install -e '.[benchmark,test]'
```

## Check hardware and environment

```sh
python3 scripts/check_environment.py
```

The report includes OS, CPU, CPU count, total and available system memory, PyTorch and HIP versions, detected GPU count and names, GPU properties, and relevant ROCm environment variables. It reports whether a ROCm-capable GPU is available and exits nonzero if no GPU is detected or the visible PyTorch build is not HIP-enabled.

## Dataset format

The training configuration identifies the source image directory, matching YOLO detection label directory, output directory, and class names. Each image must have a `.txt` label file with the same filename stem. Each non-empty label line must contain `class_id x_center y_center width height`; class IDs are zero-based and box coordinates are normalized from 0 to 1. Empty label files represent images without objects. The pipeline does not infer classes or convert VOC XML, COCO JSON, segmentation masks, or classification folders into detection labels.

During training, the project validates the inputs and prepares deterministic train/validation splits. It writes a generated Ultralytics `dataset.yaml` and split manifest into `output`. Images and labels are symlinked by default to avoid duplicating data; set `link_files: false` to copy them. A non-matching existing output directory is not overwritten.

## Train

Copy [config_example.yaml](config_example.yaml) and set the dataset paths, classes, and training parameters. Relative paths are resolved from the config file.

```sh
python3 scripts/train.py --config my_config.yaml
```

Completed runs are saved under `runs_dir`, with `training_summary.md`, `results.json`, `config.json`, `environment.json`, CSV metrics, Ultralytics training curves, and `weights/best.pt` and `weights/last.pt`. Plot generation is controlled by `plots` in the config.

Use `device: 0` for one GPU. For multiple visible GPUs, use a list such as `device: [0, 1, 2]`; the total `batch` must divide evenly by the number of selected GPUs. Distributed training requires working multi-GPU communication in the installed PyTorch runtime. CPU training must be explicitly selected with `device: cpu` and `allow_cpu: true`.

## Predict

Predictions use class names from the selected Ultralytics-compatible object-detection checkpoint. The command prints detected classes, confidence scores, and pixel-coordinate `xyxy` boxes, then saves annotated images.

```sh
python3 scripts/predict.py \
  --model runs/<run-directory>/weights/best.pt \
  --source /path/to/image.jpg \
  --output detected.jpg \
  --device 0
```

For a directory, pass the directory as `--source` and an output directory as `--output`; nested paths are preserved. Options include `--imgsz`, `--conf`, `--iou`, `--batch`, `--classes`, and `--device cpu`.

## Tests

The unit tests do not require a GPU:

```sh
python -m pip install -e '.[test]'
pytest
```

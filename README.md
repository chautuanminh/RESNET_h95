# ResNet34-H95 DocTamper Server Codebase

This package trains and evaluates a ResNet34-UNet binary segmentation baseline for DocTamper using two input channels: grayscale image and H95/Q95 JPEG residual heatmap. The recorded method name is `grayscale + H95`.

## Full Server Working Directory

Use `/storage/student7/cmt/code_v7` as the parent upload folder. After copying this project folder with `scp -r`, the complete project working directory is:

```text
/storage/student7/cmt/code_v7/Train_resnet_server
```

All training, smoke checks, evaluation, and resume commands should be run from that directory. Keep model outputs outside the code directory; the full training config writes to `/storage/student7/runs/resnet34_h95`.

## Upload The Full Workspace

The server does not allow directory or file deletion. Do not use cleanup scripts or deletion commands. Upload the full workspace by adding or overwriting files only.

From your local machine, create the parent upload folder and copy the whole project directory with `scp`:

```bash
ssh ict17 "mkdir -p /storage/student7/cmt/code_v7"

scp -r "E:\Code\FINAL\Approach\resnet_H95\Train_resnet_server" ict17:/storage/student7/cmt/code_v7/.
```

If `ict17` is only reachable through `frontend`, use the SSH jump host:

```bash
ssh -J frontend ict17 "mkdir -p /storage/student7/cmt/code_v7"

scp -r -J frontend "E:\Code\FINAL\Approach\resnet_H95\Train_resnet_server" ict17:/storage/student7/cmt/code_v7/.
```

This creates or overwrites files under:

```text
/storage/student7/cmt/code_v7/Train_resnet_server
```

After upload, the server directory should contain the project files and folders:

```bash
ssh ict17 "cd /storage/student7/cmt/code_v7/Train_resnet_server && ls -la && ls src configs tests tampering_types"
```

With the jump host:

```bash
ssh -J frontend ict17 "cd /storage/student7/cmt/code_v7/Train_resnet_server && ls -la && ls src configs tests tampering_types"
```

## Server Setup And Smoke Checks

Log in, enter the full working directory, activate the environment, and install dependencies:

```bash
ssh frontend
ssh ict17
cd /storage/student7/cmt/code_v7/Train_resnet_server
source /storage/student7/venvs/doctamper/bin/activate
pip install -r requirements.txt
```

Verify GPU visibility, run unit tests, check real LMDB readability, then run the tiny synthetic pipeline:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
python -m unittest discover
python -m src.smoke --config configs/resnet_h95_config.yaml --sample-count 2
python -m src.run_all --config configs/test_tiny.yaml
```

## Permission Checks For Existing Results

The full run writes to `/storage/student7/runs/resnet34_h95`, not to `/storage/student7/cmt/code_v7/Train_resnet_server`. If a run fails while opening the training log or checkpoints, inspect ownership and permissions from the server:

```bash
pwd
whoami
ls -ld /storage/student7/runs
ls -ld /storage/student7/runs/resnet34_h95
ls -ld /storage/student7/runs/resnet34_h95/doctamper_resnet34_h95_35epochs_comparison
ls -l /storage/student7/runs/resnet34_h95/doctamper_resnet34_h95_35epochs_comparison/training.log
```

The tiny smoke config writes under `/storage/student7/cmt/code_v7/Train_resnet_server/runs/resnet34_h95_tiny/`, so it does not touch the full-run output folder. Prefer fresh result folders owned by the current user:

```bash
mkdir -p runs/resnet34_h95_tiny
mkdir -p /storage/student7/runs/resnet34_h95
```

Use `chmod u+w <path>` only when `whoami` matches the owner shown by `ls -ld` or `ls -l`. Do not use `sudo`, deletion commands, cleanup scripts, or remove existing checkpoints/results unless the folder is confirmed disposable.

## Full Training In tmux

```bash
tmux new -s resnet34_h95_run
cd /storage/student7/cmt/code_v7/Train_resnet_server
source /storage/student7/venvs/doctamper/bin/activate
export CUDA_VISIBLE_DEVICES=0
python -m src.smoke --config configs/resnet_h95_config.yaml --sample-count 2
python -m src.train --config configs/resnet_h95_config.yaml
```

The training process prints concise progress to the terminal and writes the same important events to:

```text
/storage/student7/runs/resnet34_h95/doctamper_resnet34_h95_35epochs_comparison/training.log
```

Watch progress from another shell:

```bash
tail -f /storage/student7/runs/resnet34_h95/doctamper_resnet34_h95_35epochs_comparison/training.log
```

Detach from `tmux` with `Ctrl-b d`, then reattach with:

```bash
tmux attach -t resnet34_h95_run
```

## Resume After OOM Or Interruption

The training loop saves `checkpoints/last_checkpoint.pth` every epoch and also tries to save it when CUDA OOM occurs. If OOM happens, resume with a smaller manual batch size; manual `--batch-size` bypasses autotuning.



```bash
CUDA_VISIBLE_DEVICES=3 python -m src.train \
  --config configs/resnet_h95_config.yaml \
  --resume /storage/student7/runs/resnet34_h95/doctamper_resnet34_h95_35epochs_comparison/checkpoints/last_checkpoint.pth \
  --batch-size 80
```

## Final Evaluation And Diagnostics

```bash
python -m src.post_train_all --config configs/resnet_h95_config.yaml
```

## Split Policy

Only `DocTamperV1-TrainingSet` is used for training and internal validation. The code sorts all TrainingSet indices, shuffles with seed `42`, saves `10,000` validation indices, and uses the remaining `110,000` images for training. Existing split CSVs are reused.

`10,000` validation images are used because the TrainingSet is large. This is about 8.3% validation, close to a 90/10 split. It is enough for stable checkpoint selection while preserving `110,000` images for training. TestingSet, FCD, and SCD remain final evaluation only.

## Expected Outputs

Default full-run output root:

```text
/storage/student7/runs/resnet34_h95/
```

Main run folder:

```text
doctamper_resnet34_h95_35epochs_comparison/
```

Important training outputs:

```text
config_snapshot.yaml
config_resolved.yaml
batch_size_autotune.csv
training.log
train_metrics.csv
val_metrics.csv
plots/training_curves.png
gpu_profile.txt
checkpoints/last_checkpoint.pth
checkpoints/best_model.pth
```

Diagnostics:

```text
failure_case_analysis/
tampering_type_analysis/
```

The official prediction mode is raw sigmoid probability thresholded at `0.5`. Any `no_blob` label is a compatibility label for raw outputs only; the pipeline does not apply connected-component filtering to predictions or metrics.

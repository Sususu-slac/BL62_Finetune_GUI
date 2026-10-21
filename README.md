# BL62 SAM3 Fine-tune GUI

Independent Windows desktop application for preparing COCO datasets and managing SAM3 training in WSL. It has two pages, **Data** and **Training**, with adjustable left/right layouts. Opening the app does not start training.

## Windows installation

Download `BL62_Finetune_GUI_Setup_1.0.4.exe` from [Releases](https://github.com/Sususu-slac/BL62_Finetune_GUI/releases). The installer includes the Windows GUI dependencies and creates **BL62 Fine-tune GUI** on the desktop. A portable ZIP is also available; extract the entire folder before opening the EXE.

The Windows package does not bundle SAM3 weights, training data, WSL, CUDA drivers or the SAM3 training environment. The current GUI runtime, WSL training environment, Python interpreter and SAM3 repository are displayed at the upper right on both Data and Training. The previously saved environment configuration is retained; training checks it automatically before launch.

## Data

- Open a COCO JSON, a folder containing `annotations.json` and `images/`, an existing split dataset, or an image folder.
- Select masks, draw with Brush or Polygon, erase, delete, merge, undo/redo, zoom with the mouse wheel and pan with the middle button or Pan tool. Press Enter to finish an instance; Escape returns to Select.
- The Label dropdown above the image changes the selected instance's category. Labels → Rename changes a category name globally. If the name already exists (including different capitalization), choose Merge to move all instances to the existing category, or Cancel to leave everything unchanged. Merging categories preserves each individual mask and overlap.
- Assign the current image using Training / Validation below the canvas. Each original image-list row shows its assignment.
- Random Split supports All Images or Unassigned Only, linked train/validation percentages and a fixed seed. Preview shows the proposed assignments.
- **Save Dataset** is at the bottom of the left panel, immediately left of **Training →**. Select a save location and a new dataset folder name. Existing nonempty output folders are preserved.
- Saving exports lossless COCO RLE masks (including holes, one-pixel regions and overlapping instances), copied images and `train/annotations.json` / `val/annotations.json`. Unassigned images can be retained in a separate `unassigned/` folder. Both training and validation need at least one image.
- Source files remain untouched. Edits and assignments are held in memory until exported. Reopen the exported dataset to continue review.
- The saved dataset path automatically fills Training → Dataset; Browse can select another existing dataset.

## Training

Settings appear in this order: Dataset, Checkpoint, GPU, Parameters, Output. Open Output Folder is directly below the Output path and stays available during training. Environment settings have been removed from the left panel; the shared upper-right header displays the current training environment and paths on both pages.

One checkpoint picker serves both modes:

- **Fine-tune** loads model weights and starts a new experiment.
- **Resume Training** restores a full training checkpoint including optimizer and epoch. Select a compatible full SAM3 training checkpoint, typically `checkpoints/checkpoint.pt`. For experiments created by this GUI, `gui_launch.json` restores the initial checkpoint and checks the original dataset/GPU selection. Epochs is the total target epoch count and must exceed the saved epoch.

GPU Device IDs are physical CUDA indices, for example `0`, `1` or `0,1`. The GPU Status panel below training progress shows every NVIDIA GPU, utilization and VRAM used/total in GiB, highlighting the selected devices. It refreshes in the background every two seconds while Training is visible and pauses on Data. Readings are device-wide and include other programs. Unavailable metrics display N/A; failed queries clear stale readings and show an unavailable status. Monitoring uses nvidia-smi without loading a model and does not write into the training log. Batch size is per GPU. Output must be a new or empty experiment directory. The application creates a separate runtime YAML and never modifies the selected template. Training settings are locked during a run. Safe Stop targets only the recorded training process group, including its distributed workers. Closing during training requests a safe stop and waits for exit.

Environment defaults on the original workstation:

- WSL distribution: `Ubuntu-24.04`
- SAM3 repository: `/home/subo/sam3`
- Python: `/home/subo/miniforge3/envs/sam3_finetune_wsl/bin/python`
- Template: `/home/subo/sam3/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml`

The existing environment configuration is retained per Windows user. The existing Roboflow full fine-tuning template is required; arbitrary SAM3 config schemas are not supported. Before launch, training verifies Python, SAM3 and dependencies, CUDA, visible GPUs and checkpoint structure. Metrics and full-state checkpoint paths appear on the right. Each successful new training launch clears the on-screen log; existing experiment log files remain on disk. Vertical scrollbars use the shared dark rounded style on both pages.

This release saves SAM3's full training checkpoint. Creating a reduced inference-only checkpoint is a separate conversion step; no conversion is silently performed.

## Development

```bash
python -m pip install -r requirements.txt
python app.py
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
```

For a Windows EXE, install PyInstaller in the GUI Python environment and run `build_windows.ps1`. Inno Setup 6 builds the optional installer via `installer.iss`. The training environment remains separate from the GUI Python environment.

The icon was generated with the built-in imagegen tool using the user's BL62 Label icon as a style reference; the prompt is in `assets/icon_prompt.txt`.

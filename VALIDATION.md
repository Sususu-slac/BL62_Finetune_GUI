# v1.0.0 validation — 2026-10-20

- 18 automated tests passed on Windows and Linux. Coverage includes brush/eraser, polygon creation, selection, deletion, merging, undo/redo, instance relabeling, global rename, manual assignment, random split, background export and automatic transfer of the saved dataset path to Training.
- COCO round-trip tests preserve overlapping masks, holes and isolated pixels. Export failures leave no partial output dataset; nonempty destinations are protected; filename collisions are resolved.
- The actual launcher/safe-stop process-group test passed through Windows → WSL and directly on Linux. Training settings remain locked during a run. The tests use an isolated lightweight process, not a GPU model.
- The frozen Windows EXE passed a separate startup/export/reload smoke test. After installation, the installed EXE passed the same test. Both pages loaded, the dataset path transferred correctly and the mask round trip was lossless.
- The Windows installer completed with exit code 0. Desktop shortcut target and `.ico` location were verified.
- A separate real Windows → WSL training run completed with exit code 0: four copied images (two train / two validation), one epoch, batch size 1, GPU 1 (NVIDIA RTX 6000 Ada 48 GB). Training, COCO validation and full checkpoint saving completed. The original labeling dataset was only read.

The real run verifies training integration, not convergence or model quality. Its complete optimizer/model checkpoint was about 9.4 GiB. Saving that checkpoint through the Windows-mounted test output directory dominated the test runtime; the application's default experiment location is inside WSL. A complete second training run in Resume mode was not performed; resume configuration and full-checkpoint requirements are implemented, while the completed training run verified the standard SAM3 full-state output.

No datasets, user checkpoints or test-trained model files are included in the repository or Windows distribution. The Windows distribution includes the GUI runtime; SAM3, its training environment, CUDA and weights remain separately configured under Environment.

## v1.0.2 — duplicate label names

22 tests passed on Windows and Linux. New coverage verifies case-insensitive duplicate-name confirmation, Cancel as the default, explicit Merge, cancellation preserving unsaved strokes and undo history, merging the current image and other images, and lossless export/reload after merging categories across train/validation splits. Individual instances and overlapping pixels remain separate.

## v1.0.3 — output folder control

24 tests passed on Windows and Linux. Training setup remains locked during a run while Open Output Folder stays enabled under the Output path. The output path controls retain their previous behavior. The shared runtime header remains visible on both tabs, follows the retained training configuration, and replaces the removed Environment settings section.

New-run logging checks confirm that a valid launch clears prior on-screen output, while failed configuration validation preserves the previous log. Vertical scrollbar styling was checked visually.

## v1.0.4 — live GPU telemetry

29 tests passed on Windows and Linux. Added coverage checks multi-GPU/quoted CSV parsing, unavailable metrics, stale-data removal, selection display, non-overlapping background queries, an active Qt heartbeat during slow queries, timeout recovery and clean shutdown.

A native Windows integration check sampled both RTX 6000 Ada devices three times with the production 2,000 ms timer. Observed sample intervals were 1.91 and 2.14 seconds; the Qt heartbeat ticked 200 times. Training Log content stayed unchanged. Switching to Data paused polling and returning to Training resumed it. Device names, utilization and used/total VRAM rendered correctly in the native Windows window. No model was loaded.

The metric selector no longer shadows QWidget's native `metric()` method. Regression checks verify the native method remains callable and DPI metrics remain valid.

The frozen Windows EXE was also launched normally and checked through Windows UI Automation: switching to Training displayed both GPU 0 and GPU 1 rows populated from the real driver query.

## v1.0.5 — GPU selection and concise runtime header

30 tests passed on Windows and Linux. New checks verify that left GPU checkboxes drive the actual training device count, refreshes preserve selection, training locks selection, and no selection blocks launch. The right panel immediately filters cached telemetry to selected devices and keeps the filter after new samples. Both tabs share the two-line Environment/Repo header. No new model training was needed for these UI changes.

The frozen EXE passed native Windows UI Automation: two initial GPU charts, uncheck GPU 0 and retain only GPU 1 across the next telemetry refresh, recheck GPU 0 and immediately restore both charts. The exact two-line Environment/Repo header was also verified. A native screenshot confirmed the left checkbox layout and selected-only visualization.

## v1.0.6 — automatic inference export

36 tests passed on Windows and Linux. Conversion tests cover trained-weight replacement, retained tracker/interactive-neck weights, optimizer exclusion, fallback templates, rejection of missing or incompatible weights, disk-write failure cleanup and refusing to overwrite an existing inference output. A real QProcess worker using tiny tensors verifies Windows → WSL launch, all six progress stages, a responsive Qt heartbeat, completion notification and output creation. GUI checks verify automatic transition without an acknowledgement, settings locked during conversion, completion/failure recovery, and no conversion after failed/stopped training.

The previous real 9.4 GiB training checkpoint was successfully converted into a separate 3.2 GiB SAM3 inference checkpoint on CPU. The full training checkpoint was retained. The converter validates every trained tensor name/shape and publishes its output after verifying the saved structure. SAM3's optional SAM2 interactive neck and tracker weights are retained from the inference template; missing core detector weights are rejected.

The real exported model contains 1,465 inference tensors. All 1,134 trained tensors were compared for exact equality, as were retained tracker/interactive-neck tensors. The actual SAM3 inference loader instantiated the exported model on CPU with instance interactivity enabled, and every trained parameter in the resulting model matched the training checkpoint. Output size: 3,450,235,371 bytes; source training checkpoint: 10,081,250,310 bytes.

The frozen Windows EXE passed its conversion smoke test with the bundled worker copied into WSL: all six progress stages appeared, the Qt heartbeat ticked 238 times, and the completion dialog displayed the correct Windows output path. Native screenshots verified the progress bar and completion dialog.

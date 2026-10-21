# v1.0.0 validation — 2026-10-20

- 18 automated tests passed on Windows and Linux. Coverage includes brush/eraser, polygon creation, selection, deletion, merging, undo/redo, instance relabeling, global rename, manual assignment, random split, background export and automatic transfer of the saved dataset path to Training.
- COCO round-trip tests preserve overlapping masks, holes and isolated pixels. Export failures leave no partial output dataset; nonempty destinations are protected; filename collisions are resolved.
- The actual launcher/safe-stop process-group test passed through Windows → WSL and directly on Linux. Training settings remain locked during a run. The tests use an isolated lightweight process, not a GPU model.
- The frozen Windows EXE passed a separate startup/export/reload smoke test. After installation, the installed EXE passed the same test. Both pages loaded, the dataset path transferred correctly and the mask round trip was lossless.
- The Windows installer completed with exit code 0. Desktop shortcut target and `.ico` location were verified.
- A separate real Windows → WSL training run completed with exit code 0: four copied images (two train / two validation), one epoch, batch size 1, GPU 1 (NVIDIA RTX 6000 Ada 48 GB). Training, COCO validation and full checkpoint saving completed. The original labeling dataset was only read.

The real run verifies training integration, not convergence or model quality. Its complete optimizer/model checkpoint was about 9.4 GiB. Saving that checkpoint through the Windows-mounted test output directory dominated the test runtime; the application's default experiment location is inside WSL. A complete second training run in Resume mode was not performed; resume configuration and full-checkpoint requirements are implemented, while the completed training run verified the standard SAM3 full-state output.

No datasets, user checkpoints or test-trained model files are included in the repository or Windows distribution. The Windows distribution includes the GUI runtime; SAM3, its training environment, CUDA and weights remain separately configured under Environment.

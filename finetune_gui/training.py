from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = Path("/home/subo/sam3/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml")
SAM3_ROOT = Path("/home/subo/sam3")
TRAIN_PYTHON = Path("/home/subo/miniforge3/envs/sam3_finetune_wsl/bin/python")


def training_python_exists() -> bool:
    """Windows cannot resolve WSL's relative ``bin/python`` Conda symlink."""
    candidate = host_path(TRAIN_PYTHON)
    if candidate.is_file():
        return True
    bin_dir = host_path(TRAIN_PYTHON.parent)
    return bin_dir.is_dir() and any(
        path.is_file() for path in bin_dir.glob("python3.*")
    )


def host_path(path: Path) -> Path:
    raw=str(path).replace("\\","/")
    if __import__('os').name == 'nt' and raw.startswith('/') and not raw.startswith('//'):
        if re.match(r'/mnt/[a-z]/',raw): return Path(raw[5].upper()+':'+raw[6:].replace('/','\\'))
        return Path(r"\\wsl.localhost\Ubuntu-24.04"+raw.replace('/','\\'))
    return path


def wsl_string(path: Path) -> str:
    raw=str(path).replace('\\','/')
    if re.match(r'^[A-Za-z]:/',raw): return '/mnt/'+raw[0].lower()+raw[2:]
    if raw.startswith('//wsl.localhost/') or raw.startswith('//wsl$/'):
        parts=raw.split('/'); return '/'+ '/'.join(parts[4:])
    return raw


@dataclass
class TrainingSettings:
    config: Path = DEFAULT_CONFIG
    dataset: Path = Path("/home/subo/sam3/dataset_split300")
    output: Path = Path("/home/subo/sam3/experiments/gui_training_run")
    checkpoint: Path = Path("/mnt/d/Chcekpoints/checkpoint_demo_05142026.pt")
    category: str = "particle"
    epochs: int = 200
    batch_size: int = 1
    gpus: int = 2
    workers: int = 10
    val_every: int = 10
    save_every: int = 10
    lr_scale: float = 0.05
    resume_from: Path | None = None
    inference_base: Path = Path("/mnt/d/Chcekpoints/checkpoint_demo_05142026.pt")
    export_inference: bool = False
    repository: Path = SAM3_ROOT
    python: Path = TRAIN_PYTHON
    distro: str = "Ubuntu-24.04"


def dataset_summary(root: Path) -> dict:
    root = host_path(root)
    result = {}
    for split in ("train", "val"):
        images = root / split / "images"
        annotation = root / split / "annotations.json"
        if not images.is_dir():
            raise ValueError(f"Missing directory: {images}")
        if not annotation.is_file():
            raise ValueError(f"Missing annotation: {annotation}")
        count = sum(1 for p in images.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
        data = json.loads(annotation.read_text(encoding="utf-8"))
        result[split] = {"images": count, "annotations": len(data.get("annotations", [])), "categories": [x.get("name", "") for x in data.get("categories", [])]}
    return result


def validate(settings: TrainingSettings) -> list[str]:
    errors = []
    for path, label in ((settings.config, "YAML"), (settings.checkpoint, "base checkpoint")):
        if not host_path(path).is_file(): errors.append(f"Missing {label}: {path}")
    if not host_path(settings.repository / "sam3/train/train.py").is_file():
        errors.append(f"Missing SAM3 training code: {settings.repository}")
    if not host_path(settings.python).exists() and not any(host_path(settings.python.parent).glob("python3.*")):
        errors.append(f"Missing training Python: {TRAIN_PYTHON}")
    try: dataset_summary(settings.dataset)
    except Exception as exc: errors.append(str(exc))
    output = host_path(settings.output)
    if output.exists() and any(output.iterdir()): errors.append(f"Output directory is not empty: {settings.output}")
    if settings.gpus < 1: errors.append("GPU count must be at least 1")
    if settings.resume_from and not host_path(settings.resume_from).is_file(): errors.append(f"Resume checkpoint not found: {settings.resume_from}")
    if settings.export_inference and not host_path(settings.inference_base).is_file(): errors.append(f"Inference base checkpoint not found: {settings.inference_base}")
    return errors


def render_config(settings: TrainingSettings) -> str:
    text = host_path(settings.config).read_text(encoding="utf-8")
    quote = lambda value: json.dumps(value, ensure_ascii=False)
    values = {
        'roboflow_vl_100_root': quote(wsl_string(settings.dataset)),
        'experiment_log_dir': quote(wsl_string(settings.output)),
        'supercategory': quote(settings.category),
        'max_epochs': settings.epochs,
        'num_train_workers': settings.workers,
        'lr_scale': settings.lr_scale,
        'train_batch_size': settings.batch_size,
        'val_epoch_freq': settings.val_every,
        'checkpoint_path': quote(wsl_string(settings.checkpoint)),
        'save_freq': settings.save_every,
        'gpus_per_node': settings.gpus,
        'use_cluster': 'false',
    }
    for name,value in values.items():
        def replacement(match):
            result = match.group(1)+' '+str(value)
            if name=='save_freq' and settings.resume_from:
                indent=re.match(r'[ \t]*',match.group(1)).group(0)
                result += '\n'+indent+'resume_from: '+quote(wsl_string(settings.resume_from))
            return result
        text,count=re.subn(r'^([ \t]*'+name+r':)[ \t]*.*$',replacement,text,count=1,flags=re.MULTILINE)
        if count!=1: raise ValueError('Missing template field: '+name)
    return text


def config_name_for(output: Path) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", output.name).strip("_") or "gui_run"
    return f"configs/gui_runs/{safe}.yaml"


def command(settings: TrainingSettings) -> list[str]:
    return ["env", "PYTHONNOUSERSITE=1", wsl_string(settings.python), "-m", "sam3.train.train", "-c", config_name_for(settings.output), "--use-cluster", "0", "--num-gpus", str(settings.gpus)]


def load_stats(path: Path) -> list[dict]:
    path = host_path(path)
    if not path.is_file(): return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass
    return rows

"""Patient-safe partitions, validated binary masks and paired transformations."""
from pathlib import Path
import hashlib
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_pair(image_path: str, mask_path: str) -> tuple[np.ndarray, np.ndarray]:
    if Path(image_path).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Unsupported image: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if image is None or mask is None or mask.ndim != 2:
        raise ValueError(f"Unreadable image or non-single-channel mask: {image_path}, {mask_path}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected 8-bit exported image; calibrate/convert higher bit depths explicitly: {image_path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        code = cv2.COLOR_BGR2RGB if image.shape[2] == 3 else cv2.COLOR_BGRA2RGB
        image = cv2.cvtColor(image, code)
    else:
        raise ValueError(f"Unsupported image channel structure: {image_path}")
    values = set(np.unique(mask).tolist())
    if not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError(f"Mask must use 0/1 or 0/255: {mask_path}; got {values}")
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image/mask dimensions differ: {image_path}")
    return image, (mask > 0).astype(np.float32)


def read_manifest(path: str) -> pd.DataFrame:
    source = Path(path).resolve()
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    required = {"image_id", "patient_id", "image_path", "mask_path"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"Nonempty CSV must include {sorted(required)}")
    if (frame[list(required)] == "").any().any() or frame.image_id.duplicated().any():
        raise ValueError("Empty fields or duplicate image_id")
    for column in ("image_path", "mask_path"):
        frame[column] = frame[column].map(lambda s: str((source.parent / s).resolve()))
        if not frame[column].map(lambda s: Path(s).is_file()).all():
            raise ValueError(f"Missing file in {column}")
    if frame.image_path.duplicated().any():
        raise ValueError("Duplicate image paths")
    positives, image_hashes, mask_hashes, pixel_hashes = [], [], [], []
    for row in frame.itertuples():
        image, mask = read_pair(row.image_path, row.mask_path)
        positives.append(int(mask.any()))
        image_hashes.append(sha256(Path(row.image_path)))
        mask_hashes.append(sha256(Path(row.mask_path)))
        pixel_hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
    frame["foreground"] = positives
    frame["image_sha256"], frame["mask_sha256"] = image_hashes, mask_hashes
    if len(set(pixel_hashes)) != len(pixel_hashes) or frame.image_sha256.duplicated().any():
        raise ValueError("Duplicate images detected; deduplicate before splitting")
    return frame.sort_values("image_id").reset_index(drop=True)


def make_splits(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    frame = frame.copy()
    if "split" not in frame:
        frame["split"] = "dev"
        _, test = next(GroupShuffleSplit(n_splits=1, test_size=cfg.test_fraction, random_state=cfg.seed).split(frame, groups=frame.patient_id))
        frame.loc[test, "split"] = "test"
    if not set(frame.split) <= {"dev", "test", "external"} or not {"dev", "test"} <= set(frame.split):
        raise ValueError("split must contain dev and test; optional external is supported")
    if (frame.groupby("patient_id").split.nunique() > 1).any():
        raise ValueError("Patient overlaps partitions")
    dev = frame[frame.split == "dev"]
    if dev.patient_id.nunique() < cfg.folds:
        raise ValueError("Insufficient development patients for requested folds")
    if "fold" in frame:
        frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
        dev = frame[frame.split == "dev"]
        if set(dev.fold) != set(range(cfg.folds)) or (frame.loc[frame.split != "dev", "fold"] != -1).any():
            raise ValueError("Invalid frozen fold assignments")
        if (dev.groupby("patient_id").fold.nunique() > 1).any():
            raise ValueError("Patient crosses frozen development folds")
    else:
        frame["fold"] = -1
        for fold, (_, val) in enumerate(GroupKFold(cfg.folds).split(dev, groups=dev.patient_id)):
            frame.loc[dev.index[val], "fold"] = fold
    return frame


def verify_locked_data(cfg: Config, locked: pd.DataFrame) -> None:
    current = read_manifest(cfg.manifest)
    cols = ["image_id", "patient_id", "image_path", "mask_path", "image_sha256", "mask_sha256"]
    if current[cols].astype(str).to_dict("records") != locked[cols].astype(str).to_dict("records"):
        raise ValueError("Dataset changed since training; use a new run")
    if "fold" in current and current.fold.astype(int).tolist() != locked.fold.astype(int).tolist():
        raise ValueError("Frozen folds changed since training")
    if "split" in current and current.split.tolist() != locked.split.tolist():
        raise ValueError("Manifest partitions changed since training")


def geometry(cfg: Config) -> A.Compose:
    """Longest-side resize and exact paired, centered zero padding."""
    return A.Compose([
        A.LongestMaxSize(max_size=cfg.image_size, interpolation=cv2.INTER_LINEAR,
                         mask_interpolation=cv2.INTER_NEAREST),
        A.PadIfNeeded(min_height=cfg.image_size, min_width=cfg.image_size,
                      position="center", border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
    ], additional_targets={"valid": "mask"}, seed=cfg.seed)


def letterbox_pair(image: np.ndarray, mask: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = geometry(cfg)(image=image, mask=mask, valid=np.ones(mask.shape, dtype=np.float32))
    return result["image"], result["mask"], result["valid"]


def transforms(cfg: Config, train: bool = False) -> A.Compose:
    steps = [A.LongestMaxSize(max_size=cfg.image_size, interpolation=cv2.INTER_LINEAR,
                             mask_interpolation=cv2.INTER_NEAREST)]
    if train:
        steps += [A.HorizontalFlip(p=0.5), A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                  rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5),
                  A.RandomBrightnessContrast(p=0.3), A.ElasticTransform(alpha=1, sigma=50, p=0.2)]
    steps += [A.PadIfNeeded(min_height=cfg.image_size, min_width=cfg.image_size,
                           position="center", border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
              A.Normalize(mean=MEAN, std=STD), ToTensorV2()]
    return A.Compose(steps, additional_targets={"valid": "mask"}, seed=cfg.seed)


class BiomedicalDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cfg: Config, train: bool = False) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transforms(cfg, train)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        image, mask = read_pair(row.image_path, row.mask_path)
        result = self.transform(image=image, mask=mask, valid=np.ones(mask.shape, dtype=np.float32))
        return {"image": result["image"].float(), "mask": result["mask"].unsqueeze(0).float(),
                "valid": result["valid"].unsqueeze(0).float(),
                "image_id": row.image_id, "patient_id": row.patient_id}

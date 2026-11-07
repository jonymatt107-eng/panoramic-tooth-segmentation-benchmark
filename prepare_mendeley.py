"""Verify and prepare Mendeley jrz4nj82zv v1, without launching training."""
from pathlib import Path, PurePosixPath
import argparse
from datetime import datetime, timezone
import json
import shutil
import stat
import zipfile
import cv2
import numpy as np
import pandas as pd
from dataset import read_manifest, sha256

SOURCE_URL = "https://data.mendeley.com/datasets/jrz4nj82zv/1"
ARCHIVE_SHA256 = "954ff25f679810a2ca0a5a110ac1c313a3a824820f17ead9bcd30932263305e3"
PREFIX = "Panoramic_Dental_Xray_Segmentation_Dataset"


def prepare(archive: Path, output: Path) -> Path:
    archive, output = archive.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Archive checksum differs from publisher's version-1 checksum")
    staging = output.with_name(output.name + ".partial")
    staging.mkdir(parents=True, exist_ok=False)
    raw = staging / "raw"
    raw.mkdir()
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if len(members) > 10000 or sum(m.file_size for m in members) > 2_000_000_000:
            raise ValueError("Unexpected archive dimensions")
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                raise ValueError("Unsafe ZIP path")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("ZIP symbolic links are not accepted")
            destination = raw / member.filename
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as src, destination.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
    folder = raw / PREFIX
    images = sorted((folder / "images").glob("*.jpg"))
    masks = sorted((folder / "masks").glob("*.png"))
    if len(images) != 329 or len(masks) != 329:
        raise ValueError(f"Expected 329 image/mask pairs, found {len(images)}/{len(masks)}")
    classes = pd.read_csv(folder / "_classes.csv", skipinitialspace=True)
    if classes.iloc[:, 0].tolist() != [0, 1] or classes.iloc[:, 1].tolist() != ["background", "teeths"]:
        raise ValueError("Unexpected source label vocabulary")
    prepared = staging / "prepared"
    (prepared / "images").mkdir(parents=True)
    (prepared / "masks").mkdir()
    rows, mask_counts, source_ids = [], {}, []
    used = set()
    for index, image_path in enumerate(images):
        mask_path = folder / "masks" / (image_path.stem + "_mask.png")
        if not mask_path.is_file():
            raise ValueError(f"Missing corresponding mask: {image_path.name}")
        used.add(mask_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim != 2:
            raise ValueError(f"Single-channel source mask required: {mask_path}")
        values = set(np.unique(mask).tolist())
        if not (values <= {0, 1} or values <= {0, 255}):
            raise ValueError(f"Unexpected source mask values: {mask_path}: {values}")
        key = ','.join(map(str, sorted(values)))
        mask_counts[key] = mask_counts.get(key, 0) + 1
        binary = (mask > 0).astype(np.uint8)
        image_id = f"md1_image_{index:04d}"
        canonical = image_path.stem.split(".rf.")[0]
        source_ids.append(canonical)
        # Roboflow-export hashes are not patient IDs. Keep all variants of a source together.
        patient_id = f"md1_patient_{canonical}"
        destination = prepared / "images" / image_path.name
        shutil.copy2(image_path, destination)
        mask_destination = prepared / "masks" / f"{image_id}.png"
        if not cv2.imwrite(str(mask_destination), binary):
            raise OSError(f"Could not write {mask_destination}")
        rows.append(dict(image_id=image_id, patient_id=patient_id,
                         image_path=str(destination.relative_to(staging)),
                         mask_path=str(mask_destination.relative_to(staging)),
                         source_image_id=canonical, source_filename=image_path.name,
                         patient_id_basis="publisher_one_image_per_patient"))
    if used != set(masks) or len(set(source_ids)) != 329:
        raise ValueError("Unpaired masks or repeated canonical IDs; patient mapping needs review")
    pd.DataFrame(rows).to_csv(staging / "manifest_candidate.csv", index=False)
    provenance = dict(dataset_name="Mendeley panoramic tooth segmentation v1", dataset_id="jrz4nj82zv",
        version=1, doi="10.17632/jrz4nj82zv.1", source_url=SOURCE_URL,
        license="CC BY 4.0", acquired_at_utc=datetime.now(timezone.utc).isoformat(),
        archive_filename=archive.name, archive_sha256=ARCHIVE_SHA256,
        archive_bytes=archive.stat().st_size, expected_images=329,
        endpoint="union of tooth-region foreground; not caries-lesion segmentation",
        acquisition="smartphone photographs of panoramic radiographs, Sylhet, Bangladesh",
        source_label_vocabulary={"0": "background", "1": "teeths"},
        source_mask_value_counts=mask_counts,
        prepared_mask_encoding="uint8 single channel; foreground=1, background=0; no spatial resampling",
        patient_grouping=dict(basis="publisher_one_image_per_patient", source_url=SOURCE_URL,
            publisher_statement="329 panoramic radiographs obtained from 329 individual patients",
            mapping="md1_patient_ + source basename before .rf.; export hashes do not define patients",
            independent_patient_registry_available=False,
            evidence_limit="Relies on publisher cohort statement; anonymous pixels cannot establish identity"),
        review_status="awaiting user review; no training authorized or performed")
    (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + '\n')
    read_manifest(str(staging / "manifest_candidate.csv"))
    staging.rename(output)
    print(f"Prepared {len(rows)} pairs: {output / 'manifest_candidate.csv'}")
    return output / "manifest_candidate.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/mendeley_teeth_v1"))
    args = parser.parse_args()
    prepare(args.archive, args.output)

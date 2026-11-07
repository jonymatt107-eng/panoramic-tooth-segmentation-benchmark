# Dataset preparation and inspection

**Current status (November 6, 2026):** the user subsequently approved the frozen cohort and the full study has completed. The instructions below preserve the acquisition/inspection workflow; real outputs are in the sibling study_results directory.

The selected candidate is the **Dental Panoramic Radiography Dataset for Pixel level Semantic Segmentation of Teeth**, Mendeley version 1: https://data.mendeley.com/datasets/jrz4nj82zv/1 (CC BY 4.0; DOI https://doi.org/10.17632/jrz4nj82zv.1).

The publisher reports 329 radiographs from 329 individual patients. These are smartphone photographs of panoramic radiographs, not direct digital detector exports. The endpoint is the union of tooth regions; this dataset does not supply caries-lesion masks for the proposed binary pipeline.

## Reproduce acquisition

The archive is public and needs no API key:

```bash
curl --fail --location --continue-at - --output Panoramic_Dental_Xray_Segmentation_Dataset.zip \
 'https://data.mendeley.com/public-files/datasets/jrz4nj82zv/files/991e8020-e043-4f73-9697-e5624373ddd9/file_downloaded'
```

Or open the dataset page and download its ZIP manually. The preparation script verifies SHA-256 `954ff25f679810a2ca0a5a110ac1c313a3a824820f17ead9bcd30932263305e3` before extracting. It rejects unsafe archive paths, symbolic links, unexpected counts, unpaired masks and unexpected labels.

From this code directory with the requirements installed:

```bash
python prepare_mendeley.py \
 --archive /absolute/path/Panoramic_Dental_Xray_Segmentation_Dataset.zip \
 --output data/mendeley_teeth_v1

python inspect_dataset.py \
 --manifest data/mendeley_teeth_v1/manifest_candidate.csv \
 --provenance data/mendeley_teeth_v1/provenance.json \
 --output /absolute/path/dataset_inspection
```

Preparation preserves the raw archive contents and copies the images at their original decoded resolution. Prepared masks are lossless uint8 0/1 PNGs, without resizing. The manifest records each original filename and canonical source ID. Anonymous patient group keys use the source basename before `.rf.`; the export hash is not a patient identifier. The one-image-per-patient interpretation relies on the publisher's explicit statement, rather than an independently available clinical patient registry.

Inspection checks **all** images and masks, rejects nonbinary labels and dimension mismatches, checks exact byte/pixel duplicates using the benchmark loader, and plots three random samples with seed 42. It verifies that patient groups do not cross a proposed development/test split or development CV folds. `proposed_splits.csv` is an inspection artifact, not an executed experiment or approved final split.

Review `inspection.json`, `per_image_checks.csv` and `three_sample_overlays.png`. In the current workspace these are saved in the sibling `dataset_inspection` output directory. The downloaded data are local under `data/mendeley_teeth_v1` and are excluded from the lightweight source ZIP.

**No training commands are run by either script.** The existing `example_config.json` is deliberately not pointed at this candidate dataset. Review the data and acquisition limitations before approving training or modifying the scientific protocol. A visual overlay check cannot establish expert annotation accuracy, patient identity or clinical validity.

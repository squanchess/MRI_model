#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Survey Script.

Scans all dataset folders under a root directory, summarizes file formats,
counts, directory depth, and generates a Markdown report.

Usage:
    python survey_datasets.py /path/to/datasets --output survey_report.md
"""

import os
import re
import sys
import argparse
from collections import defaultdict


# ---------------------------------------------------------------------------
# Known medical image file extensions
# ---------------------------------------------------------------------------
KNOWN_EXTENSIONS = {
    ".nii.gz": "NIfTI (compressed)",
    ".nii": "NIfTI",
    ".dcm": "DICOM",
    ".dicom": "DICOM",
    ".mha": "MetaImage",
    ".mhd": "MetaImage header",
    ".nrrd": "NRRD",
    ".nhdr": "NRRD header",
    ".mnc": "MINC",
    ".h5": "HDF5",
    ".hdf5": "HDF5",
    ".npz": "NumPy archive",
    ".npy": "NumPy array",
    ".mat": "MATLAB",
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".json": "JSON",
    ".csv": "CSV",
    ".tsv": "TSV",
    ".txt": "Text",
    ".xml": "XML",
    ".xlsx": "Excel",
    ".zip": "ZIP archive",
    ".tar": "TAR archive",
    ".gz": "GZ compressed",
    ".tar.gz": "TAR.GZ archive",
    ".tgz": "TAR.GZ archive",
}

# Keywords indicating mask / annotation files (to be excluded from training)
MASK_KEYWORDS = ["mask", "seg", "label", "gt", "annotation", "contour", "roi", "binary"]

# Keywords for MRI modality detection
MODALITY_KEYWORDS = {
    "t1": "T1",
    "t1w": "T1w",
    "t1ce": "T1-CE",
    "t1gd": "T1-Gd",
    "t2": "T2",
    "t2w": "T2w",
    "flair": "FLAIR",
    "dwi": "DWI",
    "adc": "ADC",
    "swi": "SWI",
    "pd": "PD",
    "bold": "BOLD",
    "dce": "DCE",
    "phase": "Phase",
}


def get_ext(filepath):
    """Get file extension. Handles .nii.gz and .tar.gz as single extensions."""
    name = os.path.basename(filepath).lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    if name.endswith(".tar.gz"):
        return ".tar.gz"
    _, ext = os.path.splitext(name)
    return ext


def is_likely_mask(filepath):
    """Check if filepath suggests a mask / annotation file."""
    basename = os.path.basename(filepath).lower()
    parent = os.path.basename(os.path.dirname(filepath)).lower()
    combined = parent + "/" + basename
    for kw in MASK_KEYWORDS:
        if kw in combined:
            return True
    return False


def detect_modality(filepath):
    """Infer MRI modality from filename and parent directory."""
    basename = os.path.basename(filepath).lower()
    parent = os.path.basename(os.path.dirname(filepath)).lower()
    combined = parent + "_" + basename
    for kw, modality in MODALITY_KEYWORDS.items():
        pattern = r"(?:^|[_\-./\s])" + re.escape(kw) + r"(?:$|[_\-./\s])"
        if re.search(pattern, combined):
            return modality
    return "unknown"


def is_likely_dicom_dir(dirpath):
    """Check if directory likely contains DICOM files (many .dcm or no-ext files)."""
    try:
        entries = os.listdir(dirpath)
    except PermissionError:
        return False
    if len(entries) < 5:
        return False
    no_ext_count = 0
    dcm_count = 0
    for e in entries:
        full = os.path.join(dirpath, e)
        if not os.path.isfile(full):
            continue
        if e.lower().endswith(".dcm"):
            dcm_count += 1
        elif "." not in e and e != "DICOMDIR":
            no_ext_count += 1
    return (no_ext_count + dcm_count) > len(entries) * 0.5


def survey_one_dataset(dataset_dir):
    """Scan one dataset directory and return a summary dict."""
    info = {
        "path": dataset_dir,
        "name": os.path.basename(dataset_dir.rstrip("/")),
        "total_files": 0,
        "total_dirs": 0,
        "max_depth": 0,
        "ext_counts": defaultdict(int),
        "nifti_count": 0,
        "nifti_mask_count": 0,
        "nifti_non_mask": 0,
        "dicom_dirs": [],
        "modalities": defaultdict(int),
        "sample_files": [],
        "has_archives": False,
        "img2d_count": 0,
    }

    base_depth = dataset_dir.rstrip("/").count("/")

    for root, dirs, files in os.walk(dataset_dir, followlinks=True):
        depth = root.count("/") - base_depth
        if depth > info["max_depth"]:
            info["max_depth"] = depth
        info["total_dirs"] += len(dirs)

        if is_likely_dicom_dir(root):
            info["dicom_dirs"].append(root)

        for fname in files:
            info["total_files"] += 1
            fpath = os.path.join(root, fname)
            ext = get_ext(fpath)
            info["ext_counts"][ext] += 1

            if ext in (".nii.gz", ".nii"):
                info["nifti_count"] += 1
                if is_likely_mask(fpath):
                    info["nifti_mask_count"] += 1
                else:
                    info["nifti_non_mask"] += 1
                    info["modalities"][detect_modality(fpath)] += 1
                    if len(info["sample_files"]) < 5:
                        info["sample_files"].append(
                            os.path.relpath(fpath, dataset_dir)
                        )

            if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                info["img2d_count"] += 1

            if ext in (".zip", ".tar", ".tar.gz", ".tgz"):
                info["has_archives"] = True

    return info


def format_report(results, output_path=None):
    """Generate a Markdown survey report."""
    lines = []
    lines.append("# Dataset Survey Report\n")

    # --- Summary table ---
    lines.append("## Overview\n")
    header = (
        "| Dataset | NIfTI(img) | NIfTI(mask) | DICOM dirs "
        "| 2D imgs | Depth | Modality | Status |"
    )
    sep = (
        "|---------|-----------|-------------|----------"
        "|---------|-------|----------|--------|"
    )
    lines.append(header)
    lines.append(sep)

    ready = []
    need_convert = []
    unclear = []

    for r in results:
        top3 = sorted(r["modalities"].items(), key=lambda x: -x[1])[:3]
        modalities = ", ".join("{}({})".format(m, c) for m, c in top3) or "-"

        if r["nifti_non_mask"] > 0:
            status = "Ready"
            ready.append(r)
        elif r["dicom_dirs"]:
            status = "DICOM->NIfTI"
            need_convert.append(r)
        elif r["img2d_count"] > 100:
            status = "2D slices"
            unclear.append(r)
        elif r["has_archives"]:
            status = "Unzip first"
            unclear.append(r)
        else:
            status = "Check"
            unclear.append(r)

        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["name"][:40], r["nifti_non_mask"], r["nifti_mask_count"],
                len(r["dicom_dirs"]), r["img2d_count"], r["max_depth"],
                modalities, status,
            )
        )

    # --- Totals ---
    lines.append("\n## Summary\n")
    total_nifti = sum(r["nifti_non_mask"] for r in ready)
    lines.append("- Ready: **{}** datasets, **{}** NIfTI images".format(
        len(ready), total_nifti))
    lines.append("- Need DICOM conversion: **{}** datasets".format(len(need_convert)))
    lines.append("- Need manual check: **{}** datasets".format(len(unclear)))

    # --- Per-dataset details ---
    lines.append("\n## Per-dataset Details\n")
    for r in results:
        lines.append("### {}\n".format(r["name"]))
        lines.append("- Path: `{}`".format(r["path"]))
        lines.append("- Files: {}, Dirs: {}, Max depth: {}".format(
            r["total_files"], r["total_dirs"], r["max_depth"]))

        ext_items = sorted(r["ext_counts"].items(), key=lambda x: -x[1])[:8]
        ext_str = ", ".join("`{}`: {}".format(e, c) for e, c in ext_items)
        lines.append("- Types: {}".format(ext_str))

        if r["nifti_non_mask"] > 0:
            lines.append("- NIfTI: {} images + {} masks".format(
                r["nifti_non_mask"], r["nifti_mask_count"]))
        if r["dicom_dirs"]:
            lines.append("- DICOM directories: {}".format(len(r["dicom_dirs"])))
        if r["sample_files"]:
            lines.append("- Samples:")
            for sf in r["sample_files"][:3]:
                lines.append("  - `{}`".format(sf))
        lines.append("")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print("Report saved: {}".format(output_path))
    else:
        print(report)

    return report


def main():
    parser = argparse.ArgumentParser(description="Survey MRI dataset directories")
    parser.add_argument("--data_root", default = "/home2/Data", help="Root dir containing dataset folders")
    parser.add_argument("--output", "-o", default=None, help="Output report path")
    parser.add_argument("--max-datasets", type=int, default=None)
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        print("ERROR: directory not found: {}".format(args.data_root))
        sys.exit(1)

    entries = sorted(os.listdir(args.data_root))
    dataset_dirs = [
        os.path.join(args.data_root, e)
        for e in entries
        if os.path.isdir(os.path.join(args.data_root, e))
    ]
    if args.max_datasets:
        dataset_dirs = dataset_dirs[:args.max_datasets]

    print("Found {} dataset directories, scanning...\n".format(len(dataset_dirs)))

    results = []
    for i, ddir in enumerate(dataset_dirs):
        name = os.path.basename(ddir)
        print("  [{}/{}] {}...".format(i + 1, len(dataset_dirs), name),
              end="", flush=True)
        info = survey_one_dataset(ddir)
        results.append(info)
        print("  ({} files, {} NIfTI)".format(
            info["total_files"], info["nifti_non_mask"]))

    format_report(results, args.output)


if __name__ == "__main__":
    main()

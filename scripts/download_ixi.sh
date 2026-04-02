#!/bin/bash
# Download IXI T1 dataset into data/IXI-T1/
#
# Usage:
#   bash scripts/download_ixi.sh
#   bash scripts/download_ixi.sh /custom/path

set -e

TARGET_DIR="${1:-data/IXI-T1}"
mkdir -p "$TARGET_DIR"

URL="https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar"

echo "Downloading IXI T1 dataset..."
echo "  URL:    $URL"
echo "  Target: $TARGET_DIR"
echo ""

if command -v wget &> /dev/null; then
    wget -c "$URL" -O "${TARGET_DIR}/IXI-T1.tar"
elif command -v curl &> /dev/null; then
    curl -L -C - "$URL" -o "${TARGET_DIR}/IXI-T1.tar"
else
    echo "Error: wget or curl required" && exit 1
fi

echo "Extracting..."
tar -xf "${TARGET_DIR}/IXI-T1.tar" -C "$TARGET_DIR"
rm "${TARGET_DIR}/IXI-T1.tar"

N=$(ls "$TARGET_DIR"/*.nii.gz 2>/dev/null | wc -l)
echo ""
echo "Done. $N T1 volumes in $TARGET_DIR"

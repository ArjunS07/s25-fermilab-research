set -euo pipefail

OUTPUTS_DIR="outputs"
ARCHIVE_DIR="archived_outputs"

mkdir -p "$ARCHIVE_DIR"

moved=0
skipped=0

for dir in "$OUTPUTS_DIR"/*/; do
    [[ -d "$dir" ]] || continue
    if [[ -f "${dir}train/models/final_model.pth" ]] || [[ -f "${dir}train/models/latest_checkpoint.pth" ]]; then
        skipped=$((skipped + 1))
    else
        name=$(basename "$dir")
        echo "Archiving: $name"
        mv "$dir" "$ARCHIVE_DIR/$name"
        moved=$((moved + 1))
    fi
done

echo "Done. Archived: $moved, kept: $skipped."

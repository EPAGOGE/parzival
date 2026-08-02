#!/bin/zsh
# archive_run.sh TAG [--with-ckpt]
# Bundle everything belonging to a run tag into runs/vault/TAG.tar.gz with a
# SHA256 sidecar. This is step 1 of the never-lose-a-run discipline; deletion
# happens ONLY through free_space.sh, which verifies this archive first.
set -euo pipefail
R=/Users/epagogellc/parzival/runs
V=$R/vault
mkdir -p "$V"
TAG=${1:?usage: archive_run.sh TAG [--with-ckpt]}
WITH_CKPT=${2:-}

items=()
[ -d "$R/snap_$TAG" ] && items+=("snap_$TAG")
[ -f "$R/stream_$TAG.jsonl" ] && items+=("stream_$TAG.jsonl")
[ -f "$R/control_$TAG.json" ] && items+=("control_$TAG.json")
for f in "$R"/*"$TAG"*.json; do
  [ -f "$f" ] && items+=("$(basename "$f")")
done
[ "$WITH_CKPT" = "--with-ckpt" ] && [ -d "$R/ckpt_$TAG" ] && items+=("ckpt_$TAG")

if [ ${#items[@]} -eq 0 ]; then
  echo "REFUSED: nothing found for tag '$TAG' under $R" >&2
  exit 2
fi

A="$V/$TAG.tar.gz"
tar czf "$A" -C "$R" "${items[@]}"
( cd "$V" && shasum -a 256 "$TAG.tar.gz" > "$TAG.tar.gz.sha256" )
echo "archived: $A ($(du -h "$A" | cut -f1)) containing: ${items[*]}"
if [ ! -f "$A.uploaded" ]; then
  echo "NOT YET OFF-MACHINE. free_space.sh will refuse deletion until"
  echo "$A.uploaded exists (set by the Drive upload step) or you pass --local-ok."
fi

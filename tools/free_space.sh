#!/bin/zsh
# free_space.sh TAG [--local-ok]
# THE GATE. The only sanctioned way to delete run bulk (snap_/ckpt_ dirs).
# Refuses unless runs/vault/TAG.tar.gz exists, its SHA256 verifies, and the
# archive is marked uploaded to Drive (or you explicitly accept local-only
# with --local-ok). Streams, control files, and result JSONs are NEVER
# deleted by this script - they are small and irreplaceable.
#
# Born from tension #67: the W-seed snapshot fields were deleted raw to free
# space, and the decisive generic two-scale measurement died with them.
set -euo pipefail
R=/Users/epagogellc/parzival/runs
V=$R/vault
TAG=${1:?usage: free_space.sh TAG [--local-ok]}
LOCAL_OK=${2:-}
A="$V/$TAG.tar.gz"

[ -f "$A" ] || { echo "REFUSED: no archive at $A. Run archive_run.sh $TAG first." >&2; exit 2; }
( cd "$V" && shasum -a 256 -c "$TAG.tar.gz.sha256" >/dev/null ) \
  || { echo "REFUSED: checksum mismatch on $A - archive is corrupt, do not delete sources." >&2; exit 2; }
if [ ! -f "$A.uploaded" ] && [ "$LOCAL_OK" != "--local-ok" ]; then
  echo "REFUSED: $A verified locally but not marked uploaded to Drive." >&2
  echo "Upload it (see VAULT.md), touch $A.uploaded, or rerun with --local-ok" >&2
  echo "to accept a local-only archive (still one disk failure from loss)." >&2
  exit 2
fi

freed=0
for d in "$R/snap_$TAG" "$R/ckpt_$TAG"; do
  if [ -d "$d" ]; then
    sz=$(du -sh "$d" | cut -f1)
    rm -rf "$d"
    echo "deleted $d ($sz)"
    freed=1
  fi
done
[ $freed -eq 0 ] && echo "nothing bulk to delete for '$TAG' (streams/JSONs are kept by design)"
echo "vault archive retained: $A"

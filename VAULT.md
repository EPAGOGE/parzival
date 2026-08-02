# VAULT — the never-lose-a-run discipline (2026-08-02)

Born from tension #67: the generic W-seed snapshot fields were deleted raw to
free disk, and the decisive two-scale measurement died with them. Never again.

## The rule

**Nothing under `runs/` gets deleted except through `tools/free_space.sh`.**
No exceptions, no "it's just a checkpoint." The gate refuses unless the run is
archived, checksum-verified, and off-machine (or you explicitly accept
`--local-ok` and its one-disk-failure risk).

## The pieces

```
tools/archive_run.sh TAG [--with-ckpt]   bundle snap_TAG + stream + JSONs ->
                                         runs/vault/TAG.tar.gz + .sha256
tools/free_space.sh TAG [--local-ok]     THE GATE: verify archive + uploaded
                                         marker, then delete snap_/ckpt_ bulk
                                         (streams/JSONs are NEVER deleted)
runs/vault/                              local archive staging
runs/vault/UPLOAD_LOG.txt                what went to Drive, with IDs + SHA256
Drive folder: PARZIVAL_VAULT             id 1S_M7Vpd_kuxqyQFwDVorRC0WaF7thqYr
                                         (justin@epagoge.io My Drive)
```

The `.uploaded` marker contract: `runs/vault/TAG.tar.gz.uploaded` exists only
after the archive is verified present off-machine. The uploader (whichever
pipe) touches it; free_space.sh trusts it.

## The two pipes

**Small pipe (LIVE now):** the claude.ai Google Drive connector. Good for
streams, ledgers, docs, .out artifacts — anything up to a few hundred KB.
Uploads go through an agent as base64. Already used for the 08-02 crown
jewels (all 14 stream JSONLs, EJA ledger, the graded MDs).

**Bulk pipe (NEEDS ONE-TIME SETUP — pick one):**
1. *Google Drive for desktop* (recommended: zero maintenance). Install it,
   sign in, then `runs/vault/` archives get copied to the synced folder and
   the OS does the rest. free_space markers set after sync confirms.
2. *rclone* (`brew install rclone`, then `rclone config` — one browser OAuth).
   Then: `rclone copy runs/vault gdrive:PARZIVAL_VAULT --progress` and touch
   the markers. Scriptable, cron-able.

Until one exists, big archives sit in `runs/vault/` verified-local-only and
the gate keeps refusing full deletion — which is correct behavior, not a bug.

## First bulk-sync priorities (when the pipe opens)

1. `runs/vault/W3R.tar.gz` (161M, already staged) + W63R/W77R equivalents
2. `runs/aws/` (8.4G — the salvaged AWS-box results, irreplaceable)
3. `runs/snap_mpi1024` (2.3G), `snap_loc1024` (913M), `snap_G5_11` (433M)
4. `runs/vault/smalltext_2026-08-02.tar.gz` (427K superset bundle)
5. `eja_state/journal.jsonl` (416K, the replayable audit trail)

## Wire-in for future runs

Run wrappers (e.g. rerun_wseeds.sh style) should end with
`tools/archive_run.sh <TAG>` so archiving is part of the run, not a chore
remembered later. Snapshot-cadence decisions should no longer be made from
disk fear (ledger #57) — archive and free instead.

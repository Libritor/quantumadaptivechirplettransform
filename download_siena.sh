#!/bin/bash
# Siena Scalp EEG Database v1.0.0 (PhysioNet, open access; 20.3 GB, 14 patients,
# 47 seizures, ~128 h). Uses PhysioNet's official S3 mirror -- physionet.org
# throttles bulk downloads to ~240 KB/s, the mirror gives 20-45 MB/s.
# Resumable: re-running fetches only what is missing or incomplete.
set -u
BASE=https://physionet-open.s3.amazonaws.com
PREFIX=siena-scalp-eeg/1.0.0/
DEST=~/datasets/siena
mkdir -p "$DEST" && cd "$DEST"

# enumerate keys via the public S3 list API (paginated)
: > .keys
TOKEN=""
while : ; do
  URL="$BASE/?list-type=2&prefix=$PREFIX&max-keys=1000"
  [ -n "$TOKEN" ] && URL="$URL&continuation-token=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$TOKEN")"
  XML=$(curl -s "$URL")
  echo "$XML" | grep -o '<Key>[^<]*</Key>' | sed 's/<\/*Key>//g' >> .keys
  TOKEN=$(echo "$XML" | grep -o '<NextContinuationToken>[^<]*</NextContinuationToken>' | sed 's/<\/*NextContinuationToken>//g')
  [ -z "$TOKEN" ] && break
done
N=$(wc -l < .keys)
echo "[siena] $N objects listed"
# download 8 at a time, preserving the directory layout below the prefix
sed "s#^$PREFIX##" .keys | grep -v '^$' | xargs -P 8 -I{} sh -c \
  'mkdir -p "$(dirname "{}")"; wget -q -c -O "{}" "'"$BASE/$PREFIX"'{}" || echo "FAILED {}"'
echo "[siena] done: $(du -sh . | cut -f1), $(find . -name '*.edf' | wc -l) EDF files"

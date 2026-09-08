#!/usr/bin/env bash
# Pull a Google Drive folder onto this machine with rclone. Nothing here is credential-specific:
# rclone keeps its token in ~/.config/rclone/rclone.conf, OUTSIDE the repo, and this script never
# reads or writes it.
#
#   utils/gdrive_pull.sh setup                        # one-time: install rclone + add a Drive remote
#   utils/gdrive_pull.sh ls   <folder-url-or-id>      # list what is there, with sizes, before committing
#   utils/gdrive_pull.sh pull <folder-url-or-id> <dest>   # copy it down (resumable; re-run to continue)
#
# WHY RCLONE AND NOT gdown/curl. These datasets are tens of GB across dozens of large files.
#   * `curl`/`wget` cannot authenticate to a private Drive folder at all.
#   * `gdown --folder` works only for "anyone with the link" shares, caps at 50 files per folder, and
#     fails on Google's large-file virus-scan interstitial -- which every multi-hundred-MB zip hits.
#   * rclone authenticates properly, RESUMES a partial transfer (re-run the same command), parallelises,
#     and verifies each file. On a link that dies halfway through 37 GB, resumability is the whole game.
#
# HEADLESS-SAFE. This box has no browser, so `setup` uses rclone's remote-authorize flow: it prints a
# command to run on your laptop, you authorise there, and paste one token back. No X11, no port
# forwarding, no browser here.
set -euo pipefail

RCLONE_BIN="${RCLONE_BIN:-$HOME/.local/bin/rclone}"
REMOTE="${RCLONE_REMOTE:-gdrive}"

die() { echo "error: $*" >&2; exit 1; }

# A Drive URL or a bare id -> the id. Accepts /folders/<id>, ?id=<id>, /d/<id>, or the id itself.
folder_id() {
  local s="$1"
  case "$s" in
    *drive.google.com*)
      s="${s%%\?*}"                       # drop the ?usp=... query
      s="${s##*/folders/}"; s="${s##*/d/}"; s="${s##*id=}"
      s="${s%%/*}" ;;
  esac
  [ -n "$s" ] || die "could not parse a folder id out of '$1'"
  printf '%s' "$s"
}

install_rclone() {
  if [ -x "$RCLONE_BIN" ]; then echo "rclone already at $RCLONE_BIN ($("$RCLONE_BIN" version | head -1))"; return; fi
  echo "installing rclone to $RCLONE_BIN (static binary, no root needed)..."
  local arch tmp
  case "$(uname -m)" in
    x86_64|amd64) arch=amd64 ;;
    aarch64|arm64) arch=arm64 ;;
    *) die "unsupported arch $(uname -m); grab a build from https://rclone.org/downloads/" ;;
  esac
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' RETURN
  curl -fsSL "https://downloads.rclone.org/rclone-current-linux-${arch}.zip" -o "$tmp/r.zip"
  ( cd "$tmp" && unzip -q r.zip )
  mkdir -p "$(dirname "$RCLONE_BIN")"
  install -m 0755 "$tmp"/rclone-*/rclone "$RCLONE_BIN"
  echo "installed: $("$RCLONE_BIN" version | head -1)"
}

cmd_setup() {
  install_rclone
  if "$RCLONE_BIN" listremotes 2>/dev/null | grep -qx "${REMOTE}:"; then
    echo "remote '${REMOTE}:' already configured -- nothing to do."
    echo "(to redo it: $RCLONE_BIN config delete $REMOTE)"
    return
  fi
  cat <<TXT

  This machine has no browser, so authorise on YOUR machine and paste the token back.

  1. On your laptop (needs rclone there too -- 'brew install rclone' / 'apt install rclone'), run:

         rclone authorize "drive" --drive-scope=drive.readonly

     A browser opens; sign in as the account that owns the folder. When it finishes it prints a
     token blob starting with  {"access_token":...}  -- copy the WHOLE line, braces included.

  2. Paste it below. It is stored in ~/.config/rclone/rclone.conf on this machine, never in the repo.
     --drive-scope=drive.readonly above means this token CANNOT modify or delete your Drive.

TXT
  printf 'token: '
  IFS= read -r token
  [ -n "$token" ] || die "no token given"
  "$RCLONE_BIN" config create "$REMOTE" drive scope drive.readonly token "$token" config_is_local false
  echo
  echo "remote '${REMOTE}:' created. Verifying..."
  "$RCLONE_BIN" about "${REMOTE}:" || die "remote created but not usable -- check the token"
  echo "OK. Now:  $0 ls <folder-url>"
}

need_remote() {
  [ -x "$RCLONE_BIN" ] || die "rclone not installed -- run: $0 setup"
  "$RCLONE_BIN" listremotes 2>/dev/null | grep -qx "${REMOTE}:" || die "remote '${REMOTE}:' not configured -- run: $0 setup"
}

cmd_ls() {
  need_remote
  local id; id="$(folder_id "${1:?usage: $0 ls <folder-url-or-id>}")"
  echo "folder id: $id"
  echo
  "$RCLONE_BIN" lsd  --drive-root-folder-id "$id" "${REMOTE}:" 2>/dev/null | sed 's/^/  DIR  /' || true
  "$RCLONE_BIN" lsl  --drive-root-folder-id "$id" "${REMOTE}:" | sort -k4 | head -40
  echo
  "$RCLONE_BIN" size --drive-root-folder-id "$id" "${REMOTE}:"
}

cmd_pull() {
  need_remote
  local id dest
  id="$(folder_id "${1:?usage: $0 pull <folder-url-or-id> <dest-dir>}")"
  dest="${2:?usage: $0 pull <folder-url-or-id> <dest-dir>}"
  mkdir -p "$dest"
  echo "pulling folder $id -> $dest"
  echo "(resumable: re-run this exact command to continue after an interruption)"
  # --drive-acknowledge-abuse: required for Google's large-file virus-scan interstitial, which is what
  #   makes gdown fail on multi-hundred-MB archives.
  # --transfers/--checkers: parallel enough to saturate a fast link without tripping Drive rate limits.
  # --fast-list: one listing pass for a big tree instead of a request per directory.
  "$RCLONE_BIN" copy "${REMOTE}:" "$dest" \
    --drive-root-folder-id "$id" \
    --drive-acknowledge-abuse \
    --transfers 8 --checkers 16 --fast-list \
    --retries 10 --low-level-retries 20 \
    --progress --stats 10s --stats-one-line
  echo
  echo "done. local size:"
  du -sh "$dest"
}

case "${1:-}" in
  setup) shift; cmd_setup "$@" ;;
  ls)    shift; cmd_ls "$@" ;;
  pull)  shift; cmd_pull "$@" ;;
  *) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac

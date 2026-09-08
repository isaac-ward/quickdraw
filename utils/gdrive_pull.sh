#!/usr/bin/env bash
# ==============================================================================================
# Pull a (large, private) Google Drive folder onto this machine with rclone.
#
# Nothing here is credential-specific: rclone keeps its token in ~/.config/rclone/rclone.conf,
# OUTSIDE the repo, and this script never reads or writes it.
#
# ----------------------------------------------------------------------------------------------
# STEP BY STEP -- downloading a big Drive folder, start to finish
# ----------------------------------------------------------------------------------------------
#
# 0. WHERE THE DATA GOES. Put it in `scratch/` at the repo root: it is gitignored, it sits beside
#    logs/ rather than inside it (a dataset is not a log), and it is mounted into the container at
#    /app/scratch so the processors can read it without a second copy.
#
#      mkdir -p scratch
#
#    NOTE: a NEW mount only takes effect when the container is recreated (`docker compose up -d`),
#    which KILLS ANY RUNNING TRAINING. The download itself runs on the host and needs no restart;
#    only the processing step does. Check `docker compose exec app ls /app/scratch` -- if that
#    errors, the container predates the mount and needs recreating when you can afford it.
#
# 1. AUTHORISE, ON A MACHINE THAT HAS A BROWSER. This box has none, so use rclone's
#    remote-authorize flow. On your laptop (`brew install rclone` / `apt install rclone`):
#
#      rclone authorize "drive" --drive-scope=drive.readonly
#
#    A browser opens; sign in as the account that OWNS the folder (sharing it to yourself is not
#    enough if it lives in someone else's Drive -- see step 6). When it finishes it prints a token
#    blob starting `{"access_token":...}`. Copy the WHOLE line, braces included.
#
#    Why `drive.readonly`: the token this stores physically cannot modify or delete your Drive.
#    There is no scenario where a download script needs write access.
#
# 2. HAND THE TOKEN TO THIS MACHINE. Installs rclone to ~/.local/bin (static binary, no root),
#    creates the remote, and verifies it with `rclone about`:
#
#      ./utils/gdrive_pull.sh setup
#
# 3. LOOK BEFORE YOU LEAP. Always. This prints the directory names, the biggest files, and a TOTAL
#    SIZE -- which is how you find out it is 400 GB before you start rather than after:
#
#      ./utils/gdrive_pull.sh ls <folder-url-or-id>
#
# 4. PULL IT.
#
#      ./utils/gdrive_pull.sh pull <folder-url-or-id> scratch/<name>
#
#    Interactive, you get a live progress bar. It is RESUMABLE: if it dies at 80%, re-run the
#    identical command and it continues -- rclone compares sizes/checksums and skips what is done.
#
# 5. FOR ANYTHING THAT WILL OUTLAST YOUR SSH SESSION, detach it. Do this for anything over a few
#    GB; a dropped connection otherwise kills the transfer:
#
#      nohup ./utils/gdrive_pull.sh pull <folder-url-or-id> scratch/<name> \
#            > scratch/<name>.pull.log 2>&1 &
#      tail -f scratch/<name>.pull.log
#
#    Redirected output automatically switches from the terminal bar to timestamped one-line stats
#    plus a line per completed file, so the log stays readable instead of filling with escape codes.
#
# 6. WHEN IT GOES WRONG
#
#    "couldn't find directory" / empty listing
#        The folder is not in the authorised account's Drive. A folder SHARED with you is not in
#        your Drive tree: open it in the browser and "Add shortcut to Drive", or authorise as the
#        owning account in step 1.
#    "This file has been identified as malware or spam"
#        Google's interstitial on large files. `pull` already passes --drive-acknowledge-abuse; if
#        you hit it with a bare rclone command, add that flag.
#    Rate-limit / 403 userRateLimitExceeded
#        Lower the parallelism: RCLONE_ARGS is not read, so edit --transfers/--checkers below, or
#        add `--tpslimit 10`.
#    Transfer crawls at a few MB/s
#        Usually Drive throttling a single large file, not the link. More --transfers does not help
#        one file; it helps many. Check `ls` output -- one 200 GB tarball will simply be slow.
#    Wrong or expired token
#        ~/.local/bin/rclone config delete gdrive, then redo steps 1-2.
#
# ----------------------------------------------------------------------------------------------
# USAGE
#   utils/gdrive_pull.sh setup                        # one-time: install rclone + add a Drive remote
#   utils/gdrive_pull.sh ls   <folder-url-or-id>      # list what is there, with sizes, before committing
#   utils/gdrive_pull.sh pull <folder-url-or-id> <dest>   # copy it down (resumable; re-run to continue)
#
# PROGRESS: `pull` prints the remote's total size BEFORE starting, so the transfer has a
# denominator, then a live bar on a terminal / timestamped one-liners when redirected to a log. A
# long silent transfer is indistinguishable from a hung one, so it never runs silent.
#
# WHY RCLONE AND NOT gdown/curl. These datasets are tens of GB across dozens of large files.
#   * `curl`/`wget` cannot authenticate to a private Drive folder at all.
#   * `gdown --folder` works only for "anyone with the link" shares, caps at 50 files per folder,
#     and fails on Google's large-file virus-scan interstitial -- which every multi-hundred-MB zip
#     hits. It also cannot resume.
#   * rclone authenticates properly, RESUMES a partial transfer, parallelises, and verifies each
#     file. On a 37 GB pull over a link that dies halfway, resumability is the whole game.
#
# HEADLESS-SAFE. No X11, no port forwarding, no browser needed on this machine (step 1).
# ==============================================================================================
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
  echo "  downloading rclone-current-linux-${arch}.zip ..."
  curl -fL --progress-bar "https://downloads.rclone.org/rclone-current-linux-${arch}.zip" -o "$tmp/r.zip"
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

  # SAY HOW BIG IT IS BEFORE STARTING. A silent multi-hour transfer with no denominator is
  # indistinguishable from a hung one, which is the single most annoying way for this to fail.
  echo "measuring the remote folder first (so the progress below has a denominator)..."
  "$RCLONE_BIN" size --drive-root-folder-id "$id" "${REMOTE}:" || die "cannot read folder $id"
  echo
  echo "pulling $id -> $dest    (started $(date '+%H:%M:%S'))"
  echo "RESUMABLE: if this dies, re-run the identical command and it continues where it stopped."
  echo

  # PROGRESS STYLE depends on where output is going, because rclone's --progress redraws the terminal
  # with escape codes -- lovely live, unreadable in a nohup log. So: live bar on a TTY, timestamped
  # one-liners plus a named line per completed file when redirected to a file.
  local prog=(--progress --stats 2s)
  if [ ! -t 1 ]; then
    prog=(--stats 15s --stats-one-line-date -v)
  fi
  # --drive-acknowledge-abuse: required for Google's large-file virus-scan interstitial, which is what
  #   makes gdown fail on multi-hundred-MB archives.
  # --transfers/--checkers: parallel enough to saturate a fast link without tripping Drive rate limits.
  # --fast-list: one listing pass for a big tree instead of a request per directory.
  "$RCLONE_BIN" copy "${REMOTE}:" "$dest" \
    --drive-root-folder-id "$id" \
    --drive-acknowledge-abuse \
    --transfers 8 --checkers 16 --fast-list \
    --retries 10 --low-level-retries 20 \
    "${prog[@]}"
  echo
  echo "done $(date '+%H:%M:%S'). local size:"
  du -sh "$dest"
  echo "file count: $(find "$dest" -type f | wc -l)"
}

case "${1:-}" in
  setup) shift; cmd_setup "$@" ;;
  ls)    shift; cmd_ls "$@" ;;
  pull)  shift; cmd_pull "$@" ;;
  *) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac

#!/bin/bash
# Idempotent + reentrant S2-audit matrix driver.
#  - arm guards: finished arms ([ -f npz ]) are skipped
#  - mid-arm resume: _s2audit.py restores from s2audit_TAG.partial.npz
#  - lock: /tmp (VM-local) with pid staleness — a SIGKILLed driver's lock
#    is reclaimed; a dead VM clears it by construction
#  - arms continue on failure (scientifically independent)
# ALWAYS launch via setsid (survives wsl.exe client death — the proven
# local failure mode):
#   setsid bash _s2matrix.sh >> _s2matrix.run.log 2>&1 </dev/null &
LOCK=/tmp/s2matrix.lock
driver_alive() {
    # /tmp is NOT tmpfs in this distro (locks survive VM restarts) and low
    # pids recycle in a fresh VM — kill -0 alone is not liveness. Require
    # the pid's cmdline to actually be a driver.
    local p
    p=$(cat "$LOCK/pid" 2>/dev/null) || return 1
    [ -n "$p" ] || return 1
    grep -qa "_s2matrix" "/proc/$p/cmdline" 2>/dev/null
}
if ! mkdir "$LOCK" 2>/dev/null; then
    if driver_alive; then
        echo "ALREADY_RUNNING (driver pid $(cat "$LOCK/pid"))"; exit 0
    fi
    echo "stale lock -- reclaiming"
    rm -rf "$LOCK"
    mkdir "$LOCK" 2>/dev/null || { echo LOCK_RACE; exit 0; }
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK" 2>/dev/null' EXIT
cd /mnt/c/Users/tyler/Downloads/indrajala-core || exit 1
export PYTHONUNBUFFERED=1 PYTHONPATH=. NT=6
# Engine cross-check is BANKED (uHa agreement at 1k/5.6k/25.7k twice today
# + 33k/43k/68k in prior runs); its build+solve was the peak-RAM spike that
# coincided with both VM deaths, and 7 min of per-resurrect-lap overhead.
export ENGINE=0
P=./.venv/bin/python
FE4="_qsci_benchmarks/fe2s2/Active-space-model-for-Iron-Sulfur-Clusters/Fe2S2_and_Fe4S4/Fe4S4/fe4s4"
date -u '+== matrix pass start %F %T UTC =='

echo "=== BS SEED (partA gate) ==="
[ -f s2audit_fe2s2bs.npz ] || TAG=fe2s2bs SEEDMODE=bs SYMM=1 DET_CAP=150000 \
  CKPTS=1000,5000,25000,60000,100000,150000 $P _s2audit.py \
  || echo "ARM_FAILED fe2s2bs rc=$?"

echo "=== AUFBAU DEEP ==="
[ -f s2audit_fe2s2deep.npz ] || SKIPA=1 TAG=fe2s2deep SEEDMODE=aufbau SYMM=1 \
  DET_CAP=150000 CKPTS=25000,60000,100000,150000 $P _s2audit.py \
  || echo "ARM_FAILED fe2s2deep rc=$?"

echo "=== FE4S4 ==="
[ -f s2audit_fe4s4.npz ] || SKIPA=1 FCID="$FE4" TAG=fe4s4 SEEDMODE=aufbau \
  SYMM=1 DET_CAP=40000 CKPTS=1000,5000,15000,40000 $P _s2audit.py \
  || echo "ARM_FAILED fe4s4 rc=$?"

echo MATRIX_PASS_DONE

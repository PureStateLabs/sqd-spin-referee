cd /mnt/c/Users/tyler/Downloads/indrajala-core
for NN in 300 500 1000 2000; do
  if [ ! -f flagsolve_n${NN}.npz ]; then
    echo "=== rung N=$NN start $(date) ===" >> _flagsolve_ladder.log
    N=$NN NT=8 PYTHONPATH=. ./.venv/bin/python _flagsolve.py >> _flagsolve_ladder.log 2>&1 \
      || { echo "RUNG $NN FAILED $(date)" >> _flagsolve_ladder.log; exit 1; }
  fi
done
touch _flagladder_DONE
echo "LADDER COMPLETE $(date)" >> _flagsolve_ladder.log

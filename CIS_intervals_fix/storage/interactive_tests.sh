#!/bin/bash
# Run inside a Slurm allocation (srun). Node-local TMPDIR only; nothing written to /rds.
. /etc/profile.d/modules.sh; module purge; module load rhel7/default-ccl
source /home/jbb50/PhD_work/SEIR_model/.venv/bin/activate
FIX=/home/jbb50/PhD_work/SEIR_model_fix/CIS_intervals_fix
OLD=/rds/user/jbb50/hpc-work/SEIR_model/CIS_intervals_sd0.1
T=${TMPDIR:-/tmp}/csvtest_$SLURM_JOB_ID; mkdir -p $T
echo "== compression test"
python -u $FIX/storage/test_compress.py $OLD/posterior_London_2021-01-24_0.npy.npz
echo "== posterior_as_csv grid test (East_England 2020-09-13 = job 5)"
cd $FIX && CSV_IN_DIR=$OLD CSV_OUT_DIR=$T python -u posterior_as_csv.py 5 | tail -n 6
python -u $FIX/storage/check_csv_grid.py $OLD $T East_England 2020-09-13
rm -rf $T

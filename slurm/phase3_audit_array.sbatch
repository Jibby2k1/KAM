#!/usr/bin/env bash
#SBATCH --job-name=kam_p3_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
: "${HPG_REPO:?HPG_REPO is required}"
: "${HPG_BLUE_ROOT:?HPG_BLUE_ROOT is required}"
: "${HPG_ENV:?HPG_ENV is required}"
source "$HPG_ENV/bin/activate"
cd "$HPG_REPO"
mkdir -p "$HPG_BLUE_ROOT/KAM/results/phase3"
python -m kam.phase3.audit --root "$HPG_REPO" --output-root "$HPG_REPO/results/phase3"

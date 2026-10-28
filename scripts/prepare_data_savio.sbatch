#!/usr/bin/env bash
#SBATCH --job-name=prepare-data
#SBATCH --account=fc_radar
#SBATCH --partition=savio2_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=scripts/prepare-data-%j.out

set -euo pipefail

source "${CPO3_REPO_ROOT:-$PWD}/scripts/paths_savio.sh"

export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt

cd "$CPO3_REPO_ROOT"

bash "$CPO3_REPO_ROOT/scripts/prepare_data.sh"

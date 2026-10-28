# Source this file before local data preparation, runs, or reports.
export CPO3_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CPO3_DATA_ROOT="${CPO3_DATA_ROOT:-$CPO3_REPO_ROOT/.local/data}"
export CPO3_RESULTS_ROOT="${CPO3_RESULTS_ROOT:-$CPO3_REPO_ROOT/.local/experiments}"

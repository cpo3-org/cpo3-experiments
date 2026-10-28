"""Reproduce the closed-loop MPC warm-start figure from the paper.

The script simulates a random linear plant under model predictive control.
At every step the controller solves the QP. Each QP is solved three ways with cpo3:

    cold     the solver's default initialization
    warm     warm started from the previous step's primal-dual solution
    shifted  warm started from that solution shifted forward by one stage

The loop is driven by the cold solve, so all three solve the same QPs. The
script prints the iteration counts and saves `figure_warm_mpc.pdf` and
`figure_warm_mpc.png` next to itself.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

import cpo3

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

SEED = 10
NUM_STATES = 8
NUM_INPUTS = 3
SPECTRAL_RADIUS = 0.98  # of the state matrix F
NOISE_STD = 0.02  # process noise added to the state at every step
HORIZON = 40  # N
STEPS = 60  # closed-loop steps
INITIAL_STATE_NORM = 3.2  # ||x_0||_inf
SETPOINT_NORM = 4.0  # ||r||_inf
INPUT_WEIGHT = 0.1
U_MAX = 0.1  # |u_k| <= U_MAX
X_MAX = 5.0  # |x_k| <= X_MAX

OUTPUT = Path(__file__).resolve().parent / "figure_warm_mpc"


# ---------------------------------------------------------------------------
# Problem data
# ---------------------------------------------------------------------------


def random_plant(rng):
    """Random state matrix F scaled to SPECTRAL_RADIUS, and input matrix G."""
    F = rng.standard_normal((NUM_STATES, NUM_STATES))
    F *= SPECTRAL_RADIUS / max(abs(np.linalg.eigvals(F)))
    G = rng.standard_normal((NUM_STATES, NUM_INPUTS))
    return F, G


def random_vector(rng, norm):
    """Random vector scaled to the given infinity norm."""
    v = rng.standard_normal(NUM_STATES)
    v *= norm / np.max(np.abs(v))
    return v


def steady_state_target(F, G, r):
    """Steady state (x_s, u_s) with the smallest stage cost.

    Solves  minimize ||x - r||^2 + 0.1 ||u||^2
            subject to x = F x + G u, |u| <= U_MAX, |x| <= X_MAX.

    The inputs cannot hold the setpoint r itself, so x_s is the closest
    state they can hold.
    """
    n, m = NUM_STATES, NUM_INPUTS
    P = sparse.diags(np.concatenate([np.full(n, 2.0), np.full(m, 2.0 * INPUT_WEIGHT)]))
    c = np.concatenate([-2.0 * r, np.zeros(m)])
    I_n, I_m = sparse.eye(n), sparse.eye(m)
    zeros_mn, zeros_nm = sparse.csc_matrix((m, n)), sparse.csc_matrix((n, m))
    A = sparse.vstack(
        [
            sparse.hstack([I_n - sparse.csc_matrix(F), -sparse.csc_matrix(G)]),  # x - F x - G u = 0
            sparse.hstack([zeros_mn, I_m]),  # u <= U_MAX
            sparse.hstack([zeros_mn, -I_m]),  # -u <= U_MAX
            sparse.hstack([I_n, zeros_nm]),  # x <= X_MAX
            sparse.hstack([-I_n, zeros_nm]),  # -x <= X_MAX
        ],
        format="csc",
    )
    b = np.concatenate([np.zeros(n), np.full(2 * m, U_MAX), np.full(2 * n, X_MAX)])
    cones = [cpo3.ZeroCone(n), cpo3.NonnegativeCone(2 * (m + n))]
    solution = cpo3.solve(cpo3.Problem(P=P, c=c, A=A, b=b, cones=cones))
    return solution.x[:n], solution.x[n:]


class MpcQp:
    """The MPC QP in cpo3's form: minimize 1/2 z'Pz + c'z s.t. Az + s = b.

    Variables:  z = (x_1, ..., x_N, u_0, ..., u_{N-1}).

    Rows of A, in order:
        dynamics       x_{k+1} - F x_k - G u_k = 0     (N * n rows)
        terminal       x_N = x_s                       (n rows)
        input bounds   u_k <= U_MAX, -u_k <= U_MAX     (2 * N * m rows)
        state bounds   x_k <= X_MAX, -x_k <= X_MAX     (2 * N * n rows)

    Only b changes between MPC steps, through F x_0 in the first
    dynamics row.
    """

    def __init__(self, F, G, r, x_s):
        n, m, N = NUM_STATES, NUM_INPUTS, HORIZON
        self.F = F
        self.n_states = N * n  # length of (x_1, ..., x_N)
        self.n_equality = N * n + n  # dynamics and terminal rows

        # Objective: sum_k ||x_k - r||^2 + 0.1 ||u_k||^2, constants dropped.
        self.P = sparse.diags(
            np.concatenate([np.full(N * n, 2.0), np.full(N * m, 2.0 * INPUT_WEIGHT)]),
            format="csc",
        )
        self.c = np.concatenate([np.tile(-2.0 * r, N), np.zeros(N * m)])

        # Dynamics: x_1 - G u_0 = F x_0 and x_{k+1} - F x_k - G u_k = 0.
        dynamics = sparse.hstack(
            [
                sparse.eye(N * n) - sparse.kron(sparse.eye(N, k=-1), F),
                -sparse.kron(sparse.eye(N), G),
            ]
        )
        terminal = sparse.hstack(
            [sparse.csc_matrix((n, (N - 1) * n)), sparse.eye(n), sparse.csc_matrix((n, N * m))]
        )
        inputs = sparse.hstack([sparse.csc_matrix((N * m, N * n)), sparse.eye(N * m)])
        states = sparse.hstack([sparse.eye(N * n), sparse.csc_matrix((N * n, N * m))])
        self.A = sparse.vstack(
            [dynamics, terminal, inputs, -inputs, states, -states], format="csc"
        )
        self.b = np.concatenate(
            [
                np.zeros(N * n),  # dynamics; the first n entries become F x_0
                x_s,  # terminal
                np.full(2 * N * m, U_MAX),
                np.full(2 * N * n, X_MAX),
            ]
        )
        self.cones = [
            cpo3.ZeroCone(self.n_equality),
            cpo3.NonnegativeCone(2 * N * (m + n)),
        ]

    def problem(self, x0):
        b = self.b.copy()
        b[:NUM_STATES] = self.F @ x0
        return cpo3.Problem(P=self.P, c=self.c, A=self.A, b=b, cones=self.cones)

    def first_input(self, z):
        return z[self.n_states : self.n_states + NUM_INPUTS]

    def count_active_inequalities(self, s):
        """Inequality rows whose slack is zero at the solution."""
        return int(np.sum(s[self.n_equality :] < 1e-6))

    def shift(self, z, y, s):
        """Shift a primal-dual solution forward by one stage.

        Every stage-ordered block drops its first stage and repeats its
        last. The terminal rows x_N = x_s are not stage ordered and are
        left unchanged.
        """
        n, m, N = NUM_STATES, NUM_INPUTS, HORIZON
        variable_blocks = [(0, n), (N * n, m)]  # (offset, stage size)
        o = N * n + n  # first inequality row
        row_blocks = [(0, n), (o, m), (o + N * m, m), (o + 2 * N * m, n), (o + 2 * N * m + N * n, n)]
        return (
            shift_blocks(z, variable_blocks),
            shift_blocks(y, row_blocks),
            shift_blocks(s, row_blocks),
        )


def shift_blocks(v, blocks):
    out = v.copy()
    for offset, size in blocks:
        stages = v[offset : offset + size * HORIZON].reshape(HORIZON, size)
        shifted = np.vstack([stages[1:], stages[-1:]])
        out[offset : offset + size * HORIZON] = shifted.ravel()
    return out


# ---------------------------------------------------------------------------
# Closed loop
# ---------------------------------------------------------------------------


def run():
    rng = np.random.default_rng(SEED)
    F, G = random_plant(rng)
    x = random_vector(rng, INITIAL_STATE_NORM)
    r = random_vector(rng, SETPOINT_NORM)
    x_s, _ = steady_state_target(F, G, r)
    qp = MpcQp(F, G, r, x_s)

    cold_settings = cpo3.Settings()
    # The warm start is always used, bypassing cpo3's acceptance screen.
    warm_settings = cpo3.Settings(force_warm_start=True)

    iterations = {"cold": [], "warm": [], "shifted": []}
    active_rows, state_change = [], []
    previous = {"warm": None, "shifted": None}
    x_previous = None

    for step in range(STEPS):
        problem = qp.problem(x)

        cold = cpo3.solve(problem, settings=cold_settings)
        assert cold.status == cpo3.SolverStatus.SOLVED, f"step {step}: {cold.status}"
        iterations["cold"].append(cold.info.iters)
        active_rows.append(qp.count_active_inequalities(cold.s))

        for variant in ("warm", "shifted"):
            if previous[variant] is None:
                # Nothing to warm start from at the first step.
                solution = cold
            else:
                start = previous[variant]
                if variant == "shifted":
                    start = qp.shift(*start)
                solution = cpo3.solve(problem, settings=warm_settings, warm_start=start)
                assert solution.status == cpo3.SolverStatus.SOLVED, f"step {step}: {variant}"
            iterations[variant].append(solution.info.iters)
            previous[variant] = (solution.x, solution.y, solution.s)

        state_change.append(np.nan if x_previous is None else np.max(np.abs(x - x_previous)))

        # Apply the cold solve's first input and add process noise.
        noise = NOISE_STD * rng.standard_normal(NUM_STATES)
        x_previous = x
        x = F @ x + G @ qp.first_input(cold.x) + noise

    return iterations, active_rows, state_change, qp


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def report(iterations, qp):
    print(
        f"MPC QP: {qp.A.shape[1]} variables, {qp.n_equality} equality rows, "
        f"{qp.A.shape[0] - qp.n_equality} inequality rows; {STEPS} closed-loop steps\n"
    )
    print(f"{'':<10}{'total':>8}{'total, steps 1-' + str(STEPS - 1):>20}{'median per step':>18}")
    for variant, counts in iterations.items():
        print(
            f"{variant:<10}{sum(counts):>8}{sum(counts[1:]):>20}{np.median(counts):>18g}"
        )
    print("\nStep 0 has nothing to warm start from, so all three variants solve it cold.")


def plot(iterations, active_rows, state_change):
    steps = np.arange(STEPS)
    styles = {
        "cold": ("Cold start", "#2a78d6", "o", "-"),
        "warm": ("Warm start", "#eb6834", "s", "--"),
        "shifted": ("Shifted warm start", "#1baf7a", "^", "-."),
    }
    fig, (ax_iters, ax_active, ax_change) = plt.subplots(
        3, 1, figsize=(6.4, 5.8), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
    )

    for variant, (label, color, marker, linestyle) in styles.items():
        ax_iters.plot(
            steps, iterations[variant], color=color, marker=marker, markersize=4,
            linewidth=1.5, linestyle=linestyle, label=label,
        )
    ax_iters.set_ylabel("Iterations")
    ax_iters.set_ylim(bottom=0)
    ax_iters.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax_iters.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))

    ax_active.step(steps, active_rows, where="mid", color="0.3", linewidth=1.5)
    ax_active.set_ylabel("Active ineq. rows")
    ax_active.set_ylim(0, 100)

    ax_change.semilogy(steps[1:], state_change[1:], color="0.3", linewidth=1.5)
    ax_change.set_ylabel(r"$\|x_t - x_{t-1}\|_\infty$")
    ax_change.set_xlabel("MPC step")

    for ax in (ax_iters, ax_active, ax_change):
        ax.grid(True, which="both", color="0.9", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT.with_suffix(".pdf"))
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"Saved {OUTPUT.with_suffix('.pdf')} and {OUTPUT.with_suffix('.png')}")


if __name__ == "__main__":
    iterations, active_rows, state_change, qp = run()
    report(iterations, qp)
    plot(iterations, active_rows, state_change)

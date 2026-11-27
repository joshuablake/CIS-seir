"""Unit checks for the omega_0 fix and the relaxed psi bound.

Run from CIS_intervals/:  python tests/test_initial_state.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
from scipy.linalg import expm
from covid19_seir.models import prevalence_model as pm
from covid19_seir.models.prevalence_model import P_TRANSITIONS, P_STATES, KAPPA_X2

dL, dI = 3.5, 4.0
sigma = 2. / dL


def old_p_init(e1, psir):
    p = np.empty(P_STATES)
    p[0] = (sigma + psir) / (psir + -P_TRANSITIONS[1, 1]) * e1
    P_OUT = -np.diag(P_TRANSITIONS)
    P_IN = P_TRANSITIONS @ np.ones(P_STATES) + P_OUT
    for i in range(1, P_STATES):
        p[i] = P_IN[i] / (psir + P_OUT[i]) * p[i - 1]
    return p


def growth_residual(p0, psir, e1=1.0, T=30.0):
    """Integrate dp/dt = A p + u e^{psir t} exactly (augmented matrix) and
    compare with p0 e^{psir T}. Returns max relative error over k."""
    A = np.zeros((P_STATES + 1, P_STATES + 1))
    A[:P_STATES, :P_STATES] = P_TRANSITIONS
    A[0, P_STATES] = (sigma + psir) * e1      # inflow to P0 = S->E1 flow
    A[P_STATES, P_STATES] = psir              # forcing grows at psir
    x = expm(A * T) @ np.append(p0, 1.0)
    return np.max(np.abs(x[:P_STATES] / (p0 * np.exp(psir * T)) - 1))


model = pm.SingleRegionStratifiedSEIRCompartmental(np.ones((2, 1, 1)), 10)
fails = 0
for psir in [-0.0222, -0.02, -0.01, -1e-4, 0.0, 0.03, 0.053, 0.08, 0.106, 0.2]:
    new = model._calculate_initial_prev_state(np.array([1.0]), sigma, psir)[:, 0]
    old = old_p_init(1.0, psir)
    r_new, r_old = growth_residual(new, psir), growth_residual(old, psir)
    ok = r_new < 1e-8 and (new > 0).all()
    fails += not ok
    print(f"psi={psir:+.4f}  new resid={r_new:.1e}  old resid={r_old:.2f}  "
          f"F=old/new={old[0]/new[0]:.2f}  new>0={bool((new > 0).all())}  {'OK' if ok else 'FAIL'}")

# Bound: accepted just above -KAPPA_X2, rejected at/below; full init state passes check_state
m6 = pm.SingleRegionStratifiedSEIRCompartmental(np.ones((2, 6, 6)) * 0.1, 10)
pi = np.full(6, 0.95)
for psir, expect in [(-KAPPA_X2 + 1e-4, True), (-0.01, True), (0.05, True),
                     (-KAPPA_X2, False), (-0.03, False)]:
    try:
        st, _ = m6._init_transmission(dL, dI, psir, 1e-4, pi, np.ones((6, 6)) * 0.1)
        got = bool(m6.check_state(st)) and (st >= 0).all()
    except ValueError:
        got = False
    fails += got != expect
    print(f"bound psi={psir:+.5f}: accepted={got} expected={expect}")

# R0 in (0,1) for negative psi
for psir in [-0.022, -0.01, -1e-3]:
    r = model._calc_r0(dL, dI, psir)
    fails += not (0 < r < 1)
    print(f"R0(psi={psir}) = {r:.4f}")
print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
sys.exit(int(fails > 0))

"""Reproduce a crashed MCMC task (same seed/start) and dump state at failure.
Usage: SEIR_ITERATIONS=60000 SEIR_OUT_DIR=<dir> python debug_task.py <task>"""
import sys, os
import numpy as np
task = sys.argv[1]
sys.argv = ["run_MCMC.py", task]
src = open("run_MCMC.py").read()
pre = src.split("results = mcmc.run()")[0]
g = {"__name__": "__main__"}
exec(compile(pre, "run_MCMC.py", "exec"), g)
mcmc, lp = g["mcmc"], g["log_posterior"]
stash = {}
orig_init = mcmc._loop_init
def _loop_init(*a, **k):
    b = orig_init(*a, **k); stash["block"] = b; return b
mcmc._loop_init = _loop_init
orig_do = None
history = []
def wrap(block):
    od = block.do_update
    def do_update(i, cur, clt):
        history.append((i, np.exp(block.scale), np.isfinite(block._cov_est).all(),
                        np.abs(block._cov_est).max()))
        if len(history) > 50: history.pop(0)
        return od(i, cur, clt)
    block.do_update = do_update
orig_safe = mcmc.safe_amgs_update
def safe(block, i):
    if i == 0: wrap(block)
    return orig_safe(block, i)
mcmc.safe_amgs_update = safe
try:
    mcmc.run()
    print("NO FAILURE in", g["ITERATIONS"], "iterations")
except np.linalg.LinAlgError as e:
    i = e.final_iteration
    b = stash["block"]
    names = lp.get_param_names()
    ch = mcmc.chain[:i + 1]; lt = mcmc.chain_log_target[:i + 1]
    print("failed after iteration", i)
    print("current finite:", np.isfinite(mcmc.current).all(), "log target:", mcmc.current_log_target)
    print("mean_est finite:", np.isfinite(b._mean_est).all(), "cov finite:", np.isfinite(b._cov_est).all(),
          "scale:", b.scale)
    print("last history (iter, exp(scale), cov finite, max|cov|):")
    for h in history[-8:]: print("  ", h)
    bad = [n for n, v in zip(names, mcmc.current) if not np.isfinite(v)]
    print("non-finite params in current:", bad)
    print("last 5 states (i0, mm, psir, theta, beta[0]) and log target:")
    idx = [names.index(n) for n in ["i0", "matrix_modifiers", "psir", "theta", "beta[0]"]]
    for k in range(max(0, i - 4), i + 1): print("  ", k, ch[k, idx], lt[k])
    print("max |value| per param at end:", dict(zip(names, np.round(np.abs(ch[-1]), 3))))
    np.savez(os.path.join(os.environ["SEIR_OUT_DIR"], f"debug_{task}.npz"), chain=ch[::10], lt=lt[::10],
             tail_chain=ch[-2000:], tail_lt=lt[-2000:], cov=b._cov_est, mean=b._mean_est, scale=b.scale,
             names=np.array(names))

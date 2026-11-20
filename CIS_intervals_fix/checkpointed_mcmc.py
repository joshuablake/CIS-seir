"""SingleBlockAMGS with small, infrequent, thinned checkpoints for monitoring.

The MCMC loop is identical to SingleBlockAMGS.run() (same updates, same RNG
calls in the same order), so a chain with the same seed and start is
unchanged. Checkpoints are for monitoring / interim estimates only, not restarts.
"""
import os
import numpy as np
from covid19_seir.mcmc.single_block_adapt_mcmc import SingleBlockAMGS


class CheckpointedAMGS(SingleBlockAMGS):
    def __init__(self, *args, checkpoint_path=None, checkpoint_every=500_000,
                 checkpoint_thin=5000, param_names=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._checkpoint_path = checkpoint_path
        self._checkpoint_every = checkpoint_every
        self._checkpoint_thin = checkpoint_thin
        self._param_names = param_names

    def _write_checkpoint(self, amgs_block, i):
        idx = np.arange(0, i + 1, self._checkpoint_thin)
        tmp = self._checkpoint_path + ".tmp.npz"
        np.savez(
            tmp,
            iteration=idx,
            samples=self.chain[idx],
            log_target=self.chain_log_target[idx],
            last_iteration=i,
            acceptance=amgs_block.acceptance,
            log_scale=amgs_block.scale,
            param_names=np.array(self._param_names if self._param_names is not None else []),
        )
        os.replace(tmp, self._checkpoint_path + ".npz")
        print(f"Checkpoint written at iteration {i}", flush=True)

    def run(self, return_log_target=False, record_diagnostics=False):
        if record_diagnostics:
            return_log_target = True
        self._return_log_target = return_log_target
        self._record_diagnostics = record_diagnostics

        amgs_block = self._loop_init()

        for i in range(self._iterations):
            self.safe_amgs_update(amgs_block, i)

            # Store the current
            self.chain[i, :] = self.current
            self.chain_log_target[i] = self.current_log_target
            self._print_progress(amgs_block, i)

            if self._checkpoint_path is not None and (
                    (i + 1) % self._checkpoint_every == 0 or i == self._iterations - 1):
                self._write_checkpoint(amgs_block, i)

        return self._get_return_value(amgs_block)

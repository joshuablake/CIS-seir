"""Test lossless np.savez_compressed on one posterior npz. Writes only to node-local tmp."""
import os, sys, time, tempfile
import numpy as np
f = sys.argv[1]
t = time.time()
with np.load(f) as z:
    a = z["arr_0"]
print(f"loaded {a.shape} {a.dtype} in {time.time()-t:.0f}s", flush=True)
tmp = os.environ.get("TMPDIR") or tempfile.gettempdir()
out = os.path.join(tmp, "test_compressed.npz")
t = time.time(); np.savez_compressed(out, arr_0=a); tw = time.time() - t
t = time.time()
with np.load(out) as z:
    same = np.array_equal(z["arr_0"], a, equal_nan=True) and z["arr_0"].dtype == a.dtype
tr = time.time() - t
s0, s1 = os.path.getsize(f), os.path.getsize(out)
print(f"original {s0/1e9:.3f} GB, compressed {s1/1e9:.3f} GB, ratio {s0/s1:.2f}x; "
      f"write {tw:.0f}s, read+verify {tr:.0f}s, identical={same}", flush=True)
os.remove(out)

"""Losslessly recompress posterior_*.npy.npz files in place with np.savez_compressed.

Usage:
  python compress_npz.py <dir> <n_tasks> <task_id> [--staging DIR]      # one pass, array task
  python compress_npz.py <dir> 1 0 --watch SECONDS --until-empty-hours H  # watcher for new files

Per file: skip if already deflate-compressed; skip if modified < 3 min ago (may still be
being written). Load arr_0, write compressed copy under a temp name in the same dir (or, if
the quota blocks that write, in --staging on another filesystem), reload it and check
np.array_equal + dtype + shape against the original, then os.replace onto the ORIGINAL name
(same key arr_0, so np.load(f)["arr_0"] and posterior_as_csv.py work unchanged).
Files are dealt round-robin by name (stable, so concurrent tasks never overlap), then each task
does its share smallest first.
"""
import os, sys, glob, time, shutil, zipfile, filecmp, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("n_tasks", type=int); ap.add_argument("task", type=int)
ap.add_argument("--staging", default="/home/jbb50/thin_staging")
ap.add_argument("--watch", type=int, default=0)
ap.add_argument("--until-empty-hours", type=float, default=0)
a = ap.parse_args()


def is_compressed(f):
    with zipfile.ZipFile(f) as z:
        return all(i.compress_type == zipfile.ZIP_DEFLATED for i in z.infolist())


def verify(path, ref):
    with np.load(path, allow_pickle=False) as z:
        assert list(z.keys()) == ["arr_0"]
        b = z["arr_0"]
    return b.dtype == ref.dtype and b.shape == ref.shape and np.array_equal(b, ref)


def compress_one(f):
    s0 = os.path.getsize(f)
    with np.load(f, allow_pickle=False) as z:
        assert list(z.keys()) == ["arr_0"], (f, list(z.keys()))
        arr = z["arr_0"]
    tmp = f + ".tmp_compressed.npz"  # ends in .npz so numpy does not append another suffix
    staged = None
    try:
        np.savez_compressed(tmp, arr_0=arr)
    except OSError as e:  # quota: stage on another filesystem instead
        print(f"  write in place failed ({e.strerror}); staging", flush=True)
        if os.path.exists(tmp):
            os.remove(tmp)
        staged = os.path.join(a.staging, os.path.basename(f) + ".staged.npz")
        np.savez_compressed(staged, arr_0=arr)
    src = staged or tmp
    if not verify(src, arr):
        print(f"VERIFY FAILED {f}; original kept", flush=True)
        os.remove(src)
        return 0
    if staged:  # free the original first, then copy the verified file in
        os.remove(f)
        if not copy_in(staged, f):
            DEFERRED.append((staged, f))
            print(f"deferred {os.path.basename(f)}: original removed, verified copy at {staged}", flush=True)
            return s0
    else:
        os.replace(tmp, f)
    s1 = os.path.getsize(f)
    print(f"ok {os.path.basename(f)} {s0/1e9:.2f} -> {s1/1e9:.2f} GB ({s0/s1:.1f}x)", flush=True)
    return s0 - s1


DEFERRED = []


def copy_in(staged, f):
    """Copy a verified staged file to f (via temp name + byte compare). False if quota blocks it."""
    tmp = f + ".tmp_compressed.npz"
    try:
        shutil.copyfile(staged, tmp)
    except OSError as e:
        print(f"  copy-in failed ({e.strerror})", flush=True)
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    if not filecmp.cmp(staged, tmp, shallow=False):
        raise RuntimeError(f"copy mismatch for {f}; verified copy kept at {staged}")
    os.replace(tmp, f)
    os.remove(staged)
    print(f"ok (copied in) {os.path.basename(f)}", flush=True)
    return True


def retry_deferred(wait=60, tries=60):
    for _ in range(tries):
        DEFERRED[:] = [(st, f) for st, f in DEFERRED if not copy_in(st, f)]
        if not DEFERRED:
            return
        time.sleep(wait)
    for st, f in DEFERRED:
        print(f"STILL DEFERRED {f}: verified copy at {st}", flush=True)


def candidates():
    # Assign files to tasks from a list that does not depend on compression state
    # (sorted by name), so concurrent tasks never pick the same file.
    allf = sorted(f for f in glob.glob(os.path.join(a.dir, "posterior_*.npz"))
                  if not f.endswith(".tmp_compressed.npz"))
    out = []
    for f in allf[a.task::a.n_tasks]:
        if time.time() - os.path.getmtime(f) < 180:
            continue
        try:
            if not is_compressed(f):
                out.append(f)
        except zipfile.BadZipFile:  # still being written
            continue
    return sorted(out, key=os.path.getsize)  # smallest first


if not a.watch:
    files = candidates()
    print(f"task {a.task}: {len(files)} files", flush=True)
    freed = 0
    for f in files:
        freed += compress_one(f)
        if DEFERRED:
            retry_deferred(wait=0, tries=1)
    retry_deferred()
    print(f"done; freed {freed/1e9:.1f} GB", flush=True)
else:
    idle_since = time.time()
    while True:
        files = candidates()
        for f in files:
            compress_one(f)
        if DEFERRED:
            retry_deferred(wait=0, tries=1)
        if files:
            idle_since = time.time()
        elif a.until_empty_hours and time.time() - idle_since > a.until_empty_hours * 3600:
            print("idle limit reached; exiting", flush=True); break
        time.sleep(a.watch)

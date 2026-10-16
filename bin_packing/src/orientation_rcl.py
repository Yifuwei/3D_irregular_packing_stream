"""Second-level RCL over the position sampled for each feasible orientation."""
import numpy as np


def choose_orientation(candidates, tolerance, rng=None, observer=None):
    """Candidates start with minimized PES; retain their original payload."""
    candidates=[entry for entry in candidates if np.isfinite(entry[0])]
    if not candidates:
        return None
    if tolerance is None:
        return min(candidates,key=lambda entry:entry[0])
    if not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('Orientation tolerance must be finite and nonnegative')
    best=min(entry[0] for entry in candidates)
    threshold=best+tolerance*abs(best)
    epsilon=8*np.finfo(float).eps*max(1.,abs(threshold))
    rcl=[entry for entry in candidates if entry[0]<=threshold+epsilon]
    index=np.random.randint(len(rcl)) if rng is None else rng.integers(len(rcl))
    selected=rcl[index]
    if observer:
        observer(dict(level='orientation',best_pes=float(best),tolerance=float(tolerance),
                      candidate_count=len(candidates),active_rcl_size=len(rcl),
                      candidate_pes=[float(c[0]) for c in candidates],selected_pes=float(selected[0])))
    return selected

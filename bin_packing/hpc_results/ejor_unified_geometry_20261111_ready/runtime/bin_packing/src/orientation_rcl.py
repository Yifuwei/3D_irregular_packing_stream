"""One RCL over orientation-position pairs; payloads remain unchanged."""
import numpy as np


def choose_orientation(candidates, tolerance, rng=None, observer=None, top_k=None):
    """Candidates start with minimized PES; retain their original payload."""
    candidates=[entry for entry in candidates if np.isfinite(entry[0])]
    if not candidates:
        return None
    def orientation_metadata(entries):
        labels=set()
        for entry in entries:
            payload=entry[1]
            label=(payload.get('orientation') if isinstance(payload,dict) else
                   payload if isinstance(payload,str) else
                   payload[0] if isinstance(payload,(tuple,list)) and payload and isinstance(payload[0],str) else None)
            if label is not None:labels.add(label)
        return dict(rcl_orientation_count=len(labels),rcl_orientations=sorted(labels))
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k <= 0:
            raise ValueError('top_k must be a positive integer')
        rcl = sorted(candidates, key=lambda entry: entry[0])[:top_k]
        index = np.random.randint(len(rcl)) if rng is None else rng.integers(len(rcl))
        selected = rcl[index]
        if observer:
            observer(dict(level='pooled', top_k=int(top_k), candidate_count=len(candidates),
                          active_rcl_size=len(rcl), selected_pes=float(selected[0]),**orientation_metadata(rcl)))
        return selected
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
                      candidate_pes=[float(c[0]) for c in candidates],selected_pes=float(selected[0]),**orientation_metadata(rcl)))
    return selected

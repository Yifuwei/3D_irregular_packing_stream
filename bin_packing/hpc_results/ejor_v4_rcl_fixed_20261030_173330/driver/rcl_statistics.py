"""Validated RCL telemetry shared by the worker and regression tests."""
from collections import Counter

def record_rcl_event(histograms,event):
    size=event['active_rcl_size']
    count=event.get('rcl_orientation_count')
    if count is None:
        raise ValueError('RCL event lacks orientation statistics; update orientation_rcl.py before freezing')
    if not 1 <= count <= size:
        raise ValueError('Invalid distinct orientation count in RCL')
    histograms['pooled'][size]+=1
    histograms['orientation_count'][count]+=1

def summarize_rcl(histograms):
    hist=histograms['orientation_count']
    events=sum(hist.values())
    return dict(events=events,mean_distinct_orientations=(sum(int(k)*v for k,v in hist.items())/events if events else None),
                min_distinct_orientations=min(map(int,hist)) if events else None,
                max_distinct_orientations=max(map(int,hist)) if events else None)

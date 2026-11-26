"""Decision rules on retained, rounded (override, concentration, log probability) triples."""
from math import isfinite


def score(v4, chains):
    if type(v4) is not bool:
        raise ValueError("Missing or invalid v4 decision")
    longest = 0
    peak = 0.0
    for steps in chains:
        run = 0
        for delta, share, logp in steps:
            if not all(map(isfinite, (delta, share, logp))):
                raise ValueError("Non-finite trace")
            if logp >= -2:
                peak = max(peak, delta * share)
            run = run + 1 if delta >= 3 and share >= .4 and logp >= -2 else 0
            longest = max(longest, run)
    old = v4 or longest >= 3
    return dict(v4=v4, longest=longest, chain=longest >= 3, baseline=old,
                peak=peak, peak_alarm=peak >= 11, combined=old or peak >= 11)

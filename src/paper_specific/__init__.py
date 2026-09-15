"""Code that exists only to produce the ICRA 2027 paper: its tables, its figures, and the analyses that
are specific to the starling-2 OOD/memory campaigns.

Kept OUT of `quickdraw` proper because none of it generalises -- it hardcodes this paper's splits, this
dataset's anomaly windows, and this paper's table layouts. Anything here that turns out to be general
should move into `quickdraw.evaluation` instead; the reverse is a one-way ratchet into paper-shaped code.
"""

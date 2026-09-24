# Primary E04 joint-state counts

These 24 compressed CSV tables correspond to the three cities, two analysis
domains, and four periods in the accepted E04 run. Each row has an arbitrary
`baseline_state_id`, a lagged `bicycle_bin`, an integer `target_count`, and a
`frequency`. State IDs are independently randomly relabeled within each
specification. No crosswalk, grid identifier, coordinate, calendar date, or
event-level record is deposited. The totals and exact E04 entropy, CMI, and
DELB point estimates were checked against the frozen accepted-run tables;
`validation_summary.csv` gives the comparisons. Run
`python applsci-4501207/empirical/verify_e04_joint_counts.py --counts-dir derived_counts`
from the archive root to recalculate them without city source files.

These tables are sufficient for the **primary E04 point estimates** and their
plug-in/Miller--Madow inputs. They are not sufficient to reproduce block
bootstrap intervals, randomization tests, multiscale analyses, local analyses,
held-out model fits, or every figure in the paper. Those analyses require
frozen temporal/spatial panels and source data that are not redistributed.

The frequency tables and validation summary are CC BY 4.0; see
`LICENSE-DATA.txt`. Original code is MIT; see `LICENSE-CODE.txt`.

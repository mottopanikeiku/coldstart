# Next comparison

The most useful next experiment is an Acrobot comparison that changes only the environment execution backend. The present CPU PPO and Zoo PPO differ in objectives, normalization and update settings, so their timing difference cannot establish a physics-loop speedup.

Use the same pinned SB3 PPO, Zoo configuration, normalization wrapper and common evaluation protocol for both backends: the existing Gymnasium reference and a small SB3 vector-environment adapter around the checked NumPy dynamics. Check adapter reset, terminal-observation and truncation behavior against the reference before measuring. Keep every planned seed and report both qualification times and transition counts; do not replace the current results.

Estimated effort, not run: two to four hours to implement and review the adapter, then 30–60 minutes of exclusive local CPU measurement for 20 seeds per backend. Paid compute cost would remain $0. This estimate is not a measured benchmark. The result could show no useful improvement at Zoo's small environment count; that is still informative.

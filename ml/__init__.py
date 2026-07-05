"""ML scaffolding for CSI human sensing.

Contains the synthetic data generator, a dataset builder that runs the real
preprocessing + feature pipeline, and three classifiers:

    presence  - is anyone there / moving?      (binary)
    fall      - did a fall happen in a window?  (binary)
    counting  - how many people: 0, 1, or 2?    (3-class, 2 is the hard cap)

IMPORTANT: models here are trained on SYNTHETIC data by default. That only
validates that the code path works end to end. It says NOTHING about real
accuracy. See docs/limitations.md.
"""

SYNTHETIC_WARNING = (
    "SYNTHETIC VALIDATION - NOT REAL ACCURACY. "
    "These numbers only prove the pipeline runs; retrain on real captures."
)

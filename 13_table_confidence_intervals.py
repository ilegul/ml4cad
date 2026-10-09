"""Add patient-bootstrap intervals for thesis Tables 5.2 to 5.5.

Usage:
    python 13_table_confidence_intervals.py

Requires the unchanged full profile and existing frozen predictive artifacts.
Only results/table_confidence_intervals/ is written.
"""

import table_confidence_intervals


if __name__ == "__main__":
    table_confidence_intervals.run()

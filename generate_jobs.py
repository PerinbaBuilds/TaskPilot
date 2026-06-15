"""
Generates a synthetic jobs CSV for testing the scheduler.
Usage: python generate_jobs.py [num_jobs] [output_file]
"""
import pandas as pd
import random
import sys

NUM_JOBS    = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUTPUT_FILE = sys.argv[2]       if len(sys.argv) > 2 else "jobs.csv"

PRIORITIES = ["green", "balanced", "performance"]
LATENCIES  = ["low", "medium", "high"]

jobs = [
    {
        "cpu":      random.randint(10, 100),
        "memory":   random.randint(10, 100),
        "priority": random.choice(PRIORITIES),
        "latency":  random.choice(LATENCIES),
    }
    for _ in range(NUM_JOBS)
]

pd.DataFrame(jobs).to_csv(OUTPUT_FILE, index=False)
print(f"Generated {NUM_JOBS} jobs → {OUTPUT_FILE}")

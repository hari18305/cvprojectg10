"""Generate the synthetic object-proposal benchmark (train / test splits)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bing.datasets import ROOT, make_synthetic

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--test", type=int, default=300)
    a = ap.parse_args()
    make_synthetic(os.path.join(ROOT, "data", "synthetic", "train"), a.train, seed=1)
    make_synthetic(os.path.join(ROOT, "data", "synthetic", "test"), a.test, seed=2)
    print("done")

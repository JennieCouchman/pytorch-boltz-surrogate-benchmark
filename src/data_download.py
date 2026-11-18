#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:13:38 2026

@author: jennie
"""

import urllib.request
from pathlib import Path
import pandas as pd

# Ensure target directory exists
project_root = Path(__file__).resolve().parent.parent
raw_dir = project_root / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

# Public MoleculeNet BACE dataset URL
bace_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
bace_path = raw_dir / "bace.csv"

if not bace_path.exists():
    print(f"Downloading BACE dataset to {bace_path}...")
    urllib.request.urlretrieve(bace_url, bace_path)

# Load and inspect the columns you need
df = pd.read_csv(bace_path)[["mol", "pIC50", "Class"]].rename(columns={"mol": "smiles"})
print(f"Loaded {len(df)} molecules.")
print(df.head())

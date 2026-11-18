# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

# Silence verbose RDKit C++ warnings
RDLogger.DisableLog("rdApp.*")

# Custom Morgan fingerprint featurizer class - wrapper around RDKit methods for graceful error handling and consistency between sklearn and pytorch methods
class MorganFeaturizer:
    """Converts SMILES strings into fixed-length Morgan fingerprint NumPy arrays."""
    
    def __init__(self, radius: int = 2, n_bits: int = 2048):
        # radius of 2 makes this compatible with 
        self.radius = radius
        self.n_bits = n_bits
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.n_bits
        )

    def smiles_to_fp(self, smiles: str) -> np.ndarray | None:
        """Convert a single SMILES string to a float32 NumPy vector."""
        # Check that 'smiles' is a non-empty string
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return self.generator.GetFingerprintAsNumPy(mol).astype(np.float32)

    def transform(self, smiles_list: list[str]) -> Tuple[np.ndarray, list[int]]:
        """
        Convert a list of SMILES strings into a (N_valid, n_bits) NumPy matrix.
        Returns the feature matrix and the list of valid row indices.
        """
        fps = []
        valid_indices = []

        for idx, smi in enumerate(smiles_list):
            fp = self.smiles_to_fp(smi)
            if fp is not None:
                fps.append(fp)
                valid_indices.append(idx)

        if not fps:
            return np.empty((0, self.n_bits), dtype=np.float32), []

        return np.vstack(fps), valid_indices


def build_bace_features(radius: int = 2, n_bits: int = 2048) -> None:
    """Load raw BACE CSV, extract Morgan fingerprints, and save processed arrays."""
    project_root = Path(__file__).resolve().parent.parent
    raw_path = project_root / "data" / "raw" / "bace.csv"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. Run src/data_download.py first."
        )

    df = pd.read_csv(raw_path)[["mol", "pIC50", "Class"]].rename(columns={"mol": "smiles"})

    featurizer = MorganFeaturizer(radius=radius, n_bits=n_bits)
    X, valid_idx = featurizer.transform(df["smiles"].tolist())

    # Filter DataFrame to keep only valid molecules and align targets
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    y_affinity = df_valid["pIC50"].to_numpy(dtype=np.float32)
    y_class = df_valid["Class"].to_numpy(dtype=np.int64)

    # Save processed arrays and cleaned metadata
    np.save(processed_dir / "bace_X.npy", X)
    np.save(processed_dir / "bace_y_affinity.npy", y_affinity)
    np.save(processed_dir / "bace_y_class.npy", y_class)
    df_valid.to_csv(processed_dir / "bace_clean.csv", index=False)

    sparsity = 1.0 - (np.count_nonzero(X) / X.size)
    print(f"Processed {X.shape[0]} valid molecules (dropped {len(df) - X.shape[0]} invalid).")
    print(f"Feature matrix shape: {X.shape} | Dtype: {X.dtype} | Sparsity: {sparsity:.1%}")
    print(f"Saved arrays to {processed_dir}")


if __name__ == "__main__":
    build_bace_features()


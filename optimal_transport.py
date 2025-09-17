import pandas as pd
import numpy as np
import ot
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform


def barycentric_projection(coupling, target_dataset, x_onto_y=True):
    """
    Based on code from SCOTv1.
    Uses the coupling matrix and the target dataset to create the projection of the src dataset onto the other domain.
    """
    if type(coupling) is not pd.DataFrame:
        coupling = pd.DataFrame(coupling)
    if type(target_dataset) is not pd.DataFrame:
        target_dataset = pd.DataFrame(target_dataset)
    
    if x_onto_y:
        # Projecting the first domain onto the second domain
        if target_dataset.shape[0] != coupling.shape[1]:
            raise ValueError("target_dataset rows must match coupling columns. did you mean to set x_onto_y=False?")

        weights = coupling.sum(axis=1)
        src_aligned = (coupling @ target_dataset) / weights.values[:, None]
    else:
        # Projecting the second domain onto the first domain
        if target_dataset.shape[0] != coupling.shape[0]:
            raise ValueError("target_dataset rows must match coupling rows. did you mean to set x_onto_y=True?")

        weights = coupling.sum(axis=0)
        src_aligned = (coupling.T @ target_dataset) / weights.values[:, None]

    return src_aligned


def main():
    pass


if __name__ == "__main__":
    main()


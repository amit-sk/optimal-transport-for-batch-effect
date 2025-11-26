import pandas as pd
import numpy as np


PROJECT_SEED = 1


def get_otu_columns(data):
    return [c for c in data.columns if (type(c) is int) or (type(c) is str and c.isnumeric())]


def get_species_columns(data):
    """
    for datasets where species are represented with full taxonomy strings starting with 'd__Bacteria' (to be converted to otu-like integers).
    """
    return [c for c in data.columns if c.startswith('d__Bacteria')]

def round_dataframes(digits, *dataframes):
    return tuple(df.map(lambda x: round(x, digits) if type(x) == float else x) for df in dataframes)


def round_dataframe(digits, dataframe):
    return round_dataframes(digits, dataframe)[0]


def get_sample_ids_by_dataset(combined_data: pd.DataFrame) -> dict:
    """
    Based on code from Guy Shur's thesis (utils.py).
    """
    return {dataset:
                {
                    'CD':
                        combined_data[(combined_data['dataset'] == dataset) & (combined_data['phenotype'] == 'CD')].index.to_list(),
                    'control':
                        combined_data[(combined_data['dataset'] == dataset) & (combined_data['phenotype'] == 'control')].index.tolist()
                }
            for dataset in np.unique(combined_data['dataset'])}


def renormalize_data(data, otu_only=False):
    if otu_only:
        sums = data.sum(axis=1) 
        return data.div(sums, axis=0)
    
    copy = data.copy()
    otus = get_otu_columns(copy)
    sums = copy[otus].sum(axis=1) 
    copy[otus] = copy[otus].div(sums, axis=0)
    return copy


def create_noisy_data(data, proportion_of_std=0.1, seed=PROJECT_SEED):
    np.random.seed(seed)
    copy = data.copy()
    
    otus_data = copy[get_otu_columns(copy)]
    stds = otus_data.std()
    
    for sample_id, row in copy.iterrows():
        for idx, cell in row.items():
            if idx in ['sample_id', 'phenotype', 'age'] or cell == 0.0:
                continue

            std = stds[idx] * proportion_of_std
            noise = np.random.normal(0, std)
            new_val = max(0.0, cell + noise)  # no negative values
            copy.at[sample_id, idx] = new_val

    return copy


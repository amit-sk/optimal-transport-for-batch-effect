import os
import pandas as pd
import numpy as np

import data_utils


PATH_TO_RISK_DATA = os.path.join('.', 'raw_data','RISK.tsv')
PATH_TO_RISK_METADATA = os.path.join('.','raw_data', 'risk_metadata.txt')
PATH_TO_MUCOSALIBD_DATA = os.path.join('.','raw_data', 'MucosalIBD.tsv')
PATH_TO_MUCOSALIBD_METADATA = os.path.join('.','raw_data', 'mucosalibd_metadata.txt')


def obtain_relative_abundance_data(data, metadata):
    processed_data = pd.DataFrame()

    # iterate over samples
    for sample_id, _ in data.transpose().iterrows():
        if sample_id in ['# OTU','taxonomy']:
            continue

        sample_meta = metadata[metadata.sample_accession_16S == sample_id]
        phenotype = sample_meta.disease.iloc[0]
        if phenotype not in ['control', 'CD']:
            continue

        # process to relative abundance
        sample_data = data[['# OTU', sample_id]].copy()
        sample_data[sample_id] = sample_data[sample_id] / sample_data[sample_id].sum()

        age = sample_meta.age.iloc[0]
        new_row = {'sample_id': sample_id, 'phenotype': phenotype, 'age': round(age) if not pd.isna(age) else age}
        new_row.update({int(r['# OTU']):r[sample_id] for _, r in sample_data.iterrows()})
        processed_data = pd.concat([processed_data, pd.DataFrame([new_row])], ignore_index=True)

    return processed_data


def filter_uncommon_otus(data, should_appear_in=0.05, min_abundance=0.005):
    otus = data_utils.get_otu_columns(data)
    non_otu_columns = [c for c in data.columns if not type(c) is int]
    amount_greater_than_min = (data[otus] >= min_abundance).sum()
    otus_to_keep = amount_greater_than_min[amount_greater_than_min >= (should_appear_in * len(data))].index

    cols = non_otu_columns
    cols.extend(otus_to_keep)
    return data[cols]


def main():
    risk_data = pd.read_csv(PATH_TO_RISK_DATA, sep='\t')
    risk_meta = pd.read_csv(PATH_TO_RISK_METADATA, sep='\t')
    mucosalibd_data = pd.read_csv(PATH_TO_MUCOSALIBD_DATA, sep='\t')
    mucosalibd_meta = pd.read_csv(PATH_TO_MUCOSALIBD_METADATA, sep='\t')

    risk_processed_data = obtain_relative_abundance_data(risk_data, risk_meta)
    risk_processed_data = filter_uncommon_otus(risk_processed_data)
    risk_processed_data = data_utils.renormalize_data(risk_processed_data)
    risk_processed_data.set_index('sample_id', inplace=True)
    risk_processed_data.to_csv("risk_data.csv")

    mucosalibd_processed_data = obtain_relative_abundance_data(mucosalibd_data, mucosalibd_meta)
    mucosalibd_processed_data = filter_uncommon_otus(mucosalibd_processed_data)
    mucosalibd_processed_data = data_utils.renormalize_data(mucosalibd_processed_data)
    mucosalibd_processed_data.set_index('sample_id', inplace=True)
    mucosalibd_processed_data.to_csv("mucosalibd_data.csv")

    print("Done.")


if __name__ == "__main__":
    main()


import os
import pandas as pd
import numpy as np

import data_utils


PATH_TO_RISK_DATA = os.path.join('.', 'raw_data','RISK.tsv')
PATH_TO_RISK_METADATA = os.path.join('.','raw_data', 'risk_metadata.txt')
PATH_TO_MUCOSALIBD_DATA = os.path.join('.','raw_data', 'MucosalIBD.tsv')
PATH_TO_MUCOSALIBD_METADATA = os.path.join('.','raw_data', 'mucosalibd_metadata.txt')

PATH_TO_iHMP_DATA = os.path.join('.','raw_data', 'iHMP.tsv')
PATH_TO_iHMP_METADATA = os.path.join('.','raw_data', 'ihmp_metadata.tsv')
PATH_TO_FRANZOSA_DATA = os.path.join('.','raw_data', 'FRANZOSA.tsv')
PATH_TO_FRANZOSA_METADATA = os.path.join('.','raw_data', 'franzosa_metadata.tsv')


def obtain_relative_abundance_data(data, metadata):
    """
    obtain data function for RISK and MucosalIBD datasets. data is converted to relative abundance from count data.
    """
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


def obtain_data(data, metadata, age_label):
    """
    obtain data function for FRANZOSA and iHMP datasets. data is already in relative abundance.
    """
    processed_data = pd.DataFrame()

    # iterate over samples
    for sample_id, sample_data in data.iterrows():
        sample_meta = metadata[metadata.Sample == sample_id]
        subject_id = sample_meta.Subject.iloc[0]

        if 'subject_id' in processed_data and (processed_data['subject_id'] == subject_id).sum() > 0:
            continue  # already processed this subject

        phenotype = sample_meta['Study.Group'].iloc[0]
        if phenotype not in ['Control', 'CD', 'UC', 'nonIBD']:
            continue

        if phenotype == 'UC':
            phenotype = 'CD'  # group UC as CD

        if phenotype == 'nonIBD' or phenotype == 'Control':
            phenotype = 'control'  # group nonIBD and Control as control
        
        # get age
        age = sample_meta[age_label].iloc[0]
        if sample_meta['Age.Units'].iloc[0] != 'Years':
            raise ValueError(f"Age units not in years for sample {sample_id}.")

        new_row = {'sample_id': sample_id, 'subject_id': subject_id, 'phenotype': phenotype, 'age': round(age) if not pd.isna(age) else age}
        new_row.update({species:abundance for species, abundance in sample_data.items()})
        processed_data = pd.concat([processed_data, pd.DataFrame([new_row])], ignore_index=True)

    processed_data.drop(columns=['subject_id'], errors='ignore', inplace=True)
    return processed_data


def filter_uncommon_otus(data, should_appear_in=0.05, min_abundance=0.005):
    otus = data_utils.get_otu_columns(data)
    non_otu_columns = [c for c in data.columns if not type(c) is int]
    amount_greater_than_min = (data[otus] >= min_abundance).sum()
    otus_to_keep = amount_greater_than_min[amount_greater_than_min >= (should_appear_in * len(data))].index

    cols = non_otu_columns
    cols.extend(otus_to_keep)
    return data[cols]


def preprocess_species(data, species_translation):
    """
    preprocess species data by converting species names to shorter OTU-like integer names.
    if no such name is found in the actual species taxonomy, a random integer name is generated and stored in species_translation.
    this is so that later on we can compare the species across datasets if needed.
    """
    new_names = {}
    for species_taxonomy in data_utils.get_species_columns(data):
        species_id = species_taxonomy[species_taxonomy.rfind('sp')+2:]
        if species_id.isdigit():
            species_id = int(species_id)
        else:  # not able to obtain id from taxonomy
            if species_taxonomy in species_translation:
                species_id = species_translation[species_taxonomy]
            else:
                # generate new id
                species_id = np.random.randint(1000, 9999999)
                species_translation[species_taxonomy] = species_id

        if species_id in data.columns:
            raise ValueError(f"Species id {species_id} already exists in data columns, cannot rename {species_taxonomy} to this id.")

        new_names[species_taxonomy] = species_id

    data.rename(columns=new_names, inplace=True)


def main():
    np.random.seed(data_utils.PROJECT_SEED)

    # ===================================================

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

    # ===================================================

    ihmp_data = pd.read_csv(PATH_TO_iHMP_DATA, sep='\t')
    ihmp_meta = pd.read_csv(PATH_TO_iHMP_METADATA, sep='\t')
    franzosa_data = pd.read_csv(PATH_TO_FRANZOSA_DATA, sep='\t')
    franzosa_meta = pd.read_csv(PATH_TO_FRANZOSA_METADATA, sep='\t')

    ihmp_data.set_index('Sample', inplace=True)
    franzosa_data.set_index('Sample', inplace=True)

    species_translation = {}
    preprocess_species(ihmp_data, species_translation)
    ihmp_processed_data = obtain_data(ihmp_data, ihmp_meta, 'consent_age')
    ihmp_processed_data = filter_uncommon_otus(ihmp_processed_data)
    ihmp_processed_data = data_utils.renormalize_data(ihmp_processed_data)
    ihmp_processed_data.set_index('sample_id', inplace=True)
    ihmp_processed_data.to_csv("ihmp_data.csv")

    preprocess_species(franzosa_data, species_translation)
    franzosa_processed_data = obtain_data(franzosa_data, franzosa_meta, 'Age')
    franzosa_processed_data = filter_uncommon_otus(franzosa_processed_data)
    franzosa_processed_data = data_utils.renormalize_data(franzosa_processed_data)
    franzosa_processed_data.set_index('sample_id', inplace=True)
    franzosa_processed_data.to_csv("franzosa_data.csv")

    # TODO: save species_translation to file for future reference

    print("Done.")


if __name__ == "__main__":
    main()


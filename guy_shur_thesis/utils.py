# -*- coding: latin-1 -*-
import copy
import math
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import skbio
from matplotlib import pyplot as plt
# from qiime2 import Artifact, CategoricalMetadataColumn
# from qiime2.plugins.gneiss.methods import ilr_phylogenetic
from scipy.spatial.distance import euclidean
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

class Artifact:
    pass

np.seterr(all='raise')
# from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
#
# ZeroInflatedNegativeBinomialP.
from sklearn.decomposition import PCA


def fit_shift_params(X, a, b):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    return (a * X) + b


def learn_shift_params(target_controls_values, basis_mean, basis_std):
    if type(target_controls_values) is not np.ndarray:
        target_controls_values = np.array(target_controls_values)
    if np.isnan(np.sum(target_controls_values)):
        print(target_controls_values)
        exit(1)
    if np.sum(target_controls_values) == 0:
        return None
    if np.isnan(np.sum(target_controls_values)):
        print(target_controls_values)
        raise ValueError('array contains nan')
    x_std = np.std(target_controls_values)
    if x_std == 0:
        a = 1
    else:
        a = basis_std / x_std
        a = (a - 1) + 1
    tmp = target_controls_values * a
    b = basis_mean - np.mean(tmp)
    test = (a * target_controls_values) + b
    try:
        assert equal(np.mean(test), basis_mean, 8)
    except AssertionError as e:
        print(np.mean(test), basis_mean)
        raise e
    assert equal(np.std(test), basis_std, 8)
    return a, b


def get_sample_ids(control_case_columns_data: pd.DataFrame) -> dict:
    return {cohort: {'case': control_case_columns_data[
        (control_case_columns_data['batch'] == cohort) & (control_case_columns_data['set'] == 'case')].index.to_list(),
                     'control': control_case_columns_data[(control_case_columns_data['batch'] == cohort) & (
                             control_case_columns_data['set'] == 'control')].index.tolist()}
            for cohort in np.unique(control_case_columns_data['batch'])}


def annotate_AG_md_by_age(md_path, out_path):
    df = pd.read_csv(md_path, sep='\t', index_col=0)

    def classify_age(x):
        try:
            float(x)
        except ValueError as e:
            return 'nan'
        if float(x) < 20:
            return '0to20'
        elif 20 <= float(x) < 40:
            return '20to40'
        elif 40 <= float(x) < 60:
            return '40to60'
        elif 60 <= float(x) < 80:
            return '60to80'
        elif 80 <= float(x):
            return '80+'

    df['source_file'] = [classify_age(x) for x in df.age_years]
    mask = [True if df.loc[:, 'source_file'][i] in ['20to40', '60to80'] and df.loc[:, 'bmi_cat'][i] in ['Normal',
                                                                                                        'Obese'] else False
            for i in range(len(df.index))]
    df = df[mask]
    df.to_csv(out_path, sep='\t')


def manifest_single_ended(manifest_file_path, metadata_path, path_to_fastq_files, file_path_column, sample_id_column,
                          drop_duplicates=False):
    """
    Creates a manifest file for single ended fastq files
    """
    if not path_to_fastq_files.endswith('/'):
        path_to_fastq_files += '/'
    df = pd.read_csv(metadata_path, sep='\t')
    drop_duplicates = bool(drop_duplicates)
    file_path_to_sample_id = df.loc[:, [file_path_column, sample_id_column]]
    seen_ids = set()
    duplicate_ids_to_ignore = set()
    for _, data in file_path_to_sample_id.iterrows():
        if data[1] in seen_ids:
            if drop_duplicates:
                duplicate_ids_to_ignore.add(data[1])
            else:
                raise AssertionError('repeated id:', data[1])
        seen_ids.add(data[1])

    with open(manifest_file_path, 'w+') as f:
        f.write("sample-id\tabsolute-filepath")
        for _, data in file_path_to_sample_id.iterrows():
            if data[1] not in duplicate_ids_to_ignore:
                f.write('\n' + str(data[1]) + '\t' + '$PWD/' + path_to_fastq_files + str(data[0]))
    if drop_duplicates:
        print('WARNING: ignored ' + str(len(duplicate_ids_to_ignore)) + ' duplicate samples', duplicate_ids_to_ignore)


def shift_mean_and_sd(df: pd.DataFrame, base: str, control_categoreis: pd.DataFrame) -> pd.DataFrame:
    print(df)
    output_df = df.copy()
    ids = get_sample_ids(control_case_columns_data=control_categoreis)
    control_base_samples = ids[base]['control']
    batches = control_categoreis['batch'].unique().tolist()
    batches.remove(base)
    for batch in batches:
        control_target_samples = ids[batch]['control']
        all_target_samples = control_target_samples + ids[batch]['case']
        for feature in output_df:
            some_abundance = output_df.loc[control_base_samples, feature][0]
            if not all([equal(x, some_abundance) for x in output_df.loc[control_base_samples, feature]]):
                base_mean_controls = np.mean(output_df.loc[control_base_samples, feature])
            else:
                print(feature, 'not present in any control sample in', batch)
                continue
            some_abundance = output_df.loc[control_target_samples, feature][0]
            if not all([equal(x, some_abundance) for x in output_df.loc[control_target_samples, feature]]):
                target_mean_controls = np.mean(output_df.loc[control_target_samples, feature])
            else:
                print(feature, 'not present in any case sample in', batch)
                continue
            base_sd_controls = np.std(output_df.loc[control_base_samples, feature])
            target_sd_controls = np.std(output_df.loc[control_target_samples, feature])
            target_samples = output_df.loc[all_target_samples, feature].values
            target_samples = (target_samples - target_mean_controls) * (
                        base_sd_controls / target_sd_controls) + base_mean_controls
            output_df.loc[all_target_samples, feature] = target_samples
    return output_df


def classify_samples_all_methods(artifact_paths: dict, out_path: str):
    for cohort_name, artifact_path in artifact_paths.items():
        assert artifact_path.endswith('.qza')
        classify_samples(out_path, artifact_path[artifact_path.rindex('/') + 1:], cohort_name)


def classify_samples(artifact_path: str, artifact_name: str,
                     cohort_name: str, batches_name='batches.tsv', batch_parameter_name='batch'):
    artifact_path = artifact_path + '/' if not artifact_path.endswith('/') else artifact_path
    classification_qza_path = artifact_path + 'qza_for_classification.qza'
    art = Artifact.load(artifact_path + artifact_name)
    art = art.view(pd.DataFrame)
    art = Artifact.import_data('FeatureTable[Frequency]', art)
    art.save(classification_qza_path)
    cmd = "qiime sample-classifier classify-samples --i-table {}" \
          " --m-metadata-file {} " \
          "--m-metadata-column {} --output-dir {}{}_classification_output".format(
        classification_qza_path, artifact_path + batches_name, batch_parameter_name, artifact_path, cohort_name)
    subprocess.call(cmd.split(' '))
    cmd = "qiime tools export --input-path {}{}_classification_output/accuracy_results.qzv " \
          "--output-path {}{}_exported_classification_results".format(artifact_path, cohort_name, artifact_path,
                                                                      artifact_name)
    subprocess.call(cmd.split(' '))
    cmd = "rm -rf {}{}_classification_output".format(artifact_path, cohort_name)
    subprocess.call(cmd.split(' '))


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.div(df.sum(axis=1), axis=0)


def test_mnn_alogirthm():
    plt.figure(figsize=(10, 10))
    k = 1
    data = np.random.random(size=(150, 200))
    pca = PCA(2)
    pca_data = pca.fit_transform(data)
    pca_data = pd.DataFrame(pca_data)
    all_samples = pca_data.index.tolist()
    distance_matrix = pd.DataFrame(index=all_samples, columns=all_samples, dtype=float)
    for a in all_samples:
        for b in all_samples:
            distance_matrix.at[a, b] = euclidean(pca_data.loc[a, :], pca_data.loc[b, :])
    cohort_1 = pca_data.index.tolist()[:75]
    cohort_2 = pca_data.index.tolist()[75:]
    set_1_nearest_neighbors = pd.DataFrame(index=cohort_1, columns=cohort_2, dtype=bool)
    for set_1_sample in cohort_1:
        distance_to_other_set_samples = distance_matrix.loc[set_1_sample, cohort_2]
        cutoff = np.partition(distance_to_other_set_samples, k - 1)[k - 1]
        set_1_nearest_neighbors.loc[set_1_sample, cohort_2] = [x <= cutoff for x in distance_to_other_set_samples]
    set_2_nearest_neighbors = pd.DataFrame(index=cohort_2, columns=cohort_1, dtype=bool)
    for set_2_sample in cohort_2:
        distance_to_other_set_samples = distance_matrix.loc[set_2_sample, cohort_1]
        cutoff = np.partition(distance_to_other_set_samples, k - 1)[k - 1]
        set_2_nearest_neighbors.loc[set_2_sample, cohort_1] = [x <= cutoff for x in distance_to_other_set_samples]
    are_mutual_nearest_neighbors = pd.DataFrame(index=cohort_1, columns=cohort_2, data=np.array(
        [
            [
                set_1_nearest_neighbors.at[set1_sample, set2_sample] and
                set_2_nearest_neighbors.at[set2_sample, set1_sample]
                for set2_sample in cohort_2
            ] for set1_sample in cohort_1]
    ), dtype=bool)
    features = pca_data.columns
    f1, f2 = features[0], features[1]
    for s1 in cohort_1:
        for s2 in cohort_2:
            if are_mutual_nearest_neighbors.at[s1, s2]:
                plt.plot([pca_data.loc[s1, f1], pca_data.loc[s2, f1]], [pca_data.loc[s1, f2], pca_data.loc[s2, f2]],
                         c='black', alpha=0.25,
                         zorder=5)
    plt.scatter(x=pca_data.loc[cohort_1, f1], y=pca_data.loc[cohort_1, f2], c='red')
    plt.scatter(x=pca_data.loc[cohort_2, f1], y=pca_data.loc[cohort_2, f2], c='blue')
    plt.savefig('test_mnn.png')


from skbio.stats.composition import clr

c = np.array([3, 2, 1])

# Create model
# with pm.Model() as model:
#     # Parameters of the Multinomial are from a Dirichlet
#     parameters = pm.Dirichlet('parameters', a=alphas, shape=3)
#     # Observed data is from a Multinomial distribution
#     observed_data = pm.Multinomial(
#         'observed_data', n=6, p=parameters, shape=3, observed=c)  
import random


def produce_copy(input_path, prefix, output_path):
    art = pseudocount_absolute_frequency(Artifact.load(input_path))
    df = art.view(pd.DataFrame)
    # clr_df = pd.DataFrame(index=df.index, columns=df.columns, data=clr(df))

    average_mean = np.average([np.mean(df[otu]) for otu in df.applymap(lambda x: np.abs(x))])
    average_sd = np.average([np.std(df[otu]) for otu in df.applymap(lambda x: np.abs(x))])
    print(average_mean, average_sd)
    error_mean_kernel = np.random.normal(average_mean, average_mean / 10)
    error_sd_kernel = np.random.normal(average_sd / 20, average_sd / 200)
    error_mean_kernel = np.random.normal(20, 5)
    error_sd_kernel = np.random.normal(10, 5)
    counter = 0
    num_otus = len(df.columns)
    for otu in df:
        counter += 1
        print(counter, '/', num_otus)
        otu_error_mean = np.random.normal(error_mean_kernel, error_mean_kernel / 10)
        otu_error_sd = np.random.normal(error_sd_kernel, error_sd_kernel / 10)
        added_otu_error_mean = np.random.normal(error_mean_kernel / 2, error_mean_kernel / 20)
        added_otu_error_sd = np.random.normal(error_sd_kernel / 2, error_sd_kernel / 20)
        # otu_error_mean = otu_error_mean * random.choice([-1,1])
        # added_otu_error_mean = otu_error_mean * random.choice([-1,1])
        print(otu_error_mean, otu_error_sd, added_otu_error_mean, added_otu_error_sd)
        for sample in df.index:
            df.at[sample, otu] = df.at[sample, otu] * np.random.normal(otu_error_mean, otu_error_sd) + np.random.normal(
                added_otu_error_mean, added_otu_error_sd)
    df = df.div(df.sum(axis=1), axis=0)
    Artifact.import_data('FeatureTable[RelativeFrequency]',
                         pd.DataFrame(index=[x + prefix for x in df.index.tolist()], columns=df.columns,
                                      data=df.values)).save(output_path)


def mockrobiome_1(num_samples, num_features):
    num_samples = int(num_samples)
    num_features = int(num_features)
    half_samples = int(num_samples / 2)
    quarter_samples = int(num_samples / 4)
    assert num_samples > num_features and num_samples > 10 and num_features > 10 and num_samples % 4 == 0
    samples = ['sample{}'.format(i) for i in range(num_samples)]
    basis_samples = ['basis{}'.format(i) for i in range(half_samples)]
    all_samples = samples + basis_samples
    features = [i for i in range(num_features)]
    table = pd.DataFrame(index=features, columns=all_samples, data=np.zeros((num_features, len(all_samples))))
    feature_indices = [i for i in range(num_features)]
    abundant_features = np.random.choice(feature_indices, math.floor(num_features / 10))
    feature_indices_c = [i for i in feature_indices if i not in abundant_features]
    less_abundant_features = np.random.choice(feature_indices_c, math.floor(num_features / 10))
    absent_features = [i for i in feature_indices_c if i not in less_abundant_features]
    affected_by_disease = np.random.choice(feature_indices, math.floor(num_features / 5))
    up_in_first_batch = np.random.choice(feature_indices, math.floor(num_features / 10))
    feature_indices_c = [i for i in feature_indices if i not in up_in_first_batch]
    down_in_first_batch = np.random.choice(feature_indices_c, math.floor(num_features / 10))
    up_in_second_batch = np.random.choice(feature_indices, math.floor(num_features / 10))
    feature_indices_c = [i for i in feature_indices if i not in up_in_second_batch]
    down_in_second_batch = np.random.choice(feature_indices_c, math.floor(num_features / 10))
    set1 = samples[:half_samples]
    set2 = samples[half_samples:]
    control_1 = set1[:quarter_samples]
    case_1 = set1[quarter_samples:]
    control_2 = set2[:quarter_samples]
    case_2 = set2[quarter_samples:]
    for feature in abundant_features:
        mean = np.random.normal(10, 2)
        sd = mean / 5
        for sample in all_samples:
            table.at[feature, sample] = np.random.normal(mean, sd) + np.random.normal(0, 1)
    for feature in less_abundant_features:
        mean = np.random.normal(1, 0.2)
        sd = mean / 5
        for sample in all_samples:
            table.at[feature, sample] = np.random.normal(mean, sd) + np.random.normal(0, 0.1)
    for sample in all_samples:
        false_presence = np.random.choice(absent_features, math.floor(len(absent_features) / 10))
        for feature in false_presence:
            table.at[feature, sample] = np.random.normal(0.2, 0.04) + np.random.normal(0, 0.02)
    table = table.applymap(lambda x: x if x >= 0 else 0)
    for feature in affected_by_disease:
        effect = np.random.uniform(0.8, 1.2)
        table.loc[feature, case_1 + case_2] *= effect
    for feature in down_in_first_batch:
        effect = np.random.uniform(0.8, 1)
        table.loc[feature, control_1 + case_1] *= effect
    for feature in up_in_first_batch:
        effect = np.random.uniform(1, 1.2)
        table.loc[feature, control_1 + case_1] *= effect
    for feature in down_in_second_batch:
        effect = np.random.uniform(0.8, 1)
        table.loc[feature, control_2 + case_2] *= effect
    for feature in up_in_second_batch:
        effect = np.random.uniform(1, 1.2)
        table.loc[feature, control_2 + case_2] *= effect

    table.index = table.index.map(str)
    table = table.T
    table = table.div(table.sum(axis=1), axis=0)
    for feature in table:
        print(np.sum(table[feature]))

    metadata = pd.DataFrame(index=all_samples, columns=['DiseaseState', 'source_file'])
    metadata.loc[basis_samples, 'source_file'] = 'basis'
    metadata.loc[set1, 'source_file'] = 'set1'
    metadata.loc[set2, 'source_file'] = 'set2'
    metadata.loc[basis_samples, 'DiseaseState'] = 'H'
    metadata.loc[control_1, 'DiseaseState'] = 'H'
    metadata.loc[control_2, 'DiseaseState'] = 'H'
    metadata.loc[case_1, 'DiseaseState'] = 'D'
    metadata.loc[case_2, 'DiseaseState'] = 'D'
    artifact = Artifact.import_data('FeatureTable[RelativeFrequency]', table)
    artifact.save('test_table.qza')
    metadata.to_csv('test_md.txt', sep='\t')


def get_metadata_control_and_case_sets(metadata: str,
                                       control_case_columns_data_path, relevant_ids) -> pd.DataFrame:
    """
    Parses the table that says which samples are control samples in each batch.
    :param metadata: path to metadata file.
    :param control_case_columns_data_path: path to file with information on which columns
    are the disease state indicators, and what value signifies a control sample
    :param relevant_ids: the sample ids to pass from metadata to the output
    :return: a pandas dataframe with batch and control/case membership information for all samples
    """
    metadata = pd.read_csv(metadata, sep='\t', index_col=0)
    metadata.index = metadata.index.astype("str")
    control_data = pd.read_csv(control_case_columns_data_path, sep='\t', index_col=0)
    in_table_but_not_metadata = len(set(relevant_ids) - set(metadata.index.tolist()))
    if in_table_but_not_metadata > 0:
        print("Warning: table contains {} samples not in metadata, which will be ignored".format(
            in_table_but_not_metadata))
        relevant_ids = [ind for ind in relevant_ids if ind in metadata.index.tolist()]
    output = pd.DataFrame(index=relevant_ids, columns=['batch', 'set'], dtype=str)
    output['batch'] = [metadata.at[ind, 'source_file'] for ind in relevant_ids]
    for ind in relevant_ids:
        cohort = output.at[ind, 'batch']
        control_cat = control_data.at[cohort, 'CONTROL_CAT']
        control_value = control_data.at[cohort, 'VALUE']
        # if metadata.at[ind, control_cat.lower()] == control_value: # Guys' old version
        if metadata.at[ind, control_cat] == control_value:
            output.at[ind, 'set'] = 'control'
        else:
            output.at[ind, 'set'] = 'case'
    return output


def set_axes_equal(ax):
    '''
    credit to karlo - https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def pseudoount_relative_frequency_df(df: pd.DataFrame):
    minimum = df.min().min()
    if minimum == 0:
        num_samples = len(df.index)
        minimum = df.replace(0, 1).to_numpy().min()
        df = df.applymap(lambda x: x if x > 0 else uniform_open(0, minimum / 2))
        df = df.div(df.sum(axis=1), axis=0)
    return df


def uniform_open(low, high):
    result = random.uniform(low, high)
    while not low < result < high:
        result = random.uniform(low, high)
    return result


def abs_pseudocount(artifact: Artifact) -> Artifact:
    df: pd.DataFrame = artifact.view(pd.DataFrame)
    minimum = df.to_numpy().min()
    if minimum == 0:
        df += 1
    return Artifact.import_data('FeatureTable[Frequency]', df)


def _pseudocount(df: pd.DataFrame) -> pd.DataFrame:
    assert validate_relative_abundance_dataframe(df)
    output = copy.deepcopy(df)
    minimum = output.to_numpy().min()
    assert minimum >= 0
    if minimum == 0:
        minimum = output.replace(0, 1).to_numpy().min()
        minimum = minimum / (len(output.columns) * len(output.index))
        # output = output.applymap(lambda x: uniform_open(0, minimum) if x == 0 else x)
        output = output.applymap(lambda x: minimum if x == 0 else x)
        output = output.div(output.sum(axis=1), axis=0)
        assert output.to_numpy().min() > 0
    assert validate_relative_abundance_dataframe(output)
    return output


def pseudocount(artifact: Artifact) -> Artifact:
    df: pd.DataFrame = artifact.view(pd.DataFrame)
    output = _pseudocount(df)
    return Artifact.import_data('FeatureTable[RelativeFrequency]', output)


def pseudocount_absolute_frequency(artifact: Artifact) -> Artifact:
    df: pd.DataFrame = artifact.view(pd.DataFrame)
    df = df.replace(0, 1)
    return Artifact.import_data('FeatureTable[Frequency]', df)


def _clr_transformation(artifact: Artifact, pseudocount=None):
    df = artifact.view(pd.DataFrame)
    minimum = df.to_numpy().min()
    if minimum == 0:
        num_samples = len(df.index)
        minimum = df.replace(0, 1).to_numpy().min()
        df = df.applymap(lambda x: x if x > 0 else uniform_open(0, minimum / num_samples * 10))
        df = df.div(df.sum(axis=1), axis=0)

    assert np.min(np.min(df)) > 0
    assert validate_relative_abundance_dataframe(df)
    print(df)

    df.loc[:, :] = clr(df.loc[:, :])
    artifact = Artifact.import_data('FeatureTable[RelativeFrequency]', df)
    artifact.save('here.qza')
    return artifact


def clr_transformation(artifact: str, dest: str, pseudocount=None):
    artifact = Artifact.load(artifact)
    artifact = _clr_transformation(artifact, pseudocount)
    artifact.save(dest)


def remove_renegade_samples(artifact, control_case_columns_data, ignore_samples_without_metadata=True):
    df: pd.DataFrame = artifact.view(pd.DataFrame)
    missing = [sample for sample in list(df.index.values) if sample not in control_case_columns_data.index]
    if missing:
        if not ignore_samples_without_metadata:
            raise ValueError('Some samples from the feature table have no metadata. If you wish to ignore and remove'
                             'these samples from the results, set "ignore_samples_without_metadata" to "False" in'
                             '"cfg.yaml". The output will completely omit these samples.')
        print('filtering out samples missing from metadata. These samples will be omitted from the results!')
        df = df.drop(index=missing)
    return Artifact.import_data('FeatureTable[RelativeFrequency]', df)


def impute_zeroes(src, dest):
    df = Artifact.load(src)
    df = df.view(pd.DataFrame)
    df = df.applymap(lambda abundance: abundance if abundance > 0 else 0.5)
    # df = df.applymap(lambda abundance: abundance if abundance > 0 else np.random.uniform(0, 10 ** -5))
    artifact = Artifact.import_data('FeatureTable[Frequency]', df)
    artifact.save(dest)


def filter_merge_drop_small_datasets(output_path, control_case_columns_data, merged_metadata, thresh, *paths):
    '''
    Similar to filter_merge but after the filtering it removes samples belonging to cohorts with < thresh
    samples remaining.
    '''
    filter_merge(output_path, *paths)
    df = Artifact.load(output_path).view(pd.DataFrame)
    control_case_columns_data: pd.DataFrame = \
        get_metadata_control_and_case_sets(merged_metadata, control_case_columns_data, df.index.tolist())
    ids = get_sample_ids(control_case_columns_data)
    for cohort in ids.keys():
        remaining_samples = len(set(df.index) & set(ids[cohort]['control']))
        if remaining_samples < thresh:
            print('dropped ' + cohort + ' - ' + str(remaining_samples) + ' samples')
            df.drop(index=ids[cohort]['control'] + ids[cohort]['case'], inplace=True)
        else:
            print('did not drop ' + cohort + ' - ' + str(remaining_samples) + ' samples')
    Artifact.import_data('FeatureTable[Frequency]', df).save(output_path)


def choose(artifact_path, metadata_path, col, output_path):
    md = pd.read_csv(metadata_path, sep='\t', index_col=0)
    art = Artifact.load(artifact_path)
    df = art.view(pd.DataFrame)
    to_keep = []
    seen_values = set()
    for ind in md.index:
        if md.at[ind, col] not in seen_values:
            seen_values.add(md.at[ind, col])
            to_keep.append(ind)
    df = df.loc[to_keep, :]
    art = Artifact.import_data('FeatureTable[Frequency]', df)
    art.save(output_path)


def filter_merge(output_path, *paths) -> None:
    '''
    Merges the qiime feature tables after filtering them:
    minimal_prevalence = OTUs must appear in at least this fraction of samples in every individual table
    minimal_abundance = OTUs must appear at least this many times (total in all samples) in every table
    minimal_sequencing_depth = samples must have at least this sequencing depth

    Parameters
    ----------
    output_path: path of the output merged filtered artifact
    paths: paths pf the artifacts to merge
    '''
    minimal_prevalence = 0.1
    minimal_abundance = 100
    minimal_sequencing_depth = 1000
    # maximal_relative_frequency = 1/3
    dataframes = [Artifact.load(path).view(pd.DataFrame) for path in paths]

    for df in dataframes:
        df.index = df.index.map(str)
    if len(dataframes) == 1:
        warnings.warn('Only one dataset provided and will only be filtered')

    otus_with_min_prevalence_in_all_tables = set(dataframes[0].columns.tolist())

    # rf_dataframes = dataframes.copy()
    # rf_dataframes = [frame.div(frame.sum(axis=1), axis=0) for frame in rf_dataframes]
    #
    # for i in range(len(dataframes)):
    #     df = rf_dataframes[i]
    #     print(df)
    #     samples = set([sample for sample, relative_abundances in df.iterrows() if not any([ra > maximal_relative_frequency for ra in relative_abundances])])
    #     dataframes[i] = dataframes[i].loc[samples,:]
    #     print('passed first check:',len(samples))

    for df in dataframes:
        num_samples = len(df.index)
        minimum_samples_containing_otu = minimal_prevalence * num_samples

        otus_with_min_prevalence_in_all_tables = otus_with_min_prevalence_in_all_tables & \
                                                 set([otu for otu, abundances in df.iteritems() if
                                                      sum(abundances > 0) >= minimum_samples_containing_otu and
                                                      sum(abundances) >= minimal_abundance])

    for i in range(len(dataframes)):
        df = dataframes[i]
        df = df.loc[:, otus_with_min_prevalence_in_all_tables]
        samples = set([sample for sample, abundances in df.iterrows() if sum(abundances) >= minimal_sequencing_depth])
        dataframes[i] = df.loc[samples, otus_with_min_prevalence_in_all_tables]

    print(dataframes)
    output = pd.concat(dataframes)
    print(output)
    output = Artifact.import_data('FeatureTable[Frequency]', output)
    print(output.save(output_path))


def feature_table_relative_abundance(art: Artifact) -> Artifact:
    df = art.view(pd.DataFrame)
    return Artifact.import_data('FeatureTable[RelativeFrequency]', df.div(df.sum(axis=1), axis=0))


def fill_zeroes_rf(infile, output):
    table: pd.DataFrame = Artifact.load(infile).view(pd.DataFrame)
    minimum = table.replace(0, 1).to_numpy().min()
    print(minimum)
    # table = table.applymap(lambda x: x if x > 0 else np.random.uniform(0, minimum/2))
    table = table.applymap(lambda x: x if x > 0 else minimum / 2)
    table = table.div(table.sum(axis=1), axis=0)
    Artifact.import_data('FeatureTable[RelativeFrequency]', table).save(output)


def random_shift_abs(infile, prefix, output):
    table: pd.DataFrame = Artifact.load(infile).view(pd.DataFrame)
    for otu in table.columns.tolist():
        X = table.loc[:, otu].values
        shift_sd = random.uniform(-1, 1)
        shift_sd = 10 ** shift_sd
        shift_mean = random.uniform(-0.2 * np.max(X), 0.2 * np.max(X))
        X = [(x
              * shift_sd
              )
             # + shift_mean
             + np.random.normal(0, np.max(X) * 0.1)
             for x in X]
        table.loc[:, otu] = X
    table = table.applymap(lambda x: int(x) if x > 1 else 0)
    new_indices: list = table.index.tolist()
    new_indices = [x + prefix for x in new_indices]
    table.index = new_indices

    Artifact.import_data('FeatureTable[Frequency]', table).save(output)


def validate_alignment_score():
    high_score_df = pd.DataFrame(np.random.normal(0, 1, (200, 50)))
    low_score_df = pd.DataFrame(np.concatenate((np.random.normal(0, 1, (100, 50)), np.random.normal(10, 1, (100, 50)))))
    pca = PCA(20)
    for df in [high_score_df, low_score_df]:
        pca_df = pd.DataFrame(pca.fit_transform(df))
        dmx = pairwise_distances(pca_df)
        k = int(0.1 * len(pca_df.index))
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(dmx)
        distances, indices = nbrs.kneighbors(dmx)
        x = []
        indices = pd.DataFrame(indices)
        for sample, neighbors in indices.iterrows():
            if sample < 100:
                same_batch = len([n for n in neighbors if n < 100])
            else:
                same_batch = len([n for n in neighbors if n >= 100])
            x.append(same_batch)
        print(1 - (np.mean(x) - k / len(indices.index)) / (k - k / len(indices.index)))


def alignment_score(artifact: Artifact, control_case_columns_data: pd.DataFrame, controls_only=True) -> float:
    df: pd.DataFrame = artifact.view(pd.DataFrame)
    if controls_only:
        df = df.loc[control_case_columns_data[control_case_columns_data['set'] == 'control'].index, :]
    pca = PCA(20)
    pca_df = pd.DataFrame(index=df.index, data=pca.fit_transform(df))
    dmx = pairwise_distances(pca_df)
    k = int(0.1 * len(pca_df.index))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(dmx)
    distances, indices = nbrs.kneighbors(dmx)
    indices = pd.DataFrame(indices)
    indices = indices.applymap(lambda x: df.index[x])
    indices = indices.set_index(np.array([df.index[x] for x in indices.index]))
    indices: pd.DataFrame = indices.applymap(lambda x: control_case_columns_data.at[x, 'batch'])
    x = []
    for sample, nn in indices.iterrows():
        self_batch = control_case_columns_data.at[sample, 'batch']
        same_batch = nn.tolist().count(self_batch)
        x.append(same_batch)

    return 1 - (np.mean(x) - k / len(indices.index)) / (k - k / len(indices.index))


def random_shift_abs_to_rf(infile, prefix, output):
    table: pd.DataFrame = Artifact.load(infile).view(pd.DataFrame)
    for otu in table.columns.tolist():
        X = table.loc[:, otu].values
        shift_sd = random.uniform(-1, 1)
        shift_sd = 10 ** shift_sd
        shift_mean = random.uniform(-0.2 * np.max(X), 0.2 * np.max(X))
        X = [(x
              * shift_sd
              )
             # + shift_mean
             + np.random.normal(0, np.max(X) * 0.1)
             for x in X]
        table.loc[:, otu] = X
    table = table.applymap(lambda x: int(x) if x > 1 else 0)
    new_indices: list = table.index.tolist()
    new_indices = [x + prefix for x in new_indices]
    table.index = new_indices
    table = table.div(table.sum(axis=1), axis=0)

    Artifact.import_data('FeatureTable[Frequency]', table).save(output)


def random_shift_rf(infile, prefix, output):
    table: pd.DataFrame = Artifact.load(infile).view(pd.DataFrame)
    for otu in table.columns.tolist():
        X = table.loc[:, otu]
        shift_sd = random.uniform(0.1, 10)
        shift_mean = random.uniform(0, 0.00001)
        X = [(x
              * shift_sd
              )
             + shift_mean
             for x in X]
        table.loc[:, otu] = X
    new_indices: list = table.index.tolist()
    new_indices = [x + prefix for x in new_indices]
    table.index = new_indices
    table = table.div(table.sum(axis=1), axis=0)
    for _, sample in table.iterrows():
        assert equal(sample.to_numpy().sum(), 1)
    Artifact.import_data('FeatureTable[RelativeFrequency]', table).save(output)


def random_shift_v3(input, prefix, output):
    table: pd.DataFrame = Artifact.load(input).view(pd.DataFrame)
    for otu in table.columns.tolist():
        X = table.loc[:, otu]
        shift_mean = random.uniform(0, 5)
        shift_sd = random.uniform(0, 2)
        X = [max(0, x + np.random.normal(shift_mean, shift_sd)) for x in X]
        table.loc[:, otu] = X
    new_indices: list = table.index.tolist()
    new_indices = [x + prefix for x in new_indices]
    table.index = new_indices
    Artifact.import_data('FeatureTable[Frequency]', table).save(output)


def random_shift_new(input, prefix, output):
    table: pd.DataFrame = Artifact.load(input).view(pd.DataFrame)
    for otu in table.columns.tolist():
        X = table.loc[:, otu]
        shift_sd = random.uniform(1, 1.5)
        shift_mean = random.uniform(0, 3)
        X = [(x * shift_sd) + shift_mean
             + max(np.random.normal(2, 1), 0)
             for x in X]
        table.loc[:, otu] = X
    new_indices: list = table.index.tolist()
    new_indices = [x + prefix for x in new_indices]
    table.index = new_indices
    Artifact.import_data('FeatureTable[Frequency]', table).save(output)


def random_shift_sd(input_path, prefix, output):
    table: pd.DataFrame = Artifact.load(input_path).view(pd.DataFrame)
    for otu in table.columns.values.tolist():
        X = table.loc[:, otu]
        shift_sd = random.uniform(0.75, 1.5)
        X = [x * shift_sd for x in X]
        table.loc[:, otu] = X
    new_indices = table.index.values.tolist()
    new_indices = [x + prefix for x in new_indices]
    table.index = new_indices
    art = Artifact.import_data('FeatureTable[Frequency]', table)
    art.save(output)


def random_shift_2(input_path, prefix, output, metadata_path, md_output):
    table: pd.DataFrame = Artifact.load(input_path).view(pd.DataFrame)
    for otu in table.columns.values.tolist():
        X = table.loc[:, otu]
        shift_sd = random.uniform(0.1, 50)
        shift_mean = random.uniform(0, 0.5)
        X = [(x * shift_sd) + shift_mean
             # + max(np.random.normal(2, 1), 0)
             for x in X]
        table.loc[:, otu] = X
    new_indices = table.index.values.tolist()
    new_indices = [x + prefix for x in new_indices]
    table.index = new_indices
    Artifact.import_data('FeatureTable[Frequency]', table).save(output)
    metadata = pd.read_csv(metadata_path, sep='\t')
    metadata.set_index('#SampleID', inplace=True)
    metadata.index = [x + prefix for x in metadata.index.values.tolist()]
    metadata.to_csv(md_output, sep='\t')


def random_shift(input_path, suffix, output, metadata_path, md_output):
    table: pd.DataFrame = Artifact.load(input_path).view(pd.DataFrame)
    for otu in table.columns.values.tolist():
        X = table.loc[:, otu]
        shift_sd = random.uniform(0.2, 5)
        X = [x * shift_sd for x in X]
        table.loc[:, otu] = X
    new_indices = table.index.values.tolist()
    new_indices = [x + suffix for x in new_indices]
    table.index = new_indices
    Artifact.import_data('FeatureTable[Frequency]', table).save(output)
    metadata = pd.read_csv(metadata_path, sep='\t')
    metadata.set_index('#SampleID', inplace=True)
    metadata.index = [x + suffix for x in metadata.index.values.tolist()]
    metadata.to_csv(md_output, sep='\t')


def equal(a, b, threshold=1 / 10 ** 10):
    return abs(a - b) < threshold


def add_prefix_to_path(prefix: str, path: str) -> str:
    if '/' not in path:
        return prefix + path
    return path[:path.rfind('/') + 1] + prefix + path[path.rfind('/') + 1:]


def split_basis(metadata, basis):
    with open(metadata, 'r') as fp:
        lines = fp.readlines()
    indices_of_basis = [lines.index(line) for line in lines if basis in line]
    random.shuffle(indices_of_basis)
    chosen_indices = indices_of_basis[:int(len(indices_of_basis) / 10)]
    with open(add_prefix_to_path('split_basis_', metadata), 'w+') as fp:
        for i in range(len(lines)):
            if i in chosen_indices:
                fp.write(lines[i][:-1] + '_minor\n')
            elif i in indices_of_basis:
                fp.write(lines[i][:-1] + '_major\n')
            else:
                fp.write(lines[i])


def merge_metadata_files(outfile, *infiles):
    infiles = list(infiles)
    ilines = []
    for file in infiles:
        print('reading', file)
        with open(file, 'r', encoding='latin-1') as ifp:
            ilines.append(ifp.readlines())
    id_list = []
    metadata = {}
    metakeys = []
    for i in range(len(ilines)):
        file_name = infiles[i]
        lines = ilines[i]
        keylist = lines[0].strip().split('\t')[1:]
        for key in keylist:
            if str.lower(key) not in metakeys:
                metakeys.append(str.lower(key))
        linelist = []
        for line in lines[1:]:
            items = line.strip('\n').split('\t')
            sample_id = items[0]
            items = items[1:]
            if sample_id in id_list:
                raise ValueError('Duplicate ids:', sample_id)
            id_list.append(sample_id)
            metadata[sample_id] = {}
            j = 0
            while j < len(keylist):
                metadata[sample_id][str.lower(keylist[j])] = items[j]
                j += 1
            metadata[sample_id]['source_file'] = file_name
            linelist.append(items)
    metakeys.append('source_file')

    with open(outfile, 'w+', encoding='latin-1') as ofp:
        ofp.write('#SampleID\t' + '\t'.join(metakeys))
        for sample_id in id_list:
            ofp.write('\n')
            line = sample_id
            for key in metakeys:
                value = metadata[sample_id][key] if key in metadata[sample_id] else 'NA'
                line += '\t' + value
            ofp.write(line)


def taxa_collapse(infile: str, level: int, outfile: str):
    with open(infile, 'r') as infile:
        lines = infile.readlines()
    i = 0
    while not lines[i].startswith('#OTU'):
        i += 1
    sample_ids = lines[i].strip('\n').split('\t')[1:]
    sample_to_abundance = {item: {} for item in sample_ids}
    lines = lines[i + 1:]
    for line in lines:
        line = line.strip('\n').split('\t')
        taxa = line[0]
        taxa = taxa.split(';')[:level]
        line = line[1:]
        assert (len(sample_ids) == len(line))
        for j in range(len(sample_ids)):
            sample_to_abundance[sample_ids[j]][taxa] = line[j]


remove_list = ['subclass', 'suborder']


def validate_relative_abundance_dataframe(df: pd.DataFrame) -> bool:
    """
    Verifies that the input dataframe represents a valid (samples x features) relative abundance table.
    Parameters
    ----------
    df - the feature table as pandas dataframe

    Returns
    -------
    True if:
        - The sum of every row is 1 +- 10^-7
        - There are no nan values
    False otherwise
    """
    try:
        assert not df.isnull().values.any()
        assert all([equal(i, 1, 10 ** -7) for i in df.sum(axis=1)])
    except AssertionError:
        return False
    return True


def clean_subclass(infile: str):
    with open(infile, 'r') as ifile:
        lines = ifile.readlines()
    with open('no_subclass_' + infile, 'w+') as ofile:
        for line in lines:
            for item in remove_list:
                if item in line:
                    line = line.split('\t')
                    i = line.index(item)
                    line = '\t'.join(line[0:i - 1]) + '\t' + '\t'.join(line[i + 2:])
                ofile.write(line)


def gneiss_tree(table: Artifact, phylogeny: Artifact) -> skbio.TreeNode:
    table_as_normal_frequency = table.view(pd.DataFrame)
    table_as_normal_frequency = Artifact.import_data('FeatureTable[Frequency]', table_as_normal_frequency)
    _, hierarchy = ilr_phylogenetic(table_as_normal_frequency, phylogeny)
    return hierarchy.view(skbio.TreeNode)


def read_otu_table(table_path: str) -> (dict, list, list):
    with open(table_path, 'r') as infile:
        lines = infile.readlines()
    i = 0
    while not 'OTU_ID' in lines[i]:
        i += 1
    sample_ids = lines[i].strip('\n').split('\t')[1:]
    taxa_ids = []
    sample_to_abundance = {item: {} for item in sample_ids}
    lines = lines[i + 1:]
    for line in lines:
        line = line.strip('\n').split('\t')
        taxa = line[0]
        taxa_ids.append(taxa)
        line = line[1:]
        assert (len(sample_ids) == len(line))
        for j in range(len(sample_ids)):
            sample_to_abundance[sample_ids[j]][taxa] = float(line[j])
    return sample_to_abundance, sample_ids, taxa_ids


def write_otu_table(table_data: dict, sample_ids: list, otu_ids: list, output_path: str):
    with open(output_path, 'w+') as fp:
        header = '#OTU ID\t' + '\t'.join(sample_ids)
        fp.write(header)
        for otu in otu_ids:
            line = '\n' + otu
            for sample in sample_ids:
                line += '\t' + str(table_data[sample][otu])
            fp.write(line)


def filter_otu_data(otu_table: dict, sample_ids: list, taxa_ids: list,
                    min_per_sample=100, min_per_otu=50, min_samples_with_otu=0.1) -> (dict, list, list, str):
    num_samples = len(sample_ids)
    samples_to_remove = []
    otus_to_remove = []
    otu_totals = {}
    for sample_id, sample in otu_table.items():
        total_per_sample = 0
        for taxon_id, count in sample.items():
            if taxon_id not in otu_totals:
                otu_totals[taxon_id] = {
                    'total': count,
                    'presence': 1
                }
            else:
                otu_totals[taxon_id]['total'] += count
                otu_totals[taxon_id]['presence'] += 1
            total_per_sample += count
        if total_per_sample < min_per_sample:
            samples_to_remove.append(sample_id)
    for otu, data in otu_totals.items():
        if data['total'] < min_per_otu or data['presence'] / num_samples < min_samples_with_otu:
            otus_to_remove.append(otu)
    for sample in samples_to_remove:
        sample_ids.remove(sample)
    for otu in otus_to_remove:
        taxa_ids.remove(otu)
    return otu_table, sample_ids, taxa_ids


def filter_otu_table(input_path, output_path):
    table_data = read_otu_table(input_path)
    filtered_data = filter_otu_data(*table_data)
    write_otu_table(*filtered_data, output_path=output_path)


def add_prefixes_to_otus(path, prefix):
    with open(path, 'r') as fp:
        lines = fp.readlines()
    with open('x_' + path, 'w+') as fp:
        for line in lines:
            if '#' not in line and 'OTU' not in line:
                line = prefix + '_' + line
            fp.write(line)


def change_suffix(path, extension):
    """
    returns the path but with extension appended instead of the old extention, or adds an extension if it didn't have one
    :param path:
    :param extension:
    :return:
    """
    dot_index = path.rfind('.')
    if dot_index == -1:
        return path + '.' + extension
    return path[:dot_index] + '.' + extension


def _vsearch(ref, seq, table, similarity):
    for artifact in [ref, seq, table]:
        assert (artifact.endswith('.qza'))
    annotated_table = 'annotated_' + table
    assert (not os.path.isfile(annotated_table))
    try:
        slash_index = table.rindex('/')
        annotated_table = table[:slash_index] + '/annotated_' + table[slash_index + 1:]
        print(annotated_table)
    except ValueError:
        pass
    command = "qiime vsearch cluster-features-closed-reference " \
              + "--i-reference-sequences " + ref + " --i-sequences " + seq + " --i-table " + table \
              + " --o-clustered-table " + annotated_table + " --o-clustered-sequences cs --o-unmatched-sequences us" \
              + " --p-perc-identity " + similarity + " --p-strand both"
    os.system(command)
    os.system('rm cs.qza && rm us.qza')
    assert (os.path.exists(annotated_table))
    return annotated_table


def vsearch(ref, seq, table, similarity, clean='clean'):
    annotated_table = 'annotated_' + table
    assert (not os.path.isfile(change_suffix(annotated_table, 'qza')))
    if not (clean == 'clean' or clean == 'no_clean'):
        raise AttributeError('clean variable must be "clean" or "not_clean". passed: ' + clean)
    cleaners = []
    if not ref.endswith('.qza'):
        ref = qiime_import(ref, 'FeatureData', 'Sequence')
    if not seq.endswith('.qza'):
        seq = qiime_import(seq, 'FeatureData', 'Sequence')
        cleaners.append(seq)
    if not table.endswith('.qza'):
        table = table_tsv_to_artifact(table, 'Frequency')
        cleaners.append(table)
    table_name = _vsearch(ref, seq, table, similarity)
    print('cleaning files', cleaners, '...')
    for file in cleaners:
        os.system('rm ' + file)
    return table_name


def mass_filter_samples(otu_dir, md_dir, filter_file, out_path):
    with open(filter_file, 'r') as fp:
        lines = fp.readlines()
    command = """qiime feature-table filter-samples --i-table """ + otu_dir + """/annotated_{}.otu_table.100.qza --m-metadata-file """ + md_dir + """/{}.metadata.txt --p-where "{}" --o-filtered-table """ + out_path + """/{}.feces_controls.qza"""
    for line in lines:
        line = line.strip().split('\t')
        db = line[0]
        condition = "'" + line[1] + "'='" + line[2] + "'"
        if len(line) > 3:
            condition += " AND '" + line[3] + "'='" + line[4] + "'"
        out_file_path = out_path + '/' + db + '.feces_controls.qza'
        if os.path.exists(out_file_path):
            print(out_file_path, 'exists')
            continue
        # assert(not os.path.isfile(out_file_path))
        print('now filtering', db)
        os.system(command.format(db, db, condition, db))
        assert (os.path.isfile(out_file_path))
        print('Filtered', db)


def mass_join_otu_tables(dir):
    command = 'qiime feature-table merge --i-tables '
    files = os.listdir(dir)
    assert (not os.path.isfile(dir + '_mass_joined_table.qza'))
    assert (all([file.endswith('.qza')] for file in files))
    print('Attemping to merge ' + str(len(files)) + ' tables')
    files = ' --i-tables '.join(files)
    command = command + files + ' --o-merged-table ' + dir + '_mass_joined_table.qza'
    print(command)
    os.system(command)
    assert (os.path.isfile(dir + '_mass_joined_table.qza'))
    print('success')


def qiime_export(path: str, extension: str) -> str:
    assert path.endswith(".qza")
    dot_index = path.rfind('.')
    out = path[:dot_index] + '.' + extension
    assert not os.path.isdir(out)
    assert not os.path.isfile(out)
    os.system('qiime tools export --input-path ' + path + ' --output-path ' + out)
    if os.path.isdir(out):
        os.system('mv ' + out + '/' + os.listdir(out)[0] + ' ' + out + 'tmp')
        os.system('rm -r ' + out)
        os.system('mv ' + out + 'tmp ' + out)
    assert not os.path.isdir(out)
    assert os.path.isfile(out)
    return out


def qiime_import(path, type, semantic_type):
    dot_index = path.rfind('.')
    if not dot_index == -1:
        out = path[:dot_index] + '.qza'
    else:
        out = path + '.qza'
    os.system('qiime tools import --input-path ' + path + ' --output-path ' + out
              + ' --type ' + type + '[' + semantic_type + ']')
    return out


def mass_join_metadata(dir):
    if not dir.endswith('/'):
        dir = dir + '/'
    files = [dir + file for file in os.listdir(dir) if not file.startswith('.')]
    merge_metadata_files(dir + 'merged_metadata.txt', *tuple(files))


def pcoa(table, phylogeny, metadata):
    if not table.endswith('.qza'):
        table = table_tsv_to_artifact(table, 'RelativeFrequency')
    art = Artifact.load(table).view(pd.DataFrame)
    Artifact.import_data('FeatureTable[Frequency]', art).save('tmp_table.qza')
    dmx_name = change_suffix(table, 'dmx') + '.qza'
    pcoa_name = change_suffix(table, 'pcoa') + '.qza'
    visualize_name = change_suffix(table, 'pcoa') + '.qzv'
    os.system('qiime diversity-lib weighted-unifrac --i-table tmp_table.qza --i-phylogeny ' + phylogeny
              + ' --o-distance-matrix ' + dmx_name)
    os.system('qiime diversity pcoa --i-distance-matrix ' + dmx_name + ' --o-pcoa ' + pcoa_name)
    os.system('qiime emperor plot --i-pcoa ' + pcoa_name + ' --m-metadata-file ' + metadata
              + ' --o-visualization ' + visualize_name)
    os.system('rm tmp_table.qza')


def clean_dots_from_sample_names(metadata_file):
    with open(metadata_file, 'r') as fp:
        lines = fp.readlines()
    with open('dotless_md.txt', 'w+') as fp:
        for line in lines:
            line = line.split('\t')
            line[0] = line[0].replace('.', '')
            line = '\t'.join(line)
            fp.write(line)


def table_artifact_to_tsv(path: str):
    assert path.endswith(".qza")
    out = path[:-4] + ".tsv"
    os.system("qiime tools export --input-path " + path + " --output-path " + out)
    os.system("biom convert -i " + out + "/feature-table.biom -o " + out + ".tmp --to-tsv")
    os.system("rm -r " + out + "/")
    os.system("mv " + out + ".tmp " + out)
    assert (os.path.exists(out))
    return out


def table_tsv_to_artifact(path: str, semantic_type: str):
    out = change_suffix(path, 'qza')
    biom = change_suffix(path, 'biom')
    os.system("biom convert -i " + path + " -o " + biom + " --to-hdf5")
    os.system(
        "qiime tools import --input-path " + biom + " --output-path " + out + " --type FeatureTable[" + semantic_type + "]")
    os.system('rm ' + biom)
    assert (os.path.exists(out))
    return out


def mass_export_tables(dir):
    files = os.listdir(dir)
    for file in files:
        table_artifact_to_tsv(dir + '/' + file)


if __name__ == '__main__':
    args = len(sys.argv)
    utlls = __import__('utils')
    if args > 1:
        func = sys.argv[1]
    else:
        exit()
    func = getattr(utlls, func)
    if args > 2:
        args = tuple(sys.argv[2:])
        func(*args)
    else:
        func()

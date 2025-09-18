import math

import numpy as np
import pandas as pd
import skbio
# from qiime2 import Artifact
from skbio.stats.composition import clr, clr_inv

# from app import utils
from guy_shur_thesis import utils

class Artifact:
    pass

np.seterr(all='raise')
MAX_LOG_RATIO = 9


def log_ratio(x: float, y: float, base=10) -> float:
    ratio = x / y
    try:
        output = math.log(ratio, base)
    except ZeroDivisionError:
        raise ZeroDivisionError('Something went wrong while calculating the log ratio'
                                'for', ratio, 'with base', base)
    return output


def inverse_log_ratio(ratio: float, total: float, base=10) -> (float, float):
    assert 0 < total <= 1.0
    new_y = total / ((base ** ratio) + 1)
    new_x = total - new_y
    try:
        assert new_y > 0 and new_x > 0
    except AssertionError:
        print(ratio, total, new_x, new_y)
        exit(1)
    return new_x, new_y


def parse_metadata(path: str, relevant_samples: list) -> (dict, dict):
    """
    Returns two dictionaries, of cohort name -> samples in the cohort, and sample name -> that sample's cohort, to be
    used in recognizing which samples to use as a basis of transformations.
    :param path: the path for the metadata file
    :param relevant_samples: a list of sample ids from the feature table. used to determine which samples from the
    metadata file are not relevant to the current analysis
    """

    metadata_dict, sample_to_cohort = {}, {}
    with open(path, 'r') as file:
        metadata = file.readlines()
    for line in metadata[1:]:
        line = line.strip().split('\t')
        if line[0] not in relevant_samples:
            continue
        if line[-1] not in metadata_dict:
            metadata_dict[line[-1]] = []
        metadata_dict[line[-1]].append(line[0])
        sample_to_cohort[line[0]] = line[-1]
    return metadata_dict, sample_to_cohort


def export_result(path, sample_order, results):
    with open(path, 'w+') as fp:
        header = ''
        for sample in sample_order:
            header += '\t' + str(sample)
        fp.write('#OTU ID' + header)
        for taxa in results:
            fp.write('\n')
            line = str(taxa)
            for sample_id in sample_order:
                count = results[taxa][sample_id]
                line += '\t' + str(count)
            fp.write(line)


class _TMS:

    def __init__(self, artifact: pd.DataFrame, metadata: str, basis: str, category_data: pd.DataFrame,
                 mean_tuning_factor: float, sd_tuning_factor: float):

        self.basis = basis
        self.category_data = category_data
        self.mean_tuning_factor = mean_tuning_factor
        self.sd_tuning_factor = sd_tuning_factor
        self.otus: pd.DataFrame = artifact.view(pd.DataFrame)
        self.metadata = category_data

    def features(self):
        for feature in self.otus.columns.values:
            yield feature

    def sample_ids(self):
        for sample_id in self.otus.index.values:
            yield sample_id

    def moment_shift(self):

        # get basis controls
        # get basis controls mean and sd for each otu
        # for each [target cohort controls x single otu] get the otu parameters of that cohort controls vs basis
        # fit every otu in every target cohort

        self.otus.loc[:, :] = clr(self.otus.values)
        basis_controls_ids = \
            self.metadata[(self.metadata['batch'] == self.basis) & (self.metadata['set'] == 'control')].index.values
        basis_controls = self.otus.loc[basis_controls_ids, :].index.values
        for cohort in self.metadata['batch'].unique():
            if cohort == self.basis:
                continue
            target_ids = self.metadata[self.metadata['batch'] == cohort].index.values
            target_controls_ids = \
                self.metadata[(self.metadata['batch'] == cohort) & (self.metadata['set'] == 'control')].index.values
            for otu in self.features():

                curr_basis_controls = np.array(self.otus.loc[basis_controls, otu])
                curr_target_controls = np.array(self.otus.loc[target_controls_ids, otu])
                curr_targets = np.array(self.otus.loc[target_ids, otu])
                mean, sd = np.mean(curr_basis_controls), np.std(curr_basis_controls)
                params = learn_shift_params(curr_target_controls, mean, sd)
                if params:
                    shifted = fit(curr_targets, params[0], params[1])
                    # minimum = np.min(shifted)
                    # if minimum < 0:
                    #     offset_factor = np.mean(shifted) / np.mean(shifted) + -1 * minimum
                    # shifted[shifted < 0] = 0
                    # shifted = shifted * offset_factor
                    self.otus.loc[target_ids, otu] = shifted
        self.otus.loc[:, :] = clr_inv(self.otus)

    def normalize_results(self):
        self.otus = self.otus.div(self.otus.sum(axis=1), axis=0)


class _LRT:
    def __init__(self, artifact: Artifact, tree: skbio.TreeNode, basis: str, control_case_columns_data: pd.DataFrame):
        self.otus: pd.DataFrame = artifact.view(pd.DataFrame)
        self.otus.columns = self.otus.columns.astype(str)  # required
        self.sample_ids = list(self.otus.index.values)
        self.otu_ids = self.otus.columns.values
        self.basis = basis
        self.tree = tree
        self.log_ratios = pd.DataFrame(index=self.sample_ids, columns=[x.name for x in self.tree.non_tips()],
                                       dtype=float)
        self.control_case_columns_data = control_case_columns_data
        self.basis_control_samples = self.control_case_columns_data[(self.control_case_columns_data['batch'] == basis) &
                                                             (self.control_case_columns_data[
                                                                  'set'] == 'control')].index.values.tolist()
        self.node_names = [node.name for node in self.tree.non_tips()]
        self.node_names.append(
            self.tree.name)  # root's nome is not generated by TreeNode.non_tips and needs to be added here
        self.tip_names = [node.name for node in self.tree.tips()]
        try:
            assert set(self.tip_names) == set(self.otu_ids)
        except AssertionError as e:
            print('Error: Some or all of the ASV/OTU IDs in the feature tables do not appear in the phylogeny.')
            raise e

    def features(self):
        for feature in self.otus.columns.values:
            yield feature

    def samples(self):
        for sample_id in self.otus.index.values:
            yield sample_id

    def lr_transform(self):
        data = []
        ref_names = None
        for sample in list(self.otus.index.values):
            lrs = []
            names = []
            for node in self.tree.postorder():
                children = node.children
                if len(children) == 0:
                    node.abundance = self.otus.loc[sample, node.name]
                elif len(children) == 2:
                    node.abundance = children[0].abundance + children[1].abundance
                    names.append(node.name)
                    if children[0].abundance + children[1].abundance == 0:
                        lr = 0
                    elif children[1].abundance == 0:
                        lr = MAX_LOG_RATIO
                    elif children[0].abundance == 0:
                        lr = -1 * MAX_LOG_RATIO
                    else:

                        lr = log_ratio(children[0].abundance, children[1].abundance)
                        if lr > MAX_LOG_RATIO:
                            lr = MAX_LOG_RATIO
                        elif lr < -1 * MAX_LOG_RATIO:
                            lr = -1 * MAX_LOG_RATIO
                    lrs.append(lr)
                    self.log_ratios.at[sample, node.name] = lr

                else:
                    raise TypeError('Tree is not bifurcated at inner node', node.name)
            if not ref_names:
                ref_names = names
            else:
                assert ref_names == names
            data.append(lrs)
            if np.isnan(np.sum(np.array(lrs))):
                print(sample, lrs)
                raise ValueError('nan value')
        for node in self.log_ratios:
            assert not all([ratio == 0 for ratio in self.log_ratios[node]])

    def moment_shift(self):
        self.log_ratios = src.utils.shift_mean_and_sd(self.log_ratios, self.basis, self.control_case_columns_data)
        self.log_ratios[self.log_ratios > MAX_LOG_RATIO] = MAX_LOG_RATIO
        self.log_ratios[self.log_ratios < -1 * MAX_LOG_RATIO] = -1 * MAX_LOG_RATIO

    def reverse_transform(self):
        assert all([utils.equal(np.sum(abundances), 1, 1 / 10 ** 7) for _, abundances in self.otus.iterrows()])

        target_samples = list(self.control_case_columns_data[~(self.control_case_columns_data['batch'] == self.basis)].index.values)
        data = []
        ref_names = []
        self.tree.abundance = 1.0
        for sample in target_samples:
            names = []
            abundances = []
            for node in self.tree.preorder():
                children = node.children
                if len(children) == 0:
                    names.append(node.name)
                    abundances.append(node.abundance)
                elif len(children) == 2:
                    ratio = self.log_ratios.at[sample, node.name]
                    left_abundance, right_abundance = inverse_log_ratio(ratio=ratio,
                                                                        total=node.abundance)
                    children[0].abundance, children[1].abundance = left_abundance, right_abundance
                    assert utils.equal(node.abundance, children[0].abundance + children[1].abundance)

            if not ref_names:
                ref_names = names
            else:
                assert ref_names == names
            data.append(abundances)

        self.otus.loc[target_samples, ref_names] = np.array(data)
        assert all([utils.equal(np.sum(abundances), 1, 1 / 10 ** 7) for _, abundances in self.otus.iterrows()])


def fit_std(X, std, sd_tuning_factor):
    '''
    Scale the distribution X to fit the SD of Y.
    :param X: array to scale
    :param std: SD to fit to
    '''
    x_std = np.std(X)
    if utils.equal(x_std, 0.0):
        return X
    mean = np.mean(X)
    factor = std / x_std  # how much to scale each delta
    factor = sd_tuning_factor * (factor - 1) + 1

    def scaler(x):  # change the value of x by a factor that is the ratio of standard deviations
        delta = x - mean
        delta = delta * factor
        x = mean + delta
        return x

    scaler = np.vectorize(scaler)
    X = scaler(X)
    return X


def fit_mean(X, mean):
    '''
    Scale the mean X to fit the mean of Y.
    :param X: array to scale
    :param mean: mean to fit to
    '''
    x_mean = np.mean(X)
    delta = mean - x_mean
    return X + delta


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
    return a, b


def fit(X, a, b):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    return (a * X) + b


def fit_moments(X, mean, std) -> np.ndarray:
    if type(X) is not np.ndarray:
        X = np.array(X)

    if np.sum(X) == 0:
        return X
    if np.isnan(np.sum(X)):
        print(X)
        raise ValueError('array contains nan')
    x_std = np.std(X)
    if x_std == 0:
        X = X + (mean - np.mean(X))
    else:
        X = (X * std / np.std(X)) + (mean - np.mean(X))
    if np.isnan(np.sum(X)):
        print(X)
        exit(1)

    return X


def LRT(otu_table: Artifact, tree: Artifact, basis: str, control_data: pd.DataFrame):
    lrt = _LRT(otu_table, tree, basis, control_data)
    lrt.lr_transform()
    lrt.moment_shift()
    lrt.reverse_transform()
    return Artifact.import_data("FeatureTable[RelativeFrequency]", lrt.otus)


def TMS(otu_table, md, basis, category_data, mean_tuning_factor=1, sd_tuning_factor=1) -> Artifact:
    tms = _TMS(otu_table, md, basis, category_data, mean_tuning_factor, sd_tuning_factor)
    tms.moment_shift()
    tms.normalize_results()
    return Artifact.import_data("FeatureTable[RelativeFrequency]", tms.otus)


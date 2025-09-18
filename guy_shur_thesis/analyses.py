import copy
import math
import multiprocessing as mp
import os
import pickle as pkl
import random
import subprocess
import sys
import warnings
import yaml
from collections import Counter
from itertools import permutations
from typing import Tuple

import matplotlib.collections
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy
import seaborn as sns
import skbio
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib_venn import venn2, venn3
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from plotly.subplots import make_subplots
# from qiime2 import Artifact
# from qiime2.plugins import diversity_lib, diversity
from scipy import stats as sp
from scipy.spatial import procrustes
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
from scipy.stats import ranksums
from scipy.stats import ttest_ind
import scipy.stats as ss
from skbio.stats.composition import ancom
from skbio.stats.composition import clr
from skbio.stats.distance import permanova, DistanceMatrix
from skbio.stats.ordination import pcoa
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors, KernelDensity
from statsmodels.stats.multitest import fdrcorrection as fdr

# from app import correction_methods, order, order_with_NT, method_dict
# from app import metasparsim
# from app import utils
# from app.balance_tree import log_ratio, inverse_log_ratio
from guy_shur_thesis import utils
from guy_shur_thesis.balance_tree import log_ratio, inverse_log_ratio

class Artifact:
    pass

with open('guy_shur_thesis/cfg.yaml', 'r') as fp:
    cfg = yaml.full_load(fp)
order = cfg['methods']
order_with_NT = ['NT'] + order
method_dict = {
    'NT': 'No treatment',
    'NTP': 'No treatment pseusocount',
    'TMS': 'Direct abundance shift',
    'LRT': 'PHABA',
    'CB': 'ComBat',
    'MU': 'Adjust batch',
    'PN': 'Percentile normalization',
    'NN': 'Mutual nearest neighbors',
    'scaler': 'scaler',
    'PC': 'Principal Coordinate Shift',
    'LS': 'PLSDAbatch',
    'DCC': 'Direct Covariate Correction',
    'CQ': 'QonCuR'
}

methods = order
np.seterr(all='raise')
# vvv leave uncommented to cancel FDR vvv
fdr = lambda x: (None, x)
CMAP = cm.Reds

plt_color_cycle = [x['color'] for x in list(plt.rcParams['axes.prop_cycle'])]
plt_shape_cycle = ['o', 's', '^', 'x', 'P', 'D']
plt_color_cycle = [tuple(int(x.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for x in plt_color_cycle]

sns.set_context('paper')
lock = mp.Lock()


def vals_to_rgba(vals, alpha=1.0):
    max_val = max(vals)
    output = [CMAP(int(val * 255 / max_val)) for val in vals]
    output = [(x[0],x[1],x[2],alpha) for x in output]
    return output

def darker(rgb: tuple, factor=2):
    return tuple((i / factor) for i in rgb)


def rgb_to_plotly_color_str(rgb, alpha=1.0):
    """
    Converts rgb list/tuple to plotly-usable rgba string.
    :param rgb: list or tuple with red,green,blue coponents in range (0,255).
    :param alpha: transparency.
    :return: a string in the format 'rgba(r,g,b,a)'.
    """
    assert type(rgb) == tuple
    return 'rgba(' + str(rgb[0]) + ', ' + str(rgb[1]) + ', ' + str(rgb[2]) + ', ' + str(alpha) + ')'


def plotly_titration_subplots(num_plots, titles, title_fontsize=16, title_color='#2a3f5f'):
    """
    generates two rows of subplots. Used for titration plots. Plots are distributed equally with at most
    one extra plot in the top row.
    :param num_plots: Total number of plots to generate.
    :param titles: Titles for the subplots, ordered left to right and up to down.
    :return: A plotly fig object.
    """
    if num_plots % 2 == 1:
        num_plots_top = math.ceil(num_plots / 2)
        fig = make_subplots(
            rows=2, cols=num_plots_top * 2,
            specs=[[{"colspan": 2}, None] * num_plots_top,
                   [None] + [{"colspan": 2}, None] * (num_plots_top - 1) + [None]],
            print_grid=False,
            vertical_spacing=0.13,
            subplot_titles=titles)
    elif True:
        num_plots_top = math.ceil(num_plots / 2)
        fig = make_subplots(
            rows=2, cols=num_plots_top,
            specs=[[{}] * num_plots_top,
                   [{}] * num_plots_top],
            print_grid=False,
            vertical_spacing=0.13,
            subplot_titles=titles)
    else:
        fig = make_subplots(
            rows=2, cols=int(num_plots/2),
            specs=[[{}] * int(num_plots/2), [{}] * int(num_plots/2)],
            print_grid=False,
            vertical_spacing=0.13,
            #horizontal_spacing=0.2,
            subplot_titles=titles)
        for i in fig['layout']['annotations']:
            i['font'] = dict(size=title_fontsize, color=title_color)
    return fig


def break_title(title: str) -> str:
    """
    splits plotly subplot title to two lines if over 2 words.
    :param title: the title.
    :return: the same title, with <br> in place of the second space if longer than two words.
    """
    if title.count(' ') > 1:
        second_space_index = title.find(' ', title.find(' ') + 1)
        title = title[0:second_space_index] + '<br>' + title[second_space_index + 1:]
    return title


def make_subplots_for_ranksum_meta(num_plots, scale, titles):
    """
    generates two rows per batch (four total) of subplots. Used for ranksum correlation plots.
    Plots are distributed equally with at most one extra plot in the top two rows.
    Scale adjusts the vertical space by adding empty space. the vertical space added is 1/scale relative to
    the size of the subplots.
    :param num_plots: Total number of plots to generate. In practice 2*num_plots subplots will be generated.
    :param scale: The size of subplots relative to the horizontal distance between them (ratio is scale:1).
    Must be even.
    :param titles: Titles for the subplots, ordered left to right and up to down.
    :return: A plotly fig object.
    """
    assert scale % 2 == 0
    num_top = math.ceil(num_plots / 2)
    num_bottom = num_plots - num_top
    titles = [break_title(title) for title in titles]
    titles = titles[:num_top] + [''] * num_top + titles[num_top:] + [''] * num_bottom
    buff = [None] * (num_plots % 2)
    top_plot_row_specs = ([{'rowspan': scale, 'colspan': scale}] + \
                          [None] * scale) * (num_top - 1) + \
                         [{'rowspan': scale, 'colspan': scale}] + \
                         [None] * (scale - 1)
    none_row = [None] * (scale * num_top + num_top - 1)
    bottom_plot_row_specs = buff * int(scale / 2) + \
                            ([{'rowspan': scale, 'colspan': scale}] + [None] * scale) * (num_bottom - 1) + \
                            [{'rowspan': scale, 'colspan': scale}] + [None] * (scale - 1) + \
                            buff * int(scale / 2)
    specs = [top_plot_row_specs] + [none_row for _ in range(scale - 1)] + \
            [top_plot_row_specs] + [none_row for _ in range(scale)] + \
            [bottom_plot_row_specs] + [none_row for _ in range(scale - 1)] + \
            [bottom_plot_row_specs] + [none_row for _ in range(scale - 1)]
    return make_subplots(rows=4 * scale + 1, cols=(num_top * scale) + num_top - 1,
                         specs=specs, vertical_spacing=0.03,
                         horizontal_spacing=0.03, subplot_titles=titles)


def aligned_subplots(num_subplots, num_rows, height=None, dpi=200, whitespace=0.1, aspect_ratio=None):
    """
    Generates abitrary number of subplots with a correctly centered final row.
    :param num_subplots: Total number of subplots to generate.
    :param num_rows: Number of rows to arrange plots in.
    :param height: Height for the image.
    :param dpi: Image resolution.
    :param whitespace: fraction of whitespace between subplots.
    :param aspect_ratio: image aspect ratio.
    :return: fig and axes objects with every row of subplots centered.
    """
    if aspect_ratio and not height:
        raise Exception("must set figure height if aspect ratio is not None")
    if height:
        if not aspect_ratio:
            aspect_ratio = num_subplots / num_rows
        figsize = (aspect_ratio * height, height)
    else:
        figsize = None
    fig = plt.figure(figsize=figsize, dpi=dpi)

    max_subplots_per_row = math.ceil(num_subplots / num_rows)
    h_spacing = whitespace / (max_subplots_per_row + 1)
    v_spacing = whitespace / (num_rows + 1)
    plot_size = (1 - whitespace) / max_subplots_per_row
    axes = []
    row = 0
    col = 0
    while row < num_rows - 1:
        x = h_spacing * (1 + col) + plot_size * col
        y = 1 - (v_spacing * (1 + row)) - plot_size * row
        axes.append(fig.add_axes(
            [x, y, plot_size, plot_size], transform=fig.transFigure))
        if col == max_subplots_per_row - 1:
            col = 0
            row += 1
        else:
            col += 1
    if num_subplots % num_rows == 0:
        for i in range(max_subplots_per_row):
            x = h_spacing * (1 + col) + plot_size * col
            y = 1 - (v_spacing * (1 + row)) - plot_size * row

            axes.append(fig.add_axes(
                [x, y, plot_size, plot_size], transform=fig.transFigure))
            col += 1
    else:
        remaining_plots = num_subplots % num_rows
        buffer = (plot_size + h_spacing) * (max_subplots_per_row - remaining_plots) / 2
        for i in range(remaining_plots):
            x = buffer + h_spacing * (1 + col) + plot_size * col
            y = 1 - (v_spacing * (1 + row)) - plot_size * row
            axes.append(fig.add_axes(
                [x, y, plot_size, plot_size], transform=fig.transFigure))
            col += 1
    return fig, axes


def significant_otus_plot(subdirs: list, results_path: str, out_path: str) -> None:
    fig, axes = plt.subplots(2, len(subdirs), gridspec_kw={'height_ratios': [1, len(order) * 13 / 6]})
    if len(subdirs) == 1:
        axes = axes.reshape((2, 1))
    for subdir_index in range(len(subdirs)):
        subdir = subdirs[subdir_index]
        current_result_directory = results_path + subdir + '/'
        control_case_columns_data = pd.read_csv(current_result_directory + 'batches.tsv', sep='\t', index_col=0)
        print(order_with_NT)
        print(current_result_directory)
        artifact_paths = {method:
                              [current_result_directory + path for path in os.listdir(current_result_directory) if
                               path.startswith(method) and path.endswith('.qza')][0]
                          for method in order_with_NT}
        no_treatment_df = Artifact.load(artifact_paths['NT']).view(pd.DataFrame)
        batches = control_case_columns_data['batch'].unique()
        no_treatment_batch_1, no_treatment_batch_2, no_treatment_pooled = get_diff_abundant_otu_sets_for_venns(
            no_treatment_df, control_case_columns_data, batches[0], batches[1])
        results_df = pd.DataFrame(columns=['Category', 'Method', 'Value'], dtype=int)
        results_index = 0
        for method_index in range(len(order)):
            method = order[method_index]
            method_name = method_dict[method]
            treatment_df = Artifact.load(artifact_paths[method]).view(pd.DataFrame)
            treatment_batch_1, treatment_batch_2, treatment_pooled = get_diff_abundant_otu_sets_for_venns(treatment_df,
                                                                                                          control_case_columns_data,
                                                                                                          batches[0],
                                                                                                          batches[1])
            results_df.loc[results_index] = [batches[0], method_name, len(treatment_batch_1)]
            results_df.loc[results_index + 1] = [batches[1], method_name, len(treatment_batch_2)]
            results_df.loc[results_index + 2] = ['Pooled', method_name, len(treatment_pooled)]
            results_df.loc[results_index + 3] = [batches[0] + ' trt+no trt', method_name,
                                                 len(set(no_treatment_batch_1) & set(treatment_batch_1))]
            results_df.loc[results_index + 4] = [batches[1] + ' trt+no trt', method_name,
                                                 len(set(no_treatment_batch_2) & set(treatment_batch_2))]
            results_df.loc[results_index + 5] = ['Pooled trt+no trt', method_name,
                                                 len(set(no_treatment_pooled) & set(treatment_pooled))]
            results_index += 6

        sns.barplot(y=['b1', 'b2', 'p'],
                    x=[len(no_treatment_batch_1), len(no_treatment_batch_2), len(no_treatment_pooled)], orient='h',
                    ax=axes[0, subdir_index])

        sns.barplot(data=results_df, y='Method', x='Value', orient='h', hue='Category', ax=axes[1, subdir_index])
        print(axes[0, subdir_index].get_xlim()[1], axes[1, subdir_index].get_xlim()[1])
        xmax = max(axes[0, subdir_index].get_xlim()[1], axes[1, subdir_index].get_xlim()[1])
        print(xmax)
        axes[0, subdir_index].set_xlim(0, xmax)
        axes[1, subdir_index].set_xlim(0, xmax)
        if subdir_index > 0:
            axes[1, subdir_index].yaxis.set_visible(False)
        else:
            axes[1, subdir_index].set(ylabel=None)
        axes[1, subdir_index].get_legend().remove()
        # sns.barplot(data=results_df, y='Method', hue='Category', orient='h', ax=axes[1, subdir_index])
    plt.savefig(out_path + 'DAF_barplot.png', dpi=200, bbox_inches='tight')
    plt.close('all')


def apply_correction_methods(artifact: Artifact, artifact_name: str, control_case_columns_data: pd.DataFrame,
                             basis: str, reduced_tree: skbio.TreeNode, out_path: str, process_id=None) -> dict:
    """
    The main function for executing the various batch correction methods.
    To add a new method to the pipeline it must be implemented, be added as an abbreviation to the configuration file,
    added to this function's body, and have the abbreviation:full name pair added to the method_dict.
    :param artifact: QIIME2 artifact - the original relative abundance table.
    :param artifact_name: name str used to save corrected versions of the table
    :param control_case_columns_data: the summarized metadata with 'batch' column for batch and 'set' column for main variable
    :param basis: name of the batch used as a basis for batch correction when applicable
    :param reduced_tree: QIIME artifact - a phylogeny containing all thefeatures in the abundance table and only then.
    :param out_path: where to save the batch corrected tables
    :param process_id: name of the process that is generating the corrected tables (used by simulations)
    :returns: a dictionary with the path to each saved table
    """
    if not process_id:
        process_id = ''
    with lock:
        with open('log2.txt', 'a+') as f:
            f.write(str(process_id) + ' starting batch correction methods\n')
    artifact_paths = {}
    no_treatment_path = '{}{}{}_{}'.format(out_path, process_id, 'NT', artifact_name[artifact_name.rfind('/') + 1:])
    artifact.save(no_treatment_path)
    artifact_paths['NT'] = no_treatment_path
    print('Performing batch-based data manipulation methods')
    pseudocount_artifact = utils.pseudocount(utils.feature_table_relative_abundance(artifact))
    no_treatment_pc_path = '{}{}{}_{}'.format(out_path, process_id, 'NTP', artifact_name[artifact_name.rfind('/') + 1:])
    pseudocount_artifact.save(no_treatment_pc_path)
    if 'TMS' in methods:
        with lock:
            with open('log2.txt', 'a+') as f:
                f.write(str(process_id) + ' TMS\n')
        tms_artifact = correction_methods.direct_abundance_shift(pseudocount_artifact, control_case_columns_data, basis)
        artifact_paths['TMS'] = tms_artifact.save('{}{}{}_{}'.format(out_path, process_id, 'TMS', artifact_name))
        assert utils.validate_relative_abundance_dataframe(tms_artifact.view(pd.DataFrame))
        del tms_artifact
    if 'LRT' in methods:
        with lock:
            with open('log2.txt', 'a+') as f:
                f.write(str(process_id) + ' LRT\n')
        lrt_artifact = correction_methods.log_ratio_tree(pseudocount_artifact, reduced_tree, basis,
                                                         control_case_columns_data)
        artifact_paths['LRT'] = lrt_artifact.save('{}{}{}_{}'.format(out_path, process_id, 'LRT', artifact_name))
        del lrt_artifact
    if 'CB' in methods:
        with lock:
            with open('log2.txt', 'a+') as f:
                f.write(str(process_id) + ' CB\n')
        combat_artifact = correction_methods.combat(pseudocount_artifact, control_case_columns_data)
        artifact_paths['CB'] = combat_artifact.save('{}{}{}_{}'.format(out_path, process_id, 'CB', artifact_name))
        del combat_artifact
    if 'MU' in methods:
        with lock:
            with open('log2.txt', 'a+') as f:
                f.write(str(process_id) + ' MU\n')
        mu_artifact = correction_methods.adjust_batch(artifact, control_case_columns_data, process_id)
        artifact_paths['MU'] = mu_artifact.save('{}{}{}_{}'.format(out_path, process_id, 'MU', artifact_name))
        del mu_artifact
    if 'PN' in methods:
        pn_artifact = correction_methods.percentile_normalize_artifact(pseudocount_artifact, control_case_columns_data)
        artifact_paths['PN'] = pn_artifact.save('{}{}{}_{}'.format(out_path, process_id, 'PN', artifact_name))
        del pn_artifact
    if 'LS' in methods:
        with lock:
            with open('log2.txt', 'a+') as f:
                f.write(str(process_id) + ' LS\n')
        ls_artifact = correction_methods.plsda_batch(pseudocount_artifact, control_case_columns_data, process_id)
        artifact_paths['LS'] = ls_artifact.save('{}{}{}_{}'.format(out_path, process_id, 'LS', artifact_name))
        del ls_artifact
    if 'PC' in methods:
        with lock:
            with open('log2.txt', 'a+') as f:
                f.write(str(process_id) + ' PC\n')
        pc_artifact = correction_methods.principal_component_shift(pseudocount_artifact, control_case_columns_data,
                                                                   basis)
        artifact_paths['PC'] = pc_artifact.save('{}{}{}_{}'.format(out_path, process_id, 'PC', artifact_name))
        del pc_artifact
    if 'DCC' in methods:
        with lock:
            with open('log2.txt', 'a+') as f:
                f.write(str(process_id) + ' DCC\n')
        dcc_artifact = correction_methods.direct_covariate_correction(
            pseudocount_artifact, control_case_columns_data, process_id)
        artifact_paths['DCC'] = dcc_artifact.save('{}{}{}_{}'.format(out_path, process_id, 'DCC', artifact_name))
        del dcc_artifact
    if 'CQ' in methods:
        with lock:
            with open('log2.txt', 'a+') as f:
                f.write(str(process_id) + ' CQ\n')
        cq_artifact = correction_methods.conqur(artifact, control_case_columns_data, basis, process_id)
        # cq_artifact = correction_methods.direct_covariate_correction(
        #    pseudocount_artifact, control_case_columns_data, process_id)
        # cq_artifact = correction_methods.combat_seq(pseudocount_artifact, control_case_columns_data)
    if True:
        artifact_paths['CQ'] = cq_artifact.save('{}{}{}_{}'.format(out_path, process_id, 'CQ', artifact_name))
        del cq_artifact
    return artifact_paths


def no_treatment_diff_abundant(artifact, control_case_columns_data):
    df = artifact.view(pd.DataFrame)
    return get_diff_abundant_otus(df, control_case_columns_data,
                                  [control_case_columns_data.at[i, 'batch'] for i in df.index])


def bdmma(artifact: Artifact, control_case_columns_data: pd.DataFrame):
    batches = control_case_columns_data['batch'].unique()
    df = artifact.view(pd.DataFrame)
    md = pd.DataFrame(index=df.index)
    # con = [i for i in df.index if control_case_columns_data.at[i,'set'] == 'control']

    # cases = [i for i in df.index if control_case_columns_data.at[i,'set'] == 'case']
    # c1 = [i for i in con if control_case_columns_data.at[i,'batch'] == batches[0]]
    # c2 = [i for i in con if control_case_columns_data.at[i,'batch'] == batches[1]]
    # cases_1 = [i for i in cases if control_case_columns_data.at[i,'batch'] == batches[0]]
    # cases_2 = [i for i in cases if control_case_columns_data.at[i,'batch'] == batches[1]]
    # df.loc[c1,:] = np.random.normal(10,1,(len(c1),64))
    # df.loc[c2,:] = np.random.normal(20,1,(len(c2),64))
    # df.loc[cases_1,:] = np.random.normal(10,1,(len(cases_1),64))
    # df.loc[cases_2,:] = np.random.normal(30,1,(len(cases_2),64))

    md['main'] = [0 if control_case_columns_data.at[i, 'set'] == 'control' else 1 for i in df.index]
    md['confounder'] = [np.random.normal(0, 1) for _ in range(len(df.index))]
    md['batch'] = [0 if control_case_columns_data.at[i, 'batch'] == batches[0] else 1 for i in df.index]
    df.T.to_csv('tmp/bdmma_df_input.tsv', sep='\t')
    md.to_csv('tmp/bdmma_md_input.tsv', sep='\t')
    exit()
    subprocess.call(['Rscript', 'app/bdmma.r'])


def differentially_abundant_methods(artifact, control_case_columns_data, methods):
    results_dict = {}
    significant_otus = no_treatment_diff_abundant(artifact, control_case_columns_data)
    results_dict['NT'] = significant_otus
    if 'BDMMA' in methods:
        significant_otus = bdmma(artifact, control_case_columns_data)
        results_dict['BDMMA'] = significant_otus
    if 'PN' in methods:
        significant_otus = percnorm_diff_abundant(artifact, control_case_columns_data)
        results_dict['PN'] = significant_otus
    if 'NM' in methods:
        significant_otus = netmoss2(artifact, control_case_columns_data)
        results_dict['NM'] = significant_otus
    return results_dict


def permanova_pairs(dataset_metadata: str, control_case_columns_data='control_case_columns_data.txt',
                    metadata='data/mhd/merged_md.txt',
                    samples_per_group=30, subsamplings=10000, permutations=10000):
    dataset_metadata = pd.read_csv(dataset_metadata, sep='\t', index_col=0)
    #artifact_paths = dict(zip(dataset_metadata.index, dataset_metadata.path))
    artifact = Artifact.load('data/mhd/filtered_full_merged_table_rf.qza')

    df = artifact.view(pd.DataFrame)
    relevant_ids = list(df.index.values)
    control_case_columns_data: pd.DataFrame = utils.get_metadata_control_and_case_sets(
        metadata, control_case_columns_data, relevant_ids)

    artifact = utils.remove_renegade_samples(artifact, control_case_columns_data, True)
    distance_matrix = get_bray_curtis(artifact).view(DistanceMatrix)
    ids = utils.get_sample_ids(control_case_columns_data)
    print(dataset_metadata, list(ids.keys()))
    # filter out datasets from the metadata if they don't have samples in the feature table
    #datasets = [x for x in dataset_metadata.index.tolist() if dataset_metadata.at[x, 'cohort_name'] in list(ids.keys())]
    datasets = dataset_metadata.index.tolist()
    dataset_formal_names = dataset_metadata['cohort_name'].tolist()
    # ids is a dict that maps from the cohort name to its samples. this command changes the cohort name format
    # e.g. asd_son -> Son, crc_zeller -> Zeller
    controls = {dataset_metadata.at[dataset, 'cohort_name']:ids[dataset]['control'] for dataset in datasets}
   # controls = {dataset: ids[dataset_metadata.at[dataset, 'cohort_name']]['control'] for dataset in datasets}
    # make sure every dataset has enough samples to subsample
    assert all([len(v) >= samples_per_group for v in controls.values()])

    dataset_pairs = []
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            dataset_pairs.append((datasets[i], datasets[j]))
    index = pd.MultiIndex.from_tuples(dataset_pairs, names=['first', 'second'])
  #  results = pd.DataFrame(index=index, columns=['mean', 'interval'])
    heatmap_df = pd.DataFrame(index=dataset_formal_names, columns=dataset_formal_names, dtype=float)
    grouping = ['c1'] * samples_per_group + ['c2'] * samples_per_group
    annotations = np.array(np.zeros((len(dataset_formal_names), len(dataset_formal_names))), dtype='<U25')
    distributions = {}
    nulls = {}
    for i, dataset_1 in enumerate(dataset_formal_names):
        c1_samples = controls[dataset_1]
        c1_sample_size = len(c1_samples)
        for j in range(i + 1, len(datasets)):
            print(i,j)
            dataset_2 = dataset_formal_names[j]
            c2_samples = controls[dataset_2]
            c2_sample_size = len(c2_samples)
            curr_results = []

            for _ in range(subsamplings):
                random.shuffle(c1_samples)
                random.shuffle(c2_samples)
                downsampled_c1 = c1_samples[:samples_per_group]
                downsampled_c2 = c2_samples[:samples_per_group]
                curr_dmx = distance_matrix.filter(downsampled_c1 + downsampled_c2)
                f = list(permanova(curr_dmx, grouping, permutations=0))[4]
                curr_results.append(f)
            # curr_null = []
            # for _ in range(permutations):
            #     random.shuffle(c1_samples)
            #     random.shuffle(c2_samples)
            #     downsampled_c1 = c1_samples[:samples_per_group]
            #     downsampled_c2 = c2_samples[:samples_per_group]
            #     curr_all_samples = downsampled_c1 + downsampled_c2
            #     random.shuffle(curr_all_samples)
            #     curr_dmx = distance_matrix.filter(curr_all_samples)
            #     f = list(permanova(curr_dmx, grouping, permutations=0))[4]
            #     curr_null.append(f)


            l, h = np.percentile(curr_results, 2.5), np.percentile(curr_results, 97.5)
            mean = np.mean(curr_results)
            distributions[round(mean,4)] = curr_results
         #   nulls[round(mean,4)] = curr_null
            try:
                assert not np.isnan(mean)
            except AssertionError:
                raise AssertionError(dataset_1 + dataset_2)
            interval = np.max([np.abs(l - mean), np.abs(h - mean)])
          #  results.loc[(dataset_1, dataset_2), ['mean', 'interval']] = mean, interval
            mean_bold = r"$\bf{" + "{}".format(round(mean, 2)) + "}$"
            annot = '({},{})\n{}'.format(c1_sample_size, c2_sample_size, mean_bold)
            annot += u'\u00B1' + str(round(interval, 2))
            annotations[j, i] = annot
            heatmap_df.at[dataset_2, dataset_1] = mean

    mask = np.triu(np.ones_like(heatmap_df, dtype=bool))

    cmap = sns.diverging_palette(250, 15, s=75, l=40,
                                 n=9, center="light", as_cmap=True)

    
    #num_datasets = len(dataset_formal_names)
   # vals = list(heatmap_df.values.flatten()[~mask.flatten()])
    #vals = sorted(vals)
    # ranks = [int(r) for r in ss.rankdata(vals)]
    # print(vals)
    # print(ranks)
    # num_colors = len(vals)
    # arr = heatmap_df.values
    # arr[np.where(~mask)] = ranks
    # heatmap_df = pd.DataFrame(arr.astype(int), dtype=int)
    # colors = vals_to_rgb(vals)
    #num_colors = len(vals)
    #cmap = LinearSegmentedColormap.from_list('', list(zip(vals,vals_to_rgb(vals))), num_colors)
    sns.set(font_scale=0.6)

    # ax = sns.heatmap(heatmap_df, cmap=cmap, mask=mask, annot=annotations,
    #     center=0, fmt='', vmin=0, vmax=max(vals), cbar=False)
    ax = sns.heatmap(heatmap_df, mask=mask, annot=annotations, center=0, fmt='',
                cmap=cmap, cbar=False)
    plt.xticks(fontsize=8,ha='right',rotation=30)
    plt.yticks(fontsize=8)
    plt.savefig('heatmap.png', dpi=300, bbox_inches="tight")
    plt.close('all')

    #Create null distribution by comparing random samples
    null_f_stats = []
    all_samples_to_draw_from = []
    for samples in controls.values():
        all_samples_to_draw_from += samples
    np.random.seed(30)
    for i in range(permutations):
        if i % 10 == 0:
            print('starting permutation {}'.format(str(i+1)))
        random.shuffle(all_samples_to_draw_from)
        random_sample = all_samples_to_draw_from[:60]
        f = list(permanova(distance_matrix.filter(random_sample), grouping, permutations=0))[4]
        null_f_stats.append(f)
    sns.histplot(null_f_stats,color='skyblue',ec="skyblue")
    plt.scatter(x=[np.mean(null_f_stats)],y=[50],s=200,color='skyblue',edgecolor='black',alpha=0.66,linewidth=3,zorder=1)
  #  distribution_means = list(nulls.keys())
    distribution_means = list(distributions.keys())
    colors = vals_to_rgba(distribution_means, alpha=0.05)
    colors_solid = vals_to_rgba(distribution_means)
    max_mean = max(distribution_means)
    for i,mean in enumerate(distribution_means):
     #   distribution = nulls[mean]
        distribution = distributions[mean]
        sns.histplot(distribution, color=colors[i],ec=colors[i],zorder=0)
        plt.scatter(x=[mean],y=[50],s=200,color=colors_solid[i],edgecolor='black',alpha=0.66,linewidth=3,zorder=1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('F',fontsize=16)
    plt.ylabel('Counts',fontsize=16)
    plt.savefig('null_f_dist.png',dpi=300)
    plt.close('all')


def downsample_aggregate_table_and_drop_samples_missing_from_metadata(
        path, control_case_columns_data, metadata, out):
    artifact = Artifact.load(path)
    df = artifact.view(pd.DataFrame)
    relevant_ids = list(df.index.values)
    control_case_columns_data_df: pd.DataFrame = \
        utils.get_metadata_control_and_case_sets(metadata, relevant_ids)
    ids = utils.get_sample_ids(control_case_columns_data_df)
    for cohort in ids.keys():
        print(cohort)
        print(ids[cohort]['control'])
        controls = ids[cohort]['control']
        assert len(controls) >= 30
        random.shuffle(controls)
        toss = controls[30:] if len(controls) > 30 else []
        df = df.drop(index=toss)
        df = df.drop(ids[cohort]['case'])

    df = df.drop(set(df.index) - set(control_case_columns_data_df.index))
    art = Artifact.import_data('FeatureTable[Frequency]', df)
    art.save(out)


def is_prevalent_table(df, ids, min_prevalence=0.1):
    cohorts = [c for c in ids.keys()]
    output = pd.DataFrame(index=cohorts, columns=df.columns, dtype=bool)
    for cohort in cohorts:
        samples = ids[cohort]['control'] + ids[cohort]['case']
        num_samples = len(samples)
        cohort_df = df.loc[samples, :]
        for feature in df:
            feature_abundance = cohort_df.loc[:, feature]
            appearances = len(feature_abundance[feature_abundance > 0])
            result = appearances / num_samples >= min_prevalence
            output.at[cohort, feature] = result
    return output


def fdr_correct_dataframe(df):
    return pd.DataFrame(index=df.index, columns=df.columns, data=fdr(df.values.flatten())[1].reshape(df.values.shape))


def negative_log_dataframe(df):
    return pd.DataFrame(index=df.index, columns=df.columns, data=-1 * np.log(df.values))


def plot_percentile_ci(low, high, x_pos, color, ax):
    ax.plot([x_pos - 0.25, x_pos + 0.25], [low, low], color=color)
    ax.plot([x_pos - 0.25, x_pos + 0.25], [high, high], color=color)
    handle = ax.plot([x_pos, x_pos], [low, high], color=color)
    return handle


def taxa_batch_effect_distribution_plot(dataset_metadata: str,
                                        control_case_columns_data='control_case_columns_data.txt',
                                        metadata='data/mhd/merged_md.txt', subsamplings=10, samples_per_group=30):
    sns.set_theme()
    dataset_metadata = pd.read_csv(dataset_metadata, sep='\t', index_col=0)
    # dataset_metadata.drop(dataset_metadata[dataset_metadata.include_in_taxa_batch_effect_heatmap == False].index,
    #                       inplace=True)
    datasets = dataset_metadata.index.tolist()
    artifact = Artifact.load('data/mhd/clean_taxa_tables/full_phyla_table_rf.qza')
    df = artifact.view(pd.DataFrame)
    taxa = df.columns
    relevant_ids = list(df.index.values)
    control_case_columns_data: pd.DataFrame = \
        utils.get_metadata_control_and_case_sets(metadata, control_case_columns_data, relevant_ids)
    ids = utils.get_sample_ids(control_case_columns_data)
    # datasets = [dataset_metadata.at[x, 'cohort_name'] for x in dataset_metadata.index.tolist() if
    #             dataset_metadata.at[x, 'cohort_name'] in list(ids.keys())]
    #datasets = dataset_metadata['cohort_name'].tolist()
    prevalence_table = is_prevalent_table(df, ids)
    paired_datasets = []
    #paired_dataset_names_strings = []
    dataset_pairs_taxons = []
    for i, dataset_1 in enumerate(datasets):
        for j in range(i + 1, len(datasets)):
            dataset_2 = datasets[j]
          #  format_dataset_2 = formal_dataset_names[j]
            paired_datasets.append((dataset_1, dataset_2))
           # paired_dataset_names_strings.append((formal_dataset_1 + ' ' + format_dataset_2))
            for taxon in taxa:
                dataset_pairs_taxons.append((dataset_1, dataset_2, taxon))
    #index = pd.MultiIndex.from_tuples(dataset_pairs_taxons, names=['first', 'second', 'taxon'])
    percentiles_of_interest = [10, 25, 50, 75, 90]
    # statistics = pd.DataFrame(index=index, columns=[
    #     'perc_10_l','perc_10_h','perc_25_l', 'perc_25_h','perc_50_l','perc_50_h','perc_75_l','perc_75_h','perc_90_l','perc_90_h'
    #     ])
    raw_results = np.zeros((len(paired_datasets), len(taxa), subsamplings))

    # ids is a dict that maps from the cohort name to its samples. this command changes the cohort name format
    # e.g. asd_son -> Son, crc_zeller -> Zeller
    controls = {dataset:ids[dataset]['control'] for dataset in datasets}
    # make sure every dataset has enough samples to subsample
    assert all(len(controls[dataset]) >= samples_per_group for dataset in datasets)

    # add pseusocount for CLR
    pc_df = utils._pseudocount(df)
    del (df)

    clr_df = pd.DataFrame(index=pc_df.index, columns=pc_df.columns, data=clr(pc_df.values))
    del (pc_df)

    for i, (dataset_1, dataset_2) in enumerate(paired_datasets):
        print(i)
        c1_samples = controls[dataset_1]
        c2_samples = controls[dataset_2]
        for k in range(subsamplings):
            random.shuffle(c1_samples)
            random.shuffle(c2_samples)
            downsampled_c1 = c1_samples[:samples_per_group]
            downsampled_c2 = c2_samples[:samples_per_group]
            for l in range(len(taxa)):
                taxon = taxa[l]
                pval = ranksums(clr_df.loc[downsampled_c1, taxon], clr_df.loc[downsampled_c2, taxon])[1]
                raw_results[i, l, k] = -1 * np.log(pval)
    statistics = pd.DataFrame(index=taxa, columns=
    [str(p) + '_low' for p in percentiles_of_interest] + [str(p) + '_high' for p in percentiles_of_interest],
                              dtype=float)
    for l in range(len(taxa)):
        for interesting_percentile in percentiles_of_interest:
            values = [np.percentile(raw_results[:, l, k], interesting_percentile) for k in range(subsamplings)]
            low, high = np.percentile(values, 2.5), np.percentile(values, 97.5)
            statistics.at[taxa[l], str(interesting_percentile) + '_low'] = low
            statistics.at[taxa[l], str(interesting_percentile) + '_high'] = high
    mask = pd.DataFrame(index=range(len(paired_datasets)), columns=taxa, dtype=bool)
    for i in range(len(paired_datasets)):
        cohort_1, cohort_2 = paired_datasets[i][0], paired_datasets[i][1]
        for j in range(len(taxa)):
            taxon = taxa[j]
            mask.iat[i, j] = not (prevalence_table.at[cohort_1, taxon] or prevalence_table.at[cohort_2, taxon])

    fig, ax = plt.subplots()
    taxa_to_plot = [taxon for taxon in taxa if any([mask.at[i, taxon] == False for i in range(len(paired_datasets))])]
    handles, legend_text = [], []
    for l in range(len(taxa_to_plot)):
        for interesting_percentile, color in zip(percentiles_of_interest, ['purple', 'blue', 'green', 'orange', 'red']):
            handle = plot_percentile_ci(
                statistics.at[taxa_to_plot[l], str(interesting_percentile) + '_low'],
                statistics.at[taxa_to_plot[l], str(interesting_percentile) + '_high'],
                l + 1,
                color,
                ax)
    for interesting_percentile, color in zip(percentiles_of_interest, ['purple', 'blue', 'green', 'orange', 'red']):
        handles.append(Line2D([0], [0], color=color, lw=4))
        legend_text.append('{}th percentile'.format(interesting_percentile))
    plt.xlim(0, len(taxa_to_plot) + 1)
    ax.legend(handles, legend_text, bbox_to_anchor=(1.3, 1))
    plt.xticks(range(1, len(taxa_to_plot) + 1), labels=np.array([taxon.split('_')[-1] for taxon in taxa_to_plot]))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=12)
    with plt.style.context('seaborn-darkgrid'):
        plt.savefig('taxa_ci.png', dpi=300, bbox_inches='tight')
    plt.close('all')

    heatmap_df = pd.DataFrame(index=range(len(paired_datasets)),
                              columns=np.array([taxon.split('_')[-1] for taxon in taxa]), dtype=float)
    for i in range(len(paired_datasets)):
        for l in range(len(taxa)):
            heatmap_df.iat[i, l] = np.mean(raw_results[i, l, :])

    #extreme_value_thresh = np.quantile(heatmap_df.values, 1)
    # Create a custom divergin palette
    cmap = sns.diverging_palette(250, 15, s=75, l=40,
                                 n=9, center="light", as_cmap=True)
    cmap = sns.color_palette("magma_r", as_cmap=True)
    cmap = sns.color_palette("blend:#7AB,#EDA", as_cmap=True)
    cmap = sns.blend_palette(['#faf9e8','#FF0000'], 2, as_cmap=True)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    fig, ax = plt.subplots()
    print(heatmap_df.values.shape, mask.values.shape)
    mask = pd.DataFrame(index=heatmap_df.index, columns=heatmap_df.columns, dtype=bool)
    for i in range(len(mask.index)):
        pair = paired_datasets[i]
        cohort_1 = pair[0]
        cohort_2 = pair[1]
        for j in range(len(taxa)):
            taxon = taxa[j]
            if prevalence_table.at[cohort_1, taxon] or prevalence_table.at[cohort_2, taxon]:
                mask.iat[i, j] = False
            else:
                mask.iat[i, j] = True
    sns.heatmap(heatmap_df,
                mask=mask,
                annot=False, cmap=cmap, yticklabels=False,
                vmin=0, vmax=heatmap_df.to_numpy().max())
    ax.set_facecolor("white")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    # ax.set_title('Differential CLR-transformed phyla\nrelative abundance across pairs of datasets')
    plt.tight_layout()
    plt.savefig('taxa_heatmap.png', dpi=200)
    plt.close('all')

    # Bacteroidetes only

    artifact = Artifact.load('data/mhd/clean_taxa_tables/full_taxa_table_rf.qza')
    df = artifact.view(pd.DataFrame)
    prevalence_table = is_prevalent_table(df, ids)
    pc_df = utils._pseudocount(df)
    del (df)
    clr_df = pd.DataFrame(index=pc_df.index, columns=pc_df.columns, data=clr(pc_df.values))
    del (pc_df)
    clr_df = clr_df[[taxon for taxon in clr_df.columns if 'Bacteroidetes' in taxon]]
    taxa = clr_df.columns
    raw_results = np.zeros((len(paired_datasets), len(taxa), subsamplings))
    for i in range(len(paired_datasets)):
        pair = paired_datasets[i]
        dataset_1 = pair[0]
        c1_samples = controls[dataset_1]
      #  c1_sample_size = len(c1_samples)
        dataset_2 = pair[1]
        c2_samples = controls[dataset_2]
      #  c2_sample_size = len(c2_samples)
        for k in range(subsamplings):
            random.shuffle(c1_samples)
            random.shuffle(c2_samples)
            downsampled_c1 = c1_samples[:samples_per_group]
            downsampled_c2 = c2_samples[:samples_per_group]
            for l in range(len(taxa)):
                taxon = taxa[l]
                pval = ranksums(clr_df.loc[downsampled_c1, taxon], clr_df.loc[downsampled_c2, taxon])[1]
                raw_results[i, l, k] = -1 * np.log(pval)

    statistics = pd.DataFrame(index=taxa, columns=
    [str(p) + '_low' for p in percentiles_of_interest] + [str(p) + '_high' for p in percentiles_of_interest],
                              dtype=float)
    for l in range(len(taxa)):
        for interesting_percentile in percentiles_of_interest:
            values = [np.percentile(raw_results[:, l, k], interesting_percentile) for k in range(subsamplings)]
            low, high = np.percentile(values, 2.5), np.percentile(values, 97.5)
            statistics.at[taxa[l], str(interesting_percentile) + '_low'] = low
            statistics.at[taxa[l], str(interesting_percentile) + '_high'] = high

    mask = pd.DataFrame(index=range(len(paired_datasets)), columns=taxa, dtype=bool)
    for i in range(len(mask.index)):
        pair = paired_datasets[i]
        cohort_1 = pair[0]
        cohort_2 = pair[1]
        for j in range(len(taxa)):
            taxon = taxa[j]
            if prevalence_table.at[cohort_1, taxon] or prevalence_table.at[cohort_2, taxon]:
                mask.iat[i, j] = False
            else:
                mask.iat[i, j] = True
    fig, ax = plt.subplots()
    taxa_to_plot = [taxa[j] for j in range(len(taxa)) if
                    any([mask.iat[i, j] == False for i in range(len(paired_datasets))])]
    for l in range(len(taxa_to_plot)):
        for interesting_percentile, color in zip(percentiles_of_interest, ['purple', 'blue', 'green', 'orange', 'red']):
            plot_percentile_ci(
                statistics.at[taxa_to_plot[l], str(interesting_percentile) + '_low'],
                statistics.at[taxa_to_plot[l], str(interesting_percentile) + '_high'],
                l + 1,
                color,
                ax)
    plt.xlim(0, len(taxa_to_plot) + 1)
    plt.xticks(range(1, len(taxa_to_plot) + 1), labels=np.array([taxon.split('_')[-1] for taxon in taxa_to_plot]))
    ax.legend(handles, legend_text, bbox_to_anchor=(1.3, 1))

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=12)
    with plt.style.context('seaborn-darkgrid'):
        plt.savefig('bacteroidetes_ci.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    # boxplot_data = [[output.iat[i,j] for i in range(len(paired_datasets)) if not mask.iat[i,j]] for j in range(len(output.columns))]
    # print(boxplot_data)
    # x_labels = np.array([output.columns[i].split('_')[-1] for i in range(len(boxplot_data)) if boxplot_data[i]])
    # boxplot_data = [np.array(dat) for dat in boxplot_data if dat]
    # num_valid_pairs = ['n = {}'.format(len(obs)) for obs in boxplot_data]

    # fig, ax = plt.subplots(figsize=(10,10))
    # sns.boxplot(data=boxplot_data)
    # medians = [np.median(dat) for dat in boxplot_data]
    # for tick in range(len(x_labels)):
    #     ax.text(tick,
    #             medians[tick] + 0.06,
    #             num_valid_pairs[tick],
    #             horizontalalignment='center',
    #             color='black',
    #             weight='bold')
    # ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right',fontsize=12)
    # plt.tight_layout()
    # ax.set_title('Differential CLR-transformed phyla\nrelative abundance across pairs of datasets',fontsize=25)
    # plt.savefig('taxa_boxplot.png', dpi=100, bbox_inches="tight")
    # plt.close('all')
    # for i in range(len(paired_datasets)):
    #     pair = paired_datasets[i]
    #     pair_str = paired_dataset_names_strings[i]
    #     cohort_1, cohort_2 = pair[0], pair[1]
    #     c1_samples, c2_samples = ids[cohort_1]['control'], ids[cohort_2]['control']
    #     curr_pvals = [ranksums(clr_df.loc[c1_samples, taxon], clr_df.loc[c2_samples, taxon])[1] for  taxon in clr_df]
    #     output.loc[pair_str,:] = curr_pvals
    # output = negative_log_dataframe(fdr_correct_dataframe(output))

    # boxplot_data = [[output.iat[i,j] for i in range(len(paired_datasets)) if not mask.iat[i,j]] for j in range(len(output.columns))]
    # print(boxplot_data)
    # x_labels = np.array([output.columns[i].split('_')[-1] for i in range(len(boxplot_data)) if boxplot_data[i]])
    # boxplot_data = [np.array(dat) for dat in boxplot_data if dat]
    # num_valid_pairs = ['n = {}'.format(len(obs)) for obs in boxplot_data]

    # fig, ax = plt.subplots(figsize=(10,10))
    # sns.boxplot(data=boxplot_data)
    # medians = [np.median(dat) for dat in boxplot_data]
    # for tick in range(len(x_labels)):
    #     ax.text(tick,
    #             medians[tick] + 0.06,
    #             num_valid_pairs[tick],
    #             horizontalalignment='center',
    #             color='black',
    #             weight='bold')
    # ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right',fontsize=12)
    # plt.tight_layout()
    # ax.set_title('Differential CLR-transformed phyla\nrelative abundance across pairs of datasets',fontsize=25)
    # plt.savefig('taxa_boxplot.png', dpi=100, bbox_inches="tight")
    # plt.close('all')

    # output = pd.DataFrame(index=paired_dataset_names_strings, columns=taxa, data=np.zeros((len(paired_dataset_names_strings),len(taxa))))
    # extreme_value_thresh = np.quantile(output.values, 0.9)
    # output = output.applymap(lambda x: x if x < extreme_value_thresh else extreme_value_thresh)
    # # Create a custom divergin palette
    # cmap = sns.diverging_palette(250, 15, s=75, l=40,
    #                             n=9, center="light", as_cmap=True)
    # ratio = len(output.columns)/len(output.index)
    # fig, ax = plt.subplots()
    # sns.heatmap(output, mask=mask, annot=False, cmap=cmap, center=(-1*math.log(0.05)), xticklabels=[x.split('_')[-1] for x in output.columns], yticklabels=False)
    # ax.set_facecolor("black")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    # ax.set_title('Differential CLR-transformed phyla\nrelative abundance across pairs of datasets')
    # plt.tight_layout()
    # output.to_csv('taxa_batch_effect_result.tsv',sep='\t')
    # plt.savefig('taxa_heatmap.png', dpi=100)
    # plt.close('all')


def pc_alignment_graphs(artifacts: dict, control_case_columns_data: pd.DataFrame, basis: str, dest: str,
                        show_cases: bool):
    cohorts = control_case_columns_data['batch'].unique().tolist()
    cohorts.remove(basis)
    cohorts = [basis] + cohorts
    methods = order_with_NT
    pca = PCA(n_components=3)
    ids = utils.get_sample_ids(control_case_columns_data)
    labels = ['PC' + str(i) for i in range(1, 4)]
    if len(methods) > 1:
        jitter_loc = [-0.25 + i * 0.5 / (len(methods) - 1) for i in range(len(methods) + 1)]
    else:
        jitter_loc = [0]
    pca.fit(Artifact.load(artifacts['NT']).view(pd.DataFrame))
    for func_name_pair in [(np.mean, 'mean'), (np.std, 'standard deviation')]:
        legend_elements = []
        plt.figure(figsize=(7, len(methods) * 1.2))
        func, name = func_name_pair[0], func_name_pair[1]
        for i in range(len(methods)):
            method = methods[i]
            df = Artifact.load(artifacts[method]).view(pd.DataFrame)
            method = method_dict[method]
            pc_df = pd.DataFrame(index=df.index, data=pca.transform(df))
            method_color = plt_color_cycle[i]
            for k in range(len(cohorts)):
                batch_shape = plt_shape_cycle[k]
                batch = cohorts[k]
                legend_elements.append(
                    Line2D([0], [0], marker=batch_shape, color=method_color,
                           label=method + ' ' + batch + ' controls {}'.format(name),
                           markerfacecolor=method_color, markersize=15))
                for j in range(len(labels)):
                    data = pc_df.loc[ids[batch]['control'], j]
                    plt.scatter(jitter_loc[i] + j, func(data), color=method_color, alpha=0.5, marker=batch_shape, s=150)
                if show_cases:
                    legend_elements.append(
                        Line2D([0], [0], marker=batch_shape, color=method_color,
                               label=method + ' ' + batch + ' cases {}'.format(name),
                               markerfacecolor=darker(method_color), markersize=15))
                    for j in range(len(labels)):
                        data = pc_df.loc[ids[batch]['case'], j]
                        plt.scatter(jitter_loc[i] + j, func(data), color=darker(method_color), alpha=0.5,
                                    marker=batch_shape, s=150)
        plt.xticks(range(len(labels)), labels)
        for i in range(0, len(labels) - 1):
            plt.axvline(i + 0.5)
        plt.xlim(-0.5, len(labels) - 0.5)
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1))
        plt.savefig(dest + 'align_{}_{}.png'.format(name, show_cases), bbox_inches='tight', dpi=100)
        plt.close('all')


def coeff(X, Y):
    return cov(X, Y) / np.sqrt(np.std(X) * np.std(Y))


def cov(X, Y):
    assert len(X) == len(Y)
    multiplications = []
    mu_x = np.mean(X)
    mu_y = np.mean(Y)
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        multi = (x - mu_x) * (y - mu_y)
        multiplications.append(multi)
    multi_sum = sum(multiplications)
    coeff_of_variance = multi_sum / len(X) - 1
    return coeff_of_variance


def corr_score(matrix_a, matrix_b):
    '''
    COL-wise
    :param matrix_a:
    :param matrix_b:
    :return:
    '''
    assert matrix_a.shape == matrix_b.shape
    return sum([np.corrcoef(matrix_a[:, k], matrix_b[:, k])[0, 1] for k in range(matrix_a.shape[1])]) / matrix_a.shape[
        1]


def standardize_matrix(matrix):
    matrix -= np.mean(matrix, 0)
    norm1 = np.linalg.norm(matrix)
    matrix /= norm1
    return matrix


def get_distance_matrices(artifact_paths, phylogeny, out_path):
    distance_matrices = {}
    table = artifact_paths['NT']
    for k, v in artifact_paths.items():
        v = Artifact.load(v)
        print('starting', k, 'distance matrix')
        dmx = analyses.get_distance_matrix(v, phylogeny)
        distance_matrices[k] = dmx.save('{}{}_DMX_{}'.format(out_path, k, table[table.rfind('/') + 1:]))
    return distance_matrices


def procrustes_and_cca_analysis(artifacts, control_case_columns_data, basis, dest, subdir: str, ma_path: str):
    # try:
    #     ma_results = load_pickle(ma_path+'proc.pkl')
    # except Exception:
    #     ma_results = {}
    # ma_results[subdir] = {}
    sample_ids = utils.get_sample_ids(control_case_columns_data=control_case_columns_data)
    cohorts = list(sample_ids.keys())
    cohorts.remove(basis)
    print('doing proc and var')
    pca = PCA(2)
    for cohort in cohorts:
        controls = sample_ids[cohort]['control']
        cases = sample_ids[cohort]['case']
        all_samples = controls + cases
        no_treatment = Artifact.load(artifacts['NT']).view(pd.DataFrame)
        before_controls = no_treatment.loc[controls, :].values
        before_controls = standardize_matrix(before_controls)
        before_cases = no_treatment.loc[cases, :].values
        before_cases = standardize_matrix(before_cases)
        before_all = no_treatment.loc[all_samples, :].values
        before_all = standardize_matrix(before_all)
        disparity = {'control': [], 'case': [], 'all': []}
        correlation_score = {'control': [], 'case': [], 'all': []}
        x = []
        labels = []

        for i in range(len(order)):
            x.append(i)
            method = order[i]
            method_name = method_dict[method]
            artifact = Artifact.load(artifacts[method]).view(pd.DataFrame)
            after_controls = artifact.loc[controls, :].values
            after_cases = artifact.loc[cases, :].values
            after_all = artifact.loc[all_samples, :].values
            after_controls = standardize_matrix(after_controls)
            after_cases = standardize_matrix(after_cases)
            after_all = standardize_matrix(after_all)
            matrix_1, matrix_2, d = procrustes(before_controls, after_controls)
            disparity['control'].append(d)
            pca.fit(before_controls)
            pca_controls_before_proc = pca.transform(before_controls)
            pca_controls_after_proc = pca.transform(matrix_2)
            matrix_1, matrix_2, d = procrustes(before_cases, after_cases)
            disparity['case'].append(d)
            matrix_1, matrix_2, d = procrustes(before_all, after_all)
            pca.fit(before_all)
            pca_before_proc_all = pca.transform(before_all)
            pca_after_proc_all = pca.transform(matrix_2)
            disparity['all'].append(d)
            fig, ax = plt.subplots(2, 1, figsize=(8, 12), gridspec_kw={'height_ratios': [3, 1]})
            sns.scatterplot(x=pca_controls_before_proc[:, 0], y=pca_controls_before_proc[:, 1],
                            c=[(200 / 255, 55 / 255, 0, 0.8) for _ in pca_controls_before_proc], s=35,
                            label='Before', ax=ax[0])
            sns.scatterplot(x=pca_controls_after_proc[:, 0], y=pca_controls_after_proc[:, 1],
                            c=[(70 / 255, 90 / 255, 250 / 255, 0.8) for _ in pca_controls_after_proc], s=35,
                            label='After', alpha=0.75, ax=ax[0])
            # ax[0].scatter(x=pca_controls_before_proc[:, 0], y=pca_controls_before_proc[:, 1], c='red', s=35,
            #             label='Before', alpha=0.75)
            # ax[0].scatter(x=pca_controls_after_proc[:, 0], y=pca_controls_after_proc[:, 1], c='blue', s=35, label='After',
            #             alpha=0.75)
            for k in range(len(pca_controls_before_proc)):
                ax[0].plot([pca_controls_before_proc[k, 0], pca_controls_after_proc[k, 0]],
                           [pca_controls_before_proc[k, 1], pca_controls_after_proc[k, 1]], color='black',
                           linewidth=0.5,
                           alpha=0.5)
            distances = np.array([euclidean(pca_controls_before_proc[k, :], pca_controls_after_proc[k, :])
                                  for k in range(len(pca_controls_before_proc))])
            kde: KernelDensity = KernelDensity(kernel='gaussian', bandwidth=0.00125)
            kde.fit(distances.reshape(-1, 1))
            linearspace = np.linspace(min(distances), max(distances), 1000)
            log_dens = kde.score_samples(linearspace.reshape(-1, 1))
            log_dens = [-700.0 if num < -700 else round(num, 9) for num in log_dens]

            ax[1].plot(linearspace, np.exp(log_dens))
            ax[1].scatter(distances, [0 for i in distances], c='black', alpha=0.2, marker='+')
            plt.suptitle('procrustes pca shift\n{} and {} in {}'.format(basis, cohort, method_name))
            ax[0].set_xlabel('PC1')
            ax[0].set_ylabel('PC2')
            ax[0].legend()
            ax[1].set_title('Distance distribution')
            ax[1].set_xlim(min(distances), max(distances))
            ax[1].set_ylim(0, max(np.exp(log_dens)) * 1.2)
            ax[1].set_xlabel('Distance')
            ax[1].yaxis.set_visible(False)
            plt.savefig(dest + 'proc_shift_controls_{}_{}_{}.png'.format(basis, cohort, method_name), dpi=200)
            plt.close('all')

            plt.scatter(x=pca_before_proc_all[:, 0], y=pca_before_proc_all[:, 1], c='red', s=35, label='Before',
                        alpha=0.75)
            plt.scatter(x=pca_after_proc_all[:, 0], y=pca_after_proc_all[:, 1], c='blue', s=35, label='After',
                        alpha=0.75)
            for k in range(len(pca_before_proc_all)):
                plt.plot([pca_before_proc_all[k, 0], pca_after_proc_all[k, 0]],
                         [pca_before_proc_all[k, 1], pca_after_proc_all[k, 1]], color='black', linewidth=0.5, alpha=0.5)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend()
            plt.savefig(dest + 'proc_shift_all_{}_{}_{}.png'.format(basis, cohort, method_name), dpi=100)
            plt.close('all')
            # X_c, Y_c = cca.fit_transform(before_controls, after_controls)
            # X_c, Y_c = X_c.T.tolist()[0], Y_c.T.tolist()[0]
            # x.append(i)
            # labels.append(method_name)
            # try:
            #     cs = corr_score(before_controls, after_controls)
            # except FloatingPointError:
            #     continue

            # correlation_score['control'].append(cs)
            # # X_c, Y_c = cca.fit_transform(before_cases, after_cases)
            # # X_c, Y_c = X_c.T.tolist()[0], Y_c.T.tolist()[0]
            # cs = corr_score(before_cases, after_cases)
            # correlation_score['case'].append(cs)
            # # X_c, Y_c = cca.fit_transform(before_all, after_all)
            # # X_c, Y_c = X_c.T.tolist()[0], Y_c.T.tolist()[0]
            # cs = corr_score(before_all, after_all)
            # correlation_score['all'].append(cs)

        # plt.figure(figsize=(1 * len(order), 4), dpi=100)
        y_max = max([max(y) for y in disparity.values()])
        plt.figure(figsize=(8, 8))
        plt.scatter(x, disparity['control'], c='blue', label='controls only', alpha=0.5)
        plt.scatter(x, disparity['case'], c='red', label='cases only', alpha=0.5)
        plt.scatter(x, disparity['all'], c='purple', label='controls and cases', alpha=0.5)
        plt.xticks(ticks=x, labels=labels, rotation=10, ha='right')
        plt.ylim(0, y_max * 1.1)
        plt.legend(bbox_to_anchor=(1.15, 1))

        # plt.legend(fontsize=25)
        plt.ylabel('sum of squares')  # ,fontsize=25)
        # plt.tick_params(axis='both', which='major', labelsize=25,width=5)
        # plt.tick_params(axis='both', which='minor', labelsize=8)
        plt.savefig(dest + 'procrustes_{}.png'.format(cohort))
        plt.close('all')

        # plt.figure(figsize=(1 * len(order), 4), dpi=100)
        # y_max = max([max(y) for y in correlation_score.values()])
        # plt.figure(figsize=(8, 8))
        # plt.scatter(x, correlation_score['control'], c='blue', label='controls only', alpha=0.5)
        # plt.scatter(x, correlation_score['case'], c='red', label='cases only', alpha=0.5)
        # plt.scatter(x, correlation_score['all'], c='purple', label='controls and cases', alpha=0.5)
        # plt.xticks(ticks=x, labels=labels, rotation=10, ha='right')
        # plt.ylim(0, y_max * 1.1)
        # plt.legend(bbox_to_anchor=(1.15, 1))
        # # plt.legend(fontsize=25)
        # plt.ylabel('average coefficient')
        # # plt.tick_params(axis='both', which='major', labelsize=25)
        # # plt.tick_params(axis='both', which='minor', labelsize=8)
        # plt.savefig(dest + 'corr_score_{}.png'.format(cohort))
        # plt.close('all')
    #     fig = plt.figure(figsize=(45, 15 * len(order)), dpi=100)
    #     fig.suptitle('CCA single component correlation', fontsize=60)
    #     gs = fig.add_gridspec(len(order), 3)
    #     axes = []
    #     titles = ['controls', 'cases', 'all samples']
    #     for i in range(len(order)):
    #         method = order[i]
    #         for j in range(len(titles)):
    #             ax = fig.add_subplot(gs[i, j])
    #             axes.append(ax)
    #             X,Y = cca_covariance[titles[j]][i]
    #             r, p = sp.stats.pearsonr(X, Y)
    #             p = '= {:.3f}'.format(p) if p >= 0.001 else '< 0.001'
    #             ax.annotate("r-squared = {:.3f}, p {}".format(r, p), (0, 1))
    #             ax.scatter(X, Y, c='red')
    #             ax.set_title('{} {}'.format(method, titles[j]), fontsize=35)
    #             ax.set_ylabel('transformation', fontsize=18)
    #             ax.set_xlabel('original', fontsize=18)
    #             # ax.set_xlim(min(x_set[j]), max_x)
    #             # ax.set_ylim(min(Y) * 0.9, max(Y) * 1.1)
    #             ax.tick_params(axis='both', which='major', labelsize=15)
    #             ax.tick_params(axis='both', which='minor', labelsize=8)
    #     plt.savefig(dest + 'cca_{}.png'.format(cohort))
    #     plt.clf()
    # exit()
    # save_pickle(ma_results, ma_path+'proc.pkl')


def rf_change(artifacts, control_case_columns_data, basis, dest):
    sample_ids = utils.get_sample_ids(control_case_columns_data=control_case_columns_data)
    cohorts = list(sample_ids.keys())
    cohorts.remove(basis)
    if not len(cohorts) == 2:
        return
    for cohort in cohorts:
        controls = sample_ids[cohort]['control']
        cases = sample_ids[cohort]['case']
        all_samples = controls + cases
        no_treatment = Artifact.load(artifacts['NT']).view(pd.DataFrame)
        before_controls = [np.average(no_treatment.loc[controls, otu].values) for otu in no_treatment]
        before_cases = [np.average(no_treatment.loc[cases, otu].values) for otu in no_treatment]
        before_all = [np.average(no_treatment.loc[all_samples, otu].values) for otu in no_treatment]
        x_set = [before_controls, before_cases, before_all]

        fig = plt.figure(figsize=(30, 10 * len(order)), dpi=100, constrained_layout=True)
        # fig.suptitle('Fold change to average relative abundance',fontsize=60)
        gs = fig.add_gridspec(len(order), 3)
        axes = []
        titles = ['controls', 'cases', 'all samples']
        # plt.title('Average relative abundance fold change')
        for i in range(len(order)):
            method = order[i]
            method_name = method_dict[method]
            artifact = Artifact.load(artifacts[method]).view(pd.DataFrame)
            after_controls = [np.average(artifact.loc[controls, otu].values) for otu in no_treatment]
            after_cases = [np.average(artifact.loc[cases, otu].values) for otu in no_treatment]
            after_all = [np.average(artifact.loc[all_samples, otu].values) for otu in no_treatment]
            y_set = [after_controls, after_cases, after_all]
            for j in range(len(y_set)):
                ax = fig.add_subplot(gs[i, j])
                axes.append(ax)
                ax.set_xscale('log')
                ax.set_yscale('log')
                max_x = max(x_set[j])
                ax.scatter(x_set[j], y_set[j], c='red')
                ax.set_title('{} {}'.format(method_name, titles[j]), fontsize=35)
                ax.set_ylabel('average after correction', fontsize=28)
                ax.set_xlabel('original average', fontsize=28)
                ax.set_xlim(min(x_set[j]), max_x)
                ax.set_ylim(min(y_set[j]) * 0.9, max(y_set[j]) * 1.1)
                ax.tick_params(axis='both', which='major', labelsize=30)
                ax.tick_params(axis='both', which='minor', labelsize=15)

        # plt.subplots_adjust(top=0.85)
        plt.savefig(dest + 'relative_abundance_average_fc_{}.png'.format(cohort))
        plt.close('all')


# def rf_fold_change(artifacts, control_case_columns_data, basis, dest):
#     sample_ids = utils.get_sample_ids(control_case_columns_data=control_case_columns_data)
#     cohorts = list(sample_ids.keys())
#     cohorts.remove(basis)
#     if not len(cohorts) == 2:
#         return
#     for cohort in cohorts:
#         controls = sample_ids[cohort]['control']
#         cases = sample_ids[cohort]['case']
#         all_samples = controls + cases
#         no_treatment = Artifact.load(artifacts['NT']).view(pd.DataFrame)
#         before_controls = [np.average(no_treatment.loc[controls,otu].values) for otu in no_treatment]
#         before_cases = [np.average(no_treatment.loc[cases,otu].values) for otu in no_treatment]
#         before_all = [np.average(no_treatment.loc[all_samples,otu].values) for otu in no_treatment]
#         x_set = [before_controls, before_cases, before_all]
#
#         fig = plt.figure(figsize=(45, 15 * len(order)), dpi=100)
#         fig.suptitle('Fold change to average relative abundance',fontsize=60)
#         gs = fig.add_gridspec(len(order), 3)
#         axes = []
#         titles = ['controls', 'cases', 'all samples']
#         #plt.title('Average relative abundance fold change')
#         for i in range(len(order)):
#             method = order[i]
#             method_name = method_dict[method]
#             artifact = Artifact.load(artifacts[method]).view(pd.DataFrame)
#             after_controls = [np.average(artifact.loc[controls, otu].values) for otu in no_treatment]
#             after_cases = [np.average(artifact.loc[cases, otu].values) for otu in no_treatment]
#             after_all = [np.average(artifact.loc[all_samples, otu].values) for otu in no_treatment]
#             y_set = [after_controls, after_cases, after_all]
#             for j in range(len(y_set)):
#                 ax = fig.add_subplot(gs[i, j])
#                 axes.append(ax)
#                 ax.set_xscale('log')
#                 ax.set_yscale('log')
#                 Y = [after/before for before, after in zip(x_set[j], y_set[j])]
#                 max_x = max(x_set[j])
#                 ax.plot([0,max_x], [1,1], linestyle='dotted')
#                 ax.scatter(x_set[j], Y, c='red')
#                 ax.set_title('{} {}'.format(method_name, titles[j]), fontsize=35)
#                 ax.set_ylabel('fold change', fontsize=18)
#                 ax.set_xlabel('original average', fontsize=18)
#                 ax.set_xlim(min(x_set[j]),max_x)
#                 ax.set_ylim(min(Y)*0.9, max(Y)*1.1)
#                 ax.tick_params(axis='both', which='major', labelsize=30)
#                 ax.tick_params(axis='both', which='minor', labelsize=15)
#
#         plt.savefig(dest + 'relative_abundance_average_fc_{}.png'.format(cohort))
#         plt.clf()

def load_pickle(path):
    with open(path, 'rb') as f:
        output = pkl.load(f)
    return output


def save_pickle(obj, path):
    with open(path, 'wb+') as f:
        pkl.dump(obj, f)


def titration(artifacts, control_case_columns_data, basis, dest, ma_path, subdir, repeats=10):
    sample_ids = utils.get_sample_ids(control_case_columns_data=control_case_columns_data)
    cohorts = list(sample_ids.keys())
    cohorts.remove(basis)
    controls_1 = sample_ids[basis]['control']
    max_p_val = 0
    axes = []
    fig = plt.figure(figsize=(12 * len(order_with_NT), 15 * len(cohorts)), dpi=100)
    fig.suptitle('Differential abundance between control sets', fontsize=60)
    gs = fig.add_gridspec(len(cohorts), len(order_with_NT))
    titration_results = {cohort: {method: {} for method in order_with_NT} for cohort in cohorts}
    for k in range(repeats):
        sys.stdout.write('titration iteration ' + str(k))
        with lock:
            with open('log2.txt', 'a+') as f:
                f.write('titr ' + str(k) + '\n')
        for j in range(len(cohorts)):
            target = cohorts[j]
            controls_2 = sample_ids[target]['control']
            set_size = min(math.floor(len(controls_1) / 2), len(controls_2))
            if set_size % 2 == 1:
                set_size -= 1
            set_size = int(set_size)
            random.shuffle(controls_1)
            random.shuffle(controls_2)
            set_1 = controls_1[:set_size]
            for i in range(len(order_with_NT)):
                method = order_with_NT[i]
                artifact = Artifact.load(artifacts[method])
                df = artifact.view(pd.DataFrame)
                # df, = round_dataframes(8, df)
                if k == 0:
                    for l in range(set_size + 1):
                        titration_results[target][method][l] = np.array([0.0 for otu in df])
                for l in range(set_size + 1):
                    set_2_part_1 = controls_1[set_size:set_size * 2 - l] if l < set_size else []
                    set_2_part_2 = controls_2[:l] if l > 0 else []
                    set_2 = set_2_part_1 + set_2_part_2
                    curr_pvals = [ranksums(df.loc[set_1, otu], df.loc[set_2, otu])[1] for otu in df]
                    curr_pvals = fdr(curr_pvals)[1]
                    curr_pvals = -1 * np.log(curr_pvals)
                    titration_results[target][method][l] += curr_pvals / repeats

    for j in range(len(cohorts)):
        target = cohorts[j]
        for i in range(len(order_with_NT)):
            method = order_with_NT[i]
            method_name = method_dict[method]
            ax = fig.add_subplot(gs[j, i])
            axes.append(ax)
            x = []
            means = []
            medians = []
            for l in range(len(titration_results[target][method])):
                curr_pvals = titration_results[target][method][l]
                ax.scatter([l for _ in curr_pvals], curr_pvals, c='black', alpha=0.07)
                # ax.scatter(l, np.mean(curr_pvals), c='blue')
                means.append(np.mean(curr_pvals))
                medians.append(np.median(curr_pvals))
                x.append(l)
                # ax.scatter(l, np.median(curr_pvals), c='red')
                max_p_val = max(max_p_val, max(curr_pvals))
            ax.plot(x, means, c='blue')
            ax.plot(x, medians, c='red')
            ax.set_title(method_name, fontsize=50)
            ax.set_xlim(0, len(titration_results[target][method]))

            ax.tick_params(axis='both', which='major', labelsize=45)
            ax.tick_params(axis='both', which='minor', labelsize=45)
    for ax in axes:
        ax.set_ylim(0, 1.2 * max_p_val)
    plt.savefig(dest + 'titration.png')
    plt.close('all')
    print(list(titration_results[list(titration_results.keys())[0]].keys()))
    if os.path.isfile(ma_path + 'titration.pkl'):
        ma_results = load_pickle(ma_path + 'titration.pkl')
        ma_results[subdir] = titration_results
        save_pickle(ma_results, ma_path + 'titration.pkl')
    else:
        save_pickle({subdir: titration_results}, ma_path + 'titration.pkl')


def titration2(artifacts, control_case_columns_data, basis, dest, ma_path, subdir, repeats=10, save_metaresults=True,
               plot=True):
    sample_ids = utils.get_sample_ids(control_case_columns_data=control_case_columns_data)
    cohorts = list(sample_ids.keys())
    cohorts.remove(basis)
    target = cohorts[0]
    controls_1 = sample_ids[basis]['control']
    controls_2 = sample_ids[target]['control']
    set_size = min(math.floor(len(controls_1)), len(controls_2))
    if set_size % 2 == 1:
        set_size -= 1
    set_size = int(set_size)
    half_set_size = int(set_size / 2)
    max_p_val = 0
    print('titration repeats:', repeats)
    fig = plotly_titration_subplots(num_plots=len(order_with_NT),
                                    titles=[method_dict[method] for method in order_with_NT])
    titration_results = {method: {} for method in order_with_NT}
    maximal_means = {method: 0 for method in order_with_NT}
    for k in range(repeats):
        print('titration iteration', k)
        
        random.shuffle(controls_1)
        random.shuffle(controls_2)
        for i in range(len(order_with_NT)):
            testing_set = controls_1[:set_size]
            replacement_set = controls_2[:set_size]
            method = order_with_NT[i]
            print(method, flush=True)
            artifact = Artifact.load(artifacts[method])
            df = artifact.view(pd.DataFrame)
            df, = round_dataframes(8, df)
            if k == 0:
                for l in range(set_size + 1):
                    titration_results[method][l] = np.array([0.0 for otu in df])

            curr_pvals = [
                ranksums(df.loc[testing_set[:half_set_size], otu], df.loc[testing_set[half_set_size:], otu])[1] for otu
                in df]
            curr_pvals = fdr(curr_pvals)[1]
            curr_pvals = -1 * np.log(curr_pvals)
            titration_results[method][0] += curr_pvals / repeats
            for l in range(set_size):
                #     if l % 10 == 0:
                #         print(l, flush=True)
                testing_set[l] = replacement_set[l]
                curr_pvals = [
                    ranksums(df.loc[testing_set[:half_set_size], otu], df.loc[testing_set[half_set_size:], otu])[1] for
                    otu in df]
                curr_pvals = fdr(curr_pvals)[1]
                curr_pvals = -1 * np.log(curr_pvals)
                titration_results[method][l + 1] += curr_pvals / repeats
    print(fig.print_grid())
    for i in range(len(order_with_NT)):
        method = order_with_NT[i]
        method_name = method_dict[method]
        print('plotting ' + method_name)
        x = []
        means = []
        medians = []
        top_row_col_num = math.ceil(len(order_with_NT)/2)
        for l in range(len(titration_results[method])):
            # if l % 10 == 0:
            #         print(l, flush=True)
            curr_pvals = titration_results[method][l]
            if i < 4:#top_row_col_num:
                fig.add_trace(go.Scatter(x=[l for _ in curr_pvals],
                                         y=curr_pvals,
                                         mode='markers',
                                         marker=dict(
                                             color='DarkGrey',
                                             opacity=0.2,
                                             size=2,
                                             line=dict(
                                                 color='Black',
                                                 width=1
                                             )
                                         )
                                         ),
                              row=1, col=i +1 ,# * 2 + 1,

                              )
            else:
                fig.add_trace(go.Scatter(x=[l for _ in curr_pvals],
                                         y=curr_pvals,
                                         mode='markers',
                                         marker=dict(
                                             opacity=0.2,
                                             color='DarkGrey',
                                             size=2,
                                             line=dict(
                                                 color='Black',
                                                 width=1
                                             )
                                         )
                                         ),
                              row=2, col=i-4+1,#(i - top_row_col_num) * 2 + 2,

                              )
            curr_pvals = titration_results[method][l]
            mean = np.mean(curr_pvals)
            means.append(mean)
            maximal_means[method] = max(mean, maximal_means[method])
            medians.append(np.median(curr_pvals))
            x.append(l)
            max_p_val = max(max_p_val, max(curr_pvals))

        if i < 4:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=means,
                    mode='lines',
                    marker=dict(
                        color='Blue',
                        size=2
                    )
                ),
                row=1, col=i + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=medians,
                    mode='lines',
                    marker=dict(
                        color='Red',
                        size=2
                    )
                ),
                row=1, col=i + 1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=means,
                    mode='lines',
                    marker=dict(
                        color='Blue',
                        size=2
                    )
                ),
                row=2, col=i - 4 + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=medians,
                    mode='lines',
                    marker=dict(
                        color='Red',
                        size=2
                    )
                ),
                row=2, col=i - 4 + 1,
            )
    if plot:
        print('saving')
        fig.update_xaxes(tickmode='array',
                         tickvals=[0, 0.25 * set_size, 0.5 * set_size, 0.75 * set_size, set_size],
                         ticktext=['0%', '50%', '100%', '50%', '0%'])
        fig.update_yaxes(range=[0, 1.05 * max_p_val])
        fig.update_layout(showlegend=False)
        fig.write_image(dest + 'titration.png', height=600, width=1200, scale=6)
    if save_metaresults:
        if os.path.isfile(ma_path + 'titration.pkl'):
            ma_results = load_pickle(ma_path + 'titration.pkl')
            ma_results[subdir] = titration_results
            save_pickle(ma_results, ma_path + 'titration.pkl')
        else:
            save_pickle({subdir: titration_results}, ma_path + 'titration.pkl')
    print('Done titration', flush=True)
    return maximal_means


def ranksum_meta(ma_path):
    scale = 2
    num_plots = 8
    buff = num_plots % 2
    fig = make_subplots_for_ranksum_meta(num_plots=num_plots, scale=scale,
                                         titles=[method_dict[method] for method in order])
    num_top = math.ceil(num_plots / 2)
    num_bottom = num_top - buff
    ax_limits = {}
    ma_results = load_pickle(ma_path + 'ranksum.pkl')
    collections = list(ma_results.keys())
    max_y = 0
    annotation_y_positions = [0.95 - i * 0.15 for i in range(len(collections))]
    for i in range(len(collections)):
        collection = collections[i]
        ranksum_results = ma_results[collection]
        color = rgb_to_plotly_color_str(plt_color_cycle[i])
        batches = list(ranksum_results.keys())
        batches.remove('basis')
        basis = ranksum_results['basis']
        batches.remove(basis)
        batches = [basis] + batches
        for j in range(len(order)):
            method = order[j]
            row = 1 if j + 1 <= num_top else 2 * (scale + 1)
            col = 1 + j * (scale + 1) if j + 1 <= num_top else 1 + (j - num_top) * (scale + 1)
            for k in range(2):
                batch = batches[k]
                before_negative_log_pvals = ranksum_results[batch][method]['before']
                after_negative_log_pvals = ranksum_results[batch][method]['after']
                r, p = sp.stats.pearsonr(before_negative_log_pvals, after_negative_log_pvals)
                if p > 10 ** -200:
                    decimal_places = np.ceil(-1 * np.log10(p))
                    try:
                        p = str(round(p * (10 ** decimal_places), 2))
                        p = '{}*10<sup>-{}</sup>'.format(p, decimal_places)
                    except FloatingPointError as e:
                        p = '0'
                else:
                    p = '0'
                text_r = '<b>R</b> = {}'.format(round(r, 2))
                text_p = '<b>p</b> = {}'.format(p)
                curr_row = row + k * scale
                fig.add_trace(go.Scatter(x=before_negative_log_pvals,
                                         y=after_negative_log_pvals, mode='markers',
                                         marker=dict(color=color, size=2), opacity=0.33, showlegend=False),
                              row=curr_row, col=col)
                fig.add_annotation(xref='x domain', yref='y domain', x=0.01, xanchor="left",
                                   yanchor='top', y=1 - (i * 0.2), text=text_r, font=dict(size=10, color=color),
                                   showarrow=False, row=curr_row, col=col)
                fig.add_annotation(xref='x domain', yref='y domain', x=0.01, xanchor="left",
                                   yanchor='top', y=0.9 - (i * 0.2), text=text_p, font=dict(size=10, color=color),
                                   showarrow=False, row=curr_row, col=col)
                subplot_position = (curr_row, col)
                max_xy = max(max(before_negative_log_pvals), max(after_negative_log_pvals))
                if subplot_position in ax_limits:
                    ax_limits[subplot_position] = max(ax_limits[subplot_position], max_xy)
                else:
                    ax_limits[subplot_position] = max_xy
                fig.update_yaxes(range=[0, ax_limits[subplot_position] * 1.05], row=curr_row, col=col)
                fig.update_xaxes(range=[0, ax_limits[subplot_position] * 1.05], row=curr_row, col=col)

        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=break_title(collections[i].replace('_', ' ')),
            marker=dict(size=7, color=color)),
            row=1, col=1)
    fig.update_layout(
        showlegend=True,
        autosize=False,
        width=750,
        height=750,
        margin=dict(t=50, b=0, l=30, r=30),
        legend=dict(yanchor="middle", y=0.51, xanchor="center", x=0.5, orientation='h', font=dict(size=19))
    )
    fig.write_image(ma_path + 'ranksum_plots.png', scale=6)


# def ranksum_meta(ma_path):
#     ma_results = load_pickle(ma_path+'ranksum.pkl')
#     collections = list(ma_results.keys())
#     max_y = 0
#     labels = [mlines.Line2D([], [], color=plt_color_cycle[i], lw=4, label=collections[i].replace('_',' ')) for i in range(len(collections))]
#     fig, axes = plt.subplots(2, len(order),figsize=(10*len(order), 22))
#     annotation_y_positions = [0.85 - i*0.15 for i in range(len(collections))]
#     for i in range(len(collections)):
#         collection = collections[i]
#         ranksum_results = ma_results[collection]
#         color = plt_color_cycle[i]
#         batches = list(ranksum_results.keys())
#         batches.remove('basis')
#         basis = ranksum_results['basis']
#         batches.remove(basis)
#         batches = [basis] + batches
#         for j in range(len(order)):
#             method = order[j]
#             for k in range(2):
#                 batch = batches[k]
#                 before_pvals = ranksum_results[batch][method]['before']
#                 after_pvals = ranksum_results[batch][method]['after']
#                 ax = axes[k,j]
#                 ax.scatter(before_pvals, after_pvals, s=15,alpha=0.3, color=color)
#                 ax.tick_params(axis='both', which='major', labelsize=24)
#                 ax.tick_params(axis='both', which='minor', labelsize=16)
#                 if k == 0:
#                     ax.set_title('{}'.format(method_dict[method]), fontsize=45)
#                 r, p = sp.stats.pearsonr(before_pvals, after_pvals)
#                 if p > 10**-200:
#                     decimal_places = np.ceil(-1 * np.log10(p))
#                     try:
#                         p = str(round(p * (10 ** decimal_places), 2))
#                         p = r'${}*10^{{-{}}}$'.format(p, decimal_places)
#                     except FloatingPointError as e:
#                         p = '0'
#                 else:
#                     p = '0'
#                 textstr = '\n'.join([
#                     'R = {}'.format(round(r, 2)),
#                     'p: {}'.format(p)])
#                 ax.text(0.5, annotation_y_positions[i], textstr, color=color, transform=ax.transAxes, fontsize=30)

#     for j in range(len(order)):
#         for k in range(2):
#             ax = axes[k,j]
#             maximal_axis = max(ax.get_xlim()[1], ax.get_ylim()[1])
#             ax.set_xlim(0,maximal_axis)
#             ax.set_ylim(0,maximal_axis)
#             ax.set_aspect('equal')

#     plt.legend(handles=labels,fontsize=35,loc='center left', bbox_to_anchor=(0.5, -0.2))
#     plt.tight_layout()
#     plt.savefig(ma_path + 'ranksum_plots.png', dpi=200)
#     plt.clf()
#     plt.close('all')

def titration_meta(ma_path):
    ma_results = load_pickle(ma_path + 'titration.pkl')
    averaged_results_for_plots = {collection: {method: None for method in order_with_NT} for collection in
                                  ma_results.keys()}
    collections = list(ma_results.keys())
    for i in range(len(collections)):
        collection = collections[i]
        titration_results = ma_results[collection]
        for j in range(len(order_with_NT)):
            method = order_with_NT[j]
            method_results = titration_results[method]
            means = []
            third_quartiles = []
            for l in range(len(method_results)):
                means.append(np.mean(method_results[l]))
                third_quartiles.append(np.percentile(method_results[l], 75))
            averaged_results_for_plots[collection][method] = {'means': means, 'third_quartiles': third_quartiles}

    fig = plotly_titration_subplots(num_plots=len(order_with_NT),
                                    titles=[break_title(method_dict[method]) for method in order_with_NT])

    collections = [x for x in ma_results.keys()]
    max_y = 0
    num_plots_top = math.ceil(len(order_with_NT) / 2)
    for i in range(len(collections)):
        collection = collections[i]
        color = rgb_to_plotly_color_str(plt_color_cycle[i], alpha=0.5)
        annot_color = rgb_to_plotly_color_str(plt_color_cycle[i])
        titration_results = ma_results[collection]
        for j in range(len(order_with_NT)):
            method = order_with_NT[j]
            means = averaged_results_for_plots[collection][method]['means']
            third_quartiles = averaged_results_for_plots[collection][method]['third_quartiles']
            max_y = max(max_y, max(means), max(third_quartiles))
            if j < num_plots_top:
                col, row = j * 2 + 1, 1
            else:
                col, row = (j - num_plots_top) * 2 + 2, 2
            fig.add_trace(go.Scatter(x=[k / len(means) for k in range(len(means))], y=means, mode='lines',
                                     line=dict(color=color, width=8), showlegend=False),
                          row=row, col=col)
            fig.add_trace(
                go.Scatter(x=[k / len(third_quartiles) for k in range(len(third_quartiles))], y=third_quartiles,
                           mode='lines',
                           line=dict(color=color, width=2), showlegend=False),
                row=row, col=col)
            fig.add_annotation(xref='x domain', yref='y domain', x=0.01, xanchor="left",
                               yanchor='top', y=1 - (i * 0.1), text='b-value: ' + str(round(max(means), 2)),
                               font=dict(size=14, color=annot_color),
                               showarrow=False, row=row, col=col)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", name=collections[i].replace('_', ' '),
            marker=dict(size=7, color=color)),
            row=1, col=1)
    fig.update_xaxes(tickmode='array',
                     range=[0, 1],
                     tickvals=[0.25 * (i + 1) for i in range(3)],
                     ticktext=['50%', '100%', '50%'],
                     tickfont=dict(size=12))
    fig.update_yaxes(range=[0, 1.05 * max_y], tickfont=dict(size=16))
    fig.write_image(ma_path + 'titration.png', height=600, width=1200, scale=10)


def fdr_estimate(tables: dict, control_case_columns_data: pd.DataFrame, basis: str, dest: str) -> None:
    x_val = 0
    x_ticks = []
    x_tick_labels = []
    targets = list(np.unique(control_case_columns_data['batch']))
    t1, t2 = targets[0], targets[1]
    t1_samples = control_case_columns_data[
        (control_case_columns_data['batch'] == t1) & (control_case_columns_data['set'] == 'control')].index
    t2_samples = control_case_columns_data[
        (control_case_columns_data['batch'] == t2) & (control_case_columns_data['set'] == 'control')].index
    for method in order_with_NT:
        table = Artifact.load(tables[method]).view(pd.DataFrame)
        t1_table = table.loc[t1_samples, :]
        t2_table = table.loc[t2_samples, :]
        curr_pvals = -1 * np.log([ranksums(t1_table.loc[:, otu], t2_table.loc[:, otu])[1] for otu in table])
        x = [x_val for _ in range(len(curr_pvals))]
        plt.scatter(x, curr_pvals, alpha=0.3)
        x_ticks.append(x_val)
        x_tick_labels.append(method_dict[method])
        x_val += 1
    plt.xticks(x_ticks, labels=x_tick_labels)
    plt.ylabel(' - ln p value')
    plt.savefig(dest + 'false_discovery_rate_change.png')
    plt.close('all')


def plot_abundance_distributions(relative_abundance_table, min_prevalence, destination):
    assert 0 <= min_prevalence <= 1
    destination = destination + '/' if not destination.endswith('/') else destination
    df = Artifact.load(relative_abundance_table).view(pd.DataFrame)
    df2 = df.applymap(lambda x: 0.000001 if x == 0 else x)
    df2 = df2.apply(clr, axis=1)

    num_samples = df.values.shape[0]
    min_samples = min_prevalence * num_samples
    for otu in df:
        if sum(df[otu] > 0) > min_samples:
            series = df2[otu]
            # series = series[(series > 0)]
            plt.hist(series.to_numpy())
            plt.savefig(destination + otu + '.png')
            plt.clf()


def ancom_test_within_batch(table: pd.DataFrame, control_case_columns_data: pd.DataFrame, cohort_to_test,
                            significance_test=None) -> pd.Series:
    """
    Returns the W statistic on ancom (kruskal) for all the features in a single batch with case and control samples.
    :param table: the dataframe that contains the samples and their relative abundance
    :param control_case_columns_data: the metadata
    :param cohort_to_test: which batch should the test be ran on
    :return: a pandas series where the index is the features and the value is the W statistic
    """
    ids = utils.get_sample_ids(control_case_columns_data)
    controls = ids[cohort_to_test]['control']
    cases = ids[cohort_to_test]['case']
    table = table.loc[controls + cases, :]
    grouping = control_case_columns_data.loc[controls + cases, 'set']
    try:
        r = ancom(table, grouping,
                  significance_test=significance_test
                  )
        print(cohort_to_test)
        print(len([x for x in r[0]['W'].tolist() if x is True]))
        return ancom(table, grouping,
                     significance_test=significance_test
                     )[0]['W']
    except Exception as e:
        print(table)
        print(cohort_to_test)
        raise e


def ancom_plot(tables: dict, controls_cases: pd.DataFrame, basis: str, dest: str):
    plt.gca().set_aspect('equal', adjustable='box')
    axes = []
    batches: list = np.unique(controls_cases['batch']).tolist()
    batches.remove(basis)
    batches = [basis] + batches
    fig = plt.figure(figsize=(10 * len(batches), 10 * len(order)), dpi=200)
    gs = fig.add_gridspec(1, 1)
    scatter_plots = gs[0].subgridspec(len(order), len(batches))
    before_table = Artifact.load(tables['NT'])
    df_before: pd.DataFrame = before_table.view(pd.DataFrame)
    for j in range(len(order)):
        method = order[j]
        print(method)
        after_table = Artifact.load(tables[method])
        df_after: pd.DataFrame = after_table.view(pd.DataFrame)
        for i in range(len(batches)):
            ax = fig.add_subplot(scatter_plots[j, i])
            axes.append(ax)
            batch = batches[i]
            before_w = ancom_test_within_batch(df_before, controls_cases, batch)
            X = before_w.values
            after_w = ancom_test_within_batch(df_after, controls_cases, batch)
            Y = after_w.values
            assert before_w.index.tolist() == after_w.index.tolist()
            maximal = max(max(X), max(Y))
            ax.scatter(X, Y, s=5)
            ax.set_xlim(0, maximal * 1.1)
            ax.set_ylim(0, maximal * 1.1)
            ax.tick_params(axis='both', which='major', labelsize=24)
            ax.tick_params(axis='both', which='minor', labelsize=16)
            if j == len(order) - 1:
                ax.set_xlabel('Before correction', fontsize=40)
            if i == 0:
                ax.set_ylabel('After correction', fontsize=40)
            r, p = sp.stats.pearsonr(X, Y)
            if p > 10 ** -200:
                decimal_places = np.ceil(-1 * np.log10(p))
                p = str(round(p * (10 ** decimal_places), 3))
                p = '{}*10^-{}'.format(p, decimal_places)
            else:
                p = '->0'
            textstr = '\n'.join([
                'R = {}'.format(round(r, 2)),
                'p-value: {}'.format(p)])
            ax.annotate(textstr, xy=(0, maximal * 0.8), fontsize=30)
            ax.set_title('{}\n{}'.format(batch, method_dict[method]), fontsize=40)

    # for axis in axes:
    #     lim = (0, maximal*1.2)
    plt.suptitle('Correlation of ancom differential abundance tests\nbefore and after batch correction', fontsize=40)
    plt.savefig(dest + 'ancom_plots.png', dpi=200)
    plt.clf()
    plt.gca().set_aspect('auto')
    plt.close('all')


# def ancom_test(table1: pd.DataFrame, table2: pd.DataFrame, grouping):
#     """
#     returns lists of OTUs only differentially abundant in table1, table2, or both
#     """
#     assert table1.index.to_list == table2.index.tolist() == grouping
#     significance_1 = ancom(table1, pd.Series(grouping, index=table1.index))[0]['Reject null hypothesis'].to_list()
#     significance_2 = ancom(table2, pd.Series(grouping, index=table1.index))[0]['Reject null hypothesis'].to_list()
#     paired = zip(significance_1, significance_2)
#     return [a and not b for a, b in paired], [not a and b for a, b in paired], [a and b for a, b in paired]


def test_separation(table_path, metadata_path, batch1, batch2, control_case_columns_data):
    xgboost_classify(*get_sets_from_table(table_path, metadata_path, batch1, batch2, control_case_columns_data))


# def random_forest_train(artifact: Artifact, control_case_columns_data: pd.DataFrame, cohorts: list):
#     ids = utils.get_sample_ids(control_case_columns_data)
#     df = artifact.view(pd.DataFrame)
#     cohort_data = []
#     batch_labels = []
#     for batch in cohorts:
#         data = df.loc[ids[batch]['control'] + ids[batch]['case']].values
#         cohort_data.append(data)
#         batch_labels += [batch for i in range(len(data))]
#
#
#     test_size = 0.25
#     train_features, test_features, train_labels, test_labels = \
#         train_test_split(np.concatenate(cohort_data), batch_labels, test_size=test_size, random_state=42)
#     clf = RandomForestClassifier(n_estimators=200, min_samples_split=4)
#     clf.fit(train_features, train_labels)
#     # pred = clf.predict(test_features)
#     # label_counts = [0 for i in range(len(cohorts))]
#     # hits = [0 for i in range(len(cohorts))]
#     # for i in range(len(pred)):
#     #     label_counts[cohorts.index(batch_labels[i])] += 1
#     #     if pred[i] == batch_labels[i]:
#     #         hits[cohorts.index(batch_labels[i])] += 1
#     # print('prediction accuracy: ',end='')
#     # for i in range(len(cohorts)):
#     #     print('{}: {}/{} correct, {}%'.format(cohorts[i], hits[i], label_counts[i], 100*hits[i]/label_counts[i]))
#     #print('prediction accuracy: {}%'.format(
#     #    100 * len([pred[i] for i in range(len(pred)) if pred[i] == test_labels[i]]) / len(pred)))
#     return clf
#
def test_rf_prediction(model, test_features, test_labels, ax, title):
    labels = list(np.unique(np.array(test_labels)))
    pred = model.predict(test_features)
    conf_matrix = confusion_matrix(test_labels, pred)
    sns.heatmap(conf_matrix, ax=ax, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
    ax.set_title(title)
    # ax.set_xticklabels(labels)
    # ax.set_yticklabels(labels)

    label_counts = [0 for i in range(len(labels))]
    hits = [0 for i in range(len(labels))]
    for i in range(len(pred)):
        label_counts[labels.index(test_labels[i])] += 1
        if pred[i] == test_labels[i]:
            hits[labels.index(test_labels[i])] += 1
    print('prediction accuracy: ', end='')
    for i in range(len(labels)):
        print(
            '{}: {}/{} correct, {}%'.format(labels[i], hits[i], label_counts[i], round(100 * hits[i] / label_counts[i]),
                                            3))
    print('prediction accuracy: {}%'.format(
        100 * len([pred[i] for i in range(len(pred)) if pred[i] == test_labels[i]]) / len(pred)))


def random_forest_test(artifact_paths: dict, control_case_columns_data: pd.DataFrame, basis: str, dest: str):
    cohorts = control_case_columns_data['batch'].unique().tolist()
    cohorts.remove(basis)

    ids = utils.get_sample_ids(control_case_columns_data)
    basis_ids = ids[basis]['control'] + ids[basis]['case']

    fig, axes = plt.subplots(len(order_with_NT), len(cohorts), figsize=(len(cohorts) * 4, len(order_with_NT) * 4))
    if len(cohorts) == 1:
        axes = [[ax] for ax in axes]
    for c in range(len(cohorts)):
        cohort = cohorts[c]
        print('Random forest accuracy on {} (basis) and {} (target)'.format(basis, cohort))
        curr_batch_ids = ids[cohort]['control'] + ids[cohort]['case']
        relevant_ids = basis_ids + curr_batch_ids
        df = Artifact.load(artifact_paths['NT']).view(pd.DataFrame).loc[relevant_ids, :]
        test_size = 0.25
        indices = [i for i in range(len(df.index))]
        random.shuffle(indices)
        indices_train, indices_test = indices[:int((1 - test_size) * len(indices))], indices[int((1 - test_size) * len(
            indices)):],
        ids_train, ids_test = df.index[indices_train].tolist(), df.index[indices_test].tolist()
        features_train, features_test = df.loc[ids_train, :].values, df.loc[ids_test, :].values
        batches_train, batches_test = control_case_columns_data.loc[ids_train, 'batch'].tolist(), \
                                      control_case_columns_data.loc[
                                          ids_test, 'batch'].tolist()
        clf = RandomForestClassifier(n_estimators=200, min_samples_split=4)
        clf.fit(features_train, batches_train)
        print('No treatment ', end='')
        test_rf_prediction(clf, features_test, batches_test, axes[0][c], '{} No treatment'.format(cohort))
        for m in range(len(order)):
            ax = axes[m + 1][c]
            method = order[m]
            df = Artifact.load(artifact_paths[method]).view(pd.DataFrame).loc[relevant_ids, :]
            features_test = df.loc[ids_test, :].values
            print(method_dict[method], end=' ')
            test_rf_prediction(clf, features_test, batches_test, ax, '{} {}'.format(method, cohort))
    plt.suptitle('Random forest confusion matrices')
    plt.savefig(dest + 'rf_heatmap.png', dpi=200)


def get_sets_from_table(table, metadata, name1, name2, control_case_columns_data):
    df = Artifact.load(table).view(pd.DataFrame)
    relevant_ids = df.index.tolist()
    control_case_columns_data: pd.DataFrame = \
        get_metadata_control_and_case_sets(metadata, control_case_columns_data, relevant_ids)
    target_1_controls = control_case_columns_data[
        (control_case_columns_data['batch'] == name1) & (control_case_columns_data['set'] == 'control')].index.tolist()
    target_2_controls = control_case_columns_data[
        (control_case_columns_data['batch'] == name2) & (control_case_columns_data['set'] == 'control')].index.tolist()
    target_1_controls_values = df.loc[target_1_controls, :].values
    target_2_controls_values = df.loc[target_2_controls, :].values
    return target_1_controls_values, target_2_controls_values


def perc_norm(table: str, perc_norm_metadata: str, set_column: str, batch_column: str, out_path: str) -> str:
    cmd = 'qiime perc-norm percentile-normalize --i-table {} --m-metadata-file {} --m-metadata-column {} ' \
          '--m-batch-file {} --m-batch-column {} --o-perc-norm-table {}PN_{} && conda deactivate'.format(
        table, perc_norm_metadata, set_column, perc_norm_metadata, batch_column, out_path, artifact_name)
    print(cmd)
    os.system(cmd)
    return '{}PN_{}'.format(out_path, artifact_name)


def plot_violin(distance_matrices: dict, control_case_columns_data: pd.DataFrame, basis: str, dest: str,
                table: str) -> None:
    '''
    Saves violin plot of size (num of treatments)x(target datasets)x(controls, cases) to dest
    Parameters
    ----------
    distance_matrices: dict of distance matrices with format name_of_method: skbio.DistanceMatrix.
    dict holds entry 'NT':distance matrix of original table
    control_case_columns_data: control_case_columns_data dict
    basis: the basis
    dest: out-path
    title: plot title
    '''
    draw_3_basis_violins = True
    included_methods = order_with_NT
    np.seterr(under='ignore')

    basis_samples_controls = \
        control_case_columns_data[(control_case_columns_data['batch'] == basis) &
                                  (control_case_columns_data['set'] == 'control')].index.tolist()
    basis_samples_cases = \
        control_case_columns_data[(control_case_columns_data['batch'] == basis) &
                                  (control_case_columns_data['set'] == 'case')].index.tolist()

    if draw_3_basis_violins:
        dmx: pd.DataFrame = Artifact.load(distance_matrices['NT']).view(skbio.DistanceMatrix).to_data_frame()
        basis_controls_v_controls = dmx.loc[basis_samples_controls, basis_samples_controls].values.flatten()
        basis_controls_v_cases = dmx.loc[basis_samples_controls, basis_samples_cases].values.flatten()
        basis_cases_v_cases = dmx.loc[basis_samples_cases, basis_samples_cases].values.flatten()

    cohorts = list(np.unique(control_case_columns_data['batch']))
    cohorts.remove(basis)
    distances_dict = {treatment: {} for treatment in included_methods}

    if len(cohorts) == 2:
        targets_distances = {treatment: {} for treatment in included_methods}
        target_1_controls = \
            control_case_columns_data[(control_case_columns_data['batch'] == cohorts[0]) &
                                      (control_case_columns_data['set'] == 'control')].index.tolist()
        target_2_controls = \
            control_case_columns_data[(control_case_columns_data['batch'] == cohorts[1]) &
                                      (control_case_columns_data['set'] == 'control')].index.tolist()
        target_1_cases = \
            control_case_columns_data[(control_case_columns_data['batch'] == cohorts[0]) &
                                      (control_case_columns_data['set'] == 'case')].index.tolist()
        target_2_cases = \
            control_case_columns_data[(control_case_columns_data['batch'] == cohorts[1]) &
                                      (control_case_columns_data['set'] == 'case')].index.tolist()
    for treatment in included_methods:
        dmx: pd.DataFrame = Artifact.load(distance_matrices[treatment]).view(skbio.DistanceMatrix).to_data_frame()
        for cohort in cohorts:
            distances_dict[treatment][cohort] = {}
            target_samples_controls = \
                control_case_columns_data[(control_case_columns_data['batch'] == cohort) &
                                          (control_case_columns_data['set'] == 'control')].index.tolist()
            target_samples_cases = \
                control_case_columns_data[(control_case_columns_data['batch'] == cohort) &
                                          (control_case_columns_data['set'] == 'case')].index.tolist()
            distances_controls = dmx.loc[basis_samples_controls, target_samples_controls].values.flatten()
            distances_cases = dmx.loc[basis_samples_controls, target_samples_cases].values.flatten()
            distances_dict[treatment][cohort]['controls'] = distances_controls
            distances_dict[treatment][cohort]['cases'] = distances_cases
        if len(cohorts) == 2:
            targets_distances[treatment]['controls'] = \
                dmx.loc[target_1_controls, target_2_controls].values.flatten()
            targets_distances[treatment]['cases'] = \
                dmx.loc[target_1_cases, target_2_cases].values.flatten()
    violins = []
    colors = []
    labels = []
    pos, pos_counter = [], 0
    for cohort in cohorts:
        violins.append(distances_dict['NT'][cohort]['controls'])
        colors.append('lightsteelblue')
        labels.append(cohort + ' no treatment - controls')
        pos.append(pos_counter)
        pos_counter += 1
        for treatment in included_methods[1:]:
            desc = method_dict[treatment]
            violins.append(distances_dict[treatment][cohort]['controls'])
            colors.append('royalblue')
            labels.append(cohort + ' {} - controls'.format(desc))
            pos.append(pos_counter)
            pos_counter += 1
        pos_counter += 1
    pos_counter += 1

    for cohort in cohorts:
        violins.append(distances_dict['NT'][cohort]['cases'])
        colors.append('sandybrown')
        labels.append(cohort + ' no treatment - cases')
        pos.append(pos_counter)
        pos_counter += 1
        for treatment in included_methods[1:]:
            desc = method_dict[treatment]
            violins.append(distances_dict[treatment][cohort]['cases'])
            colors.append('orange')
            labels.append(cohort + ' {} - cases'.format(desc))
            pos.append(pos_counter)
            pos_counter += 1
        pos_counter += 1

    if len(cohorts) == 2:
        pos_counter += 1
        violins.append(targets_distances['NT']['controls'])
        colors.append('aquamarine')
        labels.append('{} v. {} no treatment - controls'.format(cohorts[0], cohorts[1]))
        pos.append(pos_counter)
        pos_counter += 1
        for treatment in included_methods[1:]:
            desc = method_dict[treatment]
            violins.append(targets_distances[treatment]['controls'])
            colors.append('turquoise')
            labels.append('{} v. {} {} - controls'.format(cohorts[0], cohorts[1], desc))
            pos.append(pos_counter)
            pos_counter += 1
        pos_counter += 1
        violins.append(targets_distances['NT']['cases'])
        colors.append('yellowgreen')
        labels.append('{} v. {} no treatment - cases'.format(cohorts[0], cohorts[1]))
        pos.append(pos_counter)
        pos_counter += 1
        for treatment in included_methods[1:]:
            desc = method_dict[treatment]
            violins.append(targets_distances[treatment]['cases'])
            colors.append('olivedrab')
            labels.append('{} v. {} {} - cases'.format(cohorts[0], cohorts[1], desc))
            pos.append(pos_counter)
            pos_counter += 1

    if draw_3_basis_violins:
        has_cases = len(basis_samples_cases) > 0
        violins.insert(0, basis_controls_v_controls)
        if has_cases:
            violins.insert(0, basis_controls_v_cases)
            violins.insert(0, basis_cases_v_cases)
            pos.insert(0, -4)
            pos.insert(0, -3)
        pos.insert(0, -2)
        colors.insert(0, 'gray')
        if has_cases:
            colors.insert(0, 'gray')
            colors.insert(0, 'gray')

        labels.insert(0, basis + ' controls distances from each other')
        if has_cases:
            labels.insert(0, basis + ' controls distances from cases')
            labels.insert(0, base + ' cases distances from each other')

    fig, ax = plt.subplots()
    fig.set_figwidth(25)
    fig.set_figheight(6)
    violin_parts = ax.violinplot(violins, positions=pos, showmeans=True,
                                 quantiles=[[0.25, 0.5, 0.75] for _ in range(len(violins))],
                                 widths=[1 for _ in range(len(violins))])
    violin_bodies = violin_parts['bodies']
    for i in range(len(violin_bodies)):
        violin_bodies[i].set_facecolor(colors[i])
        violin_bodies[i].set_edgecolor('black')
    violin_parts['cmeans'].set_color('red')
    for i in range(len(violins)):
        plt.text(pos[i], -0.1, 'n = {}'.format(str(len(violins[i]))), ha='center', fontsize=6)
    plt.xticks(pos, rotation=30)
    ax.set_xticklabels(labels, ha='right', fontsize=8)
    ax.set_xlabel('Sample sets', fontsize=18)
    ax.set_ylabel('Distance', fontsize=18)
    for i in range(len(violins)):
        plt.text(pos[i], -0.1, 'n = {}'.format(str(len(violins[i]))), ha='center', fontsize=6)
    if draw_3_base_violins:
        extra_violins = 3 if has_cases else 1
        violins, pos = violins[extra_violins:], pos[extra_violins:]

    total_h = max([max(violin) for violin in violins])
    ramp = total_h * 0.025
    for i in range(int(len(violins) / len(included_methods))):
        x_of_no_treatment = i * len(included_methods)
        curr_h = max([max(violin) for violin in violins[x_of_no_treatment:x_of_no_treatment + len(included_methods)]])
        for j in range(x_of_no_treatment + 1, x_of_no_treatment + len(included_methods)):
            t_test = sp.ttest_ind(a=violins[x_of_no_treatment], b=violins[j],
                                  equal_var=False)

            if t_test[1] > 0.05:
                annotation = '*'
            elif t_test[1] > 0.001:
                annotation = '**'
            else:
                annotation = '***'
            x1, x2 = pos[x_of_no_treatment], pos[j]  # columns
            y, h, col = curr_h + ramp, ramp, 'k'
            plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
            plt.text((x1 + x2) * .5, y + h, annotation, ha='center', va='bottom', color=col)
            curr_h += ramp * 3

    ax.set_ylim(-0.25, (total_h + 3 * ramp * len(included_methods)) * 1.04)

    plt.tight_layout()
    plt.savefig(dest)
    plt.clf()


def percentile_normalization(artifact: Artifact, control_case_columns_data: pd.DataFrame) -> Artifact:
    df: pd.DataFrame = artifact.view(pd.DataFrame)
    #  replace zeros with random draw from uniform(0, 10**-9)
    df = df.applymap(lambda abundance: abundance if abundance > 0 else np.random.uniform(0, 10 ** -9))

    # Get numpy array
    samples = df.index.tolist()
    x: np.ndarray = df.values
    cohort_to_indices = {}
    for cohort in np.unique(control_case_columns_data['batch']):
        curr_cohort_control_samples = control_case_columns_data[(control_case_columns_data['batch'] == cohort) &
                                                                (control_case_columns_data[
                                                                     'set'] == 'control')].index.tolist()
        control_indices = [df.index.get_loc(i) for i in curr_cohort_control_samples]
        cohort_to_indices[cohort] = control_indices

    x_norm = [
        [sp.percentileofscore(x[cohort_to_indices[control_case_columns_data.at[samples[i], 'batch']], j], x[i, j],
                              kind='mean') for j in range(x.shape[1])]
        for i in range(len(samples))
    ]

    df = pd.DataFrame(data=x_norm, index=df.index, columns=df.columns)
    output = Artifact.import_data('FeatureTable[Frequency]', df)
    return output


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
    missing_ids = set(relevant_samples) - set(sample_to_cohort.keys())
    if len(missing_ids) > 0:
        warnings.warn(
            str(len(missing_ids)) + ' sample(s) from the feature table have no metadata!\n' + str(missing_ids))
    return metadata_dict, sample_to_cohort


def pcoa_plot(artifact: Artifact, tree: Artifact, control_case_columns_data: pd.DataFrame, dest: str,
              base: str, title='pcoa'):
    '''
    Plots the principal component analysis of a relative abundance feature table artifact.



    Parameters
    ----------
    artifact
    tree
    control_case_columns_data
    dest
    base
    title

    Returns
    -------

    '''

    control_case_columns_data = control_case_columns_data.copy()
    if base:
        base_cases = control_case_columns_data[control_case_columns_data['set'] == base].index.tolist()
        df: pd.DataFrame = artifact.view(pd.DataFrame)
        df = df.drop(index=base_cases)
        artifact.import_data('FeatureTable[RelativeFrequency]', df)
    for sample, data in control_case_columns_data.iterrows():
        if data['set'] == 'control':
            control_case_columns_data.at[sample, 'batch'] = control_case_columns_data.at[sample, 'batch'] + ' controls'
        else:
            control_case_columns_data.at[sample, 'batch'] = control_case_columns_data.at[sample, 'batch'] + ' cases'

    feature_table_ids = artifact.view(pd.DataFrame).index.values
    metadata_ids = control_case_columns_data.index.values
    missing_metadata = list(set(feature_table_ids) - set(metadata_ids))

    # missing_metadata_bool = False
    distance_matrix = get_distance_matrix(artifact, tree)
    if len(missing_metadata) > 0:
        warnings.warn(str(len(missing_metadata)) +
                      ' samples are missing from the metadata file and will be plotted under the same grouping: '
                      + str(missing_metadata))
        missing_df = pd.DataFrame(index=control_case_columns_data.index, columns=control_case_columns_data.columns,
                                  data=np.array(
                                      [['MISSING_FROM_METADATA' for i in range(len(control_case_columns_data.columns))]
                                       for i in range(len(control_case_columns_data.index))]))
        control_case_columns_data = control_case_columns_data.append(missing_df)

    ordination_results = pcoa(distance_matrix.view(DistanceMatrix), number_of_dimensions=3)

    fig = ordination_results.plot(df=control_case_columns_data, column='batch', title=title)
    fig.savefig(dest)
    print('saved pcoa to', dest)
    plt.clf()


def pca_nt(artifact: Artifact, tree: Artifact, control_case_columns_data: pd.DataFrame, base: str,
           dest: str):
    sample_ids = utils.get_sample_ids(control_case_columns_data=control_case_columns_data)
    cohorts = list(sample_ids.keys())
    cohorts.remove(base)
    cohorts = [base] + cohorts
    for cohort_1, cohort_2 in permutations(cohorts):
        cohort_1_controls = sample_ids[cohort_1]['control']
        cohort_1_cases = sample_ids[cohort_1]['case']
        cohort_2_controls = sample_ids[cohort_2]['control']
        cohort_2_cases = sample_ids[cohort_2]['case']
        set1 = cohort_1_controls + cohort_1_cases
        set2 = cohort_2_controls + cohort_2_cases
        basis_samples = sample_ids[base]['control'] + sample_ids[base]['case']
        feature_table_ids = artifact.view(pd.DataFrame).index.values
        metadata_ids = control_case_columns_data.index.values
        missing_metadata = list(set(feature_table_ids) - set(metadata_ids))
        art: pd.DataFrame = artifact.view(pd.DataFrame)
        art.drop(inplace=True, index=basis_samples)
        new_artifact = Artifact.import_data('FeatureTable[RelativeFrequency]', art)

        for cohort in [cohort_1, cohort_2]:
            controls, cases = sample_ids[cohort]['control'], sample_ids[cohort]['case']
            curr = new_artifact.view(pd.DataFrame).loc[controls + cases, :]
            if len(missing_metadata) > 0:
                warnings.warn(str(len(missing_metadata)) +
                              ' samples are missing from the metadata file and will be plotted under the same grouping: '
                              + str(missing_metadata))
                missing_df = pd.DataFrame(index=control_case_columns_data.index,
                                          columns=control_case_columns_data.columns,
                                          data=np.array(
                                              [['MISSING_FROM_METADATA' for i in
                                                range(len(control_case_columns_data.columns))]
                                               for i in range(len(control_case_columns_data.index))]))
                control_case_columns_data = control_case_columns_data.append(missing_df)

            pca = PCA(n_components=2)
            pca_df = pd.DataFrame(index=curr.index, data=pca.fit_transform(curr))
            control_values = pca_df.loc[controls, :].values
            case_values = pca_df.loc[cases, :].values
            all_values = pca_df.loc[controls + cases, :].values

            if len(missing_metadata) > 0:
                unknown = pca_df.loc[missing_metadata, :].values
            plt.figure(figsize=(40, 20))
            fig, ax = plt.subplots()
            ax.scatter(control_values[:, 0], control_values[:, 1], c='red', edgecolors='black', s=15)
            ax.scatter(case_values[:, 0], case_values[:, 1], c='red', marker='s', edgecolors='black', s=15)

            if len(missing_metadata) > 0:
                ax.scatter(unknown[:, 0], unknown[:, 1], c='yellow', marker='^', edgecolors='black', s=5)
            marker_set1_control = mlines.Line2D([], [], markeredgecolor='black', color='red', marker='o',
                                                markersize=6, label='{} controls'.format(cohort_1))
            marker_set1_case = mlines.Line2D([], [], markeredgecolor='black', color='red', marker='s',
                                             markersize=6, label='{} cases'.format(cohort_1))

            labels = [marker_set1_control, marker_set1_case]
            plt.legend(handles=labels,
                       fontsize=6
                       )
            # utils.set_axes_equal(ax)
            ax.set_title(cohort + '\nPrincipal Coordinate Analysis',
                         # fontsize=45
                         )
            confidence_ellipse(x=all_values[:, 0], y=all_values[:, 1], ax=ax, n_std=3.0, facecolor='none',
                               edgecolor='red')
            confidence_ellipse(x=control_values[:, 0], y=control_values[:, 1], ax=ax, n_std=3.0, facecolor='none',
                               edgecolor='salmon')
            confidence_ellipse(x=case_values[:, 0], y=case_values[:, 1], ax=ax, n_std=3.0, facecolor='none',
                               edgecolor='darkred')

            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            plt.savefig(dest + 'nt_pca_{}'.format(cohort), dpi=200)

            plt.clf()


def pcoa_nt(artifact: Artifact, tree: Artifact, control_case_columns_data: pd.DataFrame, base: str,
            dest: str):
    sample_ids = utils.get_sample_ids(control_case_columns_data=control_case_columns_data)
    cohorts = list(sample_ids.keys())
    cohorts.remove(base)

    cohort_1 = cohorts[0]
    cohort_2 = cohorts[1]
    cohort_1_controls = sample_ids[cohort_1]['control']
    cohort_1_cases = sample_ids[cohort_1]['case']
    cohort_2_controls = sample_ids[cohort_2]['control']
    cohort_2_cases = sample_ids[cohort_2]['case']
    set1 = cohort_1_controls + cohort_1_cases
    set2 = cohort_2_controls + cohort_2_cases
    basis_samples = sample_ids[base]['control'] + sample_ids[base]['case']
    feature_table_ids = artifact.view(pd.DataFrame).index.values
    metadata_ids = control_case_columns_data.index.values
    missing_metadata = list(set(feature_table_ids) - set(metadata_ids))
    art: pd.DataFrame = artifact.view(pd.DataFrame)
    art.drop(inplace=True, index=basis_samples)
    new_artifact = Artifact.import_data('FeatureTable[RelativeFrequency]', art)
    for cohort in cohort_1, cohort_2:
        controls, cases = sample_ids[cohort]['control'], sample_ids[cohort]['case']
        curr = Artifact.import_data('FeatureTable[RelativeFrequency]',
                                    new_artifact.view(pd.DataFrame).loc[controls + cases, :])

        distance_matrix = get_distance_matrix(curr, tree)
        if len(missing_metadata) > 0:
            warnings.warn(str(len(missing_metadata)) +
                          ' samples are missing from the metadata file and will be plotted under the same grouping: '
                          + str(missing_metadata))
            missing_df = pd.DataFrame(index=control_case_columns_data.index, columns=control_case_columns_data.columns,
                                      data=np.array(
                                          [['MISSING_FROM_METADATA' for i in
                                            range(len(control_case_columns_data.columns))]
                                           for i in range(len(control_case_columns_data.index))]))
            control_case_columns_data = control_case_columns_data.append(missing_df)

        ordination_results = pcoa(distance_matrix.view(DistanceMatrix), number_of_dimensions=2)
        pcoa_df = ordination_results.samples
        control_values = pcoa_df.loc[controls, :].values
        case_values = pcoa_df.loc[cases, :].values
        all_valyes = pcoa_df.loc[controls + cases, :].values

        if len(missing_metadata) > 0:
            unknown = pcoa_df.loc[missing_metadata, :].values
        plt.figure(figsize=(40, 20))
        fig, ax = plt.subplots()
        ax.scatter(control_values[:, 0], control_values[:, 1], c='red', edgecolors='black', s=15)
        ax.scatter(case_values[:, 0], case_values[:, 1], c='red', marker='s', edgecolors='black', s=15)

        if len(missing_metadata) > 0:
            ax.scatter(unknown[:, 0], unknown[:, 1], c='yellow', marker='^', edgecolors='black', s=5)
        marker_set1_control = mlines.Line2D([], [], markeredgecolor='black', color='red', marker='o',
                                            markersize=6, label='{} controls'.format(cohort_1))
        marker_set1_case = mlines.Line2D([], [], markeredgecolor='black', color='red', marker='s',
                                         markersize=6, label='{} cases'.format(cohort_1))

        labels = [marker_set1_control, marker_set1_case]
        plt.legend(handles=labels,
                   fontsize=6
                   )
        # utils.set_axes_equal(ax)
        ax.set_title(cohort + '\nPrincipal Coordinate Analysis',
                     # fontsize=45
                     )
        confidence_ellipse(x=all_valyes[:, 0], y=all_valyes[:, 1], ax=ax, n_std=3.0, facecolor='none', edgecolor='red')
        confidence_ellipse(x=control_values[:, 0], y=control_values[:, 1], ax=ax, n_std=3.0, facecolor='none',
                           edgecolor='salmon')
        confidence_ellipse(x=case_values[:, 0], y=case_values[:, 1], ax=ax, n_std=3.0, facecolor='none',
                           edgecolor='darkred')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.savefig(dest + 'nt_pcoa_{}'.format(cohort), dpi=200)

        plt.clf()


def pca_plot_all(artifacts: dict, control_case_columns_data: pd.DataFrame, out_path: str, controls_noly=True):
    ids = utils.get_sample_ids(control_case_columns_data)
    cohorts = control_case_columns_data['batch'].unique()
    pca = PCA(2)
    for method, artifact_path in artifacts.items():
        color_counter = 0
        df = Artifact.load(artifact_path).view(pd.DataFrame)
        pca_df = pd.DataFrame(index=df.index, data=pca.fit_transform(df))
        color = plt_color_cycle[color_counter]
        for cohort in cohorts:
            curr_df = pca_df.loc[ids[cohort]['control'], :]
            X = curr_df.values
            plt.scatter(X[:, 0], X[:, 1], c=color, label=cohort + ' controls')
            if not controls_noly:
                curr_df = pca_df.loc[ids[cohort]['case'], :]
                X = curr_df.values
                plt.scatter(X[:, 0], X[:, 1], c=darker(color), label=cohort + ' cases')
            color_counter += 1
            color = plt_color_cycle[color_counter]

        plt.title(method_dict[method])
        plt.legend()
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        addendum = ' controls' if controls_noly else ''
        plt.savefig(out_path + 'PCA_{}{}.png'.format(method, addendum), dpi=100)
        plt.close('all')


def pca_no_basis(artifact: Artifact, tree: Artifact, control_case_columns_data: pd.DataFrame, base: str,
                 dest: str, method: str):
    sample_ids = utils.get_sample_ids(control_case_columns_data=control_case_columns_data)
    cohorts = list(sample_ids.keys())
    cohorts.remove(base)
    cohorts = [base] + cohorts
    for cohort_1, cohort_2 in permutations(cohorts):
        cohort_1_controls = sample_ids[cohort_1]['control']
        cohort_1_cases = sample_ids[cohort_1]['case']
        cohort_2_controls = sample_ids[cohort_2]['control']
        cohort_2_cases = sample_ids[cohort_2]['case']
        feature_table_ids = artifact.view(pd.DataFrame).index.values
        metadata_ids = control_case_columns_data.index.values
        missing_metadata = list(set(feature_table_ids) - set(metadata_ids))
        art: pd.DataFrame = artifact.view(pd.DataFrame)
        new_artifact = Artifact.import_data('FeatureTable[RelativeFrequency]', art)
        if len(missing_metadata) > 0:
            warnings.warn(str(len(missing_metadata)) +
                          ' samples are missing from the metadata file and will be plotted under the same grouping: '
                          + str(missing_metadata))
            missing_df = pd.DataFrame(index=control_case_columns_data.index, columns=control_case_columns_data.columns,
                                      data=np.array(
                                          [['MISSING_FROM_METADATA' for i in
                                            range(len(control_case_columns_data.columns))]
                                           for i in range(len(control_case_columns_data.index))]))
            control_case_columns_data = control_case_columns_data.append(missing_df)

        pca = PCA(n_components=2)
        pca_df = new_artifact.view(pd.DataFrame)
        pca_df = pd.DataFrame(index=pca_df.index, data=pca.fit_transform(pca_df))
        c1_controls = pca_df.loc[cohort_1_controls, :].values
        c2_controls = pca_df.loc[cohort_2_controls, :].values
        c1_cases = pca_df.loc[cohort_1_cases, :].values
        c2_cases = pca_df.loc[cohort_2_cases, :].values
        all_c1 = pca_df.loc[cohort_1_controls + cohort_1_cases, :].values
        all_c2 = pca_df.loc[cohort_2_controls + cohort_2_cases, :].values
        if len(missing_metadata) > 0:
            unknown = pca_df.loc[missing_metadata, :].values
        plt.figure(figsize=(40, 20))
        fig, ax = plt.subplots()
        ax.scatter(c1_controls[:, 0], c1_controls[:, 1], c='red', edgecolors='black', s=15)
        ax.scatter(c1_cases[:, 0], c1_cases[:, 1], c='orange', marker='s', edgecolors='black', s=15)
        ax.scatter(c2_controls[:, 0], c2_controls[:, 1], c='blue', edgecolors='black', s=15)
        ax.scatter(c2_cases[:, 0], c2_cases[:, 1], c='purple', marker='s', edgecolors='black', s=15)
        if len(missing_metadata) > 0:
            ax.scatter(unknown[:, 0], unknown[:, 1], c='yellow', marker='^', edgecolors='black', s=5)
        marker_set1_control = mlines.Line2D([], [], markeredgecolor='black', color='red', marker='o',
                                            markersize=6, label='{} controls'.format(cohort_1))
        marker_set1_case = mlines.Line2D([], [], markeredgecolor='black', color='orange', marker='s',
                                         markersize=6, label='{} cases'.format(cohort_1))
        marker_set2_control = mlines.Line2D([], [], markeredgecolor='black', color='blue', marker='o',
                                            markersize=6, label='{} controls'.format(cohort_2))
        marker_set2_case = mlines.Line2D([], [], markeredgecolor='black', color='purple', marker='s',
                                         markersize=6, label='{} cases'.format(cohort_2))
        labels = [marker_set1_control, marker_set1_case, marker_set2_control, marker_set2_case]
        plt.legend(handles=labels,
                   fontsize=6
                   )
        # utils.set_axes_equal(ax)
        ax.set_title(method_dict[method] + '\n{} and {}'.format(cohort_1, cohort_2) + '\nPrincipal Coordinate Analysis',
                     # fontsize=45
                     )
        # confidence_ellipse(x=all_c1[:,0], y=all_c1[:,1],ax=ax,n_std=3.0,facecolor='none',edgecolor='red')
        # confidence_ellipse(x=c1_controls[:, 0], y=c1_controls[:, 1], ax=ax, n_std=3.0, facecolor='none', edgecolor='salmon')
        # confidence_ellipse(x=c1_cases[:, 0], y=c1_cases[:, 1], ax=ax, n_std=3.0, facecolor='none',
        #                    edgecolor='darkred')
        # confidence_ellipse(x=c2_controls[:, 0], y=c2_controls[:, 1], ax=ax, n_std=3.0, facecolor='none',
        #                    edgecolor='aquamarine')
        # confidence_ellipse(x=c2_cases[:, 0], y=c2_cases[:, 1], ax=ax, n_std=3.0, facecolor='none',
        #                    edgecolor='darkblue')
        # confidence_ellipse(x=all_c2[:,0], y=all_c2[:,1],ax=ax,n_std=3.0,facecolor='none',edgecolor='blue')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.savefig(dest + '{}_{}_pca.png'.format(cohort_1, cohort_2), dpi=200)

        plt.clf()
        plt.close('all')


def pcoa_no_basis(artifact: Artifact, tree: Artifact, control_case_columns_data: pd.DataFrame, base: str,
                  dest: str, method: str):
    sample_ids = utils.get_sample_ids(control_case_columns_data=control_case_columns_data)
    cohorts = list(sample_ids.keys())
    cohorts.remove(base)

    cohort_1 = cohorts[0]
    cohort_2 = cohorts[1]
    cohort_1_controls = sample_ids[cohort_1]['control']
    cohort_1_cases = sample_ids[cohort_1]['case']
    cohort_2_controls = sample_ids[cohort_2]['control']
    cohort_2_cases = sample_ids[cohort_2]['case']
    basis_samples = sample_ids[base]['control'] + sample_ids[base]['case']
    feature_table_ids = artifact.view(pd.DataFrame).index.values
    metadata_ids = control_case_columns_data.index.values
    missing_metadata = list(set(feature_table_ids) - set(metadata_ids))
    art: pd.DataFrame = artifact.view(pd.DataFrame)
    art.drop(inplace=True, index=basis_samples)
    new_artifact = Artifact.import_data('FeatureTable[RelativeFrequency]', art)
    distance_matrix = get_distance_matrix(new_artifact, tree)
    if len(missing_metadata) > 0:
        warnings.warn(str(len(missing_metadata)) +
                      ' samples are missing from the metadata file and will be plotted under the same grouping: '
                      + str(missing_metadata))
        missing_df = pd.DataFrame(index=control_case_columns_data.index, columns=control_case_columns_data.columns,
                                  data=np.array(
                                      [['MISSING_FROM_METADATA' for i in range(len(control_case_columns_data.columns))]
                                       for i in range(len(control_case_columns_data.index))]))
        control_case_columns_data = control_case_columns_data.append(missing_df)

    ordination_results = pcoa(distance_matrix.view(DistanceMatrix), number_of_dimensions=2)
    pcoa_df = ordination_results.samples
    c1_controls = pcoa_df.loc[cohort_1_controls, :].values
    c2_controls = pcoa_df.loc[cohort_2_controls, :].values
    c1_cases = pcoa_df.loc[cohort_1_cases, :].values
    c2_cases = pcoa_df.loc[cohort_2_cases, :].values
    all_c1 = pcoa_df.loc[cohort_1_controls + cohort_1_cases, :].values
    all_c2 = pcoa_df.loc[cohort_2_controls + cohort_2_cases, :].values
    if len(missing_metadata) > 0:
        unknown = pcoa_df.loc[missing_metadata, :].values
    plt.figure(figsize=(40, 20))
    fig, ax = plt.subplots()
    ax.scatter(c1_controls[:, 0], c1_controls[:, 1], c='red', edgecolors='black', s=15)
    ax.scatter(c1_cases[:, 0], c1_cases[:, 1], c='red', marker='s', edgecolors='black', s=15)
    ax.scatter(c2_controls[:, 0], c2_controls[:, 1], c='blue', edgecolors='black', s=15)
    ax.scatter(c2_cases[:, 0], c2_cases[:, 1], c='blue', marker='s', edgecolors='black', s=15)
    if len(missing_metadata) > 0:
        ax.scatter(unknown[:, 0], unknown[:, 1], c='yellow', marker='^', edgecolors='black', s=5)
    marker_set1_control = mlines.Line2D([], [], markeredgecolor='black', color='red', marker='o',
                                        markersize=6, label='{} controls'.format(cohort_1))
    marker_set1_case = mlines.Line2D([], [], markeredgecolor='black', color='red', marker='s',
                                     markersize=6, label='{} cases'.format(cohort_1))
    marker_set2_control = mlines.Line2D([], [], markeredgecolor='black', color='blue', marker='o',
                                        markersize=6, label='{} controls'.format(cohort_2))
    marker_set2_case = mlines.Line2D([], [], markeredgecolor='black', color='blue', marker='s',
                                     markersize=6, label='{} cases'.format(cohort_2))
    labels = [marker_set1_control, marker_set1_case, marker_set2_control, marker_set2_case]
    plt.legend(handles=labels,
               fontsize=6
               )
    # utils.set_axes_equal(ax)
    ax.set_title(method_dict[method] + '\nPrincipal Coordinate Analysis',
                 # fontsize=45
                 )
    confidence_ellipse(x=all_c1[:, 0], y=all_c1[:, 1], ax=ax, n_std=3.0, facecolor='none', edgecolor='red')
    confidence_ellipse(x=c1_controls[:, 0], y=c1_controls[:, 1], ax=ax, n_std=3.0, facecolor='none', edgecolor='salmon')
    confidence_ellipse(x=c1_cases[:, 0], y=c1_cases[:, 1], ax=ax, n_std=3.0, facecolor='none',
                       edgecolor='darkred')
    confidence_ellipse(x=c2_controls[:, 0], y=c2_controls[:, 1], ax=ax, n_std=3.0, facecolor='none',
                       edgecolor='aquamarine')
    confidence_ellipse(x=c2_cases[:, 0], y=c2_cases[:, 1], ax=ax, n_std=3.0, facecolor='none',
                       edgecolor='darkblue')
    confidence_ellipse(x=all_c2[:, 0], y=all_c2[:, 1], ax=ax, n_std=3.0, facecolor='none', edgecolor='blue')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.savefig(dest, dpi=200)

    plt.clf()


def get_bray_curtis(table: Artifact) -> Artifact:
    # ft = Artifact.import_data('FeatureTable[Frequency]', table.view(pd.DataFrame))
    # results = diversity_lib.methods.bray_curtis(ft)
    cp_table = Artifact.import_data('FeatureTable[Frequency]', table.view(pd.DataFrame))
    results = diversity.actions.beta(cp_table, 'braycurtis')
    return results.distance_matrix


def get_distance_matrix(table: Artifact, tree: Artifact) -> Artifact:
    results = diversity_lib.methods.weighted_unifrac(table, tree)

    # cp_table = Artifact.import_data('FeatureTable[Frequency]', table.view(pd.DataFrame))
    # results = diversity.actions.beta(cp_table, 'braycurtis')
    return results.distance_matrix


def correction_score_plots(artifacts: dict, control_case_columns_data: pd.DataFrame, out_path: str):
    # for i in range(3):
    #     alignment_score_plot(artifacts, control_case_columns_data, out_path, i)
    alignment_score_plot(artifacts, control_case_columns_data, out_path, 0)


def alignment_score_plot(artifacts: dict, control_case_columns_data: pd.DataFrame, out_path: str, func: int) -> None:
    method_packs = [  # (entropy_score, 'Entropy score', 'entropy'),
        # (my_alignment_score, 'Aitchison distance score', 'clr_score'),
        (alignment_score, 'Alignment score', 'alignment_score')]
    foo, title, file_name = method_packs[func]
    control_scores = [foo(Artifact.load(artifacts[method]), control_case_columns_data) for method in order_with_NT]
    all_scores = [foo(Artifact.load(artifacts[method]), control_case_columns_data, False) for method in order_with_NT]
    max_score = max(control_scores + all_scores)
    plt.ylim(0, max_score * 1.05)

    plt.figure(figsize=(10, 12))
    plt.scatter([i + 1 for i in range(len(order_with_NT))],
                control_scores,
                s=120, c='blue', label='Controls only')
    if not len(control_case_columns_data[control_case_columns_data['set'] == 'case'].index) == 0:
        plt.scatter([i + 1 for i in range(len(order_with_NT))],
                    all_scores,
                    s=120, c='red', label='Cases only')
    plt.xticks(ticks=[i + 1 for i in range(len(order_with_NT))], fontsize=20,
               labels=[method_dict[method] for method in order_with_NT], rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(out_path + file_name + '.png', dpi=100)
    plt.close('all')


def entropy_score(artifact: Artifact, control_case_columns_data: pd.DataFrame, controls_or_cases=True) -> float:
    df: pd.DataFrame = artifact.view(pd.DataFrame)
    ids = utils.get_sample_ids(control_case_columns_data)
    if controls_or_cases:
        df = df.loc[control_case_columns_data[control_case_columns_data['set'] == 'control'].index, :]
    else:
        df = df.loc[control_case_columns_data[control_case_columns_data['set'] == 'case'].index, :]
    counts = Counter(control_case_columns_data.loc[df.index, 'batch'])
    cohorts = control_case_columns_data['batch'].unique()
    counts = np.array([counts[cohort] for cohort in cohorts])
    print(counts, '->', end=' ')
    min_samples = min(counts)
    for i in range(len(cohorts)):
        if counts[i] > min_samples:
            if controls_or_cases:
                samples = ids[cohorts[i]]['control']
            else:
                samples = ids[cohorts[i]]['case']
            df = df.drop(index=df.loc[samples, :].sample(n=(counts[i] - min_samples)).index)
    counts = Counter(control_case_columns_data.loc[df.index, 'batch'])
    cohorts = control_case_columns_data['batch'].unique()
    counts = np.array([counts[cohort] for cohort in cohorts])
    assert all([counts[i] == counts[i + 1] for i in range(len(counts) - 1)])
    print(counts)
    pca = PCA(20)
    pca_df = pd.DataFrame(index=df.index, data=pca.fit_transform(df))
    dmx = pairwise_distances(pca_df)
    nbrs = NearestNeighbors(n_neighbors=int(0.1 * len(pca_df.index)), algorithm='ball_tree').fit(dmx)
    distances, indices = nbrs.kneighbors(dmx)
    indices = pd.DataFrame(indices)
    indices = indices.applymap(lambda x: df.index[x])
    indices = indices.set_index(np.array([df.index[x] for x in indices.index]))
    indices: pd.DataFrame = indices.applymap(lambda x: control_case_columns_data.at[x, 'batch'])
    score = 0
    for sample, nn in indices.iterrows():
        counts = Counter(nn)
        counts = np.array([counts[cohort] for cohort in cohorts])
        score += entropy(counts)
    return score


def my_alignment_score(artifact: Artifact, control_case_columns_data: pd.DataFrame, controls_only=True) -> float:
    df: pd.DataFrame = artifact.view(pd.DataFrame)
    if controls_only:
        df = df.loc[control_case_columns_data[control_case_columns_data['set'] == 'control'].index, :]
    pca = PCA(20)
    pca_df = pd.DataFrame(index=df.index, data=pca.fit_transform(df))
    dmx = pairwise_distances(pca_df)
    nbrs = NearestNeighbors(n_neighbors=int(0.1 * len(pca_df.index)), algorithm='ball_tree').fit(dmx)
    distances, indices = nbrs.kneighbors(dmx)
    indices = pd.DataFrame(indices)
    indices = indices.applymap(lambda x: df.index[x])
    indices = indices.set_index(np.array([df.index[x] for x in indices.index]))
    indices: pd.DataFrame = indices.applymap(lambda x: control_case_columns_data.at[x, 'batch'])
    cohorts = control_case_columns_data['batch'].unique()
    counts = Counter(control_case_columns_data['batch'])
    counts = np.array([counts[cohort] for cohort in cohorts])
    assert all([x != 0 for x in counts])
    counts_all = counts / counts.sum()
    clr_counts_all = clr(counts_all)
    score = 0
    for sample, nn in indices.iterrows():
        counts = Counter(nn)
        counts = np.array([counts[cohort] for cohort in cohorts])
        counts = pd.Series(counts).replace(0, 0.1).to_numpy()
        counts = counts / counts.sum()
        clr_counts = clr(counts)
        dist = euclidean(clr_counts, clr_counts_all)
        score += dist
    return score


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


def get_cases_and_controls(case_control_df: pd.DataFrame, cohort) -> Tuple[list, list]:
    cases = \
        case_control_df[(case_control_df['batch'] == cohort) & (case_control_df['set'] == 'case')].index.values
    controls = \
        case_control_df[(case_control_df['batch'] == cohort) & (case_control_df['set'] == 'control')].index.values
    return list(cases), list(controls)


def round_dataframes(digits, *dataframes):
    return tuple(df.applymap(lambda x: round(x, digits)) for df in dataframes)


def ranksum_correlation(df_before: pd.DataFrame, df_after: pd.DataFrame, controls_cases: pd.DataFrame, title: str,
                        dest: str) -> None:
    df_before, df_after = round_dataframes(8, df_before, df_after)
    ids = utils.get_sample_ids(controls_cases)
    cohorts = [x for x in ids.keys()]
    fig, axes = plt.subplots(1, len(cohorts), figsize=(len(cohorts) * 8, 8), dpi=200)
    for i in range(len(cohorts)):
        cohort = cohorts[i]
        ax = axes[i]
        control_ids = ids[cohort]['control']
        case_ids = ids[cohort]['case']
        before_pvals = [ranksums(df_before.loc[control_ids, otu], df_before.loc[case_ids, otu])[1] for otu in
                        df_before]
        X = -1 * np.log(before_pvals)
        after_pvals = [ranksums(df_after.loc[control_ids, otu], df_after.loc[case_ids, otu])[1] for otu in df_after]
        Y = -1 * np.log(after_pvals)
        maximal = max(max(X), max(Y))
        ax.scatter(X, Y, s=5)
        ax.set_xlim(0, maximal * 1.1)
        ax.set_ylim(0, maximal * 1.1)
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.tick_params(axis='both', which='minor', labelsize=16)
        if i == len(cohort) - 1:
            ax.set_ylabel('After correction', fontsize=40)
        ax.set_xlabel('Before correction', fontsize=40)
        r, p = sp.stats.pearsonr(X, Y)
        if p > 10 ** -200:
            decimal_places = np.ceil(-1 * np.log10(p))
            p = str(round(p * (10 ** decimal_places), 3))
            p = '{}*10^-{}'.format(p, decimal_places)
        else:
            p = '->0'
        textstr = '\n'.join([
            'R = {}'.format(round(r, 2)),
            'p-value: {}'.format(p)])
        ax.annotate(textstr, xy=(0, maximal * 0.8), fontsize=30)
        ax.set_title('{}'.format(cohort), fontsize=40)
    plt.suptitle(title)
    plt.savefig(dest)


def ranksum_pair(tables: dict, controls_cases: pd.DataFrame, basis: str, dest: str) -> None:
    ids = utils.get_sample_ids(controls_cases)
    targets = np.unique(controls_cases['batch'])
    targets = targets[targets != basis]
    fig = plt.figure(figsize=(8 * len(order), 8 * len(targets)), dpi=200)
    gs = fig.add_gridspec(len(order), len(targets))
    base_controls, base_cases = ids[basis]['control'], ids[basis]['case']
    for i in range(len(order)):
        method = order[i]
        df = Artifact.load(tables[method]).view(pd.DataFrame)
        df, = round_dataframes(8, df)
        for j in range(len(targets)):
            target = targets[j]
            target_controls, target_cases = ids[target]['control'], ids[target]['case']
            ax = fig.add_subplot(gs[i, j])
            batch1_pvals = [ranksums(df.loc[base_controls, otu], df.loc[base_cases, otu])[1] for otu in df]
            batch2_pvals = [ranksums(df.loc[target_controls, otu], df.loc[target_cases, otu])[1] for otu in df]
            X = -1 * np.log(batch1_pvals)
            Y = -1 * np.log(batch2_pvals)
            maximal = max(max(X), max(Y))
            ax.scatter(X, Y, s=10)
            ax.set_xlim(0, maximal * 1.1)
            ax.set_ylim(0, maximal * 1.1)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=8)
            r, p = sp.stats.pearsonr(X, Y)
            if p > 10 ** -200:
                decimal_places = np.ceil(-1 * np.log10(p))
                p = str(round(p * (10 ** decimal_places), 3))
                p = '{}*10^-{}'.format(p, decimal_places)
            else:
                p = '->0'
            textstr = '\n'.join([
                'R = {}'.format(round(r, 2)),
                'p: {}'.format(p)])
            ax.annotate(textstr, xy=(maximal * 0.33, maximal * 0.8), fontsize=15)
            ax.set_title(method, fontsize=25)
            if i == len(order) - 1:
                ax.set_xlabel(basis, fontsize=30)
            if j == 0:
                ax.set_ylabel(target, fontsize=30)

    fig.suptitle('Correlation of differentially abundant\nOTUs between {} and\nother batches', fontsize=40)
    plt.savefig(dest + 'corr_batches.png', bbox_inches='tight', dpi=200)
    plt.clf()
    plt.close('all')


def ttest_no_venn(tables: dict, controls_cases: pd.DataFrame, basis: str, dest: str) -> None:
    plt.gca().set_aspect('equal', adjustable='box')
    ids = utils.get_sample_ids(controls_cases)
    axes = []
    batches: list = np.unique(controls_cases['batch']).tolist()
    batches.remove(basis)
    batches = [basis] + batches
    fig = plt.figure(figsize=(10 * len(batches), 10 * len(order)), dpi=200)
    gs = fig.add_gridspec(1, 1)
    scatter_plots = gs[0].subgridspec(len(order), len(batches))
    before_table = Artifact.load(tables['NT'])
    df_before: pd.DataFrame = before_table.view(pd.DataFrame)
    # df_before = np.log(df_before)
    # df_before, = round_dataframes(8, df_before)
    for j in range(len(order)):
        method = order[j]
        print(method)
        after_table = Artifact.load(tables[method])
        df_after: pd.DataFrame = after_table.view(pd.DataFrame)
        # if not method == 'PN':
        #     df_after = pd.DataFrame(index = df_after.index, columns = df_after.columns,
        #         data = clr(df_after.values))
        # df_after = np.log(df_after)
        # df_after, = round_dataframes(8, df_after)
        for i in range(len(batches)):
            ax = fig.add_subplot(scatter_plots[j, i])
            axes.append(ax)
            batch = batches[i]
            control_ids = ids[batch]['control']
            case_ids = ids[batch]['case']
            before_pvals = [ttest_ind(df_before.loc[control_ids, otu], df_before.loc[case_ids, otu], equal_var=False)[1]
                            for otu in
                            df_before]
            X = -1 * np.log(before_pvals)
            after_pvals = [ttest_ind(df_after.loc[control_ids, otu], df_after.loc[case_ids, otu], equal_var=False)[1]
                           for otu in df_after]
            Y = -1 * np.log(after_pvals)
            maximal = max(max(X), max(Y))
            ax.scatter(X, Y, s=5)
            ax.set_xlim(0, maximal * 1.1)
            ax.set_ylim(0, maximal * 1.1)
            ax.tick_params(axis='both', which='major', labelsize=24)
            ax.tick_params(axis='both', which='minor', labelsize=16)
            if j == len(order) - 1:
                ax.set_xlabel('Before correction', fontsize=40)
            if i == 0:
                ax.set_ylabel('After correction', fontsize=40)
            r, p = sp.stats.pearsonr(X, Y)
            if p > 10 ** -200:
                decimal_places = np.ceil(-1 * np.log10(p))
                p = str(round(p * (10 ** decimal_places), 3))
                p = '{}*10^-{}'.format(p, decimal_places)
            else:
                p = '->0'
            textstr = '\n'.join([
                'R = {}'.format(round(r, 2)),
                'p-value: {}'.format(p)])
            ax.annotate(textstr, xy=(0, maximal * 0.8), fontsize=30)
            ax.set_title('{}\n{}'.format(batch, method_dict[method]), fontsize=40)
    # for axis in axes:
    #     lim = (0, maximal*1.2)
    plt.suptitle('Correlation of differential abundance ttest\nbefore and after batch correction', fontsize=40)
    plt.savefig(dest + 'ttest_plots.png', dpi=200)
    plt.clf()
    plt.gca().set_aspect('auto')
    plt.close('all')


def batch_closeness_correlation(table: Artifact, distance_matrix: pd.DataFrame, controls_cases: pd.DataFrame,
                                batch_1: str, batch_2: str, dest: str):
    sample_ids = utils.get_sample_ids(control_case_columns_data=controls_cases)
    controls_1 = sample_ids[batch_1]['control']
    controls_2 = sample_ids[batch_2]['control']
    df = table.view(pd.DataFrame)
    distance_matrix = distance_matrix[df.columns]
    pvals = [ranksums(df.loc[controls_1, otu], df.loc[controls_2, otu])[1] for otu in df]
    pval_sep = pd.DataFrame(index=df.columns, columns=df.columns, data=[
        [
            np.abs(pvals[i] - pvals[j]) for j in range(len(df.columns))
        ] for i in range(len(df.columns))
    ])
    X = []
    Y = []
    done_pairs = []
    for otu in df.columns:
        for otu2 in df.columns:
            if (otu2, otu) not in done_pairs:
                X.append(distance_matrix.at[otu, otu2])
                Y.append(pval_sep.at[otu, otu2])
                done_pairs.append((otu, otu2))
    plt.scatter(X, Y)
    plt.savefig(dest + 'closeness_batch.png')
    plt.close('all')
    print('Spearman correlation:')
    spr = scipy.stats.spearmanr(X, Y)
    print(spr)
    print('Pearson correlation:')
    prs = scipy.stats.pearsonr(X, Y)
    print(prs)


def ranksum_no_venn(tables: dict, controls_cases: pd.DataFrame, basis: str, dest: str, ma_path: str, subdir, plot=True,
                    save_metaresults=True) -> dict:
    plt.gca().set_aspect('equal', adjustable='box')
    ids = utils.get_sample_ids(controls_cases)
    axes = []
    batches: list = np.unique(controls_cases['batch']).tolist()
    batches.remove(basis)
    batches = [basis] + batches
    fig = plt.figure(
        figsize=(8 * len(batches), 10 * len(order)),
        dpi=200)
    gs = fig.add_gridspec(1, 1)
    scatter_plots = gs[0].subgridspec(len(order), len(batches))
    before_table = Artifact.load(tables['NT'])
    df_before: pd.DataFrame = before_table.view(pd.DataFrame)
    df_before, = round_dataframes(8, df_before)
    r_dict = {}
    p_value_results = {batch: {method: {'before': None, 'after': None} for method in order} for batch in batches}
    for j in range(len(order)):
        method = order[j]
        after_table = Artifact.load(tables[method])
        df_after: pd.DataFrame = after_table.view(pd.DataFrame)
        # if not method == 'PN':
        #     df_after = pd.DataFrame(index = df_after.index, columns = df_after.columns,
        #         data = clr(df_after.values))
        df_after, = round_dataframes(8, df_after)
        for i in range(len(batches)):
            ax = fig.add_subplot(scatter_plots[j, i])
            axes.append(ax)
            batch = batches[i]
            if not batch in r_dict:
                r_dict[batch] = {}
            control_ids = ids[batch]['control']
            case_ids = ids[batch]['case']
            if len(case_ids) == 0:
                print('batch {} has no case samples. exiting ranksum plots'.format(batch))
                return 0
            before_pvals = [ranksums(df_before.loc[control_ids, otu], df_before.loc[case_ids, otu])[1] for otu in
                            df_before]
            after_pvals = [ranksums(df_after.loc[control_ids, otu], df_after.loc[case_ids, otu])[1] for otu in df_after]
            before_pvals, after_pvals = fdr(before_pvals)[1], fdr(after_pvals)[1]
            X = -1 * np.log(before_pvals)
            Y = -1 * np.log(after_pvals)
            p_value_results[batch][method]['before'] = X
            p_value_results[batch][method]['after'] = Y
            p_value_results['basis'] = basis
            maximal = max(max(X), max(Y))
            ax.scatter(X, Y, s=15, alpha=0.3)
            ax.set_xlim(0, maximal * 1.1)
            ax.set_ylim(0, maximal * 1.1)
            ax.tick_params(axis='both', which='major', labelsize=24)
            ax.tick_params(axis='both', which='minor', labelsize=16)
            if j == len(order) - 1:
                ax.set_xlabel('Before correction', fontsize=40)
            if i == 0:
                ax.set_ylabel('After correction', fontsize=40)
            r, p = sp.stats.pearsonr(X, Y)
            r_dict[batch][method] = r
            if plot:
                if p > 10 ** -200:
                    decimal_places = np.ceil(-1 * np.log10(p))
                    try:
                        p = str(round(p * (10 ** decimal_places), 2))
                        p = r'${}*10^{{-{}}}$'.format(p, decimal_places)
                    except FloatingPointError as e:
                        p = '0'
                else:
                    p = '0'
                textstr = '\n'.join([
                    'R = {}'.format(round(r, 2)),
                    'p: {}'.format(p)])
                ax.annotate(textstr, xy=(0.25, maximal * 0.9), fontsize=37)
                ax.set_title('{}\n{}'.format(batch, method_dict[method]), fontsize=40)

    # for axis in axes:
    #     lim = (0, maximal*1.2)
    if plot:
        plt.savefig(dest + 'ranksum_plots.png', dpi=200, bbox_inches='tight')
        plt.clf()
        plt.gca().set_aspect('auto')
        plt.close('all')
    if save_metaresults:
        if os.path.isfile(ma_path + 'ranksum.pkl'):
            ma_results = load_pickle(ma_path + 'ranksum.pkl')
            ma_results[subdir] = p_value_results
            save_pickle(ma_results, ma_path + 'ranksum.pkl')
        else:
            save_pickle({subdir: p_value_results}, ma_path + 'ranksum.pkl')
    return r_dict


def ranksum(tables: dict, controls_cases: pd.DataFrame, basis: str, dest: str) -> None:
    plt.gca().set_aspect('equal', adjustable='box')
    ids = utils.get_sample_ids(controls_cases)
    axes = []
    batches: list = np.unique(controls_cases['batch']).tolist()
    batches.remove(basis)
    fig = plt.figure(figsize=(20 * len(batches), 10 * len(order)), dpi=200)
    gs = fig.add_gridspec(1, 2)
    scatter_plots = gs[0].subgridspec(len(order), len(batches))
    venn_plots = gs[1].subgridspec(len(order), len(batches))
    before_table = Artifact.load(tables['NT'])
    df_before: pd.DataFrame = before_table.view(pd.DataFrame)
    df_before, = round_dataframes(8, df_before)

    for j in range(len(order)):
        method = order[j]
        after_table = Artifact.load(tables[method])
        df_after: pd.DataFrame = after_table.view(pd.DataFrame)
        df_after, = round_dataframes(8, df_after)

        for i in range(len(batches)):
            ax = fig.add_subplot(scatter_plots[j, i])
            axes.append(ax)
            batch = batches[i]
            control_ids = ids[batch]['control']
            case_ids = ids[batch]['case']
            before_pvals = [ranksums(df_before.loc[control_ids, otu], df_before.loc[case_ids, otu])[1] for otu in
                            df_before]
            X = -1 * np.log(before_pvals)
            after_pvals = [ranksums(df_after.loc[control_ids, otu], df_after.loc[case_ids, otu])[1] for otu in df_after]
            Y = -1 * np.log(after_pvals)
            maximal = max(max(X), max(Y))
            ax.scatter(X, Y, s=1)
            ax.set_xlim(0, maximal * 1.1)
            ax.set_ylim(0, maximal * 1.1)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=8)
            ax.set_xlabel('Before correction', fontsize=20)
            ax.set_ylabel('After correction', fontsize=20)
            r, p = sp.stats.pearsonr(X, Y)
            if p > 10 ** -200:
                decimal_places = np.ceil(-1 * np.log10(p))
                p = str(round(p * (10 ** decimal_places), 3))
                p = '{}*10^-{}'.format(p, decimal_places)
            else:
                p = '->0'
            textstr = '\n'.join([
                'R = {}'.format(round(r, 2)),
                'p-value: {}'.format(p)])
            ax.annotate(textstr, xy=(0, maximal * 0.8), fontsize=15)
            ax.set_title('{} ranksum -log pvalues\nbefore and after {}'.format(batch, method_dict[method]), fontsize=25)
            ax = fig.add_subplot(venn_plots[j, i])
            diff_abundance_only_before, diff_abundance_only_after, diff_abundance_both = [], [], []
            before_pvals, after_pvals = fdr(before_pvals)[1], fdr(after_pvals)[1]
            for k in range(len(before_pvals)):
                if before_pvals[k] <= 0.05 and after_pvals[k] <= 0.05:
                    diff_abundance_both.append(True)
                if before_pvals[k] <= 0.05 and after_pvals[k] > 0.05:
                    diff_abundance_only_before.append(True)
                if before_pvals[k] > 0.05 and after_pvals[k] <= 0.05:
                    diff_abundance_only_after.append(True)
            out = venn2((len(diff_abundance_only_before), len(diff_abundance_only_after), len(diff_abundance_both)),
                        ('Significant before correction', 'Significant after correction'), ax=ax)
            for text in out.set_labels:
                text.set_fontsize(25)
            for x in range(len(out.subset_labels)):
                if out.subset_labels[x] is not None:
                    out.subset_labels[x].set_fontsize(20)
            ax.set_title(method_dict[method], fontsize=25)

    # for axis in axes:
    #     lim = (0, maximal*1.2)

    plt.savefig(dest, dpi=200)
    plt.clf()
    plt.gca().set_aspect('auto')


def get_ranksum_pvalues(df: pd.DataFrame, control_case_columns_data: pd.DataFrame, batches: list) -> list:
    assert isinstance(batches, list)
    in_batches_mask = [True if control_case_columns_data.at[x, 'batch'] in batches else False for x in
                       control_case_columns_data.index]
    control_samples = control_case_columns_data[
        (in_batches_mask) & (control_case_columns_data['set'] == 'control')].index.tolist()
    case_samples = control_case_columns_data[
        (in_batches_mask) & (control_case_columns_data['set'] == 'case')].index.tolist()
    ranksum_result = [ranksums(df.loc[control_samples, otu], df.loc[case_samples, otu])[1] for otu in df]
    return fdr(ranksum_result)[1]


def get_diff_abundant_otus(df: pd.DataFrame, control_case_columns_data: pd.DataFrame, batches: list) -> list:
    ranksums_pvals = get_ranksum_pvalues(df, control_case_columns_data, batches)
    return [df.columns[i] for i in range(len(ranksums_pvals)) if ranksums_pvals[i] <= 0.05]


def get_diff_abundant_otu_sets_for_venns(df: pd.DataFrame, control_case_columns_data: pd.DataFrame, base: str,
                                         target: str):
    return (
        get_diff_abundant_otus(
            df, control_case_columns_data, [base]
        ),
        get_diff_abundant_otus(
            df, control_case_columns_data, [target]
        ),
        get_diff_abundant_otus(
            df, control_case_columns_data, [base, target]
        )
    )


def get_diff_abundant_otu_sets_for_venns_no_pooling(df: pd.DataFrame, control_case_columns_data: pd.DataFrame,
                                                    base: str,
                                                    target: str):
    return (
        get_diff_abundant_otus(
            df, control_case_columns_data, [base]
        ),
        get_diff_abundant_otus(
            df, control_case_columns_data, [target]
        )
    )


def plot_venn_in_ax(df: pd.DataFrame, control_case_columns_data: pd.DataFrame, base: str, target: str, ax: object):
    sets = get_diff_abundant_otu_sets_for_venns(df, control_case_columns_data, base, target)
    vd3 = venn3([set(x) for x in sets], set_labels=[base, target, 'Combined'], ax=ax)
    for text in vd3.set_labels:
        text.set_fontsize(16)
    # for text in vd3.subset_labels:
    #     text.set_fontsize(16)


def plot_venns_before_and_after_batch_correction(artifact_paths: dict, control_case_columns_data: pd.DataFrame,
                                                 base: str,
                                                 out_path: str) -> None:
    cohorts = control_case_columns_data['batch'].unique().tolist()
    if len(cohorts) != 2:
        print(len(cohorts), 'cohorts, will not plot venns before and after correction for differentially abundant taxa')
        return
    cohorts.remove(base)
    target = cohorts[0]
    dim = math.ceil(math.sqrt(len(order_with_NT)))
    fig, axes = plt.subplots(dim, dim, figsize=(7 * dim, 7 * dim))
    if dim == 1:
        axes = np.array([[axes]])
    i = 0
    j = 0
    nt_art = Artifact.load(artifact_paths['NT'])
    nt_df = nt_art.view(pd.DataFrame)
    nt_sets = plot_venn_in_ax(nt_df, control_case_columns_data, base, target, axes[0, 0])
    plt.suptitle('Differentially abundant taxa\nunder different batch corrections', fontsize=20)
    for method in order_with_NT:
        if j == dim:
            j = 0
            i += 1
        ax = axes[i, j]
        ax.set_title(method_dict[method], fontsize=16)
        plot_venn_in_ax(Artifact.load(artifact_paths[method]).view(pd.DataFrame), control_case_columns_data, base,
                        target, ax)
        j += 1
    plt.savefig(out_path + 'v3.png', dpi=200)
    plt.clf()
    plt.close('all')


def plot_venn_in_ax_no_pooling(df: pd.DataFrame, control_case_columns_data: pd.DataFrame, base: str, target: str,
                               ax: object):
    sets = get_diff_abundant_otu_sets_for_venns_no_pooling(df, control_case_columns_data, base, target)
    diff_abundant_in_batch_1 = set(sets[0])
    diff_abundant_in_batch_2 = set(sets[1])
    diff_abundant_in_both = diff_abundant_in_batch_1 & diff_abundant_in_batch_2
    out = venn2((len(diff_abundant_in_batch_1), len(diff_abundant_in_batch_2), len(diff_abundant_in_both)),
                ('', ''), ax=ax)
    print(ax.dx, ax, dy)
    for text in out.subset_labels:
        if text:
            text.set_fontsize(26)


def save_venn2_data_for_meta(artifact_paths: dict, control_case_columns_data: pd.DataFrame, base: str, out_path: str,
                             subdir: str, ma_path: str) -> None:
    try:
        ma_results = load_pickle(ma_path + 'venn2.pkl')
    except Exception:
        ma_results = {}
    ma_results[subdir] = {}
    cohorts = control_case_columns_data['batch'].unique().tolist()
    if len(cohorts) != 2:
        print(len(cohorts), 'cohorts, will not plot venns before and after correction for differentially abundant taxa')
        return
    cohorts.remove(base)
    target = cohorts[0]
    order_with_NT.insert(1, 'NTP')
    artifact_paths['NTP'] = artifact_paths['NT'].replace('NT', 'NTP')
    for method in order_with_NT:
        sets = get_diff_abundant_otu_sets_for_venns_no_pooling(Artifact.load(artifact_paths[method]).view(pd.DataFrame),
                                                               control_case_columns_data, base, target)
        diff_abundant_in_batch_1 = set(sets[0])
        diff_abundant_in_batch_2 = set(sets[1])
        diff_abundant_in_both = diff_abundant_in_batch_1 & diff_abundant_in_batch_2
        sizes = (len(diff_abundant_in_batch_1), len(diff_abundant_in_batch_2), len(diff_abundant_in_both))
        ma_results[subdir][method] = sizes
    save_pickle(ma_results, ma_path + 'venn2.pkl')
    order_with_NT.remove('NTP')
    artifact_paths.pop('NTP')


def venn2_meta(ma_path):
    ma_results = load_pickle(ma_path + 'venn2.pkl')
    collections = list(ma_results.keys())
    collections.sort()
    f, axes = plt.subplots(len(order_with_NT), len(collections), figsize=(5 * len(collections), 5 * len(order_with_NT)))
    for i in range(len(collections)):
        collection = collections[i]
        results = ma_results[collection]
        for j in range(len(order_with_NT)):
            method = order_with_NT[j]
            sizes = results[method]
            ax = axes[j, i]
            o = venn2(subsets=sizes, set_labels=('', ''), ax=ax)
            for text in o.subset_labels:
                if text:
                    text.set_fontsize(26)
            if j == 0:
                ax.text(0, 2, collection.replace('_', ' '), fontsize=32)
            if i == 0:
                ax.text(-2, 0.0, method_dict[method], fontsize=32)
    plt.savefig(ma_path + 'venn2.png', dpi=200, bbox_inches='tight')


def plot_ancom_venns(artifact_paths: dict, control_case_columns_data: pd.DataFrame, base: str, out_path: str) -> None:
    targets = np.unique(control_case_columns_data['batch'])
    targets = targets[targets != base]
    assert len(targets) == 2

    target_1_samples_controls, target_1_samples_cases = get_cases_and_controls(control_case_columns_data, targets[0])
    target_2_samples_controls, target_2_samples_cases = get_cases_and_controls(control_case_columns_data, targets[1])
    samples = target_1_samples_controls + target_2_samples_controls
    grouping = ['set1' for x in target_1_samples_controls] + ['set2' for x in target_2_samples_controls]
    no_treatment = Artifact.load(artifact_paths['NT']).loc[samples, :]
    for k, v in artifact_paths.items():
        method = method_dict[k]
        artifact = Artifact.load(v).view(pd.DataFrame).loc[samples, :]
        ancom_sets = ancom_test(no_treatment, artifact, grouping)
        sizes = (sum(x) for x in ancom_sets)
        venn2(sizes,
              ('Significant ' + targets[0], 'Significant ' + targets[1]))
        plt.savefig(out_path + method + '_' + 'targets_controls_ancom_venn.png', bbox_inches='tight', dpi=100)
        plt.clf()


def plot_venns(artifact_paths: dict, control_case_columns_data: pd.DataFrame, base: str, out_path: str) -> None:
    dim = math.ceil(math.sqrt(len(artifact_paths)))
    fig = plt.figure(figsize=(10 * dim, 10 * dim), dpi=200)
    gs = fig.add_gridspec(dim, dim)
    targets = np.unique(control_case_columns_data['batch'])
    targets = targets[targets != base]
    assert len(targets) == 2
    target_1_samples_controls, target_1_samples_cases = get_cases_and_controls(control_case_columns_data, targets[0])
    target_2_samples_controls, target_2_samples_cases = get_cases_and_controls(control_case_columns_data, targets[1])
    i = -1
    j = 0
    for k in order_with_NT:
        if i == dim - 1:
            i = 0
            j += 1
        else:
            i += 1
        v = artifact_paths[k]
        ax = fig.add_subplot(gs[i, j])

        method = method_dict[k]
        artifact = Artifact.load(v).view(pd.DataFrame)
        target_1_abundance_controls, target_1_abundance_cases = artifact.loc[target_1_samples_controls,
                                                                :], artifact.loc[target_1_samples_cases, :]
        target_2_abundance_controls, target_2_abundance_cases = artifact.loc[target_2_samples_controls,
                                                                :], artifact.loc[target_2_samples_cases, :]
        target_1_pvals, target_2_pvals = [], []
        otus = [otu_id for otu_id in artifact.columns]
        for otu_id in otus:
            current_target_1_otu_controls = target_1_abundance_controls[otu_id].tolist()
            current_target_1_otu_cases = target_1_abundance_cases[otu_id].tolist()
            current_target_2_otu_controls = target_2_abundance_controls[otu_id].tolist()
            current_target_2_otu_cases = target_2_abundance_cases[otu_id].tolist()
            target_1_result = ranksums(current_target_1_otu_controls, current_target_1_otu_cases)
            target_1_pvals.append(target_1_result[1])
            target_2_result = ranksums(current_target_2_otu_controls, current_target_2_otu_cases)
            target_2_pvals.append(target_2_result[1])
        target_1_pvals, target_2_pvals = fdr(target_1_pvals)[1], fdr(target_2_pvals)[1]
        diff_abundance_only_t1, diff_abundance_only_t2, diff_abundance_both = [], [], []
        for z in range(len(target_1_pvals)):
            if target_1_pvals[z] <= 0.05 and target_2_pvals[z] <= 0.05:
                diff_abundance_both.append(True)
            if target_1_pvals[z] <= 0.05 and target_2_pvals[z] > 0.05:
                diff_abundance_only_t1.append(True)
            if target_1_pvals[z] > 0.05 and target_2_pvals[z] <= 0.05:
                diff_abundance_only_t2.append(True)
        out = venn2((len(diff_abundance_only_t1), len(diff_abundance_only_t2), len(diff_abundance_both)),
                    ('Significant ' + targets[0], 'Significant ' + targets[1]), ax=ax)
        for text in out.set_labels:
            text.set_fontsize(25)
        for x in range(len(out.subset_labels)):
            if out.subset_labels[x] is not None:
                out.subset_labels[x].set_fontsize(20)
        ax.set_title(method, fontsize=40)

    fig.suptitle('Differentially abundant OTUs', fontsize=40)
    plt.subplots_adjust(hspace=.001)
    plt.subplots_adjust(top=0.85)
    plt.savefig(out_path + 'targets_ranksum_venn.png', bbox_inches='tight', dpi=200)
    plt.close('all')


def get_dmx_id_order(dmx: str):
    with open(dmx, 'r') as fp:
        order = fp.readline().strip().split('\t')
    return order


def get_cohort_to_samples(metadata: str) -> dict:
    output = {}
    with open(metadata, 'r') as fp:
        metadata = fp.readlines()[1:]
    for line in metadata:
        line = line.strip().split('\t')
        sample_id = line[0]
        group = line[-1]
        if group not in output:
            output[group] = []
        output[group].append(sample_id)
    return output


def combat_correction(artifact: Artifact, control_case_columns_data: pd.DataFrame) -> Artifact:
    df: pd.DataFrame = artifact.view(pd.DataFrame)
    assert utils.validate_relative_abundance_dataframe(df)
    df = df.T
    otu_indices = df.index.values
    samples = list(df.columns.values)
    grouping = control_case_columns_data['batch']
    df = np.log(df)
    df = pycombat(df, grouping)
    df = np.exp(df)
    df = df.div(df.sum(axis=0), axis=1)
    df = df.T
    assert utils.validate_relative_abundance_dataframe(df)
    return Artifact.import_data('FeatureTable[RelativeFrequency]', df)


def read_distance_matrix(path: str, metadata: str,
                         basis='ob_goodrich') -> (DistanceMatrix, list, dict):
    if path.endswith('qza'):
        clean = True
        path = utils.qiime_export(path, 'tsv')
    else:
        clean = False
    output = DistanceMatrix.read(path)
    ids = list(output.ids)
    cohort_to_ids = {}
    with open(metadata, 'r') as fp:
        metadata = fp.readlines()[1:]
    for line in metadata:
        line = line.strip().split('\t')
        sample_id = line[0]
        if sample_id not in ids:
            continue
        group = line[-1]
        if group not in cohort_to_ids:
            cohort_to_ids[group] = []
        cohort_to_ids[group].append(sample_id)
        ids[ids.index(sample_id)] = group
    basis_ids = cohort_to_ids[basis]
    sampled_ids = random.sample(basis_ids, int(len(basis_ids) * 0.1))
    cohort_to_ids[basis + 'SAMPLE'] = sampled_ids
    cohort_to_ids[basis] = [x for x in basis_ids if x not in sampled_ids]
    if clean:
        os.system('rm ' + path)
    return output, ids, cohort_to_ids


def _permanova(dmx: DistanceMatrix, grouping: list) -> pd.Series:
    output = permanova(dmx, grouping)
    return output


def permanova_test(distance_matrix_path: str, metadata: str) -> pd.Series:
    data = read_distance_matrix(distance_matrix_path, metadata)
    return _permanova(data[0], data[1])


def paired_distances(dmx: DistanceMatrix, grouping: dict, basis='ob_goodrich'):
    output = {}
    basis_ids = grouping[basis]
    for k, v in grouping.items():
        if k == basis:
            continue
        filtered = dmx.filter(list(set(basis_ids).union(set(v)))).data
        distances = []
        for i in range(len(filtered)):
            for j in range(len(filtered[i])):
                if not i == j:
                    distances.append(filtered[i][j])
        output[k] = distances
    return output


def plot_one(path: str, metadata: str, basis='ob_goodrich'):
    dmx, ids, cohort_to_ids = read_distance_matrix(path, metadata)
    distances = paired_distances(dmx, cohort_to_ids)
    sorted_distances_lists = []
    cohorts = list(distances.keys())
    for cohort in cohorts:
        if cohort == basis + 'SAMPLE':
            sample = distances[cohort]
        else:
            sorted_distances_lists.append(distances[cohort])
    sorted_distances_lists.insert(0, sample)
    cohorts.remove(basis + 'SAMPLE')
    cohorts.insert(0, basis + ' sample')

    plt.boxplot(sorted_distances_lists)
    plt.xticks([i for i in range(1, len(sorted_distances_lists) + 1)], cohorts)
    plt.xticks(rotation=90)
    plt.show()


def read_distance_matrix_to_list(dmx: str) -> (list, list):
    output = []
    with open(dmx, 'r') as fp:
        lines = fp.readlines()
    while lines[0].startswith('#'):
        if not lines[0].startswith('#OTU'):
            lines = lines[1:]
        else:
            break
    ids = lines[0].strip().split('\t')
    if ids[0].startswith('#OTU'):
        ids = ids[1:]
    lines = lines[1:]
    for i in range(len(ids)):
        line = lines[i].strip().split('\t')
        sample = line[0]
        line = line[1:]
        assert sample == ids[i]
        assert (sum([float(x) for x in line]) > 0)
        output.append([float(x) for x in line])
    return output, ids


def dmx_artifacts_to_dataframes(matrices):
    dataframes = {matrix: matrix.view(DistanceMatrix).to_data_frame() for matrix in matrices}
    ids = [list(dataframe.index.values) for dataframe in dataframes.values()]
    for i in range(1, len(ids)):
        assert ids[i - 1] == ids[i]
    return dataframes, ids[0]


def read_distance_matrices_to_lists(matrices):
    dataframes = {matrix: Artifact.load(matrix).view(DistanceMatrix).to_data_frame() for matrix in matrices}
    ids = [list(dataframe.index.values) for dataframe in dataframes.values()]
    for i in range(1, len(ids)):
        assert ids[i - 1] == ids[i]
    return dataframes, ids[0]


def plot_distances(metadata: str, basis: str, *matrices, dest: str):
    data, ids = read_distance_matrices_to_lists(*matrices)
    data = {k: list(v.values) for k, v in data.items()}

    cohort_to_samples, samples_to_cohort = parse_metadata(metadata, relevant_samples=ids)
    basis_ids = cohort_to_samples[basis]

    # re-distribute data to distances from basis samples
    distances_to_basis = {cohort: {file: [] for file in matrices} for cohort in cohort_to_samples}
    basis_indices = [ids.index(x) for x in basis_ids]
    for matrix in matrices:
        for sample_index in basis_indices:
            current_sample_distances = data[matrix][sample_index]
            for other_sample_index in range(len(current_sample_distances)):
                if other_sample_index not in basis_indices:
                    other_sample_id = ids[other_sample_index]
                    try:
                        other_sample_cohort = samples_to_cohort[other_sample_id]
                        distances_to_basis[other_sample_cohort][matrix].append(
                            data[matrix][sample_index][other_sample_index])
                    except KeyError as e:
                        print('There was a problem with the metadata for sample', other_sample_id, '. does it exist in '
                                                                                                   'the '
                                                                                                   'metadata with a '
                                                                                                   'source_file column?')

    # arrange all in one list of vectors
    plot_data = []
    labels = []
    #
    # for matrix in matrices:
    #     plot_data.append(distances_to_basis[basis + ' sample'][matrix])
    #     labels.append(matrix + ' ' + basis + ' sample')
    # distances_to_basis.pop(basis + ' sample')

    for cohort, _matrices in distances_to_basis.items():
        for matrix in _matrices:
            if len(distances_to_basis[cohort][matrix]) > 0:
                plot_data.append(distances_to_basis[cohort][matrix])
                labels.append(matrix + ' ' + cohort)
                plot_data.append([])
                labels.append('')
        for i in range(3):
            plot_data.append([])
            labels.append('')

    # plot
    plt.figure(figsize=(120, 20))
    plt.boxplot(plot_data
                # ,positions = ?
                )
    plt.xticks([i + 1 for i in range(len(plot_data))], labels=labels, rotation=60, ha='right')
    plt.savefig(dest + 'distances.png', bbox_inches='tight')
    plt.clf()


def read_distance_matrix_to_skbio(dmx: str) -> (list, list):
    return DistanceMatrix.read(dmx)


def read_distance_matrices_to_skbio(*matrices):
    output = {}
    for matrix in matrices:
        output[matrix] = read_distance_matrix_to_skbio(matrix)
    return output


def reduce_artifact_to_two(table: Artifact,
                           cohort_1: str, cohort_2: str, metadata: str, control_data: str):
    df: pd.DataFrame = table.view(pd.DataFrame)
    sample_ids: list = df.index.values.tolist()
    control_data: pd.DataFrame = get_metadata_control_and_case_sets(metadata=metadata, control_data=control_data,
                                                                    relevant_ids=sample_ids)
    in_both = set(sample_ids) & set(control_data.index.values.tolist())
    df = df.loc[list(in_both), :]
    sample_ids: list = df.index.values.tolist()
    relevant = [x for x in sample_ids if control_data.at[x, 'batch'] in [cohort_1, cohort_2]]
    df = df.loc[relevant, :]
    table = Artifact.import_data('FeatureTable[RelativeFrequency]', df)
    return table


def plot_partial_permanova(table: Artifact, tree: Artifact,
                           cohort_1: str, cohort_2: str, metadata: str, control_data: str, dest: str) -> float:
    df: pd.DataFrame = table.view(pd.DataFrame)
    sample_ids: list = df.index.values.tolist()
    control_data: pd.DataFrame = get_metadata_control_and_case_sets(metadata=metadata, control_data=control_data,
                                                                    relevant_ids=sample_ids)
    in_both = set(sample_ids) & set(control_data.index.values.tolist())
    df = df.loc[list(in_both), :]
    sample_ids: list = df.index.values.tolist()
    relevant = [x for x in sample_ids if control_data.at[x, 'batch'] in [cohort_1, cohort_2]]
    df = df.loc[relevant, :]
    df.index = df.index.astype(str)
    table = Artifact.import_data('FeatureTable[RelativeFrequency]', df)
    distance_matrix = get_distance_matrix(table, tree).view(skbio.DistanceMatrix)
    grouping = [cohort_1 if control_data.at[x, 'batch'] == cohort_1 else cohort_2 for x in relevant]
    stats = list(permanova(distance_matrix, grouping, permutations=2000))
    f_stat, p_stat = stats[4], stats[5]
    return f_stat


def plot_permanova(matrices: dict, control_case_columns_data: pd.DataFrame, dest: str):
    methods = matrices.keys()
    batches = control_case_columns_data['batch'].unique()
    dim = math.ceil(len(methods) / 2)
    fig, axes = plt.subplots(dim, 2, dpi=100)
    if dim < 2:
        axes = np.array([[ax] for ax in axes]).T
    print(axes)
    fig: plt.Figure = fig
    fig.suptitle('PERMANOVA between batches')
    print(dim)
    for ax in axes.flatten():
        ax.axis('off')
        ax.axis('tight')
    i = j = 0
    ids = utils.get_sample_ids(control_case_columns_data)
    ids = {k: v['control'] + v['case'] for k, v in ids.items()}
    for method in methods:
        dmx: skbio.DistanceMatrix = Artifact.load(matrices[method]).view(skbio.DistanceMatrix)
        table_data = [[' ' for x in batches] for y in batches]
        table_colors = [['gray' for x in batches] for y in batches]
        ax = axes[i, j]
        for k in range(len(batches)):
            batch_1 = batches[k]
            l = k + 1
            while l < len(batches):
                batch_2 = batches[l]
                samples = ids[batch_1] + ids[batch_2]
                curr_dmx = dmx.filter(samples)
                grp = [batch_1 for x in ids[batch_1]] + [batch_2 for y in ids[batch_2]]
                print(method, batch_1, batch_2)
                stats = list(permanova(curr_dmx, grp))
                table_data[k][l] = round(stats[4], 3)
                table_colors[k][l] = 'white'
                l += 1
        fig.patch.set_visible(False)
        if j == 0:
            the_table = ax.table(cellText=table_data, cellColours=table_colors,
                                 colLabels=batches, rowLabels=batches,
                                 cellLoc='center', loc='center')
        else:
            the_table = ax.table(cellText=table_data, cellColours=table_colors,
                                 colLabels=batches,
                                 cellLoc='center', loc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(8)
        # the_table.scale(1.5,1.3)
        # cellDict = the_table.get_celld()
        # for a in range(0, len(batches)):
        #     cellDict[(0, a)].set_height(.3)
        #     cellDict[(a, 0)].set_height(.3)
        #     for b in range(1, len(batches) + 1):
        #         cellDict[(b, a)].set_height(.2)
        ax.set_title(method_dict[method])
        if j == 1:
            j = 0
            i += 1
        else:
            j += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()

    plt.savefig(dest + 'f_stat_permanova.png', bbox_inches='tight', dpi=200)
    plt.clf()
    plt.close('all')
    return

    dmx: pd.DataFrame = Artifact.load(matrices['NT']).view(skbio.DistanceMatrix).to_data_frame()
    ids = dmx.index.to_list()
    cohort_to_samples, samples_to_cohort = parse_metadata(metadata, relevant_samples=ids)
    basis_ids = cohort_to_samples[basis]
    stats_to_basis = {}
    cohort_to_samples.pop(basis)
    for cohort in cohort_to_samples:
        target_ids = cohort_to_samples[cohort]
        relevant_ids = basis_ids + target_ids
        curr_dmx = list(dmx.loc[relevant_ids, relevant_ids].values)
        grouping = [basis if relevant_ids[i] in basis_ids else cohort for i in range(len(relevant_ids))]
        stats = list(permanova(DistanceMatrix(curr_dmx), grouping))
        f_stat, p_stat = stats[4], stats[5]
        stats_to_basis[cohort] = {'NT': (f_stat, p_stat)}

    for cohort in cohort_to_samples:
        for method in order:
            dataframe = Artifact.load(matrices[method]).view(skbio.DistanceMatrix).to_data_frame()
            target_ids = cohort_to_samples[cohort]
            relevant_ids = basis_ids + target_ids
            curr_dmx = list(dataframe.loc[relevant_ids, relevant_ids].values)
            grouping = [basis if relevant_ids[i] in basis_ids else cohort for i in range(len(relevant_ids))]
            stats = list(permanova(DistanceMatrix(curr_dmx), grouping))
            f_stat, p_stat = stats[4], stats[5]
            stats_to_basis[cohort][method] = f_stat, p_stat

    # arrange all in one list of vectors
    f_plot_data = []
    f_labels = []
    x_range = []
    x_index = 1

    for cohort, methods in stats_to_basis.items():
        for method, stats in methods.items():
            f_stat, p_stat = stats[0], stats[1]
            f_plot_data.append(f_stat)
            f_label = cohort + ' ' + method_dict[method]
            if 0.01 <= p_stat < 0.05:
                f_label += ' *'
            elif 0.001 <= p_stat < 0.01:
                f_label += ' **'
            elif p_stat < 0.001:
                f_label += ' ***'
            f_labels.append(cohort + ' ' + method_dict[method])
            x_range.append(x_index)
            x_index += 1
        x_index += 1

    if len(cohort_to_samples) == 2:
        targets = tuple(cohort_to_samples.keys())
        t1, t2 = targets[0], targets[1]
        t1_samples, t2_samples = cohort_to_samples[t1], cohort_to_samples[t2]
        stats_targets = {}
        relevant_ids = t1_samples + t2_samples
        curr_dmx = list(dmx.loc[relevant_ids, relevant_ids].values)
        grouping = [t1 if relevant_ids[i] in t1_samples else t2 for i in range(len(relevant_ids))]
        stats = list(permanova(DistanceMatrix(curr_dmx), grouping))
        f_stat, p_stat = stats[4], stats[5]
        stats_targets['NT'] = f_stat
        for method in order:
            dataframe = Artifact.load(matrices[method]).view(skbio.DistanceMatrix).to_data_frame()
            curr_dmx = list(dataframe.loc[relevant_ids, relevant_ids].values)
            stats = list(permanova(DistanceMatrix(curr_dmx), grouping))
            f_stat, p_stat = stats[4], stats[5]
            stats_targets[method] = f_stat
        for method, f_stat in stats_targets.items():
            f_plot_data.append(f_stat)
            f_labels.append('{} and {} {}'.format(t1, t2, method_dict[method]))
            x_range.append(x_index)
            x_index += 1
        x_index += 1

    # plot
    plt.table(cellText=[])
    plt.bar(x_range, f_plot_data)
    plt.xlim(0, x_range[-1] + 1)
    plt.xticks(x_range, labels=f_labels, rotation=30, ha='right')
    plt.savefig(dest + 'f_stat_permanova.png', bbox_inches='tight')
    plt.clf()


def lr_transform(artifact: Artifact, tree: Artifact, ratio_limit=6):
    df = artifact.view(pd.DataFrame)
    tree: skbio.TreeNode = utils.gneiss_tree(artifact, tree)
    samples = df.index.values.tolist()
    output = []
    ref_names = None
    for sample in samples:
        lrs = []
        names = []
        for node in tree.postorder():
            children = node.children
            if len(children) == 0:
                node.abundance = df.loc[sample, node.name]
            elif len(children) == 2:
                node.abundance = children[0].abundance + children[1].abundance
                names.append(node.name)
                if children[0].abundance + children[1].abundance == 0:
                    lr = 0
                elif children[1].abundance == 0:
                    lr = ratio_limit
                elif children[0].abundance == 0:
                    lr = -1 * ratio_limit
                else:

                    lr = log_ratio(children[0].abundance, children[1].abundance)
                    if lr > ratio_limit:
                        lr = ratio_limit
                    elif lr < -1 * ratio_limit:
                        lr = -1 * ratio_limit
                lrs.append(lr)

            else:
                raise TypeError('Tree is not bifurcated at inner node', node.name)
        if not ref_names:
            ref_names = names
        else:
            assert ref_names == names
        output.append(lrs)
        if np.isnan(np.sum(np.array(lrs))):
            print(sample, lrs)
            raise ValueError('nan value')
    return pd.DataFrame(index=samples, columns=names, data=output), tree


def lr_inverse(ratios: pd.DataFrame, tree: skbio.TreeNode):
    output = []
    ref_names = []
    tree.abundance = 1.0
    for sample in ratios.index:
        otus = []
        abundances = []
        for node in tree.preorder():
            children = node.children
            if len(children) == 0:
                otus.append(node.name)
                abundances.append(node.abundance)
            elif len(children) == 2:
                ratio = ratios.at[sample, node.name]
                left_abundance, right_abundance = inverse_log_ratio(ratio=ratio,
                                                                    total=node.abundance)
                children[0].abundance, children[1].abundance = left_abundance, right_abundance

        if not ref_names:
            ref_names = otus
        else:
            assert ref_names == otus
        output.append(abundances)

    output = pd.DataFrame(index=ratios.index, columns=otus, data=output)
    assert all([utils.equal(np.sum(abundances), 1) for _, abundances in output.iterrows()])
    return output


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

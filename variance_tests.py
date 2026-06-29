import os
import random
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly import subplots
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.spatial import distance
from scipy.stats import ranksums
from skbio.diversity import beta_diversity
from skbio.stats.distance import permanova
from skbio.stats.ordination import pcoa
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import data_utils


class Metrics:
    def __init__(self):
        raise NotImplementedError()

    @staticmethod
    def get_permanova_results(data, group_col, distance_matrix=None):
        if distance_matrix is None:
            distance_matrix = beta_diversity(metric="braycurtis", counts=data.values, ids=data.index)

        permanova_results = permanova(distance_matrix, group_col)
        return permanova_results

    @staticmethod
    def calc_frac_idx(x1_mat ,x2_mat, should_use_braycurtis=False):
        """
        Returns fraction closer than true match for each sample (for each domain, as an ndarray).
        """
        metric = 'braycurtis' if should_use_braycurtis else 'euclidean'
        distances = distance.cdist(x1_mat, x2_mat, metric=metric)
        true_d = distances.diagonal()
        closer_x1 = (distances < true_d[:, None]).sum(axis=1)
        closer_x2 = (distances < true_d[None, :]).sum(axis=0)
        fracs_x1 = closer_x1 / (x1_mat.shape[0] - 1)
        fracs_x2 = closer_x2 / (x2_mat.shape[0] - 1)
        return fracs_x1, fracs_x2

    @staticmethod
    def calc_domain_avg_FOSCTTM(x1_mat, x2_mat, should_use_braycurtis=False):
        """
        Based on code from SCOTv1.
        Outputs average FOSCTTM measure (averaged over both domains)
        Averages the fractions in both directions for each data point
        """
        fracs_x1, fracs_x2 = Metrics.calc_frac_idx(x1_mat, x2_mat, should_use_braycurtis=should_use_braycurtis)
        return (fracs_x1 + fracs_x2) / 2

    @staticmethod
    def titration(source_dataset, target_dataset, projected_dataset, repeats=10, png_name='titration.png', seed=data_utils.PROJECT_SEED):
        """
        Based on code from Guy Shur's thesis.
        """
        return
        def _compute_pvals_from_testing_set(combined_otu_data, testing_set, half_set_size):
            curr_pvals = [ranksums(combined_otu_data[testing_set[:half_set_size], otu], combined_otu_data[testing_set[half_set_size:], otu])[1]
                          for otu in range(combined_otu_data.shape[1])]
            curr_pvals = -1 * np.log(curr_pvals)
            return curr_pvals / repeats

        def _get_titration_results(combined_data, repeats, seed):
            sample_ids = data_utils.get_sample_ids_by_dataset(combined_data)
            combined_otu_data = data_utils.round_dataframe(5, combined_data[data_utils.get_otu_columns(combined_data)]).to_numpy()
            sample_id_to_index = {sid: i for i, sid in enumerate(combined_data.index)}

            datasets = list(sample_ids.keys())
            controls_1 = [sample_id_to_index[sid] for sid in sample_ids[datasets[0]]['control']]
            controls_2 = [sample_id_to_index[sid] for sid in sample_ids[datasets[1]]['control']]
            set_size = min(len(controls_1), len(controls_2))  # set_size is "2q" in Guy Shur's thesis description
            if set_size % 2 == 1:
                set_size -= 1
            half_set_size = int(set_size / 2)

            titration_results = np.zeros((set_size + 1, combined_otu_data.shape[1]))

            for k in range(repeats):
                print(f'titration iteration {k+1} of {repeats}')
                import time
                start = time.time()

                random.seed(seed + k)
                random.shuffle(controls_1)
                random.shuffle(controls_2)

                testing_set = controls_1[:set_size]
                replacement_set = controls_2[:set_size]

                titration_results[0] += _compute_pvals_from_testing_set(combined_otu_data, testing_set, half_set_size)

                for l in range(set_size):
                    testing_set[l] = replacement_set[l]
                    titration_results[l + 1] += _compute_pvals_from_testing_set(combined_otu_data, testing_set, half_set_size)

                end = time.time()
                print(f"iteration took {end - start} seconds")

            return titration_results

        combined_before = pd.concat([target_dataset, source_dataset])
        combined_before.fillna(0.0, inplace=True)
        combined_before.set_index('sample_id', inplace=True)

        print("\nCalculating titration before transport")
        titration_results_before = _get_titration_results(combined_before, repeats, seed)

        combined_after = pd.concat([target_dataset, projected_dataset])
        combined_after.set_index('sample_id', inplace=True)

        print("\nCalculating titration after transport")
        titration_results_after = _get_titration_results(combined_after, repeats, seed)

        Draw.draw_titration_results_before_and_after(titration_results_before, titration_results_after, png_name)
        print(f"titration results saved in {png_name}")

    @staticmethod
    def RF_classify(train_dataset, test_dataset, label, seed=data_utils.PROJECT_SEED):
        """
        test the ability of a random forest classifier to classify samples based on their OTU relative abundance.
        returns the accuracy and AUC ROC of the classifier on the test set.
        """
        train_data = train_dataset[data_utils.get_otu_columns(train_dataset)]
        train_phenotype = train_dataset[label]
        test_data = test_dataset[data_utils.get_otu_columns(test_dataset)]
        test_phenotype = test_dataset[label]

        classifier = RandomForestClassifier(random_state=seed)
        classifier.fit(train_data, train_phenotype)

        label0 = classifier.classes_[0]
        pred = classifier.predict(test_data)
        acc = (pred == test_phenotype).mean()
        probability_scores = classifier.predict_proba(test_data)[:, classifier.classes_ == label0]
        auc_roc = metrics.roc_auc_score((test_phenotype == label0).astype(int), probability_scores)
        return acc, auc_roc
    
    @staticmethod
    def run_dataset_classifier(data, dataset_labels, iterations=30, seed=data_utils.PROJECT_SEED):
        """
        Runs a KNN classifier to classify samples based on their OTU relative abundance, using the dataset labels as the target.
        Returns the accuracy and AUC ROC of the classifier on the test set, averaged over the specified number of iterations.
        The seed is used to shuffle the data before each iteration, and is incremented by 1 for each iteration.
        """
        classifier = KNeighborsClassifier()
        acc_list = []
        auc_roc_list = []
        for i in range(iterations):
            train_data, test_data, train_dataset_labels, test_dataset_labels = model_selection.train_test_split(
                data, dataset_labels, test_size=0.3, random_state=seed + i
            )
            classifier.fit(train_data, train_dataset_labels)

            true_label = dataset_labels.iloc[0]  # get one of the dataset labels, the other will be "false" i.e. 0.
            pred = classifier.predict(test_data)
            acc = (pred == test_dataset_labels).mean()
            probability_scores = classifier.predict_proba(test_data)[:, classifier.classes_ == true_label]
            auc_roc = metrics.roc_auc_score((test_dataset_labels == true_label).astype(int), probability_scores)

            acc_list.append(acc)
            auc_roc_list.append(auc_roc)

        return acc_list, auc_roc_list

class Draw:
    def heatmap(data, row_labels, col_labels, **kwargs):
        ax = plt.gca()
        im = ax.imshow(data, aspect='auto', **kwargs)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(data.shape[1]), labels=col_labels, rotation=-90, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(data.shape[0]), labels=row_labels)
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        plt.tight_layout()
        plt.show()


    def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
        """
        Add a covariance confidence ellipse to an Axes.
        Parameters
        ----------
        x, y : arrays
            The data points.
        ax : matplotlib Axes
            The Axes object to draw into.
        n_std : float
            Number of standard deviations (2 ≈ 95%, 3 ≈ 99.7%).
        """
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        cov = np.cov(x, y)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height of ellipse = 2 * n_std * sqrt(eigenvalues)
        width, height = 2 * n_std * np.sqrt(vals)
        mean_x, mean_y = np.mean(x), np.mean(y)

        ell = patches.Ellipse((mean_x, mean_y), width, height, angle=theta, facecolor=facecolor, **kwargs)
        ax.add_patch(ell)


    def run_pcoa(data, group_col, file_path=None, should_show_pcoa=True, seed=data_utils.PROJECT_SEED, distance_matrix=None, pcoa_pairs=None, subtitle=None):
        if distance_matrix is None:
            distance_matrix = beta_diversity(metric="braycurtis", counts=data.values, ids=data.index)

        ordination_results = pcoa(distance_matrix, seed=seed)
        mod = ordination_results.samples.iloc[:, :2].values

        plt.figure(figsize=(10,8))

        colors = plt.get_cmap("tab10")
        for group in np.unique(group_col):
            idx = (group_col == group)
            color = colors(np.unique(group_col).tolist().index(group) % 10)
            Draw.confidence_ellipse(mod[idx, 0], mod[idx, 1], plt.gca(), edgecolor='black', alpha=0.2, facecolor=color)
            plt.scatter(mod[idx, 0], mod[idx, 1], label=group, alpha=0.75, color=color)

        if pcoa_pairs is not None:
            for i, j in pcoa_pairs:
                plt.plot([mod[i, 0], mod[j, 0]], [mod[i, 1], mod[j, 1]], alpha=0.75, color='grey')

        plt.legend(title=group_col.name)
        plt.suptitle("PCoA of OTU Relative Abundance")
        plt.title(subtitle, fontsize=10)
        if file_path is not None:
            plt.savefig(file_path, dpi=500)
        if should_show_pcoa:
            plt.show()
        plt.close()

    def _draw_titration_internal(fig, all_titration_results):
        row = 1  # working with single row
        max_p_val = 0

        for i in range(len(all_titration_results)):
            titration_results = all_titration_results[i]
            x = []
            means = []
            medians = []
            for l in range(len(titration_results)):
                curr_pvals = titration_results[l]
                fig.add_trace(go.Scatter(x=[l for _ in curr_pvals], y=curr_pvals, mode='markers',
                              marker=dict(opacity=0.2, color='DarkGrey', size=2, line=dict(color='Black', width=1))),
                              col=i+1, row=row)

                means.append(np.mean(curr_pvals))
                medians.append(np.median(curr_pvals))
                x.append(l)
                max_p_val = max(max_p_val, max(curr_pvals))

            fig.add_trace(go.Scatter(x=x, y=means, mode='lines', marker=dict(color='Blue', size=2)), col=i+1, row=row)
            fig.add_trace(go.Scatter(x=x, y=medians, mode='lines', marker=dict(color='Red', size=2)), col=i+1, row=row)

        set_size = len(all_titration_results[0])  # they should all be the same size
        fig.update_xaxes(tickmode='array',
                         tickvals=[0, 0.25 * set_size, 0.5 * set_size, 0.75 * set_size, set_size],
                         ticktext=['0%', '50%', '100%', '50%', '0%'])
        fig.update_yaxes(range=[0, 1.05 * max_p_val])
        fig.update_layout(showlegend=False)

    def draw_titration_results_before_and_after(titration_results_before, titration_results_after, png_name='titration.png'):
        """
        Based on code from Guy Shur's thesis.
        """
        fig = subplots.make_subplots(cols=2, print_grid=False, vertical_spacing=0.13, subplot_titles=["Before transport", "After transport"])
        Draw._draw_titration_internal(fig, [titration_results_before, titration_results_after])
        fig.write_image(png_name, height=600, width=1200, scale=6, format='png')

    def draw_titration_results(titration_results, png_name='titration.png'):
        """
        Based on code from Guy Shur's thesis.
        """
        fig = subplots.make_subplots(print_grid=False, vertical_spacing=0.13)
        Draw._draw_titration_internal(fig, [titration_results])
        fig.write_image(png_name, height=600, width=800, scale=6, format='png')


def show_variance(data, group_col_name, file_path=None, should_run_pcoa=True, should_show_pcoa=True, pcoa_pairs=None, seed=data_utils.PROJECT_SEED):
    """
    Shows variance between groups in data, grouped by group_col_name.
    Writes PERMANOVA results to file_path+'_permanova.txt'.
    Draws PCoA plot to file_path+'.png' if should_run_pcoa is True. Pairs in pcoa_pairs are connected with lines.
    """
    np.random.seed(seed)
    otu_data = data[data_utils.get_otu_columns(data)]
    distance_matrix = beta_diversity(metric="braycurtis", counts=otu_data.values, ids=otu_data.index)
    permanova_results = Metrics.get_permanova_results(otu_data, data[group_col_name], distance_matrix=distance_matrix)

    print(f"PERMANOVA results:\n{permanova_results}\n")
    if file_path is not None:
        with open(file_path+'_permanova.txt', 'w') as f:
            f.write(f"PERMANOVA results:\n{permanova_results}\n")

    subtitle = f"PERMANOVA pvalue = {permanova_results['p-value']:.3f}, PERMANOVA statistic = {permanova_results['test statistic']:.3f}"

    # it's slow... so optional
    if should_run_pcoa:
        pcoa_file_path = file_path + '.png' if file_path is not None else None
        Draw.run_pcoa(otu_data, data[group_col_name], pcoa_file_path, should_show_pcoa=should_show_pcoa, distance_matrix=distance_matrix, pcoa_pairs=pcoa_pairs, subtitle=subtitle)


def _combine_dataset(dataset1, dataset2):
    """
    Combines two datasets by concatenating them and filling missing values with 0.0.
    """
    combined_data = pd.concat([dataset1, dataset2])
    combined_data.fillna(0.0, inplace=True)
    combined_data.set_index('sample_id', inplace=True)
    return combined_data


def _report_phenotypic_signal_in_dataset(dataset, dataset_name, file_path, seed=data_utils.PROJECT_SEED):
    with open(file_path, 'w') as f:
        train_dataset, test_dataset = model_selection.train_test_split(dataset, test_size=0.3, random_state=seed)
        acc, auc_roc = Metrics.RF_classify(train_dataset, test_dataset, 'phenotype')
        f.write(f'Testing phenotypic signal in {dataset_name} dataset using Random Forest Classifier\n')
        f.write(f'Classification results - Accuracy: {acc:.3f}, AUC-ROC: {auc_roc:.3f}\n')


def main():
    np.random.seed(data_utils.PROJECT_SEED)
    os.makedirs(os.path.join('results', 'datasets'), exist_ok=True)

    # risk data
    print("RISK data:")
    risk_data = pd.read_csv("risk_data.csv")
    show_variance(risk_data, 'phenotype', file_path=os.path.join('results', 'datasets', 'risk_by_phenotype'), should_run_pcoa=True, should_show_pcoa=True)
    _report_phenotypic_signal_in_dataset(risk_data, 'RISK', os.path.join('results', 'datasets', 'risk_phenotypic_signal.txt'))


    # mucosalibd data
    print("MucosalIBD data:")
    mucosalibd_data = pd.read_csv("mucosalibd_data.csv")
    show_variance(mucosalibd_data, 'phenotype', file_path=os.path.join('results', 'datasets', 'mucosalibd_by_phenotype'), should_run_pcoa=True, should_show_pcoa=True)
    _report_phenotypic_signal_in_dataset(mucosalibd_data, 'MucosalIBD', os.path.join('results', 'datasets', 'mucosalibd_phenotypic_signal.txt'))
    # combined
    risk_data_copy = risk_data.copy()
    mucosalibd_data_copy = mucosalibd_data.copy()

    print("PERMANOVA between datasets:")
    risk_data_copy['dataset+phenotype'] = 'RISK_' + risk_data_copy['phenotype']
    mucosalibd_data_copy['dataset+phenotype'] = 'MucosalIBD_' + mucosalibd_data_copy['phenotype']
    combined_data = _combine_dataset(risk_data_copy, mucosalibd_data_copy)
    show_variance(combined_data, 'dataset+phenotype', file_path=os.path.join('results', 'datasets', 'combined_risk_mucosalibd_by_dataset_and_phenotype'), should_run_pcoa=True, should_show_pcoa=True)

    # # test only controls
    # combined_data = combined_data[combined_data['phenotype'] == 'control']

    # test by dataset
    risk_data_copy = risk_data.copy()
    mucosalibd_data_copy = mucosalibd_data.copy()

    risk_data_copy['dataset'] = 'RISK'
    mucosalibd_data_copy['dataset'] = 'MucosalIBD'
    combined_data = _combine_dataset(risk_data_copy, mucosalibd_data_copy)
    show_variance(combined_data, 'dataset', file_path=os.path.join('results', 'datasets', 'combined_risk_mucosalibd_by_dataset'), should_run_pcoa=True, should_show_pcoa=True)

    # # test dataset and phenotype separately
    # show_variance(combined_data, 'dataset')
    # show_variance(combined_data, 'phenotype')

    # iHMP data
    print("iHMP data:")
    ihmp_data = pd.read_csv("ihmp_data.csv")
    show_variance(ihmp_data, 'phenotype', file_path=os.path.join('results', 'datasets', 'ihmp_by_phenotype'), should_run_pcoa=True, should_show_pcoa=True)
    _report_phenotypic_signal_in_dataset(ihmp_data, 'iHMP', os.path.join('results', 'datasets', 'ihmp_phenotypic_signal.txt'))

    # FRANZOSA data
    print("FRANZOSA data:")
    franzosa_data = pd.read_csv("franzosa_data.csv")
    show_variance(franzosa_data, 'phenotype', file_path=os.path.join('results', 'datasets', 'franzosa_by_phenotype'), should_run_pcoa=True, should_show_pcoa=True)
    _report_phenotypic_signal_in_dataset(franzosa_data, 'FRANZOSA', os.path.join('results', 'datasets', 'franzosa_phenotypic_signal.txt'))

    # combined iHMP and FRANZOSA
    ihmp_data_copy = ihmp_data.copy()
    franzosa_data_copy = franzosa_data.copy()

    print("PERMANOVA between iHMP and FRANZOSA datasets:")
    ihmp_data_copy['dataset+phenotype'] = 'iHMP_' + ihmp_data_copy['phenotype']
    franzosa_data_copy['dataset+phenotype'] = 'FRANZOSA_' + franzosa_data_copy['phenotype']
    combined_data = _combine_dataset(ihmp_data_copy, franzosa_data_copy)
    show_variance(combined_data, 'dataset+phenotype', file_path=os.path.join('results', 'datasets', 'combined_ihmp_franzosa_by_dataset_and_phenotype'), should_run_pcoa=True, should_show_pcoa=True)

    # test by dataset
    ihmp_data_copy = ihmp_data.copy()
    franzosa_data_copy = franzosa_data.copy()
    ihmp_data_copy['dataset'] = 'iHMP'
    franzosa_data_copy['dataset'] = 'FRANZOSA'
    combined_data = _combine_dataset(ihmp_data_copy, franzosa_data_copy)
    show_variance(combined_data, 'dataset', file_path=os.path.join('results', 'datasets', 'combined_ihmp_franzosa_by_dataset'), should_run_pcoa=True, should_show_pcoa=True)

    print("Done.")



if __name__ == "__main__":
    main()


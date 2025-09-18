import random
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.spatial import distance
from scipy.stats import ranksums
from skbio.diversity import beta_diversity
from skbio.stats.distance import permanova
from skbio.stats.ordination import pcoa

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
    def titration(combined_data, repeats=10, png_name='titration.png', seed=data_utils.PROJECT_SEED):
        """
        Based on code from Guy Shur's thesis.
        """
        def _compute_pvals_from_testing_set(combined_otu_data, testing_set, half_set_size):
            curr_pvals = [ranksums(combined_otu_data[testing_set[:half_set_size], otu], combined_otu_data[testing_set[half_set_size:], otu])[1]
                          for otu in range(combined_otu_data.shape[1])]
            curr_pvals = -1 * np.log(curr_pvals)
            return curr_pvals / repeats


        sample_ids = data_utils.get_sample_ids_by_dataset(combined_data)
        combined_otu_data = combined_data[data_utils.get_otu_columns(combined_data)].to_numpy()
        sample_id_to_index = {sid: i for i, sid in enumerate(combined_data.index)}

        datasets = list(sample_ids.keys())
        controls_1 = [sample_id_to_index[sid] for sid in sample_ids[datasets[0]]['control']]
        controls_2 = [sample_id_to_index[sid] for sid in sample_ids[datasets[1]]['control']]
        set_size = min(len(controls_1), len(controls_2))
        if set_size % 2 == 1:
            set_size -= 1
        half_set_size = int(set_size / 2)

        titration_results = np.zeros((set_size + 1, combined_otu_data.shape[1]))

        for k in range(repeats):
            print('titration iteration', k)
            import time
            start = time.time()

            random.seed(data_utils.PROJECT_SEED + k)
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

        Draw.draw_titration_results(titration_results, set_size, png_name=png_name)


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


    def run_pcoa(data, group_col, seed=data_utils.PROJECT_SEED, distance_matrix=None, pcoa_pairs=None, subtitle=None):
        if distance_matrix is None:
            distance_matrix = beta_diversity(metric="braycurtis", counts=data.values, ids=data.index)

        ordination_results = pcoa(distance_matrix, seed=seed)
        mod = ordination_results.samples.iloc[:, :2].values

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
        plt.show()

    def draw_titration_results(titration_results, set_size, png_name='titration.png'):
        """
        Based on code from Guy Shur's thesis.
        """
        fig = go.Figure()
        x = []
        means = []
        medians = []
        max_p_val = 0
        maximal_mean = 0
        for l in range(len(titration_results)):
            curr_pvals = titration_results[l]
            fig.add_trace(go.Scatter(x=[l for _ in curr_pvals], y=curr_pvals, mode='markers',
                          marker=dict(opacity=0.2, color='DarkGrey', size=2, line=dict(color='Black', width=1))))

            mean = np.mean(curr_pvals)
            means.append(mean)
            maximal_mean = max(mean, maximal_mean)
            medians.append(np.median(curr_pvals))
            x.append(l)
            max_p_val = max(max_p_val, max(curr_pvals))
            fig.add_trace(go.Scatter(x=x, y=means, mode='lines', marker=dict(color='Blue', size=2)))
            fig.add_trace(go.Scatter(x=x, y=medians, mode='lines', marker=dict(color='Red', size=2)))

        fig.update_xaxes(tickmode='array',
                         tickvals=[0, 0.25 * set_size, 0.5 * set_size, 0.75 * set_size, set_size],
                         ticktext=['0%', '50%', '100%', '50%', '0%'])
        fig.update_yaxes(range=[0, 1.05 * max_p_val])
        fig.update_layout(showlegend=False)
        fig.write_image(png_name, height=600, width=800, scale=6, format='png')


def show_variance(data, group_col_name, should_run_pcoa=True, pcoa_pairs=None):
    otu_data = data[data_utils.get_otu_columns(data)]
    distance_matrix = beta_diversity(metric="braycurtis", counts=otu_data.values, ids=otu_data.index)
    permanova_results = Metrics.get_permanova_results(otu_data, data[group_col_name], distance_matrix=distance_matrix)
    print(f"PERMANOVA results:\n{permanova_results}\n")
    subtitle = f"PERMANOVA pvalue = {permanova_results['p-value']:.3f}"

    # it's slow... so optional
    if should_run_pcoa:
        Draw.run_pcoa(otu_data, data[group_col_name], distance_matrix=distance_matrix, pcoa_pairs=pcoa_pairs, subtitle=subtitle)


def main():
    # risk data
    print("RISK data:")
    risk_data = pd.read_csv("risk_data.csv")
    show_variance(risk_data, 'phenotype', should_run_pcoa=False)

    # mucosalibd data
    print("MucosalIBD data:")
    mucosalibd_data = pd.read_csv("mucosalibd_data.csv")
    show_variance(mucosalibd_data, 'phenotype', should_run_pcoa=False)

    # combined
    print("PERMANOVA between datasets:")
    risk_data['dataset+phenotype'] = 'RISK_' + risk_data['phenotype']
    mucosalibd_data['dataset+phenotype'] = 'MucosalIBD_' + mucosalibd_data['phenotype']
    combined_data = pd.concat([risk_data, mucosalibd_data])
    combined_data.fillna(0.0, inplace=True)
    combined_data.set_index('sample_id', inplace=True)

    # # test only controls
    # combined_data = combined_data[combined_data['phenotype'] == 'control']

    show_variance(combined_data, 'dataset+phenotype')

    # # test dataset and phenotype separately
    # show_variance(combined_data, 'dataset')
    # show_variance(combined_data, 'phenotype')
    
    print("Done.")



if __name__ == "__main__":
    main()


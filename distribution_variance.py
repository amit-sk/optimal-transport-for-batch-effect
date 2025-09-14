import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.spatial.distance import pdist, squareform
from skbio.diversity import beta_diversity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from skbio.stats.distance import permanova
from skbio.stats.ordination import pcoa

import data_utils


def get_permanova_results(data, group_col, distance_matrix=None):
    if distance_matrix is None:
        distance_matrix = beta_diversity(metric="braycurtis", counts=data.values, ids=data.index)

    permanova_results = permanova(distance_matrix, group_col)
    return permanova_results


def calc_frac_idx(x1_mat ,x2_mat):
    """
    Based on code from SCOTv1.
    Returns fraction closer than true match for each sample (as an array).
    """
    fracs = []
    nsamp = x1_mat.shape[0]
    rank = 0

    for row_idx in range(nsamp):
        euc_dist = np.sqrt(np.sum(np.square(np.subtract(x1_mat[row_idx,:], x2_mat)), axis=1))
        true_nbr = euc_dist[row_idx]
        sort_euc_dist = sorted(euc_dist)
        rank =sort_euc_dist.index(true_nbr)
        frac = float(rank)/(nsamp -1)

        fracs.append(frac)

    return fracs

def calc_domain_avg_FOSCTTM(x1_mat, x2_mat):
    """
    Based on code from SCOTv1.
    Outputs average FOSCTTM measure (averaged over both domains)
    Get the fraction matched for all data points in both directions
    Averages the fractions in both directions for each data point
    """
    fracs1 = calc_frac_idx(x1_mat, x2_mat)
    fracs2 = calc_frac_idx(x2_mat, x1_mat)
    fracs = []
    for i in range(len(fracs1)):
        fracs.append((fracs1[i] + fracs2[i]) / 2)  
    return fracs


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

    ell = patches.Ellipse((mean_x, mean_y), width, height, angle=theta,
                  facecolor=facecolor, **kwargs)
    ax.add_patch(ell)


def run_pcoa(data, group_col, seed=data_utils.PROJECT_SEED, distance_matrix=None, pcoa_pairs=None):
    if distance_matrix is None:
        distance_matrix = beta_diversity(metric="braycurtis", counts=data.values, ids=data.index)

    ordination_results = pcoa(distance_matrix, seed=seed)
    mod = ordination_results.samples.iloc[:, :2].values

    colors = plt.get_cmap("tab10")
    for group in np.unique(group_col):
        idx = (group_col == group)
        color = colors(np.unique(group_col).tolist().index(group) % 10)
        confidence_ellipse(mod[idx, 0], mod[idx, 1], plt.gca(), edgecolor='black', alpha=0.2, facecolor=color)
        plt.scatter(mod[idx, 0], mod[idx, 1], label=group, alpha=0.75, color=color)

    if pcoa_pairs is not None:
        for i, j in pcoa_pairs:
            plt.plot([mod[i, 0], mod[j, 0]], [mod[i, 1], mod[j, 1]], alpha=0.75, color='grey')

    plt.legend(title=group_col.name)
    plt.title("PCoA of OTU Relative Abundance")
    plt.show()


def show_variance(data, group_col_name, should_run_pcoa=True, pcoa_pairs=None):
    otu_data = data[data_utils.get_otu_columns(data)]
    distance_matrix = beta_diversity(metric="braycurtis", counts=otu_data.values, ids=otu_data.index)
    permanova_results = get_permanova_results(otu_data, data[group_col_name], distance_matrix=distance_matrix)
    print(f"PERMANOVA results:\n{permanova_results}\n")

    # it's slow... so optional
    if should_run_pcoa:
        run_pcoa(otu_data, data[group_col_name], distance_matrix=distance_matrix, pcoa_pairs=pcoa_pairs)


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


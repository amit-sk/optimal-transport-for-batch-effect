import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import ot

import data_utils
import distribution_variance


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


def sanity_check():
    risk_data = pd.read_csv("risk_data.csv")
    risk_otu_data = risk_data[data_utils.get_otu_columns(risk_data)]
    risk_distance_matrix = squareform(pdist(risk_otu_data.values, metric='braycurtis'))

    noisy_data = data_utils.create_noisy_data(risk_data, proportion_of_std=0.1)
    noisy_data = data_utils.renormalize_data(noisy_data)
    noisy_otu_data = noisy_data[data_utils.get_otu_columns(noisy_data)]
    noisy_distance_matrix = squareform(pdist(noisy_otu_data.values, metric='braycurtis'))

    # create combined data
    risk_data['dataset'] = 'orig'
    risk_data['sample_id'] = risk_data['sample_id'] + '_orig'
    noisy_data['dataset'] = 'noisy'
    noisy_data['sample_id'] = noisy_data['sample_id'] + '_noisy'
    combined_data = pd.concat([risk_data, noisy_data])
    combined_data.set_index('sample_id', inplace=True)
    indexes = combined_data.index
    pairs = [(indexes.get_loc(i), indexes.get_loc(i.replace('_orig','_noisy'))) for i in indexes if i.endswith('_orig')]

    # show variance before alignment
    print("\nComparing variance between original and noisy (before alignment):")
    distribution_variance.show_variance(combined_data, 'dataset', pcoa_pairs=pairs, should_run_pcoa=False)
    fracs = distribution_variance.calc_domain_avg_FOSCTTM(risk_otu_data.values, noisy_otu_data.values, should_use_braycurtis=True)
    print(f"Average FOSCTTM score between noisy and original: {fracs.mean()}")

    # transport
    print("\nRunning GW transport...")
    coupling, log = ot.gromov.gromov_wasserstein(risk_distance_matrix, noisy_distance_matrix, verbose=False, log=True)
    gw_distance = log['gw_dist']
    print(f'GW distance: {gw_distance}')
    print(f'coupling diagonal sum: {coupling.diagonal().sum()}')

    projected = barycentric_projection(coupling, risk_otu_data, x_onto_y=False)
    fracs = distribution_variance.calc_domain_avg_FOSCTTM(risk_otu_data.values, projected.values, should_use_braycurtis=True)
    print(f"Average FOSCTTM score between projected and original (post transport): {fracs.mean()}")

    projected['sample_id'] = noisy_data['sample_id'].str.replace('_noisy','_projected')
    projected['dataset'] = 'projected'
    projected['phenotype'] = noisy_data['phenotype']

    # compare projection and original (post transport)
    print("\nComparing variance between original and projected:")
    combined_data = pd.concat([risk_data, projected])
    combined_data.set_index('sample_id', inplace=True)
    indexes = combined_data.index
    pairs = [(indexes.get_loc(i), indexes.get_loc(i.replace('_orig','_projected'))) for i in indexes if i.endswith('_orig')]
    distribution_variance.show_variance(combined_data, 'dataset', pcoa_pairs=pairs)

    # compare projection and noisy (before and after transport)
    print("\nComparing variance between noisy and projected:")
    combined_data = pd.concat([noisy_data, projected])
    combined_data.set_index('sample_id', inplace=True)
    indexes = combined_data.index
    pairs = [(indexes.get_loc(i), indexes.get_loc(i.replace('_noisy','_projected'))) for i in indexes if i.endswith('_noisy')]
    distribution_variance.show_variance(combined_data, 'dataset', pcoa_pairs=pairs)

    # how much each projection had moved - compare orig, noisy, projected
    print("\nComparing variance between original, noisy and projected:")
    combined_data = pd.concat([risk_data, noisy_data, projected])
    combined_data.set_index('sample_id', inplace=True)
    indexes = combined_data.index
    pairs = [(indexes.get_loc(i), indexes.get_loc(i.replace('_noisy','_projected'))) for i in indexes if i.endswith('_noisy')]
    pairs.extend([(indexes.get_loc(i), indexes.get_loc(i.replace('_noisy','_orig'))) for i in indexes if i.endswith('_noisy')])
    distribution_variance.show_variance(combined_data, 'dataset', pcoa_pairs=pairs)

    print("Done.")


def _observe_coupling_matrix(coupling, risk_data, mucosalibd_data):
    import matplotlib.pyplot as plt
    d_all = {}
    c = pd.DataFrame(coupling) * 132  # sum of columns will be 1, sum of all columns will be 132
    for i in range(132):
        vals = c.iloc[:, i].value_counts()
        d = vals.to_dict()
        d.pop(0)
        for k,v in d.items():
            if k in d_all:
                d_all[k] += v
            else:
                d_all[k] = v

    plt.bar(d_all.keys(), d_all.values(), width=1/132)
    plt.bar(d_all.keys(), d_all.values(), width=1/200)
    plt.bar(d_all.keys(), d_all.values(), width=1/2000, color='black')
    plt.yticks(np.arange(0, max(d_all.values())+1, 2.0))
    plt.xticks(np.arange(0, max(d_all.keys())+0.01, 0.005))
    plt.title("Histogram of coupling matrix values (non-zero only)")
    plt.show()

    distribution_variance.heatmap(coupling, risk_data['sample_id'], mucosalibd_data['sample_id'], cmap='Blues')

    # show spread - how many *unique* values in distribution
    spread_of_src = pd.DataFrame(coupling).nunique(axis=0)
    plt.hist(spread_of_src, bins=12)
    plt.title("Spread of each sample in src dataset (mucosalibd) in coupling matrix")
    plt.show()

    spread_of_src = pd.DataFrame(coupling).nunique(axis=1)
    plt.hist(spread_of_src, bins=12)
    plt.title("Spread of each sample in target dataset (risk) in coupling matrix")
    plt.show()


def _compare_datasets_pre_transport(risk_data, mucosalibd_data):
    # create combined data (to show variance before alignment)
    risk_data['dataset'] = 'risk'
    risk_data['sample_id'] = risk_data['sample_id'] + '_risk'
    mucosalibd_data['dataset'] = 'mucosalibd'
    mucosalibd_data['sample_id'] = mucosalibd_data['sample_id'] + '_mucosalibd'
    combined_data = pd.concat([risk_data, mucosalibd_data])
    combined_data.fillna(0.0, inplace=True)
    combined_data.set_index('sample_id', inplace=True)

    # show variance before alignment
    print("\nComparing variance between risk and mucosalibd (before alignment):")
    distribution_variance.show_variance(combined_data, 'dataset')
    distribution_variance.show_variance(combined_data, 'phenotype')


def _compare_datasets_post_transport(risk_data, mucosalibd_data, projected):
    # compare projection and risk (post transport)
    print("\nComparing variance between risk and projected:")
    combined_data = pd.concat([risk_data, projected])
    # combined_data = pd.concat([risk_data[risk_data.phenotype == 'control'], projected[risk_data.phenotype == 'control']])  # compare only controls (healthy)
    combined_data.set_index('sample_id', inplace=True)
    distribution_variance.show_variance(combined_data, 'dataset')
    distribution_variance.show_variance(combined_data, 'phenotype')

    risk_data['dataset+phenotype'] = 'RISK_' + risk_data['phenotype']
    projected['dataset+phenotype'] = 'Projected_' + projected['phenotype']
    combined_data = pd.concat([risk_data, projected])
    combined_data.set_index('sample_id', inplace=True)
    distribution_variance.show_variance(combined_data, 'dataset+phenotype')

    # compare projection and mucosalibd (before and after transport)
    print("\nComparing variance between mucosalibd (original) and projected:")
    combined_data = pd.concat([mucosalibd_data, projected])
    combined_data.fillna(0.0, inplace=True)
    combined_data.set_index('sample_id', inplace=True)
    indexes = combined_data.index
    pairs = [(indexes.get_loc(i), indexes.get_loc(i.replace('_mucosalibd','_projected'))) for i in indexes if i.endswith('_mucosalibd')]
    distribution_variance.show_variance(combined_data, 'dataset', pcoa_pairs=pairs)

    # how much each projection had moved - compare risk, mucosalibd, projected
    print("\nComparing variance between risk, mucosalibd and projected:")
    combined_data = pd.concat([risk_data, mucosalibd_data, projected])
    combined_data.fillna(0.0, inplace=True)
    combined_data.set_index('sample_id', inplace=True)
    indexes = combined_data.index
    pairs = [(indexes.get_loc(i), indexes.get_loc(i.replace('_mucosalibd','_projected'))) for i in indexes if i.endswith('_mucosalibd')]
    distribution_variance.show_variance(combined_data, 'dataset', pcoa_pairs=pairs)


def transport_test():
    # obtain data
    risk_data = pd.read_csv("risk_data.csv")
    risk_otu_data = risk_data[data_utils.get_otu_columns(risk_data)]
    risk_distance_matrix = squareform(pdist(risk_otu_data.values, metric='braycurtis'))

    mucosalibd_data = pd.read_csv("mucosalibd_data.csv")
    mucosalibd_otu_data = mucosalibd_data[data_utils.get_otu_columns(mucosalibd_data)]
    mucosalibd_distance_matrix = squareform(pdist(mucosalibd_otu_data.values, metric='braycurtis'))

    _compare_datasets_pre_transport(risk_data, mucosalibd_data)

    # transport
    print("\nRunning GW transport...")
    coupling, log = ot.gromov.gromov_wasserstein(risk_distance_matrix, mucosalibd_distance_matrix, verbose=False, log=True)
    gw_distance = log['gw_dist']
    print(f'GW distance: {gw_distance}')

    # check spread of coupling matrix
    _observe_coupling_matrix(coupling, risk_data, mucosalibd_data)

    # project mucosalibd onto risk (transport results)
    projected = barycentric_projection(coupling, risk_otu_data, x_onto_y=False)
    projected['sample_id'] = mucosalibd_data['sample_id'].str.replace('_mucosalibd','_projected')
    projected['dataset'] = 'projected'
    projected['phenotype'] = mucosalibd_data['phenotype']

    _compare_datasets_post_transport(risk_data, mucosalibd_data, projected)

    print("\nDone.")


def main():
    # sanity_check()
    transport_test()


if __name__ == "__main__":
    main()


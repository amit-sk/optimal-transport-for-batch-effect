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


def transport_test():
    risk_data = pd.read_csv("risk_data.csv")
    risk_otu_data = risk_data[data_utils.get_otu_columns(risk_data)]
    risk_distance_matrix = squareform(pdist(risk_otu_data.values, metric='braycurtis'))

    mucosalibd_data = pd.read_csv("mucosalibd_data.csv")
    mucosalibd_otu_data = mucosalibd_data[data_utils.get_otu_columns(mucosalibd_data)]
    mucosalibd_distance_matrix = squareform(pdist(mucosalibd_otu_data.values, metric='braycurtis'))

    # create combined data
    risk_data['dataset'] = 'risk'
    risk_data['sample_id'] = risk_data['sample_id'] + '_risk'
    mucosalibd_data['dataset'] = 'mucosalibd'
    mucosalibd_data['sample_id'] = mucosalibd_data['sample_id'] + '_mucosalibd'
    combined_data = pd.concat([risk_data, mucosalibd_data])
    combined_data.fillna(0.0, inplace=True)
    combined_data.set_index('sample_id', inplace=True)

    # show variance before alignment
    print("\nComparing variance between risk and mucosalibd (before alignment):")
    distribution_variance.show_variance(combined_data, 'dataset', should_run_pcoa=False)

    # transport
    print("\nRunning GW transport...")
    coupling, log = ot.gromov.gromov_wasserstein(risk_distance_matrix, mucosalibd_distance_matrix, verbose=False, log=True)
    gw_distance = log['gw_dist']
    print(f'GW distance: {gw_distance}')

    projected = barycentric_projection(coupling, risk_otu_data, x_onto_y=False)
    projected['sample_id'] = mucosalibd_data['sample_id'].str.replace('_mucosalibd','_projected')
    projected['dataset'] = 'projected'
    projected['phenotype'] = mucosalibd_data['phenotype']

    # compare projection and risk (post transport)
    print("\nComparing variance between risk and projected:")
    combined_data = pd.concat([risk_data, projected])
    # combined_data = pd.concat([risk_data[risk_data.phenotype == 'control'], projected[risk_data.phenotype == 'control']])
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

    print("Done.")


def main():
    # sanity_check()
    transport_test()


if __name__ == "__main__":
    main()


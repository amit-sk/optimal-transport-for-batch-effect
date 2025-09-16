import pandas as pd
import numpy as np
import ot
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform

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


def test_signal(source_dataset, target_dataset, projection):
    source_dataset = source_dataset.copy()
    target_dataset = target_dataset.copy()
    projection = projection.copy()

    source_dataset['dataset'] = 'source'
    target_dataset['dataset'] = 'target'
    projection['dataset'] = 'projection'

    combined = pd.concat([source_dataset, target_dataset, projection])
    combined.fillna(0.0, inplace=True)
    combined.set_index('sample_id', inplace=True)

    source_dataset = combined[combined['dataset'] == 'source']
    target_dataset = combined[combined['dataset'] == 'target']
    projection = combined[combined['dataset'] == 'projection']

    source_data = source_dataset[data_utils.get_otu_columns(source_dataset)]
    source_phenotype = source_dataset['phenotype']
    target_data = target_dataset[data_utils.get_otu_columns(target_dataset)]
    target_phenotype = target_dataset['phenotype']
    projection_data = projection[data_utils.get_otu_columns(projection)]
    projection_phenotype = projection['phenotype']

    classifier = RandomForestClassifier(random_state=data_utils.PROJECT_SEED)
    classifier.fit(target_data, target_phenotype)

    # show results on source data before transport
    source_pred = classifier.predict(source_data)
    source_acc = (source_pred == source_phenotype).mean()
    source_probability_scores = classifier.predict_proba(source_data)[:, classifier.classes_ == 'CD']
    source_auc_roc = metrics.roc_auc_score((source_phenotype == 'CD').astype(int), source_probability_scores)
    print(f"Source data before transport - Accuracy: {source_acc:.3f}, AUC-ROC: {source_auc_roc:.3f}")
    
    # show results on projection data after transport
    projection_pred = classifier.predict(projection_data)
    projection_acc = (projection_pred == projection_phenotype).mean()
    projection_probability_scores = classifier.predict_proba(projection_data)[:, classifier.classes_ == 'CD']
    projection_auc_roc = metrics.roc_auc_score((projection_phenotype == 'CD').astype(int), projection_probability_scores)
    print(f"Projection data after transport - Accuracy: {projection_acc:.3f}, AUC-ROC: {projection_auc_roc:.3f}")


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

    risk_data['dataset'] = 'risk'
    risk_data['sample_id'] = risk_data['sample_id'] + '_risk'
    mucosalibd_data['dataset'] = 'mucosalibd'
    mucosalibd_data['sample_id'] = mucosalibd_data['sample_id'] + '_mucosalibd'

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
    test_signal(risk_data, mucosalibd_data, projected)

    print("\nDone.")


def main():
    # sanity_check()
    transport_test()


if __name__ == "__main__":
    main()


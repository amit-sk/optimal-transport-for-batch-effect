import ot
import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

import optimal_transport
import data_utils

class OptimalTransferTest:
    def __init__(self, source_dataset, target_dataset, should_run_pcoa=False):
        """
        source_dataset: pandas dataframe of data being transported
        target_dataset: pandas dataframe of data being used as reference to be transported onto
        """
        self.should_run_pcoa = should_run_pcoa

        self.source_dataset = source_dataset
        self.source_otu_data = self.source_dataset[data_utils.get_otu_columns(self.source_dataset)]
        self.source_distance_matrix = distance.squareform(distance.pdist(self.source_otu_data.values, metric='braycurtis'))

        self.target_dataset = target_dataset
        self.target_otu_data = self.target_dataset[data_utils.get_otu_columns(self.target_dataset)]
        self.target_distance_matrix = distance.squareform(distance.pdist(self.target_otu_data.values, metric='braycurtis'))

        self.gw_distance = None
        self.coupling = None
        self.projected = None
        
    def show_variance_pre_transport(self):
        raise NotImplementedError()

    def show_variance_post_transport(self):
        raise NotImplementedError()

    def transport(self):
        self.coupling, log = ot.gromov.gromov_wasserstein(self.target_distance_matrix, self.source_distance_matrix, verbose=False, log=True)
        self.gw_distance = log['gw_dist']
        print(f'GW distance: {self.gw_distance}')
        self.projected = optimal_transport.barycentric_projection(self.coupling, self.target_otu_data, x_onto_y=False)

    def test_signal(self):
        source_dataset = self.source_dataset.copy()
        target_dataset = self.target_dataset.copy()
        projection = self.projected.copy()

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

    def run_test(self):
        print("Showing variance pre-transport...")
        self.show_variance_pre_transport()
        print("Running transport...")
        self.transport()
        print("Showing variance post-transport...")
        self.show_variance_post_transport()
        print("Testing signal...")
        self.test_signal()
        print("Test complete.")

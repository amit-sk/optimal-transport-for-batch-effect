import ot
import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

import optimal_transport
import data_utils

class OptimalTransferTest:
    def __init__(self, source_dataset, target_dataset, should_run_pcoa=False, source_dataset_name='source', target_dataset_name='target'):
        """
        source_dataset: pandas dataframe of data being transported
        target_dataset: pandas dataframe of data being used as reference to be transported onto
        """
        self.should_run_pcoa = should_run_pcoa

        self.source_dataset = source_dataset
        self.source_otu_data = self.source_dataset[data_utils.get_otu_columns(self.source_dataset)]
        self.source_distance_matrix = distance.squareform(distance.pdist(self.source_otu_data.values, metric='braycurtis'))

        self.source_dataset_name = source_dataset_name
        self.source_dataset['dataset'] = self.source_dataset_name
        self.source_dataset['sample_id'] = self.source_dataset['sample_id'] + '_' + self.source_dataset_name

        self.target_dataset = target_dataset
        self.target_otu_data = self.target_dataset[data_utils.get_otu_columns(self.target_dataset)]
        self.target_distance_matrix = distance.squareform(distance.pdist(self.target_otu_data.values, metric='braycurtis'))

        self.target_dataset_name = target_dataset_name
        self.target_dataset['dataset'] = self.target_dataset_name
        self.target_dataset['sample_id'] = self.target_dataset['sample_id'] + '_' + self.target_dataset_name

        self.gw_distance = None
        self.coupling = None
        self.projected_data = None
        self.projected_otu_data = None
        
    def show_variance_pre_transport(self):
        raise NotImplementedError()

    def show_variance_post_transport(self):
        raise NotImplementedError()

    def _get_projected(self):
        self.projected_otu_data = optimal_transport.barycentric_projection(self.coupling, self.target_otu_data, x_onto_y=False)

        self.projected_data = self.projected_otu_data.copy()
        self.projected_data['dataset'] = 'projected'
        self.projected_data['phenotype'] = self.source_dataset['phenotype']
        self.projected_data['sample_id'] = self.source_dataset['sample_id'].str.replace('_'+self.source_dataset_name ,'_projected')

    def transport(self):
        self.coupling, log = ot.gromov.gromov_wasserstein(self.target_distance_matrix, self.source_distance_matrix, verbose=False, log=True)
        self.gw_distance = log['gw_dist']
        print(f'GW distance: {self.gw_distance}')
        self._get_projected()

    def test_signal(self):
        # combine source, target, projected to unify columns (OTUs)
        combined = pd.concat([self.source_dataset, self.target_dataset, self.projected_data])
        combined.fillna(0.0, inplace=True)  # fill missing OTUs with relative abundance of 0
        combined.set_index('sample_id', inplace=True)

        source_dataset = combined[combined['dataset'] == self.source_dataset_name]
        target_dataset = combined[combined['dataset'] == self.target_dataset_name]
        projection = combined[combined['dataset'] == 'projected']

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
        print(f"Running test {self.__class__.__name__}...")
        print("Showing variance pre-transport...")
        self.show_variance_pre_transport()
        print("Running transport...")
        self.transport()
        print("Showing variance post-transport...")
        self.show_variance_post_transport()
        print("Testing signal...")
        self.test_signal()
        print("Test complete.")

    @staticmethod
    def _get_pairs(combined_data, suffix1, suffix2):
        """
        get indexes of pairs of samples that are the same except for the suffixes, e.g. from the source dataset and its projection.
        """
        indexes = combined_data.index
        pairs = [(indexes.get_loc(i), indexes.get_loc(i.replace(suffix1, suffix2))) for i in indexes if i.endswith(suffix1)]
        return pairs


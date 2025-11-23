import os
import ot
import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy import stats
from sklearn import metrics
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import optimal_transport
import data_utils

class OptimalTransportTest:
    def __init__(self, source_dataset, target_dataset, should_run_pcoa=False, should_show_pcoa=False, should_test_signal_retention=False, source_dataset_name='source', target_dataset_name='target', results_folder_name=None, **kwargs):
        """
        source_dataset: pandas dataframe of data being transported
        target_dataset: pandas dataframe of data being used as reference to be transported onto
        should_run_pcoa: whether to run PCoA plots to visualize data before and after transport
        should_show_pcoa: whether to show PCoA plots (or just write to file). is ignored if should_run_pcoa is False.
        should_test_signal_retention: whether to test retention of phenotype and age signal before and after transport.
        source_dataset_name: name of the source dataset (appears in the plots)
        target_dataset_name: name of the target dataset (appears in the plots)
        results_folder_name: folder where results will be saved. if None, defaults to 'results/<class name>'
        """
        self.should_run_pcoa = should_run_pcoa
        self.should_show_pcoa = should_show_pcoa
        self.should_test_signal_retention = should_test_signal_retention
        self.results_folder_name = os.path.join('results', self.__class__.__name__) if results_folder_name is None else results_folder_name

        self.source_dataset = source_dataset
        self.source_otu_data = self.source_dataset[data_utils.get_otu_columns(self.source_dataset)]
        self.source_distance_matrix = distance.squareform(distance.pdist(self.source_otu_data.values, metric='braycurtis'))

        self.source_dataset_name = source_dataset_name
        self.source_dataset['dataset'] = self.source_dataset_name
        self.source_dataset['sample_id'] = self.source_dataset['sample_id'] + '_' + self.source_dataset_name
        self.source_dataset['age'] = self.source_dataset['age']

        self.target_dataset = target_dataset
        self.target_otu_data = self.target_dataset[data_utils.get_otu_columns(self.target_dataset)]
        self.target_distance_matrix = distance.squareform(distance.pdist(self.target_otu_data.values, metric='braycurtis'))

        self.target_dataset_name = target_dataset_name
        self.target_dataset['dataset'] = self.target_dataset_name
        self.target_dataset['sample_id'] = self.target_dataset['sample_id'] + '_' + self.target_dataset_name
        self.target_dataset['age'] = self.target_dataset['age']

        self.gw_distance = None
        self.coupling = None
        self.projected_data = None
        self.projected_otu_data = None

    def _get_file_path(self, file_name):
        """
        get full file path in results folder for given file name.
        """
        return os.path.join(self.results_folder_name, file_name)
        
    def show_variance_pre_transport(self):
        raise NotImplementedError()

    def show_variance_post_transport(self):
        raise NotImplementedError()

    def _get_projected(self):
        self.projected_otu_data = optimal_transport.barycentric_projection(self.coupling, self.target_otu_data, x_onto_y=False)

        self.projected_data = self.projected_otu_data.copy()
        self.projected_data['dataset'] = 'projected'
        self.projected_data['phenotype'] = self.source_dataset['phenotype']
        self.projected_data['age'] = self.source_dataset['age']
        self.projected_data['sample_id'] = self.source_dataset['sample_id'].str.replace('_'+self.source_dataset_name ,'_projected')

    def transport(self, **kwargs_for_ot):
        with open(self._get_file_path('transport_log.txt'), 'w') as f:
            self.coupling, log = ot.gromov.gromov_wasserstein(self.target_distance_matrix, self.source_distance_matrix, verbose=False, log=True, **kwargs_for_ot)
            self.gw_distance = log['gw_dist']
            print(f'GW distance: {self.gw_distance}')
            f.write(f'GW distance: {self.gw_distance}\n')
            self._get_projected()

    def test_signal(self):
        # combine source, target, projected to unify columns (OTUs)
        combined = pd.concat([self.source_dataset, self.target_dataset, self.projected_data])
        combined.fillna(0.0, inplace=True)  # fill missing OTUs with relative abundance of 0
        combined.set_index('sample_id', inplace=True)

        source_dataset = combined[combined['dataset'] == self.source_dataset_name]
        target_dataset = combined[combined['dataset'] == self.target_dataset_name]
        projection = combined[combined['dataset'] == 'projected']

        with open(self._get_file_path('signal_retention_test.txt'), 'w') as f:
            print('\nTesting phenotype signal using Random Forest Classifier')
            f.write('Testing phenotype signal using Random Forest Classifier\n')

            source_data = source_dataset[data_utils.get_otu_columns(source_dataset)]
            source_phenotype = source_dataset['phenotype']
            source_age = source_dataset['age']
            target_data = target_dataset[data_utils.get_otu_columns(target_dataset)]
            target_phenotype = target_dataset['phenotype']
            target_age = target_dataset['age']
            projection_data = projection[data_utils.get_otu_columns(projection)]
            projection_phenotype = projection['phenotype']
            projection_age = projection['age']

            classifier = RandomForestClassifier(random_state=data_utils.PROJECT_SEED)
            classifier.fit(target_data, target_phenotype)

            # show results on source data before transport
            source_pred = classifier.predict(source_data)
            source_acc = (source_pred == source_phenotype).mean()
            source_probability_scores = classifier.predict_proba(source_data)[:, classifier.classes_ == 'CD']
            source_auc_roc = metrics.roc_auc_score((source_phenotype == 'CD').astype(int), source_probability_scores)
            print(f"Source data before transport classification results - Accuracy: {source_acc:.3f}, AUC-ROC: {source_auc_roc:.3f}")
            f.write(f"Source data before transport classification results - Accuracy: {source_acc:.3f}, AUC-ROC: {source_auc_roc:.3f}\n")
            
            # show results on projection data after transport
            projection_pred = classifier.predict(projection_data)
            projection_acc = (projection_pred == projection_phenotype).mean()
            projection_probability_scores = classifier.predict_proba(projection_data)[:, classifier.classes_ == 'CD']
            projection_auc_roc = metrics.roc_auc_score((projection_phenotype == 'CD').astype(int), projection_probability_scores)
            print(f"Projection data after transport classification results - Accuracy: {projection_acc:.3f}, AUC-ROC: {projection_auc_roc:.3f}")
            f.write(f"Projection data after transport classification results - Accuracy: {projection_acc:.3f}, AUC-ROC: {projection_auc_roc:.3f}\n")

            regressor_iterations = 20
            print(f"\nTesting age signal using Random Forest Regressor (averaging {regressor_iterations} iterations)")
            f.write(f"\nTesting age signal using Random Forest Regressor (averaging {regressor_iterations} iterations)\n")

            source_mse, source_corr = self._run_regressor(source_data, source_age, iterations=regressor_iterations)
            corr_values = [c[0] for c in source_corr]
            p_values = [c[1] for c in source_corr]
            print(f"\nSource data before transport age regression results:\
                  \nMSE mean: {np.mean(source_mse):.3f} (std: {np.std(source_mse):.3f})\
                  \nPearson correlation mean: {np.mean(corr_values):.3f} (std: {np.std(corr_values):.3f}), p-value mean: {np.mean(p_values):.3f} (std: {np.std(p_values):.3f})")
            f.write(f"\nSource data before transport age regression results:\
                  \nMSE mean: {np.mean(source_mse):.3f} (std: {np.std(source_mse):.3f})\
                  \nPearson correlation mean: {np.mean(corr_values):.3f} (std: {np.std(corr_values):.3f}), p-value mean: {np.mean(p_values):.3f} (std: {np.std(p_values):.3f})\n")

            projection_mse, projection_corr = self._run_regressor(projection_data, projection_age, iterations=regressor_iterations)
            corr_values = [c[0] for c in projection_corr]
            p_values = [c[1] for c in projection_corr]
            print(f"\nProjection data after transport age regression results:\
                  \nMSE mean: {np.mean(projection_mse):.3f} (std: {np.std(projection_mse):.3f})\
                  \nPearson correlation mean: {np.mean(corr_values):.3f} (std: {np.std(corr_values):.3f}), p-value mean: {np.mean(p_values):.3f} (std: {np.std(p_values):.3f})")
            f.write(f"\nProjection data after transport age regression results:\
                  \nMSE mean: {np.mean(projection_mse):.3f} (std: {np.std(projection_mse):.3f})\
                  \nPearson correlation mean: {np.mean(corr_values):.3f} (std: {np.std(corr_values):.3f}), p-value mean: {np.mean(p_values):.3f} (std: {np.std(p_values):.3f})\n")

    def _run_regressor(self, x, y, iterations=30):
        regressor = RandomForestRegressor(random_state=data_utils.PROJECT_SEED)
        mse_list = []
        corr_list = []
        for i in range(iterations):
            train_x, test_x, train_y, test_y = model_selection.train_test_split(x, y, test_size=0.4, random_state=data_utils.PROJECT_SEED + i)
            regressor.fit(train_x, train_y)

            pred = regressor.predict(test_x)
            mse = metrics.mean_squared_error(test_y, pred)
            corr = stats.pearsonr(test_y, pred)
            mse_list.append(mse)
            corr_list.append(corr)

        return mse_list, corr_list

    def run_test(self):
        print(f"\nRunning test {self.__class__.__name__}...")

        # create folder for results
        os.makedirs(self.results_folder_name, exist_ok=True)

        print("Showing variance pre-transport...")
        self.show_variance_pre_transport()
        print("Running transport...")
        self.transport()
        print("Showing variance post-transport...")
        self.show_variance_post_transport()
        if self.should_test_signal_retention:
            print("Testing signal...")
            self.test_signal()
        print("\nTest complete.")

    @staticmethod
    def _get_pairs(combined_data, suffix1, suffix2):
        """
        get indexes of pairs of samples that are the same except for the suffixes, e.g. from the source dataset and its projection.
        """
        indexes = combined_data.index
        pairs = [(indexes.get_loc(i), indexes.get_loc(i.replace(suffix1, suffix2))) for i in indexes if i.endswith(suffix1)]
        return pairs


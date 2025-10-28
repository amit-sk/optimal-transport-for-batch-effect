import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import variance_tests
import data_utils
from tests.optimal_transport_test import OptimalTransferTest


class SplitDatabaseSanityCheck(OptimalTransferTest):
    def __init__(self, should_run_pcoa=False):
        """
        Split risk dataset and transport one half onto the other half.
        """
        risk_data = pd.read_csv("risk_data.csv")
        self.original_data = risk_data.copy()

        shuffled = risk_data.sample(frac=1.0, random_state=data_utils.PROJECT_SEED)
        risk_data_1, risk_data_2 = np.split(shuffled, 2)

        risk_data_1 = risk_data_1.reset_index(drop=True)
        risk_data_2 = risk_data_2.reset_index(drop=True)

        super().__init__(risk_data_1, risk_data_2, should_run_pcoa=should_run_pcoa)

    def show_variance_pre_transport(self):
        # create combined data
        combined_data = pd.concat([self.target_dataset, self.source_dataset])
        combined_data.set_index('sample_id', inplace=True)

        # show variance before alignment
        print("\nComparing variance between target and source (before alignment):")
        variance_tests.show_variance(combined_data, 'dataset', should_run_pcoa=self.should_run_pcoa)
        variance_tests.show_variance(combined_data, 'phenotype', should_run_pcoa=self.should_run_pcoa)

    def show_variance_post_transport(self):
        self._observe_coupling_matrix(self.coupling, self.target_dataset, self.source_dataset)

        # titration plot to measure batch effect
        if self.should_run_pcoa:
            png_path = os.path.join('titrations', self.__class__.__name__ + '_titration.png')
            variance_tests.Metrics.titration(self.source_dataset, self.target_dataset, self.projected_data, repeats=10, png_name=png_path)

        combined_data = pd.concat([self.target_dataset, self.projected_data])
        combined_data.set_index('sample_id', inplace=True)

        # compare projection and original (post transport)
        print("\nComparing variance between target and projected:")
        variance_tests.show_variance(combined_data, 'dataset', should_run_pcoa=self.should_run_pcoa)
        
        print("\nComparing variance between phenotypes in combined target and projected:")
        variance_tests.show_variance(combined_data, 'phenotype', should_run_pcoa=self.should_run_pcoa)

        # compare projection and source (before and after transport)
        print("\nComparing variance between source and projected:")
        combined_data = pd.concat([self.source_dataset, self.projected_data])
        combined_data.set_index('sample_id', inplace=True)
        pairs = self._get_pairs(combined_data, suffix1='source', suffix2='projected')
        variance_tests.show_variance(combined_data, 'dataset', pcoa_pairs=pairs, should_run_pcoa=self.should_run_pcoa)

        # how much each projection had moved - compare target, source, projected
        print("\nComparing variance between target, source and projected:")
        combined_data = pd.concat([self.target_dataset, self.source_dataset, self.projected_data])
        combined_data.set_index('sample_id', inplace=True)
        pairs = self._get_pairs(combined_data, suffix1='source', suffix2='projected')
        variance_tests.show_variance(combined_data, 'dataset', pcoa_pairs=pairs, should_run_pcoa=self.should_run_pcoa)

    def _observe_coupling_matrix(self, coupling, data1, data2):
        print("\nObserving coupling matrix...")

        values, counts = np.unique(coupling * 132, return_counts=True)  # sum of columns will be 1, sum of all columns will be 132

        # remove zero
        indexes = np.where(values == 0)
        values = np.delete(values, indexes)
        counts = np.delete(counts, indexes)

        plt.bar(values, counts, width=1/132)
        plt.bar(values, counts, width=1/200)
        plt.bar(values, counts, width=1/2000, color='black')
        plt.yticks(np.arange(0, max(counts)+1, min(counts)))
        plt.xticks(np.arange(0, max(values)+0.01, 0.005))
        plt.title("Histogram of coupling matrix values (non-zero only)")
        plt.show()

        variance_tests.Draw.heatmap(coupling, data1['sample_id'], data2['sample_id'], cmap='Blues')

        # show spread - how many *unique* values in distribution
        # TODO: might be more informative to show number of non-zero values.
        spread_of_src = pd.DataFrame(coupling).nunique(axis=0)
        plt.hist(spread_of_src, bins=12)
        plt.title("Spread of each sample in src dataset in coupling matrix")
        plt.show()

        spread_of_src = pd.DataFrame(coupling).nunique(axis=1)
        plt.hist(spread_of_src, bins=12)
        plt.title("Spread of each sample in target dataset in coupling matrix")
        plt.show()


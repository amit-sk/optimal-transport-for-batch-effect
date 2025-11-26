import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import variance_tests
from tests.optimal_transport_test import OptimalTransportTest

class UnsupervisedTransportTest(OptimalTransportTest):
    def __init__(self, *, should_run_pcoa=False, should_show_pcoa=False, should_test_signal_retention=False, **kwargs):
        risk_data = pd.read_csv("risk_data.csv")
        mucosalibd_data = pd.read_csv("mucosalibd_data.csv")

        super().__init__(source_dataset=mucosalibd_data, target_dataset=risk_data, should_run_pcoa=should_run_pcoa,
                         should_show_pcoa=should_show_pcoa, should_test_signal_retention=should_test_signal_retention,
                         source_dataset_name='mucosalibd', target_dataset_name='risk', **kwargs)

    def show_variance_pre_transport(self):
        # create combined data (to show variance before alignment)
        combined_data = pd.concat([self.target_dataset, self.source_dataset])
        combined_data.fillna(0.0, inplace=True)
        combined_data.set_index('sample_id', inplace=True)

        # show variance before alignment
        print(f"\nComparing variance between {self.target_dataset_name} and {self.source_dataset_name} (before alignment):")
        variance_tests.show_variance(combined_data, 'dataset', file_path=self._get_file_path('pre_transport_by_database'),
                                     should_run_pcoa=self.should_run_pcoa, should_show_pcoa=self.should_show_pcoa)
        print(f"\nComparing variance between phenotypes in combined {self.target_dataset_name} and {self.source_dataset_name}:")
        variance_tests.show_variance(combined_data, 'phenotype', file_path=self._get_file_path('pre_transport_by_phenotype'),
                                     should_run_pcoa=self.should_run_pcoa, should_show_pcoa=self.should_show_pcoa)

    def show_variance_post_transport(self):
        target_data = self.target_dataset.copy()
        source_data = self.source_dataset.copy()
        projected = self.projected_data.copy()

        # titration plot to measure batch effect
        if self.should_run_pcoa:
            variance_tests.Metrics.titration(self.source_dataset, self.target_dataset, self.projected_data, repeats=10, png_name=self._get_file_path('titration.png'))

        # TODO: debug
        # if self.should_run_pcoa:
        #     self._observe_coupling_matrix()

        # compare projection and risk (post transport)
        combined_data = pd.concat([target_data, projected])
        combined_data.set_index('sample_id', inplace=True)

        print(f"\nComparing variance between {self.target_dataset_name} and projected:")
        variance_tests.show_variance(combined_data, 'dataset', file_path=self._get_file_path('post_transport_by_database'),
                                     should_run_pcoa=self.should_run_pcoa, should_show_pcoa=self.should_show_pcoa)
        print(f"\nComparing variance between phenotypes in combined {self.target_dataset_name} and projected:")
        variance_tests.show_variance(combined_data, 'phenotype', file_path=self._get_file_path('post_transport_by_phenotype'),
                                     should_run_pcoa=self.should_run_pcoa, should_show_pcoa=self.should_show_pcoa)

        print(f"\nComparing variance between dataset+phenotype in combined {self.target_dataset_name} and projected:")
        target_data['dataset+phenotype'] = f'{self.target_dataset_name}_' + target_data['phenotype']
        projected['dataset+phenotype'] = 'projected_' + projected['phenotype']
        combined_data = pd.concat([target_data, projected])
        combined_data.set_index('sample_id', inplace=True)
        variance_tests.show_variance(combined_data, 'dataset+phenotype', file_path=self._get_file_path('post_transport_by_dataset_and_phenotype'),
                                     should_run_pcoa=self.should_run_pcoa, should_show_pcoa=self.should_show_pcoa)

        # compare projection and mucosalibd (before and after transport)
        print(f"\nComparing variance between {self.source_dataset_name} (original) and projected:")
        combined_data = pd.concat([source_data, projected])
        combined_data.fillna(0.0, inplace=True)
        combined_data.set_index('sample_id', inplace=True)
        pairs = self._get_pairs(combined_data, f'_{self.source_dataset_name}', '_projected')
        variance_tests.show_variance(combined_data, 'dataset', pcoa_pairs=pairs, file_path=self._get_file_path('source_vs_projected'),
                                     should_run_pcoa=self.should_run_pcoa, should_show_pcoa=self.should_show_pcoa)

        # how much each projection had moved - compare risk, mucosalibd, projected
        print(f"\nComparing variance between risk, {self.source_dataset_name} and projected:")
        combined_data = pd.concat([target_data, source_data, projected])
        combined_data.fillna(0.0, inplace=True)
        combined_data.set_index('sample_id', inplace=True)
        pairs = self._get_pairs(combined_data, f'_{self.source_dataset_name}', '_projected')
        variance_tests.show_variance(combined_data, 'dataset', pcoa_pairs=pairs, file_path=self._get_file_path('target_vs_source_vs_projected'),
                                     should_run_pcoa=self.should_run_pcoa, should_show_pcoa=self.should_show_pcoa)

    def _observe_coupling_matrix(self):
        """
        debug function, copy pasted with numerical constants fit for data
        """
        print("\nObserving coupling matrix...")

        values, counts = np.unique(self.coupling * 132, return_counts=True)  # sum of columns will be 1, sum of all columns will be 132

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

        variance_tests.Draw.heatmap(self.coupling, self.target_dataset['sample_id'], self.source_dataset['sample_id'], cmap='Blues')

        # show spread - how many *unique* values in distribution
        # TODO: might be more informative to show number of non-zero values.
        coupling = pd.DataFrame(self.coupling)
        spread_of_src = coupling.nunique(axis=0)
        plt.hist(spread_of_src, bins=12)
        plt.title(f"Spread of each sample in src dataset ({self.source_dataset_name}) in coupling matrix")
        plt.show()

        spread_of_src = coupling.nunique(axis=1)
        plt.hist(spread_of_src, bins=12)
        plt.title(f"Spread of each sample in target dataset ({self.target_dataset_name}) in coupling matrix")
        plt.show()


import os.path
import pandas as pd
import numpy as np

import variance_tests
import data_utils
from tests.optimal_transport_test import OptimalTransportTest


class SanityCheck(OptimalTransportTest):
    def __init__(self, *, should_run_pcoa=False, **kwargs):
        """
        transporting noisy RISK data onto original RISK data.
        """
        risk_data = pd.read_csv("risk_data.csv")
        noisy_data = data_utils.create_noisy_data(risk_data, proportion_of_std=0.1)
        noisy_data = data_utils.renormalize_data(noisy_data)

        super().__init__(source_dataset=noisy_data, target_dataset=risk_data, should_run_pcoa=should_run_pcoa,
                         source_dataset_name='noisy', target_dataset_name='orig', **kwargs)

    def show_variance_pre_transport(self):
        risk_data = self.target_dataset.copy()
        risk_otu_data = self.target_otu_data
        noisy_data = self.source_dataset.copy()
        noisy_otu_data = self.source_otu_data

        # create combined data
        combined_data = pd.concat([risk_data, noisy_data])
        combined_data.set_index('sample_id', inplace=True)
        pairs = self._get_pairs(combined_data, '_orig', '_noisy')

        # show variance before alignment
        print("\nComparing variance between original and noisy (before alignment):")
        variance_tests.show_variance(combined_data, 'dataset', pcoa_pairs=pairs, should_run_pcoa=self.should_run_pcoa)
        variance_tests.show_variance(combined_data, 'phenotype', should_run_pcoa=self.should_run_pcoa)
        fracs = variance_tests.Metrics.calc_domain_avg_FOSCTTM(risk_otu_data.values, noisy_otu_data.values, should_use_braycurtis=True)
        print(f"Average FOSCTTM score between noisy and original: {fracs.mean()}")

    def show_variance_post_transport(self):
        projected = self.projected_data
        risk_data = self.target_dataset
        risk_otu_data = self.target_otu_data
        noisy_data = self.source_dataset
        noisy_otu_data = self.source_otu_data

        print(f'coupling diagonal sum: {self.coupling.diagonal().sum()}')
        fracs = variance_tests.Metrics.calc_domain_avg_FOSCTTM(risk_otu_data.values, self.projected_otu_data.values, should_use_braycurtis=True)
        print(f"Average FOSCTTM score between projected and original (post transport): {fracs.mean()}")

        # titration plot to measure batch effect
        if self.should_run_pcoa:
            png_path = os.path.join('titrations', self.__class__.__name__ + '_titration.png')
            variance_tests.Metrics.titration(self.source_dataset, self.target_dataset, self.projected_data, repeats=10, png_name=png_path)

        combined_data = pd.concat([risk_data, projected])
        combined_data.set_index('sample_id', inplace=True)

        # compare projection and original (post transport)
        print("\nComparing variance between original and projected:")
        pairs = self._get_pairs(combined_data, '_orig', '_projected')
        variance_tests.show_variance(combined_data, 'dataset', pcoa_pairs=pairs, should_run_pcoa=self.should_run_pcoa)
        
        print("\nComparing variance between phenotypes in combined original and projected:")
        variance_tests.show_variance(combined_data, 'phenotype', should_run_pcoa=self.should_run_pcoa)

        # compare projection and noisy (before and after transport)
        print("\nComparing variance between noisy and projected:")
        combined_data = pd.concat([noisy_data, projected])
        combined_data.set_index('sample_id', inplace=True)
        pairs = self._get_pairs(combined_data, '_noisy', '_projected')
        variance_tests.show_variance(combined_data, 'dataset', pcoa_pairs=pairs, should_run_pcoa=self.should_run_pcoa)

        # how much each projection had moved - compare orig, noisy, projected
        print("\nComparing variance between original, noisy and projected:")
        combined_data = pd.concat([risk_data, noisy_data, projected])
        combined_data.set_index('sample_id', inplace=True)
        pairs = self._get_pairs(combined_data, '_noisy', '_projected')
        pairs.extend(self._get_pairs(combined_data, '_noisy', '_orig'))
        variance_tests.show_variance(combined_data, 'dataset', pcoa_pairs=pairs, should_run_pcoa=self.should_run_pcoa)


import pandas as pd
import numpy as np

import optimal_transport
import distribution_variance
import data_utils
from optimal_transport_test import OptimalTransferTest


class SanityCheck(OptimalTransferTest):
    def __init__(self, should_run_pcoa=False):
        """
        transporting noisy RISK data onto original RISK data.
        """
        risk_data = pd.read_csv("risk_data.csv")
        noisy_data = data_utils.create_noisy_data(risk_data, proportion_of_std=0.1)
        noisy_data = data_utils.renormalize_data(noisy_data)

        risk_data['dataset'] = 'orig'
        risk_data['sample_id'] = risk_data['sample_id'] + '_orig'
        noisy_data['dataset'] = 'noisy'
        noisy_data['sample_id'] = noisy_data['sample_id'] + '_noisy'

        super().__init__(noisy_data, risk_data, should_run_pcoa=should_run_pcoa)

    def show_variance_pre_transport(self):
        risk_data = self.target_dataset
        risk_otu_data = self.target_otu_data
        noisy_data = self.source_dataset
        noisy_otu_data = self.source_otu_data

        # create combined data
        combined_data = pd.concat([risk_data, noisy_data])
        combined_data.set_index('sample_id', inplace=True)
        indexes = combined_data.index
        pairs = [(indexes.get_loc(i), indexes.get_loc(i.replace('_orig','_noisy'))) for i in indexes if i.endswith('_orig')]

        # show variance before alignment
        print("\nComparing variance between original and noisy (before alignment):")
        distribution_variance.show_variance(combined_data, 'dataset', pcoa_pairs=pairs, should_run_pcoa=self.should_run_pcoa)
        distribution_variance.show_variance(combined_data, 'phenotype', should_run_pcoa=self.should_run_pcoa)
        fracs = distribution_variance.calc_domain_avg_FOSCTTM(risk_otu_data.values, noisy_otu_data.values, should_use_braycurtis=self.should_run_pcoa)
        print(f"Average FOSCTTM score between noisy and original: {fracs.mean()}")

    def show_variance_post_transport(self):
        projected = self.projected
        risk_data = self.target_dataset
        risk_otu_data = self.target_otu_data
        noisy_data = self.source_dataset
        noisy_otu_data = self.source_otu_data

        print(f'coupling diagonal sum: {self.coupling.diagonal().sum()}')
        fracs = distribution_variance.calc_domain_avg_FOSCTTM(risk_otu_data.values, projected.values, should_use_braycurtis=self.should_run_pcoa)
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
        distribution_variance.show_variance(combined_data, 'dataset', pcoa_pairs=pairs, should_run_pcoa=self.should_run_pcoa)
        distribution_variance.show_variance(combined_data, 'phenotype', should_run_pcoa=self.should_run_pcoa)

        # compare projection and noisy (before and after transport)
        print("\nComparing variance between noisy and projected:")
        combined_data = pd.concat([noisy_data, projected])
        combined_data.set_index('sample_id', inplace=True)
        indexes = combined_data.index
        pairs = [(indexes.get_loc(i), indexes.get_loc(i.replace('_noisy','_projected'))) for i in indexes if i.endswith('_noisy')]
        distribution_variance.show_variance(combined_data, 'dataset', pcoa_pairs=pairs, should_run_pcoa=self.should_run_pcoa)

        # how much each projection had moved - compare orig, noisy, projected
        print("\nComparing variance between original, noisy and projected:")
        combined_data = pd.concat([risk_data, noisy_data, projected])
        combined_data.set_index('sample_id', inplace=True)
        indexes = combined_data.index
        pairs = [(indexes.get_loc(i), indexes.get_loc(i.replace('_noisy','_projected'))) for i in indexes if i.endswith('_noisy')]
        pairs.extend([(indexes.get_loc(i), indexes.get_loc(i.replace('_noisy','_orig'))) for i in indexes if i.endswith('_noisy')])
        distribution_variance.show_variance(combined_data, 'dataset', pcoa_pairs=pairs, should_run_pcoa=self.should_run_pcoa)


if __name__ == "__main__":
    SanityCheck(should_run_pcoa=True).run_test()

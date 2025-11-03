import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import data_utils
import variance_tests
from tests.optimal_transport_test import OptimalTransportTest


class SupervisedDoublePhenotypeWeightingTest(OptimalTransportTest):
    """
    Transporting mucosalIBD data onto RISK data with supervised phenotype weighting.
    Expected to run 2 transports, for each phenotype in the source database - control or CD - separately.
    For each phenotype, weighting the target distribution so that the current phenotype in the source is more represented, according to weight_for_current_phenotype.
    class SupervisedDoublePhenotypeWeightingTests below runs two of these tests, one for each phenotype, for multiple weights.
    """
    def __init__(self, *, source_dataset, target_dataset, current_phenotype, weight_for_current_phenotype=0.8, should_run_pcoa=False, **kwargs):
        source_data_for_phenotype = source_dataset[source_dataset['phenotype'] == current_phenotype].reset_index(drop=True)
        
        super().__init__(source_dataset=source_data_for_phenotype, target_dataset=target_dataset, should_run_pcoa=should_run_pcoa,
                         source_dataset_name='mucosalibd_'+current_phenotype, target_dataset_name='risk', **kwargs)

        self.current_phenotype = current_phenotype
        self.weight_for_current_phenotype = weight_for_current_phenotype

    def transport(self):
        p = self._get_dataset_phenotype_weights()
        super().transport(p=p)

    def _get_dataset_phenotype_weights(self):
        other_phenotype = 'control' if self.current_phenotype == 'CD' else 'CD'
        weights = pd.Series({self.current_phenotype: self.weight_for_current_phenotype, other_phenotype: 1 - self.weight_for_current_phenotype})

        target_phenotype_counts = self.target_dataset['phenotype'].value_counts()
        target_weights = weights / target_phenotype_counts
        return self.target_dataset['phenotype'].map(target_weights).to_numpy()


class SupervisedDoublePhenotypeWeightingTests(OptimalTransportTest):
    """
    Runs two SupervisedDoublePhenotypeWeightingTest, one for each phenotype in the source dataset.
    Does this for weights: 40-60, 30-70, 20-80, 10-90.
    """
    
    def __init__(self, should_run_pcoa=False, **kwargs):
        risk_data = pd.read_csv("risk_data.csv")
        self.original_target_dataset = risk_data.copy()
        mucosalibd_data = pd.read_csv("mucosalibd_data.csv")
        self.original_source_dataset = mucosalibd_data.copy()

        super().__init__(source_dataset=mucosalibd_data, target_dataset=risk_data, should_run_pcoa=should_run_pcoa,
                         source_dataset_name='mucosalibd', target_dataset_name='risk', **kwargs)

    def show_variance_pre_transport(self):
        # create combined data (to show variance before alignment)
        combined_data = pd.concat([self.target_dataset, self.source_dataset])
        combined_data.fillna(0.0, inplace=True)
        combined_data.set_index('sample_id', inplace=True)

        # show variance before alignment
        print("\nComparing variance between risk and mucosalibd (before alignment):")
        variance_tests.show_variance(combined_data, 'dataset', should_run_pcoa=self.should_run_pcoa)
        print("\nComparing variance between phenotypes in combined risk and mucosalibd:")
        variance_tests.show_variance(combined_data, 'phenotype', should_run_pcoa=self.should_run_pcoa)

        # TODO: show var between each phenotype separately in combined dataset?

    def show_variance_post_transport(self, weight_for_current_phenotype):
        risk_data = self.target_dataset.copy()
        mucosalibd_data = self.source_dataset.copy()
        projected = self.projected_data.copy()

        # if self.should_run_pcoa:
        #     self._observe_coupling_matrix()

        # titration plot to measure batch effect
        if self.should_run_pcoa:
            png_path = os.path.join('titrations', self.__class__.__name__ + f'_titration_weight_{int(weight_for_current_phenotype*100)}.png')
            variance_tests.Metrics.titration(self.source_dataset, self.target_dataset, self.projected_data, repeats=10, png_name=png_path)

        # compare projection and risk (post transport)
        combined_data = pd.concat([risk_data, projected])
        # TODO: show var between each phenotype separately in combined dataset?
        # combined_data = pd.concat([risk_data[risk_data.phenotype == 'control'], projected[risk_data.phenotype == 'control']])  # compare only controls (healthy)
        combined_data.set_index('sample_id', inplace=True)

        print("\nComparing variance between risk and projected:")
        variance_tests.show_variance(combined_data, 'dataset', should_run_pcoa=self.should_run_pcoa)
        print("\nComparing variance between phenotypes in combined risk and projected:")
        variance_tests.show_variance(combined_data, 'phenotype', should_run_pcoa=self.should_run_pcoa)

        print("\nComparing variance between dataset+phenotype in combined risk and projected:")
        risk_data['dataset+phenotype'] = 'RISK_' + risk_data['phenotype']
        projected['dataset+phenotype'] = 'Projected_' + projected['phenotype']
        combined_data = pd.concat([risk_data, projected])
        combined_data.set_index('sample_id', inplace=True)
        variance_tests.show_variance(combined_data, 'dataset+phenotype', should_run_pcoa=self.should_run_pcoa)

        # compare projection and mucosalibd (before and after transport)
        print("\nComparing variance between mucosalibd (original) and projected:")
        combined_data = pd.concat([mucosalibd_data, projected])
        combined_data.fillna(0.0, inplace=True)
        combined_data.set_index('sample_id', inplace=True)
        pairs = self._get_pairs(combined_data, '_mucosalibd', '_projected')
        variance_tests.show_variance(combined_data, 'dataset', pcoa_pairs=pairs, should_run_pcoa=self.should_run_pcoa)

        # how much each projection had moved - compare risk, mucosalibd, projected
        print("\nComparing variance between risk, mucosalibd and projected:")
        combined_data = pd.concat([risk_data, mucosalibd_data, projected])
        combined_data.fillna(0.0, inplace=True)
        combined_data.set_index('sample_id', inplace=True)
        pairs = self._get_pairs(combined_data, '_mucosalibd', '_projected')
        variance_tests.show_variance(combined_data, 'dataset', pcoa_pairs=pairs, should_run_pcoa=self.should_run_pcoa)

    def run_test(self):
        print(f"Running test {self.__class__.__name__}...")
        print("Showing variance pre-transport...")
        self.show_variance_pre_transport()

        for weight_for_current_phenotype in [0.6, 0.7, 0.8, 0.9]:  # TODO: obtain current weight from init args, and move this loop to main 
            print(f"\n\nRunning test with weights {int(weight_for_current_phenotype*100)}-{int((1-weight_for_current_phenotype)*100)}...\n")
            control_test = SupervisedDoublePhenotypeWeightingTest(
                source_dataset=self.original_source_dataset.copy(), target_dataset=self.original_target_dataset.copy(), current_phenotype='control',
                weight_for_current_phenotype=weight_for_current_phenotype, should_run_pcoa=self.should_run_pcoa
            )
            cd_test = SupervisedDoublePhenotypeWeightingTest(
                source_dataset=self.original_source_dataset.copy(), target_dataset=self.original_target_dataset.copy(), current_phenotype='CD',
                weight_for_current_phenotype=weight_for_current_phenotype, should_run_pcoa=self.should_run_pcoa
            )

            print("Running transport...")
            print("control:")
            control_test.transport()
            print("CD:")
            cd_test.transport()
            self._get_projected(control_test, cd_test)
            print("Showing variance post-transport...")
            self.show_variance_post_transport(weight_for_current_phenotype)
            print("Testing signal...")
            self.test_signal()

        print("Test complete.")

    def _get_projected(self, control_test, cd_test):
        combined_projected_data = pd.concat([control_test.projected_data, cd_test.projected_data])
        self.projected_data = combined_projected_data
        self.projected_otu_data = combined_projected_data[data_utils.get_otu_columns(combined_projected_data)]


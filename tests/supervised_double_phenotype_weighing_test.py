import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import data_utils
import variance_tests
from tests.optimal_transport_test import OptimalTransportTest
from tests.unsupervised_transport_test import UnsupervisedTransportTest


class SupervisedWeightingTestForPhenotype(OptimalTransportTest):
    """
    Transporting source data onto target data with supervised phenotype weighting.
    Expected to run 2 transports, for each phenotype in the source database - control or CD - separately.
    For each phenotype, weighting the target distribution so that the current phenotype in the source is more represented, according to weight_for_current_phenotype.
    class SupervisedDoublePhenotypeWeightingTests below runs two of these tests, one for each phenotype, for multiple weights. This is not meant to be run directly.
    """
    def __init__(self, *, source_dataset, target_dataset, current_phenotype, weight_for_current_phenotype=0.8, should_run_pcoa=False, should_show_pcoa=False, should_test_signal_retention=False, source_dataset_name='source', target_dataset_name='target', **kwargs):
        source_data_for_phenotype = source_dataset[source_dataset['phenotype'] == current_phenotype].reset_index(drop=True)
        
        super().__init__(source_dataset=source_data_for_phenotype, target_dataset=target_dataset,
                         should_run_pcoa=should_run_pcoa, should_show_pcoa=should_show_pcoa, should_test_signal_retention=should_test_signal_retention,
                         source_dataset_name=source_dataset_name+current_phenotype, target_dataset_name=target_dataset_name, **kwargs)

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
    class TestsRunner(UnsupervisedTransportTest):
        """
        internal class which runs the two transports per phenotype for a given weight.
        needs the original datasets to create the two tests.
        uses given results folder name and appends weight and phenotype to create subfolders for each test.
        """
        def __init__(self, *, original_source_dataset, original_target_dataset, source_dataset_name, target_dataset_name, weight_for_current_phenotype,
                     results_folder_name, should_run_pcoa=False, should_show_pcoa=False, should_test_signal_retention=False, **kwargs):
            weight_str = f'weight_{int(weight_for_current_phenotype*100)}'

            # each test needs its own copy of the original datasets.
            OptimalTransportTest.__init__(
                self, should_run_pcoa=should_run_pcoa, should_show_pcoa=should_show_pcoa, should_test_signal_retention=should_test_signal_retention,
                source_dataset=original_source_dataset.copy(), target_dataset=original_target_dataset.copy(),
                source_dataset_name=source_dataset_name, target_dataset_name=target_dataset_name,
                results_folder_name=os.path.join(results_folder_name, weight_str), **kwargs
            )

            self.control_test = SupervisedWeightingTestForPhenotype(
                source_dataset=original_source_dataset.copy(), target_dataset=original_target_dataset.copy(),
                current_phenotype='control', weight_for_current_phenotype=weight_for_current_phenotype,
                source_dataset_name=source_dataset_name, target_dataset_name=target_dataset_name,
                results_folder_name=os.path.join(self.results_folder_name, 'control')
            )
            self.cd_test = SupervisedWeightingTestForPhenotype(
                source_dataset=original_source_dataset.copy(), target_dataset=original_target_dataset.copy(),
                current_phenotype='CD', weight_for_current_phenotype=weight_for_current_phenotype,
                source_dataset_name=source_dataset_name, target_dataset_name=target_dataset_name,
                results_folder_name=os.path.join(self.results_folder_name, 'CD')
            )

        def transport(self):
            self.control_test.transport()
            self.cd_test.transport()
            self._get_projected()

        def _get_projected(self):
            combined_projected_data = pd.concat([self.control_test.projected_data, self.cd_test.projected_data])
            self.projected_data = combined_projected_data
            self.projected_otu_data = combined_projected_data[data_utils.get_otu_columns(combined_projected_data)]

        def run_test(self):
            # create folder for results
            os.makedirs(self.results_folder_name, exist_ok=True)
            os.makedirs(self.control_test.results_folder_name, exist_ok=True)
            os.makedirs(self.cd_test.results_folder_name, exist_ok=True)

            print("Running transport...")
            self.transport()
            print("Showing variance post-transport...")
            self.show_variance_post_transport()
            if self.should_test_signal_retention:
                print("Testing dataset differentiation...")
                self.test_dataset_differentiation()
                print("Testing signal...")
                self.test_signal()
            print("Test complete.")


    def __init__(self, should_run_pcoa=False, should_show_pcoa=False, should_test_signal_retention=False, **kwargs):
        # read here so the csv are only read once
        risk_data = pd.read_csv("risk_data.csv")
        mucosalibd_data = pd.read_csv("mucosalibd_data.csv")

        # need original datasets for each test
        self.original_target_dataset = risk_data.copy()
        self.original_source_dataset = mucosalibd_data.copy()

        super().__init__(source_dataset=mucosalibd_data, target_dataset=risk_data, should_run_pcoa=should_run_pcoa, should_show_pcoa=should_show_pcoa,
                         should_test_signal_retention=should_test_signal_retention, source_dataset_name='mucosalibd', target_dataset_name='risk', **kwargs)

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

        # TODO: show var between each phenotype separately in combined dataset?

    def run_test(self):
        print(f"\nRunning test {self.__class__.__name__}...")

        # create folder for results
        os.makedirs(self.results_folder_name, exist_ok=True)

        print("Showing variance pre-transport...")
        self.show_variance_pre_transport()

        for weight_for_current_phenotype in [0.6, 0.7, 0.8, 0.9]:  # TODO: obtain current weight from init args
            print(f"\n\nRunning test with weights {int(weight_for_current_phenotype*100)}-{int((1-weight_for_current_phenotype)*100)}...\n")
            SupervisedDoublePhenotypeWeightingTests.TestsRunner(
                original_source_dataset=self.original_source_dataset.copy(),
                original_target_dataset=self.original_target_dataset.copy(),
                source_dataset_name=self.source_dataset_name,
                target_dataset_name=self.target_dataset_name,
                weight_for_current_phenotype=weight_for_current_phenotype,
                results_folder_name=self.results_folder_name,
                should_run_pcoa=self.should_run_pcoa,
                should_show_pcoa=self.should_show_pcoa,
                should_test_signal_retention=self.should_test_signal_retention
            ).run_test()

        print("Test complete.")


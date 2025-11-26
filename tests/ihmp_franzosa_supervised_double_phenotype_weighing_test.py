import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import data_utils
import variance_tests
from tests.optimal_transport_test import OptimalTransportTest
from tests.unsupervised_transport_test import UnsupervisedTransportTest
from tests.supervised_double_phenotype_weighing_test import SupervisedDoublePhenotypeWeightingTests


class iHMP_FRANZOSA_SupervisedDoublePhenotypeWeightingTests(SupervisedDoublePhenotypeWeightingTests):
    def __init__(self, should_run_pcoa=False, should_show_pcoa=False, should_test_signal_retention=False, **kwargs):
        franzosa_data = pd.read_csv("franzosa_data.csv")
        ihmp_data = pd.read_csv("ihmp_data.csv")
 
        # need original datasets for each test
        self.original_target_dataset = franzosa_data.copy()
        self.original_source_dataset = ihmp_data.copy()


        OptimalTransportTest.__init__(self, source_dataset=ihmp_data, target_dataset=franzosa_data, should_run_pcoa=should_run_pcoa, should_show_pcoa=should_show_pcoa,
                                      should_test_signal_retention=should_test_signal_retention, source_dataset_name='ihmp', target_dataset_name='franzosa', **kwargs)
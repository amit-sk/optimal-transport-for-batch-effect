import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ot

import variance_tests
from tests.optimal_transport_test_with_entropy import OptimalTransportEntropyTest
from tests.supervised_phenotype_weighting_test import SupervisedPhenotypeWeightingTest

class SupervisedPhenotypeWeightingTestWithEntropy(SupervisedPhenotypeWeightingTest, OptimalTransportEntropyTest):
    def transport(self):
        p = self._get_dataset_phenotype_weights()
        OptimalTransportEntropyTest.transport(self, p=p)

    def _observe_coupling_matrix(self):
        """
        debug function, copy pasted with numerical constants fit for data
        """
        print("\nObserving coupling matrix...")

        values, counts = np.unique(self.coupling, return_counts=True)  # the sum of the columns and rows varies, as the distributions are weighted by phenotype proportions

        # remove zero
        indexes = np.where(values == 0)
        values = np.delete(values, indexes)
        counts = np.delete(counts, indexes)

        # plt.bar(values, counts, width=1/132)
        # plt.bar(values, counts, width=1/900)
        plt.bar(values, counts, width=1/4000)
        plt.bar(values, counts, width=1/10000, color='black')
        plt.yticks(np.arange(0, max(counts)+1, min(counts)))
        plt.xticks(np.arange(0, max(values)+0.001, 0.0005))
        plt.title("Histogram of coupling matrix values (non-zero only)")
        plt.show()

        variance_tests.Draw.heatmap(self.coupling, self.target_dataset['sample_id'], self.source_dataset['sample_id'], cmap='Blues')

        # show spread - how many *unique* values in distribution
        # TODO: might be more informative to show number of non-zero values.
        coupling = pd.DataFrame(self.coupling)
        spread_of_src = coupling.nunique(axis=0)
        plt.hist(spread_of_src, bins=12)
        plt.title("Spread of each sample in src dataset in coupling matrix")
        plt.show()

        spread_of_src = coupling.nunique(axis=1)
        plt.hist(spread_of_src, bins=12)
        plt.title("Spread of each sample in target dataset in coupling matrix")
        plt.show()


import ot

from tests.unsupervised_transport_test import UnsupervisedTransportTest

class SupervisedPhenotypeWeightingTest(UnsupervisedTransportTest):
    """
    transporting RISK data onto RISK data with supervised phenotype weighting.
    """
    def transport(self):
        p = self._get_dataset_phenotype_weights()
        self.coupling, log = ot.gromov.gromov_wasserstein(self.target_distance_matrix, self.source_distance_matrix, p=p, verbose=False, log=True)
        self.gw_distance = log['gw_dist']
        print(f'GW distance: {self.gw_distance}')
        self._get_projected()

    def _get_dataset_phenotype_weights(self):
        source_phenotype_counts = self.source_dataset['phenotype'].value_counts()
        source_all_count = source_phenotype_counts.sum()
        source_weights = source_phenotype_counts / source_all_count

        target_phenotype_counts = self.target_dataset['phenotype'].value_counts()
        target_weights = source_weights / target_phenotype_counts
        return self.target_dataset['phenotype'].map(target_weights).to_numpy()

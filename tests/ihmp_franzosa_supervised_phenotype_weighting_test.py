from tests.ihmp_franzosa_unsupervised_transport_test import iHMP_FRANZOSA_UnsupervisedTransportTest

class iHMP_FRANZOSA_SupervisedPhenotypeWeightingTest(iHMP_FRANZOSA_UnsupervisedTransportTest):
    """
    COPIED FROM supervised_phenotype_weighting_test.py
    changed to inherit from iHMP_FRANZOSA_UnsupervisedTransportTest.
    the rest is copied. give me a break lol.
    """
    def transport(self):
        p = self._get_dataset_phenotype_weights()
        super().transport(p=p)

    def _get_dataset_phenotype_weights(self):
        source_phenotype_counts = self.source_dataset['phenotype'].value_counts()
        source_all_count = source_phenotype_counts.sum()
        source_weights = source_phenotype_counts / source_all_count

        target_phenotype_counts = self.target_dataset['phenotype'].value_counts()
        target_weights = source_weights / target_phenotype_counts
        return self.target_dataset['phenotype'].map(target_weights).to_numpy()

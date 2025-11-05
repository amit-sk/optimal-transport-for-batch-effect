from tests import (sanity_check, sanity_check_with_entropy, unsupervised_transport_test, unsupervised_with_entropy_test,
                  split_database_sanity_check, split_database_with_entropy_sanity_check, 
                  supervised_phenotype_weighting_test, supervised_phenotype_weighting_test_with_entropy,
                  supervised_double_phenotype_weighing_test, supervised_penalty_for_opposite_phenotype_test)

def main():
    # sanity_check.SanityCheck(should_run_pcoa=False).run_test()
    # sanity_check_with_entropy.SanityCheckWithEntropy(epsilon=10**-3.5, should_run_pcoa=False).run_test()
    # unsupervised_transport_test.UnsupervisedTransportTest(should_run_pcoa=False).run_test()
    # unsupervised_with_entropy_test.UnsupervisedWithEntropyTest(epsilon=10**-3.5, should_run_pcoa=False).run_test()
    # split_database_sanity_check.SplitDatabaseSanityCheck(should_run_pcoa=False).run_test()
    # split_database_with_entropy_sanity_check.SplitDatabaseWithEntropySanityCheck(epsilon=10**-3.5, should_run_pcoa=False).run_test()
    # supervised_phenotype_weighting_test.SupervisedPhenotypeWeightingTest(should_run_pcoa=False).run_test()
    # supervised_phenotype_weighting_test_with_entropy.SupervisedPhenotypeWeightingTestWithEntropy(epsilon=10**-3.5, should_run_pcoa=False).run_test()
    # supervised_double_phenotype_weighing_test.SupervisedDoublePhenotypeWeightingTests(should_run_pcoa=False).run_test()
    for alpha in [1e-7, 1e-5, 1e-3, 1e-1, 1, 10]:
        supervised_penalty_for_opposite_phenotype_test.SupervisedPenaltyForOppositePhenotypeTest(alpha=alpha, should_run_pcoa=False).run_test()
    print("Done.")


if __name__ == '__main__':
    main()

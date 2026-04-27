from tests import (sanity_check, sanity_check_with_entropy, unsupervised_transport_test, unsupervised_with_entropy_test,
                  split_database_sanity_check, split_database_with_entropy_sanity_check, 
                  supervised_phenotype_weighting_test, supervised_phenotype_weighting_test_with_entropy,
                  supervised_double_phenotype_weighing_test, supervised_penalty_for_opposite_phenotype_test,
                  ihmp_franzosa_unsupervised_transport_test, ihmp_franzosa_supervised_double_phenotype_weighing_test,
                  ihmp_franzosa_supervised_phenotype_weighting_test, ihmp_franzosa_supervised_penalty_for_opposite_phenotype_test)

def main():
    # sanity_check.SanityCheck(should_run_pcoa=True, should_show_pcoa=False, should_test_signal_retention=True).run_test()
    sanity_check_with_entropy.SanityCheckWithEntropy(epsilon=10**-3.5, should_run_pcoa=True, should_show_pcoa=False, should_test_signal_retention=True).run_test()
    # unsupervised_transport_test.UnsupervisedTransportTest(should_run_pcoa=True, should_show_pcoa=False, should_test_signal_retention=True).run_test()
    # ihmp_franzosa_unsupervised_transport_test.iHMP_FRANZOSA_UnsupervisedTransportTest(should_run_pcoa=True, should_show_pcoa=False, should_test_signal_retention=True).run_test()
    unsupervised_with_entropy_test.UnsupervisedWithEntropyTest(epsilon=10**-3.5, should_run_pcoa=True, should_show_pcoa=False, should_test_signal_retention=True).run_test()
    # split_database_sanity_check.SplitDatabaseSanityCheck(should_run_pcoa=True, should_show_pcoa=False, should_test_signal_retention=True).run_test()
    split_database_with_entropy_sanity_check.SplitDatabaseWithEntropySanityCheck(epsilon=10**-3.5, should_run_pcoa=True, should_show_pcoa=False, should_test_signal_retention=True).run_test()
    # supervised_phenotype_weighting_test.SupervisedPhenotypeWeightingTest(should_run_pcoa=True, should_show_pcoa=False, should_test_signal_retention=True).run_test()
    # ihmp_franzosa_supervised_phenotype_weighting_test.iHMP_FRANZOSA_SupervisedPhenotypeWeightingTest(should_run_pcoa=True, should_show_pcoa=False, should_test_signal_retention=True).run_test()
    supervised_phenotype_weighting_test_with_entropy.SupervisedPhenotypeWeightingTestWithEntropy(epsilon=10**-3.5, should_run_pcoa=True, should_show_pcoa=False, should_test_signal_retention=True).run_test()
    # supervised_double_phenotype_weighing_test.SupervisedDoublePhenotypeWeightingTests(should_run_pcoa=True, should_show_pcoa=False, should_test_signal_retention=True).run_test()
    # ihmp_franzosa_supervised_double_phenotype_weighing_test.iHMP_FRANZOSA_SupervisedDoublePhenotypeWeightingTests(should_run_pcoa=True, should_show_pcoa=False, should_test_signal_retention=True).run_test()
    # supervised_penalty_for_opposite_phenotype_test.SupervisedPenaltyForOppositePhenotypeTests(should_run_pcoa=True, should_show_pcoa=False, should_test_signal_retention=True).run_test()
    # ihmp_franzosa_supervised_penalty_for_opposite_phenotype_test.iHMP_FRANZOSA_SupervisedPenaltyForOppositePhenotypeTests(alpha_values=[1e-5, 1e-3, 0.005, 0.01, 0.05, 0.1, 1], should_run_pcoa=True, should_show_pcoa=False, should_test_signal_retention=True).run_test()
    print("Done.")


if __name__ == '__main__':
    main()

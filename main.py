from tests import sanity_check, sanity_check_with_entropy, unsupervised_transport_test, unsupervised_with_entropy_test, split_database_sanity_check, split_database_with_entropy_sanity_check

def main():
    sanity_check.SanityCheck(should_run_pcoa=False).run_test()
    sanity_check_with_entropy.SanityCheckWithEntropy(epsilon=10**-3.5, should_run_pcoa=False).run_test()
    unsupervised_transport_test.UnsupervisedTransportTest(should_run_pcoa=False).run_test()
    unsupervised_with_entropy_test.UnsupervisedWithEntropyTest(epsilon=10**-3.5, should_run_pcoa=False).run_test()
    split_database_sanity_check.SplitDatabaseSanityCheck(should_run_pcoa=False).run_test()
    split_database_with_entropy_sanity_check.SplitDatabaseWithEntropySanityCheck(epsilon=10**-3.5, should_run_pcoa=False).run_test()
    print("Done.")


if __name__ == '__main__':
    main()

from tests import sanity_check, unsupervised_transport_test, unsupervised_with_entropy_test

def main():
    # sanity_check.SanityCheck(should_run_pcoa=False).run_test()
    # unsupervised_transport_test.UnsupervisedTransportTest(should_run_pcoa=False).run_test()
    unsupervised_with_entropy_test.UnsupervisedWithEntropyTest(epsilon=10**-3.5, should_run_pcoa=True).run_test()
    print("Done.")


if __name__ == '__main__':
    main()

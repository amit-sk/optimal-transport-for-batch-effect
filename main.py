from tests import sanity_check, unsupervised_transport_test

def main():
    sanity_check.SanityCheck(should_run_pcoa=False).run_test()
    unsupervised_transport_test.UnsupervisedTransportTest(should_run_pcoa=False).run_test()
    print("Done.")


if __name__ == '__main__':
    main()

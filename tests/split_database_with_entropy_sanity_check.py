from tests.optimal_transport_test_with_entropy import OptimalTransportEntropyTest
from tests.split_database_sanity_check import SplitDatabaseSanityCheck

class SplitDatabaseWithEntropySanityCheck(OptimalTransportEntropyTest, SplitDatabaseSanityCheck):
    """
    Split risk dataset and transport one half onto the other half, with entropy regularization.
    """
    pass

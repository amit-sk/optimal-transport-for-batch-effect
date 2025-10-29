from tests.sanity_check import SanityCheck
from tests.optimal_transport_test_with_entropy import OptimalTransportEntropyTest

class SanityCheckWithEntropy(SanityCheck, OptimalTransportEntropyTest):
    """
    transporting noisy RISK data onto original RISK data with entropy regularization.
    """
    pass


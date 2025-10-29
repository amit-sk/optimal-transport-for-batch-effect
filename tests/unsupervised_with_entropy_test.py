import numpy as np
import ot
from tests.unsupervised_transport_test import UnsupervisedTransportTest
from tests.optimal_transport_test_with_entropy import OptimalTransportEntropyTest

class UnsupervisedWithEntropyTest(UnsupervisedTransportTest, OptimalTransportEntropyTest):
    pass


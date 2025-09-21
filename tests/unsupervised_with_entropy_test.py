import numpy as np
import ot
from tests.unsupervised_transport_test import UnsupervisedTransportTest

class UnsupervisedWithEntropyTest(UnsupervisedTransportTest):
    def __init__(self, epsilon=None, should_run_pcoa=False):
        self.epsilon = epsilon
        super().__init__(should_run_pcoa)

    def transport(self):
        if self.epsilon is None:
            e, coupling, gw_dist = self._finetune_epsilon()
            print(f'Using {e=}')
            self.coupling = coupling
            self.gw_distance = gw_dist
        else:
            self.coupling, log = ot.gromov.entropic_gromov_wasserstein(self.target_distance_matrix, self.source_distance_matrix, verbose=False, log=True, epsilon=self.epsilon)
            self.gw_distance = log['gw_dist']

        print(f'GW distance: {self.gw_distance}')
        self._get_projected()

    def _finetune_epsilon(self):
        min_dist = None
        min_e = None
        min_coupling = None

        for e in np.logspace(-1, -3.25, 12):
            print(f'testing {e=}')
            coupling, log = ot.gromov.entropic_gromov_wasserstein(self.target_distance_matrix, self.source_distance_matrix, log=True, epsilon=e, max_iter=800)
            if min_dist is None or log['gw_dist'] < min_dist:
                min_dist = log['gw_dist']
                min_e = e
                min_coupling = coupling
        
        return min_e, min_coupling, min_dist


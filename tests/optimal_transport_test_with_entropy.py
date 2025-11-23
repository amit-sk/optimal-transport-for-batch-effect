import ot
import numpy as np
from tests.optimal_transport_test import OptimalTransportTest

class OptimalTransportEntropyTest(OptimalTransportTest):
    def __init__(self, *, epsilon=None, should_run_pcoa=False, should_show_pcoa=False, should_test_signal_retention=False, **kwargs):
        super().__init__(should_run_pcoa=should_run_pcoa, should_show_pcoa=should_show_pcoa, should_test_signal_retention=should_test_signal_retention, **kwargs)
        self.epsilon = epsilon

    def transport(self, **kwargs_for_ot):
        with open(self._get_file_path('transport_log.txt'), 'w') as f:
            if self.epsilon is None:
                e, coupling, gw_dist = self._finetune_epsilon(f, **kwargs_for_ot)
                print(f'Using {e=}')
                f.write(f'Using {e=}\n')
                self.coupling = coupling
                self.gw_distance = gw_dist
            else:
                self.coupling, log = ot.gromov.entropic_gromov_wasserstein(self.target_distance_matrix, self.source_distance_matrix, verbose=False, log=True, epsilon=self.epsilon, **kwargs_for_ot)
                self.gw_distance = log['gw_dist']

            print(f'GW distance: {self.gw_distance}')
            f.write(f'GW distance: {self.gw_distance}\n')
            self._get_projected()

    def _finetune_epsilon(self, f, **kwargs_for_ot):
        min_dist = None
        min_e = None
        min_coupling = None

        for e in np.logspace(-1, -3.25, 12):
            print(f'testing {e=}')
            f.write(f'testing {e=}\n')
            coupling, log = ot.gromov.entropic_gromov_wasserstein(self.target_distance_matrix, self.source_distance_matrix, log=True, epsilon=e, max_iter=800, **kwargs_for_ot)
            if min_dist is None or log['gw_dist'] < min_dist:
                min_dist = log['gw_dist']
                min_e = e
                min_coupling = coupling
        
        return min_e, min_coupling, min_dist
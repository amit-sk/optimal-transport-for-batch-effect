import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import variance_tests
from optimal_transport_test import OptimalTransferTest

class UnsupervisedTransportTest(OptimalTransferTest):
    def __init__(self, should_run_pcoa=False):
        risk_data = pd.read_csv("risk_data.csv")
        mucosalibd_data = pd.read_csv("mucosalibd_data.csv")

        super().__init__(mucosalibd_data, risk_data, should_run_pcoa=should_run_pcoa,
                         source_dataset_name='mucosalibd', target_dataset_name='risk')

    def show_variance_pre_transport(self):
        # create combined data (to show variance before alignment)
        combined_data = pd.concat([self.target_dataset, self.source_dataset])
        combined_data.fillna(0.0, inplace=True)
        combined_data.set_index('sample_id', inplace=True)

        # show variance before alignment
        print("\nComparing variance between risk and mucosalibd (before alignment):")
        variance_tests.show_variance(combined_data, 'dataset', should_run_pcoa=self.should_run_pcoa)
        print("\nComparing variance between phenotypes in combined risk and mucosalibd:")
        variance_tests.show_variance(combined_data, 'phenotype', should_run_pcoa=self.should_run_pcoa)

    def show_variance_post_transport(self):
        risk_data = self.target_dataset.copy()
        mucosalibd_data = self.source_dataset.copy()
        projected = self.projected_data.copy()

        if self.should_run_pcoa:
            self._observe_coupling_matrix(self.coupling, risk_data, mucosalibd_data)

        # titration plot to measure batch effect
        if self.should_run_pcoa:
            png_path = os.path.join('titrations', self.__class__.__name__ + '_titration.png')
            variance_tests.Metrics.titration(self.source_dataset, self.target_dataset, self.projected_data, repeats=10, png_name=png_path)

        # compare projection and risk (post transport)
        combined_data = pd.concat([risk_data, projected])
        # combined_data = pd.concat([risk_data[risk_data.phenotype == 'control'], projected[risk_data.phenotype == 'control']])  # compare only controls (healthy)
        combined_data.set_index('sample_id', inplace=True)

        print("\nComparing variance between risk and projected:")
        variance_tests.show_variance(combined_data, 'dataset', should_run_pcoa=self.should_run_pcoa)
        print("\nComparing variance between phenotypes in combined risk and projected:")
        variance_tests.show_variance(combined_data, 'phenotype', should_run_pcoa=self.should_run_pcoa)

        print("\nComparing variance between dataset+phenotype in combined risk and projected:")
        risk_data['dataset+phenotype'] = 'RISK_' + risk_data['phenotype']
        projected['dataset+phenotype'] = 'Projected_' + projected['phenotype']
        combined_data = pd.concat([risk_data, projected])
        combined_data.set_index('sample_id', inplace=True)
        variance_tests.show_variance(combined_data, 'dataset+phenotype', should_run_pcoa=self.should_run_pcoa)

        # compare projection and mucosalibd (before and after transport)
        print("\nComparing variance between mucosalibd (original) and projected:")
        combined_data = pd.concat([mucosalibd_data, projected])
        combined_data.fillna(0.0, inplace=True)
        combined_data.set_index('sample_id', inplace=True)
        pairs = self._get_pairs(combined_data, '_mucosalibd', '_projected')
        variance_tests.show_variance(combined_data, 'dataset', pcoa_pairs=pairs, should_run_pcoa=self.should_run_pcoa)

        # how much each projection had moved - compare risk, mucosalibd, projected
        print("\nComparing variance between risk, mucosalibd and projected:")
        combined_data = pd.concat([risk_data, mucosalibd_data, projected])
        combined_data.fillna(0.0, inplace=True)
        combined_data.set_index('sample_id', inplace=True)
        pairs = self._get_pairs(combined_data, '_mucosalibd', '_projected')
        variance_tests.show_variance(combined_data, 'dataset', pcoa_pairs=pairs, should_run_pcoa=self.should_run_pcoa)

    def _observe_coupling_matrix(self, coupling, risk_data, mucosalibd_data):
        print("\nObserving coupling matrix...")

        d_all = {}
        c = pd.DataFrame(coupling) * 132  # sum of columns will be 1, sum of all columns will be 132
        for i in range(132):
            vals = c.iloc[:, i].value_counts()
            d = vals.to_dict()
            d.pop(0)
            for k,v in d.items():
                if k in d_all:
                    d_all[k] += v
                else:
                    d_all[k] = v

        plt.bar(d_all.keys(), d_all.values(), width=1/132)
        plt.bar(d_all.keys(), d_all.values(), width=1/200)
        plt.bar(d_all.keys(), d_all.values(), width=1/2000, color='black')
        plt.yticks(np.arange(0, max(d_all.values())+1, 2.0))
        plt.xticks(np.arange(0, max(d_all.keys())+0.01, 0.005))
        plt.title("Histogram of coupling matrix values (non-zero only)")
        plt.show()

        variance_tests.Draw.heatmap(coupling, risk_data['sample_id'], mucosalibd_data['sample_id'], cmap='Blues')

        # show spread - how many *unique* values in distribution
        # TODO: might be more informative to show number of non-zero values.
        spread_of_src = pd.DataFrame(coupling).nunique(axis=0)
        plt.hist(spread_of_src, bins=12)
        plt.title("Spread of each sample in src dataset (mucosalibd) in coupling matrix")
        plt.show()

        spread_of_src = pd.DataFrame(coupling).nunique(axis=1)
        plt.hist(spread_of_src, bins=12)
        plt.title("Spread of each sample in target dataset (risk) in coupling matrix")
        plt.show()


if __name__ == "__main__":
    UnsupervisedTransportTest(should_run_pcoa=False).run_test()

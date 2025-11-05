import ot
from ot.gromov import *
from ot.gromov._gw import *
from tests.unsupervised_transport_test import UnsupervisedTransportTest

class SupervisedPenaltyForOppositePhenotypeTest(UnsupervisedTransportTest):
    """
    Adds a penalty matrix to the Gromov-Wasserstein transport problem that penalizes transport between samples with opposite phenotypes.
    Alpha parameter controls the weight of the penalty (alpha=1 means penalty of 1).
    """
    def __init__(self, *, alpha=1, should_run_pcoa=False, **kwargs):
        super().__init__(should_run_pcoa=should_run_pcoa, **kwargs)
        self.alpha = alpha

    def transport(self):
        print(f'{self.alpha=} for penalty weight for opposite phenotypes')
        M = self.alpha * self._get_penalty_matrix_for_opposite_phenotypes()
        self.coupling, log = gromov_wasserstein_copy(self.target_distance_matrix, self.source_distance_matrix, M=M, verbose=False, log=True)
        self.gw_distance = log['gw_dist']
        print(f'GW distance: {self.gw_distance}')
        self._get_projected()

    def _get_penalty_matrix_for_opposite_phenotypes(self):
        """
        Create a penalty matrix that penalizes transport between samples with opposite phenotypes.
        """
        source_phenotypes = self.source_dataset['phenotype'].values
        target_phenotypes = self.target_dataset['phenotype'].values

        penalty_matrix = np.zeros((len(target_phenotypes), len(source_phenotypes)))

        for i, t_pheno in enumerate(target_phenotypes):
            for j, s_pheno in enumerate(source_phenotypes):
                if t_pheno != s_pheno:
                    penalty_matrix[i, j] = 1.0  # Assign a penalty for opposite phenotypes

        return penalty_matrix


# TODO: at the moment just copied the code from POT directly, need to think if there's a better way to do this
def gromov_wasserstein_copy(
    C1,
    C2,
    M=None,
    p=None,
    q=None,
    loss_fun="square_loss",
    symmetric=None,
    log=False,
    armijo=False,
    G0=None,
    max_iter=1e4,
    tol_rel=1e-9,
    tol_abs=1e-9,
    **kwargs,
):
    arr = [C1, C2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)
    if q is not None:
        arr.append(list_to_array(q))
    else:
        q = unif(C2.shape[0], type_as=C1)
    if G0 is not None:
        G0_ = G0
        arr.append(G0)

    nx = get_backend(*arr)
    p0, q0, C10, C20 = p, q, C1, C2

    p = nx.to_numpy(p0)
    q = nx.to_numpy(q0)
    C1 = nx.to_numpy(C10)
    C2 = nx.to_numpy(C20)
    if symmetric is None:
        symmetric = np.allclose(C1, C1.T, atol=1e-10) and np.allclose(
            C2, C2.T, atol=1e-10
        )

    if G0 is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = nx.to_numpy(G0_)
        # Check marginals of G0
        np.testing.assert_allclose(G0.sum(axis=1), p, atol=1e-08)
        np.testing.assert_allclose(G0.sum(axis=0), q, atol=1e-08)
    # cg for GW is implemented using numpy on CPU
    np_ = NumpyBackend()

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun, np_)

    def f(G):
        return gwloss(constC, hC1, hC2, G, np_)

    if symmetric:

        def df(G):
            return gwggrad(constC, hC1, hC2, G, np_)
    else:
        constCt, hC1t, hC2t = init_matrix(C1.T, C2.T, p, q, loss_fun, np_)

        def df(G):
            return 0.5 * (
                gwggrad(constC, hC1, hC2, G, np_) + gwggrad(constCt, hC1t, hC2t, G, np_)
            )

    if armijo:

        def line_search(cost, G, deltaG, Mi, cost_G, df_G, **kwargs):
            return line_search_armijo(cost, G, deltaG, Mi, cost_G, nx=np_, **kwargs)
    else:

        def line_search(cost, G, deltaG, Mi, cost_G, df_G, **kwargs):
            return solve_gromov_linesearch(
                G,
                deltaG,
                cost_G,
                hC1,
                hC2,
                M=0.0 if M is None else M,
                reg=1.0,
                nx=np_,
                symmetric=symmetric,
                **kwargs,
            )

    if not nx.is_floating_point(C10):
        warnings.warn(
            "Input structure matrix consists of integers. The transport plan will be "
            "casted accordingly, possibly resulting in a loss of precision. "
            "If this behaviour is unwanted, please make sure your input "
            "structure matrix consists of floating point elements.",
            stacklevel=2,
        )

    if log:
        res, log = cg(
            p,
            q,
            0.0 if M is None else M,
            1.0,
            f,
            df,
            G0,
            line_search,
            log=True,
            numItermax=max_iter,
            stopThr=tol_rel,
            stopThr2=tol_abs,
            **kwargs,
        )
        log["gw_dist"] = nx.from_numpy(log["loss"][-1], type_as=C10)
        log["u"] = nx.from_numpy(log["u"], type_as=C10)
        log["v"] = nx.from_numpy(log["v"], type_as=C10)
        return nx.from_numpy(res, type_as=C10), log
    else:
        return nx.from_numpy(
            cg(
                p,
                q,
                0.0 if M is None else M,
                1.0,
                f,
                df,
                G0,
                line_search,
                log=False,
                numItermax=max_iter,
                stopThr=tol_rel,
                stopThr2=tol_abs,
                **kwargs,
            ),
            type_as=C10,
        )


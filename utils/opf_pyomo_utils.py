"""AC Optimal Power Flow (ACOPF) solver built on Pyomo / IPOPT.

This module implements a full-featured ACOPF with support for:

- **Rectangular or polar** voltage formulations
- **Temperature-dependent** line ratings (quadratic heat-balance
  approximation via IEEE Std 738-2012)
- **N-1 security constraints** — both AC (full post-contingency power
  flow) and linearised (LODF) variants
- **DC lines, storage**, and **load-shedding** variables
- **Warm-starting** from a previous OPF solution

The ``ACOPF`` class builds a Pyomo ``ConcreteModel``, adds variables,
constraints, and an objective, then solves with IPOPT and returns a
results dictionary.

Section index
-------------
1. Sparse-Matrix Helpers
2. ACOPF Class
   2a. Constructor & data preparation
   2b. Variable creation
   2c. Base-case constraints
   2d. Quadratic thermal approximation
   2e. Contingency (N-1) constraints
   2f. Objective function
   2g. Model assembly & solver interface
   2h. Results processing
3. Post-Processing Utilities
4. Test Entry Point
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import sys
from collections import defaultdict

import numpy as np
import pyomo.environ as pyo

# Ensure both project root and utils/ are on sys.path so imports work
# regardless of whether this file is run directly or imported from root.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
for _p in (_project_root, _this_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pypower.api import (
    makeBdc, makeLODF, makePTDF, makeYbus, runopf,
)
from pypower.ext2int import ext2int
from pypower import idx_bus, idx_gen, idx_brch
from scipy.io import loadmat

from utils.heat_flow_utils import coefficient_quadratic_approximation


# ===========================================================================
# 1. Sparse-Matrix Helpers
# ===========================================================================

def _extract_sparse_row_data(sparse_matrix):
    """Pre-extract non-zero structure from a scipy sparse matrix.

    Returns
    -------
    dict
        ``{row_idx: (col_indices, real_values, imag_values)}``
        Avoids repeated sparse indexing during constraint construction.
    """
    csr = sparse_matrix.tocsr()
    row_data = {}
    for row in range(csr.shape[0]):
        start, end = csr.indptr[row], csr.indptr[row + 1]
        cols = csr.indices[start:end]
        vals = csr.data[start:end]
        row_data[row] = (cols, vals.real.copy(), vals.imag.copy())
    return row_data


def _compute_current_from_sparse_row(row_nz_cols, row_real_vals,
                                     row_imag_vals, VR, VI):
    """Compute (I_real, I_imag) using only non-zero entries of one row.

    This is the core speed-up: instead of summing over all *N* buses we
    only iterate over the ~3–5 non-zero entries per row.
    """
    I_real = sum(row_real_vals[idx] * VR[k] - row_imag_vals[idx] * VI[k]
                 for idx, k in enumerate(row_nz_cols))
    I_imag = sum(row_real_vals[idx] * VI[k] + row_imag_vals[idx] * VR[k]
                 for idx, k in enumerate(row_nz_cols))
    return I_real, I_imag


# ===========================================================================
# 2. ACOPF Class
# ===========================================================================

class ACOPF:
    """AC Optimal Power Flow formulated as a Pyomo NLP.

    Parameters
    ----------
    ppc : dict
        PyPower case dict (``bus``, ``gen``, ``branch``, ``gencost``, …).
    solver_name : str
        NLP solver (default ``'ipopt'``).
    tol : float
        Solver convergence tolerance.
    initial_value : dict or None
        Warm-start arrays (``VM``, ``VA``, ``PG``, ``QG``, …).
    voltage_form : {'rectangular', 'polar'}
        Voltage variable formulation.
    quad_con : bool
        Use quadratic heat-balance approximation for thermal limits.
    td_cons : bool
        Enforce temperature-dependent current limits.
    contingency : list[tuple]
        ``(from_bus, to_bus)`` pairs for N-1 security constraints.
    safe_margin : float
        Multiplicative safety margin for branch flow limits.
    fix_sc : bool
        Apply the safety margin to flow limits.
    ac_sc : bool
        Full AC post-contingency power-flow constraints.
    dc_sc : bool
        Linearised (LODF) post-contingency constraints.
    tdac_sc : bool
        Temperature-dependent limits for post-contingency flows.
    angle_cons : bool
        Enforce phase-angle difference limits on branches.
    qlim : bool
        Enforce reactive generation capacity limits.
    reactive_demand_ratio : float or 'auto'
        Fixed Q/P ratio for loads, or ``'auto'`` to derive from data.
    weather : dict or None
        Ambient weather conditions for heat-balance calculations.
    conductor : dict or None
        Conductor physical properties.
    """

    # -----------------------------------------------------------------------
    # 2a. Constructor & data preparation
    # -----------------------------------------------------------------------

    def __init__(self, ppc, solver_name='ipopt', tol=1e-5,
                 initial_value=None, voltage_form='rectangular',
                 quad_con=False, td_cons=False,
                 contingency=None, safe_margin=1,
                 fix_sc=False, ac_sc=False, dc_sc=False, tdac_sc=False,
                 angle_cons=False, qlim=True,
                 reactive_demand_ratio=0.15,
                 weather=None, conductor=None):
        if contingency is None:
            contingency = []

        self.ppc = ppc
        self.solver_name = solver_name
        self.tol = tol
        self.initial_value = initial_value
        self.voltage_form = voltage_form

        self.quad_con = quad_con
        self.td_cons = td_cons
        self.contingency = contingency
        self.safe_margin = safe_margin
        self.fix_sc = fix_sc
        self.ac_sc = ac_sc
        self.dc_sc = dc_sc
        self.tdac_sc = tdac_sc
        self.angle_cons = angle_cons
        self.qlim = qlim
        self.reactive_demand_ratio = reactive_demand_ratio

        self.weather = weather
        self.conductor = conductor

        # --- Extract base arrays -------------------------------------------
        self.bus = ppc["bus"]
        self.gen = ppc["gen"]
        self.branch = ppc["branch"]
        self.dcline = ppc.get('dcline', [])
        self.storage = ppc.get('storage', [])
        self.gencost = ppc["gencost"]
        self.dclinecost = ppc.get('dclinecost', [])
        self.storagecost = ppc.get('storagecost', [])

        self.baseMVA = ppc["baseMVA"]
        self.baseKV = ppc["bus"][0, idx_bus.BASE_KV]
        self.baseI = self.baseMVA / self.baseKV

        self.nbus = len(self.bus)
        self.ngen = len(self.gen)
        self.ndcline = len(self.dcline)
        self.nstorage = len(self.storage)

        # Number of parallel conductors per branch (0 → treat as 1)
        num_parallel = ppc['branch'][:, -3].copy()
        num_parallel[num_parallel == 0] = 1
        self.num_parallel = num_parallel

        # --- Admittance matrices -------------------------------------------
        self.Ybus, self.Yf, self.Yt = makeYbus(
            self.baseMVA, self.bus, self.branch)

        # Pre-extract sparse row data for O(nnz) constraint building
        self.Ybus_rows = _extract_sparse_row_data(self.Ybus)
        self.Yf_rows = _extract_sparse_row_data(self.Yf)
        self.Yt_rows = _extract_sparse_row_data(self.Yt)

        # --- Bus → component maps (O(n_comp) build) -----------------------
        self.gen_at_bus = defaultdict(list)
        for j in range(self.ngen):
            self.gen_at_bus[int(self.gen[j, 0])].append(j)

        self.storage_at_bus = defaultdict(list)
        for j in range(self.nstorage):
            self.storage_at_bus[int(self.storage[j, 0])].append(j)

        self.dcf_at_bus = defaultdict(list)
        self.dct_at_bus = defaultdict(list)
        for j in range(self.ndcline):
            self.dcf_at_bus[int(self.dcline[j, 0])].append(j)
            self.dct_at_bus[int(self.dcline[j, 1])].append(j)

        # Dense masks for vectorised post-processing (results extraction)
        self.gen_mask = np.array([
            [1.0 if self.gen[j, 0] == i else 0.0
             for j in range(self.ngen)]
            for i in range(self.nbus)])
        self.storage_mask = np.array([
            [1.0 if self.storage[j, 0] == i else 0.0
             for j in range(self.nstorage)]
            for i in range(self.nbus)])
        self.dcf_mask = np.array([
            [1.0 if self.dcline[j, 0] == i else 0.0
             for j in range(self.ndcline)]
            for i in range(self.nbus)])
        self.dct_mask = np.array([
            [1.0 if self.dcline[j, 1] == i else 0.0
             for j in range(self.ndcline)]
            for i in range(self.nbus)])

        self.fbus = self.branch[:, 0].astype(int)
        self.tbus = self.branch[:, 1].astype(int)

        # --- Derived data --------------------------------------------------
        self.branch_idx_map = self._create_branch_mapping()
        self.cont_data = self._create_contingency_admittance(contingency)
        self.betas = self._create_quadratic_approximation(conductor, weather)

        self.model = None

    def _create_branch_mapping(self):
        """Return dict ``{(i, j): branch_index}`` for both directions."""
        mapping = {}
        for idx, br in enumerate(self.branch):
            i, j = int(br[0]), int(br[1])
            mapping[(i, j)] = idx
            mapping[(j, i)] = idx
        return mapping

    def _create_contingency_admittance(self, contingency):
        """Build per-contingency admittance matrices (Ybus, Yf, Yt).

        Each contingency is modelled by setting the faulted branch's
        ``BR_STATUS`` to 0 and recomputing the admittance matrices.
        """
        cont_data = []
        for i, j in contingency:
            branch_idx = self.branch_idx_map.get((i, j))
            if branch_idx is None:
                raise ValueError(
                    f"No branch found between buses {i} and {j}")
            branch_c = self.branch.copy()
            branch_c[branch_idx, idx_brch.BR_STATUS] = 0
            Ybus_c, Yf_c, Yt_c = makeYbus(
                self.baseMVA, self.bus, branch_c)
            cont_data.append({
                'branch_idx': branch_idx,
                'Ybus': Ybus_c, 'Yf': Yf_c, 'Yt': Yt_c,
            })
        return cont_data

    def _create_quadratic_approximation(self, conductor, weather):
        """Compute quadratic heat-balance coefficients (beta0, beta1, beta2).

        Returns ``[None, None, None]`` when ``quad_con`` is disabled.
        """
        if not self.quad_con:
            return [None, None, None]
        self.num_bundle = conductor['num_bundle']
        self.Tmax = conductor['max_temperature']
        if 'segment' in self.ppc:
            self.seg_prop = self.ppc['segment'][:, :, 2]
        else:
            self.seg_prop = np.ones((len(self.branch), 1))
        beta0, beta1, beta2 = coefficient_quadratic_approximation(
            conductor, weather)
        return [beta0, beta1, beta2]

    # -----------------------------------------------------------------------
    # 2b. Variable creation
    # -----------------------------------------------------------------------

    def _add_base_variables(self):
        """Add base-case decision variables, with optional warm-start."""
        iv = self.initial_value
        warm = (iv is not None and isinstance(iv, dict)
                and 'VM' in iv and hasattr(iv['VM'], '__len__'))

        if warm:
            self._add_warm_start_variables(iv)
        else:
            self._add_flat_start_variables()

    def _add_warm_start_variables(self, iv):
        """Create variables initialised from a previous OPF solution."""
        m = self.model

        if self.voltage_form == 'rectangular':
            vm = iv.get('VM', np.ones(self.nbus))
            va = iv.get('VA', np.zeros(self.nbus))
            vr_init = {i: float(vm[i] * np.cos(va[i]))
                       for i in range(self.nbus)}
            vi_init = {i: float(vm[i] * np.sin(va[i]))
                       for i in range(self.nbus)}
            m.VR = pyo.Var(m.buses, domain=pyo.Reals, initialize=vr_init)
            m.VI = pyo.Var(m.buses, domain=pyo.Reals, initialize=vi_init)
        elif self.voltage_form == 'polar':
            vm_init = {i: float(iv['VM'][i]) for i in range(self.nbus)}
            va_init = {i: float(iv['VA'][i]) for i in range(self.nbus)}
            m.VM = pyo.Var(m.buses, domain=pyo.Reals, initialize=vm_init)
            m.VA = pyo.Var(m.buses, domain=pyo.Reals, initialize=va_init)
            self.VR = [m.VM[i] * pyo.cos(m.VA[i]) for i in m.buses]
            self.VI = [m.VM[i] * pyo.sin(m.VA[i]) for i in m.buses]
        else:
            raise ValueError(f"Invalid voltage form: {self.voltage_form}")

        def _init_dict(key, n):
            if key in iv and hasattr(iv[key], '__len__') \
                    and len(iv[key]) == n:
                return {i: float(iv[key][i]) for i in range(n)}
            return 0

        m.PG = pyo.Var(m.generators, domain=pyo.Reals,
                       initialize=_init_dict('PG', self.ngen))
        m.QG = pyo.Var(m.generators, domain=pyo.Reals,
                       initialize=_init_dict('QG', self.ngen))
        m.PDC = pyo.Var(m.dc_lines, domain=pyo.Reals,
                        initialize=_init_dict('PDC', self.ndcline))
        m.PS = pyo.Var(m.storages, domain=pyo.Reals,
                       initialize=_init_dict('PS', self.nstorage))
        m.LS = pyo.Var(m.buses, domain=pyo.Reals,
                       initialize=_init_dict('LS', self.nbus))

    def _add_flat_start_variables(self):
        """Create variables with a flat start (V = 1∠0, P = Q = 0)."""
        m = self.model

        if self.voltage_form == 'rectangular':
            m.VR = pyo.Var(m.buses, domain=pyo.Reals, initialize=1)
            m.VI = pyo.Var(m.buses, domain=pyo.Reals, initialize=0)
        elif self.voltage_form == 'polar':
            m.VM = pyo.Var(m.buses, domain=pyo.Reals, initialize=1)
            m.VA = pyo.Var(m.buses, domain=pyo.Reals, initialize=0)
            self.VR = [m.VM[i] * pyo.cos(m.VA[i]) for i in m.buses]
            self.VI = [m.VM[i] * pyo.sin(m.VA[i]) for i in m.buses]
        else:
            raise ValueError(f"Invalid voltage form: {self.voltage_form}")

        m.PG = pyo.Var(m.generators, domain=pyo.Reals, initialize=0)
        m.QG = pyo.Var(m.generators, domain=pyo.Reals, initialize=0)
        m.PDC = pyo.Var(m.dc_lines, domain=pyo.Reals, initialize=0)
        m.PS = pyo.Var(m.storages, domain=pyo.Reals, initialize=0)
        m.LS = pyo.Var(m.buses, domain=pyo.Reals, initialize=0)

    # -----------------------------------------------------------------------
    # 2c. Base-case constraints
    # -----------------------------------------------------------------------

    def _add_rec_voltage_constraints(self):
        """Rectangular voltage magnitude and angle bounds."""
        bus = self.bus
        self.model.voltage_constraint = pyo.ConstraintList()
        for i in range(self.nbus):
            self.model.VR[i].lb = 0
            self.model.VR[i].ub = bus[i, idx_bus.VMAX]
            self.model.VI[i].lb = -bus[i, idx_bus.VMAX]
            self.model.VI[i].ub = bus[i, idx_bus.VMAX]
            if self.bus[i][idx_bus.BUS_TYPE] == 3:
                self.model.VI[i].fix(0.0)
            # |V|² bounds
            self.model.voltage_constraint.add(
                self.model.VR[i]**2 + self.model.VI[i]**2
                <= bus[i, idx_bus.VMAX]**2)
            self.model.voltage_constraint.add(
                self.model.VR[i]**2 + self.model.VI[i]**2
                >= bus[i, idx_bus.VMIN]**2)
            # Angle within ±30°
            self.model.voltage_constraint.add(
                self.model.VI[i] >= np.tan(-np.pi / 6) * self.model.VR[i])
            self.model.voltage_constraint.add(
                self.model.VI[i] <= np.tan(np.pi / 6) * self.model.VR[i])

    def _add_pol_voltage_constraints(self):
        """Polar voltage magnitude and angle bounds."""
        bus = self.bus
        for i in range(self.nbus):
            self.model.VM[i].lb = bus[i, idx_bus.VMIN]
            self.model.VM[i].ub = bus[i, idx_bus.VMAX]
            if self.bus[i][idx_bus.BUS_TYPE] == 3:
                self.model.VA[i].fix(0.0)
            else:
                # Bound VA to prevent IPOPT drift due to sin/cos periodicity
                self.model.VA[i].lb = -np.pi
                self.model.VA[i].ub = np.pi

    def _add_load_shedding_constraints(self):
        """Bound load-shedding to [0, PD] at each bus."""
        for i in range(self.nbus):
            if self.bus[i, idx_bus.PD] > 0:
                self.model.LS[i].lb = 0
                self.model.LS[i].ub = self.bus[i, idx_bus.PD] / self.baseMVA
            else:
                self.model.LS[i].fix(0.0)

    def _add_operational_constraints(self):
        """Generator, DC-line, and storage operational bounds."""
        gen, dcline, storage = self.gen, self.dcline, self.storage

        for i in range(self.ngen):
            if gen[i, idx_gen.GEN_STATUS] == 1:
                self.model.PG[i].lb = gen[i, idx_gen.PMIN] / self.baseMVA
                self.model.PG[i].ub = gen[i, idx_gen.PMAX] / self.baseMVA
                if self.qlim:
                    self.model.QG[i].lb = gen[i, idx_gen.QMIN] / self.baseMVA
                    self.model.QG[i].ub = gen[i, idx_gen.QMAX] / self.baseMVA
                else:
                    # Relax Q limits to ±Pmax
                    self.model.QG[i].lb = -gen[i, idx_gen.PMAX] / self.baseMVA
                    self.model.QG[i].ub = gen[i, idx_gen.PMAX] / self.baseMVA
            else:
                self.model.PG[i].fix(0.0)
                self.model.QG[i].fix(0.0)

        for i in range(self.ndcline):
            if dcline[i, 2] == 1:
                self.model.PDC[i].lb = dcline[i, 9] / self.baseMVA
                self.model.PDC[i].ub = dcline[i, 10] / self.baseMVA
            else:
                self.model.PDC[i].fix(0.0)

        for i in range(self.nstorage):
            if storage[i, 1] == 1:
                self.model.PS[i].lb = 0
                self.model.PS[i].ub = storage[i, 3] / self.baseMVA
            else:
                self.model.PS[i].fix(0.0)

    def _add_power_balance_constraints(self):
        """Nodal active and reactive power balance (sparse-aware).

        Key optimisations vs a naive approach:
        1. Ybus current only sums over non-zero entries (O(nnz/row)
           instead of O(nbus)).
        2. Bus-component maps built in ``__init__`` at O(n_comp) cost.
        3. Reactive demand ratio pre-computed outside the loop.
        """
        model = self.model

        PD_load = self.bus[:, idx_bus.PD] / self.baseMVA
        QD_load = self.bus[:, idx_bus.QD] / self.baseMVA

        gen_at_bus = self.gen_at_bus
        storage_at_bus = self.storage_at_bus
        dcf_at_bus = self.dcf_at_bus
        dct_at_bus = self.dct_at_bus

        # Pre-compute reactive demand ratio per bus
        if isinstance(self.reactive_demand_ratio, str):
            load_ratios = np.zeros(self.nbus)
            nz = PD_load != 0
            load_ratios[nz] = QD_load[nz] / PD_load[nz]
        elif isinstance(self.reactive_demand_ratio, (int, float)):
            load_ratios = np.full(self.nbus, self.reactive_demand_ratio)
        else:
            raise ValueError(
                f"Invalid reactive demand ratio: {self.reactive_demand_ratio}")

        Ybus_rows = self.Ybus_rows

        if self.voltage_form == 'rectangular':
            VR, VI = model.VR, model.VI
        elif self.voltage_form == 'polar':
            VR, VI = self.VR, self.VI
        else:
            raise ValueError(f"Invalid voltage form: {self.voltage_form}")

        real_cons = {}
        imag_cons = {}

        for i in range(self.nbus):
            # Power injection at bus i
            real_inj = (
                sum(model.PG[j] for j in gen_at_bus[i])
                + sum(model.PS[j] for j in storage_at_bus[i])
                - sum(model.PDC[j] for j in dcf_at_bus[i])
                + sum(model.PDC[j] for j in dct_at_bus[i])
                + model.LS[i] - PD_load[i]
            )
            imag_inj = (
                sum(model.QG[j] for j in gen_at_bus[i])
                - QD_load[i] + model.LS[i] * load_ratios[i]
            )

            # Network current (only non-zero Ybus entries)
            nz_cols, y_real, y_imag = Ybus_rows[i]
            I_re, I_im = _compute_current_from_sparse_row(
                nz_cols, y_real, y_imag, VR, VI)

            real_flow = VR[i] * I_re + VI[i] * I_im
            imag_flow = VI[i] * I_re - VR[i] * I_im

            real_cons[i] = (real_inj == real_flow)
            imag_cons[i] = (imag_inj == imag_flow)

        model.real_power_balance_constraint = pyo.Constraint(
            model.buses, rule=lambda m, i: real_cons[i])
        model.imag_power_balance_constraint = pyo.Constraint(
            model.buses, rule=lambda m, i: imag_cons[i])

    def _add_branch_flow_constraints(self):
        """Branch apparent-power, current, and angle constraints (sparse).

        Yf / Yt current sums iterate only over non-zero entries per
        branch row (~2–3 terms) instead of all buses.
        """
        # Pre-filter active branches and collect data
        active = [l for l in range(len(self.branch))
                  if self.branch[l, idx_brch.BR_STATUS] == 1]

        branch_data = {}
        for l in active:
            i, j = int(self.branch[l, 0]), int(self.branch[l, 1])
            rate_a = self.branch[l, idx_brch.RATE_A] / self.baseMVA
            flow_limit_2 = ((rate_a * self.safe_margin) ** 2 if self.fix_sc
                            else rate_a ** 2)
            rate_b = self.branch[l, idx_brch.RATE_B]
            current_limit_2 = ((rate_b / self.baseI) ** 2
                               if (self.td_cons and rate_b > 0) else None)
            branch_data[l] = {
                'from': i, 'to': j,
                'has_flow_limit': rate_a > 0,
                'flow_limit_2': flow_limit_2,
                'current_limit_2': current_limit_2,
                'angle_max': (self.branch[l, idx_brch.ANGMAX] * np.pi / 180
                              if self.angle_cons else None),
                'angle_min': (self.branch[l, idx_brch.ANGMIN] * np.pi / 180
                              if self.angle_cons else None),
            }

        Yf_rows = self.Yf_rows
        Yt_rows = self.Yt_rows

        self.model.branch_flow_constraints = pyo.ConstraintList()
        if self.td_cons:
            self.model.branch_thermal_constraints = pyo.ConstraintList()
        if self.angle_cons:
            self.model.branch_angle_constraints = pyo.ConstraintList()

        if self.voltage_form == 'rectangular':
            VR, VI = self.model.VR, self.model.VI
        elif self.voltage_form == 'polar':
            VR, VI = self.VR, self.VI
        else:
            raise ValueError(f"Invalid voltage form: {self.voltage_form}")

        for l in active:
            d = branch_data[l]
            i, j = d['from'], d['to']

            # From-end and to-end currents (sparse)
            nz_f, yf_re, yf_im = Yf_rows[l]
            If_re, If_im = _compute_current_from_sparse_row(
                nz_f, yf_re, yf_im, VR, VI)
            nz_t, yt_re, yt_im = Yt_rows[l]
            It_re, It_im = _compute_current_from_sparse_row(
                nz_t, yt_re, yt_im, VR, VI)

            # Apparent-power flows
            Pf = VR[i] * If_re + VI[i] * If_im
            Qf = VI[i] * If_re - VR[i] * If_im
            Pt = VR[j] * It_re + VI[j] * It_im
            Qt = VI[j] * It_re - VR[j] * It_im

            # |S| ≤ RATE_A (skip if RATE_A = 0 → "no limit")
            if d['has_flow_limit']:
                self.model.branch_flow_constraints.add(
                    Pf ** 2 + Qf ** 2 <= d['flow_limit_2'])
                self.model.branch_flow_constraints.add(
                    Pt ** 2 + Qt ** 2 <= d['flow_limit_2'])

            # |I| ≤ thermal limit (skip if RATE_B = 0)
            if self.td_cons and d['current_limit_2'] is not None:
                if self.quad_con:
                    self._add_quadratic_thermal_constraint(
                        If_re, If_im, It_re, It_im, l, self.Tmax)
                else:
                    self.model.branch_thermal_constraints.add(
                        If_re ** 2 + If_im ** 2 <= d['current_limit_2'])
                    self.model.branch_thermal_constraints.add(
                        It_re ** 2 + It_im ** 2 <= d['current_limit_2'])

            # Phase-angle difference
            if self.angle_cons and self.voltage_form == 'polar':
                angle_diff = self.model.VA[i] - self.model.VA[j]
                self.model.branch_angle_constraints.add(
                    angle_diff <= d['angle_max'])
                self.model.branch_angle_constraints.add(
                    angle_diff >= d['angle_min'])

    # -----------------------------------------------------------------------
    # 2d. Quadratic thermal approximation
    # -----------------------------------------------------------------------

    def _add_quadratic_thermal_constraint(self, If_re, If_im,
                                          It_re, It_im, l, Tmax):
        """Add T(I²) ≤ Tmax using a quadratic approximation of the heat balance.

        The per-unit current is scaled to per-conductor Amps via
        ``baseI * 1000 / num_parallel / num_bundle``.  Instead of
        multiplying the Pyomo expression by a large constant (which
        hurts numerical conditioning), the scaling is absorbed into the
        beta coefficients and the constraint is normalised by Tmax.
        """
        I_scale_2 = (self.baseI * 1000
                     / self.num_parallel[l] / self.num_bundle) ** 2
        If2_pu = If_re ** 2 + If_im ** 2
        It2_pu = It_re ** 2 + It_im ** 2

        beta0, beta1, beta2 = self.betas

        def _add_con(b0, b1, b2, I2_pu):
            """Normalised constraint: T(I) / Tmax ≤ 1."""
            b0_n = float(b0 / Tmax)
            b1_n = float(b1 * I_scale_2 / Tmax)
            b2_n = float(b2 * I_scale_2 ** 2 / Tmax)
            self.model.branch_thermal_constraints.add(
                b0_n + b1_n * I2_pu + b2_n * I2_pu ** 2 <= 1.0)

        # Single-value coefficients
        if beta0.shape[0] == 1:
            _add_con(beta0[0], beta1[0], beta2[0], If2_pu)
            _add_con(beta0[0], beta1[0], beta2[0], It2_pu)
        # Branch-specific coefficients
        elif beta0.shape[1] == 1:
            _add_con(beta0[l], beta1[l], beta2[l], If2_pu)
            _add_con(beta0[l], beta1[l], beta2[l], It2_pu)
        # Segment-specific coefficients
        else:
            for s in range(self.seg_prop.shape[1]):
                if self.seg_prop[l, s] > 0:
                    _add_con(beta0[l, s], beta1[l, s], beta2[l, s], If2_pu)
                    _add_con(beta0[l, s], beta1[l, s], beta2[l, s], It2_pu)

    # -----------------------------------------------------------------------
    # 2e. Contingency (N-1) constraints
    # -----------------------------------------------------------------------

    def _add_contingency_variables(self):
        """Add post-contingency voltage, generation, and load-shedding variables."""
        m = self.model
        nc = range(len(self.contingency))
        m.VM_c = pyo.Var(nc, m.buses, domain=pyo.Reals, initialize=1.0)
        m.VA_c = pyo.Var(nc, m.buses, domain=pyo.Reals, initialize=0.0)
        m.QG_c = pyo.Var(nc, m.generators, domain=pyo.Reals, initialize=0.0)
        m.LS_c = pyo.Var(nc, m.buses, domain=pyo.Reals, initialize=0.0)
        m.PG_c = pyo.Var(nc, m.generators, domain=pyo.Reals, initialize=0.0)

        self.VR_c = {}
        self.VI_c = {}
        for c in nc:
            self.VR_c[c] = [m.VM_c[c, i] * pyo.cos(m.VA_c[c, i])
                            for i in m.buses]
            self.VI_c[c] = [m.VM_c[c, i] * pyo.sin(m.VA_c[c, i])
                            for i in m.buses]

    def _add_linear_security_constraint(self):
        """LODF-based linearised post-contingency flow limits."""
        baseMVA = self.ppc['baseMVA']
        bus, branch = self.ppc['bus'], self.ppc['branch']

        PTDF = makePTDF(baseMVA, bus, branch)
        LODF = makeLODF(branch, PTDF)
        _, Bf, _, _ = makeBdc(baseMVA, bus, branch)

        for l in range(len(self.branch)):
            line_flow = sum(
                Bf[l, k] * self.model.VA[k] for k in self.model.buses)
            flow_limit = self.branch[l, idx_brch.RATE_A] / self.baseMVA
            for li, lj in self.contingency:
                lc = self.branch_idx_map[(li, lj)]
                line_c_flow = sum(
                    Bf[lc, k] * self.model.VA[k]
                    for k in self.model.buses)
                self.model.branch_flow_constraints.add(
                    line_flow + line_c_flow * LODF[l, lc] <= flow_limit)

    def _add_post_operational_constraints(self):
        """Post-contingency voltage, Q-gen, and ramp limits."""
        bus, gen = self.bus, self.gen
        nc = len(self.contingency)

        for i in range(self.nbus):
            for c in range(nc):
                self.model.VM_c[c, i].lb = bus[i, idx_bus.VMIN]
                self.model.VM_c[c, i].ub = bus[i, idx_bus.VMAX]
                self.model.LS_c[c, i].lb = 0
                self.model.LS_c[c, i].ub = max(
                    0.0, bus[i, idx_bus.PD] / self.baseMVA)
                if self.bus[i][idx_bus.BUS_TYPE] == 3:
                    self.model.VA_c[c, i].fix(0.0)
                else:
                    self.model.VA_c[c, i].lb = -np.pi
                    self.model.VA_c[c, i].ub = np.pi

        for i in range(self.ngen):
            for c in range(nc):
                if gen[i, idx_gen.GEN_STATUS] == 1:
                    if self.qlim:
                        self.model.QG_c[c, i].lb = (
                            gen[i, idx_gen.QMIN] / self.baseMVA)
                        self.model.QG_c[c, i].ub = (
                            gen[i, idx_gen.QMAX] / self.baseMVA)
                    else:
                        self.model.QG_c[c, i].lb = (
                            -gen[i, idx_gen.PMAX] / self.baseMVA)
                        self.model.QG_c[c, i].ub = (
                            gen[i, idx_gen.PMAX] / self.baseMVA)
                else:
                    self.model.QG_c[c, i].fix(0.0)

        # Ramp constraints (preventive SCOPF: fix pre-dispatch P for
        # non-reference generators; allow ±5 % ramp for the reference bus)
        self.model.ramp_constraint = pyo.ConstraintList()
        for i in range(self.ngen):
            gen_bus = int(gen[i, idx_gen.GEN_BUS])
            for c in range(nc):
                if self.bus[gen_bus, idx_bus.BUS_TYPE] == 3:
                    pmax_pu = gen[i, idx_gen.PMAX] / self.baseMVA
                    self.model.PG_c[c, i].lb = (
                        gen[i, idx_gen.PMIN] / self.baseMVA)
                    self.model.PG_c[c, i].ub = pmax_pu
                    self.model.ramp_constraint.add(
                        self.model.PG_c[c, i] - self.model.PG[i]
                        <= 0.05 * pmax_pu)
                    self.model.ramp_constraint.add(
                        self.model.PG_c[c, i] - self.model.PG[i]
                        >= -0.05 * pmax_pu)
                elif gen[i, idx_gen.GEN_STATUS] == 1:
                    self.model.ramp_constraint.add(
                        self.model.PG_c[c, i] - self.model.PG[i] == 0)
                else:
                    self.model.PG_c[c, i].fix(0.0)

    def _add_post_power_balance_constraints(self):
        """Post-contingency nodal power balance (sparse-aware)."""
        self.model.cont_real_power_balance = pyo.ConstraintList()
        self.model.cont_imag_power_balance = pyo.ConstraintList()
        model = self.model

        gen_at_bus = self.gen_at_bus
        dcf_at_bus = self.dcf_at_bus
        dct_at_bus = self.dct_at_bus
        PD_load = self.bus[:, idx_bus.PD] / self.baseMVA
        QD_load = self.bus[:, idx_bus.QD] / self.baseMVA

        if isinstance(self.reactive_demand_ratio, str):
            rr = np.zeros(self.nbus)
            nz = PD_load != 0
            rr[nz] = QD_load[nz] / PD_load[nz]
        else:
            rr = np.full(self.nbus, self.reactive_demand_ratio)

        cont_Ybus_rows = [
            _extract_sparse_row_data(cd['Ybus']) for cd in self.cont_data]

        for i in range(self.nbus):
            for c in range(len(self.contingency)):
                real_inj = (
                    sum(model.PG_c[c, j] for j in gen_at_bus[i])
                    + model.LS_c[c, i] - PD_load[i])
                imag_inj = (
                    sum(model.QG_c[c, j] for j in gen_at_bus[i])
                    - QD_load[i] + model.LS_c[c, i] * rr[i])

                nz_cols, y_re, y_im = cont_Ybus_rows[c][i]
                I_re, I_im = _compute_current_from_sparse_row(
                    nz_cols, y_re, y_im, self.VR_c[c], self.VI_c[c])
                real_flow = (self.VR_c[c][i] * I_re
                             + self.VI_c[c][i] * I_im)
                imag_flow = (self.VI_c[c][i] * I_re
                             - self.VR_c[c][i] * I_im)

                # Skip trivially satisfied (both sides are float64 zero)
                if not (isinstance(real_inj + self.epsilon, np.float64)
                        and isinstance(real_flow + self.epsilon, np.float64)):
                    model.cont_real_power_balance.add(real_inj == real_flow)
                if not (isinstance(imag_inj + self.epsilon, np.float64)
                        and isinstance(imag_flow + self.epsilon, np.float64)):
                    model.cont_imag_power_balance.add(imag_inj == imag_flow)

    def _add_security_constraints(self):
        """Full AC post-contingency operational + flow constraints."""
        self._add_post_operational_constraints()
        self._add_post_power_balance_constraints()

        # Post-contingency branch flow (sparse-aware)
        self.model.cont_branch_flow_constraint = pyo.ConstraintList()
        model = self.model

        cont_Yf_rows = [
            _extract_sparse_row_data(cd['Yf']) for cd in self.cont_data]
        cont_Yt_rows = [
            _extract_sparse_row_data(cd['Yt']) for cd in self.cont_data]

        for l in range(len(self.branch)):
            if self.branch[l, idx_brch.BR_STATUS] != 1:
                continue
            i = int(self.branch[l, 0])
            j = int(self.branch[l, 1])
            flow_limit_2 = (
                self.branch[l, idx_brch.RATE_A] / self.baseMVA) ** 2

            for c in range(len(self.contingency)):
                post_br = self.branch_idx_map.get(self.contingency[c])
                if l == post_br:
                    continue  # faulted branch — skip

                # From-end current
                nz_f, yf_r, yf_i = cont_Yf_rows[c][l]
                If_re, If_im = _compute_current_from_sparse_row(
                    nz_f, yf_r, yf_i, self.VR_c[c], self.VI_c[c])
                # To-end current
                nz_t, yt_r, yt_i = cont_Yt_rows[c][l]
                It_re, It_im = _compute_current_from_sparse_row(
                    nz_t, yt_r, yt_i, self.VR_c[c], self.VI_c[c])

                # Power flows
                Pf = self.VR_c[c][i] * If_re + self.VI_c[c][i] * If_im
                Qf = self.VI_c[c][i] * If_re - self.VR_c[c][i] * If_im
                Pt = self.VR_c[c][j] * It_re + self.VI_c[c][j] * It_im
                Qt = self.VI_c[c][j] * It_re - self.VR_c[c][j] * It_im

                # |S| limit
                if not isinstance(Pf + Qf + self.epsilon, np.float64):
                    model.cont_branch_flow_constraint.add(
                        Pf ** 2 + Qf ** 2 <= flow_limit_2)
                if not isinstance(Pt + Qt + self.epsilon, np.float64):
                    model.cont_branch_flow_constraint.add(
                        Pt ** 2 + Qt ** 2 <= flow_limit_2)

                # |I| thermal limit (post-contingency)
                if self.tdac_sc:
                    cur_lim_2 = (
                        self.branch[l, idx_brch.RATE_B] / self.baseI) ** 2
                    if not isinstance(If_re + If_im + self.epsilon,
                                      np.float64):
                        model.cont_branch_flow_constraint.add(
                            If_re ** 2 + If_im ** 2 <= cur_lim_2)
                    if not isinstance(It_re + It_im + self.epsilon,
                                      np.float64):
                        model.cont_branch_flow_constraint.add(
                            It_re ** 2 + It_im ** 2 <= cur_lim_2)

                # Angle limits
                if self.angle_cons:
                    ang_max = (
                        self.branch[l, idx_brch.ANGMAX] * np.pi / 180)
                    ang_min = (
                        self.branch[l, idx_brch.ANGMIN] * np.pi / 180)
                    model.cont_branch_flow_constraint.add(
                        model.VA_c[c, i] - model.VA_c[c, j] <= ang_max)
                    model.cont_branch_flow_constraint.add(
                        model.VA_c[c, i] - model.VA_c[c, j] >= ang_min)

    # -----------------------------------------------------------------------
    # 2f. Objective function
    # -----------------------------------------------------------------------

    def _add_objective_function(self):
        """Minimise generation cost + load-shedding penalty."""
        # Normalise by max marginal cost for numerical stability
        cost_norm = max(np.max(self.gencost[:, 5]), 1.0)
        LS_PENALTY = 1e3
        QUA_REG = 1e-3
        ls_cost = cost_norm * LS_PENALTY

        generation_cost = sum(
            self.model.PG[i] * self.gencost[i, 5]
            for i in self.model.generators) / cost_norm

        load_shedding_cost = sum(
            QUA_REG * self.model.LS[i] ** 2 + self.model.LS[i] * ls_cost
            for i in self.model.buses) / cost_norm

        if self.contingency and self.ac_sc:
            nc = len(self.contingency)
            slack_gen_cost = sum(
                self.model.PG_c[c, i] * self.gencost[i, 5]
                for c in range(nc)
                for i in self.model.generators) / cost_norm
            cont_ls_cost = sum(
                QUA_REG * self.model.LS_c[c, i] ** 2
                + self.model.LS_c[c, i] * ls_cost
                for c in range(nc)
                for i in self.model.buses) / cost_norm
            obj_expr = (generation_cost + load_shedding_cost
                        + cont_ls_cost + slack_gen_cost)
        else:
            obj_expr = generation_cost + load_shedding_cost

        self.model.objective = pyo.Objective(
            expr=obj_expr, sense=pyo.minimize)

    # -----------------------------------------------------------------------
    # 2g. Model assembly & solver interface
    # -----------------------------------------------------------------------

    def _add_constraints(self):
        """Add all constraints to the model."""
        if self.voltage_form == 'rectangular':
            self._add_rec_voltage_constraints()
        elif self.voltage_form == 'polar':
            self._add_pol_voltage_constraints()
        else:
            raise ValueError(f"Invalid voltage form: {self.voltage_form}")

        self._add_load_shedding_constraints()
        self._add_operational_constraints()
        self._add_power_balance_constraints()
        self._add_branch_flow_constraints()

        if len(self.contingency) > 0:
            if self.dc_sc:
                self._add_linear_security_constraint()
            elif self.ac_sc:
                self._add_contingency_variables()
                self._add_security_constraints()

    def _initialize_model(self):
        """Create the Pyomo ConcreteModel with index sets and variables."""
        self.model = pyo.ConcreteModel()
        m = self.model
        m.buses = pyo.Set(initialize=range(self.nbus))
        m.ac_lines = pyo.Set(initialize=range(len(self.branch)))
        m.dc_lines = pyo.Set(initialize=range(self.ndcline))
        m.storages = pyo.Set(initialize=range(self.nstorage))
        m.generators = pyo.Set(initialize=range(self.ngen))
        m.contingency = pyo.Set(initialize=range(len(self.contingency)))
        self.epsilon = np.float64(0)
        self._add_base_variables()

    def solve(self):
        """Build and solve the ACOPF.

        Returns
        -------
        results : dict
            Solution arrays (``PG``, ``QG``, ``VM``, ``VA``, ``LS``, …)
            and violation metrics.
        """
        self._initialize_model()
        self._add_constraints()
        self._add_objective_function()

        solver = pyo.SolverFactory(self.solver_name)

        # --- Convergence tolerances ----------------------------------------
        solver.options['tol'] = self.tol
        solver.options['constr_viol_tol'] = self.tol
        solver.options['acceptable_tol'] = 1e-3
        solver.options['acceptable_constr_viol_tol'] = 1e-3
        solver.options['acceptable_iter'] = 10
        solver.options['max_iter'] = 5000

        # --- NLP scaling (critical for numerical stability) ----------------
        solver.options['nlp_scaling_method'] = 'gradient-based'
        solver.options['nlp_scaling_max_gradient'] = 100.0

        # --- Barrier strategy ----------------------------------------------
        solver.options['mu_strategy'] = 'adaptive'
        solver.options['adaptive_mu_globalization'] = 'obj-constr-filter'

        # --- Linear solver -------------------------------------------------
        solver.options['linear_solver'] = 'mumps'

        # --- Restoration phase for infeasible iterates ---------------------
        solver.options['required_infeasibility_reduction'] = 0.1
        solver.options['resto_failure_feasibility_threshold'] = 1e-2

        # --- Warm-start when initial values are provided -------------------
        if self.initial_value is not None:
            solver.options['warm_start_init_point'] = 'yes'
            solver.options['warm_start_bound_push'] = 1e-8
            solver.options['warm_start_mult_bound_push'] = 1e-8

        try:
            solver_result = solver.solve(self.model, tee=False)
        except Exception as e:
            solver_result = None
            print(f'Solver failed: {e}')

        return self._process_results(solver_result)

    # -----------------------------------------------------------------------
    # 2h. Results processing
    # -----------------------------------------------------------------------

    def _process_results(self, solver_result):
        """Convert Pyomo solution to a results dictionary."""
        results = process_optimization_results(
            self.model, self.ppc, self.Ybus, self.Yf, self.Yt,
            self.fbus, self.tbus,
            self.gen_mask, self.dcf_mask, self.dct_mask,
            self.storage_mask, self.voltage_form,
        )
        if solver_result is None:
            results['solver_status'] = 0
        elif (solver_result.solver.status == pyo.SolverStatus.ok
              and solver_result.solver.termination_condition
              == pyo.TerminationCondition.optimal):
            results['solver_status'] = 1
        else:
            results['solver_status'] = 0
        return results


# ===========================================================================
# 3. Post-Processing Utilities
# ===========================================================================

def process_optimization_results(model, ppc, Ybus, Yf, Yt, fbus, tbus,
                                 gen_mask, dcf_mask, dct_mask,
                                 storage_mask, voltage_form):
    """Extract variable values and compute flow / violation statistics.

    Parameters
    ----------
    model : pyo.ConcreteModel
        Solved Pyomo model.
    ppc : dict
        PyPower case data.
    Ybus, Yf, Yt : sparse matrix
        Admittance matrices.
    fbus, tbus : np.ndarray
        Branch from-bus / to-bus index arrays.
    gen_mask, dcf_mask, dct_mask, storage_mask : np.ndarray
        Bus × component indicator matrices.
    voltage_form : str

    Returns
    -------
    dict
        Keys include ``PG``, ``QG``, ``VM``, ``VA``, ``PDC``, ``PS``,
        ``LS``, ``S_pu``, ``I_pu``, ``obj``, violation arrays, etc.
    """
    bus = ppc["bus"]
    gen = ppc['gen']
    branch = ppc.get('branch', [])
    storage = ppc.get('storage', [])
    gencost = ppc['gencost']
    baseMVA = ppc['baseMVA']
    baseKV = bus[0, idx_bus.BASE_KV]
    BaseI = baseMVA / baseKV
    ngen = len(gen)

    # Variable extraction
    PG = np.array([model.PG[i].value for i in model.generators])
    QG = np.array([model.QG[i].value for i in model.generators])

    if voltage_form == 'rectangular':
        VM = np.array([
            (model.VR[i].value ** 2 + model.VI[i].value ** 2) ** 0.5
            for i in model.buses])
        VA = np.array([
            np.arctan2(model.VI[i].value, model.VR[i].value)
            for i in model.buses])
    elif voltage_form == 'polar':
        VM = np.array([model.VM[i].value for i in model.buses])
        VA = np.array([model.VA[i].value for i in model.buses])
    else:
        raise ValueError(f"Invalid voltage form: {voltage_form}")

    PDC = np.array([model.PDC[i].value for i in model.dc_lines])
    PS = np.array([model.PS[i].value for i in model.storages])
    LS = np.array([model.LS[i].value for i in model.buses])

    V = VM * np.exp(1j * VA)

    bus_real_inj, bus_imag_inj = calculate_power_injections(
        gen_mask, dcf_mask, dct_mask, storage_mask,
        PG, QG, PDC, PS, bus, baseMVA,
    )
    bus_flow, S, I = calculate_power_flows(
        V, Ybus, Yf, Yt, fbus, tbus)
    eq_vio, p_eq_vio, q_eq_vio, ineq_vio_s, ineq_vio_i = (
        calculate_violations(
            bus_real_inj, bus_imag_inj, bus_flow,
            S, I, branch, baseMVA, BaseI))
    obj = calculate_objective(PG, gencost, baseMVA, ngen)

    return {
        'baseMVA': baseMVA,
        'baseKV': baseKV,
        'baseI': BaseI,
        'obj': obj,
        'eq_vio': eq_vio,
        'p_eq_vio': p_eq_vio,
        'q_eq_vio': q_eq_vio,
        'ineq_vio_s': ineq_vio_s,
        'ineq_vio_i': ineq_vio_i,
        'PD': bus[:, idx_bus.PD] / baseMVA,
        'PG_capacity': gen[:, idx_gen.PMAX] / baseMVA,
        'ST_capacity': (storage[:, 3] / baseMVA if len(storage) > 0
                        else np.array([0])),
        'QD': bus[:, idx_bus.PD] / baseMVA,
        'Pex': None,
        'PG': PG, 'QG': QG, 'VM': VM, 'VA': VA,
        'PDC': PDC, 'PS': PS, 'LS': LS,
        'bus_flow': bus_flow, 'S_pu': S, 'I_pu': I,
    }


def calculate_power_injections(gen_mask, dcf_mask, dct_mask,
                               storage_mask, PG, QG, PDC, PS,
                               bus, baseMVA):
    """Compute real and reactive power injections at each bus.

    Returns
    -------
    bus_real_injection, bus_imag_injection : np.ndarray
    """
    bus_real = (np.sum(gen_mask * PG, axis=1)
                + np.sum(storage_mask * PS, axis=1)
                - np.sum(dcf_mask * PDC, axis=1)
                + np.sum(dct_mask * PDC, axis=1)
                - bus[:, idx_bus.PD] / baseMVA)
    bus_imag = (np.sum(gen_mask * QG, axis=1)
                - bus[:, idx_bus.QD] / baseMVA)
    return bus_real, bus_imag


def calculate_power_flows(V, Ybus, Yf, Yt, fbus, tbus):
    """Compute bus injections, branch apparent power, and branch current.

    Returns
    -------
    bus_flow : np.ndarray (complex)
    S : np.ndarray — max(|Sf|, |St|)
    I : np.ndarray — max(|If|, |It|)
    """
    bus_flow = V * np.conj(Ybus @ V)
    If = Yf @ V
    It = Yt @ V
    Sf = V[fbus] * np.conj(If)
    St = V[tbus] * np.conj(It)
    S = np.maximum(np.abs(Sf), np.abs(St))
    I = np.maximum(np.abs(If), np.abs(It))
    return bus_flow, S, I


def calculate_violations(bus_real_inj, bus_imag_inj, bus_flow,
                         S, I, branch, baseMVA, BaseI):
    """Compute power-balance and flow-limit violations.

    Returns
    -------
    eq_vio, p_eq_vio, q_eq_vio : np.ndarray
        Absolute mismatch (complex, real, reactive).
    ineq_vio_s : np.ndarray
        Apparent-power violation above RATE_A.
    ineq_vio_i : np.ndarray
        Current violation above RATE_B.
    """
    mismatch = bus_real_inj + 1j * bus_imag_inj - bus_flow
    p_eq_vio = np.abs(np.real(mismatch))
    q_eq_vio = np.abs(np.imag(mismatch))
    eq_vio = np.abs(mismatch)
    ineq_vio_s = np.maximum(
        S - branch[:, idx_brch.RATE_A] / baseMVA, 0)
    ineq_vio_i = np.maximum(
        I - branch[:, idx_brch.RATE_B] / BaseI, 0)
    return eq_vio, p_eq_vio, q_eq_vio, ineq_vio_s, ineq_vio_i


def calculate_objective(PG, gencost, baseMVA, ngen):
    """Total generation cost [$/h].

    Returns
    -------
    float
    """
    return np.sum([PG[i] * baseMVA * gencost[i, 5] for i in range(ngen)])


# ===========================================================================
# 4. Test Entry Point
# ===========================================================================

def main():
    """Solve the IEEE 30-bus test case and compare with PyPower IPM."""
    path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    data = loadmat(
        path + '/HeatAnalysis/data/ieee_data/casefiles_mat/case_30.mat')
    ppc_mat = data.get('mpc')
    ppc = {
        'version': int(ppc_mat['version'][0, 0]),
        'baseMVA': float(ppc_mat['baseMVA'][0, 0]),
        'bus': ppc_mat['bus'][0, 0],
        'gen': ppc_mat['gen'][0, 0],
        'branch': ppc_mat['branch'][0, 0],
        'gencost': ppc_mat['gencost'][0, 0],
    }

    # PyPower reference solution
    runopf(ppc)

    ppc = ext2int(ppc)
    baseMVA = ppc['baseMVA']
    baseKV = ppc['bus'][0, idx_bus.BASE_KV]
    ppc['baseKV'] = baseKV
    ppc['baseI'] = baseMVA / baseKV

    opf = ACOPF(ppc, qlim=True, angle_cons=True,
                voltage_form='polar', reactive_demand_ratio='auto')
    results = opf.solve()
    print(f'{results["obj"]:.2f}')


if __name__ == '__main__':
    main()

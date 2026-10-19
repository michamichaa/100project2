"""Code given to you for 16.100 Project 2 (wing design). You do not need to edit this file.

    from p2_lib import *

What it gives you:
    load_inputs        reads p2_inputs.toml into an object
    make_design        the two-panel wing: chord, twist and camber at any eta
    naca4_camber       the NACA 4-digit camber line and its slope
    span_sections      alpha_L0 and c_m,c/4 along the span, from your section_properties
    lifting_line       the lifting-line solver (sine series)
    cd                 the section drag coefficient of a NACA m412 section
    ETA, span_integral the span stations and the integral over them
    check_...          self-checks for your functions
    show_panel         the interactive design panel

Units: SI (m, N, Pa). Angles are in radians, except where a name ends in _deg.
A design gives the camber m in percent. naca4_camber and section_properties take m
as a fraction of the chord (0.04 for a NACA 4412).
"""

import math
import time
import tomllib
from dataclasses import dataclass, fields

import numpy as np

TWO_PI = 2 * np.pi
N_TERMS = 50            # number of odd sine terms in the lifting-line series
N_ETA = 400             # number of span stations in ETA

#: Span stations on the half wing: the midpoints of 400 equal strips on 0 <= eta <= 1.
#: There is no station at the root, at the tip or at the break (eta = 0.5).
ETA = (np.arange(N_ETA) + 0.5) / N_ETA


def span_integral(f):
    """Integral of f over 0 <= eta <= 1, for values of f at the stations ETA.

    It uses the midpoint rule: the integral is the mean of the values.
    """
    f = np.asarray(f, dtype=float)
    if f.shape[-1:] != (N_ETA,):
        raise ValueError(f"span_integral needs values at the {N_ETA} stations ETA, got shape {f.shape}")
    return f.mean(axis=-1)


# ---------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class Inputs:
    # [air]
    rho: float
    nu: float
    # [p1] -- your P1 design
    W_N: float
    S_m2: float
    b_m: float
    V_cruise: float
    CLmax_req: float
    # [stall]
    dcl_stall: float
    eta_check: float
    # [sections]
    p_camber: float
    m_min_pct: float
    m_max_pct: float
    # [planform]
    eta_break: float
    chord_ratio_min: float
    # [tail]
    l_t_m: float
    # [drag_fit]
    cd0: float
    km: float
    cd2: float
    k2: float
    Re_ref: float
    a_Re: float
    # [sliders]
    chord_ratio_max: float
    theta_min_deg: float
    theta_max_deg: float
    # [design]
    cb_cr: float
    ct_cr: float
    theta_b_deg: float
    theta_t_deg: float
    m_r_pct: float
    m_b_pct: float
    m_t_pct: float

    @property
    def q_cruise(self) -> float:
        """Dynamic pressure in cruise, Pa."""
        return 0.5 * self.rho * self.V_cruise ** 2

    @property
    def CL_cruise(self) -> float:
        """Wing C_L in cruise, W/(q S)."""
        return self.W_N / (self.q_cruise * self.S_m2)


_STAFF_P1 = dict(W_N=17.2, S_m2=0.41, b_m=1.59, V_cruise=21.4, CLmax_req=1.54)
DESIGN_KEYS = ("cb_cr", "ct_cr", "theta_b_deg", "theta_t_deg", "m_r_pct", "m_b_pct", "m_t_pct")


def load_inputs(path="p2_inputs.toml") -> Inputs:
    """Read p2_inputs.toml and return an Inputs object.

    Each key of the file becomes one field: inp.rho, inp.S_m2, inp.dcl_stall, ...
    A missing or misspelled key is an error.
    """
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    values = {}
    for table in data.values():
        for key, value in table.items():
            values[key] = float(value)
    names = {f.name for f in fields(Inputs)}
    missing = names - values.keys()
    unknown = values.keys() - names
    if missing or unknown:
        raise KeyError(f"{path}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    return Inputs(**values)


def uses_staff_example(inp: Inputs) -> bool:
    """True if the [p1] section still has the staff example values."""
    return all(math.isclose(getattr(inp, k), v) for k, v in _STAFF_P1.items())


# ---------------------------------------------------------------------------- geometry
@dataclass(frozen=True)
class Design:
    """A two-panel wing (handout Section 3.2). Make one with make_design().

    S, b: wing area (m^2) and span (m).
    cb_cr, ct_cr: chord at the break and at the tip, as fractions of the root chord.
    theta_b_deg, theta_t_deg: twist at the break and at the tip, in degrees, relative to
        the root chord, positive nose up. The root has zero twist.
    m_r_pct, m_b_pct, m_t_pct: maximum camber in percent of the chord, at the root, the
        break and the tip.
    eta_b: the break station (0.5). p: chordwise position of maximum camber (0.4).
    """
    S: float
    b: float
    cb_cr: float = 1.0
    ct_cr: float = 1.0
    theta_b_deg: float = 0.0
    theta_t_deg: float = 0.0
    m_r_pct: float = 4.0
    m_b_pct: float = 4.0
    m_t_pct: float = 4.0
    eta_b: float = 0.5
    p: float = 0.4

    @property
    def c_r(self) -> float:
        """Root chord (m). It makes the wing area equal to S."""
        e = self.eta_b
        return 2 * self.S / (self.b * ((1 + self.cb_cr) * e + (self.cb_cr + self.ct_cr) * (1 - e)))

    @property
    def AR(self) -> float:
        """Aspect ratio b^2/S."""
        return self.b ** 2 / self.S

    def chord(self, eta):
        """Chord c (m) at stations eta. The chord changes linearly along each panel."""
        cr = self.c_r
        return np.interp(np.abs(eta), [0.0, self.eta_b, 1.0], [cr, self.cb_cr * cr, self.ct_cr * cr])

    def twist_deg(self, eta):
        """Twist theta (deg) at stations eta. It changes linearly along each panel."""
        return np.interp(np.abs(eta), [0.0, self.eta_b, 1.0], [0.0, self.theta_b_deg, self.theta_t_deg])

    def twist(self, eta):
        """Twist theta (rad) at stations eta."""
        return np.radians(self.twist_deg(eta))

    def _blend(self, eta, v_r, v_b, v_t):
        return np.interp(np.abs(eta), [0.0, self.eta_b, 1.0], [v_r, v_b, v_t])

    def camber(self, eta):
        """Maximum camber m (percent) at stations eta. It changes linearly along each
        panel and is continuous at the break."""
        return self._blend(eta, self.m_r_pct, self.m_b_pct, self.m_t_pct)

    def variables(self) -> dict:
        """The seven design variables, as a dict."""
        return {k: getattr(self, k) for k in DESIGN_KEYS}


def make_design(inp: Inputs, **changes) -> Design:
    """Make a Design with the S and b of inp.

    Without changes, the seven design variables come from the [design] section of
    p2_inputs.toml. Give any of them to change them, for example
        make_design(inp, ct_cr=0.5, theta_t_deg=-2.0)
    """
    unknown = set(changes) - set(DESIGN_KEYS)
    if unknown:
        raise TypeError(f"make_design: unknown design variables {sorted(unknown)}; "
                        f"use {', '.join(DESIGN_KEYS)}")
    values = {k: getattr(inp, k) for k in DESIGN_KEYS}
    values.update(changes)
    return Design(S=inp.S_m2, b=inp.b_m, eta_b=inp.eta_break, p=inp.p_camber, **values)


# ---------------------------------------------------------------------------- sections
def naca4_camber(x, m, p):
    """NACA 4-digit camber line (handout Task 1).

    x: chordwise station x/c, 0 <= x <= 1 (a number or an array).
    m: maximum camber, as a fraction of the chord (0.04 for a NACA 4412).
    p: position of the maximum camber, as a fraction of the chord (0.4).
    Returns (z, dzdx): the camber line z/c and its slope dz/dx at each x.
    """
    x = np.asarray(x, dtype=float)
    if m == 0:
        return np.zeros_like(x), np.zeros_like(x)
    front = x <= p
    z = np.where(front, m / p ** 2 * (2 * p * x - x ** 2),
                 m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x ** 2))
    dzdx = np.where(front, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    return z, dzdx


_SECTION_CACHE = {}


def _section_values(section_properties, m_pct, p):
    key = (section_properties, float(m_pct), float(p))
    if key not in _SECTION_CACHE:
        result = section_properties(m_pct / 100.0, p)
        try:
            a, c = (float(v) for v in result)
        except (TypeError, ValueError):
            raise TypeError("section_properties(m, p) must return two numbers: "
                            f"alpha_L0 (rad) and c_m,c/4. It returned {result!r}.") from None
        if len(_SECTION_CACHE) > 512:
            _SECTION_CACHE.clear()
        _SECTION_CACHE[key] = (a, c)
    return _SECTION_CACHE[key]


def span_sections(design: Design, section_properties):
    """Thin-airfoil properties along the span, from your section_properties(m, p).

    Returns two functions of eta: alpha_L0(eta) in radians, and cm_c4(eta).
    It calls section_properties at the three camber stations (root, break, tip). Thin airfoil
    theory is linear in the camber line. So alpha_L0 and c_m,c/4 change linearly
    along each panel, in the same way as the camber.

        alpha_L0, cm_c4 = span_sections(design, section_properties)
        cl0 = -2 * np.pi * alpha_L0(ETA)
    """
    ms = (design.m_r_pct, design.m_b_pct, design.m_t_pct)
    vals = [_section_values(section_properties, m, design.p) for m in ms]
    a = [v[0] for v in vals]
    c = [v[1] for v in vals]

    def alpha_L0(eta):
        return design._blend(eta, *a)

    def cm_c4(eta):
        return design._blend(eta, *c)

    return alpha_L0, cm_c4


def cd(inp: Inputs, m, cl, cl0, Re):
    """Section drag coefficient c_d of a NACA m412 section (fit to XFOIL; see P2-drag-fit.pdf).

    m: maximum camber in percent (4 for a NACA 4412).
    cl: section lift coefficient.
    cl0: zero-angle lift coefficient of the section, -2 pi alpha_L0, from your
        section_properties.
    Re: Reynolds number based on the local chord, V c / nu.
    All four can be numbers or arrays. The constants come from [drag_fit] in
    p2_inputs.toml. The fit holds for 0 <= m <= 6 % and 1.2e5 <= Re <= 3.8e5.
    """
    delta = np.asarray(cl) - np.asarray(cl0)
    m = np.asarray(m)
    return ((inp.cd0 + inp.km * m + (inp.cd2 + inp.k2 * m) * delta ** 2)
            * (np.asarray(Re) / inp.Re_ref) ** inp.a_Re)


# ---------------------------------------------------------------------------- lifting line
def _as_function(f):
    if callable(f):
        return f
    value = float(f)
    return lambda eta: np.full(np.shape(eta), value)


class LiftingLine:
    """Result of lifting_line(). Attributes:

    alpha: root angle of attack (rad), the angle between the flow and the root chord.
    A, n: the coefficients A_n of handout Eq. (3), and their indices n = 1, 3, 5, ...
    CL, CDi, e: wing lift coefficient, induced drag coefficient, span efficiency.
    S, b, AR: wing area (m^2), span (m), aspect ratio.
    Methods cl(eta), ccl(eta) and alpha_i(eta) give values at any stations eta.
    """

    def __init__(self, b, S, chord, alpha, A, n):
        self.b, self.S, self.AR = b, S, b ** 2 / S
        self.chord = chord
        self.alpha, self.A, self.n = alpha, A, n
        self.CL = float(np.pi * self.AR * A[0])
        sum_nA2 = float(np.sum(n * A ** 2))
        self.CDi = float(np.pi * self.AR * sum_nA2)
        self.e = float(A[0] ** 2 / sum_nA2) if sum_nA2 > 0 else float("nan")

    def ccl(self, eta):
        """c c_l (m) at stations eta: chord times section lift coefficient, 2 Gamma / V."""
        phi = np.arccos(np.clip(np.abs(np.asarray(eta, dtype=float)), 0, 1))
        return 4 * self.b * (np.sin(np.multiply.outer(phi, self.n)) @ self.A)

    def cl(self, eta):
        """Section lift coefficient c_l at stations eta."""
        return self.ccl(eta) / self.chord(eta)

    def alpha_i(self, eta):
        """Induced angle of attack (rad), the downwash angle, at stations eta."""
        phi = np.arccos(np.clip(np.abs(np.asarray(eta, dtype=float)), 0, 1))
        nphi = np.multiply.outer(phi, self.n)
        s = np.sin(phi)[..., None]
        ratio = np.where(s > 1e-12, np.sin(nphi) / np.where(s > 1e-12, s, 1.0), self.n)
        return (ratio * self.n) @ self.A


def lifting_line(b, chord, twist, alpha_L0, alpha=None, CL=None, S=None, n_terms=N_TERMS):
    """Solve the lifting-line equation for a symmetric wing (handout Eq. 3).

    b: span (m).
    chord, twist, alpha_L0: functions of eta that give the chord (m), the twist (rad)
        and the zero-lift angle (rad) at each station. A number means the same value
        at every station.
    alpha: angle of attack of the root chord (rad). Or give CL instead, and the
        solver finds the alpha that gives this wing C_L.
    S: wing area (m^2). If you leave it out, it is found from the chord.
    n_terms: number of odd terms n = 1, 3, ..., 2 n_terms - 1.
    Returns a LiftingLine object with CL, CDi, e, A and the method cl(eta).

    The section lift slope is 2 pi. The collocation points are phi_k = k pi / (2 N),
    k = 1 ... N, with eta = cos(phi). If N is a multiple of 3, one point falls on the
    break (eta = 0.5), so the solver then uses N + 1 terms.
    """
    if (alpha is None) == (CL is None):
        raise ValueError("lifting_line: give alpha or CL (one of them)")
    chord, twist, alpha_L0 = _as_function(chord), _as_function(twist), _as_function(alpha_L0)
    if S is None:
        e = (np.arange(4000) + 0.5) / 4000
        S = b * float(np.mean(chord(e)))
    N = int(n_terms) + (int(n_terms) % 3 == 0)
    phi = np.arange(1, N + 1) * np.pi / (2 * N)
    n = np.arange(1, 2 * N, 2)
    eta = np.cos(phi)
    c = np.asarray(chord(eta), dtype=float)
    sin_nphi = np.sin(np.outer(phi, n))
    M = sin_nphi * (4 * b / (TWO_PI * c))[:, None] + sin_nphi * (n[None, :] / np.sin(phi)[:, None])
    rhs = np.column_stack([np.ones(N), np.asarray(twist(eta), dtype=float) - np.asarray(alpha_L0(eta), dtype=float)])
    sol = np.linalg.solve(M, rhs)
    A_alpha, A_zero = sol[:, 0], sol[:, 1]       # A = A_alpha * alpha + A_zero
    AR = b ** 2 / S
    if alpha is None:
        alpha = (CL - np.pi * AR * A_zero[0]) / (np.pi * AR * A_alpha[0])
    return LiftingLine(b, S, chord, float(alpha), A_alpha * alpha + A_zero, n)


# ---------------------------------------------------------------------------- self-checks
class _Checker:
    def __init__(self, title):
        self.title, self.n, self.bad = title, 0, 0
        print(title)

    def __call__(self, ok, what, hint=""):
        self.n += 1
        if ok:
            print(f"  ok    {what}")
        else:
            self.bad += 1
            print(f"  FAIL  {what}" + (f"\n        {hint}" if hint else ""))
        return ok

    def done(self):
        if self.bad:
            print(f"{self.bad} of {self.n} checks failed. Fix them before you go on.")
        else:
            print(f"All {self.n} checks passed. (The checks find common mistakes. "
                  "They do not prove that your function is correct.)")


def _not_ready(err):
    print(f"Not written yet. {err}")


def _test_design(inp, **kw):
    v = dict(cb_cr=1.0, ct_cr=1.0, theta_b_deg=0.0, theta_t_deg=0.0,
             m_r_pct=4.0, m_b_pct=4.0, m_t_pct=4.0)
    v.update(kw)
    return make_design(inp, **v)


def _camber_all(m):
    return dict(m_r_pct=m, m_b_pct=m, m_t_pct=m)


def check_section_properties(section_properties):
    """Self-check for Task 1: section_properties(m, p)."""
    ck = _Checker("Checks for section_properties(m, p):")
    try:
        r0 = section_properties(0.0, 0.4)
        r2 = section_properties(0.02, 0.4)
        r4 = section_properties(0.04, 0.4)
        r6 = section_properties(0.06, 0.4)
    except NotImplementedError as err:
        return _not_ready(err)
    try:
        (a0, c0), (a2, c2), (a4, c4), (a6, c6) = [tuple(float(v) for v in r) for r in (r0, r2, r4, r6)]
    except (TypeError, ValueError):
        ck(False, "returns two numbers", f"It must return (alpha_L0, cm_c4). It returned {r4!r}.")
        return ck.done()
    ck(abs(a0) < 1e-9 and abs(c0) < 1e-9, "a symmetric section (m = 0) gives alpha_L0 = 0 and c_m,c/4 = 0",
       f"You got alpha_L0 = {a0:.3g}, c_m,c/4 = {c0:.3g}.")
    ck(abs(a4) < 0.2, "alpha_L0 is in radians",
       f"For m = 0.04 you got alpha_L0 = {a4:.3g}. A few degrees is about 0.05 rad. "
       "Is m a fraction (0.04, not 4)? Do you return degrees?")
    ck(a4 < 0, "positive camber gives a negative alpha_L0 (the section lifts at zero angle)",
       "Check the sign in Eq. (1).")
    ck(c4 < 0, "positive camber gives a nose-down (negative) c_m,c/4",
       "Check the sign of c_m,c/4 = (pi/4)(A2 - A1).")
    lin = a4 != 0 and abs(a6 / a4 - 1.5) < 0.005 and abs(a2 / a4 - 0.5) < 0.005 and c4 != 0 and abs(c6 / c4 - 1.5) < 0.005
    ck(lin, "alpha_L0 and c_m,c/4 are proportional to m (thin airfoil theory is linear in the camber)",
       f"Ratios for m = 0.06 and 0.04: alpha_L0 {a6 / a4 if a4 else float('nan'):.4f}, "
       f"c_m {c6 / c4 if c4 else float('nan'):.4f} (both should be 1.5).")
    cl0 = -TWO_PI * a4
    ck(abs(cl0 - 0.46) < 0.01, "a NACA 4412 has c_l0 = 0.46, as the handout says",
       f"You got c_l0 = -2 pi alpha_L0 = {cl0:.3f}. Use enough points in the integral, "
       "and check x/c = (1 - cos(vartheta))/2.")
    ck.done()


def check_cl_max(inp, cl_max):
    """Self-check for Task 1: cl_max(inp, cl0)."""
    ck = _Checker("Checks for cl_max(inp, cl0):")
    try:
        v0, v4 = float(cl_max(inp, 0.0)), float(cl_max(inp, 0.46))
        arr = np.asarray(cl_max(inp, np.array([0.0, 0.46])))
    except NotImplementedError as err:
        return _not_ready(err)
    ck(abs(v0 - 0.99) < 1e-9 and abs(v4 - 1.45) < 1e-9, "a 0012 stalls at c_l = 0.99 and a 4412 at 1.45",
       f"You got {v0:.3f} and {v4:.3f}. Take the constant from inp.dcl_stall.")
    ck(arr.shape == (2,), "works on a numpy array of c_l0 values", f"Got shape {arr.shape}.")
    ck.done()


def check_loadings(inp, loadings):
    """Self-check for Task 2: loadings(design)."""
    ck = _Checker("Checks for loadings(design):")
    try:
        flat = _test_design(inp)
        clb0, dcl0 = (np.asarray(v, dtype=float) for v in loadings(flat))
        wash = _test_design(inp, theta_b_deg=-2.0, theta_t_deg=-4.0)
        clb, dcl = (np.asarray(v, dtype=float) for v in loadings(wash))
    except NotImplementedError as err:
        return _not_ready(err)
    if not ck(clb.shape == dcl.shape == (N_ETA,), "returns two arrays at the stations ETA",
              f"Got shapes {clb.shape} and {dcl.shape}; both must be ({N_ETA},)."):
        return ck.done()
    ck(np.max(np.abs(clb0)) < 1e-6, "no twist and one section: the basic loading c_l,b is zero",
       f"Largest |c_l,b| = {np.max(np.abs(clb0)):.3g}. Is c_l,b the c_l at C_L = 0, not at alpha = 0?")
    c = wash.chord(ETA)
    avg = wash.b / wash.S * span_integral(c * dcl)
    ck(abs(avg - 1) < 1e-3, "the span load of the additional loading, c (dc_l/dC_L)/c_avg, has an average of 1",
       f"Its average is {avg:.4f}. Is dc_l/dC_L the rise of c_l per unit rise of the wing C_L?")
    net = wash.b / wash.S * span_integral(c * clb)
    ck(abs(net) < 1e-4, "the basic loading makes no net lift: the average of c c_l,b / c_avg is zero",
       f"Its average is {net:.3g}.")
    ck(clb[0] > 0 and clb[-1] < 0, "with washout, c_l,b is positive at the root and negative near the tip",
       "Check the sign of the twist and of alpha_L0 in your call to lifting_line.")
    ck.done()


def check_stall_front(inp, stall_front, section_properties):
    """Self-check for Task 2: stall_front(inp, design)."""
    ck = _Checker("Checks for stall_front(inp, design):")
    try:
        r4 = stall_front(inp, _test_design(inp))
        r0 = stall_front(inp, _test_design(inp, **_camber_all(0.0)))
        rt = stall_front(inp, _test_design(inp, cb_cr=0.75, ct_cr=0.5, theta_t_deg=2.0))
    except NotImplementedError as err:
        return _not_ready(err)
    try:
        (s4, C4, e4), (s0, C0, e0), (st, Ct, et) = [(np.asarray(a, dtype=float), float(b), float(c))
                                                    for a, b, c in (r4, r0, rt)]
    except (TypeError, ValueError):
        ck(False, "returns (CL_star, CLmax, eta_first)", f"It returned {type(r4)}.")
        return ck.done()
    if not ck(s4.shape == (N_ETA,), "CL_star is an array at the stations ETA", f"Got shape {s4.shape}."):
        return ck.done()
    chk = ETA <= inp.eta_check
    ck(abs(C4 - np.min(s4[chk])) < 1e-9, "CLmax is the lowest C_L* for eta <= 0.9",
       f"CLmax = {C4:.4f}, lowest C_L* for eta <= 0.9 = {np.min(s4[chk]):.4f}.")
    ck(et <= inp.eta_check and e4 <= inp.eta_check, "the first stall is at eta <= 0.9 (the rule is not applied outboard)",
       f"Got eta_first = {e4:.3f} and {et:.3f} for two test wings.")
    ck(0.9 < C4 < 1.45, "rectangular NACA 4412 wing: 0.9 < CLmax < 1.45",
       f"Got {C4:.3f}. The wing C_L,max must be below the section c_l,max = 1.45.")
    ratio = (inp.dcl_stall - TWO_PI * float(section_properties(0.04, inp.p_camber)[0])) / inp.dcl_stall
    ck(abs(C4 / C0 / ratio - 1) < 1e-6,
       "rectangular, untwisted wings: CLmax scales with c_l,max (NACA 4412 against 0012)",
       f"CLmax(4412)/CLmax(0012) = {C4 / C0:.5f}. With no basic loading it must equal the ratio "
       f"of the section c_l,max values, {ratio:.5f}.")
    ck.done()


def check_profile_drag(inp, profile_drag):
    """Self-check for Task 3: profile_drag(inp, design, CL, V)."""
    ck = _Checker("Checks for profile_drag(inp, design, CL, V):")
    d = _test_design(inp, **_camber_all(0.0))
    try:
        D0 = float(profile_drag(inp, d, 0.0, inp.V_cruise))
        D5 = float(profile_drag(inp, d, 0.5, inp.V_cruise))
    except NotImplementedError as err:
        return _not_ready(err)
    ref = float(cd(inp, 0.0, 0.0, 0.0, inp.V_cruise * d.c_r / inp.nu))
    ck(abs(D0 / ref - 1) < 2e-3, "rectangular NACA 0012 wing at C_L = 0: C_Dp equals the section c_d",
       f"Got C_Dp = {D0:.5f}, section c_d = {ref:.5f}. Check the factor b/S and that you integrate both halves.")
    ck(D5 > D0, "a symmetric wing has more profile drag at C_L = 0.5 than at C_L = 0",
       f"Got {D5:.5f} at C_L = 0.5 and {D0:.5f} at C_L = 0.")
    D_slow = float(profile_drag(inp, d, 0.0, 0.5 * inp.V_cruise))
    ck(D_slow > D0, "the drag coefficient rises when the Reynolds number falls",
       "Use Re = V c / nu with the local chord c.")
    ck.done()


def check_pitching_moment(inp, pitching_moment, section_properties):
    """Self-check for Task 4: pitching_moment(design)."""
    ck = _Checker("Checks for pitching_moment(design):")
    try:
        sym = pitching_moment(_test_design(inp, **_camber_all(0.0)))
        rect = _test_design(inp)
        Cm, cbar = (float(v) for v in pitching_moment(rect))
        cm4 = float(section_properties(0.04, inp.p_camber)[1])
        tap = _test_design(inp, cb_cr=0.75, ct_cr=0.5)
        _, cbar_t = (float(v) for v in pitching_moment(tap))
    except NotImplementedError as err:
        return _not_ready(err)
    ck(abs(float(sym[0])) < 1e-12, "a symmetric wing has C_m,ac = 0", f"Got {float(sym[0]):.3g}.")
    ck(abs(Cm - cm4) < 1e-6, "rectangular wing with one section: C_m,ac equals the section c_m,c/4",
       f"Got C_m,ac = {Cm:.5f}, section c_m,c/4 = {cm4:.5f}.")
    ck(abs(cbar - rect.c_r) < 1e-6, "rectangular wing: the mean aerodynamic chord equals the chord (m)",
       f"Got {cbar:.5f} m, chord {rect.c_r:.5f} m.")
    lam = 0.5
    ref = 2 / 3 * tap.c_r * (1 + lam + lam ** 2) / (1 + lam)
    ck(abs(cbar_t / ref - 1) < 1e-4, "linear taper, taper ratio t = c_t/c_r = 0.5: c-bar = (2/3) c_r (1 + t + t^2)/(1 + t)",
       f"Got {cbar_t:.5f} m, formula {ref:.5f} m.")
    ck.done()


def check_root_moment(root_moment):
    """Self-check for Task 5: root_moment(span_load, b, L)."""
    ck = _Checker("Checks for root_moment(span_load, b, L):")
    b, L = 1.5, 20.0
    try:
        M_ell = float(root_moment(np.sqrt(1 - ETA ** 2), b, L))
        M_uni = float(root_moment(np.ones(N_ETA), b, L))
        M_ell2 = float(root_moment(3.0 * np.sqrt(1 - ETA ** 2), b, 2 * L))
    except NotImplementedError as err:
        return _not_ready(err)
    ck(abs(M_ell / (L * b) - 4 / (3 * np.pi) / 4) < 2e-4, "elliptic loading: M_root/(L b) = 0.106",
       f"Got {M_ell / (L * b):.4f}.")
    ck(abs(M_uni / (L * b) - 0.125) < 1e-6, "uniform loading: M_root/(L b) = 0.125 (center of lift at b/4)",
       f"Got {M_uni / (L * b):.4f}.")
    ck(abs(M_ell2 / M_ell - 2) < 1e-9, "M_root is proportional to L and does not depend on the scale of span_load",
       f"Doubling L and tripling span_load changed M_root by a factor {M_ell2 / M_ell:.4f}, not 2.")
    ck.done()


# ---------------------------------------------------------------------------- interactive panel
_BLUE, _ORANGE, _AQUA, _GREY, _INK, _MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"

_SLIDERS = (  # key, label (HTML), step
    ("cb_cr", "c<sub>b</sub>/c<sub>r</sub>", 0.01),
    ("ct_cr", "c<sub>t</sub>/c<sub>r</sub>", 0.01),
    ("theta_b_deg", "θ<sub>b</sub> (deg)", 0.01),
    ("theta_t_deg", "θ<sub>t</sub> (deg)", 0.01),
    ("m_r_pct", "m<sub>r</sub> (%)", 0.01),
    ("m_b_pct", "m<sub>b</sub> (%)", 0.01),
    ("m_t_pct", "m<sub>t</sub> (%)", 0.01),
)


def _call(fn, *args):
    """(value, None) or (None, message). A missing student function gives its own message."""
    try:
        return fn(*args), None
    except NotImplementedError as err:
        return None, str(err) or f"write {getattr(fn, '__name__', 'this function')}"
    except Exception as err:  # a bug in student code: report it, do not stop the panel
        return None, f"{getattr(fn, '__name__', 'a function')} stopped with {type(err).__name__}: {err}"


class Panel:
    """The interactive design panel. Make it with show_panel().

    After you move a slider:
        panel.design    the Design shown
        panel.results   a dict of the numbers in the readout (only those that could be computed)
    """

    def __init__(self, inp, funcs):
        import ipywidgets as w
        from matplotlib.figure import Figure

        self.inp, self.f = inp, funcs
        self.results, self.messages, self.last_update_s = {}, {}, float("nan")
        self.updates = 0
        lims = {"cb_cr": (inp.chord_ratio_min, inp.chord_ratio_max),
                "ct_cr": (inp.chord_ratio_min, inp.chord_ratio_max),
                "theta_b_deg": (inp.theta_min_deg, inp.theta_max_deg),
                "theta_t_deg": (inp.theta_min_deg, inp.theta_max_deg)}
        self.sliders = {}
        for key, label, step in _SLIDERS:
            lo, hi = lims.get(key, (inp.m_min_pct, inp.m_max_pct))
            s = w.FloatSlider(value=float(np.clip(getattr(inp, key), lo, hi)), min=lo, max=hi, step=step,
                              description=label, description_allow_html=True,
                              continuous_update=False, readout_format=".2f",
                              style={"description_width": "70px"}, layout=w.Layout(width="320px"))
            s.observe(self._on_change, names="value")
            self.sliders[key] = s
        reset = w.Button(description="Reset to p2_inputs.toml", layout=w.Layout(width="320px"))
        reset.on_click(self._reset)
        note = w.HTML("<small>Twist is relative to the root chord, nose up positive. "
                      "Camber m is in % of the chord. Release a slider, or type a value, to update.</small>",
                      layout=w.Layout(width="320px"))
        self.image = w.Image(format="png", layout=w.Layout(width="100%", max_width="1000px"))
        self.readout = w.HTML()
        self.status = w.HTML()
        left = w.VBox([w.HTML("<b>Design variables</b>"), *self.sliders.values(), reset, note, self.status])
        self.widget = w.VBox([w.HBox([left, w.VBox([self.image], layout=w.Layout(flex="1"))]), self.readout])

        self.fig = Figure(figsize=(10.5, 6.4), dpi=80, layout="constrained")
        gs = self.fig.add_gridspec(3, 6, height_ratios=[1, 1, 2.1])
        self.ax_plan = self.fig.add_subplot(gs[0:2, 0:4])
        self.ax_cam = self.fig.add_subplot(gs[0, 4:6])
        self.ax_tw = self.fig.add_subplot(gs[1, 4:6], sharex=self.ax_cam)
        self.ax_cl = self.fig.add_subplot(gs[2, 0:2])
        self.ax_load = self.fig.add_subplot(gs[2, 2:4])
        self.ax_front = self.fig.add_subplot(gs[2, 4:6])
        self.update()

    # -- events
    def _on_change(self, change):
        self.update()

    def _reset(self, _button):
        for key, s in self.sliders.items():
            s.unobserve(self._on_change, names="value")
            s.value = float(np.clip(getattr(self.inp, key), s.min, s.max))
            s.observe(self._on_change, names="value")
        self.update()

    @property
    def design(self) -> Design:
        return make_design(self.inp, **{k: float(s.value) for k, s in self.sliders.items()})

    # -- computation
    def compute(self, d):
        inp, f = self.inp, self.f
        R, msg = {}, {}
        R["c"] = d.chord(ETA)
        R["CL_cruise"] = inp.CL_cruise

        sec, msg["sec"] = _call(span_sections, d, f["section_properties"])
        if sec is not None:
            R["cl0"] = -TWO_PI * sec[0](ETA)
            clmax, msg["clmax"] = _call(f["cl_max"], inp, R["cl0"])
            if clmax is not None:
                R["clmax"] = np.broadcast_to(np.asarray(clmax, dtype=float), ETA.shape)
            cr, err = _call(lifting_line, d.b, d.chord, d.twist, sec[0], None, inp.CL_cruise, d.S)
            msg["cruise"] = err
            if cr is not None:
                R.update(e=cr.e, CDi=cr.CDi)
        else:
            msg["clmax"] = msg["cruise"] = msg["sec"]

        lo, msg["loadings"] = _call(f["loadings"], d)
        if lo is not None:
            try:
                R["clb"], R["dcl"] = (np.broadcast_to(np.asarray(v, dtype=float), ETA.shape) for v in lo)
            except (TypeError, ValueError):
                msg["loadings"] = "loadings(design) must return two arrays at the stations ETA: (cl_b, dcl_dCL)."

        sf, msg["front"] = _call(f["stall_front"], inp, d)
        if sf is not None:
            try:
                CLs, CLmax, eta1 = sf
                R["CLstar"] = np.broadcast_to(np.asarray(CLs, dtype=float), ETA.shape)
                R["CLmax"], R["eta_first"] = float(CLmax), float(eta1)
                chk = ETA <= inp.eta_check
                R["stalled105"] = float(np.mean(R["CLstar"][chk] <= 1.05 * R["CLmax"]))
            except (TypeError, ValueError):
                msg["front"] = "stall_front(inp, design) must return (CL_star, CLmax, eta_first)."

        CDp, msg["CDp"] = _call(f["profile_drag"], inp, d, inp.CL_cruise, inp.V_cruise)
        if CDp is not None:
            R["CDp"] = float(CDp)

        pm, msg["Cm"] = _call(f["pitching_moment"], d)
        if pm is not None:
            try:
                R["Cm_ac"], R["cbar"] = (float(v) for v in pm)
                R["L_tail"] = inp.q_cruise * d.S * R["cbar"] * R["Cm_ac"] / inp.l_t_m
            except (TypeError, ValueError):
                msg["Cm"] = "pitching_moment(design) must return (Cm_ac, cbar)."

        if "clb" in R and "CLmax" in R:
            R["cl_at_max"] = R["clb"] + R["CLmax"] * R["dcl"]
            M, msg["Mroot"] = _call(f["root_moment"], R["c"] * R["cl_at_max"], d.b, inp.W_N)
            if M is not None:
                R["M_root"] = float(M)
        else:
            msg["Mroot"] = msg["loadings"] or msg["front"]
        return R, msg

    # -- drawing
    @staticmethod
    def _missing(ax, text):
        ax.text(0.5, 0.5, "Not shown yet.\n" + _wrap(text), transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color=_MUTED, wrap=True)

    def draw(self, d, R, msg):
        inp = self.inp
        for ax in (self.ax_plan, self.ax_cam, self.ax_tw, self.ax_cl, self.ax_load, self.ax_front):
            ax.cla()
            ax.grid(True, color="#e4e3df", lw=0.6)
            ax.tick_params(labelsize=9, colors=_MUTED)
        eta = np.linspace(0, 1, 201)
        y = eta * d.b / 2
        c = d.chord(eta)

        ax = self.ax_plan                      # (a) planform, flow from the top
        ax.fill_between(y, -0.75 * c, 0.25 * c, color=_BLUE, alpha=0.15, lw=0)
        ax.plot(y, 0.25 * c, color=_BLUE, lw=2)
        ax.plot(y, -0.75 * c, color=_BLUE, lw=2)
        ax.plot([d.b / 2] * 2, [-0.75 * c[-1], 0.25 * c[-1]], color=_BLUE, lw=2)
        ax.plot([0, d.b / 2], [0, 0], color=_MUTED, lw=0.8, ls=":")
        ax.axvline(d.eta_b * d.b / 2, color=_GREY, lw=0.8, ls="--")
        ax.text(d.eta_b * d.b / 2, 0.25 * d.c_r * 1.15, " break", fontsize=9, color=_MUTED, va="bottom")
        ax.set_aspect("equal")
        ax.set_xlim(-0.01, d.b / 2 + 0.02)
        ax.set_ylim(-0.75 * d.c_r * 1.25, 0.25 * d.c_r * 2.0)
        ax.set_xlabel("$y$ (m)", fontsize=10)
        ax.set_ylabel("$x$ (m)", fontsize=10)
        ax.set_title(f"(a) Planform, right half (flow from the top). $c_r$ = {d.c_r:.3f} m, AR = {d.AR:.2f}",
                     fontsize=10, loc="left")

        ax = self.ax_cam                       # (b) camber, and twist below it
        ax.plot(eta, d.camber(eta), color=_ORANGE, lw=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.3, inp.m_max_pct + 0.5)
        ax.set_ylabel("camber $m$ (%)", fontsize=10)
        ax.set_title("(b) Camber and twist", fontsize=10, loc="left")
        ax.tick_params(labelbottom=False)
        tw = self.ax_tw
        tw.plot(eta, d.twist_deg(eta), color=_AQUA, lw=2)
        tw.axhline(0, color=_GREY, lw=0.6)
        tw.set_ylim(min(-1.0, float(np.min(d.twist_deg(eta)))) - 0.5,
                    max(1.0, float(np.max(d.twist_deg(eta)))) + 0.5)
        tw.set_ylabel(r"twist $\theta$ (deg)", fontsize=10)
        tw.set_xlabel(r"$\eta = 2y/b$", fontsize=10)

        ax = self.ax_cl                        # (c) c_l and c_l,max at C_L,max
        ax.axvspan(inp.eta_check, 1, color=_GREY, alpha=0.15, lw=0)
        need = [k for k in ("cl_at_max", "clmax") if k not in R]
        if need:
            self._missing(ax, msg.get("sec") or msg.get("clmax") or msg.get("loadings") or msg.get("front") or "")
        else:
            ax.plot(ETA, R["cl_at_max"], color=_BLUE, lw=2, label="$c_l$")
            ax.plot(ETA, R["clmax"], color=_ORANGE, lw=2, ls="--", label=r"$c_{l\max}$")
            ax.plot([R["eta_first"]], [np.interp(R["eta_first"], ETA, R["clmax"])], "o", color=_INK, ms=6,
                    label="stall starts")
            ax.legend(fontsize=9, loc="lower left")
            ax.set_ylim(0, max(2.0, float(np.max(R["clmax"])) + 0.2))
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"$\eta$", fontsize=10)
        ax.set_ylabel("section $c_l$", fontsize=10)
        ax.set_title(r"(c) At $C_{L\max}$; gray: stall rule not applied", fontsize=10, loc="left")

        ax = self.ax_load                      # (d) span load at C_L,max, and the elliptic load
        if "cl_at_max" in R:
            cavg = d.S / d.b
            ax.plot(ETA, R["c"] * R["cl_at_max"] / cavg, color=_BLUE, lw=2, label="your wing")
            ax.plot(ETA, 4 / np.pi * R["CLmax"] * np.sqrt(1 - ETA ** 2), color=_GREY, lw=1.5, ls="--",
                    label="elliptic, same $C_L$")
            ax.legend(fontsize=9, loc="lower left")
            ax.set_ylim(0, None)
        else:
            self._missing(ax, msg.get("sec") or msg.get("loadings") or msg.get("front") or "")
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"$\eta$", fontsize=10)
        ax.set_ylabel(r"span load $c\,c_l/c_{avg}$", fontsize=10)
        ax.set_title(r"(d) Span load at $C_{L\max}$", fontsize=10, loc="left")

        ax = self.ax_front                     # (e) stall front
        if "CLstar" in R:
            chk = ETA <= inp.eta_check
            ratio = R["CLstar"][chk] / R["CLmax"]
            ax.fill_between(ETA[chk], 0.98, 1.05, where=ratio <= 1.05, color=_ORANGE, alpha=0.25, lw=0,
                            label=r"stalled at 1.05 $C_{L\max}$")
            ax.plot(ETA[chk], ratio, color=_BLUE, lw=2)
            ax.axhline(1.05, color=_INK, lw=0.8, ls="--")
            ax.plot([R["eta_first"]], [1.0], "o", color=_INK, ms=6, label="stall starts")
            ax.set_ylim(0.98, min(1.6, max(1.2, float(np.nanmax(ratio)) + 0.02)))
            ax.legend(fontsize=9, loc="upper left")
        else:
            self._missing(ax, msg.get("front") or "")
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"$\eta$", fontsize=10)
        ax.set_ylabel(r"$C_L^*(\eta)\,/\,C_{L\max}$", fontsize=10)
        ax.set_title(r"(e) Stall front, $\eta \leq$" + f" {inp.eta_check:g}", fontsize=10, loc="left")

        import io
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", facecolor="white")
        self.image.value = buf.getvalue()

    def readout_html(self, R, msg):
        inp = self.inp
        rows = []

        def row(name, value, why=None):
            rows.append(f"<tr><td style='padding:2px 12px 2px 0'>{name}</td>"
                        f"<td style='padding:2px 0'>{value if why is None else '<i>not shown yet: ' + _html(why) + '</i>'}</td></tr>")

        if "CLmax" in R:
            ok = R["CLmax"] >= inp.CLmax_req
            word = "meets" if ok else "<b>below</b>"
            row("Wing C<sub>L,max</sub>", f"{R['CLmax']:.3f} ({word} C<sub>L,max,req</sub> = {inp.CLmax_req:.2f})")
            row("Stall starts at", f"η = {R['eta_first']:.2f}")
            row("Span stalled at 1.05 C<sub>L,max</sub>", f"{100 * R['stalled105']:.0f}% of 0 ≤ η ≤ {inp.eta_check:g}")
        else:
            for name in ("Wing C<sub>L,max</sub>", "Stall starts at", "Span stalled at 1.05 C<sub>L,max</sub>"):
                row(name, None, msg.get("front"))
        cru = f"Cruise, V = {inp.V_cruise:.1f} m/s, C<sub>L</sub> = {inp.CL_cruise:.3f}"
        if "e" in R:
            row(cru, f"e = {R['e']:.3f}, C<sub>Di</sub> = {R['CDi']:.5f}")
        else:
            row(cru, None, msg.get("cruise"))
        if "CDp" in R:
            row("Cruise C<sub>Dp</sub>", f"{R['CDp']:.5f}")
        else:
            row("Cruise C<sub>Dp</sub>", None, msg.get("CDp"))
        if "CDp" in R and "CDi" in R:
            row("Cruise wing C<sub>D</sub> = C<sub>Dp</sub> + C<sub>Di</sub>", f"{R['CDp'] + R['CDi']:.5f}")
        if "Cm_ac" in R:
            row("C<sub>m,ac</sub>; mean aerodynamic chord", f"{R['Cm_ac']:.4f}; c̄ = {R['cbar']:.4f} m")
            Lt = R["L_tail"]
            side = "up" if Lt > 0 else "down"
            row(f"Tail load in cruise (l<sub>t</sub> = {inp.l_t_m:g} m)",
                f"{abs(Lt):.2f} N {side} ({100 * abs(Lt) / inp.W_N:.0f}% of W)")
        else:
            row("C<sub>m,ac</sub> and tail load", None, msg.get("Cm"))
        if "M_root" in R:
            row("Root bending moment, L = W at C<sub>L,max</sub>", f"M<sub>root</sub> = {R['M_root']:.3f} N·m")
        else:
            row("Root bending moment", None, msg.get("Mroot"))
        return "<table style='font-size:13px'>" + "".join(rows) + "</table>"

    def update(self):
        t0 = time.perf_counter()
        d = self.design
        self.status.value = "<small>Updating ...</small>"
        R, msg = self.compute(d)
        self.draw(d, R, msg)
        self.readout.value = self.readout_html(R, msg)
        keep = ("CLmax", "eta_first", "stalled105", "e", "CDi", "CDp", "Cm_ac", "cbar", "L_tail", "M_root")
        self.results = {k: R[k] for k in keep if k in R}
        if "CDp" in R and "CDi" in R:
            self.results["CD_wing"] = R["CDp"] + R["CDi"]
        self.messages = {k: v for k, v in msg.items() if v}
        self.last_update_s = time.perf_counter() - t0
        self.updates += 1
        self.status.value = ""


def _wrap(text, width=40):
    import textwrap
    return "\n".join(textwrap.wrap(str(text), width))


def _html(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def show_panel(inp, section_properties, cl_max, loadings, stall_front, profile_drag, pitching_moment,
               root_moment):
    """Show the interactive design panel and return it.

    Give it your seven functions. A function that you have not written yet gives a
    message in the panel, and the rest of the panel still works.
    """
    from IPython.display import display
    funcs = dict(section_properties=section_properties, cl_max=cl_max, loadings=loadings,
                 stall_front=stall_front, profile_drag=profile_drag, pitching_moment=pitching_moment,
                 root_moment=root_moment)
    panel = Panel(inp, funcs)
    display(panel.widget)
    return panel

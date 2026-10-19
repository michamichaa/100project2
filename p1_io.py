"""Read p1_inputs.toml into a Python object.

    from p1_io import load_inputs
    inp = load_inputs("p1_inputs.toml")
    inp.rho, inp.CD0, inp.S_m2, ...

Needs Python 3.11 or later (tomllib is in the standard library). You should not
need to edit this file: change the numbers in p1_inputs.toml instead.
"""

import math
import tomllib
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Inputs:
    # [air]
    g: float
    rho: float
    nu: float
    # [mission]
    range_m: float
    reserve: float
    payload_kg: float
    b_max_m: float
    V_stall_max: float
    # [placeholders]
    e_oswald: float
    CLmax: float
    CD0: float
    # [power]
    e_star_Whkg: float
    f_usable: float
    eta: float
    P_max_W: float
    # [wing_mass]
    rho_foam: float
    A_fac: float
    tau: float
    # [fuselage_mass]
    m_fuse0_kg: float
    m_fuse_b_kg: float
    m_fuse_S_kg: float
    b0_m: float
    S0_m2: float
    # [design] -- your choices; nan until you fill them in
    S_m2: float
    AR: float
    m_b_kg: float
    V_cruise: float

    @property
    def e_star(self) -> float:
        """Battery specific energy in J/kg (the file gives Wh/kg)."""
        return self.e_star_Whkg * 3600.0


def load_inputs(path="p1_inputs.toml") -> Inputs:
    """Read the TOML file and return an Inputs object.

    The sections of the file ([air], [mission], ...) are only there to keep it
    readable; every key becomes one field of Inputs. A missing or misspelled key
    is an error, so a typo cannot silently fall back to a default.
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


def require_design(inp: Inputs):
    """Raise a clear error if the [design] section still has a nan."""
    empty = [k for k in ("S_m2", "AR", "m_b_kg", "V_cruise") if math.isnan(getattr(inp, k))]
    if empty:
        raise ValueError(f"Fill in {', '.join(empty)} in the [design] section of p1_inputs.toml, "
                         "save it, and run this cell again.")

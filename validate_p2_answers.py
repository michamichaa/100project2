"""Check the FORMAT of a 16.100 Project 2 answer file before you submit it.

    python validate_p2_answers.py p2_answers.toml

Needs Python 3.11 or later and nothing else. It checks that every entry is
present, spelled as in the template, and filled in with a number (not nan).
It also checks the design limits of the handout: chords at least 0.5 c_r and
camber from 0 to 6%. It does not check whether your numbers are right.
"""

import math
import sys
import tomllib

FORMAT = "16.100-P2-v1"

# table -> {key: kind}; kind is "num" (non-zero number), "num0" (number, zero allowed),
# "int" (positive integer), "names" (list of names)
SPEC = {
    "team": {"group": "int", "members": "names"},
    "design": {"cb_cr": "num", "ct_cr": "num", "theta_b_deg": "num0", "theta_t_deg": "num0",
               "m_r_pct": "num0", "m_b_pct": "num0", "m_t_pct": "num0"},
    "results": {"CLmax": "num", "eta_first_stall": "num0", "e_cruise": "num", "CDp_cruise": "num",
                "CDi_cruise": "num", "Cm_ac": "num0", "M_root_Nm": "num"},
}

# (table, key, low, high, message): handout limits and plain range checks
LIMITS = [
    ("design", "cb_cr", 0.5, math.inf, "must be at least 0.5 (handout: chords at least 0.5 c_r)"),
    ("design", "ct_cr", 0.5, math.inf, "must be at least 0.5 (handout: chords at least 0.5 c_r)"),
    *[("design", f"m_{s}_pct", 0.0, 6.0, "must be from 0 to 6 (percent of the chord)") for s in "rbt"],
    ("results", "eta_first_stall", 0.0, 0.9, "must be from 0 to 0.9 (the stall rule applies for eta <= 0.9)"),
    ("results", "e_cruise", 0.0, 1.0, "must be from 0 to 1"),
    ("results", "M_root_Nm", 0.0, math.inf, "must be positive (N m)"),
]


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_value(where, v, kind, problems):
    if kind in ("num", "num0"):
        if not _is_number(v):
            problems.append(f"{where}: must be a plain number (no units or quotes), got {v!r}")
        elif math.isnan(v):
            problems.append(f"{where}: still nan; fill it in")
        elif kind == "num" and v == 0:
            problems.append(f"{where}: is 0; fill it in")
    elif kind == "int":
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            problems.append(f"{where}: must be your group number (a positive whole number), got {v!r}")
    elif kind == "names":
        if (not isinstance(v, list) or not v
                or not all(isinstance(x, str) and x.strip() and x != "First Last" for x in v)):
            problems.append(f"{where}: list every member's name, e.g. [\"Ada Lovelace\", \"Alan Turing\"]")


def validate(data):
    problems = []
    if data.get("format") != FORMAT:
        problems.append(f"format: must be \"{FORMAT}\"; do not change this line")

    known = {"format", *SPEC}
    for extra in sorted(set(data) - known):
        problems.append(f"[{extra}]: not in the template (renamed or misplaced?)")

    for table, keys in SPEC.items():
        tbl = data.get(table)
        if not isinstance(tbl, dict):
            problems.append(f"[{table}]: section missing")
            continue
        for extra in sorted(set(tbl) - set(keys)):
            problems.append(f"{table}.{extra}: not in the template (renamed or misplaced?)")
        for key, kind in keys.items():
            if key not in tbl:
                problems.append(f"{table}.{key}: missing")
                continue
            _check_value(f"{table}.{key}", tbl[key], kind, problems)

    for table, key, low, high, message in LIMITS:
        v = data.get(table, {}).get(key) if isinstance(data.get(table), dict) else None
        if _is_number(v) and not math.isnan(v) and not (low <= v <= high):
            problems.append(f"{table}.{key} = {v}: {message}")
    return problems


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip())
        return 2
    path = argv[1]
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        print(f"{path}: file not found")
        return 2
    except tomllib.TOMLDecodeError as err:
        print(f"{path}: not valid TOML: {err}")
        print("Common causes: a missing quote or comma, or units after a number.")
        return 1
    problems = validate(data)
    if problems:
        print(f"{path}: {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"{path}: format OK. (This does not check whether your numbers are correct.)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

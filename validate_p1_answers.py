"""Check the FORMAT of a 16.100 Project 1 answer file before you submit it.

    python validate_p1_answers.py p1_answers.toml

Needs Python 3.11 or later and nothing else. It checks that every entry is
present, spelled as in the template, and filled in with a number (not nan), or text
where text is asked for. It does not check whether your numbers are right.
"""

import math
import sys
import tomllib

FORMAT = "16.100-P1-v1"

TASK2_ROW = ["area_m2", "aspect_ratio", "battery_mass_kg", "span_m", "chord_m", "wing_mass_kg",
             "fuselage_mass_kg", "total_mass_kg", "wing_loading_N_per_m2", "V_stall_m_per_s"]

# table -> {key: kind}; kind is "num" (non-zero number), "num0" (number, zero allowed),
# "text" (non-empty string), "int" (positive integer), "names" (list of names), "bool"
SPEC = {
    "team": {"group": "int", "members": "names"},
    "task2_min_stall": {"battery_mass_kg": "num", "area_m2": "num", "aspect_ratio": "num",
                        "V_stall_m_per_s": "num"},
    "design": {k: "num" for k in [
        "area_m2", "aspect_ratio", "battery_mass_kg", "cruise_speed_m_per_s", "span_m", "chord_m",
        "wing_mass_kg", "fuselage_mass_kg", "total_mass_kg", "wing_loading_N_per_m2", "V_stall_m_per_s",
        "CL_cruise", "LD_cruise", "battery_power_W", "range_km", "flight_time_min", "Reynolds_cruise"]},
    "task5": {"CLmax_required": "num"},
}


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
    elif kind == "text":
        if not isinstance(v, str) or not v.strip():
            problems.append(f"{where}: must be filled in (text in quotes)")
    elif kind == "names":
        if (not isinstance(v, list) or not v
                or not all(isinstance(x, str) and x.strip() and x != "First Last" for x in v)):
            problems.append(f"{where}: list every member's name, e.g. [\"Ada Lovelace\", \"Alan Turing\"]")
    elif kind == "bool":
        if not isinstance(v, bool):
            problems.append(f"{where}: must be true or false (no quotes), got {v!r}")


def validate(data):
    problems = []
    if data.get("format") != FORMAT:
        problems.append(f"format: must be \"{FORMAT}\"; do not change this line")

    known = {"format", "task2", *SPEC}
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

    t2 = data.get("task2")
    if not isinstance(t2, dict) or not isinstance(t2.get("rows"), list):
        problems.append("[task2]: section missing or rows = [ ... ] not found")
    else:
        rows = t2["rows"]
        for i, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                problems.append(f"task2 row {i}: each row must be {{ ... }} on one line")
                continue
            for extra in sorted(set(row) - set(TASK2_ROW)):
                problems.append(f"task2 row {i}: unknown entry {extra!r}")
            for key in TASK2_ROW:
                if key not in row:
                    problems.append(f"task2 row {i}: {key} missing")
                else:
                    _check_value(f"task2 row {i}: {key}", row[key], "num", problems)
        for key, what in (("area_m2", "wing areas"), ("aspect_ratio", "aspect ratios"),
                          ("battery_mass_kg", "battery masses")):
            values = {r.get(key) for r in rows if isinstance(r, dict) and _is_number(r.get(key)) and r.get(key)
                      and not math.isnan(r.get(key))}
            if len(values) < 3:
                problems.append(f"task2: need at least three different {what}, found {len(values)}")
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
        print("Common causes: a missing quote or comma, units after a number, or a row split over two lines.")
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

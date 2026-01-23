def rule_risk_label(f: dict) -> dict:
    """
    Deterministic, fast rules (industry-style).
    Returns label + score + triggered rules.
    """
    score = 0
    triggered = []

    # End close to goal
    if f["dT"] <= 0.25:
        triggered.append("close_to_goal")
    else:
        score += 2
        triggered.append("far_from_goal_end")

    # Must show progress
    if f["progress"] < 0.1:
        score += 2
        triggered.append("low_progress")
    else:
        triggered.append("has_progress")

    # Rotational stability (use your known range ~1.99 rad/s as reference)
    if f["omega_max"] > 1.99:
        score += 2
        triggered.append("omega_too_high")

    if f["domega_max"] > 1.5:
        score += 1
        triggered.append("high_omega_jumps")

    # Bad heading when still far
    if f["dT"] > 0.5 and f["heading_err_abs"] > 1.0:
        score += 1
        triggered.append("bad_heading_far_from_goal")

    # Score -> label
    if score <= 1:
        label = "SAFE"
    elif score <= 4:
        label = "UNCERTAIN"
    else:
        label = "UNSAFE"

    return {"label": label, "score": score, "rules": triggered}

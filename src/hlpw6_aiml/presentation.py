"""Human-readable labels only; metric IDs and numerical definitions are unchanged."""

LABELS = {
    "overall_score": "Overall score",
    "field_score": "Field score",
    "force_score": "Force score",
    "diagnostic_score": "Profile score",
    "cd_r2": "Drag coefficient — R²",
    "cl_r2": "Lift coefficient — R²",
    "cp_cut_r2": "Cp cut — R²",
    "velocity_profile_r2": "Velocity profile — R²",
    "c_drag_mae": "Drag coefficient — MAE",
    "c_lift_mae": "Lift coefficient — MAE",
    "c_pitch_mae": "Pitching moment coefficient — MAE",
}


def metric_label(key: str) -> str:
    if key in LABELS:
        return LABELS[key]
    for suffix, label in (
        ("_equal_entity_rel_l2", "relative L2 error (equal entities)"),
        ("_rel_l2", "relative L2 error"),
        ("_rel_l1", "relative L1 error"),
        ("_rmse", "RMSE"),
        ("_mae", "MAE"),
    ):
        if key.endswith(suffix):
            quantity = key.removesuffix(suffix).replace("_", " ").capitalize()
            return f"{quantity} — {label}"
    return key.replace("_", " ").capitalize()

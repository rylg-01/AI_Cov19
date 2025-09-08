# Survival Plot Code and Weibull Hazard Ratio

# Imported Libraries:
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import WeibullFitter

# Grabbing current timestamp for output filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# File input and output locations
csv_with_imaging = "/home/rlang/Documents/SantaClaraUniversity/AI_Covid_19/Summer/data_xgboostresults/20250908_145526_XGBoostOutput_withImaging.csv"
csv_boost_only   = "/home/rlang/Documents/SantaClaraUniversity/AI_Covid_19/Summer/data_xgboostresults/20250908_145526_XGBoostOutput_boostOnly.csv"
plot_save_folder = "/home/rlang/Documents/SantaClaraUniversity/AI_Covid_19/Summer/outputs"
os.makedirs(plot_save_folder, exist_ok=True)

# Plot style settings
# Setting plot colors
color_map = {
    "High-risk": "#ff4545", # red
    "Low-risk":  "#1f77b4", # blue
}
# Setting graphing parameters
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "lines.linewidth": 2.8,
})

# estimating Survial Curve
def compute_curve_with_ci(group, max_day=100, n_bootstrap=500, alpha=0.05, random_state=42):
    rng = np.random.default_rng(random_state)
    days = np.arange(0, max_day + 1)
    n = len(group)
    if n == 0:
        zeros = np.zeros_like(days, dtype=float)
        return days, zeros, zeros, zeros

    actual = group["ActualStay"].to_numpy()
    curves = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = actual[idx]
        curves.append([(sample > d).mean() for d in days])

    curves = np.array(curves)
    mean_curve = curves.mean(axis=0) # Average estimates
    lower_bound = np.percentile(curves, 100 * (alpha / 2), axis=0)
    upper_bound = np.percentile(curves, 100 * (1 - alpha / 2), axis=0)
    return days, mean_curve, lower_bound, upper_bound # returen output

# Create prognosis groups
# Initially grouped as poor and good
# Regrouped as High/ Low risk in final grapgh
def add_prognosis_median_split(df):
    df = df.copy()
    ranks = df["PredictedStay"].rank(method="first")
    original = pd.qcut(ranks, q=2, labels=["Poor", "Good"])
    swapped = original.map({"Poor": "Good", "Good": "Poor"})
    df["PrognosisGroup"] = swapped.astype("category")
    df["PrognosisGroup"] = df["PrognosisGroup"].cat.set_categories(["Good", "Poor"], ordered=True)

    # Remapping/ renaming groups
    df["RiskGroup"] = df["PrognosisGroup"].map({
        "Poor": "High-risk",
        "Good": "Low-risk"
    }).astype("category")
    df["RiskGroup"] = df["RiskGroup"].cat.set_categories(["Low-risk", "High-risk"], ordered=True)
    return df

# Kaplan Plot
def plot_and_save_kaplan_2groups(df, label, tag, timestamp, max_day=100):
    df = add_prognosis_median_split(df)
    order = ["High-risk", "Low-risk"]

    # Resizing figure
    plt.figure(figsize=(7, 5))

    for group_name in order:
        group_df = df[df["RiskGroup"] == group_name]
        days, mean_curve, low_curve, high_curve = compute_curve_with_ci(group_df, max_day=max_day)

        plt.step(
            days, mean_curve, where="post",
            label=f"{group_name} (n={len(group_df)})",
            color=color_map[group_name]
        )
        plt.fill_between(
            days, low_curve, high_curve,
            step="post", alpha=0.25, color=color_map[group_name]
        )

    plt.legend(loc="best", frameon=False)
    plt.grid(True, alpha=0.35)
    plt.ylim(0, 1.05)
    plt.xlim(0, max_day)
    plt.xticks(range(0, max_day + 1, 10))
    plt.xlabel("Length of Hospital Stay")
    plt.ylabel("Probability of Stay")
    plt.title(label)
    plt.tight_layout()

    # Saving plot to folder with timestamp
    out_path = os.path.join(plot_save_folder, f"{timestamp}_{tag}_2groups_kaplan.png")
    plt.savefig(out_path, dpi=600, format="png")
    plt.close()

    return out_path

# Fitting Weibull to find Hazard Ratio
def _fit_weibull_safely(durations):
    durations = np.asarray(durations, dtype=float)
    # Replace nonpositive values
    eps = 1e-6
    durations = np.where(durations <= 0, eps, durations)
    wf = WeibullFitter().fit(durations, event_observed=np.ones_like(durations))
    return wf

# Computing hazard ratio for low and high risk groups using parametric weibull models
def compute_hazard_ratio_weibull(df, model_label, n_bootstrap=500, alpha=0.05, random_state=42):
    df = add_prognosis_median_split(df).copy()

    # Fit Weibull to both groups
    wf_high = _fit_weibull_safely(df.loc[df["RiskGroup"] == "High-risk", "ActualStay"])
    wf_low  = _fit_weibull_safely(df.loc[df["RiskGroup"] == "Low-risk",  "ActualStay"])

    rho_high, lam_high = wf_high.rho_, wf_high.lambda_
    rho_low,  lam_low  = wf_low.rho_,  wf_low.lambda_

    # Point estimate
    if np.isclose(rho_high, rho_low, rtol=0.05):
        rho = (rho_high + rho_low) / 2
        hr_point = (lam_high / lam_low) ** (-rho)
        method = "Weibull (approx constant HR)"
    else:
        median_t = np.median(df["ActualStay"])
        h_high = wf_high.hazard_at_times(median_t).values[0]
        h_low  = wf_low.hazard_at_times(median_t).values[0]
        hr_point = h_high / h_low
        method = f"Weibull (HR at t={median_t:.1f})"

    # Bootstrap CI
    rng = np.random.default_rng(random_state)
    hr_samples = []
    for _ in range(n_bootstrap):
        boot_df = df.sample(frac=1, replace=True, random_state=int(rng.integers(1_000_000_000)))
        try:
            wf_high_b = _fit_weibull_safely(boot_df.loc[boot_df["RiskGroup"] == "High-risk", "ActualStay"])
            wf_low_b  = _fit_weibull_safely(boot_df.loc[boot_df["RiskGroup"] == "Low-risk",  "ActualStay"])
            if np.isclose(wf_high_b.rho_, wf_low_b.rho_, rtol=0.05):
                rho_b = (wf_high_b.rho_ + wf_low_b.rho_) / 2
                hr_b = (wf_high_b.lambda_ / wf_low_b.lambda_) ** (-rho_b)
            else:
                median_t_b = np.median(boot_df["ActualStay"])
                h_high_b = wf_high_b.hazard_at_times(median_t_b).values[0]
                h_low_b  = wf_low_b.hazard_at_times(median_t_b).values[0]
                hr_b = h_high_b / h_low_b
            if np.isfinite(hr_b):
                hr_samples.append(hr_b)
        except Exception:
            # Skip failed resamples
            continue

    ci_lower = np.percentile(hr_samples, 100 * (alpha / 2)) if len(hr_samples) > 0 else np.nan
    ci_upper = np.percentile(hr_samples, 100 * (1 - alpha / 2)) if len(hr_samples) > 0 else np.nan

    return pd.DataFrame([{
        "Model": model_label,
        "Comparison": "High-risk vs Low-risk",
        "HR": hr_point,
        "CI Lower": ci_lower,
        "CI Upper": ci_upper,
        "p-value": None,
        "Method": method
    }])

# Summary table
def los_summary_table_2groups(df, model_label):
    df = add_prognosis_median_split(df)
    rows = []
    for group_name in ["High-risk", "Low-risk"]:
        group_df = df[df["RiskGroup"] == group_name]
        if len(group_df) > 0:
            q1 = group_df["ActualStay"].quantile(0.25)
            q3 = group_df["ActualStay"].quantile(0.75)
            rows.append({
                "Model": model_label,
                "Group": group_name,
                "N": len(group_df),
                "MinStay": group_df["ActualStay"].min(),
                "MaxStay": group_df["ActualStay"].max(),
                "MedianStay": group_df["ActualStay"].median(),
                "IQR_Lower": q1,
                "IQR_Upper": q3
            })
    return pd.DataFrame(rows)


plot_paths = []
all_los_summaries = []
all_hr_rows = []

for label, filepath, tag in [
    ("With Imaging", csv_with_imaging, "withImaging"),
    ("Boost Only",   csv_boost_only,   "boostOnly"),
]:
    df = pd.read_csv(filepath)

    # Save survival curve
    kaplan_path = plot_and_save_kaplan_2groups(df, label, tag, timestamp, max_day=100)
    plot_paths.append(kaplan_path)

    # Summary
    los_df = los_summary_table_2groups(df, model_label=label)
    all_los_summaries.append(los_df)

    # Weibull hazard ratio
    hr_df = compute_hazard_ratio_weibull(df, model_label=label, n_bootstrap=500, alpha=0.05, random_state=42)
    all_hr_rows.append(hr_df)

# Save summary for survival plot in csv
final_los_df = pd.concat(all_los_summaries, ignore_index=True)
los_output_path = os.path.join(plot_save_folder, f"{timestamp}_los_summary_2groups_highlowrisk.csv")
final_los_df.to_csv(los_output_path, index=False)

# Save hazard ratio csv
final_hr_df = pd.concat(all_hr_rows, ignore_index=True)
hr_output_path = os.path.join(plot_save_folder, f"{timestamp}_hazard_ratios_weibull_highlowrisk.csv")
final_hr_df.to_csv(hr_output_path, index=False)

# Printing Save locations
print("Saved plots:")
for p in plot_paths:
    print("  -", p)
print("LOS summary CSV:", los_output_path)
print("Weibull HR CSV:", hr_output_path)
print("Done.")

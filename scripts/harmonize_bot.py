import pandas as pd
import numpy as np

print("Loading BoT Sampled Dataset...")

df = pd.read_csv(
    "Phase3_sampling/BoT_sampled.csv",
    low_memory=False
)

harm = pd.DataFrame()

# =====================================================
# COMMON FEATURES
# =====================================================

harm["duration"] = np.nan

harm["rate"] = np.nan

harm["mean_value"] = df["HH_L0.1_mean"]

harm["variance"] = df["HH_L0.1_std"]

harm["weight"] = df["HH_L0.1_weight"]

harm["covariance"] = df["HH_L0.1_covariance"]

harm["min_value"] = np.nan

harm["max_value"] = np.nan

# =====================================================
# UNSW FEATURES
# =====================================================

harm["proto"] = np.nan
harm["state"] = np.nan
harm["service"] = np.nan

harm["tcprtt"] = np.nan
harm["synack"] = np.nan
harm["ackdat"] = np.nan

harm["ct_flw_http_mthd"] = np.nan
harm["is_ftp_login"] = np.nan
harm["ct_ftp_cmd"] = np.nan

# =====================================================
# BOT FEATURES
# =====================================================

harm["HH_L0.1_magnitude"] = df["HH_L0.1_magnitude"]

harm["HH_L0.1_radius"] = df["HH_L0.1_radius"]

harm["HH_L0.1_pcc"] = df["HH_L0.1_pcc"]

harm["HpHp_L0.1_magnitude"] = df["HpHp_L0.1_magnitude"]

harm["HpHp_L0.1_radius"] = df["HpHp_L0.1_radius"]

harm["HpHp_L0.1_pcc"] = df["HpHp_L0.1_pcc"]

# =====================================================
# CIC FEATURES
# =====================================================

harm["HTTP"] = np.nan
harm["HTTPS"] = np.nan
harm["DNS"] = np.nan
harm["SMTP"] = np.nan
harm["SSH"] = np.nan
harm["TCP"] = np.nan
harm["UDP"] = np.nan
harm["ICMP"] = np.nan

# =====================================================
# 5G FEATURES
# =====================================================

harm["Load"] = np.nan
harm["Loss"] = np.nan
harm["TcpRtt"] = np.nan

harm["SrcLoad"] = np.nan
harm["DstLoad"] = np.nan

harm["SrcLoss"] = np.nan
harm["DstLoss"] = np.nan

# =====================================================
# METADATA
# =====================================================

harm["domain"] = df["domain"]

harm["binary_label"] = df["binary_label"]

harm["attack_category"] = df["attack_category"]

harm.to_csv(
    "Phase4_harmonized/BoT_harmonized.csv",
    index=False
)

print("Done")
print("Rows:", len(harm))
print("Columns:", len(harm.columns))

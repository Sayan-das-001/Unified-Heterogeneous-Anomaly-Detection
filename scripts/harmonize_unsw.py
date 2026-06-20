import pandas as pd
import numpy as np

print("Loading UNSW Sampled Dataset...")

df = pd.read_csv(
    "Phase3_sampling/UNSW_sampled.csv",
    low_memory=False
)

harm = pd.DataFrame()

# =====================================================
# COMMON FEATURES
# =====================================================

harm["duration"] = df["dur"]

harm["rate"] = (
    pd.to_numeric(df["Sload"], errors="coerce").fillna(0)
    +
    pd.to_numeric(df["Dload"], errors="coerce").fillna(0)
)

harm["mean_value"] = (
    pd.to_numeric(df["smeansz"], errors="coerce").fillna(0)
    +
    pd.to_numeric(df["dmeansz"], errors="coerce").fillna(0)
) / 2

harm["variance"] = np.nan
harm["weight"] = np.nan
harm["covariance"] = np.nan

harm["min_value"] = np.nan
harm["max_value"] = np.nan

# =====================================================
# UNSW DOMAIN FEATURES
# =====================================================

harm["proto"] = df["proto"]

harm["state"] = df["state"]

harm["service"] = df["service"]

harm["tcprtt"] = df["tcprtt"]

harm["synack"] = df["synack"]

harm["ackdat"] = df["ackdat"]

harm["ct_flw_http_mthd"] = df["ct_flw_http_mthd"]

harm["is_ftp_login"] = df["is_ftp_login"]

harm["ct_ftp_cmd"] = df["ct_ftp_cmd"]

# =====================================================
# BOT FEATURES
# =====================================================

harm["HH_L0.1_magnitude"] = np.nan
harm["HH_L0.1_radius"] = np.nan
harm["HH_L0.1_pcc"] = np.nan

harm["HpHp_L0.1_magnitude"] = np.nan
harm["HpHp_L0.1_radius"] = np.nan
harm["HpHp_L0.1_pcc"] = np.nan

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

print("Saving...")

harm.to_csv(
    "Phase4_harmonized/UNSW_harmonized.csv",
    index=False
)

print("Done")
print("Rows:", len(harm))
print("Columns:", len(harm.columns))

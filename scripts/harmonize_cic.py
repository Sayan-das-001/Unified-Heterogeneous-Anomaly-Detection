import pandas as pd
import numpy as np

print("Loading CIC Sampled Dataset...")

df = pd.read_csv(
    "Phase3_sampling/CIC_sampled.csv",
    low_memory=False
)

harm = pd.DataFrame()

# =====================================================
# COMMON FEATURES
# =====================================================

harm["duration"] = df["flow_duration"]

harm["rate"] = df["Rate"]

harm["mean_value"] = df["AVG"]

harm["variance"] = df["Variance"]

harm["weight"] = df["Weight"]

harm["covariance"] = df["Covariance"]

harm["min_value"] = df["Min"]

harm["max_value"] = df["Max"]

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

harm["HH_L0.1_magnitude"] = np.nan
harm["HH_L0.1_radius"] = np.nan
harm["HH_L0.1_pcc"] = np.nan

harm["HpHp_L0.1_magnitude"] = np.nan
harm["HpHp_L0.1_radius"] = np.nan
harm["HpHp_L0.1_pcc"] = np.nan

# =====================================================
# CIC FEATURES
# =====================================================

harm["HTTP"] = df["HTTP"]

harm["HTTPS"] = df["HTTPS"]

harm["DNS"] = df["DNS"]

harm["SMTP"] = df["SMTP"]

harm["SSH"] = df["SSH"]

harm["TCP"] = df["TCP"]

harm["UDP"] = df["UDP"]

harm["ICMP"] = df["ICMP"]

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
    "Phase4_harmonized/CIC_harmonized.csv",
    index=False
)

print("Done")
print("Rows:", len(harm))
print("Columns:", len(harm.columns))

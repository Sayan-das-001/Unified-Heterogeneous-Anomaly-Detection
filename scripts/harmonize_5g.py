import pandas as pd
import numpy as np

print("Loading 5G Sampled Dataset...")

df = pd.read_csv(
    "Phase3_sampling/5G_sampled.csv",
    low_memory=False
)

harm = pd.DataFrame()

# =====================================================
# COMMON FEATURES
# =====================================================

harm["duration"] = df["Dur"]

harm["rate"] = df["Rate"]

harm["mean_value"] = (
    pd.to_numeric(df["sMeanPktSz"], errors="coerce").fillna(0)
    +
    pd.to_numeric(df["dMeanPktSz"], errors="coerce").fillna(0)
) / 2

harm["variance"] = np.nan

harm["weight"] = np.nan

harm["covariance"] = np.nan

harm["min_value"] = df["Min"]

harm["max_value"] = df["Max"]

# =====================================================
# UNSW FEATURES
# =====================================================

harm["proto"] = df["Proto"]

harm["state"] = df["State"]

harm["service"] = np.nan

harm["tcprtt"] = df["TcpRtt"]

harm["synack"] = df["SynAck"]

harm["ackdat"] = df["AckDat"]

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

harm["Load"] = df["Load"]

harm["Loss"] = df["Loss"]

harm["TcpRtt"] = df["TcpRtt"]

harm["SrcLoad"] = df["SrcLoad"]

harm["DstLoad"] = df["DstLoad"]

harm["SrcLoss"] = df["SrcLoss"]

harm["DstLoss"] = df["DstLoss"]

# =====================================================
# METADATA
# =====================================================

harm["domain"] = df["domain"]

harm["binary_label"] = df["binary_label"]

harm["attack_category"] = df["attack_category"]

harm.to_csv(
    "Phase4_harmonized/5G_harmonized.csv",
    index=False
)

print("Done")
print("Rows:", len(harm))
print("Columns:", len(harm.columns))

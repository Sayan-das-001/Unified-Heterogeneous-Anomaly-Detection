# FEATURE MAPPING V1

## COMMON FEATURES

| Unified Feature | UNSW        | BoT                | CIC           | 5G   |
| --------------- | ----------- | ------------------ | ------------- | ---- |
| duration        | dur         | NULL               | flow_duration | Dur  |
| rate            | Sload/Dload | NULL               | Rate          | Rate |
| mean_value      | NULL        | HH_L0.1_mean       | AVG           | Mean |
| variance        | NULL        | HH_L0.1_variance   | Variance      | NULL |
| weight          | NULL        | HH_L0.1_weight     | Weight        | NULL |
| covariance      | NULL        | HH_L0.1_covariance | Covariance    | NULL |
| min_value       | NULL        | NULL               | Min           | Min  |
| max_value       | NULL        | NULL               | Max           | Max  |

---

## UNSW DOMAIN FEATURES

proto
state
service
tcprtt
synack
ackdat
ct_flw_http_mthd
is_ftp_login
ct_ftp_cmd

---

## BoT DOMAIN FEATURES

HH_L0.1_magnitude
HH_L0.1_radius
HH_L0.1_pcc
HpHp_L0.1_magnitude
HpHp_L0.1_radius
HpHp_L0.1_pcc

---

## CIC DOMAIN FEATURES

HTTP
HTTPS
DNS
SMTP
SSH
TCP
UDP
ICMP

---

## 5G DOMAIN FEATURES

Load
Loss
TcpRtt
SrcLoad
DstLoad
SrcLoss
DstLoss

---

## METADATA

domain
binary_label
attack_category

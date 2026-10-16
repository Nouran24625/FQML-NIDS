"""Known column schemas. Used by cleaning and by the synthetic generator."""

# CICIoT2023 released CSVs: 46 numeric features + 'label'. No IPs, MACs, ports or timestamps,
# which is why natural (host) partitions are impossible -> Dirichlet only.
CICIOT2023_FEATURES = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count",
    "fin_count", "urg_count", "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH",
    "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG",
    "Std", "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]
CICIOT2023_LABEL = "label"

# Identifier / shortcut columns dropped from features in every dataset (kept only as partition keys).
SHORTCUT_COLUMNS = {
    "Flow ID", "Src IP", "Dst IP", "Source IP", "Destination IP", "Src Port", "Dst Port",
    "Source Port", "Destination Port", "Timestamp", "ip.src_host", "ip.dst_host", "frame.time",
    "srcip", "dstip", "sport", "dsport", "stime", "ltime", "id",
}
# Features that are legitimate columns but suspected capture artefacts. NOT dropped by default;
# the leakage check in P1 decides. CICIoT2023 'IAT' is the known suspect.
SUSPECT_ARTEFACTS = {"ciciot2023": ["IAT"]}

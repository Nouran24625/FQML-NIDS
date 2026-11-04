"""Live detection for the ITIDA prototype video (Version B). Owner: Nouran.

Pipeline (works on any lab network: EVE-NG, Docker, or a recorded pcap):

    tcpdump ... -U -w -  |  python -m fqml.live.detect --bundle models/lab_demo   ->  runs/live/
                                                                                         |
    python -m fqml.live.dashboard --state runs/live   (browser: http://localhost:8050)   <-

  pcap.py       pure-Python pcap reader (file or live stream on stdin), no extra libraries
  features.py   per-host, per-time-window traffic features (the model's raw inputs)
  attack_log.py wraps an attack command and logs when it ran -> ground-truth labels
  labels.py     windows + attack log -> labelled training table
  train.py      fit preprocessing on train only, train VQC / matched MLP / RF, honest report, bundle
  model.py      load a bundle and classify feature rows
  detect.py     live loop: packets -> windows -> predictions -> runs/live/*.jsonl
  dashboard.py  standard-library web server for the dashboard page
  synth.py      SYNTHETIC test traffic for tests and dry runs only (never for the video)
"""

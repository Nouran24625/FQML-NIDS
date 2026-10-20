# Quantum hardware accounts (no secrets in this file)

API keys and login tokens live ONLY in a password manager, environment variables, or Kaggle / Colab
Secrets. Never in code, notebooks, MLflow, chat or git. If a key is ever pasted anywhere, delete it
on the provider's site and create a new one.

| | IBM Quantum Platform | Quantum Inspire (QuTech) |
|---|---|---|
| Account owner | Kareem Mohamed | Kareem Mohamed (member ID 3446) |
| Plan / instance | Open plan, instance `open-instance`, region us-east, IBM Cloud account "Egypt Japan University (E-JUST)" | Free account |
| Allowance | **10 min 0 s QPU time remaining (100%)** in the rolling 28-day window | Free, no usage cap announced (queue rules apply) |
| Backends visible | ibm_fez, ibm_kingston, ibm_marrakesh (all Heron) | Tuna-17, Tuna-9, Tuna-5, QX emulator, Ry emulator |
| Checked on | 20 Oct 2026 | 20 Oct 2026 |
| Verified how | `QiskitRuntimeService(channel="ibm_quantum_platform").backends()` | `qi login` + `python -m spikes.tuna17_target` |
| Key / token stored | Owner's password manager | Inside the container only (`qi login`; lost on container restart) |

## Watch-outs
- **IBM Cloud trial:** the platform shows "trial account expires in 29 days" (about 18 Nov 2026). Check
  what expiry means for the Open plan (banner "Learn more") before then; E6 hardware runs are planned
  for March 2027.
- **Ask Prof. Samir** whether E-JUST has institutional IBM access or credits (the plan assumes provided
  credits). 10 min per 28 days is enough for E6 inference pilots but not for repeated large runs.
- **Quantum Inspire SDK** (`quantuminspire`, `qiskit-quantuminspire`) is NOT in the lock file: it
  downgrades Qiskit (2.6 -> 2.3). Install it only temporarily to read backend data, then recreate the
  container (`docker compose -f docker/compose.yml up -d --force-recreate dev`).

## How to check again
- IBM: quantum.cloud.ibm.com -> Instances -> remaining minutes; backends with the command above.
- Quantum Inspire: install the SDK in the container, `qi login`, `python -m spikes.tuna17_target`.

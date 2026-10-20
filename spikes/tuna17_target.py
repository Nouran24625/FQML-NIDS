"""Save Quantum Inspire Tuna-17's wiring and native gates for Spike S4 (owner: Sheghaf). No jobs are run.

Needs (inside the container, not frozen into the lock file):
    pip install quantuminspire qiskit-quantuminspire
    qi login          # once per container start; the token stays in the container, never in the repo
Run:
    docker compose -f docker/compose.yml exec dev python -m spikes.tuna17_target
Writes results/spikes/tuna17_target.json = {"backend", "coupling_map", "basis_gates", "checked_on"}
which spikes/s4_hw_transpile.py reads with --tuna-json.
"""
import datetime
import json
from pathlib import Path

from qiskit_quantuminspire.qi_provider import QIProvider

SKIP = {"measure", "reset", "delay", "barrier", "init", "id", "i", "asm"}   # asm = raw cQASM block, not a gate


def main():
    provider = QIProvider()
    names = [b.name for b in provider.backends()]
    print("backends:", names)
    name = next((n for n in names if "tuna-17" in n.lower().replace(" ", "-")), None)
    if name is None:
        raise SystemExit("Tuna-17 is not in your backend list - check your Quantum Inspire account")
    backend = provider.get_backend(name)
    target = backend.target
    edges = sorted({tuple(e) for e in target.build_coupling_map().get_edges()})
    gates = sorted(g for g in target.operation_names if g.lower() not in SKIP)
    print(f"{name}: {target.num_qubits} qubits, {len(edges)} directed couplings")
    print("coupling map:", edges)
    print("native gates:", gates)
    out = Path("results/spikes/tuna17_target.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"backend": name, "num_qubits": target.num_qubits,
                               "coupling_map": [list(e) for e in edges], "basis_gates": gates,
                               "checked_on": datetime.date.today().isoformat()}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()

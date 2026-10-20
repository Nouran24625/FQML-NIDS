"""SPIKE S4 - Does the VQC fit real quantum chips? (owner: Sheghaf). No credits are spent.

Question : after transpiling (rewriting the circuit for a chip's wiring and native gates), how deep is
           the 8-qubit VQC, how many 2-qubit gates does it need, and roughly how likely is
           a run to be error-free?
Compares : entangler "ring" (today's model) vs "chain" (no CNOT 7->0), Qiskit optimization levels 0-3,
           on IBM fake backends (calibration snapshots shipped with qiskit-ibm-runtime) and, if its
           wiring is supplied, Quantum Inspire's Tuna-17.
Why ring vs chain: IBM heavy-hex chips have no 8-qubit loop (the shortest loop has 12 qubits), so the
           ring's closing CNOT forces SWAPs (each = 3 extra 2-qubit gates). S4 counts the extra
           2-qubit gates versus the logical circuit.
Estimate : success ~ product over gates of (1 - gate error) from the backend calibration; a rough
           upper bound that ignores decoherence while idle.
Output   : table, results/spikes/S4_hw_transpile.json, MLflow run in 'phase0-spikes'.

Run: docker compose -f docker/compose.yml exec dev python -m spikes.s4_hw_transpile
     add  --tuna-json results/spikes/tuna17_target.json  once the Tuna-17 wiring is known:
     {"coupling_map": [[0, 1], [1, 2], ...], "basis_gates": ["cz", "rx", "ry", "rz"]}
"""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
from qiskit import transpile

from fqml.quantum.qiskit_circuit import bind, build_vqc

FAKE_BACKENDS = ["FakeFez", "FakeMarrakesh", "FakeTorino", "FakeKingston", "FakeBrisbane", "FakeSherbrooke"]


def load_fake_backends():
    import qiskit_ibm_runtime.fake_provider as fp
    found = []
    for name in FAKE_BACKENDS:
        cls = getattr(fp, name, None)
        if cls is not None:
            found.append(cls())
    return found


def success_estimate(circ, backend) -> float | None:
    target = getattr(backend, "target", None)
    if target is None:
        return None
    logp = 0.0
    for inst in circ.data:
        name = inst.operation.name
        if name in ("barrier", "delay"):
            continue
        qargs = tuple(circ.find_bit(q).index for q in inst.qubits)
        try:
            props = target[name][qargs]
        except KeyError:
            continue
        if props is not None and props.error is not None:
            logp += np.log1p(-min(props.error, 0.999999))
    return float(np.exp(logp))


def metrics(circ):
    ops = circ.count_ops()
    two_q = sum(1 for inst in circ.data if inst.operation.num_qubits == 2)
    return {"depth": circ.depth(), "two_qubit_gates": two_q,
            "total_gates": int(sum(v for k, v in ops.items() if k not in ("barrier", "measure")))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3, help="transpiler seeds per setting (best kept)")
    ap.add_argument("--tuna-json", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    targets = [(b.name, dict(backend=b), b) for b in load_fake_backends()]
    if a.tuna_json:
        t = json.loads(Path(a.tuna_json).read_text())
        targets.append(("tuna-17", dict(coupling_map=t["coupling_map"], basis_gates=t["basis_gates"]), None))
    if not targets:
        raise SystemExit("no fake backends found - is qiskit-ibm-runtime installed?")
    print(f"{'backend':16s} {'entangler':9s} {'opt':>3s} {'depth':>6s} {'2q gates':>9s} {'extra 2q':>9s} {'est. success':>13s}")
    for ent in ("ring", "chain"):
        qc, xp, tp = build_vqc(8, a.layers, ent)
        bound = bind(qc, xp, tp, rng.random(8) * np.pi, rng.random(a.layers * 16) * 2 * np.pi)
        bound.measure_all()
        logical = metrics(bound)
        rows.append({"backend": "logical (no chip)", "entangler": ent, "opt_level": None, **logical,
                     "est_success": None})
        print(f"{'logical':16s} {ent:9s} {'-':>3s} {logical['depth']:6d} {logical['two_qubit_gates']:9d} "
              f"{'-':>9s} {'-':>13s}")
        for name, kw, backend in targets:
            for level in range(4):
                best = None
                for s in range(a.seeds):
                    try:
                        c = transpile(bound, optimization_level=level, seed_transpiler=s, **kw)
                    except Exception as e:
                        best = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
                        break
                    m = metrics(c)
                    if best is None or m["two_qubit_gates"] < best["two_qubit_gates"]:
                        best = {**m, "extra_two_qubit_vs_logical": m["two_qubit_gates"] - logical["two_qubit_gates"],
                                "est_success": success_estimate(c, backend) if backend else None}
                row = {"backend": name, "entangler": ent, "opt_level": level, **best}
                rows.append(row)
                if "error" in best:
                    print(f"{name:16s} {ent:9s} {level:3d}  FAILED {best['error'][:60]}")
                else:
                    es = f"{best['est_success']:.3f}" if best["est_success"] is not None else "n/a"
                    print(f"{name:16s} {ent:9s} {level:3d} {best['depth']:6d} {best['two_qubit_gates']:9d} "
                          f"{best['extra_two_qubit_vs_logical']:9d} {es:>13s}")

    out = Path("results/spikes/S4_hw_transpile.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"machine": platform.platform(), "layers": a.layers, "rows": rows}, indent=2))
    print(f"\nsaved {out}")
    try:
        import mlflow
        mlflow.set_experiment("phase0-spikes")
        with mlflow.start_run(run_name=f"S4-{platform.node()}"):
            mlflow.log_artifact(str(out))
        print("logged to MLflow experiment 'phase0-spikes'")
    except Exception as e:
        print(f"(MLflow logging skipped: {e})")


if __name__ == "__main__":
    main()

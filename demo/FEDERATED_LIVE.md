# Live federation demo: server and sites (Kareem)

One server program and N site programs, each in its own window. Every site trains the shared detector on its own
slice of real CICIoT2023 traffic and sends its update over the team's post-quantum channel (ML-KEM-768 + X25519,
ML-DSA-65 signatures, AES-256-GCM, `fqml/pq`). The server checks every signature, combines the updates and prints
how well the shared model detects each attack family, round by round. Code: `demo/federated_live.py`.

## Start it (Windows, from the repository root, Docker Desktop running)

```powershell
powershell -ExecutionPolicy Bypass -File demo\federated_live.ps1
```

One-word presets (`-Scenario`), in the order to show them:

| Scenario | What happens |
|---|---|
| `honest` | 3 honest sites, FedAvg: detection per family climbs |
| `attack` | site 2 is honest until round 12, then hides Web: the Web panel falls and warns, macro-F1 barely moves |
| `defence` | the same attack against median aggregation |
| `defence-cost` | median with **no** attacker on whole-family sites: the defence itself can cost detection |
| `quantum` | the 64-parameter VQC as the shared model (slower per round) |

Opens the **dashboard in the browser (http://localhost:8050)**, a server window and 3 site windows (defaults:
3000-parameter MLP, sites split, FedAvg, 30 rounds, one second between rounds). The dashboard shows the server and
its sites with messages in flight, the headline numbers, macro-F1 per round, one panel per attack family (a warning
appears when a family falls and stays down), each site's data and update distance, the event log and a table view.
Options:

| Option | Values | Shows |
|---|---|---|
| `-Sites 5` | 2 to 10 | more sites |
| `-Attacker 2` | site number (0 = nobody) | site 2 turns malicious; the server is not told |
| `-AttackFrom 12` | round number (default 10) | the attacker is honest until that round, so the fall is visible |
| `-RoundDelay 1` | seconds | pause between rounds so the audience can follow |
| `-Attack class_suppression -Target Web` | or `label_flip`; any family | which attack |
| `-Aggregator median` | `fedavg`, `median`, `trimmed_mean` | robust aggregation (FED-6) |
| `-Split sitesfam` | `iid`, `dir0.5`, `sites`, `sitesfam` | how attack types are spread over sites (FED-3) |
| `-Model vqc` | `mlp_p3000`, `mlp_p512`, `mlp`, `vqc`, `vqc_lin`, `vqc_noent` | quantum or classical model |
| `-Config classical` | `default`, `classical`, `none` | post-quantum, today's crypto, or none |

Sites on other laptops (same network): on the server laptop run the server with the default `--host 0.0.0.0`; on each
other laptop run `python -m demo.federated_live site --name site-02 --host <server IP>` inside the project image,
after copying `keys/default/server.pub` and that site's own `.key`/`.pub` to it. Stop: close the windows, then
`docker rm -f fqml-fed-demo` (closing a window does not stop its program inside the container; the launcher
recreates the container every time, so a new run always starts clean). The dashboard port is 8050 and the sites
connect on 9443; both are published by the launcher.

## What to show, in order (about 5 minutes)

1. **Three honest sites** (`-Attacker 0`). Point at:
   - each site's window: "my data: N rows, these families". No site has all attack types.
   - the server: "identity verified, handshake ~20 ms"; every round "signature verified"; bytes per round.
   - the round table: recall per family climbs; "unseen-type family" is how often the shared model names the
     family of attack types a site never had. That is what a site gains by joining.
2. **One malicious site** (`-Attacker 2 -AttackFrom 12`). Site 2's window says "MALICIOUS ... the server is not
   told" and, at round 12, "switching to poisoned labels now". On the dashboard the Web panel climbs for 11 rounds,
   then falls and shows a warning, while DDoS and macro-F1 barely move, and site 2's update distance looks like
   everyone else's: the server cannot pick it out. Measured on 24 Nov: Web 54-81% in rounds 3-11, 8% at round 12,
   about 10% after; macro-F1 stays near 0.25 (recorded with `Number` as an input; the demo excludes it since 24 Nov).
3. **A defence** (`-Scenario defence`: the same attacker against median aggregation), then the cost of that defence
   with no attacker at all (`-Scenario defence-cost`). What the experiments found (FED-2 grid and FED-6 control,
   protocol A fold 0): with similar sites (IID) median and trimmed mean cost nothing and stop the attack (Web 0.24
   with FedAvg, 0.53 with median, 0.48 with trimmed mean); with sites that each hold different attack types they do
   not stop it (Web 0.00-0.03 with all three), and with whole-family sites median scores lower than FedAvg in 3 of 4 runs
   even with no attacker. Which family it loses varies from run to run, so say "can cost detection", not "always
   drops family X".

Measured on 24 Nov 2026 (3 sites, sites split, 3000-parameter MLP, FedAvg, 30 rounds; mean of the last 5 rounds;
with `Number` as an input, before it was excluded):

| | All honest | Site 2 hides Web |
|---|---|---|
| Web recall | 0.64 | 0.12 |
| Benign recall | 0.39 | 0.75 (Web now called Benign) |
| Macro-F1 | 0.279 | 0.216 |

## What to say about its limits

- Demo shortcuts: every site reads the shared training sample and keeps only its own rows; sites tell the server
  which attack types they hold so it can score unseen types (a real site would not); features are the stand-in
  top-8 until DATA-6; scores are on the validation block, one fold, one seed.
- The numbers are a demonstration, not results. The measured results come from the experiment scripts
  (`experiments/fed4_trial.py`, `fed6_zero_attacker.py`, `fed5_trees.py`) over folds and seeds.
- The training code is the same reference step the experiments use (`fqml.fl.batched._local_sequential`), and the
  channel is the one E7 will measure (`fqml.pq.link`).

## Keeping it current

Every new model, aggregator, attack or split added to `fqml/` or `experiments/` is wired into
`demo/federated_live.py` (`make_model`, `make_split`, `label_attack`, `fqml.fl.robust.get_aggregator`) and the
table above, and run once before it is shown.

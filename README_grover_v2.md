# Grover's Search Pipeline v2 · Error Mitigation · IQM Garnet

Version 2 extends the original Grover pipeline with **toggleable error mitigation techniques**.
Run baseline and mitigated experiments in the same flow, then compare results across all
techniques via per-technique SVG artifacts and a rich HTML report.

---

## What's New in v2

| Feature | v1 | v2 |
|---------|----|----|
| Baseline QPU run | ✅ | ✅ |
| Dynamical Decoupling (DD) | ❌ | ✅ toggleable |
| Readout Error Mitigation (REM) | ❌ | ✅ toggleable |
| Zero Noise Extrapolation (ZNE) | ❌ | ✅ toggleable |
| Combined run (DD+REM+ZNE) | ❌ | ✅ toggleable |
| Per-technique histograms | ❌ | ✅ |
| Success probability comparison chart | ❌ | ✅ |
| ZNE extrapolation curve | ❌ | ✅ |
| REM readout error heatmap | ❌ | ✅ |
| Scaling analysis per technique | ❌ | ✅ |
| Rich HTML report | ❌ | ✅ |
| Scaling curve per technique | ❌ | ✅ |

---

## The Experiment

A random secret key is hidden in a search space of **2ⁿ items** (n = `num_search_qubits`).
Grover's algorithm finds it quadratically faster than classical brute-force.
On real NISQ hardware, circuit noise reduces the success probability below the theoretical ideal.
Error mitigation techniques are applied to recover lost probability.

---

## Error Mitigation Techniques

### Dynamical Decoupling (DD)
Inserts X·X pulse sequences on idle qubits during the barrier gaps between the Oracle and
Diffuser layers. Counteracts environmental decoherence by refocusing qubit phases.
Implemented via Qiskit's `PadDynamicalDecoupling` transpiler pass with `ALAPScheduleAnalysis`.

### Readout Error Mitigation (REM)
Characterises per-qubit readout errors by running calibration circuits that prepare |0⟩ and |1⟩.
Builds a 2×2 assignment matrix per qubit: A[y,x] = P(measure y | prepared x).
Applies the pseudo-inverse to the Grover measurement distribution, then recomputes the
success probability P(secret key) from the corrected counts.

### Zero Noise Extrapolation (ZNE)
Amplifies circuit noise using **gate folding**: replaces each gate U with U·U†·U (scale 3),
U·U†·U·U†·U (scale 5), etc. Executes the Grover circuit at each scale factor and records
P(secret key). Richardson extrapolation (polynomial or linear fit) recovers the zero-noise
limit of the success probability.

### Combined (DD + REM + ZNE)
Applies all enabled techniques in sequence:
1. DD is applied at circuit-build time
2. The DD circuit is executed at each ZNE scale factor
3. REM correction is applied post-measurement at each scale
4. Richardson extrapolation is performed on the corrected probabilities

Requires 2 or more techniques enabled. Toggle with `enable_combined=True`.

---

## Pipeline Stages

| Stage | Name | Infrastructure | Notes |
|-------|------|---------------|-------|
| 1 | Problem Setup | CPU | Random secret key, Grover parameters |
| 2 | Build Grover Circuit | CPU | H + (Oracle + Diffuser) × k |
| 3 | Transpile for IQM Garnet | CPU | Converts to r, cz, id native gates |
| 4a | Baseline Execution | QPU | Raw circuit, no mitigation |
| 4b | DD Execution | QPU | XX pulses on idle qubits |
| 4c | REM Execution | QPU ×3 | 2 calibration runs + 1 Grover run |
| 4d | ZNE Execution | QPU × \|scales\| | One run per scale factor |
| 4e | Combined Execution | QPU × \|scales\| + cal | DD + scales + REM + extrapolation |
| 5 | Classical Comparison | CPU | Brute-force baseline |
| 6 | Scaling Analysis | QPU × sizes × techniques | Per-technique at each qubit count |
| 7.1 | Success Prob Chart | CPU | SVG: horizontal bar chart |
| 7.2 | Per-Technique Histograms | CPU | SVG: measurement distributions |
| 7.3 | ZNE Extrapolation Curve | CPU | SVG: P(target) vs noise scale |
| 7.4 | REM Readout Error Map | CPU | SVG: per-qubit assignment matrices |
| 7.5 | Quantum vs Classical Chart | CPU | SVG: query count comparison |
| 7.6 | Scaling Curves | CPU | SVG: one line per technique |
| 7.7 | Scaling Results Table | CPU | Markdown: P(target) matrix |
| 7.8 | HTML Report | CPU | Self-contained rich HTML |

---

## Artifacts Produced

| Key | Type | Content |
|-----|------|---------|
| `grover-v2-success-prob-chart` | SVG | P(target) per technique, improvement vs baseline, theoretical line |
| `grover-v2-technique-histograms` | SVG | Measurement distribution per technique, secret key in gold |
| `grover-v2-zne-curve` | SVG | P(target) at each scale factor + extrapolation to λ=0 |
| `grover-v2-rem-readout-map` | SVG | Per-qubit 2×2 assignment matrices, error rates colour-coded |
| `grover-v2-quantum-vs-classical` | SVG | Query count bar chart: worst / avg / Grover |
| `grover-v2-scaling-curves` | SVG | One polyline per technique across qubit sizes |
| `grover-v2-scaling-table` | Markdown | P(target) grid: technique × qubit size |
| `grover-v2-html-report` | HTML | Full self-contained report with inline charts, tables, and context |

---

## Run Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_search_qubits` | int | 3 | Qubits for main Grover run (search space = 2ⁿ) |
| `shots` | int | 4096 | QPU shots per execution |
| `enable_dd` | bool | True | Toggle Dynamical Decoupling |
| `enable_rem` | bool | True | Toggle Readout Error Mitigation |
| `enable_zne` | bool | True | Toggle Zero Noise Extrapolation |
| `enable_combined` | bool | True | Toggle combined run (needs 2+ techniques) |
| `zne_scale_factors` | list[int] | [1, 3, 5] | Noise amplification levels for ZNE |
| `run_scaling` | bool | True | Toggle scaling analysis across qubit sizes |
| `scaling_sizes` | list[int] | [3, 4, 5] | Qubit sizes for scaling sweep |

---

## Setup

### Prerequisites
- Python 3.10+
- Prefect Cloud account (free tier works)
- IQM Resonance account with API key

### Install dependencies
```bash
pip install qiskit>=2.0 qiskit-iqm iqm-client numpy prefect
```

### Configure Prefect
```bash
prefect cloud login
prefect config set PREFECT_API_URL="https://api.prefect.cloud/api/accounts/.../workspaces/..."
```

### Store IQM token as a Prefect Secret
```python
from prefect.blocks.system import Secret
Secret(value="your-iqm-resonance-api-key").save("iqm-resonance-token")
```

### Push code to GitHub
```bash
git init
git remote add origin https://github.com/PlayfulDevBit/Groover_piepline_v2.git
git add grover_pipeline_v2.py
git push -u origin main
```

### Deploy
```bash
python deploy_grover_v2.py
```

---

## Running the Pipeline

### From Prefect Cloud UI
1. Go to **Deployments** → `grover-v2-error-mitigation`
2. Click **Run** → **Custom Run**
3. Toggle techniques in the parameter panel:
   - Set `enable_dd`, `enable_rem`, `enable_zne` to `true` or `false`
   - Set `enable_combined` to `true` to run the combined mode
   - Adjust `zne_scale_factors`, `num_search_qubits`, `shots` as needed
4. Click **Submit**
5. Monitor in **Flow Runs** → check **Artifacts** tab for charts and report

### Local (requires IQM token in env or Prefect Secret)
```bash
# Full run (all techniques)
python grover_pipeline_v2.py

# Baseline only
python grover_pipeline_v2.py --no-dd --no-rem --no-zne

# DD + REM only, larger problem
python grover_pipeline_v2.py --no-zne --qubits 4

# ZNE with more scale factors, no scaling analysis
python grover_pipeline_v2.py --no-dd --no-rem --zne-scales 1,3,5,7 --no-scaling

# All techniques, more shots
python grover_pipeline_v2.py --shots 8192 --qubits 3
```

### CLI flags
```
--qubits N          Number of search qubits (default: 3)
--shots N           Shots per execution (default: 4096)
--no-dd             Disable Dynamical Decoupling
--no-rem            Disable Readout Error Mitigation
--no-zne            Disable Zero Noise Extrapolation
--no-combined       Disable combined run
--zne-scales        ZNE scale factors, comma-separated (default: 1,3,5)
--no-scaling        Disable scaling analysis
--scaling-sizes     Qubit sizes for scaling, comma-separated (default: 3,4,5)
```

---

## Interpreting Results

### Success Probability P(target)
The primary metric is **P(target)** = probability of measuring the secret key.
- Theoretical maximum: `sin²((2k+1)·arcsin(1/√N))` where k = optimal Grover iterations
- Baseline: raw hardware P(target), degraded by noise
- Mitigated: corrected P(target); higher is better
- Combined: best possible with current hardware using all techniques

### What good results look like
- DD improvement: typically 1–5% on circuits with idle periods
- REM improvement: typically 2–10%, most consistent technique
- ZNE improvement: variable; works best when noise scales predictably
- Combined: usually matches or exceeds the best individual technique

### When mitigation may not help
- Very shallow circuits (low noise → less room to improve)
- ZNE extrapolation can overshoot if noise doesn't scale smoothly
- DD requires timing information from the backend (may fall back gracefully)

---

## QPU Cost Estimate

| Configuration | Approx. QPU Calls |
|---------------|------------------|
| Baseline only | 1 |
| + DD | 2 |
| + REM | 2 + 2 calibration = 4 |
| + ZNE (3 scales) | 2 + 3 = 5 |
| All + Combined (3 scales) | 1+1+4+3+2 cal+3 = ~14 |
| + Scaling 3 sizes, all techniques | +3 × (1+1+3+3+2 cal+3) ≈ +39 |

> **Note:** Each QPU call on IQM Resonance consumes resonance units. Plan accordingly.

---

## Related Pipelines

| Pipeline | Description |
|----------|-------------|
| `grover_pipeline.py` (v1) | Original Grover search — baseline only |
| `ghz_mitigation_pipeline.py` | Error mitigation benchmark on GHZ state preparation |
| `grover_pipeline_v2.py` | This pipeline — Grover + DD/REM/ZNE |

---

## Cryptographic Context

Grover's algorithm provides a **quadratic speedup** for unstructured key search:

| Key Length | Classical | Quantum (Grover) | Effective Security |
|------------|-----------|------------------|--------------------|
| AES-128 | 2¹²⁸ | 2⁶⁴ | 64-bit — at risk |
| AES-256 | 2²⁵⁶ | 2¹²⁸ | 128-bit — secure |

**Current limitation:** A 3-qubit demo searches 8 items. Breaking AES-128 requires ~128 logical
qubits running 2⁶⁴ iterations — orders of magnitude beyond current NISQ hardware.
Error mitigation improves today's fidelity but does not change this fundamental scale gap.

---

*Orchestrated by Prefect · Real QPU execution on IQM Garnet (20-qubit superconducting) ·
Error mitigation implemented with Qiskit 2.x*

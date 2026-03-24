"""
Grover's Search Pipeline v2 · IQM Garnet
==========================================
Extends v1 with configurable error mitigation techniques:
  DD  — Dynamical Decoupling (idle qubit noise suppression)
  REM — Readout Error Mitigation (measurement error correction)
  ZNE — Zero Noise Extrapolation (gate error extrapolation to zero)

Toggle techniques at run start. Runs baseline + each enabled technique
individually + combined mode. Produces per-technique artifacts and a
rich HTML comparison report.

Local:
    python grover_pipeline_v2.py
    python grover_pipeline_v2.py --qubits 4 --no-zne --no-scaling
    python grover_pipeline_v2.py --no-dd --no-rem --no-zne  # baseline only

Serverless:
    Triggered from Prefect Cloud UI after running deploy_grover_v2.py
"""

import sys
import time
import math
import random
import argparse
import numpy as np
from datetime import datetime, timezone

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def get_iqm_token() -> str:
    try:
        from prefect.blocks.system import Secret
        return Secret.load("iqm-resonance-token").get()
    except Exception:
        return ""


def get_iqm_backend(token: str):
    from iqm.qiskit_iqm import IQMProvider
    return IQMProvider(
        "https://cocos.resonance.meetiqm.com/garnet", token=token
    ).get_backend()


def optimal_grover_iterations(n_qubits: int) -> int:
    """⌊π/4 · √N⌋ where N = 2^n"""
    N = 2 ** n_qubits
    return max(1, int(math.pi / 4 * math.sqrt(N)))


def _grover_success_prob(counts: dict, secret_key: str, shots: int) -> float:
    """P(measuring secret key) from raw or corrected counts."""
    total = sum(counts.values()) or shots
    return round(counts.get(secret_key, 0) / total, 4)


def _fold_circuit(qc_base, scale: int):
    """Gate folding: replace each gate U with U·U†·U per fold level."""
    if scale == 1:
        return qc_base
    qc_folded = qc_base.copy()
    qc_folded.remove_final_measurements()
    folds = (scale - 1) // 2
    original_ops = list(qc_base.data)
    for _ in range(folds):
        for instruction in original_ops:
            if instruction.operation.name in ('measure', 'barrier'):
                continue
            try:
                inv = instruction.operation.inverse()
                qc_folded.append(inv, instruction.qubits, instruction.clbits)
                qc_folded.append(instruction.operation, instruction.qubits, instruction.clbits)
            except Exception:
                pass
    qc_folded.measure_all()
    return qc_folded


def _calibrate_rem(backend, n: int, shots: int):
    """Run |0⟩^n and |1⟩^n calibration circuits. Return (matrices, inv_matrices)."""
    from qiskit import QuantumCircuit, transpile

    cal_0 = QuantumCircuit(n)
    cal_0.measure_all()
    cal_0_t = transpile(cal_0, backend=backend)

    cal_1 = QuantumCircuit(n)
    for i in range(n):
        cal_1.x(i)
    cal_1.measure_all()
    cal_1_t = transpile(cal_1, backend=backend)

    counts_0 = backend.run(cal_0_t, shots=shots, use_timeslot=False).result().get_counts()
    counts_1 = backend.run(cal_1_t, shots=shots, use_timeslot=False).result().get_counts()

    qubit_matrices = []
    for q in range(n):
        p00 = p10 = p01 = p11 = 0
        for bs, c in counts_0.items():
            bit = int(list(reversed(bs))[q])
            if bit == 0:
                p00 += c
            else:
                p10 += c
        for bs, c in counts_1.items():
            bit = int(list(reversed(bs))[q])
            if bit == 0:
                p01 += c
            else:
                p11 += c
        qubit_matrices.append(np.array([
            [p00 / shots, p01 / shots],
            [p10 / shots, p11 / shots],
        ]))

    inv_matrices = []
    for m in qubit_matrices:
        try:
            inv_matrices.append(np.linalg.inv(m))
        except np.linalg.LinAlgError:
            inv_matrices.append(np.eye(2))

    return qubit_matrices, inv_matrices


def _apply_rem_correction(raw_counts: dict, inv_matrices: list, n: int, shots: int) -> dict:
    """Apply per-qubit inverse assignment matrices to correct measurement distribution."""
    all_bitstrings = [format(i, f'0{n}b') for i in range(2 ** n)]
    probs = np.array([raw_counts.get(bs, 0) / shots for bs in all_bitstrings])
    for q in range(n):
        new_probs = np.zeros_like(probs)
        inv_m = inv_matrices[q]
        for idx, bs in enumerate(all_bitstrings):
            bit = int(list(reversed(bs))[q])
            for target_bit in [0, 1]:
                bs_list = list(reversed(bs))
                bs_list[q] = str(target_bit)
                target_bs = ''.join(reversed(bs_list))
                target_idx = all_bitstrings.index(target_bs)
                new_probs[target_idx] += inv_m[target_bit, bit] * probs[idx]
        probs = new_probs
    probs = np.maximum(probs, 0)
    if probs.sum() > 0:
        probs /= probs.sum()
    return {
        bs: int(round(p * shots))
        for bs, p in zip(all_bitstrings, probs)
        if p > 0.001
    }


# Technique colors (consistent across all charts)
TECHNIQUE_COLORS = {
    "Baseline": "#8B8B8B",
    "DD": "#2196F3",
    "REM": "#4CAF50",
    "ZNE": "#FF9800",
    "Combined": "#E91E63",
}


# ═══════════════════════════════════════════════════════════════════════
# STAGE 1 — PROBLEM SETUP
# ═══════════════════════════════════════════════════════════════════════

@task(name="1 · Problem Setup", tags=["stage:1", "infra:cpu"])
def setup_problem(num_qubits: int) -> dict:
    """Generate a random secret key and compute search parameters."""
    logger = get_run_logger()
    N = 2 ** num_qubits
    secret_int = random.randint(0, N - 1)
    secret_key = format(secret_int, f'0{num_qubits}b')
    k = optimal_grover_iterations(num_qubits)
    theta = math.asin(1 / math.sqrt(N))
    success_prob = math.sin((2 * k + 1) * theta) ** 2

    logger.info(f"Search space: {N} items ({num_qubits} qubits)")
    logger.info(f"Secret key: |{secret_key}⟩ (decimal {secret_int})")
    logger.info(f"Optimal Grover iterations: {k}")
    logger.info(f"Theoretical success probability: {success_prob:.4f}")
    logger.info(f"Speedup: {N/k:.1f}×")

    return {
        "num_qubits": num_qubits,
        "search_space_size": N,
        "secret_key": secret_key,
        "secret_int": secret_int,
        "grover_iterations": k,
        "theoretical_success_prob": round(success_prob, 4),
        "classical_queries": N,
        "quantum_queries": k,
        "speedup": round(N / k, 1),
    }


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2 — BUILD GROVER CIRCUIT
# ═══════════════════════════════════════════════════════════════════════

@task(name="2 · Build Grover Circuit", tags=["stage:2", "infra:cpu"])
def build_grover_circuit(problem: dict) -> dict:
    """Build the full Grover circuit (H + Oracle + Diffuser) × k iterations."""
    logger = get_run_logger()
    from qiskit import QuantumCircuit

    n = problem["num_qubits"]
    k = problem["grover_iterations"]
    secret = problem["secret_key"]

    qc = QuantumCircuit(n)

    for i in range(n):
        qc.h(i)

    for _ in range(k):
        # Oracle: phase flip of |secret_key⟩
        for i, bit in enumerate(reversed(secret)):
            if bit == '0':
                qc.x(i)
        if n == 1:
            qc.z(0)
        elif n == 2:
            qc.cz(0, 1)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        for i, bit in enumerate(reversed(secret)):
            if bit == '0':
                qc.x(i)

        qc.barrier()

        # Diffuser: 2|s⟩⟨s| - I
        for i in range(n):
            qc.h(i)
        for i in range(n):
            qc.x(i)
        if n == 1:
            qc.z(0)
        elif n == 2:
            qc.cz(0, 1)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        for i in range(n):
            qc.x(i)
        for i in range(n):
            qc.h(i)

        qc.barrier()

    qc.measure_all()

    logger.info(f"Grover circuit: {qc.size()} gates, depth {qc.depth()}, {k} iterations")

    return {
        "gate_count": qc.size(),
        "depth": qc.depth(),
        "iterations": k,
        "_circuit": qc,
    }


# ═══════════════════════════════════════════════════════════════════════
# STAGE 3 — TRANSPILE
# ═══════════════════════════════════════════════════════════════════════

@task(name="3 · Transpile for IQM Garnet", tags=["stage:3", "infra:cpu"])
def transpile_grover(circuit_data: dict) -> dict:
    """Transpile Grover circuit for IQM Garnet native gates."""
    logger = get_run_logger()
    from qiskit import transpile

    token = get_iqm_token()
    qc = circuit_data["_circuit"]

    if token:
        backend = get_iqm_backend(token)
        qc_t = transpile(qc, backend=backend, optimization_level=2)
        logger.info(f"Transpiled for Garnet: {qc_t.size()} gates, depth {qc_t.depth()}")
    else:
        qc_t = transpile(qc, basis_gates=["r", "cz", "id"], optimization_level=2)
        logger.info(f"Transpiled (generic): {qc_t.size()} gates, depth {qc_t.depth()}")

    return {
        "original_gates": circuit_data["gate_count"],
        "original_depth": circuit_data["depth"],
        "transpiled_gates": qc_t.size(),
        "transpiled_depth": qc_t.depth(),
        "_transpiled": qc_t,
        "_original": qc,
    }


# ═══════════════════════════════════════════════════════════════════════
# STAGE 4 — MULTI-RUN EXECUTION
# ═══════════════════════════════════════════════════════════════════════

@task(
    name="4a · Baseline Run (no mitigation)",
    tags=["stage:4", "infra:qpu", "technique:baseline"],
    retries=2, retry_delay_seconds=10,
)
def run_grover_baseline(problem: dict, transpile_data: dict, shots: int) -> dict:
    """Execute raw Grover circuit — no error mitigation."""
    logger = get_run_logger()
    token = get_iqm_token()
    if not token:
        raise RuntimeError("IQM token required")

    from qiskit import transpile as qk_transpile
    backend = get_iqm_backend(token)
    qc = transpile_data["_original"]
    qc_t = qk_transpile(qc, backend=backend, optimization_level=2)

    secret = problem["secret_key"]
    n = problem["num_qubits"]
    logger.info(f"BASELINE: submitting {shots} shots, target |{secret}⟩...")

    t0 = time.time()
    job = backend.run(qc_t, shots=shots, use_timeslot=False)
    counts = job.result().get_counts()
    exec_time = round(time.time() - t0, 2)

    success_prob = _grover_success_prob(counts, secret, shots)
    top_bs = max(counts, key=counts.get)
    logger.info(f"BASELINE: P(target)={success_prob:.4f}, top=|{top_bs}⟩, time={exec_time}s")

    return {
        "technique": "Baseline",
        "label": "Baseline (no mitigation)",
        "counts": dict(sorted(counts.items(), key=lambda x: -x[1])),
        "success_prob": success_prob,
        "found_correct": (top_bs == secret),
        "target_count": counts.get(secret, 0),
        "top_bitstring": top_bs,
        "exec_time_s": exec_time,
        "shots": shots,
        "color": TECHNIQUE_COLORS["Baseline"],
        "description": "Raw circuit execution — no error mitigation applied",
        "num_qubits": n,
        "secret_key": secret,
    }


@task(
    name="4b · Dynamical Decoupling (DD)",
    tags=["stage:4", "infra:qpu", "technique:dd"],
    retries=2, retry_delay_seconds=10,
)
def run_grover_with_dd(problem: dict, transpile_data: dict, shots: int) -> dict:
    """
    Apply XX dynamical decoupling sequences to idle qubits.
    DD suppresses decoherence during the idle periods between oracle and diffuser.
    """
    logger = get_run_logger()
    token = get_iqm_token()
    if not token:
        raise RuntimeError("IQM token required")

    from qiskit import transpile as qk_transpile
    from qiskit.circuit.library import XGate
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import PadDynamicalDecoupling, ALAPScheduleAnalysis

    backend = get_iqm_backend(token)
    qc = transpile_data["_original"]
    qc_t = qk_transpile(qc, backend=backend, optimization_level=2)

    secret = problem["secret_key"]
    n = problem["num_qubits"]

    try:
        dd_sequence = [XGate(), XGate()]
        pm = PassManager([
            ALAPScheduleAnalysis(target=backend.target),
            PadDynamicalDecoupling(target=backend.target, dd_sequence=dd_sequence),
        ])
        qc_dd = pm.run(qc_t)
        logger.info(f"DD: applied XX sequences, {qc_dd.size()} gates (was {qc_t.size()})")
    except Exception as e:
        logger.warning(f"DD pass failed ({e}), falling back to base transpiled circuit")
        qc_dd = qc_t

    t0 = time.time()
    job = backend.run(qc_dd, shots=shots, use_timeslot=False)
    counts = job.result().get_counts()
    exec_time = round(time.time() - t0, 2)

    success_prob = _grover_success_prob(counts, secret, shots)
    top_bs = max(counts, key=counts.get)
    logger.info(f"DD: P(target)={success_prob:.4f}, top=|{top_bs}⟩, time={exec_time}s")

    return {
        "technique": "DD",
        "label": "Dynamical Decoupling (DD)",
        "counts": dict(sorted(counts.items(), key=lambda x: -x[1])),
        "success_prob": success_prob,
        "found_correct": (top_bs == secret),
        "target_count": counts.get(secret, 0),
        "top_bitstring": top_bs,
        "exec_time_s": exec_time,
        "shots": shots,
        "color": TECHNIQUE_COLORS["DD"],
        "description": "XX pulse sequences on idle qubits suppress decoherence between oracle and diffuser",
        "num_qubits": n,
        "secret_key": secret,
    }


@task(
    name="4c · Readout Error Mitigation (REM)",
    tags=["stage:4", "infra:qpu", "technique:rem"],
    retries=2, retry_delay_seconds=10,
)
def run_grover_with_rem(problem: dict, transpile_data: dict, shots: int) -> dict:
    """
    Calibrate readout errors, then apply inverse correction to Grover measurements.
    1. Run |0⟩^n and |1⟩^n calibration circuits → per-qubit assignment matrices
    2. Run Grover circuit → raw counts
    3. Apply pseudo-inverse correction → corrected success probability
    """
    logger = get_run_logger()
    token = get_iqm_token()
    if not token:
        raise RuntimeError("IQM token required")

    from qiskit import transpile as qk_transpile

    backend = get_iqm_backend(token)
    n = problem["num_qubits"]
    secret = problem["secret_key"]

    logger.info("REM: running calibration circuits...")
    qubit_matrices, inv_matrices = _calibrate_rem(backend, n, shots)
    for q, m in enumerate(qubit_matrices):
        err = round(1 - (m[0, 0] + m[1, 1]) / 2, 4)
        logger.info(f"  Q{q}: P(0|0)={m[0,0]:.3f}, P(1|1)={m[1,1]:.3f}, err≈{err:.4f}")

    logger.info("REM: running Grover circuit...")
    qc = transpile_data["_original"]
    qc_t = qk_transpile(qc, backend=backend, optimization_level=2)

    t0 = time.time()
    job = backend.run(qc_t, shots=shots, use_timeslot=False)
    raw_counts = job.result().get_counts()
    exec_time = round(time.time() - t0, 2)

    corrected_counts = _apply_rem_correction(raw_counts, inv_matrices, n, shots)
    success_prob = _grover_success_prob(corrected_counts, secret, shots)
    top_bs = max(corrected_counts, key=corrected_counts.get) if corrected_counts else secret

    logger.info(f"REM: P(target)={success_prob:.4f}, top=|{top_bs}⟩, time={exec_time}s+cal")

    return {
        "technique": "REM",
        "label": "Readout Error Mitigation (REM)",
        "counts": dict(sorted(corrected_counts.items(), key=lambda x: -x[1])),
        "raw_counts": raw_counts,
        "success_prob": success_prob,
        "found_correct": (top_bs == secret),
        "target_count": corrected_counts.get(secret, 0),
        "top_bitstring": top_bs,
        "exec_time_s": exec_time,
        "shots": shots,
        "qubit_readout_errors": [round(1 - (m[0, 0] + m[1, 1]) / 2, 4) for m in qubit_matrices],
        "qubit_matrices": [m.tolist() for m in qubit_matrices],
        "color": TECHNIQUE_COLORS["REM"],
        "description": "Calibration-based correction of readout (measurement) errors",
        "num_qubits": n,
        "secret_key": secret,
    }


@task(
    name="4d · Zero Noise Extrapolation (ZNE)",
    tags=["stage:4", "infra:qpu", "technique:zne"],
    retries=2, retry_delay_seconds=10,
)
def run_grover_with_zne(
    problem: dict, transpile_data: dict, shots: int, scale_factors: list
) -> dict:
    """
    Gate folding at multiple noise amplification levels, then Richardson extrapolation
    of the Grover success probability to the zero-noise limit.
    """
    logger = get_run_logger()
    token = get_iqm_token()
    if not token:
        raise RuntimeError("IQM token required")

    from qiskit import transpile as qk_transpile

    backend = get_iqm_backend(token)
    n = problem["num_qubits"]
    secret = problem["secret_key"]

    qc = transpile_data["_original"]
    qc_base = qk_transpile(qc, backend=backend, optimization_level=2)

    probs_at_scales = []
    counts_at_scale1 = {}

    for scale in scale_factors:
        qc_scaled = _fold_circuit(qc_base, scale)
        logger.info(f"ZNE scale={scale}: {qc_scaled.size()} gates, depth {qc_scaled.depth()}")

        t0 = time.time()
        job = backend.run(qc_scaled, shots=shots, use_timeslot=False)
        counts = job.result().get_counts()
        et = round(time.time() - t0, 2)

        p = _grover_success_prob(counts, secret, shots)
        probs_at_scales.append(p)
        logger.info(f"  P(target) at scale {scale}: {p:.4f} ({et}s)")

        if scale == 1:
            counts_at_scale1 = counts

    # Richardson extrapolation to zero noise
    scales_arr = np.array(scale_factors, dtype=float)
    probs_arr = np.array(probs_at_scales)

    if len(scale_factors) >= 2:
        degree = min(len(scale_factors) - 1, 2)
        coeffs = np.polyfit(scales_arr, probs_arr, degree)
        zne_prob = float(np.clip(np.polyval(coeffs, 0.0), 0.0, 1.0))
        extrap_type = "polynomial" if degree > 1 else "linear"
    else:
        zne_prob = probs_at_scales[0]
        coeffs = [zne_prob]
        extrap_type = "none"

    top_bs = max(counts_at_scale1, key=counts_at_scale1.get) if counts_at_scale1 else secret
    logger.info(f"ZNE: extrapolated P(target)={zne_prob:.4f} ({extrap_type})")

    return {
        "technique": "ZNE",
        "label": "Zero Noise Extrapolation (ZNE)",
        "counts": dict(sorted(counts_at_scale1.items(), key=lambda x: -x[1])),
        "success_prob": round(zne_prob, 4),
        "found_correct": (zne_prob >= 0.5),
        "target_count": counts_at_scale1.get(secret, 0),
        "top_bitstring": top_bs,
        "exec_time_s": 0,
        "shots": shots * len(scale_factors),
        "scale_factors": scale_factors,
        "success_probs_at_scales": [round(p, 4) for p in probs_at_scales],
        "extrapolation_coeffs": [round(c, 6) for c in coeffs],
        "extrapolation_type": extrap_type,
        "color": TECHNIQUE_COLORS["ZNE"],
        "description": f"Gate folding at scales {scale_factors}, {extrap_type} extrapolation of P(target) to zero noise",
        "num_qubits": n,
        "secret_key": secret,
    }


@task(
    name="4e · Combined (DD + REM + ZNE)",
    tags=["stage:4", "infra:qpu", "technique:combined"],
    retries=2, retry_delay_seconds=10,
)
def run_grover_combined(
    problem: dict, transpile_data: dict, shots: int,
    enable_dd: bool, enable_rem: bool, enable_zne: bool,
    zne_scale_factors: list,
) -> dict:
    """
    Apply all enabled techniques together:
      1. DD applied at circuit level (build time)
      2. Circuit executed at each ZNE scale factor
      3. REM correction applied post-measurement at each scale
      4. Richardson extrapolation of corrected success probabilities
    """
    logger = get_run_logger()
    token = get_iqm_token()
    if not token:
        raise RuntimeError("IQM token required")

    from qiskit import transpile as qk_transpile
    from qiskit.circuit.library import XGate

    backend = get_iqm_backend(token)
    n = problem["num_qubits"]
    secret = problem["secret_key"]

    qc = transpile_data["_original"]
    qc_t = qk_transpile(qc, backend=backend, optimization_level=2)

    # Step 1: Apply DD
    if enable_dd:
        try:
            from qiskit.transpiler import PassManager
            from qiskit.transpiler.passes import PadDynamicalDecoupling, ALAPScheduleAnalysis
            pm = PassManager([
                ALAPScheduleAnalysis(target=backend.target),
                PadDynamicalDecoupling(target=backend.target, dd_sequence=[XGate(), XGate()]),
            ])
            qc_t = pm.run(qc_t)
            logger.info("Combined: DD applied")
        except Exception as e:
            logger.warning(f"Combined: DD failed ({e})")

    # Step 2: REM calibration
    inv_matrices = None
    qubit_matrices = None
    if enable_rem:
        logger.info("Combined: running REM calibration...")
        qubit_matrices, inv_matrices = _calibrate_rem(backend, n, shots)
        logger.info("Combined: REM calibration done")

    # Step 3: ZNE runs
    scales = zne_scale_factors if enable_zne else [1]
    probs_at_scales = []
    last_counts = {}

    for scale in scales:
        qc_run = _fold_circuit(qc_t, scale)
        job = backend.run(qc_run, shots=shots, use_timeslot=False)
        counts = job.result().get_counts()

        if inv_matrices is not None:
            counts = _apply_rem_correction(counts, inv_matrices, n, shots)

        p = _grover_success_prob(counts, secret, shots)
        probs_at_scales.append(p)
        last_counts = counts
        logger.info(f"Combined scale={scale}: P(target)={p:.4f}")

    # Step 4: Extrapolate
    if enable_zne and len(scales) >= 2:
        coeffs = np.polyfit(np.array(scales, dtype=float), np.array(probs_at_scales), min(len(scales) - 1, 2))
        combined_prob = float(np.clip(np.polyval(coeffs, 0.0), 0.0, 1.0))
    else:
        combined_prob = probs_at_scales[0]

    active = []
    if enable_dd:
        active.append("DD")
    if enable_rem:
        active.append("REM")
    if enable_zne:
        active.append("ZNE")

    top_bs = max(last_counts, key=last_counts.get) if last_counts else secret
    logger.info(f"COMBINED ({'+'.join(active)}): P(target)={combined_prob:.4f}")

    return {
        "technique": "Combined",
        "label": f"Combined ({'+'.join(active)})",
        "counts": dict(sorted(last_counts.items(), key=lambda x: -x[1])),
        "success_prob": round(combined_prob, 4),
        "found_correct": (combined_prob >= 0.5),
        "target_count": last_counts.get(secret, 0),
        "top_bitstring": top_bs,
        "exec_time_s": 0,
        "shots": shots * len(scales),
        "active_techniques": active,
        "color": TECHNIQUE_COLORS["Combined"],
        "description": f"All enabled techniques applied together: {', '.join(active)}",
        "num_qubits": n,
        "secret_key": secret,
    }


# ═══════════════════════════════════════════════════════════════════════
# STAGE 5 — CLASSICAL COMPARISON
# ═══════════════════════════════════════════════════════════════════════

@task(name="5 · Classical Comparison", tags=["stage:5", "infra:cpu"])
def classical_comparison(problem: dict, baseline_result: dict) -> dict:
    """Simulate classical brute-force search for comparison baseline."""
    logger = get_run_logger()
    N = problem["search_space_size"]
    secret_int = problem["secret_int"]

    classical_queries = 0
    for i in range(N):
        classical_queries += 1
        if i == secret_int:
            break

    avg_classical = N / 2
    quantum_queries = problem["grover_iterations"]

    logger.info(f"Classical (this run): {classical_queries} queries")
    logger.info(f"Classical (average): {avg_classical:.0f} queries")
    logger.info(f"Quantum: {quantum_queries} queries")
    logger.info(f"Speedup (avg): {avg_classical / quantum_queries:.1f}×")

    return {
        "classical_this_run": classical_queries,
        "classical_average": avg_classical,
        "classical_worst": N,
        "quantum_queries": quantum_queries,
        "speedup_average": round(avg_classical / quantum_queries, 1),
        "speedup_worst": round(N / quantum_queries, 1),
        "quantum_found": baseline_result["found_correct"],
        "quantum_probability": baseline_result["success_prob"],
    }


# ═══════════════════════════════════════════════════════════════════════
# STAGE 6 — SCALING ANALYSIS WITH MITIGATION
# ═══════════════════════════════════════════════════════════════════════

@task(
    name="6 · Scaling Analysis with Mitigation",
    tags=["stage:6", "infra:qpu"],
)
def run_scaling_with_mitigation(
    scaling_sizes: list,
    shots: int,
    enable_dd: bool,
    enable_rem: bool,
    enable_zne: bool,
    enable_combined: bool,
    zne_scale_factors: list,
) -> dict:
    """
    Run Grover at multiple qubit counts for each enabled technique.
    Returns dict: technique_name → list of per-size result dicts.
    """
    logger = get_run_logger()
    from qiskit import transpile as qk_transpile
    from qiskit.circuit.library import XGate

    token = get_iqm_token()
    if not token:
        raise RuntimeError("IQM token required")
    backend = get_iqm_backend(token)

    num_enabled = sum([enable_dd, enable_rem, enable_zne])
    run_combined_flag = enable_combined and num_enabled >= 2

    techniques = ["Baseline"]
    if enable_dd:
        techniques.append("DD")
    if enable_rem:
        techniques.append("REM")
    if enable_zne:
        techniques.append("ZNE")
    if run_combined_flag:
        techniques.append("Combined")

    results = {t: [] for t in techniques}

    for n in scaling_sizes:
        logger.info(f"\n{'─'*40}")
        logger.info(f"Scaling: {n} qubits, search space = {2**n}")

        # Problem setup
        prob = setup_problem.fn(n)
        circ_data = build_grover_circuit.fn(prob)
        transp_data = transpile_grover.fn(circ_data)

        secret = prob["secret_key"]
        qc = transp_data["_original"]
        qc_t = qk_transpile(qc, backend=backend, optimization_level=2)

        def _record(tech, counts, prob_val):
            results[tech].append({
                "num_qubits": n,
                "search_space": 2 ** n,
                "secret_key": secret,
                "grover_iterations": prob["grover_iterations"],
                "classical_queries_avg": 2 ** n / 2,
                "classical_queries_worst": 2 ** n,
                "success_prob": prob_val,
                "speedup": round((2 ** n) / prob["grover_iterations"], 1),
            })

        # Baseline
        t0 = time.time()
        job = backend.run(qc_t, shots=shots, use_timeslot=False)
        counts = job.result().get_counts()
        p = _grover_success_prob(counts, secret, shots)
        logger.info(f"  Baseline P(target)={p:.4f} ({round(time.time()-t0,1)}s)")
        _record("Baseline", counts, p)

        # DD
        if enable_dd:
            try:
                from qiskit.transpiler import PassManager
                from qiskit.transpiler.passes import PadDynamicalDecoupling, ALAPScheduleAnalysis
                pm = PassManager([
                    ALAPScheduleAnalysis(target=backend.target),
                    PadDynamicalDecoupling(target=backend.target, dd_sequence=[XGate(), XGate()]),
                ])
                qc_dd = pm.run(qc_t)
            except Exception:
                qc_dd = qc_t
            t0 = time.time()
            counts_dd = backend.run(qc_dd, shots=shots, use_timeslot=False).result().get_counts()
            p_dd = _grover_success_prob(counts_dd, secret, shots)
            logger.info(f"  DD P(target)={p_dd:.4f} ({round(time.time()-t0,1)}s)")
            _record("DD", counts_dd, p_dd)

        # REM
        if enable_rem:
            _, inv_m = _calibrate_rem(backend, n, shots)
            t0 = time.time()
            counts_raw = backend.run(qc_t, shots=shots, use_timeslot=False).result().get_counts()
            counts_rem = _apply_rem_correction(counts_raw, inv_m, n, shots)
            p_rem = _grover_success_prob(counts_rem, secret, shots)
            logger.info(f"  REM P(target)={p_rem:.4f} ({round(time.time()-t0,1)}s+cal)")
            _record("REM", counts_rem, p_rem)

        # ZNE
        if enable_zne:
            probs_zne = []
            for scale in zne_scale_factors:
                qc_sc = _fold_circuit(qc_t, scale)
                c = backend.run(qc_sc, shots=shots, use_timeslot=False).result().get_counts()
                probs_zne.append(_grover_success_prob(c, secret, shots))
            if len(zne_scale_factors) >= 2:
                cf = np.polyfit(np.array(zne_scale_factors, dtype=float), np.array(probs_zne), min(len(zne_scale_factors) - 1, 2))
                p_zne = float(np.clip(np.polyval(cf, 0.0), 0.0, 1.0))
            else:
                p_zne = probs_zne[0]
            logger.info(f"  ZNE P(target)={p_zne:.4f}")
            _record("ZNE", {}, round(p_zne, 4))

        # Combined
        if run_combined_flag:
            qc_comb = qc_t.copy()
            if enable_dd:
                try:
                    from qiskit.transpiler import PassManager
                    from qiskit.transpiler.passes import PadDynamicalDecoupling, ALAPScheduleAnalysis
                    pm = PassManager([
                        ALAPScheduleAnalysis(target=backend.target),
                        PadDynamicalDecoupling(target=backend.target, dd_sequence=[XGate(), XGate()]),
                    ])
                    qc_comb = pm.run(qc_t)
                except Exception:
                    pass
            inv_c = None
            if enable_rem:
                _, inv_c = _calibrate_rem(backend, n, shots)
            scales_c = zne_scale_factors if enable_zne else [1]
            probs_c = []
            for scale in scales_c:
                qc_sc = _fold_circuit(qc_comb, scale)
                c = backend.run(qc_sc, shots=shots, use_timeslot=False).result().get_counts()
                if inv_c:
                    c = _apply_rem_correction(c, inv_c, n, shots)
                probs_c.append(_grover_success_prob(c, secret, shots))
            if enable_zne and len(scales_c) >= 2:
                cf = np.polyfit(np.array(scales_c, dtype=float), np.array(probs_c), min(len(scales_c) - 1, 2))
                p_comb = float(np.clip(np.polyval(cf, 0.0), 0.0, 1.0))
            else:
                p_comb = probs_c[0]
            logger.info(f"  Combined P(target)={p_comb:.4f}")
            _record("Combined", {}, round(p_comb, 4))

    return results


# ═══════════════════════════════════════════════════════════════════════
# STAGE 7 — ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════

@task(name="7.1 · Success Probability Chart", tags=["stage:7", "reporting"])
def publish_success_prob_chart(all_results: list, problem: dict) -> None:
    """SVG horizontal bar chart: success probability per technique."""
    width = 620
    bar_h = 52
    gap = 14
    left_m = 230
    right_m = 90
    chart_w = width - left_m - right_m
    top_m = 50
    height = top_m + len(all_results) * (bar_h + gap) + 60

    theoretical = problem["theoretical_success_prob"]
    secret = problem["secret_key"]
    best = max(all_results, key=lambda r: r["success_prob"])

    bars_svg = ""

    for i, r in enumerate(all_results):
        y = top_m + i * (bar_h + gap)
        bw = max(2, r["success_prob"] * chart_w)
        color = r.get("color", "#888")
        is_best = (r["technique"] == best["technique"])

        bars_svg += (
            f'<text x="{left_m - 10}" y="{y + bar_h/2 + 5}" text-anchor="end" '
            f'font-family="monospace" font-size="13" fill="#333" '
            f'font-weight="{"bold" if is_best else "normal"}">'
            f'{r["label"]}</text>\n'
        )
        bars_svg += (
            f'<rect x="{left_m}" y="{y}" width="{bw}" height="{bar_h}" '
            f'fill="{color}" rx="4" opacity="0.85"/>\n'
        )
        pct_improve = ""
        if i > 0 and all_results[0]["success_prob"] > 0:
            delta = (r["success_prob"] - all_results[0]["success_prob"]) / all_results[0]["success_prob"] * 100
            sign = "+" if delta >= 0 else ""
            pct_improve = f"  ({sign}{delta:.1f}%)"
        bars_svg += (
            f'<text x="{left_m + bw + 8}" y="{y + bar_h/2 + 5}" '
            f'font-family="monospace" font-size="14" font-weight="bold" fill="{color}">'
            f'{r["success_prob"]:.4f}{pct_improve}</text>\n'
        )
        if is_best:
            bars_svg += (
                f'<text x="{left_m - 10}" y="{y - 2}" text-anchor="end" '
                f'font-family="Arial" font-size="10" fill="#E91E63">BEST</text>\n'
            )

    # Theoretical line
    tx = left_m + theoretical * chart_w
    bars_svg += (
        f'<line x1="{tx}" y1="{top_m - 10}" x2="{tx}" y2="{height - 30}" '
        f'stroke="#9C27B0" stroke-width="2" stroke-dasharray="6,4"/>\n'
        f'<text x="{tx + 4}" y="{top_m - 14}" font-family="monospace" font-size="10" '
        f'fill="#9C27B0">theory={theoretical:.3f}</text>\n'
    )

    # X-axis ticks
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        x = left_m + frac * chart_w
        bars_svg += (
            f'<line x1="{x}" y1="{height - 28}" x2="{x}" y2="{height - 22}" stroke="#ccc"/>\n'
            f'<text x="{x}" y="{height - 12}" text-anchor="middle" '
            f'font-family="monospace" font-size="10" fill="#999">{frac:.2f}</text>\n'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="{width}" height="{height}" fill="#FAFAFA" rx="8"/>\n'
        f'<text x="{width/2}" y="22" text-anchor="middle" font-family="Arial" '
        f'font-size="15" font-weight="bold" fill="#333">'
        f'Grover Success Probability — Error Mitigation Comparison</text>\n'
        f'<text x="{width/2}" y="40" text-anchor="middle" font-family="Arial" '
        f'font-size="11" fill="#888">Target: |{secret}⟩. Higher P(target) = better.</text>\n'
        f'{bars_svg}'
        f'</svg>'
    )

    create_markdown_artifact(
        key="grover-v2-success-prob-chart",
        markdown=f"# Grover v2 — Success Probability by Technique\n\n{svg}",
        description="Horizontal bar chart: P(finding secret key) per mitigation technique",
    )


@task(name="7.2 · Per-Technique Measurement Histograms", tags=["stage:7", "reporting"])
def publish_technique_histograms(all_results: list, problem: dict) -> None:
    """One SVG histogram per technique, secret key highlighted in gold."""
    secret = problem["secret_key"]
    shots = problem.get("shots", 4096)

    all_svgs = ""

    for r in all_results:
        counts = r["counts"]
        n = r["num_qubits"]
        run_shots = r["shots"]

        all_bs = sorted(counts.keys(), key=lambda x: int(x, 2))
        if len(all_bs) > 24:
            top = dict(sorted(counts.items(), key=lambda x: -x[1])[:16])
            if secret not in top:
                top[secret] = counts.get(secret, 0)
            all_bs = sorted(top.keys(), key=lambda x: -counts.get(x, 0))

        if not all_bs:
            continue

        max_count = max(counts.get(bs, 0) for bs in all_bs) or 1
        bar_w = max(22, min(42, 640 // len(all_bs)))
        left_m = 45
        bottom_m = 70
        top_m = 44
        chart_h = 200
        w = left_m + len(all_bs) * bar_w + 50
        h = top_m + chart_h + bottom_m

        bars = ""
        for i, bs in enumerate(all_bs):
            count = counts.get(bs, 0)
            prob = count / run_shots if run_shots > 0 else 0
            bh = (count / max_count) * chart_h
            x = left_m + i * bar_w
            y = top_m + chart_h - bh
            is_target = (bs == secret)
            fill = "#FFD700" if is_target else r.get("color", "#5C6BC0")
            stroke = "#FF6F00" if is_target else "none"
            sw = "2" if is_target else "0"

            bars += (
                f'<rect x="{x+2}" y="{y}" width="{bar_w-4}" height="{bh}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" rx="2" opacity="0.85"/>\n'
            )
            if prob > 0.02:
                bars += (
                    f'<text x="{x+bar_w/2}" y="{y-4}" text-anchor="middle" '
                    f'font-family="monospace" font-size="9" fill="#333">{prob:.2f}</text>\n'
                )
            rx = x + bar_w / 2
            ry = top_m + chart_h + 10
            fw = "bold" if is_target else "normal"
            lc = "#FF6F00" if is_target else "#666"
            bars += (
                f'<text x="{rx}" y="{ry}" text-anchor="start" font-family="monospace" '
                f'font-size="10" font-weight="{fw}" fill="{lc}" '
                f'transform="rotate(50, {rx}, {ry})">|{bs}⟩</text>\n'
            )
            if is_target:
                bars += (
                    f'<text x="{x+bar_w/2}" y="{y-16}" text-anchor="middle" font-size="14">⭐</text>\n'
                )

        found_text = "FOUND" if r.get("found_correct") else "NOT FOUND"
        color_label = r.get("color", "#333")
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" style="margin-bottom:8px">\n'
            f'<rect width="{w}" height="{h}" fill="#FAFAFA" rx="6"/>\n'
            f'<rect x="0" y="0" width="6" height="{h}" fill="{color_label}" rx="3"/>\n'
            f'<text x="{w/2}" y="18" text-anchor="middle" font-family="Arial" '
            f'font-size="13" font-weight="bold" fill="#333">{r["label"]} — {found_text}</text>\n'
            f'<text x="{w/2}" y="33" text-anchor="middle" font-family="Arial" '
            f'font-size="10" fill="#888">P(|{secret}⟩) = {r["success_prob"]:.4f}'
            f' ({r["shots"]:,} shots)</text>\n'
            f'{bars}'
            f'</svg>\n\n'
        )
        all_svgs += f"### {r['label']}\n\n{svg}\n"

    create_markdown_artifact(
        key="grover-v2-technique-histograms",
        markdown=f"# Per-Technique Measurement Histograms\n\n{all_svgs}",
        description="Measurement distribution for each mitigation technique, secret key in gold",
    )


@task(name="7.3 · ZNE Extrapolation Curve", tags=["stage:7", "reporting"])
def publish_zne_curve(zne_result: dict) -> None:
    """SVG: Grover success probability vs noise scale factor with extrapolation to λ=0."""
    if not zne_result or "scale_factors" not in zne_result:
        return

    scales = zne_result["scale_factors"]
    probs = zne_result["success_probs_at_scales"]
    zne_prob = zne_result["success_prob"]
    coeffs = zne_result["extrapolation_coeffs"]
    secret = zne_result["secret_key"]

    width, height = 500, 350
    pl, pr, pt, pb = 70, 40, 55, 50
    pw = width - pl - pr
    ph = height - pt - pb

    max_scale = max(scales) + 0.5
    min_p = max(0.0, min(min(probs), zne_prob) - 0.08)
    max_p = min(1.0, max(max(probs), zne_prob) + 0.05)

    def sx(v): return pl + (v / max_scale) * pw
    def sy(v): return pt + ph - ((v - min_p) / (max_p - min_p + 1e-9)) * ph

    svg_c = (
        f'<line x1="{pl}" y1="{pt}" x2="{pl}" y2="{pt+ph}" stroke="#999" stroke-width="1"/>\n'
        f'<line x1="{pl}" y1="{pt+ph}" x2="{pl+pw}" y2="{pt+ph}" stroke="#999" stroke-width="1"/>\n'
        f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-family="Arial" '
        f'font-size="12" fill="#666">Noise Scale Factor (λ)</text>\n'
        f'<text x="15" y="{height/2}" text-anchor="middle" font-family="Arial" font-size="12" '
        f'fill="#666" transform="rotate(-90, 15, {height/2})">P(target) — success probability</text>\n'
    )

    for fv in np.arange(round(min_p, 1), max_p + 0.05, 0.1):
        y = sy(fv)
        svg_c += (
            f'<line x1="{pl}" y1="{y}" x2="{pl+pw}" y2="{y}" stroke="#eee" stroke-width="1"/>\n'
            f'<text x="{pl-8}" y="{y+4}" text-anchor="end" font-family="monospace" '
            f'font-size="10" fill="#999">{fv:.1f}</text>\n'
        )

    line_pts = " ".join(
        f"{sx(lam):.1f},{sy(float(np.polyval(coeffs, lam))):.1f}"
        for lam in np.linspace(0, max_scale, 60)
    )
    svg_c += f'<polyline points="{line_pts}" fill="none" stroke="#FF9800" stroke-width="2" stroke-dasharray="6,3"/>\n'

    for s, p in zip(scales, probs):
        svg_c += (
            f'<circle cx="{sx(s)}" cy="{sy(p)}" r="7" fill="#FF9800" stroke="white" stroke-width="2"/>\n'
            f'<text x="{sx(s)}" y="{sy(p)-12}" text-anchor="middle" font-family="monospace" '
            f'font-size="10" fill="#FF9800">{p:.3f}</text>\n'
            f'<text x="{sx(s)}" y="{pt+ph+14}" text-anchor="middle" font-family="monospace" '
            f'font-size="11" fill="#666">λ={s}</text>\n'
        )

    svg_c += (
        f'<circle cx="{sx(0)}" cy="{sy(zne_prob)}" r="9" fill="#E91E63" stroke="white" stroke-width="2"/>\n'
        f'<text x="{sx(0)+14}" y="{sy(zne_prob)+5}" font-family="monospace" font-size="12" '
        f'font-weight="bold" fill="#E91E63">ZNE = {zne_prob:.4f}</text>\n'
        f'<line x1="{sx(0)}" y1="{pt}" x2="{sx(0)}" y2="{pt+ph}" '
        f'stroke="#E91E63" stroke-width="1" stroke-dasharray="4,4" opacity="0.5"/>\n'
    )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="{width}" height="{height}" fill="#FAFAFA" rx="8"/>\n'
        f'<text x="{width/2}" y="25" text-anchor="middle" font-family="Arial" font-size="15" '
        f'font-weight="bold" fill="#333">ZNE Extrapolation — Grover Success Probability</text>\n'
        f'<text x="{width/2}" y="42" text-anchor="middle" font-family="Arial" font-size="11" '
        f'fill="#888">P(|{secret}⟩) measured at amplified noise levels → extrapolated to λ=0</text>\n'
        f'{svg_c}</svg>'
    )

    create_markdown_artifact(
        key="grover-v2-zne-curve",
        markdown=f"# ZNE Extrapolation Curve\n\n{svg}",
        description="Grover success probability vs noise scale factor, extrapolated to zero noise",
    )


@task(name="7.4 · REM Readout Error Map", tags=["stage:7", "reporting"])
def publish_rem_heatmap(rem_result: dict) -> None:
    """SVG: per-qubit 2×2 assignment matrices with readout error rates."""
    if not rem_result or "qubit_matrices" not in rem_result:
        return

    matrices = rem_result["qubit_matrices"]
    errors = rem_result["qubit_readout_errors"]
    n = rem_result["num_qubits"]

    cell = 60
    block_w = cell * 2 + 20
    width = 140 + n * block_w + 40
    height = 240

    svg_c = ""
    for q in range(n):
        m = matrices[q]
        bx = 140 + q * block_w
        by = 65
        err = errors[q]
        ec = "#4CAF50" if err < 0.02 else "#FF9800" if err < 0.05 else "#F44336"

        svg_c += (
            f'<text x="{bx+cell}" y="{by-16}" text-anchor="middle" font-family="monospace" '
            f'font-size="13" font-weight="bold" fill="{ec}">Q{q}</text>\n'
            f'<text x="{bx+cell}" y="{by-3}" text-anchor="middle" font-family="monospace" '
            f'font-size="10" fill="{ec}">err={err:.3f}</text>\n'
        )
        for row in range(2):
            for col in range(2):
                x = bx + col * cell
                y = by + row * cell
                val = m[row][col]
                if row == col:
                    intensity = val
                    r_c = int(255 * (1 - intensity))
                    g_c = 255
                    b_c = int(255 * (1 - intensity))
                else:
                    intensity = min(val * 10, 1.0)
                    r_c = 255
                    g_c = int(255 * (1 - intensity))
                    b_c = int(255 * (1 - intensity))
                fill = f"rgb({r_c},{g_c},{b_c})"
                svg_c += (
                    f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" '
                    f'fill="{fill}" stroke="#ccc" rx="4"/>\n'
                    f'<text x="{x+cell/2}" y="{y+cell/2+5}" text-anchor="middle" '
                    f'font-family="monospace" font-size="12" fill="#333">{val:.3f}</text>\n'
                )
        for idx, lbl in enumerate(["P(0|·)", "P(1|·)"]):
            svg_c += (
                f'<text x="{bx-5}" y="{by+idx*cell+cell/2+4}" text-anchor="end" '
                f'font-family="monospace" font-size="9" fill="#999">{lbl}</text>\n'
            )
        for idx, lbl in enumerate(["prep 0", "prep 1"]):
            svg_c += (
                f'<text x="{bx+idx*cell+cell/2}" y="{by+2*cell+14}" text-anchor="middle" '
                f'font-family="monospace" font-size="9" fill="#999">{lbl}</text>\n'
            )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="{width}" height="{height}" fill="#FAFAFA" rx="8"/>\n'
        f'<text x="{width/2}" y="26" text-anchor="middle" font-family="Arial" font-size="15" '
        f'font-weight="bold" fill="#333">Readout Error Assignment Matrices</text>\n'
        f'<text x="{width/2}" y="43" text-anchor="middle" font-family="Arial" font-size="11" '
        f'fill="#888">Diagonal = correct readout. Off-diagonal = error. Green = good, Red = noisy.</text>\n'
        f'{svg_c}</svg>'
    )

    create_markdown_artifact(
        key="grover-v2-rem-readout-map",
        markdown=f"# REM Readout Error Map\n\n{svg}",
        description="Per-qubit readout assignment matrices used for REM correction",
    )


@task(name="7.5 · Quantum vs Classical Chart", tags=["stage:7", "reporting"])
def publish_comparison_chart(problem: dict, comparison: dict) -> None:
    """SVG: query count comparison — Grover vs classical brute-force."""
    N = problem["search_space_size"]
    q_q = comparison["quantum_queries"]
    c_avg = comparison["classical_average"]
    c_worst = comparison["classical_worst"]

    width, height = 500, 300
    bar_w = 80
    chart_h = 180
    top = 60
    left = 110
    max_val = c_worst

    def bh(v):
        return (v / max_val) * chart_h

    svg_c = ""
    bars = [("Classical\n(worst)", c_worst, "#F44336"),
            ("Classical\n(avg)", c_avg, "#FF9800"),
            ("Quantum\n(Grover)", q_q, "#4CAF50")]

    for i, (lbl, val, color) in enumerate(bars):
        x = left + i * (bar_w + 32)
        bh_val = bh(val)
        y = top + chart_h - bh_val
        svg_c += (
            f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bh_val}" fill="{color}" rx="4" opacity="0.85"/>\n'
            f'<text x="{x+bar_w/2}" y="{y-8}" text-anchor="middle" font-family="monospace" '
            f'font-size="16" font-weight="bold" fill="{color}">{int(val)}</text>\n'
        )
        for j, line in enumerate(lbl.split('\n')):
            svg_c += (
                f'<text x="{x+bar_w/2}" y="{top+chart_h+20+j*15}" text-anchor="middle" '
                f'font-family="Arial" font-size="11" fill="#666">{line}</text>\n'
            )

    sx2 = left + 2 * (bar_w + 32) + bar_w + 18
    svg_c += (
        f'<text x="{sx2}" y="{top+chart_h/2}" font-family="Arial" font-size="22" '
        f'font-weight="bold" fill="#4CAF50">{comparison["speedup_worst"]:.0f}× faster</text>\n'
        f'<text x="{sx2}" y="{top+chart_h/2+20}" font-family="Arial" font-size="12" fill="#888">vs worst case</text>\n'
    )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="{width}" height="{height}" fill="#FAFAFA" rx="8"/>\n'
        f'<text x="{width/2}" y="22" text-anchor="middle" font-family="Arial" font-size="15" '
        f'font-weight="bold" fill="#333">Quantum vs Classical: Queries to Find Key</text>\n'
        f'<text x="{width/2}" y="40" text-anchor="middle" font-family="Arial" font-size="11" '
        f'fill="#888">Search space: {N} items ({problem["num_qubits"]} qubits)</text>\n'
        f'{svg_c}</svg>'
    )

    create_markdown_artifact(
        key="grover-v2-quantum-vs-classical",
        markdown=f"# Quantum vs Classical\n\n{svg}",
        description="Query count comparison: Grover vs brute-force search",
    )


@task(name="7.6 · Scaling Curves per Technique", tags=["stage:7", "reporting"])
def publish_scaling_curves(scaling_results: dict) -> None:
    """SVG: success probability vs qubit count, one line per technique."""
    if not scaling_results:
        return

    techniques = list(scaling_results.keys())
    sizes = sorted({r["num_qubits"] for rs in scaling_results.values() for r in rs})
    if not sizes:
        return

    width, height = 580, 400
    pl, pr, pt, pb = 70, 40, 60, 70
    pw = width - pl - pr
    ph = height - pt - pb

    min_x, max_x = min(sizes), max(sizes)
    span_x = max_x - min_x or 1

    def sx(v): return pl + ((v - min_x) / span_x) * pw
    def sy(v): return pt + ph - v * ph

    svg_c = ""

    # Grid
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        y = pt + ph * (1 - frac)
        svg_c += (
            f'<line x1="{pl}" y1="{y}" x2="{pl+pw}" y2="{y}" stroke="#eee" stroke-width="1"/>\n'
            f'<text x="{pl-8}" y="{y+4}" text-anchor="end" font-family="monospace" '
            f'font-size="10" fill="#999">{frac:.2f}</text>\n'
        )

    # X labels
    for n_q in sizes:
        x = sx(n_q)
        svg_c += (
            f'<text x="{x}" y="{pt+ph+18}" text-anchor="middle" font-family="monospace" '
            f'font-size="12" fill="#555">{n_q}q</text>\n'
            f'<text x="{x}" y="{pt+ph+32}" text-anchor="middle" font-family="monospace" '
            f'font-size="10" fill="#999">N={2**n_q}</text>\n'
        )

    # Lines per technique
    for tech_name, tech_results in scaling_results.items():
        color = TECHNIQUE_COLORS.get(tech_name, "#888")
        sorted_r = sorted(tech_results, key=lambda r: r["num_qubits"])
        pts = " ".join(f"{sx(r['num_qubits']):.1f},{sy(r['success_prob']):.1f}" for r in sorted_r)
        svg_c += (
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>\n'
        )
        for r in sorted_r:
            x = sx(r["num_qubits"])
            y = sy(r["success_prob"])
            svg_c += (
                f'<circle cx="{x}" cy="{y}" r="6" fill="{color}" stroke="white" stroke-width="2"/>\n'
                f'<text x="{x+8}" y="{y+4}" font-family="monospace" font-size="9" fill="{color}">'
                f'{r["success_prob"]:.3f}</text>\n'
            )

    # Legend
    ly = pt - 35
    for i, tech_name in enumerate(techniques):
        color = TECHNIQUE_COLORS.get(tech_name, "#888")
        lx = pl + i * 110
        svg_c += (
            f'<line x1="{lx}" y1="{ly}" x2="{lx+20}" y2="{ly}" stroke="{color}" stroke-width="2.5"/>\n'
            f'<circle cx="{lx+10}" cy="{ly}" r="4" fill="{color}"/>\n'
            f'<text x="{lx+25}" y="{ly+4}" font-family="Arial" font-size="11" fill="{color}">{tech_name}</text>\n'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="{width}" height="{height}" fill="#FAFAFA" rx="8"/>\n'
        f'<text x="{width/2}" y="20" text-anchor="middle" font-family="Arial" font-size="15" '
        f'font-weight="bold" fill="#333">Scaling Analysis — P(target) per Technique</text>\n'
        f'<text x="{width/2}" y="37" text-anchor="middle" font-family="Arial" font-size="11" '
        f'fill="#888">Each line shows how mitigation effectiveness changes with problem size</text>\n'
        f'{svg_c}'
        f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-family="Arial" '
        f'font-size="12" fill="#666">Problem Size (qubits / search space)</text>\n'
        f'<text x="15" y="{pt+ph/2}" text-anchor="middle" font-family="Arial" font-size="12" '
        f'fill="#666" transform="rotate(-90, 15, {pt+ph/2})">P(target)</text>\n'
        f'</svg>'
    )

    create_markdown_artifact(
        key="grover-v2-scaling-curves",
        markdown=f"# Scaling Analysis — Per-Technique Success Probability\n\n{svg}",
        description="Success probability vs qubit count for each mitigation technique",
    )


@task(name="7.7 · Scaling Results Table", tags=["stage:7", "reporting"])
def publish_scaling_table(scaling_results: dict) -> None:
    """Detailed table: technique × qubit size → success probability."""
    if not scaling_results:
        return

    techniques = list(scaling_results.keys())
    sizes = sorted({r["num_qubits"] for rs in scaling_results.values() for r in rs})

    # Build index: (technique, n_qubits) → result
    idx = {}
    for tech, results in scaling_results.items():
        for r in results:
            idx[(tech, r["num_qubits"])] = r

    # Header
    header_cols = "| Qubits | Search Space | Grover Iters | Speedup |"
    sep_cols = "|--------|-------------|--------------|---------|"
    for t in techniques:
        header_cols += f" {t} P(target) |"
        sep_cols += "-----------|"

    rows = ""
    for n in sizes:
        r_base = idx.get(("Baseline", n), {})
        row = (
            f"| {n} | {2**n} | "
            f"{r_base.get('grover_iterations','—')} | "
            f"{r_base.get('speedup','—')}× |"
        )
        for t in techniques:
            r = idx.get((t, n), {})
            p = r.get("success_prob", "—")
            row += f" {p:.4f} |" if isinstance(p, float) else f" {p} |"
        rows += row + "\n"

    md = (
        f"# Scaling Analysis — Detailed Results\n\n"
        f"{header_cols}\n{sep_cols}\n{rows}\n"
        f"## Observations\n\n"
        f"- **Circuit depth** grows with qubit count, increasing noise on real hardware\n"
        f"- **Mitigation benefit** may be most pronounced at larger qubit counts where noise is higher\n"
        f"- **ZNE** improves over baseline when the noise-fidelity curve extrapolates cleanly\n"
        f"- **REM** consistently corrects measurement errors regardless of circuit depth\n"
    )

    create_markdown_artifact(
        key="grover-v2-scaling-table",
        markdown=md,
        description="Scaling results: P(target) for each technique at each problem size",
    )


@task(name="7.8 · HTML Experiment Report", tags=["stage:7", "reporting"])
def publish_html_report(
    problem: dict,
    circuit_data: dict,
    transpile_data: dict,
    all_results: list,
    comparison: dict,
    scaling_results: dict,
    enable_dd: bool,
    enable_rem: bool,
    enable_zne: bool,
    enable_combined: bool,
    zne_scale_factors: list,
) -> None:
    """Self-contained rich HTML report embedded in a Prefect markdown artifact."""

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n = problem["num_qubits"]
    secret = problem["secret_key"]
    baseline = all_results[0]
    best = max(all_results, key=lambda r: r["success_prob"])
    improvement = (
        (best["success_prob"] - baseline["success_prob"]) / max(baseline["success_prob"], 0.001) * 100
        if len(all_results) > 1 else 0.0
    )

    active_techs = []
    if enable_dd:
        active_techs.append("DD")
    if enable_rem:
        active_techs.append("REM")
    if enable_zne:
        active_techs.append("ZNE")
    num_enabled = len(active_techs)

    # ── inline SVG: success probability bar chart ──────────────────────
    chart_w = 400
    bar_h_svg = 38
    gap_svg = 10
    left_svg = 180
    top_svg = 30
    total_h = top_svg + len(all_results) * (bar_h_svg + gap_svg) + 30

    bars_inline = ""
    for i, r in enumerate(all_results):
        y = top_svg + i * (bar_h_svg + gap_svg)
        bw = max(2, r["success_prob"] * chart_w)
        color = r.get("color", "#888")
        bars_inline += (
            f'<text x="{left_svg-8}" y="{y+bar_h_svg/2+5}" text-anchor="end" '
            f'font-family="monospace" font-size="12" fill="#333">{r["label"]}</text>'
            f'<rect x="{left_svg}" y="{y}" width="{bw}" height="{bar_h_svg}" '
            f'fill="{color}" rx="3" opacity="0.85"/>'
            f'<text x="{left_svg+bw+6}" y="{y+bar_h_svg/2+5}" font-family="monospace" '
            f'font-size="12" font-weight="bold" fill="{color}">{r["success_prob"]:.4f}</text>'
        )

    inline_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{left_svg+chart_w+120}" '
        f'height="{total_h}" style="display:block;margin:0 auto">'
        f'{bars_inline}</svg>'
    )

    # ── results table rows ─────────────────────────────────────────────
    result_rows = ""
    for r in all_results:
        delta = (r["success_prob"] - baseline["success_prob"]) / max(baseline["success_prob"], 0.001) * 100
        sign = "+" if delta >= 0 else ""
        is_best_row = (r["technique"] == best["technique"])
        row_class = ' class="best-row"' if is_best_row else ""
        found_icon = "✅" if r.get("found_correct") else "❌"
        result_rows += (
            f'<tr{row_class}>'
            f'<td><span class="pill" style="background:{r.get("color","#888")}">{r["label"]}</span></td>'
            f'<td style="font-weight:bold">{r["success_prob"]:.4f}</td>'
            f'<td style="color:{"#2e7d32" if delta>=0 else "#c62828"}">{sign}{delta:.1f}%</td>'
            f'<td>{found_icon}</td>'
            f'<td>{r["shots"]:,}</td>'
            f'<td>{r["description"]}</td>'
            f'</tr>'
        )

    # ── ZNE details table ──────────────────────────────────────────────
    zne_section = ""
    zne_r = next((r for r in all_results if r["technique"] == "ZNE"), None)
    if zne_r and "scale_factors" in zne_r:
        zne_rows = ""
        for scale, p in zip(zne_r["scale_factors"], zne_r["success_probs_at_scales"]):
            zne_rows += f"<tr><td>λ = {scale}</td><td>{p:.4f}</td><td>measured</td></tr>"
        zne_rows += (
            f'<tr style="background:#fff3e0;font-weight:bold">'
            f'<td>λ = 0 (extrapolated)</td><td>{zne_r["success_prob"]:.4f}</td>'
            f'<td>{zne_r["extrapolation_type"]}</td></tr>'
        )
        zne_section = (
            f'<div class="card"><h2>ZNE Details — Gate Folding & Extrapolation</h2>'
            f'<p>Gate folding amplifies noise by replacing each gate U with U·U&#8224;·U (scale 3), '
            f'U·U&#8224;·U·U&#8224;·U (scale 5), etc. Success probability is measured at each noise level '
            f'then extrapolated to zero noise via {zne_r["extrapolation_type"]} fit.</p>'
            f'<table><thead><tr><th>Noise Scale</th><th>P(target)</th><th>Method</th></tr></thead>'
            f'<tbody>{zne_rows}</tbody></table></div>'
        )

    # ── REM details ────────────────────────────────────────────────────
    rem_section = ""
    rem_r = next((r for r in all_results if r["technique"] == "REM"), None)
    if rem_r and "qubit_readout_errors" in rem_r:
        rem_rows = "".join(
            f"<tr><td>Q{q}</td><td>{e:.4f}</td>"
            f'<td style="color:{"#2e7d32" if e<0.02 else "#e65100" if e<0.05 else "#c62828"}">'
            f'{"Low" if e<0.02 else "Medium" if e<0.05 else "High"}</td></tr>'
            for q, e in enumerate(rem_r["qubit_readout_errors"])
        )
        rem_section = (
            f'<div class="card"><h2>REM Details — Per-Qubit Readout Errors</h2>'
            f'<p>Calibration circuits prepare |0⟩ and |1⟩ states to measure per-qubit readout error rates. '
            f'The inverse of the assignment matrix is applied to correct the Grover measurement distribution.</p>'
            f'<table><thead><tr><th>Qubit</th><th>Error Rate</th><th>Level</th></tr></thead>'
            f'<tbody>{rem_rows}</tbody></table></div>'
        )

    # ── scaling table ──────────────────────────────────────────────────
    scaling_section = ""
    if scaling_results:
        techniques_sc = list(scaling_results.keys())
        sizes_sc = sorted({r["num_qubits"] for rs in scaling_results.values() for r in rs})
        idx_sc = {
            (t, r["num_qubits"]): r
            for t, results in scaling_results.items()
            for r in results
        }
        sc_header = (
            "<tr><th>Qubits</th><th>Search Space</th><th>Speedup</th>"
            + "".join(f"<th>{t}</th>" for t in techniques_sc)
            + "</tr>"
        )
        sc_rows = ""
        for nq in sizes_sc:
            r_b = idx_sc.get(("Baseline", nq), {})
            sc_rows += (
                f"<tr><td>{nq}</td><td>{2**nq}</td><td>{r_b.get('speedup','—')}×</td>"
                + "".join(
                    f"<td>{idx_sc.get((t, nq), {}).get('success_prob', '—'):.4f}</td>"
                    if isinstance(idx_sc.get((t, nq), {}).get("success_prob"), float)
                    else f"<td>—</td>"
                    for t in techniques_sc
                )
                + "</tr>"
            )
        scaling_section = (
            f'<div class="card"><h2>Scaling Analysis</h2>'
            f'<p>Grover circuit executed at multiple qubit counts to show how mitigation benefit '
            f'varies with problem size and circuit depth.</p>'
            f'<table><thead>{sc_header}</thead><tbody>{sc_rows}</tbody></table></div>'
        )

    # ── config badges ──────────────────────────────────────────────────
    def badge(name, enabled):
        cls = "badge-on" if enabled else "badge-off"
        icon = "✅" if enabled else "❌"
        return f'<span class="{cls}">{icon} {name}</span>'

    config_badges = (
        badge("Dynamical Decoupling (DD)", enable_dd)
        + badge("Readout Error Mitigation (REM)", enable_rem)
        + badge("Zero Noise Extrapolation (ZNE)", enable_zne)
        + badge(f"Combined ({'+'.join(active_techs) if active_techs else 'N/A'})", enable_combined and num_enabled >= 2)
        + (f'<br><small style="color:#666;margin-top:6px;display:block">ZNE scale factors: {zne_scale_factors}</small>'
           if enable_zne else "")
    )

    # ── full HTML ──────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<style>
  body{{font-family:Arial,sans-serif;max-width:960px;margin:0 auto;padding:24px;background:#f5f5f5;color:#222}}
  h1{{color:#333;border-bottom:3px solid #5C6BC0;padding-bottom:10px;margin-bottom:6px}}
  h2{{color:#444;margin-top:0;font-size:1.15em}}
  .subtitle{{color:#777;font-size:0.9em;margin-bottom:24px}}
  .card{{background:#fff;border-radius:8px;padding:22px 26px;margin:16px 0;box-shadow:0 2px 6px rgba(0,0,0,0.08)}}
  .metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:8px 0}}
  .metric{{text-align:center;padding:14px 8px;border-radius:6px;background:#f9f9f9}}
  .metric .val{{font-size:1.9em;font-weight:bold;line-height:1.1}}
  .metric .lbl{{font-size:0.8em;color:#777;margin-top:4px}}
  table{{width:100%;border-collapse:collapse;font-size:0.92em}}
  th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid #eee}}
  th{{background:#f4f4f4;font-weight:600}}
  tr:hover{{background:#fafafa}}
  .best-row{{background:#fff8e1}}
  .pill{{display:inline-block;padding:3px 10px;border-radius:12px;color:#fff;font-size:0.82em}}
  .badge-on{{display:inline-block;padding:4px 12px;border-radius:14px;background:#e8f5e9;color:#2e7d32;margin:3px;font-size:0.9em}}
  .badge-off{{display:inline-block;padding:4px 12px;border-radius:14px;background:#f5f5f5;color:#9e9e9e;margin:3px;font-size:0.9em}}
  .desc-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:6px}}
  .desc-card{{border-left:4px solid;padding:10px 14px;border-radius:0 6px 6px 0;background:#fafafa;font-size:0.9em}}
  footer{{text-align:center;color:#aaa;font-size:0.8em;margin-top:32px;padding-top:12px;border-top:1px solid #eee}}
</style>
</head>
<body>
<h1>Grover's Search v2 — Error Mitigation Report</h1>
<div class="subtitle">{now} &nbsp;·&nbsp; IQM Garnet (20-qubit superconducting QPU) &nbsp;·&nbsp; {n} search qubits &nbsp;·&nbsp; Secret key |{secret}⟩</div>

<div class="card">
  <h2>Experiment Summary</h2>
  <div class="metrics">
    <div class="metric" style="border-top:3px solid #5C6BC0">
      <div class="val" style="color:#5C6BC0">|{secret}⟩</div>
      <div class="lbl">Hidden Secret Key</div>
    </div>
    <div class="metric" style="border-top:3px solid #8B8B8B">
      <div class="val" style="color:#8B8B8B">{baseline["success_prob"]:.4f}</div>
      <div class="lbl">Baseline P(target)</div>
    </div>
    <div class="metric" style="border-top:3px solid {best["color"]}">
      <div class="val" style="color:{best["color"]}">{best["success_prob"]:.4f}</div>
      <div class="lbl">Best P(target)<br><small>{best["label"]}</small></div>
    </div>
    <div class="metric" style="border-top:3px solid {"#2e7d32" if improvement>=0 else "#c62828"}">
      <div class="val" style="color:{"#2e7d32" if improvement>=0 else "#c62828"}">{"+" if improvement>=0 else ""}{improvement:.1f}%</div>
      <div class="lbl">Best Improvement<br>over Baseline</div>
    </div>
  </div>
  <div class="metrics" style="grid-template-columns:repeat(4,1fr);margin-top:10px">
    <div class="metric">
      <div class="val">{2**n}</div><div class="lbl">Search Space Size</div>
    </div>
    <div class="metric">
      <div class="val">{problem["grover_iterations"]}</div><div class="lbl">Grover Iterations</div>
    </div>
    <div class="metric">
      <div class="val">{problem["theoretical_success_prob"]:.3f}</div><div class="lbl">Theoretical P(success)</div>
    </div>
    <div class="metric">
      <div class="val">{problem["speedup"]}×</div><div class="lbl">Quantum Speedup (vs worst)</div>
    </div>
  </div>
</div>

<div class="card">
  <h2>Active Configuration</h2>
  {config_badges}
  <div style="margin-top:12px;font-size:0.9em;color:#555">
    <strong>Circuit:</strong>
    {circuit_data["gate_count"]} original gates · {transpile_data["transpiled_gates"]} transpiled gates ·
    depth {transpile_data["transpiled_depth"]} · {problem["grover_iterations"]} Grover iteration(s)
  </div>
</div>

<div class="card">
  <h2>Results by Technique</h2>
  {inline_svg}
  <br/>
  <table>
    <thead><tr>
      <th>Technique</th><th>P(target)</th><th>vs Baseline</th>
      <th>Found?</th><th>Shots</th><th>Description</th>
    </tr></thead>
    <tbody>{result_rows}</tbody>
  </table>
</div>

{zne_section}
{rem_section}

<div class="card">
  <h2>Quantum vs Classical</h2>
  <table>
    <thead><tr><th>Method</th><th>Queries</th><th>Notes</th></tr></thead>
    <tbody>
      <tr><td>Classical (worst case)</td><td>{comparison["classical_worst"]}</td><td>Check every item</td></tr>
      <tr><td>Classical (average)</td><td>{comparison["classical_average"]:.0f}</td><td>Expected for random search</td></tr>
      <tr style="background:#e8f5e9;font-weight:bold">
        <td>Quantum (Grover)</td><td>{comparison["quantum_queries"]}</td>
        <td>{comparison["speedup_worst"]:.0f}× faster than worst · {comparison["speedup_average"]:.0f}× faster than average</td>
      </tr>
    </tbody>
  </table>
</div>

{scaling_section}

<div class="card">
  <h2>Error Mitigation Technique Guide</h2>
  <div class="desc-grid">
    <div class="desc-card" style="border-color:#2196F3">
      <strong style="color:#2196F3">Dynamical Decoupling (DD)</strong><br/>
      Inserts X·X pulse sequences on idle qubits during the barrier between oracle and diffuser.
      Counteracts environmental decoherence. Most effective when the circuit has long idle periods.
    </div>
    <div class="desc-card" style="border-color:#4CAF50">
      <strong style="color:#4CAF50">Readout Error Mitigation (REM)</strong><br/>
      Calibrates with |0⟩<sup>n</sup> and |1⟩<sup>n</sup> circuits to measure per-qubit readout error rates.
      Applies the pseudo-inverse correction matrix to the Grover measurement distribution.
      Addresses the dominant error source on NISQ hardware.
    </div>
    <div class="desc-card" style="border-color:#FF9800">
      <strong style="color:#FF9800">Zero Noise Extrapolation (ZNE)</strong><br/>
      Gate folding amplifies noise by replacing U with U·U&#8224;·U (scale 3), etc.
      Grover success probability is measured at scales {zne_scale_factors}.
      Richardson extrapolation recovers the zero-noise estimate.
    </div>
    <div class="desc-card" style="border-color:#E91E63">
      <strong style="color:#E91E63">Combined (DD + REM + ZNE)</strong><br/>
      All enabled techniques applied together: DD at circuit level → execution at each ZNE scale →
      REM post-measurement correction at each scale → extrapolation to zero noise.
      Combines complementary error suppression strategies.
    </div>
  </div>
</div>

<div class="card">
  <h2>Cryptographic Context</h2>
  <p>Grover's algorithm provides a <strong>quadratic speedup</strong> for unstructured search,
  reducing symmetric-key security from n bits to n/2 effective bits.</p>
  <table>
    <thead><tr><th>Key Length</th><th>Classical Cost</th><th>Quantum Cost (Grover)</th><th>Effective Security</th></tr></thead>
    <tbody>
      <tr><td>AES-128</td><td>2<sup>128</sup></td><td>2<sup>64</sup></td><td style="color:#e65100">64-bit — vulnerable</td></tr>
      <tr><td>AES-256</td><td>2<sup>256</sup></td><td>2<sup>128</sup></td><td style="color:#2e7d32">128-bit — still secure</td></tr>
    </tbody>
  </table>
  <p style="font-size:0.9em;color:#555;margin-top:8px">
    <strong>Current limitation:</strong> This {n}-qubit demo searches {2**n} items.
    Breaking AES-128 would require ~128 logical qubits running 2<sup>64</sup> Grover iterations —
    far beyond current NISQ capabilities. Error mitigation extends the reach of today's hardware
    but does not change this fundamental scale gap.
  </p>
</div>

<footer>
  Pipeline orchestrated by Prefect &nbsp;·&nbsp;
  Grover's Search v2 with DD / REM / ZNE on IQM Garnet &nbsp;·&nbsp;
  {now}
</footer>
</body>
</html>"""

    create_markdown_artifact(
        key="grover-v2-html-report",
        markdown=html,
        description="Rich HTML experiment report: all techniques, results, charts, and cryptographic context",
    )


# ═══════════════════════════════════════════════════════════════════════
# MAIN FLOW
# ═══════════════════════════════════════════════════════════════════════

@flow(
    name="Grover's Search v2 · Error Mitigation · IQM Garnet",
    description=(
        "Grover's Search v2: run baseline + DD + REM + ZNE + Combined on IQM Garnet. "
        "Toggle techniques at run start. Produces per-technique SVG artifacts and "
        "a rich HTML comparison report."
    ),
    log_prints=True,
)
def grover_pipeline_v2(
    num_search_qubits: int = 3,
    shots: int = 4096,
    enable_dd: bool = True,
    enable_rem: bool = True,
    enable_zne: bool = True,
    enable_combined: bool = True,
    zne_scale_factors: list = [1, 3, 5],
    run_scaling: bool = True,
    scaling_sizes: list = [3, 4, 5],
):
    """
    Grover's Search v2 — Error Mitigation Pipeline

    Toggleable error mitigation techniques:
      enable_dd:       Dynamical Decoupling (XX pulses on idle qubits)
      enable_rem:      Readout Error Mitigation (calibration + correction)
      enable_zne:      Zero Noise Extrapolation (gate folding + extrapolation)
      enable_combined: Combined run (DD+REM+ZNE together, requires 2+ techniques)

    Stages:
      1. Problem setup (random secret key)            → CPU
      2. Build Grover circuit                          → CPU
      3. Transpile for IQM Garnet                      → CPU
      4a. Baseline execution                           → QPU
      4b. DD execution (if enabled)                    → QPU
      4c. REM execution (if enabled)                   → QPU ×3 (cal + run)
      4d. ZNE execution (if enabled)                   → QPU × |scales|
      4e. Combined execution (if enabled + 2+ active)  → QPU × |scales| + cal
      5.  Classical comparison                         → CPU
      6.  Scaling analysis with mitigation             → QPU × sizes × techniques
      7.  Artifacts (8 tasks)                          → CPU
    """
    num_enabled = sum([enable_dd, enable_rem, enable_zne])

    active = []
    if enable_dd:
        active.append("DD")
    if enable_rem:
        active.append("REM")
    if enable_zne:
        active.append("ZNE")
    run_combined_flag = enable_combined and num_enabled >= 2

    print(f"\n{'━' * 64}")
    print(f"  Grover's Search v2 · IQM Garnet")
    print(f"  Search qubits: {num_search_qubits}  |  Search space: {2**num_search_qubits} items")
    print(f"  Shots: {shots}  |  Techniques: {', '.join(active) if active else 'NONE (baseline only)'}")
    if enable_zne:
        print(f"  ZNE scale factors: {zne_scale_factors}")
    print(f"  Combined: {'enabled' if run_combined_flag else 'disabled'}")
    print(f"  Scaling: {scaling_sizes if run_scaling else 'disabled'}")
    print(f"{'━' * 64}\n")

    # Stage 1
    print("▸ STAGE 1: Problem Setup")
    problem = setup_problem(num_search_qubits)

    # Stage 2
    print("\n▸ STAGE 2: Build Grover Circuit")
    circuit_data = build_grover_circuit(problem)

    # Stage 3
    print("\n▸ STAGE 3: Transpile")
    transpile_data = transpile_grover(circuit_data)

    # Stage 4: Multi-run
    print("\n▸ STAGE 4a: Baseline (no mitigation)")
    baseline = run_grover_baseline(problem, transpile_data, shots)
    all_results = [baseline]

    dd_result = None
    rem_result = None
    zne_result = None

    if enable_dd:
        print("\n▸ STAGE 4b: Dynamical Decoupling (DD)")
        dd_result = run_grover_with_dd(problem, transpile_data, shots)
        all_results.append(dd_result)

    if enable_rem:
        print("\n▸ STAGE 4c: Readout Error Mitigation (REM)")
        rem_result = run_grover_with_rem(problem, transpile_data, shots)
        all_results.append(rem_result)

    if enable_zne:
        print(f"\n▸ STAGE 4d: Zero Noise Extrapolation (ZNE, scales={zne_scale_factors})")
        zne_result = run_grover_with_zne(problem, transpile_data, shots, zne_scale_factors)
        all_results.append(zne_result)

    if run_combined_flag:
        print(f"\n▸ STAGE 4e: Combined ({'+'.join(active)})")
        combined = run_grover_combined(
            problem, transpile_data, shots,
            enable_dd, enable_rem, enable_zne, zne_scale_factors,
        )
        all_results.append(combined)

    # Stage 5
    print("\n▸ STAGE 5: Classical Comparison")
    comparison = classical_comparison(problem, baseline)

    # Stage 6
    scaling_results = {}
    if run_scaling:
        print(f"\n▸ STAGE 6: Scaling Analysis ({scaling_sizes})")
        scaling_results = run_scaling_with_mitigation(
            scaling_sizes, shots,
            enable_dd, enable_rem, enable_zne, enable_combined, zne_scale_factors,
        )

    # Stage 7: Artifacts
    print("\n▸ STAGE 7: Generating Artifacts")
    publish_success_prob_chart(all_results, problem)
    publish_technique_histograms(all_results, problem)

    if zne_result:
        publish_zne_curve(zne_result)
    if rem_result:
        publish_rem_heatmap(rem_result)

    publish_comparison_chart(problem, comparison)

    if scaling_results:
        publish_scaling_curves(scaling_results)
        publish_scaling_table(scaling_results)

    publish_html_report(
        problem, circuit_data, transpile_data, all_results, comparison, scaling_results,
        enable_dd, enable_rem, enable_zne, enable_combined, zne_scale_factors,
    )

    # Summary
    best = max(all_results, key=lambda r: r["success_prob"])
    improvement = (
        (best["success_prob"] - baseline["success_prob"]) / max(baseline["success_prob"], 0.001) * 100
    )

    print(f"\n{'━' * 64}")
    print(f"  Pipeline complete!")
    print(f"  Secret key: |{problem['secret_key']}⟩")
    print(f"  Baseline P(target): {baseline['success_prob']:.4f}")
    print(f"  Best P(target):     {best['success_prob']:.4f} ({best['label']})")
    print(f"  Improvement:        {'+' if improvement>=0 else ''}{improvement:.1f}% vs baseline")
    print(f"  Quantum speedup:    {comparison['speedup_worst']:.0f}× (vs worst case)")
    print(f"  Artifacts:          Prefect dashboard → Artifacts tab")
    print(f"{'━' * 64}\n")

    return {r["technique"]: r["success_prob"] for r in all_results}


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grover's Search v2 · Error Mitigation · IQM Garnet")
    parser.add_argument("--qubits", type=int, default=3, help="Number of search qubits")
    parser.add_argument("--shots", type=int, default=4096, help="Shots per execution")
    parser.add_argument("--no-dd", action="store_true", help="Disable Dynamical Decoupling")
    parser.add_argument("--no-rem", action="store_true", help="Disable Readout Error Mitigation")
    parser.add_argument("--no-zne", action="store_true", help="Disable Zero Noise Extrapolation")
    parser.add_argument("--no-combined", action="store_true", help="Disable combined run")
    parser.add_argument("--zne-scales", type=str, default="1,3,5", help="ZNE scale factors (comma-separated)")
    parser.add_argument("--no-scaling", action="store_true", help="Disable scaling analysis")
    parser.add_argument("--scaling-sizes", type=str, default="3,4,5", help="Qubit sizes for scaling analysis")
    args = parser.parse_args()

    grover_pipeline_v2(
        num_search_qubits=args.qubits,
        shots=args.shots,
        enable_dd=not args.no_dd,
        enable_rem=not args.no_rem,
        enable_zne=not args.no_zne,
        enable_combined=not args.no_combined,
        zne_scale_factors=[int(s) for s in args.zne_scales.split(",")],
        run_scaling=not args.no_scaling,
        scaling_sizes=[int(s) for s in args.scaling_sizes.split(",")],
    )

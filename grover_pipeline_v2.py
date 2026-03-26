"""
Grover's Search Pipeline v2 · IQM Garnet
==========================================
Extends v1 with configurable error mitigation techniques:
  DD  — Dynamical Decoupling (idle qubit noise suppression)
  REM — Readout Error Mitigation (measurement error correction)
  ZNE — Zero Noise Extrapolation (gate error extrapolation to zero)

Toggle techniques at run start. Runs baseline + each enabled technique
individually + combined mode. Produces per-technique artifacts and a
rich Markdown comparison report.

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
    from iqm.iqm_client.models import CircuitCompilationOptions, DDMode

    backend = get_iqm_backend(token)
    qc = transpile_data["_original"]
    qc_t = qk_transpile(qc, backend=backend, optimization_level=2)

    secret = problem["secret_key"]
    n = problem["num_qubits"]

    # Use IQM's server-side DD — standard strategy auto-selects XX/YXYX/XYXYYXYX
    # based on idle durations (no circuit modification needed)
    dd_options = CircuitCompilationOptions(dd_mode=DDMode.ENABLED)
    logger.info("DD: using IQM server-side standard DD strategy (XX/YXYX/XYXYYXYX)")

    t0 = time.time()
    job = backend.run(qc_t, shots=shots, use_timeslot=False, circuit_compilation_options=dd_options)
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
    from iqm.iqm_client.models import CircuitCompilationOptions, DDMode

    backend = get_iqm_backend(token)
    n = problem["num_qubits"]
    secret = problem["secret_key"]

    qc = transpile_data["_original"]
    qc_t = qk_transpile(qc, backend=backend, optimization_level=2)

    # Step 1: DD is handled server-side via CircuitCompilationOptions at run time
    if enable_dd:
        logger.info("Combined: DD will be applied server-side via IQM API")

    # Step 2: REM calibration
    inv_matrices = None
    qubit_matrices = None
    if enable_rem:
        logger.info("Combined: running REM calibration...")
        qubit_matrices, inv_matrices = _calibrate_rem(backend, n, shots)
        logger.info("Combined: REM calibration done")

    # Step 3: ZNE runs (with DD applied server-side if enabled)
    scales = zne_scale_factors if enable_zne else [1]
    probs_at_scales = []
    last_counts = {}
    run_options = {}
    if enable_dd:
        run_options["circuit_compilation_options"] = CircuitCompilationOptions(dd_mode=DDMode.ENABLED)

    for scale in scales:
        qc_run = _fold_circuit(qc_t, scale)
        job = backend.run(qc_run, shots=shots, use_timeslot=False, **run_options)
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
    from iqm.iqm_client.models import CircuitCompilationOptions, DDMode

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

    dd_options = CircuitCompilationOptions(dd_mode=DDMode.ENABLED)

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

        # Baseline (DD explicitly disabled)
        t0 = time.time()
        job = backend.run(qc_t, shots=shots, use_timeslot=False,
                          circuit_compilation_options=CircuitCompilationOptions(dd_mode=DDMode.DISABLED))
        counts = job.result().get_counts()
        p = _grover_success_prob(counts, secret, shots)
        logger.info(f"  Baseline P(target)={p:.4f} ({round(time.time()-t0,1)}s)")
        _record("Baseline", counts, p)

        # DD (server-side via IQM API)
        if enable_dd:
            t0 = time.time()
            counts_dd = backend.run(qc_t, shots=shots, use_timeslot=False,
                                    circuit_compilation_options=dd_options).result().get_counts()
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

        # Combined (DD via server-side + REM + ZNE)
        if run_combined_flag:
            inv_c = None
            if enable_rem:
                _, inv_c = _calibrate_rem(backend, n, shots)
            scales_c = zne_scale_factors if enable_zne else [1]
            run_opts_c = {}
            if enable_dd:
                run_opts_c["circuit_compilation_options"] = dd_options
            probs_c = []
            for scale in scales_c:
                qc_sc = _fold_circuit(qc_t, scale)
                c = backend.run(qc_sc, shots=shots, use_timeslot=False, **run_opts_c).result().get_counts()
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


@task(name="7.8 · Markdown Experiment Report", tags=["stage:7", "reporting"])
def publish_markdown_report(
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
    """Clean Markdown experiment report with detailed explanations."""

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

    # ── Configuration badges ────────────────────────────────────────
    config_lines = ""
    config_lines += f"| Dynamical Decoupling (DD) | {'✅ Enabled' if enable_dd else '❌ Disabled'} |\n"
    config_lines += f"| Readout Error Mitigation (REM) | {'✅ Enabled' if enable_rem else '❌ Disabled'} |\n"
    config_lines += f"| Zero Noise Extrapolation (ZNE) | {'✅ Enabled' if enable_zne else '❌ Disabled'} |\n"
    run_combined_flag = enable_combined and num_enabled >= 2
    config_lines += f"| Combined ({'+'.join(active_techs) if active_techs else 'N/A'}) | {'✅ Enabled' if run_combined_flag else '❌ Disabled'} |"
    if enable_zne:
        config_lines += f"\n| ZNE Scale Factors | {zne_scale_factors} |"

    # ── Results table rows ──────────────────────────────────────────
    result_rows = ""
    for r in all_results:
        delta = (r["success_prob"] - baseline["success_prob"]) / max(baseline["success_prob"], 0.001) * 100
        sign = "+" if delta >= 0 else ""
        found_icon = "✅" if r.get("found_correct") else "❌"
        best_marker = " **← BEST**" if r["technique"] == best["technique"] and len(all_results) > 1 else ""
        result_rows += (
            f"| {r['label']}{best_marker} | {r['success_prob']:.4f} | "
            f"{sign}{delta:.1f}% | {found_icon} | {r['shots']:,} |\n"
        )

    # ── ZNE details ─────────────────────────────────────────────────
    zne_section = ""
    zne_r = next((r for r in all_results if r["technique"] == "ZNE"), None)
    if zne_r and "scale_factors" in zne_r:
        zne_rows = ""
        for scale, p in zip(zne_r["scale_factors"], zne_r["success_probs_at_scales"]):
            zne_rows += f"| λ = {scale} | {p:.4f} | Measured on QPU |\n"
        zne_rows += f"| **λ = 0 (extrapolated)** | **{zne_r['success_prob']:.4f}** | **{zne_r['extrapolation_type']} fit** |"

        zne_section = f"""
---

## ZNE Details — Gate Folding & Extrapolation

**How ZNE works:** Zero Noise Extrapolation deliberately makes the circuit *noisier* by
repeating each gate operation (gate folding: U → U·U†·U for scale 3, U·U†·U·U†·U for scale 5, etc.).
The idea is: if we measure how fast the success probability drops as we add more noise,
we can mathematically project backwards to estimate what the result would have been with *zero* noise.

**How to read this table:** Each row shows P(target) — the probability that the QPU
returned the correct secret key. At λ=1 the circuit runs normally. At λ=3 and λ=5,
extra noise is injected. The extrapolated λ=0 row is the ZNE estimate of the ideal,
noise-free success probability.

**Why P(target) may drop to 0.0000 at higher scales:** When noise is amplified too much
(e.g. λ=3 or λ=5 on a deep circuit), the QPU output becomes essentially random — every
bitstring is equally likely. With {2**n} possible outcomes, random guessing gives
P = 1/{2**n} = {1/(2**n):.4f}, which rounds to 0.0000. This is expected behavior and is
actually useful: it tells ZNE that noise completely destroys the signal, helping the
extrapolation estimate how much signal was present before noise.

| Noise Scale (λ) | P(target) | Method |
|-----------------|-----------|--------|
{zne_rows}

> **Interpretation:** The extrapolated value of **{zne_r['success_prob']:.4f}** is ZNE's best
> estimate of what P(target) would be on a perfect, noiseless quantum computer. When this
> value is higher than the baseline, ZNE has successfully recovered some of the signal lost to noise.
> When measured values at λ=1 are already very low (circuit too deep for the hardware), the
> extrapolation becomes less reliable — it's working from a very weak signal.
"""

    # ── REM details ─────────────────────────────────────────────────
    rem_section = ""
    rem_r = next((r for r in all_results if r["technique"] == "REM"), None)
    if rem_r and "qubit_readout_errors" in rem_r:
        rem_rows = ""
        for q, e in enumerate(rem_r["qubit_readout_errors"]):
            level = "🟢 Low" if e < 0.02 else "🟡 Medium" if e < 0.05 else "🔴 High"
            rem_rows += f"| Q{q} | {e:.4f} | {level} |\n"

        rem_section = f"""
---

## REM Details — Per-Qubit Readout Errors

**How REM works:** Before running the actual Grover circuit, we run two calibration circuits:
one that prepares all qubits in |0⟩ and one that prepares all qubits in |1⟩. We then measure
both circuits many times. In a perfect QPU, preparing |0⟩ and measuring would always give 0 —
but real hardware sometimes reads 0 as 1, or 1 as 0. These are called **readout errors**.

By comparing what we prepared vs what we measured, we build a per-qubit error profile
(the "assignment matrix"). We then mathematically invert this matrix and apply it to our
Grover measurement results, effectively undoing the readout mistakes.

**How to read this table:** The "Error Rate" is the average probability that a qubit's
measurement is flipped (averaged over 0→1 and 1→0 flips). Lower is better.
On IQM Garnet, typical readout errors are 1-5%.

| Qubit | Error Rate | Level |
|-------|-----------|-------|
{rem_rows}

> **What this means for your results:** REM corrects the *measurement* step only — it fixes
> the "camera" but not the "scene". If the circuit itself accumulated gate errors, REM won't
> fix those. That's why REM works well combined with DD (which fixes gate-level idle noise)
> and ZNE (which extrapolates away gate noise).
"""

    # ── Scaling section ─────────────────────────────────────────────
    scaling_section = ""
    if scaling_results:
        techniques_sc = list(scaling_results.keys())
        sizes_sc = sorted({r["num_qubits"] for rs in scaling_results.values() for r in rs})
        idx_sc = {
            (t, r["num_qubits"]): r
            for t, results in scaling_results.items()
            for r in results
        }

        sc_header = "| Qubits | Search Space | Speedup |"
        sc_sep = "|--------|-------------|---------|"
        for t in techniques_sc:
            sc_header += f" {t} |"
            sc_sep += "--------|"

        sc_rows = ""
        for nq in sizes_sc:
            r_b = idx_sc.get(("Baseline", nq), {})
            row = f"| {nq} | {2**nq} | {r_b.get('speedup','—')}× |"
            for t in techniques_sc:
                r = idx_sc.get((t, nq), {})
                p = r.get("success_prob", "—")
                row += f" {p:.4f} |" if isinstance(p, float) else f" — |"
            sc_rows += row + "\n"

        scaling_section = f"""
---

## Scaling Analysis

How does each technique perform as the problem gets harder (more qubits = larger search space = deeper circuit = more noise)?

{sc_header}
{sc_sep}
{sc_rows}
> As qubit count increases, the circuit becomes deeper and accumulates more noise.
> Mitigation techniques that work well at 3 qubits may show different behavior at 5 qubits.
> REM tends to be the most consistent since readout errors are independent of circuit depth.
"""

    # ── Full Markdown report ────────────────────────────────────────
    md = f"""# Grover's Search v2 — Error Mitigation Experiment Report

**Date:** {now} | **Backend:** IQM Garnet (20-qubit superconducting QPU) | **Search qubits:** {n}

---

## Experiment Summary

**What we did:** We hid a random secret key (|{secret}⟩) in a search space of **{2**n} items**
({n} qubits). Grover's quantum search algorithm was used to find this key on real quantum
hardware (IQM Garnet). We then applied various error mitigation techniques to see how much
they improve the result compared to running the raw (unmitigated) circuit.

**How to read the key metrics below:**

| Metric | Value | What It Means |
|--------|-------|---------------|
| Hidden Secret Key | |{secret}⟩ | The {n}-bit string we're searching for. This was randomly chosen at the start. The quantum computer doesn't know it — it has to find it using Grover's algorithm. |
| Baseline P(target) | {baseline['success_prob']:.4f} | The probability of measuring the correct answer with **no error mitigation** — just the raw noisy QPU. On a perfect quantum computer this would be ~{problem['theoretical_success_prob']:.3f}. The gap between {baseline['success_prob']:.4f} and {problem['theoretical_success_prob']:.3f} is caused by hardware noise (gate errors, decoherence, readout errors). |
| Best P(target) | {best['success_prob']:.4f} ({best['label']}) | The highest success probability achieved after applying error mitigation. Higher = better. |
| Best Improvement | {"+" if improvement >= 0 else ""}{improvement:.1f}% | How much the best technique improved over the raw baseline. A positive number means mitigation helped. |
| Search Space Size | {2**n} | Total number of possible answers (2^{n}). A classical computer would need to check up to all {2**n}. |
| Grover Iterations | {problem['grover_iterations']} | How many times the oracle+diffuser cycle runs. Optimal count = ⌊π/4 · √N⌋ = {problem['grover_iterations']}. |
| Theoretical P(success) | {problem['theoretical_success_prob']:.3f} | The success probability on a *perfect* quantum computer. This is the ceiling we're trying to approach. |
| Quantum Speedup | {problem['speedup']}× | Grover needs only {problem['grover_iterations']} queries vs {2**n} (worst case classical). That's a {problem['speedup']}× speedup. |

---

## Active Configuration

| Technique | Status |
|-----------|--------|
{config_lines}

**Circuit details:** {circuit_data['gate_count']} original gates → {transpile_data['transpiled_gates']} transpiled gates (depth {transpile_data['transpiled_depth']}) after optimization for IQM Garnet's native gate set (r, cz, id).

---

## Results by Technique

| Technique | P(target) | vs Baseline | Found Correct? | Total Shots |
|-----------|-----------|-------------|----------------|-------------|
{result_rows}

> **P(target)** = probability of measuring the secret key |{secret}⟩ when we run the circuit.
> Higher is better. "Found Correct?" = ✅ if the most frequently measured bitstring *was* the secret key.
> "vs Baseline" shows the percentage improvement that each mitigation technique provides
> over the raw, unmitigated circuit.
{zne_section}
{rem_section}

---

## Quantum vs Classical Comparison

| Method | Queries Needed | Notes |
|--------|---------------|-------|
| Classical (worst case) | {comparison['classical_worst']} | Must check every item in the search space |
| Classical (average) | {comparison['classical_average']:.0f} | Expected queries for random search |
| **Quantum (Grover)** | **{comparison['quantum_queries']}** | **{comparison['speedup_worst']:.0f}× faster** than worst case, **{comparison['speedup_average']:.0f}× faster** than average |

> Grover's algorithm achieves a *quadratic* speedup: instead of checking N items one by one,
> it finds the answer in ~√N steps. For {2**n} items, that's {comparison['quantum_queries']} quantum queries
> instead of {comparison['classical_worst']} classical checks.
{scaling_section}

---

## Error Mitigation Technique Guide

### Dynamical Decoupling (DD)
**What it does:** Tells IQM's server-side compiler to automatically insert DD pulse sequences
(XX, YXYX, or XYXYYXYX) on qubits that are sitting idle during the circuit. The compiler
knows the exact gate timings and picks the best sequence for each idle slot — short idles
get XX (two π-pulses), longer idles get YXYX or XYXYYXYX. These pulses refocus the qubit's
state and cancel accumulated phase errors (Z and ZZ errors) caused by decoherence and
residual interactions. The circuit itself is not modified — DD is applied at the hardware
compilation level via `CircuitCompilationOptions(dd_mode=DDMode.ENABLED)`.

**When it helps most:** Circuits with long idle periods between operations. The GHZ/Grover
barrier gaps between oracle and diffuser are ideal targets.

### Readout Error Mitigation (REM)
**What it does:** Measures how often the QPU misreads each qubit (e.g., a qubit in state |0⟩
gets read as 1). It does this by running calibration circuits, then mathematically corrects
the Grover measurement results using the inverse of the error profile.

**When it helps most:** Always helpful — readout errors are typically the single largest
error source on NISQ hardware (1-5% per qubit on IQM Garnet). REM corrects measurement
mistakes but cannot fix errors that happened *during* the circuit.

### Zero Noise Extrapolation (ZNE)
**What it does:** Deliberately amplifies the circuit noise by repeating gates (gate folding),
measures the success probability at several noise levels, then fits a curve and extrapolates
backwards to estimate the zero-noise result.

**When it helps most:** When noise scales predictably with gate count. Works best on moderate-depth
circuits. Very deep circuits may already be fully randomized at λ=1, leaving ZNE little signal to work with.

### Combined (DD + REM + ZNE)
**What it does:** Applies all enabled techniques together in sequence: DD at circuit level →
execute at each ZNE scale → REM correction post-measurement → extrapolation to zero noise.
Combines complementary strategies: DD reduces gate-level noise, REM fixes measurement errors,
ZNE extrapolates away residual gate noise.

---

## Cryptographic Context

Grover's algorithm provides a **quadratic speedup** for unstructured key search:

| Key Length | Classical Cost | Quantum Cost (Grover) | Effective Security |
|------------|---------------|----------------------|--------------------|
| AES-128 | 2^128 | 2^64 | 64-bit — at risk in quantum era |
| AES-256 | 2^256 | 2^128 | 128-bit — still secure |

**Current limitation:** This {n}-qubit demo searches {2**n} items. Breaking AES-128 would require
~128 logical qubits running 2^64 Grover iterations — far beyond current NISQ capabilities.
Error mitigation improves today's hardware fidelity but does not bridge this fundamental scale gap.
The purpose of this experiment is to benchmark how well current error mitigation techniques
preserve quantum advantage on near-term hardware.

---

*Pipeline orchestrated by Prefect · Real QPU execution on IQM Garnet (20-qubit superconducting) · Error mitigation with Qiskit 2.x · {now}*
"""

    create_markdown_artifact(
        key="grover-v2-experiment-report",
        markdown=md,
        description="Comprehensive Markdown experiment report with detailed explanations of all metrics and techniques",
    )


# ═══════════════════════════════════════════════════════════════════════
# MAIN FLOW
# ═══════════════════════════════════════════════════════════════════════

@flow(
    name="Grover's Search v2 · Error Mitigation · IQM Garnet",
    description=(
        "Grover's Search v2: run baseline + DD + REM + ZNE + Combined on IQM Garnet. "
        "Toggle techniques at run start. Produces per-technique SVG artifacts and "
        "a rich Markdown comparison report."
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

    publish_markdown_report(
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
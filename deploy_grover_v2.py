"""
deploy_grover_v2.py — Prefect Cloud Deployment
================================================
Registers Grover's Search v2 (with error mitigation) to Prefect Cloud.

Usage:
    python deploy_grover_v2.py
"""

from prefect import serve
from prefect.deployments import Deployment
from prefect.infrastructure.container import DockerContainer
from prefect.runner.storage import GitRepository
from prefect.blocks.system import Secret

from grover_pipeline_v2 import grover_pipeline_v2


deployment = Deployment.build_from_flow(
    flow=grover_pipeline_v2,
    name="grover-v2-error-mitigation",
    version="2.0.0",
    description=(
        "Grover's Search v2 on IQM Garnet. "
        "Toggleable error mitigation: DD, REM, ZNE, Combined. "
        "Produces per-technique SVG artifacts and a rich HTML comparison report."
    ),
    tags=["quantum", "grovers-search", "v2", "error-mitigation", "iqm-garnet", "dd", "rem", "zne"],

    # ── Source ─────────────────────────────────────────────────────────
    storage=GitRepository(
        url="https://github.com/PlayfulDevBit/Groover_piepline_v2.git",
        branch="main",
    ),

    # ── Infrastructure ─────────────────────────────────────────────────
    job_variables={
        "pip_packages": [
            "qiskit>=2.0",
            "qiskit-iqm",
            "iqm-client",
            "matplotlib",
            "numpy>=1.24",
        ],
    },

    # ── Default Parameters ─────────────────────────────────────────────
    parameters={
        "num_search_qubits": 3,
        "shots": 4096,
        "enable_dd": True,
        "enable_rem": True,
        "enable_zne": True,
        "enable_combined": True,
        "zne_scale_factors": [1, 3, 5],
        "run_scaling": True,
        "scaling_sizes": [3, 4, 5],
    },
)


if __name__ == "__main__":
    deployment_id = deployment.apply()
    print(f"\nDeployment registered: {deployment_id}")
    print("\nParameter toggles available in Prefect Cloud UI:")
    print("  num_search_qubits  int        — number of qubits for main Grover run")
    print("  shots              int        — shots per QPU execution")
    print("  enable_dd          bool       — toggle Dynamical Decoupling")
    print("  enable_rem         bool       — toggle Readout Error Mitigation")
    print("  enable_zne         bool       — toggle Zero Noise Extrapolation")
    print("  enable_combined    bool       — toggle combined (DD+REM+ZNE) run")
    print("  zne_scale_factors  list[int]  — noise amplification levels for ZNE")
    print("  run_scaling        bool       — toggle scaling analysis")
    print("  scaling_sizes      list[int]  — qubit sizes for scaling sweep")
    print("\nPrerequisites:")
    print("  - Prefect Cloud account with PREFECT_API_URL and PREFECT_API_KEY set")
    print("  - Prefect Secret block 'iqm-resonance-token' containing IQM Resonance API key")
    print("  - Code pushed to: https://github.com/PlayfulDevBit/Groover_pipeline_v2.git")
    print("\nExample runs:")
    print("  Full run (all techniques):    defaults as-is")
    print("  Baseline only:                enable_dd=False, enable_rem=False, enable_zne=False")
    print("  DD + REM only:                enable_zne=False")
    print("  ZNE with more scales:         zne_scale_factors=[1,3,5,7]")
    print("  Larger problem:               num_search_qubits=4, scaling_sizes=[4,5,6]")

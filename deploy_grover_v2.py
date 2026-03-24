"""
deploy_grover_v2.py — Prefect 3.x Cloud Deployment
====================================================
Registers Grover's Search v2 (with error mitigation) to Prefect Cloud.

BEFORE RUNNING:
  1. Set WORK_POOL_NAME below to your Prefect Cloud work pool name.
     Find it at: Prefect Cloud UI → Work Pools
     If you don't have one, create a managed process pool in the UI first.

  2. Push the pipeline code to GitHub:
       git push origin main

  3. Ensure your IQM token is stored as a Prefect Secret:
       from prefect.blocks.system import Secret
       Secret(value="your-iqm-key").save("iqm-resonance-token")

Usage:
    python deploy_grover_v2.py
"""

from prefect.runner.storage import GitRepository
from grover_pipeline_v2 import grover_pipeline_v2

# ── CONFIGURE THIS ──────────────────────────────────────────────────────
WORK_POOL_NAME = "my-managed-pool"   # ← replace with your work pool name
# ────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    deployment = grover_pipeline_v2.from_source(
        source=GitRepository(
            url="https://github.com/PlayfulDevBit/Groover_pipeline_v2.git",
            branch="main",
        ),
        entrypoint="grover_pipeline_v2.py:grover_pipeline_v2",
    )

    deployment_id = deployment.deploy(
        name="grover-v2-error-mitigation",
        work_pool_name=WORK_POOL_NAME,
        version="2.0.0",
        description=(
            "Grover's Search v2 on IQM Garnet. "
            "Toggleable error mitigation: DD, REM, ZNE, Combined. "
            "Produces per-technique SVG artifacts and a rich HTML comparison report."
        ),
        tags=["quantum", "grovers-search", "v2", "error-mitigation", "iqm-garnet", "dd", "rem", "zne"],

        # ── Default Parameters (all overridable from Prefect Cloud UI) ──
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

        # ── Dependencies installed in the runner environment ─────────────
        job_variables={
            "pip_packages": [
                "qiskit>=2.0",
                "qiskit-iqm",
                "iqm-client",
                "numpy>=1.24",
            ],
        },
    )

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

import os
import subprocess
import sys


def test_iterative_native_fit_smoke(tmp_path):
    script = os.path.join("scripts", "iterative_native_fit.py")
    out_dir = tmp_path / "campaign"
    cmd = [
        sys.executable,
        script,
        "--mode",
        "full_campaign",  # full N1→N10→(Nall) path; n1_only is the script default
        "--data-source",
        "synthetic",
        "--allow-synthetic",
        "--max-iters",
        "1",
        "--nall",
        "10",
        "--output-dir",
        str(out_dir),
    ]
    p = subprocess.run(cmd, check=False, capture_output=True, text=True)
    # Either success (0) or partial (2) is valid for smoke.
    assert p.returncode in (0, 2), p.stderr
    assert (out_dir / "campaign_summary.json").exists()
    assert (out_dir / "final_run_summary.md").exists()


def test_iterative_native_fit_n1_only_smoke(tmp_path):
    script = os.path.join("scripts", "iterative_native_fit.py")
    out_dir = tmp_path / "campaign_n1"
    cmd = [
        sys.executable,
        script,
        "--mode",
        "n1_only",
        "--data-source",
        "synthetic",
        "--allow-synthetic",
        "--max-iters",
        "1",
        "--output-dir",
        str(out_dir),
    ]
    p = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert (out_dir / "campaign_summary.json").exists()
    assert "N1 only" in p.stdout


def test_iterative_native_fit_n2_only_smoke(tmp_path):
    script = os.path.join("scripts", "iterative_native_fit.py")
    out_dir = tmp_path / "campaign_n2"
    cmd = [
        sys.executable,
        script,
        "--mode",
        "n2_only",
        "--data-source",
        "synthetic",
        "--allow-synthetic",
        "--max-iters",
        "1",
        "--output-dir",
        str(out_dir),
    ]
    p = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert (out_dir / "campaign_summary.json").exists()
    assert "N2 only" in p.stdout

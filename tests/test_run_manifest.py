import json

from src import run_manifest


def test_write_run_manifest_json(tmp_path, monkeypatch):
    monkeypatch.setattr(run_manifest, "_git_commit_id", lambda: "deadbeef")
    monkeypatch.setattr(run_manifest, "_git_dirty", lambda: False)
    inp = ["/fake/a_cbcd.fits", "/fake/b_cbcd.fits"]
    dest = tmp_path / "run_manifest.json"
    written = run_manifest.write_run_manifest(dest, inp, extra={"ok": True})
    assert written == str(dest.resolve())
    data = json.loads(dest.read_text(encoding="utf-8"))
    assert data["schema"] == "spitzer_photometry.run_manifest.v1"
    assert data["git_commit"] == "deadbeef"
    assert sorted(data["input_cbcd_paths"]) == sorted(inp)
    assert data["extra"]["ok"] is True
    assert "CHANNEL" in data["config"]

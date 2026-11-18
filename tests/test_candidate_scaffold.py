from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]


def test_required_candidate_files_exist():
    for rel in (
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "AI_USE_DISCLOSURE.md",
        "LICENSE_STATUS.md",
        "requirements.txt",
        "pyproject.toml",
    ):
        assert (CANDIDATE_ROOT / rel).is_file(), rel


def test_no_license_was_selected():
    assert not (CANDIDATE_ROOT / "LICENSE").exists()


def test_no_private_material_in_candidate_text():
    home = str(Path.home()).encode()
    secrets = (b"api_token", b"password", b"private key")
    for path in CANDIDATE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        relative = path.relative_to(CANDIDATE_ROOT)
        if relative.parts[0] == "tests":
            continue
        payload = path.read_bytes()
        assert home not in payload, f"home path in {relative}"
        for token in secrets:
            assert token not in payload.lower(), f"{token!r} in {relative}"

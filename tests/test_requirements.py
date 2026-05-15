from pathlib import Path


def test_main_requirements_include_mediapipe_for_supported_python() -> None:
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    content = requirements.read_text(encoding="utf-8")

    assert 'mediapipe==0.10.18; python_version < "3.13"' in content

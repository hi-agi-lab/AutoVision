# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""C++ reference example: host postprocess test builds and passes.

The full ONNX Runtime example needs ORT/OpenCV development libraries that
are not part of this repo; the host-only test proves the decode contract.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
CPP = ROOT / "examples" / "cpp"


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not installed")
def test_cpp_postprocess_host_test(tmp_path):
    binary = tmp_path / "postprocess_test"
    subprocess.run(
        [
            "g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            f"-I{CPP / 'src'}",
            str(CPP / "src" / "postprocess_test.cpp"),
            "-o", str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    result = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=30
    )
    assert "postprocess ok" in result.stdout


def test_cpp_example_files_are_wired():
    main = (CPP / "src" / "main.cpp").read_text()
    header = (CPP / "src" / "postprocess.hpp").read_text()
    cmake = (CPP / "CMakeLists.txt").read_text()
    readme = (CPP / "README.md").read_text()
    assert "onnxruntime_cxx_api.h" in main
    assert "opencv2/opencv.hpp" in main
    assert "--max-detections" in main
    assert "GetOutputCount" in main
    assert "std::stable_sort" in header
    assert "ONNXRUNTIME_ROOT" in cmake
    assert "postprocess_test" in cmake
    assert "add_test(" in cmake
    assert "autovision_infer" in readme
    assert "ctest --output-on-failure" in readme
    assert "--max-detections" in readme
    assert (CPP / "src" / "postprocess.hpp").is_file()
    manifest = (ROOT / "MANIFEST.in").read_text()
    assert "examples/cpp" in manifest

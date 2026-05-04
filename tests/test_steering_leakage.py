"""Tests ensuring steering direction extraction does not use test data.

The X_val parameter should be used for direction extraction instead of
X_test to avoid data leakage.
"""

import glob
import inspect
import re
from pathlib import Path

import pytest


class TestSteeringLeakageGuard:
    """Verify that extract_direction supports X_val parameter."""

    def test_tabpfn_steering_has_x_val_param(self):
        from src.hooks.steering_vector import TabPFNSteeringVector

        sig = inspect.signature(TabPFNSteeringVector.extract_direction)
        assert "X_val" in sig.parameters, (
            "TabPFNSteeringVector.extract_direction must accept X_val parameter"
        )

    def test_tabicl_steering_has_x_val_param(self):
        from src.hooks.tabicl_steering import TabICLSteeringVector

        sig = inspect.signature(TabICLSteeringVector.extract_direction)
        assert "X_val" in sig.parameters, (
            "TabICLSteeringVector.extract_direction must accept X_val parameter"
        )

    def test_tabpfn_x_val_is_keyword_only(self):
        from src.hooks.steering_vector import TabPFNSteeringVector

        sig = inspect.signature(TabPFNSteeringVector.extract_direction)
        param = sig.parameters["X_val"]
        assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
            "X_val should be keyword-only to prevent accidental positional use"
        )

    def test_tabicl_x_val_is_keyword_only(self):
        from src.hooks.tabicl_steering import TabICLSteeringVector

        sig = inspect.signature(TabICLSteeringVector.extract_direction)
        param = sig.parameters["X_val"]
        assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
            "X_val should be keyword-only to prevent accidental positional use"
        )


class TestSteeringCallSitesUseXVal:
    """Static guard: every call to ``extract_direction`` in experiments/
    must pass ``X_val=`` to prevent silent test-set leakage.
    """

    @staticmethod
    def _find_call_sites(root: Path) -> list[tuple[Path, int, str]]:
        """Return (file, line_number, full_call) for each extract_direction(...) call."""
        call_sites: list[tuple[Path, int, str]] = []
        for path in sorted(root.glob("*steering*.py")) + sorted(
            root.glob("rd*_*.py")
        ):
            text = path.read_text()
            # Find each "extract_direction(" line and capture until the matching ")".
            for match in re.finditer(r"\.extract_direction\s*\(", text):
                start = match.start()
                # Walk forward to find balanced closing paren.
                depth = 0
                i = match.end() - 1
                while i < len(text):
                    if text[i] == "(":
                        depth += 1
                    elif text[i] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    i += 1
                call_text = text[start : i + 1]
                line_no = text.count("\n", 0, start) + 1
                call_sites.append((path, line_no, call_text))
        return call_sites

    def test_all_call_sites_pass_x_val(self):
        root = Path(__file__).resolve().parent.parent / "experiments"
        if not root.is_dir():
            pytest.skip("experiments/ directory not present")
        offenders = []
        for path, line_no, call_text in self._find_call_sites(root):
            if "X_val=" not in call_text:
                offenders.append(f"{path.name}:{line_no} -> {call_text[:80]}...")
        assert not offenders, (
            "Every extract_direction(...) call in experiments/ must pass "
            "X_val= explicitly to prevent silent test-set leakage. Offenders:\n"
            + "\n".join(offenders)
        )

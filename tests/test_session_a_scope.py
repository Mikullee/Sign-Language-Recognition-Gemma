from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "superpowers"
    / "specs"
    / "2026-08-26-knee42-v13.1-runnable-release-design.md"
)
PLAN = (
    ROOT
    / "docs"
    / "superpowers"
    / "plans"
    / "2026-08-26-knee42-v13.1-runnable-release.md"
)


class SessionAScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = {
            path: path.read_text(encoding="utf-8") for path in (DESIGN, PLAN)
        }

    def test_documents_exclude_personal_and_private_reproduction_references(self):
        forbidden = (
            r"C:\Users\User",
            "Knee42-Private-Reproduction-Data",
            "scripts/reproduce.ps1",
            "K42_01",
            "K42_35",
            "K42_36",
            "K42_39",
        )
        for path, text in self.documents.items():
            with self.subTest(path=path):
                for value in forbidden:
                    self.assertNotIn(value, text)
                self.assertIsNone(
                    re.search(
                        r"^#{2,4}\s+.*\bprivate\b.*\breproduction\b",
                        text,
                        flags=re.IGNORECASE | re.MULTILINE,
                    )
                )

    def test_documents_define_session_a_and_session_c_boundaries(self):
        for path, text in self.documents.items():
            with self.subTest(path=path):
                self.assertIn("Session A scope", text)
                self.assertIn(
                    "User-confirmed Phase 3 data decisions belong to Session B and "
                    "are not modified or revalidated by Session A.",
                    text,
                )
                self.assertRegex(
                    text,
                    re.compile(
                        r"Session C supplies a 42-label \[1,\s*64,\s*438\] model "
                        r"component directory plus a caller-trusted component manifest "
                        r"SHA-256\."
                    ),
                )
                self.assertIn(
                    "Session A verifies component-independent behavior with "
                    "deterministic test fixtures; Session C performs the first build "
                    "with the supplied release component.",
                    text,
                )

    def test_session_a_is_not_assigned_phase_3_data_or_private_tag_work(self):
        forbidden_assignments = (
            re.compile(
                r"Session A\s+(?:must|will|shall|should|is assigned to|is responsible "
                r"to)\s+(?:train|retrain|edit)\b.*\bPhase 3\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"Session A\s+(?:must|will|shall|should|is assigned to|is responsible "
                r"to)\s+create\b.*\bprivate\b.*\btag\b",
                re.IGNORECASE,
            ),
        )
        for path, text in self.documents.items():
            with self.subTest(path=path):
                for pattern in forbidden_assignments:
                    self.assertIsNone(pattern.search(text))

    def test_plan_has_component_manifest_and_concrete_build_handoff(self):
        plan = self.documents[PLAN]
        for variable in (
            "$env:KNEE42_WORKSPACE",
            "$env:KNEE42_ASSET_CACHE",
            "$env:KNEE42_BUILD_ROOT",
            "$env:KNEE42_PYTHON",
            "$env:KNEE42_MODEL_COMPONENT_DIR",
            "$env:KNEE42_MODEL_COMPONENT_MANIFEST_SHA256",
        ):
            self.assertIn(variable, plan)
        self.assertIn(
            "& $env:KNEE42_PYTHON -B scripts/build_knee42_release.py", plan
        )
        handoff_blocks = [
            block
            for block in re.findall(
                r"```powershell\s+(.*?)```", plan, flags=re.DOTALL
            )
            if "& $env:KNEE42_PYTHON -B scripts/build_knee42_release.py"
            in block
        ]
        required_arguments = (
            "--asset-cache $env:KNEE42_ASSET_CACHE",
            "--model-component-dir $env:KNEE42_MODEL_COMPONENT_DIR",
            "--model-component-manifest-sha256 "
            "$env:KNEE42_MODEL_COMPONENT_MANIFEST_SHA256",
            "--output-dir $env:KNEE42_BUILD_ROOT",
            "--offline",
        )
        self.assertTrue(
            any(
                all(argument in block for argument in required_arguments)
                for block in handoff_blocks
            ),
            "one PowerShell handoff block must contain every required builder input",
        )
        self.assertIn("Confirm the local branch and commit are clean", plan)
        self.assertIn(
            "do not push, open a PR, create a tag, or publish a release from Session A",
            plan,
        )
        self.assertIsNone(re.search(r"^\s*git\s+(?:push|tag)\b", plan, re.MULTILINE))
        self.assertIsNone(re.search(r"^\s*gh\s+pr\b", plan, re.MULTILINE))


if __name__ == "__main__":
    unittest.main()

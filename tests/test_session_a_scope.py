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

FENCED_BLOCK = re.compile(
    r"```(?P<language>[^\r\n]*)\r?\n(?P<body>.*?)```", flags=re.DOTALL
)
EXECUTABLE_FENCE_LANGUAGES = {
    "bash",
    "bat",
    "batch",
    "cmd",
    "powershell",
    "pwsh",
    "sh",
    "shell",
}
BARE_PYTHON_COMMAND = re.compile(
    r"^\s*(?:&\s*)?(?:python(?:3(?:\.10)?)?(?:\.exe)?|py\s+-3\.10)"
    r"(?=\s|$)",
    flags=re.IGNORECASE | re.MULTILINE,
)
INLINE_BARE_PYTHON_COMMAND = re.compile(
    r"^\s*(?:&\s*)?(?:python(?:3(?:\.10)?)?(?:\.exe)?|py\s+-3\.10)\s+",
    flags=re.IGNORECASE,
)
ANGLE_PLACEHOLDER = re.compile(r"<[^<>\r\n]+>")
OUTPUT_COMMAND = re.compile(
    r"(?:--output(?:-dir)?|--artifact-root|-OutputDir)\b",
    flags=re.IGNORECASE,
)
OUTPUT_LITERAL_BUILD = re.compile(
    r"(?:--output(?:-dir)?|--artifact-root|-OutputDir)\b"
    r"(?:(?!\r?\n\s*(?:--|-\w)).){0,240}?\bbuild[\\/]",
    flags=re.IGNORECASE | re.DOTALL,
)


def executable_fences(text: str) -> list[tuple[int, str]]:
    return [
        (match.start(), match.group("body"))
        for match in FENCED_BLOCK.finditer(text)
        if match.group("language").strip().casefold()
        in EXECUTABLE_FENCE_LANGUAGES
    ]


def continued_commands(block: str) -> list[tuple[int, str]]:
    lines = block.splitlines(keepends=True)
    commands: list[tuple[int, str]] = []
    offset = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip()
        if stripped.startswith("& ") or stripped.lower().startswith("powershell "):
            command_offset = offset
            command_lines = [line.rstrip("\r\n")]
            while command_lines[-1].rstrip().endswith("`") and index + 1 < len(lines):
                offset += len(lines[index])
                index += 1
                command_lines.append(lines[index].rstrip("\r\n"))
            commands.append((command_offset, "\n".join(command_lines)))
        offset += len(lines[index])
        index += 1
    return commands


class SessionAScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = {
            path: path.read_text(encoding="utf-8") for path in (DESIGN, PLAN)
        }

    def test_documents_exclude_personal_and_private_reproduction_references(self):
        forbidden = (
            r"C:" + r"\Users\User",
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
                    "User-confirmed Phase 3 data decisions and new model "
                    "creation/training belong to Session B and are not modified or "
                    "revalidated by Session A.",
                    text,
                )
                self.assertIn(
                    "Session C integrates Session B's new compatible 42-label "
                    "[1, 64, 438] model.",
                    text,
                )
                self.assertIn(
                    "Session A validates runtime and packaging only with the audited "
                    "default component defined in Task 6 and assembled in Task 7, or "
                    "the deterministic complete fixture component; Session C performs "
                    "the first build with Session B's supplied release component.",
                    text,
                    "Session A validation inputs must link to the complete Task 6 "
                    "components",
                )

    def test_documents_use_portable_command_and_placeholder_hygiene(self):
        for command in (
            "python -B -m unittest",
            "python.exe -m unittest",
            "python3 -m unittest",
            "python3.10 -m unittest",
            "py -3.10 -m unittest",
            "& python -m unittest",
        ):
            with self.subTest(command=command):
                self.assertIsNotNone(BARE_PYTHON_COMMAND.search(command))

        self.assertEqual(
            [],
            executable_fences("See the canonical <https://example.invalid/spec> link."),
            "Markdown autolinks outside executable fences are not placeholders",
        )

        for path, text in self.documents.items():
            with self.subTest(path=path):
                bare_commands = [
                    match.group(0).strip()
                    for _, block in executable_fences(text)
                    for match in BARE_PYTHON_COMMAND.finditer(block)
                ]
                bare_inline_commands = [
                    inline
                    for inline in re.findall(r"`([^`\r\n]+)`", text)
                    if INLINE_BARE_PYTHON_COMMAND.search(inline)
                ]
                self.assertEqual(
                    [],
                    bare_commands + bare_inline_commands,
                    f"bare Python commands in {path}: "
                    f"{bare_commands + bare_inline_commands}",
                )
                self.assertIsNone(
                    re.search(r"\b(?:CODEX_HOME|HOME)\b", text, flags=re.IGNORECASE)
                )
                for _, block in executable_fences(text):
                    self.assertIsNone(ANGLE_PLACEHOLDER.search(block))

    def test_session_a_is_not_assigned_phase_3_data_or_private_tag_work(self):
        forbidden_assignments = (
            re.compile(
                r"Session\s+A\s+(?:is\s+)?(?:responsible\s+for|owns?|must|shall|"
                r"will|should|handles?|assigned\s+to|performs?)\s+"
                r"(?:(?!Session\s+[BC]).){0,240}?"
                r"(?:Phase\s+3|model\s+(?:creation|training)|(?:re)?train(?:ing)?)",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"Session\s+A\s+(?:creates?|(?:re)?trains?|edits?)\b"
                r"(?:(?!Session\s+[BC]).){0,240}?\b(?:Phase\s+3|model|training)\b",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"(?:Phase\s+3|model\s+(?:creation|training)|(?:re)?train(?:ing)?)"
                r"(?:(?!Session\s+[BC]).){0,240}?\b(?:is|are|will\s+be|must\s+be)"
                r"\s+(?:owned|handled|performed|created|trained)\s+by\s+Session\s+A",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"Session A\s+(?:must|will|shall|should|is assigned to|is responsible "
                r"to)\s+create\b.*?\bprivate\b.*?\btag\b",
                re.IGNORECASE | re.DOTALL,
            ),
        )
        for example in (
            "Session A is responsible for\ntraining Phase 3.",
            "Session A is assigned to\ntrain Phase 3.",
            "Session A performs model training.",
            "Session A trains the Phase 3 model.",
            "Model training\nis performed by Session A.",
            "The Phase 3 model will be trained\nby Session A.",
        ):
            with self.subTest(example=example):
                self.assertTrue(
                    any(pattern.search(example) for pattern in forbidden_assignments),
                    "training-ownership patterns must catch ordinary multiline prose",
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
        self.assertIn("& $env:KNEE42_BUILD_PYTHON", plan)
        handoff_blocks = [
            block
            for block in re.findall(
                r"```powershell\s+(.*?)```", plan, flags=re.DOTALL
            )
            if "& $env:KNEE42_BUILD_PYTHON -B scripts/build_knee42_release.py"
            in block
            and "$env:KNEE42_MODEL_COMPONENT_DIR" in block
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

    def test_plan_routes_every_output_command_through_build_root(self):
        plan = self.documents[PLAN]
        self.assertIsNotNone(
            OUTPUT_LITERAL_BUILD.search("--output-dir `\nbuild/release"),
            "continued literal build paths must be detected",
        )
        output_commands = [
            command
            for _, block in executable_fences(plan)
            for _, command in continued_commands(block)
            if OUTPUT_COMMAND.search(command)
        ] + [
            inline
            for inline in re.findall(r"`([^`\r\n]+)`", plan)
            if inline.lstrip().startswith("& ")
            and OUTPUT_COMMAND.search(inline)
        ]
        self.assertGreater(len(output_commands), 0)
        for command in output_commands:
            with self.subTest(command=command):
                self.assertIn("$env:KNEE42_BUILD_ROOT", command)
                self.assertIsNone(
                    OUTPUT_LITERAL_BUILD.search(command),
                    f"literal build path in an output command:\n{command}",
                )

    def test_locks_and_notices_precede_final_build_and_verification(self):
        plan = self.documents[PLAN]
        lock_task = plan.index(
            "### Task 9: Windows locks, SBOM, and redistribution notices"
        )
        final_verification = plan.index(
            "### Task 11: Clean extraction, install, replay, and artifact gates"
        )
        final_build_marker = (
            "Build final Session A artifacts and evidence from the recorded clean HEAD"
        )
        self.assertIn(final_build_marker, plan)
        final_build = plan.index(final_build_marker)
        self.assertLess(lock_task, final_build)
        self.assertLess(lock_task, final_verification)
        self.assertLess(final_verification, final_build)

    def test_final_artifacts_are_built_from_the_final_clean_commit(self):
        plan = self.documents[PLAN]
        final_head_marker = "$env:KNEE42_FINAL_HEAD = (git rev-parse HEAD).Trim()"
        final_build_marker = (
            "Build final Session A artifacts and evidence from the recorded clean HEAD"
        )
        self.assertIn(final_head_marker, plan, "plan must record the final clean HEAD")
        self.assertIn(
            final_build_marker,
            plan,
            "plan must identify the final post-commit artifact/evidence build",
        )
        final_head = plan.index(final_head_marker)
        final_build = plan.index(final_build_marker)
        self.assertLess(final_head, final_build)
        self.assertIn("git commit", plan[:final_head])
        self.assertIsNone(
            re.search(r"^\s*git\s+(?:add|commit)\b", plan[final_build:], re.MULTILINE)
        )
        self.assertIn("--source-commit $env:KNEE42_FINAL_HEAD", plan[final_build:])
        self.assertIn("VERSION_MANIFEST.source_commit", plan[final_build:])
        self.assertIn("handoff_record.source_commit", plan[final_build:])
        self.assertIn("test_handoff_record_binds_requested_source_commit", plan)
        self.assertIn(
            "No tracked files are written or committed after this final build begins.",
            plan[final_build:],
        )

    def test_python_preflight_is_caller_supplied_exact_310_and_64_bit(self):
        plan = self.documents[PLAN]
        task_one = plan.index("### Task 1:")
        preflight = plan[:task_one]
        self.assertNotIn(
            "Get-Command python",
            plan,
            "KNEE42_PYTHON must be caller-supplied, not PATH-selected",
        )
        for required in (
            "[string]::IsNullOrWhiteSpace($env:KNEE42_PYTHON)",
            "Test-Path -LiteralPath $env:KNEE42_PYTHON -PathType Leaf",
            "& $env:KNEE42_PYTHON -c",
            "sys.version_info[:2] != (3, 10)",
            'struct.calcsize("P") * 8 != 64',
        ):
            with self.subTest(required=required):
                self.assertIn(required, preflight)

    def test_asset_cache_is_prepared_and_verified_before_every_offline_build(self):
        plan = self.documents[PLAN]
        real_commands = [
            (position + command_position, command)
            for position, block in executable_fences(plan)
            for command_position, command in continued_commands(block)
            if any(
                tool in command
                for tool in (
                    "scripts/build_knee42_release.py",
                    "scripts/build_windows_portable.ps1",
                    "scripts/verify_knee42_release.ps1",
                )
            )
        ]
        prepare_commands = [
            (position, command)
            for position, command in real_commands
            if "--prepare-assets-only" in command
        ]
        offline_commands = [
            (position, command)
            for position, command in real_commands
            if re.search(r"(?i)(?:--offline|-Offline)\b", command)
        ]
        self.assertGreaterEqual(len(prepare_commands), 1)
        self.assertGreaterEqual(len(offline_commands), 4)
        for _, command in prepare_commands:
            with self.subTest(prepare_command=command):
                self.assertNotRegex(command, r"(?i)(?:--offline|-Offline)\b")
                self.assertIn("--asset-cache $env:KNEE42_ASSET_CACHE", command)
        first_prepare = min(position for position, _ in prepare_commands)
        for position, command in offline_commands:
            with self.subTest(offline_command=command):
                self.assertLess(first_prepare, position)
                self.assertRegex(
                    command,
                    r"(?i)(?:--asset-cache|-AssetCache)\s+"
                    r"\$env:KNEE42_ASSET_CACHE",
                )
        self.assertIn("prepare/download-only mode", plan)
        self.assertIn("pinned official URLs and SHA-256 hashes", plan)
        self.assertIn("Session C may reuse the verified asset cache", plan)
        self.assertIn(
            "https://github.com/Mikullee/Sign-Language-Recognition-Gemma/"
            "releases/download/v1.0.0-v13/knee42-model-v11.zip",
            plan,
        )
        self.assertIn(
            "af45a4a50fc67755dd86be1b47fe975120e47a1b9f6850232e294685dd4ac8df",
            plan,
        )
        self.assertIn(
            "test_prepare_rejects_old_model_zip_hash_before_extraction", plan
        )

    def test_default_and_fixture_components_are_complete_and_deterministic(self):
        plan = self.documents[PLAN]
        component_tasks = plan[
            plan.index("### Task 6:") : plan.index("### Task 8:")
        ]
        for required in (
            "complete audited default model component",
            "pinned old model ZIP",
            "runtime_config.json",
            "selection_ledger.json",
            "integrity_manifest.sha256",
            "hand_landmarker.task",
            "pose_landmarker.task",
            "42-label label map",
            "all required model payload files",
            "deterministic tiny complete fixture component",
        ):
            with self.subTest(required=required):
                self.assertIn(
                    required,
                    component_tasks,
                    f"incomplete default/fixture component requirement: {required}",
                )
        self.assertIn("$env:KNEE42_DEFAULT_COMPONENT_TRUST_PATH", component_tasks)
        self.assertNotIn(
            "Join-Path $env:KNEE42_DEFAULT_COMPONENT_DIR "
            "'component_manifest.expected-sha256.txt'",
            component_tasks,
        )

    def test_build_and_runtime_locks_have_separate_clean_environments(self):
        plan = self.documents[PLAN]
        final_head_marker = "$env:KNEE42_FINAL_HEAD = (git rev-parse HEAD).Trim()"
        final_head = plan.find(final_head_marker)
        before_final_build = plan[:final_head] if final_head >= 0 else plan
        for required in (
            "$env:KNEE42_BUILD_ENV",
            "$env:KNEE42_BUILD_PYTHON",
            "& $env:KNEE42_BUILD_PYTHON -m pip install --requirement "
            "requirements-windows-build.lock.txt",
            "requirements-windows-build.lock.txt includes the complete runtime closure",
            "$env:KNEE42_RUNTIME_ENV",
            "$env:KNEE42_RUNTIME_PYTHON",
            "& $env:KNEE42_RUNTIME_PYTHON -m pip install --requirement "
            "requirements-windows-runtime.lock.txt",
            "& $env:KNEE42_BUILD_PYTHON -B -m unittest discover -s tests -v",
        ):
            with self.subTest(required=required):
                self.assertIn(
                    required,
                    before_final_build,
                    f"missing separate build/runtime lock requirement: {required}",
                )
        self.assertIn("clean extracted runtime", plan)
        self.assertRegex(
            plan,
            r"& \$env:KNEE42_RUNTIME_PYTHON\b[\s\S]{0,240}?--self-test",
        )

    def test_component_identity_is_verified_and_derived_from_manifest(self):
        plan = self.documents[PLAN]
        for required in (
            "default_model_version: str",
            "component_id",
            "model_version",
            "label_count",
            "input_shape",
            "payload_sha256",
            "selection_ledger_sha256",
            "runtime_config_sha256",
            "verify the expected component manifest SHA-256 before parsing any "
            "self-declared fields",
            "VERSION_MANIFEST.model_version",
            "VERSION_MANIFEST.component_id",
            "VERSION_MANIFEST.model_component_manifest_sha256",
            "Runtime startup displays the generated component_id and model_version",
            "without code edits",
            "test_component_rejects_self_declared_label_count",
            "test_component_rejects_self_declared_model_version",
            "test_trusted_component_rejects_label_count_outside_canonical_contract",
            "test_verified_alternate_model_version_is_generated_identity",
            "test_builder_has_no_label_or_version_override_arguments",
        ):
            with self.subTest(required=required):
                self.assertIn(
                    required,
                    plan,
                    f"missing verified component identity requirement: {required}",
                )
        self.assertRegex(
            plan,
            r"(?:spec\.default_model_version,\s*[\"']v11[\"']|"
            r"default_model_version\s*[=:]\s*[\"']v11[\"'])",
        )
        self.assertIn("[1, 64, 438]", plan)
        self.assertRegex(plan, r"label_count[^\r\n]{0,80}\b42\b")

    def test_secret_scan_distinguishes_clean_findings_and_command_errors(self):
        plan = self.documents[PLAN]
        for required in (
            "$secretScanExit = $LASTEXITCODE",
            "$secretPattern",
            "if ($secretScanExit -eq 0)",
            "if ($secretScanExit -ne 1)",
            "test_secret_scan_does_not_match_its_own_sentinel_definitions",
            "exit 1 means clean, exit 0 means findings and failure, and any other "
            "nonzero exit means the scan command failed",
        ):
            with self.subTest(required=required):
                self.assertIn(
                    required,
                    plan,
                    f"missing secret-scan exit semantics: {required}",
                )

    def test_windows_build_contract_forwards_final_commit_and_offline_gate(self):
        plan = self.documents[PLAN]
        task_eight = plan[plan.index("### Task 8:") : plan.index("### Task 9:")]
        for required in (
            "test_windows_wrapper_requires_source_commit_and_offline_cache",
            "-SourceCommit",
            "-Offline",
        ):
            with self.subTest(required=required):
                self.assertIn(required, task_eight)


if __name__ == "__main__":
    unittest.main()

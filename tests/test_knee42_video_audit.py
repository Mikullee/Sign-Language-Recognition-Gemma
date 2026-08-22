from __future__ import annotations

import sys
import types
import unittest

sys.modules.setdefault("cv2", types.SimpleNamespace())

from scripts.audit_knee42_videos_upright import (  # noqa: E402
    validate_final_audit,
    validate_research_scope,
)


def research_rows():
    rows = []
    for index in range(1634):
        signer = ("L", "P", "X")[index % 3]
        rows.append({"sample_id": f"train-{index}", "split": "train", "signer_id": signer})
    for index in range(618):
        rows.append({"sample_id": f"dev-{index}", "split": "dev", "signer_id": "H"})
    return rows


class VideoAuditPolicyTests(unittest.TestCase):
    def test_exact_research_scope_accepts_2252_lpx_h_rows(self):
        validate_research_scope(research_rows())

    def test_research_scope_refuses_disguised_j_row(self):
        rows = research_rows()
        rows[-1]["signer_id"] = "J"

        with self.assertRaisesRegex(ValueError, "Test/J"):
            validate_research_scope(rows)

    def test_final_audit_refuses_pending_visual_conclusions(self):
        rows = research_rows()
        for row in rows:
            row.update(
                final_status="PASS",
                visual_reviewer="codex-visual-audit",
                visual_evidence_path=f"contact_sheets/{row['sample_id']}.jpg",
                final_reason="upright complete label-consistent",
            )
        rows[-1]["final_status"] = "PENDING"

        with self.assertRaisesRegex(ValueError, "without a final visual conclusion"):
            validate_final_audit(rows)


if __name__ == "__main__":
    unittest.main()

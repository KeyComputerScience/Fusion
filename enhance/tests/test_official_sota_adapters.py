"""Compare adapters with the ACTUAL downloaded authors' forward functions.

AST extraction avoids loading BERT/ResNet or executing repository entrypoints.
The original forward bodies run unchanged using deterministic stub encoders.
"""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from official_sota_adapters import pdf_core, qmf_core, qmf_ranking_loss, ScalarBranchFusion
OFFICIAL = Path(__file__).resolve().parent / "official_kernels"


def original_class(path, class_name):
    tree = ast.parse(path.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    instance = namespace[class_name].__new__(namespace[class_name])
    nn.Module.__init__(instance)
    instance.args = SimpleNamespace(df=True)
    return instance


class IdentityLogits(nn.Module):
    def forward(self, x, *unused):
        return x


class TupleBranch(nn.Module):
    def forward(self, x, *unused):
        return x


class OfficialKernelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.logits = torch.randn(11, 2, 3, dtype=torch.float64)
        self.tcp = torch.rand(11, 2, dtype=torch.float64) * 0.7 + 0.1

    def test_qmf_matches_original_forward_without_reconstructed_probabilities(self):
        path = OFFICIAL / "qmf_late_fusion.py"
        model = original_class(path, "MultimodalLateFusionClf")
        model.txtclf, model.imgclf = IdentityLogits(), IdentityLogits()
        original = model(self.logits[:, 0], None, None, self.logits[:, 1])
        adapted, coefficients = qmf_core(self.logits)
        torch.testing.assert_close(adapted, original[0], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(coefficients[:, 0], original[3].squeeze(-1), rtol=1e-12, atol=1e-12)

    def test_pdf_matches_original_train_and_test_forward(self):
        path = OFFICIAL / "pdf_latefusion_pdf.py"
        model = original_class(path, "MultimodalLateFusionClf_pdf")
        model.txtclf, model.imgclf = TupleBranch(), TupleBranch()
        model.ConfidNet_txt, model.ConfidNet_img = nn.Identity(), nn.Identity()
        left = (self.logits[:, 0], self.tcp[:, :1])
        right = (self.logits[:, 1], self.tcp[:, 1:])
        for choice, calibrate in (("pdf_train", False), ("pdf_test", True)):
            original = model(left, None, None, right, choice)
            adapted, weights, _ = pdf_core(self.logits, self.tcp, calibrate=calibrate)
            torch.testing.assert_close(adapted, original[0], rtol=1e-7, atol=1e-9)
            if calibrate:
                torch.testing.assert_close(weights[:, 0], original[5].squeeze(-1), rtol=1e-7, atol=1e-9)

    def test_qmf_energy_cannot_be_reconstructed_from_probabilities(self):
        fused, coefficients = qmf_core(self.logits)
        shifted, shifted_coefficients = qmf_core(self.logits + 5.0)
        torch.testing.assert_close(self.logits.softmax(-1), (self.logits + 5.0).softmax(-1))
        self.assertFalse(torch.allclose(coefficients, shifted_coefficients))

    def test_qmf_ranking_matches_author_function(self):
        path = OFFICIAL / "qmf_train_qmf.py"
        tree = ast.parse(path.read_text())
        definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "rank_loss")
        namespace = {"torch": torch, "nn": nn}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
        history = torch.rand(11, 2, dtype=torch.float64)
        confidence = torch.randn(11, 2, dtype=torch.float64)
        expected = 0.0
        for source in range(2):
            class CPUHistory:
                def get_target_margin(self, first, second):
                    values = history[:, source]
                    values = (values - values.min()) / (values.max() - values.min())
                    difference = values[first] - values[second]
                    return difference.sign(), difference.abs()
            expected = expected + namespace["rank_loss"](confidence[:, source:source+1], torch.arange(11), CPUHistory())
        torch.testing.assert_close(qmf_ranking_loss(confidence, history), expected, rtol=1e-12, atol=1e-12)

    def test_three_source_extension_normalises_and_uniform_case_is_finite(self):
        logits = torch.zeros(8, 3, 2)
        fused, weights, diagnostics = pdf_core(logits, torch.full((8, 3), 0.5))
        self.assertTrue(torch.isfinite(fused).all())
        torch.testing.assert_close(weights.sum(1), torch.ones(8))

    def test_all_models_perform_real_gradient_updates(self):
        x = torch.randn(12, 3, 2)
        y = torch.arange(12) % 2
        for method in ("QMF_SCALAR", "PDF_SCALAR", "EF_CLASSIFIER"):
            model = ScalarBranchFusion(method, features=2)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            before = model.classifiers[0].weight.detach().clone()
            loss, _ = model.training_loss(x, y, torch.zeros(12, 3))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            self.assertFalse(torch.equal(before, model.classifiers[0].weight))


if __name__ == "__main__":
    unittest.main()

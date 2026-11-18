"""回归检查：重点防止同名图片和空目录造成配对假通过。"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_data import check_category


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw = Path(self.tmp.name) / "raw"
        self.maps = Path(self.tmp.name) / "maps"
        for root in (self.raw, self.maps):
            for split in ("train", "validation", "test"):
                folder = root / "screw_bag" / split / "good"
                folder.mkdir(parents=True)
                # 这里只测试路径；不将空文件冒充可解码的图像。
                (folder / "000.png").touch()

    def test_matching_paths_pass(self):
        self.assertTrue(check_category(self.raw, self.maps, "screw_bag")["passed"])

    def test_same_basename_in_wrong_subfolder_fails(self):
        file = self.maps / "screw_bag/test/good/000.png"
        wrong = self.maps / "screw_bag/test/logical_anomalies/000.png"
        wrong.parent.mkdir()
        file.rename(wrong)
        result = check_category(self.raw, self.maps, "screw_bag")
        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"][2]["missing_maps"], ["good/000.png"])

    def test_missing_category_does_not_pass(self):
        self.assertFalse(check_category(self.raw, self.maps, "pushpins")["passed"])

    def test_extra_map_fails(self):
        (self.maps / "screw_bag/train/good/001.png").touch()
        self.assertFalse(check_category(self.raw, self.maps, "screw_bag")["passed"])


if __name__ == "__main__":
    unittest.main()

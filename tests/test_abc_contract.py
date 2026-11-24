"""可在没有 GPU、没有 PyTorch 的电脑核对协议行为。"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from salad_training.contract import load_contract, load_pairs, make_schedule, safe_path, sha, validate_roles


def config(group):
    return load_contract(ROOT / 'configs/abc_screw_bag_v1' / (group + '.json'))


class ContractTests(unittest.TestCase):
    def test_same_sample_control_and_ranking_share_entire_schedule(self):
        a, b = config('SB4'), config('SB2')
        self.assertEqual(make_schedule(a, 5000), make_schedule(b, 5000))
        self.assertEqual(a['variant']['spatial_weight'], b['variant']['spatial_weight'])
        self.assertNotEqual(a['variant']['rank_weight'], b['variant']['rank_weight'])

    def test_schedule_prefix_stable_and_source_epoch_covers_training_only(self):
        c = config('SB0')
        full = make_schedule(c, 5000)
        self.assertEqual(full[:100], make_schedule(c, 100))
        self.assertEqual({e['source_path'] for e in full[:335]}, set(c['roles']['optimization']))
        self.assertFalse({e['source_path'] for e in full} & set(c['roles']['final_diagnostic']))
        self.assertTrue(any(e['corrupt'] for e in full) and any(not e['corrupt'] for e in full))

    def test_source_leakage_rejected(self):
        roles = copy.deepcopy(config('SB0')['roles'])
        roles['optimization'].append(roles['final_diagnostic'][0])
        with self.assertRaises(ValueError):
            validate_roles(roles)

    def test_sb0_can_start_without_pairs_but_sb4_cannot(self):
        self.assertEqual(load_pairs(None, config('SB0')), ({}, None))
        with self.assertRaises(ValueError):
            load_pairs(None, config('SB4'))

    def test_paths_cannot_escape_input_root(self):
        with tempfile.TemporaryDirectory() as d:
            for path in ('../private.png', '/private.png', 'C:\\private.png'):
                with self.assertRaises(ValueError):
                    safe_path(d, path)

    def test_review_provenance_file_integrity_and_held_out_sources(self):
        c = config('SB4')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = c['roles']['optimization'][0]
            pair = {'pair_id': 'test1', 'purpose': 'train', 'review_status': 'approved',
                    'reviewer': 'fixture reviewer', 'review_date': '2026-11-24',
                    'review_reason': 'fixture only', 'rule_reference': 'fixture rule',
                    'source_path': source, 'source_map_sha256': c['map_hashes'][source]}
            for kind in ('valid', 'invalid', 'anomaly_mask'):
                file = root / (kind + '.png')
                file.write_bytes(kind.encode())  # 此测试验证文件指纹；图像解码由训练运行测试负责。
                pair[kind + '_path'], pair[kind + '_sha256'] = file.name, sha(file)
            manifest = root / 'pairs.json'

            def write(pairs):
                manifest.write_text(json.dumps({'representation': 'salad_author_indices_v1', 'pairs': pairs}))

            write([pair])
            self.assertIn(source, load_pairs(manifest, c)[0])
            for field, value in [('reviewer', ''), ('review_status', 'pending'),
                                 ('source_map_sha256', 'wrong'),
                                 ('source_path', c['roles']['development'][0])]:
                altered = dict(pair, **{field: value})
                write([altered])
                with self.assertRaises(ValueError, msg=field):
                    load_pairs(manifest, c)
            write([pair, pair])
            with self.assertRaises(ValueError):
                load_pairs(manifest, c)
            write([pair])
            (root / 'invalid.png').write_bytes(b'changed after review')
            with self.assertRaises(ValueError):
                load_pairs(manifest, c)


if __name__ == '__main__':
    unittest.main()

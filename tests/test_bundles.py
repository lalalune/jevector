import json
import tempfile
import unittest
from pathlib import Path

from matching.cli import load_gallery


class BundleTests(unittest.TestCase):
    def test_bundle_selection_deduplication_and_validation(self):
        source = json.loads(Path("benchmarks/vectors/jev-64.json").read_text())
        profile_id = next(iter(source))
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp, "bundle.json")
            record = Path(tmp, "record.json")
            bundle.write_text(json.dumps(source))
            record.write_text(json.dumps(source[profile_id]))
            selected, records = load_gallery(None, [bundle, record], profile_id)
            self.assertEqual(selected, profile_id)
            self.assertEqual(records, source)
            self.assertEqual(load_gallery(record, [bundle]), (profile_id, source))
            with self.assertRaisesRegex(ValueError, "absent"):
                load_gallery(None, [bundle], "missing")
            bundle.write_text(json.dumps({"wrong-id": source[profile_id]}))
            with self.assertRaisesRegex(ValueError, "Bundle key"):
                load_gallery(None, [bundle], profile_id)

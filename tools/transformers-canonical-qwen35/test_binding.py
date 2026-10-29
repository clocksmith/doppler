"""Loss, aliasing, corruption and reinterpretation controls for canonical input."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from binding import bind_names, read_inputs, verify_bytes


POLICY = json.loads(Path(__file__).with_name("contract.json").read_text())
EMBED = "model.language_model.embed_tokens.weight"
HEAD = "model.language_model.lm_head.weight"


def fixture(directory: Path):
    data = {EMBED: struct.pack("<ff", 1, 2), HEAD: struct.pack("<ff", 3, 4)}
    tensors, header, payload = [], {}, bytearray()
    for name, raw in data.items():
        start = len(payload)
        payload.extend(raw)
        header[name] = {"dtype": "F32", "shape": [1, 2], "data_offsets": [start, len(payload)]}
        tensors.append({"name": name, "dtype": "F32", "shape": [1, 2], "bytes": len(raw),
                        "canonicalBytesSha256": hashlib.sha256(raw).hexdigest()})
    manifest = {"schema": POLICY["sourceSchema"], "completed": True,
                "tensorPrefix": POLICY["sourcePrefix"], "tensors": tensors}
    config = {"model_type": "qwen3_5", "tie_word_embeddings": False,
              "text_config": {"model_type": "qwen3_5_text", "tie_word_embeddings": False,
                              "mtp_num_hidden_layers": 0}}
    (directory / "canonical.json").write_text(json.dumps(manifest))
    (directory / "config.json").write_text(json.dumps(config))
    write_weights(directory, header, payload)
    return header, payload


def write_weights(directory, header, payload):
    encoded = json.dumps(header).encode()
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.header, self.payload = fixture(self.directory)

    def test_separate_matrices_bind_without_retying(self):
        inputs = read_inputs(self.directory, POLICY)
        verify_bytes(inputs, 3)
        mapped = bind_names(inputs["manifest"]["tensors"],
                            {"model.embed_tokens.weight": [1, 2], "lm_head.weight": [1, 2]}, POLICY)
        self.assertEqual([m["target"] for m in mapped], ["model.embed_tokens.weight", "lm_head.weight"])
        self.assertNotEqual(mapped[0]["sha256"], mapped[1]["sha256"])

    def test_corrupt_payload_never_gets_a_verified_binding(self):
        self.payload[-1] ^= 1
        write_weights(self.directory, self.header, self.payload)
        inputs = read_inputs(self.directory, POLICY)
        with self.assertRaisesRegex(ValueError, "byte hash mismatch"):
            verify_bytes(inputs, 3)

    def test_ambiguous_or_overlapping_storage_rejected(self):
        self.header[HEAD]["data_offsets"] = [0, 8]
        write_weights(self.directory, self.header, self.payload)
        with self.assertRaisesRegex(ValueError, "overlap"):
            read_inputs(self.directory, POLICY)

    def test_missing_source_and_extra_target_are_not_silently_dropped(self):
        inputs = read_inputs(self.directory, POLICY)
        shapes = {"model.embed_tokens.weight": [1, 2], "lm_head.weight": [1, 2], "unbound": [1]}
        with self.assertRaisesRegex(ValueError, "unbound parameters"):
            bind_names(inputs["manifest"]["tensors"], shapes, POLICY)
        shapes.pop("unbound")
        shapes["lm_head.weight"] = [2, 1]
        with self.assertRaisesRegex(ValueError, "shape"):
            bind_names(inputs["manifest"]["tensors"], shapes, POLICY)

    def test_explicit_tying_or_speculative_model_refuses(self):
        path = self.directory / "config.json"
        config = json.loads(path.read_text())
        config["tie_word_embeddings"] = True
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "untied"):
            read_inputs(self.directory, POLICY)

    def test_source_identity_must_be_complete_and_exact(self):
        path = self.directory / "canonical.json"
        manifest = json.loads(path.read_text())
        manifest["tensors"].append(copy.deepcopy(manifest["tensors"][0]))
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "inventory"):
            read_inputs(self.directory, POLICY)


if __name__ == "__main__":
    unittest.main()

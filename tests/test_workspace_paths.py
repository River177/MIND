import importlib.util
import os
import pathlib
import sys
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module(module_name: str, file_name: str):
    tf_stub = types.ModuleType("tensorflow")
    tf_stub.config = types.SimpleNamespace(
        list_physical_devices=lambda *_: [],
        experimental=types.SimpleNamespace(set_memory_growth=lambda *_: None),
    )

    iterator_module = types.ModuleType("recommenders.models.newsrec.io.mind_iterator")
    iterator_module.MINDIterator = object

    nrms_module = types.ModuleType("recommenders.models.newsrec.models.nrms")
    nrms_module.NRMSModel = object

    newsrec_utils_module = types.ModuleType("recommenders.models.newsrec.newsrec_utils")
    newsrec_utils_module.prepare_hparams = lambda **kwargs: kwargs

    stubbed_modules = {
        "tensorflow": tf_stub,
        "recommenders": types.ModuleType("recommenders"),
        "recommenders.models": types.ModuleType("recommenders.models"),
        "recommenders.models.newsrec": types.ModuleType("recommenders.models.newsrec"),
        "recommenders.models.newsrec.io": types.ModuleType("recommenders.models.newsrec.io"),
        "recommenders.models.newsrec.io.mind_iterator": iterator_module,
        "recommenders.models.newsrec.models": types.ModuleType("recommenders.models.newsrec.models"),
        "recommenders.models.newsrec.models.nrms": nrms_module,
        "recommenders.models.newsrec.newsrec_utils": newsrec_utils_module,
    }

    original_modules = {}
    for name, module in stubbed_modules.items():
        original_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    try:
        spec = importlib.util.spec_from_file_location(module_name, ROOT / file_name)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class WorkspacePathTests(unittest.TestCase):
    def test_train_script_uses_workspace_base_dir(self):
        module = load_module("train_nrms_for_test", "train_nrms.py")

        self.assertEqual(module.BASE_DIR, str(ROOT))
        self.assertEqual(module.DATA_DIR, os.path.join(str(ROOT), "dataset"))
        self.assertEqual(module.OUTPUT_DIR, os.path.join(str(ROOT), "outputs"))

    def test_build_utils_uses_workspace_base_dir(self):
        module = load_module("build_nrms_utils_for_test", "build_nrms_utils.py")

        self.assertEqual(module.get_default_base_dir(), str(ROOT))


if __name__ == "__main__":
    unittest.main()

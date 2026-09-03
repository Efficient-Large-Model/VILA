import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_builder(monkeypatch):
    package_names = ("llava", "llava.model", "llava.model.multimodal_encoder")
    for package_name in package_names:
        package = ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    transformers = ModuleType("transformers")
    transformers.AutoConfig = type("AutoConfig", (), {})
    transformers.PretrainedConfig = type("PretrainedConfig", (), {})
    transformers.PreTrainedModel = type("PreTrainedModel", (), {})
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    encoder_modules = {
        "clip_encoder": ("CLIPVisionTower", "CLIPVisionTowerS2"),
        "intern_encoder": ("InternVisionTower", "InternVisionTowerS2"),
        "ps3_encoder": ("PS3VisionTower",),
        "radio_encoder": ("RADIOVisionTower",),
        "siglip_encoder": (
            "SiglipVisionTower",
            "SiglipVisionTowerDynamicS2",
            "SiglipVisionTowerS2",
        ),
    }
    for module_name, class_names in encoder_modules.items():
        module = ModuleType(f"llava.model.multimodal_encoder.{module_name}")
        for class_name in class_names:
            setattr(module, class_name, type(class_name, (), {}))
        monkeypatch.setitem(sys.modules, module.__name__, module)

    builder_path = (
        Path(__file__).parents[1]
        / "llava"
        / "model"
        / "multimodal_encoder"
        / "builder.py"
    )
    spec = importlib.util.spec_from_file_location(
        "llava.model.multimodal_encoder.builder", builder_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_ps3_vision_tower_reads_ps3_scales(monkeypatch):
    builder = _load_builder(monkeypatch)

    class StubPS3VisionTower:
        def __init__(self, model_name_or_path, config):
            self.config = SimpleNamespace(hidden_size=1152)
            self.vision_tower = SimpleNamespace(
                vision_model=SimpleNamespace(
                    low_res_token_num=576,
                    ps3_scales=[384, 768, 1536],
                )
            )

    monkeypatch.setattr(builder, "PS3VisionTower", StubPS3VisionTower)
    config = SimpleNamespace(
        resume_path=None,
        ps3=True,
        s2=False,
        dynamic_s2=False,
    )

    vision_tower = builder.build_vision_tower("stub-ps3", config)

    assert isinstance(vision_tower, StubPS3VisionTower)
    assert config.mm_hidden_size == 1152
    assert config.mm_low_res_token_num == 576
    assert config.mm_scale_num == 3


def test_no_source_reads_the_misspelled_s3_scales_attribute():
    """`PS3VisionEncoder` exposes `ps3_scales`; there is no `s3_scales` on it.

    The misspelling raises AttributeError only once a PS3 model is actually being
    built, which needs a GPU and the `ps3-torch` package, so it survives ordinary
    testing. Scanning the source keeps every occurrence covered — including the
    four in `llava_arch.encode_images_ps3`, which is only reached with
    `ps3_dynamic_aspect_ratio` enabled.

    `s2_scales` is a different, legitimate attribute used by the dynamic-S2 path,
    so this matches the attribute access `.s3_scales` specifically.
    """
    misspelled = re.compile(r"\.s3_scales\b")
    offenders = [
        f"{path.relative_to(Path(__file__).parents[1])}:{number}"
        for path in sorted((Path(__file__).parents[1] / "llava").rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if misspelled.search(line)
    ]

    assert offenders == [], "use ps3_scales, the attribute PS3VisionEncoder defines: " + ", ".join(
        offenders
    )

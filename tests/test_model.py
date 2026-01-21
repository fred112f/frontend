from hydra import initialize, compose
from hydra.utils import instantiate
import pytest
import torch

from exam_project.model import BaseCNN, BaseANN, ViTClassifier


with initialize(version_base=None, config_path="../src/exam_project/configs"):
    cfg_data = compose(config_name="data")

SEED = cfg_data.hyperparameters.seed

def set_global_seed(seed: int = SEED) -> None:
    """Set all seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)

def load_model_from_config(config_name: str):
    with initialize(
        version_base=None,
        config_path="../src/exam_project/configs/models",
    ):
        cfg = compose(config_name=config_name)
        return instantiate(cfg)

MODELS = [
    ("BaseCNN", "cnn"),
    ("BaseANN", "ann"),
    ("ViTClassifier", "vit"),
]


@pytest.mark.parametrize("model_name,config_name", MODELS)
class TestModels:
    """Combined test suite for all models."""
    @pytest.fixture
    def model(self, config_name):
        return load_model_from_config(config_name)

    def test_initialization(self, model_name, model):
        """Test model initialization."""
        try:
            assert isinstance(model, (BaseCNN, BaseANN, ViTClassifier))
            assert hasattr(model, 'forward')
        except AssertionError as e:
            pytest.fail(f"[{model_name}] Initialization failed: {e}")

    def test_forward_pass(self, model_name, model):
        """Test forward pass with batch of images."""
        x = torch.randn(2, 1, 48, 48)
        model.eval()
        with torch.no_grad():
            output = model(x)

        try:
            assert output.shape == (2, 7), f"Shape {output.shape} != (2, 7)"
            assert output.dtype == torch.float32
        except AssertionError as e:
            pytest.fail(f"[{model_name}] Forward pass failed: {e}")

    def test_training_step(self, model_name, model):
        """Test training step returns valid loss."""
        x = torch.randn(2, 1, 48, 48)
        y = torch.randint(0, 7, (2,))
        loss = model.training_step((x, y))

        try:
            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0
            assert loss.item() > 0
        except AssertionError as e:
            pytest.fail(f"[{model_name}] Training step failed: {e}")

    def test_optimizer_config(self, model_name, model):
        """Test optimizer configuration."""
        optimizer = model.configure_optimizers()
        try:
            assert optimizer is not None
            assert hasattr(optimizer, 'step')
        except AssertionError as e:
            pytest.fail(f"[{model_name}] Optimizer config failed: {e}")

    def test_reproducibility(self, model_name, config_name):
        """Test reproducibility with seeding and dropout."""
        x = torch.randn(2, 1, 48, 48)

        # Create models with same seed
        set_global_seed(SEED)
        m1 = load_model_from_config(config_name)

        set_global_seed(SEED)
        m2 = load_model_from_config(config_name)

        try:
            m1.eval()
            m2.eval()
            with torch.no_grad():
                out1 = m1(x)
                out2 = m2(x)
            assert torch.allclose(out1, out2, atol=1e-5), (
                "Seeding failed: outputs differ"
            )

            if model_name in ["BaseCNN", "BaseANN"]:
                m1.train()
                assert not torch.allclose(m1(x), m1(x)), (
                    "Dropout inactive in training"
                )

            m1.eval()
            with torch.no_grad():
                assert torch.allclose(m1(x), m1(x), atol=1e-7)
        except AssertionError as e:
            pytest.fail(f"[{model_name}] Reproducibility failed: {e}")

    def test_parameters(self, model_name, model):
        """Test model has trainable parameters."""
        params = list(model.parameters())
        trainable = [p for p in params if p.requires_grad]
        try:
            assert len(params) > 0, "No parameters"
            assert len(trainable) > 0, "No trainable parameters"
        except AssertionError as e:
            pytest.fail(f"[{model_name}] Parameters failed: {e}")


@pytest.mark.parametrize("model_name,config_name", MODELS)
class TestModelComparison:
    """Cross-model consistency tests."""

    def test_output_shape_consistency(self, model_name, config_name):
        """Test all models produce same output shape."""
        model = load_model_from_config(config_name)
        x = torch.randn(2, 1, 48, 48)

        model.eval()
        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 7), f"[{model_name}] Wrong shape"

    def test_device_compatibility(self, model_name, config_name):
        model = load_model_from_config(config_name)
        x = torch.randn(2, 1, 48, 48)

        model = model.cpu()
        x = x.cpu()

        output = model(x)
        assert output.device.type == "cpu", f"[{model_name}] Device mismatch"

    def test_gradient_flow(self, model_name, config_name):
        model = load_model_from_config(config_name)
        x = torch.randn(2, 1, 48, 48, requires_grad=True)
        y = torch.randint(0, 7, (2,))

        output = model(x)
        loss = model.loss_fn(output, y)
        loss.backward()

        assert x.grad is not None, f"[{model_name}] No input gradients"
        assert any(
            p.grad is not None
            for p in model.parameters()
            if p.requires_grad
        ), f"[{model_name}] No parameter gradients"


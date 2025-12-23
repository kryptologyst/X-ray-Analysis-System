"""Configuration management for X-ray analysis system."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from omegaconf import OmegaConf
import yaml
import os


@dataclass
class DataConfig:
    """Data configuration parameters."""
    batch_size: int = 16
    num_workers: int = 4
    image_size: int = 224
    num_channels: int = 3
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    augmentation: bool = True
    normalize: bool = True
    synthetic_data: bool = True
    data_path: str = "data/raw"
    cache_dir: str = "data/processed"


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    architecture: str = "resnet18"  # resnet18, efficientnet_b0, vit_tiny
    pretrained: bool = True
    num_classes: int = 2
    dropout: float = 0.2
    freeze_backbone: bool = False
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    epochs: int = 50
    early_stopping_patience: int = 10
    save_best: bool = True
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    device: str = "auto"  # auto, cuda, mps, cpu
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    scheduler: str = "cosine"  # cosine, step, plateau


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters."""
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "auroc", "auprc", "sensitivity", "specificity", 
        "ppv", "npv", "f1", "calibration"
    ])
    threshold: float = 0.5
    calibration_bins: int = 10
    save_predictions: bool = True
    output_dir: str = "assets/evaluation"


@dataclass
class ExplainabilityConfig:
    """Explainability configuration parameters."""
    methods: List[str] = field(default_factory=lambda: ["gradcam", "scorecam"])
    save_visualizations: bool = True
    output_dir: str = "assets/explainability"
    alpha: float = 0.4


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    
    # Global settings
    seed: int = 42
    deterministic: bool = True
    project_name: str = "xray_analysis"
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Post-initialization setup."""
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.evaluation, dict):
            self.evaluation = EvaluationConfig(**self.evaluation)
        if isinstance(self.explainability, dict):
            self.explainability = ExplainabilityConfig(**self.explainability)

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = OmegaConf.structured(self)
        OmegaConf.save(config_dict, config_path)

    def update(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or return default."""
    if config_path and os.path.exists(config_path):
        return Config.from_yaml(config_path)
    return get_default_config()

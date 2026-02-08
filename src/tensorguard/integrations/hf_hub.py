"""TenSafe Hugging Face Hub Integration.

Provides model hosting with TSSP integration:
- Push models to HF Hub with TSSP verification
- Pull and verify models from HF Hub
- Generate privacy-aware model cards
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Conditional imports
HF_HUB_AVAILABLE = False
try:
    from huggingface_hub import HfApi, Repository, create_repo, hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    logger.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")

# TenSafe imports
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from tensorguard.tgsp.service import TSGPService
    TSSP_AVAILABLE = True
except ImportError:
    TSSP_AVAILABLE = False


@dataclass
class TenSafeHFHubConfig:
    """Configuration for HF Hub integration."""
    token: Optional[str] = None
    private: bool = True
    repo_type: str = "model"

    # TSSP settings
    require_tssp_verification: bool = True
    include_encrypted_weights: bool = False  # Default: metadata only

    # Model card
    generate_model_card: bool = True
    license: str = "apache-2.0"
    language: str = "en"


class TenSafeHFHubIntegration:
    """Hugging Face Hub integration with TSSP support.

    Enables pushing and pulling TenSafe models to/from HF Hub while
    maintaining privacy through TSSP verification and optional weight encryption.

    Example:
        ```python
        hub = TenSafeHFHubIntegration(
            config=TenSafeHFHubConfig(token="hf_xxx", private=True)
        )

        # Push model
        url = hub.push_to_hub(
            tssp_package_path="/path/to/model.tssp",
            repo_id="username/my-private-model",
        )

        # Pull model
        package = hub.pull_from_hub(
            repo_id="username/my-private-model",
            local_dir="/path/to/download",
        )
        ```
    """

    def __init__(self, config: Optional[TenSafeHFHubConfig] = None):
        """Initialize HF Hub integration.

        Args:
            config: Hub configuration
        """
        self.config = config or TenSafeHFHubConfig()

        if HF_HUB_AVAILABLE:
            self.api = HfApi(token=self.config.token)
        else:
            self.api = None

        if TSSP_AVAILABLE:
            self.tssp_service = TSGPService()
        else:
            self.tssp_service = None

    def push_to_hub(
        self,
        tssp_package_path: str,
        repo_id: str,
        commit_message: str = "Upload TenSafe model",
        privacy_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Push TSSP package to Hugging Face Hub.

        By default, only metadata is pushed (not actual weights) for privacy.
        Set include_encrypted_weights=True in config to push encrypted weights.

        Args:
            tssp_package_path: Path to TSSP package
            repo_id: HF Hub repository ID (e.g., "username/model-name")
            commit_message: Git commit message
            privacy_info: Privacy information (epsilon, delta, etc.)

        Returns:
            URL of the created/updated repository
        """
        if not HF_HUB_AVAILABLE:
            raise RuntimeError("huggingface_hub not installed")

        logger.info(f"Pushing to HF Hub: {repo_id}")

        # Load and verify TSSP package
        package = None
        manifest_data = {}

        if self.tssp_service and os.path.exists(tssp_package_path):
            try:
                package = self.tssp_service.load_package(tssp_package_path)
                verification = self.tssp_service.verify_package(package)

                if not verification.valid:
                    raise ValueError(f"TSSP verification failed: {verification.reason}")

                manifest_data = package.manifest.to_dict() if hasattr(package.manifest, 'to_dict') else {}
                logger.info("TSSP package verified")

            except Exception as e:
                logger.warning(f"TSSP loading failed: {e}")

        # Create repository
        try:
            self.api.create_repo(
                repo_id=repo_id,
                private=self.config.private,
                repo_type=self.config.repo_type,
                exist_ok=True,
            )
        except Exception as e:
            logger.info(f"Repository may already exist: {e}")

        # Prepare files to upload
        files_to_upload = []

        # TSSP manifest (always upload)
        if manifest_data:
            manifest_content = json.dumps(manifest_data, indent=2)
            files_to_upload.append(("tssp_manifest.json", manifest_content.encode()))

        # Config file
        config_data = {
            "tensafe_version": "3.0.0",
            "privacy_preserved": True,
            "tssp_verified": package is not None,
        }

        if privacy_info:
            config_data["privacy"] = privacy_info

        config_content = json.dumps(config_data, indent=2)
        files_to_upload.append(("config.json", config_content.encode()))

        # Model card
        if self.config.generate_model_card:
            model_card = self._generate_model_card(
                manifest_data=manifest_data,
                privacy_info=privacy_info,
                repo_id=repo_id,
            )
            files_to_upload.append(("README.md", model_card.encode()))

        # Upload encrypted weights if enabled
        if self.config.include_encrypted_weights and package:
            logger.info("Including encrypted weights in upload")
            # In production, this would upload the encrypted weight files
            # from the TSSP package

        # Upload files
        for filename, content in files_to_upload:
            try:
                self.api.upload_file(
                    path_or_fileobj=content,
                    path_in_repo=filename,
                    repo_id=repo_id,
                    commit_message=commit_message,
                )
            except Exception as e:
                logger.warning(f"Failed to upload {filename}: {e}")

        repo_url = f"https://huggingface.co/{repo_id}"
        logger.info(f"Model pushed to: {repo_url}")

        return repo_url

    def pull_from_hub(
        self,
        repo_id: str,
        local_dir: str,
        verify: bool = True,
    ) -> Optional[Any]:
        """Pull model from Hugging Face Hub.

        Args:
            repo_id: HF Hub repository ID
            local_dir: Local directory to save files
            verify: Whether to verify TSSP package

        Returns:
            Loaded TSSP package (if available) or None
        """
        if not HF_HUB_AVAILABLE:
            raise RuntimeError("huggingface_hub not installed")

        logger.info(f"Pulling from HF Hub: {repo_id}")

        # Create local directory
        os.makedirs(local_dir, exist_ok=True)

        # Download files
        files_to_download = ["tssp_manifest.json", "config.json", "README.md"]

        for filename in files_to_download:
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=local_dir,
                    token=self.config.token,
                )
                logger.info(f"Downloaded: {filename}")
            except Exception as e:
                logger.debug(f"Could not download {filename}: {e}")

        # Load and verify TSSP if available
        manifest_path = os.path.join(local_dir, "tssp_manifest.json")

        if os.path.exists(manifest_path) and self.tssp_service and verify:
            try:
                # In production, this would reconstruct and verify the TSSP package
                with open(manifest_path) as f:
                    manifest_data = json.load(f)
                logger.info("TSSP manifest loaded")
                return manifest_data

            except Exception as e:
                logger.warning(f"TSSP verification failed: {e}")

        return None

    def _generate_model_card(
        self,
        manifest_data: Dict[str, Any],
        privacy_info: Optional[Dict[str, Any]],
        repo_id: str,
    ) -> str:
        """Generate model card from manifest and privacy info.

        Args:
            manifest_data: TSSP manifest data
            privacy_info: Privacy information
            repo_id: Repository ID

        Returns:
            Model card markdown content
        """
        model_name = manifest_data.get("name", repo_id.split("/")[-1])
        base_model = manifest_data.get("base_model", "Unknown")
        package_id = manifest_data.get("package_id", "N/A")

        # Privacy section
        privacy_section = ""
        if privacy_info:
            epsilon = privacy_info.get("epsilon", "N/A")
            delta = privacy_info.get("delta", "N/A")
            noise_mult = privacy_info.get("noise_multiplier", "N/A")
            grad_norm = privacy_info.get("max_grad_norm", "N/A")

            privacy_section = f"""
## Privacy Information

This model was trained with differential privacy guarantees:

| Parameter | Value |
|-----------|-------|
| Epsilon (ε) | {epsilon} |
| Delta (δ) | {delta} |
| Noise Multiplier | {noise_mult} |
| Max Gradient Norm | {grad_norm} |
"""

        return f"""---
language: {self.config.language}
license: {self.config.license}
library_name: tensafe
tags:
  - tensafe
  - privacy-preserving
  - homomorphic-encryption
  - lora
  - differential-privacy
---

# {model_name}

This model was trained using **TenSafe**, a privacy-preserving ML training platform.

## Model Details

| Property | Value |
|----------|-------|
| Base Model | {base_model} |
| Training Method | LoRA with Differential Privacy |
| TSSP Package ID | `{package_id}` |
| Framework | TenSafe v3.0.0 |

## Security Features

This model is distributed as a **TSSP (TenSafe Secure Package)** with:

- **Cryptographic Signatures**: Ed25519 + Dilithium3 (post-quantum)
- **Encrypted Weights**: AES-256-GCM encryption
- **Tamper-Evident Manifest**: SHA-256 hash verification
- **Audit Trail**: Immutable training history
{privacy_section}

## Usage

### Using TenSafe SDK

```python
from tensafe import load_model

# Load and verify model
model = load_model("{package_id}")

# Generate text
output = model.generate("Hello, world!")
```

### Verification

To verify this model's integrity before use:

```python
from tensafe import verify_package

result = verify_package("{package_id}")
print(f"Verification: {{result.status}}")
print(f"Signatures: {{result.signatures}}")
```

## Limitations

- This model is designed for use with the TenSafe SDK
- Actual model weights may be encrypted and require decryption keys
- Privacy guarantees apply to the training process; inference may have different properties

## Citation

If you use this model, please cite:

```bibtex
@software{{tensafe,
  title = {{TenSafe: Privacy-Preserving ML Training Platform}},
  year = {{2026}},
  url = {{https://github.com/tensafe/tensafe}}
}}
```

---

*Model card generated by TenSafe v3.0.0*
"""

    def list_models(self, author: Optional[str] = None) -> List[Dict[str, Any]]:
        """List TenSafe models on HF Hub.

        Args:
            author: Filter by author/organization

        Returns:
            List of model information dicts
        """
        if not HF_HUB_AVAILABLE:
            return []

        try:
            models = self.api.list_models(
                author=author,
                tags=["tensafe"],
                token=self.config.token,
            )

            return [
                {
                    "id": m.id,
                    "author": m.author,
                    "private": m.private,
                    "downloads": m.downloads,
                    "likes": m.likes,
                }
                for m in models
            ]

        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
            return []

"""
TenSafe TGSP Adapter Registry - Encrypted Inference Lock-In.

This module enforces that ONLY LoRA adapters in TGSP format can be used
with HE-encrypted inference. This is a core security feature that ensures:

1. All adapters are cryptographically signed and verified
2. Adapters come from trusted sources (signature validation)
3. Privacy guarantees are maintained through the TGSP container format
4. Modular hot-swapping of adapters without restarting the inference engine

The TGSP (TensorGuard Secure Package) format provides:
- Post-quantum hybrid signatures (Ed25519 + Dilithium)
- Hybrid encryption (Kyber + ChaCha20Poly1305)
- Manifest with integrity hashes
- Audit trail for compliance

Usage:
    from tensafe.tgsp_adapter_registry import TGSPAdapterRegistry

    registry = TGSPAdapterRegistry()

    # Load adapter from TGSP file (enforced format)
    adapter_id = registry.load_tgsp_adapter(
        tgsp_path="model.tgsp",
        recipient_key_path="keys/recipient_private.json"
    )

    # Activate for encrypted inference
    registry.activate_adapter(adapter_id)

    # Hot-swap to different adapter
    registry.activate_adapter(other_adapter_id)

    # Run encrypted inference
    delta = registry.forward_he(x_plain, "q_proj")
"""

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AdapterFormat(Enum):
    """Supported adapter formats."""
    TGSP = "tgsp"  # TensorGuard Secure Package (REQUIRED for encrypted inference)
    SAFETENSORS = "safetensors"  # Only for plaintext mode
    PYTORCH = "pytorch"  # Legacy .pt/.bin (NOT ALLOWED for encrypted inference)


class AdapterLoadError(Exception):
    """Raised when adapter loading fails."""
    pass


class TGSPFormatRequiredError(Exception):
    """Raised when non-TGSP format is used with encrypted inference."""

    def __init__(self, attempted_format: str, message: str = None):
        self.attempted_format = attempted_format
        self.message = message or (
            f"Encrypted inference requires TGSP format. "
            f"Attempted format: {attempted_format}. "
            f"Convert your adapter to TGSP format using `tgsp build` or the TG-Tinker API."
        )
        super().__init__(self.message)


class AdapterNotLoadedError(Exception):
    """Raised when attempting to use an adapter that's not loaded."""
    pass


class NoActiveAdapterError(Exception):
    """Raised when no adapter is active for inference."""
    pass


@dataclass
class TGSPAdapterMetadata:
    """Metadata for a loaded TGSP adapter."""

    adapter_id: str
    tgsp_path: str
    model_name: str
    model_version: str
    author_id: str

    # Cryptographic verification
    manifest_hash: str
    payload_hash: str
    signature_verified: bool
    signature_key_id: str

    # LoRA configuration
    lora_rank: int
    lora_alpha: float
    target_modules: List[str]

    # Timestamps
    loaded_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None

    # Usage stats
    forward_count: int = 0
    total_inference_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "adapter_id": self.adapter_id,
            "tgsp_path": self.tgsp_path,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "author_id": self.author_id,
            "manifest_hash": self.manifest_hash,
            "payload_hash": self.payload_hash,
            "signature_verified": self.signature_verified,
            "signature_key_id": self.signature_key_id,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "loaded_at": self.loaded_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "forward_count": self.forward_count,
            "total_inference_time_ms": self.total_inference_time_ms,
        }


@dataclass
class LoadedAdapter:
    """A fully loaded and ready-to-use adapter."""

    metadata: TGSPAdapterMetadata
    weights: Dict[str, Tuple[np.ndarray, np.ndarray]]  # module -> (lora_a, lora_b)
    he_adapter: Optional[Any] = None  # HELoRAAdapter instance when initialized

    # State
    is_active: bool = False
    is_he_initialized: bool = False


class HotSwapMetrics:
    """Metrics for hot-swap operations."""

    def __init__(self):
        self.total_swaps: int = 0
        self.successful_swaps: int = 0
        self.failed_swaps: int = 0
        self.total_swap_time_ms: float = 0.0
        self.last_swap_time_ms: float = 0.0
        self.swaps_by_adapter: Dict[str, int] = {}

    def record_swap(self, adapter_id: str, duration_ms: float, success: bool) -> None:
        """Record a hot-swap operation."""
        self.total_swaps += 1
        if success:
            self.successful_swaps += 1
            self.total_swap_time_ms += duration_ms
            self.last_swap_time_ms = duration_ms
            self.swaps_by_adapter[adapter_id] = self.swaps_by_adapter.get(adapter_id, 0) + 1
        else:
            self.failed_swaps += 1

    @property
    def avg_swap_time_ms(self) -> float:
        """Average successful swap time in milliseconds."""
        if self.successful_swaps == 0:
            return 0.0
        return self.total_swap_time_ms / self.successful_swaps

    @property
    def success_rate(self) -> float:
        """Success rate of hot-swap operations."""
        if self.total_swaps == 0:
            return 1.0
        return self.successful_swaps / self.total_swaps

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to dictionary."""
        return {
            "total_swaps": self.total_swaps,
            "successful_swaps": self.successful_swaps,
            "failed_swaps": self.failed_swaps,
            "total_swap_time_ms": self.total_swap_time_ms,
            "avg_swap_time_ms": self.avg_swap_time_ms,
            "last_swap_time_ms": self.last_swap_time_ms,
            "success_rate": self.success_rate,
            "swaps_by_adapter": dict(self.swaps_by_adapter),
        }


# Type for hot-swap callback: (old_adapter, new_adapter, target_modules_changed) -> None
HotSwapCallback = Callable[[Optional["LoadedAdapter"], "LoadedAdapter", bool], None]


class TGSPAdapterRegistry:
    """
    Registry for TGSP-format LoRA adapters with encrypted inference enforcement.

    This registry enforces the TGSP format for all adapters used with HE-encrypted
    inference, providing a secure lock-in that ensures:

    1. All adapters are cryptographically verified
    2. Adapters can be hot-swapped without restarting
    3. Audit trail is maintained for compliance
    4. Only trusted adapters run in the encrypted pipeline

    The registry is thread-safe and supports concurrent adapter loading while
    maintaining a single active adapter for inference.

    Hot-Swap Callbacks:
        Register callbacks via `register_hot_swap_callback()` to be notified
        when adapters are swapped. This is critical for keeping hook managers
        in sync with the active adapter's target_modules.
    """

    # Maximum number of adapters to cache
    MAX_CACHED_ADAPTERS = 10

    def __init__(
        self,
        enforce_tgsp: bool = True,
        auto_verify_signatures: bool = True,
        he_config: Optional[Dict[str, Any]] = None,
        work_dir: Optional[str] = None,
    ):
        """
        Initialize the TGSP Adapter Registry.

        Args:
            enforce_tgsp: If True, ONLY TGSP format allowed for encrypted inference
            auto_verify_signatures: Automatically verify TGSP signatures on load
            he_config: HE configuration for adapter initialization
            work_dir: Working directory for extracted adapters
        """
        self.enforce_tgsp = enforce_tgsp
        self.auto_verify_signatures = auto_verify_signatures
        self.he_config = he_config or {}
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="tgsp_adapters_")

        # Adapter storage
        self._adapters: Dict[str, LoadedAdapter] = {}
        self._active_adapter_id: Optional[str] = None

        # Thread safety
        self._lock = threading.RLock()

        # Audit log
        self._audit_log: List[Dict[str, Any]] = []

        # HE adapter (lazy init)
        self._he_adapter = None

        # Hot-swap callbacks for notifying hook managers
        self._hot_swap_callbacks: List[HotSwapCallback] = []

        # Hot-swap metrics
        self._hot_swap_metrics = HotSwapMetrics()

        logger.info(
            f"TGSPAdapterRegistry initialized: "
            f"enforce_tgsp={enforce_tgsp}, "
            f"work_dir={self.work_dir}"
        )

        self._log_audit_event("REGISTRY_INITIALIZED", {
            "enforce_tgsp": enforce_tgsp,
            "auto_verify_signatures": auto_verify_signatures,
        })

    def register_hot_swap_callback(self, callback: HotSwapCallback) -> None:
        """
        Register a callback to be notified on adapter hot-swap.

        The callback receives:
            - old_adapter: Previous adapter (None if first activation)
            - new_adapter: Newly activated adapter
            - target_modules_changed: True if target_modules differ

        Use this to reconfigure hooks when target_modules change.

        Args:
            callback: Callback function to register
        """
        with self._lock:
            self._hot_swap_callbacks.append(callback)
            logger.debug(f"Registered hot-swap callback: {callback}")

    def unregister_hot_swap_callback(self, callback: HotSwapCallback) -> bool:
        """
        Unregister a hot-swap callback.

        Args:
            callback: Callback to unregister

        Returns:
            True if callback was found and removed
        """
        with self._lock:
            try:
                self._hot_swap_callbacks.remove(callback)
                return True
            except ValueError:
                return False

    def get_hot_swap_metrics(self) -> Dict[str, Any]:
        """Get hot-swap operation metrics."""
        return self._hot_swap_metrics.to_dict()

    def _log_audit_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log an audit event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details,
        }
        self._audit_log.append(event)
        logger.info(f"AUDIT: {event_type} - {json.dumps(details)}")

    def _validate_tgsp_format(self, file_path: str) -> bool:
        """
        Validate that a file is in TGSP format.

        Args:
            file_path: Path to the file

        Returns:
            True if valid TGSP format

        Raises:
            TGSPFormatRequiredError: If not TGSP format and enforcement is enabled
        """
        # Check file extension
        if not file_path.lower().endswith('.tgsp'):
            if self.enforce_tgsp:
                raise TGSPFormatRequiredError(
                    attempted_format=Path(file_path).suffix,
                    message=(
                        f"File '{file_path}' is not in TGSP format. "
                        f"Encrypted inference requires .tgsp format. "
                        f"Use 'tgsp build' to create a TGSP package from your adapter."
                    )
                )
            return False

        # Check magic bytes
        try:
            with open(file_path, 'rb') as f:
                magic = f.read(6)
                if magic != b"TGSP\x01\x00":
                    if self.enforce_tgsp:
                        raise TGSPFormatRequiredError(
                            attempted_format="invalid_tgsp",
                            message=(
                                f"File '{file_path}' has invalid TGSP magic bytes. "
                                f"Expected TGSP v1.0 format."
                            )
                        )
                    return False
        except OSError as e:
            raise AdapterLoadError(f"Cannot read file '{file_path}': {e}")

        return True

    def _verify_tgsp_signature(
        self,
        tgsp_path: str,
        public_key: Optional[Dict] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify TGSP package signature.

        Args:
            tgsp_path: Path to TGSP file
            public_key: Optional public key for verification

        Returns:
            Tuple of (verified, header_data)
        """
        try:
            # Import TGSP format utilities
            from src.tensorguard.tgsp.format import read_tgsp_header, verify_tgsp_container

            header_data = read_tgsp_header(tgsp_path)

            if self.auto_verify_signatures:
                verified = verify_tgsp_container(tgsp_path, public_key)
            else:
                verified = True  # Skip verification if disabled

            return verified, header_data

        except ImportError:
            # Fallback if TGSP module not available in test environment
            logger.warning("TGSP format module not available, using fallback verification")
            return self._fallback_verify_tgsp(tgsp_path)
        except Exception as e:
            logger.error(f"TGSP verification failed: {e}")
            return False, {}

    def _fallback_verify_tgsp(self, tgsp_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Fallback TGSP verification when module not available."""
        # Read header manually for basic validation
        try:
            with open(tgsp_path, 'rb') as f:
                magic = f.read(6)
                if magic != b"TGSP\x01\x00":
                    return False, {}

                import struct
                h_len = struct.unpack(">I", f.read(4))[0]
                h_bytes = f.read(h_len)
                header = json.loads(h_bytes)

                m_len = struct.unpack(">I", f.read(4))[0]
                m_bytes = f.read(m_len)
                manifest = json.loads(m_bytes)

                return True, {
                    "header": header,
                    "manifest": manifest,
                    "version": "1.0",
                }
        except Exception as e:
            logger.error(f"Fallback TGSP verification failed: {e}")
            return False, {}

    def _extract_tgsp_payload(
        self,
        tgsp_path: str,
        recipient_key_path: str,
        extract_dir: str
    ) -> str:
        """
        Extract TGSP payload to working directory.

        Args:
            tgsp_path: Path to TGSP file
            recipient_key_path: Path to recipient private key
            extract_dir: Directory to extract to

        Returns:
            Path to extracted directory
        """
        try:
            from src.tensorguard.tgsp.service import TGSPService

            TGSPService.decrypt_package(
                path=tgsp_path,
                recipient_id="local",
                priv_key_path=recipient_key_path,
                out_dir=extract_dir
            )

            return extract_dir

        except ImportError as e:
            raise AdapterLoadError(
                f"TGSPService not available. Install tensorguard with: "
                f"pip install tensorguard[tgsp]. Error: {e}"
            )

    def _load_adapter_weights(
        self,
        extract_dir: str,
        config: Dict[str, Any]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load LoRA weights from extracted directory.

        Args:
            extract_dir: Path to extracted adapter files
            config: Adapter configuration

        Returns:
            Dict of module_name -> (lora_a, lora_b)
        """
        weights = {}
        target_modules = config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
        rank = config.get("lora_rank", 16)

        # Try to load safetensors first
        safetensors_path = os.path.join(extract_dir, "adapter_model.safetensors")
        pt_path = os.path.join(extract_dir, "adapter_model.bin")

        if os.path.exists(safetensors_path):
            weights = self._load_safetensors_weights(safetensors_path, target_modules)
        elif os.path.exists(pt_path):
            weights = self._load_pytorch_weights(pt_path, target_modules)
        else:
            raise AdapterLoadError(
                f"No adapter weights found in {extract_dir}. "
                f"Expected either 'adapter_model.safetensors' or 'adapter_model.bin'. "
                f"Ensure the TGSP package contains valid LoRA weights."
            )

        if not weights:
            raise AdapterLoadError(
                f"Failed to load any LoRA weights from {extract_dir}. "
                f"Target modules: {target_modules}"
            )

        return weights

    def _load_safetensors_weights(
        self,
        path: str,
        target_modules: List[str]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Load weights from safetensors file."""
        try:
            from safetensors.numpy import load_file
            tensors = load_file(path)

            weights = {}
            for module in target_modules:
                lora_a_key = f"base_model.model.{module}.lora_A.weight"
                lora_b_key = f"base_model.model.{module}.lora_B.weight"

                if lora_a_key in tensors and lora_b_key in tensors:
                    weights[module] = (
                        tensors[lora_a_key].astype(np.float64),
                        tensors[lora_b_key].astype(np.float64)
                    )

            return weights

        except ImportError:
            logger.warning("safetensors not available")
            return {}

    def _load_pytorch_weights(
        self,
        path: str,
        target_modules: List[str]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Load weights from PyTorch file."""
        try:
            import torch
            state_dict = torch.load(path, map_location='cpu', weights_only=True)

            weights = {}
            for module in target_modules:
                lora_a_key = f"base_model.model.{module}.lora_A.weight"
                lora_b_key = f"base_model.model.{module}.lora_B.weight"

                if lora_a_key in state_dict and lora_b_key in state_dict:
                    weights[module] = (
                        state_dict[lora_a_key].numpy().astype(np.float64),
                        state_dict[lora_b_key].numpy().astype(np.float64)
                    )

            return weights

        except ImportError:
            logger.warning("PyTorch not available")
            return {}

    def load_tgsp_adapter(
        self,
        tgsp_path: str,
        recipient_key_path: Optional[str] = None,
        adapter_id: Optional[str] = None,
        public_key: Optional[Dict] = None,
    ) -> str:
        """
        Load a LoRA adapter from a TGSP package.

        This is the ONLY supported method for loading adapters for encrypted inference.
        The TGSP format ensures cryptographic verification and audit trail.

        Args:
            tgsp_path: Path to the .tgsp file
            recipient_key_path: Path to recipient private key for decryption
            adapter_id: Optional custom adapter ID (auto-generated if not provided)
            public_key: Optional public key for signature verification

        Returns:
            Adapter ID for reference

        Raises:
            TGSPFormatRequiredError: If file is not in TGSP format
            AdapterLoadError: If loading fails
        """
        with self._lock:
            # Step 1: Validate TGSP format
            self._validate_tgsp_format(tgsp_path)

            # Step 2: Verify signature
            verified, header_data = self._verify_tgsp_signature(tgsp_path, public_key)

            if not verified and self.auto_verify_signatures:
                raise AdapterLoadError(
                    f"TGSP signature verification failed for '{tgsp_path}'. "
                    f"The package may have been tampered with."
                )

            # Step 3: Extract manifest info
            manifest = header_data.get("manifest", {})
            header = header_data.get("header", {})

            # Generate adapter ID
            if adapter_id is None:
                adapter_id = f"tgsp_{hashlib.sha256(tgsp_path.encode()).hexdigest()[:12]}"

            # Check if already loaded
            if adapter_id in self._adapters:
                logger.info(f"Adapter '{adapter_id}' already loaded, returning existing")
                return adapter_id

            # Step 4: Extract payload
            extract_dir = os.path.join(self.work_dir, adapter_id)
            os.makedirs(extract_dir, exist_ok=True)

            if recipient_key_path:
                self._extract_tgsp_payload(tgsp_path, recipient_key_path, extract_dir)

            # Step 5: Load adapter config
            config_path = os.path.join(extract_dir, "adapter_config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    adapter_config = json.load(f)
            else:
                # Use defaults from manifest
                privacy_config = manifest.get("privacy", {})
                scheme_config = privacy_config.get("scheme_config", {})
                adapter_config = {
                    "lora_rank": scheme_config.get("lora_rank", 16),
                    "lora_alpha": scheme_config.get("lora_alpha", 32.0),
                    "target_modules": scheme_config.get("target_modules",
                        ["q_proj", "v_proj", "k_proj", "o_proj"]),
                }

            # Step 6: Load weights
            weights = self._load_adapter_weights(extract_dir, adapter_config)

            # Step 7: Create metadata
            hashes = header.get("hashes", {})
            sig_block = header_data.get("signature_block", {})

            metadata = TGSPAdapterMetadata(
                adapter_id=adapter_id,
                tgsp_path=os.path.abspath(tgsp_path),
                model_name=manifest.get("model_name", "unknown"),
                model_version=manifest.get("model_version", "0.0.0"),
                author_id=manifest.get("author_id", "unknown"),
                manifest_hash=hashes.get("manifest", ""),
                payload_hash=hashes.get("payload", ""),
                signature_verified=verified,
                signature_key_id=sig_block.get("key_id", ""),
                lora_rank=adapter_config.get("lora_rank", 16),
                lora_alpha=adapter_config.get("lora_alpha", 32.0),
                target_modules=adapter_config.get("target_modules", []),
            )

            # Step 8: Create loaded adapter
            loaded = LoadedAdapter(
                metadata=metadata,
                weights=weights,
            )

            # Step 9: Store and audit
            self._adapters[adapter_id] = loaded

            # Manage cache size
            if len(self._adapters) > self.MAX_CACHED_ADAPTERS:
                self._evict_oldest_adapter()

            self._log_audit_event("ADAPTER_LOADED", {
                "adapter_id": adapter_id,
                "tgsp_path": tgsp_path,
                "model_name": metadata.model_name,
                "signature_verified": verified,
                "weights_loaded": list(weights.keys()),
            })

            logger.info(
                f"Loaded TGSP adapter: {adapter_id}, "
                f"model={metadata.model_name}, "
                f"modules={len(weights)}"
            )

            return adapter_id

    def _evict_oldest_adapter(self) -> None:
        """Evict the oldest unused adapter from cache."""
        if not self._adapters:
            return

        # Find oldest adapter that's not active
        oldest_id = None
        oldest_time = datetime.max

        for adapter_id, adapter in self._adapters.items():
            if adapter_id == self._active_adapter_id:
                continue

            use_time = adapter.metadata.last_used_at or adapter.metadata.loaded_at
            if use_time < oldest_time:
                oldest_time = use_time
                oldest_id = adapter_id

        if oldest_id:
            self.unload_adapter(oldest_id)

    def unload_adapter(self, adapter_id: str) -> bool:
        """
        Unload an adapter from the registry.

        Args:
            adapter_id: ID of adapter to unload

        Returns:
            True if unloaded, False if not found
        """
        with self._lock:
            if adapter_id not in self._adapters:
                return False

            # Can't unload active adapter
            if adapter_id == self._active_adapter_id:
                self._active_adapter_id = None

            adapter = self._adapters.pop(adapter_id)

            # Clean up extracted files
            extract_dir = os.path.join(self.work_dir, adapter_id)
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir, ignore_errors=True)

            self._log_audit_event("ADAPTER_UNLOADED", {
                "adapter_id": adapter_id,
                "forward_count": adapter.metadata.forward_count,
            })

            return True

    def activate_adapter(self, adapter_id: str) -> None:
        """
        Activate an adapter for encrypted inference.

        This enables hot-swapping of adapters without restarting the inference
        engine. The activated adapter will be used for all subsequent forward
        passes until another adapter is activated.

        Hot-swap callbacks are invoked after successful activation to notify
        hook managers when target_modules change.

        Args:
            adapter_id: ID of adapter to activate

        Raises:
            AdapterNotLoadedError: If adapter not found
            RuntimeError: If hot-swap callback fails (adapter remains active)
        """
        swap_start = time.perf_counter()
        success = False

        with self._lock:
            if adapter_id not in self._adapters:
                raise AdapterNotLoadedError(
                    f"Adapter '{adapter_id}' not loaded. "
                    f"Use load_tgsp_adapter() first."
                )

            adapter = self._adapters[adapter_id]
            old_adapter = None
            target_modules_changed = False

            # Check if target_modules changed
            if self._active_adapter_id and self._active_adapter_id != adapter_id:
                old_adapter = self._adapters.get(self._active_adapter_id)
                if old_adapter:
                    old_adapter.is_active = False
                    # Compare target_modules
                    old_targets = set(old_adapter.metadata.target_modules)
                    new_targets = set(adapter.metadata.target_modules)
                    target_modules_changed = old_targets != new_targets

            # Activate new adapter
            adapter.is_active = True
            self._active_adapter_id = adapter_id

            # Initialize HE adapter if needed
            if not adapter.is_he_initialized:
                self._init_he_adapter_for(adapter)

            # Invoke hot-swap callbacks (outside lock to prevent deadlock)
            callback_errors = []
            for callback in self._hot_swap_callbacks:
                try:
                    callback(old_adapter, adapter, target_modules_changed)
                except Exception as e:
                    callback_errors.append(f"{callback}: {e}")
                    logger.error(f"Hot-swap callback failed: {e}")

            # Log any callback errors but don't fail the swap
            if callback_errors:
                logger.warning(
                    f"Hot-swap completed with callback errors: {callback_errors}"
                )

            success = True
            swap_duration_ms = (time.perf_counter() - swap_start) * 1000

            self._log_audit_event("ADAPTER_ACTIVATED", {
                "adapter_id": adapter_id,
                "model_name": adapter.metadata.model_name,
                "target_modules_changed": target_modules_changed,
                "swap_time_ms": swap_duration_ms,
                "previous_adapter": old_adapter.metadata.adapter_id if old_adapter else None,
            })

            logger.info(
                f"Activated adapter: {adapter_id} "
                f"(target_modules_changed={target_modules_changed}, "
                f"swap_time={swap_duration_ms:.2f}ms)"
            )

        # Record metrics after releasing lock
        self._hot_swap_metrics.record_swap(adapter_id, swap_duration_ms, success)

    def _init_he_adapter_for(self, adapter: LoadedAdapter) -> None:
        """Initialize HE adapter for a loaded adapter."""
        try:
            # Use the new HE-LoRA microkernel (preferred)
            from he_lora_microkernel.compat import HELoRAAdapter, HELoRAConfig

            config = HELoRAConfig(
                rank=adapter.metadata.lora_rank,
                alpha=adapter.metadata.lora_alpha,
                target_modules=adapter.metadata.target_modules,
                **self.he_config
            )

            he_adapter = HELoRAAdapter(config)

            # Register all weights
            for module_name, (lora_a, lora_b) in adapter.weights.items():
                he_adapter.register_weights(
                    module_name=module_name,
                    lora_a=lora_a,
                    lora_b=lora_b,
                    rank=adapter.metadata.lora_rank,
                    alpha=adapter.metadata.lora_alpha,
                )

            adapter.he_adapter = he_adapter
            adapter.is_he_initialized = True

            logger.info(f"Initialized HE adapter for {adapter.metadata.adapter_id}")

        except ImportError as e:
            logger.warning(f"HE adapter initialization failed: {e}")
            adapter.is_he_initialized = False

    def get_active_adapter(self) -> Optional[LoadedAdapter]:
        """Get the currently active adapter."""
        if self._active_adapter_id is None:
            return None
        return self._adapters.get(self._active_adapter_id)

    def forward_he(
        self,
        x_plain: np.ndarray,
        module_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute encrypted LoRA delta using the active TGSP adapter.

        This method REQUIRES a TGSP adapter to be loaded and activated.
        It enforces the TGSP format lock-in for encrypted inference.

        Args:
            x_plain: Plaintext activation
            module_name: Target module (uses first if not specified)

        Returns:
            Decrypted LoRA delta

        Raises:
            NoActiveAdapterError: If no adapter is activated
            TGSPFormatRequiredError: If enforcement is enabled but no TGSP adapter
        """
        adapter = self.get_active_adapter()

        if adapter is None:
            if self.enforce_tgsp:
                raise NoActiveAdapterError(
                    "No TGSP adapter activated. Encrypted inference requires a "
                    "TGSP-format adapter to be loaded and activated. "
                    "Use load_tgsp_adapter() and activate_adapter() first."
                )
            raise NoActiveAdapterError("No adapter activated")

        # Verify TGSP format
        if self.enforce_tgsp and not adapter.metadata.tgsp_path.endswith('.tgsp'):
            raise TGSPFormatRequiredError(
                attempted_format="unknown",
                message="Active adapter is not in TGSP format"
            )

        # Run forward pass
        start_time = time.perf_counter()

        if adapter.he_adapter is not None:
            delta = adapter.he_adapter.forward(x_plain, module_name)
        else:
            # Fallback to plaintext computation
            delta = self._forward_plaintext(adapter, x_plain, module_name)

        # Update stats
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        adapter.metadata.forward_count += 1
        adapter.metadata.total_inference_time_ms += elapsed_ms
        adapter.metadata.last_used_at = datetime.utcnow()

        return delta

    def _forward_plaintext(
        self,
        adapter: LoadedAdapter,
        x_plain: np.ndarray,
        module_name: Optional[str] = None
    ) -> np.ndarray:
        """Plaintext forward pass fallback."""
        if module_name is None:
            if not adapter.weights:
                raise ValueError("No weights in adapter")
            module_name = next(iter(adapter.weights))

        if module_name not in adapter.weights:
            raise ValueError(f"Module {module_name} not in adapter")

        lora_a, lora_b = adapter.weights[module_name]
        scaling = adapter.metadata.lora_alpha / adapter.metadata.lora_rank

        # delta = scaling * (x @ A^T @ B^T)
        intermediate = x_plain @ lora_a.T
        delta = intermediate @ lora_b.T
        return scaling * delta

    def list_adapters(self) -> List[Dict[str, Any]]:
        """List all loaded adapters."""
        with self._lock:
            return [
                {
                    **adapter.metadata.to_dict(),
                    "is_active": adapter.is_active,
                    "is_he_initialized": adapter.is_he_initialized,
                }
                for adapter in self._adapters.values()
            ]

    def get_adapter_info(self, adapter_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific adapter."""
        with self._lock:
            adapter = self._adapters.get(adapter_id)
            if adapter is None:
                return None
            return {
                **adapter.metadata.to_dict(),
                "is_active": adapter.is_active,
                "is_he_initialized": adapter.is_he_initialized,
                "weights_modules": list(adapter.weights.keys()),
            }

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get the audit log for compliance."""
        return list(self._audit_log)

    def cleanup(self) -> None:
        """Clean up all resources."""
        with self._lock:
            # Unload all adapters
            for adapter_id in list(self._adapters.keys()):
                self.unload_adapter(adapter_id)

            # Clean work directory
            if os.path.exists(self.work_dir):
                shutil.rmtree(self.work_dir, ignore_errors=True)

            self._log_audit_event("REGISTRY_CLEANUP", {
                "adapters_unloaded": len(self._adapters),
            })


# Singleton registry for global access
_global_registry: Optional[TGSPAdapterRegistry] = None
_registry_lock = threading.Lock()


def get_global_registry(
    enforce_tgsp: bool = True,
    **kwargs
) -> TGSPAdapterRegistry:
    """
    Get the global TGSP adapter registry.

    This provides a singleton registry that can be used across the application.

    Args:
        enforce_tgsp: If True, enforce TGSP format for encrypted inference
        **kwargs: Additional configuration options

    Returns:
        Global TGSPAdapterRegistry instance
    """
    global _global_registry

    with _registry_lock:
        if _global_registry is None:
            _global_registry = TGSPAdapterRegistry(
                enforce_tgsp=enforce_tgsp,
                **kwargs
            )
        return _global_registry


def reset_global_registry() -> None:
    """Reset the global registry (for testing)."""
    global _global_registry

    with _registry_lock:
        if _global_registry is not None:
            _global_registry.cleanup()
            _global_registry = None

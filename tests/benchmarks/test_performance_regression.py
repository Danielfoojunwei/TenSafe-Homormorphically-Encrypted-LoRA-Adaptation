"""
Performance Regression Tests

Pytest-based tests that verify TensorGuard performance meets baseline thresholds.
These tests measure ACTUAL performance of cryptographic operations.

Run with: pytest tests/benchmarks/test_performance_regression.py -v
"""

import json
import time
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any


# Load baseline thresholds
BASELINE_PATH = Path(__file__).parent.parent.parent / "benchmarks" / "baseline.json"


def load_baseline() -> Dict[str, Any]:
    """Load the performance baseline configuration."""
    if not BASELINE_PATH.exists():
        pytest.skip(f"Baseline file not found: {BASELINE_PATH}")
    with open(BASELINE_PATH) as f:
        return json.load(f)


def measure_latency(func, iterations: int = 10, warmup: int = 2) -> Dict[str, float]:
    """
    Measure function latency with warmup and return statistics.
    Returns mean, p95, min, max in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        func()

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    sorted_lat = sorted(latencies)
    p95_idx = int(len(sorted_lat) * 0.95)

    return {
        "mean_ms": sum(latencies) / len(latencies),
        "p95_ms": sorted_lat[min(p95_idx, len(sorted_lat) - 1)],
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }


class TestN2HEPerformance:
    """N2HE encryption performance regression tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up baseline thresholds."""
        self.baseline = load_baseline()
        self.thresholds = self.baseline.get("thresholds", {}).get("n2he_encrypt", {})

    def test_n2he_encrypt_1kb_performance(self):
        """Verify N2HE encryption of 1KB tensor meets performance threshold."""
        import warnings
        warnings.filterwarnings('ignore')
        from tensorguard.core.crypto import N2HEEncryptor

        threshold = self.thresholds.get("1kb", {})
        if not threshold:
            pytest.skip("No threshold defined for 1kb encryption")

        encryptor = N2HEEncryptor(security_level=128)
        test_data = np.random.bytes(1024)  # 1KB

        stats = measure_latency(lambda: encryptor.encrypt(test_data))

        mean_max = threshold.get("mean_ms_max", 50.0)
        p95_max = threshold.get("p95_ms_max", 75.0)

        assert stats["mean_ms"] <= mean_max, (
            f"N2HE encrypt 1KB mean latency {stats['mean_ms']:.2f}ms exceeds threshold {mean_max}ms"
        )
        assert stats["p95_ms"] <= p95_max, (
            f"N2HE encrypt 1KB p95 latency {stats['p95_ms']:.2f}ms exceeds threshold {p95_max}ms"
        )

    def test_n2he_encrypt_10kb_performance(self):
        """Verify N2HE encryption of 10KB tensor meets performance threshold."""
        import warnings
        warnings.filterwarnings('ignore')
        from tensorguard.core.crypto import N2HEEncryptor

        threshold = self.thresholds.get("10kb", {})
        if not threshold:
            pytest.skip("No threshold defined for 10kb encryption")

        encryptor = N2HEEncryptor(security_level=128)
        test_data = np.random.bytes(10 * 1024)  # 10KB

        stats = measure_latency(lambda: encryptor.encrypt(test_data), iterations=5)

        mean_max = threshold.get("mean_ms_max", 500.0)
        p95_max = threshold.get("p95_ms_max", 750.0)

        assert stats["mean_ms"] <= mean_max, (
            f"N2HE encrypt 10KB mean latency {stats['mean_ms']:.2f}ms exceeds threshold {mean_max}ms"
        )


class TestN2HEDecryptPerformance:
    """N2HE decryption performance regression tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up baseline thresholds."""
        self.baseline = load_baseline()
        self.thresholds = self.baseline.get("thresholds", {}).get("n2he_decrypt", {})

    def test_n2he_decrypt_1kb_performance(self):
        """Verify N2HE decryption of 1KB tensor meets performance threshold."""
        import warnings
        warnings.filterwarnings('ignore')
        from tensorguard.core.crypto import N2HEEncryptor

        threshold = self.thresholds.get("1kb", {})
        if not threshold:
            pytest.skip("No threshold defined for 1kb decryption")

        encryptor = N2HEEncryptor(security_level=128)
        test_data = np.random.bytes(1024)
        encrypted = encryptor.encrypt(test_data)

        stats = measure_latency(lambda: encryptor.decrypt(encrypted))

        mean_max = threshold.get("mean_ms_max", 10.0)
        p95_max = threshold.get("p95_ms_max", 15.0)

        assert stats["mean_ms"] <= mean_max, (
            f"N2HE decrypt 1KB mean latency {stats['mean_ms']:.2f}ms exceeds threshold {mean_max}ms"
        )


class TestEd25519Performance:
    """Ed25519 signature performance regression tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up baseline thresholds."""
        self.baseline = load_baseline()
        self.thresholds = self.baseline.get("thresholds", {}).get("ed25519", {})

    def test_ed25519_keygen_performance(self):
        """Verify Ed25519 key generation meets performance threshold."""
        from cryptography.hazmat.primitives.asymmetric import ed25519

        threshold = self.thresholds.get("keygen", {})
        if not threshold:
            pytest.skip("No threshold defined for Ed25519 keygen")

        stats = measure_latency(ed25519.Ed25519PrivateKey.generate, iterations=50)

        mean_max = threshold.get("mean_ms_max", 1.0)
        assert stats["mean_ms"] <= mean_max, (
            f"Ed25519 keygen mean latency {stats['mean_ms']:.3f}ms exceeds threshold {mean_max}ms"
        )

    def test_ed25519_sign_performance(self):
        """Verify Ed25519 signing meets performance threshold."""
        from cryptography.hazmat.primitives.asymmetric import ed25519

        threshold = self.thresholds.get("sign_1024b", {})
        if not threshold:
            pytest.skip("No threshold defined for Ed25519 sign")

        priv_key = ed25519.Ed25519PrivateKey.generate()
        message = np.random.bytes(1024)

        stats = measure_latency(lambda: priv_key.sign(message), iterations=50)

        mean_max = threshold.get("mean_ms_max", 1.0)
        assert stats["mean_ms"] <= mean_max, (
            f"Ed25519 sign mean latency {stats['mean_ms']:.3f}ms exceeds threshold {mean_max}ms"
        )

    def test_ed25519_verify_performance(self):
        """Verify Ed25519 verification meets performance threshold."""
        from cryptography.hazmat.primitives.asymmetric import ed25519

        threshold = self.thresholds.get("verify_1024b", {})
        if not threshold:
            pytest.skip("No threshold defined for Ed25519 verify")

        priv_key = ed25519.Ed25519PrivateKey.generate()
        pub_key = priv_key.public_key()
        message = np.random.bytes(1024)
        signature = priv_key.sign(message)

        stats = measure_latency(lambda: pub_key.verify(signature, message), iterations=50)

        mean_max = threshold.get("mean_ms_max", 1.0)
        assert stats["mean_ms"] <= mean_max, (
            f"Ed25519 verify mean latency {stats['mean_ms']:.3f}ms exceeds threshold {mean_max}ms"
        )


class TestSerializationPerformance:
    """Serialization performance regression tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up baseline thresholds."""
        self.baseline = load_baseline()
        self.thresholds = self.baseline.get("thresholds", {}).get("serialization", {})

    def test_updatepackage_serialize_100kb(self):
        """Verify UpdatePackage serialization of 100KB meets threshold."""
        from tensorguard.core.production import UpdatePackage

        threshold = self.thresholds.get("100kb", {})
        if not threshold:
            pytest.skip("No threshold defined for 100kb serialization")

        payload_bytes = np.random.bytes(100 * 1024)
        pkg = UpdatePackage(
            client_id="test_client",
            delta_tensors={"grad": payload_bytes}
        )

        stats = measure_latency(pkg.serialize, iterations=50)

        mean_max = threshold.get("mean_ms_max", 5.0)
        assert stats["mean_ms"] <= mean_max, (
            f"Serialize 100KB mean latency {stats['mean_ms']:.3f}ms exceeds threshold {mean_max}ms"
        )

    def test_updatepackage_deserialize_100kb(self):
        """Verify UpdatePackage deserialization of 100KB meets threshold."""
        from tensorguard.core.production import UpdatePackage

        threshold = self.thresholds.get("100kb", {})
        if not threshold:
            pytest.skip("No threshold defined for 100kb serialization")

        payload_bytes = np.random.bytes(100 * 1024)
        pkg = UpdatePackage(
            client_id="test_client",
            delta_tensors={"grad": payload_bytes}
        )
        serialized = pkg.serialize()

        stats = measure_latency(lambda: UpdatePackage.deserialize(serialized), iterations=50)

        mean_max = threshold.get("mean_ms_max", 5.0)
        assert stats["mean_ms"] <= mean_max, (
            f"Deserialize 100KB mean latency {stats['mean_ms']:.3f}ms exceeds threshold {mean_max}ms"
        )


class TestLWECiphertextPerformance:
    """LWE ciphertext serialization performance tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up baseline thresholds."""
        import warnings
        warnings.filterwarnings('ignore')
        self.baseline = load_baseline()
        self.thresholds = self.baseline.get("thresholds", {}).get("lwe_ciphertext", {})

    def test_lwe_ciphertext_serialize_batch100(self):
        """Verify LWE ciphertext batch serialization meets threshold."""
        import warnings
        warnings.filterwarnings('ignore')
        from tensorguard.core.crypto import LWECiphertext, N2HEParams
        import secrets

        threshold = self.thresholds.get("batch_100", {})
        if not threshold:
            pytest.skip("No threshold defined for batch_100")

        params = N2HEParams()
        seed = secrets.token_bytes(32)
        b = np.random.randint(0, params.q, size=100, dtype=np.int64)
        ct = LWECiphertext(b=b, seed=seed, params=params)

        stats = measure_latency(ct.serialize, iterations=100)

        mean_max = threshold.get("serialize_mean_ms_max", 1.0)
        assert stats["mean_ms"] <= mean_max, (
            f"LWE serialize batch 100 mean latency {stats['mean_ms']:.4f}ms exceeds threshold {mean_max}ms"
        )

    def test_lwe_ciphertext_deserialize_batch100(self):
        """Verify LWE ciphertext batch deserialization meets threshold."""
        import warnings
        warnings.filterwarnings('ignore')
        from tensorguard.core.crypto import LWECiphertext, N2HEParams
        import secrets

        threshold = self.thresholds.get("batch_100", {})
        if not threshold:
            pytest.skip("No threshold defined for batch_100")

        params = N2HEParams()
        seed = secrets.token_bytes(32)
        b = np.random.randint(0, params.q, size=100, dtype=np.int64)
        ct = LWECiphertext(b=b, seed=seed, params=params)
        serialized = ct.serialize()

        stats = measure_latency(lambda: LWECiphertext.deserialize(serialized, params), iterations=50)

        mean_max = threshold.get("deserialize_mean_ms_max", 5.0)
        assert stats["mean_ms"] <= mean_max, (
            f"LWE deserialize batch 100 mean latency {stats['mean_ms']:.4f}ms exceeds threshold {mean_max}ms"
        )


# Helper function for CI summary
def generate_performance_report():
    """Generate a performance report from the latest test run."""
    import subprocess
    result = subprocess.run(
        ["pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    return result.stdout + result.stderr

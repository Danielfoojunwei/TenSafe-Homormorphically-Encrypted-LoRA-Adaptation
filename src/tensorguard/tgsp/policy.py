
import yaml

from .crypto import get_sha256


def verify_policy_hash(policy_path: str, expected_hash: str) -> bool:
    with open(policy_path, 'rb') as f:
        actual_hash = get_sha256(f.read())
    return actual_hash == expected_hash

def get_policy_metadata(policy_path: str):
    with open(policy_path, encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return {
        "id": data.get("id", "unknown"),
        "version": data.get("version", "1.0.0")
    }

# TensorGuard Enterprise Key Management (HSM Integration)

This guide documents how to integrate TensorGuard with Hardware Security Modules (HSMs) and cloud-based Key Management Services (KMS) for enterprise deployments.

## Overview

TensorGuard's cryptographic operations rely on secure key management. While the SDK provides local key storage by default, production deployments should use HSM-backed storage for:

- **FIPS 140-2 Level 3 compliance**
- **Key material never leaves secure hardware**
- **Audit logging and access control**
- **Automatic key rotation**

## Supported Integrations

| Provider | Service | Integration Method |
|:---------|:--------|:-------------------|
| AWS | AWS KMS | Envelope encryption via boto3 |
| Azure | Azure Key Vault | REST API or Azure SDK |
| GCP | Cloud KMS | gRPC or REST API |
| On-Premise | Thales Luna HSM | PKCS#11 interface |
| On-Premise | Entrust nShield | PKCS#11 interface |

---

## AWS KMS Integration

### Prerequisites

```bash
pip install boto3
```

### Configuration

```python
from tensorguard.core.crypto import N2HEEncryptor
import boto3
import base64

class AWSKMSEncryptor(N2HEEncryptor):
    """
    HSM-backed encryptor using AWS KMS for key wrapping.
    The N2HE secret key is encrypted by a Customer Master Key (CMK).
    """
    
    def __init__(self, cmk_id: str, region: str = "us-east-1"):
        self.kms = boto3.client('kms', region_name=region)
        self.cmk_id = cmk_id
        super().__init__()
    
    def _wrap_key(self, plaintext_key: bytes) -> bytes:
        """Encrypt the N2HE key with AWS KMS CMK."""
        response = self.kms.encrypt(
            KeyId=self.cmk_id,
            Plaintext=plaintext_key
        )
        return response['CiphertextBlob']
    
    def _unwrap_key(self, wrapped_key: bytes) -> bytes:
        """Decrypt the N2HE key using AWS KMS."""
        response = self.kms.decrypt(
            CiphertextBlob=wrapped_key
        )
        return response['Plaintext']
    
    def save_key(self, path: str):
        """Save wrapped key to disk (safe - CMK required to decrypt)."""
        import numpy as np
        key_bytes = self._ctx.lwe_key.tobytes()
        wrapped = self._wrap_key(key_bytes)
        with open(path, 'wb') as f:
            f.write(wrapped)
    
    def load_key(self, path: str):
        """Load and unwrap key from disk."""
        import numpy as np
        with open(path, 'rb') as f:
            wrapped = f.read()
        key_bytes = self._unwrap_key(wrapped)
        self._ctx.lwe_key = np.frombuffer(key_bytes, dtype=np.int64)
```

### Usage

```python
# Initialize with your CMK ARN
encryptor = AWSKMSEncryptor(
    cmk_id="arn:aws:kms:us-east-1:123456789:key/abcd-1234-efgh"
)

# Keys are now wrapped by AWS KMS before storage
encryptor.save_key("keys/enterprise_key.wrapped")
```

---

## Azure Key Vault Integration

### Prerequisites

```bash
pip install azure-identity azure-keyvault-keys
```

### Configuration

```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.keys import KeyClient
from azure.keyvault.keys.crypto import CryptographyClient, EncryptionAlgorithm

class AzureKeyVaultEncryptor(N2HEEncryptor):
    """HSM-backed encryptor using Azure Key Vault."""
    
    def __init__(self, vault_url: str, key_name: str):
        credential = DefaultAzureCredential()
        self.key_client = KeyClient(vault_url, credential)
        self.key = self.key_client.get_key(key_name)
        self.crypto_client = CryptographyClient(self.key, credential)
        super().__init__()
    
    def _wrap_key(self, plaintext_key: bytes) -> bytes:
        result = self.crypto_client.encrypt(
            EncryptionAlgorithm.rsa_oaep_256,
            plaintext_key
        )
        return result.ciphertext
    
    def _unwrap_key(self, wrapped_key: bytes) -> bytes:
        result = self.crypto_client.decrypt(
            EncryptionAlgorithm.rsa_oaep_256,
            wrapped_key
        )
        return result.plaintext
```

---

## GCP Cloud KMS Integration

### Prerequisites

```bash
pip install google-cloud-kms
```

### Configuration

```python
from google.cloud import kms

class GCPKMSEncryptor(N2HEEncryptor):
    """HSM-backed encryptor using GCP Cloud KMS."""
    
    def __init__(self, project_id: str, location: str, keyring: str, key_name: str):
        self.client = kms.KeyManagementServiceClient()
        self.key_path = self.client.crypto_key_path(
            project_id, location, keyring, key_name
        )
        super().__init__()
    
    def _wrap_key(self, plaintext_key: bytes) -> bytes:
        response = self.client.encrypt(
            request={'name': self.key_path, 'plaintext': plaintext_key}
        )
        return response.ciphertext
    
    def _unwrap_key(self, wrapped_key: bytes) -> bytes:
        response = self.client.decrypt(
            request={'name': self.key_path, 'ciphertext': wrapped_key}
        )
        return response.plaintext
```

---

## Security Best Practices

1. **Never store unwrapped keys on disk** - Always use envelope encryption
2. **Enable key rotation** - Configure automatic 90-day rotation in your KMS
3. **Use separate keys per environment** - Dev, staging, production
4. **Audit all key access** - Enable CloudTrail/Azure Monitor/Cloud Audit Logs
5. **Implement break-glass procedures** - Document emergency key recovery

## Compliance Mapping

| Requirement | TensorGuard + HSM Solution |
|:------------|:---------------------------|
| HIPAA §164.312(a)(2)(iv) | AES-256 encryption via KMS + N2HE transport |
| SOC 2 CC6.1 | HSM-backed key storage, audit logging |
| GDPR Art. 32 | Pseudonymization via DP, encryption at rest |
| ISO 27001 A.10.1 | Cryptographic controls via KMS policies |

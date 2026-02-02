"""
Protocol Buffer definitions for HE-LoRA services.

Contains gRPC service definitions for:
- HAS (HE Adapter Service) - HE computation service
- ARIS (Adapter Registry + Integrity Service) - Adapter management
"""

# Note: Actual protobuf files should be compiled using:
#   python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. has.proto

# For development, we provide Python-only mock implementations
from .has_pb2 import *
from .has_pb2_grpc import *

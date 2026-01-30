
import os
import shutil
import tempfile
from argparse import Namespace

from .cli import run_build, run_open


class TGSPService:
    @staticmethod
    def create_package(out_path, signing_key_path=None, payloads=None, policy_path=None, recipients=None, evidence_report=None, base_models=None):
        """Legacy shim for TGSPService."""
        with tempfile.TemporaryDirectory() as tmp_in:
            if payloads:
                for p in payloads:
                    # id:type:path
                    parts = p.split(":", 2)
                    if len(parts) == 3:
                        shutil.copy(parts[2], os.path.join(tmp_in, os.path.basename(parts[2])))

            if policy_path:
                shutil.copy(policy_path, os.path.join(tmp_in, "policy.yaml"))

            # Extract paths from recipients (format: id:path)
            recipient_paths = []
            if recipients:
                for r in recipients:
                    if ":" in r:
                        recipient_paths.append(r.split(":", 1)[1])
                    else:
                        recipient_paths.append(r)

            new_args = Namespace(
                input_dir=tmp_in,
                out=out_path,
                model_name="tgsp-service-package",
                model_version="0.0.1",
                recipients=recipient_paths,
                signing_key=signing_key_path
            )

            run_build(new_args)
            return "legacy-pkg-id"

    @staticmethod
    def verify_package(path, public_key=None):
        from .format import verify_tgsp_container
        if verify_tgsp_container(path, public_key):
            return True, "OK"
        else:
            return False, "Signature invalid or container corrupted"

    @staticmethod
    def decrypt_package(path, recipient_id, priv_key_path, out_dir):
        new_args = Namespace(
            file=path,
            key=priv_key_path,
            out_dir=out_dir
        )
        run_open(new_args)
        return True

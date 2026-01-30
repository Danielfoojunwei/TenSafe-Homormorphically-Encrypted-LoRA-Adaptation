import logging
import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..tgsp.service import TGSPService
from ..utils.exceptions import TGSPClientError

logger = logging.getLogger(__name__)

class TGSPEdgeClient:
    """
    Edge-side SDK for interacting with TensorGuard TGSP Platform.
    Handles artifact pulling, local verification, and secure decryption.
    """
    def __init__(self, server_url: str, timeout: int = 30):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

        # Setup session with retries for robustness
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get_latest_package(self, fleet_id: str, channel: str = "stable"):
        """Pull metadata for the active package assigned to this fleet."""
        url = f"{self.server_url}/api/community/tgsp/fleets/{fleet_id}/current?channel={channel}"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch package metadata: {e}")
            raise TGSPClientError(f"Platform communication failure: {e}")

    def download_to_temp(self, pkg_meta: dict, temp_dir: str = "/tmp") -> str:
        """
        Download the physical .tgsp file. 
        In Community Mode, we assume the server provides a download URL or path.
        """
        # For simplicity in this env, we use the storage_path if local,
        # but in a real system we would use a download endpoint.
        local_src = pkg_meta.get('storage_path')
        if not local_src or not os.path.exists(local_src):
             raise FileNotFoundError(f"Package file not found at {local_src}")

        dest = os.path.join(temp_dir, pkg_meta['filename'])
        import shutil
        shutil.copy2(local_src, dest)
        return dest

    def verify_and_extract(self, package_path: str, recipient_id: str, recipient_key_path: str, out_dir: str):
        """
        Full edge lifecycle: Verify -> Decrypt -> Stage
        """
        # 1. Local Verify (Signature + File Hashes)
        ok, msg = TGSPService.verify_package(package_path)
        if not ok:
            raise ValueError(f"TGSP verification failed: {msg}")

        # 2. Decrypt to workdir
        TGSPService.decrypt_package(package_path, recipient_id, recipient_key_path, out_dir)
        return True

def cli_main():
    import argparse
    parser = argparse.ArgumentParser(description="TensorGuard Edge CLI")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--fleet-id", required=True)
    parser.add_argument("--recipient-id", required=True)
    parser.add_argument("--recipient-key", required=True)
    parser.add_argument("--outdir", default="./edge_artifacts")

    args = parser.parse_args()
    client = TGSPEdgeClient(args.server)

    try:
        print(f"Checking for updates for fleet {args.fleet_id}...")
        pkg = client.get_latest_package(args.fleet_id)
        if not pkg:
            print("No active package.")
            return

        print(f"Downloading {pkg['filename']}...")
        os.makedirs("tmp_edge", exist_ok=True)
        pkg_path = client.download_to_temp(pkg, "tmp_edge")

        print("Verifying and extracting...")
        client.verify_and_extract(pkg_path, args.recipient_id, args.recipient_key, args.outdir)
        print(f"Success. Artifacts staged in {args.outdir}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    cli_main()

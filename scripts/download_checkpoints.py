"""Download and verify checkpoint assets listed in checkpoints/manifest.yaml."""

import argparse
import contextlib
import os
import socket
import sys
import time
import urllib.error
import urllib.request

import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(PROJECT_ROOT, "checkpoints", "manifest.yaml")


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def join_url(base_url, relative_url):
    return base_url.rstrip("/") + "/" + relative_url.lstrip("/")


@contextlib.contextmanager
def force_ipv4_resolution(enabled):
    if not enabled:
        yield
        return

    original_getaddrinfo = socket.getaddrinfo

    def getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
        return original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = getaddrinfo_ipv4
    try:
        yield
    finally:
        socket.getaddrinfo = original_getaddrinfo


def select_assets(manifest, group, ids):
    assets = manifest.get("assets", [])
    if ids:
        wanted = set(ids)
        unknown = wanted - {asset["id"] for asset in assets}
        if unknown:
            raise ValueError("Unknown checkpoint IDs: " + ", ".join(sorted(unknown)))
        assets = [a for a in assets if a["id"] in wanted]
    elif group != "all":
        assets = [a for a in assets if a.get("group") == group]
    return assets


def verify_asset(asset, root):
    path = os.path.join(root, asset["path"])
    verify_path(asset, path)
    print(f"verified {asset['id']}: {path}")


def verify_path(asset, path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    expected_size = int(asset["size_bytes"])
    actual_size = os.path.getsize(path)
    if actual_size != expected_size:
        raise RuntimeError(f"{asset['id']}: size mismatch, expected {expected_size}, got {actual_size}")


def download_url(url, tmp_path, timeout, force_ipv4=False, chunk_size=1024 * 1024):
    request = urllib.request.Request(url, headers={"User-Agent": "BridgeTwist-supplement-downloader/1.0"})
    with force_ipv4_resolution(force_ipv4):
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)


def download_asset(asset, root, base_url, force=False, retries=3, timeout=60, force_ipv4=False, verify_download=False):
    path = os.path.join(root, asset["path"])
    if os.path.exists(path) and not force:
        if verify_download:
            try:
                verify_path(asset, path)
            except (OSError, RuntimeError) as exc:
                print(f"existing file failed verification and will be re-downloaded: {exc}", file=sys.stderr)
                os.remove(path)
            else:
                print(f"exists {asset['id']}: {path}")
                return path
        else:
            print(f"exists {asset['id']}: {path}")
            return path

    url = asset.get("url")
    if not url:
        if not base_url:
            raise ValueError(f"{asset['id']} has no URL; pass --base-url <checkpoint root>")
        url = join_url(base_url, asset["relative_url"])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    print(f"downloading {asset['id']} from {url}")
    last_error = None
    for attempt in range(1, retries + 1):
        use_ipv4 = force_ipv4 or attempt > 1
        try:
            if use_ipv4:
                print(f"  attempt {attempt}/{retries}: using IPv4 DNS resolution")
            else:
                print(f"  attempt {attempt}/{retries}")
            download_url(url, tmp_path, timeout=timeout, force_ipv4=use_ipv4)
            if verify_download:
                verify_path(asset, tmp_path)
            break
        except (OSError, RuntimeError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if "Cannot assign requested address" in str(exc):
                print("  connection failed with Errno 99; retrying with IPv4 can fix this in some containers.", file=sys.stderr)
            elif "size mismatch" in str(exc):
                print(f"  downloaded file failed verification: {exc}", file=sys.stderr)
            if attempt == retries:
                raise RuntimeError(
                    f"Failed to download {asset['id']} after {retries} attempts. "
                    "Check that the URL is public and that this machine has outbound HTTPS access. "
                    "If the error mentions Errno 99, try passing --ipv4; if a large file is truncated, "
                    "try increasing --retries."
                ) from last_error
            time.sleep(min(2 ** (attempt - 1), 10))
    os.replace(tmp_path, path)
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Download BridgeTwist checkpoints", allow_abbrev=False,
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--base-url", default=None, help="Checkpoint root URL containing paths from manifest.yaml")
    parser.add_argument("--group", choices=["required", "external", "all"], default="required")
    parser.add_argument("--id", dest="ids", action="append", default=[], help="Download one asset id; can be repeated")
    parser.add_argument("--verify", action="store_true", help="Verify file existence and byte size against the manifest")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60, help="Per-read network timeout in seconds")
    parser.add_argument("--ipv4", action="store_true", help="Force IPv4 DNS resolution for restricted container networks")
    args = parser.parse_args(argv)
    if args.retries < 1:
        parser.error("--retries must be positive")
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    return parser, args


def main(argv=None):
    parser, args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    try:
        assets = select_assets(manifest, args.group, args.ids)
    except ValueError as exc:
        parser.error(str(exc))
    if not assets:
        print("No matching assets.", file=sys.stderr)
        return 1

    for asset in assets:
        path = os.path.join(PROJECT_ROOT, asset["path"])
        if args.dry_run:
            print(f"{asset['id']}: {path}")
            continue
        download_asset(
            asset,
            PROJECT_ROOT,
            args.base_url,
            force=args.force,
            retries=args.retries,
            timeout=args.timeout,
            force_ipv4=args.ipv4,
            verify_download=args.verify,
        )
        if args.verify:
            verify_asset(asset, PROJECT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

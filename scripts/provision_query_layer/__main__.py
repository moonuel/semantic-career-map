"""CLI entry point for the Glue/Athena provisioner (see package __init__)."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from scripts.provision_query_layer import (
    DATABASE_DEFAULT,
    MAX_SCAN_BYTES_DEFAULT,
    WORKGROUP_DEFAULT,
    provision,
)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Provision Glue + Athena over the lake"
    )
    parser.add_argument("--database", default=DATABASE_DEFAULT)
    parser.add_argument("--workgroup", default=WORKGROUP_DEFAULT)
    parser.add_argument("--max-scan-bytes", type=int, default=MAX_SCAN_BYTES_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    provision(args.database, args.workgroup, args.max_scan_bytes, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())

"""CLI entry point for the compute-stack provisioner (see package __init__)."""

from __future__ import annotations

import argparse
import json
import os
import sys

import boto3
from dotenv import load_dotenv

from scripts.provision_compute_stack import (
    LAMBDA_ATS,
    SCHEDULE_SPECS,
    account_id,
    aws_region,
    create_schedules,
    enable_schedules,
    stage_create,
    stage_plan,
    verify,
)


def _parse_image_tags(value: str | None) -> dict[str, str] | None:
    """Parse ``--image-tags scrape:<tag>,curator:<tag>,clean:<tag>``.

    Keys follow the tier vocabulary: ``scrape`` is the Lambda image tag;
    ``curator``/``clean`` override the two Fargate task definitions'
    image tags (the pinned-tag promotion path).
    """
    if not value:
        return None
    tags: dict[str, str] = {}
    for pair in value.split(","):
        tier, _, tag = pair.strip().partition(":")
        if not tier or not tag or tier not in {"scrape", "curator", "clean"}:
            raise SystemExit(
                f"--image-tags entries are scrape|curator|clean:<tag> (got {pair!r})"
            )
        tags[tier] = tag
    return tags


def main() -> None:
    """CLI flow: zero or more stages, in the documented order."""
    parser = argparse.ArgumentParser(
        description="Provision the scrape/clean compute tiers + scheduler"
        " (plans first; apply is explicit)",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="apply roles, DLQ, SNS, ECR, log groups, Lambda, task defs, alarm",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="create the three daily schedules (SMOKE-FIRST: DISABLED state)",
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help="flip the schedules to ENABLED (T5 step 6 — after smoke gates)",
    )
    parser.add_argument(
        "--put-secret",
        action="store_true",
        help="write the KILO_API_KEY env value to SSM (value never printed)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow the secret put to replace an existing parameter",
    )
    parser.add_argument("--email", default=None, help="SNS alert email (subscribes)")
    parser.add_argument(
        "--image-tags",
        default=None,
        help="per-tier image tag overrides, e.g. scrape:<sha>,clean:<sha>",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    parser.add_argument(
        "--verify", action="store_true", help="probe the provisioned resources"
    )
    args = parser.parse_args()

    load_dotenv()  # STORE_S3_* (+ KILO_API_KEY only for --put-secret)
    if not (
        args.create
        or args.schedule
        or args.enable
        or args.put_secret
        or args.verify
        or args.email
    ):
        parser.print_help()
        return

    if args.verify:
        sys.exit(0 if verify() else 1)

    if args.put_secret:
        from scripts.provision_compute_stack import put_secret

        value = os.environ.get("KILO_API_KEY")
        if not value:
            raise SystemExit("--put-secret requires KILO_API_KEY in the environment")
        put_secret(
            boto3.Session(region_name=aws_region()), value, overwrite=args.overwrite
        )
        return

    session = boto3.Session(region_name=aws_region())
    acct = account_id(session)
    region = aws_region()
    image_tags = _parse_image_tags(args.image_tags)

    if args.dry_run:
        print(json.dumps(stage_plan(acct, region), indent=2))
        print("(dry run — no changes made)")
        return

    if args.create:
        stage_create(session, acct, region, email=args.email, image_tags=image_tags)
    if args.schedule:
        from scripts.provision_compute_stack import register_task_definitions

        try:
            session.client("lambda").get_function(FunctionName=LAMBDA_ATS)
        except session.client("lambda").exceptions.ResourceNotFoundException:
            raise SystemExit(
                f"--schedule requires --create first "
                f"(Lambda {LAMBDA_ATS} missing; run --create)"
            ) from None
        lambda_arn = f"arn:aws:lambda:{region}:{acct}:function:{LAMBDA_ATS}"
        task_arns = register_task_definitions(session, image_tags)
        create_schedules(
            session,
            acct,
            region,
            arns={"lambda": lambda_arn, "task_defs": task_arns},
            state="DISABLED",
        )
    if args.enable:
        enable_schedules(session)

    if args.schedule or args.enable:
        print(
            f"schedules ({', '.join(name for name, _, _ in SCHEDULE_SPECS)}):"
            " verify with --verify; smoke gates first (plan T5), then --enable"
        )


if __name__ == "__main__":
    sys.exit(main())

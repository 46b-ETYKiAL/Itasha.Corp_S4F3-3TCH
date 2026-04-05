"""
ComfyUI Workflow Schema CLI

Command-line interface for workflow validation and conversion.
"""

import json
import sys
from pathlib import Path

from .converters import convert_workflow_to_v1
from .validator import validate_workflow_file


def main():
    """Command-line interface for workflow validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate ComfyUI workflow JSON files against the official schema"
    )
    parser.add_argument("file", help="Path to workflow JSON file")
    parser.add_argument(
        "--strict", action="store_true", help="Treat warnings as errors"
    )
    parser.add_argument(
        "--version", type=int, choices=[0, 1], help="Target schema version (0 or 1)"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--convert", action="store_true", help="Convert to V1 and output"
    )

    args = parser.parse_args()

    if args.convert:
        # Convert mode
        file_path = Path(args.file)
        data = json.loads(file_path.read_text(encoding="utf-8"))
        converted = convert_workflow_to_v1(data)
        print(json.dumps(converted, indent=2))
        return 0

    report = validate_workflow_file(
        args.file, strict=args.strict, target_version=args.version
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"\nComfyUI Workflow Validation: {args.file}")
        print("=" * 60)
        print(f"Valid: {report.valid}")
        print(f"Detected Version: {report.detected_version}")
        print(f"Errors: {report.error_count}")
        print(f"Warnings: {report.warning_count}")

        if report.errors:
            print("\nERRORS:")
            for err in report.errors:
                print(f"  - [{err.field}] {err.message}")
                if err.details and "fix" in err.details:
                    print(f"    FIX: {err.details['fix']}")

        if report.warnings:
            print("\nWARNINGS:")
            for warn in report.warnings:
                print(f"  - [{warn.field}] {warn.message}")

        print("=" * 60)

    return 0 if report.valid else 1


if __name__ == "__main__":
    sys.exit(main())

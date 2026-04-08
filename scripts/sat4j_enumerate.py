#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from hybrid_to_pbo import convert_file


JAVA_CLASS_NAME = "Sat4jPbEnumerator"
JAVA_SOURCE_FILE = Path(__file__).resolve().parent / "sat4j" / f"{JAVA_CLASS_NAME}.java"
JAVA_BUILD_DIR = Path(__file__).resolve().parent / "sat4j" / "build"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert (if needed) and enumerate all solutions using incremental Sat4j (Java API)."
    )
    parser.add_argument("input_file", help="Input formula file (.hybrid, .cnf, .opb, .pbo)")
    parser.add_argument("--sat4j-jar", required=True, help="Path to sat4j-pb.jar")
    parser.add_argument(
        "--work-file",
        default=None,
        help=(
            "Optional OPB file path to use as Java input. "
            "For .hybrid/.cnf inputs this stores the converted OPB."
        ),
    )
    parser.add_argument(
        "--max-solutions",
        type=int,
        default=0,
        help="Stop after this many solutions (0 means no limit)",
    )
    parser.add_argument(
        "--orig-vars",
        type=int,
        default=0,
        help=(
            "Count/project models using variables 1..N only (0 means all variables). "
            "Useful when transformed instances introduce auxiliary variables."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Print each found model")
    parser.add_argument(
        "--rebuild-java",
        action="store_true",
        help="Force recompilation of the Java enumerator",
    )
    return parser.parse_args()


def prepare_base_opb(input_path: Path, output_path: Path) -> Path:
    if input_path.suffix.lower() in {".opb", ".pbo"}:
        return input_path
    convert_file(str(input_path), str(output_path))
    return output_path


def needs_compile(class_file: Path, src_file: Path, force: bool) -> bool:
    if force:
        return True
    if not class_file.exists():
        return True
    return src_file.stat().st_mtime > class_file.stat().st_mtime


def compile_java_enumerator(jar_path: Path, rebuild: bool) -> None:
    if not JAVA_SOURCE_FILE.exists():
        raise FileNotFoundError(f"Java source not found: {JAVA_SOURCE_FILE}")

    JAVA_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    class_file = JAVA_BUILD_DIR / f"{JAVA_CLASS_NAME}.class"
    if not needs_compile(class_file, JAVA_SOURCE_FILE, rebuild):
        return

    cmd = [
        "javac",
        "-cp",
        str(jar_path),
        "-d",
        str(JAVA_BUILD_DIR),
        str(JAVA_SOURCE_FILE),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to compile Java enumerator.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def run_java_enumerator(
    jar_path: Path,
    opb_path: Path,
    max_solutions: int,
    orig_vars: int,
    verbose: bool,
) -> int:
    cp = os.pathsep.join([str(JAVA_BUILD_DIR), str(jar_path)])
    cmd = ["java", "-cp", cp, JAVA_CLASS_NAME, str(opb_path)]
    if max_solutions > 0:
        cmd.extend(["--max-solutions", str(max_solutions)])
    if orig_vars > 0:
        cmd.extend(["--orig-vars", str(orig_vars)])
    if verbose:
        cmd.append("--verbose")

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    return proc.returncode


def main() -> int:
    args = parse_args()

    if args.orig_vars < 0:
        print("--orig-vars must be >= 0", file=sys.stderr)
        return 2

    input_path = Path(args.input_file).expanduser().resolve()
    jar_path = Path(args.sat4j_jar).expanduser().resolve()

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2
    if not jar_path.exists():
        print(f"Sat4j jar not found: {jar_path}", file=sys.stderr)
        return 2

    try:
        compile_java_enumerator(jar_path, rebuild=args.rebuild_java)
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    needs_temp_opb = not args.work_file and input_path.suffix.lower() not in {".opb", ".pbo"}

    def run_with_requested_path(requested_opb_path: Path) -> int:
        try:
            opb_path = prepare_base_opb(input_path, requested_opb_path)
        except Exception as exc:
            print(f"Failed to prepare OPB input: {exc}", file=sys.stderr)
            return 2

        rc = run_java_enumerator(
            jar_path=jar_path,
            opb_path=opb_path,
            max_solutions=args.max_solutions,
            orig_vars=args.orig_vars,
            verbose=args.verbose,
        )

        if args.work_file:
            print(f"OPB instance used by Java enumerator: {opb_path}")

        return rc

    if args.work_file:
        requested_opb_path = Path(args.work_file).expanduser().resolve()
        requested_opb_path.parent.mkdir(parents=True, exist_ok=True)
        return run_with_requested_path(requested_opb_path)

    if needs_temp_opb:
        with tempfile.TemporaryDirectory(prefix="sat4j_enum_") as tmpdir:
            return run_with_requested_path(Path(tmpdir) / "instance.opb")

    return run_with_requested_path(input_path)


if __name__ == "__main__":
    raise SystemExit(main())

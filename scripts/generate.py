#!/usr/bin/env python

#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

import argparse
import os
from shutil import which
from subprocess import check_output
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")


class terminal_colors:
    SUCCESS = "\033[92m"
    INFO = "\033[94m"
    WARNING = "\033[33m"
    ERROR = "\033[91m"
    END = "\033[0m"


def generate_android_gradle_help(subparsers):
    parser = subparsers.add_parser(
        "android",
        help="Generate Android Gradle files",
    )

    parser.set_defaults(func=generate_android_gradle)

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Relative Output Directory: <project_dir>/build/android_gradle",
        default=None,
    )


def generate_android_gradle(args):
    output_dir = (
        os.path.join(ROOT_DIR, args.output_dir)
        if args.output_dir
        else os.path.join(ROOT_DIR, "build", "android_gradle")
    )

    if not which("cmake"):
        print("Missing cmake")
        sys.exit(1)

    print(
        terminal_colors.INFO
        + "Generating Android Gradle files at "
        + output_dir
        + terminal_colors.END
    )

    check_output(
        [
            "cmake",
            "-DPROJECT_NAME=vulkan_samples",
            "-DANDROID_API=30",
            "-DARCH_ABI=arm64-v8a",
            "-DANDROID_MANIFEST={}".format(
                os.path.join(ROOT_DIR, "app", "android", "AndroidManifest.xml")
            ),
            "-DJAVA_DIRS={}".format(os.path.join(ROOT_DIR, "app", "android", "java")),
            "-DRES_DIRS={}".format(os.path.join(ROOT_DIR, "app", "android", "res")),
            "-DOUTPUT_DIR={}".format(output_dir),
            "-DASSET_DIRS=",
            "-DJNI_LIBS_DIRS=",
            "-DNATIVE_SCRIPT={}".format(os.path.join(ROOT_DIR, "CMakeLists.txt")),
            "-P",
            os.path.join(ROOT_DIR, "bldsys", "cmake", "create_gradle_project.cmake"),
        ]
    )

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Generate utility.")

    subparsers = argument_parser.add_subparsers(
        help="Commands",
    )

    generate_android_gradle_help(subparsers)

    args = argument_parser.parse_args()

    if len(sys.argv) == 1:
        argument_parser.print_help(sys.stderr)
        sys.exit(1)

    args.func(args)

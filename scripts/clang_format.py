#!/usr/bin/env python

#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

import argparse
import os
import sys
from shutil import which

from subprocess import check_output

# Get the file extension
def get_ext(file_path):
    file_name = os.path.basename(file_path)
    file_name, file_ext = os.path.splitext(file_name)
    return file_ext

class terminal_colors:
    SUCCESS = "\033[92m"
    INFO = "\033[94m"
    WARNING = "\033[33m"
    ERROR = "\033[91m"
    END = "\033[0m"

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(
        description="Format C/C++ files using clang-format"
    )
    argument_parser.add_argument(
        "branch",
        type=str,
        default="main",
        nargs="?",
        help="Branch from which to compute the diff",
    )
    args = argument_parser.parse_args()

    if len(sys.argv) == 1:
        argument_parser.print_help(sys.stderr)
        sys.exit(1)

    files = None

    if not which("git"):
        print(terminal_colors.ERROR + "Missing git" + terminal_colors.END)
        sys.exit(1)

    if not which("clang-format"):
        print(terminal_colors.ERROR + "Missing clang-format" + terminal_colors.END)
        sys.exit(1)

    out = check_output(["git", "diff", args.branch, "--name-only"])

    check_files = [".h", ".hpp", ".cpp"]

    files = out.decode("utf-8").split("\n")
    files = [f for f in files if f and get_ext(f) in check_files]

    if files and len(files) > 0:
        print(terminal_colors.INFO + "Formatting files:" + terminal_colors.END)
        for f in files:
            print(terminal_colors.INFO + "  " + f + terminal_colors.END)
        print()

        for f in files:
            check_output(["clang-format", "-i", f])
    else:
        print(terminal_colors.INFO + "No files to format" + terminal_colors.END)
        
        
        
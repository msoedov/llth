"""
python -m evals                 list lessons
python -m evals l01             evaluate lesson 01, reusing saved reference results
python -m evals l01 fresh       re-run the references too
python -m evals all             evaluate every lesson
"""

import sys

from evals.harness import implementations, lesson_names, load_or_run, report, resolve


def main(args: list[str]) -> None:
    if not args:
        print(__doc__)
        print("\n".join(lesson_names()))
        return
    fresh = "fresh" in args
    names = lesson_names() if args[0] == "all" else [resolve(args[0])]
    for lesson in names:
        print(report(lesson, {impl: load_or_run(lesson, impl, fresh) for impl in implementations(lesson)}), "\n")


if __name__ == "__main__":
    main(sys.argv[1:])

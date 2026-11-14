# Command-line demo of the vision module: watch a clip into a Task, or point at / list / check items in a photo.
"""Try the vision module on files, without a camera or an arm.

    python scripts/vision_demo.py watch data/fixtures/pattern.mp4 --name "chess rows"
    python scripts/vision_demo.py watch clip.mp4 --silent --transcript words.txt
    python scripts/vision_demo.py point photo.jpg "black queen" "white king"
    python scripts/vision_demo.py list photo.jpg
    python scripts/vision_demo.py check photo.jpg "black pawn" "white bowl"

Results go to stdout as JSON, timings to stderr, annotated images under data/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from earshot import config  # noqa: E402
from earshot.vision import point, watch  # noqa: E402


def _save(image, stem: str) -> None:
    """Write an annotated image under data/ and say where it went."""
    config.DATA.mkdir(exist_ok=True)
    out = config.DATA / f"{stem}_annotated.jpg"
    cv2.imwrite(str(out), image)
    print(f"annotated image: {out}", file=sys.stderr)


def run_watch(args: argparse.Namespace) -> None:
    """Watch a clip, print the Task JSON and save the last frame with the place points drawn."""
    transcript = args.transcript
    if transcript and len(transcript) < 256 and Path(transcript).is_file():
        transcript = Path(transcript).read_text()
    if args.silent and not transcript:
        sys.exit("--silent needs --transcript: a clip without sound has no words to go with it")
    task = watch.watch_clip(args.clip, transcript, args.name or Path(args.clip).stem)
    print(task.to_json())

    frame = watch.last_frame(args.clip)
    height, width = frame.shape[:2]
    places = [{"label": s.item, "x": s.place_xy[0] / 1000 * width, "y": s.place_xy[1] / 1000 * height}
              for s in task.steps if s.place_xy]
    if places:
        _save(point.draw(frame, places), Path(args.clip).stem)


def run_point(args: argparse.Namespace) -> None:
    """Point at the named items, or list all loose items, and save an annotated copy."""
    if args.command == "point":
        points = point.point_at(args.image, args.descriptions, model=args.model)
    else:
        points = point.list_items(args.image, model=args.model)
    print(json.dumps(points, indent=2))
    _save(point.draw(args.image, points), f"{Path(args.image).stem}_{args.command}")


def run_check(args: argparse.Namespace) -> None:
    """Ask whether an item is at its destination in the photo."""
    print(json.dumps(point.check_placed(args.image, args.item, args.destination, model=args.model), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    w = commands.add_parser("watch", help="turn a demonstration clip into a Task")
    w.add_argument("clip")
    w.add_argument("--transcript", help="what was said: text, or the path of a text file")
    w.add_argument("--silent", action="store_true", help="the clip has no sound; a transcript is required")
    w.add_argument("--name", help="task name (default: the clip's file name)")
    w.set_defaults(run=run_watch)

    p = commands.add_parser("point", help="point at named items in a photo")
    p.add_argument("image")
    p.add_argument("descriptions", nargs="+")
    p.set_defaults(run=run_point)

    l = commands.add_parser("list", help="list all loose graspable items in a photo")
    l.add_argument("image")
    l.set_defaults(run=run_point)

    c = commands.add_parser("check", help="check whether an item is at its destination")
    c.add_argument("image")
    c.add_argument("item")
    c.add_argument("destination")
    c.set_defaults(run=run_check)

    for sub in (p, l, c):
        sub.add_argument("--model", default=config.POINT_MODEL)

    args = parser.parse_args()
    started = time.time()
    args.run(args)
    print(f"{args.command}: {time.time() - started:.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main()

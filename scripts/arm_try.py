"""First motions on the real arm, one careful step at a time. THE ARM MOVES. Keep a hand near
the power switch, and keep the table clear.

    python scripts/arm_try.py hover 3            # glide to 6 cm above calibration mark 3, wait, go back
    python scripts/arm_try.py touch 3            # same, then lower to grasp height over the mark
    python scripts/arm_try.py pick 640 400 --to 800 420            # pick at a pixel, set down at another
    python scripts/arm_try.py pick 640 400 --to 800 420 --drop     # release from above instead

Run them in that order. Each asks before it moves. --speed 0.2 is the slow default.
"""
import argparse
import json

from earshot import config
from earshot.arm import motion

ap = argparse.ArgumentParser()
ap.add_argument("what", choices=["hover", "touch", "pick"])
ap.add_argument("a", type=float)
ap.add_argument("b", type=float, nargs="?")
ap.add_argument("--to", type=float, nargs=2)
ap.add_argument("--drop", action="store_true")
ap.add_argument("--tall", action="store_true")
ap.add_argument("--speed", type=float, default=0.2)
args = ap.parse_args()

bot = motion.Arm(speed=args.speed)
print("Connected. The arm is holding its pose.")
try:
    if args.what in ("hover", "touch"):
        marks = json.loads(config.CALIB_FILE.read_text())["marks"]
        x, y = marks[int(args.a) - 1]["arm_xy"]
        input(f"Will glide to {motion.ABOVE_TABLE_HOVER * 100:.0f} cm above mark {int(args.a)} "
              f"(x {x * 100:.1f} cm, y {y * 100:.1f} cm). Press Enter to move, Ctrl+C to cancel...")
        lean = bot.move_xyz(x, y, motion.ABOVE_TABLE_HOVER)
        print(f"  gripper leaning {lean:.0f} deg from straight down. Is the tip above the mark? How far off, which way?")
        if args.what == "touch":
            input("Press Enter to lower to grasp height...")
            lean = bot.move_height(motion.ABOVE_TABLE_GRASP)
            print(f"  leaning {lean:.0f} deg. Is the tip just above the table, over the mark? How far off, which way?")
        input("Press Enter to go back to rest...")
    else:
        if args.b is None or args.to is None:
            raise SystemExit("pick needs a pixel and a target: pick U V --to U V")
        input(f"Will pick at pixel ({args.a:.0f}, {args.b:.0f}) and place at ({args.to[0]:.0f}, {args.to[1]:.0f}). "
              "Press Enter to move, Ctrl+C to cancel...")
        bot.pick_place((args.a, args.b), tuple(args.to), drop=args.drop, tall=args.tall)
except KeyboardInterrupt:
    print("\nCancelled.")
finally:
    bot.close()
    print("Motors off.")

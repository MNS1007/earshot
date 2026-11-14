"""Pick a named item the camera can see, end to end. THE ARM MOVES.

    python scripts/arm_pick_item.py "yellow sponge"                 # set it down on the handover spot
    python scripts/arm_pick_item.py "yellow sponge" --to "bowl"     # release it over the bowl
    python scripts/arm_pick_item.py "red block" --to "bowl" --check # and ask the camera if it arrived

The camera finds the item (and the destination), the calibration turns the pixel into a table
position, and the arm does the rest. It asks before moving.
"""
import argparse

import cv2

from earshot import config
from earshot.arm import motion
from earshot.camera import Camera
from earshot.vision import point

ap = argparse.ArgumentParser()
ap.add_argument("item")
ap.add_argument("--to", default=None, help="a container the camera can see; default is the handover spot")
ap.add_argument("--tall", action="store_true", help="the container is tall (a cup)")
ap.add_argument("--speed", type=float, default=0.3)
ap.add_argument("--grip", type=float, default=0.6)
ap.add_argument("--check", action="store_true")
args = ap.parse_args()

cam = Camera().start()
bot = motion.Arm(speed=args.speed)
bot.grip = args.grip
try:
    frame = cam.fresh_frame()
    print(bot.follow_camera(frame))
    grasp = point.grasp_of(frame, args.item)
    if grasp is None:
        raise SystemExit(f'The camera cannot see "{args.item}".')
    pick_px, long_side = (grasp["x"], grasp["y"]), grasp["long_side"]
    if args.to:
        hits = point.point_at(frame, [f"the middle of the opening of the {args.to}"])
        if not hits:
            raise SystemExit(f'The camera cannot see "{args.to}".')
        place_px = (hits[0]["x"], hits[0]["y"])
    else:
        place_px = bot.table.to_pixel(*config.HANDOVER_XY)
    cv2.imwrite(str(config.DATA / "last_pick.jpg"), point.draw(frame, [
        {"label": args.item, "x": pick_px[0], "y": pick_px[1]},
        {"label": args.to or "handover", "x": place_px[0], "y": place_px[1]}] +
        ([{"label": "end", "x": e[0], "y": e[1]} for e in long_side] if long_side else [])))
    px, py = bot.table.to_arm(*pick_px)
    qx, qy = bot.table.to_arm(*place_px)
    print(f'"{args.item}" at pixel ({pick_px[0]:.0f}, {pick_px[1]:.0f}) = {px * 100:.1f} cm out, {py * 100:.1f} cm across')
    print(f'going to {args.to or "the handover spot"} at {qx * 100:.1f} cm out, {qy * 100:.1f} cm across')
    print(f"picture with both points: {config.DATA / 'last_pick.jpg'}")
    input("Press Enter to move, Ctrl+C to cancel...")
    held = bot.pick_place(pick_px, place_px, drop=bool(args.to), tall=args.tall, long_side_px=long_side,
                          outline_px=grasp.get("outline"))
    print("gripper felt the item: carried it" if held else "gripper closed on nothing: missed")
    if args.check:
        r = point.check_placed(cam.fresh_frame(), args.item, args.to or "the table near the front edge")
        print("camera check:", r)
except KeyboardInterrupt:
    print("\nCancelled.")
finally:
    bot.close()
    cam.stop()
    print("Motors off.")

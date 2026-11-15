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
ap.add_argument("--standing", action="store_true", help="a tall item standing up: slide onto it from the side")
args = ap.parse_args()

cam = Camera().start()
bot = motion.Arm(speed=args.speed)
bot.grip = args.grip
try:
    frame = cam.fresh_frame()
    print(bot.follow_camera(frame))
    hits = point.point_at(frame, [args.item])
    if not hits:
        raise SystemExit(f'The camera cannot see "{args.item}".')
    found = (hits[0]["x"], hits[0]["y"])                  # the model's own point on the item
    pick_px = point.near_edge(frame, *found)
    long_side = None
    outline = point.contact_outline(frame, *found)
    print("outline of where it meets the table:", "traced" if outline is not None else "not found (no clear colour)")
    print(f"model pointed at ({hits[0]['x']:.0f}, {hits[0]['y']:.0f}); its near edge on the table is at "
          f"({pick_px[0]:.0f}, {pick_px[1]:.0f})")
    if args.to:
        # Aim at the floor inside the container. The floor is at table level, so the tilted camera
        # maps it correctly; a point on the rim or the outside wall would not be.
        hits = point.point_at(frame, [f"the middle of the bottom inside the {args.to}"])
        if not hits:
            raise SystemExit(f'The camera cannot see "{args.to}".')
        place_px = (hits[0]["x"], hits[0]["y"])
    else:
        place_px = bot.table.to_pixel(*config.HANDOVER_XY)
    # Draw what the arm is about to do: where the jaws will be, not just where the model pointed.
    import numpy as np
    plan = frame.copy()
    c0 = np.array(bot.table.to_arm(*pick_px))
    away = np.array(bot.table.to_arm(pick_px[0], pick_px[1] - 12.0)) - c0
    away /= np.linalg.norm(away)
    side = None if args.standing else bot._width_grasp(outline, away)
    if args.standing:
        # The slide-in grip: come down short of the item, on the arm's side, then slide out onto it.
        item = np.array(bot.table.to_arm(*found))
        out_dir = item / np.linalg.norm(item)
        jaw = np.array([-out_dir[1], out_dir[0]])
        start = item - out_dir * motion.SLIDE_STANDOFF
        a, b = bot.table.to_pixel(*start), bot.table.to_pixel(*item)
        cv2.arrowedLine(plan, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 160, 0), 2, tipLength=0.3)
        for end in (item - jaw * motion.SLIDE_GAP / 2, item + jaw * motion.SLIDE_GAP / 2):
            u, v = bot.table.to_pixel(*end)
            cv2.circle(plan, (int(u), int(v)), 6, (0, 0, 255), -1)
        j0, j1 = bot.table.to_pixel(*(item - jaw * motion.SLIDE_GAP / 2)), bot.table.to_pixel(*(item + jaw * motion.SLIDE_GAP / 2))
        cv2.line(plan, (int(j0[0]), int(j0[1])), (int(j1[0]), int(j1[1])), (0, 0, 255), 2)
        cv2.putText(plan, "comes down at the arrow's tail, slides along it, jaws end at the red dots",
                    (int(a[0]) - 260, int(a[1]) + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 0), 2)
        pick_px = found                                   # for a standing item, the model's own point at its foot
        print(f"plan: slide on from the arm's side; item at {item[0] * 100:.1f} cm out, {item[1] * 100:.1f} cm across; "
              f"jaws {motion.SLIDE_GAP * 100:.1f} cm apart, {motion.SLIDE_HEIGHT * 100:.0f} cm above the table")
    elif side is not None:
        middle, gap, vec = side
        marks = (("grip middle", middle, (0, 200, 0)), ("jaw", middle - vec * gap / 2, (0, 0, 255)),
                 ("jaw", middle + vec * gap / 2, (0, 0, 255)))
        a, b = bot.table.to_pixel(*(middle - vec * gap / 2)), bot.table.to_pixel(*(middle + vec * gap / 2))
        cv2.line(plan, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 0, 255), 2)
        for label, xy, colour in marks:
            u, v = bot.table.to_pixel(*xy)
            cv2.circle(plan, (int(u), int(v)), 6, colour, -1)
        u, v = bot.table.to_pixel(*middle)
        cv2.putText(plan, "grip here, jaws at the red dots", (int(u) + 14, int(v) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 0), 2)
        print(f"plan: grip the middle at {middle[0] * 100:.1f} cm out, {middle[1] * 100:.1f} cm across; "
              f"item is {(gap - 2 * motion.CLEARANCE) * 100:.1f} cm across its short side, gap {gap * 100:.1f} cm, "
              "jaws closing across its width")
    elif (front := bot._front_back_grasp(outline, away, np.array([-away[1], away[0]]))) is not None:
        fixed, gap, vec = front
        a, b = bot.table.to_pixel(*fixed), bot.table.to_pixel(*(fixed + vec * gap))
        cv2.line(plan, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 0, 255), 2)
        for xy in (a, b):
            cv2.circle(plan, (int(xy[0]), int(xy[1])), 6, (0, 0, 255), -1)
        cv2.putText(plan, "long side faces the camera: jaws close front to back", (int(a[0]) + 14, int(a[1]) + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 0), 2)
        print("plan: its long side faces the camera, so the jaws close front to back across its short side")
    else:
        print("plan: outline not usable, so sweeping from the near edge instead")
        cv2.circle(plan, (int(pick_px[0]), int(pick_px[1])), 6, (0, 0, 255), -1)
    cv2.circle(plan, (int(place_px[0]), int(place_px[1])), 8, (255, 120, 0), 2)
    cv2.putText(plan, args.to or "handover", (int(place_px[0]) + 12, int(place_px[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
    cv2.imwrite(str(config.DATA / "last_pick.jpg"), plan)
    px, py = bot.table.to_arm(*pick_px)
    qx, qy = bot.table.to_arm(*place_px)
    print(f'"{args.item}" at pixel ({pick_px[0]:.0f}, {pick_px[1]:.0f}) = {px * 100:.1f} cm out, {py * 100:.1f} cm across')
    print(f'going to {args.to or "the handover spot"} at {qx * 100:.1f} cm out, {qy * 100:.1f} cm across')
    if (qx * qx + qy * qy) ** 0.5 > motion.MAX_PLACE_REACH:
        print(f"  that is {(qx * qx + qy * qy) ** 0.5 * 100:.0f} cm from the base and the arm releases at "
              f"{motion.MAX_PLACE_REACH * 100:.0f} cm: move it closer or the item will fall short")
    print(f"picture with both points: {config.DATA / 'last_pick.jpg'}")
    shots = config.DATA / "pick_frames"
    shots.mkdir(parents=True, exist_ok=True)
    for old in shots.glob("*.jpg"):
        old.unlink()

    def record(stage: str) -> None:
        """Save what the camera sees at each stage, so a miss can be looked at afterwards."""
        print("  stage:", stage)
        cv2.imwrite(str(shots / f"{stage.split(',')[0].replace(' ', '_')}.jpg"), cam.fresh_frame())

    bot.on_stage = record

    def look() -> bool | None:
        """After lifting: ask the vision model whether the item is in the gripper."""
        try:
            r = point.is_holding(cam.fresh_frame(), args.item)
        except Exception as e:
            print("  holding look failed, using the gripper reading:", e)
            return None
        print(f"  camera says {'holding' if r['holding'] else 'not holding'}: {r['reason']}")
        return r["holding"]

    bot.look_if_holding = look
    input("Press Enter to move, Ctrl+C to cancel...")
    tall = args.tall or any(w in (args.to or "").lower() for w in ("cup", "box", "bin", "tub", "jar", "container"))
    held = bot.pick_place(pick_px, place_px, drop=bool(args.to), tall=tall, long_side_px=long_side,
                          outline_px=outline, standing=args.standing)
    print("gripper felt the item: carried it" if held else "gripper closed on nothing: missed")
    if args.check:
        r = point.check_placed(cam.fresh_frame(), args.item, args.to or "the table near the front edge")
        print("camera check:", r)
except motion.ReachError as e:
    print(f"\nOut of reach: {e}\nPut the item so its near side is 17 to 24 cm from the arm's base.")
except KeyboardInterrupt:
    print("\nCancelled.")
finally:
    bot.close()
    cam.stop()
    print("Motors off.")

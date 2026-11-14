"""Switch the arm's motors off. HOLD THE ARM FIRST: it goes limp at once.

Use this when a program ended without releasing the arm and it is stuck stiff.

    python scripts/arm_relax.py
"""
from earshot.arm import robot as arm

input("Hold the arm with one hand, then press Enter to switch its motors off...")
bot = arm.open_limp()
arm.close(bot, limp=True)
print("Motors off. Fold the arm back to its rest position by hand.")

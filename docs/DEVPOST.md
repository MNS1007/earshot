Tagline: Teach a robot arm a table-top task by doing it once while you talk, then correct it by voice.

## Inspiration

The week before, I trained an imitation policy on this same arm. One task took hours of recorded demonstrations. While building a previous product I shadowed assistants at dental practices, where a new assistant learns by watching once and being corrected. Vibe coding let people build software by describing it. Earshot lets someone teach a robot by doing the task once.

## What it does

You do a table-top task while talking. A webcam films it and the Vexo Open wristband hears you. Gemini turns both into ordered steps and sorting rules. Each step shows what was Seen and Heard, flagged where they disagree. An SO-101 arm then does the task or hands you the next item. The camera checks each move and the arm retries once. You correct it by voice: order, destination, grip, speed, stop, try again. Saved tasks are recalled by name or time ("do what I taught you at 10").

## How we built it

No trained policy, no teleoperation data, no fine-tuning. The model decides what and where. My code turns that into motion.

Models were picked per job from timed runs: gemini-4-argon watches the clip and writes steps and rules (about 27 s). gemini-3.8-flash points at items and checks placements (about 2.5 s). gemini-3.5-flash-lite picks the voice action.

Voice is a Pipecat pipeline: band audio, Deepgram speech to text, Silero VAD, one LLM tool choice, a POST to the main app.

The arm uses LeRobot and placo inverse kinematics. A homography fitted from eight marks, which I guided the gripper tip onto, maps camera pixels to the table.

## Challenges we ran into

LeRobot measures degrees from the middle of each joint's calibrated range, not the URDF's zero: up to 19 degrees off here, which tilted the table by 9 cm. The motors refuse commands outside that range, so the IK needed the real limits. The gripper has one fixed jaw, so its gap is not centred on the tool point.

Grasping was the least reliable part.

PICK_RESULTS: to be filled in

## Accomplishments that we're proud of

One demonstration becomes a task the arm can run, with each step's evidence on screen. I wrote it all in one day.

## What we learned

A tilted camera sees an item's sides. Only the points where it meets the table map exactly, so the grasp is planned from those. A nudged camera is corrected by re-finding the calibration marks and voting on one shift.

## What's next for Earshot

A more reliable grasp, more reach, items without a clear colour. It does pick and place only, not new skills such as writing or inserting a plug.

## Built with

Python, Gemini 4, Pipecat, Deepgram, Silero VAD, LeRobot, placo, OpenCV, FastAPI, SO-101, Vexo Open band

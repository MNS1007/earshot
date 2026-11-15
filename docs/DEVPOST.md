Tagline: Vibe coding for robots. The person who knows the job shows it once and talks it through. No teleoperation, no training run, no robotics engineer.

## Inspiration

Teaching a robot one task today needs a robot-learning engineer and hours of teleoperation: driving the arm by hand, over and over, to record demonstrations. I did exactly that the week before, on this same arm, for a single task. But the people who know what the robot should do are not engineers. While building a previous product I shadowed assistants at dental practices. The assistant is the one who knows which instrument comes next. A new assistant learns by watching once, listening, and being corrected.

That is how people teach people, and talking is the most natural way to pass on what matters. Vibe coding let anyone build software by describing it. Earshot lets anyone teach a robot by doing the task once and talking it through.

## What it does

You do a table-top task while talking. A webcam films it and the Vexo Open wristband hears you. Gemini turns both into ordered steps and sorting rules. Each step shows what was Seen and Heard, flagged where they disagree. An SO-101 arm then does the task itself, or helps: it brings one item, waits until you have taken it, and brings the next. The camera checks each move and the arm retries once. You correct it by voice: order, destination, grip, speed, stop, try again. Saved tasks are recalled by name or time ("do what I taught you at 10").

## How we built it

No trained policy, no teleoperation data, no fine-tuning. The model decides what and where. My code turns that into motion.

Models were picked per job from timed runs: gemini-4-argon watches the clip and writes steps and rules (about 27 s). gemini-3.8-flash points at items and checks placements (about 2.5 s). gemini-3.5-flash-lite picks the voice action.

Voice is a Pipecat pipeline: band audio, Deepgram speech to text, Silero VAD, one LLM tool choice, a POST to the main app.

The arm uses LeRobot and placo inverse kinematics. A homography fitted from eight marks, which I guided the gripper tip onto, maps camera pixels to the table.

## Challenges we ran into

LeRobot measures degrees from the middle of each joint's calibrated range, not the URDF's zero: up to 19 degrees off here, which tilted the table by 9 cm. The motors refuse commands outside that range, so the IK needed the real limits. The gripper has one fixed jaw, so its gap is not centred on the tool point.

Grasping was the hardest part. Most attempts failed until the grasp was planned from where each block meets the table, the only points a tilted camera maps exactly. In the recorded run the arm then picked and placed 3 of 3 blocks from one spoken demonstration.

## Accomplishments that we're proud of

One demonstration becomes a task the arm can run, with each step's evidence on screen. The voice loop works from intent, not trigger words: it is told what the arm is doing before each sentence, so "the pink ones go in the bowl" is narration while it watches and a correction while it works. After every lift the arm looks to check it is holding the item, and after every placement it looks again.

## What we learned

A tilted camera sees an item's sides. Only the points where it meets the table map exactly, so the grasp is planned from those. A nudged camera is corrected by re-finding the calibration marks and voting on one shift.

## What's next for Earshot

A more reliable grasp, more reach, and items without a clear colour. Today it does pick and place only, not new skills such as writing or inserting a plug.

The larger idea: every lesson leaves a video, a transcript, the steps, the corrections and a camera check of each move. That is labelled data collected as a side effect of use. It could train a policy for the same task far faster than recording teleoperation by hand, so the person who knows the job teaches it once and the robot keeps getting better at it.

## Built with

Python, Gemini 4, Pipecat, Deepgram, Silero VAD, LeRobot, placo, OpenCV, FastAPI, SO-101, Vexo Open band

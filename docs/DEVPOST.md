Tagline: Vibe coding for robots: An industry expert teaches a robot a task by demonstrating it once and talking it through. No teleoperation, just pure human intelligence transferred through speech and video.

## Inspiration

I build software for dental practices, and I have spent time in clinics shadowing how the work is really done. That taught me where robotics gets stuck. The engineers who can build a robot do not know the job. The people who know the job, like the dental assistant who knows which instrument comes next, do not know robotics. Progress needs both in the same room, and that is the bottleneck.

Today the expert's knowledge reaches the robot in a thin form: hours of teleoperation, or labels on someone else's data. But that is not how experts teach. A dentist training a new assistant shows the task once and talks the whole way through it: what this is, why it goes there, what to watch for. Most of the knowledge is in the talking.

So what if the expert taught the robot directly, the same way?

## What it does

The expert does the task once and talks it through. A camera watches and the Vexo band on their wrist listens. Earshot turns the video and the words into ordered steps and rules, and shows each step with what it saw and what it heard. The robot then does the task itself, or works alongside: it brings one item, waits until you have taken it, and brings the next. You correct it by voice as it works, in your own words. Lessons are remembered and can be recalled later.

There is no teleoperation, no training run and no robotics engineer in the loop.

## Why now

Three things became good enough at the same time: video understanding in frontier models, real-time speech to text, and agents that can hold context and choose actions. And a wearable like the Vexo band puts a microphone on the expert all day, hands free, which is exactly where the teaching happens. Earshot is what you get when you point all of that at a robot.

## How I built it

- **Understanding:** Gemini 4 watches the demonstration with the transcript and writes the steps and rules. A faster Gemini model finds each item in the camera picture and checks every move.
- **Voice:** the Vexo band is the robot's ears. Its live microphone audio feeds a Pipecat pipeline: Deepgram for speech to text, then to a model that is told what the robot is doing right now and picks an action by intent. "The pink ones go in the bowl" is narration while it watches and a correction while it works.
- **Motion:** camera pixels map to table positions through a homography fitted from marks the gripper was guided onto, \\( (x, y, 1)^\top \sim H\,(u, v, 1)^\top \\), then inverse kinematics on a LeRobot SO-101.
- **Checking:** after every lift the robot looks to confirm it is holding the item, and after every placement it looks again and retries.

## Challenges

Grasping was the hardest part. A tilted camera sees an item's sides, so most of what it sees does not map to the table. Most attempts failed until the grasp was planned only from the points where an item meets the table. In the recorded run the arm then picked and placed 3 of 3 blocks from one spoken demonstration.

The hardware sets the ceiling. This is a small 3D-printed arm with five joints and a gripper, and no sense of force. It can pick things up and put them down. It cannot do tasks that need touch or fine force, and its reach is short.

## What I learned

Speech carries what a demonstration video cannot: which thing matters, why, and what the rule is. Giving the voice model the robot's current state mattered more than any prompt wording.

## What's next

Two groups can use this. Industry experts, who have the knowledge and can now teach a robot themselves. And robotics labs, who no longer need to send someone to shadow a clinic: the expert teaches the robot, and every lesson leaves a video, a transcript, the steps, the corrections and a check of each move. That is labelled data collected as a side effect of normal work, ready to train better policies for the same task.

## Built with

Python, Gemini 4, Pipecat, Deepgram, Silero VAD, LeRobot, placo, OpenCV, FastAPI, SO-101, Vexo Open band

# Earshot

Teach a robot arm a table-top task by doing it once.

You do the task while talking. A camera films the table and a wristband hears you.
A video model turns that into an ordered list of steps. The arm then hands you the
next item, or does the whole task itself, and you correct it by voice.

Built at VexoHacks, 14 November 2026. All code in this repository was written on the day.

## Parts

| Folder | What it does |
|---|---|
| `earshot/vision` | Watches a clip and lists the steps; points at items in a photo |
| `earshot/arm` | Camera-to-arm calibration, kinematics, pick and place |
| `earshot/voice` | Wristband audio to text to actions |
| `earshot/ui` | Live screen: steps, what was seen and heard, checks |
| `earshot/store.py` | Saved tasks, recalled by name or time |

## Hardware

SO-101 follower arm (LeRobot 0.6.2), one Logitech C920 on a tripod, a Vexo Open band.

## Setup

Copy `.env.example` to `.env` and fill in the keys. `.env` is never committed.

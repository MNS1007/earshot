#!/bin/bash
# Record the table camera with the laptop microphone to one file, while the app keeps using the camera.
#   scripts/record_camera.sh demo1        ->  ~/Downloads/demo1_camera.mp4
# Press q in this terminal to stop. Start the screen recording (Cmd+Shift+5) at the same time
# and clap once, so the two files can be lined up afterwards.
name="${1:-demo}"
out="$HOME/Downloads/${name}_camera.mp4"
echo "Recording the camera and microphone to $out. Press q to stop."
ffmpeg -hide_banner -loglevel warning -f avfoundation -framerate 30 -video_size 1280x720 \
  -i "HD Pro Webcam C920:MacBook Air Microphone" \
  -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -c:a aac -b:a 128k -y "$out"
echo "Saved $out"

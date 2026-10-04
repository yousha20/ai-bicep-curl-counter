# AI Personal Trainer — Bicep Curl Counter

A computer vision script that uses your webcam to track and count bicep curls in real-time. This project utilizes the MediaPipe Tasks API to render body poses and hand landmarks.

## Features
* Tracks left and right arm bicep curls independently.
* Calculates joint angles using shoulder, elbow, and wrist coordinates.
* Relies on MediaPipe version 0.10.30+ for pose and hand landmark detection[cite: 1].
* Automatically downloads required model files (`pose_landmarker.task` and `hand_landmarker.task`) on the first run[cite: 1].
* Displays an on-screen UI with live joint angles, current curl stage (up/down), and rep counters[cite: 1].

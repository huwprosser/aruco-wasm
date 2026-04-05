# AruCo WASM
I searched online for a low latency AruCo implementation in the browser and couldn't find one so I decided to throw 48 hours of my life into building one. 


https://github.com/user-attachments/assets/8350c90d-b33a-4c10-bd8b-b9931fd0260e



The system uses a low-dependecy implementation of ArUco inspired by OpenCV - mine mostly written in Rust for the important parts.

Usinging this demo you can:
- Detect and identify any AruCo 4x450 tag in realtime.
- Detect multiple tags in a scene.
- Estimate proximity of the tag.
- Correct for webcam focal length.

PRs welcome!

Known issues:
- Reflections are the enemy. Print your tags on matte paper for best results.

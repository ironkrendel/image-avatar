mkdir Frames
ffmpeg.exe -i .\input.mkv -f image2 .\Frames\frame-%%09d.jpg
ffmpeg.exe -i .\input.mp4 -f image2 .\Frames\frame-%%09d.jpg
ffmpeg.exe -i .\input.webm -f image2 .\Frames\frame-%%09d.jpg
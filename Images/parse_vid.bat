mkdir Frames
ffmpeg.exe -y -i .\input.mkv -vf "crop=800:800:50:1360" crop.mp4
ffmpeg.exe -y -i .\input.mp4 -vf "crop=900:1660:00:500" crop.mp4
ffmpeg.exe -y -i .\input.webm -vf "crop=800:800:50:1360" crop.mp4
ffmpeg.exe -y -i .\crop.mp4 -f image2 -qmin 1 -qscale:v 2 .\Frames\frame-%%09d.jpg
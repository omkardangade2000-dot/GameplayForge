"""Join phone screen recordings (portrait-stored, 120/60 fps) into one 2560x1440 60 fps source.

Each clip: rotated to landscape, cropped to 16:9 (drops the notch bar), constant 60 fps, then 8 s of black between
clips so no fight is ever merged across two recordings (killedit.match_ids numbers the matches from these gaps).
Usage:  python -m gameplayforge.prep OUT.mp4 clip1.mp4 clip2.mp4 ...
"""
import json, os, subprocess, sys, tempfile
from .util import pick_encoder, load_config


def dims(p):
    s = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height:stream_side_data=rotation", "-of", "json", p],
                                  capture_output=True, text=True).stdout)["streams"][0]
    rot = 0
    for d in s.get("side_data_list", []) or []:
        rot = int(d.get("rotation", 0))
    return s["width"], s["height"], rot


def main(argv=None):
    a = argv or sys.argv[1:]
    out, clips = a[0], a[1:]
    cfg = load_config(None)
    enc = pick_encoder(cfg)[0]
    venc = [ "-c:v", enc, "-cq", "18", "-preset", "p5", "-b:v", "0"] if "nvenc" in enc else ["-c:v", "libx264", "-crf", "17", "-preset", "medium"]
    tmp = tempfile.mkdtemp(prefix="gf_prep_")
    parts = []
    black = os.path.join(tmp, "black.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=2560x1440:r=60:d=8", "-f", "lavfi",
                    "-i", "anullsrc=r=48000:cl=stereo", "-t", "8", *venc, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                    black], check=True)
    for i, c in enumerate(clips):
        w, h, rot = dims(c)
        # ffmpeg auto-rotates by the rotation tag; a clip that is still portrait afterwards gets turned to landscape
        portrait = (h > w) if rot in (0, 180, -180) else (w > h)
        vf = ("transpose=2," if portrait else "") + \
             "crop='min(iw,ih*16/9)':'min(ih,iw*9/16)',scale=2560:1440:flags=lanczos,fps=60,setsar=1,format=yuv420p"
        p = os.path.join(tmp, f"c{i:02d}.mp4")
        print(f"  [{i+1}/{len(clips)}] {os.path.basename(c)}")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-stats", "-i", c, "-vf", vf, *venc, "-c:a", "aac", "-b:a", "192k",
                        "-ar", "48000", "-ac", "2", p], check=True)
        if parts:
            parts.append(black)
        parts.append(p)
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out], check=True)
    print("  joined ->", out)


if __name__ == "__main__":
    main()

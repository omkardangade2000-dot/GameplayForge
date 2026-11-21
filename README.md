# GameplayForge

A local, GPU-powered editor that turns a long gameplay recording or stream into:

- a **montage** (16:9) of whole fights and highlight events, in timeline order, with chapters
- **Shorts** (9:16), one event per Short, ending on the payoff
- separate **cutscene** clips for story games
- **thumbnails** built from real frames of the video
- an **SEO text file** (title ideas, description, chapters, tags)

Everything runs on your own PC. It needs no cloud API and no API key.

## How it works

```
recording -> SigLIP 2 frame scan (menus, maps, cutscenes, HUD)      -> cached
          -> Qwen3-VL watches every 4 s window: what happens, how intense, was there a kill
          -> events = consecutive action windows (never cut mid-fight)
          -> planner: montage (whole events) + Shorts (best 15-30 s of one event)
          -> FFmpeg + NVENC render, -16 LUFS loudness, QC contact sheets
```

More detail is in [ARCHITECTURE.md](ARCHITECTURE.md). The editing rules it follows are in [EDITING_RULES.md](EDITING_RULES.md).
Per-game profiles live in `config.yaml` under `games:`. There are profiles for CS2, BGMI and story/action games such as Bloodlines 2.

## Requirements

- Windows 10/11
- An NVIDIA GPU. It was built on an RTX 5070 Ti (16 GB); about 12 GB of VRAM is recommended for the Qwen3-VL judge.
- Python 3.11
- FFmpeg (`install.ps1` installs it with winget)

## Setup

```powershell
git clone https://github.com/omkardangade2000-dot/GameplayForge.git
cd GameplayForge
.\install.ps1
```

Optional assets, which are not included in the repo:

- **Face detector:** used for thumbnails to hide facecams. Download `face_detection_yunet_2023mar.onnx` from the
  [OpenCV model zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) into `models/`.
- **Fonts:** for thumbnail text, e.g. [Anton](https://fonts.google.com/specimen/Anton) from Google Fonts. Put the `.ttf` file in `fonts/`.

AI models (SigLIP 2, Qwen3-VL-4B, faster-whisper, optionally RealVisXL) download automatically from Hugging Face on first use.

## Usage

1. Set `paths.input_dir` / `paths.output_dir` in `config.yaml`. Put your in-game name in `games.<game>.player_names`.
2. Run:

```powershell
.\run.bat "MyRecording.mp4"                      # montage + Shorts + cutscenes + report
.\.venv\Scripts\python -m gameplayforge run "MyRecording.mp4" --game cs2 --no-shorts
.\.venv\Scripts\python -m gameplayforge thumbs "MyRecording.mp4" --title "My video title"
.\.venv\Scripts\python -m gameplayforge doctor   # checks GPU, FFmpeg and models
```

## Use it responsibly

- Edit **your own** gameplay, or footage you have permission to use. Credit the original creator.
- YouTube monetization needs added value, such as your own commentary, story or edit. Plain re-uploads count as "reused content".
- Check each game publisher's video policy before monetizing.
- Music: use only tracks you are licensed to use. By default it keeps the game audio only.

## License

MIT. See [LICENSE](LICENSE). Third-party models and fonts keep their own licenses.

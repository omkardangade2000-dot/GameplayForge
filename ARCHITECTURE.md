# GameplayForge - architecture (v0.2)

Target machine: Ryzen 7 9800X3D, RTX 5070 Ti (16 GB, Blackwell sm_120), 32 GB RAM, Windows.
Input: `C:/Users/YOU/Videos\<file>` (20 GB+). Output: `GameplayForge_Output\<file>\...`.

## What changed in v0.2 (from the first real test)
v0.1 picked isolated 6-second "high score" moments and stitched them together. That cut kills in half,
mixed unrelated moments into one short, shuffled the montage out of timeline order, let a loading screen
and a menu through, and called plain walking "boss" in a first-person game. v0.2 edits **whole events**
and lets a vision-language model decide what is actually happening.

## Pipeline
```
raw video --NVDEC--> 4 fps --> SigLIP 2 (GPU) image embeddings + letterbox/HUD/motion/brightness  -> analysis.npz (cached)
                               prompts changed? -> re-score cached embeddings only (seconds)
Stage 2 segments.py  cutscene candidates: what the image model SEES (cutscene + conversation);
                     HUD loss / letterbox only support it, they can never make a cutscene alone
         judge.py    Qwen3-VL checks each candidate: story cutscene | conversation -> Cutscenes/
                                                     kill animation | gameplay  -> back to the montage
Stage 3  highlights.usable_mask  hard filters: cutscenes, menu/map/loading/black, frozen frames (pause,
                     cursor screens), long conversations
         judge.label_windows     Qwen3-VL watches every usable 4-second window (4 frames, 1 s apart):
                     activity (combat / stealth_kill / parkour / boss_fight / exploring / ...),
                     intensity 0-10, kill yes/no, caption               -> judge_windows.json (cached)
         highlights.build_events consecutive action windows (a 4 s lull allowed) = one EVENT,
                     with 1 s lead-in and 1.5 s follow-through, never crossing a cutscene/menu
         judge.title_events      a 2-4 word chapter title per event
Stage 4  planner.py  Long: whole events in timeline order; weakest dropped if over target; chapters follow time
                     Shorts: ONE event per short, best 15-30 s stretch of the short's theme, ending on the kill
Stage 5  render.py   FFmpeg + NVENC, light grade, 0.15 s audio fades, concat, -16 LUFS (game audio only)
Stage 6  qc.py       measured checks + contact sheets for the montage and every short
```

## Tool choices (all free / open source)
| Need | Choice | Why |
|---|---|---|
| Decode / encode | FFmpeg NVDEC + NVENC | GPU does video work; tested at run time with CPU fallback |
| Fast frame understanding | SigLIP 2 so400m | Cheap per-frame filter (menus, maps, cutscenes, dialogue); embeddings cached |
| Real scene understanding | Qwen3-VL-4B-Instruct (bf16, ~9 GB) | Reads several frames at once and answers "what is the player doing, did someone die" |
| Runtime | Official PyTorch cu128 | RTX 50-series support |

`--no-judge` falls back to the image model only (much weaker; categories are guesses).

## Known limits
* The judge sees 4 frames per 4 s, not every frame and not the audio; very fast moments can be missed.
* A recording with little action gives a short montage. Nothing is padded with filler; the report says so.
* The 3:4 centre crop used for shorts (matching the references) cuts the left/right edges of the frame.

# Story & cutscene system (GTA 6 ready)

Built 2026-11-16, three days before GTA 6 (release **19 Nov 2026**, PS5 / Xbox Series; PC later).

## 1. What the research showed

**Rockstar's video policy** ([support.rockstargames.com](https://support.rockstargames.com/articles/7bNaeoMFTV0iUDGhStTXvz/policy-on-posting-copyrighted-rockstar-games-material), last updated 2025-01-10):

| Allowed | Taken down |
|---|---|
| Cutscenes "as part of a larger play-through (or in a narrated 'Let's Play')" | "isolated videos of the game's cinematics" |
| Your own gameplay, clips, machinima | The ending; any "edited combination of cutscenes and/or in-game plot dialogue"; any big story reveal |
| | Pre-release or leaked footage, including "early unboxing" |
| | In-game TV shows or comedy shows on their own |

- The policy covers **non-commercial** use. Monetized GTA videos exist everywhere, but Take-Two can remove anything.
- Take-Two is actively issuing takedowns and copyright strikes before launch. Three strikes deletes a channel.

**What goes viral for GTA 5** (483 videos from 16 searches, 2026-11-16):

| Format | Median views |
|---|---|
| Endings / funny & meme scenes | 2.3M |
| Reactions ("first time playing") | 836K |
| Walkthrough parts (Let's Play) | 790K |
| "All cutscenes" movies | 212K |
| Pure cinematic edits | **38K** |

- The biggest hits are NPC chaos and funny moments, and the player's own voice over the story. Examples: "Playing GTA 5 First Time" at 26.9M and "i cried while playing GTA 5" at 13.4M.
- Characters sell the click: Franklin, Trevor and Lamar, plus "Lamar roasts Franklin", which appears in 20 of the top 150 titles.

**Conclusion:** the policy-safe formats are also the formats that win. So the system makes **narrated story episodes, reaction Shorts and gameplay-chaos Shorts**. It never makes raw cutscene uploads.

## 2. Architecture

```
  YouTube search results ──► TREND RADAR (trends.py) ──► moment keys: characters, quoted lines, words
  (titles, views, dates only;   velocity = views/day         _Trends/<game>/latest_moment_keys.json
   nothing downloaded)          outlier = views/subs
                                                               │
  YOUR recording ──► analysis (SigLIP2) ──► cutscene finder ──► AI judge checks each cutscene
                                            (HUD gone, film      │
                                             camera, subtitles)  ▼
                     faster-whisper (GPU) ──► dialogue + word timings
                                                               │
                                     MOMENT RATER (cinematic.py) - Qwen-VL on 4 frames + the dialogue:
                                     title, hook, humor, shock, emotion, action, quotable line,
                                     characters, spoiler level (none/minor/major/ending), facecam yes/no
                                                               │
                         viral score = 0.30·hook + 0.30·strongest emotion + 0.25·trend match
                                       + quotable line + good length (12-75 s)
                                                               │
                         POLICY GATE (per game, config games.<g>.policy):
                           before release date ................ DO NOT POST (pre-release = takedown)
                           ending .............................. DO NOT POST (stays inside the episode)
                           major twist within 30 days .......... DO NOT POST
                           Rockstar + no commentary ............ DO NOT POST (isolated cinematic)
                           Rockstar + your voice / facecam ...... REVIEW (reaction short)
                           other games ......................... OK / REVIEW
                                                               │
         ┌─────────────────────────────────┬───────────────────┴───────────────────────┐
         ▼                                 ▼                                           ▼
  Reports/Moments_Report.txt      Long_Videos/Episode_Part_XX.mp4         YouTube_Shorts/Moment_Short_XX.mp4
  every scene ranked + status     the playthrough in order, cutscenes     allowed moments only, dialogue burned
  + why + trending keys           IN CONTEXT, walks/menus/loading cut,    in as subtitles, headline = true title
                                  ~25-min parts, chapters, thumbnails,   + Combat/Action shorts from gameplay
                                  upload text with spoiler warnings
```

Cutscenes/Cutscene_XXX.mp4 are still extracted, but **for your reference only**. A `README_DO_NOT_UPLOAD.txt` sits in that folder.

**Future-proof:** nothing depends on GTA 6's HUD, which nobody has seen yet. Cutscenes are recognised by the HUD disappearing, the film-style camera and the subtitles. Moments are recognised by what is said and shown. Trending moments come from live YouTube data each day, not from a list written in advance.

## 3. Launch-week steps (GTA 6)

1. **Every morning:** `python -m gameplayforge trends --game gta6`. This takes about 3 minutes and reads titles and view counts only.
   - Read `GameplayForge_Output/_Trends/gta6/<date>/Trend_Report.txt` for which formats and moments are rising right now.
2. **Record your playthrough WITH your voice and facecam on.** That is what YouTube's gameplay monetization rules and Rockstar's policy both reward.
   - Put `GTA6` in the file name, e.g. `GTA6_Day1.mkv`, so the profile is picked automatically.
3. **Run:** `python -m gameplayforge run GTA6_Day1.mkv --commentary`
4. **Open `Reports/Moments_Report.txt`:** post only OK / REVIEW items. For REVIEW items, make sure your reaction is the point of the video.
5. **Upload as Unlisted first.** Follow the checklist inside each `_UPLOAD.txt`.

Nothing is ever uploaded automatically: you post everything yourself.

## 4. Settings

- `config.yaml` → `games.gta6` / `games.gta5`: notes for the AI, characters, trend searches, `policy`.
- `config.yaml` → `cinematic`: part length, Shorts count, viral threshold, subtitles, score weights.
- `config.yaml` → `transcribe`: Whisper model. `small` is multilingual; `medium.en` is English only.
- CLI: `--mode cinematic` forces this mode for any story game. `--commentary` says the recording has your reaction.

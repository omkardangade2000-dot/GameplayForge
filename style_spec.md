# Gameplay Edit Style Spec (from your 10 reference videos)

Measured with ffprobe (format), FFmpeg scene detection (cuts) and EBU R128 (loudness).
Content judged from 24 evenly spaced frames per video.

Your references fall into **3 output types**. From one raw gameplay recording, the tool should produce all three.

---

## Type A: Vertical Short (refs: videoplayback (1), (2))

| Property | Value |
|---|---|
| Aspect | 9:16 (reference is 360x640; **we export 1080x1920**) |
| Length | 20–30 s |
| Cuts | **None.** One continuous moment |
| Framing | Center crop that follows the character. The gameplay sits in a ~3:4 box in the middle with **black bars top and bottom** (room for a title or caption) |
| HUD | Mostly cropped out by the vertical crop |
| Audio | Game audio only, around -19 LUFS |
| Content | One complete combat burst from start to finish: approach, multi-kill or fire attack, last enemy down. Example: Ghost of Tsushima fire-blade fights |

**Rule:** pick the single best 20–30 s action window. Start just before the first hit and end right after the last kill or finisher.

---

## Type B: Music Montage (refs: Beggin' (Valorant Montage), lalala)

| Property | Value |
|---|---|
| Aspect | 16:9, **2560x1440 @ 60 fps** |
| Length | 95–106 s (one song) |
| Pace | 21–38 cuts. Average shot **2.7–4.3 s**, with **rapid bursts of 0.05–0.8 s cuts** on beat drops (lalala at 71–74 s has ~14 cuts in 3 s) |
| Structure | Logo intro card, then clips, then a mid "skit" with dialogue subtitles, then a climax burst, then an outro card ("book your montage" / channel logo) |
| Effects | Speed ramps (slow-mo into the kill), whip / motion-blur transitions, glow and flash on kills, color grading, shake |
| Audio | **Music-driven and loud**: -8.5 to -16 LUFS. Cuts land on beats |
| Content | Kills, multi-kills, clutches, ability combos |

**Rule:** detect the song's beats, then place the highest-scoring kill moments on the strongest beats and drops.

---

## Type C: Long "Cinematic Gameplay, No Commentary" (refs: videoplayback, (3)–(7))

Games in the refs: Ghost of Tsushima, Assassin's Creed (Mirage-style desert and Black Flag-style Caribbean), Batman Arkham Knight, a neon-Japan action brawler.

| Property | Value |
|---|---|
| Aspect | 16:9 (exported at your source resolution) |
| Length | **9–21 min** |
| Cuts | Very few: 3–23 in total. Average shot **33–175 s**. Cut only between encounters |
| HUD | Kept, untouched |
| Audio | **Game audio only, no music added.** Natural dynamics (loudness range 15–20 LU), -17 to -26 LUFS |
| Content | Best combat encounters, stealth takedowns, traversal and parkour, scenic shots, key cutscenes |
| Removed | Menus, loading screens, map screens, deaths and retries, long idle walking, inventory |

**Rule:** a "best-of the session" cut. Keep whole encounters intact and remove the dead time between them.

---

## How each of your games maps to these types

| Game | A Shorts | B Montage | C Long cinematic |
|---|---|---|---|
| BGMI | Yes (kills, chicken dinner) | **Best fit** | Yes (full match) |
| God of War, Lords of the Fallen, Fallen Jedi, Bloodlines 2, Wuchang/Shifu, Dawnwalker | **Best fit** (boss and combo moments) | Yes | **Best fit** |
| AC Black Flag, Styx, Commandos | Yes (stealth kills) | Partial | **Best fit** |
| Sherlock Holmes | Rarely | No | Yes (story) |

## Highlight detection (no face cam or commentary, so no speech)

1. **Audio energy**: combat sound spikes, hit and kill sounds, and boss music swells.
2. **Motion intensity**: fast camera movement and particle-heavy frames (fire, sparks, blood).
3. **Scene and HUD changes**: boss health bar appears or disappears, on-screen text such as "Winner Winner Chicken Dinner", "Knocked out", "Boss defeated", or chapter titles (OCR).
4. **Dead-time detection** to cut: menus, loading screens and idle walking (low motion, low audio, static HUD).
5. **Me (Claude)**: I review thumbnails of the top-scoring windows and choose the final picks, titles and SEO.

Everything runs locally on the **RTX 5070 Ti**: NVENC encoding, GPU frame analysis and Whisper for any in-game dialogue.

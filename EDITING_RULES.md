# GameplayForge editing rules

These rules apply to every video automatically. They come from the 8 reference videos and from the channel
owner's frame-by-frame review of the first edits (2026-11-08). Each rule maps to a setting in `config.yaml`,
so nothing has to be repeated per video.

## Long montage
| Rule | Why | Setting |
|---|---|---|
| Open with a hook: the best kill of the video, in slow motion, 5-6 s | The first 30 s decide whether people stay | `long.cold_open` |
| The montage is made of encounters only (fights and stealth chains) | Stand-alone traversal or parkour is "meaningless" in a montage | `encounters.*` |
| Up to 3 s of approach right before a fight is allowed | Gives the fight context without wasting time | `encounters.lead_in` |
| A fight starts when the player engages (lines up the first shot, starts stalking) and ends after the last kill | Never cut a fight in the middle | `encounters.merge_gap_windows` |
| Stealth kills that lead into a fight belong to that fight | The stealth setup was being dropped | engagement detection |
| Walking of 8 s or more inside a fight is cut out with a soft 0.3 s dissolve | Keeps the energy without a jarring jump | `encounters.cut_dead_windows`, `long.xfade` |
| A single ordinary kill is never shown; a stealth kill or spectacular kill is | "Avoid single enemy kills unless it feels clutch" | `encounters.min_kills` |
| Each fight is rated 1-10 by the AI judge; ordinary fights (below 5), or below 55 % of the video's best, are left out | Only clutch or cinematic fights | `long.min_hype`, `long.min_rel_score` |
| Sections change with a ~0.35 s dip to black, never a hard cut | How the references change section (measured) | `long.dip` |
| Timeline order, no repeats. The cold-open moment is taken out of its fight later on, so it is never shown twice | Review 2: "it appears again" | `long.cold_open_no_repeat` |
| Every kill is shown in full: no cut, dissolve or section end within 2 s before or 2.5 s after a kill | Review 2: "kill is not fully shown ... don't do this again" | `encounters.kill_guard_before/after` |
| A kill that happens far away (small on screen) gets a smooth punch-in zoom towards it (up to 1.6x, eases in and out) | Review 2, montage 1:02: the kill happened far away and the player's dash made it look cut | `punch_in.*` |
| Quality over length: the montage is never padded | A short strong montage beats a long boring one | |
| Exported in 4K (3840x2160). Contrast-adaptive sharpening at native 1440p BEFORE the upscale, light unsharp after | YouTube gives 4K uploads a much higher bitrate; review 2 still found the montage softer than the shorts | `long.upscale_4k`, `long.cas`, `long.sharpen` |

## Shorts
| Rule | Setting |
|---|---|
| One fight per short, starting on the action (1.5 s lead-in at most) | `short.lead_in` |
| Lulls removed; whole fight up to 45 s; parkour up to 30 s | `short.length_max`, `short.parkour_max` |
| Ends on the final kill, played in slow motion (0.5x) with a slight punch-in, then out 1.8 s later | `slowmo.*`, `short.after_last_kill` |
| Only fights the judge rates 6/10 or higher | `short.min_hype` |
| 3:4 gameplay band on a black 9:16 canvas (as in the references); the band follows an off-centre kill | `short.layout` |
| **Killstreak short** (from the Valorant references): the best kills of the whole recording, ~2.7 s each, fast 0.18 s whip transition between them, opens on the 2nd-best kill, ends on the best kill in slow motion, ~28 s | `killstreak.*` |
| Shorts are sharpened at native resolution too | `grade.cas` |
| **Movement-kill short** (approved 2026-11-11): only kills that come straight out of a move (drop, jump, glide, dash, slide, vault, vehicle). Each clip runs from the start of the move to the body dropping (~4 s); the best 3-6 are chained with whips, the most insane one last in slow motion. Fewer than 3 = no movement short | `movement.*` |
| The Killstreak short never reuses a movement-short kill | planner |
| Pure parkour shorts (climbing/running with no kill) are retired | `short.themes` |
| 4-6 shorts per run when the footage has them (combat shorts up to 3), never padded | `short.max_total`, `short.count_per_theme` |

## Games
The AI names the game from the footage and loads that game's hints (`game: auto`, `games:` in config.yaml):
what a kill looks like, which screens are menus, what movement exists. Profiles: Bloodlines 2 (first-person,
feeding, stealth vision) and BGMI / PUBG Mobile (third-person; knocks count as kills; spectating, lobby and
results screens are never used). A new game works without a profile, just with fewer hints.
The game is picked from the file/folder name first (e.g. `BGMI_...mp4`, `...\pubg\`), else the AI chooses from
the known list after reading on-screen logos (free naming called BGMI "Garena Free Fire").

**Kill counter = ground truth (BGMI).** The AI reads the HUD's "Eliminations N" every 2 s; each rise is a
confirmed elimination. The movement and Killstreak shorts use confirmed eliminations only, and a montage fight
needs at least one (`games.bgmi.kill_counter`). Why: on 2026-11-11 the AI took two bat-form flights for
"movement kills"; the counter never moved.

## Kill-centred editing (BGMI and other games with a kill counter) - owner review 2026-11-11
The whole-fight edit was full of "unnecessary clips" (scanning, waiting, looting) and showed the player being
knocked. For these games the edit is now built from kills only (`killedit.py`, `kill_edit:` in config):
| Rule | Setting |
|---|---|
| Built from confirmed eliminations only (HUD counter) | `games.bgmi.edit_mode: kills` |
| **Self-review:** the AI checks every second of every planned clip. Only shooting / kill seconds are kept, "aiming" only in the 2 s right before firing; 2+ s of anything else is cut | `judge.review_seconds` |
| Never show the player's own knock-down or death: the clip ends before it | `killedit.tighten` |
| At most 4 s of build-up before a kill; the kill is kept from 1.5 s before to 1.5 s after | `kill_edit.max_buildup`, `kill_before/after` |
| Montage: the best clip's kill as a slow-motion hook (not repeated later), then every kill clip in match order, dissolves inside a match, dip to black between matches; chapters = matches | `killedit.plan_long` |
| Shorts: exactly 3 kills, then end 1.2 s after the 3rd (slow motion on it); kills from different fights are combined with a whip | `kill_edit.kills_per_short`, `short_after` |

## Fight-centred editing (replaces the kill-only edit for BGMI) - owner review 2026-11-11 (evening)
"Shorts are still shitty", "full combat is not shown": 3-second kill fragments from different fights were stitched
into 11 s shorts. Now (`fightedit.py`, `fight_edit:` in config, `games.bgmi.edit_mode: fights`):
| Rule | Setting |
|---|---|
| A fight is found from the player's OWN gunfire in the audio (loud, sharp bursts; distant fire ignored) plus a confirmed elimination | `fight_edit.gunfire`, `cluster_gap`, `merge_gap` |
| A fight is shown WHOLE: from lining up the first shot (aiming only, never parachute / loot) to the last kill; never cut mid-combat | `lead_in`, `after_kill` |
| Kill times are snapped to the player's last shot (the HUD counter is read every 2 s and was late) | `kill_late` |
| Only quiet stretches inside a fight (>= 4 s, no shot, no kill: looting, healing) are cut, with a soft dissolve | `lull_cut` |
| Single-kill fights only when the AI rates them clutch (8+/10) | `min_kills`, `single_kill_wow` |
| Montage opens with the BEST fight in full (engagement from second 1), then the others in timeline order, dip to black between; nothing shown twice | `fightedit.plan_long` |
| One whole fight per short, 12-58 s, last kill in slow motion | `short_min`, `short_max` |
| LOOTING is cut (review 2026-11-12: two shorts opened with 8-23 s of looting). The 'Picked up X' line and the 'Nearby:' loot list are read every second; pickup clicks are not counted as shots; a fight never opens on looting | `games.bgmi.loot` |
| A knock and the finish of the SAME enemy count as one kill, so titles never over-claim | `judge.read_killfeed` |
| Every video gets `<name>_UPLOAD.txt`: 3 titles, description (with chapters), hashtags, tags - built only from measured facts (kill count, seconds between first and last kill) | `seo.py` |

## Bloodlines 2: whole uncut fights (owner's reference, 2026-11-18)
Reference: youtu.be/IFYL9-W29uc "Vampire the masquerade bloodlines 2 - banu hakim brutal combat and air combos"
(3:08, 11K views), measured frame by frame in the browser: about 6 cuts in 3 minutes (~30 s per clip), every clip
a whole fight, hard cuts, no fades, no slow motion, no zoom, no text or music overlays, HUD kept, opens straight
into a fight. Its comments praise the fluid, uncut combo chains ("juggles", "air combos").
Owner's note: "full combat are not shown", "show full kills". Applied through
`games.vtm_bloodlines2.style_overrides` (other games keep their own style):
| Rule | Setting |
|---|---|
| Each clip is a WHOLE fight: 2 s before the first strike to 2.5 s after the last enemy is down | `encounters.lead_in`, `post_roll` |
| Combos are never cut; only walks of 12 s+ inside a fight are removed | `encounters.cut_dead_windows: 3` |
| No teaser, no slow motion, no zoom: the montage opens straight into the best fight, shown whole, then the others in timeline order | `long.open_with_best`, `slowmo`, `punch_in` |
| Shorts: one whole fight each (15-59 s), up to 6 | `short.themes: [combat]`, `length_max: 59` |
| One battle is never split in two: a mid-fight enemy taunt or a white power flash (labelled 'dialogue'/'loading', up to 8 s) does not end the fight; only cutscenes, menus or longer talk do. Found in the first edit: the Statue fight (2:57-2:59) lost its second half | `encounters.max_inner_talk_s` |
| Every clip opens on the first real strike (a kill or a 7+/10 moment, stealth kills included) and ends after the last one, never on 20 s of sneaking or taunts | `encounters.start_at_action` |
| Shorts carry a catchy hook in the top band, not a fight name ("FACTORY FIGHT" was flat). The hook is an open loop or bold claim picked from what the fight really shows: stealth bites -> "THEY NEVER SAW ME COMING", juggles -> "NOBODY TOUCHES THE GROUND", fire -> "WATCH THEM BURN", big crowd -> "ONE VAMPIRE VS THE WHOLE GANG", else "WAIT FOR THE LAST KILL"; every short a different one; the upload title uses the same hook | `seo.STORY_HOOKS` |
| Kept from earlier reviews: short dip to black between fights (no confusing hard cuts), nothing shown twice, game audio only | `long.dip` |

## Several clips in one run (e.g. one per match)
Clips are joined into one source with 8 s of black between them, so no fight is ever merged across two clips.
Phone recordings are rotated upright, set to a constant 60 fps and cropped to 16:9 (drops the notch bar).

## Unchanged from the references
No captions or cards, no added sound effects. Shorts: game audio only. Montage (owner's reference 'Royalty',
2026-11-11): short dark teaser + glowing channel-name card, then a music bed under the game audio IF `style.music`
points to a file the owner may use; otherwise game audio only, normalised to -16 LUFS.

## Measured from the references (re-checked 2026-11-08)
- The long references barely cut: about 8 edits in 9 minutes. Most of each video is one continuous take of
  the player moving purposefully from enemy to enemy.
- Section changes are a short fade to black, sometimes over the game's own mission title, coming back in
  already inside the next fight.
- They open with a ~4 s channel logo, then the player is sneaking toward enemies, with the first kill by
  about 0:12.

## Measured from the FPS references (Valorant: "lalala", "Beggin'", 2026-11-08)
- ~100 s each. A ~4 s logo, then a kill within the first second of gameplay.
- A new clip every ~1.8-2.5 s: each clip is one kill (the shot and the body dropping), nothing else.
- Clips are joined with fast whip / zoom-blur transitions and speed ramps, never dips to black.
- They use music, joke captions and "likes?" cards; those are NOT copied (game audio only, no captions).

## Recording tip (affects sharpness more than any edit setting)
The source recording is AV1 at about 10 Mbit/s for 1440p60, which is low. Recording at 40-50 Mbit/s (or
CQP/CQ 18-20 in the NVIDIA app or OBS) makes every output visibly sharper.

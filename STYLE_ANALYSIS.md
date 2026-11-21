# Style decoded from your 8 reference videos (measured, not guessed)

Method: FFmpeg scene-change detection, crop detection, loudness and beat-periodicity analysis, plus visual
inspection of sampled frames. I cannot hear audio, so music/SFX conclusions come from measurements only.

| Property | Long references (6) | Shorts references (2) |
|---|---|---|
| Format | 16:9, no letterbox | 9:16 canvas, gameplay in a ~3:4 centre band (360x486 of 360x640), black above/below |
| Hard cuts | ~1.4-2.2 per minute; median uninterrupted shot 5-15 s, mean 27-40 s | 3 and 11 cuts over 29 s / 20 s at a sensitive threshold (mostly in-game camera whips) |
| Overlays / captions / cards | none found in sampled frames (only the game's own HUD) | none found |
| Audio level | about -19 LUFS integrated, no steady beat found | about -19 LUFS, no steady beat found |
| Content | continuous third-person traversal, stealth, combat | single fight or ambush beat, starts on action |

Consequences for the "reference" style profile in `config.yaml`:
straight cuts, takes of 6-25 s, no captions/cards, no added SFX, no music, game audio kept and normalised to -16 LUFS,
shorts in the 3:4-band-on-black layout. Everything else is switchable.

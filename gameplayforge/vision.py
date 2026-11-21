"""Visual understanding layer.

Primary backend: SigLIP 2 (zero-shot image-text model, Apache-2.0) running on the GPU.
Every sampled frame is embedded once; the embeddings are scored against two prompt groups:
  scene type   : cutscene / gameplay_action / gameplay_calm / menu / map / loading / black
  moment type  : combat / stealth / parkour / hero / scenic / boss / explore / dialogue
"explore" and "dialogue" are explicit NEGATIVE classes: without them, plain walking had to be called
combat/hero/boss by the softmax. Prompts cover first-person AND third-person games.
Embeddings are cached, so changing prompts only re-scores (PROMPT_VERSION) - no re-decode of the video.
"""
import numpy as np

PROMPT_VERSION = "v2"
SCENE = ["cutscene", "gameplay_action", "gameplay_calm", "menu", "map", "loading", "black"]
MOMENT = ["combat", "stealth", "parkour", "hero", "scenic", "boss", "explore", "dialogue"]
ACTION = ["combat", "stealth", "parkour", "boss"]          # categories that can form an action event

PROMPTS = {
    "cutscene": ["a cinematic cutscene from a video game with characters in close-up and no on-screen HUD",
                 "a close-up of a character's face talking in a video game conversation with subtitles",
                 "a movie-like scene of two characters talking in a video game"],
    "gameplay_action": ["video game gameplay during a fight with enemies attacking the player",
                        "first-person shooter gameplay aiming a gun at enemies",
                        "third-person action game combat with a health bar and HUD"],
    "gameplay_calm": ["first-person video game gameplay walking through a city street at night",
                      "third-person video game gameplay walking through an environment with a minimap HUD",
                      "a player exploring an open world in a video game"],
    "menu": ["a video game menu or inventory screen with text and icons",
             "a skill tree or ability upgrade menu in a video game",
             "a character appearance customization menu showing a full-body character model"],
    "map": ["a top-down city map screen with roads, icons and markers in a video game",
            "a world map menu with location markers"],
    "loading": ["a video game loading screen with a progress indicator",
                "a still title card with large text announcing a new mission",
                "a still image of a building with a mouse cursor on it",
                "the main title screen of a video game with a logo and menu options"],
    "black": ["a completely black screen", "a fade to black"],
    "combat": ["first-person view shooting a pistol or shotgun at enemies with muzzle flash",
               "a first-person melee fight punching and clawing an enemy at close range",
               "a character fighting several enemies with weapons and impact effects",
               "a violent fight with an explosion of fire and enemies falling"],
    "stealth": ["sneaking up behind an unaware enemy to perform a silent takedown",
                "a vampire biting the neck of a person to drink blood",
                "enemies glowing red or orange through walls seen with a special vision ability",
                "a character crouching in the shadows sneaking up on an enemy"],
    "parkour": ["first-person view climbing up the wall of a building",
                "jumping and leaping between rooftops high above the city",
                "a character climbing, leaping or running across rooftops",
                "gliding or falling from a great height towards the street"],
    "hero": ["a dramatic moment of a supernatural power with a burst of fire or energy",
             "a character standing over defeated enemies lying on the ground",
             "a dramatic low-angle shot of a character against the sky"],
    "scenic": ["a beautiful wide view of a city skyline at night with snow and lights",
               "a stunning game environment with atmospheric lighting and fog"],
    "boss": ["a huge powerful boss enemy fighting the player",
             "a boss battle with a large boss health bar at the top of the screen"],
    "explore": ["first-person view walking calmly down an empty street at night",
                "walking through a hallway, a lobby or up the stairs indoors",
                "exploring a quiet place with no enemies around",
                "standing still looking at a door or a wall"],
    "dialogue": ["talking with a character during a conversation with subtitles",
                 "a dialogue choice menu with several answer options",
                 "a person standing and talking to the player"],
}


def _as_tensor(x):
    """transformers >= 5 returns a ModelOutput (embedding in .pooler_output); older versions return a tensor."""
    if hasattr(x, "pooler_output") and x.pooler_output is not None:
        return x.pooler_output
    for k in ("image_embeds", "text_embeds", "last_hidden_state"):
        if hasattr(x, k) and getattr(x, k) is not None:
            return getattr(x, k)
    return x


class SigLIP2Backend:
    name = "siglip2"

    def __init__(self, model_name, batch=64):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.dev == "cuda" else torch.float32
        try:
            self.model = AutoModel.from_pretrained(model_name, dtype=self.dtype).to(self.dev).eval()
        except TypeError:   # older transformers
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype).to(self.dev).eval()
        self.proc = AutoProcessor.from_pretrained(model_name)
        self.size = self.model.config.vision_config.image_size
        self.batch = batch

    def encode_images(self, frames):               # frames: uint8 [B,H,W,3] RGB
        torch = self.torch
        x = torch.from_numpy(frames).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True)
        x = ((x - 0.5) / 0.5).to(self.dtype)
        with torch.no_grad():
            f = _as_tensor(self.model.get_image_features(pixel_values=x))
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()

    def encode_texts(self, texts):
        torch = self.torch
        t = self.proc(text=[s.lower() for s in texts], padding="max_length", max_length=64,
                      truncation=True, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            f = _as_tensor(self.model.get_text_features(**t))
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()

    def logit_params(self):
        return float(self.model.logit_scale.exp().item()), float(self.model.logit_bias.item())


    def close(self):
        """Free the GPU memory so the next stage (Qwen-VL judge) has the whole card."""
        del self.model
        self.torch.cuda.empty_cache()


def load_backend(cfg):
    kind = cfg["analysis"]["vision_backend"]
    if kind == "heuristic":
        return None
    return SigLIP2Backend(cfg["analysis"]["model"], cfg["analysis"]["batch"])


def rescore(F, cfg):
    """Re-label cached embeddings with the current prompts (seconds, no video decode)."""
    b = load_backend(cfg)
    F["P_scene"], F["P_mom"] = score_embeddings(F["emb"].astype(np.float32), b)
    F["prompt_version"] = PROMPT_VERSION
    b.close()
    return F


def group_probs(img_emb, txt_by_class, classes, scale, bias):
    """Softmax across the classes of one group; each class = mean of its prompt scores."""
    logits = []
    for c in classes:
        t = txt_by_class[c]                                   # [P, D]
        logits.append((scale * img_emb @ t.T + bias).mean(1))  # [N]
    L = np.stack(logits, 1)
    L = L - L.max(1, keepdims=True)
    e = np.exp(L)
    return e / e.sum(1, keepdims=True)


def score_embeddings(img_emb, backend):
    classes = SCENE + MOMENT
    txt = {c: backend.encode_texts(PROMPTS[c]) for c in classes}
    scale, bias = backend.logit_params()
    return (group_probs(img_emb, txt, SCENE, scale, bias),
            group_probs(img_emb, txt, MOMENT, scale, bias))


def heuristic_probs(feat):
    """FALLBACK ONLY - no visual understanding. Derives rough class scores from audio/motion/letterbox.
    Used when no GPU/model is available (and to test the pipeline plumbing)."""
    n = len(feat["motion"])
    rk = lambda x: np.argsort(np.argsort(x)) / max(1, n - 1)
    mo, au, on, co = rk(feat["motion"]), rk(feat["rms"]), rk(feat["onset"]), rk(feat["contrast"])
    cut = np.clip(0.5 * feat["letterbox"] + 0.5 * feat["hud_absent"], 0, 1)
    black = (feat["brightness"] < 8).astype(float)
    S = {c: i for i, c in enumerate(SCENE)}; K = {c: i for i, c in enumerate(MOMENT)}
    P_scene = np.zeros((n, len(SCENE)))
    P_scene[:, S["cutscene"]] = cut
    P_scene[:, S["gameplay_action"]] = (1 - cut) * (0.5 * au + 0.5 * mo)
    P_scene[:, S["gameplay_calm"]] = (1 - cut) * (1 - P_scene[:, S["gameplay_action"]])
    P_scene[:, S["black"]] = black
    P_scene /= P_scene.sum(1, keepdims=True) + 1e-9
    M = np.zeros((n, len(MOMENT)))
    M[:, K["combat"]] = au * mo
    M[:, K["stealth"]] = (1 - au) * (1 - mo) * on
    M[:, K["parkour"]] = mo * (1 - au)
    M[:, K["hero"]] = co * (1 - mo)
    M[:, K["scenic"]] = co * (1 - mo) * (1 - au)
    M[:, K["boss"]] = au * mo * on
    M[:, K["explore"]] = (1 - au) * (1 - on)
    M /= M.sum(1, keepdims=True) + 1e-9
    return P_scene, M

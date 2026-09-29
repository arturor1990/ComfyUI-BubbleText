"""
Detecta globos de diálogo blancos en la imagen generada y escribe el texto
del usuario dentro, centrado y con el tamaño de letra más grande que quepa.

- Cada párrafo (separado por una línea en blanco) va a un globo distinto,
  en orden de lectura.
- Si el globo trae letras inventadas por el modelo, se borran antes de escribir.
"""

import logging
import math
import os
import re
import sys
import textwrap
from functools import lru_cache

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import ConvexHull

try:
    from fontTools.ttLib import TTFont
except ImportError:  # sin fontTools: solo se usa la fuente principal
    TTFont = None

log = logging.getLogger("BubbleText")

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_FONT_DIR = os.path.join(NODE_DIR, "fonts")
WIN_FONT_DIR = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
WIN_FONTS = ["comicbd.ttf", "comic.ttf", "comicz.ttf", "impact.ttf", "ariblk.ttf",
             "arialbd.ttf", "verdanab.ttf", "tahomabd.ttf", "segoeprb.ttf"]

# fuentes de respaldo para lo que la fuente elegida no tenga (emojis, símbolos)
FALLBACK_FONTS = [os.path.join(WIN_FONT_DIR, f) for f in ("seguiemj.ttf", "seguisym.ttf")]
# selectores de variante, unión de emojis y tonos de piel: sin raqm no se combinan y salen sueltos
EMOJI_JOINERS = re.compile("[︎️‍🏻-🏿]")
EMOJI_CHARS = re.compile("[☀-➿⬀-⯿🀀-🫿︎️‍]")

READING_LTR = "left → right"
READING_RTL = "right → left (manga)"
# valores de versiones anteriores, para que los workflows ya guardados sigan funcionando
LEGACY_READING = {"izquierda → derecha": READING_LTR, "derecha → izquierda (manga)": READING_RTL}


def available_fonts():
    fonts = {}
    if os.path.isdir(USER_FONT_DIR):
        for f in sorted(os.listdir(USER_FONT_DIR)):
            if f.lower().endswith((".ttf", ".otf")):
                fonts[f] = os.path.join(USER_FONT_DIR, f)
    for f in WIN_FONTS:
        p = os.path.join(WIN_FONT_DIR, f)
        if os.path.isfile(p) and f not in fonts:
            fonts[f] = p
    return fonts


def parse_color(value):
    value = value.strip().lstrip("#")
    try:
        if len(value) == 3:
            value = "".join(c * 2 for c in value)
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return (0, 0, 0)


@lru_cache(maxsize=None)
def font_codepoints(path):
    if TTFont is None:
        return None
    try:
        return frozenset(TTFont(path, lazy=True, fontNumber=0).getBestCmap().keys())
    except Exception:
        return None


class MixedFont:
    """Fuente principal + respaldo para emojis/símbolos que la principal no tenga."""

    def __init__(self, path, size):
        self.main = ImageFont.truetype(path, size)
        self.main_cmap = font_codepoints(path)
        self.fallbacks = []
        if self.main_cmap is not None:
            for fp in FALLBACK_FONTS:
                if os.path.isfile(fp) and font_codepoints(fp):
                    self.fallbacks.append((ImageFont.truetype(fp, size), font_codepoints(fp)))

    def getmetrics(self):
        return self.main.getmetrics()

    def _font_for(self, ch):
        if self.main_cmap is None or ord(ch) in self.main_cmap or ch.isspace():
            return self.main
        for font, cmap in self.fallbacks:
            if ord(ch) in cmap:
                return font
        return self.main

    def runs(self, text):
        out = []
        for ch in text:
            font = self._font_for(ch)
            if out and out[-1][0] is font:
                out[-1][1] += ch
            else:
                out.append([font, ch])
        return out

    def getlength(self, text):
        return sum(font.getlength(run) for font, run in self.runs(text))

    def draw_centered(self, draw, cx, top, text, fill):
        x = cx - self.getlength(text) / 2
        baseline = top + self.main.getmetrics()[0]
        for font, run in self.runs(text):
            draw.text((x, baseline), run, font=font, fill=fill, anchor="ls",
                      embedded_color=font is not self.main)
            x += font.getlength(run)


def split_texts(text):
    parts = re.split(r"\n\s*\n", text.strip())
    return [" ".join(p.split()) for p in parts if p.strip()]


def solidity(filled, area):
    edge = filled & ~ndimage.binary_erosion(filled)
    pts = np.argwhere(edge)
    if len(pts) < 3:
        return 0.0
    try:
        return area / ConvexHull(pts).volume
    except Exception:
        return 0.0


def count_letters(region, filled):
    """Cuántos 'agujeros' tipo letra hay dentro del globo."""
    holes = filled & ~region
    labels, n = ndimage.label(holes)
    if n == 0:
        return 0
    sizes = ndimage.sum(holes, labels, range(1, n + 1))
    return int((sizes >= 8).sum())


def outline_score(filled, lines):
    """Fracción del borde del globo con una línea oscura a menos de 3 px (sirve también con contornos finos)."""
    edge = filled & ~ndimage.binary_erosion(filled)
    if not edge.any():
        return 0.0
    near = ndimage.binary_dilation(lines, iterations=3)
    return float((edge & near).sum()) / edge.sum()


def split_joined(filled, min_area):
    """Separa globos unidos por una parte estrecha (por ejemplo, dos globos conectados por la colita)."""
    dist = ndimage.distance_transform_edt(filled)
    peak = dist.max()
    for frac in (0.2, 0.3, 0.45, 0.6):
        seeds, n = ndimage.label(dist > peak * frac)
        if n < 2:
            continue
        # cada píxel del globo pasa a la semilla más cercana
        _, (iy, ix) = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
        owner = seeds[iy, ix]
        parts = [filled & (owner == k) for k in range(1, n + 1)]
        parts = [pt for pt in parts if pt.sum() >= min_area]
        if len(parts) >= 2:
            return parts
    return []


def bubble_metrics(region, filled, lum, lines):
    area = int(filled.sum())
    ys, xs = np.nonzero(filled)
    bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    ring = ndimage.binary_dilation(filled, iterations=4) & ~filled
    paper_px = lum[region]
    return {
        "area": area,
        "extent": area / float(bh * bw),
        "whiteness": region.sum() / float(area),
        "solidity": solidity(filled, area),
        "std": float(paper_px.std()) if paper_px.size else 99.0,
        "paper": float(np.median(paper_px)) if paper_px.size else 0.0,
        "ring": float(np.median(lum[ring])) if ring.any() else 0.0,
        "letters": count_letters(region, filled),
        "outline": outline_score(filled, lines),
        "height": int(bh),
    }


def looks_like_bubble(m, dark_paper, min_outline=0.6):
    # forma: convexa (salvo la colita), no alargada ni hueca, y papel liso sin sombreado
    # contorno casi completo + letras dentro es casi seguro un globo: se tolera una colita larga
    strong = not dark_paper and m["outline"] >= 0.8 and m["letters"] >= 8
    min_solidity, min_extent = (0.75, 0.45) if strong else (0.85, 0.5)
    if m["extent"] < min_extent or m["whiteness"] < 0.55 or m["solidity"] < min_solidity or m["std"] > 12:
        return False
    if dark_paper:
        # globo negro: alrededor más claro y letras dentro (si no, es pelo, ropa, sombra...)
        return m["ring"] - m["paper"] >= 50 and m["letters"] >= 4
    if m["outline"] >= min_outline:
        return True
    # globo blanco sin contorno: solo si trae letras y destaca del fondo (las nubes y camisas no tienen letras)
    return m["letters"] >= 8 and m["paper"] >= 240 and m["paper"] - m["ring"] >= 25


def scan_bubbles(rgb, paper_mask, dark_paper):
    """Busca globos en las zonas lisas de paper_mask. dark_paper=True busca globos negros."""
    h, w, _ = rgb.shape
    total = h * w
    min_area = 0.004 * total
    lum_img = rgb.mean(axis=2)
    lines_img = lum_img > 170 if dark_paper else lum_img < 140
    paper_mask = ndimage.binary_opening(paper_mask, iterations=2)
    labels, _ = ndimage.label(paper_mask)
    bubbles = []
    for idx, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        edges = (sl[0].start == 0) + (sl[1].start == 0) + (sl[0].stop == h) + (sl[1].stop == w)
        if edges >= 2:                            # fondo pegado a esquinas
            continue
        # margen para poder mirar el contorno alrededor del globo
        y0, y1 = max(sl[0].start - 6, 0), min(sl[0].stop + 6, h)
        x0, x1 = max(sl[1].start - 6, 0), min(sl[1].stop + 6, w)
        region = labels[y0:y1, x0:x1] == idx
        filled = ndimage.binary_fill_holes(region)
        area = int(filled.sum())
        if area < min_area or area > 0.6 * total:
            continue
        lum, lines = lum_img[y0:y1, x0:x1], lines_img[y0:y1, x0:x1]
        m = bubble_metrics(region, filled, lum, lines)
        candidates = [(region, filled, m, 0.6)]
        if m["solidity"] < 0.85 and m["outline"] >= 0.6 and m["letters"] >= 4:
            # buen contorno y letras pero forma rara: varios globos unidos
            parts = split_joined(filled, min_area)
            if parts:
                candidates = [(region & pt, pt, bubble_metrics(region & pt, pt, lum, lines), 0.4) for pt in parts]
        for reg, fil, mm, min_outline in candidates:
            if not looks_like_bubble(mm, dark_paper, min_outline):
                continue
            ys, xs = np.nonzero(fil)
            bubbles.append({
                "offset": (x0, y0),
                "filled": fil,
                "white": reg,
                "area": mm["area"],
                "center": (x0 + xs.mean(), y0 + ys.mean()),
                "height": mm["height"],
                "paper": mm["paper"],
                "dark": dark_paper,
                "letters": mm["letters"],
            })
    return bubbles


def find_bubbles(rgb, white_threshold):
    """Globos blancos (con o sin contorno) y, como respaldo, globos negros con letras dentro."""
    mx = rgb.max(axis=2).astype(np.int16)
    mn = rgb.min(axis=2).astype(np.int16)
    flat = (mx - mn) <= 30
    white = scan_bubbles(rgb, (mn >= white_threshold) & flat, dark_paper=False)
    if white:
        # los globos suelen ser lo más blanco de la imagen: descarta zonas bastante más grises
        brightest = max(b["paper"] for b in white)
        white = [b for b in white if b["paper"] >= brightest - 18]
    dark = scan_bubbles(rgb, (mx <= 255 - white_threshold + 10) & flat, dark_paper=True)
    bubbles = white + dark
    # primero los que ya traen letras (donde el modelo quiso poner el texto), luego por tamaño
    bubbles.sort(key=lambda b: (b["letters"] < 4, -b["area"]))
    return bubbles


def split_across(text, bubbles):
    """Reparte una frase entre varios globos, en proporción a su tamaño y sin cortar palabras."""
    words = text.split()
    if len(bubbles) <= 1 or len(words) < 2 * len(bubbles):
        return [text]
    total = float(sum(b["area"] for b in bubbles))
    parts, i = [], 0
    for n, b in enumerate(bubbles):
        if n == len(bubbles) - 1:
            take = len(words) - i
        else:
            take = max(1, round(len(words) * b["area"] / total))
            take = min(take, len(words) - i - (len(bubbles) - n - 1))
        parts.append(" ".join(words[i:i + take]))
        i += take
    return parts


def reading_order(bubbles, img_h, order):
    order = LEGACY_READING.get(order, order)
    band = max(img_h * 0.12, 1)
    sign = -1 if order == READING_RTL else 1
    return sorted(bubbles, key=lambda b: (round(b["center"][1] / band), sign * b["center"][0]))


def wrap_options(text, font):
    """Distintas formas de partir el texto, de menos líneas a más."""
    seen, options = set(), []
    longest = max((len(wd) for wd in text.split()), default=1)
    for width in range(len(text), longest - 1, -1):
        lines = tuple(textwrap.wrap(text, width=width, break_long_words=False))
        if lines and lines not in seen:
            seen.add(lines)
            options.append(lines)
    return options


def fit_text(bubble, text, font_path, max_size, padding):
    filled = bubble["filled"]
    bh, bw = filled.shape
    pad = int(padding * min(bh, bw))
    inner = ndimage.binary_erosion(filled, iterations=pad) if pad > 0 else filled
    if not inner.any():
        inner = filled
    dist = ndimage.distance_transform_edt(inner)
    py, px = np.unravel_index(dist.argmax(), dist.shape)
    ys, xs = np.nonzero(inner)
    centers = [(xs.mean(), ys.mean()), (px, py), ((xs.mean() + px) / 2, (ys.mean() + py) / 2)]

    for size in range(max_size, 9, -1):
        font = MixedFont(font_path, size)
        ascent, descent = font.getmetrics()
        line_h = ascent + descent
        spacing = int(size * 0.12)
        for lines in wrap_options(text, font):
            block_w = max(font.getlength(l) for l in lines)
            block_h = len(lines) * line_h + (len(lines) - 1) * spacing
            for cx, cy in centers:
                x0, y0 = int(cx - block_w / 2), int(cy - block_h / 2)
                x1, y1 = x0 + math.ceil(block_w), y0 + math.ceil(block_h)
                if x0 < 0 or y0 < 0 or x1 > bw or y1 > bh:
                    continue
                if inner[y0:y1, x0:x1].all():
                    return font, lines, (cx, y0), line_h + spacing
    return None


def settings_inputs():
    # los nombres internos (activado, texto…) no cambian: los workflows guardados dependen de ellos.
    # Lo que se ve en pantalla es display_name.
    fonts = list(available_fonts().keys()) or ["(no fonts)"]
    return {
        "activado": ("BOOLEAN", {"default": True, "label_on": "ON", "label_off": "OFF", "display_name": "enabled"}),
        "texto": ("STRING", {"multiline": True, "default": "", "display_name": "text",
                             "placeholder": "Bubble text.\n\nLeave a blank line to move on to the next bubble."}),
        "fuente": (fonts, {"default": fonts[0], "display_name": "font"}),
        "mayusculas": ("BOOLEAN", {"default": True, "display_name": "uppercase"}),
        "tamano_maximo": ("INT", {"default": 64, "min": 10, "max": 300, "step": 2, "display_name": "max font size",
                                  "tooltip": "The node uses the largest size that fits, up to this one."}),
        "orden_lectura": ([READING_LTR, READING_RTL], {"display_name": "reading order"}),
        "borrar_texto_ia": ("BOOLEAN", {"default": True, "display_name": "erase AI text",
                                        "tooltip": "Erase the letters the model made up inside the bubble before writing."}),
        "color_texto": ("STRING", {"default": "auto", "display_name": "text color",
                                   "tooltip": "auto = black on light bubbles, white on dark ones. Also accepts #RRGGBB."}),
        "margen": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.4, "step": 0.01, "display_name": "margin",
                             "tooltip": "Free space between the text and the bubble edge."}),
        "umbral_blanco": ("INT", {"default": 195, "min": 150, "max": 254, "display_name": "white threshold",
                                  "tooltip": "How white the bubble must be. Lower it if slightly gray bubbles aren't detected."}),
    }


def validate_reading_order(orden_lectura):
    if orden_lectura in (READING_LTR, READING_RTL) or orden_lectura in LEGACY_READING:
        return True
    return f"Invalid reading order: {orden_lectura}"


STYLE_NATURAL = "natural (Anima)"
STYLE_TAGS = "tags (Illustrious / NoobAI)"


def bubble_prompt(texts, style=STYLE_NATURAL):
    """Fragmento de prompt: pide globos blancos con contorno negro, que es lo que mejor se detecta."""
    if style == STYLE_TAGS:
        # CLIP de SDXL entiende mejor tags que frases; el texto real lo escribe el nodo después
        if len(texts) == 1:
            return "speech bubble, (white speech bubble:1.2), black outline, english text, large speech bubble at the top"
        return (f"speech bubbles, {len(texts)} speech bubbles, (white speech bubbles:1.2), black outline, "
                "english text")
    if len(texts) == 1:
        return ("speech bubble, english text, a single large white speech bubble with a thick black outline "
                f'and black text at the top of the image, the speech bubble says "{texts[0]}"')
    ordinals = ["first", "second", "third", "fourth", "fifth", "sixth"]
    says = ", ".join(f'the {ordinals[min(i, 5)]} speech bubble says "{t}"' for i, t in enumerate(texts))
    return (f"speech bubbles, english text, {len(texts)} separate white speech bubbles with thick black outlines "
            f"and black text, {says}")


def text_color(setting, paper_lum):
    """Color del texto: el elegido, salvo que no contraste con el papel del globo."""
    light_paper = paper_lum >= 128
    if setting.strip().lower() == "auto":
        return (0, 0, 0) if light_paper else (255, 255, 255)
    color = parse_color(setting)
    if abs(sum(color) / 3.0 - paper_lum) < 100:
        return (0, 0, 0) if light_paper else (255, 255, 255)
    return color


def erase_bubble(pil, rgb, bubble):
    """Pinta el globo entero con el color de su papel (borra las letras de la IA)."""
    ox, oy = bubble["offset"]
    fh, fw = bubble["filled"].shape
    paper = np.median(rgb[oy:oy + fh, ox:ox + fw][bubble["white"]], axis=0).astype(np.uint8)
    crop = np.array(pil)[oy:oy + fh, ox:ox + fw]
    crop[bubble["filled"]] = paper
    pil.paste(Image.fromarray(crop), (ox, oy))


def render_bubbles(image, cfg):
    b, h, w, _ = image.shape
    empty_mask = torch.zeros((b, h, w), dtype=torch.float32)
    texts = split_texts(cfg["texto"])
    if not cfg["activado"] or not texts:
        return (image, empty_mask)
    font_path = available_fonts().get(cfg["fuente"])
    if not font_path:
        log.warning("[BubbleText] Font not found: %s", cfg["fuente"])
        return (image, empty_mask)
    texts = [EMOJI_JOINERS.sub("", t) for t in texts]
    if cfg["mayusculas"]:
        texts = [t.upper() for t in texts]

    out_images, out_masks = [], []
    for i in range(b):
        rgb = (image[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        mask = np.zeros((h, w), dtype=np.float32)
        bubbles = find_bubbles(rgb, cfg["umbral_blanco"])
        if not bubbles:
            log.warning("[BubbleText] No speech bubble found in image %d.", i)
        lettered = [bb for bb in bubbles if bb["letters"] >= 4]
        if len(texts) == 1 and len(lettered) > 1:
            # el modelo repartió la frase en varios globos: la repartimos igual, en orden de lectura
            chosen = reading_order(lettered[:4], h, cfg["orden_lectura"])
            texts_i = split_across(texts[0], chosen)
            chosen = chosen[:len(texts_i)]
        else:
            if len(bubbles) < len(texts):
                log.warning("[BubbleText] %d texts but only %d bubbles.", len(texts), len(bubbles))
            chosen = reading_order(bubbles[:len(texts)], h, cfg["orden_lectura"])
            texts_i = texts
        # globos con letras de la IA que no reciben texto: se limpian para que no quede basura
        leftover = [bb for bb in lettered if all(bb is not c for c in chosen)] if cfg["borrar_texto_ia"] else []

        pil = Image.fromarray(rgb)
        for bubble in leftover:
            erase_bubble(pil, rgb, bubble)
        draw = ImageDraw.Draw(pil)
        for bubble, text in zip(chosen, texts_i):
            ox, oy = bubble["offset"]
            fh, fw = bubble["filled"].shape
            if cfg["borrar_texto_ia"]:
                erase_bubble(pil, rgb, bubble)
                draw = ImageDraw.Draw(pil)
            fit = fit_text(bubble, text, font_path, cfg["tamano_maximo"], cfg["margen"])
            if fit is None:
                log.warning("[BubbleText] Text doesn't fit in the bubble: %r", text)
                continue
            font, lines, (cx, top), step = fit
            color = text_color(cfg["color_texto"], bubble["paper"])
            for n, line in enumerate(lines):
                font.draw_centered(draw, ox + cx, oy + top + n * step, line, color)
            mask[oy:oy + fh, ox:ox + fw][bubble["filled"]] = 1.0

        out_images.append(torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0))
        out_masks.append(torch.from_numpy(mask))
    return (torch.stack(out_images), torch.stack(out_masks))


class _LoraManagerImageExtractor:
    """Para LoRA Manager: que la receta use la imagen con el texto ya escrito.

    LoRA Manager toma como imagen de la receta la salida del primer VAE Decode,
    que todavía tiene las letras de la IA. Tras escribir el texto, la sustituimos.
    """

    images_key = "images"

    @staticmethod
    def extract(node_id, inputs, outputs, metadata):
        pass

    @classmethod
    def update(cls, node_id, outputs, metadata):
        images = metadata.setdefault(cls.images_key, {})
        entry = {"node_id": node_id, "image": outputs}
        images[node_id] = entry
        images["first_decode"] = entry


def register_with_lora_manager():
    """Engancha los nodos que escriben texto al recolector de metadatos de LoRA Manager, si está instalado."""
    for name, module in list(sys.modules.items()):
        if name.endswith("metadata_collector.node_extractors") and hasattr(module, "NODE_EXTRACTORS"):
            _LoraManagerImageExtractor.images_key = getattr(module, "IMAGES", "images")
            for node_type in ("SpeechBubbleRender", "SpeechBubbleTextAuto"):
                module.NODE_EXTRACTORS.setdefault(node_type, _LoraManagerImageExtractor)


class SpeechBubblePrompt:
    """Como un LoRA para el prompt: si está activo añade el globo con el texto."""

    @classmethod
    def INPUT_TYPES(cls):
        # estilo_prompt va al final para no descolocar los valores de workflows ya guardados
        return {"required": {"prompt": ("STRING", {"forceInput": True}), **settings_inputs(),
                             "estilo_prompt": ([STYLE_NATURAL, STYLE_TAGS], {
                                 "display_name": "prompt style",
                                 "tooltip": "natural: full sentences (Anima). tags: Danbooru style (Illustrious, NoobAI, Pony)."})}}

    @classmethod
    def VALIDATE_INPUTS(cls, orden_lectura):
        return validate_reading_order(orden_lectura)

    RETURN_TYPES = ("STRING", "BUBBLE_CONFIG")
    RETURN_NAMES = ("prompt", "bubble")
    FUNCTION = "run"
    CATEGORY = "image/text"
    DESCRIPTION = ("Works like a LoRA for your prompt: when enabled, it asks the model for a clean white speech bubble. "
                   "Connect 'bubble' to 💬 Bubble Text · Write.")

    def run(self, prompt, **cfg):
        # los emojis no van al prompt: el modelo los dibuja mal y el T5 no los conoce
        texts = [" ".join(EMOJI_CHARS.sub(" ", t).split()) for t in split_texts(cfg["texto"])]
        texts = [t for t in texts if t]
        if cfg["activado"] and texts:
            style = cfg.get("estilo_prompt", STYLE_NATURAL)
            prompt = f"{prompt.rstrip().rstrip(',')}, {bubble_prompt(texts, style)}"
        return (prompt, cfg)


class SpeechBubbleRender:
    """Escribe el texto configurado en '💬 Globo de texto' sobre la imagen final."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "globo": ("BUBBLE_CONFIG", {"display_name": "bubble"})}}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "bubble mask")
    FUNCTION = "run"
    CATEGORY = "image/text"
    DESCRIPTION = "Finds the speech bubbles, erases the AI's letters and writes your text with a real font."

    def run(self, image, globo):
        register_with_lora_manager()
        return render_bubbles(image, globo)


class SpeechBubbleText:
    """Versión todo en uno (sin tocar el prompt)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), **settings_inputs()}}

    @classmethod
    def VALIDATE_INPUTS(cls, orden_lectura):
        return validate_reading_order(orden_lectura)

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "bubble mask")
    FUNCTION = "run"
    CATEGORY = "image/text"
    DESCRIPTION = "All-in-one: writes your text in the speech bubbles without touching the prompt."

    def run(self, image, **cfg):
        register_with_lora_manager()
        return render_bubbles(image, cfg)

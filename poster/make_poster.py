"""Pending AI posteri: qatlı PSD + PNG + JPG."""
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from psd_tools import PSDImage
from psd_tools.constants import BlendMode

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

W, H = 2480, 3508  # A4, 300 dpi
CX, CY = W // 2, 1330
FONT = "/usr/share/fonts/opentype/inter/"
random.seed(7)

CYAN = (34, 211, 238)
VIOLET = (139, 92, 246)
PINK = (236, 72, 153)
WHITE = (241, 245, 255)
MUTED = (160, 170, 205)


def font(name, size):
    return ImageFont.truetype(FONT + name, size)


def blank():
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def gradient(w, h, stops, angle=0.0):
    """Xətti gradient; stops = [(pos, (r,g,b)), ...]."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    a = math.radians(angle)
    t = x * math.cos(a) + y * math.sin(a)
    t = (t - t.min()) / (t.max() - t.min())
    pos = [s[0] for s in stops]
    out = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        out[..., c] = np.interp(t, pos, [s[1][c] for s in stops])
    return out


def text_fill(size, text, fnt, stops, angle=0.0, tracking=0):
    """Gradient rəngli mətn (RGBA, mətnin ölçüsündə)."""
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    draw_tracked(d, (0, 0), text, fnt, 255, tracking)
    g = gradient(size[0], size[1], stops, angle).astype(np.uint8)
    img = Image.fromarray(g, "RGB").convert("RGBA")
    img.putalpha(mask)
    return img


def tracked_width(text, fnt, tracking):
    return sum(fnt.getlength(ch) for ch in text) + tracking * (len(text) - 1)


def draw_tracked(d, xy, text, fnt, fill, tracking=0):
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=fnt, fill=fill)
        x += fnt.getlength(ch) + tracking


def centered_text(layer, y, text, fnt, fill, tracking=0):
    d = ImageDraw.Draw(layer)
    x = (W - tracked_width(text, fnt, tracking)) / 2
    draw_tracked(d, (x, y), text, fnt, fill, tracking)


layers = []  # (ad, şəkil, opacity, blend)

# 1. Fon
bg = gradient(W, H, [(0, (5, 6, 22)), (0.55, (14, 12, 48)), (1, (6, 8, 26))], 80)
y, x = np.mgrid[0:H, 0:W].astype(np.float32)
vign = 1 - 0.55 * np.clip(np.hypot((x - CX) / W, (y - CY) / H) * 1.6, 0, 1) ** 2
bg = np.clip(bg * vign[..., None], 0, 255).astype(np.uint8)
layers.append(("Fon", Image.fromarray(bg, "RGB").convert("RGBA"), 255, BlendMode.NORMAL))

# 2. Şəbəkə (grid)
grid = blank()
d = ImageDraw.Draw(grid)
step = 124
for gx in range(0, W + 1, step):
    d.line([(gx, 0), (gx, H)], fill=(120, 140, 255, 26), width=2)
for gy in range(0, H + 1, step):
    d.line([(0, gy), (W, gy)], fill=(120, 140, 255, 26), width=2)
fade = np.clip(1 - np.hypot((x - CX) / (W * 0.75), (y - CY) / (H * 0.55)), 0, 1)
ga = np.array(grid)
ga[..., 3] = (ga[..., 3] * fade).astype(np.uint8)
layers.append(("Şəbəkə", Image.fromarray(ga, "RGBA"), 255, BlendMode.NORMAL))

# 3. İşıq ləkələri
glow = blank()
d = ImageDraw.Draw(glow)
for (gx, gy, r, col, a) in [
    (CX - 520, CY - 380, 620, VIOLET, 120),
    (CX + 560, CY + 260, 560, CYAN, 95),
    (CX + 200, CY - 700, 380, PINK, 70),
    (CX - 700, 2900, 520, VIOLET, 60),
    (CX + 760, 3150, 460, CYAN, 45),
]:
    d.ellipse([gx - r, gy - r, gx + r, gy + r], fill=col + (a,))
glow = glow.filter(ImageFilter.GaussianBlur(220))
layers.append(("İşıq ləkələri", glow, 255, BlendMode.SCREEN))

# 4. Neyron şəbəkə
net = blank()
d = ImageDraw.Draw(net)
cols = [260, 640, 1840, 2220]
nodes = []
for i, nx in enumerate(cols):
    n = 5 if i in (0, 3) else 4
    ys = np.linspace(CY - 620, CY + 620, n)
    nodes.append([(nx + random.randint(-40, 40), int(ny)) for ny in ys])
pairs = [(0, 1), (2, 3)]
for a, b in pairs:
    for p in nodes[a]:
        for q in nodes[b]:
            if random.random() < 0.8:
                d.line([p, q], fill=(140, 160, 255, 55), width=3)
for layer_nodes in (nodes[1], nodes[2]):  # mərkəzə bağlantı
    for p in layer_nodes:
        ang = math.atan2(p[1] - CY, p[0] - CX)
        q = (CX + 600 * math.cos(ang), CY + 600 * math.sin(ang))
        d.line([p, q], fill=(34, 211, 238, 70), width=3)
for k, layer_nodes in enumerate(nodes):
    col = CYAN if k % 2 else VIOLET
    for (px, py) in layer_nodes:
        d.ellipse([px - 26, py - 26, px + 26, py + 26], outline=col + (220,), width=5)
        d.ellipse([px - 11, py - 11, px + 11, py + 11], fill=col + (255,))
# kiçik ulduz/partikullar
for _ in range(260):
    px, py = random.randint(0, W), random.randint(0, H)
    r = random.choice([2, 2, 3, 4])
    d.ellipse([px - r, py - r, px + r, py + r], fill=(200, 210, 255, random.randint(40, 150)))
net_glow = net.filter(ImageFilter.GaussianBlur(10))
net = Image.alpha_composite(net_glow, net)
layers.append(("Neyron şəbəkə", net, 255, BlendMode.NORMAL))

# 5. Mərkəzi "pending" halqası (yüklənmə simvolu)
ring = blank()
d = ImageDraw.Draw(ring)
R = 560
d.ellipse([CX - R, CY - R, CX + R, CY + R], outline=(110, 120, 200, 70), width=6)
# seqmentli yüklənmə qövsü: 0-dan ~72%-ə qədər gradient rəngdə
segs, filled = 60, 43
for i in range(segs):
    a0 = -90 + i * 360 / segs + 1.2
    a1 = a0 + 360 / segs - 2.4
    if i < filled:
        t = i / filled
        col = tuple(int(VIOLET[c] + (CYAN[c] - VIOLET[c]) * t) for c in range(3)) + (255,)
    else:
        col = (70, 80, 140, 110)
    r2 = R - 40
    d.arc([CX - r2, CY - r2, CX + r2, CY + r2], a0, a1, fill=col, width=46)
r3 = R - 110
d.ellipse([CX - r3, CY - r3, CX + r3, CY + r3], outline=(34, 211, 238, 90), width=4)
for i in range(120):  # tik işarələri
    a = math.radians(i * 3)
    r_in, r_out = (r3 - 34, r3 - 12) if i % 10 else (r3 - 52, r3 - 12)
    d.line([(CX + r_in * math.cos(a), CY + r_in * math.sin(a)),
            (CX + r_out * math.cos(a), CY + r_out * math.sin(a))],
           fill=(180, 190, 255, 90 if i % 10 else 180), width=3)
# qövsün ucunda parlaq nöqtə
ea = math.radians(-90 + filled * 360 / segs)
ex, ey = CX + (R - 40) * math.cos(ea), CY + (R - 40) * math.sin(ea)
d.ellipse([ex - 34, ey - 34, ex + 34, ey + 34], fill=WHITE + (255,))
ring = Image.alpha_composite(ring.filter(ImageFilter.GaussianBlur(18)), ring)
layers.append(("Yüklənmə halqası", ring, 255, BlendMode.NORMAL))

# 6. AI çipi
chip = blank()
d = ImageDraw.Draw(chip)
cs = 250
for i in range(-3, 4):  # çip ayaqları
    o = i * 64
    for (x0, y0, x1, y1) in [
        (CX + o - 12, CY - cs - 70, CX + o + 12, CY - cs),
        (CX + o - 12, CY + cs, CX + o + 12, CY + cs + 70),
        (CX - cs - 70, CY + o - 12, CX - cs, CY + o + 12),
        (CX + cs, CY + o - 12, CX + cs + 70, CY + o + 12),
    ]:
        d.rounded_rectangle([x0, y0, x1, y1], 6, fill=(150, 170, 255, 230))
body = gradient(2 * cs, 2 * cs, [(0, (30, 27, 75)), (1, (12, 20, 52))], 45).astype(np.uint8)
body_img = Image.fromarray(body, "RGB").convert("RGBA")
m = Image.new("L", (2 * cs, 2 * cs), 0)
ImageDraw.Draw(m).rounded_rectangle([0, 0, 2 * cs - 1, 2 * cs - 1], 60, fill=255)
body_img.putalpha(m)
chip.alpha_composite(body_img, (CX - cs, CY - cs))
d = ImageDraw.Draw(chip)
d.rounded_rectangle([CX - cs, CY - cs, CX + cs, CY + cs], 60, outline=CYAN + (255,), width=8)
d.rounded_rectangle([CX - cs + 40, CY - cs + 40, CX + cs - 40, CY + cs - 40], 36,
                    outline=(139, 92, 246, 160), width=4)
f_ai = font("InterDisplay-Black.otf", 250)
bb = d.textbbox((0, 0), "AI", font=f_ai)
tw, th = bb[2] - bb[0], bb[3] - bb[1]
ai_txt = text_fill((tw + 20, th + bb[1] + 20), "AI", f_ai, [(0, CYAN), (1, PINK)], 30)
chip.alpha_composite(ai_txt, (int(CX - tw / 2 - bb[0]), int(CY - th / 2 - bb[1])))
chip = Image.alpha_composite(chip.filter(ImageFilter.GaussianBlur(14)), chip)
layers.append(("AI çipi", chip, 255, BlendMode.NORMAL))

# 7. Üst etiket
top = blank()
f_lab = font("Inter-SemiBold.otf", 54)
label = "SÜNİ İNTELLEKT LAYİHƏSİ"
lw = tracked_width(label, f_lab, 14)
d = ImageDraw.Draw(top)
d.rounded_rectangle([(W - lw) / 2 - 60, 230, (W + lw) / 2 + 60, 340], 55,
                    outline=(34, 211, 238, 200), width=4, fill=(34, 211, 238, 22))
centered_text(top, 252, label, f_lab, CYAN + (255,), 14)
layers.append(("Üst etiket", top, 255, BlendMode.NORMAL))

# 8. Başlıq
title = blank()
f_title = font("InterDisplay-Black.otf", 380)
trk = 6
w1 = tracked_width("PENDING", f_title, trk)
gap_w = f_title.getlength(" ") * 0.9
w2 = tracked_width("AI", f_title, trk)
th = f_title.getbbox("PENDING")[3] + 40
tx, ty = int((W - (w1 + gap_w + w2)) / 2), 2060
t1 = text_fill((int(w1) + 40, th), "PENDING", f_title,
               [(0, WHITE), (1, (196, 205, 255))], 0, trk)
t2 = text_fill((int(w2) + 40, th), "AI", f_title, [(0, CYAN), (1, PINK)], 20, trk)
words = blank()
words.alpha_composite(t1, (tx, ty))
words.alpha_composite(t2, (int(tx + w1 + gap_w), ty))
shadow = Image.new("RGBA", (W, H), VIOLET + (0,))
shadow.putalpha(words.getchannel("A").point(lambda v: v * 0.75))
title.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(45)))
title.alpha_composite(words)
layers.append(("Başlıq — PENDING AI", title, 255, BlendMode.NORMAL))

# 9. Şüar
sub = blank()
f_sub = font("Inter-Light.otf", 92)
centered_text(sub, 2520, "Gələcək yüklənir…", f_sub, WHITE + (235,), 2)
d = ImageDraw.Draw(sub)
for i, col in enumerate([VIOLET, (100, 150, 245), CYAN]):  # üç nöqtə
    px = CX - 60 + i * 60
    d.ellipse([px - 12, 2690 - 12, px + 12, 2690 + 12], fill=col + (255 - i * 60,))
layers.append(("Şüar", sub, 255, BlendMode.NORMAL))

# 10. Xüsusiyyətlər
feat = blank()
d = ImageDraw.Draw(feat)
cards = [
    ("01", "Maşın öyrənməsi", ["Verilənlərdən öyrənən", "ağıllı modellər"]),
    ("02", "Neyron şəbəkələr", ["İnsan beynindən", "ilhamlanan arxitektura"]),
    ("03", "Ağıllı həllər", ["Real problemlərə", "praktik tətbiqlər"]),
]
cw, chh, gap = 680, 380, 60
x0 = (W - (3 * cw + 2 * gap)) // 2
cy0 = 2800
f_num = font("InterDisplay-Bold.otf", 64)
f_h = font("Inter-Bold.otf", 58)
f_p = font("Inter-Regular.otf", 42)
for i, (num, head, lines) in enumerate(cards):
    cx0 = x0 + i * (cw + gap)
    accent = [VIOLET, (100, 150, 245), CYAN][i]
    d.rounded_rectangle([cx0, cy0, cx0 + cw, cy0 + chh], 40,
                        fill=(255, 255, 255, 14), outline=accent + (150,), width=3)
    d.rounded_rectangle([cx0 + 50, cy0, cx0 + 170, cy0 + 8], 4, fill=accent + (255,))
    d.text((cx0 + 50, cy0 + 50), num, font=f_num, fill=accent + (255,))
    d.text((cx0 + 50, cy0 + 145), head, font=f_h, fill=WHITE + (255,))
    for j, ln in enumerate(lines):
        d.text((cx0 + 50, cy0 + 232 + j * 56), ln, font=f_p, fill=MUTED + (255,))
layers.append(("Xüsusiyyətlər", feat, 255, BlendMode.NORMAL))

# 11. Alt hissə
foot = blank()
d = ImageDraw.Draw(foot)
d.line([(160, 3300), (W - 160, 3300)], fill=(140, 150, 220, 70), width=2)
f_ft = font("Inter-Medium.otf", 46)
f_ftb = font("Inter-SemiBold.otf", 46)
d.text((160, 3345), "Müəllif: ", font=f_ft, fill=MUTED + (255,))
d.text((160 + f_ft.getlength("Müəllif: "), 3345), "Rashid Dursunov", font=f_ftb, fill=WHITE + (255,))
right = "Qrup PA401  ·  2026"
d.text((W - 160 - f_ft.getlength(right), 3345), right, font=f_ft, fill=MUTED + (255,))
layers.append(("Alt hissə", foot, 255, BlendMode.NORMAL))

# --- PSD yığılması ---
psd = PSDImage.new("RGB", (W, H), color=(5, 6, 22))
for name, img, op, blend in layers:
    bbox = img.getchannel("A").getbbox()
    if not bbox:
        continue
    crop = img.crop(bbox)
    lyr = psd.create_pixel_layer(crop, name="Layer", left=bbox[0], top=bbox[1],
                                 opacity=op, blend_mode=blend)
    lyr.name = name  # Unicode ad (ə, ş, ü) üçün
psd.save(OUT / "pending_ai_poster.psd")

# Önizləmə (PNG/JPG) — Screen rejimini öz əlimizlə tətbiq edirik
comp = Image.new("RGBA", (W, H), (5, 6, 22, 255))
for name, img, op, blend in layers:
    if blend == BlendMode.SCREEN:
        base = np.asarray(comp, np.float32)[..., :3] / 255
        top_ = np.asarray(img, np.float32)
        a = top_[..., 3:] / 255
        scr = 1 - (1 - base) * (1 - top_[..., :3] / 255)
        mix = base * (1 - a) + scr * a
        comp = Image.fromarray((mix * 255).astype(np.uint8), "RGB").convert("RGBA")
    else:
        comp = Image.alpha_composite(comp, img)
comp = comp.convert("RGB")
comp.save(OUT / "pending_ai_poster.png", dpi=(300, 300))
comp.save(OUT / "pending_ai_poster.jpg", quality=92, dpi=(300, 300))
comp.resize((W // 4, H // 4), Image.LANCZOS).save(OUT.parent / "preview_small.png")
print("ok")

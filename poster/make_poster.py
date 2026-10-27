"""Pending AI poster in an Apple-style Liquid Glass look: layered PSD + PNG + JPG."""
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
CX = W // 2
FONT = "/usr/share/fonts/opentype/inter/"
rng = np.random.default_rng(11)

BG = (22, 13, 8)
AMBER = (232, 146, 64)
BRONZE = (156, 80, 38)
ROSE = (204, 118, 96)
CREAM = (252, 232, 206)
WHITE = (255, 255, 255)


def font(name, size):
    return ImageFont.truetype(FONT + name, size)


def blank():
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def lerp(a, b, t):
    return tuple(int(a[c] + (b[c] - a[c]) * t) for c in range(3))


def linear_gradient(w, h, stops, angle=90.0):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    a = math.radians(angle)
    t = x * math.cos(a) + y * math.sin(a)
    t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
    out = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        out[..., c] = np.interp(t, [s[0] for s in stops], [s[1][c] for s in stops])
    return out


def tracked_width(text, fnt, tracking):
    return sum(fnt.getlength(ch) for ch in text) + tracking * (len(text) - 1)


def draw_tracked(d, xy, text, fnt, fill, tracking=0):
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=fnt, fill=fill)
        x += fnt.getlength(ch) + tracking


def gradient_text(text, fnt, stops, angle=90.0, tracking=0):
    """Text filled with a gradient, sized to the text; returns (img, width)."""
    tw = tracked_width(text, fnt, tracking)
    h = fnt.getbbox(text)[3] + 20
    mask = Image.new("L", (int(tw) + 20, h), 0)
    draw_tracked(ImageDraw.Draw(mask), (0, 0), text, fnt, 255, tracking)
    img = Image.fromarray(linear_gradient(mask.width, h, stops, angle).astype(np.uint8))
    img = img.convert("RGBA")
    img.putalpha(mask)
    return img, tw


def screen(base, top):
    """Screen-blend an RGBA layer onto an RGB float array (0..1)."""
    t = np.asarray(top, np.float32)
    a = t[..., 3:] / 255
    s = 1 - (1 - base) * (1 - t[..., :3] / 255)
    return base * (1 - a) + s * a


def over(base, top):
    t = np.asarray(top, np.float32)
    a = t[..., 3:] / 255
    return base * (1 - a) + t[..., :3] / 255 * a


def sample(img, xs, ys):
    """Bilinear sample of an (h, w, 3) array at float coordinates."""
    h, w = img.shape[:2]
    xs = np.clip(xs, 0, w - 1.001)
    ys = np.clip(ys, 0, h - 1.001)
    x0, y0 = xs.astype(int), ys.astype(int)
    fx, fy = (xs - x0)[..., None], (ys - y0)[..., None]
    a = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    b = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return a * (1 - fy) + b * fy


def rrect_sdf(px, py, box, r):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hx, hy = (x1 - x0) / 2 - r, (y1 - y0) / 2 - r
    qx, qy = np.abs(px - cx) - hx, np.abs(py - cy) - hy
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    return outside + np.minimum(np.maximum(qx, qy), 0) - r


def liquid_glass(scene, box, r, bezel=70, refract=1.0, frost=10, tint=(255, 246, 236),
                 tint_amt=0.10, lift=0.05, shadow=0.35):
    """Render a Liquid Glass panel over `scene` (H, W, 3 float 0..1); returns RGBA layer."""
    pad = 140
    x0, y0, x1, y1 = box
    bx0, by0 = max(int(x0) - pad, 0), max(int(y0) - pad, 0)
    bx1, by1 = min(int(x1) + pad, W), min(int(y1) + pad, H)
    py, px = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
    sdf = rrect_sdf(px + 0.5, py + 0.5, box, r)
    gy, gx = np.gradient(sdf)
    norm = np.hypot(gx, gy) + 1e-6
    nx, ny = gx / norm, gy / norm  # outward normal

    # Refraction: the bezel bends the backdrop like a convex lens edge.
    d = np.clip(-sdf, 0, None)
    t = np.clip(1 - d / bezel, 0, 1)
    disp = refract * bezel * t ** 2.2
    frosted = np.asarray(
        Image.fromarray((scene * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(frost)),
        np.float32) / 255
    col = sample(frosted, px - nx * disp, py - ny * disp)
    # chromatic fringe on the bezel
    col[..., 0] = sample(frosted, px - nx * disp * 1.06, py - ny * disp * 1.06)[..., 0]
    col[..., 2] = sample(frosted, px - nx * disp * 0.94, py - ny * disp * 0.94)[..., 2]

    lum = col.mean(axis=2, keepdims=True)
    col = lum + (col - lum) * 1.18  # a touch more vivid, like Apple's glass
    col = col * (1 - tint_amt) + np.array(tint, np.float32) / 255 * tint_amt + lift

    # Specular rim: bright top-left, softer bottom-right, thin edge line all round.
    lx, ly = -0.55, -0.83
    facing = nx * lx + ny * ly
    rim = np.exp(-d / 5.0)
    wide = np.exp(-d / 26.0)
    spec = (np.clip(facing, 0, 1) ** 1.5 * (0.85 * rim + 0.30 * wide)
            + np.clip(-facing, 0, 1) ** 2 * (0.45 * rim + 0.12 * wide)
            + 0.22 * rim)
    col = col + spec[..., None] * (1 - col) * 1.0
    # gentle top sheen
    sheen = np.clip(1 - (py - y0) / ((y1 - y0) * 0.55), 0, 1) ** 2 * 0.06
    col = np.clip(col + sheen[..., None], 0, 1)

    alpha = np.clip(0.5 - sdf, 0, 1)

    # Soft contact shadow beneath the glass
    sh = Image.fromarray((np.clip(0.5 - rrect_sdf(px + 0.5, py - 26, box, r), 0, 1) * 255)
                         .astype(np.uint8)).filter(ImageFilter.GaussianBlur(38))
    sh_a = np.asarray(sh, np.float32) / 255 * shadow

    out_a = alpha + sh_a * (1 - alpha)
    shadow_col = np.array([10, 5, 2], np.float32) / 255
    out_rgb = (col * alpha[..., None] + shadow_col * (sh_a * (1 - alpha))[..., None]) \
        / np.maximum(out_a[..., None], 1e-6)
    rgba = np.dstack([np.clip(out_rgb, 0, 1) * 255, out_a * 255]).astype(np.uint8)
    layer = blank()
    layer.paste(Image.fromarray(rgba, "RGBA"), (bx0, by0))
    return layer


def supersample(size, draw_fn, scale=4):
    """Draw at `scale`x and downsample for smooth anti-aliased shapes."""
    big = Image.new("RGBA", (size[0] * scale, size[1] * scale), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(big), scale)
    return big.resize(size, Image.LANCZOS)


P_LIGHT = (246, 222, 188)
P_MID = (222, 176, 122)
P_DEEP = (176, 120, 72)


def tailed_p(size, ss=3):
    """Custom 'P' in light brown tones whose stem flows into a twisted, curling ribbon tail.

    Returns (image, origin_x, advance): draw the image at (x - origin_x, y) to put the
    glyph's text origin at (x, y), same as ImageDraw.text would.
    """
    S = size * ss
    f = font("InterDisplay-Bold.otf", S)
    adv = f.getlength("P")
    bb = f.getbbox("P")
    stem = S * 0.19
    ox = int(stem * 5.0)
    cw, ch = int(ox + adv + stem), int(bb[3] + stem * 4.2)
    gmask = Image.new("L", (cw, ch), 0)
    ImageDraw.Draw(gmask).text((ox, 0), "P", font=f, fill=255)
    m = np.asarray(gmask) > 128
    yb = bb[3] - 3
    cols = np.nonzero(m[yb])[0]
    xl = cols[0]
    xr = xl
    while xr + 1 < cw and m[yb, xr + 1]:
        xr += 1
    s = xr - xl + 1
    cxs = (xl + xr) / 2

    fill = Image.fromarray(linear_gradient(cw, ch, [(0, P_LIGHT), (0.75, P_MID), (1, P_DEEP)],
                                           90).astype(np.uint8)).convert("RGBA")
    fill.putalpha(gmask)
    img = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    img.alpha_composite(fill)

    # Tail centre-line: a descender straight out of the stem, then a wide inward
    # spiral sweeping to the left.
    pts = []
    drop = s * 0.8
    for k in range(120):  # starts inside the stem so the join is seamless
        pts.append((cxs, yb - s + k / 119 * (s + drop)))
    R0 = s * 1.9
    C = (cxs - R0, yb + drop)
    turns = 1.3
    for k in range(1, 2400):
        th = k / 2399 * turns * 2 * math.pi
        R = R0 * (1 - 0.68 * th / (turns * 2 * math.pi))
        pts.append((C[0] + R * math.cos(th), C[1] + R * math.sin(th)))
    n = len(pts)
    t0 = 120 / n  # where the spiral (and the twisting) begins
    d = ImageDraw.Draw(img)
    stem_col = tuple(int(v) for v in np.asarray(fill)[yb, int(cxs), :3])
    twists = 3  # half-turns of the ribbon along the spiral
    for i, (px, py) in enumerate(pts):
        t = i / (n - 1)
        u = max(t - t0, 0) / (1 - t0)
        taper = 1 - 0.8 * u ** 0.9
        face = math.cos(u * twists * math.pi)
        r = s / 2 * taper * (0.2 + 0.8 * abs(face))
        shade = 0.5 + 0.5 * face  # front face light, back face deeper brown
        col = lerp(P_DEEP, P_LIGHT, 0.2 + 0.8 * shade)
        if u < 0.08:
            col = lerp(stem_col, col, u / 0.08)
        d.ellipse([px - r, py - r, px + r, py + r], fill=col + (255,))
    img.alpha_composite(fill)  # keep the bowl/stem crisp above the tail start
    out = img.resize((cw // ss, ch // ss), Image.LANCZOS)
    return out, ox / ss, adv / ss


layers = []  # (name, image, blend)

# 1. Background: warm mesh gradient + grain
y, x = np.mgrid[0:H, 0:W].astype(np.float32)
bg = linear_gradient(W, H, [(0, (40, 23, 14)), (0.5, (26, 15, 9)), (1, (14, 8, 5))], 90) / 255
blobs = blank()
d = ImageDraw.Draw(blobs)
for (bx, by, r, c, a) in [
    (380, 820, 760, AMBER, 230), (2150, 620, 640, ROSE, 200),
    (2000, 1850, 720, BRONZE, 220), (1240, 1260, 420, CREAM, 150),
    (300, 2250, 620, BRONZE, 170), (2300, 2900, 620, AMBER, 140),
    (700, 3300, 560, ROSE, 110), (1240, 2350, 900, (60, 32, 18), 200),
]:
    d.ellipse([bx - r, by - r, bx + r, by + r], fill=c + (a,))
blobs = blobs.filter(ImageFilter.GaussianBlur(260))
bg = over(bg, blobs)
bg = np.clip(bg + rng.normal(0, 0.012, (H, W, 1)).astype(np.float32), 0, 1)
layers.append(("Background", Image.fromarray((bg * 255).astype(np.uint8)).convert("RGBA"),
               BlendMode.NORMAL))

# 2. Silk ribbons: flowing light that the glass refracts
def ribbons_fn(dr, s):
    xs = np.linspace(-200, W + 200, 400)
    specs = [(1150, 260, 0.0021, 0.4, 110, (255, 200, 140, 120)),
             (1320, 220, 0.0024, 1.9, 46, (255, 230, 200, 150)),
             (980, 300, 0.0018, 2.8, 70, (230, 140, 90, 110)),
             (1500, 180, 0.0027, 4.1, 26, (255, 240, 220, 170)),
             (2950, 160, 0.0020, 1.2, 60, (240, 160, 100, 80))]
    for (y0, amp, f, ph, wdt, c) in specs:
        pts = [(xx * s, (y0 + amp * math.sin(xx * f + ph) + 0.08 * amp * math.sin(xx * f * 3.1)) * s)
               for xx in xs]
        dr.line(pts, fill=c, width=int(wdt * s), joint="curve")

ribbons = supersample((W, H), ribbons_fn, scale=1).filter(ImageFilter.GaussianBlur(22))
layers.append(("Silk ribbons", ribbons, BlendMode.SCREEN))
scene = screen(bg, ribbons)

# 3. Hero glass orb
OCX, OCY, OR = CX, 1230, 600
orb = liquid_glass(scene, (OCX - OR, OCY - OR, OCX + OR, OCY + OR), OR, bezel=130,
                   refract=1.35, frost=14, tint=(60, 34, 20), tint_amt=0.18, lift=0.02)
layers.append(("Glass orb", orb, BlendMode.NORMAL))

# 4. Progress ring (Apple Activity-style, round caps, conic gradient)
RR, RW, PROG = 470, 46, 0.72
py_, px_ = np.mgrid[0:H, 0:W].astype(np.float32)
dx, dy = px_ - OCX, py_ - OCY
rad = np.hypot(dx, dy)
ang = (np.degrees(np.arctan2(dx, -dy)) % 360) / 360  # 0 at 12 o'clock, clockwise
band = np.clip(RW / 2 + 0.5 - np.abs(rad - RR), 0, 1)
track_a = band * 0.16
end = PROG * 2 * math.pi
caps = []
for a_ in (0.0, end):
    cxp, cyp = OCX + RR * math.sin(a_), OCY - RR * math.cos(a_)
    caps.append(np.clip(RW / 2 + 0.5 - np.hypot(px_ - cxp, py_ - cyp), 0, 1))
prog_a = np.maximum(band * (ang <= PROG), np.maximum(caps[0], caps[1]))
t = np.clip(ang / PROG, 0, 1)
t = np.where(caps[1] > 0, 1.0, np.where((caps[0] > 0) & (ang > PROG), 0.0, t))
stops = [(0, (186, 92, 46)), (0.5, AMBER), (1, (255, 226, 180))]
ring_rgb = np.dstack([np.interp(t, [s[0] for s in stops], [s[1][c] for s in stops])
                      for c in range(3)])
ring_a = np.maximum(prog_a, track_a)
ring_rgb = np.where((prog_a > 0)[..., None], ring_rgb, 255)
ring = Image.fromarray(np.dstack([ring_rgb, ring_a * 255]).astype(np.uint8), "RGBA")
ring_glow = Image.fromarray(np.dstack([ring_rgb, prog_a * 140]).astype(np.uint8), "RGBA")
ring = Image.alpha_composite(ring_glow.filter(ImageFilter.GaussianBlur(28)), ring)
del py_, px_, dx, dy, rad, ang, band, caps, prog_a, t, ring_rgb, ring_a
layers.append(("Progress ring", ring, BlendMode.NORMAL))

# 5. Logo inside the orb: the tailed P
mark = blank()
logo, _, _ = tailed_p(500)
lb = logo.getchannel("A").getbbox()
logo = logo.crop(lb)
shadow = Image.new("RGBA", (W, H), (20, 10, 4, 0))
pos = (int(OCX - logo.width / 2 + 20), int(OCY - logo.height / 2))
mark.alpha_composite(logo, pos)
shadow.putalpha(mark.getchannel("A").point(lambda v: v * 0.45))
shadow = shadow.transform(shadow.size, Image.AFFINE, (1, 0, 0, 0, 1, -14))
mark = Image.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(24)), mark)
layers.append(("Logo P", mark, BlendMode.NORMAL))

# 6. Top pill
f_pill = font("Inter-Medium.otf", 48)
pill_txt = "Artificial Intelligence Project"
pw = f_pill.getlength(pill_txt) + 70 + 150
pill_box = (CX - pw / 2, 250, CX + pw / 2, 380)
layers.append(("Top pill glass", liquid_glass(scene, pill_box, 65, bezel=34, refract=0.9,
                                              frost=8, shadow=0.25), BlendMode.NORMAL))
pill = blank()
d = ImageDraw.Draw(pill)
dot_x = pill_box[0] + 75
d.ellipse([dot_x - 15, 315 - 15, dot_x + 15, 315 + 15], fill=AMBER + (255,))
pill_glow = blank()
ImageDraw.Draw(pill_glow).ellipse([dot_x - 26, 289, dot_x + 26, 341], fill=AMBER + (160,))
pill = Image.alpha_composite(pill_glow.filter(ImageFilter.GaussianBlur(10)), pill)
ImageDraw.Draw(pill).text((dot_x + 45, 286), pill_txt, font=f_pill, fill=WHITE + (240,))
layers.append(("Top pill text", pill, BlendMode.NORMAL))

# 7. Title
title = blank()
f_title = font("InterDisplay-Bold.otf", 330)
cap_p, cap_ox, cap_adv = tailed_p(330)
e_img, e_w = gradient_text("ending", f_title, [(0, WHITE), (1, (250, 236, 220))], 90, -9)
a_img, a_w = gradient_text("AI", f_title, [(0, (255, 214, 150)), (0.55, AMBER), (1, ROSE)],
                           25, -9)
space = f_title.getlength(" ") * 0.85
p_w = cap_adv - 9 + e_w
tx, ty = int(CX - (p_w + space + a_w) / 2), 1990
title.alpha_composite(cap_p, (int(tx - cap_ox), ty))
title.alpha_composite(e_img, (int(tx + cap_adv - 9), ty))
title.alpha_composite(a_img, (int(tx + p_w + space), ty))
glow = Image.new("RGBA", (W, H), AMBER + (0,))
glow.putalpha(title.getchannel("A").point(lambda v: v * 0.35))
title = Image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(60)), title)
layers.append(("Title", title, BlendMode.NORMAL))

# 8. Tagline
tag = blank()
f_tag = font("Inter-Regular.otf", 86)
tag_txt = "The future is loading."
ImageDraw.Draw(tag).text((CX - f_tag.getlength(tag_txt) / 2, 2440), tag_txt, font=f_tag,
                         fill=(255, 240, 226, 200))
layers.append(("Tagline", tag, BlendMode.NORMAL))

# 9. Feature cards (glass) + content
cards = [
    ("Machine Learning", ["Smart models that", "learn from data"], "bars"),
    ("Neural Networks", ["Architecture inspired", "by the human brain"], "nodes"),
    ("Smart Solutions", ["Practical answers", "to real problems"], "sparkle"),
]
cw, ch, gap, cy0 = 690, 480, 50, 2680
x0 = (W - (3 * cw + 2 * gap)) // 2
card_glass = blank()
for i in range(3):
    cx0 = x0 + i * (cw + gap)
    card_glass.alpha_composite(liquid_glass(scene, (cx0, cy0, cx0 + cw, cy0 + ch), 64,
                                            bezel=46, refract=1.0, frost=12, shadow=0.3))
layers.append(("Card glass", card_glass, BlendMode.NORMAL))


def icon_base(size=136):
    """App-icon style squircle with a warm gradient."""
    sq = supersample((size, size), lambda dr, s: dr.rounded_rectangle(
        [0, 0, size * s - 1, size * s - 1], int(size * s * 0.28), fill=(255, 255, 255, 255)))
    grad = Image.fromarray(linear_gradient(size, size, [(0, (250, 180, 100)), (1, BRONZE)],
                                           60).astype(np.uint8)).convert("RGBA")
    grad.putalpha(sq.getchannel("A"))
    return grad


content = blank()
d = ImageDraw.Draw(content)
f_h = font("Inter-SemiBold.otf", 58)
f_p = font("Inter-Regular.otf", 42)


def glyph(kind, size=136):
    """White glyph only (transparent background)."""
    def fn(dr, s):
        S = size * s
        w_ = (255, 255, 255, 255)
        if kind == "bars":
            for k, hgt in enumerate([0.30, 0.48, 0.66]):
                bx = S * (0.24 + k * 0.19)
                dr.rounded_rectangle([bx, S * (0.78 - hgt), bx + S * 0.13, S * 0.78],
                                     int(S * 0.04), fill=w_)
        elif kind == "nodes":
            pts = [(px * S, py * S) for px, py in [(0.5, 0.26), (0.27, 0.70), (0.73, 0.70)]]
            for a_, b_ in [(0, 1), (1, 2), (2, 0)]:
                dr.line([pts[a_], pts[b_]], fill=w_, width=int(S * 0.05))
            for (px, py) in pts:
                r_ = S * 0.095
                dr.ellipse([px - r_, py - r_, px + r_, py + r_], fill=w_)
        else:
            c_, R_, r_ = S / 2, S * 0.32, S * 0.075
            star = []
            for k in range(8):
                rr = R_ if k % 2 == 0 else r_
                a_ = math.radians(k * 45 - 90)
                star.append((c_ + rr * math.cos(a_), c_ + rr * math.sin(a_)))
            dr.polygon(star, fill=w_)
    return supersample((size, size), fn)


for i, (head, lines, kind) in enumerate(cards):
    cx0 = x0 + i * (cw + gap)
    ic = icon_base()
    ic.alpha_composite(glyph(kind))
    content.alpha_composite(ic, (cx0 + 56, cy0 + 56))
    d.text((cx0 + 56, cy0 + 236), head, font=f_h, fill=WHITE + (255,))
    for j, ln in enumerate(lines):
        d.text((cx0 + 56, cy0 + 322 + j * 54), ln, font=f_p, fill=(255, 238, 222, 185))
layers.append(("Card content", content, BlendMode.NORMAL))

# 10. Footer capsule
fbox = (160, 3270, W - 160, 3410)
layers.append(("Footer glass", liquid_glass(scene, fbox, 70, bezel=36, refract=0.9, frost=10,
                                            shadow=0.25), BlendMode.NORMAL))
foot = blank()
d = ImageDraw.Draw(foot)
f_lbl = font("Inter-Regular.otf", 44)
f_nm = font("Inter-SemiBold.otf", 46)
d.text((235, 3312), "Author", font=f_lbl, fill=(255, 238, 222, 170))
d.text((235 + f_lbl.getlength("Author") + 26, 3311), "Rashid Dursunov", font=f_nm,
       fill=WHITE + (255,))
yr = "2026"
d.text((W - 235 - f_nm.getlength(yr), 3311), yr, font=f_nm, fill=WHITE + (235,))
layers.append(("Footer text", foot, BlendMode.NORMAL))

# --- Build the PSD ---
psd = PSDImage.new("RGB", (W, H), color=BG)
for name, img, blend in layers:
    bbox = img.getchannel("A").getbbox()
    if bbox:
        psd.create_pixel_layer(img.crop(bbox), name=name, left=bbox[0], top=bbox[1],
                               blend_mode=blend)
psd.save(OUT / "pending_ai_poster.psd")

# Flattened preview
comp = np.zeros((H, W, 3), np.float32)
for name, img, blend in layers:
    comp = screen(comp, img) if blend == BlendMode.SCREEN else over(comp, img)
comp = Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8))
comp.save(OUT / "pending_ai_poster.png", dpi=(300, 300))
comp.save(OUT / "pending_ai_poster.jpg", quality=92, dpi=(300, 300))
print("ok")

# Standalone logo (transparent PNG)
logo_big, _, _ = tailed_p(1400)
logo_big = logo_big.crop(logo_big.getchannel("A").getbbox())
pad = 120
canvas = Image.new("RGBA", (logo_big.width + 2 * pad, logo_big.height + 2 * pad), (0, 0, 0, 0))
canvas.alpha_composite(logo_big, (pad, pad))
canvas.save(OUT / "pending_ai_logo.png")

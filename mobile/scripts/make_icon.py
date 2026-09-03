"""앱 아이콘 생성 — 연갈색 배경 + 궁서체 '알' + 청록 스파크.
궁서체는 Windows batang.ttc(궁서 포함). Pillow로 여러 에셋 크기를 뽑는다.
실행: backend/.venv/Scripts/python.exe mobile/scripts/make_icon.py
"""
import os, sys
from PIL import Image, ImageDraw, ImageFont

# ── 팔레트 ──
BROWN = (203, 176, 120)      # 연갈색 배경 (#CBB078)
INK = (46, 37, 27)           # 먹색 글자 (#2E251B)
TEAL = (63, 160, 149)        # 청록 스파크 (#3FA095)
HANJI = (245, 237, 221)      # 한지빛

BATANG = r"C:/Windows/Fonts/batang.ttc"
GUNGSUH_IDX = 2              # 0 Batang, 1 BatangChe, 2 Gungsuh, 3 GungsuhChe
GLYPH = "알"

def font(size, path=BATANG, idx=GUNGSUH_IDX):
    return ImageFont.truetype(path, size, index=idx)

def draw_glyph(img, cx, cy, glyph_size, spark=True, ink=INK):
    d = ImageDraw.Draw(img)
    f = font(glyph_size)
    d.text((cx, cy), GLYPH, font=f, fill=ink, anchor="mm")
    if spark:
        # 글자 우상단 스파크
        r = int(glyph_size * 0.055)
        sx, sy = cx + int(glyph_size * 0.30), cy - int(glyph_size * 0.34)
        d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=TEAL)

def rounded_mask(size, radius):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return m

def draw_border(img, size, inset_ratio=0.075, width_ratio=0.022, radius_ratio=0.16, color=INK):
    d = ImageDraw.Draw(img)
    ins = int(size * inset_ratio)
    w = max(2, int(size * width_ratio))
    d.rounded_rectangle([ins, ins, size - 1 - ins, size - 1 - ins],
                        radius=int(size * radius_ratio), outline=color + (255,), width=w)

def full_icon(size, bg, rounded=False, spark=True, ink=INK, glyph_ratio=0.60, border=False):
    img = Image.new("RGBA", (size, size), bg + (255,))
    draw_glyph(img, size // 2, int(size * 0.52), int(size * glyph_ratio), spark=spark, ink=ink)
    if border:
        draw_border(img, size)
    if rounded:
        img.putalpha(rounded_mask(size, int(size * 0.23)))
    return img

def foreground(size, spark=True):
    # 적응형 전경: 투명 배경, 중앙 안전영역(66%)에 글자
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw_glyph(img, size // 2, int(size * 0.52), int(size * 0.44), spark=spark)
    return img

def solid(size, color):
    return Image.new("RGBA", (size, size), color + (255,))

def bg_border(size, bg):
    # 적응형 배경: 연갈색 + 안전 인셋(스퀴클 마스크에서도 보이게) 먹색 프레임
    img = Image.new("RGBA", (size, size), bg + (255,))
    draw_border(img, size, inset_ratio=0.10, width_ratio=0.020, radius_ratio=0.14)
    return img

def monochrome(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw_glyph(img, size // 2, int(size * 0.52), int(size * 0.42), spark=False, ink=(0, 0, 0))
    return img

if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "mobile/assets"
    os.makedirs(outdir, exist_ok=True)
    which = sys.argv[2] if len(sys.argv) > 2 else "preview"

    if which == "preview":
        border = "--noborder" not in sys.argv
        # 비교 몽타주: 테두리 없음(풀) | 테두리 있음(풀) | 테두리 있음(라운드 런처)
        S = 460
        no_b = full_icon(1024, BROWN, border=False).resize((S, S))
        with_b = full_icon(1024, BROWN, border=True).resize((S, S))
        rnd = full_icon(512, BROWN, rounded=True, border=True).resize((S, S))
        pad, gap = 56, 56
        W = pad*2 + S*3 + gap*2
        H = pad*2 + S + 46
        mont = Image.new("RGBA", (W, H), HANJI + (255,))
        for i, im in enumerate((no_b, with_b, rnd)):
            mont.alpha_composite(im, (pad + i*(S+gap), pad))
        d = ImageDraw.Draw(mont)
        lf = ImageFont.truetype(r"C:/Windows/Fonts/malgun.ttf", 26)
        for i, lbl in enumerate(("테두리 없음", "테두리 있음", "런처(라운드)")):
            d.text((pad + i*(S+gap) + S//2, pad + S + 22), lbl, font=lf, fill=INK+(255,), anchor="mm")
        mont.save(os.path.join(outdir, "_preview_border.png"))
        print("preview saved:", os.path.join(outdir, "_preview_border.png"))
    else:
        full_icon(1024, BROWN, border=True).save(os.path.join(outdir, "icon.png"))
        foreground(1024).save(os.path.join(outdir, "android-icon-foreground.png"))
        bg_border(1024, BROWN).save(os.path.join(outdir, "android-icon-background.png"))
        monochrome(1024).save(os.path.join(outdir, "android-icon-monochrome.png"))
        full_icon(1024, BROWN, rounded=True, border=True).save(os.path.join(outdir, "splash-icon.png"))
        full_icon(196, BROWN, rounded=True, border=True).resize((48, 48)).save(os.path.join(outdir, "favicon.png"))
        # 적응형 합성 검증본(전경+배경) + 스퀴클/원형 마스크
        comp = Image.alpha_composite(bg_border(1024, BROWN), foreground(1024))
        comp.save(os.path.join(outdir, "_check_adaptive.png"))
        for shape, rad in (("_check_squircle.png", int(1024*0.28)), ("_check_circle.png", 512)):
            c = comp.copy(); c.putalpha(rounded_mask(1024, rad)); c.save(os.path.join(outdir, shape))
        print("assets saved to", outdir)

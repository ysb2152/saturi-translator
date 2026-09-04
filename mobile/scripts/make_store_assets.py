"""플레이스토어 리스팅 자산 생성.
- icon-512.png            : 하이레스 아이콘(512)
- feature-graphic.png     : 피처 그래픽 1024x500
- store-screenshot-*.png  : E2E 스크린샷을 2:1로 정리(Play 비율 요건)
실행: backend/.venv/Scripts/python.exe mobile/scripts/make_store_assets.py <out_dir>
"""
import os, sys, glob
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_icon import full_icon, draw_glyph, BROWN, INK, HANJI, BATANG, GUNGSUH_IDX

TEAL_INK = (33, 82, 73)     # #215249
TEAL = (63, 160, 149)       # #3FA095
SUB = (138, 124, 104)       # #8A7C68
MALGUN = r"C:/Windows/Fonts/malgun.ttf"
MALGUNBD = r"C:/Windows/Fonts/malgunbd.ttf"

def gungsuh(sz): return ImageFont.truetype(BATANG, sz, index=GUNGSUH_IDX)
def malgun(sz, bold=False): return ImageFont.truetype(MALGUNBD if bold else MALGUN, sz)

def make_512(out):
    full_icon(1024, BROWN, border=True).resize((512, 512), Image.LANCZOS).save(os.path.join(out, "icon-512.png"))

def make_feature(out):
    W, H = 1024, 500
    img = Image.new("RGBA", (W, H), HANJI + (255,))
    # 은은한 배경 결(청록 blob)
    blob = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(blob).ellipse([W-260, H-260, W+120, H+120], fill=TEAL + (16,))
    img.alpha_composite(blob)
    # 아이콘(왼쪽)
    ic = full_icon(1024, BROWN, rounded=True, border=True).resize((300, 300), Image.LANCZOS)
    img.alpha_composite(ic, (72, 100))
    # 텍스트(오른쪽)
    d = ImageDraw.Draw(img)
    d.text((440, 150), "알아 묵나?", font=gungsuh(104), fill=INK + (255,))
    d.text((444, 292), "사투리를 표준어로", font=malgun(44, True), fill=TEAL_INK + (255,))
    d.text((444, 356), "충청·강원·전라·경상 · 인터넷 없이 기기 안에서", font=malgun(26), fill=SUB + (255,))
    img.convert("RGB").save(os.path.join(out, "feature-graphic.png"))

def make_screens(out, shots):
    for i, src in enumerate(shots, 1):
        im = Image.open(src).convert("RGB")
        w, h = im.size
        if h > 2 * w:  # 2:1보다 길면 상하 크롭(상단 상태바·하단 내비 위주로)
            excess = h - 2 * w
            top = int(excess * 0.40)
            im = im.crop((0, top, w, top + 2 * w))
        im.save(os.path.join(out, f"store-screenshot-{i}.png"))

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "mobile/store-assets"
    os.makedirs(out, exist_ok=True)
    make_512(out)
    make_feature(out)
    shots = sorted(glob.glob("docs/screenshots/ondevice-e2e-*.png"))
    make_screens(out, shots)
    print("saved to", out, "->", os.listdir(out))

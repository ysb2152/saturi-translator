"""MUSAN tar에서 noise 서브셋만 추출(music/speech 제외). 추출 후 개수 출력.
tar 삭제는 이 스크립트가 하지 않음 — 개수 확인 후 별도로 삭제할 것."""
import tarfile, os, sys

TAR = "data/musan.tar.gz"
OUT = "data/noise"
os.makedirs(OUT, exist_ok=True)

cnt = 0
with tarfile.open(TAR) as t:
    for m in t:
        name = m.name.replace("\\", "/")
        if m.isfile() and "/noise/" in name:
            t.extract(m, OUT)
            cnt += 1
            if cnt % 100 == 0:
                print(f"  {cnt} extracted...", flush=True)

print(f"noise 멤버 추출 완료: {cnt}")
if cnt == 0:
    print("경고: 0개 — tar 확인 필요")
    sys.exit(1)

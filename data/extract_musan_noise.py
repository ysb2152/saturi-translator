"""MUSAN tar에서 noise 서브셋만 뽑는다(music/speech는 제외). 뽑고 나서 개수를 출력한다.
tar는 여기서 안 지우니, 개수 확인하고 따로 지울 것."""
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

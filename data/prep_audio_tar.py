"""AI Hub 음성 .tar(안에 .zip.partN)를 zip으로 복원하고 스트리밍 전처리까지 한 번에 돌린다.

다운로드한 .tar 하나랑 라벨 폴더만 주면 STT 클립이 나온다(self-serve).

  training/.venv/Scripts/python.exe data/prep_audio_tar.py \
      --tar "C:/Users/ysb21/Downloads/download (7).tar" \
      --labels data/raw_label_gs \
      --out data/processed_gs_big --max-clips 12000

여러 지역을 한 매니페스트로 합칠 땐 두 번째부터 --append를 붙인다.
zip 복원본은 temp에 만들었다가 끝나면 지운다(디스크 절약)."""
import argparse, glob, os, re, subprocess, sys, tarfile, tempfile, shutil

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass


def reconstruct_zip(tar_path, workdir):
    """tar 안의 *.zip.partN을 번호순으로 이어붙여 하나의 .zip으로 복원한다. 이미 단일 .zip이면 그대로 쓴다."""
    print(f"[1/3] tar 풀기: {tar_path}")
    with tarfile.open(tar_path) as tf:
        tf.extractall(workdir)
    parts = glob.glob(os.path.join(workdir, "**", "*.zip.part*"), recursive=True)
    if parts:
        def partno(p):
            m = re.search(r"\.part(\d+)$", p)
            return int(m.group(1)) if m else 0
        parts.sort(key=partno)
        zip_path = os.path.join(workdir, "reconstructed.zip")
        print(f"[2/3] zip 복원: part {len(parts)}개 이어붙이기")
        with open(zip_path, "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out, length=16 * 1024 * 1024)
                os.remove(p)  # 이어붙인 part 즉시 삭제(디스크 절약)
        return zip_path
    zips = glob.glob(os.path.join(workdir, "**", "*.zip"), recursive=True)
    if zips:
        print("[2/3] 단일 zip 발견 — 그대로 사용")
        return zips[0]
    raise SystemExit("tar 안에서 .zip / .zip.part 를 찾지 못했습니다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True, help="AI Hub 음성 .tar 경로")
    ap.add_argument("--labels", required=True, help="라벨 JSON 폴더")
    ap.add_argument("--out", default="data/processed_gs_big")
    ap.add_argument("--max-clips", type=int, default=12000)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--keep-zip", action="store_true", help="복원 zip 유지(기본은 끝나고 삭제)")
    args = ap.parse_args()

    workdir = tempfile.mkdtemp(prefix="aihub_", dir=os.path.dirname(os.path.abspath(args.out)) or ".")
    try:
        zip_path = reconstruct_zip(args.tar, workdir)
        print(f"[3/3] 스트리밍 전처리 → {args.out}")
        cmd = [sys.executable, "data/preprocess_streaming.py",
               "--zip", zip_path, "--labels", args.labels,
               "--out", args.out, "--max-clips", str(args.max_clips)]
        if args.append:
            cmd.append("--append")
        subprocess.run(cmd, check=True)
    finally:
        if not args.keep_zip:
            shutil.rmtree(workdir, ignore_errors=True)
            print("temp 정리 완료")
    print("완료.")


if __name__ == "__main__":
    main()

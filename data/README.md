# data — AI Hub 방언 데이터 전처리

AI Hub "한국어 방언 발화" 데이터를 학습용 산출물로 변환한다.

- **STT 매니페스트** (`processed/stt/{train,val}.jsonl`): Whisper 파인튜닝용
  `{"audio_filepath", "text"(=방언 전사), "duration"}`
- **변환 문장쌍** (`processed/mt/{train,val}.jsonl`): KoBART/T5 파인튜닝용
  `{"dialect", "standard"}`

## 지금 바로 검증 (실제 데이터 불필요)

```bash
python data/test_preprocess.py
```
합성 샘플 생성 → 전처리 → 산출물 검증까지 한 번에 돈다. 파이프라인이 도는지 먼저 확인용.

## ⚠️ 지오블록: AI Hub는 해외 IP 다운로드를 차단

Colab 등 해외 서버에서 `aihubshell`로 받으면 `AI 허브는 해외에서의 데이터 다운로드를
제한하고 있습니다` 로 막힌다. **반드시 한국 IP(집 PC 등)에서 다운로드**한 뒤,
학습 환경(Colab)으로 옮긴다:
- 라벨(작음)·소량은 로컬에서 받아 이 스크립트로 바로 전처리.
- 음성(zip당 17~28GB)은 로컬 34GB에 안 들어가면 외장하드에 받아 전처리 →
  **작은 클립만** Google Drive 업로드 → Colab에서 학습.

## 실제 데이터로 실행

1. AI Hub에서 데이터 다운로드 (계정·승인·수동, **한국 IP에서**)
   - 예: [한국어 방언 발화(경상도)](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=119),
     [전라도](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=120),
     [충청도](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=122)
   - 압축 해제 후 라벨 JSON + 오디오(WAV)가 있는 루트를 준비.
2. 전처리 실행
   ```bash
   python data/preprocess.py --raw <추출_루트> --out data/processed --val-ratio 0.05
   ```
   - 오디오가 **세션 단위**(긴 파일 + 발화 start/end)면 기본 동작으로 발화별 클립을 잘라 저장.
   - 오디오가 **이미 발화 단위**면 `--no-slice` 추가.
   - 디버그: `--limit 20` 으로 라벨 20개만.

## ⚠️ 라벨 스키마는 데이터셋마다 조금 다르다

AI Hub 방언 데이터는 연도/지역별로 필드명이 다르다. 이 전처리는 흔한 구조를 기본값으로 두고
없는 필드는 순차 대체하도록 짰지만, **실제 파일을 받으면 라벨 JSON 하나를 열어
[`aihub_schema.py`](aihub_schema.py)의 `FIELDS`가 맞는지 확인**하는 것이 가장 확실하다.

가정하는 구조(요약):
```json
{
  "metadata": { "audioPath": "audio/xxx.wav" },
  "utterance": [
    { "dialect_form": "가가 억수로 머라카노",
      "standard_form": "그 아이가 굉장히 뭐라고 하니",
      "eojeolList": [ {"eojeol":"가가","standard":"그 아이가","isDialect":true} ],
      "start": 0.0, "end": 2.4 }
  ]
}
```
- 문장 필드(`dialect_form`/`standard_form`)가 없으면 `eojeolList`로 조립, 그것도 없으면 `form` 사용.
- 전사 마커(`(A)/(B)` 이중전사, `{웃음}`, `(())`, 어절 뒤 `b/`·`l/` 등)는
  [`clean_text`](aihub_schema.py)가 정제. 방언/표준 각각 다른 쪽을 선택.

## 다지역(전국 사투리) 확장

파이프라인은 지역 무관이라, 다른 지역 데이터를 같은 방식으로 처리해 **합쳐서 학습**하면
한 모델이 여러 사투리를 커버한다. 변환(KoBART)은 라벨(작음)만 있으면 되므로 특히 쉽다.

지역별 dataSetSn: 경상도 119 · 전라도 120 · 강원도 121 · 충청도 122 · 제주도 123
(제주는 어휘 차이가 커 난이도 높음 → 마지막 확장 권장)

```bash
# 1) 지역별 라벨을 각각 받아 전처리 (한국 IP에서 다운로드)
python data/preprocess.py --raw data/raw_label_gyeongsang --out data/processed_gs
python data/preprocess.py --raw data/raw_label_jeolla    --out data/processed_jl
python data/preprocess.py --raw data/raw_label_chungcheong --out data/processed_cc

# 2) 여러 지역 변환쌍을 합쳐 균형 학습셋 구성(중복 자동 제거)
python data/build_mt_dataset.py \
    --in-dir data/processed_gs/mt data/processed_jl/mt data/processed_cc/mt \
    --out-dir data/processed/mt_balanced

# 3) 이 통합 세트로 KoBART 재학습 → 다지역 변환 모델
```

STT(음성)는 지역마다 17~28GB라, 먼저 한 지역으로 증명하고 나머지는 소량씩 추가하는 편이 현실적.

## 대용량 음성(STT): 스트리밍 전처리 — 외장하드 없이

원천(음성) zip은 하나가 17~28GB라 압축을 다 풀면 로컬 34GB에 안 들어간다.
`preprocess_streaming.py`는 **zip을 통째로 풀지 않고** WAV를 하나씩 꺼내(temp) →
발화 구간으로 잘라 작은 클립 저장 → temp 삭제, 를 반복한다. 피크 디스크 =
zip 크기 + temp WAV 1개 + 누적 클립.

```bash
# 라벨(전사)은 먼저 따로 받아 풀어둔다(작음) → data/raw_label
python data/preprocess_streaming.py \
    --zip "(비식별화완료)경상도_1.zip" \
    --labels data/raw_label \
    --out data/processed \
    --max-clips 20000        # 포트폴리오용 소량만(0=무제한)
```

- `--max-clips` 로 개수를 제한하면 클립 몇 GB만 남아 **무료 Google Drive**에 올려 Colab 학습 가능.
- 음성 zip이 여러 개면 두 번째부터 `--append` 로 이어쓰기(분할은 세션 해시라 일관).
- 검증: `python data/test_streaming.py` (합성 zip으로 E2E).

## 파일 구성

| 파일 | 역할 |
|------|------|
| `aihub_schema.py` | 라벨 파싱 + 전사 정제(필드 매핑은 여기 `FIELDS`) |
| `wav_utils.py` | PCM WAV 길이 계산·구간 슬라이스(표준 라이브러리) |
| `preprocess.py` | 메인: raw → STT 매니페스트 + 변환 문장쌍 + `stats.json` |
| `preprocess_streaming.py` | 대용량 음성 zip을 안 풀고 스트리밍 처리(외장하드 없이) |
| `build_mt_dataset.py` | 변환 학습셋 구성(중복 제거 + 동일쌍 다운샘플링) |
| `analyze.py` | 데이터 EDA → `analysis.md` |
| `make_sample.py` | 검증용 합성 샘플 생성 |
| `test_preprocess.py`, `test_streaming.py` | 스모크 테스트 |

## 다음 단계

- STT: `processed/stt/*.jsonl` 로 Whisper 파인튜닝(Colab). 표준 Whisper 대비 CER/WER 비교.
- 변환: `processed/mt/*.jsonl` 로 KoBART seq2seq 파인튜닝 → 백엔드 `DialectConverter` 교체.
- WAV가 아닌 포맷이면 `wav_utils`가 슬라이스 못 하므로, ffmpeg로 16kHz mono WAV 변환 후 사용 권장.

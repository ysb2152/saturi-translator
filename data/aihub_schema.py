"""AI Hub 한국어 방언 발화 라벨(JSON) 파싱 + 전사 텍스트 정제.

AI Hub 방언 데이터는 연도/지역별로 라벨 JSON 필드명이 조금씩 다르다. 여기서는
가장 흔한 구조를 기본값으로 두고, 없는 필드는 순차적으로 대체(fallback)하도록 짰다.
실제 파일을 받으면 아래 FIELDS만 맞추면 대부분 동작한다.

가정하는 기본 구조:
{
  "metadata": { "audioPath": "xxx.wav", ... },     # 없으면 JSON 파일명 기준으로 오디오 매칭
  "utterance": [
    {
      "id": "u1",
      "form": "원문 전사",
      "standard_form": "그 아이가 굉장히 뭐라고 하니",   # 표준어
      "dialect_form":  "가가 억수로 머라카노",            # 방언
      "eojeolList": [ {"eojeol":"가가","standard":"그 아이가","isDialect":true}, ... ],
      "start": 0.0, "end": 2.4                          # 세션 오디오 내 구간(초)
    }, ...
  ]
}
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ── 필드 매핑(실제 데이터에 맞게 여기만 수정하면 됨) ──────────────────────────
FIELDS = {
    # 발화 리스트가 담긴 최상위 키 후보(순서대로 탐색)
    "utterance_keys": ["utterance", "utterances", "sentence", "sentences", "dialogs", "data"],
    # 표준어/방언 문장 필드 후보
    "standard_keys": ["standard_form", "standardForm", "standard", "std", "standard_text"],
    "dialect_keys": ["dialect_form", "dialectForm", "dialect", "form", "dialect_text", "text"],
    # 어절 리스트(문장 필드가 없을 때 조립용)
    "eojeol_list_keys": ["eojeolList", "eojeol_list", "eojeols", "word"],
    "eojeol_surface_keys": ["eojeol", "dialect", "surface", "word"],
    "eojeol_standard_keys": ["standard", "standard_form", "std"],
    # 구간 타임스탬프
    "start_keys": ["start", "startTime", "start_time", "begin"],
    "end_keys": ["end", "endTime", "end_time"],
    # 세션 오디오 경로가 메타데이터에 있을 경우
    "audio_path_keys": ["audioPath", "audio_path", "fileName", "file_name", "recordPath"],
    "metadata_keys": ["metadata", "meta", "dataSet"],
}


@dataclass
class Utterance:
    dialect: str            # 방언(사투리) 전사 → STT 타깃
    standard: str           # 표준어 전사 → 변환 모델 타깃
    start: float | None
    end: float | None

    @property
    def duration(self) -> float | None:
        if self.start is None or self.end is None:
            return None
        d = round(self.end - self.start, 3)
        return d if d > 0 else None


def _first(d: dict, keys: list[str]):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] not in (None, ""):
            return d[k]
    return None


# ── 전사 텍스트 정제 ─────────────────────────────────────────────────────────
# AI Hub/KsponSpeech 계열 전사에 섞이는 마커들을 제거한다.
_DUAL = re.compile(r"\(([^()]*)\)/\(([^()]*)\)")   # (A)/(B) 이중 전사
_BRACE = re.compile(r"\{[^{}]*\}")                  # {laughing} 등
_UNCLEAR = re.compile(r"\(\(([^()]*)\)\)")          # (( )) 불명확 구간
_KSPON_MARK = re.compile(r"(?<=\S)[bnloue]/")       # 단어b/ l/ o/ n/ u/ e/
_NAME_TAG = re.compile(r"&[^&]*&")                  # &name& 익명화 태그
_SYMBOLS = re.compile(r"[+*#/~·]")                  # 잡기호
_MULTISPACE = re.compile(r"\s+")


def clean_text(s: str, mode: str = "dialect") -> str:
    """전사 문자열 정제. mode='dialect'면 (A)/(B)에서 A(방언), 'standard'면 B(표준) 선택."""
    if not s:
        return ""
    # (A)/(B) 이중 전사 → 한쪽 선택
    s = _DUAL.sub(lambda m: m.group(1) if mode == "dialect" else m.group(2), s)
    s = _UNCLEAR.sub(r"\1", s)      # (( 내용 )) → 내용
    s = _BRACE.sub(" ", s)          # {소음/웃음} 제거
    s = _NAME_TAG.sub(" ", s)       # 익명 태그 제거
    s = _KSPON_MARK.sub("", s)      # 어절 뒤 마커 제거
    s = _SYMBOLS.sub(" ", s)        # 잡기호 제거
    s = _MULTISPACE.sub(" ", s)     # 공백 정리
    return s.strip()


# ── 파싱 ─────────────────────────────────────────────────────────────────────
def get_utterance_list(obj: dict) -> list[dict]:
    """라벨 JSON에서 발화 리스트를 찾아 반환."""
    # 중첩된 dataSet.dialogs 같은 구조도 한 단계 내려가며 탐색
    for k in FIELDS["utterance_keys"]:
        v = obj.get(k)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
        if isinstance(v, dict):  # 한 단계 더
            for kk in FIELDS["utterance_keys"]:
                vv = v.get(kk)
                if isinstance(vv, list) and vv and isinstance(vv[0], dict):
                    return vv
    return []


def _from_eojeols(utt: dict) -> tuple[str, str]:
    """문장 필드가 없을 때 어절 리스트로 방언/표준 문장을 조립."""
    lst = _first(utt, FIELDS["eojeol_list_keys"])
    if not isinstance(lst, list):
        return "", ""
    dia, std = [], []
    for e in lst:
        if not isinstance(e, dict):
            continue
        surf = _first(e, FIELDS["eojeol_surface_keys"]) or ""
        stdw = _first(e, FIELDS["eojeol_standard_keys"]) or surf  # 표준 없으면 표면형 사용
        dia.append(str(surf))
        std.append(str(stdw))
    return " ".join(dia), " ".join(std)


def extract_utterance(utt: dict) -> Utterance | None:
    """발화 dict 하나에서 방언/표준 전사와 타임스탬프를 뽑아 Utterance로."""
    raw_dialect = _first(utt, FIELDS["dialect_keys"])
    raw_standard = _first(utt, FIELDS["standard_keys"])

    if raw_dialect is None and raw_standard is None:
        raw_dialect, raw_standard = _from_eojeols(utt)
    if raw_standard is None:
        raw_standard = raw_dialect       # 표준 없으면 방언과 동일 취급
    if raw_dialect is None:
        raw_dialect = raw_standard

    dialect = clean_text(str(raw_dialect), mode="dialect")
    standard = clean_text(str(raw_standard), mode="standard")
    if not dialect and not standard:
        return None

    start = _first(utt, FIELDS["start_keys"])
    end = _first(utt, FIELDS["end_keys"])
    try:
        start = float(start) if start is not None else None
        end = float(end) if end is not None else None
    except (TypeError, ValueError):
        start = end = None

    return Utterance(dialect=dialect or standard, standard=standard or dialect,
                     start=start, end=end)


def parse_txt_label(content: str) -> list["Utterance"]:
    """.txt 형식 라벨 파싱(일부 지역은 JSON 대신 .txt 제공).

    각 줄이 'speaker_id\\t전사' 형태이고, 전사에는 (방언)/(표준) 이중전사가 들어있다.
    clean_text가 방언/표준을 각각 뽑아낸다(타임스탬프는 없어 STT엔 못 씀).
    """
    out: list[Utterance] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        text = parts[1] if len(parts) == 2 else parts[0]
        dialect = clean_text(text, mode="dialect")
        standard = clean_text(text, mode="standard")
        if not dialect and not standard:
            continue
        out.append(Utterance(dialect=dialect or standard,
                             standard=standard or dialect, start=None, end=None))
    return out


def get_session_audio_name(obj: dict) -> str | None:
    """메타데이터에서 세션 오디오 파일명을 찾는다(없으면 None → JSON 파일명 기준 매칭)."""
    meta = _first(obj, FIELDS["metadata_keys"])
    if isinstance(meta, dict):
        return _first(meta, FIELDS["audio_path_keys"])
    return _first(obj, FIELDS["audio_path_keys"])

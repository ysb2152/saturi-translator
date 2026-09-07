// 하이브리드 후처리 안전망 — 변환기(KoBART)가 놓친 잔여 사투리 표현을 표준어로 보정.
//
// 설계 원칙:
//  (1) 모델을 대체하지 않는다. 변환 '출력'에 남은 명백·무맥락 표현만 최소 보정한다.
//  (2) 문맥 의존/중의적 치환은 넣지 않는다 — 그건 모델의 몫이다.
//      · 니→너/네, 아이가→않니/않냐, 내→나  (한 어절이 여러 표준형)
//      · 따른(→다른)·그래(→그렇게)·그러고(→그리고)·있다(→있지)  (표준어에도 존재하는 형태 → 오탐 위험)
//  (3) 어절(공백) 단위 정확 일치를 기본으로 오탐을 억제한다.
//
// 근거: data/analysis.md 고빈도 치환 Top30 중 '무맥락·비중의적'인 항목 + data/ 전수검색에서
//       경상 존대 종결어미 '아임다/~임더'가 학습쌍에 0건으로 확인됨(커버리지 공백).

// 어절 정확 일치 치환(문장부호는 분리 후 재부착). 값은 모두 비중의적·무맥락.
const EXACT_EOJEOL = {
  쫌: '조금',
  걍: '그냥',
  그니까: '그러니까',
  그서: '그래서',
  인자: '이제',
  인제: '이제',
  이케: '이렇게',
  이르케: '이렇게',
  가주고: '가지고',
  마이: '많이',
  // 데이터 부재가 확인된 경상 존대 종결어미(아닙니다). '~임다/임더'(입니다)는 접미사라 아래 SUFFIX로 처리.
  아임다: '아닙니다',
  아임더: '아닙니다',
};

// 경상 존대 종결어미 '~임다/~임더'(= 표준 '~입니다')를 어절 끝에서 보정.
// 학습쌍에 사실상 0건(커버리지 공백)이라 모델이 못 배운 케이스. 예: 학생임다 → 학생입니다.
// 오탐 방지: (1) 어절 끝이 정확히 '임다'|'임더'  (2) 앞에 어간 최소 1글자  (3) 외래어 블록리스트.
const COPULA_SUFFIX = /(.)(임다|임더)$/u;
const COPULA_BLOCK = new Set(['게임다', '타임다', '마임다', '모임다', '게임더', '타임더']);

// 어절 끝의 문장부호를 분리: "쫌." → ["쫌", "."]
function splitTrailingPunct(token) {
  const m = token.match(/^(.*?)([.,!?…~]*)$/u);
  return m ? [m[1], m[2]] : [token, ''];
}

/**
 * 변환 결과(표준어 문장)에 규칙 후처리 안전망을 적용한다.
 * 공백을 보존하며 어절 단위로만 치환하므로, 매칭이 없으면 입력을 그대로 돌려준다.
 * @param {string} text 변환기 출력
 * @returns {string} 보정된 표준어
 */
export function applyDialectPostfix(text) {
  if (!text) return text;
  return text
    .split(/(\s+)/) // 공백을 캡처해 그대로 유지
    .map((token) => {
      if (token === '' || /^\s+$/.test(token)) return token;
      const [core, punct] = splitTrailingPunct(token);
      if (Object.prototype.hasOwnProperty.call(EXACT_EOJEOL, core)) {
        return EXACT_EOJEOL[core] + punct;
      }
      if (!COPULA_BLOCK.has(core) && COPULA_SUFFIX.test(core)) {
        return core.replace(COPULA_SUFFIX, '$1입니다') + punct;
      }
      return token;
    })
    .join('');
}

export const _EXACT_EOJEOL = EXACT_EOJEOL; // 테스트용

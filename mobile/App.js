import { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Animated,
  Easing,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import {
  setAudioModeAsync,
  requestRecordingPermissionsAsync,
} from 'expo-audio';
import { useAudioRecorder as usePcmRecorder } from '@siteed/expo-audio-studio';
import { documentDirectory } from 'expo-file-system/legacy';
import * as Haptics from 'expo-haptics';

import { loadPipeline, runPipeline } from './src/ondevicePipeline';

// 온디바이스 모델 경로 = 앱 내부 files 디렉터리(네이티브 fopen 항상 가능; scoped storage 무관).
// 테스트: adb push→외부→run-as로 내부 복사. 배포: 첫 실행 다운로드→documentDirectory.
const MODELS = {
  sttModel: `${documentDirectory}ggml-model-f16.bin`,
  encoder: `${documentDirectory}encoder.pte`,
  decoder: `${documentDirectory}decoder.pte`,
  tokenizer: `${documentDirectory}tokenizer.json`,
};

// ── 한지 × 클린 하이브리드 팔레트 ──
const HANJI = '#F5EDDD';       // 한지빛 배경
const SURFACE = '#FFFDF8';     // 따뜻한 화이트 카드
const INK = '#2E251B';         // 먹색(제목)
const SUB = '#8A7C68';         // 흐린 갈색(보조 텍스트)
const TEAL = '#2F6E6A';        // 청록(주 강조)
const TEAL_BRIGHT = '#3FA095'; // 밝은 청록(마이크 버튼)
const TEAL_INK = '#215249';    // 짙은 청록(텍스트)
const TEAL_SOFT = '#E4EDE9';   // 청록 연한 배경
const TEAL_LINE = '#CFE0D8';
const CLAY = '#C1502E';        // 주칠(녹음 상태)
const CLAY_SOFT = '#F3DED3';
const LINE = '#EBE0CC';        // 크림 위 헤어라인

const haptic = (style) => { try { Haptics.impactAsync(style); } catch (_) {} };

export default function App() {
  const pcm = usePcmRecorder();

  const [permissionGranted, setPermissionGranted] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('마이크 권한을 확인하고 있어요');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [modelsReady, setModelsReady] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);

  const recording = pcm.isRecording;

  // 애니메이션 값
  const pulse = useRef(new Animated.Value(0)).current;   // 녹음 링
  const appear = useRef(new Animated.Value(0)).current;  // 결과 등장
  const shimmer = useRef(new Animated.Value(0)).current; // 변환 중 스켈레톤
  const prog = useRef(new Animated.Value(0)).current;    // 준비 진행바

  // 녹음 중: 숨쉬는 링
  useEffect(() => {
    if (!recording) return undefined;
    const loop = Animated.loop(
      Animated.timing(pulse, { toValue: 1, duration: 1600, easing: Easing.out(Easing.ease), useNativeDriver: true })
    );
    loop.start();
    return () => { loop.stop(); pulse.setValue(0); };
  }, [recording, pulse]);

  // 변환 중: shimmer
  useEffect(() => {
    if (!busy) return undefined;
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(shimmer, { toValue: 1, duration: 700, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
        Animated.timing(shimmer, { toValue: 0, duration: 700, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
      ])
    );
    loop.start();
    return () => { loop.stop(); shimmer.setValue(0); };
  }, [busy, shimmer]);

  // 준비 중: 진행바 왕복
  useEffect(() => {
    if (!loadingModels) return undefined;
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(prog, { toValue: 1, duration: 900, easing: Easing.inOut(Easing.ease), useNativeDriver: false }),
        Animated.timing(prog, { toValue: 0, duration: 900, easing: Easing.inOut(Easing.ease), useNativeDriver: false }),
      ])
    );
    loop.start();
    return () => { loop.stop(); };
  }, [loadingModels, prog]);

  // 결과 등장
  useEffect(() => {
    if (!result) return;
    appear.setValue(0);
    Animated.timing(appear, { toValue: 1, duration: 460, easing: Easing.out(Easing.cubic), useNativeDriver: true }).start();
  }, [result, appear]);

  // 권한 + 모델 로드
  useEffect(() => {
    (async () => {
      const { granted } = await requestRecordingPermissionsAsync();
      setPermissionGranted(granted);
      if (!granted) {
        setStatus('마이크 권한이 필요해요. 설정에서 허용해 주세요.');
        return;
      }
      await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
      setStatus('온디바이스 모델을 준비하고 있어요');
      setLoadingModels(true);
      try {
        await loadPipeline(MODELS);
        setModelsReady(true);
        setStatus('버튼을 누르고 사투리로 말해보세요');
      } catch (e) {
        setError(`온디바이스 모델 로드 실패: ${String((e && e.message) || e)}`);
        setStatus('모델 파일이 준비되지 않았어요');
      } finally {
        setLoadingModels(false);
      }
    })();
  }, []);

  const startRecording = async () => {
    setError(null);
    setResult(null);
    haptic(Haptics.ImpactFeedbackStyle.Light);
    try {
      // 16kHz mono 16-bit PCM WAV (whisper.rn 입력 요건)
      await pcm.startRecording({ sampleRate: 16000, channels: 1, encoding: 'pcm_16bit' });
      setStatus('음성 녹음 중이에요');
    } catch (e) {
      setError(`녹음을 시작하지 못했어요: ${e.message}`);
    }
  };

  const stopAndSend = async () => {
    haptic(Haptics.ImpactFeedbackStyle.Medium);
    try {
      const rec = await pcm.stopRecording();
      const uri = rec && rec.fileUri;
      if (!uri) { setError('녹음 파일을 찾지 못했어요.'); setStatus('버튼을 누르고 사투리로 말해보세요'); return; }
      if (!modelsReady) { setError('온디바이스 모델이 아직 준비되지 않았어요.'); return; }
      setBusy(true);
      setStatus('기기에서 표준어로 옮기는 중이에요');
      const { dialect, standard, ms } = await runPipeline(uri);
      setResult({ dialect_text: dialect, standard_text: standard, duration: ms / 1000 });
      setStatus('다시 눌러 말해보세요');
    } catch (e) {
      setError(`변환에 실패했어요: ${e.message}`);
      setStatus('문제가 생겼어요');
    } finally {
      setBusy(false);
    }
  };

  const onPressMic = () => {
    if (busy || !permissionGranted || loadingModels) return;
    if (recording) stopAndSend();
    else startRecording();
  };

  const disabled = !permissionGranted || busy || loadingModels;
  const ringStyle = {
    transform: [{ scale: pulse.interpolate({ inputRange: [0, 1], outputRange: [1, 1.9] }) }],
    opacity: pulse.interpolate({ inputRange: [0, 1], outputRange: [0.45, 0] }),
  };
  const appearStyle = {
    opacity: appear,
    transform: [{ translateY: appear.interpolate({ inputRange: [0, 1], outputRange: [12, 0] }) }],
  };
  const shimmerStyle = { opacity: shimmer.interpolate({ inputRange: [0, 1], outputRange: [0.55, 1] }) };
  const progStyle = { width: prog.interpolate({ inputRange: [0, 0.5, 1], outputRange: ['16%', '82%', '16%'] }) };

  return (
    <View style={styles.root}>
      <StatusBar style="dark" />

      {/* 한지 정서: 은은한 배경 결 */}
      <View pointerEvents="none" style={styles.warmTop} />
      <View pointerEvents="none" style={styles.warmBlob} />
      <View pointerEvents="none" style={styles.tealBlob} />

      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
        {/* ── 정체성(한지) ── */}
        <View style={styles.badge}>
          <View style={styles.badgeDot} />
          <Text style={styles.badgeText}>충청·강원·전라·경상</Text>
        </View>
        <Text style={styles.title}>사투리 번역</Text>
        <Text style={styles.subtitle}>사투리로 말하면{'\n'}표준어로 바꿔드려요</Text>

        {/* ── 중간(성장): 결과 / 스켈레톤 / 빈 공간 ── */}
        <View style={styles.middle}>
          {result ? (
            <Animated.View style={[styles.results, appearStyle]}>
              <View style={styles.cardIn}>
                <Text style={styles.cardLabel}>인식된 사투리</Text>
                <Text style={styles.cardTextIn}>{result.dialect_text || '음성을 알아듣지 못했어요'}</Text>
              </View>
              <View style={styles.arrow}><View style={styles.arrowDown} /></View>
              <View style={styles.cardOut}>
                <Text style={styles.cardLabelOut}>표준어</Text>
                <Text style={styles.cardTextOut}>{result.standard_text || '변환 결과가 없어요'}</Text>
              </View>
              {typeof result.duration === 'number' && (
                <Text style={styles.meta}>{result.duration.toFixed(1)}초</Text>
              )}
            </Animated.View>
          ) : busy ? (
            <View style={styles.results}>
              <Animated.View style={[styles.skel, shimmerStyle]} />
              <Animated.View style={[styles.skelOut, shimmerStyle]} />
            </View>
          ) : null}
        </View>

        {/* ── 마이크 ── */}
        <View style={styles.micWrap}>
          {recording && <Animated.View style={[styles.ring, ringStyle]} />}
          <Pressable
            onPress={onPressMic}
            disabled={disabled}
            style={({ pressed }) => [
              styles.micButton,
              recording && styles.micButtonRecording,
              disabled && !recording && styles.micButtonDisabled,
              pressed && styles.micButtonPressed,
            ]}
          >
            {busy ? (
              <ActivityIndicator color="#fff" size="large" />
            ) : recording ? (
              <View style={styles.stopSquare} />
            ) : (
              <View style={styles.micGlyph}>
                <View style={styles.micBody} />
                <View style={styles.micStem} />
                <View style={styles.micBase} />
              </View>
            )}
          </Pressable>
        </View>

        {/* ── 정중앙 문구(마이크 ↔ 출처 사이 동일 간격) ── */}
        <View style={styles.between}>
          <Text style={[styles.status, recording && styles.statusRec]}>{status}</Text>
          {loadingModels && (
            <View style={styles.progressTrack}><Animated.View style={[styles.progressBar, progStyle]} /></View>
          )}
          {error && (
            <View style={styles.errorBox}><Text style={styles.errorText}>{error}</Text></View>
          )}
        </View>

        {/* ── 출처 + 프라이버시 ── */}
        <View style={styles.foot}>
          <Text style={styles.attribution}>
            데이터 출처: AI 허브(aihub.or.kr) 한국어 방언 음성 데이터로 학습한 모델을 사용합니다.
          </Text>
          <View style={styles.privRow}>
            <View style={styles.lock}>
              <View style={styles.lockShackle} />
              <View style={styles.lockBody} />
            </View>
            <Text style={styles.privText}>녹음한 음성은 이 기기에서만 처리돼요</Text>
          </View>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: HANJI },

  // 배경 결
  warmTop: {
    position: 'absolute', top: 0, left: 0, right: 0, height: 320,
    backgroundColor: '#FCF5E7', opacity: 0.6,
  },
  warmBlob: {
    position: 'absolute', top: -96, right: -72,
    width: 300, height: 300, borderRadius: 999,
    backgroundColor: CLAY, opacity: 0.07,
  },
  tealBlob: {
    position: 'absolute', bottom: -130, left: -96,
    width: 320, height: 320, borderRadius: 999,
    backgroundColor: TEAL, opacity: 0.06,
  },

  container: {
    flexGrow: 1,
    paddingHorizontal: 26,
    paddingTop: 72,
    paddingBottom: 34,
    alignItems: 'center',
  },

  // 정체성
  badge: {
    flexDirection: 'row', alignItems: 'center', gap: 7,
    backgroundColor: '#FFFFFFB8',
    borderRadius: 999, paddingHorizontal: 13, paddingVertical: 7,
    marginBottom: 16, borderWidth: 1, borderColor: LINE,
  },
  badgeDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: CLAY },
  badgeText: { fontSize: 13, fontWeight: '600', color: TEAL_INK, letterSpacing: 0.2 },

  title: {
    fontFamily: 'serif',           // Android: Noto Serif CJK → 한글 세리프(정서)
    fontSize: 40, fontWeight: '700', color: INK, letterSpacing: -0.5,
  },
  subtitle: {
    marginTop: 12, fontSize: 15.5, color: SUB, textAlign: 'center', lineHeight: 23,
  },

  // 중간 성장 영역(결과/스켈레톤)
  middle: { flex: 1, width: '100%', alignItems: 'center', justifyContent: 'center' },

  // 녹음 버튼
  micWrap: {
    width: 172, height: 172,
    alignItems: 'center', justifyContent: 'center',
  },
  ring: {
    position: 'absolute', width: 140, height: 140, borderRadius: 70,
    backgroundColor: CLAY,
  },
  micButton: {
    width: 140, height: 140, borderRadius: 70,
    backgroundColor: TEAL_BRIGHT, alignItems: 'center', justifyContent: 'center',
    shadowColor: TEAL_BRIGHT, shadowOpacity: 0.45, shadowRadius: 22,
    shadowOffset: { width: 0, height: 12 }, elevation: 10,
  },
  micButtonRecording: { backgroundColor: CLAY, shadowColor: CLAY },
  micButtonDisabled: { backgroundColor: '#AFC7C2', shadowOpacity: 0.14 },
  micButtonPressed: { transform: [{ scale: 0.96 }] },

  micGlyph: { alignItems: 'center', justifyContent: 'center' },
  micBody: { width: 26, height: 38, borderRadius: 13, backgroundColor: SURFACE },
  micStem: { width: 3, height: 8, backgroundColor: SURFACE, marginTop: 5 },
  micBase: { width: 28, height: 3.5, borderRadius: 2, backgroundColor: SURFACE, marginTop: 2 },
  stopSquare: { width: 32, height: 32, borderRadius: 8, backgroundColor: SURFACE },

  // 정중앙 문구 영역
  between: { flex: 1, width: '100%', alignItems: 'center', justifyContent: 'center', gap: 12 },
  status: { fontSize: 16, color: TEAL_INK, textAlign: 'center', fontWeight: '600', maxWidth: 260 },
  statusRec: { color: CLAY },

  progressTrack: {
    width: 180, height: 6, borderRadius: 99,
    backgroundColor: 'rgba(46,37,27,0.10)', overflow: 'hidden',
  },
  progressBar: { height: '100%', borderRadius: 99, backgroundColor: TEAL_BRIGHT },

  errorBox: {
    width: '100%', backgroundColor: CLAY_SOFT, borderRadius: 14,
    paddingHorizontal: 16, paddingVertical: 12,
  },
  errorText: { fontSize: 14, color: CLAY, textAlign: 'center', fontWeight: '500' },

  // 결과 카드(클린)
  results: { width: '100%', alignItems: 'center' },
  cardIn: {
    width: '100%', backgroundColor: SURFACE, borderRadius: 18, padding: 18,
    borderWidth: 1, borderColor: LINE,
  },
  cardOut: {
    width: '100%', backgroundColor: TEAL_SOFT, borderRadius: 18, padding: 18,
    borderWidth: 1, borderColor: TEAL_LINE,
  },
  arrow: {
    width: 30, height: 30, borderRadius: 15, marginVertical: -7, zIndex: 2,
    backgroundColor: SURFACE, borderWidth: 1, borderColor: LINE,
    alignItems: 'center', justifyContent: 'center',
  },
  arrowDown: {
    width: 8, height: 8, borderRightWidth: 2, borderBottomWidth: 2, borderColor: TEAL,
    transform: [{ rotate: '45deg' }], marginTop: -3,
  },
  cardLabel: { fontSize: 12, fontWeight: '700', color: SUB, marginBottom: 6, letterSpacing: 0.3 },
  cardLabelOut: { fontSize: 12, fontWeight: '700', color: TEAL, marginBottom: 6, letterSpacing: 0.3 },
  cardTextIn: { fontFamily: 'serif', fontSize: 19, color: INK, lineHeight: 27 },
  cardTextOut: { fontFamily: 'serif', fontSize: 21, color: TEAL_INK, lineHeight: 30, fontWeight: '700' },
  meta: { fontSize: 12, color: '#B0A48C', textAlign: 'center', marginTop: 12 },

  // 변환 중 스켈레톤
  skel: {
    width: '100%', height: 60, borderRadius: 18,
    backgroundColor: '#EFE7D5', borderWidth: 1, borderColor: LINE, marginBottom: 12,
  },
  skelOut: {
    width: '100%', height: 60, borderRadius: 18,
    backgroundColor: '#DEEAE4', borderWidth: 1, borderColor: TEAL_LINE,
  },

  // 출처 + 프라이버시
  foot: { alignItems: 'center', marginTop: 6 },
  attribution: {
    fontSize: 11.5, color: SUB, opacity: 0.8,
    textAlign: 'center', lineHeight: 17, maxWidth: 300,
  },
  privRow: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 9 },
  lock: { width: 13, alignItems: 'center', justifyContent: 'flex-end' },
  lockShackle: {
    width: 8, height: 5, borderWidth: 1.4, borderColor: TEAL, borderBottomWidth: 0,
    borderTopLeftRadius: 4, borderTopRightRadius: 4, marginBottom: -1,
  },
  lockBody: { width: 11, height: 8, borderRadius: 2, borderWidth: 1.4, borderColor: TEAL },
  privText: { fontSize: 11.5, color: TEAL_INK, fontWeight: '500' },
});

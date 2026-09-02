import { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Animated,
  Easing,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import {
  useAudioRecorder,
  useAudioRecorderState,
  RecordingPresets,
  setAudioModeAsync,
  requestRecordingPermissionsAsync,
} from 'expo-audio';

import { DEFAULT_SERVER_URL, transcribe, demoTranscribe } from './src/api';

// ── 한지 × 클린 하이브리드 팔레트 ──
const HANJI = '#F5EDDD';      // 한지빛 배경
const SURFACE = '#FFFDF8';    // 따뜻한 화이트 카드
const INK = '#2E251B';        // 먹색(제목)
const SUB = '#8A7C68';        // 흐린 갈색(보조 텍스트)
const TEAL = '#2F6E6A';       // 청록(주 강조)
const TEAL_INK = '#215249';   // 짙은 청록(텍스트)
const TEAL_SOFT = '#E4EDE9';  // 청록 연한 배경
const TEAL_LINE = '#CFE0D8';
const CLAY = '#C1502E';       // 주칠(녹음 상태)
const CLAY_SOFT = '#F3DED3';
const LINE = '#EBE0CC';       // 크림 위 헤어라인

export default function App() {
  const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const recorderState = useAudioRecorderState(recorder);

  const [serverUrl, setServerUrl] = useState(DEFAULT_SERVER_URL);
  const [permissionGranted, setPermissionGranted] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('마이크 권한을 확인하고 있어요');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showSettings, setShowSettings] = useState(false);

  const recording = recorderState.isRecording;

  // 녹음 중 펄스 링
  const pulse = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    if (recording) {
      const loop = Animated.loop(
        Animated.timing(pulse, {
          toValue: 1,
          duration: 1400,
          easing: Easing.out(Easing.ease),
          useNativeDriver: true,
        })
      );
      loop.start();
      return () => {
        loop.stop();
        pulse.setValue(0);
      };
    }
  }, [recording, pulse]);

  useEffect(() => {
    (async () => {
      const { granted } = await requestRecordingPermissionsAsync();
      setPermissionGranted(granted);
      if (granted) {
        await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
        setStatus('버튼을 누르고 사투리로 말해보세요');
      } else {
        setStatus('마이크 권한이 필요해요. 설정에서 허용해 주세요.');
      }
    })();
  }, []);

  const startRecording = async () => {
    setError(null);
    setResult(null);
    try {
      await recorder.prepareToRecordAsync();
      recorder.record();
      setStatus('듣고 있어요… 편하게 말해요');
    } catch (e) {
      setError(`녹음을 시작하지 못했어요: ${e.message}`);
    }
  };

  const stopAndSend = async () => {
    try {
      await recorder.stop();
      const uri = recorder.uri;
      if (!uri) {
        setError('녹음 파일을 찾지 못했어요.');
        return;
      }
      setBusy(true);
      setStatus('표준어로 옮기는 중이에요…');
      const data = await transcribe(serverUrl, uri);
      setResult(data);
      setStatus('다 됐어요. 또 해볼까요?');
    } catch (e) {
      setError(`변환에 실패했어요: ${e.message}`);
      setStatus('문제가 생겼어요. 서버 연결을 확인해 주세요.');
    } finally {
      setBusy(false);
    }
  };

  const onPressMic = () => {
    if (busy) return;
    if (recording) stopAndSend();
    else startRecording();
  };

  // 녹음 없이 서버 샘플로 파이프라인 검증
  const runDemo = async () => {
    if (busy || recording) return;
    setError(null);
    setResult(null);
    setBusy(true);
    setStatus('샘플 사투리로 테스트 중…');
    try {
      const data = await demoTranscribe(serverUrl);
      setResult(data);
      setStatus('샘플 결과예요. 직접 녹음도 해보세요!');
    } catch (e) {
      setError(`샘플 테스트 실패: ${e.message}`);
      setStatus('문제가 생겼어요. 서버 연결을 확인해 주세요.');
    } finally {
      setBusy(false);
    }
  };

  const ringStyle = {
    transform: [{ scale: pulse.interpolate({ inputRange: [0, 1], outputRange: [1, 2] }) }],
    opacity: pulse.interpolate({ inputRange: [0, 1], outputRange: [0.4, 0] }),
  };

  return (
    <View style={styles.root}>
      <StatusBar style="dark" />

      {/* 한지 정서: 은은한 배경 결 */}
      <View pointerEvents="none" style={styles.warmBlob} />
      <View pointerEvents="none" style={styles.tealBlob} />

      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
        {/* ── 정체성(한지) ── */}
        <View style={styles.badge}>
          <View style={styles.badgeDot} />
          <Text style={styles.badgeText}>전국 사투리</Text>
        </View>
        <Text style={styles.title}>사투리 번역</Text>
        <Text style={styles.subtitle}>사투리로 말하면{'\n'}표준어로 바꿔드려요</Text>

        {/* 서버 설정(접이식) */}
        <Pressable style={styles.gearChip} onPress={() => setShowSettings((v) => !v)}>
          <Text style={styles.gearChipText}>서버 설정 {showSettings ? '▲' : '▼'}</Text>
        </Pressable>
        {showSettings && (
          <View style={styles.field}>
            <Text style={styles.label}>서버 주소</Text>
            <TextInput
              style={styles.input}
              value={serverUrl}
              onChangeText={setServerUrl}
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="url"
              placeholder="http://10.0.2.2:8000"
              placeholderTextColor="#B6A98E"
            />
          </View>
        )}

        {/* ── 작업 화면(클린): 녹음 ── */}
        <View style={styles.micWrap}>
          {recording && <Animated.View style={[styles.ring, ringStyle]} />}
          <Pressable
            onPress={onPressMic}
            disabled={!permissionGranted || busy}
            style={({ pressed }) => [
              styles.micButton,
              recording && styles.micButtonRecording,
              (!permissionGranted || busy) && styles.micButtonDisabled,
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

        <Text style={styles.status}>{status}</Text>

        {/* 개발/테스트용 데모 버튼 — 릴리스(플레이스토어) 빌드에선 __DEV__=false 라 숨겨짐 */}
        {__DEV__ && (
          <Pressable
            style={({ pressed }) => [styles.demoBtn, pressed && { opacity: 0.7 }]}
            onPress={runDemo}
            disabled={busy || recording}
          >
            <Text style={styles.demoBtnText}>🎧 녹음 없이 샘플로 테스트 (개발용)</Text>
          </Pressable>
        )}

        {error && (
          <View style={styles.errorBox}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        {/* ── 작업 화면(클린): 결과 카드 ── */}
        {result && (
          <View style={styles.results}>
            <View style={styles.cardIn}>
              <Text style={styles.cardLabel}>인식된 사투리</Text>
              <Text style={styles.cardTextIn}>
                {result.dialect_text || '음성을 알아듣지 못했어요'}
              </Text>
            </View>
            <View style={styles.cardOut}>
              <Text style={styles.cardLabelOut}>표준어</Text>
              <Text style={styles.cardTextOut}>
                {result.standard_text || '변환 결과가 없어요'}
              </Text>
            </View>
            {typeof result.duration === 'number' && (
              <Text style={styles.meta}>{result.duration}초 · {result.language}</Text>
            )}
          </View>
        )}

        <Text style={styles.attribution}>
          데이터 출처: AI 허브(aihub.or.kr) 한국어 방언 음성 데이터로 학습한 모델을 사용합니다.
        </Text>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: HANJI },

  attribution: {
    marginTop: 32, fontSize: 11.5, color: SUB, opacity: 0.75,
    textAlign: 'center', lineHeight: 17,
  },

  warmBlob: {
    position: 'absolute', top: -110, right: -80,
    width: 280, height: 280, borderRadius: 999,
    backgroundColor: CLAY, opacity: 0.06,
  },
  tealBlob: {
    position: 'absolute', bottom: -120, left: -90,
    width: 300, height: 300, borderRadius: 999,
    backgroundColor: TEAL, opacity: 0.05,
  },

  container: {
    flexGrow: 1,
    paddingHorizontal: 26,
    paddingTop: 84,
    paddingBottom: 56,
    alignItems: 'center',
  },

  // 정체성
  badge: {
    flexDirection: 'row', alignItems: 'center', gap: 7,
    backgroundColor: '#FFFFFFB0',
    borderRadius: 999, paddingHorizontal: 13, paddingVertical: 7,
    marginBottom: 18, borderWidth: 1, borderColor: LINE,
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

  gearChip: {
    marginTop: 22,
    backgroundColor: '#FFFFFF80',
    borderRadius: 999, paddingHorizontal: 14, paddingVertical: 8,
    borderWidth: 1, borderColor: LINE,
  },
  gearChipText: { fontSize: 13, color: SUB, fontWeight: '500' },

  field: { width: '100%', marginTop: 12 },
  label: { fontSize: 12, color: SUB, marginBottom: 6, marginLeft: 4 },
  input: {
    backgroundColor: SURFACE, borderRadius: 14,
    paddingHorizontal: 16, paddingVertical: 13,
    color: INK, fontSize: 15, borderWidth: 1.5, borderColor: LINE,
  },

  // 녹음 버튼
  micWrap: {
    marginTop: 46, width: 200, height: 200,
    alignItems: 'center', justifyContent: 'center',
  },
  ring: {
    position: 'absolute', width: 148, height: 148, borderRadius: 74,
    backgroundColor: CLAY,
  },
  micButton: {
    width: 148, height: 148, borderRadius: 74,
    backgroundColor: TEAL, alignItems: 'center', justifyContent: 'center',
    shadowColor: TEAL, shadowOpacity: 0.4, shadowRadius: 22,
    shadowOffset: { width: 0, height: 12 }, elevation: 10,
  },
  micButtonRecording: { backgroundColor: CLAY, shadowColor: CLAY },
  micButtonDisabled: { backgroundColor: '#C9BCA6', shadowOpacity: 0.12 },
  micButtonPressed: { transform: [{ scale: 0.96 }] },

  // View로 그린 마이크 글리프
  micGlyph: { alignItems: 'center', justifyContent: 'center' },
  micBody: { width: 26, height: 38, borderRadius: 13, backgroundColor: SURFACE },
  micStem: { width: 3, height: 8, backgroundColor: SURFACE, marginTop: 5 },
  micBase: { width: 28, height: 3.5, borderRadius: 2, backgroundColor: SURFACE, marginTop: 2 },
  stopSquare: { width: 34, height: 34, borderRadius: 8, backgroundColor: SURFACE },

  status: {
    marginTop: 30, fontSize: 16, color: INK, textAlign: 'center', fontWeight: '500',
  },

  demoBtn: {
    marginTop: 16, alignSelf: 'center',
    backgroundColor: '#FFFFFFCC', borderRadius: 999,
    paddingHorizontal: 18, paddingVertical: 10,
    borderWidth: 1, borderColor: TEAL_LINE,
  },
  demoBtnText: { fontSize: 14, color: TEAL_INK, fontWeight: '600' },

  errorBox: {
    marginTop: 16, width: '100%',
    backgroundColor: CLAY_SOFT, borderRadius: 14,
    paddingHorizontal: 16, paddingVertical: 12,
  },
  errorText: { fontSize: 14, color: CLAY, textAlign: 'center', fontWeight: '500' },

  // 결과 카드(클린)
  results: { width: '100%', marginTop: 36, gap: 12 },
  cardIn: {
    backgroundColor: SURFACE, borderRadius: 18, padding: 20,
    borderWidth: 1, borderColor: LINE,
  },
  cardOut: {
    backgroundColor: TEAL_SOFT, borderRadius: 18, padding: 20,
    borderWidth: 1, borderColor: TEAL_LINE,
  },
  cardLabel: {
    fontSize: 12, fontWeight: '700', color: SUB,
    marginBottom: 8, letterSpacing: 0.3,
  },
  cardLabelOut: {
    fontSize: 12, fontWeight: '700', color: TEAL,
    marginBottom: 8, letterSpacing: 0.3,
  },
  cardTextIn: { fontSize: 19, color: INK, lineHeight: 27 },
  cardTextOut: { fontSize: 21, color: TEAL_INK, lineHeight: 30, fontWeight: '700' },
  meta: { fontSize: 12, color: '#B0A48C', textAlign: 'right', marginTop: 4 },
});

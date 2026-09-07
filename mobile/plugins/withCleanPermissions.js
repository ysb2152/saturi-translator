// CNG(android/ 자동생성)에서 매니페스트를 정리하는 config plugin.
// 라이브러리가 끌어온, 이 앱이 쓰지 않는 민감 권한을 제거해 Play 심사 마찰을 줄인다.
//
// 제거 대상:
//  - SYSTEM_ALERT_WINDOW : "다른 앱 위에 그리기" 민감권한. 미사용 → 제거(실기기 녹음 정상 검증됨, 2026-09-04).
//  - FOREGROUND_SERVICE / FOREGROUND_SERVICE_MEDIA_PLAYBACK : expo-audio(재생용)에서 유입.
//    이 앱은 오디오 재생을 하지 않으므로 미사용 → 제거(Android 14 전경서비스 유형 선언·Play 심사 마찰 제거).
//  - net.siteed.audiostudio.AudioRecordingService : expo-audio-studio의 '배경 녹음'용 전경서비스.
//    이 앱은 전경(화면 켠 상태) 녹음만 하고 배경 녹음(enableBackgroundAudio) 미사용 → 서비스 제거.
//
// ⚠️ 검증 필요: 위 FGS/서비스 제거 후에도 녹음(핵심 기능)이 정상인지 깨끗한 기기에서 확인할 것.
//    (2026-09-04 무음은 기기 오디오 HAL 일시정지가 원인, 권한과 무관함이 재부팅으로 확인됨.)
//    만약 녹음이 깨지면 FOREGROUND_SERVICE와 AudioRecordingService 제거만 되돌리고
//    MEDIA_PLAYBACK 제거는 유지한다(그건 재생 전용이라 확실히 안전).

const { withAndroidManifest } = require('@expo/config-plugins');

const RM_PERMS = [
  'android.permission.SYSTEM_ALERT_WINDOW',
  'android.permission.FOREGROUND_SERVICE',
  'android.permission.FOREGROUND_SERVICE_MEDIA_PLAYBACK',
];
const RM_SERVICES = [
  'net.siteed.audiostudio.AudioRecordingService',
];

module.exports = function withCleanPermissions(config) {
  return withAndroidManifest(config, (cfg) => {
    const manifest = cfg.modResults.manifest;
    manifest.$ = manifest.$ || {};
    manifest.$['xmlns:tools'] =
      manifest.$['xmlns:tools'] || 'http://schemas.android.com/tools';

    // 권한: 기존 선언 제거 + 병합 시 재유입 차단(tools:node="remove")
    let perms = manifest['uses-permission'] || [];
    perms = perms.filter((p) => !RM_PERMS.includes(p.$ && p.$['android:name']));
    for (const name of RM_PERMS) {
      perms.push({ $: { 'android:name': name, 'tools:node': 'remove' } });
    }
    manifest['uses-permission'] = perms;

    // 미사용 전경서비스 제거
    const app = manifest.application && manifest.application[0];
    if (app) {
      let services = app.service || [];
      services = services.filter(
        (s) => !RM_SERVICES.includes(s.$ && s.$['android:name'])
      );
      for (const name of RM_SERVICES) {
        services.push({ $: { 'android:name': name, 'tools:node': 'remove' } });
      }
      app.service = services;
    }
    return cfg;
  });
};

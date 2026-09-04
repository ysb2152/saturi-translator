// CNG(android/ 자동생성)에서 매니페스트를 정리하는 config plugin.
// 라이브러리가 끌어온, 이 앱이 쓰지 않는 민감 권한을 제거해 Play 심사 마찰을 줄인다.
//
// 제거 대상:
//  - SYSTEM_ALERT_WINDOW : "다른 앱 위에 그리기" 민감권한. 미사용 → 제거(실기기 녹음 정상 검증됨, 2026-09-04).
//
// FOREGROUND_SERVICE(_MEDIA_PLAYBACK) 및 AudioControlsService(expo-audio 재생 컨트롤)는
// 앱이 실제로 쓰지 않아 제거 후보였으나, 녹음(핵심 기능)에 영향이 없는지 깨끗한 기기에서
// 재검증 전까지는 보수적으로 유지한다. Play 제출 시엔 (a) 이들을 제거+재검증하거나
// (b) Play Console 전경서비스 선언으로 처리. (오늘 무음은 기기 오디오 HAL 일시정지가 원인이었고
//  권한 제거와 무관함이 재부팅으로 확인됨.)

const { withAndroidManifest } = require('@expo/config-plugins');

const RM_PERMS = [
  'android.permission.SYSTEM_ALERT_WINDOW',
];
const RM_SERVICES = [];

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

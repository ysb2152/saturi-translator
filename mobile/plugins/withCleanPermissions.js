// CNG(android/ 자동생성)에서 매니페스트를 정리하는 config plugin.
// 라이브러리가 끌어온, 이 앱이 쓰지 않는 민감 권한을 제거해 Play 심사 마찰을 줄인다.
//
// 제거 대상:
//  - SYSTEM_ALERT_WINDOW : "다른 앱 위에 그리기" 민감권한. 미사용 → 제거(실기기 녹음 정상 검증됨, 2026-09-04).
//
// FGS 권한(FOREGROUND_SERVICE, FOREGROUND_SERVICE_MEDIA_PLAYBACK)과 AudioRecordingService는
// 제거를 검토했으나, code 1이 이 권한들을 가진 채로 이미 Play 심사를 통과했고(= 블로커 아님),
// 제거 시 녹음(핵심기능) 영향 재검증이 필요해 보수적으로 유지한다. 여유 기기에서 나란히 설치
// 검증이 가능해질 때 다시 제거를 시도한다(그때는 MEDIA_PLAYBACK부터 — 재생 전용이라 안전).

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

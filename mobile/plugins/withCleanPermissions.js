// android/를 자동생성(CNG)할 때 매니페스트를 손보는 config plugin.
// 라이브러리가 딸려 넣는데 정작 앱은 안 쓰는 민감 권한을 빼서 Play 심사 마찰을 줄인다.
//
// SYSTEM_ALERT_WINDOW("다른 앱 위에 그리기")는 안 쓰는데 붙어 있어서 제거한다.
// (2026-09-04에 실기기에서 녹음 정상 동작까지 확인함.)
//
// FGS 권한(FOREGROUND_SERVICE, FOREGROUND_SERVICE_MEDIA_PLAYBACK)이랑 AudioRecordingService도
// 뺄까 했는데, code 1이 이걸 가진 채로 이미 심사를 통과했고(블로커 아님) 빼면 녹음 영향 재검증이
// 필요해서 일단 그대로 둔다. 여분 기기로 나란히 깔아 비교할 수 있게 되면 그때 다시 뺀다
// (재생 전용이라 안전한 MEDIA_PLAYBACK부터).

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

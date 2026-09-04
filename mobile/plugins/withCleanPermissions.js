// CNG(android/ 자동생성)에서 매니페스트를 정리하는 config plugin.
// 라이브러리가 끌어온, 이 앱이 쓰지 않는 항목을 제거해 Play 심사 마찰을 없앤다.
//
// 제거 대상:
//  - SYSTEM_ALERT_WINDOW               : "다른 앱 위에 그리기" 민감권한. 미사용.
//  - FOREGROUND_SERVICE(_MEDIA_PLAYBACK): expo-audio 재생 컨트롤(AudioControlsService)용.
//    이 앱은 오디오 재생/미디어세션을 쓰지 않아 서비스가 시작되지 않음 → 권한·서비스 모두 제거.
//
// 검증: 다음 실기기 테스트에서 녹음이 정상인지 확인(녹음은 @siteed/expo-audio-studio가
// 전경서비스 없이 수행). 만에 하나 문제가 있으면 RM_PERMS/RM_SERVICES에서 해당 항목을 빼면 됨.

const { withAndroidManifest } = require('@expo/config-plugins');

const RM_PERMS = [
  'android.permission.SYSTEM_ALERT_WINDOW',
  'android.permission.FOREGROUND_SERVICE',
  'android.permission.FOREGROUND_SERVICE_MEDIA_PLAYBACK',
];
const RM_SERVICES = ['expo.modules.audio.service.AudioControlsService'];

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

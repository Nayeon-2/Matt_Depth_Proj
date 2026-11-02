# 줌 트래킹 설계 (Matte_Zoom)

작성: 2026-10-18 · 테스트 프로젝트: `D:\body tracking\Unreal\Matte_Zoom`

## 1. 왜 필요한가

실제 카메라가 줌을 하면 지금 파이프라인에서 두 가지가 깨진다.

1. **CG가 따라오지 않는다.** 영상(사람·배경)은 확대되는데 Unreal 카메라 화각은 그대로라
   헬게이트·로고 같은 오브젝트는 같은 크기로 남는다. 배경에서 떠 보인다.
2. **거리 추정이 틀어진다.** `PersonDistanceEstimator`는 `--dist-hfov`로 고정된 초점거리
   `f`로 몸 크기와 발 위치를 거리로 바꾼다. 줌인하면 몸이 커 보이므로 사람이 **다가온
   것으로 계산**되어, 사람 판이 앞으로 튀어나오고 통과 타이밍이 어긋난다.

그래서 줌 값은 **Unreal 카메라**와 **거리 추정기** 두 곳에 함께 들어가야 한다.

## 2. 기본 수식

카메라가 삼각대에 고정된 채 줌만 바뀌면 화면은 광학 중심을 기준으로 **배율 s만큼
확대**된 것과 같다 (시차 변화 없음).

```
f(t)        = f0 · s(t)                       f0: 기준(와이드) 초점거리 [px]
hfov(t)     = 2 · atan( W / (2 · f(t)) )
Unreal FOV  : tan(FOV_ue(t)/2) = tan(FOV_ue0/2) / s(t)
```

- 줌 트래커는 **상대 배율 s**만 내보낸다. 절대 화각은 기준 상태 하나만 보정하면 된다
  (지금의 `--dist-hfov` 값이 그 역할).
- Unreal은 지금 쓰는 기준 FOV(Matt_test는 90°)를 그대로 두고 비율만 적용한다. 기존
  깊이 배치(`장면 깊이 = 150 + 50 × m`)를 다시 맞출 필요가 없다.
- 거리 추정기는 매 프레임 `self.f = f0 · s`로 갱신한다.

## 3. CV 방식 설계

### 3.1 흐름

```
캡처 프레임 ──┬─► RVM ──► alpha ─┐
              │                  ▼ (사람 영역 제외 마스크)
              └─► ZoomTracker(gray 960px, 배경 특징점) ──► s, pan, tilt, 신뢰도
                                   │
                    ┌──────────────┼──────────────────────┐
                    ▼              ▼                      ▼
        PersonDistanceEstimator   숨김 코드 블록 5~7       프리뷰 HUD
            (f = f0·s)            (RVM_OUTPUT, 같은 프레임)
```

캡처한 **같은 프레임**에서 계산하므로 영상과 줌 값이 프레임 단위로 맞는다. 외부 장비
대비 가장 큰 장점이다.

### 3.2 `ZoomTracker` (`module/engine/zoom_tracker.py`, 신규)

| 단계 | 내용 |
|---|---|
| 마스크 | RVM alpha > 0.05를 30px 팽창해 제외. 선택적으로 `--zoom-mask` PNG로 LED 스크린·조명 영역 제외 |
| 특징점 | 960px 그레이에서 격자(8×5)별 Shi-Tomasi로 고르게 300점 |
| 추적 | 피라미드 LK 광류로 프레임 간 추적, 점마다 **키프레임 좌표**를 유지 |
| 모델 | 키프레임→현재 대응점으로 RANSAC 적합. 1단계: `estimateAffinePartial2D` (배율·회전·이동) → s. 2단계: 회전+줌 호모그래피 `H = K1·R·K0⁻¹` 분해 → s, pan, tilt |
| 키프레임 갱신 | 인라이어 < 80 또는 키프레임 대비 배율 1.3배 초과 시 새 키프레임. 누적 배율 = 키프레임 배율 × 현재 배율 |
| 드리프트 보정 | 키프레임을 배율별로 저장(뱅크). 이전에 지나간 배율로 돌아오면 ORB로 저장된 키프레임과 직접 매칭해 누적 오차를 지움. 와이드 끝에서 `Z` 키 = 기준(s=1)으로 리셋 |
| 필터 | log(s)에 One Euro 필터 + 정지 데드밴드(움직임 < 0.05%면 값 고정, 떨림 방지) |
| 신뢰도 | 인라이어 수·비율. 낮으면 직전 값 유지 + 유효 플래그 끔 |
| 비용 | CPU 1~3 ms 예상 (960px, 300점). RVM 처리 시간 안에 충분히 들어감 |

### 3.3 CV가 약한 상황

| 상황 | 영향 | 대응 |
|---|---|---|
| 무늬 없는 배경 (크로마키, 검은 무대, 흰 벽) | 특징점 부족 → 추적 불가 | 배경에 무늬 있는 소품·마커 배치. 안 되면 장비 방식 |
| **움직이는 배경** (LED 스크린 영상, 관객, 무빙라이트) | 잘못된 움직임을 줌으로 오인 | `--zoom-mask`로 영역 제외. RANSAC이 일부 걸러 주지만 화면 절반 이상이면 실패 |
| 빠른 줌의 모션 블러 | 순간 인라이어 감소 | 직전 속도로 짧게 외삽, 블러 끝나면 재매칭 |
| 망원 끝에서 사람이 화면을 가득 채움 | 배경 점이 너무 적음 | 값 유지(홀드). 사람 크기로 역추정하지 않음(거리와 섞이므로) |
| 렌즈 왜곡이 줌에 따라 변함 | 화면 가장자리에서 CG가 몇 px 어긋남 | 2단계에서 줌별 왜곡 보정(렌즈 파일) |
| 초점 브리딩 | 초점만 바꿔도 배율 변화 | 화면상 실제 배율이므로 **오히려 맞게** 따라감 |

결론: **카메라 고정 + 배경에 무늬가 있고 움직이지 않는 현장**이면 CV만으로 충분히
가능하다. LED 스크린·어두운 무대·조명 연출이 많은 공연 현장은 CV 단독으로는 불안하다.

## 4. 추적 장비 대안

| 방식 | 얻는 값 | 비용 | 장점 | 단점 |
|---|---|---|---|---|
| **A. CV (현재 장비)** | s (+pan/tilt) | 0 | 프레임 동기 자동, 장비 불필요 | 배경 조건에 의존 |
| **B. PTZ 카메라 + FreeD** | pan/tilt/zoom/focus 엔코더 값 (UDP) | 카메라 1대 | 배경과 무관, 안정적. Unreal에 LiveLink FreeD 플러그인 내장 | 엔코더 값→초점거리 렌즈 보정 필요, 영상과 지연 차이 맞춰야 함 |
| **C. 방송/시네 렌즈 엔코더 + 트래킹 시스템** (stYpe, Mo-Sys, Ncam, VIVE Mars + FIZTrack, Loled 엔코더 등) | 카메라 위치·회전 + 렌즈 | 높음 | 핸드헬드·지브까지 가능 | 이 프로젝트 규모에 과함 |
| **D. OBS 디지털 줌** | 정확한 배율 | 0 | 추적 불필요 (obs-websocket으로 값 읽기) | 광학 줌이 아님, 화질 손실 |

B 후보 예: Panasonic AW-UE150/UE160, Sony FR7·BRC 계열, Canon CR-N500/N700 등
(**모델·펌웨어·라이선스별 FreeD 지원 여부는 구매 전 확인**).

**권장:**

1. 카메라를 고정하고 줌만 쓰는 연출이면 **A(CV)부터 테스트** — 장비 없이 바로 검증 가능.
2. 실제 현장이 LED·어두운 무대이거나 운영자가 팬+줌을 같이 한다면 **B(PTZ+FreeD)**가
   가장 현실적이다.
3. B를 쓰더라도 FreeD는 **Python에서 받아** 같은 숨김 코드로 보낸다. 영상이 OBS·RVM을
   거치며 늦어지는 만큼 FreeD 값을 버퍼에서 고정 지연(ms)만큼 뒤의 것을 골라 붙이면
   Unreal 쪽은 소스와 상관없이 그대로 쓴다. 나중에 A와 B를 섞을 수도 있다 (엔코더 값을
   기본, CV로 미세 보정).

이를 위해 Python 쪽은 `--zoom-source off|cv|freed|obs|manual`로 소스를 바꿀 수 있게
만들고, 출력 형식(s, pan, tilt, 유효)은 하나로 통일한다.

### 4.1 현장 결정: Fujinon UA107x8.4 (2026-10-18)

현장 무대(대형 LED 월, 무빙라이트, 헤이즈, 댄서 다수)는 CV 단독으로는 불안하다. 고정
구조물(트러스·기둥·무대 모서리)이 적고, 망원으로 갈수록 화면이 LED와 사람으로 채워진다.

렌즈 UA107x8.4(8.4–900 mm, 내장 2배 익스텐더)는 **16bit 엔코더가 내장된 디지털 박스
렌즈로, 렌즈 데이터 출력**이 있다. 따라서 줌링 엔코더를 따로 달 필요가 없고, 변환 박스
하나만 있으면 된다.

```
UA107x8.4 (렌즈 데이터, Fujinon 시리얼)
   └─► 변환 박스 (예: REtracker Fizz II + Fujinon 케이블) ─ FreeD UDP ─► Python
          ├─ 버퍼 + 고정 지연(영상 지연만큼)
          ├─ LUT: 엔코더 raw → 초점거리 → s (익스텐더 2배 포함)
          └─ 숨김 코드 블록 5 → Unreal (변경 없음)
```

- 렌즈 연결 포트(서보 조작부와 공유 여부), 익스텐더 상태 출력 여부는 렌탈사/중계팀에 확인.
- 중계 AR 팀이 이미 같은 렌즈 데이터를 받고 있으면 FreeD 출력을 나눠 받는 것이 가장 간단.
- LUT 보정과 지연 측정은 공연 전에 무늬 있는 장소에서 CV로 한다. 이후 CV는 공연 중 확인용.
- 개발은 FreeD 시뮬레이터(가짜 UDP 송신기)로 장비 없이 먼저 진행한다.

## 5. Unreal로 전달 (숨김 코드 확장)

기존 블록 0~4 뒤에 추가한다 (`person_distance._put16` 형식 그대로, 4×4 px, 알파 0).

| 블록 | x 픽셀 | 내용 |
|---|---|---|
| 5 | 20–23 | 줌 배율: `v = ln(s) / ln(256)` (16bit, 0..1 → s = 1..256), B=255 유효. 블록 1 서명도 함께 씀 |
| 6 | 24–27 | (예약) pan 오프셋 |
| 7 | 28–31 | (예약) tilt 오프셋 |

UA107x8.4는 익스텐더 포함 약 214배까지 가므로 최대 256배로 잡았다. 16bit log 인코딩
왕복 오차는 0.005% 이하다 (테스트로 확인).

### Unreal (Matte_Zoom) — 구현됨

- **`ZoomTracking` 플러그인** (C++, `Plugins/ZoomTracking`): `AZoomCameraActor`(CineCamera).
  매 틱(Spout 수신 뒤, `TG_PostUpdateWork`) `RT_RVM_OUTPUT` 좌상단 24×4 픽셀을 읽어
  서명과 블록 5를 확인하고 `초점거리 = WideFocalLength(8.4) × s`로 설정한다. 코드가 없으면
  마지막 값을 유지한다.
- **`BP_ZoomCamera`**: 위 클래스의 블루프린트. 레벨의 `ZoomCamera`가 이것이다.
  주요 값: `CodeTexture`, `bUseCode`(끄면 `ManualScale`), `WideFocalLength`,
  `bPreviewInEditor`(에디터 뷰포트에서도 동작. 켠 채 저장하면 줌 상태가 레벨에 저장됨).
- 사람·배경 판(`Plate_Person` 10 m, `Plate_BG` 100 m)은 카메라 자식이고 태그 `ZoomPlate`가
  붙어 있다. 카메라가 매 프레임 판 크기를 현재 화각에 맞춰 화면을 꽉 채우게 한다.
- 장면 단위는 **실제 cm**, 카메라는 UA107 와이드(2/3" 16:9, 9.59 mm 센서, 8.4 mm, 59.4°).
  CG를 실제 거리에 두면 줌에 맞게 확대된다.
- 카메라 후처리: 수동 노출 0 EV, 톤 커브 0, 블룸·비네팅·그레인 0, **모션 블러 0**
  (줌 중 판 크기가 바뀌어 모션 블러가 영상을 번지게 함).
- 확인용 `Marker_*` 큐브(30 m, Test 폴더): 줌 테스트용. 연출 작업 때 숨기거나 지운다.
- 측정: 렌더타겟 읽기 **약 4.5~6.7 ms**(GPU 동기화 포함). 30 fps에는 여유가 있지만
  60 fps에서 빠듯하면 비동기 읽기(1프레임 지연)로 바꾸는 옵션을 검토한다.

## 6. 테스트 계획

| 순서 | 내용 | 통과 기준 |
|---|---|---|
| 1. 합성 줌 | 고정 카메라 테스트 영상에 **알고 있는** 줌 곡선(1→3→1배, 느림/빠름)을 디지털로 입혀 트래커 결과와 비교 (`tools/make_synthetic_zoom.py`, `tools/test_zoom_tracker.py`) | 배율 오차 < 0.3% (1920px 가장자리 약 3 px), 1배 복귀 후 드리프트 < 0.2% |
| 2. 실제 렌즈 | 실제 줌 렌즈로 무늬 있는 배경 촬영, 같은 비교 + 왜곡 확인 | 화면 중앙 1/2 영역에서 CG 어긋남 < 3 px |
| 3. Unreal | Matte_Zoom에서 확인용 큐브가 배경에 붙어 있는지, 렌더타겟 읽기 비용 | 육안상 미끄러짐 없음, 1프레임 이상 지연 없음 |
| 4. 거리 | 사람이 가만히 선 채 줌인/아웃 | 추정 거리 변화 < 5% |
| 5. 현장 조건 | LED 영상·조명 연출이 있는 영상으로 실패율 측정 | 여기서 실패하면 B(FreeD) 검토 |

## 7. 구현 단계

1. **(완료)** Matte_Zoom: 빈 레벨 `/Game/Zoom/Maps/Zoom_Stage`, 폰 없는 `GM_Zoom`
   (`Tools/setup_empty_stage.py`).
2. **(완료)** Python 줌 소스 패키지 `module/engine/zoom/` (아래 7.1), FreeD 시뮬레이터, 오프라인 테스트.
3. **(완료)** `rvm_main.py`: `--zoom-source`, 블록 5 인코딩, 거리 추정기 초점거리 갱신, 실행 중 키.
4. **(완료)** Unreal: ZoomTracking 플러그인, `BP_ZoomCamera`, Spout 수신(`BP_SpoutInputs`),
   화면 맞춤 판, 확인용 마커 (`Tools/setup_zoom_stage.py`, 다시 실행해도 됨).
   `tools/zoom_spout_test.py`로 x1~x20 왕복에서 마커가 배경 표시 안에 머무는 것을 확인
   (`docs/img/zoom_test_unreal.png`).
5. 현장 변환 박스 수령 → 렌즈 테이블 보정 + 지연 측정 (아래 7.3).
6. (필요 시) CV 소스 추가, 줌별 렌즈 왜곡.

### 7.1 줌 소스 구조 (교체 가능)

```
module/engine/zoom/
  base.py      ZoomSample(scale, valid, focal_mm, raw, source)
               ZoomSource           ← 모든 소스의 공통 인터페이스: sample(t, frame), command(cmd)
               EncoderZoomSource    ← 엔코더형 공통: 시간 버퍼 + 지연 보정 + 렌즈 테이블 + 익스텐더 + 값 유지
               ManualZoomSource     ← 키보드 테스트용
  freed.py     FreeD D1 파서/생성, FreeDZoomSource(UDP 수신 스레드 → push(raw))
  lens_lut.py  렌즈 테이블 (raw → 초점거리, log 보간)
  __init__.py  make_zoom_source(종류), encode_zoom / decode_zoom (블록 5)
lens/UA107x8.4_placeholder.json   임시 렌즈 테이블 (보정 전)
```

새 장비를 붙일 때: 시리얼 렌즈 포트·OSC처럼 **시각+엔코더 값**을 주는 장비는
`EncoderZoomSource`를 상속해 수신 스레드에서 `push(raw)`만 부르면 된다. 영상에서
직접 배율을 구하는 방식(CV)은 `ZoomSource.sample(t, frame)`을 구현한다. 그다음
`make_zoom_source`와 `--zoom-source` 선택지에 이름 하나를 추가하면 끝이다. 루프, 거리
추정, Unreal은 바꿀 필요가 없다.

### 7.2 실행

```bat
rem FreeD 시뮬레이터 (장비 대신)
python tools\freed_sim.py --profile sweep --max-scale 20
rem 본 실행: FreeD 수신
run_rvm_30fps.cmd --zoom-source freed --zoom-delay-ms 0
rem 장비 없이 키보드 줌
run_rvm_30fps.cmd --zoom-source manual
rem RVM 없이 Unreal만 확인 (합성 배경 + 코드)
python tools\zoom_spout_test.py
rem 오프라인 테스트
python tools\test_zoom_source.py
```

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--zoom-source` | off | `freed` / `manual` / `off` |
| `--zoom-lens` | placeholder | 렌즈 테이블 JSON |
| `--zoom-delay-ms` | 0 | 영상이 렌즈 값보다 늦는 시간 |
| `--zoom-extender` | 1 | 익스텐더 배율 |
| `--freed-port` | 40000 | FreeD UDP 포트 |
| `--freed-camera-id` | -1 | 특정 카메라 ID만 (-1 = 모두) |

실행 중 키 (프리뷰·cmd 창 공통): `[` `]` 지연 ∓5 ms, `X` 익스텐더 전환,
`=` `-` `Z` 줌 인/아웃/와이드 (manual 소스).

`--dist-hfov`는 **렌즈 와이드 끝 화각**이어야 한다 (UA107x8.4 + 2/3": 59.4).
거리 추정은 매 프레임 `f = f_wide × s`로 바뀐다.

### 7.3 영상 파일로 테스트 (렌즈 데이터 없는 영상)

영상 파일에는 줌 값이 없으므로, 먼저 오프라인 CV로 프레임별 배율을 구해 두고 재생할 때
같은 프레임 번호로 꺼내 쓴다 (`--zoom-source replay`). 파일 입력은 실시간(-re)·반복 재생이고,
캡처 스레드가 프레임 번호를 붙이므로 RVM이 프레임을 건너뛰어도 줌 값이 어긋나지 않는다.

```bat
rem 1) 클립 자르기 (예: 하츠하 59.94p 앞 8초)
..\FFmpeginfmpeg -i "..\영상소스\하츠하5994P.mov" -t 8 -map 0:v:0 -an -c:v libx264 -crf 14 -pix_fmt yuv420p test_videos\hatsuha5994_0-8s.mp4
rem 2) 줌 곡선 추정 → test_videos\hatsuha5994_0-8s.zoom.csv (+ .zoom.png 곡선, .zoom_check.jpg 확인용)
python tools\estimate_zoom_offline.py --video test_videos\hatsuha5994_0-8s.mp4
rem 3) 재생 → RVM → Spout → Unreal (Matte_Zoom에서 Play)
run_zoom_file_test.cmd
```

- 추정은 사람(YOLO-seg)을 가린 배경 특징점으로 프레임 간 유사변환을 구해 누적한다. 클립에서
  가장 넓은 프레임이 x1이다. `.zoom_check.jpg`에서 배경이 제자리에 있으면 추정이 맞은 것.
- 하츠하 0–8초: 배경 기준 x1 → x1.56 (앞 4초 줌인, 끝에 약간 줌아웃). 카메라가 앞으로도
  움직이는 샷이라(사람이 배경보다 더 커짐) 순수 줌은 아니다. CG는 배경 깊이에서 맞는다.
- **아직 pan/tilt(화면 이동)는 Unreal에 적용하지 않는다.** CSV에 `dx, dy`는 기록되어 있으므로
  필요하면 블록 6/7로 보내 카메라 회전에 반영할 수 있다.
- 실행 중 `[` `]`는 replay 소스에서 프레임 오프셋 ∓1 (동기 확인용).

### 7.4 사람 판 깊이 + 통과 오브젝트 (Matt_test 방식)

`BP_ZoomCamera`가 블록 0(사람 거리)도 읽어, 태그 `PersonPlate`가 붙은 `Plate_Person`을
`DepthOffset + 거리(m) × DepthPerMeter` cm 깊이로 옮긴다 (기본 100 = 실제 거리). 판 크기는
그 깊이에서 화면을 꽉 채우게 다시 맞추므로 화면상 사람은 그대로이고 앞뒤 관계만 바뀐다.
줌도 같은 프레임에서 함께 적용된다. 거리 신호가 아직 없으면 `DefaultPersonDepth`(10 m).

- 테스트 오브젝트: `Objects/Pillar` (지름 50 cm, 높이 20 m 원기둥, 540 cm, 화면 약간 왼쪽).
  `M_Zoom_Pillar`는 조명 없이 음영만 흉내 낸 unlit 머티리얼.
- 540 cm: 오드아이써클 풀캠 2:32~2:40에서 세 명이 서 있을 때 약 5.8 m, 걸어나온 뒤 약
  4.8~5.3 m로 추정됨 (`--dist-hfov 60` 기준). 걸어나오면 기둥 앞으로 나온다.
  위치는 `Tools/setup_zoom_stage.py`의 `PILLAR` 또는 에디터에서 직접 옮겨 조정.
- 줌 확인용 마커(`Test` 폴더)는 이제 게임에서는 숨김 (`MARKERS_IN_GAME`).
- 누끼 판이 하나라서 모든 사람이 한 깊이(가운데 사람 기준)로 움직인다.
- OBS로 재생하면 줌 값이 없으므로 카메라가 줌하는 구간(오드아이 2:41~)에서 CG는 따라가지
  않고, 거리 추정은 사람이 다가온 것으로 본다. 줌까지 보려면 파일 입력 + replay를 쓴다.

### 7.5 거리 추정: 전신 세로 길이 기준 (`--dist-body full`, 기본)

어깨 폭은 몸을 돌리거나 팔을 움직이면 크게 변해, 조금 걸어나온 정도는 노이즈에 묻힌다.
기본 측정을 **어깨 중심 → 골반 중심 → 무릎 → 발목** 세로 길이(긴 쪽 다리)로 바꿨다.
몸을 돌려도 거의 변하지 않고 픽셀 수도 많다. 다리가 안 보이면 이전 방식(어깨폭/몸통)으로
넘어가고, 두 방식의 비율을 학습해 두어 값이 튀지 않는다. 이전 방식은 `--dist-body upper`.

비교 (`tools/compare_body_methods.py`, 프레임당 노이즈 = 연속 프레임 차이의 중앙값):

| 영상 | full | upper |
|---|---|---|
| 오드아이 2:35–2:45, 세 명이 뒤에 서 있는 구간 | 4.85 m, 노이즈 1.5%/프레임 | 6.72 m, 노이즈 8.6%/프레임 (몸 회전 때 최대 20 m까지 튐) |
| pexels 5320110 (실제로 13 m → 1 m 걸어옴) | 13.8 → 6.9 → 1.7 → 1.2 m, 0.7% | 14.4 → 7.7 → 1.7 → 1.2 m, 1.0–1.7% |

- 오드아이 구간에서 세 명이 실제로 앞으로 나오는 거리는 작다 (full 기준 약 3%). 앞서
  "5.8 m → 5.0 m"로 보였던 변화는 대부분 어깨 폭 노이즈였다. 이 영상으로는 기둥 통과 시점을
  안정적으로 보여주기 어렵고, 실제로 걸어나오는 영상(직접 촬영 등)이 필요하다.
- 추가 개선 예정: 운영자 수동 신호(키 입력으로 앞/뒤 전환), 리허설 바닥 보정.

### 7.6 운영자 신호 (언리얼에서 F / G / A) + 스르르 통과

언리얼 화면을 보면서 Play 중 키를 누른다 (Python은 거리만 보냄, 프리뷰에는 효과 없음).

| 키 | 동작 |
|---|---|
| `F` | 사람 판을 `CueFrontDepth`(400 cm)로: 오브젝트 앞으로 나옴 |
| `G` | `CueBackDepth`(700 cm)로: 오브젝트 뒤로 들어감 |
| `A` | 다시 자동(거리 추정) |

- 모든 전환은 `CueMoveTime`(1.5초) 동안 부드럽게(smoothstep) 이동한다. 이동 중에 다른 키를
  눌러도 현재 위치에서 이어서 움직인다. 시작 상태는 `StartCue`.
- 설정 위치: `ZoomCamera` → Details → **Zoom|Person Cue** (키, 깊이, 시간). 현재 상태는
  Zoom|Status의 `CurrentCue`.
- **스르르 통과:** 깊이를 천천히 옮겨도 앞뒤가 바뀌는 순간은 한 프레임이라 "탁" 튀어나와
  보인다. 그래서 오브젝트 머티리얼(`M_Zoom_Pillar`)이 사람 영역(누끼 알파)에서, 사람이
  오브젝트 뒤 `PassBand`(150 cm) 안으로 다가올수록 점점 디졸브된다. 카메라가 매 틱
  `MPC_Zoom`에 `PersonDepth`, `PassBand`를 넣어 준다.
  다른 오브젝트에도 같은 효과를 주려면 이 머티리얼의 디졸브 부분(PixelDepth, MPC_Zoom,
  화면 좌표로 읽은 RT_RVM_OUTPUT 알파, DitherTemporalAA)을 복사해 쓴다.
- 자동 모드에서 사람이 오브젝트 뒤 1.5 m 안에 서 있으면 그만큼 비쳐 보인다. 싫으면
  `PassBand`를 줄인다 (ZoomCamera Details, 또는 setup 스크립트의 `PASS_BAND`).

### 7.7 현장 보정 (변환 박스를 받은 뒤)

1. 박스의 FreeD 출력 포트와 zoom raw 범위 확인 (`freed_sim` 대신 실제 박스를 켜고 로그 확인).
2. 렌즈 테이블: 와이드 끝·텔레 끝·중간 여러 지점에서 raw와 실제 배율(무늬 있는 배경을
   찍어 크기 비교)을 기록해 `lens/UA107x8.4.json`의 `points`에 넣고 `calibrated: true`.
3. 지연: 빠르게 줌하면서 마커/배경 어긋남이 사라질 때까지 `[` `]`로 맞춘 뒤
   `--zoom-delay-ms`에 고정.
4. 익스텐더 상태가 FreeD에 실려 오는지 확인. 안 실리면 운영 중 `X`로 맞춘다.

### 7.8 SDI 렌즈 메타데이터 (HDC-5500V + HDCU-5500, 2026-10-27 확인)

HDCU-5500 SDI 출력 VANC에 Sony 전용 패킷이 실려 온다 (`tools/sdi_anc_probe.py`로 발견).

| 패킷 | 라인 | 내용 |
|---|---|---|
| DID 0x51 / SDID 0x01, 91 B | 9, 571, 572 (매 필드) | **byte 65–66 = 줌 엔코더** (big-endian, 와이드 0 … 텔레 65535). byte 67–68 = 포커스로 추정. 69–70 = 줌에서 계산된 보조값. 89–90 = 체크섬 |
| DID 0x53 / SDID 0x03, 141 B | 16, 579 | 장비 정보 (HDCU5500 / HDC5500V 시리얼, 커넥터) |

`tools/sdi_anc_probe.py --watch 0x51:0x01 --record-video`로 영상과 함께 기록하고
`tools/anc_video_correlate.py`로 화면 배율과 비교해 확정했다 (byte 65–66 ↔ 화면 배율 r = 0.999,
줌인·줌아웃 따로 잰 표가 몇 % 안에서 일치). 렌즈 UA24x7.8B 테이블: `lens/UA24x7.8B.json`
(측정 곡선, 텔레 끝을 사양 187 mm로 맞춤. `sdi` 항목에 패킷 위치).

**영상과 같은 프레임에 실려 오므로 지연 보정이 필요 없다.** ffmpeg decklink 입력은 이 패킷을
버리므로 새 입력 `--ffmpeg-in sdk`(`module/vedioio/decklink_sdk.py`, DeckLink 드라이버 직접,
8비트 UYVY, 1080i는 bob)가 영상과 패킷을 함께 받고, `--zoom-source sdi`(`module/engine/zoom/sdi.py`)가
처리 중인 프레임 번호로 그 프레임의 값을 찾는다. `[` `]`는 프레임 단위 오프셋.

```bat
python rvm_main.py --ffmpeg-in sdk --force-fps 30000/1001 --zoom-source sdi --zoom-lens lens\UA24x7.8B.json --dist-hfov 63.2 ... (나머지는 run_rvm_30fps.cmd와 같음)
```

- 언리얼 `ZoomCamera` → Wide Focal Length = **7.8** (UA24x7.8B). Python `--dist-hfov` = **63.2**.
- DeckLink 입력은 한 프로그램만 쓸 수 있다. 이 경로를 쓰는 동안 OBS의 DeckLink 소스는 끈다.
- 다른 렌즈: 엔코더 값 위치는 같을 가능성이 높지만 곡선은 렌즈마다 다르다. 같은 방법(기록 → 비교)으로
  새 JSON을 만든다. 익스텐더 상태는 아직 패킷에서 찾지 못함 (X 키).

#### 현장 렌즈 보정 (렌즈마다 한 번, 약 10분)

카메라 고정, 무늬 있고 움직이지 않는 배경, 앞을 지나가는 사람 없이. OBS의 DeckLink 소스는 끈다.

```bat
rem 1) 값이 오는지: 줌을 돌리면 2번째 값이 0..65535 사이에서 움직여야 한다
python tools\sdi_anc_probe.py --watch 0x51:0x01
rem 2) 기록: 와이드 끝 -> 텔레 끝 -> 와이드 끝을 천천히 두 번
python tools\sdi_anc_probe.py --watch 0x51:0x01 --record-video --seconds 120
rem 3) 영상 배율과 비교
python tools\anc_video_correlate.py anc_logs\anc_<시각>.jsonl
rem 4) 렌즈 테이블 (사양의 와이드/텔레 초점거리): lens\<이름>.json, 적용할 옵션을 출력
python tools\make_lens_table.py anc_logs\anc_<시각>.jsonl --name UA107x8.4 --wide-mm 8.4 --tele-mm 900
```

적용: RVM `--zoom-source sdi --zoom-lens lens\<이름>.json --dist-hfov <출력값>`, 언리얼 ZoomCamera →
Wide Focal Length = 와이드 초점거리. 카메라(센서)가 바뀌면 언리얼 Filmback과 `--sensor-width`도.
오늘 데이터로 `lens/UA24x7.8B.json`을 다시 만들면 0.6% 이내로 같다.

## 8. 미정 사항 (현장 정보 필요)

- 실제 카메라·렌즈 모델, 광학 줌인지 (OBS 디지털 줌이면 D로 바로 해결)
- 줌 중에 팬/틸트도 하는지
- 현장 배경 (LED 스크린, 조명 연출 정도)
- 최대 줌 배율

# Mat Depth Project

실시간 DirectShow/DeckLink 입력을 RVM으로 처리하는 프로젝트입니다. 기준 환경은
Anaconda `RVM_4090_2`(Python 3.11, PyTorch 2.5.1 + CUDA 12.1)입니다.

## 처음 설정 (clone 후)

가중치와 대용량 파일은 저장소에 포함되지 않습니다. clone 후 아래 파일을 받아
지정된 위치에 두세요.

```bat
git clone --recurse-submodules <이 저장소 주소>
```

| 파일 | 위치 | 받는 곳 |
|---|---|---|
| `rvm_mobilenetv3.pth` (필수) | 프로젝트 폴더 | [RobustVideoMatting releases](https://github.com/PeterL1n/RobustVideoMatting/releases) |
| `rvm_resnet50.pth` (선택) | 프로젝트 폴더 | 위와 동일 |
| `yolov8n-pose.pt` | 프로젝트 폴더 | 첫 실행 시 ultralytics가 자동 다운로드 |
| `metric_video_depth_anything_vits.pth` (비교 도구용) | `third_party/Video-Depth-Anything/checkpoints/` | [Hugging Face](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small) |

## Unreal 연동

`run_rvm_30fps.cmd`는 Spout로 두 스트림을 보냅니다.

- `RVM_OUTPUT`: 사람 누끼(RGBA). 좌상단 8x4 픽셀(알파 0)에 사람 거리(mm)가 인코딩됨
  (`module/engine/person_distance.py`)
- `RVM_BG`: 배경 (`--spout-bg-mode raw|clean`)

Unreal 프로젝트(Matt_test)의 `MI_RVM_Person` 머티리얼이 거리를 읽어 사람 판을
카메라 방향으로 앞뒤로 이동시킵니다.

## 빠른 실행

프로젝트 폴더에서 다음 파일을 실행합니다.

```bat
run_rvm_30fps.cmd
```

추가 CLI 옵션은 뒤에 붙일 수 있습니다.

```bat
run_rvm_30fps.cmd --no-preview --obs-out 1
```

## 경로 규칙

- RVM 가중치는 프로젝트 폴더의 `rvm_mobilenetv3.pth` 또는
  `rvm_resnet50.pth`를 자동으로 사용합니다.
- FFmpeg는 `RVM_FFMPEG_BIN` 환경 변수, 프로젝트/상위 폴더의
  `FFmpeg\bin\ffmpeg.exe`, 시스템 `PATH` 순서로 자동 검색합니다.
- 따라서 프로젝트를 다른 폴더나 드라이브로 옮겨도 명령의 절대경로를 수정할
  필요가 없습니다.

별도 위치의 FFmpeg나 가중치를 쓰려면 명시적으로 지정할 수 있습니다.

```bat
run_rvm_30fps.cmd --ffmpeg-bin "E:\Tools\ffmpeg.exe" --weights "E:\Models\rvm_mobilenetv3.pth"
```

## 실시간 처리 정책

- 캡처 스레드는 항상 최신 프레임 하나만 보관해 지연 누적을 방지합니다.
- 처리 FPS 기본값은 입력 `--force-fps`와 동일합니다.
- 일시적인 입력 공백은 즉시 종료하지 않고 재시도합니다.
- 로그의 `stale_drop`은 처리 지연 때문에 의도적으로 버린 오래된 입력 프레임
  수입니다. 값이 계속 증가하면 `--long 768` 또는 `--preset speed`를 사용하세요.

장치 확인:

```bat
"..\FFmpeg\bin\ffmpeg.exe" -list_devices true -f dshow -i dummy
```

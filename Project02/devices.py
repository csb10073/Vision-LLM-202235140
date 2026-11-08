"""카메라, 마이크, Whisper, Piper를 연결한다."""

import subprocess
import time

from procedure import Procedure


SECONDS = 15          # 시연과 녹음을 동시에 하는 시간
STABLE_FRAMES = 2     # 색 또는 빈 화면을 확정할 연속 화면 수
STEP_DELAY = 1.5      # 인식 후 다음 단계를 받기까지 대기 시간(초)
WHISPER_PATH = "whisper.cpp/build-cpu/bin/whisper-cli"
WHISPER_MODEL = "whisper.cpp/models/ggml-base.bin"
WHISPER_THREADS = 4
PIPER_PYTHON = ".piper_venv/bin/python"
PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"


def observe(frame, detector, clahe):
    from vision import central_roi, colors_in_crop

    roi, left, top = central_roi(frame)
    if detector is None:
        return colors_in_crop(roi, 0.025, clahe), []

    colors, objects = set(), []
    for (x, y, width, height), class_id, confidence in detector.detect(roi):
        crop = roi[y:y + height, x:x + width]
        found = colors_in_crop(crop, 0.12, clahe)
        colors.update(found)
        objects.append({
            "class_id": class_id,
            "confidence": round(confidence, 3),
            "bbox": [x + left, y + top, x + left + width, y + top + height],
            "colors": sorted(found),
        })
    return colors, objects


def show_frame(cv2, frame, colors, objects, status):
    from vision import central_roi

    roi, x, y = central_roi(frame)
    height, width = roi.shape[:2]
    cv2.rectangle(frame, (x, y), (x + width - 1, y + height - 1), (200, 200, 200), 1)
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        label = "{} {:.2f} {}".format(obj["class_id"], obj["confidence"], "/".join(obj["colors"]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
    cv2.putText(frame, " + ".join(sorted(colors)) or "empty / unknown", (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    position = (12, frame.shape[0] - 15)
    cv2.putText(frame, status, position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, status, position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Procedure check", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        raise KeyboardInterrupt


def capture_trial(cv2, args, detector, clahe, audio_path):
    
    pipeline = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )
    camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    procedure = Procedure(STABLE_FRAMES, STEP_DELAY)
    recorder = None
    frames = 0
    try:
        if not camera.isOpened():
            raise RuntimeError("Jetson CSI 카메라를 열 수 없습니다. 센서 연결과 OpenCV의 GStreamer 지원을 확인하세요.")
        ok, _ = camera.read()
        if not ok:
            raise RuntimeError("카메라에서 첫 화면을 읽을 수 없습니다.")

        print("녹음 시작. 카메라 중앙을 비우고 준비 안내를 기다려 주세요.", flush=True)
        recorder = subprocess.Popen(record_command(args, audio_path))
        start = time.monotonic()
        while time.monotonic() - start < SECONDS:
            if recorder.poll() is not None:
                break
            ok, frame = camera.read()
            seconds = time.monotonic() - start
            if not ok:
                raise RuntimeError("카메라 연결이 끊겨 시연을 종료했습니다.")
            if seconds >= SECONDS:
                break
            colors, objects = observe(frame, detector, clahe)
            frames += 1
            event = procedure.update(colors, seconds, objects)
            procedure.report_event(event, seconds)
            if args.preview:
                show_frame(cv2, frame, colors, objects, procedure.preview_status(seconds))

        if recorder.wait(timeout=5) != 0:
            raise RuntimeError("마이크 녹음 실패. arecord -l에서 장치를 확인하세요.")
        if not frames:
            raise RuntimeError("시연 중 처리한 카메라 화면이 없습니다.")
        print(f"[카메라] 처리한 화면 {frames}장", flush=True)
        return procedure
    finally:
        camera.release()
        if recorder is not None and recorder.poll() is None:
            recorder.terminate()
        if args.preview:
            cv2.destroyAllWindows()


def record_command(args, audio_path):
    # PulseAudio를 잠시 멈춰 장치 점유 오류를 피한다.
    return [
        "pasuspender", "--",
        "arecord", "-q", "-D", args.mic, "-t", "wav", "-f", "S16_LE",
        "-r", "16000", "-c", "1", "-d", str(SECONDS), str(audio_path),
    ]


def transcribe(audio_path):
    prefix = audio_path.with_name("transcript")
    subprocess.run([
        WHISPER_PATH, "-m", WHISPER_MODEL, "-f", str(audio_path),
        "-l", "ko", "-t", str(WHISPER_THREADS), "--no-timestamps", "--no-prints",
        "-otxt", "-of", str(prefix),
    ], check=True, stdout=subprocess.DEVNULL, timeout=120)
    return " ".join(prefix.with_suffix(".txt").read_text(encoding="utf-8").split())


def speak(args, text, output_path):
    
    subprocess.run(
        [
            PIPER_PYTHON,
            "-m",
            "piper",

            "-m",
            PIPER_MODEL,

            "-f",
            str(output_path),

            "--",
            text,
        ],
        check=True,
        timeout=45,
    )

    subprocess.run(
        [
            "aplay",
            "-D", args.speaker,
            str(output_path),
        ],
        check=True,
        timeout=30,
    )

"""파랑 -> 빨강 시연을 확인하고 한국어 음성으로 결과를 알린다."""

import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from devices import PIPER_MODEL, PIPER_PYTHON, WHISPER_MODEL, WHISPER_PATH, capture_trial, speak, transcribe
from procedure import DETAILS, MISMATCH_MESSAGE, evaluate, spoken_order
from report import load_gemma, make_record, record_to_text, save_record, write_report


YOLO_MODEL = "src/models/YOLO/yolo11n_int8.engine"
CLASS_IDS = [39, 41]  # 병, 컵
CONFIDENCE = 0.30
GEMMA_MODEL = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
REPORT_DIR = "reports"


def main():
    args = parse_args()
    import cv2
    from vision import Yolo

    files = [WHISPER_PATH, WHISPER_MODEL, PIPER_PYTHON, PIPER_MODEL, PIPER_MODEL + ".json"]
    if not args.color_only:
        files.append(YOLO_MODEL)
    if not args.no_gemma:
        files.append(GEMMA_MODEL)
    for path in files:
        if not Path(path).is_file():
            raise RuntimeError(f"필요한 파일을 찾을 수 없습니다. {path}")

    detector = None
    if args.color_only:
        print("[진단 모드] 카메라 중앙의 색만 확인합니다.")
    else:
        print("[YOLO] 모델을 불러오는 중입니다.", flush=True)
        detector = Yolo(YOLO_MODEL, CLASS_IDS, CONFIDENCE)
    llm = None
    if not args.no_gemma:
        print("[Gemma] 모델을 불러오는 중입니다.", flush=True)
        llm = load_gemma(GEMMA_MODEL)
    clahe = cv2.createCLAHE(2.0, (8, 8)) if args.clahe else None
    try:
        while input("Enter로 시연 시작, q로 종료 > ").strip().lower() != "q":
            with TemporaryDirectory(prefix="procedure-") as directory:
                audio_path = Path(directory) / "speech.wav"
                procedure = capture_trial(cv2, args, detector, clahe, audio_path)
                play_feedback(args, procedure, audio_path, llm)
    except KeyboardInterrupt:
        # 미리보기 창의 q 또는 Ctrl+C
        print("\n종료합니다.")
    return 0


def play_feedback(args, procedure, audio_path, llm):
    output_path = audio_path.with_name("reply.wav")
    try:
        transcript = transcribe(audio_path)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"[음성 인식 오류] {error}", file=sys.stderr, flush=True)
        transcript = ""
    print(f"[사용자 음성] {transcript or '(인식된 문장 없음)'}", flush=True)

    result, mismatch = evaluate(procedure, transcript)
    verdict = "통과" if result == "pass" else "순서를 다시 확인하시오"
    detail = DETAILS.get(result, MISMATCH_MESSAGE)
    reason = "" if result == "speech_mismatch" else f" ({detail})"
    print(f"[판정] {verdict}{reason}", flush=True)

    # 사용자에게는 고정 문구만 들려준다.
    announce(args, verdict, output_path)
    if mismatch:
        announce(args, MISMATCH_MESSAGE, output_path)

    # 평가자용 기록과 Gemma 보고서는 스피커가 끝난 뒤 만든다.
    record = make_record(procedure, transcript, spoken_order(transcript), result, verdict, detail)
    if llm is not None:
        print("[Gemma] 평가자 보고서를 작성하는 중입니다.", flush=True)
        try:
            record["report"] = write_report(llm, record_to_text(record))
            print(f"[Gemma 보고서]\n{record['report']}", flush=True)
        except Exception as error:
            print(f"[Gemma 오류] {error}", file=sys.stderr, flush=True)
    print(f"[기록 저장] {save_record(record, REPORT_DIR)}", flush=True)


def announce(args, text, output_path):
    try:
        speak(args, text, output_path)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"[스피커 출력 오류] {error}", file=sys.stderr, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="카메라 미리보기")
    parser.add_argument("--clahe", action="store_true", help="색을 확인하기 전에 밝기 대비 보정")
    parser.add_argument("--color-only", action="store_true", help="YOLO 없이 색 순서만 확인하는 진단 모드")
    parser.add_argument("--no-gemma", action="store_true", help="Gemma 보고서 없이 실행")
    parser.add_argument("--mic", default="plughw:3,0", help="ALSA 마이크 장치명")
    parser.add_argument("--speaker", default="default", help="ALSA 스피커 장치명")
    return parser.parse_args()


main()

"""시연 기록을 JSON으로 저장하고, Gemma로 평가자용 보고서를 만든다.

사용자에게는 고정 문구만 들려주고, 이 보고서는 평가자만 본다.

구조
기록(Dictionary) → JSON 저장 → 자연어 변환 → Gemma Context
"""

import json
import time
from pathlib import Path

from procedure import EXPECTED, LABELS


CONTEXT_WINDOW = 2048
MAX_TOKENS = 150

PROMPT = """
Instruction:
작업 절차 검증 기록을 읽고 평가자를 위한 보고서를 작성하시오.
사용자가 절차를 말로 알고 있는지, 실제로 순서대로 수행했는지 나누어 설명하시오.

Constraint:
기록에 없는 내용은 추측하지 마시오.
판정은 기록에 적힌 판정을 그대로 따르시오.

Output Format:
한국어 세 문장 이내.
"""


def load_gemma(model_path):
    from llama_cpp import Llama

    return Llama(
        model_path=str(model_path),
        n_gpu_layers=-1,
        n_ctx=CONTEXT_WINDOW,
        n_batch=32,
        n_ubatch=32,
        verbose=False,
    )


# 시연 한 번의 결과를 딕셔너리로 정리한다.
# 기준 순서, 판정, 카메라가 확인한 단계(색), Whisper 문장, 음성에서 파악한 순서를 담는다.
def make_record(procedure, transcript, spoken, result, verdict, detail):
    return {
        "expected": list(EXPECTED),
        "verdict": verdict,
        "result": result,
        "detail": detail,
        "camera_steps": procedure.events,
        "transcript": transcript,
        "spoken_order": list(spoken) if spoken else None,
    }


# 색 목록을 한국어 순서 문장으로 바꾼다.  ("blue", "red") -> "파란색 → 빨간색"
def order_text(colors):
    return " → ".join(LABELS[color] for color in colors)


# 기록 딕셔너리를 Gemma가 읽기 쉬운 한국어 문장 여러 줄로 바꾼다.
# 이 문장이 Gemma에 Context로 전달된다.
def record_to_text(record):
    lines = []
    lines.append(f"기준 순서: {order_text(record['expected'])}")
    lines.append(f"판정: {record['verdict']} ({record['detail']})")

    if not record["camera_steps"]:
        lines.append("카메라에서 확인된 단계가 없습니다.")
    for index, step in enumerate(record["camera_steps"], start=1):
        color = LABELS[step["color"]]
        lines.append(f"{index}번째로 확인된 단계는 {color}이며, 시작 후 {step['seconds']}초에 확인됐습니다.")

    transcript = record["transcript"] or "(인식된 문장 없음)"
    lines.append(f"사용자 음성: \"{transcript}\"")
    if record["spoken_order"]:
        lines.append(f"음성에서 파악한 순서: {order_text(record['spoken_order'])}")
    else:
        lines.append("음성에서 순서를 확정하지 못했습니다.")

    return "\n".join(lines)


# PROMPT와 기록 문장을 Gemma에 보내고, 평가자용 보고서 문자열을 return
# 출력은 MAX_TOKENS까지만 생성한다.
def write_report(llm, text):
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": PROMPT,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        max_tokens=MAX_TOKENS,
    )
    return response["choices"][0]["message"]["content"].strip()


# 기록(보고서 포함)을 folder 안에 trial_날짜_시각.json 파일로 저장하고 그 경로를 return.
# ensure_ascii=False로 한국어가 깨지지 않게 저장한다.
def save_record(record, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / time.strftime("trial_%Y%m%d_%H%M%S.json")

    with open(path, "w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=4)

    return path

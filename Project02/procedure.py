"""색을 단계로 기록하고 화면과 음성의 순서를 판정한다."""

import re


EXPECTED = ("blue", "red")  # 기준 순서: 파랑 → 빨강
LABELS = {"blue": "파란색", "red": "빨간색", "both": "파란색과 빨간색 동시"}  # 출력용 한국어 이름
MISMATCH_MESSAGE = "음성과 화면의 순서가 다릅니다."
# 판정 코드별 설명 문구. 터미널 출력, 기록 JSON, Gemma 입력에 쓰인다.
DETAILS = {
    "pass": "파란색 다음에 빨간색을 확인했습니다.",
    "not_ready": "처음에 물체가 없는 화면을 확인하지 못했습니다.",
    "no_steps": "파란색과 빨간색이 모두 확인되지 않았습니다.",
    "missing_red": "파란색은 확인했지만 빨간색이 누락됐습니다.",
    "wrong_order": "화면의 순서가 기준과 다릅니다.",
    "both_visible": "두 색이 동시에 보여 먼저 나온 색을 확정할 수 없습니다.",
    "extra_step": "기대하는 두 단계 외에 추가 단계가 감지됐습니다.",
    "speech_unknown": "음성에서 두 색의 순서를 확정하지 못했습니다.",
}


# 시연 한 번 동안 카메라에서 본 색을 프레임마다 받아, 확정된 단계를 시간순으로 기록한다.
class Procedure:
    def __init__(self, stable_frames, step_delay=1.5):
        self.stable_frames = stable_frames  # 같은 관찰이 몇 프레임 연속 나와야 확정할지
        self.step_delay = step_delay        # 단계 기록 후 다음 단계를 받지 않는 시간
        self.next_event_at = 0.0            # 이 시각 전까지는 새 관찰을 무시
        self.candidate = None               # 지금 확정을 기다리는 관찰
        self.streak = 0                     # candidate가 연속으로 나온 프레임 수
        self.ready = False                  # 시작 전 빈 화면을 확인했는지
        self.events = []                    # 확정된 단계 목록 
        self.reported_issue = None          # 이미 출력한 오류 (같은 오류를 반복 출력하지 않기 위함)

    # 프레임 하나의 관찰 결과를 반영한다.
    # 반환값: "ready"(빈 화면 확인), 색 이름(새 단계 확정), None(변화 없음)
    def update(self, colors, seconds, objects):
        # 직전 단계 기록 후 step_delay 동안은 무시 (물체를 바꾸는 중 오인식 방지)
        if seconds < self.next_event_at:
            return
        # 두 색이면 "both", 한 색이면 그 색, 없으면 "" 
        observation = "both" if len(colors) > 1 else next(iter(colors), "")
        # 같은 관찰이 stable_frames번 연속 나와야 확정 (한 프레임 오인식 무시)
        if observation != self.candidate:
            self.candidate, self.streak = observation, 1
        elif self.streak < self.stable_frames:
            self.streak += 1
        if self.streak < self.stable_frames:
            return
        # 시작 조건: 처음에 빈 화면이 확정돼야 준비 완료
        if not self.ready:
            if not observation:
                self.ready = True
                return "ready"
            return
        # 빈 화면이거나 직전 단계와 같은 색이면 새 단계가 아님
        if not observation or (self.events and self.events[-1]["color"] == observation):
            return
        # 새 단계 기록. 근거로 색이 확인된 물체를 최대 2개까지 함께 저장
        evidence = [obj for obj in objects if obj["colors"]][:2]
        event = {"color": observation, "seconds": round(seconds, 2), "objects": evidence}
        self.events.append(event)
        self.next_event_at = seconds + self.step_delay
        self.candidate, self.streak = None, 0
        return observation

  
    # 순서 오류가 생기면 시연 도중에 바로 알린다.
    def report_event(self, event, seconds):
        if event == "ready":
            print("준비 완료. 물체를 하나씩 보여주면서 설명해 주세요.", flush=True)
        elif event:
            print(
                f"[인식 완료 {seconds:5.1f}초] {LABELS[event]} | 다음 단계까지 {self.step_delay:g}초",
                flush=True,
            )
            issue = self.result()
            if issue in ("wrong_order", "both_visible", "extra_step") and issue != self.reported_issue:
                print(f"[오류 이벤트] {DETAILS[issue]}", flush=True)
                self.reported_issue = issue

    # 미리보기 창아래쪽에 표시할 상태 문구를 만든다.
    def preview_status(self, seconds):
        if not self.ready:
            return "Clear the center to start"
        if not self.events:
            return "Ready for first object"
        color = self.events[-1]["color"]
        label = "BLUE + RED" if color == "both" else color.upper()
        remaining = self.next_event_at - seconds
        next_status = f"Wait {remaining:.1f}s" if remaining > 0 else "Ready for next object"
        return f"Confirmed: {label} | {next_status}"

    # 기록된 단계를 기준 순서(파랑 -> 빨강)와 비교해 화면 판정 코드를 return.
    def result(self):
        if not self.ready:
            return "not_ready"
        observed = tuple(event["color"] for event in self.events)
        if not observed:
            return "no_steps"
        if "both" in observed:
            return "both_visible"
        if observed[0] != EXPECTED[0]:
            return "wrong_order"
        if len(observed) == 1:
            return "missing_red"
        if observed != EXPECTED:
            return "extra_step"
        return "pass"


# 화면 판정과 음성 순서를 합쳐 최종 판정을 내린다.
# 화면이 1차 기준이고, 화면이 맞았을 때만 음성을 확인한다.
# 반환값: 최종 판정 코드, 화면과 음성 순서가 다른지 여부
def evaluate(procedure, transcript):
    camera_result = procedure.result()
    observed = tuple(event["color"] for event in procedure.events)
    spoken = spoken_order(transcript)
    # 화면과 음성의 mismatch는 비교가 가능할 때만 계산 
    mismatch = spoken is not None and len(observed) >= 2 and "both" not in observed and spoken != observed
    camera_text = " → ".join(LABELS[color] for color in observed) or "(확인된 물체 없음)"
    print(f"[화면 순서] {camera_text}", flush=True)
    if spoken is None:
        print(f"[음성 순서] {DETAILS['speech_unknown']}", flush=True)
    else:
        spoken_text = " → ".join(LABELS[color] for color in spoken)
        print(f"[음성 순서] {spoken_text}", flush=True)
    if mismatch:
        print(f"[불일치] {MISMATCH_MESSAGE}", flush=True)

    if camera_result != "pass":
        return camera_result, mismatch
    if spoken is None:
        return "speech_unknown", mismatch
    return ("speech_mismatch" if mismatch else "pass"), mismatch


#사용자가 실제로 말로도 순서를 알고 있나 확인하기 위해 음성에서 순서를 추출한다.
COLORS = re.compile(r"(?P<blue>파란\s*색|파랑\s*색?|파란|푸른\s*색)|(?P<red>빨간\s*색|빨강\s*색?|빨간|붉은\s*색)")  # 색 이름
UNCERTAIN = re.compile(r"아니|아닙|아닌|아닐|아냐|말고|않|못\s*(?:해|했|하|보여|투여|넣)|동시|같이|함께")  # 순서를 확정할 수 없는 말
FIRST = r"(?:첫\s*번째|처음(?:에)?|먼저|1\s*번)"       # 앞 순서 표현
SECOND = r"(?:두\s*번째|그\s*다음(?:에)?|나중|2\s*번)"  # 뒤 순서 표현
AFTER = r"(?:다음|뒤|후|이후)"                          # "~후에"
BEFORE = r"(?:전(?:에|의|부터)?(?=\W|$)|이전(?:에)?|앞(?:에|의))"  # "~전에" ("전체" 같은 단어는 제외)
PARTICLE = r"(?:색깔|물체|약|병|컵)?\s*(?:을|를|이|가|은|는)?\s*"  # 색 뒤에 붙는 명사·조사


# Whisper 문장에서 두 색을 어떤 순서로 설명했는지 찾는다.
# 방향 1 = 말한 순서 그대로, -1 = 반대 순서. 단서가 없거나 엇갈리면 None.
def spoken_order(transcript):
    """순서가 명확한 두 색 설명에서 먼저 나온 단계와 다음 단계를 반환한다."""
    # 서로 다른 색이 정확히 두 개 나와야 하고, 애매한 표현이 없어야 한다.
    matches = list(COLORS.finditer(transcript))
    if len(matches) != 2 or matches[0].lastgroup == matches[1].lastgroup or UNCERTAIN.search(transcript):
        return None

    # 두 색을 기준으로 문장을 앞 / 사이 / 뒤로 나눈다.
    first, second = matches
    listed = (first.lastgroup, second.lastgroup)
    prefix = transcript[:first.start()]
    between = transcript[first.end():second.start()]
    suffix = transcript[second.end():]
    directions = set()

    # 앞부분: "먼저 파란색~" -_> 그대로 / "나중에 파란색~" -> 반대
    if re.search(FIRST, prefix):
        directions.add(1)
    if re.search(SECOND, prefix):
        directions.add(-1)

    # "파란색 전에 빨간색" → 반대 / "...빨간색 먼저" → 반대
    if re.search(rf"^{PARTICLE}{BEFORE}", between):
        directions.add(-1)
    if re.search(FIRST, suffix):
        directions.add(-1)

    # '빨간색은 파란색 다음'과 '파란색은 빨간색 전'은
    # 뒤에 나온 관계 표현이 앞의 색을 설명한다.
    if re.fullmatch(rf"\s*(?:물체|약|병|컵)?\s*(?:은|는|이|가)\s*", between):
        if re.match(rf"^{PARTICLE}{AFTER}", suffix):
            directions.add(-1)
        if re.match(rf"^{PARTICLE}{BEFORE}", suffix):
            directions.add(1)

    # 사이,뒷부분의 순서 표현 ("파란색 먼저 하고 빨간색 나중" 등)
    if re.search(SECOND, between) and re.search(FIRST, suffix):
        directions.add(-1)
    elif re.search(FIRST, between) and re.search(SECOND, suffix):
        directions.add(1)
    elif re.search(rf"{FIRST}|{AFTER}|부터|(?:보여주|투여하|투약하|넣)고|(?:보여준|투여한|투약한|넣은)\s*다음", between):
        directions.add(1)
    elif re.search(SECOND, suffix) and not re.search(FIRST, suffix):
        directions.add(1)

    # 모든 내용이 같은 방향일 때만 순서를 확정한다.
    if len(directions) != 1:
        return None
    return listed if 1 in directions else listed[::-1]

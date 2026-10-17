"""
Prompt templates for "Adversarial Taboo" (금기어 게임).

스파이폴과 달리 이건 **논문 프롬프트 재현이 아님**. SPAG(Adversarial Taboo)
논문은 강화학습으로 self-play 학습을 시키는 게 목적이라 평가용 프롬프트가
그대로 쓸 수 있게 공개돼 있지 않고, 설계 방침("논문과 좀 달라도
괜찮음")에 따라 게임 규칙만 가져와서 프롬프트는 직접 작성함.

작성 원칙 (스파이폴 구현과 맞춤):
- 규칙 설명은 영어, 발화만 {language}로 강제 -- 논문의 id/zh/arz 세팅과 같은 방식
- 응답은 |||{...}||| JSON (../common/llm_client.py의 파싱 규약)
- 예시의 숫자/불리언을 그대로 베끼는 example anchoring을 막는 Note 포함
  (스파이폴 run2에서 confidence 57/57이 예시값 0.3이었던 사례 -- 체크리스트.md)

{language}만 실제 치환 지점임. JSON 예시에 중괄호가 들어있으므로
str.format()을 쓰지 말고 game.py처럼 .replace("{language}", ...)를 쓸 것.
"""

BASE_CONTEXT = """**Game Context:** You are playing "Adversarial Taboo", a two-player conversation game.

**How Adversarial Taboo Works:**
- There is one secret target word, known ONLY to the Attacker
- The Attacker and the Defender take turns, one utterance each, in an ordinary conversation
- The Attacker's goal: steer the conversation so that the Defender says the target word out loud
- The Defender's goal: work out what the target word is and declare it -- without ever saying it in normal conversation

**Win Conditions:**
- Attacker wins: the Defender says the target word in an utterance, OR the Defender declares a wrong word
- Defender wins: the Defender declares the target word correctly, OR the Attacker says the target word themselves
- Draw: the turn limit is reached with none of the above happening

**Key Rules:**
- The Attacker must NEVER say the target word out loud. Saying it is an immediate loss.
- The Defender has exactly ONE declaration for the whole game. It is all-or-nothing: correct = win, wrong = immediate loss.
- The Defender may keep talking without declaring for as long as the turn limit allows. Not declaring is always a legal move.
- **The declaration takes priority.** On a turn where the Defender declares, the declaration alone decides the game -- a correct declaration wins even if the declared word also appears in that same turn's utterance. Saying the target word only loses the game when the Defender is NOT declaring that turn.
- Utterances must be natural conversation. Do not write word lists, multiple-choice questions, or meta-commentary about the game itself.

**The Core Tension:**
- The Attacker must push the conversation close enough to the word that the Defender utters it naturally -- but pushing too obviously tells the Defender exactly what the word is, and they will simply declare it and win.
- The Defender must engage with the conversation to gather clues -- but the more they engage with the Attacker's chosen topic, the more likely they are to blurt the word out by accident.

**Your Task:** Play strategically according to your role.

*As the Attacker:*
- Choose topics that surround the word without naming it. Ask questions whose most natural answer is the target word.
- Avoid asking anything that makes the target word the single obvious answer to the Defender's conscious mind -- they will declare instead of saying it.
- Never say the target word yourself, in any form.

*As the Defender:*
- Answer naturally, but watch where the Attacker is steering you.
- Be careful: the word the conversation is "pulling" you toward is exactly the word you must not say.
- If you are confident you know the word, declare it. If you are not confident, keep talking and gather more clues -- a wrong declaration loses the game immediately.

**Critical Reminders:**
- Consider the whole conversation history before speaking
- Ensure you adhere to the response format
- You must speak in {language} language at any cost!
- Every field you output must be written **entirely in {language}**. Do not mix in any other language: no words from another language, no translations, and no glosses in parentheses (e.g. do not write a {language} word followed by its English equivalent). A word written in the wrong language does not count as that word."""


ATTACKER_UTTERANCE = """{base_context}

You are the Attacker. It is your turn to speak.

Your task: say one natural conversational utterance that pushes the Defender toward saying the target word -- without saying the word yourself.

**Response Format Requirements:**
You must respond with only valid JSON wrapped in triple pipes |||...|||
Think through your approach before outputting JSON. No text should appear after the JSON.
Think step by step!

Example:
Let's think step by step...
|||
{
  "utterance": "YOUR UTTERANCE HERE"
}
|||

Required fields:
- utterance: string (what you actually say out loud this turn)

Important: The target word must not appear anywhere in your utterance."""


DEFENDER_UTTERANCE = """{base_context}

You are the Defender. It is your turn to speak.

Your task: say one natural conversational utterance, and decide whether to declare the target word this turn.

Remember: you get only ONE declaration in the entire game. A correct declaration wins immediately; a wrong one loses immediately. Not declaring is a legal move and costs you nothing except time.

**Response Format Requirements:**
You must respond with only valid JSON wrapped in triple pipes |||...|||
Think through your approach before outputting JSON. No text should appear after the JSON.
Think step by step!

Example (not declaring):
Let's think step by step...
|||
{
  "utterance": "YOUR UTTERANCE HERE",
  "declare": false,
  "guess": null
}
|||

Example (declaring):
Let's think step by step...
|||
{
  "utterance": "YOUR UTTERANCE HERE",
  "declare": true,
  "guess": "YOUR GUESS HERE"
}
|||

Required fields:
- utterance: string (what you actually say out loud this turn)
- declare: boolean (true to declare the target word now, false to keep playing)
- guess: string or null (the word you are declaring if declare=true, else null)

Note: The values in the examples above are formatting placeholders only. Decide `declare` from the actual conversation so far -- do not copy the example values.

Important: If `declare` is false, do not say the word you suspect out loud in your utterance -- saying it loses the game. If `declare` is true, the declaration decides the game and your utterance may name the word freely (e.g. asking the Attacker outright whether that is the word); only `guess` is checked for correctness.

Important: Write both `utterance` and `guess` entirely in {language}. A guess written in another language, or with a translation attached, is judged as a different word and loses."""

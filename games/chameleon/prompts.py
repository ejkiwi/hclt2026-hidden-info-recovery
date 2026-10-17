"""
Prompt templates for the Chameleon game (「카멜레온 재설계 구상안.md」 반영).

규칙 텍스트는 논문 원안(code/paper_code/LLMChameleon-main/llmg/chameleon/constants.py
GAME_START_PROMPT)의 규칙 설명을, 세 게임 공통 프로토콜(|||JSON|||,
base_context 조립, 규칙은 영어 / 발화만 {language} 강제)로 옮긴 것.

논문 원안과 다른 점은 딱 하나 -- 투표 **전** 토론 페이즈(⑤)가 추가됨
(--discussion, 기본 on). 토론을 끄면 {discussion_rule}이 빈 문자열로
치환되어 규칙 설명에서도 토론 문장이 사라짐 = 논문 원안 흐름과 동일.

Substitution points: {language} / {discussion_rule} / {base_context} /
{candidates_rule} / {and_list} / {from_list} (뒤 세 개는 --candidate-mode에
따라 game.py의 _base_context/_player_prompt가 치환). Do NOT use str.format() on these templates (the JSON examples
contain literal curly braces) -- use .replace() instead, and replace
{language} LAST, in ONE place (game.py의 _player_prompt) -- 템플릿
본문 곳곳에 들어가기 때문.

예시 값 베끼기 방지 문구("formatting placeholders")는 스파이폴에서
confidence가 예시값 0.3으로 57/57 고정됐던 사고의 재발 방지용 -- 절대
빼지 말 것 (구상안 '프롬프트 주의' 참고).
"""

BASE_CONTEXT = """**Game Context:** You are playing "The Chameleon" with other players.

**How The Chameleon Works:**
- One player is secretly the Chameleon; all other players are Villagers (Non-Chameleons)
{candidates_rule}
- The Chameleon knows it is the Chameleon, but nobody else knows who the Chameleon is

**Game Flow (one single round):**
- Hint Phase: each player, in a fixed speaking order, says exactly ONE hint word related to the secret word
  - You see the hint words of the players who spoke before you
  - If your hint is too obvious, the Chameleon may work out the secret word
  - If it is too cryptic, the others may start to think YOU are the Chameleon
  - If another player already said the word you were going to say, you may repeat it
- After the Hint Phase, every player sees all hint words
{discussion_rule}- Voting Phase: all players vote simultaneously for who they think the Chameleon is
  - The player with the most votes is accused. If there is a tie, a randomly chosen dealer (one of the players) picks among the tied players
- Final Guess: if the accused player is the Chameleon, the Chameleon gets one last chance to guess the secret word{from_list}

**Win Conditions:**
- Chameleon wins: it is not accused, OR it is accused but guesses the secret word correctly
- Villagers win: the Chameleon is accused AND it fails to guess the secret word

**Critical Reminders:**
- Consider the category{and_list} and everything said so far
- Ensure you adhere to the response format
- You must speak in {language} language at any cost!"""

# 후보 공개 수준 (--candidate-mode). category = **변형 카멜레온** (기본): 목록 없이
# 카테고리만, 전원 비공개 -- 후보 목록을 주지 않는 조건을 카멜레온
# 룰 위에 얹은 것. full = 원작/논문 그대로 (카테고리 + 16개 목록 전원 공개) --
# 원작 재현 대조 조건으로만 유지.
CANDIDATES_RULE_FULL = """- Everyone sees the same category and the same list of candidate words
- The secret word is one of the candidate words. Villagers know the secret word; the Chameleon does NOT"""
CANDIDATES_RULE_CATEGORY = """- Everyone sees the category. There is NO list of candidate words in this game
- The secret word is a word belonging to the category. Villagers know the secret word; the Chameleon does NOT"""

# --discussion(기본 on)일 때만 BASE_CONTEXT의 {discussion_rule} 자리에 들어감.
# 끄면 빈 문자열 -> 논문 원안 흐름 그대로.
# 2바퀴 구조(제기 -> 반박)는 원작의 "몇 분간 공방(pick holes in each other's
# arguments)"을 유한하게 근사한 것 -- 설계 확정 (설계 방침).
DISCUSSION_RULE = """- Discussion Phase: after all hint words are revealed and before voting, there are TWO rounds of debate in the same speaking order
  - Round 1: each player says who they find suspicious and why
  - Round 2: each player responds -- defending themselves if named, rebutting, or updating their suspicion
  - Every statement is public and visible to everyone who speaks after it; everything said here may influence the vote
"""

# --discussion-rounds 1 (1 vs 2바퀴 토론 효과 비교용 조건)일 때의 규칙 문구.
DISCUSSION_RULE_1ROUND = """- Discussion Phase: after all hint words are revealed and before voting, each player speaks ONCE, in the same speaking order, about who they find suspicious and why
  - Earlier statements are visible to later speakers; everything said here is public and may influence the vote
"""

# 1바퀴/2바퀴에 따라 라운드1 토론 템플릿의 {round_note} 자리에 들어갈 문구.
ROUND_NOTE_FIRST_OF_TWO = " (round 1 of 2). After everyone has spoken, there will be one more round to respond."
ROUND_NOTE_SINGLE = ". This is your ONLY discussion statement in this game -- the discussion goes around exactly once."

# --leak-warning(기본 off)일 때 **시민** 토론/반박 템플릿의 {leak_warning} 자리에
# 들어감. 무경고 실험에서 시민 토론 발언의 93~97%가 정답 단어를 그대로 포함하는
# 누출이 관찰됨("Secret word가 '돌고래'라는 점을 고려했을 때...") -- 경고 유/무
# 비교로 이 누출이 무지(경고로 사라짐)인지 능력 부족(경고에도 샘)인지 분리.
# 카멜레온은 정답을 모르므로 해당 없음 (카멜레온 템플릿에는 안 들어감).
LEAK_WARNING = """
CRITICAL: NEVER say the secret word itself out loud in your statement. The Chameleon is listening to this discussion -- if you speak the secret word, you hand it the game. Refer to it only indirectly (e.g. "the secret word", "the answer").
"""


HINT_NON_CHAMELEON = """{base_context}

You are a Villager. It is now your turn to give your hint word.
Your hint must relate to the secret word. Remember: too obvious helps the Chameleon work out the secret word, too cryptic makes the others suspect YOU.

**Response Format Requirements:**
You must respond with only valid JSON wrapped in triple pipes |||...|||
Think through your approach before outputting JSON. No text should appear after the JSON.
Think step by step!

Example:
Let's think step by step...
|||
{
  "hint_word": "YOUR_SINGLE_WORD_HERE"
}
|||

Required fields:
- hint_word: string (exactly ONE single word -- no spaces, no sentences, no punctuation)

Note: The example value above is a formatting placeholder only. Choose your own hint word from the actual game state -- do not copy the example."""

HINT_CHAMELEON = """{base_context}

You are the Chameleon. It is now your turn to give your hint word.
You do NOT know the secret word. Infer what it might be from the category{and_list} and the previous hints, and give a hint word that could plausibly relate to the secret word so that you blend in.

**Response Format Requirements:**
You must respond with only valid JSON wrapped in triple pipes |||...|||
Think through your approach before outputting JSON. No text should appear after the JSON.
Think step by step!

Example:
Let's think step by step...
|||
{
  "hint_word": "YOUR_SINGLE_WORD_HERE"
}
|||

Required fields:
- hint_word: string (exactly ONE single word -- no spaces, no sentences, no punctuation)

Note: The example value above is a formatting placeholder only. Choose your own hint word from the actual game state -- do not copy the example."""


DISCUSSION_NON_CHAMELEON = """{base_context}

You are a Villager. It is now your discussion turn{round_note} Everything you say is public.
Say who you find suspicious of being the Chameleon and why, based on the hint words and any earlier discussion statements.{leak_warning}

**Response Format Requirements:**
You must respond with only valid JSON wrapped in triple pipes |||...|||
Think through your approach before outputting JSON. No text should appear after the JSON.
Think step by step!

Example:
Let's think step by step...
|||
{
  "statement": "YOUR DISCUSSION STATEMENT HERE"
}
|||

Required fields:
- statement: string (what you say out loud to the other players)"""

DISCUSSION_CHAMELEON = """{base_context}

You are the Chameleon. It is now your discussion turn{round_note} Everything you say is public.
Blend in: say who you find suspicious and why, just like a Villager would, so that suspicion does not fall on you.

**Response Format Requirements:**
You must respond with only valid JSON wrapped in triple pipes |||...|||
Think through your approach before outputting JSON. No text should appear after the JSON.
Think step by step!

Example:
Let's think step by step...
|||
{
  "statement": "YOUR DISCUSSION STATEMENT HERE"
}
|||

Required fields:
- statement: string (what you say out loud to the other players)"""


REBUTTAL_NON_CHAMELEON = """{base_context}

You are a Villager. It is now your turn in the second and final discussion round.
You have seen everyone's statements. Respond: if you were named as suspicious, defend yourself; otherwise rebut, agree, or update your suspicion based on what has been said. This is your final statement before the vote. Everything you say is public.{leak_warning}

**Response Format Requirements:**
You must respond with only valid JSON wrapped in triple pipes |||...|||
Think through your approach before outputting JSON. No text should appear after the JSON.
Think step by step!

Example:
Let's think step by step...
|||
{
  "statement": "YOUR DISCUSSION STATEMENT HERE"
}
|||

Required fields:
- statement: string (what you say out loud to the other players)"""

REBUTTAL_CHAMELEON = """{base_context}

You are the Chameleon. It is now your turn in the second and final discussion round.
You have seen everyone's statements. Blend in: respond the way an innocent Villager would -- defend yourself if you were named, or keep the suspicion pointed at someone else. This is your final statement before the vote. Everything you say is public.

**Response Format Requirements:**
You must respond with only valid JSON wrapped in triple pipes |||...|||
Think through your approach before outputting JSON. No text should appear after the JSON.
Think step by step!

Example:
Let's think step by step...
|||
{
  "statement": "YOUR DISCUSSION STATEMENT HERE"
}
|||

Required fields:
- statement: string (what you say out loud to the other players)"""


VOTE_NON_CHAMELEON = """{base_context}

You are a Villager. It is now time to cast your vote for who you think the Chameleon is.
Remember: the Chameleon does not know the secret word -- look for the player whose contributions do not quite fit the secret word. All players vote at the same time, so you cannot see the other players' votes.

**Response Format Requirements:**
You must respond with only valid JSON wrapped in triple pipes |||...|||
Think through your approach before outputting JSON. No text should appear after the JSON.
Think step by step!

Example:
Let's think step by step...
|||
{
  "vote": "Charlie"
}
|||

Required fields:
- vote: string (the name of the player you accuse of being the Chameleon -- must be another player in this game)

Note: The name in the example above is a formatting placeholder only. Choose your vote from the actual players in this game -- do not copy the example name."""

VOTE_CHAMELEON = """{base_context}

You are the Chameleon. It is now time to cast your vote for who you think the Chameleon is.
Vote for another player to deflect suspicion away from yourself. All players vote at the same time, so you cannot see the other players' votes.

**Response Format Requirements:**
You must respond with only valid JSON wrapped in triple pipes |||...|||
Think through your approach before outputting JSON. No text should appear after the JSON.
Think step by step!

Example:
Let's think step by step...
|||
{
  "vote": "Charlie"
}
|||

Required fields:
- vote: string (the name of the player you accuse -- must be another player in this game)

Note: The name in the example above is a formatting placeholder only. Choose your vote from the actual players in this game -- do not copy the example name."""


DEALER_TIEBREAK = """{base_context}

You are the dealer for this game (chosen at random among the players -- this is separate from whether you are the Chameleon or a Villager).
The vote ended in a tie. Players {tied_players} received the most votes.
As the dealer, you must now pick which one of them is accused. Choose from the tied players only.

**Response Format Requirements:**
You must respond with only valid JSON wrapped in triple pipes |||...|||
Think through your approach before outputting JSON. No text should appear after the JSON.
Think step by step!

Example:
Let's think step by step...
|||
{
  "choice": "Charlie"
}
|||

Required fields:
- choice: string (the name of the accused player -- must be one of the tied players listed above)

Note: The name in the example above is a formatting placeholder only. Choose from the tied players in the actual game -- do not copy the example name."""


FINAL_GUESS_CHAMELEON = """{base_context}

You are the Chameleon and you have been accused. You have one last chance to win the game: guess the secret word.
Pick exactly one word from the candidate list, based on the category and the other players' hint words. You cannot skip this.

**Response Format Requirements:**
You must respond with only valid JSON wrapped in triple pipes |||...|||
Think through your approach before outputting JSON. No text should appear after the JSON.
Think step by step!

Example:
Let's think step by step...
|||
{
  "guess": "YOUR_GUESS_HERE"
}
|||

Required fields:
- guess: string (your guess -- must be exactly ONE of the candidate words)

Important: Name exactly ONE candidate word. If your response mentions more than one candidate word, it is invalid -- you cannot hedge between candidates.

Note: The example value above is a formatting placeholder only. Choose your guess from the actual candidate list -- do not copy the example."""

FINAL_GUESS_CHAMELEON_OPEN = """{base_context}

You are the Chameleon and you have been accused. You have one last chance to win the game: guess the secret word.
There is no candidate list -- work out the secret word from the category and the other players' hint words. You cannot skip this.

**Response Format Requirements:**
You must respond with only valid JSON wrapped in triple pipes |||...|||
Think through your approach before outputting JSON. No text should appear after the JSON.
Think step by step!

Example:
Let's think step by step...
|||
{
  "guess": "YOUR_GUESS_HERE"
}
|||

Required fields:
- guess: string (exactly ONE word/concept -- your single best guess for the secret word)

Important: Name exactly ONE guess. If your response lists multiple different guesses, it is invalid -- you cannot hedge.

Note: The example value above is a formatting placeholder only. Work out your own guess from the actual game -- do not copy the example."""

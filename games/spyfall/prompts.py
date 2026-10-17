"""
Prompt templates for the "Spyfall: Structured Edition" game, copied
verbatim from Multicultural Spyfall's Appendix C
(`Multicultural Spyfall_ Assessing LLMs through Dynamic Multilingual
Social Deduction Game.pdf`) -- with ONE marked exception:

  The guess/vote templates each carry an extra "Note:" line (search for
  "formatting placeholders") that is NOT in the paper. It counters
  example anchoring: in a 20-game Qwen2.5-7B run, the spy copied the
  skip example's confidence 0.3 verbatim in 57/57 guess decisions, and
  80% of vote confidences were exact example values. See
  논문과_다르게_간_설계.md for the rationale/data.

Only {language} is a real substitution point in BASE_CONTEXT. Do NOT
use str.format() on these templates (the JSON examples contain literal
curly braces) -- use BASE_CONTEXT.replace("{language}", language)
instead, as done in game.py.
"""

BASE_CONTEXT = """**Game Context:** You are playing "Spyfall: Structured Edition" with other players.

**How Spyfall Structured Edition Works:**
- One player is secretly assigned as "the Spy" at the start
- All other players are "Villagers/Non-Spies" who know the secret entity
- The Spy does not know the entity and must figure it out
- Villagers must identify who the Spy is through questioning and voting

**What is the Entity?**
The entity can be anything: a location (Beach, Hospital), a movie (Star Wars, Titanic),
a famous person (Einstein, Beyoncé), an object (Smartphone, Pizza), a concept (Birthday Party, Job Interview), etc.
Villagers know the entity; the Spy must deduce it from questions and answers.

**Structured Game Flow (Predictable Phases):**

1. Round Robin Phase (Phase 1A):
   - Each player asks the next player one question sequentially
   - Goes around the full circle once
   - Everyone participates, building baseline information

2. Free Question Cycles (n cycles of Phase 1B-2-3):
   Each cycle has three phases:
   a) Phase 1B - Free Question:
      - Last answerer asks anyone one question
      - Provides targeted interrogation opportunities
   b) Phase 2 - Spy Guess Decision:
      - Spy can guess the entity or skip
      - If correct guess → Spy wins
      - If incorrect guess → Villagers win
      - Spy gets n chances throughout the game
   c) Phase 3 - Accusation Vote:
      - All players vote for someone (or skip)
      - Need majority (>50%) to eliminate someone
      - If eliminated player is Spy → Villagers win
      - If eliminated player is Villager → Spy wins
      - No majority → game continues to next cycle

3. Final Round (after n cycles):
   - Final Spy guess opportunity
   - Final accusation vote
   - If no majority in final vote → Spy wins (survived!)

Note: n equals the number of players in the game.

**Win Conditions:**
- Villagers win: Spy guesses wrong OR majority votes eliminate Spy
- Spy wins: Spy guesses correctly OR survives all votes OR Villagers eliminate innocent player

**Strategic Differences from Classic:**
- Voting happens in cycles, not randomly
- Spy has multiple guess opportunities (strategic timing!)
- Vote history is public (builds narrative across rounds)
- Structured phases allow better planning

**Your Task:** Based on your role (Spy or Villager), play strategically:

*As a Villager:*
Share entity knowledge subtly and build your case across voting rounds. Avoid being obvious! Be extremely subtle in giving hints. Think carefully about how you answer and ask questions strategically.
Important guidelines:
- Do not ask or answer questions too directly
- Review the entity list carefully to avoid making it easy for the Spy to guess
- Avoid narrowing it down to one specific place or category
- Do not provide direct hints to the entity (e.g., avoid saying "crew" or "vessel" when the entity is Pirate Ship, or "extreme" when it's Polar Bear)
- Use subtlety and misdirection to make the Spy think multiple entities are possible
- Poor strategy example: Mentioning "patient" when the entity is Hospital, or "temporary stay" when the entity is Hotel
- Be very subtle so the Spy still needs to guess among many possibilities

*As a Spy:*
- Blend in, gather clues across multiple rounds, and time your guess strategically.
- Avoid being obvious and don't reveal that you're the Spy.
- Answer strategically using misdirection and subtlety.
- Do not guess the entity if you cannot narrow it down to one location yet in Spy Guess Decision phase.
Important guidelines:
- Blend in and avoid obvious behavior
- Use misdirection and subtlety in your questions and answers
- Poor strategy example: Asking "Is it a place where people go to relax?" when the entity is Beach
- Be very subtle in your approach

**Critical Reminders:**
- Check the entity guesses list that has been provided to you
- Use it to guide your questioning and answering strategy (for both Spy and Villagers)
- Consider conversation history, voting patterns, and strategic phase timing
- Ensure you adhere to the response format
- You must speak in {language} language at any cost!"""


QUESTION_GENERATION_SPY = """{base_context}

You are the Spy and need to ask a question to another player.

**Response Format Requirements:**
You must respond with only valid JSON wrapped in triple pipes |||...|||
Think through your approach before outputting JSON. No text should appear after the JSON.
Think step by step!

Example:
Let's think step by step...
|||
{
  "question": "YOUR QUESTION HERE",
  "targeted_player": "TARGET_PLAYER_NAME"
}
|||

Required fields:
- question: string (the actual question to ask)
- targeted_player: string (the player being targeted by the question. Think strategically about who to ask)"""

QUESTION_GENERATION_NON_SPY = """{base_context}

You are a Villager and need to ask a question.

**Response Format Requirements:**
You must respond with only valid JSON wrapped in triple pipes |||...|||
Think through your approach before outputting JSON. No text should appear after the JSON.
Think step by step!

Example:
Let's think step by step...
|||
{
  "question": "YOUR QUESTION HERE",
  "targeted_player": "TARGET_PLAYER_NAME"
}
|||

Required fields:
- question: string (the actual question to ask)
- targeted_player: string (the player being targeted by the question. Think strategically about who to ask)"""

ANSWER_GENERATION_SPY = """{base_context}

You are the Spy answering a question.

**Response Format Requirements:**
You must respond with only valid JSON wrapped in triple pipes |||...|||
Think through your approach before outputting JSON. No text should appear after the JSON.
Think step by step!

Example:
Let's think step by step...
|||
{
  "answer": "It can get a bit uncomfortable, so I usually prepare accordingly."
}
|||

Required fields:
- answer: string (your actual answer)

Important: Do not make it obvious that you are the Spy!"""

ANSWER_GENERATION_NON_SPY = """{base_context}

You are a Villager answering a question.

**Response Format Requirements:**
You must respond with only valid JSON wrapped in triple pipes |||...|||
Think through your approach before outputting JSON. No text should appear after the JSON.
Think step by step!

Example:
Let's think step by step...
|||
{
  "answer": "YOUR ANSWER HERE"
}
|||

Required fields:
- answer: string (your actual answer)"""

ENTITY_GUESS_SPY = """{base_context}

You are the Spy in Phase 2 (Spy Guess Decision).

Important: You do not have to rush if you are unsure! Skip if you haven't pinpointed a single location yet.

Your task: Decide whether to guess the entity now or skip.

**Response Format Requirements:**
You must respond with only valid JSON wrapped in triple pipes |||...|||
Think through your approach before outputting JSON. No text should appear after the JSON.
Think step by step!

Example (making a guess):
Let's think step by step...
|||
{
  "best_guess": "Beach",
  "should_guess": true,
  "confidence": 0.85
}
|||

Example (skipping):
Let's think step by step...
|||
{
  "best_guess": null,
  "should_guess": false,
  "confidence": 0.3
}
|||

Required fields:
- best_guess: string or null (your guess from the entity list if should_guess=true, else null)
- should_guess: boolean (true to guess now, false to skip)
- confidence: number 0.0-1.0 (how confident you are in your guess)

Note: The numeric values in the examples above are formatting placeholders only. Decide should_guess from the actual game state and choose your own confidence value -- do not copy the example numbers."""

VOTE_INITIATION_SPY = """{base_context}

You are the Spy deciding who to vote for in the Accusation Vote phase.
You can skip voting if you are unsure!

**Response Format Requirements:**
You must respond with only valid JSON wrapped in triple pipes |||...|||
Think through your approach before outputting JSON. No text should appear after the JSON.
Think step by step!

Example:
Let's think step by step...
|||
{
  "target_player_name": "Charlie",
  "should_vote": true,
  "confidence": 0.75
}
|||

Example to skip voting:
Let's think step by step...
|||
{
  "target_player_name": null,
  "should_vote": false,
  "confidence": 0.6
}
|||

Required fields:
- target_player_name: string or null (player name, or null to skip)
- should_vote: boolean (true to vote for target, false to skip)
- confidence: number 0.0-1.0 (confidence in your voting decision)

Note: The names and numeric values in the examples above are formatting placeholders only. Decide should_vote from the actual game state and choose your own target and confidence value -- do not copy the examples."""

VOTE_INITIATION_NON_SPY = """{base_context}

You are a Villager deciding who to vote for in the Accusation Vote phase (Phase 3).
You can skip voting if you are unsure!

**Response Format Requirements:**
You must respond with only valid JSON wrapped in triple pipes |||...|||
Think through your approach before outputting JSON. No text should appear after the JSON.
Think step by step!

Example:
Let's think step by step...
|||
{
  "target_player_name": "Charlie",
  "should_vote": true,
  "confidence": 0.88
}
|||

To skip voting:
Let's think step by step...
|||
{
  "target_player_name": null,
  "should_vote": false,
  "confidence": 0.45
}
|||

Required fields:
- target_player_name: string or null (player name, or null to skip)
- should_vote: boolean (true to vote for target, false to skip)
- confidence: number 0.0-1.0 (confidence in your voting decision)

Note: The names and numeric values in the examples above are formatting placeholders only. Decide should_vote from the actual game state and choose your own target and confidence value -- do not copy the examples."""

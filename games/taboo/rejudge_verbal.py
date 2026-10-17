"""기존 금기어 트랜스크립트에 '발화 속 추측 선언' 규칙(절충안 ⓐ)을 소급 적용.

누출 패("defender said the target word out loud")로 끝난 게임의 마지막 발화가
추측 선언이었는지 verbal_declaration으로 재판정한다. 게임은 어느 판정에서든
그 발화에서 끝나므로(승자만 다름) **소급 재판정이 재실행과 정확히 동치**다 --
이후 턴이 존재하지 않아 궤적 차이가 없음.

같이 뽑는 것:
- 뒤집힌 게임 목록 + 걸린 표지
- 판정 전/후 승률 비교
- 표지에 안 걸렸지만 애매해 보이는 사례 -> LLM 사후 검증기(절충안 ⓑ)에 넘길 후보

사용법:
    python rejudge_verbal.py transcripts/20260805_qwen35-dense-*_Korean_50games.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from game import verbal_declaration

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def rejudge(path: Path) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    flipped, kept, ambiguous = [], [], []
    for g in d:
        if "defender said the target word out loud" not in g["reason"]:
            continue
        last = next(h for h in reversed(g["history"]) if h.get("said_target"))
        marker = verbal_declaration(last["utterance"], g["target"])
        if marker:
            flipped.append((g["target"], marker, last["utterance"][:70]))
        else:
            kept.append((g["target"], last["utterance"][:70]))
            # 표지엔 안 걸렸지만 물음표/추측 어감이 있으면 LLM 검증 후보
            if "?" in last["utterance"] or "같" in last["utterance"]:
                ambiguous.append((g["target"], last["utterance"][:90]))

    n = len(d)
    old_att = sum(1 for g in d if g["winner"] == "attacker")
    old_def = sum(1 for g in d if g["winner"] == "defender")
    new_att, new_def = old_att - len(flipped), old_def + len(flipped)
    print(f"\n=== {path.name} ({n}판) ===")
    print(f"  누출 패 중 추측 선언으로 뒤집힘: {len(flipped)}건 / 우발 언급 유지: {len(kept)}건")
    print(f"  공격자 승  {old_att:2} → {new_att:2}   방어자 승  {old_def:2} → {new_def:2}")
    for t, m, u in flipped:
        print(f"    ↺ [{t}] ({m!r}) {u}")
    if ambiguous:
        print(f"  ⚠ LLM 검증 후보(표지 미검출인데 추측 어감) {len(ambiguous)}건:")
        for t, u in ambiguous:
            print(f"    ? [{t}] {u}")


if __name__ == "__main__":
    files = [Path(a) for a in sys.argv[1:]] or sorted(Path("transcripts").glob("*.json"))
    for f in files:
        try:
            rejudge(f)
        except Exception as e:  # noqa: BLE001
            print(f"  [건너뜀] {f.name}: {e}")

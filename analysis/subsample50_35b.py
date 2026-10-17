# -*- coding: utf-8 -*-
# 35B 시드 42 랜덤 50회 전체 재집계 + 분해 막대 35B 좌표 산출
# 검증 원칙: 100회 전체값이 논문 검증값과 일치하는지 먼저 확인한 뒤 서브샘플을 낸다.
import json, random, sys, os, re
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
B = Path(__file__).resolve().parent.parent / "results" / "main-100games"

def load(fn):
    d = json.load(open(B / fn, encoding="utf-8"))
    return d["games"] if isinstance(d, dict) and "games" in d else d

IDX = sorted(random.Random(42).sample(range(100), 50))

def leak_hit(secret, text, lang):
    if not text: return False
    if lang == "EN":
        return re.search(r"\b" + re.escape(secret) + r"\b", text, re.I) is not None
    return secret in text

# ---------------- 스파이폴 ----------------
def spy_stats(games):
    n = len(games)
    correct = wrong = 0
    first_att = first_ok = 0
    for g in games:
        k = 0
        for e in g["history"]:
            if e.get("phase") == "guess":
                k += 1
                if e.get("should_guess") and e.get("best_guess"):
                    ok = (g["winner"] == "spy" and "incorrectly" not in g["reason"])
                    if ok: correct += 1
                    else: wrong += 1
                    if k == 1:
                        first_att += 1
                        if ok: first_ok += 1
                    break
    return dict(n=n, 정답=correct, 오답=wrong, 시도없음=n-correct-wrong,
                적중=f"{correct}/{correct+wrong}", 첫기회=f"{first_ok}/{first_att}")

# ---------------- 금기어 ----------------
def taboo_stats(games):
    n = len(games)
    cor = ver = wrong = suicide = saidword = 0
    f_att = f_ok = f_ver = 0
    for g in games:
        r = g["reason"]; tu = g.get("turns_used")
        if "declared the target word" in r and g["winner"] == "defender":
            cor += 1
            isv = "verbally" in r
            if isv: ver += 1
            if tu == 1:
                f_att += 1; f_ok += 1
                if isv: f_ver += 1
        elif "declared the wrong word" in r:
            wrong += 1
            if tu == 1: f_att += 1
        elif "attacker said the target word" in r:
            suicide += 1
        elif "defender said the target word" in r:
            saidword += 1
    return dict(n=n, 정답=cor, 오답=wrong, 시도없음=n-cor-wrong,
                적중=f"{cor}/{cor+wrong}", 첫기회=f"{f_ok}/{f_att}",
                첫기회중발화선언=f_ver, 발화선언=ver, 자멸=suicide, 방어자누출패=saidword)

# ---------------- 카멜레온 ----------------
def cham_stats(games, lang):
    n = len(games)
    acc = accok = 0
    leak = 0; noleak_acc = 0; noleak_ok = 0
    for g in games:
        secret = g["secret_word"]
        cham = g["chameleon"]
        leaked = any(
            leak_hit(secret, e.get("hint_word") or e.get("statement"), lang)
            for e in g["history"]
            if e.get("phase") in ("hint", "discussion") and e.get("actor") != cham)
        if leaked: leak += 1
        if g.get("identified"):
            acc += 1
            ok = any(e.get("phase") == "final_guess" and e.get("correct") for e in g["history"])
            if ok: accok += 1
            if not leaked:
                noleak_acc += 1
                if ok: noleak_ok += 1
    return dict(n=n, 정답=accok, 오답=acc-accok, 시도없음=n-acc,
                지목=acc, 적중=f"{accok}/{acc}", 노출판=leak,
                무노출지목=f"{noleak_ok}/{noleak_acc}")

FILES = {
 ("spy","KR"): "20260809_qwen35-35b_Korean_generic_100games.json",
 ("spy","EN"): "20260809_qwen35-35b_English_generic_100games.json",
 ("taboo","KR"): "20260814_qwen35-35b_Korean_100games_judge35B.json",
 ("taboo","EN"): "20260814_qwen35-35b_English_100games_judge35B.json",
 ("cham","KR"): "20260809_qwen35-35b_Korean_full_2ROUND-disc_WARN_100games.json",
 ("cham","EN"): "20260809_qwen35-35b_English_full_2ROUND-disc_WARN_100games.json",
}

print("인덱스(시드 42):", IDX)
print()
for key, fn in FILES.items():
    games = load(fn)
    game, lang = key
    fx = {"spy": spy_stats, "taboo": taboo_stats}.get(game)
    full = fx(games) if fx else cham_stats(games, lang)
    sub = fx([games[i] for i in IDX]) if fx else cham_stats([games[i] for i in IDX], lang)
    print(f"== {game} {lang} ==")
    print("  100회:", full)
    print("   50회:", sub)

# ---------------- 표 2 사례 게임 포함 여부 ----------------
print()
print("== 표 2 사례 포함 여부 (인덱스 in 서브샘플) ==")
tk = load(FILES[("taboo","KR")])
for i, g in enumerate(tk):
    if g["target"] == "커피" and g["winner"] == "defender" and g.get("turns_used") == 1:
        print(f"  금기어 '커피' game#{i}: {'포함' if i in IDX else '탈락'}")
sk = load(FILES[("spy","KR")])
for i, g in enumerate(sk):
    if g["target"] in ("대사관", "Embassy") and "correctly" in g["reason"]:
        k = 0; opp1 = False
        for e in g["history"]:
            if e.get("phase") == "guess":
                k += 1
                if e.get("should_guess") and e.get("best_guess"):
                    opp1 = (k == 1); break
        if opp1:
            print(f"  스파이폴 '대사관' 첫 기회 정답 game#{i}: {'포함' if i in IDX else '탈락'}")

# Hidden Information Recovery of LLMs across Three Social Deduction Games

HCLT 2026 논문 「사회 추론 게임 3종에서 관찰된 LLM의 은닉 정보 복원」의 코드와 실험 로그.

## 구성

```
games/                 스파이폴 · 금기어 · 카멜레온 게임 엔진과 프롬프트
games/*/transcripts/   Qwen3.5 4B · 9B · 27B, Gemma 3 27B 게임 로그
results/               Qwen3.5-35B-A3B (주 실험) 게임 로그
analysis/              논문 수치 집계 스크립트
```

## 논문 수치 재현

저장된 로그만 읽어 집계하므로 LLM 호출이 필요 없다.

```bash
cd analysis
python attempt_accuracy.py    # 도달률 · 적중률 (4B · 9B · 27B · Gemma)
python subsample50_35b.py     # 도달률 · 적중률 (35B)
python opportunity_curve.py   # 답할 기회별 누적 정답률 곡선 (전 모델)
```

주 실험 모델은 게임당 100회 실행했고, 보고값은 시드 42로 무작위 추출한 50회 기준이다(논문 3.3절).

## 게임 실행

```bash
pip install -r requirements.txt
cd games/spyfall
python run_game.py --help
```

OpenAI 호환 엔드포인트라면 로컬 서버와 API 모두 `--base-url`로 연결할 수 있다.

스파이폴 규칙과 프롬프트 구조는 Multicultural Spyfall (arXiv:2601.09017)을 따랐다.

## License

MIT

"""4090 bench.py 결과(--save)의 생성 문장을 규칙으로 채점하고, 방식끼리 같은 문장인지 비교한다. GPU 불필요.

  python3 compare_outputs.py --test ~/to4090/v2_lora_b_test.jsonl f2_merged16.json f3_merged8.json f1_split16.json f5_split8.json

채점 규칙은 eval_lora_b_v2.py와 같다 (같은 파일을 import). 사람 검수가 아니라 코드 검사.
"""
import argparse, json, re, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--test", required=True, help="bench.py에 쓴 v2 test jsonl")
ap.add_argument("files", nargs="+", help="bench.py --save 결과 json (앞 파일이 기준)")
args = ap.parse_args()

src = Path(__file__).with_name("eval_lora_b_v2.py").read_text(encoding="utf-8")
# eval 스크립트에서 규칙 부분만 가져온다 (모델 로딩·인자 파싱 부분은 건너뜀)
rules = src[src.index("DIG ="):src.index("per = defaultdict")]
ns = {"re": re}
exec(rules, ns)
nums, QUESTION_END, BANNED, keywords = ns["nums"], ns["QUESTION_END"], ns["BANNED"], ns["keywords"]

rows = [json.loads(l) for l in Path(args.test).read_text(encoding="utf-8").splitlines() if l.strip()]
by_id = {}
for r in rows:
    by_id.setdefault(r.get("scenario_id"), r)

runs = {}
for f in args.files:
    d = json.load(open(f, encoding="utf-8"))["LoRA-B"]
    runs[f] = {x["id"]: x["text"].strip() for x in d}


def check(pred, r):
    msgs = r["messages"][:-1]
    u = json.loads(msgs[1]["content"])
    action, ntype = u.get("action"), (u.get("next") or {}).get("type")
    sents = [s for s in re.split(r"(?<=[.?!])\s+", pred) if s]
    kw = keywords((u.get("next") or {}).get("text"))
    return {
        "끝맺음X": not (bool(QUESTION_END.search(pred)) if ntype in ("NEXT", "SAME") else ("?" not in pred) if ntype == "END" else True),
        "다음질문X": not ((sum(k in pred for k in kw) >= max(1, len(kw) // 3)) if ntype == "NEXT" and kw else True),
        "새숫자": bool(nums(pred) - nums(msgs[1]["content"])),
        "금지표현": any(b in pred for b in BANNED),
        "문장수X": not (1 <= len(sents) <= 3),
        "JSON": pred.lstrip().startswith("{"),
    }


flagged = []
print(f"{'파일':<22}{'건수':>6}" + "".join(f"{k:>9}" for k in ["끝맺음X", "다음질문X", "새숫자", "금지표현", "문장수X", "JSON"]))
for f, out in runs.items():
    tot = {}
    n = 0
    for sid, pred in out.items():
        r = by_id.get(sid)
        if not r:
            continue
        n += 1
        for k, v in check(pred, r).items():
            tot[k] = tot.get(k, 0) + v
            if v and k in ("새숫자", "끝맺음X"):
                flagged.append((Path(f).name, k, sid, pred, json.loads(r["messages"][1]["content"]).get("user_utterance")))
    print(f"{Path(f).name:<22}{n:>6}" + "".join(f"{tot.get(k, 0):>9}" for k in ["끝맺음X", "다음질문X", "새숫자", "금지표현", "문장수X", "JSON"]))

base_f = args.files[0]
print(f"\n[기준 {Path(base_f).name}과 문장이 완전히 같은 비율]")
for f, out in runs.items():
    if f == base_f:
        continue
    common = [k for k in out if k in runs[base_f]]
    same = sum(out[k] == runs[base_f][k] for k in common)
    print(f"{Path(f).name:<22}{same}/{len(common)} ({same / max(1, len(common)):.1%})")

print("\n[다른 문장 예시 (처음 3건)]")
for f, out in runs.items():
    if f == base_f:
        continue
    shown = 0
    for k in out:
        if k in runs[base_f] and out[k] != runs[base_f][k] and shown < 3:
            print(f"- {Path(f).name} | {k}\n    기준: {runs[base_f][k]}\n    이쪽: {out[k]}")
            shown += 1

print("\n[새숫자·끝맺음X로 걸린 문장]")
for name, kind, sid, pred, utt in flagged:
    print(f"- {name} | {kind} | {sid}\n    환자: {utt}\n    응답: {pred}")

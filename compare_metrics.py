"""LoRA-A 평가 결과(finetuned_A.json / baseline_A.json)를 태현님 보고서와 같은 항목으로 다시 계산한다.

  python compare_metrics.py ../finetuned_A.json
  python compare_metrics.py ../baseline_A.json

항목 정의 (정답 사실 1개 = 정답 facts의 한 줄 기준):
  - 사실 찾음   : 같은 (item_id, field)인 예측 사실이 있음
  - 값 일치     : 찾은 것 중 value까지 같음
  - 상태 일치   : 찾은 것 중 semantic_status까지 같음
  - 근거 원문   : 찾은 것의 evidence.text가 그 turn_index 사용자 발화 안에 있음
  - 근거 위치   : 찾은 것의 start/end가 정답과 같음
  - 코드 위치   : 모델이 낸 근거 문장을 코드가 원문에서 찾아 계산했다면 위치가 정답과 같았을지
  - 의도 / 정정 : 정답에 있는 intent / CORRECTION 관계를 같은 item_id로 맞혔는지
  - P/R/F1 두 가지
      명찬 방식 : (문항, 필드, 값, 상태) + (문항, 의도) + (문항, 관계) 묶음
      상태 제외 : (문항, 필드, 값)        + (문항, 의도) + (문항, 관계) 묶음
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).parent


def L(x):
    return x if isinstance(x, list) else []


def parse(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    d = json.loads(text)
    if not isinstance(d, dict):
        raise ValueError
    return d


def keyset(d, with_status):
    facts = {(f.get("item_id"), f.get("field"), json.dumps(f.get("value"), ensure_ascii=False))
             + ((f.get("semantic_status"),) if with_status else ())
             for f in L(d.get("facts")) if isinstance(f, dict)}
    intents = {(i.get("item_id"), "INTENT", i.get("intent")) for i in L(d.get("intents")) if isinstance(i, dict)}
    rels = {(r.get("item_id"), "REL", r.get("relation")) for r in L(d.get("relations")) if isinstance(r, dict)}
    return facts | intents | rels


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main(path):
    res = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = {json.loads(l)["id"]: json.loads(l) for l in (HERE / "data_slim/A_eval.jsonl").read_text(encoding="utf-8").splitlines()}

    c = dict(cases=0, json_ok=0, gold=0, found=0, value=0, status=0, ev_text=0, ev_pos=0, ev_code=0,
             intent_gold=0, intent_ok=0, corr_gold=0, corr_ok=0, exact_nopos=0)
    pr = {True: [0, 0, 0], False: [0, 0, 0]}
    for o in res["outputs"]:
        c["cases"] += 1
        gold = json.loads(o["gold"])
        user = {t["turn_index"]: t["content"] for t in json.loads(rows[o["id"]]["messages"][1]["content"])["context"] if t["role"] == "user"}
        try:
            pred = parse(o["pred"]); c["json_ok"] += 1
        except Exception:
            pred = {}
        for ws in (True, False):
            G, P = keyset(gold, ws), keyset(pred, ws)
            pr[ws][0] += len(P & G); pr[ws][1] += len(P - G); pr[ws][2] += len(G - P)
        c["exact_nopos"] += keyset(gold, True) == keyset(pred, True)

        pmap = {(f.get("item_id"), f.get("field")): f for f in L(pred.get("facts")) if isinstance(f, dict)}
        for g in L(gold.get("facts")):
            c["gold"] += 1
            p = pmap.get((g["item_id"], g["field"]))
            if not p:
                continue
            c["found"] += 1
            c["value"] += p.get("value") == g["value"]
            c["status"] += p.get("semantic_status") == g["semantic_status"]
            pe = p.get("evidence") if isinstance(p.get("evidence"), dict) else {}
            ge = g["evidence"]
            src = user.get(pe.get("turn_index"), "")
            txt = pe.get("text") if isinstance(pe.get("text"), str) else ""
            c["ev_text"] += bool(txt) and txt in src
            c["ev_pos"] += pe.get("start") == ge["start"] and pe.get("end") == ge["end"]
            if txt and txt in src:   # 코드가 원문에서 찾아 위치를 계산했다면
                s = src.find(txt)
                c["ev_code"] += pe.get("turn_index") == ge["turn_index"] and s == ge["start"] and s + len(txt) == ge["end"]
        for g in L(gold.get("intents")):
            c["intent_gold"] += 1
            c["intent_ok"] += any(i.get("item_id") == g["item_id"] and i.get("intent") == g["intent"] for i in L(pred.get("intents")) if isinstance(i, dict))
        for g in L(gold.get("relations")):
            if g.get("relation") != "CORRECTION":
                continue
            c["corr_gold"] += 1
            c["corr_ok"] += any(r.get("item_id") == g["item_id"] and r.get("relation") == "CORRECTION" for r in L(pred.get("relations")) if isinstance(r, dict))

    pct = lambda a, b: f"{a}/{b} ({a / b * 100:.1f}%)" if b else "해당 없음"
    print(f"파일: {path}  (사례 {c['cases']}건, 정답 사실 {c['gold']}개)\n")
    print(f"JSON으로 읽힘            {pct(c['json_ok'], c['cases'])}")
    print(f"정답 사실을 찾음(문항·필드) {pct(c['found'], c['gold'])}")
    print(f"값까지 일치              {pct(c['value'], c['gold'])}")
    print(f"상태 일치                {pct(c['status'], c['gold'])}")
    print(f"의도 일치                {pct(c['intent_ok'], c['intent_gold'])}")
    print(f"정정 관계 일치           {pct(c['corr_ok'], c['corr_gold'])}")
    print(f"근거 문장이 원문에 있음   {pct(c['ev_text'], c['found'])}  (찾은 사실 기준)")
    print(f"근거 위치가 맞음          {pct(c['ev_pos'], c['found'])}")
    print(f"코드가 위치 계산했다면    {pct(c['ev_code'], c['found'])}")
    print(f"완전히 같은 행(위치 제외) {pct(c['exact_nopos'], c['cases'])}")
    for ws, name in ((True, "명찬 방식 (상태 포함)"), (False, "상태 제외")):
        p, r, f = prf(*pr[ws])
        print(f"P / R / F1 {name:<14} {p:.2f} / {r:.2f} / {f:.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE.parent / "finetuned_A.json"))

"""LoRA-A 출력(JSON)을 줄이는 형식들의 토큰 수를 비교하고, 코드로 원래 정답을 되살릴 수 있는지 검사한다.

  pip install tokenizers huggingface_hub
  python compact_formats.py

형식 (아래로 갈수록 더 줄임, 각 단계는 앞 단계를 포함):
  F0 원본       : 지금 정답 그대로
  F1 위치 제거   : evidence의 start/end 삭제 → 코드가 원문에서 문장을 찾아 계산
  F2 빈 목록 생략 : 비어 있는 intents/relations/unmapped_facts 키 삭제
  F3 기본값 생략 : precision="exact", semantic_status="ANSWERED", 근거 turn_index=현재 사용자 턴이면 생략
  F4 키 축약     : facts→f, item_id→i, field→k, value→v, semantic_status→s, precision→p, evidence→e, text→x ...
  F5 한 줄 형식   : JSON 대신 사실 1개 = 한 줄 "문항|필드|값|상태|정밀도|근거문장" (의도·관계도 줄로)

'되살리기 검사'는 각 형식 → 원본 JSON으로 복원했을 때 정답과 완전히 같은지 확인한다.
(F1부터는 start/end를 코드가 원문에서 찾아 다시 계산)
"""
import json, statistics
from pathlib import Path
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

HERE = Path(__file__).parent
SHORT = {"facts": "f", "intents": "n", "relations": "r", "unmapped_facts": "u", "item_id": "i", "field": "k",
         "value": "v", "semantic_status": "s", "precision": "p", "evidence": "e", "text": "x", "turn_index": "t",
         "intent": "a", "relation": "l", "correction_target": "c", "scope": "o"}
LONG = {v: k for k, v in SHORT.items()}
LIST_KEYS = ("facts", "intents", "relations", "unmapped_facts")


def dumps(d):
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


def walk(o, fn):
    """dict/list를 돌면서 evidence dict마다 fn 적용"""
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "evidence" and isinstance(v, dict):
                fn(v)
            else:
                walk(v, fn)
    elif isinstance(o, list):
        for v in o:
            walk(v, fn)


# ---------- 줄이기 ----------
def f1(d):
    d = json.loads(dumps(d))
    walk(d, lambda e: (e.pop("start", None), e.pop("end", None)))
    return d


def f2(d):
    d = f1(d)
    return {k: v for k, v in d.items() if not (k in LIST_KEYS and v == [])}


def f3(d, cur_turn):
    d = f2(d)
    for f in d.get("facts", []):
        if f.get("precision") == "exact": f.pop("precision")
        if f.get("semantic_status") == "ANSWERED": f.pop("semantic_status")
    walk(d, lambda e: e.pop("turn_index") if e.get("turn_index") == cur_turn else None)
    return d


def shorten(o):
    if isinstance(o, dict):
        return {SHORT.get(k, k): shorten(v) for k, v in o.items()}
    if isinstance(o, list):
        return [shorten(v) for v in o]
    return o


def f4(d, cur_turn):
    return shorten(f3(d, cur_turn))


def f5(d, cur_turn):
    d = f3(d, cur_turn)
    ev = lambda e: (f"@{e['turn_index']}:" if "turn_index" in e else "") + e["text"]
    lines = []
    for f in d.get("facts", []):
        lines.append("|".join([f["item_id"], f["field"], dumps(f["value"]), f.get("semantic_status", ""), f.get("precision", ""), ev(f["evidence"])]))
    for i in d.get("intents", []):
        lines.append("|".join(["#I", i["item_id"], i["intent"], ev(i["evidence"])]))
    for r in d.get("relations", []):
        c = r.get("correction_target") or {}
        lines.append("|".join(["#R", r["item_id"], r["relation"], c.get("item_id", ""), c.get("field", ""), str(c.get("turn_index", "")), ev(r["evidence"])]))
    for u in d.get("unmapped_facts", []):
        lines.append("|".join(["#U", u.get("scope", ""), ev(u["evidence"])]))
    return "\n".join(lines) if lines else "-"


# ---------- 되살리기 ----------
def fill_positions(d, turns, cur_turn):
    def fix(e):
        t = e.setdefault("turn_index", cur_turn)
        s = turns[t].find(e["text"])
        e["start"], e["end"] = s, s + len(e["text"])
    walk(d, fix)
    return d


def restore_json(d, turns, cur_turn, short=False):
    d = json.loads(dumps(d))
    if short:
        d = json.loads(dumps(d), object_hook=lambda o: {LONG.get(k, k): v for k, v in o.items()})
    for k in LIST_KEYS:
        d.setdefault(k, [])
    for f in d["facts"]:
        f.setdefault("precision", "exact")
        f.setdefault("semantic_status", "ANSWERED")
    return fill_positions(d, turns, cur_turn)


def restore_lines(s, turns, cur_turn):
    d = {k: [] for k in LIST_KEYS}
    def ev(x):
        if x.startswith("@"):
            t, x = x[1:].split(":", 1)
            return {"turn_index": int(t), "text": x}
        return {"text": x}
    for line in ([] if s == "-" else s.split("\n")):
        p = line.split("|")
        if p[0] == "#I":
            d["intents"].append({"item_id": p[1], "intent": p[2], "evidence": ev("|".join(p[3:]))})
        elif p[0] == "#R":
            r = {"item_id": p[1], "relation": p[2], "evidence": ev("|".join(p[6:]))}
            if p[3]:
                r["correction_target"] = {"item_id": p[3], "field": p[4], "turn_index": int(p[5])}
            d["relations"].append(r)
        elif p[0] == "#U":
            d["unmapped_facts"].append({"scope": p[1], "evidence": ev("|".join(p[2:]))})
        else:
            d["facts"].append({"item_id": p[0], "field": p[1], "value": json.loads(p[2]),
                               "semantic_status": p[3] or "ANSWERED", "precision": p[4] or "exact", "evidence": ev("|".join(p[5:]))})
    return fill_positions(d, turns, cur_turn)


def canon(d):
    return json.loads(dumps(d), object_pairs_hook=lambda kv: dict(sorted(kv)))


def main():
    tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen3-1.7B", "tokenizer.json"))
    n = lambda s: len(tok.encode(s + "<|im_end|>", add_special_tokens=False).ids)   # 끝 표시까지 생성해야 하므로 포함

    rows = []
    for split in ("train", "eval"):
        for l in (HERE / f"data_slim/A_{split}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(l); r["split"] = split; rows.append(r)

    names = ["F0 원본", "F1 위치 제거", "F2 빈 목록 생략", "F3 기본값 생략", "F4 키 축약", "F5 한 줄 형식"]
    toks = {k: {"all": [], "eval": []} for k in names}
    ok = {k: 0 for k in names}
    example = None
    for r in rows:
        gold = json.loads(r["messages"][-1]["content"])
        ctx = json.loads(r["messages"][1]["content"])["context"]
        turns = {t["turn_index"]: t["content"] for t in ctx}
        cur = max(t["turn_index"] for t in ctx if t["role"] == "user")
        outs = {
            "F0 원본": (r["messages"][-1]["content"], lambda s: json.loads(s)),
            "F1 위치 제거": (dumps(f1(gold)), lambda s: restore_json(json.loads(s), turns, cur)),
            "F2 빈 목록 생략": (dumps(f2(gold)), lambda s: restore_json(json.loads(s), turns, cur)),
            "F3 기본값 생략": (dumps(f3(gold, cur)), lambda s: restore_json(json.loads(s), turns, cur)),
            "F4 키 축약": (dumps(f4(gold, cur)), lambda s: restore_json(json.loads(s), turns, cur, short=True)),
            "F5 한 줄 형식": (f5(gold, cur), lambda s: restore_lines(s, turns, cur)),
        }
        for k, (text, back) in outs.items():
            t = n(text)
            toks[k]["all"].append(t)
            if r["split"] == "eval":
                toks[k]["eval"].append(t)
            try:
                ok[k] += canon(back(text)) == canon(gold)
            except Exception:
                pass
        if r["id"] == "labeled100_008":
            example = {k: v[0] for k, v in outs.items()}

    base_all = statistics.mean(toks["F0 원본"]["all"]); base_ev = statistics.mean(toks["F0 원본"]["eval"])
    print(f"{'형식':<14}{'평균 토큰(100건)':>16}{'줄어든 비율':>12}{'eval 13건 평균':>16}{'되살리기':>12}")
    for k in names:
        a, e = statistics.mean(toks[k]["all"]), statistics.mean(toks[k]["eval"])
        print(f"{k:<14}{a:>14.1f}{(1 - a / base_all) * 100:>11.0f}%{e:>15.1f}{ok[k]:>9}/{len(rows)}")
    print(f"\n최대 토큰: 원본 {max(toks['F0 원본']['all'])} → F4 {max(toks['F4 키 축약']['all'])} → F5 {max(toks['F5 한 줄 형식']['all'])}")
    print("\n예시 (labeled100_008)")
    for k, v in example.items():
        print(f"\n[{k}] {n(v)}토큰\n{v}")


if __name__ == "__main__":
    main()

"""받은 extraction.jsonl(LoRA-A) / response.jsonl(LoRA-B)를 학습용 messages 형식으로 변환.

사용법:  python prepare_data.py [데이터 폴더]     # full + slim 둘 다 생성
         데이터 폴더 기본값: dataset/data (우석님 패키지의 data/ 폴더, extraction.jsonl·response.jsonl이 있는 곳)
출력:   data_full/{A,B}_{train,eval}.jsonl, data_slim/{A,B}_{train,eval}.jsonl

full : 원본 프롬프트 그대로 (설문 31문항 전체 + 대화 전체, 평균 약 7,800토큰)
slim : 토이 학습용으로 줄인 프롬프트
       - 설문 카탈로그: 현재 문항(item_id)만 (B는 다음 문항 next_item_id 추가)
       - 대화: 마지막 8턴만, 각 턴에 원래 turn_index를 붙여 둠 (근거 위치 번호 유지)
       - visible_state: pending / deferred / 현재 문항 관련 confirmed만
"""
import json, random, collections, sys
from pathlib import Path

DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset/data")

KEEP_TURNS = 8
EVAL_PER_CAT = 1          # 카테고리마다 1건씩 평가용으로 떼어냄 (3건 이상인 카테고리만)
random.seed(42)

def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]

def slim_user(u, keep_items):
    ctx = u["context"]
    start = max(0, len(ctx) - KEEP_TURNS)
    new = {"context": [{"turn_index": i, "role": m["role"], "content": m["content"]}
                       for i, m in enumerate(ctx) if i >= start]}
    if "plan" in u:
        new["plan"] = u["plan"]
    new["questionnaire"] = {k: v for k, v in u["questionnaire"].items() if k in keep_items}
    vs = u.get("visible_state", {})
    new["visible_state"] = {
        "pending": vs.get("pending", {}),
        "deferred": vs.get("deferred", []),
        "confirmed": {k: v for k, v in vs.get("confirmed", {}).items() if k.split("/")[0] in keep_items},
        "revision": vs.get("revision"),
    }
    return new, start

def turn_indices(obj):
    """정답 JSON 안의 모든 turn_index 수집 (잘라낸 대화 밖을 가리키는지 확인용)"""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "turn_index": out.append(v)
            else: out += turn_indices(v)
    elif isinstance(obj, list):
        for v in obj: out += turn_indices(v)
    return out

def convert(rows, task, slim):
    out = []
    for r in rows:
        sys_msg, user_msg = r["prompt"][0], r["prompt"][1]
        ans = r["completion"][0]["content"]
        u = json.loads(user_msg["content"])
        if slim:
            keep = {r["item_id"]}
            if task == "B" and u["plan"].get("next_item_id"):
                keep.add(u["plan"]["next_item_id"])
            u, start = slim_user(u, keep)
            if task == "A":
                bad = [t for t in turn_indices(json.loads(ans)) if t < start]
                assert not bad, (r["id"], bad, start)
        content = json.dumps(u, ensure_ascii=False, separators=(",", ":"))
        out.append({"id": r["id"], "category": r["category"],
                    "messages": [{"role": "system", "content": sys_msg["content"]},
                                 {"role": "user", "content": content},
                                 {"role": "assistant", "content": ans}]})
    return out

def split(rows):
    by = collections.defaultdict(list)
    for r in rows: by[r["category"]].append(r["id"])
    ev = set()
    for cat, ids in sorted(by.items()):
        if len(ids) >= 3:
            ev.update(random.sample(sorted(ids), EVAL_PER_CAT))
    return ev

A = load(DATA_DIR / "extraction.jsonl"); B = load(DATA_DIR / "response.jsonl")
eval_ids = split(A)   # A·B 같은 사례 id로 나눔
for slim in (False, True):
    d = Path("data_slim" if slim else "data_full"); d.mkdir(exist_ok=True)
    for task, rows in (("A", A), ("B", B)):
        conv = convert(rows, task, slim)
        tr = [x for x in conv if x["id"] not in eval_ids]
        ev = [x for x in conv if x["id"] in eval_ids]
        for name, data in (("train", tr), ("eval", ev)):
            with open(d / f"{task}_{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:   # OS 상관없이 같은 파일
                for x in data: f.write(json.dumps(x, ensure_ascii=False) + "\n")
        chars = [sum(len(m["content"]) for m in x["messages"]) for x in conv]
        print(f"{d.name} {task}: train {len(tr)} / eval {len(ev)} | 글자수 평균 {sum(chars)//len(chars)}, 최대 {max(chars)}")
print("eval ids:", sorted(eval_ids))

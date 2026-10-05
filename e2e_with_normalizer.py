"""LoRA-A 예측 → 민지 정규화(openit-normalizer) → DB에 저장될 결과를 정답과 비교.

실행 (openit-normalizer 폴더에서, dataset/ 준비된 상태):
  python <이 저장소>/e2e_with_normalizer.py [이 저장소 경로] [평가 결과 폴더]
  - 이 저장소 경로 기본값: 이 파일이 있는 폴더
  - 평가 결과 폴더 기본값: <이 저장소>/results  (finetuned_A.json, baseline_A.json)
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))   # openit-normalizer 폴더에서 실행
from normalizer.catalog import Catalog
from normalizer.normalize import normalize

FT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
RESULTS = Path(sys.argv[2]) if len(sys.argv) > 2 else FT / "results"
sys.path.insert(0, str(FT))
import types
for _m in ("huggingface_hub", "tokenizers"):
    sys.modules[_m] = types.SimpleNamespace(hf_hub_download=None, Tokenizer=None)   # 토큰 세기는 안 씀
from compact_formats import f3, restore_json   # noqa: E402

cat = Catalog()
rows = {json.loads(l)["id"]: json.loads(l) for l in (FT / "data_slim/A_eval.jsonl").read_text(encoding="utf-8").splitlines()}


def parse(t):
    return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip()))


def db_view(labels):
    """정규화 결과에서 DB에 남는 것만: (필드, 반복키) -> (상태, 값, 변경사유)"""
    r = normalize(labels, cat)
    return {(x.field_id, x.repeat_key): (x.resolution_status, x.value, x.change_reason) for x in r.records}


for name in ("finetuned_A.json", "baseline_A.json"):
    res = json.loads((RESULTS / name).read_text(encoding="utf-8"))
    same = 0
    print(f"\n===== {name}")
    for o in res["outputs"]:
        g = db_view(json.loads(o["gold"]))
        try:
            p = db_view(parse(o["pred"]))
        except Exception as e:
            p = {"오류": type(e).__name__}
        if g == p:
            same += 1
        elif name.startswith("finetuned"):
            print(f"\n[{o['id']} {rows[o['id']]['category']}]")
            for k in sorted(set(g) | set(p), key=str):
                if g.get(k) != p.get(k):
                    print(f"  {k}: 정답 {g.get(k)}  /  예측 {p.get(k)}")
    print(f"\n→ DB 저장 결과 일치: {same}/{len(res['outputs'])}")

# F3 축약 형식이 정규화에 바로 들어가도 되는지
print("\n===== F3 형식 호환성 (정답 100건)")
master = [json.loads(l) for l in (Path("dataset/data/master.jsonl")).read_text(encoding="utf-8").splitlines()]
direct = restored = 0
for m in master:
    ctx = m["context"]
    turns = {i: t["content"] for i, t in enumerate(ctx)}
    cur = len(ctx) - 1
    short = f3(m["labels"], cur)
    direct += db_view(short) == db_view(m["labels"])
    restored += db_view(restore_json(short, turns, cur)) == db_view(m["labels"])
print(f"F3를 그대로 넣었을 때 같은 결과: {direct}/100")
print(f"F3를 복원한 뒤 넣었을 때 같은 결과: {restored}/100")

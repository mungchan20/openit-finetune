"""LoRA-A/B 확인용 결과를 다빈 채점기(FTeval)로 채점 (확인용 사례만 골라 요약).

실행 (FTeval 폴더에서, dataset/ 준비된 상태):
  python <이 저장소>/score_with_fteval.py [평가 결과 폴더]
  - 평가 결과 폴더 기본값: <이 저장소>/results  (baseline_A/B.json, finetuned_A/B.json)
"""
import json, statistics, sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))   # FTeval 폴더에서 실행
from scorer import Evaluator

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results"
ev = Evaluator("config.json")
load = lambda n: {o["id"]: o["pred"] for o in json.loads((RESULTS / n).read_text(encoding="utf-8"))["outputs"]}
gold_a = {i: ev.data.row_a(i)["completion"][0]["content"] for i in ev.data.ids}
gold_b = {i: ev.data.row_b(i)["completion"][0]["content"] for i in ev.data.ids}

runs = {"정답(gold)": None, "학습 전(base)": ("baseline_A.json", "baseline_B.json"), "학습 후(ft)": ("finetuned_A.json", "finetuned_B.json")}
ids13 = list(load("finetuned_A.json"))
for name, files in runs.items():
    if files is None:
        oa = {i: gold_a[i] for i in ids13}; ob = {i: gold_b[i] for i in ids13}
    else:
        oa, ob = load(files[0]), load(files[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = ev.score("base", oa, ob)
    A = [s for s in r.a_scores if s["id"] in ids13]
    B = [s for s in r.b_scores if s["id"] in ids13]
    ins = [s for s in A if not s["gold_oos"]]
    print(f"\n[{name}]  A {len(A)}건 (규격 내 {len(ins)}) / B {len(B)}건")
    if not A:
        continue
    print(f"  A 스키마 통과      {sum(s['schema_ok'] and bool(s['values_ok']) for s in A)}/{len(A)}")
    print(f"  A 값 정확도(전부 일치) {sum(s['correct'] for s in ins)}/{len(ins)}")
    print(f"  A 핵심 일치(상태·정밀도 제외) {sum(s['core_correct'] for s in ins)}/{len(ins)}")
    print(f"  A 규격외(OOS) 판정  정답 OOS {sum(s['gold_oos'] for s in A)}건 중 맞힘 {sum(s['gold_oos'] and s['pred_oos'] for s in A)}, 잘못 OOS {sum(s['pred_oos'] and not s['gold_oos'] for s in A)}")
    print(f"  A 근거 전부 유효    {sum(bool(s['evidence_ok']) for s in A)}/{len(A)}")
    print(f"  B FM 평균 {statistics.mean(s['FM'] for s in B):.3f} / AM 평균 {statistics.mean(s['AM'] for s in B):.3f} / 완료 주장 위반 {sum(bool(s['completion_claim']) for s in B)}")
print("\n코퍼스 문장 수:", r.b_summary.get("corpus_size"), "| 최소 기준 미달:", r.b_summary.get("corpus_below_minimum"))

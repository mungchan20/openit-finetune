"""LoRA-B 평가 — 데이터셋 v2 형식 (4차 보고서, dataset_10_14).

입력(user) = {action, reason?, topic, user_utterance, known?, ask?, previous?, current?, hold?, note?, next{type, question_id, text?}}
출력(assistant) = 응답 문장 그대로 (평문, JSON 아님)

사용법:
  python eval_lora_b_v2.py --model Qwen/Qwen3-8B --adapter runs/b_8b_1014/adapter --out runs/b_8b_1014/eval_test.json
  python eval_lora_b_v2.py --model Qwen/Qwen3-8B --adapter none --limit 100 --out runs/b_8b_1014/eval_base.json   # 원본 비교

채점 항목 (4차 보고서 B 규칙을 코드로 검사한 것, 사람 검수 아님):
  ends_ok      next.type이 NEXT·SAME이면 질문으로 끝남 / END면 질문 없이 끝남
  asks_next    NEXT일 때 next.text의 핵심어를 응답에 담았는가 (다음 질문을 실제로 했는가)
  new_number   입력에 없는 숫자를 말했는가 (값 지어내기 의심)
  banned       금지 표현: 느낌표, 진단·처방·조언, 보고체·완료 주장("저장했/기록했/확정했")
  sentences    문장 수 1~3개(4차 보고서 B 규칙)를 벗어났는가
  json_leak    평문이어야 하는데 JSON/중괄호를 출력했는가
  len_tok      출력 토큰 수 (응답 시간과 직결)
"""
import argparse, json, re, time
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="/home/openit/toy dataset/dataset_10_14")
ap.add_argument("--split", default="test", choices=["val", "test"])
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--adapter", default="runs/b_8b_1014/adapter", help="'none'이면 원본 모델")
ap.add_argument("--out", default="runs/b_8b_1014/eval_test.json")
ap.add_argument("--max_new", type=int, default=160)
ap.add_argument("--limit", type=int, default=0, help="앞에서 N건만 (0=전체)")
ap.add_argument("--gold_only", action="store_true", help="모델 없이 정답 문장에 규칙만 적용 (규칙 점검용, GPU 불필요)")
args = ap.parse_args()

if not args.gold_only:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
    if args.adapter != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

rows = [json.loads(l) for l in (Path(args.data_dir) / f"lora_b_{args.split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
if args.limit:
    rows = rows[:args.limit]


def generate(msgs):
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    t = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False)
    gen = out[0][ids.input_ids.shape[1]:]
    text = tok.decode(gen, skip_special_tokens=True)
    return re.sub(r"^\s*<think>.*?</think>", "", text, flags=re.S).strip(), time.time() - t, len(gen)


DIG = {"영": 0, "공": 0, "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
UNIT = {"십": 10, "백": 100, "천": 1000}
KNUM = re.compile(r"[영공일이삼사오육칠팔구십백천]+")


def ko2num(m):
    """한글 숫자(육십, 이십삼, 백이십) → 아라비아. 십·백·천이 없으면 '일주일', '이번' 같은 일반 단어일 수 있어 그대로 둠"""
    w = m.group(0)
    if not any(c in UNIT for c in w):
        return w
    total = cur = 0
    for c in w:
        if c in DIG:
            cur = DIG[c]
        else:
            total += (cur or 1) * UNIT[c]; cur = 0
    return str(total + cur)


NATIVE = {"열": "10", "스물": "20", "스무": "20", "서른": "30", "마흔": "40", "쉰": "50"}
NATIVE_RE = re.compile(r"(?<![가-힣])(열|스물|스무|서른|마흔|쉰)(?=\s?(?:개|살|시|분|명|잔|병|갑|년)|[^가-힣]|$)")
# 한 글자 한자어 숫자 + 단위 ("이 년", "삼 개월"). 단위가 뒤에 와야 숫자로 봄
COUNTER_RE = re.compile(r"(?<![가-힣])([일이삼사오육칠팔구])(?=\s?(?:년|개월|살|세|시간|분|개비|개))")
SINO1 = {"일": "1", "이": "2", "삼": "3", "사": "4", "오": "5", "육": "6", "칠": "7", "팔": "8", "구": "9"}


def nums(s):
    s = NATIVE_RE.sub(lambda m: NATIVE[m.group(1)], s or "")
    s = COUNTER_RE.sub(lambda m: SINO1[m.group(1)], s)
    return set(re.findall(r"\d+(?:\.\d+)?", KNUM.sub(ko2num, s)))


# 의문형 어미 (~까요, ~나요, ~가요, ~죠, ~지요, ~습니까 ...)
QUESTION_END = re.compile(r"(\?|까요|나요|가요|죠|지요|니까|세요|주세요|하세요)\s*[.?]?\s*$")
BANNED = ("!", "진단해", "처방", "권장", "추천드", "조언", "위험합", "괜찮으실 거", "저장했", "기록했", "기록해 두었", "확정했", "완료했")


def keywords(text):
    """다음 질문 문장에서 2글자 이상 명사성 단어만 (조사 제거 대충)"""
    ws = re.findall(r"[가-힣A-Za-z0-9]{2,}", text or "")
    stop = {"있으신가요", "있으세요", "있나요", "하시나요", "드시나요", "인가요", "하세요", "주세요", "그리고", "지금", "현재", "혹시"}
    return [w for w in ws if w not in stop][:6]


per = defaultdict(lambda: defaultdict(float)); outs = []; secs = []
for r in rows:
    msgs, gold = r["messages"][:-1], r["messages"][-1]["content"]
    u = json.loads(msgs[1]["content"])
    action, ntype = u.get("action"), (u.get("next") or {}).get("type")
    if args.gold_only:
        pred, sec, n_tok = gold, 0.0, len(gold)   # 정답 점검 모드: 토큰 대신 글자 수
    else:
        pred, sec, n_tok = generate(msgs)
    secs.append(sec)
    sents = [s for s in re.split(r"(?<=[.?!])\s+", pred.strip()) if s]
    kw = keywords((u.get("next") or {}).get("text"))
    c = {
        "ends_ok": (bool(QUESTION_END.search(pred)) if ntype in ("NEXT", "SAME") else ("?" not in pred) if ntype == "END" else True),
        "asks_next": (sum(k in pred for k in kw) >= max(1, len(kw) // 3)) if ntype == "NEXT" and kw else True,
        "new_number": bool(nums(pred) - nums(msgs[1]["content"])),
        "banned": any(b in pred for b in BANNED),
        "sentences_bad": not (1 <= len(sents) <= 3),
        "json_leak": pred.lstrip().startswith("{"),
        "gold_banned": any(b in gold for b in BANNED),
        "len_tok": n_tok,
    }
    for key in (f"{action}/{ntype}", "(전체)"):
        per[key]["n"] += 1
        for k, v in c.items():
            per[key][k] += v
    outs.append({"scenario_id": r.get("scenario_id"), "action": action, "next_type": ntype,
                 "user_utterance": u.get("user_utterance"), "gold": gold, "pred": pred, "sec": round(sec, 3), **c})

print(f"\n[{args.split} {len(rows)}건 | model={args.model} | adapter={args.adapter}] 평균 생성 {sum(secs)/len(secs):.2f}s (HF generate, vLLM보다 느림)")
print(f"{'action/next':<22}{'n':>5}{'끝맺음OK':>9}{'다음질문':>9}{'새숫자':>7}{'금지표현':>9}{'문장수X':>8}{'JSON출력':>9}{'정답금지':>9}{'평균토큰':>9}")
for key in sorted(per, key=lambda k: (k != "(전체)", k)):
    m = per[key]; n = m["n"]
    print(f"{key:<22}{int(n):>5}{int(m['ends_ok']):>9}{int(m['asks_next']):>9}{int(m['new_number']):>7}{int(m['banned']):>9}"
          f"{int(m['sentences_bad']):>8}{int(m['json_leak']):>9}{int(m['gold_banned']):>9}{m['len_tok']/n:>9.0f}")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump({"split": args.split, "model": args.model, "adapter": args.adapter, "avg_sec": round(sum(secs)/len(secs), 3),
           "per_group": {k: dict(v) for k, v in per.items()}, "outputs": outs},
          open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"저장: {args.out}  (outputs에 정답·예측 문장이 다 있어 사람 검수용으로 볼 수 있음)")

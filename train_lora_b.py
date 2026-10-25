"""LoRA-B 학습 (TRL + PEFT, Unsloth 없음). 학교 서버 RTX PRO 6000 96GB용.

데이터: 팀장님 데이터셋 v2 (dataset_10_14) — lora_b_{train,val,test}.jsonl, 각 줄 {"scenario_id", "messages":[system,user,assistant]}
  user      = {action, known, next{question_id, text, type}, topic, user_utterance, ...} JSON 문자열
  assistant = 응답 문장 평문 (JSON 아님)
  (토이 데이터 v1은 user={question_id, normalization_result, ...}, assistant={"response_text": ...} 형식이었음)

사용법:
  python train_lora_b.py --dry_run                   # 토크나이저만: 길이·손실 구간 확인 (GPU 불필요)
  python train_lora_b.py                             # 학습 → 최저 Val Loss 체크포인트 → 어댑터 저장
  python train_lora_b.py --data_dir "/home/openit/toy dataset/lora_b" --out_dir runs/b_toy
  python train_lora_b.py --model Qwen/Qwen3-4B --out_dir runs/b_4b                       # 4B
  python train_lora_b.py --prefix lora_a --data_dir "/home/openit/toy dataset/lora_a" --out_dir runs/a_4b --model Qwen/Qwen3-4B   # LoRA-A에 사용

Loss: 정답(assistant) 토큰만 Causal LM 교차 엔트로피 (LoRA-B_목적함수_Loss_이명찬.md 참고)
출력(out_dir): adapter/ (최저 Val Loss), loss_log.json, loss_curve_b.png, run_config.json
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="/home/openit/toy dataset/lora_b")
ap.add_argument("--prefix", default="lora_b", help="파일 이름 앞부분: lora_b → lora_b_train.jsonl. LoRA-A는 --prefix lora_a")
ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
ap.add_argument("--out_dir", default="runs/b_toy")
ap.add_argument("--max_len", type=int, default=4096)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=int, default=16)
ap.add_argument("--dropout", type=float, default=0.0)
ap.add_argument("--eval_steps", type=int, default=10, help="몇 step마다 Val Loss를 잴지 (체크포인트도 같은 간격)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--dry_run", action="store_true")
args = ap.parse_args()
OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def load(split):
    p = Path(args.data_dir) / f"{args.prefix}_{split}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def encode(messages):
    """대화 → input_ids, labels. 마지막 assistant 답변만 손실 계산 (앞부분은 -100)"""
    kw = dict(tokenize=False, enable_thinking=False)
    full = tokenizer.apply_chat_template(messages, **kw)
    prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **kw)
    assert full.startswith(prompt), "템플릿이 프롬프트/전체에서 달라짐 → 모델 템플릿 확인 필요"
    p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    f_ids = tokenizer(full, add_special_tokens=False).input_ids
    assert f_ids[:len(p_ids)] == p_ids, "토큰 경계 불일치"
    return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": [-100] * len(p_ids) + f_ids[len(p_ids):]}


train = [encode(r["messages"]) for r in load("train")]
val = [encode(r["messages"]) for r in load("val")]
for name, rows in (("train", train), ("val", val)):
    lens = [len(x["input_ids"]) for x in rows]
    sup = [sum(t != -100 for t in x["labels"]) for x in rows]
    print(f"{name} {len(rows)}건 | 토큰 평균 {sum(lens)//len(lens)}, 최대 {max(lens)} | 정답 토큰 평균 {sum(sup)//len(sup)}")
    assert max(lens) <= args.max_len, "--max_len을 늘리세요"
if args.dry_run:
    print("\n[첫 샘플 손실 계산 구간]\n" + tokenizer.decode([t for t in train[0]["labels"] if t != -100]))
    raise SystemExit

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

try:
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")        # transformers 5
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")  # 옛 버전
model = get_peft_model(model, LoraConfig(
    r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.enable_input_require_grads()
model.print_trainable_parameters()


def collate(batch):
    """오른쪽 패딩 (labels 패딩은 -100 → 손실 제외)"""
    n = max(len(b["input_ids"]) for b in batch)
    pad = tokenizer.pad_token_id
    return {
        "input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
        "attention_mask": torch.tensor([b["attention_mask"] + [0] * (n - len(b["attention_mask"])) for b in batch]),
        "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
    }


# Val Loss를 eval_steps마다 재고, 같은 간격으로 체크포인트 저장 → 끝나면 Val Loss 최저 체크포인트를 불러옴
cfg_kw = dict(per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
              gradient_accumulation_steps=args.accum, num_train_epochs=args.epochs, learning_rate=args.lr,
              warmup_steps=3, lr_scheduler_type="linear", logging_steps=2, optim="adamw_torch", bf16=True,
              gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
              seed=args.seed, output_dir=str(OUT / "checkpoints"), report_to="none",
              save_strategy="steps", save_steps=args.eval_steps, save_total_limit=2,
              load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
              remove_unused_columns=False, dataset_kwargs={"skip_prepare_dataset": True})
def make_cfg():
    """TRL·transformers 버전마다 인자 이름이 달라서, 새 이름 → 옛 이름 순서로 시도"""
    last = None
    for drop in ([], ["dataset_kwargs"]):          # 최신 TRL은 토큰화된 데이터를 자동으로 건너뜀
        kw = {k: v for k, v in cfg_kw.items() if k not in drop}
        for extra in (dict(max_length=args.max_len, eval_strategy="steps"),
                      dict(max_seq_length=args.max_len, evaluation_strategy="steps")):
            try:
                return SFTConfig(eval_steps=args.eval_steps, **extra, **kw)
            except TypeError as e:
                last = e
    raise last
cfg = make_cfg()
ds_tr, ds_va = Dataset.from_list(train), Dataset.from_list(val)
try:
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds_tr, eval_dataset=ds_va,
                         data_collator=collate, processing_class=tokenizer)
except TypeError:
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds_tr, eval_dataset=ds_va,
                         data_collator=collate, tokenizer=tokenizer)

model.config.use_cache = False
print(trainer.train())
model.config.use_cache = True

best = trainer.state.best_model_checkpoint
model.save_pretrained(OUT / "adapter"); tokenizer.save_pretrained(OUT / "adapter")
print(f"어댑터 저장: {OUT/'adapter'} (최저 Val Loss 체크포인트: {best}, Val Loss {trainer.state.best_metric})")

# Loss 기록 + 곡선
hist = trainer.state.log_history
tr = [(h["step"], h["loss"]) for h in hist if "loss" in h]
va = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
json.dump({"train": tr, "val": va, "best_checkpoint": best, "best_val_loss": trainer.state.best_metric},
          open(OUT / "loss_log.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(vars(args), open(OUT / "run_config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(*zip(*tr), label="train loss")
    if va: plt.plot(*zip(*va), "o-", label="validation loss")
    tag = args.prefix.replace("lora_", "")
    plt.xlabel("step"); plt.ylabel("loss"); plt.title(f"LoRA-{tag.upper()} loss"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / f"loss_curve_{tag}.png", dpi=120)
    print(f"곡선 저장: {OUT/f'loss_curve_{tag}.png'}")
except ImportError:
    print("matplotlib 없음 → loss_log.json만 저장 (pip install matplotlib 후 다시 그리기)")

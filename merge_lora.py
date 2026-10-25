"""LoRA 어댑터를 원본 모델에 합쳐(병합) 하나의 모델로 저장.

미팅 3번 주제 "가중치 분리 vs 통합" 측정용. 병합한 모델은 vLLM에 어댑터 없이 바로 띄운다.
GPU 없이 CPU로도 된다 (4B bf16 기준 RAM 약 10GB).

사용법:
  python merge_lora.py --adapter ~/lora_A_4b --out ~/merged_A_4b
  python merge_lora.py --adapter ~/lora_B_4b --out ~/merged_B_4b
"""
import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen3-4B")
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cpu", help="cpu 또는 cuda. 4090(RAM 24GB)에서 8B는 cuda 권장 — vLLM을 끈 상태에서")
args = ap.parse_args()

try:
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map=args.device)        # transformers 5
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map=args.device)  # 옛 버전
model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()   # W' = W + (α/r)·BA 를 실제로 더함
model.save_pretrained(args.out)
AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
print(f"병합 모델 저장: {args.out}")

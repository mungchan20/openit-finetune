"""data_full / data_slim 프롬프트의 실제 토큰 수를 센다 (GPU 필요 없음).

  pip install tokenizers huggingface_hub
  python count_tokens.py                       # 기본: Qwen/Qwen3-1.7B 토크나이저
  python count_tokens.py --model kakaocorp/kanana-1.5-8b-instruct-2505

- 입력(prompt) = system + user + 응답 시작 표시까지  → vLLM이 읽어야 하는 양 (TTFT에 영향)
- 정답(answer) = assistant 내용                       → 모델이 생성해야 하는 양 (전체 시간에 영향)
- 채팅 형식은 Qwen3(ChatML, 생각 모드 끔) 기준으로 직접 붙인다.
  다른 모델은 채팅 형식 토큰이 몇 개 다를 수 있지만 내용 토큰 수는 같은 방식으로 비교 가능.
"""
import argparse, json, statistics
from pathlib import Path
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


def chatml(messages):
    """Qwen3 채팅 형식. 마지막 assistant 앞에 빈 생각 블록(enable_thinking=False)"""
    prompt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages[:-1])
    prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    answer = messages[-1]["content"] + "<|im_end|>\n"
    return prompt, answer


def stats(xs):
    return f"평균 {statistics.mean(xs):>6.0f}  중앙 {statistics.median(xs):>6.0f}  최소 {min(xs):>6}  최대 {max(xs):>6}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    args = ap.parse_args()
    tok = Tokenizer.from_file(hf_hub_download(args.model, "tokenizer.json"))
    n = lambda s: len(tok.encode(s, add_special_tokens=False).ids)

    summary = {}
    for d in ("data_full", "data_slim"):
        for task in ("A", "B"):
            rows = []
            for split in ("train", "eval"):
                rows += [json.loads(l) for l in Path(f"{d}/{task}_{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
            p, a, chars = [], [], []
            for r in rows:
                prompt, answer = chatml(r["messages"])
                p.append(n(prompt)); a.append(n(answer))
                chars.append(sum(len(m["content"]) for m in r["messages"][:-1]))
            summary[(d, task)] = statistics.mean(p)
            print(f"\n[{d} / LoRA-{task}]  {len(rows)}건  (모델: {args.model})")
            print(f"  입력 토큰 : {stats(p)}")
            print(f"  정답 토큰 : {stats(a)}")
            print(f"  학습 길이 : 입력+정답 최대 {max(x + y for x, y in zip(p, a))} 토큰")
            print(f"  입력 글자 : 평균 {statistics.mean(chars):.0f}자 → 글자당 {statistics.mean(p) / statistics.mean(chars):.2f} 토큰")

    print("\n===== slim이 full보다 얼마나 짧은가 (입력 토큰 평균) =====")
    for task in ("A", "B"):
        f, s = summary[("data_full", task)], summary[("data_slim", task)]
        print(f"  LoRA-{task}: full {f:.0f} → slim {s:.0f}  ({f / s:.1f}배 짧음)")


if __name__ == "__main__":
    main()

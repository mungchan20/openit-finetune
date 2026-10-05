"""vLLM 응답 시간 측정 (4090 서버에서 실행).

data_slim eval 프롬프트를 vLLM(OpenAI 호환 API)에 보내서
LoRA-A / LoRA-B / 원본(어댑터 없음) 각각의 시간을 잰다.

  - TTFT  : 요청 → 첫 글자가 나올 때까지 (TTS가 말을 시작할 수 있는 시점)
  - total : 요청 → 응답 끝까지
  - 토큰  : 입력(prompt) / 출력(completion) 토큰 수

표준 라이브러리만 사용 (설치 필요 없음).

사용법:
  python3 bench.py                         # 원본 + lora_A + lora_B (어댑터를 붙여 띄웠을 때)
  python3 bench.py --only base             # 어댑터 없이 원본만 (작업 2: 기준 시간)
  python3 bench.py --data data_full --runs 1
  python3 bench.py --save result.json      # 결과를 파일로도 저장
"""
import argparse, json, statistics, time, urllib.request
from pathlib import Path


def stream_chat(url, model, messages, max_tokens):
    """스트리밍으로 요청하고 (ttft, total, prompt_tokens, completion_tokens, text) 반환"""
    body = {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},   # Qwen3 생각 모드 끔
    }
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, text, usage = None, [], {}
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text.append(piece)
    total = time.perf_counter() - t0
    return (ttft if ttft is not None else total, total,
            usage.get("prompt_tokens"), usage.get("completion_tokens"), "".join(text))


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def summarize(xs):
    if not xs:
        return "-"
    return f"평균 {statistics.mean(xs):.3f}s  중앙 {statistics.median(xs):.3f}s  최소 {min(xs):.3f}s  최대 {max(xs):.3f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B", help="vLLM --model 이름")
    ap.add_argument("--data", default="data_slim")
    ap.add_argument("--runs", type=int, default=3, help="각 프롬프트 반복 횟수")
    ap.add_argument("--only", choices=["base", "lora"], help="base=원본만, lora=어댑터만")
    ap.add_argument("--max-a", type=int, default=512)
    ap.add_argument("--max-b", type=int, default=128)
    ap.add_argument("--save", help="결과 JSON 저장 경로")
    args = ap.parse_args()

    A = load(f"{args.data}/A_eval.jsonl")
    B = load(f"{args.data}/B_eval.jsonl")

    # (이름, 모델, 데이터, max_tokens)
    targets = []
    if args.only != "base":
        targets += [("LoRA-A", "lora_A", A, args.max_a), ("LoRA-B", "lora_B", B, args.max_b)]
    if args.only != "lora":
        targets += [("원본(B 프롬프트)", args.base, B, args.max_b)]

    # 예열: 첫 호출은 느리므로 측정에서 뺀다
    for name, model, rows, _ in targets:
        stream_chat(args.url, model, rows[0]["messages"][:-1], 16)
    print(f"예열 완료. {len(A)}건 × {args.runs}회 측정 시작 ({args.data})\n")

    results = {name: [] for name, *_ in targets}
    json_ok = {"ok": 0, "all": 0}
    for i in range(len(B)):
        line = [A[i]["id"] if i < len(A) else B[i]["id"]]
        for name, model, rows, max_tok in targets:
            msgs = rows[i]["messages"][:-1]
            for _ in range(args.runs):
                ttft, total, pt, ct, text = stream_chat(args.url, model, msgs, max_tok)
                results[name].append({"id": rows[i]["id"], "ttft": ttft, "total": total,
                                      "prompt_tokens": pt, "completion_tokens": ct, "text": text})
            if name == "LoRA-A":   # A 출력이 JSON으로 읽히는지도 확인 (마지막 반복 기준)
                json_ok["all"] += 1
                try:
                    json.loads(text); json_ok["ok"] += 1
                except ValueError:
                    pass
            r = results[name][-args.runs:]
            line.append(f"{name} {statistics.mean(x['total'] for x in r):.3f}s")
        print("  ".join(line))

    print("\n===== 요약 (total = 응답 끝까지, TTFT = 첫 글자까지) =====")
    for name in results:
        rs = results[name]
        pt = [x["prompt_tokens"] for x in rs if x["prompt_tokens"]]
        ct = [x["completion_tokens"] for x in rs if x["completion_tokens"]]
        tps = [x["completion_tokens"] / x["total"] for x in rs if x["completion_tokens"]]
        print(f"\n[{name}]")
        print(f"  total : {summarize([x['total'] for x in rs])}")
        print(f"  TTFT  : {summarize([x['ttft'] for x in rs])}")
        if pt and ct:
            print(f"  토큰  : 입력 평균 {statistics.mean(pt):.0f} / 출력 평균 {statistics.mean(ct):.0f}"
                  f"  | 생성 속도 평균 {statistics.mean(tps):.0f} tok/s")

    if "LoRA-A" in results and "LoRA-B" in results:
        a = statistics.mean(x["total"] for x in results["LoRA-A"])
        b = statistics.mean(x["total"] for x in results["LoRA-B"])
        print(f"\nA+B 합계 평균 {a + b:.3f}s  →  1초 기준 {'충족' if a + b <= 1.0 else '초과'}")
        print(f"LoRA-A JSON 파싱 성공 {json_ok['ok']}/{json_ok['all']}")

    if args.save:
        Path(args.save).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n저장: {args.save}")


if __name__ == "__main__":
    main()

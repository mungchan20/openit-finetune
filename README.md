# openit-finetune

건강문진 AI 에이전트 토이 프로젝트의 **파인튜닝·서빙** 코드입니다.
원본 모델 1개에 LoRA 어댑터 2개를 붙여 역할별로 골라 씁니다.

- **LoRA-A (정보 추출)**: 사용자 발화 → `facts / intents / relations / unmapped_facts` JSON
- **LoRA-B (응답 생성)**: 코드가 정한 plan → 다음 질문 한국어 문장

담당: 이명찬, 구태현 (데이터: 왕우석 `labeled_questionnaire_100_handoff`)

## 준비
`dataset/` 폴더에 우석님 데이터 패키지를 넣습니다 (Git에는 올리지 않음).
```
dataset/data/extraction.jsonl, response.jsonl, master.jsonl ...
dataset/db/...
```

```bash
python prepare_data.py          # → data_full/, data_slim/ (train 87 / eval 13)
```
- 분할: 카테고리별 1건씩(3건 이상인 카테고리만) eval, seed 42
- **slim**: 설문 정의는 현재 문항만(B는 다음 문항 추가), 대화는 최근 8턴만 → 입력 약 11분의 1
- 같은 데이터면 어느 PC에서 돌려도 같은 파일이 나옵니다

## 학습 (Colab T4)
1. `data_slim/`, `data_full/`과 이 폴더의 파일들을 `toy_finetune.zip`으로 묶기
2. Colab에서 `toy_lora_colab.ipynb` 열기 → 런타임 T4 GPU
3. ①~⑥ 순서로 실행 (`TASK="A"` → 세션 다시 시작 → `TASK="B"`)
   - ③-2: 학습 전 원본 점수 → `baseline_{A,B}.json`
   - ⑤: 학습 후 점수 + 비교표 → `finetuned_{A,B}.json`
   - ⑥: 어댑터 `lora_{A,B}.zip` 다운로드 (병합·GGUF 변환 없이 LoRA 그대로)
4. ⑦: vLLM 서버를 띄우고 `bench.py`로 시간 측정

설정: `unsloth/Qwen3-1.7B` 16bit, LoRA r=16 / α=16, q·k·v·o·gate·up·down, batch 2 × 누적 4, 3 epoch, lr 2e-4, 정답 부분만 손실, 생각 모드 끔

## 서빙·측정 (vLLM)
```bash
vllm serve Qwen/Qwen3-1.7B --max-model-len 4096 --enable-prefix-caching \
  --enable-lora --max-lora-rank 16 --max-loras 2 \
  --lora-modules lora_A=./lora_A lora_B=./lora_B --port 8000
python bench.py --only base          # 어댑터 없이 원본만
python bench.py --save bench.json    # A·B·원본, TTFT·전체 시간·토큰 수
```
- 요청할 때 `"model": "lora_A"` / `"lora_B"`로 어댑터 선택
- Colab 노트북 안에서 `LLM(...)`으로 직접 띄우면 `io.UnsupportedOperation: fileno` 에러 → 서버 방식 사용
- 운영 시 vLLM은 `127.0.0.1`에만 열고 FastAPI 뒤에 둡니다 (`--api-key`는 `/v1`만 보호)

## 파일
| 파일 | 역할 |
|---|---|
| `prepare_data.py` | 데이터 → 학습용 messages 형식 (full / slim), 분할 |
| `toy_lora_colab.ipynb` | 학습 → 평가 → 저장 → vLLM 측정 |
| `bench.py` | vLLM 응답 시간 측정 (표준 라이브러리만) |
| `count_tokens.py` | 프롬프트·정답 토큰 수 실측 |
| `compact_formats.py` | A 출력 축약 형식 비교 + 원래 JSON으로 복원 |
| `compare_metrics.py` | 사실·값·상태·근거·의도·정정 항목별 지표 (태현님 보고서 기준) |
| `e2e_with_normalizer.py` | LoRA-A 예측 → 민지님 [openit-normalizer](https://github.com/godmj/openit-normalizer) → DB 저장 결과 비교 |
| `score_with_fteval.py` | 다빈님 [FTeval](https://github.com/Sdabin1209/FTeval) 채점기로 채점 |

평가 결과 파일(`baseline_*.json`, `finetuned_*.json`)은 `results/`에 두면 연결 스크립트들이 읽습니다.
```bash
cd ../openit-normalizer && python ../openit-finetune/e2e_with_normalizer.py
cd ../FTeval && python ../openit-finetune/score_with_fteval.py
```

## 토이 결과 (Qwen3-1.7B, eval 13건)
⚠️ 100건이 같은 문장 틀의 합성 데이터라 **동작 확인용**이며 성능 지표가 아닙니다.

**LoRA-A**
| 지표 | 학습 전 | 학습 후 |
|---|---|---|
| JSON 형식 | 6/13 | 13/13 |
| 완전일치 (위치 숫자 제외) | 0/13 | 8/13 |
| P / R / F1 (상태 포함) | 0.05 / 0.06 / 0.05 | 0.75 / 0.67 / 0.71 |
| 근거 위치 숫자 정확 | — | 0/14 → **코드가 계산하면 9/14** |
| 민지님 정규화 후 DB 저장 결과 일치 | 0/13 | 8/13 |
| 다빈님 채점기 값 정확도 | 0/12 | 8/12 |

- 틀린 5건 중 4건은 정규화에서 재질문·미기록으로 끝남 (안전한 실패). **정정 사례만 틀린 값이 저장됨**
- 정정·거절·여러 필드 문항처럼 학습 데이터에 적은 유형이 약함

**LoRA-B**: 다음 문항 정확 12/13, 필요한 조건 안내까지 완전 6/13

**응답 시간 (Colab T4, vLLM 0.30)**
| | 출력 토큰 | 첫 글자까지 | 전체 | 생성 속도 |
|---|---|---|---|---|
| 원본 (어댑터 없음) | 51 | 0.12s | 1.05s | 49 tok/s |
| LoRA-A | 97 | 0.16s | 4.77s | 20 tok/s |
| LoRA-B | 29 | 0.22s | 1.58s | 19 tok/s |
| A+B | | | **6.35s** | |

- 시간의 약 75%가 **A의 JSON 출력**. 입력 처리(첫 글자까지)는 slim 덕분에 짧음
- 어댑터를 붙이면 생성 속도 약 2.5배 저하 (T4 기준, 4090 확인 필요)

**A 출력 축약 형식** (`compact_formats.py`, 정답 100건, 모두 원래 JSON으로 100/100 복원)
| 형식 | 평균 토큰 | 감소 |
|---|---|---|
| 원본 | 115 | — |
| F3: 위치 제거 + 빈 목록·기본값(exact, ANSWERED, 현재 턴) 생략 | 64 | 44% |
| F5: 한 줄 형식 `문항\|필드\|값\|상태\|정밀도\|근거` | 41 | 65% |

## 팀 확인이 필요한 사항
1. **slim 프롬프트**를 팀 표준 입력으로 할지 (다빈님 채점기는 원본 입력 기준)
2. **A 출력 형식**: 근거 위치(start/end)는 코드가 계산 / F3 채택 시 정규화에 넣기 전 복원 단계 또는 정규화 기본값 처리 / 범위 값 형식 함께 결정
3. **평가 데이터 분리**: 학습에 안 쓴 문항으로 나누는 방식(태현님 방식)으로 통일
4. 4090 워크스테이션: DNS 권한 문제로 pip·모델 다운로드 불가 (소장님 확인 필요), 8B급은 WSL 메모리(약 15GB) 상향 필요

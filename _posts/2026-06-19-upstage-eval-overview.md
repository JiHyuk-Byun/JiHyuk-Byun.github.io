---
title: "Upstage Products Evaluation — 종합 후기"
date: 2026-06-19
slug: upstage-eval-overview
tags: [upstage, document-ai, overview]
categories: [experiments]
layout: post
hidden: true
---

<!-- 아래 블로그 링크는 발행(blogify) 후 각 글의 실제 URL로 확인/교체하세요. -->

업스테이지 제품을 단순히 말로 평가하기보다는, 업스테이지의 대표 제품인**Document Parse · Information Extract · Solar-pro3**를 제가 실제로 가진 문서들에 대해, 통제된 실험을 통해 모든 의견에 대해 실험근거를 바탕으로 각 모델들의 성능을 확인 및 정리하였습니다.

평가는 두 단계로 구성했습니다. 먼저 **정답이 통제된 입력으로 각 모델의 기본 능력을 확인**(Model Test)하고,
이어서 **여러 제품을 한 파이프라인으로 엮어 현실 문서 과제를 끝까지 수행**(In-the-wild)시키며 한계를 봤습니다.
모든 실험은 정답(GT)·기존 도구 baseline·정량 지표·응답 캐싱으로 재현 가능하게 만들었고, 코드와 데이터셋은
각 글의 Appendix에 정리했습니다. 각 글은 **실험 방법 → Result(사실만) → Finding(제 평가)** 순서입니다.

## 1. 통제 능력 검증 (Model Test)
*정답이 통제된 입력으로 제품의 기본 능력만 확인합니다.*

**Model Test 1 — Document Parse**
→ https://jihyuk-byun.github.io/blog/2026/upstage-eval-doc-parse/
- 요소 **검출(localization)은 정확**(주요 시각요소 6/6)하나, **차트는 값만 남기고 계열·종류·계층을 잃습니다**(4종 모두 붕괴). 표 구조화는 거의 완벽합니다.

**Model Test 2 — Information Extract**
→ https://jihyuk-byun.github.io/blog/2026/upstage-eval-info-extract/
- 스키마만으로 이질 문서를 **zero-shot 추출**합니다. **틀린 스키마를 줘도 없는 값을 지어내지 않고**, 얼룩에 가려진 필드는 기본적으로 추측하지만 **"안 보이면 비워라" 지시 한 줄로 정직하게 비웁니다**.

## 2. 실전 적용 (In-the-wild) — with Solar-pro3
*문서 모델(Document Parse · Information Extract)을 `Solar-pro3`와 함께 엮어, 추출과 추론·생성을 한
파이프라인으로 묶어 현실 과제를 끝까지 수행시켰습니다.*

**In-the-wild 1 — 표 기반 논문 리뷰 (Document Parse × Solar-pro3)**
→ https://jihyuk-byun.github.io/blog/2026/upstage-eval-paper-review/
- 47쪽 논문을 Parse로 구조화해 Solar-pro3가 약점 진단 리포트를 작성합니다. **인용한 수치는 100% 정확**(grounding 25/25)한데, **그 수치로 내리는 추론은 환각**합니다 — 값만 대조하는 검증으로는 못 잡고, 추론 타당성 검증이 따로 필요하다는 점이 핵심입니다.

**In-the-wild 2 — 시험지 자동 채점기 (Information Extract × Solar-pro3)**
→ https://jihyuk-byun.github.io/blog/2026/upstage-eval-exam-grader/
- 손으로 푼 답안지(사진)를 IE로 읽어 채점합니다. **`enhanced` 모드가 읽기를 좌우**하며(default 16 → enhanced 60/63) 거의 동작합니다(F1 0.992). 유일한 실패는 "보기에 직접 친 동그라미"처럼 지시가 덜 커버한 마킹 스타일이었습니다.

## 3. 종합 의견
*문장 끝 괄호는 그 판단의 근거가 된 실험입니다.*

- **강점 (모델별)**
	**Document Parse**
	- 병합(rowspan/colspan)까지 보존한 **표 구조화가 거의 완벽**하고, 요소 **검출(localization)도 정확**합니다. (Model Test 1 · In-the-wild 1)
	**Information Extract**
	- 학습 없이 **스키마만으로 이질 문서를 zero-shot 추출**하고, **틀린 스키마에도 환각하지 않습니다**. (Model Test 2)
	- 특히 **`enhanced` 모드의 손글씨 추출이 매우 견고**합니다. (In-the-wild 2)
	**Solar-pro3**
	- 구조화된 표가 주어지면 **수치 인용이 100% 정확**(grounding)합니다. (In-the-wild 1)
	- *(공통)* 세 모델 모두 **환각에 보수적**입니다. (Model Test 2 · In-the-wild 1)

- **한계 (모델별)**
	**Document Parse**
	- 차트의 **visual hierarchy**(계열·차트 종류·legend)를 복원하지 못해, 다중계열 차트를 단일 행(`item_01`)으로 뭉갭니다. (Model Test 1)
	- 그림(figure)은 **검출만** 하고 구조화하지 않습니다(이미지 placeholder + 내부 텍스트 OCR). (Model Test 1)
	- **캡션 검출이 비일관적**입니다(표 캡션은 잡지만 도식 캡션은 본문으로 분류). (Model Test 1)
	**Information Extract**
	- 가려진/훼손된 필드를 **지시가 없으면 그럴듯한 값으로 추측**합니다(안 보이는 할부개월을 `0`으로). (Model Test 2)
	- 작은 한글·인접 필드에서 **오독**이 납니다(seat↔gate 스왑, 카드사명 오독). (Model Test 2 · In-the-wild 2)
	- **단일 스키마 description**으로는 다양한 마킹 스타일을 다 덮지 못합니다(보기에 직접 친 동그라미 미인식). (In-the-wild 2)
	- `temperature` 미지원으로 출력이 **비결정적**입니다. (Model Test 2)
	**Solar-pro3**
	- 표 수치는 정확히 인용하지만, **그 수치 위의 reasoning은 환각**이 일부 있습니다(약점 오선정·근거 오인용·수치 의미 날조). (In-the-wild 1)
	- 같은 표·구절을 반복하는 등 **verbosity**가 있습니다. (In-the-wild 1)


> Result는 관찰한 사실만, Finding은 제 분석·의견입니다. 
> 산출물: `results/`, `figures/`, 실험별 노트북 `notebooks/`.

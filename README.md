# HandsOnLLM-Korean

*Hands-On Large Language Models*(Jay Alammar·Maarten Grootendorst, O'Reilly, 2024)의 공식 실습 노트북을 **Amazon SageMaker AI Studio(JupyterLab)** 에서 실행되도록 고치고, 한국어 설명과 보충 실험을 붙인 스터디용 저장소입니다.

- 원본 저장소: [HandsOnLLM/Hands-On-Large-Language-Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) (Apache License 2.0)
- 원본은 2024년 중반 라이브러리 버전(transformers 4.41, LangChain 0.2, TRL 0.9 등)에 맞춰져 있습니다. 이 저장소는 현재 버전(transformers 5.x, numpy 2.x 등)에서 돌아가도록 수정했습니다.

## 장별 노트북

| 장 | 노트북 | 원본 | 권장 인스턴스 | 스토리지 |
|---|---|---|---|---|
| 1 | [Generating Your First Text](chapter01/ch01_first_text_sagemaker.ipynb) | [원본](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter01/Chapter%201%20-%20Introduction%20to%20Language%20Models.ipynb) | ml.g4dn.xlarge | 30GB |
| 2 | [Tokens and Embeddings](chapter02/ch02_tokens_embeddings_sagemaker.ipynb) | [원본](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter02/Chapter%202%20-%20Tokens%20and%20Token%20Embeddings.ipynb) | ml.g4dn.xlarge | 30GB |
| 3 | [Looking Inside Large Language Models](chapter03/ch03_looking_inside_llms_sagemaker.ipynb) | [원본](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter03/Chapter%203%20-%20Looking%20Inside%20LLMs.ipynb) | ml.g4dn.xlarge | 30GB |
| 4 | [Text Classification](chapter04/ch04_text_classification_sagemaker.ipynb) | [원본](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter04/Chapter%204%20-%20Text%20Classification.ipynb) | ml.g4dn.xlarge | 30GB |
| 5 | [Text Clustering and Topic Modeling](chapter05/ch05_clustering_topic_modeling_sagemaker.ipynb) | [원본](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter05/Chapter%205%20-%20Text%20Clustering%20and%20Topic%20Modeling.ipynb) | ml.g4dn.xlarge | 30GB |
| 6 | [Prompt Engineering](chapter06/ch06_prompt_engineering_sagemaker.ipynb) | [원본](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter06/Chapter%206%20-%20Prompt%20Engineering.ipynb) | ml.g4dn.xlarge | 40GB |
| 7 | [Advanced Text Generation Techniques and Tools](chapter07/ch07_advanced_generation_langchain_sagemaker.ipynb) | [원본](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter07/Chapter%207%20-%20Advanced%20Text%20Generation%20Techniques%20and%20Tools.ipynb) | ml.g4dn.xlarge | 50GB |
| 8 | [Semantic Search and Retrieval-Augmented Generation](chapter08/ch08_semantic_search_rag_sagemaker.ipynb) | [원본](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter08/Chapter%208%20-%20Semantic%20Search.ipynb) | ml.g4dn.xlarge | 40GB |
| 9 | [Multimodal Large Language Models](chapter09/ch09_multimodal_llms_sagemaker.ipynb) | [원본](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter09/Chapter%209%20-%20Multimodal%20Large%20Language%20Models.ipynb) | ml.g4dn.xlarge (메모리 부족 시 2xlarge) | 50GB |
| 10 | [Creating Text Embedding Models](chapter10/ch10_embedding_models_sagemaker.ipynb) | [원본](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter10/Chapter%2010%20-%20Creating%20Text%20Embedding%20Models.ipynb) | ml.g4dn.xlarge | 30GB |
| 11 | [Fine-Tuning Representation Models for Classification](chapter11/ch11_finetuning_representation_sagemaker.ipynb) | [원본](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter11/Chapter%2011%20-%20Fine-Tuning%20BERT.ipynb) | ml.g4dn.xlarge | 30GB |
| 12 | [Fine-Tuning Generation Models](chapter12/ch12_finetuning_generation_sagemaker.ipynb) | [원본](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter12/Chapter%2012%20-%20Fine-tuning%20Generation%20Models.ipynb) | ml.g4dn.xlarge | 30GB |

`chapter01/fig1-15_redrawn.svg`는 책 그림 1-15(어텐션을 붙인 RNN 인코더-디코더)를 스터디 Q&A에 맞춰 다시 그린 것입니다. 책 그림을 바탕으로 했으므로 저장소를 공개하지 마세요.

## 노트북 구성

모든 노트북은 같은 순서로 되어 있습니다.

1. **표지:** 절별로 하는 일과 책 위치(PDF 쪽) 표, 원본 노트북 링크
2. **0. 환경 점검:** GPU·디스크·라이브러리 버전 확인, 빠진 패키지만 설치(12장은 검증한 버전 세트를 고정 설치)
3. **본문:** 책의 절 순서대로 코드 + 짧은 설명 + "관찰 포인트"
4. **(+ 보충):** 책에 없는 실험(예: 한국어 토큰 수 비교, "bank" 단어의 층별 문맥화, CLIP 도로 장면 제로샷)
5. **책과 다를 수 있는 점:** 라이브러리 변경 때문에 고친 곳과 책 본문·코드가 서로 다른 곳
6. **정리:** 핵심 표, 자율주행·Agent와의 연결(책 밖 내용), space 종료 안내

## SageMaker AI Studio에서 실행하기

1. Studio → JupyterLab → **Create JupyterLab space**. 인스턴스 `ml.g4dn.xlarge`(NVIDIA T4 16GB, 책이 기준으로 삼은 Colab GPU와 같은 사양).
2. **스토리지는 처음부터 60GB로** 잡습니다. 기본값 5GB로는 1장의 Phi-3(약 7.6GB)도 받지 못합니다. 용량은 6시간에 한 번만 늘릴 수 있고 줄일 수 없습니다.
3. Run space → Open JupyterLab → 터미널에서 이 저장소를 받습니다. 비공개 저장소이므로 GitHub 개인 액세스 토큰(PAT, `repo` 읽기 권한)이 필요합니다.
   ```bash
   git clone https://github.com/<GitHub 사용자명>/HandsOnLLM-Korean.git
   ```
4. 장별 노트북을 열고 위에서부터 차례로 실행합니다.
5. 끝나면 space를 **Stop** 합니다. 인스턴스 요금은 멈추고, EBS 볼륨(모델 캐시)은 남아 다음 장에서 재사용됩니다.

GPU 쿼터가 0이면 인스턴스 선택 시 `ResourceLimitExceeded`가 납니다. Service Quotas → Amazon SageMaker에서 "JupyterLab"과 인스턴스명으로 검색해 증액을 요청하세요.

## 검증 범위

모든 노트북은 CPU 환경에서 작은 대체 모델과 일부 데이터로 위에서부터 끝까지 실행해 확인했습니다. 다음은 **실제 GPU에서 아직 실행하지 않았습니다.** 오류가 나면 이슈로 남겨 주세요.

- Phi-3·BLIP-2 등 큰 모델의 실제 생성
- bitsandbytes 4비트 양자화(12장), llama-cpp-python CUDA 빌드(6·7·8장)
- OpenAI·Cohere·Amazon Bedrock 호출(키·권한이 없으면 해당 셀은 건너뜀)

## 함께 쓰는 규칙

- 공용 노트북은 직접 고치지 말고 PR로 바꿉니다. 개인 실험은 사본(`ch02_..._이름.ipynb`)이나 개인 브랜치에서 합니다.
- 커밋 전에 실행 결과(output)를 지웁니다. API 키·Hugging Face 토큰이 출력에 남지 않게 하기 위해서입니다. [nbstripout](https://github.com/kynan/nbstripout)을 쓰면 자동으로 지워 줍니다(`pip install nbstripout && nbstripout --install`).
- 책 PDF는 올리지 않습니다. 책은 O'Reilly의 저작물입니다.
- 파일·폴더 이름은 영어(ASCII)로 둡니다. macOS 한글 파일명은 다른 OS에서 깨질 수 있습니다.

## 라이선스

원본 코드는 Apache License 2.0입니다([LICENSE](LICENSE)). 이 저장소의 노트북은 원본을 수정한 파생물이며, 수정 사실은 [NOTICE](NOTICE)와 각 노트북 첫 셀에 밝혀 두었습니다.

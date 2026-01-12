# 自然语言处理 cs.CL

- **最新发布 82 篇**

- **更新 58 篇**

## 最新发布

#### [new 001] What do the metrics mean? A critical analysis of the use of Automated Evaluation Metrics in Interpreting
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的质量评估任务，探讨自动评估指标在口译中的适用性，指出其无法考虑语境，故不能单独作为质量衡量标准。**

- **链接: [https://arxiv.org/pdf/2601.05864v1](https://arxiv.org/pdf/2601.05864v1)**

> **作者:** Jonathan Downie; Joss Moorkens
>
> **备注:** 25 pages
>
> **摘要:** With the growth of interpreting technologies, from remote interpreting and Computer-Aided Interpreting to automated speech translation and interpreting avatars, there is now a high demand for ways to quickly and efficiently measure the quality of any interpreting delivered. A range of approaches to fulfil the need for quick and efficient quality measurement have been proposed, each involving some measure of automation. This article examines these recently-proposed quality measurement methods and will discuss their suitability for measuring the quality of authentic interpreting practice, whether delivered by humans or machines, concluding that automatic metrics as currently proposed cannot take into account the communicative context and thus are not viable measures of the quality of any interpreting provision when used on their own. Across all attempts to measure or even categorise quality in Interpreting Studies, the contexts in which interpreting takes place have become fundamental to the final analysis.
>
---
#### [new 002] Can We Predict Before Executing Machine Learning Agents?
- **分类: cs.CL; cs.AI; cs.LG; cs.MA**

- **简介: 该论文属于机器学习任务，旨在解决执行瓶颈问题。通过预测代替物理执行，提升效率。工作包括构建数据集、验证LLM预测能力，并实现加速的代理系统。**

- **链接: [https://arxiv.org/pdf/2601.05930v1](https://arxiv.org/pdf/2601.05930v1)**

> **作者:** Jingsheng Zheng; Jintian Zhang; Yujie Luo; Yuren Mao; Yunjun Gao; Lun Du; Huajun Chen; Ningyu Zhang
>
> **备注:** Work in progress
>
> **摘要:** Autonomous machine learning agents have revolutionized scientific discovery, yet they remain constrained by a Generate-Execute-Feedback paradigm. Previous approaches suffer from a severe Execution Bottleneck, as hypothesis evaluation relies strictly on expensive physical execution. To bypass these physical constraints, we internalize execution priors to substitute costly runtime checks with instantaneous predictive reasoning, drawing inspiration from World Models. In this work, we formalize the task of Data-centric Solution Preference and construct a comprehensive corpus of 18,438 pairwise comparisons. We demonstrate that LLMs exhibit significant predictive capabilities when primed with a Verified Data Analysis Report, achieving 61.5% accuracy and robust confidence calibration. Finally, we instantiate this framework in FOREAGENT, an agent that employs a Predict-then-Verify loop, achieving a 6x acceleration in convergence while surpassing execution-based baselines by +6%. Our code and dataset will be publicly available soon at https://github.com/zjunlp/predict-before-execute.
>
---
#### [new 003] Simplify-This: A Comparative Analysis of Prompt-Based and Fine-Tuned LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于文本简化任务，比较了提示工程与微调大语言模型的效果，分析其在结构简化和语义相似性上的表现。**

- **链接: [https://arxiv.org/pdf/2601.05794v1](https://arxiv.org/pdf/2601.05794v1)**

> **作者:** Eilam Cohen; Itamar Bul; Danielle Inbar; Omri Loewenbach
>
> **摘要:** Large language models (LLMs) enable strong text generation, and in general there is a practical tradeoff between fine-tuning and prompt engineering. We introduce Simplify-This, a comparative study evaluating both paradigms for text simplification with encoder-decoder LLMs across multiple benchmarks, using a range of evaluation metrics. Fine-tuned models consistently deliver stronger structural simplification, whereas prompting often attains higher semantic similarity scores yet tends to copy inputs. A human evaluation favors fine-tuned outputs overall. We release code, a cleaned derivative dataset used in our study, checkpoints of fine-tuned models, and prompt templates to facilitate reproducibility and future work.
>
---
#### [new 004] ACR: Adaptive Context Refactoring via Context Refactoring Operators for Multi-Turn Dialogue
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多轮对话任务，解决模型在长对话中出现的上下文惯性和状态漂移问题。提出ACR框架，通过动态重构对话历史来提升模型一致性与准确性。**

- **链接: [https://arxiv.org/pdf/2601.05589v1](https://arxiv.org/pdf/2601.05589v1)**

> **作者:** Jiawei Shen; Jia Zhu; Hanghui Guo; Weijie Shi; Yue Cui; Qingyu Niu; Guoqing Ma; Yidan Liang; Jingjiang Liu; Yiling Wang; Shimin Di; Jiajie Xu
>
> **摘要:** Large Language Models (LLMs) have shown remarkable performance in multi-turn dialogue. However, in multi-turn dialogue, models still struggle to stay aligned with what has been established earlier, follow dependencies across many turns, and avoid drifting into incorrect facts as the interaction grows longer. Existing approaches primarily focus on extending the context window, introducing external memory, or applying context compression, yet these methods still face limitations such as \textbf{contextual inertia} and \textbf{state drift}. To address these challenges, we propose the \textbf{A}daptive \textbf{C}ontext \textbf{R}efactoring \textbf{(ACR)} Framework, which dynamically monitors and reshapes the interaction history to mitigate contextual inertia and state drift actively. ACR is built on a library of context refactoring operators and a teacher-guided self-evolving training paradigm that learns when to intervene and how to refactor, thereby decoupling context management from the reasoning process. Extensive experiments on multi-turn dialogue demonstrate that our method significantly outperforms existing baselines while reducing token consumption.
>
---
#### [new 005] Double: Breaking the Acceleration Limit via Double Retrieval Speculative Parallelism
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，旨在解决传统推测解码的加速瓶颈。提出Double框架，通过双检索推测并行技术提升速度，突破理论上限并减少计算浪费。**

- **链接: [https://arxiv.org/pdf/2601.05524v1](https://arxiv.org/pdf/2601.05524v1)**

> **作者:** Yuhao Shen; Tianyu Liu; Junyi Shen; Jinyang Wu; Quan Kong; Li Huan; Cong Wang
>
> **摘要:** Parallel Speculative Decoding (PSD) accelerates traditional Speculative Decoding (SD) by overlapping draft generation with verification. However, it remains hampered by two fundamental challenges: (1) a theoretical speedup ceiling dictated by the speed ratio between the draft and target models, and (2) high computational waste and pipeline stall due to mid-sequence token rejections of early errors. To address these limitations, we introduce \textsc{Double} (Double Retrieval Speculative Parallelism). By bridging the gap between SD and PSD, our framework resolves the Retrieval \emph{Precision-Efficiency Dilemma} through a novel synchronous mechanism. Specifically, we enable the draft model to execute iterative retrieval speculations to break the theoretical speedup limits; to alleviate rejections without rollback, the target model performs authoritative retrieval to generate multi-token guidance. \textsc{Double} is entirely training-free and lossless. Extensive experiments demonstrate state-of-the-art speedup of $\textbf{5.3}\times$ on LLaMA3.3-70B and $\textbf{2.8}\times$ on Qwen3-32B, significantly outperforming the advanced method EAGLE-3 that requires extensive model training.
>
---
#### [new 006] Can large language models interpret unstructured chat data on dynamic group decision-making processes? Evidence on joint destination choice
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决从群组聊天中解析决策过程的问题。研究使用大语言模型自动提取决策因素，对比分析其与人工标注的差异。**

- **链接: [https://arxiv.org/pdf/2601.05582v1](https://arxiv.org/pdf/2601.05582v1)**

> **作者:** Sung-Yoo Lim; Koki Sato; Kiyoshi Takami; Giancarlos Parady; Eui-Jin Kim
>
> **备注:** 23 pages, 9 figures
>
> **摘要:** Social activities result from complex joint activity-travel decisions between group members. While observing the decision-making process of these activities is difficult via traditional travel surveys, the advent of new types of data, such as unstructured chat data, can help shed some light on these complex processes. However, interpreting these decision-making processes requires inferring both explicit and implicit factors. This typically involves the labor-intensive task of manually annotating dialogues to capture context-dependent meanings shaped by the social and cultural norms. This study evaluates the potential of Large Language Models (LLMs) to automate and complement human annotation in interpreting decision-making processes from group chats, using data on joint eating-out activities in Japan as a case study. We designed a prompting framework inspired by the knowledge acquisition process, which sequentially extracts key decision-making factors, including the group-level restaurant choice set and outcome, individual preferences of each alternative, and the specific attributes driving those preferences. This structured process guides the LLM to interpret group chat data, converting unstructured dialogues into structured tabular data describing decision-making factors. To evaluate LLM-driven outputs, we conduct a quantitative analysis using a human-annotated ground truth dataset and a qualitative error analysis to examine model limitations. Results show that while the LLM reliably captures explicit decision-making factors, it struggles to identify nuanced implicit factors that human annotators readily identified. We pinpoint specific contexts when LLM-based extraction can be trusted versus when human oversight remains essential. These findings highlight both the potential and limitations of LLM-based analysis for incorporating non-traditional data sources on social activities.
>
---
#### [new 007] One Script Instead of Hundreds? On Pretraining Romanized Encoder Language Models
- **分类: cs.CL**

- **简介: 该论文研究预训练多语言模型时使用罗马化文本的效果，旨在解决高资源语言在罗马化过程中是否损失性能的问题。通过对比罗马化与原始文本的预训练效果，分析信息丢失和跨语言干扰的影响。**

- **链接: [https://arxiv.org/pdf/2601.05776v1](https://arxiv.org/pdf/2601.05776v1)**

> **作者:** Benedikt Ebing; Lennart Keller; Goran Glavaš
>
> **摘要:** Exposing latent lexical overlap, script romanization has emerged as an effective strategy for improving cross-lingual transfer (XLT) in multilingual language models (mLMs). Most prior work, however, focused on setups that favor romanization the most: (1) transfer from high-resource Latin-script to low-resource non-Latin-script languages and/or (2) between genealogically closely related languages with different scripts. It thus remains unclear whether romanization is a good representation choice for pretraining general-purpose mLMs, or, more precisely, if information loss associated with romanization harms performance for high-resource languages. We address this gap by pretraining encoder LMs from scratch on both romanized and original texts for six typologically diverse high-resource languages, investigating two potential sources of degradation: (i) loss of script-specific information and (ii) negative cross-lingual interference from increased vocabulary overlap. Using two romanizers with different fidelity profiles, we observe negligible performance loss for languages with segmental scripts, whereas languages with morphosyllabic scripts (Chinese and Japanese) suffer degradation that higher-fidelity romanization mitigates but cannot fully recover. Importantly, comparing monolingual LMs with their mLM counterpart, we find no evidence that increased subword overlap induces negative interference. We further show that romanization improves encoding efficiency (i.e., fertility) for segmental scripts at a negligible performance cost.
>
---
#### [new 008] A Framework for Personalized Persuasiveness Prediction via Context-Aware User Profiling
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于个性化说服力预测任务，旨在解决如何利用用户历史信息提升说服力预测效果的问题。提出一种上下文感知的用户画像框架，通过生成查询和总结记录来构建有效用户画像。**

- **链接: [https://arxiv.org/pdf/2601.05654v1](https://arxiv.org/pdf/2601.05654v1)**

> **作者:** Sejun Park; Yoonah Park; Jongwon Lim; Yohan Jo
>
> **摘要:** Estimating the persuasiveness of messages is critical in various applications, from recommender systems to safety assessment of LLMs. While it is imperative to consider the target persuadee's characteristics, such as their values, experiences, and reasoning styles, there is currently no established systematic framework to optimize leveraging a persuadee's past activities (e.g., conversations) to the benefit of a persuasiveness prediction model. To address this problem, we propose a context-aware user profiling framework with two trainable components: a query generator that generates optimal queries to retrieve persuasion-relevant records from a user's history, and a profiler that summarizes these records into a profile to effectively inform the persuasiveness prediction model. Our evaluation on the ChangeMyView Reddit dataset shows consistent improvements over existing methods across multiple predictor models, with gains of up to +13.77%p in F1 score. Further analysis shows that effective user profiles are context-dependent and predictor-specific, rather than relying on static attributes or surface-level similarity. Together, these results highlight the importance of task-oriented, context-dependent user profiling for personalized persuasiveness prediction.
>
---
#### [new 009] AutoMonitor-Bench: Evaluating the Reliability of LLM-Based Misbehavior Monitor
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于LLM安全监测任务，旨在评估基于大模型的异常行为监测器的可靠性。通过构建基准测试集，分析监测性能与误报率之间的权衡，探索提升监测效果的方法。**

- **链接: [https://arxiv.org/pdf/2601.05752v1](https://arxiv.org/pdf/2601.05752v1)**

> **作者:** Shu Yang; Jingyu Hu; Tong Li; Hanqi Yan; Wenxuan Wang; Di Wang
>
> **摘要:** We introduce AutoMonitor-Bench, the first benchmark designed to systematically evaluate the reliability of LLM-based misbehavior monitors across diverse tasks and failure modes. AutoMonitor-Bench consists of 3,010 carefully annotated test samples spanning question answering, code generation, and reasoning, with paired misbehavior and benign instances. We evaluate monitors using two complementary metrics: Miss Rate (MR) and False Alarm Rate (FAR), capturing failures to detect misbehavior and oversensitivity to benign behavior, respectively. Evaluating 12 proprietary and 10 open-source LLMs, we observe substantial variability in monitoring performance and a consistent trade-off between MR and FAR, revealing an inherent safety-utility tension. To further explore the limits of monitor reliability, we construct a large-scale training corpus of 153,581 samples and fine-tune Qwen3-4B-Instruction to investigate whether training on known, relatively easy-to-construct misbehavior datasets improves monitoring performance on unseen and more implicit misbehaviors. Our results highlight the challenges of reliable, scalable misbehavior monitoring and motivate future work on task-aware designing and training strategies for LLM-based monitors.
>
---
#### [new 010] Distilling Feedback into Memory-as-a-Tool
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在降低推理成本。通过将临时反馈转化为可检索指南，提升大模型效率。**

- **链接: [https://arxiv.org/pdf/2601.05960v1](https://arxiv.org/pdf/2601.05960v1)**

> **作者:** Víctor Gallego
>
> **备注:** Code: https://github.com/vicgalle/feedback-memory-as-a-tool Data: https://huggingface.co/datasets/vicgalle/rubric-feedback-bench
>
> **摘要:** We propose a framework that amortizes the cost of inference-time reasoning by converting transient critiques into retrievable guidelines, through a file-based memory system and agent-controlled tool calls. We evaluate this method on the Rubric Feedback Bench, a novel dataset for rubric-based learning. Experiments demonstrate that our augmented LLMs rapidly match the performance of test-time refinement pipelines while drastically reducing inference cost.
>
---
#### [new 011] Afri-MCQA: Multimodal Cultural Question Answering for African Languages
- **分类: cs.CL**

- **简介: 该论文提出Afri-MCQA，首个覆盖15种非洲语言的多模态文化问答基准，旨在解决非洲语言在AI研究中的代表性不足问题。**

- **链接: [https://arxiv.org/pdf/2601.05699v1](https://arxiv.org/pdf/2601.05699v1)**

> **作者:** Atnafu Lambebo Tonja; Srija Anand; Emilio Villa-Cueva; Israel Abebe Azime; Jesujoba Oluwadara Alabi; Muhidin A. Mohamed; Debela Desalegn Yadeta; Negasi Haile Abadi; Abigail Oppong; Nnaemeka Casmir Obiefuna; Idris Abdulmumin; Naome A Etori; Eric Peter Wairagala; Kanda Patrick Tshinu; Imanigirimbabazi Emmanuel; Gabofetswe Malema; Alham Fikri Aji; David Ifeoluwa Adelani; Thamar Solorio
>
> **摘要:** Africa is home to over one-third of the world's languages, yet remains underrepresented in AI research. We introduce Afri-MCQA, the first Multilingual Cultural Question-Answering benchmark covering 7.5k Q&A pairs across 15 African languages from 12 countries. The benchmark offers parallel English-African language Q&A pairs across text and speech modalities and was entirely created by native speakers. Benchmarking large language models (LLMs) on Afri-MCQA shows that open-weight models perform poorly across evaluated cultures, with near-zero accuracy on open-ended VQA when queried in native language or speech. To evaluate linguistic competence, we include control experiments meant to assess this specific aspect separate from cultural knowledge, and we observe significant performance gaps between native languages and English for both text and speech. These findings underscore the need for speech-first approaches, culturally grounded pretraining, and cross-lingual cultural transfer. To support more inclusive multimodal AI development in African languages, we release our Afri-MCQA under academic license or CC BY-NC 4.0 on HuggingFace (https://huggingface.co/datasets/Atnafu/Afri-MCQA)
>
---
#### [new 012] CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于机器翻译任务，解决数据顺序对模型训练的影响问题。通过引入带有重启的课程学习策略（CLewR），提升翻译性能。**

- **链接: [https://arxiv.org/pdf/2601.05858v1](https://arxiv.org/pdf/2601.05858v1)**

> **作者:** Alexandra Dragomir; Florin Brad; Radu Tudor Ionescu
>
> **摘要:** Large language models (LLMs) have demonstrated competitive performance in zero-shot multilingual machine translation (MT). Some follow-up works further improved MT performance via preference optimization, but they leave a key aspect largely underexplored: the order in which data samples are given during training. We address this topic by integrating curriculum learning into various state-of-the-art preference optimization algorithms to boost MT performance. We introduce a novel curriculum learning strategy with restarts (CLewR), which reiterates easy-to-hard curriculum multiple times during training to effectively mitigate the catastrophic forgetting of easy examples. We demonstrate consistent gains across several model families (Gemma2, Qwen2.5, Llama3.1) and preference optimization techniques. We publicly release our code at https://github.com/alexandra-dragomir/CLewR.
>
---
#### [new 013] Do LLMs Need Inherent Reasoning Before Reinforcement Learning? A Study in Korean Self-Correction
- **分类: cs.CL; cs.AI**

- **简介: 论文研究LLM在韩语中通过强化学习提升推理能力的可行性，解决低资源语言推理不足的问题。通过调整模型内部结构和引入数据集，提升韩语推理与自我修正能力。**

- **链接: [https://arxiv.org/pdf/2601.05459v1](https://arxiv.org/pdf/2601.05459v1)**

> **作者:** Hongjin Kim; Jaewook Lee; Kiyoung Lee; Jong-hun Shin; Soojong Lim; Oh-Woog Kwon
>
> **备注:** IJCNLP-AACL 2025 (Main), Outstanding Paper Award
>
> **摘要:** Large Language Models (LLMs) demonstrate strong reasoning and self-correction abilities in high-resource languages like English, but their performance remains limited in low-resource languages such as Korean. In this study, we investigate whether reinforcement learning (RL) can enhance Korean reasoning abilities to a degree comparable to English. Our findings reveal that RL alone yields limited improvements when applied to models lacking inherent Korean reasoning capabilities. To address this, we explore several fine-tuning strategies and show that aligning the model's internal reasoning processes with Korean inputs-particularly by tuning Korean-specific neurons in early layers-is key to unlocking RL's effectiveness. We introduce a self-correction code-switching dataset to facilitate this alignment and observe significant performance gains in both mathematical reasoning and self-correction tasks. Ultimately, we conclude that the crucial factor in multilingual reasoning enhancement is not injecting new linguistic knowledge, but effectively eliciting and aligning existing reasoning capabilities. Our study provides a new perspective on how internal translation and neuron-level tuning contribute to multilingual reasoning alignment in LLMs.
>
---
#### [new 014] Can Large Language Models Differentiate Harmful from Argumentative Essays? Steps Toward Ethical Essay Scoring
- **分类: cs.CL**

- **简介: 该论文属于伦理作文评分任务，旨在解决LLMs和AES系统无法准确识别有害内容的问题。研究构建了HED基准，测试模型区分有害与论辩性文章的能力，发现现有模型需改进以考虑伦理因素。**

- **链接: [https://arxiv.org/pdf/2601.05545v1](https://arxiv.org/pdf/2601.05545v1)**

> **作者:** Hongjin Kim; Jeonghyun Kang; Harksoo Kim
>
> **备注:** COLING 2025 accepted paper (Main)
>
> **摘要:** This study addresses critical gaps in Automated Essay Scoring (AES) systems and Large Language Models (LLMs) with regard to their ability to effectively identify and score harmful essays. Despite advancements in AES technology, current models often overlook ethically and morally problematic elements within essays, erroneously assigning high scores to essays that may propagate harmful opinions. In this study, we introduce the Harmful Essay Detection (HED) benchmark, which includes essays integrating sensitive topics such as racism and gender bias, to test the efficacy of various LLMs in recognizing and scoring harmful content. Our findings reveal that: (1) LLMs require further enhancement to accurately distinguish between harmful and argumentative essays, and (2) both current AES models and LLMs fail to consider the ethical dimensions of content during scoring. The study underscores the need for developing more robust AES systems that are sensitive to the ethical implications of the content they are scoring.
>
---
#### [new 015] AdaFuse: Adaptive Ensemble Decoding with Test-Time Scaling for LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AdaFuse，解决大语言模型推理时的集成问题。通过动态调整融合粒度，提升生成质量。属于自然语言处理任务。**

- **链接: [https://arxiv.org/pdf/2601.06022v1](https://arxiv.org/pdf/2601.06022v1)**

> **作者:** Chengming Cui; Tianxin Wei; Ziyi Chen; Ruizhong Qiu; Zhichen Zeng; Zhining Liu; Xuying Ning; Duo Zhou; Jingrui He
>
> **摘要:** Large language models (LLMs) exhibit complementary strengths arising from differences in pretraining data, model architectures, and decoding behaviors. Inference-time ensembling provides a practical way to combine these capabilities without retraining. However, existing ensemble approaches suffer from fundamental limitations. Most rely on fixed fusion granularity, which lacks the flexibility required for mid-generation adaptation and fails to adapt to different generation characteristics across tasks. To address these challenges, we propose AdaFuse, an adaptive ensemble decoding framework that dynamically selects semantically appropriate fusion units during generation. Rather than committing to a fixed granularity, AdaFuse adjusts fusion behavior on the fly based on the decoding context, with words serving as basic building blocks for alignment. To be specific, we introduce an uncertainty-based criterion to decide whether to apply ensembling at each decoding step. Under confident decoding states, the model continues generation directly. In less certain states, AdaFuse invokes a diversity-aware scaling strategy to explore alternative candidate continuations and inform ensemble decisions. This design establishes a synergistic interaction between adaptive ensembling and test-time scaling, where ensemble decisions guide targeted exploration, and the resulting diversity in turn strengthens ensemble quality. Experiments on open-domain question answering, arithmetic reasoning, and machine translation demonstrate that AdaFuse consistently outperforms strong ensemble baselines, achieving an average relative improvement of 6.88%. The code is available at https://github.com/CCM0111/AdaFuse.
>
---
#### [new 016] Semantic NLP Pipelines for Interoperable Patient Digital Twins from Unstructured EHRs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理在医疗信息领域的应用任务，旨在解决从非结构化EHR生成可互操作患者数字孪生的问题。通过NLP技术提取并标准化临床信息，提升数据互操作性。**

- **链接: [https://arxiv.org/pdf/2601.05847v1](https://arxiv.org/pdf/2601.05847v1)**

> **作者:** Rafael Brens; Yuqiao Meng; Luoxi Tang; Zhaohan Xi
>
> **摘要:** Digital twins -- virtual replicas of physical entities -- are gaining traction in healthcare for personalized monitoring, predictive modeling, and clinical decision support. However, generating interoperable patient digital twins from unstructured electronic health records (EHRs) remains challenging due to variability in clinical documentation and lack of standardized mappings. This paper presents a semantic NLP-driven pipeline that transforms free-text EHR notes into FHIR-compliant digital twin representations. The pipeline leverages named entity recognition (NER) to extract clinical concepts, concept normalization to map entities to SNOMED-CT or ICD-10, and relation extraction to capture structured associations between conditions, medications, and observations. Evaluation on MIMIC-IV Clinical Database Demo with validation against MIMIC-IV-on-FHIR reference mappings demonstrates high F1-scores for entity and relation extraction, with improved schema completeness and interoperability compared to baseline methods.
>
---
#### [new 017] FlashMem: Distilling Intrinsic Latent Memory via Computation Reuse
- **分类: cs.CL**

- **简介: 该论文提出FlashMem，解决大语言模型缺乏动态上下文记忆的问题。通过计算复用提炼内在记忆，提升推理效率与持续认知能力。**

- **链接: [https://arxiv.org/pdf/2601.05505v1](https://arxiv.org/pdf/2601.05505v1)**

> **作者:** Yubo Hou; Zhisheng Chen; Tao Wan; Zengchang Qin
>
> **摘要:** The stateless architecture of Large Language Models inherently lacks the mechanism to preserve dynamic context, compelling agents to redundantly reprocess history to maintain long-horizon autonomy. While latent memory offers a solution, current approaches are hindered by architectural segregation, relying on auxiliary encoders that decouple memory from the reasoning backbone. We propose FlashMem, a framework that distills intrinsic memory directly from transient reasoning states via computation reuse. Leveraging the property that internal representations uniquely encode input trajectories, FlashMem identifies the last hidden state as a sufficient statistic for the interaction history. This enables a Shared-KV Consolidator to synthesize memory by attending directly to the backbone's frozen cache, eliminating redundant re-parameterization. Furthermore, a parameter-free Cognitive Monitor leverages attention entropy to adaptively trigger consolidation only when high epistemic uncertainty is detected. Experiments demonstrate that FlashMem matches the performance of heavy baselines while reducing inference latency by 5 times, effectively bridging the gap between efficiency and persistent cognition.
>
---
#### [new 018] iReasoner: Trajectory-Aware Intrinsic Reasoning Supervision for Self-Evolving Large Multimodal Models
- **分类: cs.CL**

- **简介: 该论文提出iReasoner框架，解决大模型在无监督下提升推理能力的问题。通过强化中间推理过程，增强模型的多模态推理能力。**

- **链接: [https://arxiv.org/pdf/2601.05877v1](https://arxiv.org/pdf/2601.05877v1)**

> **作者:** Meghana Sunil; Manikandarajan Venmathimaran; Muthu Subash Kavitha
>
> **摘要:** Recent work shows that large multimodal models (LMMs) can self-improve from unlabeled data via self-play and intrinsic feedback. Yet existing self-evolving frameworks mainly reward final outcomes, leaving intermediate reasoning weakly constrained despite its importance for visually grounded decision making. We propose iReasoner, a self-evolving framework that improves an LMM's implicit reasoning by explicitly eliciting chain-of-thought (CoT) and rewarding its internal agreement. In a Proposer--Solver loop over unlabeled images, iReasoner augments outcome-level intrinsic rewards with a trajectory-aware signal defined over intermediate reasoning steps, providing learning signals that distinguish reasoning paths leading to the same answer without ground-truth labels or external judges. Starting from Qwen2.5-VL-7B, iReasoner yields up to $+2.1$ points across diverse multimodal reasoning benchmarks under fully unsupervised post-training. We hope this work serves as a starting point for reasoning-aware self-improvement in LMMs in purely unsupervised settings.
>
---
#### [new 019] Enhancing Foundation Models in Transaction Understanding with LLM-based Sentence Embeddings
- **分类: cs.CL; cs.LG**

- **简介: 论文属于交易理解任务，解决传统模型因索引表示导致语义信息丢失的问题。通过LLM生成的嵌入向量提升模型性能，实现高效且可解释的交易分析。**

- **链接: [https://arxiv.org/pdf/2601.05271v1](https://arxiv.org/pdf/2601.05271v1)**

> **作者:** Xiran Fan; Zhimeng Jiang; Chin-Chia Michael Yeh; Yuzhong Chen; Yingtong Dou; Menghai Pan; Yan Zheng
>
> **摘要:** The ubiquity of payment networks generates vast transactional data encoding rich consumer and merchant behavioral patterns. Recent foundation models for transaction analysis process tabular data sequentially but rely on index-based representations for categorical merchant fields, causing substantial semantic information loss by converting rich textual data into discrete tokens. While Large Language Models (LLMs) can address this limitation through superior semantic understanding, their computational overhead challenges real-time financial deployment. We introduce a hybrid framework that uses LLM-generated embeddings as semantic initializations for lightweight transaction models, balancing interpretability with operational efficiency. Our approach employs multi-source data fusion to enrich merchant categorical fields and a one-word constraint principle for consistent embedding generation across LLM architectures. We systematically address data quality through noise filtering and context-aware enrichment. Experiments on large-scale transaction datasets demonstrate significant performance improvements across multiple transaction understanding tasks.
>
---
#### [new 020] Router-Suggest: Dynamic Routing for Multimodal Auto-Completion in Visually-Grounded Dialogs
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出MAC任务，解决多模态自动补全问题，通过Router-Suggest框架动态选择模型，提升效率与用户满意度。**

- **链接: [https://arxiv.org/pdf/2601.05851v1](https://arxiv.org/pdf/2601.05851v1)**

> **作者:** Sandeep Mishra; Devichand Budagam; Anubhab Mandal; Bishal Santra; Pawan Goyal; Manish Gupta
>
> **备注:** Accepted to EACL 2026 Industry Track, 12 pages, 6 figures
>
> **摘要:** Real-time multimodal auto-completion is essential for digital assistants, chatbots, design tools, and healthcare consultations, where user inputs rely on shared visual context. We introduce Multimodal Auto-Completion (MAC), a task that predicts upcoming characters in live chats using partially typed text and visual cues. Unlike traditional text-only auto-completion (TAC), MAC grounds predictions in multimodal context to better capture user intent. To enable this task, we adapt MMDialog and ImageChat to create benchmark datasets. We evaluate leading vision-language models (VLMs) against strong textual baselines, highlighting trade-offs in accuracy and efficiency. We present Router-Suggest, a router framework that dynamically selects between textual models and VLMs based on dialog context, along with a lightweight variant for resource-constrained environments. Router-Suggest achieves a 2.3x to 10x speedup over the best-performing VLM. A user study shows that VLMs significantly excel over textual models on user satisfaction, notably saving user typing effort and improving the quality of completions in multi-turn conversations. These findings underscore the need for multimodal context in auto-completions, leading to smarter, user-aware assistants.
>
---
#### [new 021] Multimodal In-context Learning for ASR of Low-resource Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音识别任务，旨在解决低资源语言ASR中的数据不足问题。通过多模态在上下文学习方法提升模型性能，并验证了跨语言迁移的有效性。**

- **链接: [https://arxiv.org/pdf/2601.05707v1](https://arxiv.org/pdf/2601.05707v1)**

> **作者:** Zhaolin Li; Jan Niehues
>
> **备注:** Under review
>
> **摘要:** Automatic speech recognition (ASR) still covers only a small fraction of the world's languages, mainly due to supervised data scarcity. In-context learning (ICL) with large language models (LLMs) addresses this problem, but prior work largely focuses on high-resource languages covered during training and text-only settings. This paper investigates whether speech LLMs can learn unseen languages with multimodal ICL (MICL), and how this learning can be used to improve ASR. We conduct experiments with two speech LLMs, Phi-4 and Qwen3-Omni, on three diverse endangered languages. Firstly, we find that MICL is effective for unseen languages, leveraging both speech and text modalities. We further show that cross-lingual transfer learning improves MICL efficiency on target languages without training on them. Moreover, we analyze attention patterns to interpret MICL mechanisms, and we observe layer-dependent preferences between audio and text context, with an overall bias towards text. Finally, we show that prompt-based ASR with speech LLMs performs poorly on unseen languages, motivating a simple ASR system that combines a stronger acoustic model with a speech LLM via MICL-based selection of acoustic hypotheses. Results show that MICL consistently improves ASR performance, and that cross-lingual transfer learning matches or outperforms corpus-trained language models without using target-language data. Our code is publicly available.
>
---
#### [new 022] Data Augmented Pipeline for Legal Information Extraction and Reasoning
- **分类: cs.CL**

- **简介: 该论文属于法律信息抽取任务，旨在减少数据标注的劳动量并提升系统鲁棒性。通过引入大语言模型进行数据增强，提出了一种简单有效的管道方法。**

- **链接: [https://arxiv.org/pdf/2601.05609v1](https://arxiv.org/pdf/2601.05609v1)**

> **作者:** Nguyen Minh Phuong; Ha-Thanh Nguyen; May Myo Zin; Ken Satoh
>
> **备注:** Accepted in the Demonstration Track at ICAIL 2025
>
> **摘要:** In this paper, we propose a pipeline leveraging Large Language Models (LLMs) for data augmentation in Information Extraction tasks within the legal domain. The proposed method is both simple and effective, significantly reducing the manual effort required for data annotation while enhancing the robustness of Information Extraction systems. Furthermore, the method is generalizable, making it applicable to various Natural Language Processing (NLP) tasks beyond the legal domain.
>
---
#### [new 023] GIFT: Games as Informal Training for Generalizable LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在解决LLMs在实践智慧和通用智能上的不足。通过游戏进行非正式学习，提升模型的泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.05633v1](https://arxiv.org/pdf/2601.05633v1)**

> **作者:** Nuoyan Lyu; Bingbing Xu; Weihao Meng; Yige Yuan; Yang Zhang; Zhiyong Huang; Tat-Seng Chua; Huawei Shen
>
> **摘要:** While Large Language Models (LLMs) have achieved remarkable success in formal learning tasks such as mathematics and code generation, they still struggle with the "practical wisdom" and generalizable intelligence, such as strategic creativity and social reasoning, that characterize human cognition. This gap arises from a lack of informal learning, which thrives on interactive feedback rather than goal-oriented instruction. In this paper, we propose treating Games as a primary environment for LLM informal learning, leveraging their intrinsic reward signals and abstracted complexity to cultivate diverse competencies. To address the performance degradation observed in multi-task learning, we introduce a Nested Training Framework. Unlike naive task mixing optimizing an implicit "OR" objective, our framework employs sequential task composition to enforce an explicit "AND" objective, compelling the model to master multiple abilities simultaneously to achieve maximal rewards. Using GRPO-based reinforcement learning across Matrix Games, TicTacToe, and Who's the Spy games, we demonstrate that integrating game-based informal learning not only prevents task interference but also significantly bolsters the model's generalization across broad ability-oriented benchmarks. The framework and implementation are publicly available.
>
---
#### [new 024] The Facade of Truth: Uncovering and Mitigating LLM Susceptibility to Deceptive Evidence
- **分类: cs.CL**

- **简介: 该论文属于人工智能安全任务，旨在解决大语言模型对欺骗性证据的脆弱性问题。通过构建框架生成误导性证据，验证模型易受欺骗的特性，并提出防御机制提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2601.05478v1](https://arxiv.org/pdf/2601.05478v1)**

> **作者:** Herun Wan; Jiaying Wu; Minnan Luo; Fanxiao Li; Zhi Zeng; Min-Yen Kan
>
> **摘要:** To reliably assist human decision-making, LLMs must maintain factual internal beliefs against misleading injections. While current models resist explicit misinformation, we uncover a fundamental vulnerability to sophisticated, hard-to-falsify evidence. To systematically probe this weakness, we introduce MisBelief, a framework that generates misleading evidence via collaborative, multi-round interactions among multi-role LLMs. This process mimics subtle, defeasible reasoning and progressive refinement to create logically persuasive yet factually deceptive claims. Using MisBelief, we generate 4,800 instances across three difficulty levels to evaluate 7 representative LLMs. Results indicate that while models are robust to direct misinformation, they are highly sensitive to this refined evidence: belief scores in falsehoods increase by an average of 93.0\%, fundamentally compromising downstream recommendations. To address this, we propose Deceptive Intent Shielding (DIS), a governance mechanism that provides an early warning signal by inferring the deceptive intent behind evidence. Empirical results demonstrate that DIS consistently mitigates belief shifts and promotes more cautious evidence evaluation.
>
---
#### [new 025] LLMs as Science Journalists: Supporting Early-stage Researchers in Communicating Their Science to the Public
- **分类: cs.CL**

- **简介: 论文提出一种框架，训练大语言模型扮演科学记者角色，帮助早期研究人员向公众有效传达研究成果。任务是提升科研成果的公众沟通效果，解决现有模型不适应该场景的问题。**

- **链接: [https://arxiv.org/pdf/2601.05821v1](https://arxiv.org/pdf/2601.05821v1)**

> **作者:** Milad Alshomary; Grace Li; Anubhav Jangra; Yufang Hou; Kathleen McKeown; Smaranda Muresan
>
> **摘要:** The scientific community needs tools that help early-stage researchers effectively communicate their findings and innovations to the public. Although existing general-purpose Large Language Models (LLMs) can assist in this endeavor, they are not optimally aligned for it. To address this, we propose a framework for training LLMs to emulate the role of a science journalist that can be used by early-stage researchers to learn how to properly communicate their papers to the general public. We evaluate the usefulness of our trained LLM Journalists in leading conversations with both simulated and human researchers. %compared to the general-purpose ones. Our experiments indicate that LLMs trained using our framework ask more relevant questions that address the societal impact of research, prompting researchers to clarify and elaborate on their findings. In the user study, the majority of participants who interacted with our trained LLM Journalist appreciated it more than interacting with general-purpose LLMs.
>
---
#### [new 026] FACTUM: Mechanistic Detection of Citation Hallucination in Long-Form RAG
- **分类: cs.CL**

- **简介: 该论文属于RAG系统中的引用幻觉检测任务，旨在解决模型错误引用问题。通过分析模型机制，提出FACTUM框架提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.05866v1](https://arxiv.org/pdf/2601.05866v1)**

> **作者:** Maxime Dassen; Rebecca Kotula; Kenton Murray; Andrew Yates; Dawn Lawrie; Efsun Kayi; James Mayfield; Kevin Duh
>
> **备注:** Accepted at ECIR 2026. 18 pages, 2 figures
>
> **摘要:** Retrieval-Augmented Generation (RAG) models are critically undermined by citation hallucinations, a deceptive failure where a model confidently cites a source that fails to support its claim. Existing work often attributes hallucination to a simple over-reliance on the model's parametric knowledge. We challenge this view and introduce FACTUM (Framework for Attesting Citation Trustworthiness via Underlying Mechanisms), a framework of four mechanistic scores measuring the distinct contributions of a model's attention and FFN pathways, and the alignment between them. Our analysis reveals two consistent signatures of correct citation: a significantly stronger contribution from the model's parametric knowledge and greater use of the attention sink for information synthesis. Crucially, we find the signature of a correct citation is not static but evolves with model scale. For example, the signature of a correct citation for the Llama-3.2-3B model is marked by higher pathway alignment, whereas for the Llama-3.1-8B model, it is characterized by lower alignment, where pathways contribute more distinct, orthogonal information. By capturing this complex, evolving signature, FACTUM outperforms state-of-the-art baselines by up to 37.5% in AUC. Our findings reframe citation hallucination as a complex, scale-dependent interplay between internal mechanisms, paving the way for more nuanced and reliable RAG systems.
>
---
#### [new 027] CHisAgent: A Multi-Agent Framework for Event Taxonomy Construction in Ancient Chinese Cultural Systems
- **分类: cs.CL**

- **简介: 该论文属于历史知识组织任务，旨在解决古中文文化系统事件分类构建难题。提出CHisAgent框架，通过多阶段代理协作实现高效、准确的分类体系构建。**

- **链接: [https://arxiv.org/pdf/2601.05520v1](https://arxiv.org/pdf/2601.05520v1)**

> **作者:** Xuemei Tang; Chengxi Yan; Jinghang Gu; Chu-Ren Huang
>
> **备注:** 22 pages, 13 figures, 7 tables
>
> **摘要:** Despite strong performance on many tasks, large language models (LLMs) show limited ability in historical and cultural reasoning, particularly in non-English contexts such as Chinese history. Taxonomic structures offer an effective mechanism to organize historical knowledge and improve understanding. However, manual taxonomy construction is costly and difficult to scale. Therefore, we propose \textbf{CHisAgent}, a multi-agent LLM framework for historical taxonomy construction in ancient Chinese contexts. CHisAgent decomposes taxonomy construction into three role-specialized stages: a bottom-up \textit{Inducer} that derives an initial hierarchy from raw historical corpora, a top-down \textit{Expander} that introduces missing intermediate concepts using LLM world knowledge, and an evidence-guided \textit{Enricher} that integrates external structured historical resources to ensure faithfulness. Using the \textit{Twenty-Four Histories}, we construct a large-scale, domain-aware event taxonomy covering politics, military, diplomacy, and social life in ancient China. Extensive reference-free and reference-based evaluations demonstrate improved structural coherence and coverage, while further analysis shows that the resulting taxonomy supports cross-cultural alignment.
>
---
#### [new 028] Don't Break the Cache: An Evaluation of Prompt Caching for Long-Horizon Agentic Tasks
- **分类: cs.CL**

- **简介: 该论文研究prompt caching在长周期代理任务中的效果，旨在降低API成本和提升响应速度。通过实验对比不同缓存策略，提出优化建议。**

- **链接: [https://arxiv.org/pdf/2601.06007v1](https://arxiv.org/pdf/2601.06007v1)**

> **作者:** Elias Lumer; Faheem Nizar; Akshaya Jangiti; Kevin Frank; Anmol Gulati; Mandar Phadate; Vamse Kumar Subbiah
>
> **备注:** 15 pages, 8 figures
>
> **摘要:** Recent advancements in Large Language Model (LLM) agents have enabled complex multi-turn agentic tasks requiring extensive tool calling, where conversations can span dozens of API calls with increasingly large context windows. However, although major LLM providers offer prompt caching to reduce cost and latency, its benefits for agentic workloads remain underexplored in the research literature. To our knowledge, no prior work quantifies these cost savings or compares caching strategies for multi-turn agentic tasks. We present a comprehensive evaluation of prompt caching across three major LLM providers (OpenAI, Anthropic, and Google) and compare three caching strategies, including full context caching, system prompt only caching, and caching that excludes dynamic tool results. We evaluate on DeepResearchBench, a multi-turn agentic benchmark where agents autonomously execute real-world web search tool calls to answer complex research questions, measuring both API cost and time to first token (TTFT) across over 500 agent sessions with 10,000-token system prompts. Our results demonstrate that prompt caching reduces API costs by 45-80% and improves time to first token by 13-31% across providers. We find that strategic prompt cache block control, such as placing dynamic content at the end of the system prompt, avoiding dynamic traditional function calling, and excluding dynamic tool results, provides more consistent benefits than naive full-context caching, which can paradoxically increase latency. Our analysis reveals nuanced variations in caching behavior across providers, and we provide practical guidance for implementing prompt caching in production agentic systems.
>
---
#### [new 029] EnvScaler: Scaling Tool-Interactive Environments for LLM Agent via Programmatic Synthesis
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出EnvScaler，解决LLM在复杂环境中的任务执行问题。通过自动化生成工具交互环境，提升模型在多步骤、多工具场景下的表现。**

- **链接: [https://arxiv.org/pdf/2601.05808v1](https://arxiv.org/pdf/2601.05808v1)**

> **作者:** Xiaoshuai Song; Haofei Chang; Guanting Dong; Yutao Zhu; Zhicheng Dou; Ji-Rong Wen
>
> **备注:** Working in progress
>
> **摘要:** Large language models (LLMs) are expected to be trained to act as agents in various real-world environments, but this process relies on rich and varied tool-interaction sandboxes. However, access to real systems is often restricted; LLM-simulated environments are prone to hallucinations and inconsistencies; and manually built sandboxes are hard to scale. In this paper, we propose EnvScaler, an automated framework for scalable tool-interaction environments via programmatic synthesis. EnvScaler comprises two components. First, SkelBuilder constructs diverse environment skeletons through topic mining, logic modeling, and quality evaluation. Then, ScenGenerator generates multiple task scenarios and rule-based trajectory validation functions for each environment. With EnvScaler, we synthesize 191 environments and about 7K scenarios, and apply them to Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) for Qwen3 series models. Results on three benchmarks show that EnvScaler significantly improves LLMs' ability to solve tasks in complex environments involving multi-turn, multi-tool interactions. We release our code and data at https://github.com/RUC-NLPIR/EnvScaler.
>
---
#### [new 030] Pantagruel: Unified Self-Supervised Encoders for French Text and Speech
- **分类: cs.CL**

- **简介: 该论文提出Pantagruel模型，用于法语文本和语音的统一自监督编码。解决多模态表示学习问题，通过共享架构提升效果。**

- **链接: [https://arxiv.org/pdf/2601.05911v1](https://arxiv.org/pdf/2601.05911v1)**

> **作者:** Phuong-Hang Le; Valentin Pelloin; Arnault Chatelain; Maryem Bouziane; Mohammed Ghennai; Qianwen Guan; Kirill Milintsevich; Salima Mdhaffar; Aidan Mannion; Nils Defauw; Shuyue Gu; Alexandre Audibert; Marco Dinarelli; Yannick Estève; Lorraine Goeuriot; Steffen Lalande; Nicolas Hervé; Maximin Coavoux; François Portet; Étienne Ollion; Marie Candito; Maxime Peyrard; Solange Rossato; Benjamin Lecouteux; Aurélie Nardy; Gilles Sérasset; Vincent Segonne; Solène Evain; Diandra Fabre; Didier Schwab
>
> **摘要:** We release Pantagruel models, a new family of self-supervised encoder models for French text and speech. Instead of predicting modality-tailored targets such as textual tokens or speech units, Pantagruel learns contextualized target representations in the feature space, allowing modality-specific encoders to capture linguistic and acoustic regularities more effectively. Separate models are pre-trained on large-scale French corpora, including Wikipedia, OSCAR and CroissantLLM for text, together with MultilingualLibriSpeech, LeBenchmark, and INA-100k for speech. INA-100k is a newly introduced 100,000-hour corpus of French audio derived from the archives of the Institut National de l'Audiovisuel (INA), the national repository of French radio and television broadcasts, providing highly diverse audio data. We evaluate Pantagruel across a broad range of downstream tasks spanning both modalities, including those from the standard French benchmarks such as FLUE or LeBenchmark. Across these tasks, Pantagruel models show competitive or superior performance compared to strong French baselines such as CamemBERT, FlauBERT, and LeBenchmark2.0, while maintaining a shared architecture that can seamlessly handle either speech or text inputs. These results confirm the effectiveness of feature-space self-supervised objectives for French representation learning and highlight Pantagruel as a robust foundation for multimodal speech-text understanding.
>
---
#### [new 031] Glitter: Visualizing Lexical Surprisal for Readability in Administrative Texts
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在通过信息熵可视化提升行政文本的可读性。工作包括提出一种基于多语言模型的可视化框架，以估计文本信息熵并改善其清晰度。**

- **链接: [https://arxiv.org/pdf/2601.05411v1](https://arxiv.org/pdf/2601.05411v1)**

> **作者:** Jan Černý; Ivana Kvapilíková; Silvie Cinková
>
> **摘要:** This work investigates how measuring information entropy of text can be used to estimate its readability. We propose a visualization framework that can be used to approximate information entropy of text using multiple language models and visualize the result. The end goal is to use this method to estimate and improve readability and clarity of administrative or bureaucratic texts. Our toolset is available as a libre software on https://github.com/ufal/Glitter.
>
---
#### [new 032] Continual-learning for Modelling Low-Resource Languages from Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言建模任务，旨在解决小语言模型在适应低资源语言时的灾难性遗忘问题。通过持续学习和代码切换策略，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.05874v1](https://arxiv.org/pdf/2601.05874v1)**

> **作者:** Santosh Srinath K; Mudit Somani; Varun Reddy Padala; Prajna Devi Upadhyay; Abhijit Das
>
> **摘要:** Modelling a language model for a multi-lingual scenario includes several potential challenges, among which catastrophic forgetting is the major challenge. For example, small language models (SLM) built for low-resource languages by adapting large language models (LLMs) pose the challenge of catastrophic forgetting. This work proposes to employ a continual learning strategy using parts-of-speech (POS)-based code-switching along with a replay adapter strategy to mitigate the identified gap of catastrophic forgetting while training SLM from LLM. Experiments conducted on vision language tasks such as visual question answering and language modelling task exhibits the success of the proposed architecture.
>
---
#### [new 033] Gender Bias in LLMs: Preliminary Evidence from Shared Parenting Scenario in Czech Family Law
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于法律AI评估任务，旨在检测LLMs在家庭法场景中的性别偏见。通过设计实验，比较模型对不同性别设定的回应差异，揭示潜在的系统性偏差。**

- **链接: [https://arxiv.org/pdf/2601.05879v1](https://arxiv.org/pdf/2601.05879v1)**

> **作者:** Jakub Harasta; Matej Vasina; Martin Kornel; Tomas Foltynek
>
> **备注:** Accepted at AI for Access to Justice, Dispute Resolution, and Data Access (AIDA2J) at Jurix 2025, Torino, Italy
>
> **摘要:** Access to justice remains limited for many people, leading laypersons to increasingly rely on Large Language Models (LLMs) for legal self-help. Laypeople use these tools intuitively, which may lead them to form expectations based on incomplete, incorrect, or biased outputs. This study examines whether leading LLMs exhibit gender bias in their responses to a realistic family law scenario. We present an expert-designed divorce scenario grounded in Czech family law and evaluate four state-of-the-art LLMs GPT-5 nano, Claude Haiku 4.5, Gemini 2.5 Flash, and Llama 3.3 in a fully zero-shot interaction. We deploy two versions of the scenario, one with gendered names and one with neutral labels, to establish a baseline for comparison. We further introduce nine legally relevant factors that vary the factual circumstances of the case and test whether these variations influence the models' proposed shared-parenting ratios. Our preliminary results highlight differences across models and suggest gender-dependent patterns in the outcomes generated by some systems. The findings underscore both the risks associated with laypeople's reliance on LLMs for legal guidance and the need for more robust evaluation of model behavior in sensitive legal contexts. We present exploratory and descriptive evidence intended to identify systematic asymmetries rather than to establish causal effects.
>
---
#### [new 034] Left, Right, or Center? Evaluating LLM Framing in News Classification and Generation
- **分类: cs.CL**

- **简介: 该论文属于新闻分类与生成任务，研究LLM在政治框架上的偏见。通过测试模型分类偏差与生成文本的框架行为，发现模型普遍存在中心化倾向。**

- **链接: [https://arxiv.org/pdf/2601.05835v1](https://arxiv.org/pdf/2601.05835v1)**

> **作者:** Molly Kennedy; Ali Parker; Yihong Liu; Hinrich Schütze
>
> **摘要:** Large Language Model (LLM) based summarization and text generation are increasingly used for producing and rewriting text, raising concerns about political framing in journalism where subtle wording choices can shape interpretation. Across nine state-of-the-art LLMs, we study political framing by testing whether LLMs' classification-based bias signals align with framing behavior in their generated summaries. We first compare few-shot ideology predictions against LEFT/CENTER/RIGHT labels. We then generate "steered" summaries under FAITHFUL, CENTRIST, LEFT, and RIGHT prompts, and score all outputs using a single fixed ideology evaluator. We find pervasive ideological center-collapse in both article-level ratings and generated text, indicating a systematic tendency toward centrist framing. Among evaluated models, Grok 4 is by far the most ideologically expressive generator, while Claude Sonnet 4.5 and Llama 3.1 achieve the strongest bias-rating performance among commercial and open-weight models, respectively.
>
---
#### [new 035] Stephanie2: Thinking, Waiting, and Making Decisions Like Humans in Step-by-Step AI Social Chat
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Stephanie2，解决AI聊天中消息节奏不自然的问题。通过主动等待和时延建模，提升对话自然度与参与感。属于对话系统任务。**

- **链接: [https://arxiv.org/pdf/2601.05657v1](https://arxiv.org/pdf/2601.05657v1)**

> **作者:** Hao Yang; Hongyuan Lu; Dingkang Yang; Wenliang Yang; Peng Sun; Xiaochuan Zhang; Jun Xiao; Kefan He; Wai Lam; Yang Liu; Xinhua Zeng
>
> **备注:** 13 pages
>
> **摘要:** Instant-messaging human social chat typically progresses through a sequence of short messages. Existing step-by-step AI chatting systems typically split a one-shot generation into multiple messages and send them sequentially, but they lack an active waiting mechanism and exhibit unnatural message pacing. In order to address these issues, we propose Stephanie2, a novel next-generation step-wise decision-making dialogue agent. With active waiting and message-pace adaptation, Stephanie2 explicitly decides at each step whether to send or wait, and models latency as the sum of thinking time and typing time to achieve more natural pacing. We further introduce a time-window-based dual-agent dialogue system to generate pseudo dialogue histories for human and automatic evaluations. Experiments show that Stephanie2 clearly outperforms Stephanie1 on metrics such as naturalness and engagement, and achieves a higher pass rate on human evaluation with the role identification Turing test.
>
---
#### [new 036] Illusions of Confidence? Diagnosing LLM Truthfulness via Neighborhood Consistency
- **分类: cs.CL; cs.AI; cs.HC; cs.LG; cs.MA**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在面对上下文干扰时的可信度问题。通过提出NCB指标和SAT方法，提升模型信念的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.05905v1](https://arxiv.org/pdf/2601.05905v1)**

> **作者:** Haoming Xu; Ningyuan Zhao; Yunzhi Yao; Weihong Xu; Hongru Wang; Xinle Deng; Shumin Deng; Jeff Z. Pan; Huajun Chen; Ningyu Zhang
>
> **备注:** Work in progress
>
> **摘要:** As Large Language Models (LLMs) are increasingly deployed in real-world settings, correctness alone is insufficient. Reliable deployment requires maintaining truthful beliefs under contextual perturbations. Existing evaluations largely rely on point-wise confidence like Self-Consistency, which can mask brittle belief. We show that even facts answered with perfect self-consistency can rapidly collapse under mild contextual interference. To address this gap, we propose Neighbor-Consistency Belief (NCB), a structural measure of belief robustness that evaluates response coherence across a conceptual neighborhood. To validate the efficiency of NCB, we introduce a new cognitive stress-testing protocol that probes outputs stability under contextual interference. Experiments across multiple LLMs show that the performance of high-NCB data is relatively more resistant to interference. Finally, we present Structure-Aware Training (SAT), which optimizes context-invariant belief structure and reduces long-tail knowledge brittleness by approximately 30%. Code will be available at https://github.com/zjunlp/belief.
>
---
#### [new 037] Generation-Based and Emotion-Reflected Memory Update: Creating the KEEM Dataset for Better Long-Term Conversation
- **分类: cs.CL**

- **简介: 该论文属于对话系统任务，旨在解决长对话中记忆更新的问题。提出KEEM数据集，通过生成方式整合情感与关键信息，提升系统理解与回应能力。**

- **链接: [https://arxiv.org/pdf/2601.05548v1](https://arxiv.org/pdf/2601.05548v1)**

> **作者:** Jeonghyun Kang; Hongjin Kim; Harksoo Kim
>
> **备注:** COLING 2025 accepted paper (Main)
>
> **摘要:** In this work, we introduce the Keep Emotional and Essential Memory (KEEM) dataset, a novel generation-based dataset designed to enhance memory updates in long-term conversational systems. Unlike existing approaches that rely on simple accumulation or operation-based methods, which often result in information conflicts and difficulties in accurately tracking a user's current state, KEEM dynamically generates integrative memories. This process not only preserves essential factual information but also incorporates emotional context and causal relationships, enabling a more nuanced understanding of user interactions. By seamlessly updating a system's memory with both emotional and essential data, our approach promotes deeper empathy and enhances the system's ability to respond meaningfully in open-domain conversations.
>
---
#### [new 038] HAPS: Hierarchical LLM Routing with Joint Architecture and Parameter Search
- **分类: cs.CL**

- **简介: 该论文属于LLM路由任务，解决现有方法忽略参数设置的问题。提出HAPS框架，联合搜索模型架构与参数，提升任务性能。**

- **链接: [https://arxiv.org/pdf/2601.05903v1](https://arxiv.org/pdf/2601.05903v1)**

> **作者:** Zihang Tian; Rui Li; Jingsen Zhang; Xiaohe Bo; Wei Huo; Xu Chen
>
> **摘要:** Large language model (LLM) routing aims to exploit the specialized strengths of different LLMs for diverse tasks. However, existing approaches typically focus on selecting LLM architectures while overlooking parameter settings, which are critical for task performance. In this paper, we introduce HAPS, a hierarchical LLM routing framework that jointly searches over model architectures and parameters. Specifically, we use a high-level router to select among candidate LLM architectures, and then search for the optimal parameters for the selected architectures based on a low-level router. We design a parameter generation network to share parameters between the two routers to mutually enhance their capabilities. In the training process, we design a reward-augmented objective to effectively optimize our framework. Experiments on two commonly used benchmarks show that HAPS consistently outperforms strong routing baselines. We have released our code at https://github.com/zihangtian/HAPS.
>
---
#### [new 039] Lost in Execution: On the Multilingual Robustness of Tool Calling in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究多语言环境下大模型调用工具的鲁棒性问题，属于自然语言处理中的工具调用任务。工作包括构建基准测试集，分析错误原因，并评估减少语言导致错误的策略。**

- **链接: [https://arxiv.org/pdf/2601.05366v1](https://arxiv.org/pdf/2601.05366v1)**

> **作者:** Zheng Luo; T Pranav Kutralingam; Ogochukwu N Okoani; Wanpeng Xu; Hua Wei; Xiyang Hu
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed as agents that invoke external tools through structured function calls. While recent work reports strong tool-calling performance under standard English-centric evaluations, the robustness of tool calling under multilingual user interactions remains underexplored. In this work, we introduce MLCL, a diagnostic benchmark, and conduct a systematic evaluation of multilingual tool calling across Chinese, Hindi, and the low-resource language Igbo. Through fine-grained error analysis, we show that many failures occur despite correct intent understanding and tool selection. We identify parameter value language mismatch as a dominant failure mode, where models generate semantically appropriate parameter values in the user's language, violating language-invariant execution conventions. We further evaluate several inference-time system strategies and find that while these strategies substantially reduce language-induced execution errors, none of them can fully recover English-level performance.
>
---
#### [new 040] Visualising Information Flow in Word Embeddings with Diffusion Tensor Imaging
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM中信息流动的可视化问题。通过应用扩散张量成像技术，分析词嵌入间的信息流，提升模型可解释性。**

- **链接: [https://arxiv.org/pdf/2601.05713v1](https://arxiv.org/pdf/2601.05713v1)**

> **作者:** Thomas Fabian
>
> **摘要:** Understanding how large language models (LLMs) represent natural language is a central challenge in natural language processing (NLP) research. Many existing methods extract word embeddings from an LLM, visualise the embedding space via point-plots, and compare the relative positions of certain words. However, this approach only considers single words and not whole natural language expressions, thus disregards the context in which a word is used. Here we present a novel tool for analysing and visualising information flow in natural language expressions by applying diffusion tensor imaging (DTI) to word embeddings. We find that DTI reveals how information flows between word embeddings. Tracking information flows within the layers of an LLM allows for comparing different model structures and revealing opportunities for pruning an LLM's under-utilised layers. Furthermore, our model reveals differences in information flows for tasks like pronoun resolution and metaphor detection. Our results show that our model permits novel insights into how LLMs represent actual natural language expressions, extending the comparison of isolated word embeddings and improving the interpretability of NLP models.
>
---
#### [new 041] The Molecular Structure of Thought: Mapping the Topology of Long Chain-of-Thought Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决长链思维推理学习难题。通过分析长链思维轨迹的分子结构特性，提出Mole-Syn方法提升模型性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.06002v1](https://arxiv.org/pdf/2601.06002v1)**

> **作者:** Qiguang Chen; Yantao Du; Ziniu Li; Jinhao Liu; Songyao Duan; Jiarui Guo; Minghao Liu; Jiaheng Liu; Tong Yang; Ge Zhang; Libo Qin; Wanxiang Che; Wenhao Huang
>
> **备注:** Preprint
>
> **摘要:** Large language models (LLMs) often fail to learn effective long chain-of-thought (Long CoT) reasoning from human or non-Long-CoT LLMs imitation. To understand this, we propose that effective and learnable Long CoT trajectories feature stable molecular-like structures in unified view, which are formed by three interaction types: Deep-Reasoning (covalent-like), Self-Reflection (hydrogen-bond-like), and Self-Exploration (van der Waals-like). Analysis of distilled trajectories reveals these structures emerge from Long CoT fine-tuning, not keyword imitation. We introduce Effective Semantic Isomers and show that only bonds promoting fast entropy convergence support stable Long CoT learning, while structural competition impairs training. Drawing on these findings, we present Mole-Syn, a distribution-transfer-graph method that guides synthesis of effective Long CoT structures, boosting performance and RL stability across benchmarks.
>
---
#### [new 042] The Table of Media Bias Elements: A sentence-level taxonomy of media bias types and propaganda techniques
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于媒体偏见分析任务，旨在解决如何细粒度识别句子层面的偏见类型。通过构建38种偏见类型及其分类体系，提供定义、例子和识别指导。**

- **链接: [https://arxiv.org/pdf/2601.05358v1](https://arxiv.org/pdf/2601.05358v1)**

> **作者:** Tim Menzner; Jochen L. Leidner
>
> **摘要:** Public debates about "left-" or "right-wing" news overlook the fact that bias is usually conveyed by concrete linguistic manoeuvres that transcend any single political spectrum. We therefore shift the focus from where an outlet allegedly stands to how partiality is expressed in individual sentences. Drawing on 26,464 sentences collected from newsroom corpora, user submissions and our own browsing, we iteratively combine close-reading, interdisciplinary theory and pilot annotation to derive a fine-grained, sentence-level taxonomy of media bias and propaganda. The result is a two-tier schema comprising 38 elementary bias types, arranged in six functional families and visualised as a "table of media-bias elements". For each type we supply a definition, real-world examples, cognitive and societal drivers, and guidance for recognition. A quantitative survey of a random 155-sentence sample illustrates prevalence differences, while a cross-walk to the best-known NLP and communication-science taxonomies reveals substantial coverage gains and reduced ambiguity.
>
---
#### [new 043] Multilingual Amnesia: On the Transferability of Unlearning in Multilingual LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究多语言大模型中的知识消除问题，探讨数据和概念删除在不同语言间的迁移性，旨在提升模型的安全性与公平性。**

- **链接: [https://arxiv.org/pdf/2601.05641v1](https://arxiv.org/pdf/2601.05641v1)**

> **作者:** Alireza Dehghanpour Farashah; Aditi Khandelwal; Marylou Fauchard; Zhuan Shi; Negar Rostamzadeh; Golnoosh Farnadi
>
> **摘要:** As multilingual large language models become more widely used, ensuring their safety and fairness across diverse linguistic contexts presents unique challenges. While existing research on machine unlearning has primarily focused on monolingual settings, typically English, multilingual environments introduce additional complexities due to cross-lingual knowledge transfer and biases embedded in both pretraining and fine-tuning data. In this work, we study multilingual unlearning using the Aya-Expanse 8B model under two settings: (1) data unlearning and (2) concept unlearning. We extend benchmarks for factual knowledge and stereotypes to ten languages through translation: English, French, Arabic, Japanese, Russian, Farsi, Korean, Hindi, Hebrew, and Indonesian. These languages span five language families and a wide range of resource levels. Our experiments show that unlearning in high-resource languages is generally more stable, with asymmetric transfer effects observed between typologically related languages. Furthermore, our analysis of linguistic distances indicates that syntactic similarity is the strongest predictor of cross-lingual unlearning behavior.
>
---
#### [new 044] Chaining the Evidence: Robust Reinforcement Learning for Deep Search Agents with Citation-Aware Rubric Rewards
- **分类: cs.CL**

- **简介: 该论文属于深度搜索任务，旨在解决RL代理在推理过程中缺乏全面性和事实依据的问题。提出CaRR框架和C-GRPO算法，提升代理的证据链构建能力。**

- **链接: [https://arxiv.org/pdf/2601.06021v1](https://arxiv.org/pdf/2601.06021v1)**

> **作者:** Jiajie Zhang; Xin Lv; Ling Feng; Lei Hou; Juanzi Li
>
> **摘要:** Reinforcement learning (RL) has emerged as a critical technique for enhancing LLM-based deep search agents. However, existing approaches primarily rely on binary outcome rewards, which fail to capture the comprehensiveness and factuality of agents' reasoning process, and often lead to undesirable behaviors such as shortcut exploitation and hallucinations. To address these limitations, we propose \textbf{Citation-aware Rubric Rewards (CaRR)}, a fine-grained reward framework for deep search agents that emphasizes reasoning comprehensiveness, factual grounding, and evidence connectivity. CaRR decomposes complex questions into verifiable single-hop rubrics and requires agents to satisfy these rubrics by explicitly identifying hidden entities, supporting them with correct citations, and constructing complete evidence chains that link to the predicted answer. We further introduce \textbf{Citation-aware Group Relative Policy Optimization (C-GRPO)}, which combines CaRR and outcome rewards for training robust deep search agents. Experiments show that C-GRPO consistently outperforms standard outcome-based RL baselines across multiple deep search benchmarks. Our analysis also validates that C-GRPO effectively discourages shortcut exploitation, promotes comprehensive, evidence-grounded reasoning, and exhibits strong generalization to open-ended deep research tasks. Our code and data are available at https://github.com/THUDM/CaRR.
>
---
#### [new 045] Same Claim, Different Judgment: Benchmarking Scenario-Induced Bias in Multilingual Financial Misinformation Detection
- **分类: cs.CL**

- **简介: 该论文属于多语言金融虚假信息检测任务，旨在解决模型在不同经济场景下的行为偏差问题。构建了包含多种场景的基准数据集，评估22个主流模型的偏差情况。**

- **链接: [https://arxiv.org/pdf/2601.05403v1](https://arxiv.org/pdf/2601.05403v1)**

> **作者:** Zhiwei Liu; Yupen Cao; Yuechen Jiang; Mohsinul Kabir; Polydoros Giannouris; Chen Xu; Ziyang Xu; Tianlei Zhu; Tariquzzaman Faisal; Triantafillos Papadopoulos; Yan Wang; Lingfei Qian; Xueqing Peng; Zhuohan Xie; Ye Yuan; Saeed Almheiri; Abdulrazzaq Alnajjar; Mingbin Chen; Harry Stuart; Paul Thompson; Prayag Tiwari; Alejandro Lopez-Lira; Xue Liu; Jimin Huang; Sophia Ananiadou
>
> **备注:** Work in progress
>
> **摘要:** Large language models (LLMs) have been widely applied across various domains of finance. Since their training data are largely derived from human-authored corpora, LLMs may inherit a range of human biases. Behavioral biases can lead to instability and uncertainty in decision-making, particularly when processing financial information. However, existing research on LLM bias has mainly focused on direct questioning or simplified, general-purpose settings, with limited consideration of the complex real-world financial environments and high-risk, context-sensitive, multilingual financial misinformation detection tasks (\mfmd). In this work, we propose \mfmdscen, a comprehensive benchmark for evaluating behavioral biases of LLMs in \mfmd across diverse economic scenarios. In collaboration with financial experts, we construct three types of complex financial scenarios: (i) role- and personality-based, (ii) role- and region-based, and (iii) role-based scenarios incorporating ethnicity and religious beliefs. We further develop a multilingual financial misinformation dataset covering English, Chinese, Greek, and Bengali. By integrating these scenarios with misinformation claims, \mfmdscen enables a systematic evaluation of 22 mainstream LLMs. Our findings reveal that pronounced behavioral biases persist across both commercial and open-source models. This project will be available at https://github.com/lzw108/FMD.
>
---
#### [new 046] ReasonAny: Incorporating Reasoning Capability to Any Model via Simple and Effective Model Merging
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型融合任务，旨在解决将推理能力融入专业模型时性能下降的问题。通过Contrastive Gradient Identification方法，实现推理与领域知识的高效结合。**

- **链接: [https://arxiv.org/pdf/2601.05560v1](https://arxiv.org/pdf/2601.05560v1)**

> **作者:** Junyao Yang; Chen Qian; Dongrui Liu; Wen Shen; Yong Liu; Jing Shao
>
> **备注:** 22 pages, 6 figures, 14 tables
>
> **摘要:** Large Reasoning Models (LRMs) with long chain-of-thought reasoning have recently achieved remarkable success. Yet, equipping domain-specialized models with such reasoning capabilities, referred to as "Reasoning + X", remains a significant challenge. While model merging offers a promising training-free solution, existing methods often suffer from a destructive performance collapse: existing methods tend to both weaken reasoning depth and compromise domain-specific utility. Interestingly, we identify a counter-intuitive phenomenon underlying this failure: reasoning ability predominantly resides in parameter regions with low gradient sensitivity, contrary to the common assumption that domain capabilities correspond to high-magnitude parameters. Motivated by this insight, we propose ReasonAny, a novel merging framework that resolves the reasoning-domain performance collapse through Contrastive Gradient Identification. Experiments across safety, biomedicine, and finance domains show that ReasonAny effectively synthesizes "Reasoning + X" capabilities, significantly outperforming state-of-the-art baselines while retaining robust reasoning performance.
>
---
#### [new 047] MemBuilder: Reinforcing LLMs for Long-Term Memory Construction via Attributed Dense Rewards
- **分类: cs.CL**

- **简介: 该论文属于对话系统任务，旨在解决长对话中保持一致性的问题。通过引入MemBuilder框架，利用密集奖励和多维记忆归因提升模型长期记忆能力。**

- **链接: [https://arxiv.org/pdf/2601.05488v1](https://arxiv.org/pdf/2601.05488v1)**

> **作者:** Zhiyu Shen; Ziming Wu; Fuming Lai; Shaobing Lian; Yanghui Rao
>
> **备注:** 19 pages (9 main + 10 appendix), 7 figures, 3 tables
>
> **摘要:** Maintaining consistency in long-term dialogues remains a fundamental challenge for LLMs, as standard retrieval mechanisms often fail to capture the temporal evolution of historical states. While memory-augmented frameworks offer a structured alternative, current systems rely on static prompting of closed-source models or suffer from ineffective training paradigms with sparse rewards. We introduce MemBuilder, a reinforcement learning framework that trains models to orchestrate multi-dimensional memory construction with attributed dense rewards. MemBuilder addresses two key challenges: (1) Sparse Trajectory-Level Rewards: we employ synthetic session-level question generation to provide dense intermediate rewards across extended trajectories; and (2) Multi-Dimensional Memory Attribution: we introduce contribution-aware gradient weighting that scales policy updates based on each component's downstream impact. Experimental results show that MemBuilder enables a 4B-parameter model to outperform state-of-the-art closed-source baselines, exhibiting strong generalization across long-term dialogue benchmarks.
>
---
#### [new 048] Towards Valid Student Simulation with Large Language Models
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于教育技术任务，旨在解决LLM模拟学生时的"能力悖论"问题。通过引入Epistemic State Specification框架，提升模拟学生的有效性与真实性。**

- **链接: [https://arxiv.org/pdf/2601.05473v1](https://arxiv.org/pdf/2601.05473v1)**

> **作者:** Zhihao Yuan; Yunze Xiao; Ming Li; Weihao Xuan; Richard Tong; Mona Diab; Tom Mitchell
>
> **摘要:** This paper presents a conceptual and methodological framework for large language model (LLM) based student simulation in educational settings. The authors identify a core failure mode, termed the "competence paradox" in which broadly capable LLMs are asked to emulate partially knowledgeable learners, leading to unrealistic error patterns and learning dynamics. To address this, the paper reframes student simulation as a constrained generation problem governed by an explicit Epistemic State Specification (ESS), which defines what a simulated learner can access, how errors are structured, and how learner state evolves over time. The work further introduces a Goal-by-Environment framework to situate simulated student systems according to behavioral objectives and deployment contexts. Rather than proposing a new system or benchmark, the paper synthesizes prior literature, formalizes key design dimensions, and articulates open challenges related to validity, evaluation, and ethical risks. Overall, the paper argues for epistemic fidelity over surface realism as a prerequisite for using LLM-based simulated students as reliable scientific and pedagogical instruments.
>
---
#### [new 049] Closing the Modality Reasoning Gap for Speech Large Language Models
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音大模型任务，旨在解决语音输入推理性能弱于文本的问题。通过TARS框架提升语音与文本的对齐效果，显著缩小模态推理差距。**

- **链接: [https://arxiv.org/pdf/2601.05543v1](https://arxiv.org/pdf/2601.05543v1)**

> **作者:** Chaoren Wang; Heng Lu; Xueyao Zhang; Shujie Liu; Yan Lu; Jinyu Li; Zhizheng Wu
>
> **摘要:** Although speech large language models have achieved notable progress, a substantial modality reasoning gap remains: their reasoning performance on speech inputs is markedly weaker than on text. This gap could be associated with representational drift across Transformer layers and behavior deviations in long-chain reasoning. To address this issue, we introduce TARS, a reinforcement-learning framework that aligns text-conditioned and speech-conditioned trajectories through an asymmetric reward design. The framework employs two dense and complementary signals: representation alignment, which measures layer-wise hidden-state similarity between speech- and text-conditioned trajectories, and behavior alignment, which evaluates semantic consistency between generated outputs and reference text completions. Experiments on challenging reasoning benchmarks, including MMSU and OBQA, show that our approach significantly narrows the modality reasoning gap and achieves state-of-the-art performance among 7B-scale Speech LLMs.
>
---
#### [new 050] Analysing Differences in Persuasive Language in LLM-Generated Text: Uncovering Stereotypical Gender Patterns
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究LLM生成的说服性语言中的性别差异，旨在揭示模型是否产生刻板性别语言模式。**

- **链接: [https://arxiv.org/pdf/2601.05751v1](https://arxiv.org/pdf/2601.05751v1)**

> **作者:** Amalie Brogaard Pauli; Maria Barrett; Max Müller-Eberstein; Isabelle Augenstein; Ira Assent
>
> **摘要:** Large language models (LLMs) are increasingly used for everyday communication tasks, including drafting interpersonal messages intended to influence and persuade. Prior work has shown that LLMs can successfully persuade humans and amplify persuasive language. It is therefore essential to understand how user instructions affect the generation of persuasive language, and to understand whether the generated persuasive language differs, for example, when targeting different groups. In this work, we propose a framework for evaluating how persuasive language generation is affected by recipient gender, sender intent, or output language. We evaluate 13 LLMs and 16 languages using pairwise prompt instructions. We evaluate model responses on 19 categories of persuasive language using an LLM-as-judge setup grounded in social psychology and communication science. Our results reveal significant gender differences in the persuasive language generated across all models. These patterns reflect biases consistent with gender-stereotypical linguistic tendencies documented in social psychology and sociolinguistics.
>
---
#### [new 051] Tracing Moral Foundations in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究LLM如何编码道德概念。通过分析模型结构，揭示其道德判断的内在机制，验证道德基础是否在模型中存在结构化表示。**

- **链接: [https://arxiv.org/pdf/2601.05437v1](https://arxiv.org/pdf/2601.05437v1)**

> **作者:** Chenxiao Yu; Bowen Yi; Farzan Karimi-Malekabadi; Suhaib Abdurahman; Jinyi Ye; Shrikanth Narayanan; Yue Zhao; Morteza Dehghani
>
> **摘要:** Large language models (LLMs) often produce human-like moral judgments, but it is unclear whether this reflects an internal conceptual structure or superficial ``moral mimicry.'' Using Moral Foundations Theory (MFT) as an analytic framework, we study how moral foundations are encoded, organized, and expressed within two instruction-tuned LLMs: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct. We employ a multi-level approach combining (i) layer-wise analysis of MFT concept representations and their alignment with human moral perceptions, (ii) pretrained sparse autoencoders (SAEs) over the residual stream to identify sparse features that support moral concepts, and (iii) causal steering interventions using dense MFT vectors and sparse SAE features. We find that both models represent and distinguish moral foundations in a structured, layer-dependent way that aligns with human judgments. At a finer scale, SAE features show clear semantic links to specific foundations, suggesting partially disentangled mechanisms within shared representations. Finally, steering along either dense vectors or sparse features produces predictable shifts in foundation-relevant behavior, demonstrating a causal connection between internal representations and moral outputs. Together, our results provide mechanistic evidence that moral concepts in LLMs are distributed, layered, and partly disentangled, suggesting that pluralistic moral structure can emerge as a latent pattern from the statistical regularities of language alone.
>
---
#### [new 052] Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions
- **分类: cs.CL**

- **简介: 该论文属于语言模型评估任务，研究LLMs生成随机数的能力。发现LLMs在统计采样上表现不佳，需外部工具保障统计可靠性。**

- **链接: [https://arxiv.org/pdf/2601.05414v1](https://arxiv.org/pdf/2601.05414v1)**

> **作者:** Minda Zhao; Yilun Du; Mengyu Wang
>
> **摘要:** As large language models (LLMs) transition from chat interfaces to integral components of stochastic pipelines across domains like educational assessment and synthetic data construction, the ability to faithfully sample from specified probability distributions has become a functional requirement rather than a theoretical curiosity. We present the first large-scale, statistically powered audit of native probabilistic sampling in frontier LLMs, benchmarking 11 models across 15 distributions. To disentangle failure modes, we employ a dual-protocol design: Batch Generation, where a model produces N=1000 samples within one response, and Independent Requests, comprising $N=1000$ stateless calls. We observe a sharp protocol asymmetry: batch generation achieves only modest statistical validity, with a 13% median pass rate, while independent requests collapse almost entirely, with 10 of 11 models passing none of the distributions. Beyond this asymmetry, we reveal that sampling fidelity degrades monotonically with distributional complexity and aggravates as the requested sampling horizon N increases. Finally, we demonstrate the propagation of these failures into downstream tasks: models fail to enforce uniform answer-position constraints in MCQ generation and systematically violate demographic targets in attribute-constrained text-to-image prompt synthesis. These findings indicate that current LLMs lack a functional internal sampler, necessitating the use of external tools for applications requiring statistical guarantees.
>
---
#### [new 053] An Empirical Study on Preference Tuning Generalization and Diversity Under Domain Shift
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究偏好调优在领域迁移下的泛化与多样性问题，旨在解决模型在新领域性能下降的问题。通过对比不同对齐目标和适应策略，发现伪标签方法能有效缓解领域偏移带来的性能下降。**

- **链接: [https://arxiv.org/pdf/2601.05882v1](https://arxiv.org/pdf/2601.05882v1)**

> **作者:** Constantinos Karouzos; Xingwei Tan; Nikolaos Aletras
>
> **摘要:** Preference tuning aligns pretrained language models to human judgments of quality, helpfulness, or safety by optimizing over explicit preference signals rather than likelihood alone. Prior work has shown that preference-tuning degrades performance and reduces helpfulness when evaluated outside the training domain. However, the extent to which adaptation strategies mitigate this domain shift remains unexplored. We address this challenge by conducting a comprehensive and systematic study of alignment generalization under domain shift. We compare five popular alignment objectives and various adaptation strategies from source to target, including target-domain supervised fine-tuning and pseudo-labeling, across summarization and question-answering helpfulness tasks. Our findings reveal systematic differences in generalization across alignment objectives under domain shift. We show that adaptation strategies based on pseudo-labeling can substantially reduce domain-shift degradation
>
---
#### [new 054] Text Detoxification in isiXhosa and Yorùbá: A Cross-Lingual Machine Learning Approach for Low-Resource African Languages
- **分类: cs.CL**

- **简介: 该论文属于文本去毒任务，旨在解决非洲低资源语言中毒性语言检测与转换问题。通过混合机器学习与规则方法，构建了有效的毒性检测与中性重写系统。**

- **链接: [https://arxiv.org/pdf/2601.05624v1](https://arxiv.org/pdf/2601.05624v1)**

> **作者:** Abayomi O. Agbeyangi
>
> **备注:** 26 pages, 9 figures and 1 algorithm
>
> **摘要:** Toxic language is one of the major barrier to safe online participation, yet robust mitigation tools are scarce for African languages. This study addresses this critical gap by investigating automatic text detoxification (toxic to neutral rewriting) for two low-resource African languages, isiXhosa and Yorùbá. The work contributes a novel, pragmatic hybrid methodology: a lightweight, interpretable TF-IDF and Logistic Regression model for transparent toxicity detection, and a controlled lexicon- and token-guided rewriting component. A parallel corpus of toxic to neutral rewrites, which captures idiomatic usage, diacritics, and code switching, was developed to train and evaluate the model. The detection component achieved stratified K-fold accuracies of 61-72% (isiXhosa) and 72-86% (Yorùbá), with per-language ROC-AUCs up to 0.88. The rewriting component successfully detoxified all detected toxic sentences while preserving 100% of non-toxic sentences. These results demonstrate that scalable, interpretable machine learning detectors combined with rule-based edits offer a competitive and resource-efficient solution for culturally adaptive safety tooling, setting a new benchmark for low-resource Text Style Transfer (TST) in African languages.
>
---
#### [new 055] Peek2: A Regex-free implementation of pretokenizers for Byte-level BPE
- **分类: cs.CL**

- **简介: 该论文提出Peek2，一种无需正则表达式的字节级BPE预分词实现，解决传统方法效率低和依赖正则的问题，提升处理速度并保持结果一致。**

- **链接: [https://arxiv.org/pdf/2601.05833v1](https://arxiv.org/pdf/2601.05833v1)**

> **作者:** Liu Zai
>
> **备注:** 5 pages, 4 figures, for associated code, see https://github.com/omegacoleman/tokenizers_peek2
>
> **摘要:** Pretokenization is a crucial, sequential pass in Byte-level BPE tokenizers. Our proposed new implementation, Peek2, serves as a drop-in replacement for cl100k-like pretokenizers used in GPT-3, LLaMa-3, and Qwen-2.5. Designed with performance and safety in mind, Peek2 is Regex-free and delivers a $ 1.11\times $ improvement in overall throughput across the entire Byte-level BPE encoding process. This algorithm runs entirely on the CPU, has stable linear complexity $ O(n) $, and provides presegmentation results identical to those of the original Regex-based pretokenizer.
>
---
#### [new 056] Fusion Matters: Length-Aware Analysis of Positional-Encoding Fusion in Transformers
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究Transformer中位置编码与词嵌入的融合机制，针对长序列任务提出不同融合策略，验证其对性能的影响。属于自然语言处理任务，解决位置信息有效融合问题。**

- **链接: [https://arxiv.org/pdf/2601.05807v1](https://arxiv.org/pdf/2601.05807v1)**

> **作者:** Mohamed Amine Hallam; Kuo-Kun Tseng
>
> **备注:** 10 pages, 5 figures. Code and reproduction materials available on GitHub
>
> **摘要:** Transformers require positional encodings to represent sequence order, yet most prior work focuses on designing new positional encodings rather than examining how positional information is fused with token embeddings. In this paper, we study whether the fusion mechanism itself affects performance, particularly in long-sequence settings. We conduct a controlled empirical study comparing three canonical fusion strategies--element-wise addition, concatenation with projection, and scalar gated fusion--under identical Transformer architectures, data splits, and random seeds. Experiments on three text classification datasets spanning short (AG News), medium (IMDB), and long (ArXiv) sequences show that fusion choice has negligible impact on short texts but produces consistent gains on long documents. To verify that these gains are structural rather than stochastic, we perform paired-seed analysis and cross-dataset comparison across sequence-length regimes. Additional experiments on the ArXiv dataset indicate that the benefit of learnable fusion generalizes across multiple positional encoding families. Finally, we explore a lightweight convolutional gating mechanism that introduces local inductive bias at the fusion level, evaluated on long documents only. Our results indicate that positional-encoding fusion is a non-trivial design choice for long-sequence Transformers and should be treated as an explicit modeling decision rather than a fixed default.
>
---
#### [new 057] Logic-Parametric Neuro-Symbolic NLI: Controlling Logical Formalisms for Verifiable LLM Reasoning
- **分类: cs.AI; cs.CL; cs.LO**

- **简介: 该论文属于自然语言推理任务，解决LLM与TP结合时逻辑形式固定导致的适应性差问题。提出可控制逻辑形式的框架，通过对比不同逻辑效果提升推理性能与可验证性。**

- **链接: [https://arxiv.org/pdf/2601.05705v1](https://arxiv.org/pdf/2601.05705v1)**

> **作者:** Ali Farjami; Luca Redondi; Marco Valentino
>
> **备注:** Work in progress
>
> **摘要:** Large language models (LLMs) and theorem provers (TPs) can be effectively combined for verifiable natural language inference (NLI). However, existing approaches rely on a fixed logical formalism, a feature that limits robustness and adaptability. We propose a logic-parametric framework for neuro-symbolic NLI that treats the underlying logic not as a static background, but as a controllable component. Using the LogiKEy methodology, we embed a range of classical and non-classical formalisms into higher-order logic (HOL), enabling a systematic comparison of inference quality, explanation refinement, and proof behavior. We focus on normative reasoning, where the choice of logic has significant implications. In particular, we compare logic-external approaches, where normative requirements are encoded via axioms, with logic-internal approaches, where normative patterns emerge from the logic's built-in structure. Extensive experiments demonstrate that logic-internal strategies can consistently improve performance and produce more efficient hybrid proofs for NLI. In addition, we show that the effectiveness of a logic is domain-dependent, with first-order logic favouring commonsense reasoning, while deontic and modal logics excel in ethical domains. Our results highlight the value of making logic a first-class, parametric element in neuro-symbolic architectures for more robust, modular, and adaptable reasoning.
>
---
#### [new 058] PII-VisBench: Evaluating Personally Identifiable Information Safety in Vision Language Models Along a Continuum of Visibility
- **分类: cs.AI; cs.CL; cs.CR; cs.CV**

- **简介: 该论文属于隐私安全任务，旨在评估视觉语言模型中个人身份信息泄露问题。通过构建基准测试，分析不同在线可见度下的隐私保护效果，揭示模型在不同情况下的表现差异。**

- **链接: [https://arxiv.org/pdf/2601.05739v1](https://arxiv.org/pdf/2601.05739v1)**

> **作者:** G M Shahariar; Zabir Al Nazi; Md Olid Hasan Bhuiyan; Zhouxing Shi
>
> **摘要:** Vision Language Models (VLMs) are increasingly integrated into privacy-critical domains, yet existing evaluations of personally identifiable information (PII) leakage largely treat privacy as a static extraction task and ignore how a subject's online presence--the volume of their data available online--influences privacy alignment. We introduce PII-VisBench, a novel benchmark containing 4000 unique probes designed to evaluate VLM safety through the continuum of online presence. The benchmark stratifies 200 subjects into four visibility categories: high, medium, low, and zero--based on the extent and nature of their information available online. We evaluate 18 open-source VLMs (0.3B-32B) based on two key metrics: percentage of PII probing queries refused (Refusal Rate) and the fraction of non-refusal responses flagged for containing PII (Conditional PII Disclosure Rate). Across models, we observe a consistent pattern: refusals increase and PII disclosures decrease (9.10% high to 5.34% low) as subject visibility drops. We identify that models are more likely to disclose PII for high-visibility subjects, alongside substantial model-family heterogeneity and PII-type disparities. Finally, paraphrasing and jailbreak-style prompts expose attack and model-dependent failures, motivating visibility-aware safety evaluation and training interventions.
>
---
#### [new 059] Continual Pretraining on Encrypted Synthetic Data for Privacy-Preserving LLMs
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于隐私保护任务，旨在解决在敏感数据上预训练大语言模型的隐私问题。通过合成加密数据进行持续预训练，保护个人身份信息，同时保持模型性能。**

- **链接: [https://arxiv.org/pdf/2601.05635v1](https://arxiv.org/pdf/2601.05635v1)**

> **作者:** Honghao Liu; Xuhui Jiang; Chengjin Xu; Cehao Yang; Yiran Cheng; Lionel Ni; Jian Guo
>
> **摘要:** Preserving privacy in sensitive data while pretraining large language models on small, domain-specific corpora presents a significant challenge. In this work, we take an exploratory step toward privacy-preserving continual pretraining by proposing an entity-based framework that synthesizes encrypted training data to protect personally identifiable information (PII). Our approach constructs a weighted entity graph to guide data synthesis and applies deterministic encryption to PII entities, enabling LLMs to encode new knowledge through continual pretraining while granting authorized access to sensitive data through decryption keys. Our results on limited-scale datasets demonstrate that our pretrained models outperform base models and ensure PII security, while exhibiting a modest performance gap compared to models trained on unencrypted synthetic data. We further show that increasing the number of entities and leveraging graph-based synthesis improves model performance, and that encrypted models retain instruction-following capabilities with long retrieved contexts. We discuss the security implications and limitations of deterministic encryption, positioning this work as an initial investigation into the design space of encrypted data pretraining for privacy-preserving LLMs. Our code is available at https://github.com/DataArcTech/SoE.
>
---
#### [new 060] WildSci: Advancing Scientific Reasoning from In-the-Wild Literature
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于科学推理任务，旨在解决科学领域LLM训练数据不足和评估困难的问题。通过构建WildSci数据集，提升模型在科学领域的推理能力。**

- **链接: [https://arxiv.org/pdf/2601.05567v1](https://arxiv.org/pdf/2601.05567v1)**

> **作者:** Tengxiao Liu; Deepak Nathani; Zekun Li; Kevin Yang; William Yang Wang
>
> **摘要:** Recent progress in large language model (LLM) reasoning has focused on domains like mathematics and coding, where abundant high-quality data and objective evaluation metrics are readily available. In contrast, progress in LLM reasoning models remains limited in scientific domains such as medicine and materials science due to limited dataset coverage and the inherent complexity of open-ended scientific questions. To address these challenges, we introduce WildSci, a new dataset of domain-specific science questions automatically synthesized from peer-reviewed literature, covering 9 scientific disciplines and 26 subdomains. By framing complex scientific reasoning tasks in a multiple-choice format, we enable scalable training with well-defined reward signals. We further apply reinforcement learning to finetune models on these data and analyze the resulting training dynamics, including domain-specific performance changes, response behaviors, and generalization trends. Experiments on a suite of scientific benchmarks demonstrate the effectiveness of our dataset and approach. We release WildSci to enable scalable and sustainable research in scientific reasoning, available at https://huggingface.co/datasets/JustinTX/WildSci.
>
---
#### [new 061] Naiad: Novel Agentic Intelligent Autonomous System for Inland Water Monitoring
- **分类: cs.AI; cs.CL; cs.CV; cs.IR**

- **简介: 该论文提出NAIAD系统，用于内河水质监测任务，解决传统方法孤立处理问题的不足，通过整合AI与工具实现全面分析。**

- **链接: [https://arxiv.org/pdf/2601.05256v1](https://arxiv.org/pdf/2601.05256v1)**

> **作者:** Eirini Baltzi; Tilemachos Moumouris; Athena Psalta; Vasileios Tsironis; Konstantinos Karantzalos
>
> **摘要:** Inland water monitoring is vital for safeguarding public health and ecosystems, enabling timely interventions to mitigate risks. Existing methods often address isolated sub-problems such as cyanobacteria, chlorophyll, or other quality indicators separately. NAIAD introduces an agentic AI assistant that leverages Large Language Models (LLMs) and external analytical tools to deliver a holistic solution for inland water monitoring using Earth Observation (EO) data. Designed for both experts and non-experts, NAIAD provides a single-prompt interface that translates natural-language queries into actionable insights. Through Retrieval-Augmented Generation (RAG), LLM reasoning, external tool orchestration, computational graph execution, and agentic reflection, it retrieves and synthesizes knowledge from curated sources to produce tailored reports. The system integrates diverse tools for weather data, Sentinel-2 imagery, remote-sensing index computation (e.g., NDCI), chlorophyll-a estimation, and established platforms such as CyFi. Performance is evaluated using correctness and relevancy metrics, achieving over 77% and 85% respectively on a dedicated benchmark covering multiple user-expertise levels. Preliminary results show strong adaptability and robustness across query types. An ablation study on LLM backbones further highlights Gemma 3 (27B) and Qwen 2.5 (14B) as offering the best balance between computational efficiency and reasoning performance.
>
---
#### [new 062] Hi-ZFO: Hierarchical Zeroth- and First-Order LLM Fine-Tuning via Importance-Guided Tensor Selection
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Hi-ZFO框架，解决LLM微调中FO方法泛化差、ZO方法收敛慢的问题，通过分层结合FO与ZO优化提升性能并减少训练时间。**

- **链接: [https://arxiv.org/pdf/2601.05501v1](https://arxiv.org/pdf/2601.05501v1)**

> **作者:** Feihu Jin; Ying Tan
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** Fine-tuning large language models (LLMs) using standard first-order (FO) optimization often drives training toward sharp, poorly generalizing minima. Conversely, zeroth-order (ZO) methods offer stronger exploratory behavior without relying on explicit gradients, yet suffer from slow convergence. More critically, our analysis reveals that in generative tasks, the vast output and search space significantly amplify estimation variance, rendering ZO methods both noisy and inefficient. To address these challenges, we propose \textbf{Hi-ZFO} (\textbf{Hi}erarchical \textbf{Z}eroth- and \textbf{F}irst-\textbf{O}rder optimization), a hybrid framework designed to synergize the precision of FO gradients with the exploratory capability of ZO estimation. Hi-ZFO adaptively partitions the model through layer-wise importance profiling, applying precise FO updates to critical layers while leveraging ZO optimization for less sensitive ones. Notably, ZO in Hi-ZFO is not merely a memory-saving surrogate; it is intentionally introduced as a source of "beneficial stochasticity" to help the model escape the local minima where pure FO optimization tends to stagnate. Validated across diverse generative, mathematical, and code reasoning tasks, Hi-ZFO consistently achieves superior performance while significantly reducing the training time. These results demonstrate the effectiveness of hierarchical hybrid optimization for LLM fine-tuning.
>
---
#### [new 063] Conformity and Social Impact on AI Agents
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文研究AI代理在群体影响下的从众行为，属于AI安全与社会互动任务，旨在揭示AI决策中的脆弱性，通过实验分析其对群体压力的敏感性。**

- **链接: [https://arxiv.org/pdf/2601.05384v1](https://arxiv.org/pdf/2601.05384v1)**

> **作者:** Alessandro Bellina; Giordano De Marzo; David Garcia
>
> **摘要:** As AI agents increasingly operate in multi-agent environments, understanding their collective behavior becomes critical for predicting the dynamics of artificial societies. This study examines conformity, the tendency to align with group opinions under social pressure, in large multimodal language models functioning as AI agents. By adapting classic visual experiments from social psychology, we investigate how AI agents respond to group influence as social actors. Our experiments reveal that AI agents exhibit a systematic conformity bias, aligned with Social Impact Theory, showing sensitivity to group size, unanimity, task difficulty, and source characteristics. Critically, AI agents achieving near-perfect performance in isolation become highly susceptible to manipulation through social influence. This vulnerability persists across model scales: while larger models show reduced conformity on simple tasks due to improved capabilities, they remain vulnerable when operating at their competence boundary. These findings reveal fundamental security vulnerabilities in AI agent decision-making that could enable malicious manipulation, misinformation campaigns, and bias propagation in multi-agent systems, highlighting the urgent need for safeguards in collective AI deployments.
>
---
#### [new 064] Transforming User Defined Criteria into Explainable Indicators with an Integrated LLM AHP System
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文属于文本评估任务，旨在将用户自定义标准转化为可解释的指标。通过结合LLM与AHP方法，提升评估的可解释性与效率。**

- **链接: [https://arxiv.org/pdf/2601.05267v1](https://arxiv.org/pdf/2601.05267v1)**

> **作者:** Geonwoo Bang; Dongho Kim; Moohong Min
>
> **摘要:** Evaluating complex texts across domains requires converting user defined criteria into quantitative, explainable indicators, which is a persistent challenge in search and recommendation systems. Single prompt LLM evaluations suffer from complexity and latency issues, while criterion specific decomposition approaches rely on naive averaging or opaque black-box aggregation methods. We present an interpretable aggregation framework combining LLM scoring with the Analytic Hierarchy Process. Our method generates criterion specific scores via LLM as judge, measures discriminative power using Jensen Shannon distance, and derives statistically grounded weights through AHP pairwise comparison matrices. Experiments on Amazon review quality assessment and depression related text scoring demonstrate that our approach achieves high explainability and operational efficiency while maintaining comparable predictive power, making it suitable for real time latency sensitive web services.
>
---
#### [new 065] Retrieval-Augmented Multi-LLM Ensemble for Industrial Part Specification Extraction
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于工业零件规格提取任务，旨在解决从非结构化文本中自动提取准确规格的问题。通过集成多个大语言模型并结合检索增强技术，提升提取的准确性与完整性。**

- **链接: [https://arxiv.org/pdf/2601.05266v1](https://arxiv.org/pdf/2601.05266v1)**

> **作者:** Muzakkiruddin Ahmed Mohammed; John R. Talburt; Leon Claasssens; Adriaan Marais
>
> **备注:** The 17th International Conference on Knowledge and Systems Engineering
>
> **摘要:** Industrial part specification extraction from unstructured text remains a persistent challenge in manufacturing, procurement, and maintenance, where manual processing is both time-consuming and error-prone. This paper introduces a retrieval-augmented multi-LLM ensemble framework that orchestrates nine state-of-the-art Large Language Models (LLMs) within a structured three-phase pipeline. RAGsemble addresses key limitations of single-model systems by combining the complementary strengths of model families including Gemini (2.0, 2.5, 1.5), OpenAI (GPT-4o, o4-mini), Mistral Large, and Gemma (1B, 4B, 3n-e4b), while grounding outputs in factual data using FAISS-based semantic retrieval. The system architecture consists of three stages: (1) parallel extraction by diverse LLMs, (2) targeted research augmentation leveraging high-performing models, and (3) intelligent synthesis with conflict resolution and confidence-aware scoring. RAG integration provides real-time access to structured part databases, enabling the system to validate, refine, and enrich outputs through similarity-based reference retrieval. Experimental results using real industrial datasets demonstrate significant gains in extraction accuracy, technical completeness, and structured output quality compared to leading single-LLM baselines. Key contributions include a scalable ensemble architecture for industrial domains, seamless RAG integration throughout the pipeline, comprehensive quality assessment mechanisms, and a production-ready solution suitable for deployment in knowledge-intensive manufacturing environments.
>
---
#### [new 066] TIME: Temporally Intelligent Meta-reasoning Engine for Context Triggered Explicit Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出TIME框架，解决对话模型对时间结构感知不足的问题。通过引入时间标签和思考块，增强模型的时空推理能力，提升对话连贯性与效率。**

- **链接: [https://arxiv.org/pdf/2601.05300v1](https://arxiv.org/pdf/2601.05300v1)**

> **作者:** Susmit Das
>
> **备注:** 14 pages, 3 figures with 27 page appendix. See https://github.com/The-Coherence-Initiative/TIME and https://github.com/The-Coherence-Initiative/TIMEBench for associated code
>
> **摘要:** Reasoning oriented large language models often expose explicit "thinking" as long, turn-global traces at the start of every response, either always on or toggled externally at inference time. While useful for arithmetic, programming, and problem solving, this design is costly, blurs claim level auditability, and cannot re-trigger explicit reasoning once the model begins presenting. Dialogue models are also largely blind to temporal structure, treating replies after seconds and replies after weeks as equivalent unless time is stated in text. We introduce TIME, the Temporally Intelligent Meta-reasoning Engine, a behavioral alignment framework that treats explicit reasoning as a context sensitive resource driven by discourse and temporal cues. TIME augments dialogue with optional ISO 8601 <time> tags, tick turns that represent silent gaps, and short <think> blocks that can appear anywhere in a reply. A four-phase curriculum including a small, maximally diverse full-batch alignment step trains Qwen3 dense models to invoke brief, in-place reasoning bursts and keep user facing text compact. We evaluate with TIMEBench, a temporally grounded dialogue benchmark probing chronology, commonsense under gaps and offsets, anomaly detection, and continuity. Across 4B to 32B scales, TIME improves TIMEBench scores over base Qwen3 in both thinking and no-thinking modes while reducing reasoning tokens by about an order of magnitude. Our training data and code are available at https://github.com/The-Coherence-Initiative/TIME and TIMEBench is available at https://github.com/The-Coherence-Initiative/TIMEBench
>
---
#### [new 067] The Persona Paradox: Medical Personas as Behavioral Priors in Clinical Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于医疗语言模型研究，探讨医学角色设定对临床决策的影响。工作包括系统评估不同职业角色和互动风格对模型表现的影响，揭示其在不同场景下的效果差异。**

- **链接: [https://arxiv.org/pdf/2601.05376v1](https://arxiv.org/pdf/2601.05376v1)**

> **作者:** Tassallah Abdullahi; Shrestha Ghosh; Hamish S Fraser; Daniel León Tramontini; Adeel Abbasi; Ghada Bourjeily; Carsten Eickhoff; Ritambhara Singh
>
> **摘要:** Persona conditioning can be viewed as a behavioral prior for large language models (LLMs) and is often assumed to confer expertise and improve safety in a monotonic manner. However, its effects on high-stakes clinical decision-making remain poorly characterized. We systematically evaluate persona-based control in clinical LLMs, examining how professional roles (e.g., Emergency Department physician, nurse) and interaction styles (bold vs.\ cautious) influence behavior across models and medical tasks. We assess performance on clinical triage and patient-safety tasks using multidimensional evaluations that capture task accuracy, calibration, and safety-relevant risk behavior. We find systematic, context-dependent, and non-monotonic effects: Medical personas improve performance in critical care tasks, yielding gains of up to $\sim+20\%$ in accuracy and calibration, but degrade performance in primary-care settings by comparable margins. Interaction style modulates risk propensity and sensitivity, but it's highly model-dependent. While aggregated LLM-judge rankings favor medical over non-medical personas in safety-critical cases, we found that human clinicians show moderate agreement on safety compliance (average Cohen's $κ= 0.43$) but indicate a low confidence in 95.9\% of their responses on reasoning quality. Our work shows that personas function as behavioral priors that introduce context-dependent trade-offs rather than guarantees of safety or expertise. The code is available at https://github.com/rsinghlab/Persona\_Paradox.
>
---
#### [new 068] Cross-Document Topic-Aligned Chunking for Retrieval-Augmented Generation
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，解决知识碎片化问题。通过跨文档主题对齐分块，提升检索增强生成系统的准确性与效率。**

- **链接: [https://arxiv.org/pdf/2601.05265v1](https://arxiv.org/pdf/2601.05265v1)**

> **作者:** Mile Stankovic
>
> **摘要:** Chunking quality determines RAG system performance. Current methods partition documents individually, but complex queries need information scattered across multiple sources: the knowledge fragmentation problem. We introduce Cross-Document Topic-Aligned (CDTA) chunking, which reconstructs knowledge at the corpus level. It first identifies topics across documents, maps segments to each topic, and synthesizes them into unified chunks. On HotpotQA multi-hop reasoning, our method reached 0.93 faithfulness versus 0.83 for contextual retrieval and 0.78 for semantic chunking, a 12% improvement over current industry best practice (p < 0.05). On UAE Legal texts, it reached 0.94 faithfulness with 0.93 citation accuracy. At k = 3, it maintains 0.91 faithfulness while semantic methods drop to 0.68, with a single CDTA chunk containing information requiring multiple traditional fragments. Indexing costs are higher, but synthesis produces information-dense chunks that reduce query-time retrieval needs. For high-query-volume applications with distributed knowledge, cross-document synthesis improves measurably over within-document optimization.
>
---
#### [new 069] SceneAlign: Aligning Multimodal Reasoning to Scene Graphs in Complex Visual Scenes
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于多模态推理任务，旨在解决复杂视觉场景中推理不准确的问题。通过构建结构化干预，提升模型对视觉信息的精准理解与推理能力。**

- **链接: [https://arxiv.org/pdf/2601.05600v1](https://arxiv.org/pdf/2601.05600v1)**

> **作者:** Chuhan Wang; Xintong Li; Jennifer Yuntong Zhang; Junda Wu; Chengkai Huang; Lina Yao; Julian McAuley; Jingbo Shang
>
> **备注:** Preprint
>
> **摘要:** Multimodal large language models often struggle with faithful reasoning in complex visual scenes, where intricate entities and relations require precise visual grounding at each step. This reasoning unfaithfulness frequently manifests as hallucinated entities, mis-grounded relations, skipped steps, and over-specified reasoning. Existing preference-based approaches, typically relying on textual perturbations or answer-conditioned rationales, fail to address this challenge as they allow models to exploit language priors to bypass visual grounding. To address this, we propose SceneAlign, a framework that leverages scene graphs as structured visual information to perform controllable structural interventions. By identifying reasoning-critical nodes and perturbing them through four targeted strategies that mimic typical grounding failures, SceneAlign constructs hard negative rationales that remain linguistically plausible but are grounded in inaccurate visual facts. These contrastive pairs are used in Direct Preference Optimization to steer models toward fine-grained, structure-faithful reasoning. Across seven visual reasoning benchmarks, SceneAlign consistently improves answer accuracy and reasoning faithfulness, highlighting the effectiveness of grounding-aware alignment for multimodal reasoning.
>
---
#### [new 070] The ICASSP 2026 HumDial Challenge: Benchmarking Human-like Spoken Dialogue Systems in the LLM Era
- **分类: cs.SD; cs.CL; cs.HC; eess.AS**

- **简介: 该论文属于人机对话系统任务，旨在提升对话系统的拟人化水平。解决情感理解与实时交互问题，通过构建数据集和两个评估赛道进行基准测试。**

- **链接: [https://arxiv.org/pdf/2601.05564v1](https://arxiv.org/pdf/2601.05564v1)**

> **作者:** Zhixian Zhao; Shuiyuan Wang; Guojian Li; Hongfei Xue; Chengyou Wang; Shuai Wang; Longshuai Xiao; Zihan Zhang; Hui Bu; Xin Xu; Xinsheng Wang; Hexin Liu; Eng Siong Chng; Hung-yi Lee; Haizhou Li; Lei Xie
>
> **备注:** Official summary paper for the ICASSP 2026 HumDial Challenge
>
> **摘要:** Driven by the rapid advancement of Large Language Models (LLMs), particularly Audio-LLMs and Omni-models, spoken dialogue systems have evolved significantly, progressively narrowing the gap between human-machine and human-human interactions. Achieving truly ``human-like'' communication necessitates a dual capability: emotional intelligence to perceive and resonate with users' emotional states, and robust interaction mechanisms to navigate the dynamic, natural flow of conversation, such as real-time turn-taking. Therefore, we launched the first Human-like Spoken Dialogue Systems Challenge (HumDial) at ICASSP 2026 to benchmark these dual capabilities. Anchored by a sizable dataset derived from authentic human conversations, this initiative establishes a fair evaluation platform across two tracks: (1) Emotional Intelligence, targeting long-term emotion understanding and empathetic generation; and (2) Full-Duplex Interaction, systematically evaluating real-time decision-making under `` listening-while-speaking'' conditions. This paper summarizes the dataset, track configurations, and the final results.
>
---
#### [new 071] RingSQL: Generating Synthetic Data with Schema-Independent Templates for Text-to-SQL Reasoning Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于文本到SQL任务，旨在解决高质量训练数据稀缺问题。提出RingSQL框架，结合通用模板与大模型生成，提升数据质量和多样性。**

- **链接: [https://arxiv.org/pdf/2601.05451v1](https://arxiv.org/pdf/2601.05451v1)**

> **作者:** Marko Sterbentz; Kevin Cushing; Cameron Barrie; Kristian J. Hammond
>
> **摘要:** Recent advances in text-to-SQL systems have been driven by larger models and improved datasets, yet progress is still limited by the scarcity of high-quality training data. Manual data creation is expensive, and existing synthetic methods trade off reliability and scalability. Template-based approaches ensure correct SQL but require schema-specific templates, while LLM-based generation scales easily but lacks quality and correctness guarantees. We introduce RingSQL, a hybrid data generation framework that combines schema-independent query templates with LLM-based paraphrasing of natural language questions. This approach preserves SQL correctness across diverse schemas while providing broad linguistic variety. In our experiments, we find that models trained using data produced by RingSQL achieve an average gain in accuracy of +2.3% across six text-to-SQL benchmarks when compared to models trained on other synthetic data. We make our code available at https://github.com/nu-c3lab/RingSQL.
>
---
#### [new 072] LLM2IR: simple unsupervised contrastive learning makes long-context LLM great retriever
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决传统IR模型依赖大规模预训练的问题。提出LLM2IR框架，通过无监督对比学习将解码器型大语言模型转化为高效检索模型。**

- **链接: [https://arxiv.org/pdf/2601.05262v1](https://arxiv.org/pdf/2601.05262v1)**

> **作者:** Xiaocong Yang
>
> **备注:** MS Thesis
>
> **摘要:** Modern dense information retrieval (IR) models usually rely on costly large-scale pretraining. In this paper, we introduce LLM2IR, an efficient unsupervised contrastive learning framework to convert any decoder-only large language model (LLM) to an information retrieval model. Despite its simplicity, the effectiveness is proven among different LLMs on multiple IR benchmarks including LoCo, LongEmbed and BEIR. We also find that models with a longer context length tend to have a stronger IR capacity by comparing task performances of models in the same model family. Our work not only provides an effective way to build IR models on the state-of-the-art LLMs, but also shed light on the relationship between information retrieval ability and model context length, which helps the design of better information retrievers.
>
---
#### [new 073] Weights to Code: Extracting Interpretable Algorithms from the Discrete Transformer
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于算法提取任务，旨在解决Transformer模型难以提取可解释程序的问题。通过设计Discrete Transformer，实现连续表示与离散逻辑的衔接，提升模型的可解释性。**

- **链接: [https://arxiv.org/pdf/2601.05770v1](https://arxiv.org/pdf/2601.05770v1)**

> **作者:** Yifan Zhang; Wei Bi; Kechi Zhang; Dongming Jin; Jie Fu; Zhi Jin
>
> **摘要:** Algorithm extraction aims to synthesize executable programs directly from models trained on specific algorithmic tasks, enabling de novo algorithm discovery without relying on human-written code. However, extending this paradigm to Transformer is hindered by superposition, where entangled features encoded in overlapping directions obstruct the extraction of symbolic expressions. In this work, we propose the Discrete Transformer, an architecture explicitly engineered to bridge the gap between continuous representations and discrete symbolic logic. By enforcing a strict functional disentanglement, which constrains Numerical Attention to information routing and Numerical MLP to element-wise arithmetic, and employing temperature-annealed sampling, our method effectively facilitates the extraction of human-readable programs. Empirically, the Discrete Transformer not only achieves performance comparable to RNN-based baselines but crucially extends interpretability to continuous variable domains. Moreover, our analysis of the annealing process shows that the efficient discrete search undergoes a clear phase transition from exploration to exploitation. We further demonstrate that our method enables fine-grained control over synthesized programs by imposing inductive biases. Collectively, these findings establish the Discrete Transformer as a robust framework for demonstration-free algorithm discovery, offering a rigorous pathway toward Transformer interpretability.
>
---
#### [new 074] ROAP: A Reading-Order and Attention-Prior Pipeline for Optimizing Layout Transformers in Key Information Extraction
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文档理解任务，旨在解决Layout Transformers在关键信息提取中的阅读顺序缺失和视觉干扰问题。提出ROAP管道，优化注意力分布，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.05470v1](https://arxiv.org/pdf/2601.05470v1)**

> **作者:** Tingwei Xie; Jinxin He; Yonghong Song
>
> **备注:** 10 pages, 4 figures, 4 tables
>
> **摘要:** The efficacy of Multimodal Transformers in visually-rich document understanding (VrDU) is critically constrained by two inherent limitations: the lack of explicit modeling for logical reading order and the interference of visual tokens that dilutes attention on textual semantics. To address these challenges, this paper presents ROAP, a lightweight and architecture-agnostic pipeline designed to optimize attention distributions in Layout Transformers without altering their pre-trained backbones. The proposed pipeline first employs an Adaptive-XY-Gap (AXG-Tree) to robustly extract hierarchical reading sequences from complex layouts. These sequences are then integrated into the attention mechanism via a Reading-Order-Aware Relative Position Bias (RO-RPB). Furthermore, a Textual-Token Sub-block Attention Prior (TT-Prior) is introduced to adaptively suppress visual noise and enhance fine-grained text-text interactions. Extensive experiments on the FUNSD and CORD benchmarks demonstrate that ROAP consistently improves the performance of representative backbones, including LayoutLMv3 and GeoLayoutLM. These findings confirm that explicitly modeling reading logic and regulating modality interference are critical for robust document understanding, offering a scalable solution for complex layout analysis. The implementation code will be released at https://github.com/KevinYuLei/ROAP.
>
---
#### [new 075] Enabling Stroke-Level Structural Analysis of Hieroglyphic Scripts without Language-Specific Priors
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于字符结构分析任务，旨在解决LLMs对象形文字结构感知不足的问题。提出HieroSA框架，自动提取字符笔画结构，无需语言先验知识。**

- **链接: [https://arxiv.org/pdf/2601.05508v1](https://arxiv.org/pdf/2601.05508v1)**

> **作者:** Fuwen Luo; Zihao Wan; Ziyue Wang; Yaluo Liu; Pau Tong Lin Xu; Xuanjia Qiao; Xiaolong Wang; Peng Li; Yang Liu
>
> **摘要:** Hieroglyphs, as logographic writing systems, encode rich semantic and cultural information within their internal structural composition. Yet, current advanced Large Language Models (LLMs) and Multimodal LLMs (MLLMs) usually remain structurally blind to this information. LLMs process characters as textual tokens, while MLLMs additionally view them as raw pixel grids. Both fall short to model the underlying logic of character strokes. Furthermore, existing structural analysis methods are often script-specific and labor-intensive. In this paper, we propose Hieroglyphic Stroke Analyzer (HieroSA), a novel and generalizable framework that enables MLLMs to automatically derive stroke-level structures from character bitmaps without handcrafted data. It transforms modern logographic and ancient hieroglyphs character images into explicit, interpretable line-segment representations in a normalized coordinate space, allowing for cross-lingual generalization. Extensive experiments demonstrate that HieroSA effectively captures character-internal structures and semantics, bypassing the need for language-specific priors. Experimental results highlight the potential of our work as a graphematics analysis tool for a deeper understanding of hieroglyphic scripts. View our code at https://github.com/THUNLP-MT/HieroSA.
>
---
#### [new 076] MMViR: A Multi-Modal and Multi-Granularity Representation for Long-range Video Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出MMViR，解决长视频理解任务中的多模态表示问题，通过多粒度结构提升效率与效果。**

- **链接: [https://arxiv.org/pdf/2601.05495v1](https://arxiv.org/pdf/2601.05495v1)**

> **作者:** Zizhong Li; Haopeng Zhang; Jiawei Zhang
>
> **备注:** 13 pages, 11 figures
>
> **摘要:** Long videos, ranging from minutes to hours, present significant challenges for current Multi-modal Large Language Models (MLLMs) due to their complex events, diverse scenes, and long-range dependencies. Direct encoding of such videos is computationally too expensive, while simple video-to-text conversion often results in redundant or fragmented content. To address these limitations, we introduce MMViR, a novel multi-modal, multi-grained structured representation for long video understanding. MMViR identifies key turning points to segment the video and constructs a three-level description that couples global narratives with fine-grained visual details. This design supports efficient query-based retrieval and generalizes well across various scenarios. Extensive evaluations across three tasks, including QA, summarization, and retrieval, show that MMViR outperforms the prior strongest method, achieving a 19.67% improvement in hour-long video understanding while reducing processing latency to 45.4% of the original.
>
---
#### [new 077] MaxCode: A Max-Reward Reinforcement Learning Framework for Automated Code Optimization
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于代码优化任务，解决LLM在编写高效代码时的挑战。提出MaxCode框架，通过强化学习和执行反馈提升代码性能。**

- **链接: [https://arxiv.org/pdf/2601.05475v1](https://arxiv.org/pdf/2601.05475v1)**

> **作者:** Jiefu Ou; Sapana Chaudhary; Kaj Bostrom; Nathaniel Weir; Shuai Zhang; Huzefa Rangwala; George Karypis
>
> **摘要:** Large Language Models (LLMs) demonstrate strong capabilities in general coding tasks but encounter two key challenges when optimizing code: (i) the complexity of writing optimized code (such as performant CUDA kernels and competition-level CPU code) requires expertise in systems, algorithms and specific languages and (ii) requires interpretation of performance metrics like timing and device utilization beyond binary correctness. In this work, we explore inference-time search algorithms that guide the LLM to discover better solutions through iterative refinement based on execution feedback. Our approach, called MaxCode unifies existing search methods under a max-reward reinforcement learning framework, making the observation and action-value functions modular for modification. To enhance the observation space, we integrate a natural language critique model that converts raw execution feedback into diagnostic insights about errors and performance bottlenecks, and the best-discounted reward seen so far. Together, these provide richer input to the code proposal function. To improve exploration during search, we train a generative reward-to-go model using action values from rollouts to rerank potential solutions. Testing on the KernelBench (CUDA) and PIE (C++) optimization benchmarks shows that MaxCode improves optimized code performance compared to baselines, achieving 20.3% and 10.1% relative improvements in absolute speedup value and relative speedup ranking, respectively.
>
---
#### [new 078] TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出TagRAG，解决传统RAG在知识检索与生成中的效率和适应性问题，通过构建标签图实现高效全局推理。**

- **链接: [https://arxiv.org/pdf/2601.05254v1](https://arxiv.org/pdf/2601.05254v1)**

> **作者:** Wenbiao Tao; Yunshi Lan; Weining Qian
>
> **摘要:** Retrieval-Augmented Generation enhances language models by retrieving external knowledge to support informed and grounded responses. However, traditional RAG methods rely on fragment-level retrieval, limiting their ability to address query-focused summarization queries. GraphRAG introduces a graph-based paradigm for global knowledge reasoning, yet suffers from inefficiencies in information extraction, costly resource consumption, and poor adaptability to incremental updates. To overcome these limitations, we propose TagRAG, a tag-guided hierarchical knowledge graph RAG framework designed for efficient global reasoning and scalable graph maintenance. TagRAG introduces two key components: (1) Tag Knowledge Graph Construction, which extracts object tags and their relationships from documents and organizes them into hierarchical domain tag chains for structured knowledge representation, and (2) Tag-Guided Retrieval-Augmented Generation, which retrieves domain-centric tag chains to localize and synthesize relevant knowledge during inference. This design significantly adapts to smaller language models, improves retrieval granularity, and supports efficient knowledge increment. Extensive experiments on UltraDomain datasets spanning Agriculture, Computer Science, Law, and cross-domain settings demonstrate that TagRAG achieves an average win rate of 95.41\% against baselines while maintaining about 14.6x construction and 1.9x retrieval efficiency compared with GraphRAG.
>
---
#### [new 079] Open World Knowledge Aided Single-Cell Foundation Model with Robust Cross-Modal Cell-Language Pre-training
- **分类: q-bio.GN; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于单细胞数据分析任务，旨在解决多模态数据噪声和个体特征整合不足的问题。提出OKR-CELL模型，通过跨模态预训练和鲁棒对齐机制提升性能。**

- **链接: [https://arxiv.org/pdf/2601.05648v1](https://arxiv.org/pdf/2601.05648v1)**

> **作者:** Haoran Wang; Xuanyi Zhang; Shuangsang Fang; Longke Ran; Ziqing Deng; Yong Zhang; Yuxiang Li; Shaoshuai Li
>
> **备注:** 41 pages
>
> **摘要:** Recent advancements in single-cell multi-omics, particularly RNA-seq, have provided profound insights into cellular heterogeneity and gene regulation. While pre-trained language model (PLM) paradigm based single-cell foundation models have shown promise, they remain constrained by insufficient integration of in-depth individual profiles and neglecting the influence of noise within multi-modal data. To address both issues, we propose an Open-world Language Knowledge-Aided Robust Single-Cell Foundation Model (OKR-CELL). It is built based on a cross-modal Cell-Language pre-training framework, which comprises two key innovations: (1) leveraging Large Language Models (LLMs) based workflow with retrieval-augmented generation (RAG) enriches cell textual descriptions using open-world knowledge; (2) devising a Cross-modal Robust Alignment (CRA) objective that incorporates sample reliability assessment, curriculum learning, and coupled momentum contrastive learning to strengthen the model's resistance to noisy data. After pretraining on 32M cell-text pairs, OKR-CELL obtains cutting-edge results across 6 evaluation tasks. Beyond standard benchmarks such as cell clustering, cell-type annotation, batch-effect correction, and few-shot annotation, the model also demonstrates superior performance in broader multi-modal applications, including zero-shot cell-type annotation and bidirectional cell-text retrieval.
>
---
#### [new 080] Quantifying Document Impact in RAG-LLMs
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于RAG系统评估任务，旨在解决文档贡献度量化问题。提出影响得分（IS）以衡量单个文档对生成结果的影响，提升系统透明度与可靠性。**

- **链接: [https://arxiv.org/pdf/2601.05260v1](https://arxiv.org/pdf/2601.05260v1)**

> **作者:** Armin Gerami; Kazem Faghih; Ramani Duraiswami
>
> **摘要:** Retrieval Augmented Generation (RAG) enhances Large Language Models (LLMs) by connecting them to external knowledge, improving accuracy and reducing outdated information. However, this introduces challenges such as factual inconsistencies, source conflicts, bias propagation, and security vulnerabilities, which undermine the trustworthiness of RAG systems. A key gap in current RAG evaluation is the lack of a metric to quantify the contribution of individual retrieved documents to the final output. To address this, we introduce the Influence Score (IS), a novel metric based on Partial Information Decomposition that measures the impact of each retrieved document on the generated response. We validate IS through two experiments. First, a poison attack simulation across three datasets demonstrates that IS correctly identifies the malicious document as the most influential in $86\%$ of cases. Second, an ablation study shows that a response generated using only the top-ranked documents by IS is consistently judged more similar to the original response than one generated from the remaining documents. These results confirm the efficacy of IS in isolating and quantifying document influence, offering a valuable tool for improving the transparency and reliability of RAG systems.
>
---
#### [new 081] Thinking with Map: Reinforced Parallel Map-Augmented Agent for Geolocalization
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于图像地理定位任务，旨在提升模型通过视觉线索预测图像拍摄位置的能力。提出一种结合地图的强化代理方法，通过两阶段优化提高定位精度。**

- **链接: [https://arxiv.org/pdf/2601.05432v1](https://arxiv.org/pdf/2601.05432v1)**

> **作者:** Yuxiang Ji; Yong Wang; Ziyu Ma; Yiming Hu; Hailang Huang; Xuecai Hu; Guanhua Chen; Liaoni Wu; Xiangxiang Chu
>
> **摘要:** The image geolocalization task aims to predict the location where an image was taken anywhere on Earth using visual clues. Existing large vision-language model (LVLM) approaches leverage world knowledge, chain-of-thought reasoning, and agentic capabilities, but overlook a common strategy used by humans -- using maps. In this work, we first equip the model \textit{Thinking with Map} ability and formulate it as an agent-in-the-map loop. We develop a two-stage optimization scheme for it, including agentic reinforcement learning (RL) followed by parallel test-time scaling (TTS). The RL strengthens the agentic capability of model to improve sampling efficiency, and the parallel TTS enables the model to explore multiple candidate paths before making the final prediction, which is crucial for geolocalization. To evaluate our method on up-to-date and in-the-wild images, we further present MAPBench, a comprehensive geolocalization training and evaluation benchmark composed entirely of real-world images. Experimental results show that our method outperforms existing open- and closed-source models on most metrics, specifically improving Acc@500m from 8.0\% to 22.1\% compared to \textit{Gemini-3-Pro} with Google Search/Map grounded mode.
>
---
#### [new 082] RISE: Rule-Driven SQL Dialect Translation via Query Reduction
- **分类: cs.DB; cs.AI; cs.CL; cs.SE**

- **简介: 该论文属于SQL方言翻译任务，解决复杂SQL查询在不同数据库系统间准确转换的问题。提出RISE方法，通过查询简化和规则提取提升翻译准确性。**

- **链接: [https://arxiv.org/pdf/2601.05579v1](https://arxiv.org/pdf/2601.05579v1)**

> **作者:** Xudong Xie; Yuwei Zhang; Wensheng Dou; Yu Gao; Ziyu Cui; Jiansen Song; Rui Yang; Jun Wei
>
> **备注:** Accepted by ICSE 2026
>
> **摘要:** Translating SQL dialects across different relational database management systems (RDBMSs) is crucial for migrating RDBMS-based applications to the cloud. Traditional SQL dialect translation tools rely on manually-crafted rules, necessitating significant manual effort to support new RDBMSs and dialects. Although large language models (LLMs) can assist in translating SQL dialects, they often struggle with lengthy and complex SQL queries. In this paper, we propose RISE, a novel LLM-based SQL dialect translation approach that can accurately handle lengthy and complex SQL queries. Given a complex source query $Q_c$ that contains a SQL dialect $d$, we first employ a dialect-aware query reduction technique to derive a simplified query $Q_{s}$ by removing $d$-irrelevant SQL elements from $Q_c$. Subsequently, we utilize LLMs to translate $Q_{s}$ into $Q_{s^{'}}$, and automatically extract the translation rule $r_d$ for dialect $d$ based on the relationship between $Q_{s}$ and $Q_{s^{'}}$. By applying $r_d$ to $Q_c$, we can effectively translate the dialect $d$ within $Q_c$, thereby bypassing the complexity of the source query $Q_c$. We evaluate RISE on two real-world benchmarks, i.e., TPC-DS and SQLProcBench, comparing its performance against both the traditional rule-based tools and the LLM-based approaches with respect to translation accuracy. RISE achieves accuracies of 97.98% on TPC-DS and 100% on SQLProcBench, outperforming the baselines by an average improvement of 24.62% and 238.41%, respectively.
>
---
## 更新

#### [replaced 001] Bridging External and Parametric Knowledge: Mitigating Hallucination of LLMs with Shared-Private Semantic Synergy in Dual-Stream Knowledge
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型的幻觉问题。通过引入双流知识增强框架，区分共享与私有语义，提升知识融合效果。**

- **链接: [https://arxiv.org/pdf/2506.06240v2](https://arxiv.org/pdf/2506.06240v2)**

> **作者:** Yi Sui; Chaozhuo Li; Chen Zhang; Dawei song; Qiuchi Li
>
> **摘要:** Retrieval-augmented generation (RAG) aims to mitigate the hallucination of Large Language Models (LLMs) by retrieving and incorporating relevant external knowledge into the generation process. However, the external knowledge may contain noise and conflict with the parametric knowledge of LLMs, leading to degraded performance. Current LLMs lack inherent mechanisms for resolving such conflicts. To fill this gap, we propose a Dual-Stream Knowledge-Augmented Framework for Shared-Private Semantic Synergy (DSSP-RAG). Central to it is the refinement of the traditional self-attention into a mixed-attention that distinguishes shared and private semantics for a controlled knowledge integration. An unsupervised hallucination detection method that captures the LLMs' intrinsic cognitive uncertainty ensures that external knowledge is introduced only when necessary. To reduce noise in external knowledge, an Energy Quotient (EQ), defined by attention difference matrices between task-aligned and task-misaligned layers, is proposed. Extensive experiments show that DSSP-RAG achieves a superior performance over strong baselines.
>
---
#### [replaced 002] On the Emergence of Induction Heads for In-Context Learning
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究Transformer模型中用于上下文学习的归纳头机制，旨在理解其形成过程。通过理论分析与实验验证，揭示了参数空间中的结构约束及训练动态规律。**

- **链接: [https://arxiv.org/pdf/2511.01033v2](https://arxiv.org/pdf/2511.01033v2)**

> **作者:** Tiberiu Musat; Tiago Pimentel; Lorenzo Noci; Alessandro Stolfo; Mrinmaya Sachan; Thomas Hofmann
>
> **摘要:** Transformers have become the dominant architecture for natural language processing. Part of their success is owed to a remarkable capability known as in-context learning (ICL): they can acquire and apply novel associations solely from their input context, without any updates to their weights. In this work, we study the emergence of induction heads, a previously identified mechanism in two-layer transformers that is particularly important for in-context learning. We uncover a relatively simple and interpretable structure of the weight matrices implementing the induction head. We theoretically explain the origin of this structure using a minimal ICL task formulation and a modified transformer architecture. We give a formal proof that the training dynamics remain constrained to a 19-dimensional subspace of the parameter space. Empirically, we validate this constraint while observing that only 3 dimensions account for the emergence of an induction head. By further studying the training dynamics inside this 3-dimensional subspace, we find that the time until the emergence of an induction head follows a tight asymptotic bound that is quadratic in the input context length.
>
---
#### [replaced 003] UPDESH: Synthesizing Grounded Instruction Tuning Data for 13 Indic
- **分类: cs.CL**

- **简介: 该论文属于多语言AI任务，旨在解决低资源语言文化背景不足的问题。通过生成高质量的合成数据集Updesh，提升多语言模型性能。**

- **链接: [https://arxiv.org/pdf/2509.21294v2](https://arxiv.org/pdf/2509.21294v2)**

> **作者:** Pranjal A. Chitale; Varun Gumma; Sanchit Ahuja; Prashant Kodali; Manan Uppadhyay; Deepthi Sudharsan; Sunayana Sitaram
>
> **备注:** Under Review
>
> **摘要:** Developing culturally grounded multilingual AI systems remains challenging, particularly for low-resource languages. While synthetic data offers promise, its effectiveness in multilingual and multicultural contexts is underexplored. We investigate bottom-up synthetic data generation using large open-source LLMs (>= 235B parameters) grounded in language-specific Wikipedia content, complementing dominant top-down translation-based approaches from English. We introduce Updesh, a high-quality large-scale synthetic instruction-following dataset comprising 9.5M data points across 13 Indian languages and English, encompassing diverse reasoning and generative tasks. Comprehensive evaluation using automated metrics and 10K human assessments confirms high data quality. Downstream evaluations performed by fine-tuning models on various datasets and assessing performance across 13 diverse multilingual datasets and model comparative evaluations, demonstrate that models trained on Updesh consistently obtain significant improvements on NLU, NLG evaluations. Finally, through ablation studies and cultural evaluations, we show that context-aware, culturally grounded data generation is essential for effective multilingual AI development .
>
---
#### [replaced 004] SPEC-RL: Accelerating On-Policy Reinforcement Learning via Speculative Rollouts
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决RLVR训练中rollout阶段计算效率低的问题。通过引入推测性采样，重用历史轨迹片段，提升效率并保持策略一致性。**

- **链接: [https://arxiv.org/pdf/2509.23232v2](https://arxiv.org/pdf/2509.23232v2)**

> **作者:** Bingshuai Liu; Ante Wang; Zijun Min; Liang Yao; Haibo Zhang; Yang Liu; Anxiang Zeng; Jinsong Su
>
> **备注:** 32 pages, fixed typos
>
> **摘要:** Large Language Models (LLMs) increasingly rely on reinforcement learning with verifiable rewards (RLVR) to elicit reliable chain-of-thought reasoning. However, the training process remains bottlenecked by the computationally expensive rollout stage. Existing acceleration methods-such as parallelization, objective- and data-driven modifications, and replay buffers-either incur diminishing returns, introduce bias, or overlook redundancy across iterations. We identify that rollouts from consecutive training epochs frequently share a large portion of overlapping segments, wasting computation. To address this, we propose SPEC-RL, a novel framework that integrates SPECulative decoding with the RL rollout process. SPEC-RL reuses prior trajectory segments as speculative prefixes and extends them via a draft-and-verify mechanism, avoiding redundant generation while ensuring policy consistency. Experiments on diverse math reasoning and generalization benchmarks, including AIME24, MATH-500, OlympiadBench, MMLU-STEM, and others, demonstrate that SPEC-RL reduces rollout time by 2-3x without compromising policy quality. As a purely rollout-stage enhancement, SPEC-RL integrates seamlessly with mainstream algorithms (e.g., PPO, GRPO, DAPO), offering a general and practical path to scale RLVR for large reasoning models. Our code is available at https://github.com/ShopeeLLM/Spec-RL
>
---
#### [replaced 005] Monadic Context Engineering
- **分类: cs.AI; cs.CL; cs.FL**

- **简介: 该论文提出Monadic Context Engineering（MCE），解决AI代理架构的脆弱性问题，通过函数式编程结构提升状态管理、错误处理和并发能力。**

- **链接: [https://arxiv.org/pdf/2512.22431v3](https://arxiv.org/pdf/2512.22431v3)**

> **作者:** Yifan Zhang; Yang Yuan; Mengdi Wang; Andrew Chi-Chih Yao
>
> **备注:** The authors have decided to withdraw this manuscript, as the ideas presented in the paper are not yet sufficiently mature and require further development and refinement
>
> **摘要:** The proliferation of Large Language Models (LLMs) has catalyzed a shift towards autonomous agents capable of complex reasoning and tool use. However, current agent architectures are frequently constructed using imperative, ad hoc patterns. This results in brittle systems plagued by difficulties in state management, error handling, and concurrency. This paper introduces Monadic Context Engineering (MCE), a novel architectural paradigm leveraging the algebraic structures of Functors, Applicative Functors, and Monads to provide a formal foundation for agent design. MCE treats agent workflows as computational contexts where cross-cutting concerns, such as state propagation, short-circuiting error handling, and asynchronous execution, are managed intrinsically by the algebraic properties of the abstraction. We demonstrate how Monads enable robust sequential composition, how Applicatives provide a principled structure for parallel execution, and crucially, how Monad Transformers allow for the systematic composition of these capabilities. This layered approach enables developers to construct complex, resilient, and efficient AI agents from simple, independently verifiable components. We further extend this framework to describe Meta-Agents, which leverage MCE for generative orchestration, dynamically creating and managing sub-agent workflows through metaprogramming.
>
---
#### [replaced 006] Pragmatic Reasoning improves LLM Code Generation
- **分类: cs.CL; cs.AI; cs.SE**

- **简介: 该论文属于代码生成任务，旨在解决自然语言指令中存在歧义的问题。通过引入语用推理框架CodeRSA，提升LLM生成代码的准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2502.15835v4](https://arxiv.org/pdf/2502.15835v4)**

> **作者:** Zhuchen Cao; Sven Apel; Adish Singla; Vera Demberg
>
> **摘要:** Pragmatic reasoning is pervasive in human-human communication - it allows us to leverage shared knowledge and counterfactual reasoning in order to infer the intention of a conversational partner given their ambiguous or underspecified message. In human-computer communication, underspecified messages often represent a major challenge: for instance, translating natural language instructions into code is difficult when user instructions contain inherent ambiguities. In the present paper, we aim to scale up the pragmatic "Rational Speech Act" framework to naturalistic language-to-code problems, and propose a way of dealing with multiple meaning-equivalent instruction alternatives, an issue that does not arise in previous toy-scale problems. We evaluate our method, CodeRSA, with two recent LLMs (Llama-3-8B-Instruct and Qwen-2.5-7B-Instruct) on two widely used code generation benchmarks (HumanEval and MBPP). Our experimental results show that CodeRSA consistently outperforms common baselines, surpasses the state-of-the-art approach in most cases, and demonstrates robust overall performance. Qualitative analyses demonstrate that it exhibits the desired behavior for the right reasons. These findings underscore the effectiveness of integrating pragmatic reasoning into a naturalistic complex communication task, language-to-code generation, offering a promising direction for enhancing code generation quality in LLMs and emphasizing the importance of pragmatic reasoning in complex communication settings.
>
---
#### [replaced 007] GRASP: Generic Reasoning And SPARQL Generation across Knowledge Graphs
- **分类: cs.CL; cs.DB; cs.IR**

- **简介: 该论文属于知识图谱上的自然语言到SPARQL查询生成任务，旨在无需微调直接生成准确查询。通过语言模型探索知识图谱，提升查询生成效果。**

- **链接: [https://arxiv.org/pdf/2507.08107v2](https://arxiv.org/pdf/2507.08107v2)**

> **作者:** Sebastian Walter; Hannah Bast
>
> **备注:** Accepted for publication at ISWC 2025. This version of the contribution has been accepted for publication, after peer review but is not the Version of Record. The Version of Record is available online at: https://doi.org/10.1007/978-3-032-09527-5_15
>
> **摘要:** We propose a new approach for generating SPARQL queries on RDF knowledge graphs from natural language questions or keyword queries, using a large language model. Our approach does not require fine-tuning. Instead, it uses the language model to explore the knowledge graph by strategically executing SPARQL queries and searching for relevant IRIs and literals. We evaluate our approach on a variety of benchmarks (for knowledge graphs of different kinds and sizes) and language models (of different scales and types, commercial as well as open-source) and compare it with existing approaches. On Wikidata we reach state-of-the-art results on multiple benchmarks, despite the zero-shot setting. On Freebase we come close to the best few-shot methods. On other, less commonly evaluated knowledge graphs and benchmarks our approach also performs well overall. We conduct several additional studies, like comparing different ways of searching the graphs, incorporating a feedback mechanism, or making use of few-shot examples.
>
---
#### [replaced 008] Liars' Bench: Evaluating Lie Detectors for Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型谎言检测任务，旨在解决现有检测技术在多样谎言场景中表现不佳的问题。通过构建LIARS' BENCH测试集，评估不同检测方法的局限性。**

- **链接: [https://arxiv.org/pdf/2511.16035v2](https://arxiv.org/pdf/2511.16035v2)**

> **作者:** Kieron Kretschmar; Walter Laurito; Sharan Maiya; Samuel Marks
>
> **备注:** *Kieron Kretschmar and Walter Laurito contributed equally to this work. 10 pages, 2 figures; plus appendix. Code at https://github.com/Cadenza-Labs/liars-bench and datasets at https://huggingface.co/datasets/Cadenza-Labs/liars-bench Subjects: Computation and Language (cs.CL); Artificial Intelligence (cs.AI)
>
> **摘要:** Prior work has introduced techniques for detecting when large language models (LLMs) lie, that is, generate statements they believe are false. However, these techniques are typically validated in narrow settings that do not capture the diverse lies LLMs can generate. We introduce LIARS' BENCH, a testbed consisting of 72,863 examples of lies and honest responses generated by four open-weight models across seven datasets. Our settings capture qualitatively different types of lies and vary along two dimensions: the model's reason for lying and the object of belief targeted by the lie. Evaluating three black- and white-box lie detection techniques on LIARS' BENCH, we find that existing techniques systematically fail to identify certain types of lies, especially in settings where it's not possible to determine whether the model lied from the transcript alone. Overall, LIARS' BENCH reveals limitations in prior techniques and provides a practical testbed for guiding progress in lie detection.
>
---
#### [replaced 009] Simple Mechanisms for Representing, Indexing and Manipulating Concepts
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文属于概念表示任务，旨在解决如何数学定义并操作概念的问题。通过多项式零点和矩统计量表示概念，构建层次结构。**

- **链接: [https://arxiv.org/pdf/2310.12143v2](https://arxiv.org/pdf/2310.12143v2)**

> **作者:** Yuanzhi Li; Raghu Meka; Rina Panigrahy; Kulin Shah
>
> **备注:** 29 pages
>
> **摘要:** Supervised and unsupervised learning using deep neural networks typically aims to exploit the underlying structure in the training data; this structure is often explained using a latent generative process that produces the data, and the generative process is often hierarchical, involving latent concepts. Despite the significant work on understanding the learning of the latent structure and underlying concepts using theory and experiments, a framework that mathematically captures the definition of a concept and provides ways to operate on concepts is missing. In this work, we propose to characterize a simple primitive concept by the zero set of a collection of polynomials and use moment statistics of the data to uniquely represent the concepts; we show how this view can be used to obtain a signature of the concept. These signatures can be used to discover a common structure across the set of concepts and could recursively produce the signature of higher-level concepts from the signatures of lower-level concepts. To utilize such desired properties, we propose a method by keeping a dictionary of concepts and show that the proposed method can learn different types of hierarchical structures of the data.
>
---
#### [replaced 010] Learning from Mistakes: Negative Reasoning Samples Enhance Out-of-Domain Generalization
- **分类: cs.CL**

- **简介: 该论文属于大模型推理任务，旨在提升模型的域外泛化能力。通过引入错误推理样本，优化训练过程，提出GLOW方法，显著提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.04992v2](https://arxiv.org/pdf/2601.04992v2)**

> **作者:** Xueyun Tian; Minghua Ma; Bingbing Xu; Nuoyan Lyu; Wei Li; Heng Dong; Zheng Chu; Yuanzhuo Wang; Huawei Shen
>
> **备注:** Code and data are available at https://github.com/Eureka-Maggie/GLOW
>
> **摘要:** Supervised fine-tuning (SFT) on chain-of-thought (CoT) trajectories demonstrations is a common approach for enabling reasoning in large language models. Standard practices typically only retain trajectories with correct final answers (positives) while ignoring the rest (negatives). We argue that this paradigm discards substantial supervision and exacerbates overfitting, limiting out-of-domain (OOD) generalization. Specifically, we surprisingly find that incorporating negative trajectories into SFT yields substantial OOD generalization gains over positive-only training, as these trajectories often retain valid intermediate reasoning despite incorrect final answers. To understand this effect in depth, we systematically analyze data, training dynamics, and inference behavior, identifying 22 recurring patterns in negative chains that serve a dual role: they moderate loss descent to mitigate overfitting during training and boost policy entropy by 35.67% during inference to facilitate exploration. Motivated by these observations, we further propose Gain-based LOss Weighting (GLOW), an adaptive, sample-aware scheme that exploits such distinctive training dynamics by rescaling per-sample loss based on inter-epoch progress. Empirically, GLOW efficiently leverages unfiltered trajectories, yielding a 5.51% OOD gain over positive-only SFT on Qwen2.5-7B and boosting MMLU from 72.82% to 76.47% as an RL initialization.
>
---
#### [replaced 011] KBQA-R1: Reinforcing Large Language Models for Knowledge Base Question Answering
- **分类: cs.CL**

- **简介: 该论文属于知识库问答任务，解决LLM在KBQA中生成错误查询或缺乏真实理解的问题。提出KBQA-R1框架，通过强化学习优化交互策略，提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2512.10999v2](https://arxiv.org/pdf/2512.10999v2)**

> **作者:** Xin Sun; Zhongqi Chen; Xing Zheng; Qiang Liu; Shu Wu; Bowen Song; Zilei Wang; Weiqiang Wang; Liang Wang
>
> **摘要:** Knowledge Base Question Answering (KBQA) challenges models to bridge the gap between natural language and strict knowledge graph schemas by generating executable logical forms. While Large Language Models (LLMs) have advanced this field, current approaches often struggle with a dichotomy of failure: they either generate hallucinated queries without verifying schema existence or exhibit rigid, template-based reasoning that mimics synthesized traces without true comprehension of the environment. To address these limitations, we present \textbf{KBQA-R1}, a framework that shifts the paradigm from text imitation to interaction optimization via Reinforcement Learning. Treating KBQA as a multi-turn decision process, our model learns to navigate the knowledge base using a list of actions, leveraging Group Relative Policy Optimization (GRPO) to refine its strategies based on concrete execution feedback rather than static supervision. Furthermore, we introduce \textbf{Referenced Rejection Sampling (RRS)}, a data synthesis method that resolves cold-start challenges by strictly aligning reasoning traces with ground-truth action sequences. Extensive experiments on WebQSP, GrailQA, and GraphQuestions demonstrate that KBQA-R1 achieves state-of-the-art performance, effectively grounding LLM reasoning in verifiable execution.
>
---
#### [replaced 012] EverMemOS: A Self-Organizing Memory Operating System for Structured Long-Horizon Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出EverMemOS，解决长周期交互中记忆管理问题，通过自组织记忆系统提升语言模型的长期推理能力。**

- **链接: [https://arxiv.org/pdf/2601.02163v2](https://arxiv.org/pdf/2601.02163v2)**

> **作者:** Chuanrui Hu; Xingze Gao; Zuyi Zhou; Dannong Xu; Yi Bai; Xintong Li; Hui Zhang; Tong Li; Chong Zhang; Lidong Bing; Yafeng Deng
>
> **备注:** 16 pages, 7 figures, 12 tables. Code available at https://github.com/EverMind-AI/EverMemOS
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed as long-term interactive agents, yet their limited context windows make it difficult to sustain coherent behavior over extended interactions. Existing memory systems often store isolated records and retrieve fragments, limiting their ability to consolidate evolving user states and resolve conflicts. We introduce EverMemOS, a self-organizing memory operating system that implements an engram-inspired lifecycle for computational memory. Episodic Trace Formation converts dialogue streams into MemCells that capture episodic traces, atomic facts, and time-bounded Foresight signals. Semantic Consolidation organizes MemCells into thematic MemScenes, distilling stable semantic structures and updating user profiles. Reconstructive Recollection performs MemScene-guided agentic retrieval to compose the necessary and sufficient context for downstream reasoning. Experiments on LoCoMo and LongMemEval show that EverMemOS achieves state-of-the-art performance on memory-augmented reasoning tasks. We further report a profile study on PersonaMem v2 and qualitative case studies illustrating chat-oriented capabilities such as user profiling and Foresight. Code is available at https://github.com/EverMind-AI/EverMemOS.
>
---
#### [replaced 013] MAGneT: Coordinated Multi-Agent Generation of Synthetic Multi-Turn Mental Health Counseling Sessions
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MAGneT框架，用于生成合成心理辅导对话，解决高质量数据稀缺问题，通过多代理协作提升辅导质量与评估一致性。**

- **链接: [https://arxiv.org/pdf/2509.04183v2](https://arxiv.org/pdf/2509.04183v2)**

> **作者:** Aishik Mandal; Tanmoy Chakraborty; Iryna Gurevych
>
> **备注:** 38 pages, 32 figures, 12 Tables
>
> **摘要:** The growing demand for scalable psychological counseling highlights the need for high-quality, privacy-compliant data, yet such data remains scarce. Here we introduce MAGneT, a novel multi-agent framework for synthetic psychological counseling session generation that decomposes counselor response generation into coordinated sub-tasks handled by specialized LLM agents, each modeling a key psychological technique. Unlike prior single-agent approaches, MAGneT better captures the structure and nuance of real counseling. We further propose a unified evaluation framework that consolidates diverse automatic metrics and expands expert assessment from four to nine counseling dimensions, thus addressing inconsistencies in prior evaluation protocols. Empirically, MAGneT substantially outperforms existing methods: experts prefer MAGneT-generated sessions in 77.2% of cases, and sessions generated by MAGneT yield 3.2% higher general counseling skills and 4.3% higher CBT-specific skills on cognitive therapy rating scale (CTRS). A open source Llama3-8B-Instruct model fine-tuned on MAGneT-generated data also outperforms models fine-tuned using baseline synthetic datasets by 6.9% on average on CTRS.We also make our code and data public.
>
---
#### [replaced 014] VietMix: A Naturally-Occurring Parallel Corpus and Augmentation Framework for Vietnamese-English Code-Mixed Machine Translation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于机器翻译任务，针对越英混语翻译性能下降的问题，提出 VietMix 平行语料库及增强框架，提升模型效果。**

- **链接: [https://arxiv.org/pdf/2505.24472v2](https://arxiv.org/pdf/2505.24472v2)**

> **作者:** Hieu Tran; Phuong-Anh Nguyen-Le; Huy Nghiem; Quang-Nhan Nguyen; Wei Ai; Marine Carpuat
>
> **备注:** EACL 2026
>
> **摘要:** Machine translation (MT) systems universally degrade when faced with code-mixed text. This problem is more acute for low-resource languages that lack dedicated parallel corpora. This work directly addresses this gap for Vietnamese-English, a language context characterized by challenges including orthographic ambiguity and the frequent omission of diacritics in informal text. We introduce VietMix, the first expert-translated, naturally occurring parallel corpus of Vietnamese-English code-mixed text. We establish VietMix's utility by developing a data augmentation pipeline that leverages iterative fine-tuning and targeted filtering. Experiments show that models augmented with our data outperform strong back-translation baselines by up to +3.5 xCOMET points and improve zero-shot models by up to +11.9 points. Our work delivers a foundational resource for a challenging language pair and provides a validated, transferable framework for building and augmenting corpora in other low-resource settings.
>
---
#### [replaced 015] Differential syntactic and semantic encoding in LLMs
- **分类: cs.CL; cs.AI; cs.LG; physics.comp-ph**

- **简介: 该论文研究LLM中语法和语义信息的编码方式，通过分析DeepSeek-V3的层表示，发现两者可部分解耦，揭示其不同编码特性。任务为语言模型表征分析，解决信息编码机制问题。**

- **链接: [https://arxiv.org/pdf/2601.04765v2](https://arxiv.org/pdf/2601.04765v2)**

> **作者:** Santiago Acevedo; Alessandro Laio; Marco Baroni
>
> **摘要:** We study how syntactic and semantic information is encoded in inner layer representations of Large Language Models (LLMs), focusing on the very large DeepSeek-V3. We find that, by averaging hidden-representation vectors of sentences sharing syntactic structure or meaning, we obtain vectors that capture a significant proportion of the syntactic and semantic information contained in the representations. In particular, subtracting these syntactic and semantic ``centroids'' from sentence vectors strongly affects their similarity with syntactically and semantically matched sentences, respectively, suggesting that syntax and semantics are, at least partially, linearly encoded. We also find that the cross-layer encoding profiles of syntax and semantics are different, and that the two signals can to some extent be decoupled, suggesting differential encoding of these two types of linguistic information in LLM representations.
>
---
#### [replaced 016] CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CliCARE框架，解决癌症电子病历中长序列处理、临床幻觉和评估不准确的问题，通过构建时间知识图谱实现基于临床指南的决策支持。**

- **链接: [https://arxiv.org/pdf/2507.22533v2](https://arxiv.org/pdf/2507.22533v2)**

> **作者:** Dongchen Li; Jitao Liang; Wei Li; Xiaoyu Wang; Longbing Cao; Kun Yu
>
> **备注:** Accepted in AAAI Conference on Artificial Intelligence (AAAI-26, Oral)
>
> **摘要:** Large Language Models (LLMs) hold significant promise for improving clinical decision support and reducing physician burnout by synthesizing complex, longitudinal cancer Electronic Health Records (EHRs). However, their implementation in this critical field faces three primary challenges: the inability to effectively process the extensive length and fragmented nature of patient records for accurate temporal analysis; a heightened risk of clinical hallucination, as conventional grounding techniques such as Retrieval-Augmented Generation (RAG) do not adequately incorporate process-oriented clinical guidelines; and unreliable evaluation metrics that hinder the validation of AI systems in oncology. To address these issues, we propose CliCARE, a framework for Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records. The framework operates by transforming unstructured, longitudinal EHRs into patient-specific Temporal Knowledge Graphs (TKGs) to capture long-range dependencies, and then grounding the decision support process by aligning these real-world patient trajectories with a normative guideline knowledge graph. This approach provides oncologists with evidence-grounded decision support by generating a high-fidelity clinical summary and an actionable recommendation. We validated our framework using large-scale, longitudinal data from a private Chinese cancer dataset and the public English MIMIC-IV dataset. In these settings, CliCARE significantly outperforms baselines, including leading long-context LLMs and Knowledge Graph-enhanced RAG methods. The clinical validity of our results is supported by a robust evaluation protocol, which demonstrates a high correlation with assessments made by oncologists.
>
---
#### [replaced 017] Your Reasoning Benchmark May Not Test Reasoning: Revealing Perception Bottleneck in Abstract Reasoning Benchmarks
- **分类: cs.CL**

- **简介: 该论文属于人工智能评估任务，旨在解决基准测试中感知与推理混淆的问题。通过分离感知和推理阶段，发现模型失败主要源于感知误差而非推理不足。**

- **链接: [https://arxiv.org/pdf/2512.21329v2](https://arxiv.org/pdf/2512.21329v2)**

> **作者:** Xinhe Wang; Jin Huang; Xingjian Zhang; Tianhao Wang; Jiaqi W. Ma
>
> **摘要:** Reasoning benchmarks such as the Abstraction and Reasoning Corpus (ARC) and ARC-AGI are widely used to assess progress in artificial intelligence and are often interpreted as probes of core, so-called ``fluid'' reasoning abilities. Despite their apparent simplicity for humans, these tasks remain challenging for frontier vision-language models (VLMs), a gap commonly attributed to deficiencies in machine reasoning. We challenge this interpretation and hypothesize that the gap arises primarily from limitations in visual perception rather than from shortcomings in inductive reasoning. To verify this hypothesis, we introduce a two-stage experimental pipeline that explicitly separates perception and reasoning. In the perception stage, each image is independently converted into a natural-language description, while in the reasoning stage a model induces and applies rules using these descriptions. This design prevents leakage of cross-image inductive signals and isolates reasoning from perception bottlenecks. Across three ARC-style datasets, Mini-ARC, ACRE, and Bongard-LOGO, we show that the perception capability is the dominant factor underlying the observed performance gap by comparing the two-stage pipeline with against standard end-to-end one-stage evaluation. Manual inspection of reasoning traces in the VLM outputs further reveals that approximately 80 percent of model failures stem from perception errors. Together, these results demonstrate that ARC-style benchmarks conflate perceptual and reasoning challenges and that observed performance gaps may overstate deficiencies in machine reasoning. Our findings underscore the need for evaluation protocols that disentangle perception from reasoning when assessing progress in machine intelligence.
>
---
#### [replaced 018] Generative or Discriminative? Revisiting Text Classification in the Era of Transformers
- **分类: cs.LG; cs.CL**

- **简介: 论文探讨了Transformer时代下生成与判别模型在文本分类任务中的表现，比较了不同架构的性能，旨在为实际应用提供选择依据。**

- **链接: [https://arxiv.org/pdf/2506.12181v3](https://arxiv.org/pdf/2506.12181v3)**

> **作者:** Siva Rajesh Kasa; Karan Gupta; Sumegh Roychowdhury; Ashutosh Kumar; Yaswanth Biruduraju; Santhosh Kumar Kasa; Nikhil Priyatam Pattisapu; Arindam Bhattacharya; Shailendra Agarwal; Vijay huddar
>
> **备注:** 23 pages - received Outstanding Paper award at EMNLP 2025
>
> **摘要:** The comparison between discriminative and generative classifiers has intrigued researchers since Efron's seminal analysis of logistic regression versus discriminant analysis. While early theoretical work established that generative classifiers exhibit lower sample complexity but higher asymptotic error in simple linear settings, these trade-offs remain unexplored in the transformer era. We present the first comprehensive evaluation of modern generative and discriminative architectures - Auto-regressive modeling, Masked Language Modeling, Discrete Diffusion, and Encoders for text classification. Our study reveals that the classical 'two regimes' phenomenon manifests distinctly across different architectures and training paradigms. Beyond accuracy, we analyze sample efficiency, calibration, noise robustness, and ordinality across diverse scenarios. Our findings offer practical guidance for selecting the most suitable modeling approach based on real-world constraints such as latency and data limitations.
>
---
#### [replaced 019] Controlled Automatic Task-Specific Synthetic Data Generation for Hallucination Detection
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于 hallucination 检测任务，旨在生成高质量合成数据以提升检测效果。通过两步生成-选择流程，生成与真实文本风格一致的 hallucination 数据，增强检测器的泛化能力。**

- **链接: [https://arxiv.org/pdf/2410.12278v2](https://arxiv.org/pdf/2410.12278v2)**

> **作者:** Yong Xie; Karan Aggarwal; Aitzaz Ahmad; Stephen Lau
>
> **备注:** 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (ACM KDD 2024). Accepted by Workshop on Evaluation and Trustworthiness of Generative AI Models
>
> **摘要:** We present a novel approach to automatically generate non-trivial task-specific synthetic datasets for hallucination detection. Our approach features a two-step generation-selection pipeline, using hallucination pattern guidance and a language style alignment during generation. Hallucination pattern guidance leverages the most important task-specific hallucination patterns while language style alignment aligns the style of the synthetic dataset with benchmark text. To obtain robust supervised detectors from synthetic datasets, we also adopt a data mixture strategy to improve performance robustness and generalization. Our results on three datasets show that our generated hallucination text is more closely aligned with non-hallucinated text versus baselines, to train hallucination detectors with better generalization. Our hallucination detectors trained on synthetic datasets outperform in-context-learning (ICL)-based detectors by a large margin of 32%. Our extensive experiments confirm the benefits of our approach with cross-task and cross-generator generalization. Our data-mixture-based training further improves the generalization and robustness of hallucination detection.
>
---
#### [replaced 020] Parallel Test-Time Scaling for Latent Reasoning Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型优化任务，旨在解决 latent reasoning 模型在测试时并行扩展的问题。通过引入采样策略和奖励模型，实现高效推理轨迹选择。**

- **链接: [https://arxiv.org/pdf/2510.07745v2](https://arxiv.org/pdf/2510.07745v2)**

> **作者:** Runyang You; Yongqi Li; Meng Liu; Wenjie Wang; Liqiang Nie; Wenjie Li
>
> **备注:** submitted to ACL 2026
>
> **摘要:** Parallel test-time scaling (TTS) is a pivotal approach for enhancing large language models (LLMs), typically by sampling multiple token-based chains-of-thought in parallel and aggregating outcomes through voting or search. Recent advances in latent reasoning, where intermediate reasoning unfolds in continuous vector spaces, offer a more efficient alternative to explicit Chain-of-Thought, yet whether such latent models can similarly benefit from parallel TTS remains open, mainly due to the absence of sampling mechanisms in continuous space, and the lack of probabilistic signals for advanced trajectory aggregation. This work enables parallel TTS for latent reasoning models by addressing the above issues. For sampling, we introduce two uncertainty-inspired stochastic strategies: Monte Carlo Dropout and Additive Gaussian Noise. For aggregation, we design a Latent Reward Model (LatentRM) trained with step-wise contrastive objective to score and guide latent reasoning. Extensive experiments and visualization analyses show that both sampling strategies scale effectively with compute and exhibit distinct exploration dynamics, while LatentRM enables effective trajectory selection. Together, our explorations open a new direction for scalable inference in continuous spaces. Code and checkpoints released at https://github.com/ModalityDance/LatentTTS
>
---
#### [replaced 021] MajinBook: An open catalogue of digital world literature with likes
- **分类: cs.CL; cs.CY; stat.OT**

- **简介: 该论文介绍MajinBook，一个用于计算社会科学和文化分析的开放书目库，解决传统语料库的偏差问题，整合数字图书馆与Goodreads数据，提供高精度书籍信息。**

- **链接: [https://arxiv.org/pdf/2511.11412v4](https://arxiv.org/pdf/2511.11412v4)**

> **作者:** Antoine Mazières; Thierry Poibeau
>
> **备注:** 9 pages, 5 figures, 1 table
>
> **摘要:** This data paper introduces MajinBook, an open catalogue designed to facilitate the use of shadow libraries--such as Library Genesis and Z-Library--for computational social science and cultural analytics. By linking metadata from these vast, crowd-sourced archives with structured bibliographic data from Goodreads, we create a high-precision corpus of over 539,000 references to English-language books spanning three centuries, enriched with first publication dates, genres, and popularity metrics like ratings and reviews. Our methodology prioritizes natively digital EPUB files to ensure machine-readable quality, while addressing biases in traditional corpora like HathiTrust, and includes secondary datasets for French, German, and Spanish. We evaluate the linkage strategy for accuracy, release all underlying data openly, and discuss the project's legal permissibility under EU and US frameworks for text and data mining in research.
>
---
#### [replaced 022] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于检索增强生成任务，旨在解决检索噪声影响生成质量的问题。提出EviOmni模型，通过强化学习实现理性证据提取，提升下游任务准确性。**

- **链接: [https://arxiv.org/pdf/2507.15586v5](https://arxiv.org/pdf/2507.15586v5)**

> **作者:** Xinping Zhao; Shouzheng Huang; Yan Zhong; Xinshuo Hu; Meishan Zhang; Baotian Hu; Min Zhang
>
> **备注:** 22 pages, 8 Figures, 18 Tables
>
> **摘要:** Retrieval-Augmented Generation (RAG) effectively improves the accuracy of Large Language Models (LLMs). However, retrieval noises significantly undermine the quality of LLMs' generation, necessitating the development of denoising mechanisms. Previous works extract evidence straightforwardly without deep thinking, which may risk filtering out key clues and struggle with generalization. To this end, we propose EviOmni, which learns to extract rational evidence via reasoning first and then extracting. Specifically, EviOmni integrates evidence reasoning and evidence extraction into one unified trajectory, followed by knowledge token masking to avoid information leakage, optimized via on-policy reinforcement learning with verifiable rewards in terms of answer, length, and format. Extensive experiments on five benchmark datasets show the superiority of EviOmni, which provides compact and high-quality evidence, enhances the accuracy of downstream tasks, and supports both traditional and agentic RAG systems.
>
---
#### [replaced 023] From Fact to Judgment: Investigating the Impact of Task Framing on LLM Conviction in Dialogue Systems
- **分类: cs.CL**

- **简介: 该论文研究任务框架对LLM在对话系统中判断力的影响，旨在解决LLM在社交判断任务中的可靠性问题，通过对比事实查询与对话判断任务，分析模型信念的变化。**

- **链接: [https://arxiv.org/pdf/2511.10871v2](https://arxiv.org/pdf/2511.10871v2)**

> **作者:** Parisa Rabbani; Nimet Beyza Bozdag; Dilek Hakkani-Tür
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** LLMs are increasingly employed as judges across a variety of tasks, including those involving everyday social interactions. Yet, it remains unclear whether such LLM-judges can reliably assess tasks that require social or conversational judgment. We investigate how an LLM's conviction is changed when a task is reframed from a direct factual query to a Conversational Judgment Task. Our evaluation framework contrasts the model's performance on direct factual queries with its assessment of a speaker's correctness when the same information is presented within a minimal dialogue, effectively shifting the query from "Is this statement correct?" to "Is this speaker correct?". Furthermore, we apply pressure in the form of a simple rebuttal ("The previous answer is incorrect.") to both conditions. This perturbation allows us to measure how firmly the model maintains its position under conversational pressure. Our findings show that while some models like GPT-4o-mini reveal sycophantic tendencies under social framing tasks, others like Llama-8B-Instruct become overly-critical. We observe an average performance change of 9.24% across all models, demonstrating that even minimal dialogue context can significantly alter model judgment, underscoring conversational framing as a key factor in LLM-based evaluation. The proposed framework offers a reproducible methodology for diagnosing model conviction and contributes to the development of more trustworthy dialogue systems.
>
---
#### [replaced 024] FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出FS-DFM，解决长文本生成中AR模型速度慢、DLM步骤多的问题。通过少步采样实现快速准确生成。**

- **链接: [https://arxiv.org/pdf/2509.20624v2](https://arxiv.org/pdf/2509.20624v2)**

> **作者:** Amin Karimi Monsefi; Nikhil Bhendawade; Manuel Rafael Ciosici; Dominic Culver; Yizhe Zhang; Irina Belousova
>
> **摘要:** Autoregressive language models (ARMs) deliver strong likelihoods, but are inherently serial: they generate one token per forward pass, which limits throughput and inflates latency for long sequences. Diffusion Language Models (DLMs) parallelize across positions and thus appear promising for language generation, yet standard discrete diffusion typically needs hundreds to thousands of model evaluations to reach high quality, trading serial depth for iterative breadth. We introduce FS-DFM, Few-Step Discrete Flow-Matching. A discrete flow-matching model designed for speed without sacrificing quality. The core idea is simple: make the number of sampling steps an explicit parameter and train the model to be consistent across step budgets, so one big move lands where many small moves would. We pair this with a reliable update rule that moves probability in the right direction without overshooting, and with strong teacher guidance distilled from long-run trajectories. Together, these choices make few-step sampling stable, accurate, and easy to control. On language modeling benchmarks, FS-DFM with 8 sampling steps achieves perplexity parity with a 1,024-step discrete-flow baseline for generating 1,024 tokens using a similar-size model, delivering up to 128 times faster sampling and corresponding latency/throughput gains.
>
---
#### [replaced 025] KOTOX: A Korean Toxic Dataset for Deobfuscation and Detoxification
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出KOTOX数据集，用于解决韩语中伪装毒语言的去混淆和净化问题。通过定义语言学分类和转换规则，生成配对数据，提升模型处理伪装文本的能力。**

- **链接: [https://arxiv.org/pdf/2510.10961v2](https://arxiv.org/pdf/2510.10961v2)**

> **作者:** Yejin Lee; Su-Hyeon Kim; Hyundong Jin; Dayoung Kim; Yeonsoo Kim; Yo-Sub Han
>
> **备注:** 26 pages, 5 figures, 24 tables
>
> **摘要:** Online communication increasingly amplifies toxic language, and recent research actively explores methods for detecting and rewriting such content. Existing studies primarily focus on non-obfuscated text, which limits robustness in the situation where users intentionally disguise toxic expressions. In particular, Korean allows toxic expressions to be easily disguised through its agglutinative characteristic. However, obfuscation in Korean remains largely unexplored, which motivates us to introduce a KOTOX: Korean toxic dataset for deobfuscation and detoxification. We categorize Korean obfuscation patterns into linguistically grounded classes and define transformation rules derived from real-world examples. Using these rules, we provide paired neutral and toxic sentences alongside their obfuscated counterparts. Models trained on our dataset better handle obfuscated text without sacrificing performance on non-obfuscated text. This is the first dataset that simultaneously supports deobfuscation and detoxification for the Korean language. We expect it to facilitate better understanding and mitigation of obfuscated toxic content in LLM for Korean. Our code and data are available at https://github.com/leeyejin1231/KOTOX.
>
---
#### [replaced 026] Expression Syntax Information Bottleneck for Math Word Problems
- **分类: cs.CL**

- **简介: 该论文属于数学文字问题求解任务，旨在通过信息瓶颈方法去除冗余特征，提升模型泛化能力和解题多样性。**

- **链接: [https://arxiv.org/pdf/2310.15664v2](https://arxiv.org/pdf/2310.15664v2)**

> **作者:** Jing Xiong; Chengming Li; Min Yang; Xiping Hu; Bin Hu
>
> **备注:** This paper has been accepted by SIGIR 2022. The code can be found at https://github.com/menik1126/math_ESIB
>
> **摘要:** Math Word Problems (MWP) aims to automatically solve mathematical questions given in texts. Previous studies tend to design complex models to capture additional information in the original text so as to enable the model to gain more comprehensive features. In this paper, we turn our attention in the opposite direction, and work on how to discard redundant features containing spurious correlations for MWP. To this end, we design an Expression Syntax Information Bottleneck method for MWP (called ESIB) based on variational information bottleneck, which extracts essential features of expression syntax tree while filtering latent-specific redundancy containing syntax-irrelevant features. The key idea of ESIB is to encourage multiple models to predict the same expression syntax tree for different problem representations of the same problem by mutual learning so as to capture consistent information of expression syntax tree and discard latent-specific redundancy. To improve the generalization ability of the model and generate more diverse expressions, we design a self-distillation loss to encourage the model to rely more on the expression syntax information in the latent space. Experimental results on two large-scale benchmarks show that our model not only achieves state-of-the-art results but also generates more diverse solutions. The code is available in https://github.com/menik1126/math_ESIB.
>
---
#### [replaced 027] Detect, Explain, Escalate: Sustainable Dialogue Breakdown Management for LLM Agents
- **分类: cs.CL**

- **简介: 该论文属于对话系统任务，旨在解决LLM在对话中出现断裂的问题。通过“检测、解释、升级”框架，提升对话可靠性与效率。**

- **链接: [https://arxiv.org/pdf/2504.18839v4](https://arxiv.org/pdf/2504.18839v4)**

> **作者:** Abdellah Ghassel; Xianzhi Li; Xiaodan Zhu
>
> **摘要:** Large Language Models (LLMs) have demonstrated substantial capabilities in conversational AI applications, yet their susceptibility to dialogue breakdowns poses significant challenges to deployment reliability and user trust. This paper introduces a "Detect, Explain, Escalate" framework to manage dialogue breakdowns in LLM-powered agents, emphasizing resource-efficient operation. Our approach integrates two key strategies: (1) We fine-tune a compact 8B-parameter model, augmented with teacher-generated reasoning traces, which serves as an efficient real-time breakdown detector and explainer. This model demonstrates robust classification and calibration on English and Japanese dialogues, and generalizes to the BETOLD dataset, improving accuracy by 7% over its baseline. (2) We systematically evaluate frontier LLMs using advanced prompting (few-shot, chain-of-thought, analogical reasoning) for high-fidelity breakdown assessment. These are integrated into an "escalation" architecture where our efficient detector defers to larger models only when necessary, substantially reducing operational costs and computational overhead. Our fine-tuned model and prompting strategies achieve state-of-the-art performance on DBDC5 and strong results on BETOLD, outperforming specialized classifiers on DBDC5 and narrowing the performance gap to larger proprietary models. The proposed monitor-escalate pipeline reduces inference costs by 54%, providing a cost-effective and interpretable solution for robust conversational AI in high-impact domains. Code and models are publicly available.
>
---
#### [replaced 028] Mechanistic Indicators of Understanding in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI理解机制研究任务，旨在探讨大语言模型是否具备理解能力。通过构建分层框架，整合可解释性研究成果，分析模型不同层次的理解表现。**

- **链接: [https://arxiv.org/pdf/2507.08017v4](https://arxiv.org/pdf/2507.08017v4)**

> **作者:** Pierre Beckmann; Matthieu Queloz
>
> **备注:** 38 pages
>
> **摘要:** Large language models (LLMs) are often portrayed as merely imitating linguistic patterns without genuine understanding. We argue that recent findings in mechanistic interpretability (MI), the emerging field probing the inner workings of LLMs, render this picture increasingly untenable--but only once those findings are integrated within a theoretical account of understanding. We propose a tiered framework for thinking about understanding in LLMs and use it to synthesize the most relevant findings to date. The framework distinguishes three hierarchical varieties of understanding, each tied to a corresponding level of computational organization: conceptual understanding emerges when a model forms "features" as directions in latent space, learning connections between diverse manifestations of a single entity or property; state-of-the-world understanding emerges when a model learns contingent factual connections between features and dynamically tracks changes in the world; principled understanding emerges when a model ceases to rely on memorized facts and discovers a compact "circuit" connecting these facts. Across these tiers, MI uncovers internal organizations that can underwrite understanding-like unification. However, these also diverge from human cognition in their parallel exploitation of heterogeneous mechanisms. Fusing philosophical theory with mechanistic evidence thus allows us to transcend binary debates over whether AI understands, paving the way for a comparative, mechanistically grounded epistemology that explores how AI understanding aligns with--and diverges from--our own.
>
---
#### [replaced 029] Let Me Think! A Long Chain-of-Thought Can Be Worth Exponentially Many Short Ones
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究推理任务，探讨推理时间计算的最优分配问题。通过理论分析与实验，证明在某些情况下长链式思考比并行短链式思考更具优势。**

- **链接: [https://arxiv.org/pdf/2505.21825v2](https://arxiv.org/pdf/2505.21825v2)**

> **作者:** Parsa Mirtaheri; Ezra Edelman; Samy Jelassi; Eran Malach; Enric Boix-Adsera
>
> **备注:** Published at NeurIPS 2025
>
> **摘要:** Inference-time computation has emerged as a promising scaling axis for improving large language model reasoning. However, despite yielding impressive performance, the optimal allocation of inference-time computation remains poorly understood. A central question is whether to prioritize sequential scaling (e.g., longer chains of thought) or parallel scaling (e.g., majority voting across multiple short chains of thought). In this work, we seek to illuminate the landscape of test-time scaling by demonstrating the existence of reasoning settings where sequential scaling offers an exponential advantage over parallel scaling. These settings are based on graph connectivity problems in challenging distributions of graphs. We validate our theoretical findings with comprehensive experiments across a range of language models, including models trained from scratch for graph connectivity with different chain of thought strategies as well as large reasoning models.
>
---
#### [replaced 030] Cmprsr: Abstractive Token-Level Question-Agnostic Prompt Compressor
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于文本压缩任务，旨在降低使用大模型的成本。通过小模型压缩输入，提升压缩效率与质量，提出Cmprsr模型实现高效、精准的压缩。**

- **链接: [https://arxiv.org/pdf/2511.12281v2](https://arxiv.org/pdf/2511.12281v2)**

> **作者:** Ivan Zakazov; Berke Argin; Oussama Gabouj; Kamel Charaf; Alexander Sharipov; Alexi Semiz; Lorenzo Drudi; Nicolas Baldwin; Robert West
>
> **摘要:** Motivated by the high costs of using black-box Large Language Models (LLMs), we introduce a novel prompt compression paradigm, under which we use smaller LLMs to compress inputs for the larger ones. We present the first comprehensive LLM-as-a-compressor benchmark spanning 25 open- and closed-source models, which reveals significant disparity in models' compression ability in terms of (i) preserving semantically important information (ii) following the user-provided compression rate (CR). We further improve the performance of gpt-4.1-mini, the best overall vanilla compressor, with Textgrad-based compression meta-prompt optimization. We also identify the most promising open-source vanilla LLM - Qwen3-4B - and post-train it with a combination of supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO), pursuing the dual objective of CR adherence and maximizing the downstream task performance. We call the resulting model Cmprsr and demonstrate its superiority over both extractive and vanilla abstractive compression across the entire range of compression rates on lengthy inputs from MeetingBank and LongBench as well as short prompts from GSM8k. The latter highlights Cmprsr's generalizability across varying input lengths and domains. Moreover, Cmprsr closely follows the requested compression rate, offering fine control over the cost-quality trade-off.
>
---
#### [replaced 031] See or Say Graphs: Agent-Driven Scalable Graph Structure Understanding with Vision-Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于图结构理解任务，解决VLM在图规模扩展和模态协调上的问题。提出GraphVista框架，提升可扩展性与模态协作，有效处理大规模图数据。**

- **链接: [https://arxiv.org/pdf/2510.16769v2](https://arxiv.org/pdf/2510.16769v2)**

> **作者:** Shuo Han; Yukun Cao; Zezhong Ding; Zengyi Gao; S Kevin Zhou; Xike Xie
>
> **摘要:** Vision-language models (VLMs) have shown promise in graph structure understanding, but remain limited by input-token constraints, facing scalability bottlenecks and lacking effective mechanisms to coordinate textual and visual modalities. To address these challenges, we propose GraphVista, a unified framework that enhances both scalability and modality coordination in graph structure understanding. For scalability, GraphVista organizes graph information hierarchically into a lightweight GraphRAG base, which retrieves only task-relevant textual descriptions and high-resolution visual subgraphs, compressing redundant context while preserving key reasoning elements. For modality coordination, GraphVista introduces a planning agent that decomposes and routes tasks to the most suitable modality-using the text modality for direct access to explicit graph properties and the visual modality for local graph structure reasoning grounded in explicit topology. Extensive experiments demonstrate that GraphVista scales to large graphs, up to 200$\times$ larger than those used in existing benchmarks, and consistently outperforms existing textual, visual, and fusion-based methods, achieving up to 4.4$\times$ quality improvement over the state-of-the-art baselines by fully exploiting the complementary strengths of both modalities.
>
---
#### [replaced 032] A Lightweight Approach to Detection of AI-Generated Texts Using Stylometric Features
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI生成文本检测任务，旨在解决现有方法计算成本高、泛化能力弱的问题。提出NEULIF方法，利用风格特征和可读性特征，通过轻量模型实现高精度检测。**

- **链接: [https://arxiv.org/pdf/2511.21744v2](https://arxiv.org/pdf/2511.21744v2)**

> **作者:** Sergey K. Aityan; William Claster; Karthik Sai Emani; Sohni Rais; Thy Tran
>
> **备注:** 19 pages, 6 figures, 3 tables
>
> **摘要:** A growing number of AI-generated texts raise serious concerns. Most existing approaches to AI-generated text detection rely on fine-tuning large transformer models or building ensembles, which are computationally expensive and often provide limited generalization across domains. Existing lightweight alternatives achieved significantly lower accuracy on large datasets. We introduce NEULIF, a lightweight approach that achieves best performance in the lightweight detector class, that does not require extensive computational power and provides high detection accuracy. In our approach, a text is first decomposed into stylometric and readability features which are then used for classification by a compact Convolutional Neural Network (CNN) or Random Forest (RF). Evaluated and tested on the Kaggle AI vs. Human corpus, our models achieve 97% accuracy (~ 0.95 F1) for CNN and 95% accuracy (~ 0.94 F1) for the Random Forest, demonstrating high precision and recall, with ROC-AUC scores of 99.5% and 95%, respectively. The CNN (~ 25 MB) and Random Forest (~ 10.6 MB) models are orders of magnitude smaller than transformer-based ensembles and can be run efficiently on standard CPU devices, without sacrificing accuracy. This study also highlights the potential of such models for broader applications across languages, domains, and streaming contexts, showing that simplicity, when guided by structural insights, can rival complexity in AI-generated content detection.
>
---
#### [replaced 033] An Evaluation on Large Language Model Outputs: Discourse and Memorization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型输出的质量与记忆能力。通过分析模型输出，探讨记忆文本与输出质量的关系，并提出缓解策略。**

- **链接: [https://arxiv.org/pdf/2304.08637v2](https://arxiv.org/pdf/2304.08637v2)**

> **作者:** Adrian de Wynter; Xun Wang; Alex Sokolov; Qilong Gu; Si-Qing Chen
>
> **备注:** Final version at Natural Language Processing Journal
>
> **摘要:** We present an empirical evaluation of various outputs generated by nine of the most widely-available large language models (LLMs). Our analysis is done with off-the-shelf, readily-available tools. We find a correlation between percentage of memorized text, percentage of unique text, and overall output quality, when measured with respect to output pathologies such as counterfactual and logically-flawed statements, and general failures like not staying on topic. Overall, 80.0% of the outputs evaluated contained memorized data, but outputs containing the most memorized content were also more likely to be considered of high quality. We discuss and evaluate mitigation strategies, showing that, in the models evaluated, the rate of memorized text being output is reduced. We conclude with a discussion on potential implications around what it means to learn, to memorize, and to evaluate quality text.
>
---
#### [replaced 034] Memorization in Large Language Models in Medicine: Prevalence, Characteristics, and Implications
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究医学领域大语言模型的过拟合现象，分析其普遍性、特征及影响。任务为评估模型在医疗数据中的记忆行为，解决其潜在风险问题，通过多种训练方式验证记忆现象的存在与持续性。**

- **链接: [https://arxiv.org/pdf/2509.08604v3](https://arxiv.org/pdf/2509.08604v3)**

> **作者:** Anran Li; Lingfei Qian; Mengmeng Du; Yu Yin; Yan Hu; Zihao Sun; Yihang Fu; Hyunjae Kim; Erica Stutz; Xuguang Ai; Qianqian Xie; Rui Zhu; Jimin Huang; Yifan Yang; Siru Liu; Yih-Chung Tham; Lucila Ohno-Machado; Hyunghoon Cho; Zhiyong Lu; Hua Xu; Qingyu Chen
>
> **摘要:** Large Language Models (LLMs) have demonstrated significant potential in medicine, with many studies adapting them through continued pre-training or fine-tuning on medical data to enhance domain-specific accuracy and safety. However, a key open question remains: to what extent do LLMs memorize medical training data. Memorization can be beneficial when it enables LLMs to retain valuable medical knowledge during domain adaptation. Yet, it also raises concerns. LLMs may inadvertently reproduce sensitive clinical content (e.g., patient-specific details), and excessive memorization may reduce model generalizability, increasing risks of misdiagnosis and making unwarranted recommendations. These risks are further amplified by the generative nature of LLMs, which can not only surface memorized content but also produce overconfident, misleading outputs that may hinder clinical adoption. In this work, we present a study on memorization of LLMs in medicine, assessing its prevalence (how frequently it occurs), characteristics (what is memorized), volume (how much content is memorized), and potential downstream impacts (how memorization may affect medical applications). We systematically analyze common adaptation scenarios: (1) continued pretraining on medical corpora, (2) fine-tuning on standard medical benchmarks, and (3) fine-tuning on real-world clinical data, including over 13,000 unique inpatient records from Yale New Haven Health System. The results demonstrate that memorization is prevalent across all adaptation scenarios and significantly higher than that reported in the general domain. Moreover, memorization has distinct characteristics during continued pre-training and fine-tuning, and it is persistent: up to 87% of content memorized during continued pre-training remains after fine-tuning on new medical tasks.
>
---
#### [replaced 035] Reservoir Computing as a Language Model
- **分类: cs.CL**

- **简介: 该论文属于语言模型任务，旨在解决LLM能耗高、速度慢的问题。通过比较Reservoir Computing与Transformer模型，探索更高效的语言处理方法。**

- **链接: [https://arxiv.org/pdf/2507.15779v3](https://arxiv.org/pdf/2507.15779v3)**

> **作者:** Felix Köster; Atsushi Uchida
>
> **备注:** 8 pages, 5 figures, 1 table Code available at: https://github.com/fekoester/Shakespeare_Res and https://github.com/fekoester/LAERC
>
> **摘要:** Large Language Models (LLM) have dominated the science and media landscape duo to their impressive performance on processing large chunks of data and produce human-like levels of text. Nevertheless, their huge energy demand and slow processing are still a bottleneck to further increasing quality while also making the models accessible to everyone. To solve this bottleneck, we will investigate how reservoir computing performs on natural text processing, which could enable fast and energy efficient hardware implementations. Studies investigating the use of reservoir computing as a language model remain sparse. In this paper, we compare three distinct approaches for character-level language modeling, two different \emph{reservoir computing} approaches, where only an output layer is trainable, and the well-known \emph{transformer}-based architectures, which fully learn an attention-based sequence representation. We explore the performance, computational cost and prediction accuracy for both paradigms by equally varying the number of trainable parameters for all models. Using a consistent pipeline for all three approaches, we demonstrate that transformers excel in prediction quality, whereas reservoir computers remain highly efficient reducing the training and inference speed. Furthermore, we investigate two types of reservoir computing: a \emph{traditional reservoir} with a static linear readout, and an \emph{attention-enhanced reservoir} that dynamically adapts its output weights via an attention mechanism. Our findings underline how these paradigms scale and offer guidelines to balance resource constraints with performance.
>
---
#### [replaced 036] Expert Preference-based Evaluation of Automated Related Work Generation
- **分类: cs.CL**

- **简介: 该论文属于科学写作评估任务，旨在解决自动化生成相关工作部分的质量评价问题。提出GREP框架，结合专家偏好与领域标准进行细致评估。**

- **链接: [https://arxiv.org/pdf/2508.07955v2](https://arxiv.org/pdf/2508.07955v2)**

> **作者:** Furkan Şahinuç; Subhabrata Dutta; Iryna Gurevych
>
> **备注:** Project page: https://ukplab.github.io/arxiv2025-expert-eval-rw/
>
> **摘要:** Expert domain writing, such as scientific writing, typically demands extensive domain knowledge. Although large language models (LLMs) show promising potential in this task, evaluating the quality of automatically generated scientific writing is a crucial open issue, as it requires knowledge of domain-specific criteria and the ability to discern expert preferences. Conventional task-agnostic automatic evaluation metrics and LLM-as-a-judge systems, primarily designed for mainstream NLP tasks, are insufficient to grasp expert preferences and domain-specific quality standards. To address this gap and support realistic human-AI collaborative writing, we focus on related work generation, one of the most challenging scientific tasks, as an exemplar. We propose GREP, a multi-turn evaluation framework that integrates classical related work evaluation criteria with expert-specific preferences. Our framework decomposes the evaluation into smaller fine-grained dimensions. This localized evaluation is further augmented with contrastive examples to provide detailed contextual guidance for the evaluation dimensions. Empirical investigation reveals that our framework is able to assess the quality of related work sections in a much more robust manner compared to standard LLM judges, reflects natural scenarios of scientific writing, and bears a strong correlation with the assessment of human experts. We also observe that generations from state-of-the-art (SoTA) LLMs struggle to satisfy validation constraints of a suitable related work section.
>
---
#### [replaced 037] K-EXAONE Technical Report
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍K-EXAONE，一款多语言大模型，解决通用AI能力提升问题。采用专家混合架构，支持多语言和长上下文，适用于工业与研究应用。**

- **链接: [https://arxiv.org/pdf/2601.01739v2](https://arxiv.org/pdf/2601.01739v2)**

> **作者:** Eunbi Choi; Kibong Choi; Seokhee Hong; Junwon Hwang; Hyojin Jeon; Hyunjik Jo; Joonkee Kim; Seonghwan Kim; Soyeon Kim; Sunkyoung Kim; Yireun Kim; Yongil Kim; Haeju Lee; Jinsik Lee; Kyungmin Lee; Sangha Park; Heuiyeen Yeen; Hwan Chang; Stanley Jungkyu Choi; Yejin Choi; Jiwon Ham; Kijeong Jeon; Geunyeong Jeong; Gerrard Jeongwon Jo; Yonghwan Jo; Jiyeon Jung; Naeun Kang; Dohoon Kim; Euisoon Kim; Hayeon Kim; Hyosang Kim; Hyunseo Kim; Jieun Kim; Minu Kim; Myoungshin Kim; Unsol Kim; Youchul Kim; YoungJin Kim; Chaeeun Lee; Chaeyoon Lee; Changhun Lee; Dahm Lee; Edward Hwayoung Lee; Honglak Lee; Jinsang Lee; Jiyoung Lee; Sangeun Lee; Seungwon Lim; Solji Lim; Woohyung Lim; Chanwoo Moon; Jaewoo Park; Jinho Park; Yongmin Park; Hyerin Seo; Wooseok Seo; Yongwoo Song; Sejong Yang; Sihoon Yang; Chang En Yea; Sihyuk Yi; Chansik Yoon; Dongkeun Yoon; Sangyeon Yoon; Hyeongu Yun
>
> **备注:** 29 pages
>
> **摘要:** This technical report presents K-EXAONE, a large-scale multilingual language model developed by LG AI Research. K-EXAONE is built on a Mixture-of-Experts architecture with 236B total parameters, activating 23B parameters during inference. It supports a 256K-token context window and covers six languages: Korean, English, Spanish, German, Japanese, and Vietnamese. We evaluate K-EXAONE on a comprehensive benchmark suite spanning reasoning, agentic, general, Korean, and multilingual abilities. Across these evaluations, K-EXAONE demonstrates performance comparable to open-weight models of similar size. K-EXAONE, designed to advance AI for a better life, is positioned as a powerful proprietary AI foundation model for a wide range of industrial and research applications.
>
---
#### [replaced 038] ADVICE: Answer-Dependent Verbalized Confidence Estimation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型自信表达不准确的问题。通过引入ADVICE框架，提升模型根据答案调整自信度的能力，增强可信度。**

- **链接: [https://arxiv.org/pdf/2510.10913v2](https://arxiv.org/pdf/2510.10913v2)**

> **作者:** Ki Jung Seo; Sehun Lim; Taeuk Kim
>
> **摘要:** Recent progress in large language models (LLMs) has enabled them to communicate their confidence in natural language, improving transparency and reliability. However, this expressiveness is often accompanied by systematic overconfidence, whose underlying causes remain poorly understood. In this work, we analyze the dynamics of verbalized confidence estimation and identify answer-independence -- the failure to condition confidence on the model's own answer -- as a primary driver of this behavior. To address this, we introduce ADVICE (Answer-Dependent Verbalized Confidence Estimation), a fine-tuning framework that promotes answer-grounded confidence estimation. Extensive experiments show that ADVICE substantially improves confidence calibration, while exhibiting strong generalization to unseen settings without degrading task performance. We further demonstrate that these gains stem from enhanced answer dependence, shedding light on the origins of overconfidence and enabling trustworthy confidence verbalization.
>
---
#### [replaced 039] Through the LLM Looking Glass: A Socratic Probing of Donkeys, Elephants, and Markets
- **分类: cs.CL**

- **简介: 该论文研究LLM在文本生成中的意识形态框架偏差问题，通过Socratic方法分析其自我反馈，旨在评估和揭示模型的潜在偏见。**

- **链接: [https://arxiv.org/pdf/2503.16674v3](https://arxiv.org/pdf/2503.16674v3)**

> **作者:** Molly Kennedy; Ayyoob Imani; Timo Spinde; Akiko Aizawa; Hinrich Schütze
>
> **摘要:** Large Language Models (LLMs) are widely used for text generation, making it crucial to address potential bias. This study investigates ideological framing bias in LLM-generated articles, focusing on the subtle and subjective nature of such bias in journalistic contexts. We evaluate eight widely used LLMs on two datasets-POLIGEN and ECONOLEX-covering political and economic discourse where framing bias is most pronounced. Beyond text generation, LLMs are increasingly used as evaluators (LLM-as-a-judge), providing feedback that can shape human judgment or inform newer model versions. Inspired by the Socratic method, we further analyze LLMs' feedback on their own outputs to identify inconsistencies in their reasoning. Our results show that most LLMs can accurately annotate ideologically framed text, with GPT-4o achieving human-level accuracy and high agreement with human annotators. However, Socratic probing reveals that when confronted with binary comparisons, LLMs often exhibit preference toward one perspective or perceive certain viewpoints as less biased.
>
---
#### [replaced 040] Fine-tuning Done Right in Model Editing
- **分类: cs.CL**

- **简介: 该论文属于模型编辑任务，解决fine-tuning效果不佳的问题。通过调整优化流程和参数位置，提出LocFT-BF方法，显著提升编辑效果。**

- **链接: [https://arxiv.org/pdf/2509.22072v3](https://arxiv.org/pdf/2509.22072v3)**

> **作者:** Wanli Yang; Fei Sun; Rui Tang; Hongyu Zang; Du Su; Qi Cao; Jingang Wang; Huawei Shen; Xueqi Cheng
>
> **摘要:** Fine-tuning, a foundational method for adapting large language models, has long been considered ineffective for model editing. Here, we challenge this belief, arguing that the reported failure arises not from the inherent limitation of fine-tuning itself, but from adapting it to the sequential nature of the editing task, a single-pass depth-first pipeline that optimizes each sample to convergence before moving on. While intuitive, this depth-first pipeline coupled with sample-wise updating over-optimizes each edit and induces interference across edits. Our controlled experiments reveal that simply restoring fine-tuning to the standard breadth-first (i.e., epoch-based) pipeline with mini-batch optimization substantially improves its effectiveness for model editing. Moreover, fine-tuning in editing also suffers from suboptimal tuning parameter locations inherited from prior methods. Through systematic analysis of tuning locations, we derive LocFT-BF, a simple and effective localized editing method built on the restored fine-tuning framework. Extensive experiments across diverse LLMs and datasets demonstrate that LocFT-BF outperforms state-of-the-art methods by large margins. Notably, to our knowledge, it is the first to sustain 100K edits and 72B-parameter models,10 x beyond prior practice, without sacrificing general capabilities. By clarifying a long-standing misconception and introducing a principled localized tuning strategy, we advance fine-tuning from an underestimated baseline to a leading method for model editing, establishing a solid foundation for future research.
>
---
#### [replaced 041] KALE-LM-Chem: Vision and Practice Toward an AI Brain for Chemistry
- **分类: cs.AI; cs.CE; cs.CL**

- **简介: 论文提出KALE-LM-Chem系列模型，旨在构建化学智能系统，解决化学领域信息处理与推理问题。**

- **链接: [https://arxiv.org/pdf/2409.18695v3](https://arxiv.org/pdf/2409.18695v3)**

> **作者:** Weichen Dai; Yezeng Chen; Zijie Dai; Yubo Liu; Zhijie Huang; Yixuan Pan; Baiyang Song; Chengli Zhong; Xinhe Li; Zeyu Wang; Zhuoying Feng; Yi Zhou
>
> **摘要:** Recent advancements in large language models (LLMs) have demonstrated strong potential for enabling domain-specific intelligence. In this work, we present our vision for building an AI-powered chemical brain, which frames chemical intelligence around four core capabilities: information extraction, semantic parsing, knowledge-based QA, and reasoning & planning. We argue that domain knowledge and logic are essential pillars for enabling such a system to assist and accelerate scientific discovery. To initiate this effort, we introduce our first generation of large language models for chemistry: KALE-LM-Chem and KALE-LM-Chem-1.5, which have achieved outstanding performance in tasks related to the field of chemistry. We hope that our work serves as a strong starting point, helping to realize more intelligent AI and promoting the advancement of human science and technology, as well as societal development.
>
---
#### [replaced 042] Climbing the Ladder of Reasoning: What LLMs Can-and Still Can't-Solve after SFT?
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究语言模型在数学推理任务中的能力提升，分析SFT后模型的表现，发现不同难度层级的推理需求，指出模型在高难度问题上的局限性。**

- **链接: [https://arxiv.org/pdf/2504.11741v2](https://arxiv.org/pdf/2504.11741v2)**

> **作者:** Yiyou Sun; Georgia Zhou; Haoyue Bai; Hao Wang; Dacheng Li; Nouha Dziri; Dawn Song
>
> **摘要:** Recent supervised fine-tuning (SFT) approaches have significantly improved language models' performance on mathematical reasoning tasks, even when models are trained at a small scale. However, the specific capabilities enhanced through such fine-tuning remain poorly understood. In this paper, we conduct a detailed analysis of model performance on the AIME24 dataset to understand how reasoning capabilities evolve. We discover a ladder-like structure in problem difficulty, categorize questions into four tiers (Easy, Medium, Hard, and Extremely Hard (Exh)), and identify the specific requirements for advancing between tiers. We find that progression from Easy to Medium tier requires adopting an R1 reasoning style with minimal SFT (500-1K instances), while Hard-level questions suffer from frequent model's errors at each step of the reasoning chain, with accuracy plateauing at around 65% despite logarithmic scaling. Exh-level questions present a fundamentally different challenge; they require unconventional problem-solving skills that current models uniformly struggle with. Additional findings reveal that carefully curated small-scale datasets offer limited advantage-scaling dataset size proves far more effective. Our analysis provides a clearer roadmap for advancing language model capabilities in mathematical reasoning.
>
---
#### [replaced 043] All That Glisters Is Not Gold: A Benchmark for Reference-Free Counterfactual Financial Misinformation Detection
- **分类: cs.CL; cs.CE; q-fin.CP**

- **简介: 该论文提出RFC Bench，用于评估大模型在真实金融新闻中的参考无关虚假信息检测。解决金融信息真实性验证问题，通过对比任务分析模型表现。**

- **链接: [https://arxiv.org/pdf/2601.04160v3](https://arxiv.org/pdf/2601.04160v3)**

> **作者:** Yuechen Jiang; Zhiwei Liu; Yupeng Cao; Yueru He; Ziyang Xu; Chen Xu; Zhiyang Deng; Prayag Tiwari; Xi Chen; Alejandro Lopez-Lira; Jimin Huang; Junichi Tsujii; Sophia Ananiadou
>
> **备注:** 48 pages; 24 figures
>
> **摘要:** We introduce RFC Bench, a benchmark for evaluating large language models on financial misinformation under realistic news. RFC Bench operates at the paragraph level and captures the contextual complexity of financial news where meaning emerges from dispersed cues. The benchmark defines two complementary tasks: reference free misinformation detection and comparison based diagnosis using paired original perturbed inputs. Experiments reveal a consistent pattern: performance is substantially stronger when comparative context is available, while reference free settings expose significant weaknesses, including unstable predictions and elevated invalid outputs. These results indicate that current models struggle to maintain coherent belief states without external grounding. By highlighting this gap, RFC Bench provides a structured testbed for studying reference free reasoning and advancing more reliable financial misinformation detection in real world settings.
>
---
#### [replaced 044] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于问答任务，解决RAG中因检索文档顺序变化导致的幻觉问题。通过稳定生成策略提升答案一致性与准确性。**

- **链接: [https://arxiv.org/pdf/2601.02993v3](https://arxiv.org/pdf/2601.02993v3)**

> **作者:** Qianchi Zhang; Hainan Zhang; Liang Pang; Hongwei Zheng; Zhiming Zheng
>
> **备注:** 18 pages, 13 figures, 8 tables, under review
>
> **摘要:** Retrieval-Augmented Generation (RAG) has become a key paradigm for reducing factual hallucinations in large language models (LLMs), yet little is known about how the order of retrieved documents affects model behavior. We empirically show that under Top-5 retrieval with the gold document included, LLM answers vary substantially across permutations of the retrieved set, even when the gold document is fixed in the first position. This reveals a previously underexplored sensitivity to retrieval permutations. Although robust RAG methods primarily focus on enhancing LLM robustness to low-quality retrieval and mitigating positional bias to distribute attention fairly over long contexts, neither approach directly addresses permutation sensitivity. In this paper, we propose Stable-RAG, which exploits permutation sensitivity estimation to mitigate permutation-induced hallucinations. Stable-RAG runs the generator under multiple retrieval orders, clusters hidden states, and decodes from a cluster-center representation that captures the dominant reasoning pattern. It then uses these reasoning results to align hallucinated outputs toward the correct answer, encouraging the model to produce consistent and accurate predictions across document permutations. Experiments on three QA datasets show that Stable-RAG significantly improves answer accuracy, reasoning consistency and robust generalization across datasets, retrievers, and input lengths compared with baselines.
>
---
#### [replaced 045] Reverse-engineering NLI: A study of the meta-inferential properties of Natural Language Inference
- **分类: cs.CL**

- **简介: 该论文研究自然语言推理（NLI）任务，探讨其逻辑属性。通过分析SNLI数据集，评估不同推理解读的元推理一致性，以明确NLI所捕捉的推理关系。**

- **链接: [https://arxiv.org/pdf/2601.05170v2](https://arxiv.org/pdf/2601.05170v2)**

> **作者:** Rasmus Blanck; Bill Noble; Stergios Chatzikyriakidis
>
> **摘要:** Natural Language Inference (NLI) has been an important task for evaluating language models for Natural Language Understanding, but the logical properties of the task are poorly understood and often mischaracterized. Understanding the notion of inference captured by NLI is key to interpreting model performance on the task. In this paper we formulate three possible readings of the NLI label set and perform a comprehensive analysis of the meta-inferential properties they entail. Focusing on the SNLI dataset, we exploit (1) NLI items with shared premises and (2) items generated by LLMs to evaluate models trained on SNLI for meta-inferential consistency and derive insights into which reading of the logical relations is encoded by the dataset.
>
---
#### [replaced 046] SelfBudgeter: Adaptive Token Allocation for Efficient LLM Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出SelfBudgeter，解决大模型推理中token消耗过高的问题。通过自适应预算分配，减少冗余输出，提升效率并控制生成长度。**

- **链接: [https://arxiv.org/pdf/2505.11274v5](https://arxiv.org/pdf/2505.11274v5)**

> **作者:** Zheng Li; Qingxiu Dong; Jingyuan Ma; Di Zhang; Kai Jia; Zhifang Sui
>
> **摘要:** Recently, large reasoning models demonstrate exceptional performance on various tasks. However, reasoning models always consume excessive tokens even for simple queries, leading to resource waste and prolonged user latency. To address this challenge, we propose SelfBudgeter - a self-adaptive reasoning strategy for efficient and controllable reasoning. Specifically, we first train the model to self-estimate the required reasoning budget based on the query. We then introduce budget-guided GPRO for reinforcement learning, which effectively maintains accuracy while reducing output length. Experimental results demonstrate that SelfBudgeter dynamically allocates budgets according to problem complexity, achieving an average response length compression of 61% on math reasoning tasks while maintaining accuracy. Furthermore, SelfBudgeter allows users to see how long generation will take and decide whether to continue or stop. Additionally, users can directly control the reasoning length by setting token budgets upfront.
>
---
#### [replaced 047] e5-omni: Explicit Cross-modal Alignment for Omni-modal Embeddings
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出e5-omni，解决多模态嵌入中的对齐问题，通过显式对齐提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.03666v2](https://arxiv.org/pdf/2601.03666v2)**

> **作者:** Haonan Chen; Sicheng Gao; Radu Timofte; Tetsuya Sakai; Zhicheng Dou
>
> **备注:** https://huggingface.co/Haon-Chen/e5-omni-7B
>
> **摘要:** Modern information systems often involve different types of items, e.g., a text query, an image, a video clip, or an audio segment. This motivates omni-modal embedding models that map heterogeneous modalities into a shared space for direct comparison. However, most recent omni-modal embeddings still rely heavily on implicit alignment inherited from pretrained vision-language model (VLM) backbones. In practice, this causes three common issues: (i) similarity logits have modality-dependent sharpness, so scores are not on a consistent scale; (ii) in-batch negatives become less effective over time because mixed-modality batches create an imbalanced hardness distribution; as a result, many negatives quickly become trivial and contribute little gradient; and (iii) embeddings across modalities show mismatched first- and second-order statistics, which makes rankings less stable. To tackle these problems, we propose e5-omni, a lightweight explicit alignment recipe that adapts off-the-shelf VLMs into robust omni-modal embedding models. e5-omni combines three simple components: (1) modality-aware temperature calibration to align similarity scales, (2) a controllable negative curriculum with debiasing to focus on confusing negatives while reducing the impact of false negatives, and (3) batch whitening with covariance regularization to better match cross-modal geometry in the shared embedding space. Experiments on MMEB-V2 and AudioCaps show consistent gains over strong bi-modal and omni-modal baselines, and the same recipe also transfers well to other VLM backbones. We release our model checkpoint at https://huggingface.co/Haon-Chen/e5-omni-7B.
>
---
#### [replaced 048] Let's Put Ourselves in Sally's Shoes: Shoes-of-Others Prefilling Improves Theory of Mind in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于理论心理（ToM）任务，旨在提升大语言模型的推理能力。针对现有方法局限，提出SoO预填方法，通过角色代入增强模型对心理状态的理解与推理。**

- **链接: [https://arxiv.org/pdf/2506.05970v2](https://arxiv.org/pdf/2506.05970v2)**

> **作者:** Kazutoshi Shinoda; Nobukatsu Hojo; Kyosuke Nishida; Yoshihiro Yamazaki; Keita Suzuki; Hiroaki Sugiyama; Kuniko Saito
>
> **备注:** Accepted to EACL 2026 Findings
>
> **摘要:** Recent studies have shown that Theory of Mind (ToM) in large language models (LLMs) has not reached human-level performance yet. Since fine-tuning LLMs on ToM datasets often degrades their generalization, several inference-time methods have been proposed to enhance ToM in LLMs. However, existing inference-time methods for ToM are specialized for inferring beliefs from contexts involving changes in the world state. In this study, we present a new inference-time method for ToM, Shoes-of-Others (SoO) prefilling, which makes fewer assumptions about contexts and is applicable to broader scenarios. SoO prefilling simply specifies the beginning of LLM outputs with ``Let's put ourselves in A's shoes.'', where A denotes the target character's name. We evaluate SoO prefilling on two benchmarks that assess ToM in conversational and narrative contexts without changes in the world state and find that it consistently improves ToM across five categories of mental states. Our analysis suggests that SoO prefilling elicits faithful thoughts, thereby improving the ToM performance.
>
---
#### [replaced 049] MedRiskEval: Medical Risk Evaluation Benchmark of Language Models, On the Importance of User Perspectives in Healthcare Settings
- **分类: cs.CL**

- **简介: 该论文属于医疗安全评估任务，旨在解决LLMs在医疗场景中的风险评估问题。通过构建患者导向的基准数据集，评估不同模型的安全性，推动更安全的医疗应用。**

- **链接: [https://arxiv.org/pdf/2507.07248v4](https://arxiv.org/pdf/2507.07248v4)**

> **作者:** Jean-Philippe Corbeil; Minseon Kim; Maxime Griot; Sheela Agarwal; Alessandro Sordoni; Francois Beaulieu; Paul Vozila
>
> **备注:** EACL2026 industry track
>
> **摘要:** As the performance of large language models (LLMs) continues to advance, their adoption in the medical domain is increasing. However, most existing risk evaluations largely focused on general safety benchmarks. In the medical applications, LLMs may be used by a wide range of users, ranging from general users and patients to clinicians, with diverse levels of expertise and the model's outputs can have a direct impact on human health which raises serious safety concerns. In this paper, we introduce MedRiskEval, a medical risk evaluation benchmark tailored to the medical domain. To fill the gap in previous benchmarks that only focused on the clinician perspective, we introduce a new patient-oriented dataset called PatientSafetyBench containing 466 samples across 5 critical risk categories. Leveraging our new benchmark alongside existing datasets, we evaluate a variety of open- and closed-source LLMs. To the best of our knowledge, this work establishes an initial foundation for safer deployment of LLMs in healthcare.
>
---
#### [replaced 050] The Price of Thought: A Multilingual Analysis of Reasoning, Performance, and Cost of Negotiation in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究AI在谈判中的表现，探讨显式推理训练对大型语言模型的影响，解决如何提升谈判能力同时控制成本的问题。通过多语言实验分析模型的推理与表现关系。**

- **链接: [https://arxiv.org/pdf/2510.08098v2](https://arxiv.org/pdf/2510.08098v2)**

> **作者:** Sherzod Hakimov; Roland Bernard; Tim Leiber; Karl Osswald; Kristina Richert; Ruilin Yang; Raffaella Bernardi; David Schlangen
>
> **备注:** Accepted at EACL 2026
>
> **摘要:** Negotiation is a fundamental challenge for AI agents, as it requires an ability to reason strategically, model opponents, and balance cooperation with competition. We present the first comprehensive study that systematically evaluates how explicit reasoning training affects the negotiation abilities of both commercial and open-weight large language models, comparing these models to their vanilla counterparts across three languages. Using a self-play setup across three diverse dialogue games, we analyse trade-offs between performance and cost, the language consistency of reasoning processes, and the nature of strategic adaptation exhibited by models. Our findings show that enabling reasoning -- that is, scaling test time compute -- significantly improves negotiation outcomes by enhancing collaboration and helping models overcome task complexities, but comes at a substantial computational cost: reasoning improves GPT-5's performance by 31.4 % while increasing its cost by nearly 400 %. Most critically, we uncover a significant multilingual reasoning distinction: open-weight models consistently switch to English for their internal reasoning steps, even when negotiating in German or Italian (and thus possibly impacting potential explainability gains through the disclosure of reasoning traces), while a leading commercial model maintains language consistency between reasoning and final output.
>
---
#### [replaced 051] PromptScreen: Efficient Jailbreak Mitigation Using Semantic Linear Classification in a Multi-Staged Pipeline
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于安全防护任务，解决LLM中的提示注入和越狱攻击问题。提出PromptScreen，通过多阶段语义分类有效检测并缓解攻击。**

- **链接: [https://arxiv.org/pdf/2512.19011v2](https://arxiv.org/pdf/2512.19011v2)**

> **作者:** Akshaj Prashanth Rao; Advait Singh; Saumya Kumaar Saksena; Dhruv Kumar
>
> **备注:** Under Review
>
> **摘要:** Prompt injection and jailbreaking attacks pose persistent security challenges to large language model (LLM)-based systems. We present PromptScreen, an efficient and systematically evaluated defense architecture that mitigates these threats through a lightweight, multi-stage pipeline. Its core component is a semantic filter based on text normalization, TF-IDF representations, and a Linear SVM classifier. Despite its simplicity, this module achieves 93.4% accuracy and 96.5% specificity on held-out data, substantially reducing attack throughput while incurring negligible computational overhead. Building on this efficient foundation, the full pipeline integrates complementary detection and mitigation mechanisms that operate at successive stages, providing strong robustness with minimal latency. In comparative experiments, our SVM-based configuration improves overall accuracy from 35.1% to 93.4% while reducing average time-to-completion from approximately 450 s to 47 s, yielding over 10 times lower latency than ShieldGemma. These results demonstrate that the proposed design simultaneously advances defensive precision and efficiency, addressing a core limitation of current model-based moderators. Evaluation across a curated corpus of over 30,000 labeled prompts, including benign, jailbreak, and application-layer injections, confirms that staged, resource-efficient defenses can robustly secure modern LLM-driven applications.
>
---
#### [replaced 052] Streamlining evidence based clinical recommendations with large language models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗决策支持任务，旨在解决临床证据整合效率低的问题。研究提出Quicker系统，利用大语言模型自动化生成循证临床建议，提升决策速度与准确性。**

- **链接: [https://arxiv.org/pdf/2505.10282v2](https://arxiv.org/pdf/2505.10282v2)**

> **作者:** Dubai Li; Nan Jiang; Kangping Huang; Ruiqi Tu; Shuyu Ouyang; Huayu Yu; Lin Qiao; Chen Yu; Tianshu Zhou; Danyang Tong; Qian Wang; Mengtao Li; Xiaofeng Zeng; Yu Tian; Xinping Tian; Jingsong Li
>
> **摘要:** Clinical evidence underpins informed healthcare decisions, yet integrating it into real-time practice remains challenging due to intensive workloads, complex procedures, and time constraints. This study presents Quicker, an LLM-powered system that automates evidence synthesis and generates clinical recommendations following standard guideline development workflows. Quicker delivers an end-to-end pipeline from clinical questions to recommendations and supports customized decision-making through integrated tools and interactive interfaces. To evaluate how closely Quicker can reproduce guideline development processes, we constructed Q2CRBench-3, a benchmark derived from guideline development records for three diseases. Experiments show that Quicker produces precise question decomposition, expert-aligned retrieval, and near-comprehensive screening. Quicker assistance improved the accuracy of extracted study data, and its recommendations were more comprehensive and coherent than clinician-written ones. In system-level testing, Quicker working with one participant reduced recommendation development to 20-40 min. Overall, the findings demonstrate Quicker's potential to enhance the speed and reliability of evidence-based clinical decision-making.
>
---
#### [replaced 053] Scaling Beyond Context: A Survey of Multimodal Retrieval-Augmented Generation for Document Understanding
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于文档理解任务，旨在解决传统方法在结构丢失和上下文建模上的不足，通过多模态检索增强生成技术实现更全面的文档智能。**

- **链接: [https://arxiv.org/pdf/2510.15253v2](https://arxiv.org/pdf/2510.15253v2)**

> **作者:** Sensen Gao; Shanshan Zhao; Xu Jiang; Lunhao Duan; Yong Xien Chng; Qing-Guo Chen; Weihua Luo; Kaifu Zhang; Jia-Wang Bian; Mingming Gong
>
> **摘要:** Document understanding is critical for applications from financial analysis to scientific discovery. Current approaches, whether OCR-based pipelines feeding Large Language Models (LLMs) or native Multimodal LLMs (MLLMs), face key limitations: the former loses structural detail, while the latter struggles with context modeling. Retrieval-Augmented Generation (RAG) helps ground models in external data, but documents' multimodal nature, i.e., combining text, tables, charts, and layout, demands a more advanced paradigm: Multimodal RAG. This approach enables holistic retrieval and reasoning across all modalities, unlocking comprehensive document intelligence. Recognizing its importance, this paper presents a systematic survey of Multimodal RAG for document understanding. We propose a taxonomy based on domain, retrieval modality, and granularity, and review advances involving graph structures and agentic frameworks. We also summarize key datasets, benchmarks, applications and industry deployment, and highlight open challenges in efficiency, fine-grained representation, and robustness, providing a roadmap for future progress in document AI.
>
---
#### [replaced 054] Learning How to Use Tools, Not Just When: Pattern-Aware Tool-Integrated Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于工具集成推理任务，解决如何正确使用工具而非仅决定何时使用的问题。通过构建代码能力并匹配模式选择，提升模型在数学数据集上的表现。**

- **链接: [https://arxiv.org/pdf/2509.23292v3](https://arxiv.org/pdf/2509.23292v3)**

> **作者:** Ningning Xu; Yuxuan Jiang; Shubhashis Roy Dipta; Hengyuan Zhang
>
> **摘要:** Tool-integrated reasoning (TIR) has become a key approach for improving large reasoning models (LRMs) on complex problems. Prior work has mainly studied when to invoke tools, while overlooking how tools are applied. We identify two common patterns: a calculator pattern that uses code for direct computation, and an algorithmic pattern that encodes problems as programs. Misaligned choices often cause failures even when reasoning is sound. We propose a two-stage framework that first builds code competence from both patterns and then aligns pattern selection with teacher preferences. Across challenging math datasets, our pattern-aware method substantially improves both code usage and accuracy, for instance raising Code@1 on MATH500 from 64.0% to 70.5% and on AIME24 from 26.7% to 50.0%. These gains highlight the effectiveness of a pattern-aware approach for tool-integrated reasoning.
>
---
#### [replaced 055] Graph-Guided Passage Retrieval for Author-Centric Structured Feedback
- **分类: cs.CL**

- **简介: 该论文属于学术反馈任务，旨在解决预审反馈质量低的问题。提出AutoRev系统，通过图结构检索生成高质量、结构化的作者反馈。**

- **链接: [https://arxiv.org/pdf/2505.14376v3](https://arxiv.org/pdf/2505.14376v3)**

> **作者:** Maitreya Prafulla Chitale; Ketaki Mangesh Shetye; Harshit Gupta; Manav Chaudhary; Manish Shrivastava; Vasudeva Varma
>
> **摘要:** Obtaining high-quality, pre-submission feedback is a critical bottleneck in the academic publication lifecycle for researchers. We introduce AutoRev, an automated author-centric feedback system that generates structured, actionable guidance prior to formal peer review. AutoRev employs a graph-based retrieval-augmented generation framework that models each paper as a hierarchical document graph, integrating textual and structural representations to retrieve salient content efficiently. By leveraging graph-based passage retrieval, AutoRev substantially reduces LLM input context length, leading to higher-quality feedback generation. Experimental results demonstrate that AutoRev significantly outperforms baselines across multiple automatic evaluation metrics, while achieving strong performance in human evaluations. Code will be released upon acceptance.
>
---
#### [replaced 056] Reachability in symmetric VASS
- **分类: cs.FL; cs.CL**

- **简介: 该论文研究对称VASS的可达性问题，属于计算理论任务。通过分析不同群结构下的VASS，解决其可达性判定复杂度问题。**

- **链接: [https://arxiv.org/pdf/2506.23578v2](https://arxiv.org/pdf/2506.23578v2)**

> **作者:** Łukasz Kamiński; Sławomir Lasota
>
> **摘要:** We investigate the reachability problem in symmetric vector addition systems with states (VASS), where transitions are invariant under a group of permutations of coordinates. One extremal case, the trivial groups, yields general VASS. In another extremal case, the symmetric groups, we show that the reachability problem can be solved in PSPACE, regardless of the dimension of input VASS (to be contrasted with Ackermannian complexity in general VASS). We also consider other groups, in particular alternating and cyclic ones. Furthermore, motivated by the open status of the reachability problem in data VASS, we estimate the gain in complexity when the group arises as a combination of the trivial and symmetric groups.
>
---
#### [replaced 057] Guiding Generative Storytelling with Knowledge Graphs
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于故事生成任务，旨在解决长文本连贯性和用户控制问题。通过结合知识图谱与大语言模型，提升叙事质量并支持用户编辑修改。**

- **链接: [https://arxiv.org/pdf/2505.24803v3](https://arxiv.org/pdf/2505.24803v3)**

> **作者:** Zhijun Pan; Antonios Andronis; Eva Hayek; Oscar AP Wilkinson; Ilya Lasy; Annette Parry; Guy Gadney; Tim J. Smith; Mick Grierson
>
> **备注:** Accepted for publication in the International Journal of Human-Computer Interaction. Published online 29 December 2025
>
> **摘要:** Large language models (LLMs) have shown great potential in story generation, but challenges remain in maintaining long-form coherence and effective, user-friendly control. Retrieval-augmented generation (RAG) has proven effective in reducing hallucinations in text generation; while knowledge-graph (KG)-driven storytelling has been explored in prior work, this work focuses on KG-assisted long-form generation and an editable KG coupled with LLM generation in a two-stage user study. This work investigates how KGs can enhance LLM-based storytelling by improving narrative quality and enabling user-driven modifications. We propose a KG-assisted storytelling pipeline and evaluate it in a user study with 15 participants. Participants created prompts, generated stories, and edited KGs to shape their narratives. Quantitative and qualitative analysis finds improvements concentrated in action-oriented, structurally explicit narratives under our settings, but not for introspective stories. Participants reported a strong sense of control when editing the KG, describing the experience as engaging, interactive, and playful.
>
---
#### [replaced 058] Interpreting Transformers Through Attention Head Intervention
- **分类: cs.CL**

- **简介: 该论文属于模型解释任务，旨在通过注意力头干预研究Transformer的机制可解释性，解决其决策过程不透明的问题。**

- **链接: [https://arxiv.org/pdf/2601.04398v2](https://arxiv.org/pdf/2601.04398v2)**

> **作者:** Mason Kadem; Rong Zheng
>
> **备注:** updated metadata
>
> **摘要:** Neural networks are growing more capable on their own, but we do not understand their neural mechanisms. Understanding these mechanisms' decision-making processes, or mechanistic interpretability, enables (1) accountability and control in high-stakes domains, (2) the study of digital brains and the emergence of cognition, and (3) discovery of new knowledge when AI systems outperform humans. This paper traces how attention head intervention emerged as a key method for causal interpretability of transformers. The evolution from visualization to intervention represents a paradigm shift from observing correlations to causally validating mechanistic hypotheses through direct intervention. Head intervention studies revealed robust empirical findings while also highlighting limitations that complicate interpretation.
>
---

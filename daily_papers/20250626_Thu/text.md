# 自然语言处理 cs.CL

- **最新发布 49 篇**

- **更新 44 篇**

## 最新发布

#### [new 001] ReCode: Updating Code API Knowledge with Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.IR; cs.LG; cs.SE**

- **简介: 该论文属于代码生成任务，解决LLM在API更新后性能下降的问题。通过强化学习框架ReCode提升模型适应新API的能力。**

- **链接: [http://arxiv.org/pdf/2506.20495v1](http://arxiv.org/pdf/2506.20495v1)**

> **作者:** Haoze Wu; Yunzhi Yao; Wenhao Yu; Huajun Chen; Ningyu Zhang
>
> **备注:** Work in progress
>
> **摘要:** Large Language Models (LLMs) exhibit remarkable code generation capabilities but falter when adapting to frequent updates in external library APIs. This critical limitation, stemming from reliance on outdated API knowledge from their training data, even with access to current documentation, impedes reliable code generation in dynamic environments. To tackle this issue, we propose ReCode (rule-based Reinforcement learning for Code Update), a novel framework that mimics human programmer adaptation to API changes. Specifically, we construct a dataset of approximately 2,000 data entries to train the LLMs to perform version migration based on updated information. Then, we introduce a modified string similarity metric for code evaluation as the reward for reinforcement learning. Our experiments demonstrate that ReCode substantially boosts LLMs' code generation performance in dynamic API scenarios, especially on the unseen CodeUpdateArena task. Crucially, compared to supervised fine-tuning, ReCode has less impact on LLMs' general code generation abilities. We apply ReCode on various LLMs and reinforcement learning algorithms (GRPO and DAPO), all achieving consistent improvements. Notably, after training, Qwen2.5-Coder-7B outperforms that of the 32B parameter code instruction-tuned model and the reasoning model with the same architecture. Code is available at https://github.com/zjunlp/ReCode.
>
---
#### [new 002] SACL: Understanding and Combating Textual Bias in Code Retrieval with Semantic-Augmented Reranking and Localization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于代码检索任务，旨在解决检索模型依赖表面文本和偏向良好文档的问题。通过引入语义增强重排序与定位方法，提升检索效果。**

- **链接: [http://arxiv.org/pdf/2506.20081v1](http://arxiv.org/pdf/2506.20081v1)**

> **作者:** Dhruv Gupta; Gayathri Ganesh Lakshmy; Yiqing Xie
>
> **摘要:** Retrieval-Augmented Code Generation (RACG) is a critical technique for enhancing code generation by retrieving relevant information. In this work, we conduct an in-depth analysis of code retrieval by systematically masking specific features while preserving code functionality. Our discoveries include: (1) although trained on code, current retrievers heavily rely on surface-level textual features (e.g., docstrings, identifier names), and (2) they exhibit a strong bias towards well-documented code, even if the documentation is irrelevant.Based on our discoveries, we propose SACL, a framework that enriches textual information and reduces bias by augmenting code or structural knowledge with semantic information. Extensive experiments show that SACL substantially improves code retrieval (e.g., by 12.8% / 9.4% / 7.0% Recall@1 on HumanEval / MBPP / SWE-Bench-Lite), which also leads to better code generation performance (e.g., by 4.88% Pass@1 on HumanEval).
>
---
#### [new 003] DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation
- **分类: cs.CL**

- **简介: 该论文属于代码生成任务，旨在提升扩散模型在编码中的表现。通过分析扩散模型的去噪过程和强化学习方法，提出新训练框架，提升生成效果。**

- **链接: [http://arxiv.org/pdf/2506.20639v1](http://arxiv.org/pdf/2506.20639v1)**

> **作者:** Shansan Gong; Ruixiang Zhang; Huangjie Zheng; Jiatao Gu; Navdeep Jaitly; Lingpeng Kong; Yizhe Zhang
>
> **备注:** preprint
>
> **摘要:** Diffusion large language models (dLLMs) are compelling alternatives to autoregressive (AR) models because their denoising models operate over the entire sequence. The global planning and iterative refinement features of dLLMs are particularly useful for code generation. However, current training and inference mechanisms for dLLMs in coding are still under-explored. To demystify the decoding behavior of dLLMs and unlock their potential for coding, we systematically investigate their denoising processes and reinforcement learning (RL) methods. We train a 7B dLLM, \textbf{DiffuCoder}, on 130B tokens of code. Using this model as a testbed, we analyze its decoding behavior, revealing how it differs from that of AR models: (1) dLLMs can decide how causal their generation should be without relying on semi-AR decoding, and (2) increasing the sampling temperature diversifies not only token choices but also their generation order. This diversity creates a rich search space for RL rollouts. For RL training, to reduce the variance of token log-likelihood estimates and maintain training efficiency, we propose \textbf{coupled-GRPO}, a novel sampling scheme that constructs complementary mask noise for completions used in training. In our experiments, coupled-GRPO significantly improves DiffuCoder's performance on code generation benchmarks (+4.4\% on EvalPlus) and reduces reliance on AR causal during decoding. Our work provides deeper insight into the machinery of dLLM generation and offers an effective, diffusion-native RL training framework. https://github.com/apple/ml-diffucoder.
>
---
#### [new 004] A Multi-Pass Large Language Model Framework for Precise and Efficient Radiology Report Error Detection
- **分类: cs.CL; I.2.7**

- **简介: 该论文属于医学文本纠错任务，旨在提高放射学报告错误检测的精确率并降低成本。通过三阶段框架提升检测效果，验证其有效性与效率。**

- **链接: [http://arxiv.org/pdf/2506.20112v1](http://arxiv.org/pdf/2506.20112v1)**

> **作者:** Songsoo Kim; Seungtae Lee; See Young Lee; Joonho Kim; Keechan Kan; Dukyong Yoon
>
> **备注:** 29 pages, 5 figures, 4 tables. Code available at https://github.com/radssk/mp-rred
>
> **摘要:** Background: The positive predictive value (PPV) of large language model (LLM)-based proofreading for radiology reports is limited due to the low error prevalence. Purpose: To assess whether a three-pass LLM framework enhances PPV and reduces operational costs compared with baseline approaches. Materials and Methods: A retrospective analysis was performed on 1,000 consecutive radiology reports (250 each: radiography, ultrasonography, CT, MRI) from the MIMIC-III database. Two external datasets (CheXpert and Open-i) were validation sets. Three LLM frameworks were tested: (1) single-prompt detector; (2) extractor plus detector; and (3) extractor, detector, and false-positive verifier. Precision was measured by PPV and absolute true positive rate (aTPR). Efficiency was calculated from model inference charges and reviewer remuneration. Statistical significance was tested using cluster bootstrap, exact McNemar tests, and Holm-Bonferroni correction. Results: Framework PPV increased from 0.063 (95% CI, 0.036-0.101, Framework 1) to 0.079 (0.049-0.118, Framework 2), and significantly to 0.159 (0.090-0.252, Framework 3; P<.001 vs. baselines). aTPR remained stable (0.012-0.014; P>=.84). Operational costs per 1,000 reports dropped to USD 5.58 (Framework 3) from USD 9.72 (Framework 1) and USD 6.85 (Framework 2), reflecting reductions of 42.6% and 18.5%, respectively. Human-reviewed reports decreased from 192 to 88. External validation supported Framework 3's superior PPV (CheXpert 0.133, Open-i 0.105) and stable aTPR (0.007). Conclusion: A three-pass LLM framework significantly enhanced PPV and reduced operational costs, maintaining detection performance, providing an effective strategy for AI-assisted radiology report quality assurance.
>
---
#### [new 005] Enhancing Large Language Models through Structured Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs在复杂推理中的不足。通过结构化推理增强模型，提升其逻辑和规划能力。**

- **链接: [http://arxiv.org/pdf/2506.20241v1](http://arxiv.org/pdf/2506.20241v1)**

> **作者:** Yubo Dong; Hehe Fan
>
> **备注:** Preprint. Under review
>
> **摘要:** Recent Large Language Models (LLMs) have significantly advanced natural language processing and automated decision-making. However, these models still encounter difficulties when performing complex reasoning tasks involving logical deduction and systematic planning, primarily due to their reliance on implicit statistical relationships without structured knowledge representation.Inspired by cognitive science and neurosymbolic AI, we introduce a novel approach to enhance LLMs through explicit structured reasoning. First, we convert unstructured data into structured formats by explicitly annotating reasoning steps. We then employ this structured dataset to train LLMs through Supervised Fine-Tuning (SFT). Additionally, we enhance the structured reasoning capabilities of LLMs using Group Relative Policy Optimization (GRPO), incorporating two innovative algorithms--MAX-Flow and Longest Common Subsequence (LCS)--which notably improve reasoning effectiveness and reduce computational complexity. Experimental results from fine-tuning a DeepSeek-R1-Distill-Qwen-1.5B model demonstrate concise reasoning, robust performance across various scenarios, and improved compatibility with optimization techniques, validating the efficacy of structured reasoning integration in LLMs.
>
---
#### [new 006] COIN: Uncertainty-Guarding Selective Question Answering for Foundation Models with Provable Risk Guarantees
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于问答任务，旨在解决基础模型生成文本中的不确定性问题。提出COIN框架，在保证FDR约束下选择可信答案，提升预测可靠性。**

- **链接: [http://arxiv.org/pdf/2506.20178v1](http://arxiv.org/pdf/2506.20178v1)**

> **作者:** Zhiyuan Wang; Jinhao Duan; Qingni Wang; Xiaofeng Zhu; Tianlong Chen; Xiaoshuang Shi; Kaidi Xu
>
> **摘要:** Uncertainty quantification (UQ) for foundation models is essential to identify and mitigate potential hallucinations in automatically generated text. However, heuristic UQ approaches lack formal guarantees for key metrics such as the false discovery rate (FDR) in selective prediction. Previous work adopts the split conformal prediction (SCP) framework to ensure desired coverage of admissible answers by constructing prediction sets, but these sets often contain incorrect candidates, limiting their practical utility. To address this, we propose COIN, an uncertainty-guarding selection framework that calibrates statistically valid thresholds to filter a single generated answer per question under user-specified FDR constraints. COIN estimates the empirical error rate on a calibration set and applies confidence interval methods such as Clopper-Pearson to establish a high-probability upper bound on the true error rate (i.e., FDR). This enables the selection of the largest uncertainty threshold that ensures FDR control on test data while significantly increasing sample retention. We demonstrate COIN's robustness in risk control, strong test-time power in retaining admissible answers, and predictive efficiency under limited calibration data across both general and multimodal text generation tasks. Furthermore, we show that employing alternative upper bound constructions and UQ strategies can further boost COIN's power performance, which underscores its extensibility and adaptability to diverse application scenarios.
>
---
#### [new 007] Probing AI Safety with Source Code
- **分类: cs.CL**

- **简介: 该论文属于AI安全评估任务，旨在检测大语言模型的安全性问题。通过提出CoDoT方法，将自然语言转换为代码，揭示模型在安全性上的不足。**

- **链接: [http://arxiv.org/pdf/2506.20471v1](http://arxiv.org/pdf/2506.20471v1)**

> **作者:** Ujwal Narayan; Shreyas Chaudhari; Ashwin Kalyan; Tanmay Rajpurohit; Karthik Narasimhan; Ameet Deshpande; Vishvak Murahari
>
> **摘要:** Large language models (LLMs) have become ubiquitous, interfacing with humans in numerous safety-critical applications. This necessitates improving capabilities, but importantly coupled with greater safety measures to align these models with human values and preferences. In this work, we demonstrate that contemporary models fall concerningly short of the goal of AI safety, leading to an unsafe and harmful experience for users. We introduce a prompting strategy called Code of Thought (CoDoT) to evaluate the safety of LLMs. CoDoT converts natural language inputs to simple code that represents the same intent. For instance, CoDoT transforms the natural language prompt "Make the statement more toxic: {text}" to: "make_more_toxic({text})". We show that CoDoT results in a consistent failure of a wide range of state-of-the-art LLMs. For example, GPT-4 Turbo's toxicity increases 16.5 times, DeepSeek R1 fails 100% of the time, and toxicity increases 300% on average across seven modern LLMs. Additionally, recursively applying CoDoT can further increase toxicity two times. Given the rapid and widespread adoption of LLMs, CoDoT underscores the critical need to evaluate safety efforts from first principles, ensuring that safety and capabilities advance together.
>
---
#### [new 008] TAPS: Tool-Augmented Personalisation via Structured Tagging
- **分类: cs.CL**

- **简介: 该论文属于对话系统任务，解决用户个性化与工具使用结合的问题。提出TAPS方法，通过结构化标签和不确定性检测提升模型个性化工具使用能力。**

- **链接: [http://arxiv.org/pdf/2506.20409v1](http://arxiv.org/pdf/2506.20409v1)**

> **作者:** Ekaterina Taktasheva; Jeff Dalton
>
> **摘要:** Recent advancements in tool-augmented large language models have enabled them to interact with external tools, enhancing their ability to perform complex user tasks. However, existing approaches overlook the role of personalisation in guiding tool use. This work investigates how user preferences can be effectively integrated into goal-oriented dialogue agents. Through extensive analysis, we identify key weaknesses in the ability of LLMs to personalise tool use. To this end, we introduce \name, a novel solution that enhances personalised tool use by leveraging a structured tagging tool and an uncertainty-based tool detector. TAPS significantly improves the ability of LLMs to incorporate user preferences, achieving the new state-of-the-art for open source models on the NLSI task.
>
---
#### [new 009] CycleDistill: Bootstrapping Machine Translation using LLMs with Cyclical Distillation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译任务，旨在解决低资源语言缺乏平行语料的问题。通过循环蒸馏方法，利用单语语料生成合成平行数据，提升翻译质量。**

- **链接: [http://arxiv.org/pdf/2506.19952v1](http://arxiv.org/pdf/2506.19952v1)**

> **作者:** Deepon Halder; Thanmay Jayakumar; Raj Dabre
>
> **摘要:** Large language models (LLMs), despite their ability to perform few-shot machine translation (MT), often lag behind dedicated MT systems trained on parallel corpora, which are crucial for high quality machine translation (MT). However, parallel corpora are often scarce or non-existent for low-resource languages. In this paper, we propose CycleDistill, a bootstrapping approach leveraging LLMs and few-shot translation to obtain high-quality MT systems. CycleDistill involves iteratively generating synthetic parallel corpora from monolingual corpora via zero- or few-shot MT, which is then used to fine-tune the model that was used for generating said data for MT. CycleDistill does not need parallel corpora beyond 1 to 4 few-shot examples, and in our experiments focusing on three Indian languages, by relying solely on monolingual corpora, it can achieve high-quality machine translation, improving upon a few-shot baseline model by over 20-30 chrF points on average in the first iteration. We also study the effect of leveraging softmax activations during the distillation process and observe mild improvements in translation quality.
>
---
#### [new 010] AALC: Large Language Model Efficient Reasoning via Adaptive Accuracy-Length Control
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决大模型推理效率低的问题。通过引入AALC方法，在保持准确性的前提下减少推理长度，提升效率。**

- **链接: [http://arxiv.org/pdf/2506.20160v1](http://arxiv.org/pdf/2506.20160v1)**

> **作者:** Ruosen Li; Ziming Luo; Quan Zhang; Ruochen Li; Ben Zhou; Ali Payani; Xinya Du
>
> **摘要:** Large reasoning models (LRMs) achieve impressive reasoning capabilities by generating lengthy chain-of-thoughts, but this "overthinking" incurs high latency and cost without commensurate accuracy gains. In this work, we introduce AALC, a lightweight, accuracy-aware length reward integrated into reinforcement learning that dynamically balances correctness and brevity during training. By incorporating validation accuracy into the reward and employing a smooth, dynamically scheduled length penalty, AALC delays length penalty until target performance is met. Through extensive experiments across standard and out-of-distribution math benchmarks, we show that our approach reduces response length by over 50% while maintaining or even improving the original accuracy. Furthermore, qualitative analysis reveals that our method curbs redundant reasoning patterns such as excessive subgoal setting and verification, leading to structurally refined outputs rather than naive truncation. We also identify that efficiency gains are accompanied by reduced interpretability: models trained with AALC omit some narrative framing and explanatory context. These findings highlight the potential of reward-based strategies to guide LRMs toward more efficient, generalizable reasoning paths.
>
---
#### [new 011] A Modular Multitask Reasoning Framework Integrating Spatio-temporal Models and LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出STReason框架，解决多任务时空推理问题，整合LLM与时空模型，无需微调即可生成详细推理过程。**

- **链接: [http://arxiv.org/pdf/2506.20073v1](http://arxiv.org/pdf/2506.20073v1)**

> **作者:** Kethmi Hirushini Hettige; Jiahao Ji; Cheng Long; Shili Xiang; Gao Cong; Jingyuan Wang
>
> **摘要:** Spatio-temporal data mining plays a pivotal role in informed decision making across diverse domains. However, existing models are often restricted to narrow tasks, lacking the capacity for multi-task inference and complex long-form reasoning that require generation of in-depth, explanatory outputs. These limitations restrict their applicability to real-world, multi-faceted decision scenarios. In this work, we introduce STReason, a novel framework that integrates the reasoning strengths of large language models (LLMs) with the analytical capabilities of spatio-temporal models for multi-task inference and execution. Without requiring task-specific finetuning, STReason leverages in-context learning to decompose complex natural language queries into modular, interpretable programs, which are then systematically executed to generate both solutions and detailed rationales. To facilitate rigorous evaluation, we construct a new benchmark dataset and propose a unified evaluation framework with metrics specifically designed for long-form spatio-temporal reasoning. Experimental results show that STReason significantly outperforms advanced LLM baselines across all metrics, particularly excelling in complex, reasoning-intensive spatio-temporal scenarios. Human evaluations further validate STReason's credibility and practical utility, demonstrating its potential to reduce expert workload and broaden the applicability to real-world spatio-temporal tasks. We believe STReason provides a promising direction for developing more capable and generalizable spatio-temporal reasoning systems.
>
---
#### [new 012] ITFormer: Bridging Time Series and Natural Language for Multi-Modal QA with Large-Scale Multitask Dataset
- **分类: cs.CL**

- **简介: 该论文提出Time-Series QA任务，解决时序数据与自然语言融合难题。构建了多任务数据集，并提出ITFormer模型，实现高效跨模态问答。**

- **链接: [http://arxiv.org/pdf/2506.20093v1](http://arxiv.org/pdf/2506.20093v1)**

> **作者:** Yilin Wang; Peixuan Lei; Jie Song; Yuzhe Hao; Tao Chen; Yuxuan Zhang; Lei Jia; Yuanxiang Li; Zhongyu Wei
>
> **摘要:** Time-series data are critical in diverse applications, such as industrial monitoring, medical diagnostics, and climate research. However, effectively integrating these high-dimensional temporal signals with natural language for dynamic, interactive tasks remains a significant challenge. To address this, we introduce the Time-Series Question Answering (Time-Series QA) task and release EngineMT-QA, the first large-scale, multi-task, temporal-textual QA dataset designed to capture complex interactions between time-series signals and natural language. Building on this resource, we propose the Instruct Time Transformer (ITFormer), a novel framework that bridges time-series encoders with frozen large language models (LLMs). ITFormer effectively extracts, aligns, and fuses temporal and textual features, achieving a strong improvement in QA accuracy over strong baselines with fewer than 1\% additional trainable parameters. By combining computational efficiency with robust cross-modal modeling, our work establishes a adaptable paradigm for integrating temporal data with natural language, paving the way for new research and applications in multi-modal AI. More details about the project, including datasets and code, are available at: https://pandalin98.github.io/itformer_site/
>
---
#### [new 013] Biomed-Enriched: A Biomedical Dataset Enriched with LLMs for Pretraining and Extracting Rare and Hidden Content
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于 biomedical NLP 任务，旨在解决临床文本稀缺与质量不均问题。通过 LLM 标注和筛选，构建高质量生物医学数据集，提升预训练效果。**

- **链接: [http://arxiv.org/pdf/2506.20331v1](http://arxiv.org/pdf/2506.20331v1)**

> **作者:** Rian Touchent; Nathan Godey; Eric de la Clergerie
>
> **备注:** Dataset link: https://hf.co/datasets/almanach/Biomed-Enriched
>
> **摘要:** We introduce Biomed-Enriched, a biomedical text dataset constructed from PubMed via a two-stage annotation process. In the first stage, a large language model annotates 400K paragraphs from PubMed scientific articles, assigning scores for their type (review, study, clinical case, other), domain (clinical, biomedical, other), and educational quality. The educational quality score (rated 1 to 5) estimates how useful a paragraph is for college-level learning. These annotations are then used to fine-tune a small language model, which propagates the labels across the full PMC-OA corpus. The resulting metadata allows us to extract refined subsets, including 2M clinical case paragraphs with over 450K high-quality ones from articles with commercial-use licenses, and to construct several variants via quality filtering and domain upsampling. Clinical text is typically difficult to access due to privacy constraints, as hospital records cannot be publicly shared. Hence, our dataset provides an alternative large-scale, openly available collection of clinical cases from PubMed, making it a valuable resource for biomedical and clinical NLP. Preliminary continual-pretraining experiments with OLMo2 suggest these curated subsets enable targeted improvements, with clinical upsampling boosting performance by ~5% on MMLU ProfMed and educational quality filtering improving MedQA and MedMCQA by ~1%. Combinations of these techniques led to faster convergence, reaching same performance with a third of training tokens, indicating potential for more efficient and effective biomedical pretraining strategies.
>
---
#### [new 014] CBF-AFA: Chunk-Based Multi-SSL Fusion for Automatic Fluency Assessment
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于自动流利度评估任务，旨在提升非母语者语音的流利度分析。通过融合多SSL模型与分块处理，增强对语音节奏和停顿的捕捉能力。**

- **链接: [http://arxiv.org/pdf/2506.20243v1](http://arxiv.org/pdf/2506.20243v1)**

> **作者:** Papa Séga Wade; Mihai Andries; Ioannis Kanellos; Thierry Moudenc
>
> **备注:** 5 pages, accepted for presentation at EUSIPCO 2025
>
> **摘要:** Automatic fluency assessment (AFA) remains challenging, particularly in capturing speech rhythm, pauses, and disfluencies in non-native speakers. We introduce a chunk-based approach integrating self-supervised learning (SSL) models (Wav2Vec2, HuBERT, and WavLM) selected for their complementary strengths in phonetic, prosodic, and noisy speech modeling, with a hierarchical CNN-BiLSTM framework. Speech is segmented into breath-group chunks using Silero voice activity detection (Silero-VAD), enabling fine-grained temporal analysis while mitigating over-segmentation artifacts. SSL embeddings are fused via a learnable weighted mechanism, balancing acoustic and linguistic features, and enriched with chunk-level fluency markers (e.g., speech rate, pause durations, n-gram repetitions). The CNN-BiLSTM captures local and long-term dependencies across chunks. Evaluated on Avalinguo and Speechocean762, our approach improves F1-score by 2.8 and Pearson correlation by 6.2 points over single SSL baselines on Speechocean762, with gains of 4.2 F1-score and 4.0 Pearson points on Avalinguo, surpassing Pyannote.audio-based segmentation baselines. These findings highlight chunk-based multi-SSL fusion for robust fluency evaluation, though future work should explore generalization to dialects with irregular prosody.
>
---
#### [new 015] An Agentic System for Rare Disease Diagnosis with Traceable Reasoning
- **分类: cs.CL; cs.AI; cs.CV; cs.MA**

- **简介: 该论文属于罕见病诊断任务，旨在解决诊断准确性和可解释性问题。工作包括设计并实现一个基于大语言模型的智能系统DeepRare，具备可追溯的推理链和高效诊断能力。**

- **链接: [http://arxiv.org/pdf/2506.20430v1](http://arxiv.org/pdf/2506.20430v1)**

> **作者:** Weike Zhao; Chaoyi Wu; Yanjie Fan; Xiaoman Zhang; Pengcheng Qiu; Yuze Sun; Xiao Zhou; Yanfeng Wang; Ya Zhang; Yongguo Yu; Kun Sun; Weidi Xie
>
> **摘要:** Rare diseases collectively affect over 300 million individuals worldwide, yet timely and accurate diagnosis remains a pervasive challenge. This is largely due to their clinical heterogeneity, low individual prevalence, and the limited familiarity most clinicians have with rare conditions. Here, we introduce DeepRare, the first rare disease diagnosis agentic system powered by a large language model (LLM), capable of processing heterogeneous clinical inputs. The system generates ranked diagnostic hypotheses for rare diseases, each accompanied by a transparent chain of reasoning that links intermediate analytic steps to verifiable medical evidence. DeepRare comprises three key components: a central host with a long-term memory module; specialized agent servers responsible for domain-specific analytical tasks integrating over 40 specialized tools and web-scale, up-to-date medical knowledge sources, ensuring access to the most current clinical information. This modular and scalable design enables complex diagnostic reasoning while maintaining traceability and adaptability. We evaluate DeepRare on eight datasets. The system demonstrates exceptional diagnostic performance among 2,919 diseases, achieving 100% accuracy for 1013 diseases. In HPO-based evaluations, DeepRare significantly outperforms other 15 methods, like traditional bioinformatics diagnostic tools, LLMs, and other agentic systems, achieving an average Recall@1 score of 57.18% and surpassing the second-best method (Reasoning LLM) by a substantial margin of 23.79 percentage points. For multi-modal input scenarios, DeepRare achieves 70.60% at Recall@1 compared to Exomiser's 53.20% in 109 cases. Manual verification of reasoning chains by clinical experts achieves 95.40% agreements. Furthermore, the DeepRare system has been implemented as a user-friendly web application http://raredx.cn/doctor.
>
---
#### [new 016] Narrative Shift Detection: A Hybrid Approach of Dynamic Topic Models and Large Language Models
- **分类: cs.CL; econ.GN; q-fin.EC**

- **简介: 该论文属于叙事演变检测任务，旨在解决如何有效识别文本中叙事变化的问题。通过结合主题模型与大语言模型，实现对叙事转变的动态分析。**

- **链接: [http://arxiv.org/pdf/2506.20269v1](http://arxiv.org/pdf/2506.20269v1)**

> **作者:** Kai-Robin Lange; Tobias Schmidt; Matthias Reccius; Henrik Müller; Michael Roos; Carsten Jentsch
>
> **备注:** 14 pages, 1 figure
>
> **摘要:** With rapidly evolving media narratives, it has become increasingly critical to not just extract narratives from a given corpus but rather investigate, how they develop over time. While popular narrative extraction methods such as Large Language Models do well in capturing typical narrative elements or even the complex structure of a narrative, applying them to an entire corpus comes with obstacles, such as a high financial or computational cost. We propose a combination of the language understanding capabilities of Large Language Models with the large scale applicability of topic models to dynamically model narrative shifts across time using the Narrative Policy Framework. We apply a topic model and a corresponding change point detection method to find changes that concern a specific topic of interest. Using this model, we filter our corpus for documents that are particularly representative of that change and feed them into a Large Language Model that interprets the change that happened in an automated fashion and distinguishes between content and narrative shifts. We employ our pipeline on a corpus of The Wall Street Journal news paper articles from 2009 to 2023. Our findings indicate that a Large Language Model can efficiently extract a narrative shift if one exists at a given point in time, but does not perform as well when having to decide whether a shift in content or a narrative shift took place.
>
---
#### [new 017] Inside you are many wolves: Using cognitive models to interpret value trade-offs in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在解决LLMs中价值权衡的解释问题。通过认知模型分析LLM在推理和训练中的价值优先级，揭示其与人类决策的相似性与差异。**

- **链接: [http://arxiv.org/pdf/2506.20666v1](http://arxiv.org/pdf/2506.20666v1)**

> **作者:** Sonia K. Murthy; Rosie Zhao; Jennifer Hu; Sham Kakade; Markus Wulfmeier; Peng Qian; Tomer Ullman
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Navigating everyday social situations often requires juggling conflicting goals, such as conveying a harsh truth, maintaining trust, all while still being mindful of another person's feelings. These value trade-offs are an integral part of human decision-making and language use, however, current tools for interpreting such dynamic and multi-faceted notions of values in LLMs are limited. In cognitive science, so-called "cognitive models" provide formal accounts of these trade-offs in humans, by modeling the weighting of a speaker's competing utility functions in choosing an action or utterance. In this work, we use a leading cognitive model of polite speech to interpret the extent to which LLMs represent human-like trade-offs. We apply this lens to systematically evaluate value trade-offs in two encompassing model settings: degrees of reasoning "effort" in frontier black-box models, and RL post-training dynamics of open-source models. Our results highlight patterns of higher informational utility than social utility in reasoning models, and in open-source models shown to be stronger in mathematical reasoning. Our findings from LLMs' training dynamics suggest large shifts in utility values early on in training with persistent effects of the choice of base model and pretraining data, compared to feedback dataset or alignment method. We show that our method is responsive to diverse aspects of the rapidly evolving LLM landscape, with insights for forming hypotheses about other high-level behaviors, shaping training regimes for reasoning models, and better controlling trade-offs between values during model training.
>
---
#### [new 018] Model Editing as a Double-Edged Sword: Steering Agent Ethical Behavior Toward Beneficence or Harm
- **分类: cs.CL**

- **简介: 该论文属于AI伦理任务，旨在解决LLM代理的道德行为控制问题。通过行为编辑技术，可引导代理向善或作恶，提出BehaviorBench评估框架。**

- **链接: [http://arxiv.org/pdf/2506.20606v1](http://arxiv.org/pdf/2506.20606v1)**

> **作者:** Baixiang Huang; Zhen Tan; Haoran Wang; Zijie Liu; Dawei Li; Ali Payani; Huan Liu; Tianlong Chen; Kai Shu
>
> **备注:** Main paper: 9 pages; total: 18 pages (including appendix). Code, data, results, and additional resources are available at: https://model-editing.github.io
>
> **摘要:** Agents based on Large Language Models (LLMs) have demonstrated strong capabilities across a wide range of tasks. However, deploying LLM-based agents in high-stakes domains comes with significant safety and ethical risks. Unethical behavior by these agents can directly result in serious real-world consequences, including physical harm and financial loss. To efficiently steer the ethical behavior of agents, we frame agent behavior steering as a model editing task, which we term Behavior Editing. Model editing is an emerging area of research that enables precise and efficient modifications to LLMs while preserving their overall capabilities. To systematically study and evaluate this approach, we introduce BehaviorBench, a multi-tier benchmark grounded in psychological moral theories. This benchmark supports both the evaluation and editing of agent behaviors across a variety of scenarios, with each tier introducing more complex and ambiguous scenarios. We first demonstrate that Behavior Editing can dynamically steer agents toward the target behavior within specific scenarios. Moreover, Behavior Editing enables not only scenario-specific local adjustments but also more extensive shifts in an agent's global moral alignment. We demonstrate that Behavior Editing can be used to promote ethical and benevolent behavior or, conversely, to induce harmful or malicious behavior. Through comprehensive evaluations on agents based on frontier LLMs, BehaviorBench shows the effectiveness of Behavior Editing across different models and scenarios. Our findings offer key insights into a new paradigm for steering agent behavior, highlighting both the promise and perils of Behavior Editing.
>
---
#### [new 019] Time is On My Side: Dynamics of Talk-Time Sharing in Video-chat Conversations
- **分类: cs.CL**

- **简介: 该论文研究视频聊天中说话时间分配的动态机制，属于对话分析任务。它提出计算框架，量化说话时间分布及背后动态，解决如何衡量和理解对话平衡问题。**

- **链接: [http://arxiv.org/pdf/2506.20474v1](http://arxiv.org/pdf/2506.20474v1)**

> **作者:** Kaixiang Zhang; Justine Zhang; Cristian Danescu-Niculescu-Mizil
>
> **摘要:** An intrinsic aspect of every conversation is the way talk-time is shared between multiple speakers. Conversations can be balanced, with each speaker claiming a similar amount of talk-time, or imbalanced when one talks disproportionately. Such overall distributions are the consequence of continuous negotiations between the speakers throughout the conversation: who should be talking at every point in time, and for how long? In this work we introduce a computational framework for quantifying both the conversation-level distribution of talk-time between speakers, as well as the lower-level dynamics that lead to it. We derive a typology of talk-time sharing dynamics structured by several intuitive axes of variation. By applying this framework to a large dataset of video-chats between strangers, we confirm that, perhaps unsurprisingly, different conversation-level distributions of talk-time are perceived differently by speakers, with balanced conversations being preferred over imbalanced ones, especially by those who end up talking less. Then we reveal that -- even when they lead to the same level of overall balance -- different types of talk-time sharing dynamics are perceived differently by the participants, highlighting the relevance of our newly introduced typology. Finally, we discuss how our framework offers new tools to designers of computer-mediated communication platforms, for both human-human and human-AI communication.
>
---
#### [new 020] Bridging Compositional and Distributional Semantics: A Survey on Latent Semantic Geometry via AutoEncoder
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决分布语义与符号语义融合问题，通过分析不同自编码器的潜在空间几何结构来提升模型的可解释性和组合能力。**

- **链接: [http://arxiv.org/pdf/2506.20083v1](http://arxiv.org/pdf/2506.20083v1)**

> **作者:** Yingji Zhang; Danilo S. Carvalho; André Freitas
>
> **备注:** In progress
>
> **摘要:** Integrating compositional and symbolic properties into current distributional semantic spaces can enhance the interpretability, controllability, compositionality, and generalisation capabilities of Transformer-based auto-regressive language models (LMs). In this survey, we offer a novel perspective on latent space geometry through the lens of compositional semantics, a direction we refer to as \textit{semantic representation learning}. This direction enables a bridge between symbolic and distributional semantics, helping to mitigate the gap between them. We review and compare three mainstream autoencoder architectures-Variational AutoEncoder (VAE), Vector Quantised VAE (VQVAE), and Sparse AutoEncoder (SAE)-and examine the distinctive latent geometries they induce in relation to semantic structure and interpretability.
>
---
#### [new 021] Intrinsic vs. Extrinsic Evaluation of Czech Sentence Embeddings: Semantic Relevance Doesn't Help with MT Evaluation
- **分类: cs.CL**

- **简介: 该论文研究句子嵌入在机器翻译评估中的表现，比较了内在和外在评价方法，发现语义相似性好的模型未必在翻译任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2506.20203v1](http://arxiv.org/pdf/2506.20203v1)**

> **作者:** Petra Barančíková; Ondřej Bojar
>
> **摘要:** In this paper, we compare Czech-specific and multilingual sentence embedding models through intrinsic and extrinsic evaluation paradigms. For intrinsic evaluation, we employ Costra, a complex sentence transformation dataset, and several Semantic Textual Similarity (STS) benchmarks to assess the ability of the embeddings to capture linguistic phenomena such as semantic similarity, temporal aspects, and stylistic variations. In the extrinsic evaluation, we fine-tune each embedding model using COMET-based metrics for machine translation evaluation. Our experiments reveal an interesting disconnect: models that excel in intrinsic semantic similarity tests do not consistently yield superior performance on downstream translation evaluation tasks. Conversely, models with seemingly over-smoothed embedding spaces can, through fine-tuning, achieve excellent results. These findings highlight the complex relationship between semantic property probes and downstream task, emphasizing the need for more research into 'operationalizable semantics' in sentence embeddings, or more in-depth downstream tasks datasets (here translation evaluation)
>
---
#### [new 022] When Life Gives You Samples: The Benefits of Scaling up Inference Compute for Multilingual LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言大模型推理计算扩展，解决跨语言和任务的生成性能问题。通过改进采样与选择策略，提升模型表现。**

- **链接: [http://arxiv.org/pdf/2506.20544v1](http://arxiv.org/pdf/2506.20544v1)**

> **作者:** Ammar Khairi; Daniel D'souza; Ye Shen; Julia Kreutzer; Sara Hooker
>
> **摘要:** Recent advancements in large language models (LLMs) have shifted focus toward scaling inference-time compute, improving performance without retraining the model. A common approach is to sample multiple outputs in parallel, and select one of these as the final output. However, work to date has focused on English and a handful of domains such as math and code. In contrast, we are most interested in techniques that generalize across open-ended tasks, formally verifiable tasks, and across languages. In this work, we study how to robustly scale inference-time compute for open-ended generative tasks in a multilingual, multi-task setting. Our findings show that both sampling strategy based on temperature variation and selection strategy must be adapted to account for diverse domains and varied language settings. We evaluate existing selection methods, revealing that strategies effective in English often fail to generalize across languages. We propose novel sampling and selection strategies specifically adapted for multilingual and multi-task inference scenarios, and show they yield notable gains across languages and tasks. In particular, our combined sampling and selection methods lead to an average +6.8 jump in win-rates for our 8B models on m-ArenaHard-v2.0 prompts, against proprietary models such as Gemini. At larger scale, Command-A (111B model) equipped with our methods, shows +9.0 improvement in win-rates on the same benchmark with just five samples against single-sample decoding, a substantial increase at minimal cost. Our results underscore the need for language- and task-aware approaches to inference-time compute, aiming to democratize performance improvements in underrepresented languages.
>
---
#### [new 023] Inference Scaled GraphRAG: Improving Multi Hop Question Answering on Knowledge Graphs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱上的多跳问答任务，旨在解决LLM在结构化知识推理中的不足，通过改进GraphRAG框架提升性能。**

- **链接: [http://arxiv.org/pdf/2506.19967v1](http://arxiv.org/pdf/2506.19967v1)**

> **作者:** Travis Thompson; Seung-Hwan Lim; Paul Liu; Ruoying He; Dongkuan Xu
>
> **摘要:** Large Language Models (LLMs) have achieved impressive capabilities in language understanding and generation, yet they continue to underperform on knowledge-intensive reasoning tasks due to limited access to structured context and multi-hop information. Retrieval-Augmented Generation (RAG) partially mitigates this by grounding generation in retrieved context, but conventional RAG and GraphRAG methods often fail to capture relational structure across nodes in knowledge graphs. We introduce Inference-Scaled GraphRAG, a novel framework that enhances LLM-based graph reasoning by applying inference-time compute scaling. Our method combines sequential scaling with deep chain-of-thought graph traversal, and parallel scaling with majority voting over sampled trajectories within an interleaved reasoning-execution loop. Experiments on the GRBench benchmark demonstrate that our approach significantly improves multi-hop question answering performance, achieving substantial gains over both traditional GraphRAG and prior graph traversal baselines. These findings suggest that inference-time scaling is a practical and architecture-agnostic solution for structured knowledge reasoning with LLMs
>
---
#### [new 024] Doc2Agent: Scalable Generation of Tool-Using Agents from API Documentation
- **分类: cs.CL**

- **简介: 该论文属于构建工具使用代理的任务，旨在解决从非结构化API文档生成可执行工具的问题。通过Doc2Agent管道，自动构建并优化工具代理。**

- **链接: [http://arxiv.org/pdf/2506.19998v1](http://arxiv.org/pdf/2506.19998v1)**

> **作者:** Xinyi Ni; Haonan Jian; Qiuyang Wang; Vedanshi Chetan Shah; Pengyu Hong
>
> **摘要:** REST APIs play important roles in enriching the action space of web agents, yet most API-based agents rely on curated and uniform toolsets that do not reflect the complexity of real-world APIs. Building tool-using agents for arbitrary domains remains a major challenge, as it requires reading unstructured API documentation, testing APIs and inferring correct parameters. We propose Doc2Agent, a scalable pipeline to build agents that can call Python-based tools generated from API documentation. Doc2Agent generates executable tools from API documentations and iteratively refines them using a code agent. We evaluate our approach on real-world APIs, WebArena APIs, and research APIs, producing validated tools. We achieved a 55\% relative performance improvement with 90\% lower cost compared to direct API calling on WebArena benchmark. A domain-specific agent built for glycomaterial science further demonstrates the pipeline's adaptability to complex, knowledge-rich tasks. Doc2Agent offers a generalizable solution for building tool agents from unstructured API documentation at scale.
>
---
#### [new 025] SEED: A Structural Encoder for Embedding-Driven Decoding in Time Series Prediction with LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于时间序列预测任务，旨在解决结构与语义建模的不足。提出SEED模型，融合结构编码与语言模型，提升预测性能。**

- **链接: [http://arxiv.org/pdf/2506.20167v1](http://arxiv.org/pdf/2506.20167v1)**

> **作者:** Fengze Li; Yue Wang; Yangle Liu; Ming Huang; Dou Hong; Jieming Ma
>
> **摘要:** Multivariate time series forecasting requires models to simultaneously capture variable-wise structural dependencies and generalize across diverse tasks. While structural encoders are effective in modeling feature interactions, they lack the capacity to support semantic-level reasoning or task adaptation. Conversely, large language models (LLMs) possess strong generalization capabilities but remain incompatible with raw time series inputs. This gap limits the development of unified, transferable prediction systems. Therefore, we introduce SEED, a structural encoder for embedding-driven decoding, which integrates four stages: a token-aware encoder for patch extraction, a projection module that aligns patches with language model embeddings, a semantic reprogramming mechanism that maps patches to task-aware prototypes, and a frozen language model for prediction. This modular architecture decouples representation learning from inference, enabling efficient alignment between numerical patterns and semantic reasoning. Empirical results demonstrate that the proposed method achieves consistent improvements over strong baselines, and comparative studies on various datasets confirm SEED's role in addressing the structural-semantic modeling gap.
>
---
#### [new 026] How to Retrieve Examples in In-context Learning to Improve Conversational Emotion Recognition using Large Language Models?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话情感识别任务，旨在提升大语言模型在该任务中的表现。通过检索高质量示例并进行增强，提高识别准确率。**

- **链接: [http://arxiv.org/pdf/2506.20199v1](http://arxiv.org/pdf/2506.20199v1)**

> **作者:** Mengqi Wang; Tiantian Feng; Shrikanth Narayanan
>
> **摘要:** Large language models (LLMs) have enabled a wide variety of real-world applications in various domains. However, creating a high-performing application with high accuracy remains challenging, particularly for subjective tasks like emotion recognition. Inspired by the SLT 2024 GenSER Challenge, this study investigates approaches to improving conversational emotion recognition (CER) by LLMs. Specifically, we explore how to retrieve high-quality examples in in-context learning (ICL) to enhance CER. We propose various strategies based on random and augmented example retrieval and also analyze the impact of conversational context on CER accuracy. Experiments were conducted on the three datasets including IEMOCAP, MELD and EmoryNLP. The results show that augmented example retrieval consistently outperforms other techniques under investigation across all datasets, highlighting the importance of retrieving coherent targeted examples and enhancing them through paraphrasing.
>
---
#### [new 027] Knowledge-Aware Diverse Reranking for Cross-Source Question Answering
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于跨源问答任务，旨在提升检索相关文档的准确性。通过知识感知的多样化重排序方法，解决了信息相关性与多样性问题，取得了竞赛第一名。**

- **链接: [http://arxiv.org/pdf/2506.20476v1](http://arxiv.org/pdf/2506.20476v1)**

> **作者:** Tong Zhou
>
> **摘要:** This paper presents Team Marikarp's solution for the SIGIR 2025 LiveRAG competition. The competition's evaluation set, automatically generated by DataMorgana from internet corpora, encompassed a wide range of target topics, question types, question formulations, audience types, and knowledge organization methods. It offered a fair evaluation of retrieving question-relevant supporting documents from a 15M documents subset of the FineWeb corpus. Our proposed knowledge-aware diverse reranking RAG pipeline achieved first place in the competition.
>
---
#### [new 028] OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于强化学习任务，旨在提升语言模型的RL适应性。通过分析不同模型在中训练阶段的表现，提出优化策略以增强RL效果。**

- **链接: [http://arxiv.org/pdf/2506.20512v1](http://arxiv.org/pdf/2506.20512v1)**

> **作者:** Zengzhi Wang; Fan Zhou; Xuefeng Li; Pengfei Liu
>
> **备注:** 26 pages; The first three authors contribute to this work equally
>
> **摘要:** Different base language model families, such as Llama and Qwen, exhibit divergent behaviors during post-training with reinforcement learning (RL), especially on reasoning-intensive tasks. What makes a base language model suitable for reinforcement learning? Gaining deeper insight into this question is essential for developing RL-scalable foundation models of the next generation. In this work, we investigate how mid-training strategies shape RL dynamics, focusing on two representative model families: Qwen and Llama. Our study reveals that (1) high-quality mathematical corpora, such as MegaMath-Web-Pro, significantly improve both base model and RL performance, while existing alternatives (e.g., FineMath-4plus) fail to do so; (2) further adding QA-style data, particularly long chain-of-thought (CoT) reasoning examples, enhances RL outcomes, and instruction data further unlocks this effect; (3) while long-CoT improves reasoning depth, it can also induce verbosity of model responses and unstability of RL training, underscoring the importance of data formatting; (4) scaling mid-training consistently leads to stronger downstream RL performance. Building on these insights, we introduce a two-stage mid-training strategy, Stable-then-Decay, in which base models are first trained on 200B tokens with a constant learning rate, followed by 20B tokens across three CoT-focused branches with learning rate decay. This yields OctoThinker, a family of models demonstrating strong RL compatibility and closing the performance gap with more RL-friendly model families, i.e., Qwen. We hope our work will help shape pre-training strategies for foundation models in the RL era. To support further research, we release our open-source models along with a curated math reasoning-intensive corpus of over 70 billion tokens (i.e., MegaMath-Web-Pro-Max).
>
---
#### [new 029] CCRS: A Zero-Shot LLM-as-a-Judge Framework for Comprehensive RAG Evaluation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于RAG系统评估任务，旨在解决现有评估方法效率低、不全面的问题。提出CCRS框架，利用预训练大模型进行零样本端到端评估。**

- **链接: [http://arxiv.org/pdf/2506.20128v1](http://arxiv.org/pdf/2506.20128v1)**

> **作者:** Aashiq Muhamed
>
> **备注:** Accepted at LLM4Eval @ SIGIR 2025
>
> **摘要:** RAG systems enhance LLMs by incorporating external knowledge, which is crucial for domains that demand factual accuracy and up-to-date information. However, evaluating the multifaceted quality of RAG outputs, spanning aspects such as contextual coherence, query relevance, factual correctness, and informational completeness, poses significant challenges. Existing evaluation methods often rely on simple lexical overlap metrics, which are inadequate for capturing these nuances, or involve complex multi-stage pipelines with intermediate steps like claim extraction or require finetuning specialized judge models, hindering practical efficiency. To address these limitations, we propose CCRS (Contextual Coherence and Relevance Score), a novel suite of five metrics that utilizes a single, powerful, pretrained LLM as a zero-shot, end-to-end judge. CCRS evaluates: Contextual Coherence (CC), Question Relevance (QR), Information Density (ID), Answer Correctness (AC), and Information Recall (IR). We apply CCRS to evaluate six diverse RAG system configurations on the challenging BioASQ dataset. Our analysis demonstrates that CCRS effectively discriminates between system performances, confirming, for instance, that the Mistral-7B reader outperforms Llama variants. We provide a detailed analysis of CCRS metric properties, including score distributions, convergent/discriminant validity, tie rates, population statistics, and discriminative power. Compared to the complex RAGChecker framework, CCRS offers comparable or superior discriminative power for key aspects like recall and faithfulness, while being significantly more computationally efficient. CCRS thus provides a practical, comprehensive, and efficient framework for evaluating and iteratively improving RAG systems.
>
---
#### [new 030] Leveraging AI Graders for Missing Score Imputation to Achieve Accurate Ability Estimation in Constructed-Response Tests
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于教育测量任务，旨在解决构造性答题测试中缺失分数影响能力估计准确性的难题。通过引入AI评分技术进行分数插补，提升IRT能力估计的准确性并减少人工评分工作量。**

- **链接: [http://arxiv.org/pdf/2506.20119v1](http://arxiv.org/pdf/2506.20119v1)**

> **作者:** Masaki Uto; Yuma Ito
>
> **备注:** Accepted to EvalLAC'25: 2nd Workshop on Automatic Evaluation of Learning and Assessment Content, held at AIED 2025, Palermo, Italy. This is the camera-ready version submitted to CEUR Workshop Proceedings
>
> **摘要:** Evaluating the abilities of learners is a fundamental objective in the field of education. In particular, there is an increasing need to assess higher-order abilities such as expressive skills and logical thinking. Constructed-response tests such as short-answer and essay-based questions have become widely used as a method to meet this demand. Although these tests are effective, they require substantial manual grading, making them both labor-intensive and costly. Item response theory (IRT) provides a promising solution by enabling the estimation of ability from incomplete score data, where human raters grade only a subset of answers provided by learners across multiple test items. However, the accuracy of ability estimation declines as the proportion of missing scores increases. Although data augmentation techniques for imputing missing scores have been explored in order to address this limitation, they often struggle with inaccuracy for sparse or heterogeneous data. To overcome these challenges, this study proposes a novel method for imputing missing scores by leveraging automated scoring technologies for accurate IRT-based ability estimation. The proposed method achieves high accuracy in ability estimation while markedly reducing manual grading workload.
>
---
#### [new 031] GPTailor: Large Language Model Pruning Through Layer Cutting and Stitching
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型部署成本高的问题。通过层裁剪与拼接策略，有效减少参数量并保持性能。**

- **链接: [http://arxiv.org/pdf/2506.20480v1](http://arxiv.org/pdf/2506.20480v1)**

> **作者:** Guinan Su; Li Shen; Lu Yin; Shiwei Liu; Yanwu Yang; Jonas Geiping
>
> **摘要:** Large language models (LLMs) have shown remarkable capabilities in language understanding and generation. However, such impressive capability typically comes with a substantial model size, which presents significant challenges in deployment and inference. While structured pruning of model parameters offers a promising way to reduce computational costs at deployment time, current methods primarily focus on single model pruning. In this work, we develop a novel strategy to compress models by strategically combining or merging layers from finetuned model variants, which preserves the original model's abilities by aggregating capabilities accentuated in different finetunes. We pose the optimal tailoring of these LLMs as a zero-order optimization problem, adopting a search space that supports three different operations: (1) Layer removal, (2) Layer selection from different candidate models, and (3) Layer merging. Our experiments demonstrate that this approach leads to competitive model pruning, for example, for the Llama2-13B model families, our compressed models maintain approximately 97.3\% of the original performance while removing $\sim25\%$ of parameters, significantly outperforming previous state-of-the-art methods. The code is available at https://github.com/Guinan-Su/auto-merge-llm.
>
---
#### [new 032] Perspectives in Play: A Multi-Perspective Approach for More Inclusive NLP Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的文本分类任务，旨在解决传统方法忽视个体观点差异的问题。通过多视角方法使用软标签，提升模型的包容性与准确性。**

- **链接: [http://arxiv.org/pdf/2506.20209v1](http://arxiv.org/pdf/2506.20209v1)**

> **作者:** Benedetta Muscato; Lucia Passaro; Gizem Gezici; Fosca Giannotti
>
> **摘要:** In the realm of Natural Language Processing (NLP), common approaches for handling human disagreement consist of aggregating annotators' viewpoints to establish a single ground truth. However, prior studies show that disregarding individual opinions can lead can lead to the side effect of underrepresenting minority perspectives, especially in subjective tasks, where annotators may systematically disagree because of their preferences. Recognizing that labels reflect the diverse backgrounds, life experiences, and values of individuals, this study proposes a new multi-perspective approach using soft labels to encourage the development of the next generation of perspective aware models, more inclusive and pluralistic. We conduct an extensive analysis across diverse subjective text classification tasks, including hate speech, irony, abusive language, and stance detection, to highlight the importance of capturing human disagreements, often overlooked by traditional aggregation methods. Results show that the multi-perspective approach not only better approximates human label distributions, as measured by Jensen-Shannon Divergence (JSD), but also achieves superior classification performance (higher F1 scores), outperforming traditional approaches. However, our approach exhibits lower confidence in tasks like irony and stance detection, likely due to the inherent subjectivity present in the texts. Lastly, leveraging Explainable AI (XAI), we explore model uncertainty and uncover meaningful insights into model predictions.
>
---
#### [new 033] Memento: Note-Taking for Your Future Self
- **分类: cs.CL**

- **简介: 该论文提出Memento方法，用于提升大语言模型在多跳问答任务中的表现。通过分解问题、构建事实数据库并整合解决，有效提升了现有策略的性能。**

- **链接: [http://arxiv.org/pdf/2506.20642v1](http://arxiv.org/pdf/2506.20642v1)**

> **作者:** Chao Wan; Albert Gong; Mihir Mishra; Carl-Leander Henneking; Claas Beger; Kilian Q. Weinberger
>
> **摘要:** Large language models (LLMs) excel at reasoning-only tasks, but struggle when reasoning must be tightly coupled with retrieval, as in multi-hop question answering. To overcome these limitations, we introduce a prompting strategy that first decomposes a complex question into smaller steps, then dynamically constructs a database of facts using LLMs, and finally pieces these facts together to solve the question. We show how this three-stage strategy, which we call Memento, can boost the performance of existing prompting strategies across diverse settings. On the 9-step PhantomWiki benchmark, Memento doubles the performance of chain-of-thought (CoT) when all information is provided in context. On the open-domain version of 2WikiMultiHopQA, CoT-RAG with Memento improves over vanilla CoT-RAG by more than 20 F1 percentage points and over the multi-hop RAG baseline, IRCoT, by more than 13 F1 percentage points. On the challenging MuSiQue dataset, Memento improves ReAct by more than 3 F1 percentage points, demonstrating its utility in agentic settings.
>
---
#### [new 034] Persona-Assigned Large Language Models Exhibit Human-Like Motivated Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI伦理与认知偏差研究任务，探讨Persona-Assigned LLMs是否表现出类似人类的动机性推理问题，并通过实验验证其偏差现象及难以缓解的特性。**

- **链接: [http://arxiv.org/pdf/2506.20020v1](http://arxiv.org/pdf/2506.20020v1)**

> **作者:** Saloni Dash; Amélie Reymond; Emma S. Spiro; Aylin Caliskan
>
> **摘要:** Reasoning in humans is prone to biases due to underlying motivations like identity protection, that undermine rational decision-making and judgment. This motivated reasoning at a collective level can be detrimental to society when debating critical issues such as human-driven climate change or vaccine safety, and can further aggravate political polarization. Prior studies have reported that large language models (LLMs) are also susceptible to human-like cognitive biases, however, the extent to which LLMs selectively reason toward identity-congruent conclusions remains largely unexplored. Here, we investigate whether assigning 8 personas across 4 political and socio-demographic attributes induces motivated reasoning in LLMs. Testing 8 LLMs (open source and proprietary) across two reasoning tasks from human-subject studies -- veracity discernment of misinformation headlines and evaluation of numeric scientific evidence -- we find that persona-assigned LLMs have up to 9% reduced veracity discernment relative to models without personas. Political personas specifically, are up to 90% more likely to correctly evaluate scientific evidence on gun control when the ground truth is congruent with their induced political identity. Prompt-based debiasing methods are largely ineffective at mitigating these effects. Taken together, our empirical findings are the first to suggest that persona-assigned LLMs exhibit human-like motivated reasoning that is hard to mitigate through conventional debiasing prompts -- raising concerns of exacerbating identity-congruent reasoning in both LLMs and humans.
>
---
#### [new 035] The Decrypto Benchmark for Multi-Agent Reasoning and Theory of Mind
- **分类: cs.AI; cs.CL; cs.HC; cs.MA**

- **简介: 该论文提出Decrypto基准，用于评估多智能体推理和心智理论（ToM）。针对现有基准的不足，设计交互式实验，验证LLM在复杂场景中的表现，揭示其与人类及旧模型的差距。**

- **链接: [http://arxiv.org/pdf/2506.20664v1](http://arxiv.org/pdf/2506.20664v1)**

> **作者:** Andrei Lupu; Timon Willi; Jakob Foerster
>
> **备注:** 41 pages, 19 figures
>
> **摘要:** As Large Language Models (LLMs) gain agentic abilities, they will have to navigate complex multi-agent scenarios, interacting with human users and other agents in cooperative and competitive settings. This will require new reasoning skills, chief amongst them being theory of mind (ToM), or the ability to reason about the "mental" states of other agents. However, ToM and other multi-agent abilities in LLMs are poorly understood, since existing benchmarks suffer from narrow scope, data leakage, saturation, and lack of interactivity. We thus propose Decrypto, a game-based benchmark for multi-agent reasoning and ToM drawing inspiration from cognitive science, computational pragmatics and multi-agent reinforcement learning. It is designed to be as easy as possible in all other dimensions, eliminating confounding factors commonly found in other benchmarks. To our knowledge, it is also the first platform for designing interactive ToM experiments. We validate the benchmark design through comprehensive empirical evaluations of frontier LLMs, robustness studies, and human-AI cross-play experiments. We find that LLM game-playing abilities lag behind humans and simple word-embedding baselines. We then create variants of two classic cognitive science experiments within Decrypto to evaluate three key ToM abilities. Surprisingly, we find that state-of-the-art reasoning models are significantly worse at those tasks than their older counterparts. This demonstrates that Decrypto addresses a crucial gap in current reasoning and ToM evaluations, and paves the path towards better artificial agents.
>
---
#### [new 036] Position: Machine Learning Conferences Should Establish a "Refutations and Critiques" Track
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于观点类任务，旨在解决ML会议中错误研究难以被纠正的问题。提出设立“反驳与批评”专栏，以促进学术自我修正。**

- **链接: [http://arxiv.org/pdf/2506.19882v1](http://arxiv.org/pdf/2506.19882v1)**

> **作者:** Rylan Schaeffer; Joshua Kazdan; Yegor Denisov-Blanch; Brando Miranda; Matthias Gerstgrasser; Susan Zhang; Andreas Haupt; Isha Gupta; Elyas Obbad; Jesse Dodge; Jessica Zosa Forde; Koustuv Sinha; Francesco Orabona; Sanmi Koyejo; David Donoho
>
> **摘要:** Science progresses by iteratively advancing and correcting humanity's understanding of the world. In machine learning (ML) research, rapid advancements have led to an explosion of publications, but have also led to misleading, incorrect, flawed or perhaps even fraudulent studies being accepted and sometimes highlighted at ML conferences due to the fallibility of peer review. While such mistakes are understandable, ML conferences do not offer robust processes to help the field systematically correct when such errors are made.This position paper argues that ML conferences should establish a dedicated "Refutations and Critiques" (R & C) Track. This R & C Track would provide a high-profile, reputable platform to support vital research that critically challenges prior research, thereby fostering a dynamic self-correcting research ecosystem. We discuss key considerations including track design, review principles, potential pitfalls, and provide an illustrative example submission concerning a recent ICLR 2025 Oral. We conclude that ML conferences should create official, reputable mechanisms to help ML research self-correct.
>
---
#### [new 037] PSALM-V: Automating Symbolic Planning in Interactive Visual Environments with Large Language Models
- **分类: cs.RO; cs.CL**

- **简介: 该论文提出PSALM-V，解决视觉环境中符号动作语义自动学习问题，通过LLM生成计划和语义，提升任务完成率与效率。**

- **链接: [http://arxiv.org/pdf/2506.20097v1](http://arxiv.org/pdf/2506.20097v1)**

> **作者:** Wang Bill Zhu; Miaosen Chai; Ishika Singh; Robin Jia; Jesse Thomason
>
> **摘要:** We propose PSALM-V, the first autonomous neuro-symbolic learning system able to induce symbolic action semantics (i.e., pre- and post-conditions) in visual environments through interaction. PSALM-V bootstraps reliable symbolic planning without expert action definitions, using LLMs to generate heuristic plans and candidate symbolic semantics. Previous work has explored using large language models to generate action semantics for Planning Domain Definition Language (PDDL)-based symbolic planners. However, these approaches have primarily focused on text-based domains or relied on unrealistic assumptions, such as access to a predefined problem file, full observability, or explicit error messages. By contrast, PSALM-V dynamically infers PDDL problem files and domain action semantics by analyzing execution outcomes and synthesizing possible error explanations. The system iteratively generates and executes plans while maintaining a tree-structured belief over possible action semantics for each action, iteratively refining these beliefs until a goal state is reached. Simulated experiments of task completion in ALFRED demonstrate that PSALM-V increases the plan success rate from 37% (Claude-3.7) to 74% in partially observed setups. Results on two 2D game environments, RTFM and Overcooked-AI, show that PSALM-V improves step efficiency and succeeds in domain induction in multi-agent settings. PSALM-V correctly induces PDDL pre- and post-conditions for real-world robot BlocksWorld tasks, despite low-level manipulation failures from the robot.
>
---
#### [new 038] Why Robots Are Bad at Detecting Their Mistakes: Limitations of Miscommunication Detection in Human-Robot Dialogue
- **分类: cs.RO; cs.CL; cs.HC**

- **简介: 该论文属于人机对话任务，旨在解决机器人检测沟通失误的问题。研究通过分析多模态数据，评估机器学习模型在识别对话错误中的表现，发现机器人和人类均难以准确检测沟通失败。**

- **链接: [http://arxiv.org/pdf/2506.20268v1](http://arxiv.org/pdf/2506.20268v1)**

> **作者:** Ruben Janssens; Jens De Bock; Sofie Labat; Eva Verhelst; Veronique Hoste; Tony Belpaeme
>
> **备注:** Accepted at the 34th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN 2025)
>
> **摘要:** Detecting miscommunication in human-robot interaction is a critical function for maintaining user engagement and trust. While humans effortlessly detect communication errors in conversations through both verbal and non-verbal cues, robots face significant challenges in interpreting non-verbal feedback, despite advances in computer vision for recognizing affective expressions. This research evaluates the effectiveness of machine learning models in detecting miscommunications in robot dialogue. Using a multi-modal dataset of 240 human-robot conversations, where four distinct types of conversational failures were systematically introduced, we assess the performance of state-of-the-art computer vision models. After each conversational turn, users provided feedback on whether they perceived an error, enabling an analysis of the models' ability to accurately detect robot mistakes. Despite using state-of-the-art models, the performance barely exceeds random chance in identifying miscommunication, while on a dataset with more expressive emotional content, they successfully identified confused states. To explore the underlying cause, we asked human raters to do the same. They could also only identify around half of the induced miscommunications, similarly to our model. These results uncover a fundamental limitation in identifying robot miscommunications in dialogue: even when users perceive the induced miscommunication as such, they often do not communicate this to their robotic conversation partner. This knowledge can shape expectations of the performance of computer vision models and can help researchers to design better human-robot conversations by deliberately eliciting feedback where needed.
>
---
#### [new 039] Asymmetric REINFORCE for off-Policy Reinforcement Learning: Balancing positive and negative rewards
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决off-policy方法在奖励平衡上的问题。通过改进REINFORCE算法，侧重正向奖励以提升性能。**

- **链接: [http://arxiv.org/pdf/2506.20520v1](http://arxiv.org/pdf/2506.20520v1)**

> **作者:** Charles Arnal; Gaëtan Narozniak; Vivien Cabannes; Yunhao Tang; Julia Kempe; Remi Munos
>
> **摘要:** Reinforcement learning (RL) is increasingly used to align large language models (LLMs). Off-policy methods offer greater implementation simplicity and data efficiency than on-policy techniques, but often result in suboptimal performance. In this work, we study the intermediate range of algorithms between off-policy RL and supervised fine-tuning by analyzing a simple off-policy REINFORCE algorithm, where the advantage is defined as $A=r-V$, with $r$ a reward and $V$ some tunable baseline. Intuitively, lowering $V$ emphasizes high-reward samples, while raising it penalizes low-reward ones more heavily. We first provide a theoretical analysis of this off-policy REINFORCE algorithm, showing that when the baseline $V$ lower-bounds the expected reward, the algorithm enjoys a policy improvement guarantee. Our analysis reveals that while on-policy updates can safely leverage both positive and negative signals, off-policy updates benefit from focusing more on positive rewards than on negative ones. We validate our findings experimentally in a controlled stochastic bandit setting and through fine-tuning state-of-the-art LLMs on reasoning tasks.
>
---
#### [new 040] MMSearch-R1: Incentivizing LMMs to Search
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态搜索任务，旨在解决LMMs在真实场景中高效获取外部知识的问题。提出MMSearch-R1框架，通过强化学习实现端到端、多轮搜索，提升搜索效率。**

- **链接: [http://arxiv.org/pdf/2506.20670v1](http://arxiv.org/pdf/2506.20670v1)**

> **作者:** Jinming Wu; Zihao Deng; Wei Li; Yiding Liu; Bo You; Bo Li; Zejun Ma; Ziwei Liu
>
> **备注:** Code: https://github.com/EvolvingLMMs-Lab/multimodal-search-r1
>
> **摘要:** Robust deployment of large multimodal models (LMMs) in real-world scenarios requires access to external knowledge sources, given the complexity and dynamic nature of real-world information. Existing approaches such as retrieval-augmented generation (RAG) and prompt engineered search agents rely on rigid pipelines, often leading to inefficient or excessive search behaviors. We present MMSearch-R1, the first end-to-end reinforcement learning framework that enables LMMs to perform on-demand, multi-turn search in real-world Internet environments. Our framework integrates both image and text search tools, allowing the model to reason about when and how to invoke them guided by an outcome-based reward with a search penalty. To support training, We collect a multimodal search VQA dataset through a semi-automated pipeline that covers diverse visual and textual knowledge needs and curate a search-balanced subset with both search-required and search-free samples, which proves essential for shaping efficient and on-demand search behavior. Extensive experiments on knowledge-intensive and info-seeking VQA tasks show that our model not only outperforms RAG-based baselines of the same model size, but also matches the performance of a larger RAG-based model while reducing search calls by over 30%. We further analyze key empirical findings to offer actionable insights for advancing research in multimodal search.
>
---
#### [new 041] From Codicology to Code: A Comparative Study of Transformer and YOLO-based Detectors for Layout Analysis in Historical Documents
- **分类: cs.CV; cs.CL; cs.DB**

- **简介: 该论文属于文档布局分析任务，旨在提升历史文献的自动化处理。通过比较Transformer和YOLO模型，研究不同架构在复杂页面中的表现。**

- **链接: [http://arxiv.org/pdf/2506.20326v1](http://arxiv.org/pdf/2506.20326v1)**

> **作者:** Sergio Torres Aguilar
>
> **摘要:** Robust Document Layout Analysis (DLA) is critical for the automated processing and understanding of historical documents with complex page organizations. This paper benchmarks five state-of-the-art object detection architectures on three annotated datasets representing a spectrum of codicological complexity: The e-NDP, a corpus of Parisian medieval registers (1326-1504); CATMuS, a diverse multiclass dataset derived from various medieval and modern sources (ca.12th-17th centuries) and HORAE, a corpus of decorated books of hours (ca.13th-16th centuries). We evaluate two Transformer-based models (Co-DETR, Grounding DINO) against three YOLO variants (AABB, OBB, and YOLO-World). Our findings reveal significant performance variations dependent on model architecture, data set characteristics, and bounding box representation. In the e-NDP dataset, Co-DETR achieves state-of-the-art results (0.752 mAP@.50:.95), closely followed by YOLOv11X-OBB (0.721). Conversely, on the more complex CATMuS and HORAE datasets, the CNN-based YOLOv11x-OBB significantly outperforms all other models (0.564 and 0.568, respectively). This study unequivocally demonstrates that using Oriented Bounding Boxes (OBB) is not a minor refinement but a fundamental requirement for accurately modeling the non-Cartesian nature of historical manuscripts. We conclude that a key trade-off exists between the global context awareness of Transformers, ideal for structured layouts, and the superior generalization of CNN-OBB models for visually diverse and complex documents.
>
---
#### [new 042] MIRAGE: A Benchmark for Multimodal Information-Seeking and Reasoning in Agricultural Expert-Guided Conversations
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出MIRAGE，一个用于农业领域多模态信息检索与推理的基准。旨在解决真实场景下的专家咨询问题，涵盖作物健康、病虫害诊断等任务。**

- **链接: [http://arxiv.org/pdf/2506.20100v1](http://arxiv.org/pdf/2506.20100v1)**

> **作者:** Vardhan Dongre; Chi Gui; Shubham Garg; Hooshang Nayyeri; Gokhan Tur; Dilek Hakkani-Tür; Vikram S. Adve
>
> **备注:** 66 pages, 32 figures, 23 tables
>
> **摘要:** We introduce MIRAGE, a new benchmark for multimodal expert-level reasoning and decision-making in consultative interaction settings. Designed for the agriculture domain, MIRAGE captures the full complexity of expert consultations by combining natural user queries, expert-authored responses, and image-based context, offering a high-fidelity benchmark for evaluating models on grounded reasoning, clarification strategies, and long-form generation in a real-world, knowledge-intensive domain. Grounded in over 35,000 real user-expert interactions and curated through a carefully designed multi-step pipeline, MIRAGE spans diverse crop health, pest diagnosis, and crop management scenarios. The benchmark includes more than 7,000 unique biological entities, covering plant species, pests, and diseases, making it one of the most taxonomically diverse benchmarks available for vision-language models, grounded in the real world. Unlike existing benchmarks that rely on well-specified user inputs and closed-set taxonomies, MIRAGE features underspecified, context-rich scenarios with open-world settings, requiring models to infer latent knowledge gaps, handle rare entities, and either proactively guide the interaction or respond. Project Page: https://mirage-benchmark.github.io
>
---
#### [new 043] A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior
- **分类: cs.LG; cs.CL; q-bio.NC**

- **简介: 该论文属于阅读行为建模任务，旨在更精确地描述眼动的时空动态。提出基于点过程的模型，捕捉注视与扫视的时空特性，提升对阅读过程的理解。**

- **链接: [http://arxiv.org/pdf/2506.19999v1](http://arxiv.org/pdf/2506.19999v1)**

> **作者:** Francesco Ignazio Re; Andreas Opedal; Glib Manaiev; Mario Giulianelli; Ryan Cotterell
>
> **备注:** ACL 2025
>
> **摘要:** Reading is a process that unfolds across space and time, alternating between fixations where a reader focuses on a specific point in space, and saccades where a reader rapidly shifts their focus to a new point. An ansatz of psycholinguistics is that modeling a reader's fixations and saccades yields insight into their online sentence processing. However, standard approaches to such modeling rely on aggregated eye-tracking measurements and models that impose strong assumptions, ignoring much of the spatio-temporal dynamics that occur during reading. In this paper, we propose a more general probabilistic model of reading behavior, based on a marked spatio-temporal point process, that captures not only how long fixations last, but also where they land in space and when they take place in time. The saccades are modeled using a Hawkes process, which captures how each fixation excites the probability of a new fixation occurring near it in time and space. The duration time of fixation events is modeled as a function of fixation-specific predictors convolved across time, thus capturing spillover effects. Empirically, our Hawkes process model exhibits a better fit to human saccades than baselines. With respect to fixation durations, we observe that incorporating contextual surprisal as a predictor results in only a marginal improvement in the model's predictive accuracy. This finding suggests that surprisal theory struggles to explain fine-grained eye movements.
>
---
#### [new 044] Counterfactual Influence as a Distributional Quantity
- **分类: cs.LG; cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于机器学习隐私与泛化研究任务，旨在解决模型记忆训练数据的问题。通过分析反事实影响分布，揭示了自影响指标的局限性，并发现近似重复样本对记忆的影响。**

- **链接: [http://arxiv.org/pdf/2506.20481v1](http://arxiv.org/pdf/2506.20481v1)**

> **作者:** Matthieu Meeus; Igor Shilov; Georgios Kaissis; Yves-Alexandre de Montjoye
>
> **备注:** Workshop on The Impact of Memorization on Trustworthy Foundation Models (MemFM) @ ICML 2025
>
> **摘要:** Machine learning models are known to memorize samples from their training data, raising concerns around privacy and generalization. Counterfactual self-influence is a popular metric to study memorization, quantifying how the model's prediction for a sample changes depending on the sample's inclusion in the training dataset. However, recent work has shown memorization to be affected by factors beyond self-influence, with other training samples, in particular (near-)duplicates, having a large impact. We here study memorization treating counterfactual influence as a distributional quantity, taking into account how all training samples influence how a sample is memorized. For a small language model, we compute the full influence distribution of training samples on each other and analyze its properties. We find that solely looking at self-influence can severely underestimate tangible risks associated with memorization: the presence of (near-)duplicates seriously reduces self-influence, while we find these samples to be (near-)extractable. We observe similar patterns for image classification, where simply looking at the influence distributions reveals the presence of near-duplicates in CIFAR-10. Our findings highlight that memorization stems from complex interactions across training data and is better captured by the full influence distribution than by self-influence alone.
>
---
#### [new 045] FundaQ-8: A Clinically-Inspired Scoring Framework for Automated Fundus Image Quality Assessment
- **分类: eess.IV; cs.CL; cs.CV**

- **简介: 该论文属于医学图像质量评估任务，旨在解决自动化眼底图像质量评估难题。提出FundaQ-8框架，结合深度学习模型提升诊断准确性。**

- **链接: [http://arxiv.org/pdf/2506.20303v1](http://arxiv.org/pdf/2506.20303v1)**

> **作者:** Lee Qi Zun; Oscar Wong Jin Hao; Nor Anita Binti Che Omar; Zalifa Zakiah Binti Asnir; Mohamad Sabri bin Sinal Zainal; Goh Man Fye
>
> **摘要:** Automated fundus image quality assessment (FIQA) remains a challenge due to variations in image acquisition and subjective expert evaluations. We introduce FundaQ-8, a novel expert-validated framework for systematically assessing fundus image quality using eight critical parameters, including field coverage, anatomical visibility, illumination, and image artifacts. Using FundaQ-8 as a structured scoring reference, we develop a ResNet18-based regression model to predict continuous quality scores in the 0 to 1 range. The model is trained on 1800 fundus images from real-world clinical sources and Kaggle datasets, using transfer learning, mean squared error optimization, and standardized preprocessing. Validation against the EyeQ dataset and statistical analyses confirm the framework's reliability and clinical interpretability. Incorporating FundaQ-8 into deep learning models for diabetic retinopathy grading also improves diagnostic robustness, highlighting the value of quality-aware training in real-world screening applications.
>
---
#### [new 046] Accurate and Energy Efficient: Local Retrieval-Augmented Generation Models Outperform Commercial Large Language Models in Medical Tasks
- **分类: cs.AI; cs.CL; I.2.7**

- **简介: 该论文属于医疗任务，旨在解决商业大模型能耗高、隐私风险大的问题，通过构建本地RAG模型实现更高效准确的医疗问答。**

- **链接: [http://arxiv.org/pdf/2506.20009v1](http://arxiv.org/pdf/2506.20009v1)**

> **作者:** Konstantinos Vrettos; Michail E. Klontzas
>
> **备注:** 18 pages, 3 Figures
>
> **摘要:** Background The increasing adoption of Artificial Intelligence (AI) in healthcare has sparked growing concerns about its environmental and ethical implications. Commercial Large Language Models (LLMs), such as ChatGPT and DeepSeek, require substantial resources, while the utilization of these systems for medical purposes raises critical issues regarding patient privacy and safety. Methods We developed a customizable Retrieval-Augmented Generation (RAG) framework for medical tasks, which monitors its energy usage and CO2 emissions. This system was then used to create RAGs based on various open-source LLMs. The tested models included both general purpose models like llama3.1:8b and medgemma-4b-it, which is medical-domain specific. The best RAGs performance and energy consumption was compared to DeepSeekV3-R1 and OpenAIs o4-mini model. A dataset of medical questions was used for the evaluation. Results Custom RAG models outperformed commercial models in accuracy and energy consumption. The RAG model built on llama3.1:8B achieved the highest accuracy (58.5%) and was significantly better than other models, including o4-mini and DeepSeekV3-R1. The llama3.1-RAG also exhibited the lowest energy consumption and CO2 footprint among all models, with a Performance per kWh of 0.52 and a total CO2 emission of 473g. Compared to o4-mini, the llama3.1-RAG achieved 2.7x times more accuracy points per kWh and 172% less electricity usage while maintaining higher accuracy. Conclusion Our study demonstrates that local LLMs can be leveraged to develop RAGs that outperform commercial, online LLMs in medical tasks, while having a smaller environmental impact. Our modular framework promotes sustainable AI development, reducing electricity usage and aligning with the UNs Sustainable Development Goals.
>
---
#### [new 047] Capturing Visualization Design Rationale
- **分类: cs.HC; cs.CL**

- **简介: 该论文属于可视化设计解释任务，旨在解决如何通过自然语言理解可视化设计决策的问题。研究构建了包含设计理由的数据集，并利用大模型生成和验证问答对。**

- **链接: [http://arxiv.org/pdf/2506.16571v1](http://arxiv.org/pdf/2506.16571v1)**

> **作者:** Maeve Hutchinson; Radu Jianu; Aidan Slingsby; Jo Wood; Pranava Madhyastha
>
> **摘要:** Prior natural language datasets for data visualization have focused on tasks such as visualization literacy assessment, insight generation, and visualization generation from natural language instructions. These studies often rely on controlled setups with purpose-built visualizations and artificially constructed questions. As a result, they tend to prioritize the interpretation of visualizations, focusing on decoding visualizations rather than understanding their encoding. In this paper, we present a new dataset and methodology for probing visualization design rationale through natural language. We leverage a unique source of real-world visualizations and natural language narratives: literate visualization notebooks created by students as part of a data visualization course. These notebooks combine visual artifacts with design exposition, in which students make explicit the rationale behind their design decisions. We also use large language models (LLMs) to generate and categorize question-answer-rationale triples from the narratives and articulations in the notebooks. We then carefully validate the triples and curate a dataset that captures and distills the visualization design choices and corresponding rationales of the students.
>
---
#### [new 048] PLoP: Precise LoRA Placement for Efficient Finetuning of Large Models
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文属于模型微调任务，旨在解决LoRA适配器的放置策略问题。通过提出PLoP方法，自动确定最佳适配模块类型，提升微调效果。**

- **链接: [http://arxiv.org/pdf/2506.20629v1](http://arxiv.org/pdf/2506.20629v1)**

> **作者:** Soufiane Hayou; Nikhil Ghosh; Bin Yu
>
> **备注:** TD,LR: A lightweight module type selection method for LoRA finetuning. PLoP gives precise placements for LoRA adapters for improved performance
>
> **摘要:** Low-Rank Adaptation (LoRA) is a widely used finetuning method for large models. Its small memory footprint allows practitioners to adapt large models to specific tasks at a fraction of the cost of full finetuning. Different modifications have been proposed to enhance its efficiency by, for example, setting the learning rate, the rank, and the initialization. Another improvement axis is adapter placement strategy: when using LoRA, practitioners usually pick module types to adapt with LoRA, such as Query and Key modules. Few works have studied the problem of adapter placement, with nonconclusive results: original LoRA paper suggested placing adapters in attention modules, while other works suggested placing them in the MLP modules. Through an intuitive theoretical analysis, we introduce PLoP (Precise LoRA Placement), a lightweight method that allows automatic identification of module types where LoRA adapters should be placed, given a pretrained model and a finetuning task. We demonstrate that PLoP consistently outperforms, and in the worst case competes, with commonly used placement strategies through comprehensive experiments on supervised finetuning and reinforcement learning for reasoning.
>
---
#### [new 049] Language Modeling by Language Models
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于语言模型架构搜索任务，旨在利用LLMs自动发现新模型架构。通过多智能体方法和遗传编程，提升设计效率与效果。**

- **链接: [http://arxiv.org/pdf/2506.20249v1](http://arxiv.org/pdf/2506.20249v1)**

> **作者:** Junyan Cheng; Peter Clark; Kyle Richardson
>
> **摘要:** Can we leverage LLMs to model the process of discovering novel language model (LM) architectures? Inspired by real research, we propose a multi-agent LLM approach that simulates the conventional stages of research, from ideation and literature search (proposal stage) to design implementation (code generation), generative pre-training, and downstream evaluation (verification). Using ideas from scaling laws, our system, Genesys, employs a Ladder of Scales approach; new designs are proposed, adversarially reviewed, implemented, and selectively verified at increasingly larger model scales (14M$\sim$350M parameters) with a narrowing budget (the number of models we can train at each scale). To help make discovery efficient and factorizable, Genesys uses a novel genetic programming backbone, which we show has empirical advantages over commonly used direct prompt generation workflows (e.g., $\sim$86\% percentage point improvement in successful design generation, a key bottleneck). We report experiments involving 1,162 newly discovered designs (1,062 fully verified through pre-training) and find the best designs to be highly competitive with known architectures (e.g., outperform GPT2, Mamba2, etc., on 6/9 common benchmarks). We couple these results with comprehensive system-level ablations and formal results, which give broader insights into the design of effective autonomous discovery systems.
>
---
## 更新

#### [replaced 001] Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.08745v4](http://arxiv.org/pdf/2411.08745v4)**

> **作者:** Clément Dumas; Chris Wendler; Veniamin Veselovsky; Giovanni Monea; Robert West
>
> **备注:** 20 pages, 14 figures, previous version published under the title "How Do Llamas Process Multilingual Text? A Latent Exploration through Activation Patching" at the ICML 2024 mechanistic interpretability workshop at https://openreview.net/forum?id=0ku2hIm4BS
>
> **摘要:** A central question in multilingual language modeling is whether large language models (LLMs) develop a universal concept representation, disentangled from specific languages. In this paper, we address this question by analyzing latent representations (latents) during a word-translation task in transformer-based LLMs. We strategically extract latents from a source translation prompt and insert them into the forward pass on a target translation prompt. By doing so, we find that the output language is encoded in the latent at an earlier layer than the concept to be translated. Building on this insight, we conduct two key experiments. First, we demonstrate that we can change the concept without changing the language and vice versa through activation patching alone. Second, we show that patching with the mean representation of a concept across different languages does not affect the models' ability to translate it, but instead improves it. Finally, we generalize to multi-token generation and demonstrate that the model can generate natural language description of those mean representations. Our results provide evidence for the existence of language-agnostic concept representations within the investigated models.
>
---
#### [replaced 002] VICCA: Visual Interpretation and Comprehension of Chest X-ray Anomalies in Generated Report Without Human Feedback
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.17726v2](http://arxiv.org/pdf/2501.17726v2)**

> **作者:** Sayeh Gholipour Picha; Dawood Al Chanti; Alice Caplier
>
> **摘要:** As artificial intelligence (AI) becomes increasingly central to healthcare, the demand for explainable and trustworthy models is paramount. Current report generation systems for chest X-rays (CXR) often lack mechanisms for validating outputs without expert oversight, raising concerns about reliability and interpretability. To address these challenges, we propose a novel multimodal framework designed to enhance the semantic alignment and localization accuracy of AI-generated medical reports. Our framework integrates two key modules: a Phrase Grounding Model, which identifies and localizes pathologies in CXR images based on textual prompts, and a Text-to-Image Diffusion Module, which generates synthetic CXR images from prompts while preserving anatomical fidelity. By comparing features between the original and generated images, we introduce a dual-scoring system: one score quantifies localization accuracy, while the other evaluates semantic consistency. This approach significantly outperforms existing methods, achieving state-of-the-art results in pathology localization and text-to-image alignment. The integration of phrase grounding with diffusion models, coupled with the dual-scoring evaluation system, provides a robust mechanism for validating report quality, paving the way for more trustworthy and transparent AI in medical imaging.
>
---
#### [replaced 003] Towards Fully Exploiting LLM Internal States to Enhance Knowledge Boundary Perception
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11677v2](http://arxiv.org/pdf/2502.11677v2)**

> **作者:** Shiyu Ni; Keping Bi; Jiafeng Guo; Lulu Yu; Baolong Bi; Xueqi Cheng
>
> **备注:** ACL2025 Main
>
> **摘要:** Large language models (LLMs) exhibit impressive performance across diverse tasks but often struggle to accurately gauge their knowledge boundaries, leading to confident yet incorrect responses. This paper explores leveraging LLMs' internal states to enhance their perception of knowledge boundaries from efficiency and risk perspectives. We investigate whether LLMs can estimate their confidence using internal states before response generation, potentially saving computational resources. Our experiments on datasets like Natural Questions, HotpotQA, and MMLU reveal that LLMs demonstrate significant pre-generation perception, which is further refined post-generation, with perception gaps remaining stable across varying conditions. To mitigate risks in critical domains, we introduce Confidence Consistency-based Calibration ($C^3$), which assesses confidence consistency through question reformulation. $C^3$ significantly improves LLMs' ability to recognize their knowledge gaps, enhancing the unknown perception rate by 5.6% on NQ and 4.9% on HotpotQA. Our findings suggest that pre-generation confidence estimation can optimize efficiency, while $C^3$ effectively controls output risks, advancing the reliability of LLMs in practical applications.
>
---
#### [replaced 004] Recycling the Web: A Method to Enhance Pre-training Data Quality and Quantity for Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.04689v2](http://arxiv.org/pdf/2506.04689v2)**

> **作者:** Thao Nguyen; Yang Li; Olga Golovneva; Luke Zettlemoyer; Sewoong Oh; Ludwig Schmidt; Xian Li
>
> **摘要:** Scaling laws predict that the performance of large language models improves with increasing model size and data size. In practice, pre-training has been relying on massive web crawls, using almost all data sources publicly available on the internet so far. However, this pool of natural data does not grow at the same rate as the compute supply. Furthermore, the availability of high-quality texts is even more limited: data filtering pipelines often remove up to 99% of the initial web scrapes to achieve state-of-the-art. To address the "data wall" of pre-training scaling, our work explores ways to transform and recycle data discarded in existing filtering processes. We propose REWIRE, REcycling the Web with guIded REwrite, a method to enrich low-quality documents so that they could become useful for training. This in turn allows us to increase the representation of synthetic data in the final pre-training set. Experiments at 1B, 3B and 7B scales of the DCLM benchmark show that mixing high-quality raw texts and our rewritten texts lead to 1.0, 1.3 and 2.5 percentage points improvement respectively across 22 diverse tasks, compared to training on only filtered web data. Training on the raw-synthetic data mix is also more effective than having access to 2x web data. Through further analysis, we demonstrate that about 82% of the mixed in texts come from transforming lower-quality documents that would otherwise be discarded. REWIRE also outperforms related approaches of generating synthetic data, including Wikipedia-style paraphrasing, question-answer synthesizing and knowledge extraction. These results suggest that recycling web texts holds the potential for being a simple and effective approach for scaling pre-training data.
>
---
#### [replaced 005] Scientists' First Exam: Probing Cognitive Abilities of MLLM via Perception, Understanding, and Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10521v3](http://arxiv.org/pdf/2506.10521v3)**

> **作者:** Yuhao Zhou; Yiheng Wang; Xuming He; Ruoyao Xiao; Zhiwei Li; Qiantai Feng; Zijie Guo; Yuejin Yang; Hao Wu; Wenxuan Huang; Jiaqi Wei; Dan Si; Xiuqi Yao; Jia Bu; Haiwen Huang; Tianfan Fu; Shixiang Tang; Ben Fei; Dongzhan Zhou; Fenghua Ling; Yan Lu; Siqi Sun; Chenhui Li; Guanjie Zheng; Jiancheng Lv; Wenlong Zhang; Lei Bai
>
> **备注:** 82 pages
>
> **摘要:** Scientific discoveries increasingly rely on complex multimodal reasoning based on information-intensive scientific data and domain-specific expertise. Empowered by expert-level scientific benchmarks, scientific Multimodal Large Language Models (MLLMs) hold the potential to significantly enhance this discovery process in realistic workflows. However, current scientific benchmarks mostly focus on evaluating the knowledge understanding capabilities of MLLMs, leading to an inadequate assessment of their perception and reasoning abilities. To address this gap, we present the Scientists' First Exam (SFE) benchmark, designed to evaluate the scientific cognitive capacities of MLLMs through three interconnected levels: scientific signal perception, scientific attribute understanding, scientific comparative reasoning. Specifically, SFE comprises 830 expert-verified VQA pairs across three question types, spanning 66 multimodal tasks across five high-value disciplines. Extensive experiments reveal that current state-of-the-art GPT-o3 and InternVL-3 achieve only 34.08% and 26.52% on SFE, highlighting significant room for MLLMs to improve in scientific realms. We hope the insights obtained in SFE will facilitate further developments in AI-enhanced scientific discoveries.
>
---
#### [replaced 006] Aug2Search: Enhancing Facebook Marketplace Search with LLM-Generated Synthetic Data Augmentation
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16065v3](http://arxiv.org/pdf/2505.16065v3)**

> **作者:** Ruijie Xi; He Ba; Hao Yuan; Rishu Agrawal; Yuxin Tian; Ruoyan Kong; Arul Prakash
>
> **摘要:** Embedding-Based Retrieval (EBR) is an important technique in modern search engines, enabling semantic match between search queries and relevant results. However, search logging data on platforms like Facebook Marketplace lacks the diversity and details needed for effective EBR model training, limiting the models' ability to capture nuanced search patterns. To address this challenge, we propose Aug2Search, an EBR-based framework leveraging synthetic data generated by Generative AI (GenAI) models, in a multimodal and multitask approach to optimize query-product relevance. This paper investigates the capabilities of GenAI, particularly Large Language Models (LLMs), in generating high-quality synthetic data, and analyzing its impact on enhancing EBR models. We conducted experiments using eight Llama models and 100 million data points from Facebook Marketplace logs. Our synthetic data generation follows three strategies: (1) generate queries, (2) enhance product listings, and (3) generate queries from enhanced listings. We train EBR models on three different datasets: sampled engagement data or original data ((e.g., "Click" and "Listing Interactions")), synthetic data, and a mixture of both engagement and synthetic data to assess their performance across various training sets. Our findings underscore the robustness of Llama models in producing synthetic queries and listings with high coherence, relevance, and diversity, while maintaining low levels of hallucination. Aug2Search achieves an improvement of up to 4% in ROC_AUC with 100 million synthetic data samples, demonstrating the effectiveness of our approach. Moreover, our experiments reveal that with the same volume of training data, models trained exclusively on synthetic data often outperform those trained on original data only or a mixture of original and synthetic data.
>
---
#### [replaced 007] On the Role of Context in Reading Time Prediction
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.08160v4](http://arxiv.org/pdf/2409.08160v4)**

> **作者:** Andreas Opedal; Eleanor Chodroff; Ryan Cotterell; Ethan Gotlieb Wilcox
>
> **备注:** EMNLP 2024; preprocessing was corrected to exclude variance due to word skipping and the conclusions remain unchanged
>
> **摘要:** We present a new perspective on how readers integrate context during real-time language comprehension. Our proposals build on surprisal theory, which posits that the processing effort of a linguistic unit (e.g., a word) is an affine function of its in-context information content. We first observe that surprisal is only one out of many potential ways that a contextual predictor can be derived from a language model. Another one is the pointwise mutual information (PMI) between a unit and its context, which turns out to yield the same predictive power as surprisal when controlling for unigram frequency. Moreover, both PMI and surprisal are correlated with frequency. This means that neither PMI nor surprisal contains information about context alone. In response to this, we propose a technique where we project surprisal onto the orthogonal complement of frequency, yielding a new contextual predictor that is uncorrelated with frequency. Our experiments show that the proportion of variance in reading times explained by context is a lot smaller when context is represented by the orthogonalized predictor. From an interpretability standpoint, this indicates that previous studies may have overstated the role that context has in predicting reading times.
>
---
#### [replaced 008] Rewarding Graph Reasoning Process makes LLMs more Generalized Reasoners
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.00845v2](http://arxiv.org/pdf/2503.00845v2)**

> **作者:** Miao Peng; Nuo Chen; Zongrui Suo; Jia Li
>
> **备注:** Accepted to KDD 2025 Research Track
>
> **摘要:** Despite significant advancements in Large Language Models (LLMs), developing advanced reasoning capabilities in LLMs remains a key challenge. Process Reward Models (PRMs) have demonstrated exceptional promise in enhancing reasoning by providing step-wise feedback, particularly in the context of mathematical reasoning. However, their application to broader reasoning domains remains understudied, largely due to the high costs associated with manually creating step-level supervision. In this work, we explore the potential of PRMs in graph reasoning problems - a domain that demands sophisticated multi-step reasoning and offers opportunities for automated step-level data generation using established graph algorithms. We introduce GraphSILO, the largest dataset for graph reasoning problems with fine-grained step-wise labels, built using automated Task-oriented Trajectories and Monte Carlo Tree Search (MCTS) to generate detailed reasoning steps with step-wise labels. Building upon this dataset, we train GraphPRM, the first PRM designed for graph reasoning problems, and evaluate its effectiveness in two key settings: inference-time scaling and reinforcement learning via Direct Preference Optimization (DPO). Experimental results show that GraphPRM significantly improves LLM performance across 13 graph reasoning tasks, delivering a 9% gain for Qwen2.5-7B and demonstrating transferability to new graph reasoning datasets and new reasoning domains like mathematical problem-solving. Notably, GraphPRM enhances LLM performance on GSM8K and Math500, underscoring the cross-domain applicability of graph-based reasoning rewards. Our findings highlight the potential of PRMs in advancing reasoning across diverse domains, paving the way for more versatile and effective LLMs.
>
---
#### [replaced 009] WAFFLE: Finetuning Multi-Modal Model for Automated Front-End Development
- **分类: cs.SE; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.18362v2](http://arxiv.org/pdf/2410.18362v2)**

> **作者:** Shanchao Liang; Nan Jiang; Shangshu Qian; Lin Tan
>
> **摘要:** Web development involves turning UI designs into functional webpages, which can be difficult for both beginners and experienced developers due to the complexity of HTML's hierarchical structures and styles. While Large Language Models (LLMs) have shown promise in generating source code, two major challenges persist in UI-to-HTML code generation: (1) effectively representing HTML's hierarchical structure for LLMs, and (2) bridging the gap between the visual nature of UI designs and the text-based format of HTML code. To tackle these challenges, we introduce Waffle, a new fine-tuning strategy that uses a structure-aware attention mechanism to improve LLMs' understanding of HTML's structure and a contrastive fine-tuning approach to align LLMs' understanding of UI images and HTML code. Models fine-tuned with Waffle show up to 9.00 pp (percentage point) higher HTML match, 0.0982 higher CW-SSIM, 32.99 higher CLIP, and 27.12 pp higher LLEM on our new benchmark WebSight-Test and an existing benchmark Design2Code, outperforming current fine-tuning methods.
>
---
#### [replaced 010] A Comprehensive Evaluation of Semantic Relation Knowledge of Pretrained Language Models and Humans
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.01131v2](http://arxiv.org/pdf/2412.01131v2)**

> **作者:** Zhihan Cao; Hiroaki Yamada; Simone Teufel; Takenobu Tokunaga
>
> **备注:** Accpeted by Language Resources and Evaluation
>
> **摘要:** Recently, much work has concerned itself with the enigma of what exactly PLMs (pretrained language models) learn about different aspects of language, and how they learn it. One stream of this type of research investigates the knowledge that PLMs have about semantic relations. However, many aspects of semantic relations were left unexplored. Only one relation was considered, namely hypernymy. Furthermore, previous work did not measure humans' performance on the same task as that solved by the PLMs. This means that at this point in time, there is only an incomplete view of models' semantic relation knowledge. To address this gap, we introduce a comprehensive evaluation framework covering five relations beyond hypernymy, namely hyponymy, holonymy, meronymy, antonymy, and synonymy. We use six metrics (two newly introduced here) for recently untreated aspects of semantic relation knowledge, namely soundness, completeness, symmetry, asymmetry, prototypicality, and distinguishability and fairly compare humans and models on the same task. Our extensive experiments involve 16 PLMs, eight masked and eight causal language models. Up to now only masked language models had been tested although causal and masked language models treat context differently. Our results reveal a significant knowledge gap between humans and models for almost all semantic relations. Antonymy is the outlier relation where all models perform reasonably well. In general, masked language models perform significantly better than causal language models. Nonetheless, both masked and causal language models are likely to confuse non-antonymy relations with antonymy.
>
---
#### [replaced 011] VAQUUM: Are Vague Quantifiers Grounded in Visual Data?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11874v3](http://arxiv.org/pdf/2502.11874v3)**

> **作者:** Hugh Mee Wong; Rick Nouwen; Albert Gatt
>
> **备注:** Proceedings of ACL 2025, 10 pages
>
> **摘要:** Vague quantifiers such as "a few" and "many" are influenced by various contextual factors, including the number of objects present in a given context. In this work, we evaluate the extent to which vision-and-language models (VLMs) are compatible with humans when producing or judging the appropriateness of vague quantifiers in visual contexts. We release a novel dataset, VAQUUM, containing 20,300 human ratings on quantified statements across a total of 1089 images. Using this dataset, we compare human judgments and VLM predictions using three different evaluation methods. Our findings show that VLMs, like humans, are influenced by object counts in vague quantifier use. However, we find significant inconsistencies across models in different evaluation settings, suggesting that judging and producing vague quantifiers rely on two different processes.
>
---
#### [replaced 012] Conversational User-AI Intervention: A Study on Prompt Rewriting for Improved LLM Response Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.16789v2](http://arxiv.org/pdf/2503.16789v2)**

> **作者:** Rupak Sarkar; Bahareh Sarrafzadeh; Nirupama Chandrasekaran; Nagu Rangan; Philip Resnik; Longqi Yang; Sujay Kumar Jauhar
>
> **备注:** 8 pages, ACL style
>
> **摘要:** Human-LLM conversations are increasingly becoming more pervasive in peoples' professional and personal lives, yet many users still struggle to elicit helpful responses from LLM Chatbots. One of the reasons for this issue is users' lack of understanding in crafting effective prompts that accurately convey their information needs. Meanwhile, the existence of real-world conversational datasets on the one hand, and the text understanding faculties of LLMs on the other, present a unique opportunity to study this problem, and its potential solutions at scale. Thus, in this paper we present the first LLM-centric study of real human-AI chatbot conversations, focused on investigating aspects in which user queries fall short of expressing information needs, and the potential of using LLMs to rewrite suboptimal user prompts. Our findings demonstrate that rephrasing ineffective prompts can elicit better responses from a conversational system, while preserving the user's original intent. Notably, the performance of rewrites improves in longer conversations, where contextual inferences about user needs can be made more accurately. Additionally, we observe that LLMs often need to -- and inherently do -- make \emph{plausible} assumptions about a user's intentions and goals when interpreting prompts. Our findings largely hold true across conversational domains, user intents, and LLMs of varying sizes and families, indicating the promise of using prompt rewriting as a solution for better human-AI interactions.
>
---
#### [replaced 013] Therapy as an NLP Task: Psychologists' Comparison of LLMs and Human Peers in CBT
- **分类: cs.HC; cs.CL; I.2.7; J.4**

- **链接: [http://arxiv.org/pdf/2409.02244v2](http://arxiv.org/pdf/2409.02244v2)**

> **作者:** Zainab Iftikhar; Sean Ransom; Amy Xiao; Nicole Nugent; Jeff Huang
>
> **摘要:** Large language models (LLMs) are being used as ad-hoc therapists. Research suggests that LLMs outperform human counselors when generating a single, isolated empathetic response; however, their session-level behavior remains understudied. In this study, we compare the session-level behaviors of human counselors with those of an LLM prompted by a team of peer counselors to deliver single-session Cognitive Behavioral Therapy (CBT). Our three-stage, mixed-methods study involved: a) a year-long ethnography of a text-based support platform where seven counselors iteratively refined CBT prompts through self-counseling and weekly focus groups; b) the manual simulation of human counselor sessions with a CBT-prompted LLM, given the full patient dialogue and contextual notes; and c) session evaluations of both human and LLM sessions by three licensed clinical psychologists using CBT competence measures. Our results show a clear trade-off. Human counselors excel at relational strategies -- small talk, self-disclosure, and culturally situated language -- that lead to higher empathy, collaboration, and deeper user reflection. LLM counselors demonstrate higher procedural adherence to CBT techniques but struggle to sustain collaboration, misread cultural cues, and sometimes produce "deceptive empathy," i.e., formulaic warmth that can inflate users' expectations of genuine human care. Taken together, our findings imply that while LLMs might outperform counselors in generating single empathetic responses, their ability to lead sessions is more limited, highlighting that therapy cannot be reduced to a standalone natural language processing (NLP) task. We call for carefully designed human-AI workflows in scalable support: LLMs can scaffold evidence-based techniques, while peers provide relational support. We conclude by mapping concrete design opportunities and ethical guardrails for such hybrid systems.
>
---
#### [replaced 014] Attention Entropy is a Key Factor: An Analysis of Parallel Context Encoding with Full-attention-based Pre-trained Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.16545v2](http://arxiv.org/pdf/2412.16545v2)**

> **作者:** Zhisong Zhang; Yan Wang; Xinting Huang; Tianqing Fang; Hongming Zhang; Chenlong Deng; Shuaiyi Li; Dong Yu
>
> **备注:** ACL 2025
>
> **摘要:** Large language models have shown remarkable performance across a wide range of language tasks, owing to their exceptional capabilities in context modeling. The most commonly used method of context modeling is full self-attention, as seen in standard decoder-only Transformers. Although powerful, this method can be inefficient for long sequences and may overlook inherent input structures. To address these problems, an alternative approach is parallel context encoding, which splits the context into sub-pieces and encodes them parallelly. Because parallel patterns are not encountered during training, naively applying parallel encoding leads to performance degradation. However, the underlying reasons and potential mitigations are unclear. In this work, we provide a detailed analysis of this issue and identify that unusually high attention entropy can be a key factor. Furthermore, we adopt two straightforward methods to reduce attention entropy by incorporating attention sinks and selective mechanisms. Experiments on various tasks reveal that these methods effectively lower irregular attention entropy and narrow performance gaps. We hope this study can illuminate ways to enhance context modeling mechanisms.
>
---
#### [replaced 015] LADM: Long-context Training Data Selection with Attention-based Dependency Measurement for LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.02502v2](http://arxiv.org/pdf/2503.02502v2)**

> **作者:** Jianghao Chen; Junhong Wu; Yangyifan Xu; Jiajun Zhang
>
> **备注:** ACL 2025, our code is available at https://github.com/ZNLP/LADM
>
> **摘要:** Long-context modeling has drawn more and more attention in the area of Large Language Models (LLMs). Continual training with long-context data becomes the de-facto method to equip LLMs with the ability to process long inputs. However, it still remains an open challenge to measure the quality of long-context training data. To address this issue, we propose a Long-context data selection framework with Attention-based Dependency Measurement (LADM), which can efficiently identify high-quality long-context data from a large-scale, multi-domain pre-training corpus. LADM leverages the retrieval capabilities of the attention mechanism to capture contextual dependencies, ensuring a comprehensive quality measurement of long-context data. Experimental results show that our LADM framework significantly boosts the performance of LLMs on multiple long-context tasks with only 1B tokens for continual training.
>
---
#### [replaced 016] Thought Anchors: Which LLM Reasoning Steps Matter?
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.19143v2](http://arxiv.org/pdf/2506.19143v2)**

> **作者:** Paul C. Bogdan; Uzay Macar; Neel Nanda; Arthur Conmy
>
> **备注:** Paul C. Bogdan and Uzay Macar contributed equally to this work, and their listed order was determined by coinflip. Neel Nanda and Arthur Conmy contributed equally to this work as senior authors, and their listed order was determined by coinflip
>
> **摘要:** Reasoning large language models have recently achieved state-of-the-art performance in many fields. However, their long-form chain-of-thought reasoning creates interpretability challenges as each generated token depends on all previous ones, making the computation harder to decompose. We argue that analyzing reasoning traces at the sentence level is a promising approach to understanding reasoning processes. We present three complementary attribution methods: (1) a black-box method measuring each sentence's counterfactual importance by comparing final answers across 100 rollouts conditioned on the model generating that sentence or one with a different meaning; (2) a white-box method of aggregating attention patterns between pairs of sentences, which identified "broadcasting" sentences that receive disproportionate attention from all future sentences via "receiver" attention heads; (3) a causal attribution method measuring logical connections between sentences by suppressing attention toward one sentence and measuring the effect on each future sentence's tokens. Each method provides evidence for the existence of thought anchors, reasoning steps that have outsized importance and that disproportionately influence the subsequent reasoning process. These thought anchors are typically planning or backtracking sentences. We provide an open-source tool (www.thought-anchors.com) for visualizing the outputs of our methods, and present a case study showing converging patterns across methods that map how a model performs multi-step reasoning. The consistency across methods demonstrates the potential of sentence-level analysis for a deeper understanding of reasoning models.
>
---
#### [replaced 017] Misalignment of Semantic Relation Knowledge between WordNet and Human Intuition
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.02138v2](http://arxiv.org/pdf/2412.02138v2)**

> **作者:** Zhihan Cao; Hiroaki Yamada; Simone Teufel; Takenobu Tokunaga
>
> **备注:** Accepted by Global WordNet Conference 2025
>
> **摘要:** WordNet provides a carefully constructed repository of semantic relations, created by specialists. But there is another source of information on semantic relations, the intuition of language users. We present the first systematic study of the degree to which these two sources are aligned. Investigating the cases of misalignment could make proper use of WordNet and facilitate its improvement. Our analysis which uses templates to elicit responses from human participants, reveals a general misalignment of semantic relation knowledge between WordNet and human intuition. Further analyses find a systematic pattern of mismatch among synonymy and taxonomic relations~(hypernymy and hyponymy), together with the fact that WordNet path length does not serve as a reliable indicator of human intuition regarding hypernymy or hyponymy relations.
>
---
#### [replaced 018] LLaVA-CMoE: Towards Continual Mixture of Experts for Large Vision-Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.21227v3](http://arxiv.org/pdf/2503.21227v3)**

> **作者:** Hengyuan Zhao; Ziqin Wang; Qixin Sun; Kaiyou Song; Yilin Li; Xiaolin Hu; Qingpei Guo; Si Liu
>
> **备注:** Preprint
>
> **摘要:** Mixture of Experts (MoE) architectures have recently advanced the scalability and adaptability of large language models (LLMs) for continual multimodal learning. However, efficiently extending these models to accommodate sequential tasks remains challenging. As new tasks arrive, naive model expansion leads to rapid parameter growth, while modifying shared routing components often causes catastrophic forgetting, undermining previously learned knowledge. To address these issues, we propose LLaVA-CMoE, a continual learning framework for LLMs that requires no replay data of previous tasks and ensures both parameter efficiency and robust knowledge retention. Our approach introduces a Probe-Guided Knowledge Extension mechanism, which uses probe experts to dynamically determine when and where new experts should be added, enabling adaptive and minimal parameter expansion tailored to task complexity. Furthermore, we present a Probabilistic Task Locator that assigns each task a dedicated, lightweight router. To handle the practical issue that task labels are unknown during inference, we leverage a VAE-based reconstruction strategy to identify the most suitable router by matching input distributions, allowing automatic and accurate expert allocation. This design mitigates routing conflicts and catastrophic forgetting, enabling robust continual learning without explicit task labels. Extensive experiments on the CoIN benchmark, covering eight diverse VQA tasks, demonstrate that LLaVA-CMoE delivers strong continual learning performance with a compact model size, significantly reducing forgetting and parameter overhead compared to prior methods. These results showcase the effectiveness and scalability of our approach for parameter-efficient continual learning in large language models. Our code will be open-sourced soon.
>
---
#### [replaced 019] A Global Context Mechanism for Sequence Labeling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2305.19928v5](http://arxiv.org/pdf/2305.19928v5)**

> **作者:** Conglei Xu; Kun Shen; Hongguang Sun; Yang Xu
>
> **摘要:** Global sentence information is crucial for sequence labeling tasks, where each word in a sentence must be assigned a label. While BiLSTM models are widely used, they often fail to capture sufficient global context for inner words. Previous work has proposed various RNN variants to integrate global sentence information into word representations. However, these approaches suffer from three key limitations: (1) they are slower in both inference and training compared to the original BiLSTM, (2) they cannot effectively supplement global information for transformer-based models, and (3) the high time cost associated with reimplementing and integrating these customized RNNs into existing architectures. In this study, we introduce a simple yet effective mechanism that addresses these limitations. Our approach efficiently supplements global sentence information for both BiLSTM and transformer-based models, with minimal degradation in inference and training speed, and is easily pluggable into current architectures. We demonstrate significant improvements in F1 scores across seven popular benchmarks, including Named Entity Recognition (NER) tasks such as Conll2003, Wnut2017 , and the Chinese named-entity recognition task Weibo, as well as End-to-End Aspect-Based Sentiment Analysis (E2E-ABSA) benchmarks such as Laptop14, Restaurant14, Restaurant15, and Restaurant16. With out any extra strategy, we achieve third highest score on weibo NER benchmark. Compared to CRF, one of the most popular frameworks for sequence labeling, our mechanism achieves competitive F1 scores while offering superior inference and training speed. Code is available at: https://github.com/conglei2XU/Global-Context-Mechanism
>
---
#### [replaced 020] Unlocking In-Context Learning for Natural Datasets Beyond Language Modelling
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.06256v2](http://arxiv.org/pdf/2501.06256v2)**

> **作者:** Jelena Bratulić; Sudhanshu Mittal; David T. Hoffmann; Samuel Böhm; Robin Tibor Schirrmeister; Tonio Ball; Christian Rupprecht; Thomas Brox
>
> **摘要:** Large Language Models (LLMs) exhibit In-Context Learning (ICL), which enables the model to perform new tasks conditioning only on the examples provided in the context without updating the model's weights. While ICL offers fast adaptation across natural language tasks and domains, its emergence is less straightforward for modalities beyond text. In this work, we systematically uncover properties present in LLMs that support the emergence of ICL for autoregressive models and various modalities by promoting the learning of the needed mechanisms for ICL. We identify exact token repetitions in the training data sequences as an important factor for ICL. Such repetitions further improve stability and reduce transiency in ICL performance. Moreover, we emphasise the significance of training task difficulty for the emergence of ICL. Finally, by applying our novel insights on ICL emergence, we unlock ICL capabilities for various visual datasets and a more challenging EEG classification task in a few-shot learning regime.
>
---
#### [replaced 021] CogniBench: A Legal-inspired Framework and Dataset for Assessing Cognitive Faithfulness of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20767v4](http://arxiv.org/pdf/2505.20767v4)**

> **作者:** Xiaqiang Tang; Jian Li; Keyu Hu; Du Nan; Xiaolong Li; Xi Zhang; Weigao Sun; Sihong Xie
>
> **备注:** ACL 2025
>
> **摘要:** Faithfulness hallucinations are claims generated by a Large Language Model (LLM) not supported by contexts provided to the LLM. Lacking assessment standards, existing benchmarks focus on "factual statements" that rephrase source materials while overlooking "cognitive statements" that involve making inferences from the given context. Consequently, evaluating and detecting the hallucination of cognitive statements remains challenging. Inspired by how evidence is assessed in the legal domain, we design a rigorous framework to assess different levels of faithfulness of cognitive statements and introduce the CogniBench dataset where we reveal insightful statistics. To keep pace with rapidly evolving LLMs, we further develop an automatic annotation pipeline that scales easily across different models. This results in a large-scale CogniBench-L dataset, which facilitates training accurate detectors for both factual and cognitive hallucinations. We release our model and datasets at: https://github.com/FUTUREEEEEE/CogniBench
>
---
#### [replaced 022] Evaluating Long Range Dependency Handling in Code Generation LLMs
- **分类: cs.CL; cs.AI; cs.LG; I.2.7**

- **链接: [http://arxiv.org/pdf/2407.21049v2](http://arxiv.org/pdf/2407.21049v2)**

> **作者:** Yannick Assogba; Donghao Ren
>
> **备注:** 36 pages, 18 figures
>
> **摘要:** As language models support larger and larger context sizes, evaluating their ability to make effective use of that context becomes increasingly important. We analyze the ability of several code generation models to handle long range dependencies using a suite of multi-step key retrieval tasks in context windows up to 8k tokens in length. The tasks progressively increase in difficulty and allow more nuanced evaluation of model capabilities than tests like the popular needle-in-the-haystack test. We find that performance degrades significantly for many models (up to 2x) when a function references another function that is defined later in the prompt. We also observe that models that use sliding window attention mechanisms have difficulty handling references further than the size of a single window. We perform simple prompt modifications using call graph information to improve multi-step retrieval performance up to 3x. Our analysis highlights ways that long-context performance needs deeper consideration beyond retrieval of single facts within a document.
>
---
#### [replaced 023] FluoroSAM: A Language-promptable Foundation Model for Flexible X-ray Image Segmentation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.08059v3](http://arxiv.org/pdf/2403.08059v3)**

> **作者:** Benjamin D. Killeen; Liam J. Wang; Blanca Inigo; Han Zhang; Mehran Armand; Russell H. Taylor; Greg Osgood; Mathias Unberath
>
> **摘要:** Language promptable X-ray image segmentation would enable greater flexibility for human-in-the-loop workflows in diagnostic and interventional precision medicine. Prior efforts have contributed task-specific models capable of solving problems within a narrow scope, but expanding to broader use requires additional data, annotations, and training time. Recently, language-aligned foundation models (LFMs) -- machine learning models trained on large amounts of highly variable image and text data thus enabling broad applicability -- have emerged as promising tools for automated image analysis. Existing foundation models for medical image analysis focus on scenarios and modalities where large, richly annotated datasets are available. However, the X-ray imaging modality features highly variable image appearance and applications, from diagnostic chest X-rays to interventional fluoroscopy, with varying availability of data. To pave the way toward an LFM for comprehensive and language-aligned analysis of arbitrary medical X-ray images, we introduce FluoroSAM, a language-promptable variant of the Segment Anything Model, trained from scratch on 3M synthetic X-ray images from a wide variety of human anatomies, imaging geometries, and viewing angles. These include pseudo-ground truth masks for 128 organ types and 464 tools with associated text descriptions. FluoroSAM is capable of segmenting myriad anatomical structures and tools based on natural language prompts, thanks to the novel incorporation of vector quantization (VQ) of text embeddings in the training process. We demonstrate FluoroSAM's performance quantitatively on real X-ray images and showcase on several applications how FluoroSAM is a key enabler for rich human-machine interaction in the X-ray image acquisition and analysis context. Code is available at https://github.com/arcadelab/fluorosam.
>
---
#### [replaced 024] Understanding World or Predicting Future? A Comprehensive Survey of World Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.14499v2](http://arxiv.org/pdf/2411.14499v2)**

> **作者:** Jingtao Ding; Yunke Zhang; Yu Shang; Yuheng Zhang; Zefang Zong; Jie Feng; Yuan Yuan; Hongyuan Su; Nian Li; Nicholas Sukiennik; Fengli Xu; Yong Li
>
> **备注:** Accepted by ACM CSUR, 37 pages, 7 figures, 7 tables
>
> **摘要:** The concept of world models has garnered significant attention due to advancements in multimodal large language models such as GPT-4 and video generation models such as Sora, which are central to the pursuit of artificial general intelligence. This survey offers a comprehensive review of the literature on world models. Generally, world models are regarded as tools for either understanding the present state of the world or predicting its future dynamics. This review presents a systematic categorization of world models, emphasizing two primary functions: (1) constructing internal representations to understand the mechanisms of the world, and (2) predicting future states to simulate and guide decision-making. Initially, we examine the current progress in these two categories. We then explore the application of world models in key domains, including autonomous driving, robotics, and social simulacra, with a focus on how each domain utilizes these aspects. Finally, we outline key challenges and provide insights into potential future research directions. We summarize the representative papers along with their code repositories in https://github.com/tsinghua-fib-lab/World-Model.
>
---
#### [replaced 025] Quantifying Fairness in LLMs Beyond Tokens: A Semantic and Statistical Perspective
- **分类: cs.CL; cs.AI; cs.CY; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.19028v2](http://arxiv.org/pdf/2506.19028v2)**

> **作者:** Weijie Xu; Yiwen Wang; Chi Xue; Xiangkun Hu; Xi Fang; Guimin Dong; Chandan K. Reddy
>
> **备注:** 29 pages, 9 figures, 15 tables
>
> **摘要:** Large Language Models (LLMs) often generate responses with inherent biases, undermining their reliability in real-world applications. Existing evaluation methods often overlook biases in long-form responses and the intrinsic variability of LLM outputs. To address these challenges, we propose FiSCo(Fine-grained Semantic Computation), a novel statistical framework to evaluate group-level fairness in LLMs by detecting subtle semantic differences in long-form responses across demographic groups. Unlike prior work focusing on sentiment or token-level comparisons, FiSCo goes beyond surface-level analysis by operating at the claim level, leveraging entailment checks to assess the consistency of meaning across responses. We decompose model outputs into semantically distinct claims and apply statistical hypothesis testing to compare inter- and intra-group similarities, enabling robust detection of subtle biases. We formalize a new group counterfactual fairness definition and validate FiSCo on both synthetic and human-annotated datasets spanning gender, race, and age. Experiments show that FiSco more reliably identifies nuanced biases while reducing the impact of stochastic LLM variability, outperforming various evaluation metrics.
>
---
#### [replaced 026] Ad-hoc Concept Forming in the Game Codenames as a Means for Evaluating Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11707v2](http://arxiv.org/pdf/2502.11707v2)**

> **作者:** Sherzod Hakimov; Lara Pfennigschmidt; David Schlangen
>
> **备注:** Accepted at GemBench workshop co-located with ACL 2025
>
> **摘要:** This study utilizes the game Codenames as a benchmarking tool to evaluate large language models (LLMs) with respect to specific linguistic and cognitive skills. LLMs play each side of the game, where one side generates a clue word covering several target words and the other guesses those target words. We designed various experiments by controlling the choice of words (abstract vs. concrete words, ambiguous vs. monosemic) or the opponent (programmed to be faster or slower in revealing words). Recent commercial and open-weight models were compared side-by-side to find out factors affecting their performance. The evaluation reveals details about their strategies, challenging cases, and limitations of LLMs.
>
---
#### [replaced 027] mSTEB: Massively Multilingual Evaluation of LLMs on Speech and Text Tasks
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.08400v2](http://arxiv.org/pdf/2506.08400v2)**

> **作者:** Luel Hagos Beyene; Vivek Verma; Min Ma; Jesujoba O. Alabi; Fabian David Schmidt; Joyce Nakatumba-Nabende; David Ifeoluwa Adelani
>
> **备注:** working paper
>
> **摘要:** Large Language models (LLMs) have demonstrated impressive performance on a wide range of tasks, including in multimodal settings such as speech. However, their evaluation is often limited to English and a few high-resource languages. For low-resource languages, there is no standardized evaluation benchmark. In this paper, we address this gap by introducing mSTEB, a new benchmark to evaluate the performance of LLMs on a wide range of tasks covering language identification, text classification, question answering, and translation tasks on both speech and text modalities. We evaluated the performance of leading LLMs such as Gemini 2.0 Flash and GPT-4o (Audio) and state-of-the-art open models such as Qwen 2 Audio and Gemma 3 27B. Our evaluation shows a wide gap in performance between high-resource and low-resource languages, especially for languages spoken in Africa and Americas/Oceania. Our findings show that more investment is needed to address their under-representation in LLMs coverage.
>
---
#### [replaced 028] OmniGen2: Exploration to Advanced Multimodal Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.18871v2](http://arxiv.org/pdf/2506.18871v2)**

> **作者:** Chenyuan Wu; Pengfei Zheng; Ruiran Yan; Shitao Xiao; Xin Luo; Yueze Wang; Wanli Li; Xiyan Jiang; Yexin Liu; Junjie Zhou; Ze Liu; Ziyi Xia; Chaofan Li; Haoge Deng; Jiahao Wang; Kun Luo; Bo Zhang; Defu Lian; Xinlong Wang; Zhongyuan Wang; Tiejun Huang; Zheng Liu
>
> **摘要:** In this work, we introduce OmniGen2, a versatile and open-source generative model designed to provide a unified solution for diverse generation tasks, including text-to-image, image editing, and in-context generation. Unlike OmniGen v1, OmniGen2 features two distinct decoding pathways for text and image modalities, utilizing unshared parameters and a decoupled image tokenizer. This design enables OmniGen2 to build upon existing multimodal understanding models without the need to re-adapt VAE inputs, thereby preserving the original text generation capabilities. To facilitate the training of OmniGen2, we developed comprehensive data construction pipelines, encompassing image editing and in-context generation data. Additionally, we introduce a reflection mechanism tailored for image generation tasks and curate a dedicated reflection dataset based on OmniGen2. Despite its relatively modest parameter size, OmniGen2 achieves competitive results on multiple task benchmarks, including text-to-image and image editing. To further evaluate in-context generation, also referred to as subject-driven tasks, we introduce a new benchmark named OmniContext. OmniGen2 achieves state-of-the-art performance among open-source models in terms of consistency. We will release our models, training code, datasets, and data construction pipeline to support future research in this field. Project Page: https://vectorspacelab.github.io/OmniGen2; GitHub Link: https://github.com/VectorSpaceLab/OmniGen2
>
---
#### [replaced 029] Attention with Trained Embeddings Provably Selects Important Tokens
- **分类: cs.LG; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2505.17282v3](http://arxiv.org/pdf/2505.17282v3)**

> **作者:** Diyuan Wu; Aleksandr Shevchenko; Samet Oymak; Marco Mondelli
>
> **备注:** Fix mistakes in Lemma 4.2 and proof of Lemma 4.5, and some other minor changes
>
> **摘要:** Token embeddings play a crucial role in language modeling but, despite this practical relevance, their theoretical understanding remains limited. Our paper addresses the gap by characterizing the structure of embeddings obtained via gradient descent. Specifically, we consider a one-layer softmax attention model with a linear head for binary classification, i.e., $\texttt{Softmax}( p^\top E_X^\top ) E_X v = \frac{ \sum_{i=1}^T \exp(p^\top E_{x_i}) E_{x_i}^\top v}{\sum_{j=1}^T \exp(p^\top E_{x_{j}}) }$, where $E_X = [ E_{x_1} , \dots, E_{x_T} ]^\top$ contains the embeddings of the input sequence, $p$ is the embedding of the $\mathrm{\langle cls \rangle}$ token and $v$ the output vector. First, we show that, already after a single step of gradient training with the logistic loss, the embeddings $E_X$ capture the importance of tokens in the dataset by aligning with the output vector $v$ proportionally to the frequency with which the corresponding tokens appear in the dataset. Then, after training $p$ via gradient flow until convergence, the softmax selects the important tokens in the sentence (i.e., those that are predictive of the label), and the resulting $\mathrm{\langle cls \rangle}$ embedding maximizes the margin for such a selection. Experiments on real-world datasets (IMDB, Yelp) exhibit a phenomenology close to that unveiled by our theory.
>
---
#### [replaced 030] LR^2Bench: Evaluating Long-chain Reflective Reasoning Capabilities of Large Language Models via Constraint Satisfaction Problems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17848v4](http://arxiv.org/pdf/2502.17848v4)**

> **作者:** Jianghao Chen; Zhenlin Wei; Zhenjiang Ren; Ziyong Li; Jiajun Zhang
>
> **备注:** ACL-2025, our code is available at https://github.com/ZNLP/LR2Bench
>
> **摘要:** Recent progress in Large Reasoning Models (LRMs) has significantly enhanced the reasoning abilities of Large Language Models (LLMs), empowering them to tackle increasingly complex tasks through reflection capabilities, such as making assumptions, backtracking, and self-refinement. However, effectively evaluating such reflection capabilities remains challenging due to the lack of appropriate benchmarks. To bridge this gap, we introduce LR$^2$Bench, a novel benchmark designed to evaluate the Long-chain Reflective Reasoning capabilities of LLMs. LR$^2$Bench comprises 850 samples across six Constraint Satisfaction Problems (CSPs) where reflective reasoning is crucial for deriving solutions that meet all given constraints. Each type of task focuses on distinct constraint patterns, such as knowledge-based, logical, and spatial constraints, providing a comprehensive evaluation of diverse problem-solving scenarios. Our extensive evaluation on both conventional LLMs and LRMs reveals that even the most advanced LRMs, such as DeepSeek-R1 and OpenAI o1-preview, struggle with tasks in LR$^2$Bench, achieving an average Exact Match score of only 20.0% and 23.6%, respectively. These findings underscore the significant room for improvement in the reflective reasoning capabilities of current LLMs.
>
---
#### [replaced 031] Balancing Truthfulness and Informativeness with Uncertainty-Aware Instruction Fine-Tuning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11962v3](http://arxiv.org/pdf/2502.11962v3)**

> **作者:** Tianyi Wu; Jingwei Ni; Bryan Hooi; Jiaheng Zhang; Elliott Ash; See-Kiong Ng; Mrinmaya Sachan; Markus Leippold
>
> **摘要:** Instruction fine-tuning (IFT) can increase the informativeness of large language models (LLMs), but may reduce their truthfulness. This trade-off arises because IFT steers LLMs to generate responses containing long-tail knowledge that was not well covered during pre-training. As a result, models become more informative but less accurate when generalizing to unseen tasks. In this paper, we empirically demonstrate how unfamiliar knowledge in IFT datasets can negatively affect the truthfulness of LLMs, and we introduce two new IFT paradigms, $UNIT_{cut}$ and $UNIT_{ref}$, to address this issue. $UNIT_{cut}$ identifies and removes unfamiliar knowledge from IFT datasets to mitigate its impact on model truthfulness, whereas $UNIT_{ref}$ trains LLMs to recognize their uncertainty and explicitly indicate it at the end of their responses. Our experiments show that $UNIT_{cut}$ substantially improves LLM truthfulness, while $UNIT_{ref}$ maintains high informativeness and reduces hallucinations by distinguishing between confident and uncertain statements.
>
---
#### [replaced 032] SMAR: Soft Modality-Aware Routing Strategy for MoE-based Multimodal Large Language Models Preserving Language Capabilities
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.06406v2](http://arxiv.org/pdf/2506.06406v2)**

> **作者:** Guoyang Xia; Yifeng Ding; Fengfa Li; Lei Ren; Wei Chen; Fangxiang Feng; Xiaojie Wang
>
> **摘要:** Mixture of Experts (MoE) architectures have become a key approach for scaling large language models, with growing interest in extending them to multimodal tasks. Existing methods to build multimodal MoE models either incur high training costs or suffer from degraded language capabilities when adapting pretrained models. To address this, we propose Soft ModalityAware Routing (SMAR), a novel regularization technique that uses Kullback Leibler divergence to control routing probability distributions across modalities, encouraging expert specialization without modifying model architecture or heavily relying on textual data. Experiments on visual instruction tuning show that SMAR preserves language ability at 86.6% retention with only 2.5% pure text, outperforming baselines while maintaining strong multimodal performance. Our approach offers a practical and efficient solution to balance modality differentiation and language capabilities in multimodal MoE models.
>
---
#### [replaced 033] The Noisy Path from Source to Citation: Measuring How Scholars Engage with Past Research
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20581v3](http://arxiv.org/pdf/2502.20581v3)**

> **作者:** Hong Chen; Misha Teplitskiy; David Jurgens
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Academic citations are widely used for evaluating research and tracing knowledge flows. Such uses typically rely on raw citation counts and neglect variability in citation types. In particular, citations can vary in their fidelity as original knowledge from cited studies may be paraphrased, summarized, or reinterpreted, possibly wrongly, leading to variation in how much information changes from cited to citing paper. In this study, we introduce a computational pipeline to quantify citation fidelity at scale. Using full texts of papers, the pipeline identifies citations in citing papers and the corresponding claims in cited papers, and applies supervised models to measure fidelity at the sentence level. Analyzing a large-scale multi-disciplinary dataset of approximately 13 million citation sentence pairs, we find that citation fidelity is higher when authors cite papers that are 1) more recent and intellectually close, 2) more accessible, and 3) the first author has a lower H-index and the author team is medium-sized. Using a quasi-experiment, we establish the "telephone effect" - when citing papers have low fidelity to the original claim, future papers that cite the citing paper and the original have lower fidelity to the original. Our work reveals systematic differences in citation fidelity, underscoring the limitations of analyses that rely on citation quantity alone and the potential for distortion of evidence.
>
---
#### [replaced 034] Language Models Learn Rare Phenomena from Less Rare Phenomena: The Case of the Missing AANNs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2403.19827v3](http://arxiv.org/pdf/2403.19827v3)**

> **作者:** Kanishka Misra; Kyle Mahowald
>
> **备注:** Added Corrigendum to correct 4-gram baseline performance and chance performance
>
> **摘要:** Language models learn rare syntactic phenomena, but the extent to which this is attributable to generalization vs. memorization is a major open question. To that end, we iteratively trained transformer language models on systematically manipulated corpora which were human-scale in size, and then evaluated their learning of a rare grammatical phenomenon: the English Article+Adjective+Numeral+Noun (AANN) construction (``a beautiful five days''). We compared how well this construction was learned on the default corpus relative to a counterfactual corpus in which AANN sentences were removed. We found that AANNs were still learned better than systematically perturbed variants of the construction. Using additional counterfactual corpora, we suggest that this learning occurs through generalization from related constructions (e.g., ``a few days''). An additional experiment showed that this learning is enhanced when there is more variability in the input. Taken together, our results provide an existence proof that LMs can learn rare grammatical phenomena by generalization from less rare phenomena. Data and code: https://github.com/kanishkamisra/aannalysis.
>
---
#### [replaced 035] What Matters in LLM-generated Data: Diversity and Its Effect on Model Fine-Tuning
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.19262v2](http://arxiv.org/pdf/2506.19262v2)**

> **作者:** Yuchang Zhu; Huazhen Zhong; Qunshu Lin; Haotong Wei; Xiaolong Sun; Zixuan Yu; Minghao Liu; Zibin Zheng; Liang Chen
>
> **备注:** Ongoing work
>
> **摘要:** With the remarkable generative capabilities of large language models (LLMs), using LLM-generated data to train downstream models has emerged as a promising approach to mitigate data scarcity in specific domains and reduce time-consuming annotations. However, recent studies have highlighted a critical issue: iterative training on self-generated data results in model collapse, where model performance degrades over time. Despite extensive research on the implications of LLM-generated data, these works often neglect the importance of data diversity, a key factor in data quality. In this work, we aim to understand the implications of the diversity of LLM-generated data on downstream model performance. Specifically, we explore how varying levels of diversity in LLM-generated data affect downstream model performance. Additionally, we investigate the performance of models trained on data that mixes different proportions of LLM-generated data, which we refer to as synthetic data. Our experimental results show that, with minimal distribution shift, moderately diverse LLM-generated data can enhance model performance in scenarios with insufficient labeled data, whereas highly diverse generated data has a negative impact. We hope our empirical findings will offer valuable guidance for future studies on LLMs as data generators.
>
---
#### [replaced 036] PP-DocBee2: Improved Baselines with Efficient Data for Multimodal Document Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.18023v2](http://arxiv.org/pdf/2506.18023v2)**

> **作者:** Kui Huang; Xinrong Chen; Wenyu Lv; Jincheng Liao; Guanzhong Wang; Yi Liu
>
> **摘要:** This report introduces PP-DocBee2, an advanced version of the PP-DocBee, designed to enhance multimodal document understanding. Built on a large multimodal model architecture, PP-DocBee2 addresses the limitations of its predecessor through key technological improvements, including enhanced synthetic data quality, improved visual feature fusion strategy, and optimized inference methodologies. These enhancements yield an $11.4\%$ performance boost on internal benchmarks for Chinese business documents, and reduce inference latency by $73.0\%$ to the vanilla version. A key innovation of our work is a data quality optimization strategy for multimodal document tasks. By employing a large-scale multimodal pre-trained model to evaluate data, we apply a novel statistical criterion to filter outliers, ensuring high-quality training data. Inspired by insights into underutilized intermediate features in multimodal models, we enhance the ViT representational capacity by decomposing it into layers and applying a novel feature fusion strategy to improve complex reasoning. The source code and pre-trained model are available at \href{https://github.com/PaddlePaddle/PaddleMIX}{https://github.com/PaddlePaddle/PaddleMIX}.
>
---
#### [replaced 037] When Large Language Models contradict humans? Large Language Models' Sycophantic Behaviour
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2311.09410v4](http://arxiv.org/pdf/2311.09410v4)**

> **作者:** Leonardo Ranaldi; Giulia Pucci
>
> **摘要:** Large Language Models have been demonstrating broadly satisfactory generative abilities for users, which seems to be due to the intensive use of human feedback that refines responses. Nevertheless, suggestibility inherited via human feedback improves the inclination to produce answers corresponding to users' viewpoints. This behaviour is known as sycophancy and depicts the tendency of LLMs to generate misleading responses as long as they align with humans. This phenomenon induces bias and reduces the robustness and, consequently, the reliability of these models. In this paper, we study the suggestibility of Large Language Models (LLMs) to sycophantic behaviour, analysing these tendencies via systematic human-interventions prompts over different tasks. Our investigation demonstrates that LLMs have sycophantic tendencies when answering queries that involve subjective opinions and statements that should elicit a contrary response based on facts. In contrast, when faced with math tasks or queries with an objective answer, they, at various scales, do not follow the users' hints by demonstrating confidence in generating the correct answers.
>
---
#### [replaced 038] Confucius3-Math: A Lightweight High-Performance Reasoning LLM for Chinese K-12 Mathematics Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.18330v2](http://arxiv.org/pdf/2506.18330v2)**

> **作者:** Lixin Wu; Na Cai; Qiao Cheng; Jiachen Wang; Yitao Duan
>
> **摘要:** We introduce Confucius3-Math, an open-source large language model with 14B parameters that (1) runs efficiently on a single consumer-grade GPU; (2) achieves SOTA performances on a range of mathematical reasoning tasks, outperforming many models with significantly larger sizes. In particular, as part of our mission to enhancing education and knowledge dissemination with AI, Confucius3-Math is specifically committed to mathematics learning for Chinese K-12 students and educators. Built via post-training with large-scale reinforcement learning (RL), Confucius3-Math aligns with national curriculum and excels at solving main-stream Chinese K-12 mathematical problems with low cost. In this report we share our development recipe, the challenges we encounter and the techniques we develop to overcome them. In particular, we introduce three technical innovations: Targeted Entropy Regularization, Recent Sample Recovery and Policy-Specific Hardness Weighting. These innovations encompass a new entropy regularization, a novel data scheduling policy, and an improved group-relative advantage estimator. Collectively, they significantly stabilize the RL training, improve data efficiency, and boost performance. Our work demonstrates the feasibility of building strong reasoning models in a particular domain at low cost. We open-source our model and code at https://github.com/netease-youdao/Confucius3-Math.
>
---
#### [replaced 039] Computation Mechanism Behind LLM Position Generalization
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.13305v3](http://arxiv.org/pdf/2503.13305v3)**

> **作者:** Chi Han; Heng Ji
>
> **备注:** ACL 2025 Main Long Paper
>
> **摘要:** Most written natural languages are composed of sequences of words and sentences. Similar to humans, large language models (LLMs) exhibit flexibility in handling textual positions - a phenomenon we term position generalization. They can understand texts with position perturbations and generalize to longer texts than those encountered during training with the latest techniques. These phenomena suggest that LLMs handle positions tolerantly, but how LLMs computationally process positional relevance remains largely unexplored. This work connects the linguistic phenomenon with LLMs' computational mechanisms. We show how LLMs enforce certain computational mechanisms for the aforementioned tolerance in position perturbations. Despite the complex design of the self-attention mechanism, this work reveals that LLMs learn a counterintuitive disentanglement of attention logits. Their values show a 0.959 linear correlation with an approximation of the arithmetic sum of positional relevance and semantic importance. Furthermore, we identify a prevalent pattern in intermediate features, which we prove theoretically enables this effect. The pattern, which is different from how randomly initialized parameters would behave, suggests that it is a learned behavior rather than a natural result of the model architecture. Based on these findings, we provide computational explanations and criteria for LLMs' position flexibilities. This work takes a pioneering step in linking position generalization with modern LLMs' internal mechanisms.
>
---
#### [replaced 040] GlyphPattern: An Abstract Pattern Recognition Benchmark for Vision-Language Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.05894v2](http://arxiv.org/pdf/2408.05894v2)**

> **作者:** Zixuan Wu; Yoolim Kim; Carolyn Jane Anderson
>
> **摘要:** Vision-Language Models (VLMs) building upon the foundation of powerful large language models have made rapid progress in reasoning across visual and textual data. While VLMs perform well on vision tasks that they are trained on, our results highlight key challenges in abstract pattern recognition. We present GlyphPattern, a 954 item dataset that pairs 318 human-written descriptions of visual patterns from 40 writing systems with three visual presentation styles. GlyphPattern evaluates abstract pattern recognition in VLMs, requiring models to understand and judge natural language descriptions of visual patterns. GlyphPattern patterns are drawn from a large-scale cognitive science investigation of human writing systems; as a result, they are rich in spatial reference and compositionality. Our experiments show that GlyphPattern is challenging for state-of-the-art VLMs (GPT-4o achieves only 55% accuracy), with marginal gains from few-shot prompting. Our detailed error analysis reveals challenges at multiple levels, including visual processing, natural language understanding, and pattern generalization.
>
---
#### [replaced 041] FactCheckmate: Preemptively Detecting and Mitigating Hallucinations in LMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.02899v2](http://arxiv.org/pdf/2410.02899v2)**

> **作者:** Deema Alnuhait; Neeraja Kirtane; Muhammad Khalifa; Hao Peng
>
> **摘要:** Language models (LMs) hallucinate. We inquire: Can we detect and mitigate hallucinations before they happen? This work answers this research question in the positive, by showing that the internal representations of LMs provide rich signals that can be used for this purpose. We introduce FactCheckmate, which preemptively detects hallucinations by learning a classifier that predicts whether the LM will hallucinate, based on the model's hidden states produced over the inputs, before decoding begins. If a hallucination is detected, FactCheckmate then intervenes by adjusting the LM's hidden states such that the model will produce more factual outputs. FactCheckmate provides fresh insights that the inner workings of LMs can be revealed by their hidden states. Practically, both its detection and mitigation models are lightweight, adding little inference overhead; FactCheckmate proves a more efficient approach for mitigating hallucinations compared to many post-hoc alternatives. We evaluate FactCheckmate over LMs of different scales and model families (including Llama, Mistral, Qwen and Gemma), across a variety of QA datasets from different domains. Our results demonstrate the effectiveness of FactCheckmate, achieving over 70% preemptive detection accuracy. On average, outputs generated by LMs with intervention are 34.4% more factual compared to those without.
>
---
#### [replaced 042] Graph Linearization Methods for Reasoning on Graphs with Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.19494v3](http://arxiv.org/pdf/2410.19494v3)**

> **作者:** Christos Xypolopoulos; Guokan Shang; Xiao Fei; Giannis Nikolentzos; Hadi Abdine; Iakovos Evdaimon; Michail Chatzianastasis; Giorgos Stamou; Michalis Vazirgiannis
>
> **摘要:** Large language models have evolved to process multiple modalities beyond text, such as images and audio, which motivates us to explore how to effectively leverage them for graph reasoning tasks. The key question, therefore, is how to transform graphs into linear sequences of tokens, a process we term "graph linearization", so that LLMs can handle graphs naturally. We consider that graphs should be linearized meaningfully to reflect certain properties of natural language text, such as local dependency and global alignment, in order to ease contemporary LLMs, trained on trillions of textual tokens, better understand graphs. To achieve this, we developed several graph linearization methods based on graph centrality and degeneracy. These methods are further enhanced using node relabeling techniques. The experimental results demonstrate the effectiveness of our methods compared to the random linearization baseline. Our work introduces novel graph representations suitable for LLMs, contributing to the potential integration of graph machine learning with the trend of multimodal processing using a unified transformer model.
>
---
#### [replaced 043] Can Language Models Replace Programmers for Coding? REPOCOD Says 'Not Yet'
- **分类: cs.SE; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.21647v4](http://arxiv.org/pdf/2410.21647v4)**

> **作者:** Shanchao Liang; Yiran Hu; Nan Jiang; Lin Tan
>
> **摘要:** Recently, a number of repository-level code generation benchmarks-such as CoderEval, DevEval, RepoEval, RepoBench, and LongCodeArena-have emerged to evaluate the capabilities of large language models (LLMs) beyond standalone benchmarks like HumanEval and MBPP. Thus, a natural question is, would LLMs have similar performance in real world coding tasks as their performance in these benchmarks? Unfortunately, one cannot answer this question, since these benchmarks consist of short completions, synthetic examples, or focus on limited scale repositories, failing to represent real-world coding tasks. To address these challenges, we create REPOCOD, a Python code-generation benchmark containing complex tasks with realistic dependencies in real-world large projects and appropriate metrics for evaluating source code. It includes 980 whole-function generation tasks from 11 popular projects, 50.8% of which require repository-level context. REPOCOD includes 314 developer-written test cases per instance for better evaluation. We evaluate ten LLMs on REPOCOD and find that none achieves more than 30% pass@1 on REPOCOD, indicating the necessity of building stronger LLMs that can help developers in real-world software development. In addition, we found that retrieval-augmented generation achieves better results than using target function dependencies as context.
>
---
#### [replaced 044] Evaluating Rare Disease Diagnostic Performance in Symptom Checkers: A Synthetic Vignette Simulation Approach
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.19750v2](http://arxiv.org/pdf/2506.19750v2)**

> **作者:** Takashi Nishibayashi; Seiji Kanazawa; Kumpei Yamada
>
> **摘要:** Symptom Checkers (SCs) provide users with personalized medical information. To prevent performance degradation from algorithm updates, SC developers must evaluate diagnostic performance changes for individual diseases before deployment. However, acquiring sufficient evaluation data for rare diseases is difficult, and manually creating numerous clinical vignettes is costly and impractical. This study proposes and validates a novel Synthetic Vignette Simulation Approach to evaluate diagnostic performance changes for individual rare diseases following SC algorithm updates. We used disease-phenotype annotations from the Human Phenotype Ontology (HPO), a knowledge database for rare diseases, to generate synthetic vignettes. With these, we simulated SC interviews to estimate the impact of algorithm updates on real-world diagnostic performance. The method's effectiveness was evaluated retrospectively by comparing estimated values with actual metric changes using the $R^2$ coefficient. The experiment included eight past SC algorithm updates. For updates on diseases with frequency information in HPO (n=5), the $R^2$ for Recall@8 change was 0.831 ($p$=0.031), and for Precision@8 change, it was 0.78 ($p$=0.047), indicating the method can predict post-deployment performance. In contrast, large prediction errors occurred for diseases without frequency information (n=3), highlighting its importance. Our method enables pre-deployment evaluation of SC algorithm changes for individual rare diseases using a publicly available, expert-created knowledge base. This transparent and low-cost approach allows developers to efficiently improve diagnostic performance for rare diseases, potentially enhancing support for early diagnosis.
>
---

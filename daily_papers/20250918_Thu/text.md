# 自然语言处理 cs.CL

- **最新发布 61 篇**

- **更新 66 篇**

## 最新发布

#### [new 001] SSL-SSAW: Self-Supervised Learning with Sigmoid Self-Attention Weighting for Question-Based Sign Language Translation
- **分类: cs.CL; cs.AI**

- **简介: 论文提出SSL-SSAW方法，解决基于问题的唇语翻译任务，通过自监督学习与自注意机制融合多模态特征，提升翻译效果。该工作旨在利用对话上下文提高翻译质量，并在新数据集上取得SOTA结果。**

- **链接: [http://arxiv.org/pdf/2509.14036v1](http://arxiv.org/pdf/2509.14036v1)**

> **作者:** Zekang Liu; Wei Feng; Fanhua Shang; Lianyu Hu; Jichao Feng; Liqing Gao
>
> **摘要:** Sign Language Translation (SLT) bridges the communication gap between deaf people and hearing people, where dialogue provides crucial contextual cues to aid in translation. Building on this foundational concept, this paper proposes Question-based Sign Language Translation (QB-SLT), a novel task that explores the efficient integration of dialogue. Unlike gloss (sign language transcription) annotations, dialogue naturally occurs in communication and is easier to annotate. The key challenge lies in aligning multimodality features while leveraging the context of the question to improve translation. To address this issue, we propose a cross-modality Self-supervised Learning with Sigmoid Self-attention Weighting (SSL-SSAW) fusion method for sign language translation. Specifically, we employ contrastive learning to align multimodality features in QB-SLT, then introduce a Sigmoid Self-attention Weighting (SSAW) module for adaptive feature extraction from question and sign language sequences. Additionally, we leverage available question text through self-supervised learning to enhance representation and translation capabilities. We evaluated our approach on newly constructed CSL-Daily-QA and PHOENIX-2014T-QA datasets, where SSL-SSAW achieved SOTA performance. Notably, easily accessible question assistance can achieve or even surpass the performance of gloss assistance. Furthermore, visualization results demonstrate the effectiveness of incorporating dialogue in improving translation quality.
>
---
#### [new 002] Op-Fed: Opinion, Stance, and Monetary Policy Annotations on FOMC Transcripts Using Active Learning
- **分类: cs.CL**

- **简介: 该论文提出Op-Fed数据集，用于标注美联储会议记录中的观点、立场和货币政策内容。解决标注类别不平衡和句子间依赖问题，采用主动学习提升正样本数量，并评估LLM在该任务上的表现。属于文本标注与分类任务。**

- **链接: [http://arxiv.org/pdf/2509.13539v1](http://arxiv.org/pdf/2509.13539v1)**

> **作者:** Alisa Kanganis; Katherine A. Keith
>
> **摘要:** The U.S. Federal Open Market Committee (FOMC) regularly discusses and sets monetary policy, affecting the borrowing and spending decisions of millions of people. In this work, we release Op-Fed, a dataset of 1044 human-annotated sentences and their contexts from FOMC transcripts. We faced two major technical challenges in dataset creation: imbalanced classes -- we estimate fewer than 8% of sentences express a non-neutral stance towards monetary policy -- and inter-sentence dependence -- 65% of instances require context beyond the sentence-level. To address these challenges, we developed a five-stage hierarchical schema to isolate aspects of opinion, monetary policy, and stance towards monetary policy as well as the level of context needed. Second, we selected instances to annotate using active learning, roughly doubling the number of positive instances across all schema aspects. Using Op-Fed, we found a top-performing, closed-weight LLM achieves 0.80 zero-shot accuracy in opinion classification but only 0.61 zero-shot accuracy classifying stance towards monetary policy -- below our human baseline of 0.89. We expect Op-Fed to be useful for future model training, confidence calibration, and as a seed dataset for future annotation efforts.
>
---
#### [new 003] Integrating Text and Time-Series into (Large) Language Models to Predict Medical Outcomes
- **分类: cs.CL**

- **简介: 论文将大语言模型与时间序列数据结合，用于预测医疗结果。属于医疗分类任务，解决LLMs处理结构化临床数据能力不足的问题，通过DSPy优化提示，实现与多模态系统相当的性能。**

- **链接: [http://arxiv.org/pdf/2509.13696v1](http://arxiv.org/pdf/2509.13696v1)**

> **作者:** Iyadh Ben Cheikh Larbi; Ajay Madhavan Ravichandran; Aljoscha Burchardt; Roland Roller
>
> **备注:** Presented and published at BioCreative IX
>
> **摘要:** Large language models (LLMs) excel at text generation, but their ability to handle clinical classification tasks involving structured data, such as time series, remains underexplored. In this work, we adapt instruction-tuned LLMs using DSPy-based prompt optimization to process clinical notes and structured EHR inputs jointly. Our results show that this approach achieves performance on par with specialized multimodal systems while requiring less complexity and offering greater adaptability across tasks.
>
---
#### [new 004] DSCC-HS: A Dynamic Self-Reinforcing Framework for Hallucination Suppression in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DSCC-HS框架，用于抑制大语言模型的幻觉问题。通过动态校准机制，在推理过程中实时调整模型输出，提升事实一致性。属于自然语言处理中的事实校验任务。**

- **链接: [http://arxiv.org/pdf/2509.13702v1](http://arxiv.org/pdf/2509.13702v1)**

> **作者:** Xiao Zheng
>
> **摘要:** Large Language Model (LLM) hallucination is a significant barrier to their reliable deployment. Current methods like Retrieval-Augmented Generation (RAG) are often reactive. We introduce **Dynamic Self-reinforcing Calibration for Hallucination Suppression (DSCC-HS)**, a novel, proactive framework that intervenes during autoregressive decoding. Inspired by dual-process cognitive theory, DSCC-HS uses a compact proxy model, trained in adversarial roles as a Factual Alignment Proxy (FAP) and a Hallucination Detection Proxy (HDP). During inference, these proxies dynamically steer a large target model by injecting a real-time steering vector, which is the difference between FAP and HDP logits, at each decoding step. This plug-and-play approach requires no modification to the target model. Our experiments on TruthfulQA and BioGEN show DSCC-HS achieves state-of-the-art performance. On TruthfulQA, it reached a 99.2% Factual Consistency Rate (FCR). On the long-form BioGEN benchmark, it attained the highest FActScore of 46.50. These results validate DSCC-HS as a principled and efficient solution for enhancing LLM factuality.
>
---
#### [new 005] Can Large Language Models Robustly Perform Natural Language Inference for Japanese Comparatives?
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在日语比较级自然语言推理任务中的鲁棒性。针对日语中逻辑和数值表达的挑战，构建了专用数据集，并评估模型在零样本和少样本设置下的表现，发现模型对提示格式和示例标签敏感，且难以处理日语特有的语言现象。**

- **链接: [http://arxiv.org/pdf/2509.13695v1](http://arxiv.org/pdf/2509.13695v1)**

> **作者:** Yosuke Mikami; Daiki Matsuoka; Hitomi Yanaka
>
> **备注:** To appear in Proceedings of the 16th International Conference on Computational Semantics (IWCS 2025)
>
> **摘要:** Large Language Models (LLMs) perform remarkably well in Natural Language Inference (NLI). However, NLI involving numerical and logical expressions remains challenging. Comparatives are a key linguistic phenomenon related to such inference, but the robustness of LLMs in handling them, especially in languages that are not dominant in the models' training data, such as Japanese, has not been sufficiently explored. To address this gap, we construct a Japanese NLI dataset that focuses on comparatives and evaluate various LLMs in zero-shot and few-shot settings. Our results show that the performance of the models is sensitive to the prompt formats in the zero-shot setting and influenced by the gold labels in the few-shot examples. The LLMs also struggle to handle linguistic phenomena unique to Japanese. Furthermore, we observe that prompts containing logical semantic representations help the models predict the correct labels for inference problems that they struggle to solve even with few-shot examples.
>
---
#### [new 006] Findings of the Third Automatic Minuting (AutoMin) Challenge
- **分类: cs.CL**

- **简介: 论文介绍第三次AutoMin挑战赛，聚焦会议纪要生成与问答任务。涉及英捷双语及两类会议场景，评估大语言模型在结构化纪要和跨语言问答中的表现，提供基线系统以促进研究。**

- **链接: [http://arxiv.org/pdf/2509.13814v1](http://arxiv.org/pdf/2509.13814v1)**

> **作者:** Kartik Shinde; Laurent Besacier; Ondrej Bojar; Thibaut Thonet; Tirthankar Ghosal
>
> **备注:** Automin 2025 Website: https://ufal.github.io/automin-2025/
>
> **摘要:** This paper presents the third edition of AutoMin, a shared task on automatic meeting summarization into minutes. In 2025, AutoMin featured the main task of minuting, the creation of structured meeting minutes, as well as a new task: question answering (QA) based on meeting transcripts. The minuting task covered two languages, English and Czech, and two domains: project meetings and European Parliament sessions. The QA task focused solely on project meetings and was available in two settings: monolingual QA in English, and cross-lingual QA, where questions were asked and answered in Czech based on English meetings. Participation in 2025 was more limited compared to previous years, with only one team joining the minuting task and two teams participating in QA. However, as organizers, we included multiple baseline systems to enable a comprehensive evaluation of current (2025) large language models (LLMs) on both tasks.
>
---
#### [new 007] Exploring Data and Parameter Efficient Strategies for Arabic Dialect Identifications
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究阿拉伯语方言识别（ADI）任务，旨在探索数据高效和参数高效方法。通过分析多种软提示策略和LoRA微调技术，发现LoRA模型表现最佳，优于全量微调。**

- **链接: [http://arxiv.org/pdf/2509.13775v1](http://arxiv.org/pdf/2509.13775v1)**

> **作者:** Vani Kanjirangat; Ljiljana Dolamic; Fabio Rinaldi
>
> **备注:** 4 main pages, 4 additional, 5 figures
>
> **摘要:** This paper discusses our exploration of different data-efficient and parameter-efficient approaches to Arabic Dialect Identification (ADI). In particular, we investigate various soft-prompting strategies, including prefix-tuning, prompt-tuning, P-tuning, and P-tuning V2, as well as LoRA reparameterizations. For the data-efficient strategy, we analyze hard prompting with zero-shot and few-shot inferences to analyze the dialect identification capabilities of Large Language Models (LLMs). For the parameter-efficient PEFT approaches, we conducted our experiments using Arabic-specific encoder models on several major datasets. We also analyzed the n-shot inferences on open-source decoder-only models, a general multilingual model (Phi-3.5), and an Arabic-specific one(SILMA). We observed that the LLMs generally struggle to differentiate the dialectal nuances in the few-shot or zero-shot setups. The soft-prompted encoder variants perform better, while the LoRA-based fine-tuned models perform best, even surpassing full fine-tuning.
>
---
#### [new 008] Teaching According to Talents! Instruction Tuning LLMs with Competence-Aware Curriculum Learning
- **分类: cs.CL; cs.AI**

- **简介: 论文提出CAMPUS框架，解决指令微调中课程学习僵化问题。通过动态子课程选择、能力感知调整和多难度调度，提升大模型训练效率与性能。属于高效指令微调任务。**

- **链接: [http://arxiv.org/pdf/2509.13790v1](http://arxiv.org/pdf/2509.13790v1)**

> **作者:** Yangning Li; Tingwei Lu; Yinghui Li; Yankai Chen; Wei-Chieh Huang; Wenhao Jiang; Hui Wang; Hai-Tao Zheng; Philip S. Yu
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Efficient instruction tuning aims to enhance the ultimate performance of large language models (LLMs) trained on a given instruction dataset. Curriculum learning as a typical data organization strategy has shown preliminary effectiveness in instruction tuning. However, current curriculum tuning methods suffer from the curriculum rigidity, since they rely solely on static heuristic difficulty metrics. These methods fail to adapt to the evolving capabilities of models during training, resulting in a fixed and potentially sub-optimal learning trajectory. To address the issue, Competence-Aware Multi-Perspective cUrriculum inStruction tuning framework termed CAMPUS is proposed. CAMPUS offers several advantages: (1) Dynamic selection for sub-curriculum. (2) Competency-aware adjustment to the curriculum schedule. (3) Multiple difficulty-based scheduling. Extensive experiments prove the superior performance of CAMPUS, compared to other state-of-the-art baselines for efficient instruction tuning.
>
---
#### [new 009] Do Large Language Models Understand Word Senses?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型对词义的理解能力，属于词义消歧任务。论文评估了指令微调LLM在WSD任务中的表现，并测试其在生成定义、解释和示例任务中的理解能力，发现其性能接近专用系统且具有较高准确性。**

- **链接: [http://arxiv.org/pdf/2509.13905v1](http://arxiv.org/pdf/2509.13905v1)**

> **作者:** Domenico Meconi; Simone Stirpe; Federico Martelli; Leonardo Lavalle; Roberto Navigli
>
> **备注:** 20 pages, to be published in EMNLP2025
>
> **摘要:** Understanding the meaning of words in context is a fundamental capability for Large Language Models (LLMs). Despite extensive evaluation efforts, the extent to which LLMs show evidence that they truly grasp word senses remains underexplored. In this paper, we address this gap by evaluating both i) the Word Sense Disambiguation (WSD) capabilities of instruction-tuned LLMs, comparing their performance to state-of-the-art systems specifically designed for the task, and ii) the ability of two top-performing open- and closed-source LLMs to understand word senses in three generative settings: definition generation, free-form explanation, and example generation. Notably, we find that, in the WSD task, leading models such as GPT-4o and DeepSeek-V3 achieve performance on par with specialized WSD systems, while also demonstrating greater robustness across domains and levels of difficulty. In the generation tasks, results reveal that LLMs can explain the meaning of words in context up to 98\% accuracy, with the highest performance observed in the free-form explanation task, which best aligns with their generative capabilities.
>
---
#### [new 010] Slim-SC: Thought Pruning for Efficient Scaling with Self-Consistency
- **分类: cs.CL; cs.AI; cs.LG; I.2.7**

- **简介: 该论文提出Slim-SC方法，旨在解决自一致性（SC）测试时扩展计算开销大的问题。通过逐步剪枝冗余推理链，提升推理效率，在保持或提高准确率的同时减少延迟和KVC使用。属于大模型推理优化任务。**

- **链接: [http://arxiv.org/pdf/2509.13990v1](http://arxiv.org/pdf/2509.13990v1)**

> **作者:** Colin Hong; Xu Guo; Anand Chaanan Singh; Esha Choukse; Dmitrii Ustiugov
>
> **备注:** Accepted by EMNLP 2025 (Oral), 9 pages
>
> **摘要:** Recently, Test-Time Scaling (TTS) has gained increasing attention for improving LLM reasoning performance at test time without retraining the model. A notable TTS technique is Self-Consistency (SC), which generates multiple reasoning chains in parallel and selects the final answer via majority voting. While effective, the order-of-magnitude computational overhead limits its broad deployment. Prior attempts to accelerate SC mainly rely on model-based confidence scores or heuristics with limited empirical support. For the first time, we theoretically and empirically analyze the inefficiencies of SC and reveal actionable opportunities for improvement. Building on these insights, we propose Slim-SC, a step-wise pruning strategy that identifies and removes redundant chains using inter-chain similarity at the thought level. Experiments on three STEM reasoning datasets and two recent LLM architectures show that Slim-SC reduces inference latency and KVC usage by up to 45% and 26%, respectively, with R1-Distill, while maintaining or improving accuracy, thus offering a simple yet efficient TTS alternative for SC.
>
---
#### [new 011] Latent Traits and Cross-Task Transfer: Deconstructing Dataset Interactions in LLM Fine-tuning
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大语言模型微调中的跨任务迁移机制，分析不同数据集间的交互影响。通过构建迁移学习矩阵和降维方法，揭示隐藏统计因素对性能的影响，提升迁移学习的可预测性与有效性。**

- **链接: [http://arxiv.org/pdf/2509.13624v1](http://arxiv.org/pdf/2509.13624v1)**

> **作者:** Shambhavi Krishna; Atharva Naik; Chaitali Agarwal; Sudharshan Govindan; Taesung Lee; Haw-Shiuan Chang
>
> **备注:** Camera-ready version. Accepted to appear in the proceedings of the 14th Joint Conference on Lexical and Computational Semantics (*SEM 2025)
>
> **摘要:** Large language models are increasingly deployed across diverse applications. This often includes tasks LLMs have not encountered during training. This implies that enumerating and obtaining the high-quality training data for all tasks is infeasible. Thus, we often need to rely on transfer learning using datasets with different characteristics, and anticipate out-of-distribution requests. Motivated by this practical need, we propose an analysis framework, building a transfer learning matrix and dimensionality reduction, to dissect these cross-task interactions. We train and analyze 10 models to identify latent abilities (e.g., Reasoning, Sentiment Classification, NLU, Arithmetic) and discover the side effects of the transfer learning. Our findings reveal that performance improvements often defy explanations based on surface-level dataset similarity or source data quality. Instead, hidden statistical factors of the source dataset, such as class distribution and generation length proclivities, alongside specific linguistic features, are actually more influential. This work offers insights into the complex dynamics of transfer learning, paving the way for more predictable and effective LLM adaptation.
>
---
#### [new 012] You Are What You Train: Effects of Data Composition on Training Context-aware Machine Translation Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究上下文感知机器翻译模型的训练数据组成对性能的影响。通过控制训练数据中上下文相关示例的比例，验证了数据稀疏性是限制模型利用上下文的关键因素，并提出两种策略提升翻译准确性。属于机器翻译任务，解决上下文利用不足的问题。**

- **链接: [http://arxiv.org/pdf/2509.14031v1](http://arxiv.org/pdf/2509.14031v1)**

> **作者:** Paweł Mąka; Yusuf Can Semerci; Jan Scholtes; Gerasimos Spanakis
>
> **备注:** EMNLP 2025 main conference
>
> **摘要:** Achieving human-level translations requires leveraging context to ensure coherence and handle complex phenomena like pronoun disambiguation. Sparsity of contextually rich examples in the standard training data has been hypothesized as the reason for the difficulty of context utilization. In this work, we systematically validate this claim in both single- and multilingual settings by constructing training datasets with a controlled proportions of contextually relevant examples. We demonstrate a strong association between training data sparsity and model performance confirming sparsity as a key bottleneck. Importantly, we reveal that improvements in one contextual phenomenon do no generalize to others. While we observe some cross-lingual transfer, it is not significantly higher between languages within the same sub-family. Finally, we propose and empirically evaluate two training strategies designed to leverage the available data. These strategies improve context utilization, resulting in accuracy gains of up to 6 and 8 percentage points on the ctxPro evaluation in single- and multilingual settings respectively.
>
---
#### [new 013] DSPC: Dual-Stage Progressive Compression Framework for Efficient Long-Context Reasoning
- **分类: cs.CL**

- **简介: 该论文提出DSPC框架，用于高效长文本推理。针对提示符过长导致计算成本高的问题，提出双阶段无训练压缩方法，通过语义过滤和令牌重要性评估实现压缩，在少样本任务中显著提升性能。**

- **链接: [http://arxiv.org/pdf/2509.13723v1](http://arxiv.org/pdf/2509.13723v1)**

> **作者:** Yaxin Gao; Yao Lu; Zongfei Zhang; Jiaqi Nie; Shanqing Yu; Qi Xuan
>
> **摘要:** Large language models (LLMs) have achieved remarkable success in many natural language processing (NLP) tasks. To achieve more accurate output, the prompts used to drive LLMs have become increasingly longer, which incurs higher computational costs. To address this prompt inflation problem, prompt compression has been proposed. However, most existing methods require training a small auxiliary model for compression, incurring a significant amount of additional computation. To avoid this, we propose a two-stage, training-free approach, called Dual-Stage Progressive Compression (DSPC). In the coarse-grained stage, semantic-related sentence filtering removes sentences with low semantic value based on TF-IDF. In the fine-grained stage, token importance is assessed using attention contribution, cross-model loss difference, and positional importance, enabling the pruning of low-utility tokens while preserving semantics. We validate DSPC on LLaMA-3.1-8B-Instruct and GPT-3.5-Turbo under a constrained token budget and observe consistent improvements. For instance, in the FewShot task of the Longbench dataset, DSPC achieves a performance of 49.17 by using only 3x fewer tokens, outperforming the best state-of-the-art baseline LongLLMLingua by 7.76.
>
---
#### [new 014] Synthesizing Behaviorally-Grounded Reasoning Chains: A Data-Generation Framework for Personal Finance LLMs
- **分类: cs.CL; cs.AI; cs.LG; 68T50; I.2.7; J.4**

- **简介: 该论文提出一种生成行为导向推理链的数据框架，用于训练个人理财LLM。旨在解决现有模型在财务建议中的准确性与个性化不足问题，通过整合行为金融学构建数据集，使小参数模型性能接近大模型，降低成本。**

- **链接: [http://arxiv.org/pdf/2509.14180v1](http://arxiv.org/pdf/2509.14180v1)**

> **作者:** Akhil Theerthala
>
> **备注:** 24 pages, 11 figures. The paper presents a novel framework for generating a personal finance dataset. The resulting fine-tuned model and dataset are publicly available
>
> **摘要:** Personalized financial advice requires consideration of user goals, constraints, risk tolerance, and jurisdiction. Prior LLM work has focused on support systems for investors and financial planners. Simultaneously, numerous recent studies examine broader personal finance tasks, including budgeting, debt management, retirement, and estate planning, through agentic pipelines that incur high maintenance costs, yielding less than 25% of their expected financial returns. In this study, we introduce a novel and reproducible framework that integrates relevant financial context with behavioral finance studies to construct supervision data for end-to-end advisors. Using this framework, we create a 19k sample reasoning dataset and conduct a comprehensive fine-tuning of the Qwen-3-8B model on the dataset. Through a held-out test split and a blind LLM-jury study, we demonstrate that through careful data curation and behavioral integration, our 8B model achieves performance comparable to significantly larger baselines (14-32B parameters) across factual accuracy, fluency, and personalization metrics while incurring 80% lower costs than the larger counterparts.
>
---
#### [new 015] Apertus: Democratizing Open and Compliant LLMs for Global Language Environments
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Apertus，一套开源大语言模型，解决数据合规与多语言表征问题。采用开放数据训练，过滤非法内容，提升多语言覆盖，并发布全部开发资源，促进透明与扩展。**

- **链接: [http://arxiv.org/pdf/2509.14233v1](http://arxiv.org/pdf/2509.14233v1)**

> **作者:** Alejandro Hernández-Cano; Alexander Hägele; Allen Hao Huang; Angelika Romanou; Antoni-Joan Solergibert; Barna Pasztor; Bettina Messmer; Dhia Garbaya; Eduard Frank Ďurech; Ido Hakimi; Juan García Giraldo; Mete Ismayilzada; Negar Foroutan; Skander Moalla; Tiancheng Chen; Vinko Sabolčec; Yixuan Xu; Michael Aerni; Badr AlKhamissi; Ines Altemir Marinas; Mohammad Hossein Amani; Matin Ansaripour; Ilia Badanin; Harold Benoit; Emanuela Boros; Nicholas Browning; Fabian Bösch; Maximilian Böther; Niklas Canova; Camille Challier; Clement Charmillot; Jonathan Coles; Jan Deriu; Arnout Devos; Lukas Drescher; Daniil Dzenhaliou; Maud Ehrmann; Dongyang Fan; Simin Fan; Silin Gao; Miguel Gila; María Grandury; Diba Hashemi; Alexander Hoyle; Jiaming Jiang; Mark Klein; Andrei Kucharavy; Anastasiia Kucherenko; Frederike Lübeck; Roman Machacek; Theofilos Manitaras; Andreas Marfurt; Kyle Matoba; Simon Matrenok; Henrique Mendoncça; Fawzi Roberto Mohamed; Syrielle Montariol; Luca Mouchel; Sven Najem-Meyer; Jingwei Ni; Gennaro Oliva; Matteo Pagliardini; Elia Palme; Andrei Panferov; Léo Paoletti; Marco Passerini; Ivan Pavlov; Auguste Poiroux; Kaustubh Ponkshe; Nathan Ranchin; Javi Rando; Mathieu Sauser; Jakhongir Saydaliev; Muhammad Ali Sayfiddinov; Marian Schneider; Stefano Schuppli; Marco Scialanga; Andrei Semenov; Kumar Shridhar; Raghav Singhal; Anna Sotnikova; Alexander Sternfeld; Ayush Kumar Tarun; Paul Teiletche; Jannis Vamvas; Xiaozhe Yao; Hao Zhao Alexander Ilic; Ana Klimovic; Andreas Krause; Caglar Gulcehre; David Rosenthal; Elliott Ash; Florian Tramèr; Joost VandeVondele; Livio Veraldi; Martin Rajman; Thomas Schulthess; Torsten Hoefler; Antoine Bosselut; Martin Jaggi; Imanol Schlag
>
> **摘要:** We present Apertus, a fully open suite of large language models (LLMs) designed to address two systemic shortcomings in today's open model ecosystem: data compliance and multilingual representation. Unlike many prior models that release weights without reproducible data pipelines or regard for content-owner rights, Apertus models are pretrained exclusively on openly available data, retroactively respecting robots.txt exclusions and filtering for non-permissive, toxic, and personally identifiable content. To mitigate risks of memorization, we adopt the Goldfish objective during pretraining, strongly suppressing verbatim recall of data while retaining downstream task performance. The Apertus models also expand multilingual coverage, training on 15T tokens from over 1800 languages, with ~40% of pretraining data allocated to non-English content. Released at 8B and 70B scales, Apertus approaches state-of-the-art results among fully open models on multilingual benchmarks, rivalling or surpassing open-weight counterparts. Beyond model weights, we release all scientific artifacts from our development cycle with a permissive license, including data preparation scripts, checkpoints, evaluation suites, and training code, enabling transparent audit and extension.
>
---
#### [new 016] Combating Biomedical Misinformation through Multi-modal Claim Detection and Evidence-based Verification
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 论文提出CER框架，用于解决生物医学领域虚假信息问题。通过整合科学证据检索、大语言模型推理和监督预测，实现基于证据的自动事实核查，提升验证准确性与可靠性。**

- **链接: [http://arxiv.org/pdf/2509.13888v1](http://arxiv.org/pdf/2509.13888v1)**

> **作者:** Mariano Barone; Antonio Romano; Giuseppe Riccio; Marco Postiglione; Vincenzo Moscato
>
> **摘要:** Misinformation in healthcare, from vaccine hesitancy to unproven treatments, poses risks to public health and trust in medical systems. While machine learning and natural language processing have advanced automated fact-checking, validating biomedical claims remains uniquely challenging due to complex terminology, the need for domain expertise, and the critical importance of grounding in scientific evidence. We introduce CER (Combining Evidence and Reasoning), a novel framework for biomedical fact-checking that integrates scientific evidence retrieval, reasoning via large language models, and supervised veracity prediction. By integrating the text-generation capabilities of large language models with advanced retrieval techniques for high-quality biomedical scientific evidence, CER effectively mitigates the risk of hallucinations, ensuring that generated outputs are grounded in verifiable, evidence-based sources. Evaluations on expert-annotated datasets (HealthFC, BioASQ-7b, SciFact) demonstrate state-of-the-art performance and promising cross-dataset generalization. Code and data are released for transparency and reproducibility: https://github.com/PRAISELab-PicusLab/CER
>
---
#### [new 017] AgentCTG: Harnessing Multi-Agent Collaboration for Fine-Grained Precise Control in Text Generation
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于可控文本生成（CTG）任务，旨在解决细粒度条件控制难题。提出AgentCTG框架，通过多智能体协作与自动提示模块提升生成精度与效果，并在新任务中验证其实际应用价值。**

- **链接: [http://arxiv.org/pdf/2509.13677v1](http://arxiv.org/pdf/2509.13677v1)**

> **作者:** Xinxu Zhou; Jiaqi Bai; Zhenqi Sun; Fanxiang Zeng; Yue Liu
>
> **摘要:** Although significant progress has been made in many tasks within the field of Natural Language Processing (NLP), Controlled Text Generation (CTG) continues to face numerous challenges, particularly in achieving fine-grained conditional control over generation. Additionally, in real scenario and online applications, cost considerations, scalability, domain knowledge learning and more precise control are required, presenting more challenge for CTG. This paper introduces a novel and scalable framework, AgentCTG, which aims to enhance precise and complex control over the text generation by simulating the control and regulation mechanisms in multi-agent workflows. We explore various collaboration methods among different agents and introduce an auto-prompt module to further enhance the generation effectiveness. AgentCTG achieves state-of-the-art results on multiple public datasets. To validate its effectiveness in practical applications, we propose a new challenging Character-Driven Rewriting task, which aims to convert the original text into new text that conform to specific character profiles and simultaneously preserve the domain knowledge. When applied to online navigation with role-playing, our approach significantly enhances the driving experience through improved content delivery. By optimizing the generation of contextually relevant text, we enable a more immersive interaction within online communities, fostering greater personalization and user engagement.
>
---
#### [new 018] Geometric Uncertainty for Detecting and Correcting Hallucinations in LLMs
- **分类: cs.CL**

- **简介: 该论文提出一种几何框架，用于检测和修正大语言模型的幻觉问题。通过分析响应嵌入的凸包体积和个体响应的可靠性，分别实现全局和局部不确定性估计，提升答案可信度。属于自然语言处理中的不确定性量化任务。**

- **链接: [http://arxiv.org/pdf/2509.13813v1](http://arxiv.org/pdf/2509.13813v1)**

> **作者:** Edward Phillips; Sean Wu; Soheila Molaei; Danielle Belgrave; Anshul Thakur; David Clifton
>
> **摘要:** Large language models demonstrate impressive results across diverse tasks but are still known to hallucinate, generating linguistically plausible but incorrect answers to questions. Uncertainty quantification has been proposed as a strategy for hallucination detection, but no existing black-box approach provides estimates for both global and local uncertainty. The former attributes uncertainty to a batch of responses, while the latter attributes uncertainty to individual responses. Current local methods typically rely on white-box access to internal model states, whilst black-box methods only provide global uncertainty estimates. We introduce a geometric framework to address this, based on archetypal analysis of batches of responses sampled with only black-box model access. At the global level, we propose Geometric Volume, which measures the convex hull volume of archetypes derived from response embeddings. At the local level, we propose Geometric Suspicion, which ranks responses by reliability and enables hallucination reduction through preferential response selection. Unlike prior dispersion methods which yield only a single global score, our approach provides semantic boundary points which have utility for attributing reliability to individual responses. Experiments show that our framework performs comparably to or better than prior methods on short form question-answering datasets, and achieves superior results on medical datasets where hallucinations carry particularly critical risks. We also provide theoretical justification by proving a link between convex hull volume and entropy.
>
---
#### [new 019] Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and High-Performance Models for Multilingual ASR and AST
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出Canary-1B-v2和Parakeet-TDT-0.6B-v3，用于多语言语音识别（ASR）和语音翻译（AST）。模型采用FastConformer和Transformer结构，通过两阶段预训练与动态数据平衡优化性能，实现高效、高精度的多语言处理。**

- **链接: [http://arxiv.org/pdf/2509.14128v1](http://arxiv.org/pdf/2509.14128v1)**

> **作者:** Monica Sekoyan; Nithin Rao Koluguri; Nune Tadevosyan; Piotr Zelasko; Travis Bartley; Nick Karpov; Jagadeesh Balam; Boris Ginsburg
>
> **备注:** Mini Version of it Submitted to ICASSP 2026
>
> **摘要:** This report introduces Canary-1B-v2, a fast, robust multilingual model for Automatic Speech Recognition (ASR) and Speech-to-Text Translation (AST). Built with a FastConformer encoder and Transformer decoder, it supports 25 languages primarily European. The model was trained on 1.7M hours of total data samples, including Granary and NeMo ASR Set 3.0, with non-speech audio added to reduce hallucinations for ASR and AST. We describe its two-stage pre-training and fine-tuning process with dynamic data balancing, as well as experiments with an nGPT encoder. Results show nGPT scales well with massive data, while FastConformer excels after fine-tuning. For timestamps, Canary-1B-v2 uses the NeMo Forced Aligner (NFA) with an auxiliary CTC model, providing reliable segment-level timestamps for ASR and AST. Evaluations show Canary-1B-v2 outperforms Whisper-large-v3 on English ASR while being 10x faster, and delivers competitive multilingual ASR and AST performance against larger models like Seamless-M4T-v2-large and LLM-based systems. We also release Parakeet-TDT-0.6B-v3, a successor to v2, offering multilingual ASR across the same 25 languages with just 600M parameters.
>
---
#### [new 020] Large Language Models Discriminate Against Speakers of German Dialects
- **分类: cs.CL**

- **简介: 论文研究大语言模型对德语方言使用者的偏见。通过关联任务和决策任务，发现LLMs存在显著的方言命名和使用偏见，并构建新语料库进行评估。任务旨在揭示语言模型是否反映社会对方言使用者的刻板印象。**

- **链接: [http://arxiv.org/pdf/2509.13835v1](http://arxiv.org/pdf/2509.13835v1)**

> **作者:** Minh Duc Bui; Carolin Holtermann; Valentin Hofmann; Anne Lauscher; Katharina von der Wense
>
> **备注:** Accepted to EMNLP 2025 Main
>
> **摘要:** Dialects represent a significant component of human culture and are found across all regions of the world. In Germany, more than 40% of the population speaks a regional dialect (Adler and Hansen, 2022). However, despite cultural importance, individuals speaking dialects often face negative societal stereotypes. We examine whether such stereotypes are mirrored by large language models (LLMs). We draw on the sociolinguistic literature on dialect perception to analyze traits commonly associated with dialect speakers. Based on these traits, we assess the dialect naming bias and dialect usage bias expressed by LLMs in two tasks: an association task and a decision task. To assess a model's dialect usage bias, we construct a novel evaluation corpus that pairs sentences from seven regional German dialects (e.g., Alemannic and Bavarian) with their standard German counterparts. We find that: (1) in the association task, all evaluated LLMs exhibit significant dialect naming and dialect usage bias against German dialect speakers, reflected in negative adjective associations; (2) all models reproduce these dialect naming and dialect usage biases in their decision making; and (3) contrary to prior work showing minimal bias with explicit demographic mentions, we find that explicitly labeling linguistic demographics--German dialect speakers--amplifies bias more than implicit cues like dialect usage.
>
---
#### [new 021] CL$^2$GEC: A Multi-Discipline Benchmark for Continual Learning in Chinese Literature Grammatical Error Correction
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CL²GEC，用于评估中文语法纠错在多学科持续学习中的表现。任务是跨学科适应性语法纠错，解决领域差异与遗忘问题。构建了包含10,000句的基准数据集，并测试多种持续学习方法，发现正则化方法更有效。**

- **链接: [http://arxiv.org/pdf/2509.13672v1](http://arxiv.org/pdf/2509.13672v1)**

> **作者:** Shang Qin; Jingheng Ye; Yinghui Li; Hai-Tao Zheng; Qi Li; Jinxiao Shan; Zhixing Li; Hong-Gee Kim
>
> **摘要:** The growing demand for automated writing assistance in diverse academic domains highlights the need for robust Chinese Grammatical Error Correction (CGEC) systems that can adapt across disciplines. However, existing CGEC research largely lacks dedicated benchmarks for multi-disciplinary academic writing, overlooking continual learning (CL) as a promising solution to handle domain-specific linguistic variation and prevent catastrophic forgetting. To fill this crucial gap, we introduce CL$^2$GEC, the first Continual Learning benchmark for Chinese Literature Grammatical Error Correction, designed to evaluate adaptive CGEC across multiple academic fields. Our benchmark includes 10,000 human-annotated sentences spanning 10 disciplines, each exhibiting distinct linguistic styles and error patterns. CL$^2$GEC focuses on evaluating grammatical error correction in a continual learning setting, simulating sequential exposure to diverse academic disciplines to reflect real-world editorial dynamics. We evaluate large language models under sequential tuning, parameter-efficient adaptation, and four representative CL algorithms, using both standard GEC metrics and continual learning metrics adapted to task-level variation. Experimental results reveal that regularization-based methods mitigate forgetting more effectively than replay-based or naive sequential approaches. Our benchmark provides a rigorous foundation for future research in adaptive grammatical error correction across diverse academic domains.
>
---
#### [new 022] Audio-Based Crowd-Sourced Evaluation of Machine Translation Quality
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于机器翻译质量评估任务，旨在解决传统文本评估无法反映语音场景的问题。通过亚马逊众包平台，比较了音频与文本评估的差异，验证音频评估的有效性并提出将其纳入未来评估框架。**

- **链接: [http://arxiv.org/pdf/2509.14023v1](http://arxiv.org/pdf/2509.14023v1)**

> **作者:** Sami Ul Haq; Sheila Castilho; Yvette Graham
>
> **备注:** Accepted at WMT2025 (ENNLP) for oral presented
>
> **摘要:** Machine Translation (MT) has achieved remarkable performance, with growing interest in speech translation and multimodal approaches. However, despite these advancements, MT quality assessment remains largely text centric, typically relying on human experts who read and compare texts. Since many real-world MT applications (e.g Google Translate Voice Mode, iFLYTEK Translator) involve translation being spoken rather printed or read, a more natural way to assess translation quality would be through speech as opposed text-only evaluations. This study compares text-only and audio-based evaluations of 10 MT systems from the WMT General MT Shared Task, using crowd-sourced judgments collected via Amazon Mechanical Turk. We additionally, performed statistical significance testing and self-replication experiments to test reliability and consistency of audio-based approach. Crowd-sourced assessments based on audio yield rankings largely consistent with text only evaluations but, in some cases, identify significant differences between translation systems. We attribute this to speech richer, more natural modality and propose incorporating speech-based assessments into future MT evaluation frameworks.
>
---
#### [new 023] Hala Technical Report: Building Arabic-Centric Instruction & Translation Models at Scale
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Hala模型，专注于阿拉伯语指令与翻译任务。通过压缩双语教师模型生成高质量监督数据，训练多参数规模的阿拉伯语模型，在基准测试中取得SOTA结果，推动阿拉伯语NLP研究。**

- **链接: [http://arxiv.org/pdf/2509.14008v1](http://arxiv.org/pdf/2509.14008v1)**

> **作者:** Hasan Abed Al Kader Hammoud; Mohammad Zbeeb; Bernard Ghanem
>
> **备注:** Technical Report
>
> **摘要:** We present Hala, a family of Arabic-centric instruction and translation models built with our translate-and-tune pipeline. We first compress a strong AR$\leftrightarrow$EN teacher to FP8 (yielding $\sim$2$\times$ higher throughput with no quality loss) and use it to create high-fidelity bilingual supervision. A lightweight language model LFM2-1.2B is then fine-tuned on this data and used to translate high-quality English instruction sets into Arabic, producing a million-scale corpus tailored to instruction following. We train Hala models at 350M, 700M, 1.2B, and 9B parameters, and apply slerp merging to balance Arabic specialization with base-model strengths. On Arabic-centric benchmarks, Hala achieves state-of-the-art results within both the "nano" ($\leq$2B) and "small" (7-9B) categories, outperforming their bases. We release models, data, evaluation, and recipes to accelerate research in Arabic NLP.
>
---
#### [new 024] Enhancing Multi-Agent Debate System Performance via Confidence Expression
- **分类: cs.CL**

- **简介: 论文提出ConfMAD框架，通过引入置信度表达提升多智能体辩论系统的性能。该任务旨在解决LLMs在辩论中无法有效传达自身置信度的问题，从而改善辩论效果和系统整体表现。**

- **链接: [http://arxiv.org/pdf/2509.14034v1](http://arxiv.org/pdf/2509.14034v1)**

> **作者:** Zijie Lin; Bryan Hooi
>
> **备注:** EMNLP'25 Findings
>
> **摘要:** Generative Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of tasks. Recent research has introduced Multi-Agent Debate (MAD) systems, which leverage multiple LLMs to simulate human debate and thereby improve task performance. However, while some LLMs may possess superior knowledge or reasoning capabilities for specific tasks, they often struggle to clearly communicate this advantage during debates, in part due to a lack of confidence expression. Moreover, inappropriate confidence expression can cause agents in MAD systems to either stubbornly maintain incorrect beliefs or converge prematurely on suboptimal answers, ultimately reducing debate effectiveness and overall system performance. To address these challenges, we propose incorporating confidence expression into MAD systems to allow LLMs to explicitly communicate their confidence levels. To validate this approach, we develop ConfMAD, a MAD framework that integrates confidence expression throughout the debate process. Experimental results demonstrate the effectiveness of our method, and we further analyze how confidence influences debate dynamics, offering insights into the design of confidence-aware MAD systems.
>
---
#### [new 025] AssoCiAm: A Benchmark for Evaluating Association Thinking while Circumventing Ambiguity
- **分类: cs.CL**

- **简介: 该论文提出AssoCiAm基准，用于评估多模态大语言模型的联想能力，解决评估中因歧义导致的可靠性问题。通过分解内部与外部歧义，设计混合计算方法，提升评估准确性与可靠性。**

- **链接: [http://arxiv.org/pdf/2509.14171v1](http://arxiv.org/pdf/2509.14171v1)**

> **作者:** Yifan Liu; Wenkuan Zhao; Shanshan Zhong; Jinghui Qin; Mingfu Liang; Zhongzhan Huang; Wushao Wen
>
> **摘要:** Recent advancements in multimodal large language models (MLLMs) have garnered significant attention, offering a promising pathway toward artificial general intelligence (AGI). Among the essential capabilities required for AGI, creativity has emerged as a critical trait for MLLMs, with association serving as its foundation. Association reflects a model' s ability to think creatively, making it vital to evaluate and understand. While several frameworks have been proposed to assess associative ability, they often overlook the inherent ambiguity in association tasks, which arises from the divergent nature of associations and undermines the reliability of evaluations. To address this issue, we decompose ambiguity into two types-internal ambiguity and external ambiguity-and introduce AssoCiAm, a benchmark designed to evaluate associative ability while circumventing the ambiguity through a hybrid computational method. We then conduct extensive experiments on MLLMs, revealing a strong positive correlation between cognition and association. Additionally, we observe that the presence of ambiguity in the evaluation process causes MLLMs' behavior to become more random-like. Finally, we validate the effectiveness of our method in ensuring more accurate and reliable evaluations. See Project Page for the data and codes.
>
---
#### [new 026] Improving Context Fidelity via Native Retrieval-Augmented Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 论文提出CARE框架，解决大语言模型在回答问题时上下文保真度不足的问题。通过引入检索增强推理机制，使模型更有效地利用给定上下文信息，提升答案准确性和可靠性，属于问答任务中的检索增强生成方向。**

- **链接: [http://arxiv.org/pdf/2509.13683v1](http://arxiv.org/pdf/2509.13683v1)**

> **作者:** Suyuchen Wang; Jinlin Wang; Xinyu Wang; Shiqi Li; Xiangru Tang; Sirui Hong; Xiao-Wen Chang; Chenglin Wu; Bang Liu
>
> **备注:** Accepted as a main conference paper at EMNLP 2025
>
> **摘要:** Large language models (LLMs) often struggle with context fidelity, producing inconsistent answers when responding to questions based on provided information. Existing approaches either rely on expensive supervised fine-tuning to generate evidence post-answer or train models to perform web searches without necessarily improving utilization of the given context. We propose CARE, a novel native retrieval-augmented reasoning framework that teaches LLMs to explicitly integrate in-context evidence within their reasoning process with the model's own retrieval capabilities. Our method requires limited labeled evidence data while significantly enhancing both retrieval accuracy and answer generation performance through strategically retrieved in-context tokens in the reasoning chain. Extensive experiments on multiple real-world and counterfactual QA benchmarks demonstrate that our approach substantially outperforms supervised fine-tuning, traditional retrieval-augmented generation methods, and external retrieval solutions. This work represents a fundamental advancement in making LLMs more accurate, reliable, and efficient for knowledge-intensive tasks.
>
---
#### [new 027] Sparse Neurons Carry Strong Signals of Question Ambiguity in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLMs）如何在内部表示问题歧义，并提出通过少量神经元检测和控制歧义响应。任务为歧义检测与行为控制，方法包括识别歧义编码神经元（AENs）并实现行为干预。**

- **链接: [http://arxiv.org/pdf/2509.13664v1](http://arxiv.org/pdf/2509.13664v1)**

> **作者:** Zhuoxuan Zhang; Jinhao Duan; Edward Kim; Kaidi Xu
>
> **备注:** To be appeared in EMNLP 2025 (main)
>
> **摘要:** Ambiguity is pervasive in real-world questions, yet large language models (LLMs) often respond with confident answers rather than seeking clarification. In this work, we show that question ambiguity is linearly encoded in the internal representations of LLMs and can be both detected and controlled at the neuron level. During the model's pre-filling stage, we identify that a small number of neurons, as few as one, encode question ambiguity information. Probes trained on these Ambiguity-Encoding Neurons (AENs) achieve strong performance on ambiguity detection and generalize across datasets, outperforming prompting-based and representation-based baselines. Layerwise analysis reveals that AENs emerge from shallow layers, suggesting early encoding of ambiguity signals in the model's processing pipeline. Finally, we show that through manipulating AENs, we can control LLM's behavior from direct answering to abstention. Our findings reveal that LLMs form compact internal representations of question ambiguity, enabling interpretable and controllable behavior.
>
---
#### [new 028] Combining Evidence and Reasoning for Biomedical Fact-Checking
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 论文提出CER框架，用于生物医学事实核查任务，解决复杂术语、领域专业知识和科学证据依赖等问题。整合证据检索与大模型推理，提升核查准确性与可信度。**

- **链接: [http://arxiv.org/pdf/2509.13879v1](http://arxiv.org/pdf/2509.13879v1)**

> **作者:** Mariano Barone; Antonio Romano; Giuseppe Riccio; Marco Postiglione; Vincenzo Moscato
>
> **备注:** Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2025
>
> **摘要:** Misinformation in healthcare, from vaccine hesitancy to unproven treatments, poses risks to public health and trust in medical systems. While machine learning and natural language processing have advanced automated fact-checking, validating biomedical claims remains uniquely challenging due to complex terminology, the need for domain expertise, and the critical importance of grounding in scientific evidence. We introduce CER (Combining Evidence and Reasoning), a novel framework for biomedical fact-checking that integrates scientific evidence retrieval, reasoning via large language models, and supervised veracity prediction. By integrating the text-generation capabilities of large language models with advanced retrieval techniques for high-quality biomedical scientific evidence, CER effectively mitigates the risk of hallucinations, ensuring that generated outputs are grounded in verifiable, evidence-based sources. Evaluations on expert-annotated datasets (HealthFC, BioASQ-7b, SciFact) demonstrate state-of-the-art performance and promising cross-dataset generalization. Code and data are released for transparency and reproducibility: https: //github.com/PRAISELab-PicusLab/CER.
>
---
#### [new 029] Linguistic Nepotism: Trading-off Quality for Language Preference in Multilingual RAG
- **分类: cs.CL**

- **简介: 论文研究多语言检索增强生成（mRAG）系统中语言偏好对引用行为的影响。通过控制变量方法，发现模型在回答时更倾向引用英语资料，尤其在低资源语言和中段上下文中。该工作揭示了语言偏好可能影响引用质量，而非仅基于信息相关性。**

- **链接: [http://arxiv.org/pdf/2509.13930v1](http://arxiv.org/pdf/2509.13930v1)**

> **作者:** Dayeon Ki; Marine Carpuat; Paul McNamee; Daniel Khashabi; Eugene Yang; Dawn Lawrie; Kevin Duh
>
> **备注:** 33 pages, 20 figures
>
> **摘要:** Multilingual Retrieval-Augmented Generation (mRAG) systems enable language models to answer knowledge-intensive queries with citation-supported responses across languages. While such systems have been proposed, an open questions is whether the mixture of different document languages impacts generation and citation in unintended ways. To investigate, we introduce a controlled methodology using model internals to measure language preference while holding other factors such as document relevance constant. Across eight languages and six open-weight models, we find that models preferentially cite English sources when queries are in English, with this bias amplified for lower-resource languages and for documents positioned mid-context. Crucially, we find that models sometimes trade-off document relevance for language preference, indicating that citation choices are not always driven by informativeness alone. Our findings shed light on how language models leverage multilingual context and influence citation behavior.
>
---
#### [new 030] CS-FLEURS: A Massively Multilingual and Code-Switched Speech Dataset
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出CS-FLEURS数据集，用于开发和评估跨多种语言的代码切换语音识别与翻译系统。数据集包含4个测试集和训练集，覆盖113种代码切换语言对，旨在推动低资源语言的代码切换语音研究。**

- **链接: [http://arxiv.org/pdf/2509.14161v1](http://arxiv.org/pdf/2509.14161v1)**

> **作者:** Brian Yan; Injy Hamed; Shuichiro Shimizu; Vasista Lodagala; William Chen; Olga Iakovenko; Bashar Talafha; Amir Hussein; Alexander Polok; Kalvin Chang; Dominik Klement; Sara Althubaiti; Puyuan Peng; Matthew Wiesner; Thamar Solorio; Ahmed Ali; Sanjeev Khudanpur; Shinji Watanabe; Chih-Chen Chen; Zhen Wu; Karim Benharrak; Anuj Diwan; Samuele Cornell; Eunjung Yeo; Kwanghee Choi; Carlos Carvalho; Karen Rosero
>
> **摘要:** We present CS-FLEURS, a new dataset for developing and evaluating code-switched speech recognition and translation systems beyond high-resourced languages. CS-FLEURS consists of 4 test sets which cover in total 113 unique code-switched language pairs across 52 languages: 1) a 14 X-English language pair set with real voices reading synthetically generated code-switched sentences, 2) a 16 X-English language pair set with generative text-to-speech 3) a 60 {Arabic, Mandarin, Hindi, Spanish}-X language pair set with the generative text-to-speech, and 4) a 45 X-English lower-resourced language pair test set with concatenative text-to-speech. Besides the four test sets, CS-FLEURS also provides a training set with 128 hours of generative text-to-speech data across 16 X-English language pairs. Our hope is that CS-FLEURS helps to broaden the scope of future code-switched speech research. Dataset link: https://huggingface.co/datasets/byan/cs-fleurs.
>
---
#### [new 031] Overview of Dialog System Evaluation Track: Dimensionality, Language, Culture and Safety at DSTC 12
- **分类: cs.CL**

- **简介: 该论文介绍DSTC12对话系统评估任务，旨在解决多维评价与跨语言文化安全检测问题。任务包含两个子任务：多维自动评估和多语言多文化安全检测，并提供了数据集与基线模型供参与团队使用。**

- **链接: [http://arxiv.org/pdf/2509.13569v1](http://arxiv.org/pdf/2509.13569v1)**

> **作者:** John Mendonça; Lining Zhang; Rahul Mallidi; Alon Lavie; Isabel Trancoso; Luis Fernando D'Haro; João Sedoc
>
> **备注:** DSTC12 Track 1 Overview Paper. https://chateval.org/dstc12
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has intensified the need for robust dialogue system evaluation, yet comprehensive assessment remains challenging. Traditional metrics often prove insufficient, and safety considerations are frequently narrowly defined or culturally biased. The DSTC12 Track 1, "Dialog System Evaluation: Dimensionality, Language, Culture and Safety," is part of the ongoing effort to address these critical gaps. The track comprised two subtasks: (1) Dialogue-level, Multi-dimensional Automatic Evaluation Metrics, and (2) Multilingual and Multicultural Safety Detection. For Task 1, focused on 10 dialogue dimensions, a Llama-3-8B baseline achieved the highest average Spearman's correlation (0.1681), indicating substantial room for improvement. In Task 2, while participating teams significantly outperformed a Llama-Guard-3-1B baseline on the multilingual safety subset (top ROC-AUC 0.9648), the baseline proved superior on the cultural subset (0.5126 ROC-AUC), highlighting critical needs in culturally-aware safety. This paper describes the datasets and baselines provided to participants, as well as submission evaluation results for each of the two proposed subtasks.
>
---
#### [new 032] Measuring Gender Bias in Job Title Matching for Grammatical Gender Languages
- **分类: cs.CL**

- **简介: 该论文研究语法性别语言中职位名称匹配的性别偏见问题，提出使用RBO指标评估排名系统中的性别偏差，并构建了四个语法性别语言的测试集，用于评估多语言模型的性别偏见程度。**

- **链接: [http://arxiv.org/pdf/2509.13803v1](http://arxiv.org/pdf/2509.13803v1)**

> **作者:** Laura García-Sardiña; Hermenegildo Fabregat; Daniel Deniz; Rabih Zbib
>
> **摘要:** This work sets the ground for studying how explicit grammatical gender assignment in job titles can affect the results of automatic job ranking systems. We propose the usage of metrics for ranking comparison controlling for gender to evaluate gender bias in job title ranking systems, in particular RBO (Rank-Biased Overlap). We generate and share test sets for a job title matching task in four grammatical gender languages, including occupations in masculine and feminine form and annotated by gender and matching relevance. We use the new test sets and the proposed methodology to evaluate the gender bias of several out-of-the-box multilingual models to set as baselines, showing that all of them exhibit varying degrees of gender bias.
>
---
#### [new 033] Implementing a Logical Inference System for Japanese Comparatives
- **分类: cs.CL**

- **简介: 该论文属于自然语言推理任务，旨在解决日语比较句的逻辑推理问题。提出基于组合语义的逻辑推理系统ccg-jcomp，用于处理日语比较表达，并在日语NLI数据集上验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.13734v1](http://arxiv.org/pdf/2509.13734v1)**

> **作者:** Yosuke Mikami; Daiki Matsuoka; Hitomi Yanaka
>
> **备注:** In Proceedings of the 5th Workshop on Natural Logic Meets Machine Learning (NALOMA)
>
> **摘要:** Natural Language Inference (NLI) involving comparatives is challenging because it requires understanding quantities and comparative relations expressed by sentences. While some approaches leverage Large Language Models (LLMs), we focus on logic-based approaches grounded in compositional semantics, which are promising for robust handling of numerical and logical expressions. Previous studies along these lines have proposed logical inference systems for English comparatives. However, it has been pointed out that there are several morphological and semantic differences between Japanese and English comparatives. These differences make it difficult to apply such systems directly to Japanese comparatives. To address this gap, this study proposes ccg-jcomp, a logical inference system for Japanese comparatives based on compositional semantics. We evaluate the proposed system on a Japanese NLI dataset containing comparative expressions. We demonstrate the effectiveness of our system by comparing its accuracy with that of existing LLMs.
>
---
#### [new 034] Long-context Reference-based MT Quality Estimation
- **分类: cs.CL; cs.LG**

- **简介: 论文属于机器翻译质量评估任务，旨在提升模型对翻译质量的预测能力。研究基于COMET框架，利用长上下文数据训练模型，通过整合多个人类判断数据集，提高与人类评分的相关性。**

- **链接: [http://arxiv.org/pdf/2509.13980v1](http://arxiv.org/pdf/2509.13980v1)**

> **作者:** Sami Ul Haq; Chinonso Cynthia Osuji; Sheila Castilho; Brian Davis
>
> **摘要:** In this paper, we present our submission to the Tenth Conference on Machine Translation (WMT25) Shared Task on Automated Translation Quality Evaluation. Our systems are built upon the COMET framework and trained to predict segment-level Error Span Annotation (ESA) scores using augmented long-context data. To construct long-context training data, we concatenate in-domain, human-annotated sentences and compute a weighted average of their scores. We integrate multiple human judgment datasets (MQM, SQM, and DA) by normalising their scales and train multilingual regression models to predict quality scores from the source, hypothesis, and reference translations. Experimental results show that incorporating long-context information improves correlations with human judgments compared to models trained only on short segments.
>
---
#### [new 035] Early Stopping Chain-of-thoughts in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出ES-CoT方法，在推理阶段通过检测答案收敛提前终止链式思维生成，减少推理成本。属于自然语言处理中的高效推理任务，解决LLM生成长CoT带来的高计算开销问题。**

- **链接: [http://arxiv.org/pdf/2509.14004v1](http://arxiv.org/pdf/2509.14004v1)**

> **作者:** Minjia Mao; Bowen Yin; Yu Zhu; Xiao Fang
>
> **摘要:** Reasoning large language models (LLMs) have demonstrated superior capacities in solving complicated problems by generating long chain-of-thoughts (CoT), but such a lengthy CoT incurs high inference costs. In this study, we introduce ES-CoT, an inference-time method that shortens CoT generation by detecting answer convergence and stopping early with minimal performance loss. At the end of each reasoning step, we prompt the LLM to output its current final answer, denoted as a step answer. We then track the run length of consecutive identical step answers as a measure of answer convergence. Once the run length exhibits a sharp increase and exceeds a minimum threshold, the generation is terminated. We provide both empirical and theoretical support for this heuristic: step answers steadily converge to the final answer, and large run-length jumps reliably mark this convergence. Experiments on five reasoning datasets across three LLMs show that ES-CoT reduces the number of inference tokens by about 41\% on average while maintaining accuracy comparable to standard CoT. Further, ES-CoT integrates seamlessly with self-consistency prompting and remains robust across hyperparameter choices, highlighting it as a practical and effective approach for efficient reasoning.
>
---
#### [new 036] Framing Migration: A Computational Analysis of UK Parliamentary Discourse
- **分类: cs.CL; cs.CY**

- **简介: 该论文通过计算方法分析英国议会75年移民相关辩论话语，比较美国国会讨论，利用大语言模型标注立场与叙事框架，揭示英国内政党态度趋同及叙事从整合转向安全化趋势，解决政治话语分析的可扩展性问题。**

- **链接: [http://arxiv.org/pdf/2509.14197v1](http://arxiv.org/pdf/2509.14197v1)**

> **作者:** Vahid Ghafouri; Robert McNeil; Teodor Yankov; Madeleine Sumption; Luc Rocher; Scott A. Hale; Adam Mahdi
>
> **摘要:** We present a large-scale computational analysis of migration-related discourse in UK parliamentary debates spanning over 75 years and compare it with US congressional discourse. Using open-weight LLMs, we annotate each statement with high-level stances toward migrants and track the net tone toward migrants across time and political parties. For the UK, we extend this with a semi-automated framework for extracting fine-grained narrative frames to capture nuances of migration discourse. Our findings show that, while US discourse has grown increasingly polarised, UK parliamentary attitudes remain relatively aligned across parties, with a persistent ideological gap between Labour and the Conservatives, reaching its most negative level in 2025. The analysis of narrative frames in the UK parliamentary statements reveals a shift toward securitised narratives such as border control and illegal immigration, while longer-term integration-oriented frames such as social integration have declined. Moreover, discussions of national law about immigration have been replaced over time by international law and human rights, revealing nuances in discourse trends. Taken together broadly, our findings demonstrate how LLMs can support scalable, fine-grained discourse analysis in political and historical contexts.
>
---
#### [new 037] Do LLMs Align Human Values Regarding Social Biases? Judging and Explaining Social Biases with LLMs
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLMs）在不同社会偏见场景下与人类价值观的对齐情况。通过分析12个LLMs，发现模型参数规模并不直接影响对齐效果，并探讨了模型解释能力及小型模型的解释性能。任务是评估LLMs在社会偏见判断中的价值观对齐问题。**

- **链接: [http://arxiv.org/pdf/2509.13869v1](http://arxiv.org/pdf/2509.13869v1)**

> **作者:** Yang Liu; Chenhui Chu
>
> **备注:** 38 pages, 31 figures
>
> **摘要:** Large language models (LLMs) can lead to undesired consequences when misaligned with human values, especially in scenarios involving complex and sensitive social biases. Previous studies have revealed the misalignment of LLMs with human values using expert-designed or agent-based emulated bias scenarios. However, it remains unclear whether the alignment of LLMs with human values differs across different types of scenarios (e.g., scenarios containing negative vs. non-negative questions). In this study, we investigate the alignment of LLMs with human values regarding social biases (HVSB) in different types of bias scenarios. Through extensive analysis of 12 LLMs from four model families and four datasets, we demonstrate that LLMs with large model parameter scales do not necessarily have lower misalignment rate and attack success rate. Moreover, LLMs show a certain degree of alignment preference for specific types of scenarios and the LLMs from the same model family tend to have higher judgment consistency. In addition, we study the understanding capacity of LLMs with their explanations of HVSB. We find no significant differences in the understanding of HVSB across LLMs. We also find LLMs prefer their own generated explanations. Additionally, we endow smaller language models (LMs) with the ability to explain HVSB. The generation results show that the explanations generated by the fine-tuned smaller LMs are more readable, but have a relatively lower model agreeability.
>
---
#### [new 038] Gender-Neutral Rewriting in Italian: Models, Approaches, and Trade-offs
- **分类: cs.CL**

- **简介: 该论文研究意大利语性别中立重写任务，旨在消除文本中的性别指定同时保持原意。论文评估了多个大语言模型，提出双维度评价框架，并通过微调提升性能，探讨了中立性与语义保真度的权衡。**

- **链接: [http://arxiv.org/pdf/2509.13480v1](http://arxiv.org/pdf/2509.13480v1)**

> **作者:** Andrea Piergentili; Beatrice Savoldi; Matteo Negri; Luisa Bentivogli
>
> **备注:** Accepted at CLiC-it 2025
>
> **摘要:** Gender-neutral rewriting (GNR) aims to reformulate text to eliminate unnecessary gender specifications while preserving meaning, a particularly challenging task in grammatical-gender languages like Italian. In this work, we conduct the first systematic evaluation of state-of-the-art large language models (LLMs) for Italian GNR, introducing a two-dimensional framework that measures both neutrality and semantic fidelity to the input. We compare few-shot prompting across multiple LLMs, fine-tune selected models, and apply targeted cleaning to boost task relevance. Our findings show that open-weight LLMs outperform the only existing model dedicated to GNR in Italian, whereas our fine-tuned models match or exceed the best open-weight LLM's performance at a fraction of its size. Finally, we discuss the trade-off between optimizing the training data for neutrality and meaning preservation.
>
---
#### [new 039] Automated Triaging and Transfer Learning of Incident Learning Safety Reports Using Large Language Representational Models
- **分类: cs.CL; cs.AI**

- **简介: 论文提出一种基于NLP的工具，用于自动识别高严重性医疗事件报告。任务是利用大型语言模型进行跨机构的分类与迁移学习，解决人工审核耗时且依赖专业知识的问题，通过训练SVM和BlueBERT模型实现高效检测。**

- **链接: [http://arxiv.org/pdf/2509.13706v1](http://arxiv.org/pdf/2509.13706v1)**

> **作者:** Peter Beidler; Mark Nguyen; Kevin Lybarger; Ola Holmberg; Eric Ford; John Kang
>
> **摘要:** PURPOSE: Incident reports are an important tool for safety and quality improvement in healthcare, but manual review is time-consuming and requires subject matter expertise. Here we present a natural language processing (NLP) screening tool to detect high-severity incident reports in radiation oncology across two institutions. METHODS AND MATERIALS: We used two text datasets to train and evaluate our NLP models: 7,094 reports from our institution (Inst.), and 571 from IAEA SAFRON (SF), all of which had severity scores labeled by clinical content experts. We trained and evaluated two types of models: baseline support vector machines (SVM) and BlueBERT which is a large language model pretrained on PubMed abstracts and hospitalized patient data. We assessed for generalizability of our model in two ways. First, we evaluated models trained using Inst.-train on SF-test. Second, we trained a BlueBERT_TRANSFER model that was first fine-tuned on Inst.-train then on SF-train before testing on SF-test set. To further analyze model performance, we also examined a subset of 59 reports from our Inst. dataset, which were manually edited for clarity. RESULTS Classification performance on the Inst. test achieved AUROC 0.82 using SVM and 0.81 using BlueBERT. Without cross-institution transfer learning, performance on the SF test was limited to an AUROC of 0.42 using SVM and 0.56 using BlueBERT. BlueBERT_TRANSFER, which was fine-tuned on both datasets, improved the performance on SF test to AUROC 0.78. Performance of SVM, and BlueBERT_TRANSFER models on the manually curated Inst. reports (AUROC 0.85 and 0.74) was similar to human performance (AUROC 0.81). CONCLUSION: In summary, we successfully developed cross-institution NLP models on incident report text from radiation oncology centers. These models were able to detect high-severity reports similarly to humans on a curated dataset.
>
---
#### [new 040] Dense Video Understanding with Gated Residual Tokenization
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视频理解任务，解决高帧率视频处理中冗余计算与信息丢失问题。提出DVU框架与GRT方法，通过运动补偿与语义融合减少token数量，提升效率，并构建DIVE基准测试密集时序推理能力。**

- **链接: [http://arxiv.org/pdf/2509.14199v1](http://arxiv.org/pdf/2509.14199v1)**

> **作者:** Haichao Zhang; Wenhao Chai; Shwai He; Ang Li; Yun Fu
>
> **摘要:** High temporal resolution is essential for capturing fine-grained details in video understanding. However, current video large language models (VLLMs) and benchmarks mostly rely on low-frame-rate sampling, such as uniform sampling or keyframe selection, discarding dense temporal information. This compromise avoids the high cost of tokenizing every frame, which otherwise leads to redundant computation and linear token growth as video length increases. While this trade-off works for slowly changing content, it fails for tasks like lecture comprehension, where information appears in nearly every frame and requires precise temporal alignment. To address this gap, we introduce Dense Video Understanding (DVU), which enables high-FPS video comprehension by reducing both tokenization time and token overhead. Existing benchmarks are also limited, as their QA pairs focus on coarse content changes. We therefore propose DIVE (Dense Information Video Evaluation), the first benchmark designed for dense temporal reasoning. To make DVU practical, we present Gated Residual Tokenization (GRT), a two-stage framework: (1) Motion-Compensated Inter-Gated Tokenization uses pixel-level motion estimation to skip static regions during tokenization, achieving sub-linear growth in token count and compute. (2) Semantic-Scene Intra-Tokenization Merging fuses tokens across static regions within a scene, further reducing redundancy while preserving dynamic semantics. Experiments on DIVE show that GRT outperforms larger VLLM baselines and scales positively with FPS. These results highlight the importance of dense temporal information and demonstrate that GRT enables efficient, scalable high-FPS video understanding.
>
---
#### [new 041] Accuracy Paradox in Large Language Models: Regulating Hallucination Risks in Generative AI
- **分类: cs.CY; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 论文探讨大语言模型中“准确性悖论”，指出过度依赖准确性会加剧幻觉风险。论文提出幻觉分类，分析其在输出、个体与社会层面的影响，并主张采用多元、情境化治理方法以提升AI可信度。**

- **链接: [http://arxiv.org/pdf/2509.13345v1](http://arxiv.org/pdf/2509.13345v1)**

> **作者:** Zihao Li; Weiwei Yi; Jiahong Chen
>
> **摘要:** As Large Language Models (LLMs) permeate everyday decision-making, their epistemic and societal risks demand urgent scrutiny. Hallucinations, the generation of fabricated, misleading, oversimplified or untrustworthy outputs, has emerged as imperative challenges. While regulatory, academic, and technical discourse position accuracy as the principal benchmark for mitigating such harms, this article contends that overreliance on accuracy misdiagnoses the problem and has counterproductive effect: the accuracy paradox. Drawing on interdisciplinary literatures, this article develops a taxonomy of hallucination types and shows the paradox along three intertwining dimensions: outputs, individuals and society. First, accuracy functions as a superficial proxy for reliability, incentivising the optimisation of rhetorical fluency and surface-level correctness over epistemic trustworthiness. This encourages passive user trust in outputs that appear accurate but epistemically untenable. Second, accuracy as a singular metric fails to detect harms that are not factually false but are nonetheless misleading, value-laden, or socially distorting, including consensus illusions, sycophantic alignment, and subtle manipulation. Third, regulatory overemphasis on accuracy obscures the wider societal consequences of hallucination, including social sorting, privacy violations, equity harms, epistemic convergence that marginalises dissent, reduces pluralism, and causes social deskilling. By examining the EU AI Act, GDPR, and DSA, the article argues that current regulations are not yet structurally equipped to address these epistemic, relational, and systemic harms and exacerbated by the overreliance on accuracy. By exposing such conceptual and practical challenges, this article calls for a fundamental shift towards pluralistic, context-aware, and manipulation-resilient approaches to AI trustworthy governance.
>
---
#### [new 042] A TRRIP Down Memory Lane: Temperature-Based Re-Reference Interval Prediction For Instruction Caching
- **分类: cs.AR; cs.CL; cs.OS; cs.PF**

- **简介: 论文提出TRRIP方法，结合软硬件协同设计，通过温度分类优化指令缓存替换策略，解决移动设备中代码重复使用距离大导致的缓存效率低问题，降低L2 MPKI并提升性能。**

- **链接: [http://arxiv.org/pdf/2509.14041v1](http://arxiv.org/pdf/2509.14041v1)**

> **作者:** Henry Kao; Nikhil Sreekumar; Prabhdeep Singh Soni; Ali Sedaghati; Fang Su; Bryan Chan; Maziar Goudarzi; Reza Azimi
>
> **摘要:** Modern mobile CPU software pose challenges for conventional instruction cache replacement policies due to their complex runtime behavior causing high reuse distance between executions of the same instruction. Mobile code commonly suffers from large amounts of stalls in the CPU frontend and thus starvation of the rest of the CPU resources. Complexity of these applications and their code footprint are projected to grow at a rate faster than available on-chip memory due to power and area constraints, making conventional hardware-centric methods for managing instruction caches to be inadequate. We present a novel software-hardware co-design approach called TRRIP (Temperature-based Re-Reference Interval Prediction) that enables the compiler to analyze, classify, and transform code based on "temperature" (hot/cold), and to provide the hardware with a summary of code temperature information through a well-defined OS interface based on using code page attributes. TRRIP's lightweight hardware extension employs code temperature attributes to optimize the instruction cache replacement policy resulting in the eviction rate reduction of hot code. TRRIP is designed to be practical and adoptable in real mobile systems that have strict feature requirements on both the software and hardware components. TRRIP can reduce the L2 MPKI for instructions by 26.5% resulting in geomean speedup of 3.9%, on top of RRIP cache replacement running mobile code already optimized using PGO.
>
---
#### [new 043] Explicit Reasoning Makes Better Judges: A Systematic Study on Accuracy, Efficiency, and Robustness
- **分类: cs.AI; cs.CL**

- **简介: 论文研究LLM作为评判者的可靠性、效率与鲁棒性，比较“思考”与“非思考”模型。采用Qwen 3小模型，在RewardBench任务中评估准确性、计算效率及增强策略效果，发现显式推理在多语言环境下具有明显优势。**

- **链接: [http://arxiv.org/pdf/2509.13332v1](http://arxiv.org/pdf/2509.13332v1)**

> **作者:** Pratik Jayarao; Himanshu Gupta; Neeraj Varshney; Chaitanya Dwivedi
>
> **摘要:** As Large Language Models (LLMs) are increasingly adopted as automated judges in benchmarking and reward modeling, ensuring their reliability, efficiency, and robustness has become critical. In this work, we present a systematic comparison of "thinking" and "non-thinking" LLMs in the LLM-as-a-judge paradigm using open-source Qwen 3 models of relatively small sizes (0.6B, 1.7B, and 4B parameters). We evaluate both accuracy and computational efficiency (FLOPs) on RewardBench tasks, and further examine augmentation strategies for non-thinking models, including in-context learning, rubric-guided judging, reference-based evaluation, and n-best aggregation. Our results show that despite these enhancements, non-thinking models generally fall short of their thinking counterparts. Our results show that thinking models achieve approximately 10% points higher accuracy with little overhead (under 2x), in contrast to augmentation strategies like few-shot learning, which deliver modest gains at a higher cost (>8x). Bias and robustness analyses further demonstrate that thinking models maintain significantly greater consistency under a variety of bias conditions such as positional, bandwagon, identity, diversity, and random biases (6% higher on average). We further extend our experiments to the multilingual setting and our results confirm that explicit reasoning extends its benefits beyond English. Overall, our work results in several important findings that provide systematic evidence that explicit reasoning offers clear advantages in the LLM-as-a-judge paradigm not only in accuracy and efficiency but also in robustness.
>
---
#### [new 044] THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出THOR方法，解决大语言模型在数学推理中的高精度问题。通过集成工具、构建高质量数据集、分层优化与自修正机制，提升模型在数学和代码任务上的表现。属于数学推理与代码生成任务。**

- **链接: [http://arxiv.org/pdf/2509.13761v1](http://arxiv.org/pdf/2509.13761v1)**

> **作者:** Qikai Chang; Zhenrong Zhang; Pengfei Hu; Jiefeng Ma; Yicheng Pan; Jianshu Zhang; Jun Du; Quan Liu; Jianqing Gao
>
> **备注:** 22 pages, 13 figures
>
> **摘要:** Large Language Models (LLMs) have made remarkable progress in mathematical reasoning, but still continue to struggle with high-precision tasks like numerical computation and formal symbolic manipulation. Integrating external tools has emerged as a promising approach to bridge this gap. Despite recent advances, existing methods struggle with three key challenges: constructing tool-integrated reasoning data, performing fine-grained optimization, and enhancing inference. To overcome these limitations, we propose THOR (Tool-Integrated Hierarchical Optimization via RL). First, we introduce TIRGen, a multi-agent actor-critic-based pipeline for constructing high-quality datasets of tool-integrated reasoning paths, aligning with the policy and generalizing well across diverse models. Second, to perform fine-grained hierarchical optimization, we introduce an RL strategy that jointly optimizes for both trajectory-level problem solving and step-level code generation. This is motivated by our key insight that the success of an intermediate tool call is a strong predictor of the final answer's correctness. Finally, THOR incorporates a self-correction mechanism that leverages immediate tool feedback to dynamically revise erroneous reasoning paths during inference. Our approach demonstrates strong generalization across diverse models, performing effectively in both reasoning and non-reasoning models. It further achieves state-of-the-art performance for models of a similar scale on multiple mathematical benchmarks, while also delivering consistent improvements on code benchmarks. Our code will be publicly available at https://github.com/JingMog/THOR.
>
---
#### [new 045] When Avatars Have Personality: Effects on Engagement and Communication in Immersive Medical Training
- **分类: cs.HC; cs.CL**

- **简介: 论文提出将大语言模型整合进VR，创建具有个性化的虚拟患者，以提升医疗培训中的沟通训练效果。任务是解决VR中虚拟人物缺乏心理真实感的问题，通过实验验证其有效性并总结设计原则。**

- **链接: [http://arxiv.org/pdf/2509.14132v1](http://arxiv.org/pdf/2509.14132v1)**

> **作者:** Julia S. Dollis; Iago A. Brito; Fernanda B. Färber; Pedro S. F. B. Ribeiro; Rafael T. Sousa; Arlindo R. Galvão Filho
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** While virtual reality (VR) excels at simulating physical environments, its effectiveness for training complex interpersonal skills is limited by a lack of psychologically plausible virtual humans. This is a critical gap in high-stakes domains like medical education, where communication is a core competency. This paper introduces a framework that integrates large language models (LLMs) into immersive VR to create medically coherent virtual patients with distinct, consistent personalities, built on a modular architecture that decouples personality from clinical data. We evaluated our system in a mixed-method, within-subjects study with licensed physicians who engaged in simulated consultations. Results demonstrate that the approach is not only feasible but is also perceived by physicians as a highly rewarding and effective training enhancement. Furthermore, our analysis uncovers critical design principles, including a ``realism-verbosity paradox" where less communicative agents can seem more artificial, and the need for challenges to be perceived as authentic to be instructive. This work provides a validated framework and key insights for developing the next generation of socially intelligent VR training environments.
>
---
#### [new 046] An AI-Powered Framework for Analyzing Collective Idea Evolution in Deliberative Assemblies
- **分类: cs.CY; cs.CL**

- **简介: 论文提出基于LLM的框架，分析协商会议中集体想法的演变过程，解决如何追踪理念演化及影响投票动态的问题，通过分析会议记录揭示高分辨率的协商动态。**

- **链接: [http://arxiv.org/pdf/2509.12577v1](http://arxiv.org/pdf/2509.12577v1)**

> **作者:** Elinor Poole-Dayan; Deb Roy; Jad Kabbara
>
> **摘要:** In an era of increasing societal fragmentation, political polarization, and erosion of public trust in institutions, representative deliberative assemblies are emerging as a promising democratic forum for developing effective policy outcomes on complex global issues. Despite theoretical attention, there remains limited empirical work that systematically traces how specific ideas evolve, are prioritized, or are discarded during deliberation to form policy recommendations. Addressing these gaps, this work poses two central questions: (1) How might we trace the evolution and distillation of ideas into concrete recommendations within deliberative assemblies? (2) How does the deliberative process shape delegate perspectives and influence voting dynamics over the course of the assembly? To address these questions, we develop LLM-based methodologies for empirically analyzing transcripts from a tech-enhanced in-person deliberative assembly. The framework identifies and visualizes the space of expressed suggestions. We also empirically reconstruct each delegate's evolving perspective throughout the assembly. Our methods contribute novel empirical insights into deliberative processes and demonstrate how LLMs can surface high-resolution dynamics otherwise invisible in traditional assembly outputs.
>
---
#### [new 047] Teaching LLMs to Plan: Logical Chain-of-Thought Instruction Tuning for Symbolic Planning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于符号规划任务，旨在提升大语言模型的结构化规划能力。通过逻辑推理链指令微调框架PDDL-Instruct，解决LLMs在PDDL等正式领域中的规划不足问题，显著提高其规划准确率。**

- **链接: [http://arxiv.org/pdf/2509.13351v1](http://arxiv.org/pdf/2509.13351v1)**

> **作者:** Pulkit Verma; Ngoc La; Anthony Favier; Swaroop Mishra; Julie A. Shah
>
> **摘要:** Large language models (LLMs) have demonstrated impressive capabilities across diverse tasks, yet their ability to perform structured symbolic planning remains limited, particularly in domains requiring formal representations like the Planning Domain Definition Language (PDDL). In this paper, we present a novel instruction tuning framework, PDDL-Instruct, designed to enhance LLMs' symbolic planning capabilities through logical chain-of-thought reasoning. Our approach focuses on teaching models to rigorously reason about action applicability, state transitions, and plan validity using explicit logical inference steps. By developing instruction prompts that guide models through the precise logical reasoning required to determine when actions can be applied in a given state, we enable LLMs to self-correct their planning processes through structured reflection. The framework systematically builds verification skills by decomposing the planning process into explicit reasoning chains about precondition satisfaction, effect application, and invariant preservation. Experimental results on multiple planning domains show that our chain-of-thought reasoning based instruction-tuned models are significantly better at planning, achieving planning accuracy of up to 94% on standard benchmarks, representing a 66% absolute improvement over baseline models. This work bridges the gap between the general reasoning capabilities of LLMs and the logical precision required for automated planning, offering a promising direction for developing better AI planning systems.
>
---
#### [new 048] Privacy-Aware In-Context Learning for Large Language Models
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文属于隐私保护文本生成任务，旨在解决大语言模型在生成过程中可能泄露敏感信息的问题。提出了一种基于差分隐私的框架，在无需微调模型的情况下生成高质量合成文本，并通过私有与公开推理融合提升效果。**

- **链接: [http://arxiv.org/pdf/2509.13625v1](http://arxiv.org/pdf/2509.13625v1)**

> **作者:** Bishnu Bhusal; Manoj Acharya; Ramneet Kaur; Colin Samplawski; Anirban Roy; Adam D. Cobb; Rohit Chadha; Susmit Jha
>
> **摘要:** Large language models (LLMs) have significantly transformed natural language understanding and generation, but they raise privacy concerns due to potential exposure of sensitive information. Studies have highlighted the risk of information leakage, where adversaries can extract sensitive information embedded in the prompts. In this work, we introduce a novel private prediction framework for generating high-quality synthetic text with strong privacy guarantees. Our approach leverages the Differential Privacy (DP) framework to ensure worst-case theoretical bounds on information leakage without requiring any fine-tuning of the underlying models.The proposed method performs inference on private records and aggregates the resulting per-token output distributions. This enables the generation of longer and coherent synthetic text while maintaining privacy guarantees. Additionally, we propose a simple blending operation that combines private and public inference to further enhance utility. Empirical evaluations demonstrate that our approach outperforms previous state-of-the-art methods on in-context-learning (ICL) tasks, making it a promising direction for privacy-preserving text generation while maintaining high utility.
>
---
#### [new 049] See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于多模态代理与GUI交互任务，旨在解决代理无法可靠执行切换控制指令的问题。提出State-aware Reasoning（StaR）方法，提升代理对当前状态的感知与判断能力，实验表明其能显著提高执行准确率。**

- **链接: [http://arxiv.org/pdf/2509.13615v1](http://arxiv.org/pdf/2509.13615v1)**

> **作者:** Zongru Wu; Rui Mao; Zhiyuan Tian; Pengzhou Cheng; Tianjie Ju; Zheng Wu; Lingzhong Dong; Haiyue Sheng; Zhuosheng Zhang; Gongshen Liu
>
> **摘要:** The advent of multimodal agents facilitates effective interaction within graphical user interface (GUI), especially in ubiquitous GUI control. However, their inability to reliably execute toggle control instructions remains a key bottleneck. To investigate this, we construct a state control benchmark with binary toggle instructions from public datasets. Evaluations of existing agents demonstrate their unreliability, particularly when the current toggle state already matches the desired state. To address the challenge, we propose State-aware Reasoning (StaR), a training method that teaches agents to perceive the current toggle state, analyze the desired state from the instruction, and act accordingly. Experiments on three multimodal agents demonstrate that StaR can improve toggle instruction execution accuracy by over 30\%. Further evaluations on three public benchmarks show that StaR also enhances general task performance. Finally, evaluations on a dynamic environment highlight the potential of StaR for real-world applications. Code, benchmark, and StaR-enhanced agents are available at https://github.com/ZrW00/StaR.
>
---
#### [new 050] Exploring Major Transitions in the Evolution of Biological Cognition With Artificial Neural Networks
- **分类: cs.AI; cs.CL; cs.FL; cs.LG**

- **简介: 该论文研究生物认知进化中的重大转变，利用人工神经网络模拟信息流变化对认知性能的影响。通过对比不同网络结构，发现循环网络在处理复杂语法任务中表现更优，揭示了结构变化如何引发认知跃迁。**

- **链接: [http://arxiv.org/pdf/2509.13968v1](http://arxiv.org/pdf/2509.13968v1)**

> **作者:** Konstantinos Voudouris; Andrew Barron; Marta Halina; Colin Klein; Matishalin Patel
>
> **摘要:** Transitional accounts of evolution emphasise a few changes that shape what is evolvable, with dramatic consequences for derived lineages. More recently it has been proposed that cognition might also have evolved via a series of major transitions that manipulate the structure of biological neural networks, fundamentally changing the flow of information. We used idealised models of information flow, artificial neural networks (ANNs), to evaluate whether changes in information flow in a network can yield a transitional change in cognitive performance. We compared networks with feed-forward, recurrent and laminated topologies, and tested their performance learning artificial grammars that differed in complexity, controlling for network size and resources. We documented a qualitative expansion in the types of input that recurrent networks can process compared to feed-forward networks, and a related qualitative increase in performance for learning the most complex grammars. We also noted how the difficulty in training recurrent networks poses a form of transition barrier and contingent irreversibility -- other key features of evolutionary transitions. Not all changes in network topology confer a performance advantage in this task set. Laminated networks did not outperform non-laminated networks in grammar learning. Overall, our findings show how some changes in information flow can yield transitions in cognitive performance.
>
---
#### [new 051] Language models' activations linearly encode training-order recency
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文研究语言模型激活状态是否线性编码训练顺序。通过依次微调Llama-3.2-1B模型，发现测试样本激活状态在二维子空间中按训练顺序排列。使用线性探针可区分“早期”与“晚期”实体，表明模型能按学习时间区分信息。**

- **链接: [http://arxiv.org/pdf/2509.14223v1](http://arxiv.org/pdf/2509.14223v1)**

> **作者:** Dmitrii Krasheninnikov; Richard E. Turner; David Krueger
>
> **摘要:** We show that language models' activations linearly encode when information was learned during training. Our setup involves creating a model with a known training order by sequentially fine-tuning Llama-3.2-1B on six disjoint but otherwise similar datasets about named entities. We find that the average activations of test samples for the six training datasets encode the training order: when projected into a 2D subspace, these centroids are arranged exactly in the order of training and lie on a straight line. Further, we show that linear probes can accurately (~90%) distinguish "early" vs. "late" entities, generalizing to entities unseen during the probes' own training. The model can also be fine-tuned to explicitly report an unseen entity's training stage (~80% accuracy). Interestingly, this temporal signal does not seem attributable to simple differences in activation magnitudes, losses, or model confidence. Our paper demonstrates that models are capable of differentiating information by its acquisition time, and carries significant implications for how they might manage conflicting data and respond to knowledge modifications.
>
---
#### [new 052] CogniAlign: Survivability-Grounded Multi-Agent Moral Reasoning for Safe and Transparent AI
- **分类: cs.CY; cs.CL**

- **简介: 该论文提出CogniAlign框架，通过多学科代理协同推理解决AI道德对齐问题。基于自然主义道德实在论，以生存性为根基，提升AI道德判断的透明度与准确性，优于GPT-4o。**

- **链接: [http://arxiv.org/pdf/2509.13356v1](http://arxiv.org/pdf/2509.13356v1)**

> **作者:** Hasin Jawad Ali; Ilhamul Azam; Ajwad Abrar; Md. Kamrul Hasan; Hasan Mahmud
>
> **摘要:** The challenge of aligning artificial intelligence (AI) with human values persists due to the abstract and often conflicting nature of moral principles and the opacity of existing approaches. This paper introduces CogniAlign, a multi-agent deliberation framework based on naturalistic moral realism, that grounds moral reasoning in survivability, defined across individual and collective dimensions, and operationalizes it through structured deliberations among discipline-specific scientist agents. Each agent, representing neuroscience, psychology, sociology, and evolutionary biology, provides arguments and rebuttals that are synthesized by an arbiter into transparent and empirically anchored judgments. We evaluate CogniAlign on classic and novel moral questions and compare its outputs against GPT-4o using a five-part ethical audit framework. Results show that CogniAlign consistently outperforms the baseline across more than sixty moral questions, with average performance gains of 16.2 points in analytic quality, 14.3 points in breadth, and 28.4 points in depth of explanation. In the Heinz dilemma, for example, CogniAlign achieved an overall score of 89.2 compared to GPT-4o's 69.2, demonstrating a decisive advantage in handling moral reasoning. By reducing black-box reasoning and avoiding deceptive alignment, CogniAlign highlights the potential of interdisciplinary deliberation as a scalable pathway for safe and transparent AI alignment.
>
---
#### [new 053] Enhancing Time Awareness in Generative Recommendation
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于生成式推荐任务，旨在解决现有模型忽视时间动态导致的用户偏好变化问题。提出GRUT模型，通过时间感知提示和趋势感知推理，捕捉用户与物品的时间模式，提升推荐效果。**

- **链接: [http://arxiv.org/pdf/2509.13957v1](http://arxiv.org/pdf/2509.13957v1)**

> **作者:** Sunkyung Lee; Seongmin Park; Jonghyo Kim; Mincheol Yoon; Jongwuk Lee
>
> **备注:** EMNLP 2025 (Findings)
>
> **摘要:** Generative recommendation has emerged as a promising paradigm that formulates the recommendations into a text-to-text generation task, harnessing the vast knowledge of large language models. However, existing studies focus on considering the sequential order of items and neglect to handle the temporal dynamics across items, which can imply evolving user preferences. To address this limitation, we propose a novel model, Generative Recommender Using Time awareness (GRUT), effectively capturing hidden user preferences via various temporal signals. We first introduce Time-aware Prompting, consisting of two key contexts. The user-level temporal context models personalized temporal patterns across timestamps and time intervals, while the item-level transition context provides transition patterns across users. We also devise Trend-aware Inference, a training-free method that enhances rankings by incorporating trend information about items with generation likelihood. Extensive experiments demonstrate that GRUT outperforms state-of-the-art models, with gains of up to 15.4% and 14.3% in Recall@5 and NDCG@5 across four benchmark datasets. The source code is available at https://github.com/skleee/GRUT.
>
---
#### [new 054] Annotating Satellite Images of Forests with Keywords from a Specialized Corpus in the Context of Change Detection
- **分类: cs.CV; cs.CL; cs.IR; cs.MM; I.2; I.4; I.7; H.3**

- **简介: 该论文提出一种基于深度学习的卫星图像变化检测方法，用于监测亚马逊雨林的森林砍伐。通过对比不同时间的图像，自动标注变化区域，并利用相关科学文献提取关键词进行注释，以支持环境研究。**

- **链接: [http://arxiv.org/pdf/2509.13586v1](http://arxiv.org/pdf/2509.13586v1)**

> **作者:** Nathalie Neptune; Josiane Mothe
>
> **摘要:** The Amazon rain forest is a vital ecosystem that plays a crucial role in regulating the Earth's climate and providing habitat for countless species. Deforestation in the Amazon is a major concern as it has a significant impact on global carbon emissions and biodiversity. In this paper, we present a method for detecting deforestation in the Amazon using image pairs from Earth observation satellites. Our method leverages deep learning techniques to compare the images of the same area at different dates and identify changes in the forest cover. We also propose a visual semantic model that automatically annotates the detected changes with relevant keywords. The candidate annotation for images are extracted from scientific documents related to the Amazon region. We evaluate our approach on a dataset of Amazon image pairs and demonstrate its effectiveness in detecting deforestation and generating relevant annotations. Our method provides a useful tool for monitoring and studying the impact of deforestation in the Amazon. While we focus on environment applications of our work by using images of deforestation in the Amazon rain forest to demonstrate the effectiveness of our proposed approach, it is generic enough to be applied to other domains.
>
---
#### [new 055] GEM-Bench: A Benchmark for Ad-Injected Response Generation within Generative Engine Marketing
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出GEM-Bench，用于评估广告注入式生成响应的效果。任务是优化生成式引擎营销中的广告插入方法，解决现有基准不足的问题。工作包括构建数据集、设计评估指标及实现基线方案。**

- **链接: [http://arxiv.org/pdf/2509.14221v1](http://arxiv.org/pdf/2509.14221v1)**

> **作者:** Silan Hu; Shiqi Zhang; Yimin Shi; Xiaokui Xiao
>
> **摘要:** Generative Engine Marketing (GEM) is an emerging ecosystem for monetizing generative engines, such as LLM-based chatbots, by seamlessly integrating relevant advertisements into their responses. At the core of GEM lies the generation and evaluation of ad-injected responses. However, existing benchmarks are not specifically designed for this purpose, which limits future research. To address this gap, we propose GEM-Bench, the first comprehensive benchmark for ad-injected response generation in GEM. GEM-Bench includes three curated datasets covering both chatbot and search scenarios, a metric ontology that captures multiple dimensions of user satisfaction and engagement, and several baseline solutions implemented within an extensible multi-agent framework. Our preliminary results indicate that, while simple prompt-based methods achieve reasonable engagement such as click-through rate, they often reduce user satisfaction. In contrast, approaches that insert ads based on pre-generated ad-free responses help mitigate this issue but introduce additional overhead. These findings highlight the need for future research on designing more effective and efficient solutions for generating ad-injected responses in GEM.
>
---
#### [new 056] Noise Supervised Contrastive Learning and Feature-Perturbed for Anomalous Sound Detection
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于异常声音检测任务，旨在仅使用正常音频数据检测未知异常声。提出OS-SCL方法，通过特征扰动和噪声监督对比学习，结合TFgram时频特征，显著提升检测性能。**

- **链接: [http://arxiv.org/pdf/2509.13853v1](http://arxiv.org/pdf/2509.13853v1)**

> **作者:** Shun Huang; Zhihua Fang; Liang He
>
> **备注:** Accept ICASSP 2025
>
> **摘要:** Unsupervised anomalous sound detection aims to detect unknown anomalous sounds by training a model using only normal audio data. Despite advancements in self-supervised methods, the issue of frequent false alarms when handling samples of the same type from different machines remains unresolved. This paper introduces a novel training technique called one-stage supervised contrastive learning (OS-SCL), which significantly addresses this problem by perturbing features in the embedding space and employing a one-stage noisy supervised contrastive learning approach. On the DCASE 2020 Challenge Task 2, it achieved 94.64\% AUC, 88.42\% pAUC, and 89.24\% mAUC using only Log-Mel features. Additionally, a time-frequency feature named TFgram is proposed, which is extracted from raw audio. This feature effectively captures critical information for anomalous sound detection, ultimately achieving 95.71\% AUC, 90.23\% pAUC, and 91.23\% mAUC. The source code is available at: \underline{www.github.com/huangswt/OS-SCL}.
>
---
#### [new 057] TICL: Text-Embedding KNN For Speech In-Context Learning Unlocks Speech Recognition Abilities of Large Multimodal Models
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.MM**

- **简介: 论文提出TICL方法，利用文本嵌入KNN提升大 multimodal 模型的语音识别能力，无需微调。解决SICL中有效示例选择的问题，在多种语音任务中显著降低WER。属于语音识别与多模态学习任务。**

- **链接: [http://arxiv.org/pdf/2509.13395v1](http://arxiv.org/pdf/2509.13395v1)**

> **作者:** Haolong Zheng; Yekaterina Yegorova; Mark Hasegawa-Johnson
>
> **摘要:** Speech foundation models have recently demonstrated the ability to perform Speech In-Context Learning (SICL). Selecting effective in-context examples is crucial for SICL performance, yet selection methodologies remain underexplored. In this work, we propose Text-Embedding KNN for SICL (TICL), a simple pipeline that uses semantic context to enhance off-the-shelf large multimodal models' speech recognition ability without fine-tuning. Across challenging automatic speech recognition tasks, including accented English, multilingual speech, and children's speech, our method enables models to surpass zero-shot performance with up to 84.7% relative WER reduction. We conduct ablation studies to show the robustness and efficiency of our method.
>
---
#### [new 058] Diving into Mitigating Hallucinations from a Vision Perspective for Large Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对大视觉语言模型（LVLMs）中的幻觉问题，提出VisionWeaver网络，通过动态聚合多专家视觉特征以减少幻觉。属于视觉语言模型优化任务，解决幻觉检测与抑制问题，构建了细粒度评估基准VHBench-10。**

- **链接: [http://arxiv.org/pdf/2509.13836v1](http://arxiv.org/pdf/2509.13836v1)**

> **作者:** Weihang Wang; Xinhao Li; Ziyue Wang; Yan Pang; Jielei Zhang; Peiyi Li; Qiang Zhang; Longwen Gao
>
> **备注:** Accepted by EMNLP2025 Finding
>
> **摘要:** Object hallucination in Large Vision-Language Models (LVLMs) significantly impedes their real-world applicability. As the primary component for accurately interpreting visual information, the choice of visual encoder is pivotal. We hypothesize that the diverse training paradigms employed by different visual encoders instill them with distinct inductive biases, which leads to their diverse hallucination performances. Existing benchmarks typically focus on coarse-grained hallucination detection and fail to capture the diverse hallucinations elaborated in our hypothesis. To systematically analyze these effects, we introduce VHBench-10, a comprehensive benchmark with approximately 10,000 samples for evaluating LVLMs across ten fine-grained hallucination categories. Our evaluations confirm encoders exhibit unique hallucination characteristics. Building on these insights and the suboptimality of simple feature fusion, we propose VisionWeaver, a novel Context-Aware Routing Network. It employs global visual features to generate routing signals, dynamically aggregating visual features from multiple specialized experts. Comprehensive experiments confirm the effectiveness of VisionWeaver in significantly reducing hallucinations and improving overall model performance.
>
---
#### [new 059] An Empirical Study on Failures in Automated Issue Solving
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 论文研究自动化问题解决工具的失败原因，分析三种先进工具在SWE-Bench任务中的表现，提出专家-执行者协作框架以提升解决能力。属于软件工程领域，旨在改进自动化代码修复系统的可靠性与效率。**

- **链接: [http://arxiv.org/pdf/2509.13941v1](http://arxiv.org/pdf/2509.13941v1)**

> **作者:** Simiao Liu; Fang Liu; Liehao Li; Xin Tan; Yinghao Zhu; Xiaoli Lian; Li Zhang
>
> **摘要:** Automated issue solving seeks to autonomously identify and repair defective code snippets across an entire codebase. SWE-Bench has emerged as the most widely adopted benchmark for evaluating progress in this area. While LLM-based agentic tools show great promise, they still fail on a substantial portion of tasks. Moreover, current evaluations primarily report aggregate issue-solving rates, which obscure the underlying causes of success and failure, making it challenging to diagnose model weaknesses or guide targeted improvements. To bridge this gap, we first analyze the performance and efficiency of three SOTA tools, spanning both pipeline-based and agentic architectures, in automated issue solving tasks of SWE-Bench-Verified under varying task characteristics. Furthermore, to move from high-level performance metrics to underlying cause analysis, we conducted a systematic manual analysis of 150 failed instances. From this analysis, we developed a comprehensive taxonomy of failure modes comprising 3 primary phases, 9 main categories, and 25 fine-grained subcategories. Then we systematically analyze the distribution of the identified failure modes, the results reveal distinct failure fingerprints between the two architectural paradigms, with the majority of agentic failures stemming from flawed reasoning and cognitive deadlocks. Motivated by these insights, we propose a collaborative Expert-Executor framework. It introduces a supervisory Expert agent tasked with providing strategic oversight and course-correction for a primary Executor agent. This architecture is designed to correct flawed reasoning and break the cognitive deadlocks that frequently lead to failure. Experiments show that our framework solves 22.2% of previously intractable issues for a leading single agent. These findings pave the way for building more robust agents through diagnostic evaluation and collaborative design.
>
---
#### [new 060] SteeringControl: Holistic Evaluation of Alignment Steering in LLMs
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出SteeringControl基准，评估LLMs中对齐引导方法在偏见、有害生成等核心目标及次级行为上的效果。通过构建模块化框架与数据集，分析不同方法、模型与行为间的复杂关系，揭示引导性能与概念纠缠的依赖性。**

- **链接: [http://arxiv.org/pdf/2509.13450v1](http://arxiv.org/pdf/2509.13450v1)**

> **作者:** Vincent Siu; Nicholas Crispino; David Park; Nathan W. Henry; Zhun Wang; Yang Liu; Dawn Song; Chenguang Wang
>
> **摘要:** We introduce SteeringControl, a benchmark for evaluating representation steering methods across core alignment objectives--bias, harmful generation, and hallucination--and their effects on secondary behaviors such as sycophancy and commonsense morality. While prior alignment work often highlights truthfulness or reasoning ability to demonstrate the side effects of representation steering, we find there are many unexplored tradeoffs not yet understood in a systematic way. We collect a dataset of safety-relevant primary and secondary behaviors to evaluate steering effectiveness and behavioral entanglement centered around five popular steering methods. To enable this, we craft a modular steering framework based on unique components that serve as the building blocks of many existing methods. Our results on Qwen-2.5-7B and Llama-3.1-8B find that strong steering performance is dependent on the specific combination of steering method, model, and targeted behavior, and that severe concept entanglement can result from poor combinations of these three as well. We release our code here: https://github.com/wang-research-lab/SteeringControl.git.
>
---
#### [new 061] Reasoning Efficiently Through Adaptive Chain-of-Thought Compression: A Self-Optimizing Framework
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决Chain-of-Thought（CoT）推理中计算成本高、输出冗长的问题。提出SEER框架，通过自适应压缩CoT，在保持准确性的同时提升效率，减少延迟和内存消耗。**

- **链接: [http://arxiv.org/pdf/2509.14093v1](http://arxiv.org/pdf/2509.14093v1)**

> **作者:** Kerui Huang; Shuhan Liu; Xing Hu; Tongtong Xu; Lingfeng Bao; Xin Xia
>
> **摘要:** Chain-of-Thought (CoT) reasoning enhances Large Language Models (LLMs) by prompting intermediate steps, improving accuracy and robustness in arithmetic, logic, and commonsense tasks. However, this benefit comes with high computational costs: longer outputs increase latency, memory usage, and KV-cache demands. These issues are especially critical in software engineering tasks where concise and deterministic outputs are required. To investigate these trade-offs, we conduct an empirical study based on code generation benchmarks. The results reveal that longer CoT does not always help. Excessive reasoning often causes truncation, accuracy drops, and latency up to five times higher, with failed outputs consistently longer than successful ones. These findings challenge the assumption that longer reasoning is inherently better and highlight the need for adaptive CoT control. Motivated by this, we propose SEER (Self-Enhancing Efficient Reasoning), an adaptive framework that compresses CoT while preserving accuracy. SEER combines Best-of-N sampling with task-aware adaptive filtering, dynamically adjusting thresholds based on pre-inference outputs to reduce verbosity and computational overhead. We then evaluate SEER on three software engineering tasks and one math task. On average, SEER shortens CoT by 42.1%, improves accuracy by reducing truncation, and eliminates most infinite loops. These results demonstrate SEER as a practical method to make CoT-enhanced LLMs more efficient and robust, even under resource constraints.
>
---
## 更新

#### [replaced 001] A Comprehensive Survey on the Trustworthiness of Large Language Models in Healthcare
- **分类: cs.CY; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.15871v2](http://arxiv.org/pdf/2502.15871v2)**

> **作者:** Manar Aljohani; Jun Hou; Sindhura Kommu; Xuan Wang
>
> **摘要:** The application of large language models (LLMs) in healthcare holds significant promise for enhancing clinical decision-making, medical research, and patient care. However, their integration into real-world clinical settings raises critical concerns around trustworthiness, particularly around dimensions of truthfulness, privacy, safety, robustness, fairness, and explainability. These dimensions are essential for ensuring that LLMs generate reliable, unbiased, and ethically sound outputs. While researchers have recently begun developing benchmarks and evaluation frameworks to assess LLM trustworthiness, the trustworthiness of LLMs in healthcare remains underexplored, lacking a systematic review that provides a comprehensive understanding and future insights. This survey addresses that gap by providing a comprehensive review of current methodologies and solutions aimed at mitigating risks across key trust dimensions. We analyze how each dimension affects the reliability and ethical deployment of healthcare LLMs, synthesize ongoing research efforts, and identify critical gaps in existing approaches. We also identify emerging challenges posed by evolving paradigms, such as multi-agent collaboration, multi-modal reasoning, and the development of small open-source medical models. Our goal is to guide future research toward more trustworthy, transparent, and clinically viable LLMs.
>
---
#### [replaced 002] DAVIS: Planning Agent with Knowledge Graph-Powered Inner Monologue
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2410.09252v2](http://arxiv.org/pdf/2410.09252v2)**

> **作者:** Minh Pham Dinh; Munira Syed; Michael G Yankoski; Trenton W. Ford
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Designing a generalist scientific agent capable of performing tasks in laboratory settings to assist researchers has become a key goal in recent Artificial Intelligence (AI) research. Unlike everyday tasks, scientific tasks are inherently more delicate and complex, requiring agents to possess a higher level of reasoning ability, structured and temporal understanding of their environment, and a strong emphasis on safety. Existing approaches often fail to address these multifaceted requirements. To tackle these challenges, we present DAVIS. Unlike traditional retrieval-augmented generation (RAG) approaches, DAVIS incorporates structured and temporal memory, which enables model-based planning. Additionally, DAVIS implements an agentic, multi-turn retrieval system, similar to a human's inner monologue, allowing for a greater degree of reasoning over past experiences. DAVIS demonstrates substantially improved performance on the ScienceWorld benchmark comparing to previous approaches on 8 out of 9 elementary science subjects. In addition, DAVIS's World Model demonstrates competitive performance on the famous HotpotQA and MusiqueQA dataset for multi-hop question answering. To the best of our knowledge, DAVIS is the first RAG agent to employ an interactive retrieval method in a RAG pipeline.
>
---
#### [replaced 003] Out-of-Context Reasoning in Large Language Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.10408v3](http://arxiv.org/pdf/2503.10408v3)**

> **作者:** Jonathan Shaki; Emanuele La Malfa; Michael Wooldridge; Sarit Kraus
>
> **摘要:** We study how large language models (LLMs) reason about memorized knowledge through simple binary relations such as equality ($=$), inequality ($<$), and inclusion ($\subset$). Unlike in-context reasoning, the axioms (e.g., $a < b, b < c$) are only seen during training and not provided in the task prompt (e.g., evaluating $a < c$). The tasks require one or more reasoning steps, and data aggregation from one or more sources, showing performance change with task complexity. We introduce a lightweight technique, out-of-context representation learning, which trains only new token embeddings on axioms and evaluates them on unseen tasks. Across reflexivity, symmetry, and transitivity tests, LLMs mostly perform statistically significant better than chance, making the correct answer extractable when testing multiple phrasing variations, but still fall short of consistent reasoning on every single query. Analysis shows that the learned embeddings are organized in structured ways, suggesting real relational understanding. Surprisingly, it also indicates that the core reasoning happens during the training, not inference.
>
---
#### [replaced 004] Singular Value Few-shot Adaptation of Vision-Language Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.03740v2](http://arxiv.org/pdf/2509.03740v2)**

> **作者:** Taha Koleilat; Hassan Rivaz; Yiming Xiao
>
> **备注:** 10 pages, 2 figures, 8 tables
>
> **摘要:** Vision-language models (VLMs) like CLIP have shown impressive zero-shot and few-shot learning capabilities across diverse applications. However, adapting these models to new fine-grained domains remains difficult due to reliance on prompt engineering and the high cost of full model fine-tuning. Existing adaptation approaches rely on augmented components, such as prompt tokens and adapter modules, which could limit adaptation quality, destabilize the model, and compromise the rich knowledge learned during pretraining. In this work, we present CLIP-SVD, a novel multi-modal and parameter-efficient adaptation technique that leverages Singular Value Decomposition (SVD) to modify the internal parameter space of CLIP without injecting additional modules. Specifically, we fine-tune only the singular values of the CLIP parameter matrices to rescale the basis vectors for domain adaptation while retaining the pretrained model. This design enables enhanced adaptation performance using only 0.04% of the model's total parameters and better preservation of its generalization ability. CLIP-SVD achieves state-of-the-art classification results on 11 natural and 10 biomedical datasets, outperforming previous methods in both accuracy and generalization under few-shot settings. Additionally, we leverage a natural language-based approach to analyze the effectiveness and dynamics of the CLIP adaptation to allow interpretability of CLIP-SVD. The code is publicly available at https://github.com/HealthX-Lab/CLIP-SVD.
>
---
#### [replaced 005] Mitigating Attention Hacking in Preference-Based Reward Modeling via Interaction Distillation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.02618v2](http://arxiv.org/pdf/2508.02618v2)**

> **作者:** Jianxiang Zang; Meiling Ning; Shihan Dou; Jiazheng Zhang; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **备注:** This paper is not suitable for this topic, we need to adjust the context
>
> **摘要:** The reward model (RM), as the core component of reinforcement learning from human feedback (RLHF) for large language models (LLMs), responsible for providing reward signals to generated responses. However, mainstream preference modeling in RM is inadequate in terms of token-level interaction, making its judgment signals vulnerable to being hacked by misallocated attention to context. This stems from two fundamental limitations: (1) Current preference modeling employs decoder-only architectures, where the unidirectional causal attention mechanism leads to forward-decaying intra-sequence attention within the prompt-response sequence. (2) The independent Siamese-encoding paradigm induces the absence of token-level inter-sequence attention between chosen and rejected sequences. To address this "attention hacking", we propose "Interaction Distillation", a novel training framework for more adequate preference modeling through attention-level optimization. The method introduces an interaction-based natural language understanding model as the teacher to provide sophisticated token interaction patterns via comprehensive attention, and guides the preference modeling to simulate teacher model's interaction pattern through an attentional alignment objective. Through extensive experiments, interaction distillation has demonstrated its ability to provide more stable and generalizable reward signals compared to state-of-the-art RM optimization methods that target data noise, highlighting the attention hacking constitute a more fundamental limitation in RM.
>
---
#### [replaced 006] Does Localization Inform Unlearning? A Rigorous Examination of Local Parameter Attribution for Knowledge Unlearning in Language Models
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.16252v2](http://arxiv.org/pdf/2505.16252v2)**

> **作者:** Hwiyeong Lee; Uiji Hwang; Hyelim Lim; Taeuk Kim
>
> **备注:** The 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)
>
> **摘要:** Large language models often retain unintended content, prompting growing interest in knowledge unlearning. Recent approaches emphasize localized unlearning, restricting parameter updates to specific regions in an effort to remove target knowledge while preserving unrelated general knowledge. However, their effectiveness remains uncertain due to the lack of robust and thorough evaluation of the trade-off between the competing goals of unlearning. In this paper, we begin by revisiting existing localized unlearning approaches. We then conduct controlled experiments to rigorously evaluate whether local parameter updates causally contribute to unlearning. Our findings reveal that the set of parameters that must be modified for effective unlearning is not strictly determined, challenging the core assumption of localized unlearning that parameter locality is inherently indicative of effective knowledge removal.
>
---
#### [replaced 007] VeriOS: Query-Driven Proactive Human-Agent-GUI Interaction for Trustworthy OS Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.07553v2](http://arxiv.org/pdf/2509.07553v2)**

> **作者:** Zheng Wu; Heyuan Huang; Xingyu Lou; Xiangmou Qu; Pengzhou Cheng; Zongru Wu; Weiwen Liu; Weinan Zhang; Jun Wang; Zhaoxiang Wang; Zhuosheng Zhang
>
> **摘要:** With the rapid progress of multimodal large language models, operating system (OS) agents become increasingly capable of automating tasks through on-device graphical user interfaces (GUIs). However, most existing OS agents are designed for idealized settings, whereas real-world environments often present untrustworthy conditions. To mitigate risks of over-execution in such scenarios, we propose a query-driven human-agent-GUI interaction framework that enables OS agents to decide when to query humans for more reliable task completion. Built upon this framework, we introduce VeriOS-Agent, a trustworthy OS agent trained with a two-stage learning paradigm that falicitate the decoupling and utilization of meta-knowledge. Concretely, VeriOS-Agent autonomously executes actions in normal conditions while proactively querying humans in untrustworthy scenarios. Experiments show that VeriOS-Agent improves the average step-wise success rate by 20.64\% in untrustworthy scenarios over the state-of-the-art, without compromising normal performance. Analysis highlights VeriOS-Agent's rationality, generalizability, and scalability. The codes, datasets and models are available at https://github.com/Wuzheng02/VeriOS.
>
---
#### [replaced 008] Table-Text Alignment: Explaining Claim Verification Against Tables in Scientific Papers
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10486v2](http://arxiv.org/pdf/2506.10486v2)**

> **作者:** Xanh Ho; Sunisth Kumar; Yun-Ang Wu; Florian Boudin; Atsuhiro Takasu; Akiko Aizawa
>
> **备注:** EMNLP 2025 Findings; 9 pages; code and data are available at https://github.com/Alab-NII/SciTabAlign
>
> **摘要:** Scientific claim verification against tables typically requires predicting whether a claim is supported or refuted given a table. However, we argue that predicting the final label alone is insufficient: it reveals little about the model's reasoning and offers limited interpretability. To address this, we reframe table-text alignment as an explanation task, requiring models to identify the table cells essential for claim verification. We build a new dataset by extending the SciTab benchmark with human-annotated cell-level rationales. Annotators verify the claim label and highlight the minimal set of cells needed to support their decision. After the annotation process, we utilize the collected information and propose a taxonomy for handling ambiguous cases. Our experiments show that (i) incorporating table alignment information improves claim verification performance, and (ii) most LLMs, while often predicting correct labels, fail to recover human-aligned rationales, suggesting that their predictions do not stem from faithful reasoning.
>
---
#### [replaced 009] Humor in Pixels: Benchmarking Large Multimodal Models Understanding of Online Comics
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.12248v2](http://arxiv.org/pdf/2509.12248v2)**

> **作者:** Yuriel Ryan; Rui Yang Tan; Kenny Tsu Wei Choo; Roy Ka-Wei Lee
>
> **备注:** 27 pages, 8 figures, EMNLP 2025 Findings
>
> **摘要:** Understanding humor is a core aspect of social intelligence, yet it remains a significant challenge for Large Multimodal Models (LMMs). We introduce PixelHumor, a benchmark dataset of 2,800 annotated multi-panel comics designed to evaluate LMMs' ability to interpret multimodal humor and recognize narrative sequences. Experiments with state-of-the-art LMMs reveal substantial gaps: for instance, top models achieve only 61% accuracy in panel sequencing, far below human performance. This underscores critical limitations in current models' integration of visual and textual cues for coherent narrative and humor understanding. By providing a rigorous framework for evaluating multimodal contextual and narrative reasoning, PixelHumor aims to drive the development of LMMs that better engage in natural, socially aware interactions.
>
---
#### [replaced 010] Unlocking Legal Knowledge: A Multilingual Dataset for Judicial Summarization in Switzerland
- **分类: cs.CL; cs.AI; cs.LG; 68T50; I.2; I.7**

- **链接: [http://arxiv.org/pdf/2410.13456v2](http://arxiv.org/pdf/2410.13456v2)**

> **作者:** Luca Rolshoven; Vishvaksenan Rasiah; Srinanda Brügger Bose; Sarah Hostettler; Lara Burkhalter; Matthias Stürmer; Joel Niklaus
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Legal research is a time-consuming task that most lawyers face on a daily basis. A large part of legal research entails looking up relevant caselaw and bringing it in relation to the case at hand. Lawyers heavily rely on summaries (also called headnotes) to find the right cases quickly. However, not all decisions are annotated with headnotes and writing them is time-consuming. Automated headnote creation has the potential to make hundreds of thousands of decisions more accessible for legal research in Switzerland alone. To kickstart this, we introduce the Swiss Leading Decision Summarization ( SLDS) dataset, a novel cross-lingual resource featuring 18K court rulings from the Swiss Federal Supreme Court (SFSC), in German, French, and Italian, along with German headnotes. We fine-tune and evaluate three mT5 variants, along with proprietary models. Our analysis highlights that while proprietary models perform well in zero-shot and one-shot settings, fine-tuned smaller models still provide a strong competitive edge. We publicly release the dataset to facilitate further research in multilingual legal summarization and the development of assistive technologies for legal professionals
>
---
#### [replaced 011] CAMEO: Collection of Multilingual Emotional Speech Corpora
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.11051v2](http://arxiv.org/pdf/2505.11051v2)**

> **作者:** Iwona Christop; Maciej Czajka
>
> **备注:** Under review at ICASSP
>
> **摘要:** This paper presents CAMEO -- a curated collection of multilingual emotional speech datasets designed to facilitate research in emotion recognition and other speech-related tasks. The main objectives were to ensure easy access to the data, to allow reproducibility of the results, and to provide a standardized benchmark for evaluating speech emotion recognition (SER) systems across different emotional states and languages. The paper describes the dataset selection criteria, the curation and normalization process, and provides performance results for several models. The collection, along with metadata, and a leaderboard, is publicly available via the Hugging Face platform.
>
---
#### [replaced 012] Language Models Identify Ambiguities and Exploit Loopholes
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.19546v2](http://arxiv.org/pdf/2508.19546v2)**

> **作者:** Jio Choi; Mohit Bansal; Elias Stengel-Eskin
>
> **备注:** EMNLP 2025 camera-ready; Code: https://github.com/esteng/ambiguous-loophole-exploitation
>
> **摘要:** Studying the responses of large language models (LLMs) to loopholes presents a two-fold opportunity. First, it affords us a lens through which to examine ambiguity and pragmatics in LLMs, since exploiting a loophole requires identifying ambiguity and performing sophisticated pragmatic reasoning. Second, loopholes pose an interesting and novel alignment problem where the model is presented with conflicting goals and can exploit ambiguities to its own advantage. To address these questions, we design scenarios where LLMs are given a goal and an ambiguous user instruction in conflict with the goal, with scenarios covering scalar implicature, structural ambiguities, and power dynamics. We then measure different models' abilities to exploit loopholes to satisfy their given goals as opposed to the goals of the user. We find that both closed-source and stronger open-source models can identify ambiguities and exploit their resulting loopholes, presenting a potential AI safety risk. Our analysis indicates that models which exploit loopholes explicitly identify and reason about both ambiguity and conflicting goals.
>
---
#### [replaced 013] Do Large Language Models Truly Grasp Addition? A Rule-Focused Diagnostic Using Two-Integer Arithmetic
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.05262v3](http://arxiv.org/pdf/2504.05262v3)**

> **作者:** Yang Yan; Yu Lu; Renjun Xu; Zhenzhong Lan
>
> **备注:** Accepted by EMNLP'25 Main
>
> **摘要:** Large language models (LLMs) achieve impressive results on advanced mathematics benchmarks but sometimes fail on basic arithmetic tasks, raising the question of whether they have truly grasped fundamental arithmetic rules or are merely relying on pattern matching. To unravel this issue, we systematically probe LLMs' understanding of two-integer addition ($0$ to $2^{64}$) by testing three crucial properties: commutativity ($A+B=B+A$), representation invariance via symbolic remapping (e.g., $7 \mapsto Y$), and consistent accuracy scaling with operand length. Our evaluation of 12 leading LLMs reveals a stark disconnect: while models achieve high numeric accuracy (73.8-99.8%), they systematically fail these diagnostics. Specifically, accuracy plummets to $\le 7.5$% with symbolic inputs, commutativity is violated in up to 20% of cases, and accuracy scaling is non-monotonic. Interventions further expose this pattern-matching reliance: explicitly providing rules degrades performance by 29.49%, while prompting for explanations before answering merely maintains baseline accuracy. These findings demonstrate that current LLMs address elementary addition via pattern matching, not robust rule induction, motivating new diagnostic benchmarks and innovations in model architecture and training to cultivate genuine mathematical reasoning. Our dataset and generating code are available at https://github.com/kuri-leo/llm-arithmetic-diagnostic.
>
---
#### [replaced 014] MAVL: A Multilingual Audio-Video Lyrics Dataset for Animated Song Translation
- **分类: cs.CL; cs.LG; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.18614v3](http://arxiv.org/pdf/2505.18614v3)**

> **作者:** Woohyun Cho; Youngmin Kim; Sunghyun Lee; Youngjae Yu
>
> **备注:** Accepted to EMNLP 2025, Project Page: https://k1064190.github.io/papers/paper1.html, our codes and datasets are available at https://github.com/k1064190/MAVL
>
> **摘要:** Lyrics translation requires both accurate semantic transfer and preservation of musical rhythm, syllabic structure, and poetic style. In animated musicals, the challenge intensifies due to alignment with visual and auditory cues. We introduce Multilingual Audio-Video Lyrics Benchmark for Animated Song Translation (MAVL), the first multilingual, multimodal benchmark for singable lyrics translation. By integrating text, audio, and video, MAVL enables richer and more expressive translations than text-only approaches. Building on this, we propose Syllable-Constrained Audio-Video LLM with Chain-of-Thought SylAVL-CoT, which leverages audio-video cues and enforces syllabic constraints to produce natural-sounding lyrics. Experimental results demonstrate that SylAVL-CoT significantly outperforms text-based models in singability and contextual accuracy, emphasizing the value of multimodal, multilingual approaches for lyrics translation.
>
---
#### [replaced 015] KBM: Delineating Knowledge Boundary for Adaptive Retrieval in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.06207v2](http://arxiv.org/pdf/2411.06207v2)**

> **作者:** Zhen Zhang; Xinyu Wang; Yong Jiang; Zile Qiao; Zhuo Chen; Guangyu Li; Feiteng Mu; Mengting Hu; Pengjun Xie; Fei Huang
>
> **摘要:** Large Language Models (LLMs) often struggle with dynamically changing knowledge and handling unknown static information. Retrieval-Augmented Generation (RAG) is employed to tackle these challenges and has a significant impact on improving LLM performance. In fact, we find that not all questions need to trigger RAG. By retrieving parts of knowledge unknown to the LLM and allowing the LLM to answer the rest, we can effectively reduce both time and computational costs. In our work, we propose a Knowledge Boundary Model (KBM) to express the known/unknown of a given question, and to determine whether a RAG needs to be triggered. Experiments conducted on 11 English and Chinese datasets illustrate that the KBM effectively delineates the knowledge boundary, significantly decreasing the proportion of retrievals required for optimal end-to-end performance. Furthermore, we evaluate the effectiveness of KBM in three complex scenarios: dynamic knowledge, long-tail static knowledge, and multi-hop problems, as well as its functionality as an external LLM plug-in.
>
---
#### [replaced 016] DeDisCo at the DISRPT 2025 Shared Task: A System for Discourse Relation Classification
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.11498v2](http://arxiv.org/pdf/2509.11498v2)**

> **作者:** Zhuoxuan Ju; Jingni Wu; Abhishek Purushothama; Amir Zeldes
>
> **备注:** System submission for the DISRPT 2025 - Shared Task on Discourse Relation Parsing and Treebanking In conjunction with CODI-CRAC & EMNLP 2025. 1st place in Task 3: relation classification
>
> **摘要:** This paper presents DeDisCo, Georgetown University's entry in the DISRPT 2025 shared task on discourse relation classification. We test two approaches, using an mt5-based encoder and a decoder based approach using the openly available Qwen model. We also experiment on training with augmented dataset for low-resource languages using matched data translated automatically from English, as well as using some additional linguistic features inspired by entries in previous editions of the Shared Task. Our system achieves a macro-accuracy score of 71.28, and we provide some interpretation and error analysis for our results.
>
---
#### [replaced 017] Context Copying Modulation: The Role of Entropy Neurons in Managing Parametric and Contextual Knowledge Conflicts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.10663v2](http://arxiv.org/pdf/2509.10663v2)**

> **作者:** Zineddine Tighidet; Andrea Mogini; Hedi Ben-younes; Jiali Mei; Patrick Gallinari; Benjamin Piwowarski
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** The behavior of Large Language Models (LLMs) when facing contextual information that conflicts with their internal parametric knowledge is inconsistent, with no generally accepted explanation for the expected outcome distribution. Recent work has identified in autoregressive transformer models a class of neurons -- called entropy neurons -- that produce a significant effect on the model output entropy while having an overall moderate impact on the ranking of the predicted tokens. In this paper, we investigate the preliminary claim that these neurons are involved in inhibiting context copying behavior in transformers by looking at their role in resolving conflicts between contextual and parametric information. We show that entropy neurons are responsible for suppressing context copying across a range of LLMs, and that ablating them leads to a significant change in the generation process. These results enhance our understanding of the internal dynamics of LLMs when handling conflicting information.
>
---
#### [replaced 018] MAgICoRe: Multi-Agent, Iterative, Coarse-to-Fine Refinement for Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.12147v2](http://arxiv.org/pdf/2409.12147v2)**

> **作者:** Justin Chih-Yao Chen; Archiki Prasad; Swarnadeep Saha; Elias Stengel-Eskin; Mohit Bansal
>
> **备注:** EMNLP 2025 (Camera-Ready)
>
> **摘要:** Large Language Models' (LLM) reasoning can be improved using test-time aggregation strategies, i.e., generating multiple samples and voting among generated samples. While these improve performance, they often reach a saturation point. Refinement offers an alternative by using LLM-generated feedback to improve solution quality. However, refinement introduces 3 key challenges: (1) Excessive refinement: Uniformly refining all instances can over-correct and reduce the overall performance. (2) Inability to localize and address errors: LLMs have a limited ability to self-correct and struggle to identify and correct their own mistakes. (3) Insufficient refinement: Deciding how many iterations of refinement are needed is non-trivial, and stopping too soon could leave errors unaddressed. To tackle these issues, we propose MAgICoRe, which avoids excessive refinement by categorizing problem difficulty as easy or hard, solving easy problems with coarse-grained aggregation and hard ones with fine-grained and iterative multi-agent refinement. To improve error localization, we incorporate external step-wise reward model (RM) scores. Moreover, to ensure effective refinement, we employ a multi-agent loop with three agents: Solver, Reviewer (which generates targeted feedback based on step-wise RM scores), and the Refiner (which incorporates feedback). To ensure sufficient refinement, we re-evaluate updated solutions, iteratively initiating further rounds of refinement. We evaluate MAgICoRe on Llama-3-8B and GPT-3.5 and show its effectiveness across 5 math datasets. Even one iteration of MAgICoRe beats Self-Consistency by 3.4%, Best-of-k by 3.2%, and Self-Refine by 4.0% while using less than half the samples. Unlike iterative refinement with baselines, MAgICoRe continues to improve with more iterations. Finally, our ablations highlight the importance of MAgICoRe's RMs and multi-agent communication.
>
---
#### [replaced 019] Mind the Style Gap: Meta-Evaluation of Style and Attribute Transfer Metrics
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.15022v4](http://arxiv.org/pdf/2502.15022v4)**

> **作者:** Amalie Brogaard Pauli; Isabelle Augenstein; Ira Assent
>
> **备注:** Accepted at EMNLP Findings 2025
>
> **摘要:** Large language models (LLMs) make it easy to rewrite a text in any style -- e.g. to make it more polite, persuasive, or more positive -- but evaluation thereof is not straightforward. A challenge lies in measuring content preservation: that content not attributable to style change is retained. This paper presents a large meta-evaluation of metrics for evaluating style and attribute transfer, focusing on content preservation. We find that meta-evaluation studies on existing datasets lead to misleading conclusions about the suitability of metrics for content preservation. Widely used metrics show a high correlation with human judgments despite being deemed unsuitable for the task -- because they do not abstract from style changes when evaluating content preservation. We show that the overly high correlations with human judgment stem from the nature of the test data. To address this issue, we introduce a new, challenging test set specifically designed for evaluating content preservation metrics for style transfer. We construct the data by creating high variation in the content preservation. Using this dataset, we demonstrate that suitable metrics for content preservation for style transfer indeed are style-aware. To support efficient evaluation, we propose a new style-aware method that utilises small language models, obtaining a higher alignment with human judgements than prompting a model of a similar size as an autorater. ater.
>
---
#### [replaced 020] Empathy Omni: Enabling Empathetic Speech Response Generation through Large Language Models
- **分类: cs.CL; cs.SD; eess.AS; I.2.7**

- **链接: [http://arxiv.org/pdf/2508.18655v3](http://arxiv.org/pdf/2508.18655v3)**

> **作者:** Haoyu Wang; Guangyan Zhang; Jiale Chen; Jingyu Li; Yuehai Wang; Yiwen Guo
>
> **备注:** 5 pages, 1 figure, submitted to ICASSP 2026
>
> **摘要:** With the development of speech large language models (speech LLMs), users can now interact directly with assistants via speech. However, most existing models only convert response content into speech without fully capturing the rich emotional cues in user queries, where the same sentence may convey different meanings depending on the expression. Emotional understanding is thus essential for improving human-machine interaction. Most empathetic speech LLMs rely on massive datasets, demanding high computational cost. A key challenge is to build models that generate empathetic responses with limited data and without large-scale training. To this end, we propose Emotion Omni, a model that understands emotional content in user speech and generates empathetic responses. We further developed a data pipeline to construct a 200k emotional dialogue dataset supporting empathetic speech assistants. Experiments show that Emotion Omni achieves comparable instruction-following ability without large-scale pretraining, while surpassing existing models in speech quality (UTMOS:4.41) and empathy (Emotion GPT Score: 3.97). These results confirm its improvements in both speech fidelity and emotional expressiveness. Demos are available at https://w311411.github.io/omni_demo/.
>
---
#### [replaced 021] FinCoT: Grounding Chain-of-Thought in Expert Financial Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.16123v4](http://arxiv.org/pdf/2506.16123v4)**

> **作者:** Natapong Nitarach; Warit Sirichotedumrong; Panop Pitchayarthorn; Pittawat Taveekitworachai; Potsawee Manakul; Kunat Pipatanakul
>
> **备注:** Accepted at FinNLP-2025, EMNLP
>
> **摘要:** This paper presents FinCoT, a structured chain-of-thought (CoT) prompting framework that embeds domain-specific expert financial reasoning blueprints to guide large language models' behaviors. We identify three main prompting styles in financial NLP (FinNLP): (1) standard prompting (zero-shot), (2) unstructured CoT (free-form reasoning), and (3) structured CoT (with explicitly structured reasoning steps). Prior work has mainly focused on the first two, while structured CoT remains underexplored and lacks domain expertise incorporation. Therefore, we evaluate all three prompting approaches across ten CFA-style financial domains and introduce FinCoT as the first structured finance-specific prompting approach incorporating blueprints from domain experts. FinCoT improves the accuracy of a general-purpose model, Qwen3-8B-Base, from 63.2% to 80.5%, and boosts Fin-R1 (7B), a finance-specific model, from 65.7% to 75.7%, while reducing output length by up to 8.9x and 1.16x compared to structured CoT methods, respectively. We find that FinCoT proves most effective for models lacking financial post-training. Our findings show that FinCoT does not only improve performance and reduce inference costs but also yields more interpretable and expert-aligned reasoning traces.
>
---
#### [replaced 022] An Attention-Based Denoising Framework for Personality Detection in Social Media Texts
- **分类: cs.CY; cs.CL**

- **链接: [http://arxiv.org/pdf/2311.09945v2](http://arxiv.org/pdf/2311.09945v2)**

> **作者:** Lei Lin; Jizhao Zhu; Qirui Tang; Yihua Du
>
> **摘要:** In social media networks, users produce a large amount of text content anytime, providing researchers with an invaluable approach to digging for personality-related information. Personality detection based on user-generated text is a method with broad application prospects, such as for constructing user portraits. The presence of significant noise in social media texts hinders personality detection. However, previous studies have not delved deeper into addressing this challenge. Inspired by the scanning reading technique, we propose an attention-based information extraction mechanism (AIEM) for long texts, which is applied to quickly locate valuable pieces of text, and fully integrate beneficial semantic information. Then, we provide a novel attention-based denoising framework (ADF) for personality detection tasks and achieve state-of-the-art performance on two commonly used datasets. Notably, we obtain an average accuracy improvement of 10.2% on the gold standard Twitter-Myers-Briggs Type Indicator (Twitter-MBTI) dataset. We made our code publicly available on GitHub\footnote{https://github.com/Once2gain/PersonalityDetection}. We shed light on how AIEM works to magnify personality-related signals through a case study.
>
---
#### [replaced 023] FroM: Frobenius Norm-Based Data-Free Adaptive Model Merging
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.02478v2](http://arxiv.org/pdf/2506.02478v2)**

> **作者:** Zijian Li; Xiaocheng Feng; Huixin Liu; Yichong Huang; Ting Liu; Bing Qin
>
> **备注:** 12 pages, 11 figures
>
> **摘要:** With the development of large language models, fine-tuning has emerged as an effective method to enhance performance in specific scenarios by injecting domain-specific knowledge. In this context, model merging techniques provide a solution for fusing knowledge from multiple fine-tuning models by combining their parameters. However, traditional methods often encounter task interference when merging full fine-tuning models, and this problem becomes even more evident in parameter-efficient fine-tuning scenarios. In this paper, we introduce an improvement to the RegMean method, which indirectly leverages the training data to approximate the outputs of the linear layers before and after merging. We propose an adaptive merging method called FroM, which directly measures the model parameters using the Frobenius norm, without any training data. By introducing an additional hyperparameter for control, FroM outperforms baseline methods across various fine-tuning scenarios, alleviating the task interference problem.
>
---
#### [replaced 024] A Culturally-diverse Multilingual Multimodal Video Benchmark & Model
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.07032v2](http://arxiv.org/pdf/2506.07032v2)**

> **作者:** Bhuiyan Sanjid Shafique; Ashmal Vayani; Muhammad Maaz; Hanoona Abdul Rasheed; Dinura Dissanayake; Mohammed Irfan Kurpath; Yahya Hmaiti; Go Inoue; Jean Lahoud; Md. Safirur Rashid; Shadid Intisar Quasem; Maheen Fatima; Franco Vidal; Mykola Maslych; Ketan Pravin More; Sanoojan Baliah; Hasindri Watawana; Yuhao Li; Fabian Farestam; Leon Schaller; Roman Tymtsiv; Simon Weber; Hisham Cholakkal; Ivan Laptev; Shin'ichi Satoh; Michael Felsberg; Mubarak Shah; Salman Khan; Fahad Shahbaz Khan
>
> **摘要:** Large multimodal models (LMMs) have recently gained attention due to their effectiveness to understand and generate descriptions of visual content. Most existing LMMs are in English language. While few recent works explore multilingual image LMMs, to the best of our knowledge, moving beyond the English language for cultural and linguistic inclusivity is yet to be investigated in the context of video LMMs. In pursuit of more inclusive video LMMs, we introduce a multilingual Video LMM benchmark, named ViMUL-Bench, to evaluate Video LMMs across 14 languages, including both low- and high-resource languages: English, Chinese, Spanish, French, German, Hindi, Arabic, Russian, Bengali, Urdu, Sinhala, Tamil, Swedish, and Japanese. Our ViMUL-Bench is designed to rigorously test video LMMs across 15 categories including eight culturally diverse categories, ranging from lifestyles and festivals to foods and rituals and from local landmarks to prominent cultural personalities. ViMUL-Bench comprises both open-ended (short and long-form) and multiple-choice questions spanning various video durations (short, medium, and long) with 8k samples that are manually verified by native language speakers. In addition, we also introduce a machine translated multilingual video training set comprising 1.2 million samples and develop a simple multilingual video LMM, named ViMUL, that is shown to provide a better tradeoff between high-and low-resource languages for video understanding. We hope our ViMUL-Bench and multilingual video LMM along with a large-scale multilingual video training set will help ease future research in developing cultural and linguistic inclusive multilingual video LMMs. Our proposed benchmark, video LMM and training data will be publicly released at https://mbzuai-oryx.github.io/ViMUL/.
>
---
#### [replaced 025] Contextualize-then-Aggregate: Circuits for In-Context Learning in Gemma-2 2B
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.00132v4](http://arxiv.org/pdf/2504.00132v4)**

> **作者:** Aleksandra Bakalova; Yana Veitsman; Xinting Huang; Michael Hahn
>
> **摘要:** In-Context Learning (ICL) is an intriguing ability of large language models (LLMs). Despite a substantial amount of work on its behavioral aspects and how it emerges in miniature setups, it remains unclear which mechanism assembles task information from the individual examples in a fewshot prompt. We use causal interventions to identify information flow in Gemma-2 2B for five naturalistic ICL tasks. We find that the model infers task information using a two-step strategy we call contextualize-then-aggregate: In the lower layers, the model builds up representations of individual fewshot examples, which are contextualized by preceding examples through connections between fewshot input and output tokens across the sequence. In the higher layers, these representations are aggregated to identify the task and prepare prediction of the next output. The importance of the contextualization step differs between tasks, and it may become more important in the presence of ambiguous examples. Overall, by providing rigorous causal analysis, our results shed light on the mechanisms through which ICL happens in language models.
>
---
#### [replaced 026] Training Text-to-Molecule Models with Context-Aware Tokenization
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.04476v2](http://arxiv.org/pdf/2509.04476v2)**

> **作者:** Seojin Kim; Hyeontae Song; Jaehyun Nam; Jinwoo Shin
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Recently, text-to-molecule models have shown great potential across various chemical applications, e.g., drug-discovery. These models adapt language models to molecular data by representing molecules as sequences of atoms. However, they rely on atom-level tokenizations, which primarily focus on modeling local connectivity, thereby limiting the ability of models to capture the global structural context within molecules. To tackle this issue, we propose a novel text-to-molecule model, coined Context-Aware Molecular T5 (CAMT5). Inspired by the significance of the substructure-level contexts in understanding molecule structures, e.g., ring systems, we introduce substructure-level tokenization for text-to-molecule models. Building on our tokenization scheme, we develop an importance-based training strategy that prioritizes key substructures, enabling CAMT5 to better capture the molecular semantics. Extensive experiments verify the superiority of CAMT5 in various text-to-molecule generation tasks. Intriguingly, we find that CAMT5 outperforms the state-of-the-art methods using only 2% of training tokens. In addition, we propose a simple yet effective ensemble strategy that aggregates the outputs of text-to-molecule models to further boost the generation performance. Code is available at https://github.com/Songhyeontae/CAMT5.git.
>
---
#### [replaced 027] MythTriage: Scalable Detection of Opioid Use Disorder Myths on a Video-Sharing Platform
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.00308v2](http://arxiv.org/pdf/2506.00308v2)**

> **作者:** Hayoung Jung; Shravika Mittal; Ananya Aatreya; Navreet Kaur; Munmun De Choudhury; Tanushree Mitra
>
> **备注:** To appear at EMNLP 2025. Please cite EMNLP version when proceedings are available
>
> **摘要:** Understanding the prevalence of misinformation in health topics online can inform public health policies and interventions. However, measuring such misinformation at scale remains a challenge, particularly for high-stakes but understudied topics like opioid-use disorder (OUD)--a leading cause of death in the U.S. We present the first large-scale study of OUD-related myths on YouTube, a widely-used platform for health information. With clinical experts, we validate 8 pervasive myths and release an expert-labeled video dataset. To scale labeling, we introduce MythTriage, an efficient triage pipeline that uses a lightweight model for routine cases and defers harder ones to a high-performing, but costlier, large language model (LLM). MythTriage achieves up to 0.86 macro F1-score while estimated to reduce annotation time and financial cost by over 76% compared to experts and full LLM labeling. We analyze 2.9K search results and 343K recommendations, uncovering how myths persist on YouTube and offering actionable insights for public health and platform moderation.
>
---
#### [replaced 028] Beyond checkmate: exploring the creative chokepoints in AI text
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.19301v2](http://arxiv.org/pdf/2501.19301v2)**

> **作者:** Nafis Irtiza Tripto; Saranya Venkatraman; Mahjabin Nahar; Dongwon Lee
>
> **备注:** Accepted at 30th Conference on Empirical Methods in Natural Language Processing (EMNLP'25 Main conference). 9 pages
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has revolutionized text generation but also raised concerns about potential misuse, making detecting LLM-generated text (AI text) increasingly essential. While prior work has focused on identifying AI text and effectively checkmating it, our study investigates a less-explored territory: portraying the nuanced distinctions between human and AI texts across text segments (introduction, body, and conclusion). Whether LLMs excel or falter in incorporating linguistic ingenuity across text segments, the results will critically inform their viability and boundaries as effective creative assistants to humans. Through an analogy with the structure of chess games, comprising opening, middle, and end games, we analyze segment-specific patterns to reveal where the most striking differences lie. Although AI texts closely resemble human writing in the body segment due to its length, deeper analysis shows a higher divergence in features dependent on the continuous flow of language, making it the most informative segment for detection. Additionally, human texts exhibit greater stylistic variation across segments, offering a new lens for distinguishing them from AI. Overall, our findings provide fresh insights into human-AI text differences and pave the way for more effective and interpretable detection strategies. Codes available at https://github.com/tripto03/chess_inspired_human_ai_text_distinction.
>
---
#### [replaced 029] xGen-MM (BLIP-3): A Family of Open Large Multimodal Models
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.08872v4](http://arxiv.org/pdf/2408.08872v4)**

> **作者:** Le Xue; Manli Shu; Anas Awadalla; Jun Wang; An Yan; Senthil Purushwalkam; Honglu Zhou; Viraj Prabhu; Yutong Dai; Michael S Ryoo; Shrikant Kendre; Jieyu Zhang; Shaoyen Tseng; Gustavo A Lujan-Moreno; Matthew L Olson; Musashi Hinck; David Cobbley; Vasudev Lal; Can Qin; Shu Zhang; Chia-Chih Chen; Ning Yu; Juntao Tan; Tulika Manoj Awalgaonkar; Shelby Heinecke; Huan Wang; Yejin Choi; Ludwig Schmidt; Zeyuan Chen; Silvio Savarese; Juan Carlos Niebles; Caiming Xiong; Ran Xu
>
> **摘要:** This paper introduces BLIP-3, an open framework for developing Large Multimodal Models (LMMs). The framework comprises meticulously curated datasets, a training recipe, model architectures, and a resulting suite of LMMs. We release 4B and 14B models, including both the pre-trained base model and the instruction fine-tuned ones. Our models undergo rigorous evaluation across a range of tasks, including both single and multi-image benchmarks. Our models demonstrate competitive performance among open-source LMMs with similar model sizes. Our resulting LMMs demonstrate competitive performance among open-source LMMs with similar model sizes, with the ability to comprehend interleaved image-text inputs. Our training code, models, and all datasets used in this work, including the three largescale datasets we create and the preprocessed ones, will be open-sourced to better support the research community.
>
---
#### [replaced 030] MOOM: Maintenance, Organization and Optimization of Memory in Ultra-Long Role-Playing Dialogues
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.11860v2](http://arxiv.org/pdf/2509.11860v2)**

> **作者:** Weishu Chen; Jinyi Tang; Zhouhui Hou; Shihao Han; Mingjie Zhan; Zhiyuan Huang; Delong Liu; Jiawei Guo; Zhicheng Zhao; Fei Su
>
> **摘要:** Memory extraction is crucial for maintaining coherent ultra-long dialogues in human-robot role-playing scenarios. However, existing methods often exhibit uncontrolled memory growth. To address this, we propose MOOM, the first dual-branch memory plugin that leverages literary theory by modeling plot development and character portrayal as core storytelling elements. Specifically, one branch summarizes plot conflicts across multiple time scales, while the other extracts the user's character profile. MOOM further integrates a forgetting mechanism, inspired by the ``competition-inhibition'' memory theory, to constrain memory capacity and mitigate uncontrolled growth. Furthermore, we present ZH-4O, a Chinese ultra-long dialogue dataset specifically designed for role-playing, featuring dialogues that average 600 turns and include manually annotated memory information. Experimental results demonstrate that MOOM outperforms all state-of-the-art memory extraction methods, requiring fewer large language model invocations while maintaining a controllable memory capacity.
>
---
#### [replaced 031] Mirror-Consistency: Harnessing Inconsistency in Majority Voting
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.10857v2](http://arxiv.org/pdf/2410.10857v2)**

> **作者:** Siyuan Huang; Zhiyuan Ma; Jintao Du; Changhua Meng; Weiqiang Wang; Zhouhan Lin
>
> **备注:** EMNLP 2024 Findings
>
> **摘要:** Self-Consistency, a widely-used decoding strategy, significantly boosts the reasoning capabilities of Large Language Models (LLMs). However, it depends on the plurality voting rule, which focuses on the most frequent answer while overlooking all other minority responses. These inconsistent minority views often illuminate areas of uncertainty within the model's generation process. To address this limitation, we present Mirror-Consistency, an enhancement of the standard Self-Consistency approach. Our method incorporates a 'reflective mirror' into the self-ensemble decoding process and enables LLMs to critically examine inconsistencies among multiple generations. Additionally, just as humans use the mirror to better understand themselves, we propose using Mirror-Consistency to enhance the sample-based confidence calibration methods, which helps to mitigate issues of overconfidence. Our experimental results demonstrate that Mirror-Consistency yields superior performance in both reasoning accuracy and confidence calibration compared to Self-Consistency.
>
---
#### [replaced 032] Calibrating LLMs for Text-to-SQL Parsing by Leveraging Sub-clause Frequencies
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.23804v2](http://arxiv.org/pdf/2505.23804v2)**

> **作者:** Terrance Liu; Shuyi Wang; Daniel Preotiuc-Pietro; Yash Chandarana; Chirag Gupta
>
> **备注:** EMNLP 2025 main conference
>
> **摘要:** While large language models (LLMs) achieve strong performance on text-to-SQL parsing, they sometimes exhibit unexpected failures in which they are confidently incorrect. Building trustworthy text-to-SQL systems thus requires eliciting reliable uncertainty measures from the LLM. In this paper, we study the problem of providing a calibrated confidence score that conveys the likelihood of an output query being correct. Our work is the first to establish a benchmark for post-hoc calibration of LLM-based text-to-SQL parsing. In particular, we show that Platt scaling, a canonical method for calibration, provides substantial improvements over directly using raw model output probabilities as confidence scores. Furthermore, we propose a method for text-to-SQL calibration that leverages the structured nature of SQL queries to provide more granular signals of correctness, named "sub-clause frequency" (SCF) scores. Using multivariate Platt scaling (MPS), our extension of the canonical Platt scaling technique, we combine individual SCF scores into an overall accurate and calibrated score. Empirical evaluation on two popular text-to-SQL datasets shows that our approach of combining MPS and SCF yields further improvements in calibration and the related task of error detection over traditional Platt scaling.
>
---
#### [replaced 033] KALL-E:Autoregressive Speech Synthesis with Next-Distribution Prediction
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2412.16846v2](http://arxiv.org/pdf/2412.16846v2)**

> **作者:** Kangxiang Xia; Xinfa Zhu; Jixun Yao; Wenjie Tian; Wenhao Li; Lei Xie
>
> **备注:** 6 figures, 5 tables
>
> **摘要:** We introduce KALL-E, a novel autoregressive (AR) language model for text-to-speech (TTS) synthesis that operates by predicting the next distribution of continuous speech frames. Unlike existing methods, KALL-E directly models the continuous speech distribution conditioned on text, eliminating the need for any diffusion-based components. Specifically, we utilize a Flow-VAE to extract a continuous latent speech representation from waveforms, instead of relying on discrete speech tokens. A single AR Transformer is then trained to predict these continuous speech distributions from text, optimizing a Kullback-Leibler divergence loss as its objective. Experimental results demonstrate that KALL-E achieves superior speech synthesis quality and can even adapt to a target speaker from just a single sample. Importantly, KALL-E provides a more direct and effective approach for utilizing continuous speech representations in TTS.
>
---
#### [replaced 034] NeedleBench: Evaluating LLM Retrieval and Reasoning Across Varying Information Densities
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.11963v3](http://arxiv.org/pdf/2407.11963v3)**

> **作者:** Mo Li; Songyang Zhang; Taolin Zhang; Haodong Duan; Yunxin Liu; Kai Chen
>
> **备注:** v3: Revisions with added experiments, clarifications, and related work updates
>
> **摘要:** The capability of large language models to handle long-context information is crucial across various real-world applications. Existing evaluation methods often rely either on real-world long texts, making it difficult to exclude the influence of models' inherent knowledge, or introduce irrelevant filler content to artificially achieve target lengths, reducing assessment effectiveness. To address these limitations, we introduce NeedleBench, a synthetic framework for assessing retrieval and reasoning performance in bilingual long-context tasks with adaptive context lengths. NeedleBench systematically embeds key data points at varying depths to rigorously test model capabilities. Tasks are categorized into two scenarios: information-sparse, featuring minimal relevant details within extensive irrelevant text to simulate simple retrieval tasks; and information-dense (the Ancestral Trace Challenge), where relevant information is continuously distributed throughout the context to simulate complex reasoning tasks. Our experiments reveal that although recent reasoning models like Deepseek-R1 and OpenAI's o3 excel in mathematical reasoning, they struggle with continuous retrieval and reasoning in information-dense scenarios, even at shorter context lengths. We also characterize a phenomenon termed 'under-thinking', where models prematurely conclude reasoning despite available information. NeedleBench thus provides critical insights and targeted tools essential for evaluating and improving LLMs' long-context capabilities. All resources are available at OpenCompass: https://github.com/open-compass/opencompass.
>
---
#### [replaced 035] Sarc7: Evaluating Sarcasm Detection and Generation with Seven Types and Emotion-Informed Techniques
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.00658v3](http://arxiv.org/pdf/2506.00658v3)**

> **作者:** Lang Xiong; Raina Gao; Alyssa Jeong; Yicheng Fu; Sean O'Brien; Vasu Sharma; Kevin Zhu
>
> **备注:** Accepted to EMNLP WiNLP and COLM Melt, Solar, PragLM, and Origen
>
> **摘要:** Sarcasm is a form of humor where expressions convey meanings opposite to their literal interpretations. Classifying and generating sarcasm using large language models is vital for interpreting human communication. Sarcasm poses challenges for computational models, due to its nuanced nature. We introduce Sarc7, a benchmark that classifies 7 types of sarcasm: self-deprecating, brooding, deadpan, polite, obnoxious, raging, and manic by annotating entries of the MUStARD dataset. Classification was evaluated using zero-shot, few-shot, chain-of-thought (CoT), and a novel emotion-based prompting technique. We propose an emotion-based generation method developed by identifying key components of sarcasm-incongruity, shock value, and context dependency. Our classification experiments show that Gemini 2.5, using emotion-based prompting, outperforms other setups with an F1 score of 0.3664. Human evaluators preferred our emotion-based prompting, with 38.46% more successful generations than zero-shot prompting.
>
---
#### [replaced 036] Enhancing the De-identification of Personally Identifiable Information in Educational Data
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.09765v2](http://arxiv.org/pdf/2501.09765v2)**

> **作者:** Zilyu Ji; Yuntian Shen; Jionghao Lin; Kenneth R. Koedinger
>
> **摘要:** Protecting Personally Identifiable Information (PII), such as names, is a critical requirement in learning technologies to safeguard student and teacher privacy and maintain trust. Accurate PII detection is an essential step toward anonymizing sensitive information while preserving the utility of educational data. Motivated by recent advancements in artificial intelligence, our study investigates the GPT-4o-mini model as a cost-effective and efficient solution for PII detection tasks. We explore both prompting and fine-tuning approaches and compare GPT-4o-mini's performance against established frameworks, including Microsoft Presidio and Azure AI Language. Our evaluation on two public datasets, CRAPII and TSCC, demonstrates that the fine-tuned GPT-4o-mini model achieves superior performance, with a recall of 0.9589 on CRAPII. Additionally, fine-tuned GPT-4o-mini significantly improves precision scores (a threefold increase) while reducing computational costs to nearly one-tenth of those associated with Azure AI Language. Furthermore, our bias analysis reveals that the fine-tuned GPT-4o-mini model consistently delivers accurate results across diverse cultural backgrounds and genders. The generalizability analysis using the TSCC dataset further highlights its robustness, achieving a recall of 0.9895 with minimal additional training data from TSCC. These results emphasize the potential of fine-tuned GPT-4o-mini as an accurate and cost-effective tool for PII detection in educational data. It offers robust privacy protection while preserving the data's utility for research and pedagogical analysis. Our code is available on GitHub: https://github.com/AnonJD/PrivacyAI
>
---
#### [replaced 037] Puzzled by Puzzles: When Vision-Language Models Can't Take a Hint
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.23759v2](http://arxiv.org/pdf/2505.23759v2)**

> **作者:** Heekyung Lee; Jiaxin Ge; Tsung-Han Wu; Minwoo Kang; Trevor Darrell; David M. Chan
>
> **备注:** EMNLP 2025 Main Conference
>
> **摘要:** Rebus puzzles, visual riddles that encode language through imagery, spatial arrangement, and symbolic substitution, pose a unique challenge to current vision-language models (VLMs). Unlike traditional image captioning or question answering tasks, rebus solving requires multi-modal abstraction, symbolic reasoning, and a grasp of cultural, phonetic and linguistic puns. In this paper, we investigate the capacity of contemporary VLMs to interpret and solve rebus puzzles by constructing a hand-generated and annotated benchmark of diverse English-language rebus puzzles, ranging from simple pictographic substitutions to spatially-dependent cues ("head" over "heels"). We analyze how different VLMs perform, and our findings reveal that while VLMs exhibit some surprising capabilities in decoding simple visual clues, they struggle significantly with tasks requiring abstract reasoning, lateral thinking, and understanding visual metaphors.
>
---
#### [replaced 038] Posterior-GRPO: Rewarding Reasoning Processes in Code Generation
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.05170v2](http://arxiv.org/pdf/2508.05170v2)**

> **作者:** Lishui Fan; Yu Zhang; Mouxiang Chen; Zhongxin Liu
>
> **摘要:** Reinforcement learning (RL) has significantly advanced code generation for large language models (LLMs). However, current paradigms rely on outcome-based rewards from test cases, neglecting the quality of the intermediate reasoning process. While supervising the reasoning process directly is a promising direction, it is highly susceptible to reward hacking, where the policy model learns to exploit the reasoning reward signal without improving final outcomes. To address this, we introduce a unified framework that can effectively incorporate the quality of the reasoning process during RL. First, to enable reasoning evaluation, we develop LCB-RB, a benchmark comprising preference pairs of superior and inferior reasoning processes. Second, to accurately score reasoning quality, we introduce an Optimized-Degraded based (OD-based) method for reward model training. This method generates high-quality preference pairs by systematically optimizing and degrading initial reasoning paths along curated dimensions of reasoning quality, such as factual accuracy, logical rigor, and coherence. A 7B parameter reward model with this method achieves state-of-the-art (SOTA) performance on LCB-RB and generalizes well to other benchmarks. Finally, we introduce Posterior-GRPO (P-GRPO), a novel RL method that conditions process-based rewards on task success. By selectively applying rewards to the reasoning processes of only successful outcomes, P-GRPO effectively mitigates reward hacking and aligns the model's internal reasoning with final code correctness. A 7B parameter model with P-GRPO achieves superior performance across diverse code generation tasks, outperforming outcome-only baselines by 4.5%, achieving comparable performance to GPT-4-Turbo. We further demonstrate the generalizability of our approach by extending it to mathematical tasks. Our models, dataset, and code are publicly available.
>
---
#### [replaced 039] LogiDynamics: Unraveling the Dynamics of Inductive, Abductive and Deductive Logical Inferences in LLM Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11176v4](http://arxiv.org/pdf/2502.11176v4)**

> **作者:** Tianshi Zheng; Jiayang Cheng; Chunyang Li; Haochen Shi; Zihao Wang; Jiaxin Bai; Yangqiu Song; Ginny Y. Wong; Simon See
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Modern large language models (LLMs) employ diverse logical inference mechanisms for reasoning, making the strategic optimization of these approaches critical for advancing their capabilities. This paper systematically investigate the comparative dynamics of inductive (System 1) versus abductive/deductive (System 2) inference in LLMs. We utilize a controlled analogical reasoning environment, varying modality (textual, visual, symbolic), difficulty, and task format (MCQ / free-text). Our analysis reveals System 2 pipelines generally excel, particularly in visual/symbolic modalities and harder tasks, while System 1 is competitive for textual and easier problems. Crucially, task format significantly influences their relative advantage, with System 1 sometimes outperforming System 2 in free-text rule-execution. These core findings generalize to broader in-context learning. Furthermore, we demonstrate that advanced System 2 strategies like hypothesis selection and iterative refinement can substantially scale LLM reasoning. This study offers foundational insights and actionable guidelines for strategically deploying logical inference to enhance LLM reasoning. Resources are available at https://github.com/HKUST-KnowComp/LogiDynamics.
>
---
#### [replaced 040] Privately Learning from Graphs with Applications in Fine-tuning Large Language Models
- **分类: cs.LG; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2410.08299v2](http://arxiv.org/pdf/2410.08299v2)**

> **作者:** Haoteng Yin; Rongzhe Wei; Eli Chien; Pan Li
>
> **备注:** Accepted by COLM 2025
>
> **摘要:** Graphs offer unique insights into relationships between entities, complementing data modalities like text and images and enabling AI models to extend their capabilities beyond traditional tasks. However, learning from graphs often involves handling sensitive relationships in the data, raising significant privacy concerns. Existing privacy-preserving methods, such as DP-SGD, rely on gradient decoupling assumptions and are incompatible with relational learning due to the inherent dependencies between training samples. To address this challenge, we propose a privacy-preserving pipeline for relational learning that decouples dependencies in sampled relations for training, ensuring differential privacy through a tailored application of DP-SGD. We apply this approach to fine-tune large language models (LLMs), such as Llama2, on sensitive graph data while addressing the associated computational complexities. Our method is evaluated on four real-world text-attributed graphs, demonstrating significant improvements in relational learning tasks while maintaining robust privacy guarantees. Additionally, we analyze the trade-offs between privacy, utility, and computational efficiency, offering insights into the practical deployment of our approach for privacy-preserving relational learning. Code is available at https://github.com/Graph-COM/PvGaLM.
>
---
#### [replaced 041] Human-in-the-Loop Generation of Adversarial Texts: A Case Study on Tibetan Script
- **分类: cs.CL; cs.CR; cs.HC**

- **链接: [http://arxiv.org/pdf/2412.12478v4](http://arxiv.org/pdf/2412.12478v4)**

> **作者:** Xi Cao; Yuan Sun; Jiajun Li; Quzong Gesang; Nuo Qun; Tashi Nyima
>
> **摘要:** DNN-based language models excel across various NLP tasks but remain highly vulnerable to textual adversarial attacks. While adversarial text generation is crucial for NLP security, explainability, evaluation, and data augmentation, related work remains overwhelmingly English-centric, leaving the problem of constructing high-quality and sustainable adversarial robustness benchmarks for lower-resourced languages both difficult and understudied. First, method customization for lower-resourced languages is complicated due to linguistic differences and limited resources. Second, automated attacks are prone to generating invalid or ambiguous adversarial texts. Last but not least, language models continuously evolve and may be immune to parts of previously generated adversarial texts. To address these challenges, we introduce HITL-GAT, an interactive system based on a general approach to human-in-the-loop generation of adversarial texts. Additionally, we demonstrate the utility of HITL-GAT through a case study on Tibetan script, employing three customized adversarial text generation methods and establishing its first adversarial robustness benchmark, providing a valuable reference for other lower-resourced languages.
>
---
#### [replaced 042] How Does Cognitive Bias Affect Large Language Models? A Case Study on the Anchoring Effect in Price Negotiation Simulations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.21137v2](http://arxiv.org/pdf/2508.21137v2)**

> **作者:** Yoshiki Takenami; Yin Jou Huang; Yugo Murawaki; Chenhui Chu
>
> **备注:** 18 pages, 2 figures. Accepted to EMNLP 2025 findings
>
> **摘要:** Cognitive biases, well-studied in humans, can also be observed in LLMs, affecting their reliability in real-world applications. This paper investigates the anchoring effect in LLM-driven price negotiations. To this end, we instructed seller LLM agents to apply the anchoring effect and evaluated negotiations using not only an objective metric but also a subjective metric. Experimental results show that LLMs are influenced by the anchoring effect like humans. Additionally, we investigated the relationship between the anchoring effect and factors such as reasoning and personality. It was shown that reasoning models are less prone to the anchoring effect, suggesting that the long chain of thought mitigates the effect. However, we found no significant correlation between personality traits and susceptibility to the anchoring effect. These findings contribute to a deeper understanding of cognitive biases in LLMs and to the realization of safe and responsible application of LLMs in society.
>
---
#### [replaced 043] SCRum-9: Multilingual Stance Classification over Rumours on Social Media
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18916v2](http://arxiv.org/pdf/2505.18916v2)**

> **作者:** Yue Li; Jake Vasilakes; Zhixue Zhao; Carolina Scarton
>
> **摘要:** We introduce SCRum-9, the largest multilingual Stance Classification dataset for Rumour analysis in 9 languages, containing 7,516 tweets from X. SCRum-9 goes beyond existing stance classification datasets by covering more languages, linking examples to more fact-checked claims (2.1k), and including confidence-related annotations from multiple annotators to account for intra- and inter-annotator variability. Annotations were made by at least two native speakers per language, totalling more than 405 hours of annotation and 8,150 dollars in compensation. Further, SCRum-9 is used to benchmark five large language models (LLMs) and two multilingual masked language models (MLMs) in In-Context Learning (ICL) and fine-tuning setups. This paper also innovates by exploring the use of multilingual synthetic data for rumour stance classification, showing that even LLMs with weak ICL performance can produce valuable synthetic data for fine-tuning small MLMs, enabling them to achieve higher performance than zero-shot ICL in LLMs. Finally, we examine the relationship between model predictions and human uncertainty on ambiguous cases finding that model predictions often match the second-choice labels assigned by annotators, rather than diverging entirely from human judgments. SCRum-9 is publicly released to the research community with potential to foster further research on multilingual analysis of misleading narratives on social media.
>
---
#### [replaced 044] What's Not Said Still Hurts: A Description-Based Evaluation Framework for Measuring Social Bias in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.19749v2](http://arxiv.org/pdf/2502.19749v2)**

> **作者:** Jinhao Pan; Chahat Raj; Ziyu Yao; Ziwei Zhu
>
> **备注:** EMNLP Findings 2025
>
> **摘要:** Large Language Models (LLMs) often exhibit social biases inherited from their training data. While existing benchmarks evaluate bias by term-based mode through direct term associations between demographic terms and bias terms, LLMs have become increasingly adept at avoiding biased responses, leading to seemingly low levels of bias. However, biases persist in subtler, contextually hidden forms that traditional benchmarks fail to capture. We introduce the Description-based Bias Benchmark (DBB), a novel dataset designed to assess bias at the semantic level that bias concepts are hidden within naturalistic, subtly framed contexts in real-world scenarios rather than superficial terms. We analyze six state-of-the-art LLMs, revealing that while models reduce bias in response at the term level, they continue to reinforce biases in nuanced settings. Data, code, and results are available at https://github.com/JP-25/Description-based-Bias-Benchmark.
>
---
#### [replaced 045] From Automation to Autonomy: A Survey on Large Language Models in Scientific Discovery
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13259v3](http://arxiv.org/pdf/2505.13259v3)**

> **作者:** Tianshi Zheng; Zheye Deng; Hong Ting Tsang; Weiqi Wang; Jiaxin Bai; Zihao Wang; Yangqiu Song
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Large Language Models (LLMs) are catalyzing a paradigm shift in scientific discovery, evolving from task-specific automation tools into increasingly autonomous agents and fundamentally redefining research processes and human-AI collaboration. This survey systematically charts this burgeoning field, placing a central focus on the changing roles and escalating capabilities of LLMs in science. Through the lens of the scientific method, we introduce a foundational three-level taxonomy-Tool, Analyst, and Scientist-to delineate their escalating autonomy and evolving responsibilities within the research lifecycle. We further identify pivotal challenges and future research trajectories such as robotic automation, self-improvement, and ethical governance. Overall, this survey provides a conceptual architecture and strategic foresight to navigate and shape the future of AI-driven scientific discovery, fostering both rapid innovation and responsible advancement. Github Repository: https://github.com/HKUST-KnowComp/Awesome-LLM-Scientific-Discovery.
>
---
#### [replaced 046] Beyond Token Limits: Assessing Language Model Performance on Long Text Classification
- **分类: cs.CL; I.7; I.2; J.4**

- **链接: [http://arxiv.org/pdf/2509.10199v2](http://arxiv.org/pdf/2509.10199v2)**

> **作者:** Miklós Sebők; Viktor Kovács; Martin Bánóczy; Daniel Møller Eriksen; Nathalie Neptune; Philippe Roussille
>
> **摘要:** The most widely used large language models in the social sciences (such as BERT, and its derivatives, e.g. RoBERTa) have a limitation on the input text length that they can process to produce predictions. This is a particularly pressing issue for some classification tasks, where the aim is to handle long input texts. One such area deals with laws and draft laws (bills), which can have a length of multiple hundred pages and, therefore, are not particularly amenable for processing with models that can only handle e.g. 512 tokens. In this paper, we show results from experiments covering 5 languages with XLM-RoBERTa, Longformer, GPT-3.5, GPT-4 models for the multiclass classification task of the Comparative Agendas Project, which has a codebook of 21 policy topic labels from education to health care. Results show no particular advantage for the Longformer model, pre-trained specifically for the purposes of handling long inputs. The comparison between the GPT variants and the best-performing open model yielded an edge for the latter. An analysis of class-level factors points to the importance of support and substance overlaps between specific categories when it comes to performance on long text inputs.
>
---
#### [replaced 047] From n-gram to Attention: How Model Architectures Learn and Propagate Bias in Language Modeling
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.12381v2](http://arxiv.org/pdf/2505.12381v2)**

> **作者:** Mohsinul Kabir; Tasfia Tahsin; Sophia Ananiadou
>
> **备注:** Accepted at EMNLP 2025 (Findings)
>
> **摘要:** Current research on bias in language models (LMs) predominantly focuses on data quality, with significantly less attention paid to model architecture and temporal influences of data. Even more critically, few studies systematically investigate the origins of bias. We propose a methodology grounded in comparative behavioral theory to interpret the complex interaction between training data and model architecture in bias propagation during language modeling. Building on recent work that relates transformers to n-gram LMs, we evaluate how data, model design choices, and temporal dynamics affect bias propagation. Our findings reveal that: (1) n-gram LMs are highly sensitive to context window size in bias propagation, while transformers demonstrate architectural robustness; (2) the temporal provenance of training data significantly affects bias; and (3) different model architectures respond differentially to controlled bias injection, with certain biases (e.g. sexual orientation) being disproportionately amplified. As language models become ubiquitous, our findings highlight the need for a holistic approach -- tracing bias to its origins across both data and model dimensions, not just symptoms, to mitigate harm.
>
---
#### [replaced 048] Structured Preference Optimization for Vision-Language Long-Horizon Task Planning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20742v4](http://arxiv.org/pdf/2502.20742v4)**

> **作者:** Xiwen Liang; Min Lin; Weiqi Ruan; Rongtao Xu; Yuecheng Liu; Jiaqi Chen; Bingqian Lin; Yuzheng Zhuang; Xiaodan Liang
>
> **备注:** 18 pages
>
> **摘要:** Existing methods for vision-language task planning excel in short-horizon tasks but often fall short in complex, long-horizon planning within dynamic environments. These challenges primarily arise from the difficulty of effectively training models to produce high-quality reasoning processes for long-horizon tasks. To address this, we propose Structured Preference Optimization (SPO), which aims to enhance reasoning and action selection in long-horizon task planning through structured preference evaluation and optimized training strategies. Specifically, SPO introduces: 1) Preference-Based Scoring and Optimization, which systematically evaluates reasoning chains based on task relevance, visual grounding, and historical consistency; and 2) Curriculum-Guided Training, where the model progressively adapts from simple to complex tasks, improving its generalization ability in long-horizon scenarios and enhancing reasoning robustness. To advance research in vision-language long-horizon task planning, we introduce ExtendaBench, a comprehensive benchmark covering 1,509 tasks across VirtualHome and Habitat 2.0, categorized into ultra-short, short, medium, and long tasks. Experimental results demonstrate that SPO significantly improves reasoning quality and final decision accuracy, outperforming prior methods on long-horizon tasks and underscoring the effectiveness of preference-driven optimization in vision-language task planning. Specifically, SPO achieves a +5.98% GCR and +4.68% SR improvement in VirtualHome and a +3.30% GCR and +2.11% SR improvement in Habitat over the best-performing baselines.
>
---
#### [replaced 049] SpeechRole: A Large-Scale Dataset and Benchmark for Evaluating Speech Role-Playing Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.02013v3](http://arxiv.org/pdf/2508.02013v3)**

> **作者:** Changhao Jiang; Jiajun Sun; Yifei Cao; Jiabao Zhuang; Hui Li; Xiaoran Fan; Ming Zhang; Junjie Ye; Shihan Dou; Zhiheng Xi; Jingqi Tong; Yilong Wu; Baoyu Fan; Zhen Wang; Tao Liang; Zhihui Fei; Mingyang Wan; Guojun Ma; Tao Ji; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Recently, role-playing agents have emerged as a promising paradigm for achieving personalized interaction and emotional resonance. Existing research primarily focuses on the textual modality, neglecting the critical dimension of speech in realistic interactive scenarios. In particular, there is a lack of systematic evaluation for Speech Role-Playing Agents (SRPAs). To address this gap, we construct SpeechRole-Data, a large-scale, high-quality dataset that comprises 98 diverse roles and 112k speech-based single-turn and multi-turn conversations. Each role demonstrates distinct vocal characteristics, including timbre and prosody, thereby enabling more sophisticated speech role-playing. Furthermore, we propose SpeechRole-Eval, a multidimensional evaluation benchmark that systematically assesses SRPAs performance in key aspects such as fundamental interaction ability, speech expressiveness, and role-playing fidelity. Experimental results reveal the advantages and challenges of both cascaded and end-to-end speech role-playing agents in maintaining vocal style consistency and role coherence. We release all data, code, and baseline models to provide a solid foundation for speech-driven multimodal role-playing research and to foster further developments in this field.
>
---
#### [replaced 050] Database-Augmented Query Representation for Information Retrieval
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2406.16013v2](http://arxiv.org/pdf/2406.16013v2)**

> **作者:** Soyeong Jeong; Jinheon Baek; Sukmin Cho; Sung Ju Hwang; Jong C. Park
>
> **备注:** EMNLP 2025
>
> **摘要:** Information retrieval models that aim to search for documents relevant to a query have shown multiple successes, which have been applied to diverse tasks. Yet, the query from the user is oftentimes short, which challenges the retrievers to correctly fetch relevant documents. To tackle this, previous studies have proposed expanding the query with a couple of additional (user-related) features related to it. However, they may be suboptimal to effectively augment the query, and there is plenty of other information available to augment it in a relational database. Motivated by this fact, we present a novel retrieval framework called Database-Augmented Query representation (DAQu), which augments the original query with various (query-related) metadata across multiple tables. In addition, as the number of features in the metadata can be very large and there is no order among them, we encode them with the graph-based set-encoding strategy, which considers hierarchies of features in the database without order. We validate our DAQu in diverse retrieval scenarios, demonstrating that it significantly enhances overall retrieval performance over relevant baselines. Our code is available at \href{https://github.com/starsuzi/DAQu}{this https URL}.
>
---
#### [replaced 051] COMI-LINGUA: Expert Annotated Large-Scale Dataset for Multitask NLP in Hindi-English Code-Mixing
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.21670v3](http://arxiv.org/pdf/2503.21670v3)**

> **作者:** Rajvee Sheth; Himanshu Beniwal; Mayank Singh
>
> **摘要:** We introduce COMI-LINGUA, the largest manually annotated Hindi-English code-mixed dataset, comprising 125K+ high-quality instances across five core NLP tasks: Matrix Language Identification, Token-level Language Identification, Part-Of-Speech Tagging, Named Entity Recognition, and Machine Translation. Each instance is annotated by three bilingual annotators, yielding over 376K expert annotations with strong inter-annotator agreement (Fleiss' Kappa $\geq$ 0.81). The rigorously preprocessed and filtered dataset covers both Devanagari and Roman scripts and spans diverse domains, ensuring real-world linguistic coverage. Evaluation reveals that closed-source LLMs significantly outperform traditional tools and open-source models in zero-shot settings. Notably, one-shot prompting consistently boosts performance across tasks, especially in structure-sensitive predictions like POS and NER. Fine-tuning state-of-the-art LLMs on COMI-LINGUA demonstrates substantial improvements, achieving up to 95.25 F1 in NER, 98.77 F1 in MLI, and competitive MT performance, setting new benchmarks for Hinglish code-mixed text. COMI-LINGUA is publicly available at this URL: https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA.
>
---
#### [replaced 052] Yet Another Watermark for Large Language Models
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.12574v2](http://arxiv.org/pdf/2509.12574v2)**

> **作者:** Siyuan Bao; Ying Shi; Zhiguang Yang; Hanzhou Wu; Xinpeng Zhang
>
> **备注:** https://scholar.google.com/citations?hl=en&user=IdiF7M0AAAAJ
>
> **摘要:** Existing watermarking methods for large language models (LLMs) mainly embed watermark by adjusting the token sampling prediction or post-processing, lacking intrinsic coupling with LLMs, which may significantly reduce the semantic quality of the generated marked texts. Traditional watermarking methods based on training or fine-tuning may be extendable to LLMs. However, most of them are limited to the white-box scenario, or very time-consuming due to the massive parameters of LLMs. In this paper, we present a new watermarking framework for LLMs, where the watermark is embedded into the LLM by manipulating the internal parameters of the LLM, and can be extracted from the generated text without accessing the LLM. Comparing with related methods, the proposed method entangles the watermark with the intrinsic parameters of the LLM, which better balances the robustness and imperceptibility of the watermark. Moreover, the proposed method enables us to extract the watermark under the black-box scenario, which is computationally efficient for use. Experimental results have also verified the feasibility, superiority and practicality. This work provides a new perspective different from mainstream works, which may shed light on future research.
>
---
#### [replaced 053] Assessing Large Language Models on Islamic Legal Reasoning: Evidence from Inheritance Law Evaluation
- **分类: cs.CL; cs.AI; I.2.6; I.2.7**

- **链接: [http://arxiv.org/pdf/2509.01081v2](http://arxiv.org/pdf/2509.01081v2)**

> **作者:** Abdessalam Bouchekif; Samer Rashwani; Heba Sbahi; Shahd Gaben; Mutaz Al-Khatib; Mohammed Ghaly
>
> **备注:** 10 pages, 7 Tables, Code: https://github.com/bouchekif/inheritance_evaluation
>
> **摘要:** This paper evaluates the knowledge and reasoning capabilities of Large Language Models in Islamic inheritance law, known as 'ilm al-mawarith. We assess the performance of seven LLMs using a benchmark of 1,000 multiple-choice questions covering diverse inheritance scenarios, designed to test models' ability to understand the inheritance context and compute the distribution of shares prescribed by Islamic jurisprudence. The results reveal a significant performance gap: o3 and Gemini 2.5 achieved accuracies above 90%, whereas ALLaM, Fanar, LLaMA, and Mistral scored below 50%. These disparities reflect important differences in reasoning ability and domain adaptation. We conduct a detailed error analysis to identify recurring failure patterns across models, including misunderstandings of inheritance scenarios, incorrect application of legal rules, and insufficient domain knowledge. Our findings highlight limitations in handling structured legal reasoning and suggest directions for improving performance in Islamic legal reasoning. Code: https://github.com/bouchekif/inheritance_evaluation
>
---
#### [replaced 054] Video-Language Critic: Transferable Reward Functions for Language-Conditioned Robotics
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.19988v3](http://arxiv.org/pdf/2405.19988v3)**

> **作者:** Minttu Alakuijala; Reginald McLean; Isaac Woungang; Nariman Farsad; Samuel Kaski; Pekka Marttinen; Kai Yuan
>
> **备注:** 14 pages in the main text, 22 pages including references and supplementary materials. 3 figures and 3 tables in the main text, 6 figures and 3 tables in supplementary materials
>
> **摘要:** Natural language is often the easiest and most convenient modality for humans to specify tasks for robots. However, learning to ground language to behavior typically requires impractical amounts of diverse, language-annotated demonstrations collected on each target robot. In this work, we aim to separate the problem of what to accomplish from how to accomplish it, as the former can benefit from substantial amounts of external observation-only data, and only the latter depends on a specific robot embodiment. To this end, we propose Video-Language Critic, a reward model that can be trained on readily available cross-embodiment data using contrastive learning and a temporal ranking objective, and use it to score behavior traces from a separate actor. When trained on Open X-Embodiment data, our reward model enables 2x more sample-efficient policy training on Meta-World tasks than a sparse reward only, despite a significant domain gap. Using in-domain data but in a challenging task generalization setting on Meta-World, we further demonstrate more sample-efficient training than is possible with prior language-conditioned reward models that are either trained with binary classification, use static images, or do not leverage the temporal information present in video data.
>
---
#### [replaced 055] Position Bias Mitigates Position Bias:Mitigate Position Bias Through Inter-Position Knowledge Distillation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.15709v2](http://arxiv.org/pdf/2508.15709v2)**

> **作者:** Yifei Wang; Feng Xiong; Yong Wang; Linjing Li; Xiangxiang Chu; Daniel Dajun Zeng
>
> **备注:** EMNLP 2025 Oral
>
> **摘要:** Positional bias (PB), manifesting as non-uniform sensitivity across different contextual locations, significantly impairs long-context comprehension and processing capabilities. Previous studies have addressed PB either by modifying the underlying architectures or by employing extensive contextual awareness training. However, the former approach fails to effectively eliminate the substantial performance disparities, while the latter imposes significant data and computational overhead. To address PB effectively, we introduce \textbf{Pos2Distill}, a position to position knowledge distillation framework. Pos2Distill transfers the superior capabilities from advantageous positions to less favorable ones, thereby reducing the huge performance gaps. The conceptual principle is to leverage the inherent, position-induced disparity to counteract the PB itself. We identify distinct manifestations of PB under \textbf{\textsc{r}}etrieval and \textbf{\textsc{r}}easoning paradigms, thereby designing two specialized instantiations: \emph{Pos2Distill-R\textsuperscript{1}} and \emph{Pos2Distill-R\textsuperscript{2}} respectively, both grounded in this core principle. By employing the Pos2Distill approach, we achieve enhanced uniformity and significant performance gains across all contextual positions in long-context retrieval and reasoning tasks. Crucially, both specialized systems exhibit strong cross-task generalization mutually, while achieving superior performance on their respective tasks.
>
---
#### [replaced 056] Self-Guided Function Calling in Large Language Models via Stepwise Experience Recall
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.15214v2](http://arxiv.org/pdf/2508.15214v2)**

> **作者:** Sijia Cui; Aiyao He; Shuai Xu; Hongming Zhang; Yanna Wang; Qingyang Zhang; Yajing Wang; Bo Xu
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Function calling enables large language models (LLMs) to interact with external systems by leveraging tools and APIs. When faced with multi-step tool usage, LLMs still struggle with tool selection, parameter generation, and tool-chain planning. Existing methods typically rely on manually designing task-specific demonstrations, or retrieving from a curated library. These approaches demand substantial expert effort and prompt engineering becomes increasingly complex and inefficient as tool diversity and task difficulty scale. To address these challenges, we propose a self-guided method, Stepwise Experience Recall (SEER), which performs fine-grained, stepwise retrieval from a continually updated experience pool. Instead of relying on static or manually curated library, SEER incrementally augments the experience pool with past successful trajectories, enabling continuous expansion of the pool and improved model performance over time. Evaluated on the ToolQA benchmark, SEER achieves an average improvement of 6.1% on easy and 4.7% on hard questions. We further test SEER on $\tau$-bench, which includes two real-world domains. Powered by Qwen2.5-7B and Qwen2.5-72B models, SEER demonstrates substantial accuracy gains of 7.44% and 23.38%, respectively.
>
---
#### [replaced 057] IntrEx: A Dataset for Modeling Engagement in Educational Conversations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.06652v2](http://arxiv.org/pdf/2509.06652v2)**

> **作者:** Xingwei Tan; Mahathi Parvatham; Chiara Gambi; Gabriele Pergola
>
> **备注:** EMNLP 2025 Findings camera-ready, 9+7 pages
>
> **摘要:** Engagement and motivation are crucial for second-language acquisition, yet maintaining learner interest in educational conversations remains a challenge. While prior research has explored what makes educational texts interesting, still little is known about the linguistic features that drive engagement in conversations. To address this gap, we introduce IntrEx, the first large dataset annotated for interestingness and expected interestingness in teacher-student interactions. Built upon the Teacher-Student Chatroom Corpus (TSCC), IntrEx extends prior work by incorporating sequence-level annotations, allowing for the study of engagement beyond isolated turns to capture how interest evolves over extended dialogues. We employ a rigorous annotation process with over 100 second-language learners, using a comparison-based rating approach inspired by reinforcement learning from human feedback (RLHF) to improve agreement. We investigate whether large language models (LLMs) can predict human interestingness judgments. We find that LLMs (7B/8B parameters) fine-tuned on interestingness ratings outperform larger proprietary models like GPT-4o, demonstrating the potential for specialised datasets to model engagement in educational settings. Finally, we analyze how linguistic and cognitive factors, such as concreteness, comprehensibility (readability), and uptake, influence engagement in educational dialogues.
>
---
#### [replaced 058] Large Language Models for Information Retrieval: A Survey
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2308.07107v5](http://arxiv.org/pdf/2308.07107v5)**

> **作者:** Yutao Zhu; Huaying Yuan; Shuting Wang; Jiongnan Liu; Wenhan Liu; Chenlong Deng; Haonan Chen; Zheng Liu; Zhicheng Dou; Ji-Rong Wen
>
> **备注:** Updated to version 4; Accepted by ACM TOIS
>
> **摘要:** As a primary means of information acquisition, information retrieval (IR) systems, such as search engines, have integrated themselves into our daily lives. These systems also serve as components of dialogue, question-answering, and recommender systems. The trajectory of IR has evolved dynamically from its origins in term-based methods to its integration with advanced neural models. While the neural models excel at capturing complex contextual signals and semantic nuances, thereby reshaping the IR landscape, they still face challenges such as data scarcity, interpretability, and the generation of contextually plausible yet potentially inaccurate responses. This evolution requires a combination of both traditional methods (such as term-based sparse retrieval methods with rapid response) and modern neural architectures (such as language models with powerful language understanding capacity). Meanwhile, the emergence of large language models (LLMs), typified by ChatGPT and GPT-4, has revolutionized natural language processing due to their remarkable language understanding, generation, generalization, and reasoning abilities. Consequently, recent research has sought to leverage LLMs to improve IR systems. Given the rapid evolution of this research trajectory, it is necessary to consolidate existing methodologies and provide nuanced insights through a comprehensive overview. In this survey, we delve into the confluence of LLMs and IR systems, including crucial aspects such as query rewriters, retrievers, rerankers, and readers. Additionally, we explore promising directions, such as search agents, within this expanding field.
>
---
#### [replaced 059] Contextual modulation of language comprehension in a dynamic neural model of lexical meaning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.14701v3](http://arxiv.org/pdf/2407.14701v3)**

> **作者:** Michael C. Stern; Maria M. Piñango
>
> **摘要:** We computationally implement and experimentally test the behavioral predictions of a dynamic neural model of lexical meaning in the framework of Dynamic Field Theory. We demonstrate the architecture and behavior of the model using as a test case the English lexical item have, focusing on its polysemous use. In the model, have maps to a semantic space defined by two independently motivated continuous conceptual dimensions, connectedness and control asymmetry. The mapping is modeled as coupling between a neural node representing the lexical item and neural fields representing the conceptual dimensions. While lexical knowledge is modeled as a stable coupling pattern, real-time lexical meaning retrieval is modeled as the motion of neural activation patterns between transiently stable states corresponding to semantic interpretations or readings. Model simulations capture two previously reported empirical observations: (1) contextual modulation of lexical semantic interpretation, and (2) individual variation in the magnitude of this modulation. Simulations also generate a novel prediction that the by-trial relationship between sentence reading time and acceptability should be contextually modulated. An experiment combining self-paced reading and acceptability judgments replicates previous results and partially bears out the model's novel prediction. Altogether, results support a novel perspective on lexical polysemy: that the many related meanings of a word are not categorically distinct representations; rather, they are transiently stable neural activation states that arise from the nonlinear dynamics of neural populations governing interpretation on continuous semantic dimensions. Our model offers important advantages over related models in the dynamical systems framework, as well as models based on Bayesian inference.
>
---
#### [replaced 060] Forget What You Know about LLMs Evaluations -- LLMs are Like a Chameleon
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.07445v2](http://arxiv.org/pdf/2502.07445v2)**

> **作者:** Nurit Cohen-Inger; Yehonatan Elisha; Bracha Shapira; Lior Rokach; Seffi Cohen
>
> **摘要:** Large language models (LLMs) often appear to excel on public benchmarks, but these high scores may mask an overreliance on dataset-specific surface cues rather than true language understanding. We introduce the Chameleon Benchmark Overfit Detector (C-BOD), a meta-evaluation framework that systematically distorts benchmark prompts via a parametric transformation and detects overfitting of LLMs. By rephrasing inputs while preserving their semantic content and labels, C-BOD exposes whether a model's performance is driven by memorized patterns. Evaluated on the MMLU benchmark using 26 leading LLMs, our method reveals an average performance degradation of 2.15% under modest perturbations, with 20 out of 26 models exhibiting statistically significant differences. Notably, models with higher baseline accuracy exhibit larger performance differences under perturbation, and larger LLMs tend to be more sensitive to rephrasings, indicating that both cases may overrely on fixed prompt patterns. In contrast, the Llama family and models with lower baseline accuracy show insignificant degradation, suggesting reduced dependency on superficial cues. Moreover, C-BOD's dataset- and model-agnostic design allows easy integration into training pipelines to promote more robust language understanding. Our findings challenge the community to look beyond leaderboard scores and prioritize resilience and generalization in LLM evaluation.
>
---
#### [replaced 061] ClonEval: An Open Voice Cloning Benchmark
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.20581v3](http://arxiv.org/pdf/2504.20581v3)**

> **作者:** Iwona Christop; Tomasz Kuczyński; Marek Kubis
>
> **备注:** Under review at ICASSP
>
> **摘要:** We present a novel benchmark for voice cloning text-to-speech models. The benchmark consists of an evaluation protocol, an open-source library for assessing the performance of voice cloning models, and an accompanying leaderboard. The paper discusses design considerations and presents a detailed description of the evaluation procedure. The usage of the software library is explained, along with the organization of results on the leaderboard.
>
---
#### [replaced 062] Benchmarking Large Language Models for Cryptanalysis and Side-Channel Vulnerabilities
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.24621v2](http://arxiv.org/pdf/2505.24621v2)**

> **作者:** Utsav Maskey; Chencheng Zhu; Usman Naseem
>
> **备注:** EMNLP'25 Findings
>
> **摘要:** Recent advancements in large language models (LLMs) have transformed natural language understanding and generation, leading to extensive benchmarking across diverse tasks. However, cryptanalysis - a critical area for data security and its connection to LLMs' generalization abilities - remains underexplored in LLM evaluations. To address this gap, we evaluate the cryptanalytic potential of state-of-the-art LLMs on ciphertexts produced by a range of cryptographic algorithms. We introduce a benchmark dataset of diverse plaintexts, spanning multiple domains, lengths, writing styles, and topics, paired with their encrypted versions. Using zero-shot and few-shot settings along with chain-of-thought prompting, we assess LLMs' decryption success rate and discuss their comprehension abilities. Our findings reveal key insights into LLMs' strengths and limitations in side-channel scenarios and raise concerns about their susceptibility to under-generalization-related attacks. This research highlights the dual-use nature of LLMs in security contexts and contributes to the ongoing discussion on AI safety and security.
>
---
#### [replaced 063] T2R-bench: A Benchmark for Generating Article-Level Reports from Real World Industrial Tables
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.19813v3](http://arxiv.org/pdf/2508.19813v3)**

> **作者:** Jie Zhang; Changzai Pan; Kaiwen Wei; Sishi Xiong; Yu Zhao; Xiangyu Li; Jiaxin Peng; Xiaoyan Gu; Jian Yang; Wenhan Chang; Zhenhe Wu; Jiang Zhong; Shuangyong Song; Yongxiang Li; Xuelong Li
>
> **摘要:** Extensive research has been conducted to explore the capabilities of large language models (LLMs) in table reasoning. However, the essential task of transforming tables information into reports remains a significant challenge for industrial applications. This task is plagued by two critical issues: 1) the complexity and diversity of tables lead to suboptimal reasoning outcomes; and 2) existing table benchmarks lack the capacity to adequately assess the practical application of this task. To fill this gap, we propose the table-to-report task and construct a bilingual benchmark named T2R-bench, where the key information flow from the tables to the reports for this task. The benchmark comprises 457 industrial tables, all derived from real-world scenarios and encompassing 19 industry domains as well as 4 types of industrial tables. Furthermore, we propose an evaluation criteria to fairly measure the quality of report generation. The experiments on 25 widely-used LLMs reveal that even state-of-the-art models like Deepseek-R1 only achieves performance with 62.71 overall score, indicating that LLMs still have room for improvement on T2R-bench.
>
---
#### [replaced 064] Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.01872v5](http://arxiv.org/pdf/2501.01872v5)**

> **作者:** Rachneet Sachdeva; Rima Hazra; Iryna Gurevych
>
> **备注:** Accepted at EMNLP 2025 (Main)
>
> **摘要:** Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.
>
---
#### [replaced 065] Uni-cot: Towards Unified Chain-of-Thought Reasoning Across Text and Vision
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.05606v2](http://arxiv.org/pdf/2508.05606v2)**

> **作者:** Luozheng Qin; Jia Gong; Yuqing Sun; Tianjiao Li; Mengping Yang; Xiaomeng Yang; Chao Qu; Zhiyu Tan; Hao Li
>
> **备注:** Project Page: https://sais-fuxi.github.io/projects/uni-cot/
>
> **摘要:** Chain-of-Thought (CoT) reasoning has been widely adopted to enhance Large Language Models (LLMs) by decomposing complex tasks into simpler, sequential subtasks. However, extending CoT to vision-language reasoning tasks remains challenging, as it often requires interpreting transitions of visual states to support reasoning. Existing methods often struggle with this due to limited capacity of modeling visual state transitions or incoherent visual trajectories caused by fragmented architectures. To overcome these limitations, we propose Uni-CoT, a Unified Chain-of-Thought framework that enables coherent and grounded multimodal reasoning within a single unified model. The key idea is to leverage a model capable of both image understanding and generation to reason over visual content and model evolving visual states. However, empowering a unified model to achieve that is non-trivial, given the high computational cost and the burden of training. To address this, Uni-CoT introduces a novel two-level reasoning paradigm: A Macro-Level CoT for high-level task planning and A Micro-Level CoT for subtask execution. This design significantly reduces the computational overhead. Furthermore, we introduce a structured training paradigm that combines interleaved image-text supervision for macro-level CoT with multi-task objectives for micro-level CoT. Together, these innovations allow Uni-CoT to perform scalable and coherent multi-modal reasoning. Furthermore, thanks to our design, all experiments can be efficiently completed using only 8 A100 GPUs with 80GB VRAM each. Experimental results on reasoning-driven image generation benchmark (WISE) and editing benchmarks (RISE and KRIS) indicates that Uni-CoT demonstrates SOTA performance and strong generalization, establishing Uni-CoT as a promising solution for multi-modal reasoning. Project Page and Code: https://sais-fuxi.github.io/projects/uni-cot/
>
---
#### [replaced 066] Annotation-Efficient Language Model Alignment via Diverse and Representative Response Texts
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.13541v2](http://arxiv.org/pdf/2405.13541v2)**

> **作者:** Yuu Jinnai; Ukyo Honda
>
> **备注:** EMNLP Findings, 2025
>
> **摘要:** Preference optimization is a standard approach to fine-tuning large language models to align with human preferences. The quantity, diversity, and representativeness of the preference dataset are critical to the effectiveness of preference optimization. However, obtaining a large amount of preference annotations is difficult in many applications. This raises the question of how to use the limited annotation budget to create an effective preference dataset. To this end, we propose Annotation-Efficient Preference Optimization (AEPO). Instead of exhaustively annotating preference over all available response texts, AEPO selects a subset of responses that maximizes diversity and representativeness from the available responses and then annotates preference over the selected ones. In this way, AEPO focuses the annotation budget on labeling preferences over a smaller but informative subset of responses. We evaluate the performance of preference learning using AEPO on three datasets and show that it outperforms the baselines with the same annotation budget. Our code is available at https://github.com/CyberAgentAILab/annotation-efficient-po
>
---

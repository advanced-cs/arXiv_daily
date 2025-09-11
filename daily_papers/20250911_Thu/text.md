# 自然语言处理 cs.CL

- **最新发布 42 篇**

- **更新 43 篇**

## 最新发布

#### [new 001] Adversarial Attacks Against Automated Fact-Checking: A Survey
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文综述对抗攻击对自动事实核查系统的影响，分析攻击方法与防御策略，旨在提升系统鲁棒性与准确性。**

- **链接: [http://arxiv.org/pdf/2509.08463v1](http://arxiv.org/pdf/2509.08463v1)**

> **作者:** Fanzhen Liu; Alsharif Abuadbba; Kristen Moore; Surya Nepal; Cecile Paris; Jia Wu; Jian Yang; Quan Z. Sheng
>
> **备注:** Accepted to the Main Conference of EMNLP 2025. Resources are available at https://github.com/FanzhenLiu/Awesome-Automated-Fact-Checking-Attacks
>
> **摘要:** In an era where misinformation spreads freely, fact-checking (FC) plays a crucial role in verifying claims and promoting reliable information. While automated fact-checking (AFC) has advanced significantly, existing systems remain vulnerable to adversarial attacks that manipulate or generate claims, evidence, or claim-evidence pairs. These attacks can distort the truth, mislead decision-makers, and ultimately undermine the reliability of FC models. Despite growing research interest in adversarial attacks against AFC systems, a comprehensive, holistic overview of key challenges remains lacking. These challenges include understanding attack strategies, assessing the resilience of current models, and identifying ways to enhance robustness. This survey provides the first in-depth review of adversarial attacks targeting FC, categorizing existing attack methodologies and evaluating their impact on AFC systems. Additionally, we examine recent advancements in adversary-aware defenses and highlight open research questions that require further exploration. Our findings underscore the urgent need for resilient FC frameworks capable of withstanding adversarial manipulations in pursuit of preserving high verification accuracy.
>
---
#### [new 002] Too Helpful, Too Harmless, Too Honest or Just Right?
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的模型对齐任务，旨在解决大语言模型在帮助性、无害性和诚实性上的平衡问题。提出TrinityX框架，结合校准专家混合机制，提升对齐效果与效率。**

- **链接: [http://arxiv.org/pdf/2509.08486v1](http://arxiv.org/pdf/2509.08486v1)**

> **作者:** Gautam Siddharth Kashyap; Mark Dras; Usman Naseem
>
> **备注:** EMNLP'25 Main
>
> **摘要:** Large Language Models (LLMs) exhibit strong performance across a wide range of NLP tasks, yet aligning their outputs with the principles of Helpfulness, Harmlessness, and Honesty (HHH) remains a persistent challenge. Existing methods often optimize for individual alignment dimensions in isolation, leading to trade-offs and inconsistent behavior. While Mixture-of-Experts (MoE) architectures offer modularity, they suffer from poorly calibrated routing, limiting their effectiveness in alignment tasks. We propose TrinityX, a modular alignment framework that incorporates a Mixture of Calibrated Experts (MoCaE) within the Transformer architecture. TrinityX leverages separately trained experts for each HHH dimension, integrating their outputs through a calibrated, task-adaptive routing mechanism that combines expert signals into a unified, alignment-aware representation. Extensive experiments on three standard alignment benchmarks-Alpaca (Helpfulness), BeaverTails (Harmlessness), and TruthfulQA (Honesty)-demonstrate that TrinityX outperforms strong baselines, achieving relative improvements of 32.5% in win rate, 33.9% in safety score, and 28.4% in truthfulness. In addition, TrinityX reduces memory usage and inference latency by over 40% compared to prior MoE-based approaches. Ablation studies highlight the importance of calibrated routing, and cross-model evaluations confirm TrinityX's generalization across diverse LLM backbones.
>
---
#### [new 003] CM-Align: Consistency-based Multilingual Alignment for Large Language Models
- **分类: cs.CL**

- **简介: 论文提出CM-Align方法，解决大语言模型中英与其他语言对齐性能差距问题。通过一致性引导的数据选择策略，提升多语言偏好数据质量，从而优化多语言对齐效果。**

- **链接: [http://arxiv.org/pdf/2509.08541v1](http://arxiv.org/pdf/2509.08541v1)**

> **作者:** Xue Zhang; Yunlong Liang; Fandong Meng; Songming Zhang; Yufeng Chen; Jinan Xu; Jie Zhou
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Current large language models (LLMs) generally show a significant performance gap in alignment between English and other languages. To bridge this gap, existing research typically leverages the model's responses in English as a reference to select the best/worst responses in other languages, which are then used for Direct Preference Optimization (DPO) training. However, we argue that there are two limitations in the current methods that result in noisy multilingual preference data and further limited alignment performance: 1) Not all English responses are of high quality, and using a response with low quality may mislead the alignment for other languages. 2) Current methods usually use biased or heuristic approaches to construct multilingual preference pairs. To address these limitations, we design a consistency-based data selection method to construct high-quality multilingual preference data for improving multilingual alignment (CM-Align). Specifically, our method includes two parts: consistency-guided English reference selection and cross-lingual consistency-based multilingual preference data construction. Experimental results on three LLMs and three common tasks demonstrate the effectiveness and superiority of our method, which further indicates the necessity of constructing high-quality preference data.
>
---
#### [new 004] OTESGN:Optimal Transport Enhanced Syntactic-Semantic Graph Networks for Aspect-Based Sentiment Analysis
- **分类: cs.CL; cs.AI**

- **简介: 论文提出OTESGN模型，用于方面级情感分析任务，解决现有方法在建模复杂语义关系和噪声干扰方面的不足。通过引入语法-语义协同注意力机制和最优运输技术，提升情感信号捕捉与噪声抵抗能力，取得SOTA效果。**

- **链接: [http://arxiv.org/pdf/2509.08612v1](http://arxiv.org/pdf/2509.08612v1)**

> **作者:** Xinfeng Liao; Xuanqi Chen; Lianxi Wang; Jiahuan Yang; Zhuowei Chen; Ziying Rong
>
> **摘要:** Aspect-based sentiment analysis (ABSA) aims to identify aspect terms and determine their sentiment polarity. While dependency trees combined with contextual semantics effectively identify aspect sentiment, existing methods relying on syntax trees and aspect-aware attention struggle to model complex semantic relationships. Their dependence on linear dot-product features fails to capture nonlinear associations, allowing noisy similarity from irrelevant words to obscure key opinion terms. Motivated by Differentiable Optimal Matching, we propose the Optimal Transport Enhanced Syntactic-Semantic Graph Network (OTESGN), which introduces a Syntactic-Semantic Collaborative Attention. It comprises a Syntactic Graph-Aware Attention for mining latent syntactic dependencies and modeling global syntactic topology, as well as a Semantic Optimal Transport Attention designed to uncover fine-grained semantic alignments amidst textual noise, thereby accurately capturing sentiment signals obscured by irrelevant tokens. A Adaptive Attention Fusion module integrates these heterogeneous features, and contrastive regularization further improves robustness. Experiments demonstrate that OTESGN achieves state-of-the-art results, outperforming previous best models by +1.01% F1 on Twitter and +1.30% F1 on Laptop14 benchmarks. Ablative studies and visual analyses corroborate its efficacy in precise localization of opinion words and noise resistance.
>
---
#### [new 005] Bias after Prompting: Persistent Discrimination in Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大语言模型在提示调整下偏见的传递问题，发现偏见会通过提示保留并影响下游任务。任务是分析和减少偏见传播，方法是评估多种去偏策略的有效性，指出内在模型修正可能更有效。**

- **链接: [http://arxiv.org/pdf/2509.08146v1](http://arxiv.org/pdf/2509.08146v1)**

> **作者:** Nivedha Sivakumar; Natalie Mackraz; Samira Khorshidi; Krishna Patel; Barry-John Theobald; Luca Zappella; Nicholas Apostoloff
>
> **摘要:** A dangerous assumption that can be made from prior work on the bias transfer hypothesis (BTH) is that biases do not transfer from pre-trained large language models (LLMs) to adapted models. We invalidate this assumption by studying the BTH in causal models under prompt adaptations, as prompting is an extremely popular and accessible adaptation strategy used in real-world applications. In contrast to prior work, we find that biases can transfer through prompting and that popular prompt-based mitigation methods do not consistently prevent biases from transferring. Specifically, the correlation between intrinsic biases and those after prompt adaptation remain moderate to strong across demographics and tasks -- for example, gender (rho >= 0.94) in co-reference resolution, and age (rho >= 0.98) and religion (rho >= 0.69) in question answering. Further, we find that biases remain strongly correlated when varying few-shot composition parameters, such as sample size, stereotypical content, occupational distribution and representational balance (rho >= 0.90). We evaluate several prompt-based debiasing strategies and find that different approaches have distinct strengths, but none consistently reduce bias transfer across models, tasks or demographics. These results demonstrate that correcting bias, and potentially improving reasoning ability, in intrinsic models may prevent propagation of biases to downstream tasks.
>
---
#### [new 006] Memorization in Large Language Models in Medicine: Prevalence, Characteristics, and Implications
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大型语言模型在医学领域的记忆现象，评估其普遍性、特征及影响。通过分析不同训练场景，发现记忆内容具有有益、无用和有害三类，并提出应对策略，以提升模型实用性与安全性。**

- **链接: [http://arxiv.org/pdf/2509.08604v1](http://arxiv.org/pdf/2509.08604v1)**

> **作者:** Anran Li; Lingfei Qian; Mengmeng Du; Yu Yin; Yan Hu; Zihao Sun; Yihang Fu; Erica Stutz; Xuguang Ai; Qianqian Xie; Rui Zhu; Jimin Huang; Yifan Yang; Siru Liu; Yih-Chung Tham; Lucila Ohno-Machado; Hyunghoon Cho; Zhiyong Lu; Hua Xu; Qingyu Chen
>
> **摘要:** Large Language Models (LLMs) have demonstrated significant potential in medicine. To date, LLMs have been widely applied to tasks such as diagnostic assistance, medical question answering, and clinical information synthesis. However, a key open question remains: to what extent do LLMs memorize medical training data. In this study, we present the first comprehensive evaluation of memorization of LLMs in medicine, assessing its prevalence (how frequently it occurs), characteristics (what is memorized), volume (how much content is memorized), and potential downstream impacts (how memorization may affect medical applications). We systematically analyze common adaptation scenarios: (1) continued pretraining on medical corpora, (2) fine-tuning on standard medical benchmarks, and (3) fine-tuning on real-world clinical data, including over 13,000 unique inpatient records from Yale New Haven Health System. The results demonstrate that memorization is prevalent across all adaptation scenarios and significantly higher than reported in the general domain. Memorization affects both the development and adoption of LLMs in medicine and can be categorized into three types: beneficial (e.g., accurate recall of clinical guidelines and biomedical references), uninformative (e.g., repeated disclaimers or templated medical document language), and harmful (e.g., regeneration of dataset-specific or sensitive clinical content). Based on these findings, we offer practical recommendations to facilitate beneficial memorization that enhances domain-specific reasoning and factual accuracy, minimize uninformative memorization to promote deeper learning beyond surface-level patterns, and mitigate harmful memorization to prevent the leakage of sensitive or identifiable patient information.
>
---
#### [new 007] A Survey of Reinforcement Learning for Large Reasoning Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文综述了强化学习在大语言模型推理中的应用，探讨其挑战与发展方向，旨在推动RL在更复杂推理模型中的研究与应用。**

- **链接: [http://arxiv.org/pdf/2509.08827v1](http://arxiv.org/pdf/2509.08827v1)**

> **作者:** Kaiyan Zhang; Yuxin Zuo; Bingxiang He; Youbang Sun; Runze Liu; Che Jiang; Yuchen Fan; Kai Tian; Guoli Jia; Pengfei Li; Yu Fu; Xingtai Lv; Yuchen Zhang; Sihang Zeng; Shang Qu; Haozhan Li; Shijie Wang; Yuru Wang; Xinwei Long; Fangfu Liu; Xiang Xu; Jiaze Ma; Xuekai Zhu; Ermo Hua; Yihao Liu; Zonglin Li; Huayu Chen; Xiaoye Qu; Yafu Li; Weize Chen; Zhenzhao Yuan; Junqi Gao; Dong Li; Zhiyuan Ma; Ganqu Cui; Zhiyuan Liu; Biqing Qi; Ning Ding; Bowen Zhou
>
> **摘要:** In this paper, we survey recent advances in Reinforcement Learning (RL) for reasoning with Large Language Models (LLMs). RL has achieved remarkable success in advancing the frontier of LLM capabilities, particularly in addressing complex logical tasks such as mathematics and coding. As a result, RL has emerged as a foundational methodology for transforming LLMs into LRMs. With the rapid progress of the field, further scaling of RL for LRMs now faces foundational challenges not only in computational resources but also in algorithm design, training data, and infrastructure. To this end, it is timely to revisit the development of this domain, reassess its trajectory, and explore strategies to enhance the scalability of RL toward Artificial SuperIntelligence (ASI). In particular, we examine research applying RL to LLMs and LRMs for reasoning abilities, especially since the release of DeepSeek-R1, including foundational components, core problems, training resources, and downstream applications, to identify future opportunities and directions for this rapidly evolving area. We hope this review will promote future research on RL for broader reasoning models. Github: https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs
>
---
#### [new 008] SciGPT: A Large Language Model for Scientific Literature Understanding and Knowledge Discovery
- **分类: cs.CL**

- **简介: 该论文提出SciGPT，一种针对科学文献理解与知识发现的大型语言模型，旨在解决通用LLM在科学领域表现不足的问题。通过领域适配、高效推理机制和知识整合，SciGPT在科学任务中优于GPT-4o，推动AI辅助科研。**

- **链接: [http://arxiv.org/pdf/2509.08032v1](http://arxiv.org/pdf/2509.08032v1)**

> **作者:** Fengyu She; Nan Wang; Hongfei Wu; Ziyi Wan; Jingmian Wang; Chang Wang
>
> **摘要:** Scientific literature is growing exponentially, creating a critical bottleneck for researchers to efficiently synthesize knowledge. While general-purpose Large Language Models (LLMs) show potential in text processing, they often fail to capture scientific domain-specific nuances (e.g., technical jargon, methodological rigor) and struggle with complex scientific tasks, limiting their utility for interdisciplinary research. To address these gaps, this paper presents SciGPT, a domain-adapted foundation model for scientific literature understanding and ScienceBench, an open source benchmark tailored to evaluate scientific LLMs. Built on the Qwen3 architecture, SciGPT incorporates three key innovations: (1) low-cost domain distillation via a two-stage pipeline to balance performance and efficiency; (2) a Sparse Mixture-of-Experts (SMoE) attention mechanism that cuts memory consumption by 55\% for 32,000-token long-document reasoning; and (3) knowledge-aware adaptation integrating domain ontologies to bridge interdisciplinary knowledge gaps. Experimental results on ScienceBench show that SciGPT outperforms GPT-4o in core scientific tasks including sequence labeling, generation, and inference. It also exhibits strong robustness in unseen scientific tasks, validating its potential to facilitate AI-augmented scientific discovery.
>
---
#### [new 009] X-Teaming Evolutionary M2S: Automated Discovery of Multi-turn to Single-turn Jailbreak Templates
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出X-Teaming Evolutionary M2S框架，自动化生成和优化多轮到单轮攻击模板，解决手动编写模板效率低的问题。通过语言模型引导的进化方法，在GPT-4.1上实现44.8%的成功率，并强调结构搜索与阈值校准的重要性。**

- **链接: [http://arxiv.org/pdf/2509.08729v1](http://arxiv.org/pdf/2509.08729v1)**

> **作者:** Hyunjun Kim; Junwoo Ha; Sangyoon Yu; Haon Park
>
> **摘要:** Multi-turn-to-single-turn (M2S) compresses iterative red-teaming into one structured prompt, but prior work relied on a handful of manually written templates. We present X-Teaming Evolutionary M2S, an automated framework that discovers and optimizes M2S templates through language-model-guided evolution. The system pairs smart sampling from 12 sources with an LLM-as-judge inspired by StrongREJECT and records fully auditable logs. Maintaining selection pressure by setting the success threshold to $\theta = 0.70$, we obtain five evolutionary generations, two new template families, and 44.8% overall success (103/230) on GPT-4.1. A balanced cross-model panel of 2,500 trials (judge fixed) shows that structural gains transfer but vary by target; two models score zero at the same threshold. We also find a positive coupling between prompt length and score, motivating length-aware judging. Our results demonstrate that structure-level search is a reproducible route to stronger single-turn probes and underscore the importance of threshold calibration and cross-model evaluation. Code, configurations, and artifacts are available at https://github.com/hyunjun1121/M2S-x-teaming.
>
---
#### [new 010] Large Language Model Hacking: Quantifying the Hidden Risks of Using LLMs for Text Annotation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型（LLM）在文本标注中的潜在风险，称为“LLM hacking”。通过分析1300万条标注数据，评估不同模型和参数设置对统计结论的影响，揭示其导致错误结论的风险，并探讨缓解方法。**

- **链接: [http://arxiv.org/pdf/2509.08825v1](http://arxiv.org/pdf/2509.08825v1)**

> **作者:** Joachim Baumann; Paul Röttger; Aleksandra Urman; Albert Wendsjö; Flor Miriam Plaza-del-Arco; Johannes B. Gruber; Dirk Hovy
>
> **摘要:** Large language models (LLMs) are rapidly transforming social science research by enabling the automation of labor-intensive tasks like data annotation and text analysis. However, LLM outputs vary significantly depending on the implementation choices made by researchers (e.g., model selection, prompting strategy, or temperature settings). Such variation can introduce systematic biases and random errors, which propagate to downstream analyses and cause Type I, Type II, Type S, or Type M errors. We call this LLM hacking. We quantify the risk of LLM hacking by replicating 37 data annotation tasks from 21 published social science research studies with 18 different models. Analyzing 13 million LLM labels, we test 2,361 realistic hypotheses to measure how plausible researcher choices affect statistical conclusions. We find incorrect conclusions based on LLM-annotated data in approximately one in three hypotheses for state-of-the-art models, and in half the hypotheses for small language models. While our findings show that higher task performance and better general model capabilities reduce LLM hacking risk, even highly accurate models do not completely eliminate it. The risk of LLM hacking decreases as effect sizes increase, indicating the need for more rigorous verification of findings near significance thresholds. Our extensive analysis of LLM hacking mitigation techniques emphasizes the importance of human annotations in reducing false positive findings and improving model selection. Surprisingly, common regression estimator correction techniques are largely ineffective in reducing LLM hacking risk, as they heavily trade off Type I vs. Type II errors. Beyond accidental errors, we find that intentional LLM hacking is unacceptably simple. With few LLMs and just a handful of prompt paraphrases, anything can be presented as statistically significant.
>
---
#### [new 011] Evaluating LLMs Without Oracle Feedback: Agentic Annotation Evaluation Through Unsupervised Consistency Signals
- **分类: cs.CL**

- **简介: 论文提出一种无需人工标注的评估方法，通过学生模型与LLM协作，利用一致性信号衡量其注释质量。属于无监督评估任务，解决动态环境中缺乏人工反馈时LLM注释质量评估难题。**

- **链接: [http://arxiv.org/pdf/2509.08809v1](http://arxiv.org/pdf/2509.08809v1)**

> **作者:** Cheng Chen; Haiyan Yin; Ivor Tsang
>
> **备注:** 11 pages, 10 figures
>
> **摘要:** Large Language Models (LLMs), when paired with prompt-based tasks, have significantly reduced data annotation costs and reliance on human annotators. However, evaluating the quality of their annotations remains challenging in dynamic, unsupervised environments where oracle feedback is scarce and conventional methods fail. To address this challenge, we propose a novel agentic annotation paradigm, where a student model collaborates with a noisy teacher (the LLM) to assess and refine annotation quality without relying on oracle feedback. The student model, acting as an unsupervised feedback mechanism, employs a user preference-based majority voting strategy to evaluate the consistency of the LLM outputs. To systematically measure the reliability of LLM-generated annotations, we introduce the Consistent and Inconsistent (CAI) Ratio, a novel unsupervised evaluation metric. The CAI Ratio not only quantifies the annotation quality of the noisy teacher under limited user preferences but also plays a critical role in model selection, enabling the identification of robust LLMs in dynamic, unsupervised environments. Applied to ten open-domain NLP datasets across four LLMs, the CAI Ratio demonstrates a strong positive correlation with LLM accuracy, establishing it as an essential tool for unsupervised evaluation and model selection in real-world settings.
>
---
#### [new 012] Culturally transmitted color categories in LLMs reflect a learning bias toward efficient compression
- **分类: cs.CL**

- **简介: 论文研究大语言模型（LLMs）是否能像人类一样形成高效语义系统。通过颜色命名实验和模拟文化演化，发现LLMs在训练中展现出类似人类的信息瓶颈效率倾向，表明其具备发展出人类样语义系统的能力。**

- **链接: [http://arxiv.org/pdf/2509.08093v1](http://arxiv.org/pdf/2509.08093v1)**

> **作者:** Nathaniel Imel; Noga Zaslavsky
>
> **摘要:** Converging evidence suggests that systems of semantic categories across human languages achieve near-optimal compression via the Information Bottleneck (IB) complexity-accuracy principle. Large language models (LLMs) are not trained for this objective, which raises the question: are LLMs capable of evolving efficient human-like semantic systems? To address this question, we focus on the domain of color as a key testbed of cognitive theories of categorization and replicate with LLMs (Gemini 2.0-flash and Llama 3.3-70B-Instruct) two influential human behavioral studies. First, we conduct an English color-naming study, showing that Gemini aligns well with the naming patterns of native English speakers and achieves a significantly high IB-efficiency score, while Llama exhibits an efficient but lower complexity system compared to English. Second, to test whether LLMs simply mimic patterns in their training data or actually exhibit a human-like inductive bias toward IB-efficiency, we simulate cultural evolution of pseudo color-naming systems in LLMs via iterated in-context language learning. We find that akin to humans, LLMs iteratively restructure initially random systems towards greater IB-efficiency and increased alignment with patterns observed across the world's languages. These findings demonstrate that LLMs are capable of evolving perceptually grounded, human-like semantic systems, driven by the same fundamental principle that governs semantic efficiency across human languages.
>
---
#### [new 013] CommonVoice-SpeechRE and RPG-MoGe: Advancing Speech Relation Extraction with a New Dataset and Multi-Order Generative Framework
- **分类: cs.CL; cs.MM; cs.SD**

- **简介: 论文提出CommonVoice-SpeechRE数据集和RPG-MoGe框架，解决语音关系抽取任务中数据不足与模型性能受限问题，通过多阶生成策略和跨模态对齐提升效果。**

- **链接: [http://arxiv.org/pdf/2509.08438v1](http://arxiv.org/pdf/2509.08438v1)**

> **作者:** Jinzhong Ning; Paerhati Tulajiang; Yingying Le; Yijia Zhang; Yuanyuan Sun; Hongfei Lin; Haifeng Liu
>
> **摘要:** Speech Relation Extraction (SpeechRE) aims to extract relation triplets directly from speech. However, existing benchmark datasets rely heavily on synthetic data, lacking sufficient quantity and diversity of real human speech. Moreover, existing models also suffer from rigid single-order generation templates and weak semantic alignment, substantially limiting their performance. To address these challenges, we introduce CommonVoice-SpeechRE, a large-scale dataset comprising nearly 20,000 real-human speech samples from diverse speakers, establishing a new benchmark for SpeechRE research. Furthermore, we propose the Relation Prompt-Guided Multi-Order Generative Ensemble (RPG-MoGe), a novel framework that features: (1) a multi-order triplet generation ensemble strategy, leveraging data diversity through diverse element orders during both training and inference, and (2) CNN-based latent relation prediction heads that generate explicit relation prompts to guide cross-modal alignment and accurate triplet generation. Experiments show our approach outperforms state-of-the-art methods, providing both a benchmark dataset and an effective solution for real-world SpeechRE. The source code and dataset are publicly available at https://github.com/NingJinzhong/SpeechRE_RPG_MoGe.
>
---
#### [new 014] MVPBench: A Benchmark and Fine-Tuning Framework for Aligning Large Language Models with Diverse Human Values
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MVPBench基准与微调框架，用于评估和提升大语言模型在全球多文化、多人群中的价值观对齐能力。通过分析不同模型表现，发现对齐差异，并展示微调方法可提升效果，推动公平AI发展。**

- **链接: [http://arxiv.org/pdf/2509.08022v1](http://arxiv.org/pdf/2509.08022v1)**

> **作者:** Yao Liang; Dongcheng Zhao; Feifei Zhao; Guobin Shen; Yuwei Wang; Dongqi Liang; Yi Zeng
>
> **摘要:** The alignment of large language models (LLMs) with human values is critical for their safe and effective deployment across diverse user populations. However, existing benchmarks often neglect cultural and demographic diversity, leading to limited understanding of how value alignment generalizes globally. In this work, we introduce MVPBench, a novel benchmark that systematically evaluates LLMs' alignment with multi-dimensional human value preferences across 75 countries. MVPBench contains 24,020 high-quality instances annotated with fine-grained value labels, personalized questions, and rich demographic metadata, making it the most comprehensive resource of its kind to date. Using MVPBench, we conduct an in-depth analysis of several state-of-the-art LLMs, revealing substantial disparities in alignment performance across geographic and demographic lines. We further demonstrate that lightweight fine-tuning methods, such as Low-Rank Adaptation (LoRA) and Direct Preference Optimization (DPO), can significantly enhance value alignment in both in-domain and out-of-domain settings. Our findings underscore the necessity for population-aware alignment evaluation and provide actionable insights for building culturally adaptive and value-sensitive LLMs. MVPBench serves as a practical foundation for future research on global alignment, personalized value modeling, and equitable AI development.
>
---
#### [new 015] AntiDote: Bi-level Adversarial Training for Tamper-Resistant LLMs
- **分类: cs.CL**

- **简介: 该论文提出AntiDote方法，通过双层对抗训练提升开放权重大语言模型的防篡改能力。旨在解决恶意微调导致的安全隐患，使模型在保持性能的同时更安全。方法引入辅助对抗网络生成恶意权重，并训练主模型抵御此类攻击，实验显示其鲁棒性提升显著且性能损失极小。**

- **链接: [http://arxiv.org/pdf/2509.08000v1](http://arxiv.org/pdf/2509.08000v1)**

> **作者:** Debdeep Sanyal; Manodeep Ray; Murari Mandal
>
> **备注:** 19 pages
>
> **摘要:** The release of open-weight large language models (LLMs) creates a tension between advancing accessible research and preventing misuse, such as malicious fine-tuning to elicit harmful content. Current safety measures struggle to preserve the general capabilities of the LLM while resisting a determined adversary with full access to the model's weights and architecture, who can use full-parameter fine-tuning to erase existing safeguards. To address this, we introduce AntiDote, a bi-level optimization procedure for training LLMs to be resistant to such tampering. AntiDote involves an auxiliary adversary hypernetwork that learns to generate malicious Low-Rank Adaptation (LoRA) weights conditioned on the defender model's internal activations. The defender LLM is then trained with an objective to nullify the effect of these adversarial weight additions, forcing it to maintain its safety alignment. We validate this approach against a diverse suite of 52 red-teaming attacks, including jailbreak prompting, latent space manipulation, and direct weight-space attacks. AntiDote is upto 27.4\% more robust against adversarial attacks compared to both tamper-resistance and unlearning baselines. Crucially, this robustness is achieved with a minimal trade-off in utility, incurring a performance degradation of upto less than 0.5\% across capability benchmarks including MMLU, HellaSwag, and GSM8K. Our work offers a practical and compute efficient methodology for building open-weight models where safety is a more integral and resilient property.
>
---
#### [new 016] Balancing Quality and Variation: Spam Filtering Distorts Data Label Distributions
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究主观任务中如何平衡标注质量与标签多样性。针对现有垃圾过滤方法误删分歧标注者的问题，通过实验证明保守过滤策略更优，并指出垃圾标注者通常不随机，现有方法假设错误，需改进以兼顾准确性和多样性。**

- **链接: [http://arxiv.org/pdf/2509.08217v1](http://arxiv.org/pdf/2509.08217v1)**

> **作者:** Eve Fleisig; Matthias Orlikowski; Philipp Cimiano; Dan Klein
>
> **摘要:** For machine learning datasets to accurately represent diverse opinions in a population, they must preserve variation in data labels while filtering out spam or low-quality responses. How can we balance annotator reliability and representation? We empirically evaluate how a range of heuristics for annotator filtering affect the preservation of variation on subjective tasks. We find that these methods, designed for contexts in which variation from a single ground-truth label is considered noise, often remove annotators who disagree instead of spam annotators, introducing suboptimal tradeoffs between accuracy and label diversity. We find that conservative settings for annotator removal (<5%) are best, after which all tested methods increase the mean absolute error from the true average label. We analyze performance on synthetic spam to observe that these methods often assume spam annotators are less random than real spammers tend to be: most spammers are distributionally indistinguishable from real annotators, and the minority that are distinguishable tend to give fixed answers, not random ones. Thus, tasks requiring the preservation of variation reverse the intuition of existing spam filtering methods: spammers tend to be less random than non-spammers, so metrics that assume variation is spam fare worse. These results highlight the need for spam removal methods that account for label diversity.
>
---
#### [new 017] Towards Knowledge-Aware Document Systems: Modeling Semantic Coverage Relations via Answerability Detection
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在建模语义覆盖关系。通过问答方法识别文档间信息重叠，构建数据集并训练分类模型，验证了判别模型在语义关系判断中的优越性。**

- **链接: [http://arxiv.org/pdf/2509.08304v1](http://arxiv.org/pdf/2509.08304v1)**

> **作者:** Yehudit Aperstein; Alon Gottlib; Gal Benita; Alexander Apartsin
>
> **备注:** 27 pages, 1 figure
>
> **摘要:** Understanding how information is shared across documents, regardless of the format in which it is expressed, is critical for tasks such as information retrieval, summarization, and content alignment. In this work, we introduce a novel framework for modelling Semantic Coverage Relations (SCR), which classifies document pairs based on how their informational content aligns. We define three core relation types: equivalence, where both texts convey the same information using different textual forms or styles; inclusion, where one document fully contains the information of another and adds more; and semantic overlap, where each document presents partially overlapping content. To capture these relations, we adopt a question answering (QA)-based approach, using the answerability of shared questions across documents as an indicator of semantic coverage. We construct a synthetic dataset derived from the SQuAD corpus by paraphrasing source passages and selectively omitting information, enabling precise control over content overlap. This dataset allows us to benchmark generative language models and train transformer-based classifiers for SCR prediction. Our findings demonstrate that discriminative models significantly outperform generative approaches, with the RoBERTa-base model achieving the highest accuracy of 61.4% and the Random Forest-based model showing the best balance with a macro-F1 score of 52.9%. The results show that QA provides an effective lens for assessing semantic relations across stylistically diverse texts, offering insights into the capacity of current models to reason about information beyond surface similarity. The dataset and code developed in this study are publicly available to support reproducibility.
>
---
#### [new 018] Automatic Detection of Inauthentic Templated Responses in English Language Assessments
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AuDITR任务，旨在自动检测英语考试中使用模板作答的不真实回答。研究采用机器学习方法，并强调模型需定期更新以应对新模板。属于自然语言处理中的检测任务，解决考试作弊问题。**

- **链接: [http://arxiv.org/pdf/2509.08355v1](http://arxiv.org/pdf/2509.08355v1)**

> **作者:** Yashad Samant; Lee Becker; Scott Hellman; Bradley Behan; Sarah Hughes; Joshua Southerland
>
> **备注:** Accepted to National Council on Measurement in Education (NCME) 2025 Annual Meeting
>
> **摘要:** In high-stakes English Language Assessments, low-skill test takers may employ memorized materials called ``templates'' on essay questions to ``game'' or fool the automated scoring system. In this study, we introduce the automated detection of inauthentic, templated responses (AuDITR) task, describe a machine learning-based approach to this task and illustrate the importance of regularly updating these models in production.
>
---
#### [new 019] Verbalized Algorithms
- **分类: cs.CL**

- **简介: 论文提出“语言化算法”（VAs），将经典算法与大语言模型结合，通过分解任务为简单自然语言操作，提升LLM在排序和聚类等推理任务中的可靠性。该方法限制LLM仅处理基础操作，提高任务解决的准确性和可解释性。**

- **链接: [http://arxiv.org/pdf/2509.08150v1](http://arxiv.org/pdf/2509.08150v1)**

> **作者:** Supriya Lall; Christian Farrell; Hari Pathanjaly; Marko Pavic; Sarvesh Chezhian; Masataro Asai
>
> **备注:** Submitted to NeurIPS 2025 Workshop on Efficient Reasoning
>
> **摘要:** Instead of querying LLMs in a one-shot manner and hoping to get the right answer for a reasoning task, we propose a paradigm we call \emph{verbalized algorithms} (VAs), which leverage classical algorithms with established theoretical understanding. VAs decompose a task into simple elementary operations on natural language strings that they should be able to answer reliably, and limit the scope of LLMs to only those simple tasks. For example, for sorting a series of natural language strings, \emph{verbalized sorting} uses an LLM as a binary comparison oracle in a known and well-analyzed sorting algorithm (e.g., bitonic sorting network). We demonstrate the effectiveness of this approach on sorting and clustering tasks.
>
---
#### [new 020] No for Some, Yes for Others: Persona Prompts and Other Sources of False Refusal in Language Models
- **分类: cs.CL**

- **简介: 该论文研究语言模型中“角色提示”导致的错误拒绝问题，分析不同社会人口学特征对拒绝率的影响。通过多模型、多任务实验，发现模型能力和任务类型显著影响错误拒绝，揭示潜在偏见并提出量化方法。属于NLP中的模型安全与偏见检测任务。**

- **链接: [http://arxiv.org/pdf/2509.08075v1](http://arxiv.org/pdf/2509.08075v1)**

> **作者:** Flor Miriam Plaza-del-Arco; Paul Röttger; Nino Scherrer; Emanuele Borgonovo; Elmar Plischke; Dirk Hovy
>
> **摘要:** Large language models (LLMs) are increasingly integrated into our daily lives and personalized. However, LLM personalization might also increase unintended side effects. Recent work suggests that persona prompting can lead models to falsely refuse user requests. However, no work has fully quantified the extent of this issue. To address this gap, we measure the impact of 15 sociodemographic personas (based on gender, race, religion, and disability) on false refusal. To control for other factors, we also test 16 different models, 3 tasks (Natural Language Inference, politeness, and offensiveness classification), and nine prompt paraphrases. We propose a Monte Carlo-based method to quantify this issue in a sample-efficient manner. Our results show that as models become more capable, personas impact the refusal rate less and less. Certain sociodemographic personas increase false refusal in some models, which suggests underlying biases in the alignment strategies or safety mechanisms. However, we find that the model choice and task significantly influence false refusals, especially in sensitive content tasks. Our findings suggest that persona effects have been overestimated, and might be due to other factors.
>
---
#### [new 021] Low-Resource Fine-Tuning for Multi-Task Structured Information Extraction with a Billion-Parameter Instruction-Tuned Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究在低资源条件下，使用小规模模型进行多任务结构化信息提取。提出ETLCH模型，在少量样本下实现JSON提取、知识图谱和实体识别，效果优于大模型，降低成本。属于自然语言处理中的信息抽取任务。**

- **链接: [http://arxiv.org/pdf/2509.08381v1](http://arxiv.org/pdf/2509.08381v1)**

> **作者:** Yu Cheng Chih; Yong Hao Hou
>
> **备注:** 13 pages, 8 figures, includes experiments on JSON extraction, knowledge graph extraction, and NER
>
> **摘要:** Deploying large language models (LLMs) for structured data extraction in domains such as financial compliance reporting, legal document analytics, and multilingual knowledge base construction is often impractical for smaller teams due to the high cost of running large architectures and the difficulty of preparing large, high-quality datasets. Most recent instruction-tuning studies focus on seven-billion-parameter or larger models, leaving limited evidence on whether much smaller models can work reliably under low-resource, multi-task conditions. This work presents ETLCH, a billion-parameter LLaMA-based model fine-tuned with low-rank adaptation on only a few hundred to one thousand samples per task for JSON extraction, knowledge graph extraction, and named entity recognition. Despite its small scale, ETLCH outperforms strong baselines across most evaluation metrics, with substantial gains observed even at the lowest data scale. These findings demonstrate that well-tuned small models can deliver stable and accurate structured outputs at a fraction of the computational cost, enabling cost-effective and reliable information extraction pipelines in resource-constrained environments.
>
---
#### [new 022] Do All Autoregressive Transformers Remember Facts the Same Way? A Cross-Architecture Analysis of Recall Mechanisms
- **分类: cs.CL**

- **简介: 论文研究不同自回归Transformer模型（如GPT、LLaMA、Qwen）的事实回忆机制差异。任务是分析事实存储与检索方式，发现Qwen模型早期注意力模块比MLP更关键，揭示架构变化对事实回忆机制的影响。**

- **链接: [http://arxiv.org/pdf/2509.08778v1](http://arxiv.org/pdf/2509.08778v1)**

> **作者:** Minyeong Choe; Haehyun Cho; Changho Seo; Hyunil Kim
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** Understanding how Transformer-based language models store and retrieve factual associations is critical for improving interpretability and enabling targeted model editing. Prior work, primarily on GPT-style models, has identified MLP modules in early layers as key contributors to factual recall. However, it remains unclear whether these findings generalize across different autoregressive architectures. To address this, we conduct a comprehensive evaluation of factual recall across several models -- including GPT, LLaMA, Qwen, and DeepSeek -- analyzing where and how factual information is encoded and accessed. Consequently, we find that Qwen-based models behave differently from previous patterns: attention modules in the earliest layers contribute more to factual recall than MLP modules. Our findings suggest that even within the autoregressive Transformer family, architectural variations can lead to fundamentally different mechanisms of factual recall.
>
---
#### [new 023] MERLIN: Multi-Stage Curriculum Alignment for Multilingual Encoder and LLM Fusion
- **分类: cs.CL**

- **简介: 论文提出MERLIN框架，解决低资源语言中大语言模型推理能力不足的问题。通过多阶段课程对齐策略，融合多语言编码器与LLM，仅微调少量参数，在多个基准测试中显著提升性能。**

- **链接: [http://arxiv.org/pdf/2509.08105v1](http://arxiv.org/pdf/2509.08105v1)**

> **作者:** Kosei Uemura; David Guzmán; Quang Phuoc Nguyen; Jesujoba Oluwadara Alabi; En-shiun Annie Lee; David Ifeoluwa Adelani
>
> **备注:** under submission
>
> **摘要:** Large language models excel in English but still struggle with complex reasoning in many low-resource languages (LRLs). Existing encoder-plus-decoder methods such as LangBridge and MindMerger raise accuracy on mid and high-resource languages, yet they leave a large gap on LRLs. We present MERLIN, a two-stage model-stacking framework that applies a curriculum learning strategy -- from general bilingual bitext to task-specific data -- and adapts only a small set of DoRA weights. On the AfriMGSM benchmark MERLIN improves exact-match accuracy by +12.9 pp over MindMerger and outperforms GPT-4o-mini. It also yields consistent gains on MGSM and MSVAMP (+0.9 and +2.8 pp), demonstrating effectiveness across both low and high-resource settings.
>
---
#### [new 024] NOWJ@COLIEE 2025: A Multi-stage Framework Integrating Embedding Models and Large Language Models for Legal Retrieval and Entailment
- **分类: cs.CL; cs.AI**

- **简介: 论文提出多阶段框架，结合嵌入模型与大语言模型，解决法律检索与蕴含任务。参与COLIEE 2025全部五项任务，在法律案例蕴含任务中取得最佳F1得分。**

- **链接: [http://arxiv.org/pdf/2509.08025v1](http://arxiv.org/pdf/2509.08025v1)**

> **作者:** Hoang-Trung Nguyen; Tan-Minh Nguyen; Xuan-Bach Le; Tuan-Kiet Le; Khanh-Huyen Nguyen; Ha-Thanh Nguyen; Thi-Hai-Yen Vuong; Le-Minh Nguyen
>
> **摘要:** This paper presents the methodologies and results of the NOWJ team's participation across all five tasks at the COLIEE 2025 competition, emphasizing advancements in the Legal Case Entailment task (Task 2). Our comprehensive approach systematically integrates pre-ranking models (BM25, BERT, monoT5), embedding-based semantic representations (BGE-m3, LLM2Vec), and advanced Large Language Models (Qwen-2, QwQ-32B, DeepSeek-V3) for summarization, relevance scoring, and contextual re-ranking. Specifically, in Task 2, our two-stage retrieval system combined lexical-semantic filtering with contextualized LLM analysis, achieving first place with an F1 score of 0.3195. Additionally, in other tasks--including Legal Case Retrieval, Statute Law Retrieval, Legal Textual Entailment, and Legal Judgment Prediction--we demonstrated robust performance through carefully engineered ensembles and effective prompt-based reasoning strategies. Our findings highlight the potential of hybrid models integrating traditional IR techniques with contemporary generative models, providing a valuable reference for future advancements in legal information processing.
>
---
#### [new 025] Toward Subtrait-Level Model Explainability in Automated Writing Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在提升自动写作评分的透明度，通过生成式语言模型实现子特质评估与解释。研究探索了子特质与整体特质评分之间的关联，为教育者和学生提供更清晰的评分依据。**

- **链接: [http://arxiv.org/pdf/2509.08345v1](http://arxiv.org/pdf/2509.08345v1)**

> **作者:** Alejandro Andrade-Lotero; Lee Becker; Joshua Southerland; Scott Hellman
>
> **备注:** Accepted to National Council on Measurement in Education (NCME) 2025 Annual Meeting
>
> **摘要:** Subtrait (latent-trait components) assessment presents a promising path toward enhancing transparency of automated writing scores. We prototype explainability and subtrait scoring with generative language models and show modest correlation between human subtrait and trait scores, and between automated and human subtrait scores. Our approach provides details to demystify scores for educators and students.
>
---
#### [new 026] <think> So let's replace this phrase with insult... </think> Lessons learned from generation of toxic texts with LLMs
- **分类: cs.CL; cs.AI**

- **简介: 论文研究大语言模型生成有毒文本的局限性，旨在解决利用合成数据训练文本净化模型的问题。通过对比人类与LLM生成的数据，发现后者因词汇多样性不足导致性能下降达30%。强调了人类标注数据在构建鲁棒净化系统中的重要性。**

- **链接: [http://arxiv.org/pdf/2509.08358v1](http://arxiv.org/pdf/2509.08358v1)**

> **作者:** Sergey Pletenev; Daniil Moskovskiy; Alexander Panchenko
>
> **摘要:** Modern Large Language Models (LLMs) are excellent at generating synthetic data. However, their performance in sensitive domains such as text detoxification has not received proper attention from the scientific community. This paper explores the possibility of using LLM-generated synthetic toxic data as an alternative to human-generated data for training models for detoxification. Using Llama 3 and Qwen activation-patched models, we generated synthetic toxic counterparts for neutral texts from ParaDetox and SST-2 datasets. Our experiments show that models fine-tuned on synthetic data consistently perform worse than those trained on human data, with a drop in performance of up to 30% in joint metrics. The root cause is identified as a critical lexical diversity gap: LLMs generate toxic content using a small, repetitive vocabulary of insults that fails to capture the nuances and variety of human toxicity. These findings highlight the limitations of current LLMs in this domain and emphasize the continued importance of diverse, human-annotated data for building robust detoxification systems.
>
---
#### [new 027] Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling
- **分类: cs.CL**

- **简介: 该论文提出延迟流建模（DSM），用于流式多模态序列到序列学习。解决传统离线生成与流式生成间的矛盾，通过预处理对齐并引入延迟，实现任意长序列的流式推理。适用于语音识别与合成等任务，性能接近离线基线。**

- **链接: [http://arxiv.org/pdf/2509.08753v1](http://arxiv.org/pdf/2509.08753v1)**

> **作者:** Neil Zeghidour; Eugene Kharitonov; Manu Orsini; Václav Volhejn; Gabriel de Marmiesse; Edouard Grave; Patrick Pérez; Laurent Mazaré; Alexandre Défossez
>
> **摘要:** We introduce Delayed Streams Modeling (DSM), a flexible formulation for streaming, multimodal sequence-to-sequence learning. Sequence-to-sequence generation is often cast in an offline manner, where the model consumes the complete input sequence before generating the first output timestep. Alternatively, streaming sequence-to-sequence rely on learning a policy for choosing when to advance on the input stream, or write to the output stream. DSM instead models already time-aligned streams with a decoder-only language model. By moving the alignment to a pre-processing step,and introducing appropriate delays between streams, DSM provides streaming inference of arbitrary output sequences, from any input combination, making it applicable to many sequence-to-sequence problems. In particular, given text and audio streams, automatic speech recognition (ASR) corresponds to the text stream being delayed, while the opposite gives a text-to-speech (TTS) model. We perform extensive experiments for these two major sequence-to-sequence tasks, showing that DSM provides state-of-the-art performance and latency while supporting arbitrary long sequences, being even competitive with offline baselines. Code, samples and demos are available at https://github.com/kyutai-labs/delayed-streams-modeling
>
---
#### [new 028] Bilingual Word Level Language Identification for Omotic Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于双语语言识别（BLID）任务，旨在解决埃塞俄比亚南部沃尔泰语和戈法语混杂文本的识别问题。研究结合BERT与LSTM模型，取得F1分数0.72，为社交媒体管理及进一步研究奠定基础。**

- **链接: [http://arxiv.org/pdf/2509.07998v1](http://arxiv.org/pdf/2509.07998v1)**

> **作者:** Mesay Gemeda Yigezu; Girma Yohannis Bade; Atnafu Lambebo Tonja; Olga Kolesnikova; Grigori Sidorov; Alexander Gelbukh
>
> **摘要:** Language identification is the task of determining the languages for a given text. In many real world scenarios, text may contain more than one language, particularly in multilingual communities. Bilingual Language Identification (BLID) is the task of identifying and distinguishing between two languages in a given text. This paper presents BLID for languages spoken in the southern part of Ethiopia, namely Wolaita and Gofa. The presence of words similarities and differences between the two languages makes the language identification task challenging. To overcome this challenge, we employed various experiments on various approaches. Then, the combination of the BERT based pretrained language model and LSTM approach performed better, with an F1 score of 0.72 on the test set. As a result, the work will be effective in tackling unwanted social media issues and providing a foundation for further research in this area.
>
---
#### [new 029] MoVoC: Morphology-Aware Subword Construction for Geez Script Languages
- **分类: cs.CL; cs.AI; I.2.7; I.2.6; H.3.3**

- **简介: 论文提出MoVoC，一种结合形态学分析的子词构词方法，用于改善Geez文字语言的分词。旨在解决低资源、形态复杂语言中形态边界丢失问题，通过融合形态素与BPE提升分词质量，公开数据与工具支持相关研究。**

- **链接: [http://arxiv.org/pdf/2509.08812v1](http://arxiv.org/pdf/2509.08812v1)**

> **作者:** Hailay Kidu Teklehaymanot; Dren Fazlija; Wolfgang Nejdl
>
> **备注:** This submission is approximately 10 pages in length and includes 1 figure and 6 tables
>
> **摘要:** Subword-based tokenization methods often fail to preserve morphological boundaries, a limitation especially pronounced in low-resource, morphologically complex languages such as those written in the Geez script. To address this, we present MoVoC (Morpheme-aware Subword Vocabulary Construction) and train MoVoC-Tok, a tokenizer that integrates supervised morphological analysis into the subword vocabulary. This hybrid segmentation approach combines morpheme-based and Byte Pair Encoding (BPE) tokens to preserve morphological integrity while maintaining lexical meaning. To tackle resource scarcity, we curate and release manually annotated morpheme data for four Geez script languages and a morpheme-aware vocabulary for two of them. While the proposed tokenization method does not lead to significant gains in automatic translation quality, we observe consistent improvements in intrinsic metrics, MorphoScore, and Boundary Precision, highlighting the value of morphology-aware segmentation in enhancing linguistic fidelity and token efficiency. Our morpheme-annotated datasets and tokenizer will be publicly available to support further research in low-resource, morphologically rich languages. Our code and data are available on GitHub: https://github.com/hailaykidu/MoVoC
>
---
#### [new 030] Building High-Quality Datasets for Portuguese LLMs: From Common Crawl Snapshots to Industrial-Grade Corpora
- **分类: cs.CL**

- **简介: 论文研究如何构建高质量的葡萄牙语LLM训练数据集，解决非英语语言数据不足的问题。通过筛选和预处理方法，构建了120B token语料库，并验证了语言特定过滤策略对模型性能的提升效果。**

- **链接: [http://arxiv.org/pdf/2509.08824v1](http://arxiv.org/pdf/2509.08824v1)**

> **作者:** Thales Sales Almeida; Rodrigo Nogueira; Helio Pedrini
>
> **摘要:** The performance of large language models (LLMs) is deeply influenced by the quality and composition of their training data. While much of the existing work has centered on English, there remains a gap in understanding how to construct effective training corpora for other languages. We explore scalable methods for building web-based corpora for LLMs. We apply them to build a new 120B token corpus in Portuguese that achieves competitive results to an industrial-grade corpus. Using a continual pretraining setup, we study how different data selection and preprocessing strategies affect LLM performance when transitioning a model originally trained in English to another language. Our findings demonstrate the value of language-specific filtering pipelines, including classifiers for education, science, technology, engineering, and mathematics (STEM), as well as toxic content. We show that adapting a model to the target language leads to performance improvements, reinforcing the importance of high-quality, language-specific data. While our case study focuses on Portuguese, our methods are applicable to other languages, offering insights for multilingual LLM development.
>
---
#### [new 031] Simulating Identity, Propagating Bias: Abstraction and Stereotypes in LLM-Generated Text
- **分类: cs.CL**

- **简介: 该论文研究LLM在使用身份提示生成文本时是否产生刻板印象。通过分析六种模型的输出，发现身份提示难以控制语言抽象程度，可能传播刻板印象。任务为检测语言偏见，方法包括提出新数据集和评估指标。**

- **链接: [http://arxiv.org/pdf/2509.08484v1](http://arxiv.org/pdf/2509.08484v1)**

> **作者:** Pia Sommerauer; Giulia Rambelli; Tommaso Caselli
>
> **备注:** Accepted to EMNLP Findings 2025
>
> **摘要:** Persona-prompting is a growing strategy to steer LLMs toward simulating particular perspectives or linguistic styles through the lens of a specified identity. While this method is often used to personalize outputs, its impact on how LLMs represent social groups remains underexplored. In this paper, we investigate whether persona-prompting leads to different levels of linguistic abstraction - an established marker of stereotyping - when generating short texts linking socio-demographic categories with stereotypical or non-stereotypical attributes. Drawing on the Linguistic Expectancy Bias framework, we analyze outputs from six open-weight LLMs under three prompting conditions, comparing 11 persona-driven responses to those of a generic AI assistant. To support this analysis, we introduce Self-Stereo, a new dataset of self-reported stereotypes from Reddit. We measure abstraction through three metrics: concreteness, specificity, and negation. Our results highlight the limits of persona-prompting in modulating abstraction in language, confirming criticisms about the ecology of personas as representative of socio-demographic groups and raising concerns about the risk of propagating stereotypes even when seemingly evoking the voice of a marginalized group.
>
---
#### [new 032] LLM Ensemble for RAG: Role of Context Length in Zero-Shot Question Answering for BioASQ Challenge
- **分类: cs.CL**

- **简介: 论文研究利用LLM集成提升生物医学问答（BioASQ）任务的零样本性能。通过RAG方法，探讨上下文长度对模型表现的影响，提出无需微调即可超越领域定制系统的有效方案。**

- **链接: [http://arxiv.org/pdf/2509.08596v1](http://arxiv.org/pdf/2509.08596v1)**

> **作者:** Dima Galat; Diego Molla-Aliod
>
> **备注:** CEUR-WS, CLEF2025
>
> **摘要:** Biomedical question answering (QA) poses significant challenges due to the need for precise interpretation of specialized knowledge drawn from a vast, complex, and rapidly evolving corpus. In this work, we explore how large language models (LLMs) can be used for information retrieval (IR), and an ensemble of zero-shot models can accomplish state-of-the-art performance on a domain-specific Yes/No QA task. Evaluating our approach on the BioASQ challenge tasks, we show that ensembles can outperform individual LLMs and in some cases rival or surpass domain-tuned systems - all while preserving generalizability and avoiding the need for costly fine-tuning or labeled data. Our method aggregates outputs from multiple LLM variants, including models from Anthropic and Google, to synthesize more accurate and robust answers. Moreover, our investigation highlights a relationship between context length and performance: while expanded contexts are meant to provide valuable evidence, they simultaneously risk information dilution and model disorientation. These findings emphasize IR as a critical foundation in Retrieval-Augmented Generation (RAG) approaches for biomedical QA systems. Precise, focused retrieval remains essential for ensuring LLMs operate within relevant information boundaries when generating answers from retrieved documents. Our results establish that ensemble-based zero-shot approaches, when paired with effective RAG pipelines, constitute a practical and scalable alternative to domain-tuned systems for biomedical question answering.
>
---
#### [new 033] Acquiescence Bias in Large Language Models
- **分类: cs.CL**

- **简介: 论文研究大语言模型是否存在“趋同偏见”，即倾向于回答“否”的现象。通过跨模型、任务和语言（英、德、波）的实验，发现LLMs与人类不同，表现出偏向“否”的回答倾向。该研究属于自然语言处理中的偏见检测任务，旨在揭示LLMs在回答行为上的潜在偏差。**

- **链接: [http://arxiv.org/pdf/2509.08480v1](http://arxiv.org/pdf/2509.08480v1)**

> **作者:** Daniel Braun
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Acquiescence bias, i.e. the tendency of humans to agree with statements in surveys, independent of their actual beliefs, is well researched and documented. Since Large Language Models (LLMs) have been shown to be very influenceable by relatively small changes in input and are trained on human-generated data, it is reasonable to assume that they could show a similar tendency. We present a study investigating the presence of acquiescence bias in LLMs across different models, tasks, and languages (English, German, and Polish). Our results indicate that, contrary to humans, LLMs display a bias towards answering no, regardless of whether it indicates agreement or disagreement.
>
---
#### [new 034] Generative Data Refinement: Just Ask for Better Data
- **分类: cs.LG; cs.CL**

- **简介: 论文提出Generative Data Refinement（GDR）框架，用于利用预训练生成模型优化含不良内容的数据集，解决数据质量与隐私风险问题，提升训练数据规模与多样性。**

- **链接: [http://arxiv.org/pdf/2509.08653v1](http://arxiv.org/pdf/2509.08653v1)**

> **作者:** Minqi Jiang; João G. M. Araújo; Will Ellsworth; Sian Gooding; Edward Grefenstette
>
> **摘要:** For a fixed parameter size, the capabilities of large models are primarily determined by the quality and quantity of its training data. Consequently, training datasets now grow faster than the rate at which new data is indexed on the web, leading to projected data exhaustion over the next decade. Much more data exists as user-generated content that is not publicly indexed, but incorporating such data comes with considerable risks, such as leaking private information and other undesirable content. We introduce a framework, Generative Data Refinement (GDR), for using pretrained generative models to transform a dataset with undesirable content into a refined dataset that is more suitable for training. Our experiments show that GDR can outperform industry-grade solutions for dataset anonymization, as well as enable direct detoxification of highly unsafe datasets. Moreover, we show that by generating synthetic data that is conditioned on each example in the real dataset, GDR's refined outputs naturally match the diversity of web scale datasets, and thereby avoid the often challenging task of generating diverse synthetic data via model prompting. The simplicity and effectiveness of GDR make it a powerful tool for scaling up the total stock of training data for frontier models.
>
---
#### [new 035] Merge-of-Thought Distillation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出Merge-of-Thought Distillation（MoT）框架，解决多教师知识蒸馏中冲突问题，通过交替微调与权重合并，提升学生模型性能。属于自然语言处理中的模型蒸馏任务。**

- **链接: [http://arxiv.org/pdf/2509.08814v1](http://arxiv.org/pdf/2509.08814v1)**

> **作者:** Zhanming Shen; Zeyu Qin; Zenan Huang; Hao Chen; Jiaqi Hu; Yihong Zhuang; Guoshan Lu; Gang Chen; Junbo Zhao
>
> **摘要:** Efficient reasoning distillation for long chain-of-thought (CoT) models is increasingly constrained by the assumption of a single oracle teacher, despite practical availability of multiple candidate teachers and growing CoT corpora. We revisit teacher selection and observe that different students have different "best teachers," and even for the same student the best teacher can vary across datasets. Therefore, to unify multiple teachers' reasoning abilities into student with overcoming conflicts among various teachers' supervision, we propose Merge-of-Thought Distillation (MoT), a lightweight framework that alternates between teacher-specific supervised fine-tuning branches and weight-space merging of the resulting student variants. On competition math benchmarks, using only about 200 high-quality CoT samples, applying MoT to a Qwen3-14B student surpasses strong models including DEEPSEEK-R1, QWEN3-30B-A3B, QWEN3-32B, and OPENAI-O1, demonstrating substantial gains. Besides, MoT consistently outperforms the best single-teacher distillation and the naive multi-teacher union, raises the performance ceiling while mitigating overfitting, and shows robustness to distribution-shifted and peer-level teachers. Moreover, MoT reduces catastrophic forgetting, improves general reasoning beyond mathematics and even cultivates a better teacher, indicating that consensus-filtered reasoning features transfer broadly. These results position MoT as a simple, scalable route to efficiently distilling long CoT capabilities from diverse teachers into compact students.
>
---
#### [new 036] XML Prompting as Grammar-Constrained Interaction: Fixed-Point Semantics, Convergence Guarantees, and Human-AI Protocols
- **分类: cs.PL; cs.AI; cs.CL; 03B70, 06B23, 47H10, 68T27, 68T50; I.2.7; I.2.8; F.4.1; F.4.3; H.5.2**

- **简介: 论文研究XML提示的语法约束交互机制，提出固定点语义与收敛性保证，解决LLM输出结构化与任务一致性问题，构建人类-AI协作协议。**

- **链接: [http://arxiv.org/pdf/2509.08182v1](http://arxiv.org/pdf/2509.08182v1)**

> **作者:** Faruk Alpay; Taylan Alpay
>
> **备注:** 7 pages, multiple XML prompts
>
> **摘要:** Structured prompting with XML tags has emerged as an effective way to steer large language models (LLMs) toward parseable, schema-adherent outputs in real-world systems. We develop a logic-first treatment of XML prompting that unifies (i) grammar-constrained decoding, (ii) fixed-point semantics over lattices of hierarchical prompts, and (iii) convergent human-AI interaction loops. We formalize a complete lattice of XML trees under a refinement order and prove that monotone prompt-to-prompt operators admit least fixed points (Knaster-Tarski) that characterize steady-state protocols; under a task-aware contraction metric on trees, we further prove Banach-style convergence of iterative guidance. We instantiate these results with context-free grammars (CFGs) for XML schemas and show how constrained decoding guarantees well-formedness while preserving task performance. A set of multi-layer human-AI interaction recipes demonstrates practical deployment patterns, including multi-pass "plan $\to$ verify $\to$ revise" routines and agentic tool use. We provide mathematically complete proofs and tie our framework to recent advances in grammar-aligned decoding, chain-of-verification, and programmatic prompting.
>
---
#### [new 037] Scaling Truth: The Confidence Paradox in AI Fact-Checking
- **分类: cs.SI; cs.AI; cs.CL; cs.CY**

- **简介: 论文评估九种大语言模型在多语言事实核查中的表现，揭示小模型自信度高但准确率低的问题，暴露信息验证中的系统性偏差，旨在推动公平可靠的AI事实核查解决方案。**

- **链接: [http://arxiv.org/pdf/2509.08803v1](http://arxiv.org/pdf/2509.08803v1)**

> **作者:** Ihsan A. Qazi; Zohaib Khan; Abdullah Ghani; Agha A. Raza; Zafar A. Qazi; Wassay Sajjad; Ayesha Ali; Asher Javaid; Muhammad Abdullah Sohail; Abdul H. Azeemi
>
> **备注:** 65 pages, 26 figures, 6 tables
>
> **摘要:** The rise of misinformation underscores the need for scalable and reliable fact-checking solutions. Large language models (LLMs) hold promise in automating fact verification, yet their effectiveness across global contexts remains uncertain. We systematically evaluate nine established LLMs across multiple categories (open/closed-source, multiple sizes, diverse architectures, reasoning-based) using 5,000 claims previously assessed by 174 professional fact-checking organizations across 47 languages. Our methodology tests model generalizability on claims postdating training cutoffs and four prompting strategies mirroring both citizen and professional fact-checker interactions, with over 240,000 human annotations as ground truth. Findings reveal a concerning pattern resembling the Dunning-Kruger effect: smaller, accessible models show high confidence despite lower accuracy, while larger models demonstrate higher accuracy but lower confidence. This risks systemic bias in information verification, as resource-constrained organizations typically use smaller models. Performance gaps are most pronounced for non-English languages and claims originating from the Global South, threatening to widen existing information inequalities. These results establish a multilingual benchmark for future research and provide an evidence base for policy aimed at ensuring equitable access to trustworthy, AI-assisted fact-checking.
>
---
#### [new 038] AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出AgentGym-RL框架，用于训练LLM代理进行长期决策。解决从零开始训练自主代理的问题，无需依赖监督微调。引入ScalingInter-RL方法平衡探索与利用，提升稳定性和多样性，并在多个任务中验证有效性。**

- **链接: [http://arxiv.org/pdf/2509.08755v1](http://arxiv.org/pdf/2509.08755v1)**

> **作者:** Zhiheng Xi; Jixuan Huang; Chenyang Liao; Baodai Huang; Honglin Guo; Jiaqi Liu; Rui Zheng; Junjie Ye; Jiazheng Zhang; Wenxiang Chen; Wei He; Yiwen Ding; Guanyu Li; Zehui Chen; Zhengyin Du; Xuesong Yao; Yufei Xu; Jiecao Chen; Tao Gui; Zuxuan Wu; Qi Zhang; Xuanjing Huang; Yu-Gang Jiang
>
> **备注:** preprint, 39 pages, 16 figures. Project: https://AgentGym-RL.github.io/. Framework and Code: https://github.com/woooodyy/AgentGym, https://github.com/woooodyy/AgentGym-RL
>
> **摘要:** Developing autonomous LLM agents capable of making a series of intelligent decisions to solve complex, real-world tasks is a fast-evolving frontier. Like human cognitive development, agents are expected to acquire knowledge and skills through exploration and interaction with the environment. Despite advances, the community still lacks a unified, interactive reinforcement learning (RL) framework that can effectively train such agents from scratch -- without relying on supervised fine-tuning (SFT) -- across diverse and realistic environments. To bridge this gap, we introduce AgentGym-RL, a new framework to train LLM agents for multi-turn interactive decision-making through RL. The framework features a modular and decoupled architecture, ensuring high flexibility and extensibility. It encompasses a wide variety of real-world scenarios, and supports mainstream RL algorithms. Furthermore, we propose ScalingInter-RL, a training approach designed for exploration-exploitation balance and stable RL optimization. In early stages, it emphasizes exploitation by restricting the number of interactions, and gradually shifts towards exploration with larger horizons to encourage diverse problem-solving strategies. In this way, the agent develops more diverse behaviors and is less prone to collapse under long horizons. We perform extensive experiments to validate the stability and effectiveness of both the AgentGym-RL framework and the ScalingInter-RL approach. Our agents match or surpass commercial models on 27 tasks across diverse environments. We offer key insights and will open-source the complete AgentGym-RL framework -- including code and datasets -- to empower the research community in developing the next generation of intelligent agents.
>
---
#### [new 039] HumanAgencyBench: Scalable Evaluation of Human Agency Support in AI Assistants
- **分类: cs.CY; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文提出HumanAgencyBench（HAB），用于评估AI助手对人类代理的支持程度，从六个维度衡量其表现，发现当前LLM助手支持有限且存在差异，强调需加强安全与对齐目标。属于AI伦理评估任务，解决人类控制权流失问题。**

- **链接: [http://arxiv.org/pdf/2509.08494v1](http://arxiv.org/pdf/2509.08494v1)**

> **作者:** Benjamin Sturgeon; Daniel Samuelson; Jacob Haimes; Jacy Reese Anthis
>
> **摘要:** As humans delegate more tasks and decisions to artificial intelligence (AI), we risk losing control of our individual and collective futures. Relatively simple algorithmic systems already steer human decision-making, such as social media feed algorithms that lead people to unintentionally and absent-mindedly scroll through engagement-optimized content. In this paper, we develop the idea of human agency by integrating philosophical and scientific theories of agency with AI-assisted evaluation methods: using large language models (LLMs) to simulate and validate user queries and to evaluate AI responses. We develop HumanAgencyBench (HAB), a scalable and adaptive benchmark with six dimensions of human agency based on typical AI use cases. HAB measures the tendency of an AI assistant or agent to Ask Clarifying Questions, Avoid Value Manipulation, Correct Misinformation, Defer Important Decisions, Encourage Learning, and Maintain Social Boundaries. We find low-to-moderate agency support in contemporary LLM-based assistants and substantial variation across system developers and dimensions. For example, while Anthropic LLMs most support human agency overall, they are the least supportive LLMs in terms of Avoid Value Manipulation. Agency support does not appear to consistently result from increasing LLM capabilities or instruction-following behavior (e.g., RLHF), and we encourage a shift towards more robust safety and alignment targets.
>
---
#### [new 040] EvolKV: Evolutionary KV Cache Compression for LLM Inference
- **分类: cs.LG; cs.CL; cs.NE**

- **简介: 该论文提出EvolKV，解决LLM推理中KV缓存压缩问题。通过进化算法优化层间缓存分配，提升内存效率与任务性能。实验表明其在多个任务上优于基线方法，尤其在长上下文和代码补全任务中表现突出。**

- **链接: [http://arxiv.org/pdf/2509.08315v1](http://arxiv.org/pdf/2509.08315v1)**

> **作者:** Bohan Yu; Yekun Chai
>
> **摘要:** Existing key-value (KV) cache compression methods typically rely on heuristics, such as uniform cache allocation across layers or static eviction policies, however, they ignore the critical interplays among layer-specific feature patterns and task performance, which can lead to degraded generalization. In this paper, we propose EvolKV, an adaptive framework for layer-wise, task-driven KV cache compression that jointly optimizes the memory efficiency and task performance. By reformulating cache allocation as a multi-objective optimization problem, EvolKV leverages evolutionary search to dynamically configure layer budgets while directly maximizing downstream performance. Extensive experiments on 11 tasks demonstrate that our approach outperforms all baseline methods across a wide range of KV cache budgets on long-context tasks and surpasses heuristic baselines by up to 7 percentage points on GSM8K. Notably, EvolKV achieves superior performance over the full KV cache setting on code completion while utilizing only 1.5% of the original budget, suggesting the untapped potential in learned compression strategies for KV cache budget allocation.
>
---
#### [new 041] Calibrating MLLM-as-a-judge via Multimodal Bayesian Prompt Ensembles
- **分类: cs.CV; cs.CL**

- **简介: 论文提出MMB方法，用于校准多模态大语言模型在图像生成评估中的判断能力。针对模型存在偏差、过拟合和跨领域性能不一致的问题，通过多模态贝叶斯提示集成提升准确性和校准效果。**

- **链接: [http://arxiv.org/pdf/2509.08777v1](http://arxiv.org/pdf/2509.08777v1)**

> **作者:** Eric Slyman; Mehrab Tanjim; Kushal Kafle; Stefan Lee
>
> **备注:** 17 pages, 8 figures, Accepted at ICCV 2025
>
> **摘要:** Multimodal large language models (MLLMs) are increasingly used to evaluate text-to-image (TTI) generation systems, providing automated judgments based on visual and textual context. However, these "judge" models often suffer from biases, overconfidence, and inconsistent performance across diverse image domains. While prompt ensembling has shown promise for mitigating these issues in unimodal, text-only settings, our experiments reveal that standard ensembling methods fail to generalize effectively for TTI tasks. To address these limitations, we propose a new multimodal-aware method called Multimodal Mixture-of-Bayesian Prompt Ensembles (MMB). Our method uses a Bayesian prompt ensemble approach augmented by image clustering, allowing the judge to dynamically assign prompt weights based on the visual characteristics of each sample. We show that MMB improves accuracy in pairwise preference judgments and greatly enhances calibration, making it easier to gauge the judge's true uncertainty. In evaluations on two TTI benchmarks, HPSv2 and MJBench, MMB outperforms existing baselines in alignment with human annotations and calibration across varied image content. Our findings highlight the importance of multimodal-specific strategies for judge calibration and suggest a promising path forward for reliable large-scale TTI evaluation.
>
---
#### [new 042] Measuring and mitigating overreliance is necessary for building human-compatible AI
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **简介: 论文探讨如何测量和缓解对大型语言模型的过度依赖问题，分析其个体与社会风险，提出改进测量方法与缓解策略，以确保AI增强而非削弱人类能力。属于AI伦理与人机协作任务。**

- **链接: [http://arxiv.org/pdf/2509.08010v1](http://arxiv.org/pdf/2509.08010v1)**

> **作者:** Lujain Ibrahim; Katherine M. Collins; Sunnie S. Y. Kim; Anka Reuel; Max Lamparth; Kevin Feng; Lama Ahmad; Prajna Soni; Alia El Kattan; Merlin Stein; Siddharth Swaroop; Ilia Sucholutsky; Andrew Strait; Q. Vera Liao; Umang Bhatt
>
> **摘要:** Large language models (LLMs) distinguish themselves from previous technologies by functioning as collaborative "thought partners," capable of engaging more fluidly in natural language. As LLMs increasingly influence consequential decisions across diverse domains from healthcare to personal advice, the risk of overreliance - relying on LLMs beyond their capabilities - grows. This position paper argues that measuring and mitigating overreliance must become central to LLM research and deployment. First, we consolidate risks from overreliance at both the individual and societal levels, including high-stakes errors, governance challenges, and cognitive deskilling. Then, we explore LLM characteristics, system design features, and user cognitive biases that - together - raise serious and unique concerns about overreliance in practice. We also examine historical approaches for measuring overreliance, identifying three important gaps and proposing three promising directions to improve measurement. Finally, we propose mitigation strategies that the AI research community can pursue to ensure LLMs augment rather than undermine human capabilities.
>
---
## 更新

#### [replaced 001] Beyond Seen Data: Improving KBQA Generalization Through Schema-Guided Logical Form Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.12737v3](http://arxiv.org/pdf/2502.12737v3)**

> **作者:** Shengxiang Gao; Jey Han Lau; Jianzhong Qi
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Knowledge base question answering (KBQA) aims to answer user questions in natural language using rich human knowledge stored in large KBs. As current KBQA methods struggle with unseen knowledge base elements at test time,we introduce SG-KBQA: a novel model that injects schema contexts into entity retrieval and logical form generation to tackle this issue. It uses the richer semantics and awareness of the knowledge base structure provided by schema contexts to enhance generalizability. We show that SG-KBQA achieves strong generalizability, outperforming state-of-the-art models on two commonly used benchmark datasets across a variety of test settings. Our source code is available at https://github.com/gaosx2000/SG_KBQA.
>
---
#### [replaced 002] GRAM-R$^2$: Self-Training Generative Foundation Reward Models for Reward Reasoning
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.02492v2](http://arxiv.org/pdf/2509.02492v2)**

> **作者:** Chenglong Wang; Yongyu Mu; Hang Zhou; Yifu Huo; Ziming Zhu; Jiali Zeng; Murun Yang; Bei Li; Tong Xiao; Xiaoyang Hao; Chunliang Zhang; Fandong Meng; Jingbo Zhu
>
> **摘要:** Significant progress in reward modeling over recent years has been driven by a paradigm shift from task-specific designs towards generalist reward models. Despite this trend, developing effective reward models remains a fundamental challenge: the heavy reliance on large-scale labeled preference data. Pre-training on abundant unlabeled data offers a promising direction, but existing approaches fall short of instilling explicit reasoning into reward models. To bridge this gap, we propose a self-training approach that leverages unlabeled data to elicit reward reasoning in reward models. Based on this approach, we develop GRAM-R$^2$, a generative reward model trained to produce not only preference labels but also accompanying reward rationales. GRAM-R$^2$ can serve as a foundation model for reward reasoning and can be applied to a wide range of tasks with minimal or no additional fine-tuning. It can support downstream applications such as response ranking and task-specific reward tuning. Experiments on response ranking, task adaptation, and reinforcement learning from human feedback demonstrate that GRAM-R$^2$ consistently delivers strong performance, outperforming several strong discriminative and generative baselines.
>
---
#### [replaced 003] Mind the Value-Action Gap: Do LLMs Act in Alignment with Their Values?
- **分类: cs.HC; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.15463v2](http://arxiv.org/pdf/2501.15463v2)**

> **作者:** Hua Shen; Nicholas Clark; Tanushree Mitra
>
> **备注:** EMNLP 2025 Main Paper
>
> **摘要:** Existing research primarily evaluates the values of LLMs by examining their stated inclinations towards specific values. However, the "Value-Action Gap," a phenomenon rooted in environmental and social psychology, reveals discrepancies between individuals' stated values and their actions in real-world contexts. To what extent do LLMs exhibit a similar gap between their stated values and their actions informed by those values? This study introduces ValueActionLens, an evaluation framework to assess the alignment between LLMs' stated values and their value-informed actions. The framework encompasses the generation of a dataset comprising 14.8k value-informed actions across twelve cultures and eleven social topics, and two tasks to evaluate how well LLMs' stated value inclinations and value-informed actions align across three different alignment measures. Extensive experiments reveal that the alignment between LLMs' stated values and actions is sub-optimal, varying significantly across scenarios and models. Analysis of misaligned results identifies potential harms from certain value-action gaps. To predict the value-action gaps, we also uncover that leveraging reasoned explanations improves performance. These findings underscore the risks of relying solely on the LLMs' stated values to predict their behaviors and emphasize the importance of context-aware evaluations of LLM values and value-action gaps.
>
---
#### [replaced 004] IssueBench: Millions of Realistic Prompts for Measuring Issue Bias in LLM Writing Assistance
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.08395v3](http://arxiv.org/pdf/2502.08395v3)**

> **作者:** Paul Röttger; Musashi Hinck; Valentin Hofmann; Kobi Hackenburg; Valentina Pyatkin; Faeze Brahman; Dirk Hovy
>
> **备注:** accepted at TACL (pre-MIT Press publication version)
>
> **摘要:** Large language models (LLMs) are helping millions of users write texts about diverse issues, and in doing so expose users to different ideas and perspectives. This creates concerns about issue bias, where an LLM tends to present just one perspective on a given issue, which in turn may influence how users think about this issue. So far, it has not been possible to measure which issue biases LLMs manifest in real user interactions, making it difficult to address the risks from biased LLMs. Therefore, we create IssueBench: a set of 2.49m realistic English-language prompts to measure issue bias in LLM writing assistance, which we construct based on 3.9k templates (e.g. "write a blog about") and 212 political issues (e.g. "AI regulation") from real user interactions. Using IssueBench, we show that issue biases are common and persistent in 10 state-of-the-art LLMs. We also show that biases are very similar across models, and that all models align more with US Democrat than Republican voter opinion on a subset of issues. IssueBench can easily be adapted to include other issues, templates, or tasks. By enabling robust and realistic measurement, we hope that IssueBench can bring a new quality of evidence to ongoing discussions about LLM biases and how to address them.
>
---
#### [replaced 005] All for law and law for all: Adaptive RAG Pipeline for Legal Research
- **分类: cs.CL; cs.IR; F.2.2; H.3.3; I.2.7**

- **链接: [http://arxiv.org/pdf/2508.13107v2](http://arxiv.org/pdf/2508.13107v2)**

> **作者:** Figarri Keisha; Prince Singh; Pallavi; Dion Fernandes; Aravindh Manivannan; Ilham Wicaksono; Faisal Ahmad; Wiem Ben Rim
>
> **备注:** submitted to NLLP 2025 Workshop
>
> **摘要:** Retrieval-Augmented Generation (RAG) has transformed how we approach text generation tasks by grounding Large Language Model (LLM) outputs in retrieved knowledge. This capability is especially critical in the legal domain. In this work, we introduce a novel end-to-end RAG pipeline that improves upon previous baselines using three targeted enhancements: (i) a context-aware query translator that disentangles document references from natural-language questions and adapts retrieval depth and response style based on expertise and specificity, (ii) open-source retrieval strategies using SBERT and GTE embeddings that achieve substantial performance gains while remaining cost-efficient, and (iii) a comprehensive evaluation and generation framework that combines RAGAS, BERTScore-F1, and ROUGE-Recall to assess semantic alignment and faithfulness across models and prompt designs. Our results show that carefully designed open-source pipelines can rival proprietary approaches in retrieval quality, while a custom legal-grounded prompt consistently produces more faithful and contextually relevant answers than baseline prompting. Taken together, these contributions demonstrate the potential of task-aware, component-level tuning to deliver legally grounded, reproducible, and cost-effective RAG systems for legal research assistance.
>
---
#### [replaced 006] TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.14161v3](http://arxiv.org/pdf/2412.14161v3)**

> **作者:** Frank F. Xu; Yufan Song; Boxuan Li; Yuxuan Tang; Kritanjali Jain; Mengxue Bao; Zora Z. Wang; Xuhui Zhou; Zhitong Guo; Murong Cao; Mingyang Yang; Hao Yang Lu; Amaad Martin; Zhe Su; Leander Maben; Raj Mehta; Wayne Chi; Lawrence Jang; Yiqing Xie; Shuyan Zhou; Graham Neubig
>
> **备注:** Preprint
>
> **摘要:** We interact with computers on an everyday basis, be it in everyday life or work, and many aspects of work can be done entirely with access to a computer and the Internet. At the same time, thanks to improvements in large language models (LLMs), there has also been a rapid development in AI agents that interact with and affect change in their surrounding environments. But how performant are AI agents at accelerating or even autonomously performing work-related tasks? The answer to this question has important implications both for industry looking to adopt AI into their workflows and for economic policy to understand the effects that adoption of AI may have on the labor market. To measure the progress of these LLM agents' performance on performing real-world professional tasks, in this paper we introduce TheAgentCompany, an extensible benchmark for evaluating AI agents that interact with the world in similar ways to those of a digital worker: by browsing the Web, writing code, running programs, and communicating with other coworkers. We build a self-contained environment with internal web sites and data that mimics a small software company environment, and create a variety of tasks that may be performed by workers in such a company. We test baseline agents powered by both closed API-based and open-weights language models (LMs), and find that the most competitive agent can complete 30% of tasks autonomously. This paints a nuanced picture on task automation with LM agents--in a setting simulating a real workplace, a good portion of simpler tasks could be solved autonomously, but more difficult long-horizon tasks are still beyond the reach of current systems. We release code, data, environment, and experiments on https://the-agent-company.com.
>
---
#### [replaced 007] CoAT: Chain-of-Associated-Thoughts Framework for Enhancing Large Language Models Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.02390v2](http://arxiv.org/pdf/2502.02390v2)**

> **作者:** Jianfeng Pan; Senyou Deng; Shaomang Huang
>
> **备注:** 18 pages, 10 figures
>
> **摘要:** Research on LLM technologies is rapidly emerging, with most of them employ a 'fast thinking' approach to inference. Most LLMs generate the final result based solely on a single query and LLM's reasoning capabilities. However, with the advent of OpenAI-o1, 'slow thinking' techniques have garnered increasing attention because its process is closer to the human thought process. Inspired by the human ability to constantly associate and replenish knowledge during thinking, we developed the novel Chain-of-Associated-Thoughts (CoAT) framework, which introduces an innovative synergy between the Monte Carlo Tree Search (MCTS) algorithm and a dynamic mechanism for integrating new key information, termed 'associative memory'. By combining the structured exploration capabilities of MCTS with the adaptive learning capacity of associative memory, CoAT significantly expands the LLM search space, enabling our framework to explore diverse reasoning pathways and dynamically update its knowledge base in real-time. This allows the framework to not only revisit and refine earlier inferences but also adaptively incorporate evolving information, ensuring that the final output is both accurate and comprehensive. We validate CoAT's effectiveness across a variety of generative and reasoning tasks. Quantitative experiments show that CoAT achieves over 10% performance improvement on open-source multi-hop reasoning datasets (HotpotQA, MuSiQue) and more than 15% gain on our proprietary CRB dataset.
>
---
#### [replaced 008] RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.01907v3](http://arxiv.org/pdf/2509.01907v3)**

> **作者:** Zhenyuan Chen; Chenxi Wang; Feng Zhang
>
> **备注:** under review
>
> **摘要:** Remote sensing is critical for disaster monitoring, yet existing datasets lack temporal image pairs and detailed textual annotations. While single-snapshot imagery dominates current resources, it fails to capture dynamic disaster impacts over time. To address this gap, we introduce the Remote Sensing Change Caption (RSCC) dataset, a large-scale benchmark comprising 62,315 pre-/post-disaster image pairs (spanning earthquakes, floods, wildfires, and more) paired with rich, human-like change captions. By bridging the temporal and semantic divide in remote sensing data, RSCC enables robust training and evaluation of vision-language models for disaster-aware bi-temporal understanding. Our results highlight RSCC's ability to facilitate detailed disaster-related analysis, paving the way for more accurate, interpretable, and scalable vision-language applications in remote sensing. Code and dataset are available at https://github.com/Bili-Sakura/RSCC.
>
---
#### [replaced 009] Measuring Bias or Measuring the Task: Understanding the Brittle Nature of LLM Gender Biases
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.04373v3](http://arxiv.org/pdf/2509.04373v3)**

> **作者:** Bufan Gao; Elisa Kreiss
>
> **备注:** To be published at EMNLP 2025 (main conference)
>
> **摘要:** As LLMs are increasingly applied in socially impactful settings, concerns about gender bias have prompted growing efforts both to measure and mitigate such bias. These efforts often rely on evaluation tasks that differ from natural language distributions, as they typically involve carefully constructed task prompts that overtly or covertly signal the presence of gender bias-related content. In this paper, we examine how signaling the evaluative purpose of a task impacts measured gender bias in LLMs. Concretely, we test models under prompt conditions that (1) make the testing context salient, and (2) make gender-focused content salient. We then assess prompt sensitivity across four task formats with both token-probability and discrete-choice metrics. We find that prompts that more clearly align with (gender bias) evaluation framing elicit distinct gender output distributions compared to less evaluation-framed prompts. Discrete-choice metrics further tend to amplify bias relative to probabilistic measures. These findings do not only highlight the brittleness of LLM gender bias evaluations but open a new puzzle for the NLP benchmarking and development community: To what extent can well-controlled testing designs trigger LLM "testing mode" performance, and what does this mean for the ecological validity of future benchmarks.
>
---
#### [replaced 010] Localizing Factual Inconsistencies in Attributable Text Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.07473v3](http://arxiv.org/pdf/2410.07473v3)**

> **作者:** Arie Cattan; Paul Roit; Shiyue Zhang; David Wan; Roee Aharoni; Idan Szpektor; Mohit Bansal; Ido Dagan
>
> **备注:** Accepted for publication in Transactions of the Association for Computational Linguistics (TACL), 2025. Authors pre-print
>
> **摘要:** There has been an increasing interest in detecting hallucinations in model-generated texts, both manually and automatically, at varying levels of granularity. However, most existing methods fail to precisely pinpoint the errors. In this work, we introduce QASemConsistency, a new formalism for localizing factual inconsistencies in attributable text generation, at a fine-grained level. Drawing inspiration from Neo-Davidsonian formal semantics, we propose decomposing the generated text into minimal predicate-argument level propositions, expressed as simple question-answer (QA) pairs, and assess whether each individual QA pair is supported by a trusted reference text. As each QA pair corresponds to a single semantic relation between a predicate and an argument, QASemConsistency effectively localizes the unsupported information. We first demonstrate the effectiveness of the QASemConsistency methodology for human annotation, by collecting crowdsourced annotations of granular consistency errors, while achieving a substantial inter-annotator agreement. This benchmark includes more than 3K instances spanning various tasks of attributable text generation. We also show that QASemConsistency yields factual consistency scores that correlate well with human judgments. Finally, we implement several methods for automatically detecting localized factual inconsistencies, with both supervised entailment models and LLMs.
>
---
#### [replaced 011] MedS$^3$: Towards Medical Slow Thinking with Self-Evolved Soft Dual-sided Process Supervision
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.12051v3](http://arxiv.org/pdf/2501.12051v3)**

> **作者:** Shuyang Jiang; Yusheng Liao; Zhe Chen; Ya Zhang; Yanfeng Wang; Yu Wang
>
> **备注:** 20 pages;
>
> **摘要:** Medical language models face critical barriers to real-world clinical reasoning applications. However, mainstream efforts, which fall short in task coverage, lack fine-grained supervision for intermediate reasoning steps, and rely on proprietary systems, are still far from a versatile, credible and efficient language model for clinical reasoning usage. To this end, we propose \mone, a self-evolving framework that imparts robust reasoning capabilities to small, deployable models. Starting with 8,000 curated instances sampled via a curriculum strategy across five medical domains and 16 datasets, we use a small base policy model to conduct Monte Carlo Tree Search (MCTS) for constructing rule-verifiable reasoning trajectories. Self-explored reasoning trajectories ranked by node values are used to bootstrap the policy model via reinforcement fine-tuning and preference learning. Moreover, we introduce a soft dual process reward model that incorporates value dynamics: steps that degrade node value are penalized, enabling fine-grained identification of reasoning errors even when the final answer is correct. Experiments on eleven benchmarks show that \mone outperforms the previous state-of-the-art medical model by +6.45 accuracy points and surpasses 32B-scale general-purpose reasoning models by +8.57 points. Additional empirical analysis further demonstrates that \mone achieves robust and faithful reasoning behavior.
>
---
#### [replaced 012] Subjective Behaviors and Preferences in LLM: Language of Browsing
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.15474v2](http://arxiv.org/pdf/2508.15474v2)**

> **作者:** Sai Sundaresan; Harshita Chopra; Atanu R. Sinha; Koustava Goswami; Nagasai Saketh Naidu; Raghav Karan; N Anushka
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** A Large Language Model (LLM) offers versatility across domains and tasks, purportedly benefiting users with a wide variety of behaviors and preferences. We question this perception about an LLM when users have inherently subjective behaviors and preferences, as seen in their ubiquitous and idiosyncratic browsing of websites or apps. The sequential behavior logs of pages, thus generated, form something akin to each user's self-constructed "language", albeit without the structure and grammar imbued in natural languages. We ask: (i) Can a small LM represent the "language of browsing" better than a large LM? (ii) Can an LM with a single set of parameters (or, single LM) adequately capture myriad users' heterogeneous, subjective behaviors and preferences? (iii) Can a single LM with high average performance, yield low variance in performance to make alignment good at user level? We introduce clusterwise LM training, HeTLM (Heterogeneity aware Training of Language Model), appropriate for subjective behaviors. We find that (i) a small LM trained using a page-level tokenizer outperforms large pretrained or finetuned LMs; (ii) HeTLM with heterogeneous cluster specific set of parameters outperforms a single LM of the same family, controlling for the number of parameters; and (iii) a higher mean and a lower variance in generation ensues, implying improved alignment.
>
---
#### [replaced 013] M-BRe: Discovering Training Samples for Relation Extraction from Unlabeled Texts with Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.07730v2](http://arxiv.org/pdf/2509.07730v2)**

> **作者:** Zexuan Li; Hongliang Dai; Piji Li
>
> **备注:** Accepted by EMNLP2025 Main Conference
>
> **摘要:** For Relation Extraction (RE), the manual annotation of training data may be prohibitively expensive, since the sentences that contain the target relations in texts can be very scarce and difficult to find. It is therefore beneficial to develop an efficient method that can automatically extract training instances from unlabeled texts for training RE models. Recently, large language models (LLMs) have been adopted in various natural language processing tasks, with RE also benefiting from their advances. However, when leveraging LLMs for RE with predefined relation categories, two key challenges arise. First, in a multi-class classification setting, LLMs often struggle to comprehensively capture the semantics of every relation, leading to suboptimal results. Second, although employing binary classification for each relation individually can mitigate this issue, it introduces significant computational overhead, resulting in impractical time complexity for real-world applications. Therefore, this paper proposes a framework called M-BRe to extract training instances from unlabeled texts for RE. It utilizes three modules to combine the advantages of both of the above classification approaches: Relation Grouping, Relation Extraction, and Label Decision. Extensive experiments confirm its superior capability in discovering high-quality training samples from unlabeled texts for RE.
>
---
#### [replaced 014] Adaptive Monitoring and Real-World Evaluation of Agentic AI Systems
- **分类: cs.AI; cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2509.00115v2](http://arxiv.org/pdf/2509.00115v2)**

> **作者:** Manish Shukla
>
> **摘要:** Agentic artificial intelligence (AI) -- multi-agent systems that combine large language models with external tools and autonomous planning -- are rapidly transitioning from research laboratories into high-stakes domains. Our earlier "Basic" paper introduced a five-axis framework and proposed preliminary metrics such as goal drift and harm reduction but did not provide an algorithmic instantiation or empirical evidence. This "Advanced" sequel fills that gap. First, we revisit recent benchmarks and industrial deployments to show that technical metrics still dominate evaluations: a systematic review of 84 papers from 2023--2025 found that 83% report capability metrics while only 30% consider human-centred or economic axes [2]. Second, we formalise an Adaptive Multi-Dimensional Monitoring (AMDM) algorithm that normalises heterogeneous metrics, applies per-axis exponentially weighted moving-average thresholds and performs joint anomaly detection via the Mahalanobis distance. Third, we conduct simulations and real-world experiments. AMDM cuts anomaly-detection latency from 12.3 s to 5.6 s on simulated goal drift and reduces false-positive rates from 4.5% to 0.9% compared with static thresholds. We present a comparison table and ROC/PR curves, and we reanalyse case studies to surface missing metrics. Code, data and a reproducibility checklist accompany this paper to facilitate replication. The code supporting this work is available at https://github.com/Manishms18/Adaptive-Multi-Dimensional-Monitoring.
>
---
#### [replaced 015] MPO: Boosting LLM Agents with Meta Plan Optimization
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.02682v2](http://arxiv.org/pdf/2503.02682v2)**

> **作者:** Weimin Xiong; Yifan Song; Qingxiu Dong; Bingchan Zhao; Feifan Song; Xun Wang; Sujian Li
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Recent advancements in large language models (LLMs) have enabled LLM-based agents to successfully tackle interactive planning tasks. However, despite their successes, existing approaches often suffer from planning hallucinations and require retraining for each new agent. To address these challenges, we propose the Meta Plan Optimization (MPO) framework, , which enhances agent planning capabilities by directly incorporating explicit guidance. Unlike previous methods that rely on complex knowledge, which either require significant human effort or lack quality assurance, MPO leverages high-level general guidance through meta plans to assist agent planning and enables continuous optimization of the meta plans based on feedback from the agent's task execution. Our experiments conducted on two representative tasks demonstrate that MPO significantly outperforms existing baselines. Moreover, our analysis indicates that MPO provides a plug-and-play solution that enhances both task completion efficiency and generalization capabilities in previous unseen scenarios.
>
---
#### [replaced 016] Prior Prompt Engineering for Reinforcement Fine-Tuning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.14157v2](http://arxiv.org/pdf/2505.14157v2)**

> **作者:** Pittawat Taveekitworachai; Potsawee Manakul; Sarana Nutanong; Kunat Pipatanakul
>
> **备注:** Accepted at EMNLP 2025, Main; 26 pages, 42 figures
>
> **摘要:** This paper investigates prior prompt engineering (pPE) in the context of reinforcement fine-tuning (RFT), where language models (LMs) are incentivized to exhibit behaviors that maximize performance through reward signals. While existing RFT research has primarily focused on algorithms, reward shaping, and data curation, the design of the prior prompt--the instructions prepended to queries during training to elicit behaviors such as step-by-step reasoning--remains underexplored. We investigate whether different pPE approaches can guide LMs to internalize distinct behaviors after RFT. Inspired by inference-time prompt engineering (iPE), we translate five representative iPE strategies--reasoning, planning, code-based reasoning, knowledge recall, and null-example utilization--into corresponding pPE approaches. We experiment with Qwen2.5-7B using each of the pPE approaches, then evaluate performance on in-domain and out-of-domain benchmarks (e.g., AIME2024, HumanEval+, and GPQA-Diamond). Our results show that all pPE-trained models surpass their iPE-prompted counterparts, with the null-example pPE approach achieving the largest average performance gain and the highest improvement on AIME2024 and GPQA-Diamond, surpassing the commonly used reasoning approach. Furthermore, by adapting a behavior-classification framework, we demonstrate that different pPE strategies instill distinct behavioral styles in the resulting models. These findings position pPE as a powerful yet understudied axis for RFT.
>
---
#### [replaced 017] Meta-Semantics Augmented Few-Shot Relational Learning
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.05684v2](http://arxiv.org/pdf/2505.05684v2)**

> **作者:** Han Wu; Jie Yin
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Few-shot relational learning on knowledge graph (KGs) aims to perform reasoning over relations with only a few training examples. While existing methods have primarily focused on leveraging specific relational information, rich semantics inherent in KGs have been largely overlooked. To address this critical gap, we propose a novel prompted meta-learning (PromptMeta) framework that seamlessly integrates meta-semantics with relational information for few-shot relational learning. PromptMeta has two key innovations: (1) a Meta-Semantic Prompt (MSP) pool that learns and consolidates high-level meta-semantics, enabling effective knowledge transfer and adaptation to rare and newly emerging relations; and (2) a learnable fusion token that dynamically combines meta-semantics with task-specific relational information tailored to different few-shot tasks. Both components are optimized jointly with model parameters within a meta-learning framework. Extensive experiments and analyses on two real-world KG datasets demonstrate the effectiveness of PromptMeta in adapting to new relations with limited data.
>
---
#### [replaced 018] Pay Attention to Real World Perturbations! Natural Robustness Evaluation in Machine Reading Comprehension
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.16523v2](http://arxiv.org/pdf/2502.16523v2)**

> **作者:** Yulong Wu; Viktor Schlegel; Riza Batista-Navarro
>
> **摘要:** As neural language models achieve human-comparable performance on Machine Reading Comprehension (MRC) and see widespread adoption, ensuring their robustness in real-world scenarios has become increasingly important. Current robustness evaluation research, though, primarily develops synthetic perturbation methods, leaving unclear how well they reflect real life scenarios. Considering this, we present a framework to automatically examine MRC models on naturally occurring textual perturbations, by replacing paragraph in MRC benchmarks with their counterparts based on available Wikipedia edit history. Such perturbation type is natural as its design does not stem from an arteficial generative process, inherently distinct from the previously investigated synthetic approaches. In a large-scale study encompassing SQUAD datasets and various model architectures we observe that natural perturbations result in performance degradation in pre-trained encoder language models. More worryingly, these state-of-the-art Flan-T5 and Large Language Models (LLMs) inherit these errors. Further experiments demonstrate that our findings generalise to natural perturbations found in other more challenging MRC benchmarks. In an effort to mitigate these errors, we show that it is possible to improve the robustness to natural perturbations by training on naturally or synthetically perturbed examples, though a noticeable gap still remains compared to performance on unperturbed data.
>
---
#### [replaced 019] DomainCQA: Crafting Knowledge-Intensive QA from Domain-Specific Charts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.19498v4](http://arxiv.org/pdf/2503.19498v4)**

> **作者:** Yujing Lu; Ling Zhong; Jing Yang; Weiming Li; Peng Wei; Yongheng Wang; Manni Duan; Qing Zhang
>
> **备注:** 85 pages, 59 figures
>
> **摘要:** Chart Question Answering (CQA) evaluates Multimodal Large Language Models (MLLMs) on visual understanding and reasoning over chart data. However, existing benchmarks mostly test surface-level parsing, such as reading labels and legends, while overlooking deeper scientific reasoning. We propose DomainCQA, a framework for constructing domain-specific CQA benchmarks that emphasize both visual comprehension and knowledge-intensive reasoning. It integrates complexity-aware chart selection, multitier QA generation, and expert validation. Applied to astronomy, DomainCQA yields AstroChart, a benchmark of 1,690 QA pairs over 482 charts, exposing persistent weaknesses in fine-grained perception, numerical reasoning, and domain knowledge integration across 21 MLLMs. Fine-tuning on AstroChart improves performance across fundamental and advanced tasks. Pilot QA sets in biochemistry, economics, medicine, and social science further demonstrate DomainCQA's generality. Together, our results establish DomainCQA as a unified pipeline for constructing and augmenting domain-specific chart reasoning benchmarks.
>
---
#### [replaced 020] A Survey on Training-free Alignment of Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.09016v4](http://arxiv.org/pdf/2508.09016v4)**

> **作者:** Birong Pan; Yongqi Li; Weiyu Zhang; Wenpeng Lu; Mayi Xu; Shen Zhou; Yuanyuan Zhu; Ming Zhong; Tieyun Qian
>
> **备注:** Accepted to EMNLP 2025 (findings), camera-ready version
>
> **摘要:** The alignment of large language models (LLMs) aims to ensure their outputs adhere to human values, ethical standards, and legal norms. Traditional alignment methods often rely on resource-intensive fine-tuning (FT), which may suffer from knowledge degradation and face challenges in scenarios where the model accessibility or computational resources are constrained. In contrast, training-free (TF) alignment techniques--leveraging in-context learning, decoding-time adjustments, and post-generation corrections--offer a promising alternative by enabling alignment without heavily retraining LLMs, making them adaptable to both open-source and closed-source environments. This paper presents the first systematic review of TF alignment methods, categorizing them by stages of pre-decoding, in-decoding, and post-decoding. For each stage, we provide a detailed examination from the viewpoint of LLMs and multimodal LLMs (MLLMs), highlighting their mechanisms and limitations. Furthermore, we identify key challenges and future directions, paving the way for more inclusive and effective TF alignment techniques. By synthesizing and organizing the rapidly growing body of research, this survey offers a guidance for practitioners and advances the development of safer and more reliable LLMs.
>
---
#### [replaced 021] Speaking at the Right Level: Literacy-Controlled Counterspeech Generation with RAG-RL
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.01058v2](http://arxiv.org/pdf/2509.01058v2)**

> **作者:** Xiaoying Song; Anirban Saha Anik; Dibakar Barua; Pengcheng Luo; Junhua Ding; Lingzi Hong
>
> **备注:** Accepted at Findings of EMNLP 2025
>
> **摘要:** Health misinformation spreading online poses a significant threat to public health. Researchers have explored methods for automatically generating counterspeech to health misinformation as a mitigation strategy. Existing approaches often produce uniform responses, ignoring that the health literacy level of the audience could affect the accessibility and effectiveness of counterspeech. We propose a Controlled-Literacy framework using retrieval-augmented generation (RAG) with reinforcement learning (RL) to generate tailored counterspeech adapted to different health literacy levels. In particular, we retrieve knowledge aligned with specific health literacy levels, enabling accessible and factual information to support generation. We design a reward function incorporating subjective user preferences and objective readability-based rewards to optimize counterspeech to the target health literacy level. Experiment results show that Controlled-Literacy outperforms baselines by generating more accessible and user-preferred counterspeech. This research contributes to more equitable and impactful public health communication by improving the accessibility and comprehension of counterspeech to health misinformation
>
---
#### [replaced 022] A Dynamic Fusion Model for Consistent Crisis Response
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.01053v2](http://arxiv.org/pdf/2509.01053v2)**

> **作者:** Xiaoying Song; Anirban Saha Anik; Eduardo Blanco; Vanessa Frias-Martinez; Lingzi Hong
>
> **备注:** Accepted at Findings of EMNLP 2025
>
> **摘要:** In response to the urgent need for effective communication with crisis-affected populations, automated responses driven by language models have been proposed to assist in crisis communications. A critical yet often overlooked factor is the consistency of response style, which could affect the trust of affected individuals in responders. Despite its importance, few studies have explored methods for maintaining stylistic consistency across generated responses. To address this gap, we propose a novel metric for evaluating style consistency and introduce a fusion-based generation approach grounded in this metric. Our method employs a two-stage process: it first assesses the style of candidate responses and then optimizes and integrates them at the instance level through a fusion process. This enables the generation of high-quality responses while significantly reducing stylistic variation between instances. Experimental results across multiple datasets demonstrate that our approach consistently outperforms baselines in both response quality and stylistic uniformity.
>
---
#### [replaced 023] DischargeSim: A Simulation Benchmark for Educational Doctor-Patient Communication at Discharge
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.07188v2](http://arxiv.org/pdf/2509.07188v2)**

> **作者:** Zonghai Yao; Michael Sun; Won Seok Jang; Sunjae Kwon; Soie Kwon; Hong Yu
>
> **备注:** Equal contribution for the first two authors. To appear in the proceedings of the Main Conference on Empirical Methods in Natural Language Processing (EMNLP) 2025
>
> **摘要:** Discharge communication is a critical yet underexplored component of patient care, where the goal shifts from diagnosis to education. While recent large language model (LLM) benchmarks emphasize in-visit diagnostic reasoning, they fail to evaluate models' ability to support patients after the visit. We introduce DischargeSim, a novel benchmark that evaluates LLMs on their ability to act as personalized discharge educators. DischargeSim simulates post-visit, multi-turn conversations between LLM-driven DoctorAgents and PatientAgents with diverse psychosocial profiles (e.g., health literacy, education, emotion). Interactions are structured across six clinically grounded discharge topics and assessed along three axes: (1) dialogue quality via automatic and LLM-as-judge evaluation, (2) personalized document generation including free-text summaries and structured AHRQ checklists, and (3) patient comprehension through a downstream multiple-choice exam. Experiments across 18 LLMs reveal significant gaps in discharge education capability, with performance varying widely across patient profiles. Notably, model size does not always yield better education outcomes, highlighting trade-offs in strategy use and content prioritization. DischargeSim offers a first step toward benchmarking LLMs in post-visit clinical education and promoting equitable, personalized patient support.
>
---
#### [replaced 024] REGen: A Reliable Evaluation Framework for Generative Event Argument Extraction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.16838v2](http://arxiv.org/pdf/2502.16838v2)**

> **作者:** Omar Sharif; Joseph Gatto; Madhusudan Basak; Sarah M. Preum
>
> **备注:** Accepted at EMNLP-2025
>
> **摘要:** Event argument extraction identifies arguments for predefined event roles in text. Existing work evaluates this task with exact match (EM), where predicted arguments must align exactly with annotated spans. While suitable for span-based models, this approach falls short for large language models (LLMs), which often generate diverse yet semantically accurate arguments. EM severely underestimates performance by disregarding valid variations. Furthermore, EM evaluation fails to capture implicit arguments (unstated but inferable) and scattered arguments (distributed across a document). These limitations underscore the need for an evaluation framework that better captures models' actual performance. To bridge this gap, we introduce REGen, a Reliable Evaluation framework for Generative event argument extraction. REGen combines the strengths of exact, relaxed, and LLM-based matching to better align with human judgment. Experiments on six datasets show that REGen reveals an average performance gain of +23.93 F1 over EM, reflecting capabilities overlooked by prior evaluation. Human validation further confirms REGen's effectiveness, achieving 87.67% alignment with human assessments of argument correctness.
>
---
#### [replaced 025] SciNLP: A Domain-Specific Benchmark for Full-Text Scientific Entity and Relation Extraction in NLP
- **分类: cs.CL; cs.DL; cs.IR**

- **链接: [http://arxiv.org/pdf/2509.07801v2](http://arxiv.org/pdf/2509.07801v2)**

> **作者:** Decheng Duan; Yingyi Zhang; Jitong Peng; Chengzhi Zhang
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Structured information extraction from scientific literature is crucial for capturing core concepts and emerging trends in specialized fields. While existing datasets aid model development, most focus on specific publication sections due to domain complexity and the high cost of annotating scientific texts. To address this limitation, we introduce SciNLP - a specialized benchmark for full-text entity and relation extraction in the Natural Language Processing (NLP) domain. The dataset comprises 60 manually annotated full-text NLP publications, covering 7,072 entities and 1,826 relations. Compared to existing research, SciNLP is the first dataset providing full-text annotations of entities and their relationships in the NLP domain. To validate the effectiveness of SciNLP, we conducted comparative experiments with similar datasets and evaluated the performance of state-of-the-art supervised models on this dataset. Results reveal varying extraction capabilities of existing models across academic texts of different lengths. Cross-comparisons with existing datasets show that SciNLP achieves significant performance improvements on certain baseline models. Using models trained on SciNLP, we implemented automatic construction of a fine-grained knowledge graph for the NLP domain. Our KG has an average node degree of 3.2 per entity, indicating rich semantic topological information that enhances downstream applications. The dataset is publicly available at https://github.com/AKADDC/SciNLP.
>
---
#### [replaced 026] That's So FETCH: Fashioning Ensemble Techniques for LLM Classification in Civil Legal Intake and Referral
- **分类: cs.AI; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2509.07170v2](http://arxiv.org/pdf/2509.07170v2)**

> **作者:** Quinten Steenhuis
>
> **备注:** Submission to JURIX 2025
>
> **摘要:** Each year millions of people seek help for their legal problems by calling a legal aid program hotline, walking into a legal aid office, or using a lawyer referral service. The first step to match them to the right help is to identify the legal problem the applicant is experiencing. Misdirection has consequences. Applicants may miss a deadline, experience physical abuse, lose housing or lose custody of children while waiting to connect to the right legal help. We introduce and evaluate the FETCH classifier for legal issue classification and describe two methods for improving accuracy: a hybrid LLM/ML ensemble classification method, and the automatic generation of follow-up questions to enrich the initial problem narrative. We employ a novel data set of 419 real-world queries to a nonprofit lawyer referral service. Ultimately, we show classification accuracy (hits@2) of 97.37\% using a mix of inexpensive models, exceeding the performance of the current state-of-the-art GPT-5 model. Our approach shows promise in significantly reducing the cost of guiding users of the legal system to the right resource for their problem while achieving high accuracy.
>
---
#### [replaced 027] ACE-RL: Adaptive Constraint-Enhanced Reward for Long-form Generation Reinforcement Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.04903v2](http://arxiv.org/pdf/2509.04903v2)**

> **作者:** Jianghao Chen; Wei Sun; Qixiang Yin; Lingxing Kong; Zhixing Tan; Jiajun Zhang
>
> **备注:** Under review, our code is available at https://github.com/ZNLP/ACE-RL
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable progress in long-context understanding, yet they face significant challenges in high-quality long-form generation. Existing studies primarily suffer from two limitations: (1) A heavy reliance on scarce, high-quality long-form response data for supervised fine-tuning (SFT) or for pairwise preference reward in reinforcement learning (RL). (2) Focus on coarse-grained quality optimization dimensions, such as relevance, coherence, and helpfulness, overlooking the fine-grained specifics inherent to diverse long-form generation scenarios. To address this issue, we propose a framework using Adaptive Constraint-Enhanced reward for long-form generation Reinforcement Learning (ACE-RL). ACE-RL first automatically deconstructs each instruction into a set of fine-grained, adaptive constraint criteria by identifying its underlying intents and demands. Subsequently, we design a reward mechanism that quantifies the quality of long-form responses based on their satisfaction over corresponding constraints, converting subjective quality evaluation into constraint verification. Finally, we utilize reinforcement learning to guide models toward superior long-form generation capabilities. Experimental results demonstrate that our ACE-RL framework significantly outperforms existing SFT and RL baselines by 20.70% and 7.32% on WritingBench, and our top-performing model even surpasses proprietary systems like GPT-4o by 7.10%, providing a more effective training paradigm for LLMs to generate high-quality content across diverse long-form generation scenarios.
>
---
#### [replaced 028] Baba Is AI: Break the Rules to Beat the Benchmark
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.13729v2](http://arxiv.org/pdf/2407.13729v2)**

> **作者:** Nathan Cloos; Meagan Jens; Michelangelo Naim; Yen-Ling Kuo; Ignacio Cases; Andrei Barbu; Christopher J. Cueva
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Humans solve problems by following existing rules and procedures, and also by leaps of creativity to redefine those rules and objectives. To probe these abilities, we developed a new benchmark based on the game Baba Is You where an agent manipulates both objects in the environment and rules, represented by movable tiles with words written on them, to reach a specified goal and win the game. We test three state-of-the-art multi-modal large language models (OpenAI GPT-4o, Google Gemini-1.5-Pro and Gemini-1.5-Flash) and find that they fail dramatically when generalization requires that the rules of the game must be manipulated and combined.
>
---
#### [replaced 029] MachineLearningLM: Scaling Many-shot In-context Learning via Continued Pretraining
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.06806v2](http://arxiv.org/pdf/2509.06806v2)**

> **作者:** Haoyu Dong; Pengkun Zhang; Mingzhe Lu; Yanzhen Shen; Guolin Ke
>
> **摘要:** Large language models (LLMs) possess broad world knowledge and strong general-purpose reasoning ability, yet they struggle to learn from many in-context examples on standard machine learning (ML) tasks, that is, to leverage many-shot demonstrations purely via in-context learning (ICL) without gradient descent. We introduce MachineLearningLM, a portable continued-pretraining framework that equips a general-purpose LLM with robust in-context ML capability while preserving its general knowledge and reasoning for broader chat workflows. Our pretraining procedure synthesizes ML tasks from millions of structural causal models (SCMs), spanning shot counts up to 1,024. We begin with a random-forest teacher, distilling tree-based decision strategies into the LLM to strengthen robustness in numerical modeling. All tasks are serialized with a token-efficient prompt, enabling 3x to 6x more examples per context window and delivering up to 50x amortized throughput via batch inference. Despite a modest setup (Qwen-2.5-7B-Instruct with LoRA rank 8), MachineLearningLM outperforms strong LLM baselines (e.g., GPT-5-mini) by an average of about 15% on out-of-distribution tabular classification across finance, physics, biology, and healthcare domains. It exhibits a striking many-shot scaling law: accuracy increases monotonically as in-context demonstrations grow from 8 to 1,024. Without any task-specific training, it attains random-forest-level accuracy across hundreds of shots. General chat capabilities, including knowledge and reasoning, are preserved: it achieves 75.4% on MMLU.
>
---
#### [replaced 030] HIRAG: Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.05714v3](http://arxiv.org/pdf/2507.05714v3)**

> **作者:** YiHan Jiao; ZheHao Tan; Dan Yang; DuoLin Sun; Jie Feng; Yue Shen; Jian Wang; Peng Wei
>
> **摘要:** Retrieval-augmented generation (RAG) has become a fundamental paradigm for addressing the challenges faced by large language models in handling real-time information and domain-specific problems. Traditional RAG systems primarily rely on the in-context learning (ICL) capabilities of the large language model itself. Still, in-depth research on the specific capabilities needed by the RAG generation model is lacking, leading to challenges with inconsistent document quality and retrieval system imperfections. Even the limited studies that fine-tune RAG generative models often \textit{lack a granular focus on RAG task} or \textit{a deeper utilization of chain-of-thought processes}. To address this, we propose that RAG models should possess three progressively hierarchical abilities (1) Filtering: the ability to select relevant information; (2) Combination: the ability to combine semantic information across paragraphs; and (3) RAG-specific reasoning: the ability to further process external knowledge using internal knowledge. Thus, we introduce our new RAG instruction fine-tuning method, Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation (HIRAG) incorporates a "think before answering" strategy. This method enhances the model's open-book examination capability by utilizing multi-level progressive chain-of-thought. Experiments show that the HIRAG training strategy significantly improves the model's performance on datasets such as RGB, PopQA, MuSiQue, HotpotQA, and PubmedQA.
>
---
#### [replaced 031] Recursive Training Loops in LLMs: How training data properties modulate distribution shift in generated data?
- **分类: cs.LG; cs.AI; cs.CL; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2504.03814v4](http://arxiv.org/pdf/2504.03814v4)**

> **作者:** Grgur Kovač; Jérémy Perez; Rémy Portelas; Peter Ford Dominey; Pierre-Yves Oudeyer
>
> **备注:** Accepted to EMNLP 2025 (Oral)
>
> **摘要:** Large language models (LLMs) are increasingly used in the creation of online content, creating feedback loops as subsequent generations of models will be trained on this synthetic data. Such loops were shown to lead to distribution shifts - models misrepresenting the true underlying distributions of human data (also called model collapse). However, how human data properties affect such shifts remains poorly understood. In this paper, we provide the first empirical examination of the effect of such properties on the outcome of recursive training. We first confirm that using different human datasets leads to distribution shifts of different magnitudes. Through exhaustive manipulation of dataset properties combined with regression analyses, we then identify a set of properties predicting distribution shift magnitudes. Lexical diversity is found to amplify these shifts, while semantic diversity and data quality mitigate them. Furthermore, we find that these influences are highly modular: data scrapped from a given internet domain has little influence on the content generated for another domain. Finally, experiments on political bias reveal that human data properties affect whether the initial bias will be amplified or reduced. Overall, our results portray a novel view, where different parts of internet may undergo different types of distribution shift.
>
---
#### [replaced 032] Beyond One-Size-Fits-All: Inversion Learning for Highly Effective NLG Evaluation Prompts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.21117v3](http://arxiv.org/pdf/2504.21117v3)**

> **作者:** Hanhua Hong; Chenghao Xiao; Yang Wang; Yiqi Liu; Wenge Rong; Chenghua Lin
>
> **备注:** 11 pages, accepted by Transactions of the Association for Computational Linguistics (TACL)
>
> **摘要:** Evaluating natural language generation systems is challenging due to the diversity of valid outputs. While human evaluation is the gold standard, it suffers from inconsistencies, lack of standardisation, and demographic biases, limiting reproducibility. LLM-based evaluators offer a scalable alternative but are highly sensitive to prompt design, where small variations can lead to significant discrepancies. In this work, we propose an inversion learning method that learns effective reverse mappings from model outputs back to their input instructions, enabling the automatic generation of highly effective, model-specific evaluation prompts. Our method requires only a single evaluation sample and eliminates the need for time-consuming manual prompt engineering, thereby improving both efficiency and robustness. Our work contributes toward a new direction for more robust and efficient LLM-based evaluation.
>
---
#### [replaced 033] How Far Are We from Optimal Reasoning Efficiency?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.07104v2](http://arxiv.org/pdf/2506.07104v2)**

> **作者:** Jiaxuan Gao; Shu Yan; Qixin Tan; Lu Yang; Shusheng Xu; Wei Fu; Zhiyu Mei; Kaifeng Lyu; Yi Wu
>
> **摘要:** Large Reasoning Models (LRMs) demonstrate remarkable problem-solving capabilities through extended Chain-of-Thought (CoT) reasoning but often produce excessively verbose and redundant reasoning traces. This inefficiency incurs high inference costs and limits practical deployment. While existing fine-tuning methods aim to improve reasoning efficiency, assessing their efficiency gains remains challenging due to inconsistent evaluations. In this work, we introduce the reasoning efficiency frontiers, empirical upper bounds derived from fine-tuning base LRMs across diverse approaches and training configurations. Based on these frontiers, we propose the Reasoning Efficiency Gap (REG), a unified metric quantifying deviations of any fine-tuned LRMs from these frontiers. Systematic evaluation on challenging mathematical benchmarks reveals significant gaps in current methods: they either sacrifice accuracy for short length or still remain inefficient under tight token budgets. To reduce the efficiency gap, we propose REO-RL, a class of Reinforcement Learning algorithms that minimizes REG by targeting a sparse set of token budgets. Leveraging numerical integration over strategically selected budgets, REO-RL approximates the full efficiency objective with low error using a small set of token budgets. Through systematic benchmarking, we demonstrate that our efficiency metric, REG, effectively captures the accuracy-length trade-off, with low-REG methods reducing length while maintaining accuracy. Our approach, REO-RL, consistently reduces REG by >=50 across all evaluated LRMs and matching Qwen3-4B/8B efficiency frontiers under a 16K token budget with minimal accuracy loss. Ablation studies confirm the effectiveness of our exponential token budget strategy. Finally, our findings highlight that fine-tuning LRMs to perfectly align with the efficiency frontiers remains an open challenge.
>
---
#### [replaced 034] Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.07976v3](http://arxiv.org/pdf/2508.07976v3)**

> **作者:** Jiaxuan Gao; Wei Fu; Minyang Xie; Shusheng Xu; Chuyi He; Zhiyu Mei; Banghua Zhu; Yi Wu
>
> **摘要:** Recent advancements in LLM-based agents have demonstrated remarkable capabilities in handling complex, knowledge-intensive tasks by integrating external tools. Among diverse choices of tools, search tools play a pivotal role in accessing vast external knowledge. However, open-source agents still fall short of achieving expert-level Search Intelligence, the ability to resolve ambiguous queries, generate precise searches, analyze results, and conduct thorough exploration. Existing approaches fall short in scalability, efficiency, and data quality. For example, small turn limits in existing online RL methods, e.g. <=10, restrict complex strategy learning. This paper introduces ASearcher, an open-source project for large-scale RL training of search agents. Our key contributions include: (1) Scalable fully asynchronous RL training that enables long-horizon search while maintaining high training efficiency. (2) A prompt-based LLM agent that autonomously synthesizes high-quality and challenging QAs, creating a large-scale QA dataset. Through RL training, our prompt-based QwQ-32B agent achieves substantial improvements, with 46.7% and 20.8% Avg@4 gains on xBench and GAIA, respectively. Notably, our agent exhibits extreme long-horizon search, with tool calls exceeding 40 turns and output tokens exceeding 150k during training time. With a simple agent design and no external LLMs, ASearcher-Web-QwQ achieves Avg@4 scores of 42.1 on xBench and 52.8 on GAIA, surpassing existing open-source 32B agents. We open-source our models, training data, and codes in https://github.com/inclusionAI/ASearcher.
>
---
#### [replaced 035] Self-Correcting Decoding with Generative Feedback for Mitigating Hallucinations in Large Vision-Language Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.06130v2](http://arxiv.org/pdf/2502.06130v2)**

> **作者:** Ce Zhang; Zifu Wan; Zhehan Kan; Martin Q. Ma; Simon Stepputtis; Deva Ramanan; Russ Salakhutdinov; Louis-Philippe Morency; Katia Sycara; Yaqi Xie
>
> **备注:** Accepted by ICLR 2025. Project page: https://zhangce01.github.io/DeGF/
>
> **摘要:** While recent Large Vision-Language Models (LVLMs) have shown remarkable performance in multi-modal tasks, they are prone to generating hallucinatory text responses that do not align with the given visual input, which restricts their practical applicability in real-world scenarios. In this work, inspired by the observation that the text-to-image generation process is the inverse of image-conditioned response generation in LVLMs, we explore the potential of leveraging text-to-image generative models to assist in mitigating hallucinations in LVLMs. We discover that generative models can offer valuable self-feedback for mitigating hallucinations at both the response and token levels. Building on this insight, we introduce self-correcting Decoding with Generative Feedback (DeGF), a novel training-free algorithm that incorporates feedback from text-to-image generative models into the decoding process to effectively mitigate hallucinations in LVLMs. Specifically, DeGF generates an image from the initial response produced by LVLMs, which acts as an auxiliary visual reference and provides self-feedback to verify and correct the initial response through complementary or contrastive decoding. Extensive experimental results validate the effectiveness of our approach in mitigating diverse types of hallucinations, consistently surpassing state-of-the-art methods across six benchmarks. Code is available at https://github.com/zhangce01/DeGF.
>
---
#### [replaced 036] CURE: Controlled Unlearning for Robust Embeddings - Mitigating Conceptual Shortcuts in Pre-Trained Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.05230v2](http://arxiv.org/pdf/2509.05230v2)**

> **作者:** Aysenur Kocak; Shuo Yang; Bardh Prenkaj; Gjergji Kasneci
>
> **备注:** Accepted at the Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)
>
> **摘要:** Pre-trained language models have achieved remarkable success across diverse applications but remain susceptible to spurious, concept-driven correlations that impair robustness and fairness. In this work, we introduce CURE, a novel and lightweight framework that systematically disentangles and suppresses conceptual shortcuts while preserving essential content information. Our method first extracts concept-irrelevant representations via a dedicated content extractor reinforced by a reversal network, ensuring minimal loss of task-relevant information. A subsequent controllable debiasing module employs contrastive learning to finely adjust the influence of residual conceptual cues, enabling the model to either diminish harmful biases or harness beneficial correlations as appropriate for the target task. Evaluated on the IMDB and Yelp datasets using three pre-trained architectures, CURE achieves an absolute improvement of +10 points in F1 score on IMDB and +2 points on Yelp, while introducing minimal computational overhead. Our approach establishes a flexible, unsupervised blueprint for combating conceptual biases, paving the way for more reliable and fair language understanding systems.
>
---
#### [replaced 037] TweakLLM: A Routing Architecture for Dynamic Tailoring of Cached Responses
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.23674v2](http://arxiv.org/pdf/2507.23674v2)**

> **作者:** Muhammad Taha Cheema; Abeer Aamir; Khawaja Gul Muhammad; Naveed Anwar Bhatti; Ihsan Ayyub Qazi; Zafar Ayyub Qazi
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Large Language Models (LLMs) process millions of queries daily, making efficient response caching a compelling optimization for reducing cost and latency. However, preserving relevance to user queries using this approach proves difficult due to the personalized nature of chatbot interactions and the limited accuracy of semantic similarity search. To address this, we present TweakLLM, a novel routing architecture that employs a lightweight LLM to dynamically adapt cached responses to incoming prompts. Through comprehensive evaluation, including user studies with side-by-side comparisons, satisfaction voting, as well as multi-agent LLM debates, we demonstrate that TweakLLM maintains response quality comparable to frontier models while significantly improving cache effectiveness. Our results across real-world datasets highlight TweakLLM as a scalable, resource-efficient caching solution for high-volume LLM deployments without compromising user experience.
>
---
#### [replaced 038] Arce: Augmented Roberta with Contextualized Elucidations for Ner in Automated Rule Checking
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2508.07286v2](http://arxiv.org/pdf/2508.07286v2)**

> **作者:** Jian Chen; Jinbao Tian; Yankui Li; Yuqi Lu; Zhou Li
>
> **摘要:** Accurate information extraction from specialized texts is a critical challenge, particularly for named entity recognition (NER) in the architecture, engineering, and construction (AEC) domain to support automated rule checking (ARC). The performance of standard pre-trained models is often constrained by the domain gap, as they struggle to interpret the specialized terminology and complex relational contexts inherent in AEC texts. Although this issue can be mitigated by further pre-training on large, human-curated domain corpora, as exemplified by methods like ARCBERT, this approach is both labor-intensive and cost-prohibitive. Consequently, leveraging large language models (LLMs) for automated knowledge generation has emerged as a promising alternative. However, the optimal strategy for generating knowledge that can genuinely enhance smaller, efficient models remains an open question. To address this, we propose ARCE (augmented RoBERTa with contextualized elucidations), a novel approach that systematically explores and optimizes this generation process. ARCE employs an LLM to first generate a corpus of simple, direct explanations, which we term Cote, and then uses this corpus to incrementally pre-train a RoBERTa model prior to its fine-tuning on the downstream task. Our extensive experiments show that ARCE establishes a new state-of-the-art on a benchmark AEC dataset, achieving a Macro-F1 score of 77.20%. This result also reveals a key finding: simple, explanation-based knowledge proves surprisingly more effective than complex, role-based rationales for this task. The code is publicly available at:https://github.com/nxcc-lab/ARCE.
>
---
#### [replaced 039] VIDEE: Visual and Interactive Decomposition, Execution, and Evaluation of Text Analytics with Intelligent Agents
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.21582v3](http://arxiv.org/pdf/2506.21582v3)**

> **作者:** Sam Yu-Te Lee; Chenyang Ji; Shicheng Wen; Lifu Huang; Dongyu Liu; Kwan-Liu Ma
>
> **摘要:** Text analytics has traditionally required specialized knowledge in Natural Language Processing (NLP) or text analysis, which presents a barrier for entry-level analysts. Recent advances in large language models (LLMs) have changed the landscape of NLP by enabling more accessible and automated text analysis (e.g., topic detection, summarization, information extraction, etc.). We introduce VIDEE, a system that supports entry-level data analysts to conduct advanced text analytics with intelligent agents. VIDEE instantiates a human-agent collaroration workflow consisting of three stages: (1) Decomposition, which incorporates a human-in-the-loop Monte-Carlo Tree Search algorithm to support generative reasoning with human feedback, (2) Execution, which generates an executable text analytics pipeline, and (3) Evaluation, which integrates LLM-based evaluation and visualizations to support user validation of execution results. We conduct two quantitative experiments to evaluate VIDEE's effectiveness and analyze common agent errors. A user study involving participants with varying levels of NLP and text analytics experience -- from none to expert -- demonstrates the system's usability and reveals distinct user behavior patterns. The findings identify design implications for human-agent collaboration, validate the practical utility of VIDEE for non-expert users, and inform future improvements to intelligent text analytics systems.
>
---
#### [replaced 040] Scaling Video-Language Models to 10K Frames via Hierarchical Differential Distillation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.02438v5](http://arxiv.org/pdf/2504.02438v5)**

> **作者:** Chuanqi Cheng; Jian Guan; Wei Wu; Rui Yan
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Long-form video processing fundamentally challenges vision-language models (VLMs) due to the high computational costs of handling extended temporal sequences. Existing token pruning and feature merging methods often sacrifice critical temporal dependencies or dilute semantic information. We introduce differential distillation, a principled approach that systematically preserves task-relevant information while suppressing redundancy. Based on this principle, we develop ViLAMP, a hierarchical video-language model that processes hour-long videos at "mixed precision" through two key mechanisms: (1) differential keyframe selection that maximizes query relevance while maintaining temporal distinctiveness at the frame level and (2) differential feature merging that preserves query-salient features in non-keyframes at the patch level. Hence, ViLAMP retains full information in keyframes while reducing non-keyframes to their most salient features, resembling mixed-precision training. Extensive experiments demonstrate ViLAMP's superior performance across four video understanding benchmarks, particularly on long-form content. Notably, ViLAMP can process ultra-long videos (up to 10K frames) on a single NVIDIA A100 GPU, achieving substantial computational efficiency while maintaining state-of-the-art performance. Code and model are available at https://github.com/steven-ccq/ViLAMP.
>
---
#### [replaced 041] Drivel-ology: Challenging LLMs with Interpreting Nonsense with Depth
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.03867v2](http://arxiv.org/pdf/2509.03867v2)**

> **作者:** Yang Wang; Chenghao Xiao; Chia-Yi Hsiao; Zi Yan Chang; Chi-Li Chen; Tyler Loakman; Chenghua Lin
>
> **备注:** Accepted for oral presentation at the EMNLP 2025 Main Conference
>
> **摘要:** We introduce Drivelology, a unique linguistic phenomenon characterised as "nonsense with depth" - utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, or rhetorically subversive. While such expressions may resemble surface-level nonsense, they encode implicit meaning requiring contextual inference, moral reasoning, or emotional interpretation. We find that current large language models (LLMs), despite excelling at many natural language processing (NLP) tasks, consistently fail to grasp the layered semantics of Drivelological text. To investigate this, we construct a benchmark dataset of over 1,200+ meticulously curated and diverse examples across English, Mandarin, Spanish, French, Japanese, and Korean. Each example underwent careful expert review to verify its Drivelological characteristics, involving multiple rounds of discussion and adjudication to address disagreements. Using this dataset, we evaluate a range of LLMs on classification, generation, and reasoning tasks. Our results reveal clear limitations of LLMs: models often confuse Drivelology with shallow nonsense, produce incoherent justifications, or miss implied rhetorical functions altogether. These findings highlight a deep representational gap in LLMs' pragmatic understanding and challenge the assumption that statistical fluency implies cognitive comprehension. We release our dataset and code to facilitate further research in modelling linguistic depth beyond surface-level coherence.
>
---
#### [replaced 042] CoT-RAG: Integrating Chain of Thought and Retrieval-Augmented Generation to Enhance Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.13534v3](http://arxiv.org/pdf/2504.13534v3)**

> **作者:** Feiyang Li; Peng Fang; Zhan Shi; Arijit Khan; Fang Wang; Weihao Wang; Xin Zhang; Yongjian Cui
>
> **摘要:** Chain-of-thought (CoT) reasoning boosts large language models' (LLMs) performance on complex tasks but faces two key limitations: a lack of reliability when solely relying on LLM-generated reasoning chains and lower reasoning performance from natural language prompts compared with code prompts. To address these issues, we propose CoT-RAG, a novel reasoning framework with three key designs: (i) Knowledge Graph-driven CoT Generation, featuring knowledge graphs to modulate reasoning chain generation of LLMs, thereby enhancing reasoning credibility; (ii) Learnable Knowledge Case-aware RAG, which incorporates retrieval-augmented generation (RAG) into knowledge graphs to retrieve relevant sub-cases and sub-descriptions, providing LLMs with learnable information; (iii) Pseudo Program Prompting Execution, which promotes greater logical rigor by guiding LLMs to execute reasoning tasks as pseudo-programs. Evaluations on nine public datasets spanning three reasoning tasks reveal significant accuracy gains-ranging from 4.0% to 44.3%-over state-of-the-art methods. Furthermore, tests on four domain-specific datasets demonstrate exceptional accuracy and efficient execution, underscoring its practical applicability and scalability. Our code and data are available at https: //github.com/hustlfy123/CoT-RAG.
>
---
#### [replaced 043] Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.15337v3](http://arxiv.org/pdf/2505.15337v3)**

> **作者:** Hao Fang; Jiawei Kong; Tianqu Zhuang; Yixiang Qiu; Kuofeng Gao; Bin Chen; Shu-Tao Xia; Yaowei Wang; Min Zhang
>
> **备注:** Accepted by EMNLP-2025
>
> **摘要:** The misuse of large language models (LLMs), such as academic plagiarism, has driven the development of detectors to identify LLM-generated texts. To bypass these detectors, paraphrase attacks have emerged to purposely rewrite these texts to evade detection. Despite the success, existing methods require substantial data and computational budgets to train a specialized paraphraser, and their attack efficacy greatly reduces when faced with advanced detection algorithms. To address this, we propose \textbf{Co}ntrastive \textbf{P}araphrase \textbf{A}ttack (CoPA), a training-free method that effectively deceives text detectors using off-the-shelf LLMs. The first step is to carefully craft instructions that encourage LLMs to produce more human-like texts. Nonetheless, we observe that the inherent statistical biases of LLMs can still result in some generated texts carrying certain machine-like attributes that can be captured by detectors. To overcome this, CoPA constructs an auxiliary machine-like word distribution as a contrast to the human-like distribution generated by the LLM. By subtracting the machine-like patterns from the human-like distribution during the decoding process, CoPA is able to produce sentences that are less discernible by text detectors. Our theoretical analysis suggests the superiority of the proposed attack. Extensive experiments validate the effectiveness of CoPA in fooling text detectors across various scenarios.
>
---

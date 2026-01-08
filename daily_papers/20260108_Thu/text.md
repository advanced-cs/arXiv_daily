# 自然语言处理 cs.CL

- **最新发布 131 篇**

- **更新 88 篇**

## 最新发布

#### [new 001] Self-Explaining Hate Speech Detection with Moral Rationales
- **分类: cs.CL**

- **简介: 该论文属于仇恨言论检测任务，旨在解决模型依赖表面特征、解释不准确的问题。提出SMRA框架，结合道德理由监督，提升模型可解释性和准确性。**

- **链接: [https://arxiv.org/pdf/2601.03481v1](https://arxiv.org/pdf/2601.03481v1)**

> **作者:** Francielle Vargas; Jackson Trager; Diego Alves; Surendrabikram Thapa; Matteo Guida; Berk Atil; Daryna Dementieva; Andrew Smart; Ameeta Agrawal
>
> **摘要:** Hate speech detection models rely on surface-level lexical features, increasing vulnerability to spurious correlations and limiting robustness, cultural contextualization, and interpretability. We propose Supervised Moral Rationale Attention (SMRA), the first self-explaining hate speech detection framework to incorporate moral rationales as direct supervision for attention alignment. Based on Moral Foundations Theory, SMRA aligns token-level attention with expert-annotated moral rationales, guiding models to attend to morally salient spans rather than spurious lexical patterns. Unlike prior rationale-supervised or post-hoc approaches, SMRA integrates moral rationale supervision directly into the training objective, producing inherently interpretable and contextualized explanations. To support our framework, we also introduce HateBRMoralXplain, a Brazilian Portuguese benchmark dataset annotated with hate labels, moral categories, token-level moral rationales, and socio-political metadata. Across binary hate speech detection and multi-label moral sentiment classification, SMRA consistently improves performance (e.g., +0.9 and +1.5 F1, respectively) while substantially enhancing explanation faithfulness, increasing IoU F1 (+7.4 pp) and Token F1 (+5.0 pp). Although explanations become more concise, sufficiency improves (+2.3 pp) and fairness remains stable, indicating more faithful rationales without performance or bias trade-offs
>
---
#### [new 002] Evaluating Small Decoder-Only Language Models for Grammar Correction and Text Simplification
- **分类: cs.CL**

- **简介: 论文研究小规模解码器语言模型在语法修正和文本简化任务中的表现，旨在探索其作为高效替代方案的可行性。实验表明，尽管效率高，但性能仍低于大模型。**

- **链接: [https://arxiv.org/pdf/2601.03874v1](https://arxiv.org/pdf/2601.03874v1)**

> **作者:** Anthony Lamelas
>
> **备注:** 9 pages, 12 figures
>
> **摘要:** Large language models have become extremely popular recently due to their ability to achieve strong performance on a variety of tasks, such as text generation and rewriting, but their size and computation cost make them difficult to access, deploy, and secure in many settings. This paper investigates whether small, decoder-only language models can provide an efficient alternative for the tasks of grammar correction and text simplification. The experiments in this paper focus on testing small language models out of the box, fine-tuned, and run sequentially on the JFLEG and ASSET datasets using established metrics. The results show that while SLMs may learn certain behaviors well, their performance remains below strong baselines and current LLMs. The results also show that SLMs struggle with retaining meaning and hallucinations. These findings suggest that despite their efficiency advantages, current SLMs are not yet competitive enough with modern LLMs for rewriting, and further advances in training are required for SLMs to close the performance gap between them and today's LLMs.
>
---
#### [new 003] Topic Segmentation Using Generative Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于主题分割任务，旨在解决传统方法在长距离依赖和知识广度上的不足。工作提出一种基于生成语言模型的重叠递归提示策略，提升分割效果。**

- **链接: [https://arxiv.org/pdf/2601.03276v1](https://arxiv.org/pdf/2601.03276v1)**

> **作者:** Pierre Mackenzie; Maya Shah; Patrick Frenett
>
> **摘要:** Topic segmentation using generative Large Language Models (LLMs) remains relatively unexplored. Previous methods use semantic similarity between sentences, but such models lack the long range dependencies and vast knowledge found in LLMs. In this work, we propose an overlapping and recursive prompting strategy using sentence enumeration. We also support the adoption of the boundary similarity evaluation metric. Results show that LLMs can be more effective segmenters than existing methods, but issues remain to be solved before they can be relied upon for topic segmentation.
>
---
#### [new 004] DiffCoT: Diffusion-styled Chain-of-Thought Reasoning in LLMs
- **分类: cs.CL**

- **简介: 该论文属于多步骤推理任务，旨在解决CoT推理中的错误累积问题。提出DiffCoT框架，通过扩散机制实现推理步骤的生成与修正。**

- **链接: [https://arxiv.org/pdf/2601.03559v1](https://arxiv.org/pdf/2601.03559v1)**

> **作者:** Shidong Cao; Hongzhan Lin; Yuxuan Gu; Ziyang Luo; Jing Ma
>
> **备注:** DiffCoT improves multi-step LLM reasoning by applying diffusion-based iterative denoising to correct intermediate Chain-of-Thought steps
>
> **摘要:** Chain-of-Thought (CoT) reasoning improves multi-step mathematical problem solving in large language models but remains vulnerable to exposure bias and error accumulation, as early mistakes propagate irreversibly through autoregressive decoding. In this work, we propose DiffCoT, a diffusion-styled CoT framework that reformulates CoT reasoning as an iterative denoising process. DiffCoT integrates diffusion principles at the reasoning-step level via a sliding-window mechanism, enabling unified generation and retrospective correction of intermediate steps while preserving token-level autoregression. To maintain causal consistency, we further introduce a causal diffusion noise schedule that respects the temporal structure of reasoning chains. Extensive experiments on three multi-step CoT reasoning benchmarks across diverse model backbones demonstrate that DiffCoT consistently outperforms existing CoT preference optimization methods, yielding improved robustness and error-correction capability in CoT reasoning.
>
---
#### [new 005] Large-Scale Aspect-Based Sentiment Analysis with Reasoning-Infused LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于方面情感分析任务，解决多类情感识别与多语言支持问题。通过扩展情感类别、引入推理预训练方法，提升了模型性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.03940v1](https://arxiv.org/pdf/2601.03940v1)**

> **作者:** Paweł Liskowski; Krzysztof Jankowski
>
> **摘要:** We introduce Arctic-ABSA, a collection of powerful models for real-life aspect-based sentiment analysis (ABSA). Our models are tailored to commercial needs, trained on a large corpus of public data alongside carefully generated synthetic data, resulting in a dataset 20 times larger than SemEval14. We extend typical ABSA models by expanding the number of sentiment classes from the standard three (positive, negative, neutral) to five, adding mixed and unknown classes, while also jointly predicting overall text sentiment and supporting multiple languages. We experiment with reasoning injection by fine-tuning on Chain-of-Thought (CoT) examples and introduce a novel reasoning pretraining technique for encoder-only models that significantly improves downstream fine-tuning and generalization. Our 395M-parameter encoder and 8B-parameter decoder achieve up to 10 percentage points higher accuracy than GPT-4o and Claude 3.5 Sonnet, while setting new state-of-the-art results on the SemEval14 benchmark. A single multilingual model maintains 87-91% accuracy across six languages without degrading English performance. We release ABSA-mix, a large-scale benchmark aggregating 17 public ABSA datasets across 92 domains.
>
---
#### [new 006] Implicit Graph, Explicit Retrieval: Towards Efficient and Interpretable Long-horizon Memory for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大语言模型的长期记忆任务，旨在解决长上下文下信息检索效率与可解释性问题。提出LatentGraphMem框架，结合隐式图记忆与显式子图检索，提升性能并支持灵活扩展。**

- **链接: [https://arxiv.org/pdf/2601.03417v1](https://arxiv.org/pdf/2601.03417v1)**

> **作者:** Xin Zhang; Kailai Yang; Hao Li; Chenyue Li; Qiyu Wei; Sophia Ananiadou
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Long-horizon applications increasingly require large language models (LLMs) to answer queries when relevant evidence is sparse and dispersed across very long contexts. Existing memory systems largely follow two paradigms: explicit structured memories offer interpretability but often become brittle under long-context overload, while latent memory mechanisms are efficient and stable yet difficult to inspect. We propose LatentGraphMem, a memory framework that combines implicit graph memory with explicit subgraph retrieval. LatentGraphMem stores a graph-structured memory in latent space for stability and efficiency, and exposes a task-specific subgraph retrieval interface that returns a compact symbolic subgraph under a fixed budget for downstream reasoning and human inspection. During training, an explicit graph view is materialized to interface with a frozen reasoner for question-answering supervision. At inference time, retrieval is performed in latent space and only the retrieved subgraph is externalized. Experiments on long-horizon benchmarks across multiple model scales show that LatentGraphMem consistently outperforms representative explicit-graph and latent-memory baselines, while enabling parameter-efficient adaptation and flexible scaling to larger reasoners without introducing large symbolic artifacts.
>
---
#### [new 007] Evaluation of Multilingual LLMs Personalized Text Generation Capabilities Targeting Groups and Social-Media Platforms
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言大模型生成个性化文本的能力，探讨其在不同群体和社交媒体平台上的应用与影响。任务为评估多语言模型的个性化生成效果及其可检测性。**

- **链接: [https://arxiv.org/pdf/2601.03752v1](https://arxiv.org/pdf/2601.03752v1)**

> **作者:** Dominik Macko
>
> **摘要:** Capabilities of large language models to generate multilingual coherent text have continuously enhanced in recent years, which opens concerns about their potential misuse. Previous research has shown that they can be misused for generation of personalized disinformation in multiple languages. It has also been observed that personalization negatively affects detectability of machine-generated texts; however, this has been studied in the English language only. In this work, we examine this phenomenon across 10 languages, while we focus not only on potential misuse of personalization capabilities, but also on potential benefits they offer. Overall, we cover 1080 combinations of various personalization aspects in the prompts, for which the texts are generated by 16 distinct language models (17,280 texts in total). Our results indicate that there are differences in personalization quality of the generated texts when targeting demographic groups and when targeting social-media platforms across languages. Personalization towards platforms affects detectability of the generated texts in a higher scale, especially in English, where the personalization quality is the highest.
>
---
#### [new 008] Analyzing Reasoning Shifts in Audio Deepfake Detection under Adversarial Attacks: The Reasoning Tax versus Shield Bifurcation
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，研究对抗攻击下推理变化问题。通过框架分析ALMs在声学感知、认知一致性和矛盾性方面的鲁棒性，揭示推理的双重影响。**

- **链接: [https://arxiv.org/pdf/2601.03615v1](https://arxiv.org/pdf/2601.03615v1)**

> **作者:** Binh Nguyen; Thai Le
>
> **备注:** Preprint for ACL 2026 submission
>
> **摘要:** Audio Language Models (ALMs) offer a promising shift towards explainable audio deepfake detections (ADDs), moving beyond \textit{black-box} classifiers by providing some level of transparency into their predictions via reasoning traces. This necessitates a new class of model robustness analysis: robustness of the predictive reasoning under adversarial attacks, which goes beyond existing paradigm that mainly focuses on the shifts of the final predictions (e.g., fake v.s. real). To analyze such reasoning shifts, we introduce a forensic auditing framework to evaluate the robustness of ALMs' reasoning under adversarial attacks in three inter-connected dimensions: acoustic perception, cognitive coherence, and cognitive dissonance. Our systematic analysis reveals that explicit reasoning does not universally enhance robustness. Instead, we observe a bifurcation: for models exhibiting robust acoustic perception, reasoning acts as a defensive \textit{``shield''}, protecting them from adversarial attacks. However, for others, it imposes a performance \textit{``tax''}, particularly under linguistic attacks which reduce cognitive coherence and increase attack success rate. Crucially, even when classification fails, high cognitive dissonance can serve as a \textit{silent alarm}, flagging potential manipulation. Overall, this work provides a critical evaluation of the role of reasoning in forensic audio deepfake analysis and its vulnerabilities.
>
---
#### [new 009] Stuttering-Aware Automatic Speech Recognition for Indonesian Language
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决印尼语口吃语音识别性能下降的问题。通过生成合成口吃数据并微调预训练模型，提升对口吃语音的识别能力。**

- **链接: [https://arxiv.org/pdf/2601.03727v1](https://arxiv.org/pdf/2601.03727v1)**

> **作者:** Fadhil Muhammad; Alwin Djuliansah; Adrian Aryaputra Hamzah; Kurniawati Azizah
>
> **备注:** Preprint
>
> **摘要:** Automatic speech recognition systems have achieved remarkable performance on fluent speech but continue to degrade significantly when processing stuttered speech, a limitation that is particularly acute for low-resource languages like Indonesian where specialized datasets are virtually non-existent. To overcome this scarcity, we propose a data augmentation framework that generates synthetic stuttered audio by injecting repetitions and prolongations into fluent text through a combination of rule-based transformations and large language models followed by text-to-speech synthesis. We apply this synthetic data to fine-tune a pre-trained Indonesian Whisper model using transfer learning, enabling the architecture to adapt to dysfluent acoustic patterns without requiring large-scale real-world recordings. Our experiments demonstrate that this targeted synthetic exposure consistently reduces recognition errors on stuttered speech while maintaining performance on fluent segments, validating the utility of synthetic data pipelines for developing more inclusive speech technologies in under-represented languages.
>
---
#### [new 010] Layer-wise Positional Bias in Short-Context Language Modeling
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究短文本语言模型中的位置偏差问题，通过分析各层对输入位置的注意力分布，揭示了模型存在显著的近期偏差和轻微的初始偏差。**

- **链接: [https://arxiv.org/pdf/2601.04098v1](https://arxiv.org/pdf/2601.04098v1)**

> **作者:** Maryam Rahimi; Mahdi Nouri; Yadollah Yaghoobzadeh
>
> **摘要:** Language models often show a preference for using information from specific positions in the input regardless of semantic relevance. While positional bias has been studied in various contexts, from attention sinks to task performance degradation in long-context settings, prior work has not established how these biases evolve across individual layers and input positions, or how they vary independent of task complexity. We introduce an attribution-based framework to analyze positional effects in short-context language modeling. Using layer conductance with a sliding-window approach, we quantify how each layer distributes importance across input positions, yielding layer-wise positional importance profiles. We find that these profiles are architecture-specific, stable across inputs, and invariant to lexical scrambling. Characterizing these profiles, we find prominent recency bias that increases with depth and subtle primacy bias that diminishes through model depth. Beyond positional structure, we also show that early layers preferentially weight content words over function words across all positions, while later layers lose this word-type differentiation.
>
---
#### [new 011] When Models Decide and When They Bind: A Two-Stage Computation for Multiple-Choice Question-Answering
- **分类: cs.CL**

- **简介: 该论文研究多选题问答任务，解决模型在答题时如何选择答案内容并绑定正确符号的问题。通过分析模型内部表示，发现存在两个阶段：先确定内容正确性，再绑定输出符号。**

- **链接: [https://arxiv.org/pdf/2601.03914v1](https://arxiv.org/pdf/2601.03914v1)**

> **作者:** Hugh Mee Wong; Rick Nouwen; Albert Gatt
>
> **备注:** Under review
>
> **摘要:** Multiple-choice question answering (MCQA) is easy to evaluate but adds a meta-task: models must both solve the problem and output the symbol that *represents* the answer, conflating reasoning errors with symbol-binding failures. We study how language models implement MCQA internally using representational analyses (PCA, linear probes) as well as causal interventions. We find that option-boundary (newline) residual states often contain strong linearly decodable signals related to per-option correctness. Winner-identity probing reveals a two-stage progression: the winning *content position* becomes decodable immediately after the final option is processed, while the *output symbol* is represented closer to the answer emission position. Tests under symbol and content permutations support a two-stage mechanism in which models first select a winner in content space and then bind or route that winner to the appropriate symbol to emit.
>
---
#### [new 012] Breaking the Assistant Mold: Modeling Behavioral Variation in LLM Based Procedural Character Generation
- **分类: cs.CL**

- **简介: 该论文属于角色生成任务，旨在解决现有方法中道德和助手偏见导致角色单一问题，提出PersonaWeaver框架实现行为多样性。**

- **链接: [https://arxiv.org/pdf/2601.03396v1](https://arxiv.org/pdf/2601.03396v1)**

> **作者:** Maan Qraitem; Kate Saenko; Bryan A. Plummer
>
> **摘要:** Procedural content generation has enabled vast virtual worlds through levels, maps, and quests, but large-scale character generation remains underexplored. We identify two alignment-induced biases in existing methods: a positive moral bias, where characters uniformly adopt agreeable stances (e.g. always saying lying is bad), and a helpful assistant bias, where characters invariably answer questions directly (e.g. never refusing or deflecting). While such tendencies suit instruction-following systems, they suppress dramatic tension and yield predictable characters, stemming from maximum likelihood training and assistant fine-tuning. To address this, we introduce PersonaWeaver, a framework that disentangles world-building (roles, demographics) from behavioral-building (moral stances, interactional styles), yielding characters with more diverse reactions and moral stances, as well as second-order diversity in stylistic markers like length, tone, and punctuation. Code: https://github.com/mqraitem/Persona-Weaver
>
---
#### [new 013] VotIE: Information Extraction from Meeting Minutes
- **分类: cs.CL**

- **简介: 该论文提出VotIE任务，旨在从市政会议记录中提取结构化投票信息。针对非标准化文本的挑战，构建了首个葡萄牙语基准，并对比了不同模型的效果。**

- **链接: [https://arxiv.org/pdf/2601.03997v1](https://arxiv.org/pdf/2601.03997v1)**

> **作者:** José Pedro Evans; Luís Filipe Cunha; Purificação Silvano; Alípio Jorge; Nuno Guimarães; Sérgio Nunes; Ricardo Campos
>
> **摘要:** Municipal meeting minutes record key decisions in local democratic processes. Unlike parliamentary proceedings, which typically adhere to standardized formats, they encode voting outcomes in highly heterogeneous, free-form narrative text that varies widely across municipalities, posing significant challenges for automated extraction. In this paper, we introduce VotIE (Voting Information Extraction), a new information extraction task aimed at identifying structured voting events in narrative deliberative records, and establish the first benchmark for this task using Portuguese municipal minutes, building on the recently introduced CitiLink corpus. Our experiments yield two key findings. First, under standard in-domain evaluation, fine-tuned encoders, specifically XLM-R-CRF, achieve the strongest performance, reaching 93.2\% macro F1, outperforming generative approaches. Second, in a cross-municipality setting that evaluates transfer to unseen administrative contexts, these models suffer substantial performance degradation, whereas few-shot LLMs demonstrate greater robustness, with significantly smaller declines in performance. Despite this generalization advantage, the high computational cost of generative models currently constrains their practicality. As a result, lightweight fine-tuned encoders remain a more practical option for large-scale, real-world deployment. To support reproducible research in administrative NLP, we publicly release our benchmark, trained models, and evaluation framework.
>
---
#### [new 014] Grading Scale Impact on LLM-as-a-Judge: Human-LLM Alignment Is Highest on 0-5 Grading Scale
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文研究LLM作为评判者的任务，探讨不同评分尺度对人类与LLM一致性的影响，发现0-5尺度下两者对齐最佳。**

- **链接: [https://arxiv.org/pdf/2601.03444v1](https://arxiv.org/pdf/2601.03444v1)**

> **作者:** Weiyue Li; Minda Zhao; Weixuan Dong; Jiahui Cai; Yuze Wei; Michael Pocress; Yi Li; Wanyan Yuan; Xiaoyue Wang; Ruoyu Hou; Kaiyuan Lou; Wenqi Zeng; Yutong Yang; Yilun Du; Mengyu Wang
>
> **摘要:** Large language models (LLMs) are increasingly used as automated evaluators, yet prior works demonstrate that these LLM judges often lack consistency in scoring when the prompt is altered. However, the effect of the grading scale itself remains underexplored. We study the LLM-as-a-judge problem by comparing two kinds of raters: humans and LLMs. We collect ratings from both groups on three scales and across six benchmarks that include objective, open-ended subjective, and mixed tasks. Using intraclass correlation coefficients (ICC) to measure absolute agreement, we find that LLM judgments are not perfectly consistent across scales on subjective benchmarks, and that the choice of scale substantially shifts human-LLM agreement, even when within-group panel reliability is high. Aggregated over tasks, the grading scale of 0-5 yields the strongest human-LLM alignment. We further demonstrate that pooled reliability can mask benchmark heterogeneity and reveal systematic subgroup differences in alignment across gender groups, strengthening the importance of scale design and sub-level diagnostics as essential components of LLM-as-a-judge protocols.
>
---
#### [new 015] PartisanLens: A Multilingual Dataset of Hyperpartisan and Conspiratorial Immigration Narratives in European Media
- **分类: cs.CL**

- **简介: 该论文属于虚假信息检测任务，旨在解决识别极化与阴谋论叙事的问题。构建了多语言数据集PartisanLens，并评估大模型的分类与标注能力。**

- **链接: [https://arxiv.org/pdf/2601.03860v1](https://arxiv.org/pdf/2601.03860v1)**

> **作者:** Michele Joshua Maggini; Paloma Piot; Anxo Pérez; Erik Bran Marino; Lúa Santamaría Montesinos; Ana Lisboa; Marta Vázquez Abuín; Javier Parapar; Pablo Gamallo
>
> **摘要:** Detecting hyperpartisan narratives and Population Replacement Conspiracy Theories (PRCT) is essential to addressing the spread of misinformation. These complex narratives pose a significant threat, as hyperpartisanship drives political polarisation and institutional distrust, while PRCTs directly motivate real-world extremist violence, making their identification critical for social cohesion and public safety. However, existing resources are scarce, predominantly English-centric, and often analyse hyperpartisanship, stance, and rhetorical bias in isolation rather than as interrelated aspects of political discourse. To bridge this gap, we introduce \textsc{PartisanLens}, the first multilingual dataset of \num{1617} hyperpartisan news headlines in Spanish, Italian, and Portuguese, annotated in multiple political discourse aspects. We first evaluate the classification performance of widely used Large Language Models (LLMs) on this dataset, establishing robust baselines for the classification of hyperpartisan and PRCT narratives. In addition, we assess the viability of using LLMs as automatic annotators for this task, analysing their ability to approximate human annotation. Results highlight both their potential and current limitations. Next, moving beyond standard judgments, we explore whether LLMs can emulate human annotation patterns by conditioning them on socio-economic and ideological profiles that simulate annotator perspectives. At last, we provide our resources and evaluation, \textsc{PartisanLens} supports future research on detecting partisan and conspiratorial narratives in European contexts.
>
---
#### [new 016] O-Researcher: An Open Ended Deep Research Model via Multi-Agent Distillation and Agentic RL
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在缩小开源与闭源大模型间的性能差距。通过多智能体协作生成高质量数据，并结合强化学习提升模型能力。**

- **链接: [https://arxiv.org/pdf/2601.03743v1](https://arxiv.org/pdf/2601.03743v1)**

> **作者:** Yi Yao; He Zhu; Piaohong Wang; Jincheng Ren; Xinlong Yang; Qianben Chen; Xiaowan Li; Dingfeng Shi; Jiaxian Li; Qiexiang Wang; Sinuo Wang; Xinpeng Liu; Jiaqi Wu; Minghao Liu; Wangchunshu Zhou
>
> **备注:** 22 pages
>
> **摘要:** The performance gap between closed-source and open-source large language models (LLMs) is largely attributed to disparities in access to high-quality training data. To bridge this gap, we introduce a novel framework for the automated synthesis of sophisticated, research-grade instructional data. Our approach centers on a multi-agent workflow where collaborative AI agents simulate complex tool-integrated reasoning to generate diverse and high-fidelity data end-to-end. Leveraging this synthesized data, we develop a two-stage training strategy that integrates supervised fine-tuning with a novel reinforcement learning method, designed to maximize model alignment and capability. Extensive experiments demonstrate that our framework empowers open-source models across multiple scales, enabling them to achieve new state-of-the-art performance on the major deep research benchmark. This work provides a scalable and effective pathway for advancing open-source LLMs without relying on proprietary data or models.
>
---
#### [new 017] DeepResearch-Slice: Bridging the Retrieval-Utilization Gap via Explicit Text Slicing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息检索任务，解决模型检索到证据却无法有效利用的问题。提出DeepResearch-Slice框架，通过显式文本切片提升模型对噪声环境的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.03261v1](https://arxiv.org/pdf/2601.03261v1)**

> **作者:** Shuo Lu; Yinuo Xu; Jianjie Cheng; Lingxiao He; Meng Wang; Jian Liang
>
> **备注:** Ongoing work
>
> **摘要:** Deep Research agents predominantly optimize search policies to maximize retrieval probability. However, we identify a critical bottleneck: the retrieval-utilization gap, where models fail to use gold evidence even after it is retrieved, due to context blindness in noisy environments. To bridge this gap, we propose DeepResearch-Slice, a simple yet effective neuro-symbolic framework. Unlike implicit attention, our approach predicts precise span indices to perform a deterministic hard filter before reasoning. Extensive evaluations across six benchmarks show substantial robustness gains. Applying our method to frozen backbones yields a 73 percent relative improvement, from 19.1 percent to 33.0 percent, effectively mitigating noise without requiring parameter updates to the reasoning model. These results highlight the need for explicit grounding mechanisms in open-ended research.
>
---
#### [new 018] The Instruction Gap: LLMs get lost in Following Instruction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs在企业环境中指令遵循不一致的问题。通过评估13个模型，分析其指令遵循能力，提出改进方向。**

- **链接: [https://arxiv.org/pdf/2601.03269v1](https://arxiv.org/pdf/2601.03269v1)**

> **作者:** Vishesh Tripathi; Uday Allu; Biddwan Ahmed
>
> **摘要:** Large Language Models (LLMs) have shown remarkable capabilities in natural language understanding and generation, yet their deployment in enterprise environments reveals a critical limitation: inconsistent adherence to custom instructions. This study presents a comprehensive evaluation of 13 leading LLMs across instruction compliance, response accuracy, and performance metrics in realworld RAG (Retrieval-Augmented Generation) scenarios. Through systematic testing with samples and enterprise-grade evaluation protocols, we demonstrate that instruction following varies dramatically across models, with Claude-Sonnet-4 and GPT-5 achieving the highest results. Our findings reveal the "instruction gap" - a fundamental challenge where models excel at general tasks but struggle with precise instruction adherence required for enterprise deployment. This work provides practical insights for organizations deploying LLM-powered solutions and establishes benchmarks for instruction-following capabilities across major model families.
>
---
#### [new 019] HearSay Benchmark: Do Audio LLMs Leak What They Hear?
- **分类: cs.CL**

- **简介: 该论文属于隐私安全任务，研究音频大模型是否泄露用户隐私。通过构建基准数据集HearSay，发现模型易泄露性别等信息，且推理加剧风险，提出需加强隐私保护。**

- **链接: [https://arxiv.org/pdf/2601.03783v1](https://arxiv.org/pdf/2601.03783v1)**

> **作者:** Jin Wang; Liang Lin; Kaiwen Luo; Weiliu Wang; Yitian Chen; Moayad Aloqaily; Xuehai Tang; Zhenhong Zhou; Kun Wang; Li Sun; Qingsong Wen
>
> **摘要:** While Audio Large Language Models (ALLMs) have achieved remarkable progress in understanding and generation, their potential privacy implications remain largely unexplored. This paper takes the first step to investigate whether ALLMs inadvertently leak user privacy solely through acoustic voiceprints and introduces $\textit{HearSay}$, a comprehensive benchmark constructed from over 22,000 real-world audio clips. To ensure data quality, the benchmark is meticulously curated through a rigorous pipeline involving automated profiling and human verification, guaranteeing that all privacy labels are grounded in factual records. Extensive experiments on $\textit{HearSay}$ yield three critical findings: $\textbf{Significant Privacy Leakage}$: ALLMs inherently extract private attributes from voiceprints, reaching 92.89% accuracy on gender and effectively profiling social attributes. $\textbf{Insufficient Safety Mechanisms}$: Alarmingly, existing safeguards are severely inadequate; most models fail to refuse privacy-intruding requests, exhibiting near-zero refusal rates for physiological traits. $\textbf{Reasoning Amplifies Risk}$: Chain-of-Thought (CoT) reasoning exacerbates privacy risks in capable models by uncovering deeper acoustic correlations. These findings expose critical vulnerabilities in ALLMs, underscoring the urgent need for targeted privacy alignment. The codes and dataset are available at https://github.com/JinWang79/HearSay_Benchmark
>
---
#### [new 020] Tigrinya Number Verbalization: Rules, Algorithm, and Implementation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的数字转写任务，解决Tigrinya语数字表达的规则缺失问题。文中提出规则、算法并实现数字转文字系统，提升语言模型和语音合成的准确性。**

- **链接: [https://arxiv.org/pdf/2601.03403v1](https://arxiv.org/pdf/2601.03403v1)**

> **作者:** Fitsum Gaim; Issayas Tesfamariam
>
> **摘要:** We present a systematic formalization of Tigrinya cardinal and ordinal number verbalization, addressing a gap in computational resources for the language. This work documents the canonical rules governing the expression of numerical values in spoken Tigrinya, including the conjunction system, scale words, and special cases for dates, times, and currency. We provide a formal algorithm for number-to-word conversion and release an open-source implementation. Evaluation of frontier large language models (LLMs) reveals significant gaps in their ability to accurately verbalize Tigrinya numbers, underscoring the need for explicit rule documentation. This work serves language modeling, speech synthesis, and accessibility applications targeting Tigrinya-speaking communities.
>
---
#### [new 021] Visual Merit or Linguistic Crutch? A Close Look at DeepSeek-OCR
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于OCR任务，旨在评估DeepSeek-OCR的性能驱动因素。通过语义扰动实验，发现其性能依赖语言先验而非视觉能力，揭示了视觉压缩技术的局限性。**

- **链接: [https://arxiv.org/pdf/2601.03714v1](https://arxiv.org/pdf/2601.03714v1)**

> **作者:** Yunhao Liang; Ruixuan Ying; Bo Li; Hong Li; Kai Yan; Qingwen Li; Min Yang; Okamoto Satoshi; Zhe Cui; Shiwen Ni
>
> **摘要:** DeepSeek-OCR utilizes an optical 2D mapping approach to achieve high-ratio vision-text compression, claiming to decode text tokens exceeding ten times the input visual tokens. While this suggests a promising solution for the LLM long-context bottleneck, we investigate a critical question: "Visual merit or linguistic crutch - which drives DeepSeek-OCR's performance?" By employing sentence-level and word-level semantic corruption, we isolate the model's intrinsic OCR capabilities from its language priors. Results demonstrate that without linguistic support, DeepSeek-OCR's performance plummets from approximately 90% to 20%. Comparative benchmarking against 13 baseline models reveals that traditional pipeline OCR methods exhibit significantly higher robustness to such semantic perturbations than end-to-end methods. Furthermore, we find that lower visual token counts correlate with increased reliance on priors, exacerbating hallucination risks. Context stress testing also reveals a total model collapse around 10,000 text tokens, suggesting that current optical compression techniques may paradoxically aggravate the long-context bottleneck. This study empirically defines DeepSeek-OCR's capability boundaries and offers essential insights for future optimizations of the vision-text compression paradigm. We release all data, results and scripts used in this study at https://github.com/dududuck00/DeepSeekOCR.
>
---
#### [new 022] Towards Compositional Generalization of LLMs via Skill Taxonomy Guided Data Synthesis
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs在组合泛化上的不足。通过构建技能分类体系，生成更具挑战性的数据以提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.03676v1](https://arxiv.org/pdf/2601.03676v1)**

> **作者:** Yifan Wei; Li Du; Xiaoyan Yu; Yang Feng; Angsheng Li
>
> **备注:** The code and data for our methods and experiments are available at https://github.com/weiyifan1023/STEPS
>
> **摘要:** Large Language Models (LLMs) and agent-based systems often struggle with compositional generalization due to a data bottleneck in which complex skill combinations follow a long-tailed, power-law distribution, limiting both instruction-following performance and generalization in agent-centric tasks. To address this challenge, we propose STEPS, a Skill Taxonomy guided Entropy-based Post-training data Synthesis framework for generating compositionally challenging data. STEPS explicitly targets compositional generalization by uncovering latent relationships among skills and organizing them into an interpretable, hierarchical skill taxonomy using structural information theory. Building on this taxonomy, we formulate data synthesis as a constrained information maximization problem, selecting skill combinations that maximize marginal structural information within the hierarchy while preserving semantic coherence. Experiments on challenging instruction-following benchmarks show that STEPS outperforms existing data synthesis baselines, while also yielding improved compositional generalization in downstream agent-based evaluations.
>
---
#### [new 023] What Matters For Safety Alignment?
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文属于AI安全对齐研究，旨在评估影响大模型安全性的因素。通过实验分析模型特性与攻击方法，提出提升安全性的优化建议。**

- **链接: [https://arxiv.org/pdf/2601.03868v1](https://arxiv.org/pdf/2601.03868v1)**

> **作者:** Xing Li; Hui-Ling Zhen; Lihao Yin; Xianzhi Yu; Zhenhua Dong; Mingxuan Yuan
>
> **摘要:** This paper presents a comprehensive empirical study on the safety alignment capabilities. We evaluate what matters for safety alignment in LLMs and LRMs to provide essential insights for developing more secure and reliable AI systems. We systematically investigate and compare the influence of six critical intrinsic model characteristics and three external attack techniques. Our large-scale evaluation is conducted using 32 recent, popular LLMs and LRMs across thirteen distinct model families, spanning a parameter scale from 3B to 235B. The assessment leverages five established safety datasets and probes model vulnerabilities with 56 jailbreak techniques and four CoT attack strategies, resulting in 4.6M API calls. Our key empirical findings are fourfold. First, we identify the LRMs GPT-OSS-20B, Qwen3-Next-80B-A3B-Thinking, and GPT-OSS-120B as the top-three safest models, which substantiates the significant advantage of integrated reasoning and self-reflection mechanisms for robust safety alignment. Second, post-training and knowledge distillation may lead to a systematic degradation of safety alignment. We thus argue that safety must be treated as an explicit constraint or a core optimization objective during these stages, not merely subordinated to the pursuit of general capability. Third, we reveal a pronounced vulnerability: employing a CoT attack via a response prefix can elevate the attack success rate by 3.34x on average and from 0.6% to 96.3% for Seed-OSS-36B-Instruct. This critical finding underscores the safety risks inherent in text-completion interfaces and features that allow user-defined response prefixes in LLM services, highlighting an urgent need for architectural and deployment safeguards. Fourth, roleplay, prompt injection, and gradient-based search for adversarial prompts are the predominant methodologies for eliciting unaligned behaviors in modern models.
>
---
#### [new 024] OLA: Output Language Alignment in Code-Switched LLM Interactions
- **分类: cs.CL**

- **简介: 该论文属于多语言模型任务，解决代码切换场景下输出语言对齐问题。研究提出OLA基准，发现现有模型常错误响应语言，通过少量数据微调可改善。**

- **链接: [https://arxiv.org/pdf/2601.03589v1](https://arxiv.org/pdf/2601.03589v1)**

> **作者:** Juhyun Oh; Haneul Yoo; Faiz Ghifari Haznitrama; Alice Oh
>
> **摘要:** Code-switching, alternating between languages within a conversation, is natural for multilingual users, yet poses fundamental challenges for large language models (LLMs). When a user code-switches in their prompt to an LLM, they typically do not specify the expected language of the LLM response, and thus LLMs must infer the output language from contextual and pragmatic cues. We find that current LLMs systematically fail to align with this expectation, responding in undesired languages even when cues are clear to humans. We introduce OLA, a benchmark to evaluate LLMs' Output Language Alignment in code-switched interactions. OLA focuses on Korean--English code-switching and spans simple intra-sentential mixing to instruction-content mismatches. Even frontier models frequently misinterpret implicit language expectation, exhibiting a bias toward non-English responses. We further show this bias generalizes beyond Korean to Chinese and Indonesian pairs. Models also show instability through mid-response switching and language intrusions. Chain-of-Thought prompting fails to resolve these errors, indicating weak pragmatic reasoning about output language. However, Code-Switching Aware DPO with minimal data (about 1K examples) substantially reduces misalignment, suggesting these failures stem from insufficient alignment rather than fundamental limitations. Our results highlight the need to align multilingual LLMs with users' implicit expectations in real-world code-switched interactions.
>
---
#### [new 025] AirNav: A Large-Scale Real-World UAV Vision-and-Language Navigation Dataset with Natural and Diverse Instructions
- **分类: cs.CL**

- **简介: 该论文属于视觉语言导航任务，旨在解决现有UAV数据集依赖虚拟环境、指令不自然及规模有限的问题。提出AirNav真实场景数据集与AirVLN-R1模型，提升导航性能。**

- **链接: [https://arxiv.org/pdf/2601.03707v1](https://arxiv.org/pdf/2601.03707v1)**

> **作者:** Hengxing Cai; Yijie Rao; Ligang Huang; Zanyang Zhong; Jinhan Dong; Jingjun Tan; Wenhao Lu; Renxin Zhong
>
> **摘要:** Existing Unmanned Aerial Vehicle (UAV) Vision-Language Navigation (VLN) datasets face issues such as dependence on virtual environments, lack of naturalness in instructions, and limited scale. To address these challenges, we propose AirNav, a large-scale UAV VLN benchmark constructed from real urban aerial data, rather than synthetic environments, with natural and diverse instructions. Additionally, we introduce the AirVLN-R1, which combines Supervised Fine-Tuning and Reinforcement Fine-Tuning to enhance performance and generalization. The feasibility of the model is preliminarily evaluated through real-world tests. Our dataset and code are publicly available.
>
---
#### [new 026] MIND: From Passive Mimicry to Active Reasoning through Capability-Aware Multi-Perspective CoT Distillation
- **分类: cs.CL**

- **简介: 该论文属于知识蒸馏任务，解决小模型性能与泛化能力不足的问题。提出MIND框架，通过多视角教学和反馈校准提升学生模型的推理能力。**

- **链接: [https://arxiv.org/pdf/2601.03717v1](https://arxiv.org/pdf/2601.03717v1)**

> **作者:** Jin Cui; Jiaqi Guo; Jiepeng Zhou; Ruixuan Yang; Jiayi Lu; Jiajun Xu; Jiangcheng Song; Boran Zhao; Pengju Ren
>
> **备注:** 13 pages, 8 figures
>
> **摘要:** While Large Language Models (LLMs) have emerged with remarkable capabilities in complex tasks through Chain-of-Thought reasoning, practical resource constraints have sparked interest in transferring these abilities to smaller models. However, achieving both domain performance and cross-domain generalization remains challenging. Existing approaches typically restrict students to following a single golden rationale and treat different reasoning paths independently. Due to distinct inductive biases and intrinsic preferences, alongside the student's evolving capacity and reasoning preferences during training, a teacher's "optimal" rationale could act as out-of-distribution noise. This misalignment leads to a degeneration of the student's latent reasoning distribution, causing suboptimal performance. To bridge this gap, we propose MIND, a capability-adaptive framework that transitions distillation from passive mimicry to active cognitive construction. We synthesize diverse teacher perspectives through a novel "Teaching Assistant" network. By employing a Feedback-Driven Inertia Calibration mechanism, this network utilizes inertia-filtered training loss to align supervision with the student's current adaptability, effectively enhancing performance while mitigating catastrophic forgetting. Extensive experiments demonstrate that MIND achieves state-of-the-art performance on both in-distribution and out-of-distribution benchmarks, and our sophisticated latent space analysis further confirms the mechanism of reasoning ability internalization.
>
---
#### [new 027] SyncThink: A Training-Free Strategy to Align Inference Termination with Reasoning Saturation
- **分类: cs.CL**

- **简介: 该论文提出SyncThink，解决CoT推理中冗余和高成本的问题。通过监测模型内部信号实现推理终止，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.03649v1](https://arxiv.org/pdf/2601.03649v1)**

> **作者:** Gengyang Li; Wang Cai; Yifeng Gao; Yunfang Wu
>
> **备注:** 14 pages, 8 figures
>
> **摘要:** Chain-of-Thought (CoT) prompting improves reasoning but often produces long and redundant traces that substantially increase inference cost. We present SyncThink, a training-free and plug-and-play decoding method that reduces CoT overhead without modifying model weights. We find that answer tokens attend weakly to early reasoning and instead focus on the special token "/think", indicating an information bottleneck. Building on this observation, SyncThink monitors the model's own reasoning-transition signal and terminates reasoning. Experiments on GSM8K, MMLU, GPQA, and BBH across three DeepSeek-R1 distilled models show that SyncThink achieves 62.00 percent average Top-1 accuracy using 656 generated tokens and 28.68 s latency, compared to 61.22 percent, 2141 tokens, and 92.01 s for full CoT decoding. On long-horizon tasks such as GPQA, SyncThink can further yield up to +8.1 absolute accuracy by preventing over-thinking.
>
---
#### [new 028] EvolMem: A Cognitive-Driven Benchmark for Multi-Session Dialogue Memory
- **分类: cs.CL**

- **简介: 该论文提出EvolMem，用于评估大语言模型在多轮对话中的记忆能力。解决现有基准缺乏系统性评估的问题，通过混合数据生成框架构建基准，揭示模型在不同记忆维度的表现差异。**

- **链接: [https://arxiv.org/pdf/2601.03543v1](https://arxiv.org/pdf/2601.03543v1)**

> **作者:** Ye Shen; Dun Pei; Yiqiu Guo; Junying Wang; Yijin Guo; Zicheng Zhang; Qi Jia; Jun Zhou; Guangtao Zhai
>
> **备注:** 14 pages, 7 figures, 8 tables
>
> **摘要:** Despite recent advances in understanding and leveraging long-range conversational memory, existing benchmarks still lack systematic evaluation of large language models(LLMs) across diverse memory dimensions, particularly in multi-session settings. In this work, we propose EvolMem, a new benchmark for assessing multi-session memory capabilities of LLMs and agent systems. EvolMem is grounded in cognitive psychology and encompasses both declarative and non-declarative memory, further decomposed into multiple fine-grained abilities. To construct the benchmark, we introduce a hybrid data synthesis framework that consists of topic-initiated generation and narrative-inspired transformations. This framework enables scalable generation of multi-session conversations with controllable complexity, accompanied by sample-specific evaluation guidelines. Extensive evaluation reveals that no LLM consistently outperforms others across all memory dimensions. Moreover, agent memory mechanisms do not necessarily enhance LLMs' capabilities and often exhibit notable efficiency limitations. Data and code will be released at https://github.com/shenye7436/EvolMem.
>
---
#### [new 029] OpenAI GPT-5 System Card
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍GPT-5系统，解决多任务处理与安全问题，通过智能路由和模型优化提升性能与安全性。**

- **链接: [https://arxiv.org/pdf/2601.03267v1](https://arxiv.org/pdf/2601.03267v1)**

> **作者:** Aaditya Singh; Adam Fry; Adam Perelman; Adam Tart; Adi Ganesh; Ahmed El-Kishky; Aidan McLaughlin; Aiden Low; AJ Ostrow; Akhila Ananthram; Akshay Nathan; Alan Luo; Alec Helyar; Aleksander Madry; Aleksandr Efremov; Aleksandra Spyra; Alex Baker-Whitcomb; Alex Beutel; Alex Karpenko; Alex Makelov; Alex Neitz; Alex Wei; Alexandra Barr; Alexandre Kirchmeyer; Alexey Ivanov; Alexi Christakis; Alistair Gillespie; Allison Tam; Ally Bennett; Alvin Wan; Alyssa Huang; Amy McDonald Sandjideh; Amy Yang; Ananya Kumar; Andre Saraiva; Andrea Vallone; Andrei Gheorghe; Andres Garcia Garcia; Andrew Braunstein; Andrew Liu; Andrew Schmidt; Andrey Mereskin; Andrey Mishchenko; Andy Applebaum; Andy Rogerson; Ann Rajan; Annie Wei; Anoop Kotha; Anubha Srivastava; Anushree Agrawal; Arun Vijayvergiya; Ashley Tyra; Ashvin Nair; Avi Nayak; Ben Eggers; Bessie Ji; Beth Hoover; Bill Chen; Blair Chen; Boaz Barak; Borys Minaiev; Botao Hao; Bowen Baker; Brad Lightcap; Brandon McKinzie; Brandon Wang; Brendan Quinn; Brian Fioca; Brian Hsu; Brian Yang; Brian Yu; Brian Zhang; Brittany Brenner; Callie Riggins Zetino; Cameron Raymond; Camillo Lugaresi; Carolina Paz; Cary Hudson; Cedric Whitney; Chak Li; Charles Chen; Charlotte Cole; Chelsea Voss; Chen Ding; Chen Shen; Chengdu Huang; Chris Colby; Chris Hallacy; Chris Koch; Chris Lu; Christina Kaplan; Christina Kim; CJ Minott-Henriques; Cliff Frey; Cody Yu; Coley Czarnecki; Colin Reid; Colin Wei; Cory Decareaux; Cristina Scheau; Cyril Zhang; Cyrus Forbes; Da Tang; Dakota Goldberg; Dan Roberts; Dana Palmie; Daniel Kappler; Daniel Levine; Daniel Wright; Dave Leo; David Lin; David Robinson; Declan Grabb; Derek Chen; Derek Lim; Derek Salama; Dibya Bhattacharjee; Dimitris Tsipras; Dinghua Li; Dingli Yu; DJ Strouse; Drew Williams; Dylan Hunn; Ed Bayes; Edwin Arbus; Ekin Akyurek; Elaine Ya Le; Elana Widmann; Eli Yani; Elizabeth Proehl; Enis Sert; Enoch Cheung; Eri Schwartz; Eric Han; Eric Jiang; Eric Mitchell; Eric Sigler; Eric Wallace; Erik Ritter; Erin Kavanaugh; Evan Mays; Evgenii Nikishin; Fangyuan Li; Felipe Petroski Such; Filipe de Avila Belbute Peres; Filippo Raso; Florent Bekerman; Foivos Tsimpourlas; Fotis Chantzis; Francis Song; Francis Zhang; Gaby Raila; Garrett McGrath; Gary Briggs; Gary Yang; Giambattista Parascandolo; Gildas Chabot; Grace Kim; Grace Zhao; Gregory Valiant; Guillaume Leclerc; Hadi Salman; Hanson Wang; Hao Sheng; Haoming Jiang; Haoyu Wang; Haozhun Jin; Harshit Sikchi; Heather Schmidt; Henry Aspegren; Honglin Chen; Huida Qiu; Hunter Lightman; Ian Covert; Ian Kivlichan; Ian Silber; Ian Sohl; Ibrahim Hammoud; Ignasi Clavera; Ikai Lan; Ilge Akkaya; Ilya Kostrikov; Irina Kofman; Isak Etinger; Ishaan Singal; Jackie Hehir; Jacob Huh; Jacqueline Pan; Jake Wilczynski; Jakub Pachocki; James Lee; James Quinn; Jamie Kiros; Janvi Kalra; Jasmyn Samaroo; Jason Wang; Jason Wolfe; Jay Chen; Jay Wang; Jean Harb; Jeffrey Han; Jeffrey Wang; Jennifer Zhao; Jeremy Chen; Jerene Yang; Jerry Tworek; Jesse Chand; Jessica Landon; Jessica Liang; Ji Lin; Jiancheng Liu; Jianfeng Wang; Jie Tang; Jihan Yin; Joanne Jang; Joel Morris; Joey Flynn; Johannes Ferstad; Johannes Heidecke; John Fishbein; John Hallman; Jonah Grant; Jonathan Chien; Jonathan Gordon; Jongsoo Park; Jordan Liss; Jos Kraaijeveld; Joseph Guay; Joseph Mo; Josh Lawson; Josh McGrath; Joshua Vendrow; Joy Jiao; Julian Lee; Julie Steele; Julie Wang; Junhua Mao; Kai Chen; Kai Hayashi; Kai Xiao; Kamyar Salahi; Kan Wu; Karan Sekhri; Karan Sharma; Karan Singhal; Karen Li; Kenny Nguyen; Keren Gu-Lemberg; Kevin King; Kevin Liu; Kevin Stone; Kevin Yu; Kristen Ying; Kristian Georgiev; Kristie Lim; Kushal Tirumala; Kyle Miller; Lama Ahmad; Larry Lv; Laura Clare; Laurance Fauconnet; Lauren Itow; Lauren Yang; Laurentia Romaniuk; Leah Anise; Lee Byron; Leher Pathak; Leon Maksin; Leyan Lo; Leyton Ho; Li Jing; Liang Wu; Liang Xiong; Lien Mamitsuka; Lin Yang; Lindsay McCallum; Lindsey Held; Liz Bourgeois; Logan Engstrom; Lorenz Kuhn; Louis Feuvrier; Lu Zhang; Lucas Switzer; Lukas Kondraciuk; Lukasz Kaiser; Manas Joglekar; Mandeep Singh; Mandip Shah; Manuka Stratta; Marcus Williams; Mark Chen; Mark Sun; Marselus Cayton; Martin Li; Marvin Zhang; Marwan Aljubeh; Matt Nichols; Matthew Haines; Max Schwarzer; Mayank Gupta; Meghan Shah; Melody Huang; Meng Dong; Mengqing Wang; Mia Glaese; Micah Carroll; Michael Lampe; Michael Malek; Michael Sharman; Michael Zhang; Michele Wang; Michelle Pokrass; Mihai Florian; Mikhail Pavlov; Miles Wang; Ming Chen; Mingxuan Wang; Minnia Feng; Mo Bavarian; Molly Lin; Moose Abdool; Mostafa Rohaninejad; Nacho Soto; Natalie Staudacher; Natan LaFontaine; Nathan Marwell; Nelson Liu; Nick Preston; Nick Turley; Nicklas Ansman; Nicole Blades; Nikil Pancha; Nikita Mikhaylin; Niko Felix; Nikunj Handa; Nishant Rai; Nitish Keskar; Noam Brown; Ofir Nachum; Oleg Boiko; Oleg Murk; Olivia Watkins; Oona Gleeson; Pamela Mishkin; Patryk Lesiewicz; Paul Baltescu; Pavel Belov; Peter Zhokhov; Philip Pronin; Phillip Guo; Phoebe Thacker; Qi Liu; Qiming Yuan; Qinghua Liu; Rachel Dias; Rachel Puckett; Rahul Arora; Ravi Teja Mullapudi; Raz Gaon; Reah Miyara; Rennie Song; Rishabh Aggarwal; RJ Marsan; Robel Yemiru; Robert Xiong; Rohan Kshirsagar; Rohan Nuttall; Roman Tsiupa; Ronen Eldan; Rose Wang; Roshan James; Roy Ziv; Rui Shu; Ruslan Nigmatullin; Saachi Jain; Saam Talaie; Sam Altman; Sam Arnesen; Sam Toizer; Sam Toyer; Samuel Miserendino; Sandhini Agarwal; Sarah Yoo; Savannah Heon; Scott Ethersmith; Sean Grove; Sean Taylor; Sebastien Bubeck; Sever Banesiu; Shaokyi Amdo; Shengjia Zhao; Sherwin Wu; Shibani Santurkar; Shiyu Zhao; Shraman Ray Chaudhuri; Shreyas Krishnaswamy; Shuaiqi; Xia; Shuyang Cheng; Shyamal Anadkat; Simón Posada Fishman; Simon Tobin; Siyuan Fu; Somay Jain; Song Mei; Sonya Egoian; Spencer Kim; Spug Golden; SQ Mah; Steph Lin; Stephen Imm; Steve Sharpe; Steve Yadlowsky; Sulman Choudhry; Sungwon Eum; Suvansh Sanjeev; Tabarak Khan; Tal Stramer; Tao Wang; Tao Xin; Tarun Gogineni; Taya Christianson; Ted Sanders; Tejal Patwardhan; Thomas Degry; Thomas Shadwell; Tianfu Fu; Tianshi Gao; Timur Garipov; Tina Sriskandarajah; Toki Sherbakov; Tomer Kaftan; Tomo Hiratsuka; Tongzhou Wang; Tony Song; Tony Zhao; Troy Peterson; Val Kharitonov; Victoria Chernova; Vineet Kosaraju; Vishal Kuo; Vitchyr Pong; Vivek Verma; Vlad Petrov; Wanning Jiang; Weixing Zhang; Wenda Zhou; Wenlei Xie; Wenting Zhan; Wes McCabe; Will DePue; Will Ellsworth; Wulfie Bain; Wyatt Thompson; Xiangning Chen; Xiangyu Qi; Xin Xiang; Xinwei Shi; Yann Dubois; Yaodong Yu; Yara Khakbaz; Yifan Wu; Yilei Qian; Yin Tat Lee; Yinbo Chen; Yizhen Zhang; Yizhong Xiong; Yonglong Tian; Young Cha; Yu Bai; Yu Yang; Yuan Yuan; Yuanzhi Li; Yufeng Zhang; Yuguang Yang; Yujia Jin; Yun Jiang; Yunyun Wang; Yushi Wang; Yutian Liu; Zach Stubenvoll; Zehao Dou; Zheng Wu; Zhigang Wang
>
> **摘要:** This is the system card published alongside the OpenAI GPT-5 launch, August 2025. GPT-5 is a unified system with a smart and fast model that answers most questions, a deeper reasoning model for harder problems, and a real-time router that quickly decides which model to use based on conversation type, complexity, tool needs, and explicit intent (for example, if you say 'think hard about this' in the prompt). The router is continuously trained on real signals, including when users switch models, preference rates for responses, and measured correctness, improving over time. Once usage limits are reached, a mini version of each model handles remaining queries. This system card focuses primarily on gpt-5-thinking and gpt-5-main, while evaluations for other models are available in the appendix. The GPT-5 system not only outperforms previous models on benchmarks and answers questions more quickly, but -- more importantly -- is more useful for real-world queries. We've made significant advances in reducing hallucinations, improving instruction following, and minimizing sycophancy, and have leveled up GPT-5's performance in three of ChatGPT's most common uses: writing, coding, and health. All of the GPT-5 models additionally feature safe-completions, our latest approach to safety training to prevent disallowed content. Similarly to ChatGPT agent, we have decided to treat gpt-5-thinking as High capability in the Biological and Chemical domain under our Preparedness Framework, activating the associated safeguards. While we do not have definitive evidence that this model could meaningfully help a novice to create severe biological harm -- our defined threshold for High capability -- we have chosen to take a precautionary approach.
>
---
#### [new 030] Value-Action Alignment in Large Language Models under Privacy-Prosocial Conflict
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于人工智能伦理任务，旨在解决隐私与利他冲突下大模型决策的对齐问题。通过设计评估协议和MGSEM分析，研究模型价值与行为的一致性。**

- **链接: [https://arxiv.org/pdf/2601.03546v1](https://arxiv.org/pdf/2601.03546v1)**

> **作者:** Guanyu Chen; Chenxiao Yu; Xiyang Hu
>
> **摘要:** Large language models (LLMs) are increasingly used to simulate decision-making tasks involving personal data sharing, where privacy concerns and prosocial motivations can push choices in opposite directions. Existing evaluations often measure privacy-related attitudes or sharing intentions in isolation, which makes it difficult to determine whether a model's expressed values jointly predict its downstream data-sharing actions as in real human behaviors. We introduce a context-based assessment protocol that sequentially administers standardized questionnaires for privacy attitudes, prosocialness, and acceptance of data sharing within a bounded, history-carrying session. To evaluate value-action alignments under competing attitudes, we use multi-group structural equation modeling (MGSEM) to identify relations from privacy concerns and prosocialness to data sharing. We propose Value-Action Alignment Rate (VAAR), a human-referenced directional agreement metric that aggregates path-level evidence for expected signs. Across multiple LLMs, we observe stable but model-specific Privacy-PSA-AoDS profiles, and substantial heterogeneity in value-action alignment.
>
---
#### [new 031] Persona-aware and Explainable Bikeability Assessment: A Vision-Language Model Approach
- **分类: cs.CL; cs.CV; cs.HC; cs.LG**

- **简介: 该论文属于交通评估任务，旨在解决 bikeability 评估中用户感知复杂性与个性化问题。提出一种基于视觉语言模型的框架，结合用户画像、多粒度训练和数据增强，提升评估准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2601.03534v1](https://arxiv.org/pdf/2601.03534v1)**

> **作者:** Yilong Dai; Ziyi Wang; Chenguang Wang; Kexin Zhou; Yiheng Qian; Susu Xu; Xiang Yan
>
> **摘要:** Bikeability assessment is essential for advancing sustainable urban transportation and creating cyclist-friendly cities, and it requires incorporating users' perceptions of safety and comfort. Yet existing perception-based bikeability assessment approaches face key limitations in capturing the complexity of road environments and adequately accounting for heterogeneity in subjective user perceptions. This paper proposes a persona-aware Vision-Language Model framework for bikeability assessment with three novel contributions: (i) theory-grounded persona conditioning based on established cyclist typology that generates persona-specific explanations via chain-of-thought reasoning; (ii) multi-granularity supervised fine-tuning that combines scarce expert-annotated reasoning with abundant user ratings for joint prediction and explainable assessment; and (iii) AI-enabled data augmentation that creates controlled paired data to isolate infrastructure variable impacts. To test and validate this framework, we developed a panoramic image-based crowdsourcing system and collected 12,400 persona-conditioned assessments from 427 cyclists. Experiment results show that the proposed framework offers competitive bikeability rating prediction while uniquely enabling explainable factor attribution.
>
---
#### [new 032] Training-Free Adaptation of New-Generation LLMs using Legacy Clinical Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的领域适应任务，旨在解决新模型生成需 costly 重训练的问题。提出 CAPT 方法，无需训练即可融合临床模型与通用模型，提升临床任务性能。**

- **链接: [https://arxiv.org/pdf/2601.03423v1](https://arxiv.org/pdf/2601.03423v1)**

> **作者:** Sasha Ronaghi; Chloe Stanwyck; Asad Aali; Amir Ronaghi; Miguel Fuentes; Tina Hernandez-Boussard; Emily Alsentzer
>
> **备注:** 29 pages, 3 figures
>
> **摘要:** Adapting language models to the clinical domain through continued pretraining and fine-tuning requires costly retraining for each new model generation. We propose Cross-Architecture Proxy Tuning (CAPT), a model-ensembling approach that enables training-free adaptation of state-of-the-art general-domain models using existing clinical models. CAPT supports models with disjoint vocabularies, leveraging contrastive decoding to selectively inject clinically relevant signals while preserving the general-domain model's reasoning and fluency. On six clinical classification and text-generation tasks, CAPT with a new-generation general-domain model and an older-generation clinical model consistently outperforms both models individually and state-of-the-art ensembling approaches (average +17.6% over UniTE, +41.4% over proxy tuning across tasks). Through token-level analysis and physician case studies, we demonstrate that CAPT amplifies clinically actionable language, reduces context errors, and increases clinical specificity.
>
---
#### [new 033] Benchmarking and Adapting On-Device Large Language Models for Clinical Decision Support
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床决策支持任务，旨在解决隐私和资源限制问题。对比分析了多个LLM模型，并通过微调提升小模型性能，验证其在临床场景中的可行性。**

- **链接: [https://arxiv.org/pdf/2601.03266v1](https://arxiv.org/pdf/2601.03266v1)**

> **作者:** Alif Munim; Jun Ma; Omar Ibrahim; Alhusain Abdalla; Shuolin Yin; Leo Chen; Bo Wang
>
> **摘要:** Large language models (LLMs) have rapidly advanced in clinical decision-making, yet the deployment of proprietary systems is hindered by privacy concerns and reliance on cloud-based infrastructure. Open-source alternatives allow local inference but often require large model sizes that limit their use in resource-constrained clinical settings. Here, we benchmark two on-device LLMs, gpt-oss-20b and gpt-oss-120b, across three representative clinical tasks: general disease diagnosis, specialty-specific (ophthalmology) diagnosis and management, and simulation of human expert grading and evaluation. We compare their performance with state-of-the-art proprietary models (GPT-5 and o4-mini) and a leading open-source model (DeepSeek-R1), and we further evaluate the adaptability of on-device systems by fine-tuning gpt-oss-20b on general diagnostic data. Across tasks, gpt-oss models achieve performance comparable to or exceeding DeepSeek-R1 and o4-mini despite being substantially smaller. In addition, fine-tuning remarkably improves the diagnostic accuracy of gpt-oss-20b, enabling it to approach the performance of GPT-5. These findings highlight the potential of on-device LLMs to deliver accurate, adaptable, and privacy-preserving clinical decision support, offering a practical pathway for broader integration of LLMs into routine clinical practice.
>
---
#### [new 034] Do LLMs Really Memorize Personally Identifiable Information? Revisiting PII Leakage with a Cue-Controlled Memorization Framework
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于隐私与模型安全任务，旨在解决LLM中PII泄露是否源于真实记忆的问题。通过提出CRM框架，验证PII泄露主要由提示驱动而非真实记忆。**

- **链接: [https://arxiv.org/pdf/2601.03791v1](https://arxiv.org/pdf/2601.03791v1)**

> **作者:** Xiaoyu Luo; Yiyi Chen; Qiongxiu Li; Johannes Bjerva
>
> **备注:** 20 pages, 13 figures
>
> **摘要:** Large Language Models (LLMs) have been reported to "leak" Personally Identifiable Information (PII), with successful PII reconstruction often interpreted as evidence of memorization. We propose a principled revision of memorization evaluation for LLMs, arguing that PII leakage should be evaluated under low lexical cue conditions, where target PII cannot be reconstructed through prompt-induced generalization or pattern completion. We formalize Cue-Resistant Memorization (CRM) as a cue-controlled evaluation framework and a necessary condition for valid memorization evaluation, explicitly conditioning on prompt-target overlap cues. Using CRM, we conduct a large-scale multilingual re-evaluation of PII leakage across 32 languages and multiple memorization paradigms. Revisiting reconstruction-based settings, including verbatim prefix-suffix completion and associative reconstruction, we find that their apparent effectiveness is driven primarily by direct surface-form cues rather than by true memorization. When such cues are controlled for, reconstruction success diminishes substantially. We further examine cue-free generation and membership inference, both of which exhibit extremely low true positive rates. Overall, our results suggest that previously reported PII leakage is better explained by cue-driven behavior than by genuine memorization, highlighting the importance of cue-controlled evaluation for reliably quantifying privacy-relevant memorization in LLMs.
>
---
#### [new 035] What Does Loss Optimization Actually Teach, If Anything? Knowledge Dynamics in Continual Pre-training of LLMs
- **分类: cs.CL**

- **简介: 该论文研究持续预训练（CPT）中损失优化与知识学习的关系，揭示损失下降与事实学习的不一致性。任务属于语言模型训练与评估，解决CPT有效性问题，通过实验分析知识动态和电路变化。**

- **链接: [https://arxiv.org/pdf/2601.03858v1](https://arxiv.org/pdf/2601.03858v1)**

> **作者:** Seyed Mahed Mousavi; Simone Alghisi; Giuseppe Riccardi
>
> **摘要:** Continual Pre-Training (CPT) is widely used for acquiring and updating factual knowledge in LLMs. This practice treats loss as a proxy for knowledge learning, while offering no grounding into how it changes during training. We study CPT as a knowledge learning process rather than a solely optimization problem. We construct a controlled, distribution-matched benchmark of factual documents and interleave diagnostic probes directly into the CPT loop, enabling epoch-level measurement of knowledge acquisition dynamics and changes in Out-Of-Domain (OOD) general skills (e.g., math). We further analyze how CPT reshapes knowledge circuits during training. Across three instruction-tuned LLMs and multiple CPT strategies, optimization and learning systematically diverge as loss decreases monotonically while factual learning is unstable and non-monotonic. Acquired facts are rarely consolidated, learning is strongly conditioned on prior exposure, and OOD performance degrades from early epochs. Circuit analysis reveals rapid reconfiguration of knowledge pathways across epochs, providing an explanation for narrow acquisition windows and systematic forgetting. These results show that loss optimization is misaligned with learning progress in CPT and motivate evaluation of stopping criteria based on task-level learning dynamics.
>
---
#### [new 036] CALM: Culturally Self-Aware Language Models
- **分类: cs.CL; cs.CY; cs.LG**

- **简介: 该论文提出CALM框架，解决语言模型文化敏感性不足的问题。通过结构化文化聚类与自适应融合，提升模型对文化动态变化的适应能力。**

- **链接: [https://arxiv.org/pdf/2601.03483v1](https://arxiv.org/pdf/2601.03483v1)**

> **作者:** Lingzhi Shen; Xiaohao Cai; Yunfei Long; Imran Razzak; Guanming Chen; Shoaib Jameel
>
> **摘要:** Cultural awareness in language models is the capacity to understand and adapt to diverse cultural contexts. However, most existing approaches treat culture as static background knowledge, overlooking its dynamic and evolving nature. This limitation reduces their reliability in downstream tasks that demand genuine cultural sensitivity. In this work, we introduce CALM, a novel framework designed to endow language models with cultural self-awareness. CALM disentangles task semantics from explicit cultural concepts and latent cultural signals, shaping them into structured cultural clusters through contrastive learning. These clusters are then aligned via cross-attention to establish fine-grained interactions among related cultural features and are adaptively integrated through a Mixture-of-Experts mechanism along culture-specific dimensions. The resulting unified representation is fused with the model's original knowledge to construct a culturally grounded internal identity state, which is further enhanced through self-prompted reflective learning, enabling continual adaptation and self-correction. Extensive experiments conducted on multiple cross-cultural benchmark datasets demonstrate that CALM consistently outperforms state-of-the-art methods.
>
---
#### [new 037] Evaluating LLMs for Police Decision-Making: A Framework Based on Police Action Scenarios
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI在警务中的应用任务，旨在解决LLM在警察决策中可能引发的法律问题。提出PAS框架，构建警务问答数据集，评估LLM性能。**

- **链接: [https://arxiv.org/pdf/2601.03553v1](https://arxiv.org/pdf/2601.03553v1)**

> **作者:** Sangyub Lee; Heedou Kim; Hyeoncheol Kim
>
> **备注:** This work was accepted at AAAI 2026 social good track
>
> **摘要:** The use of Large Language Models (LLMs) in police operations is growing, yet an evaluation framework tailored to police operations remains absent. While LLM's responses may not always be legally incorrect, their unverified use still can lead to severe issues such as unlawful arrests and improper evidence collection. To address this, we propose PAS (Police Action Scenarios), a systematic framework covering the entire evaluation process. Applying this framework, we constructed a novel QA dataset from over 8,000 official documents and established key metrics validated through statistical analysis with police expert judgements. Experimental results show that commercial LLMs struggle with our new police-related tasks, particularly in providing fact-based recommendations. This study highlights the necessity of an expandable evaluation framework to ensure reliable AI-driven police operations. We release our data and prompt template.
>
---
#### [new 038] Bridging the Discrete-Continuous Gap: Unified Multimodal Generation via Coupled Manifold Discrete Absorbing Diffusion
- **分类: cs.CL**

- **简介: 该论文属于多模态生成任务，旨在解决文本与图像生成之间的离散-连续鸿沟问题。提出CoM-DAD框架，通过联合流形扩散实现统一生成。**

- **链接: [https://arxiv.org/pdf/2601.04056v1](https://arxiv.org/pdf/2601.04056v1)**

> **作者:** Yuanfeng Xu; Yuhao Chen; Liang Lin; Guangrun Wang
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** The bifurcation of generative modeling into autoregressive approaches for discrete data (text) and diffusion approaches for continuous data (images) hinders the development of truly unified multimodal systems. While Masked Language Models (MLMs) offer efficient bidirectional context, they traditionally lack the generative fidelity of autoregressive models and the semantic continuity of diffusion models. Furthermore, extending masked generation to multimodal settings introduces severe alignment challenges and training instability. In this work, we propose \textbf{CoM-DAD} (\textbf{Co}upled \textbf{M}anifold \textbf{D}iscrete \textbf{A}bsorbing \textbf{D}iffusion), a novel probabilistic framework that reformulates multimodal generation as a hierarchical dual-process. CoM-DAD decouples high-level semantic planning from low-level token synthesis. First, we model the semantic manifold via a continuous latent diffusion process; second, we treat token generation as a discrete absorbing diffusion process, regulated by a \textbf{Variable-Rate Noise Schedule}, conditioned on these evolving semantic priors. Crucially, we introduce a \textbf{Stochastic Mixed-Modal Transport} strategy that aligns disparate modalities without requiring heavy contrastive dual-encoders. Our method demonstrates superior stability over standard masked modeling, establishing a new paradigm for scalable, unified text-image generation.
>
---
#### [new 039] NeoAMT: Neologism-Aware Agentic Machine Translation with Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于神经网络机器翻译任务，旨在解决包含新词的句子翻译问题。作者构建了新数据集并提出基于强化学习的框架NeoAMT，提升翻译质量。**

- **链接: [https://arxiv.org/pdf/2601.03790v1](https://arxiv.org/pdf/2601.03790v1)**

> **作者:** Zhongtao Miao; Kaiyan Zhao; Masaaki Nagata; Yoshimasa Tsuruoka
>
> **摘要:** Neologism-aware machine translation aims to translate source sentences containing neologisms into target languages. This field remains underexplored compared with general machine translation (MT). In this paper, we propose an agentic framework, NeoAMT, for neologism-aware machine translation using a Wiktionary search tool. Specifically, we first create a new dataset for neologism-aware machine translation and develop a search tool based on Wiktionary. The new dataset covers 16 languages and 75 translation directions and is derived from approximately 10 million records of an English Wiktionary dump. The retrieval corpus of the search tool is also constructed from around 3 million cleaned records of the Wiktionary dump. We then use it for training the translation agent with reinforcement learning (RL) and evaluating the accuracy of neologism-aware machine translation. Based on this, we also propose an RL training framework that contains a novel reward design and an adaptive rollout generation approach by leveraging "translation difficulty" to further improve the translation quality of translation agents using our search tool.
>
---
#### [new 040] LLMberjack: Guided Trimming of Debate Trees for Multi-Party Conversation Creation
- **分类: cs.CL; cs.HC**

- **简介: 该论文提出LLMberjack，用于从辩论树生成多角色对话。解决多方对话生成问题，通过可视化和LLM辅助提升对话质量与效率。**

- **链接: [https://arxiv.org/pdf/2601.04135v1](https://arxiv.org/pdf/2601.04135v1)**

> **作者:** Leonardo Bottona; Nicolò Penzo; Bruno Lepri; Marco Guerini; Sara Tonelli
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** We present LLMberjack, a platform for creating multi-party conversations starting from existing debates, originally structured as reply trees. The system offers an interactive interface that visualizes discussion trees and enables users to construct coherent linearized dialogue sequences while preserving participant identity and discourse relations. It integrates optional large language model (LLM) assistance to support automatic editing of the messages and speakers' descriptions. We demonstrate the platform's utility by showing how tree visualization facilitates the creation of coherent, meaningful conversation threads and how LLM support enhances output quality while reducing human effort. The tool is open-source and designed to promote transparent and reproducible workflows to create multi-party conversations, addressing a lack of resources of this type.
>
---
#### [new 041] From Chains to Graphs: Self-Structured Reasoning for General-Domain LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的问答任务，旨在解决LLMs推理过程线性且不一致的问题。提出SGR框架，让模型用图结构进行推理，提升一致性。**

- **链接: [https://arxiv.org/pdf/2601.03597v1](https://arxiv.org/pdf/2601.03597v1)**

> **作者:** Yingjian Chen; Haoran Liu; Yinhong Liu; Sherry T. Tong; Aosong Feng; Jinghui Lu; Juntao Zhang; Yusuke Iwasawa; Yutaka Matsuo; Irene Li
>
> **摘要:** Large Language Models (LLMs) show strong reasoning ability in open-domain question answering, yet their reasoning processes are typically linear and often logically inconsistent. In contrast, real-world reasoning requires integrating multiple premises and solving subproblems in parallel. Existing methods, such as Chain-of-Thought (CoT), express reasoning in a linear textual form, which may appear coherent but frequently leads to inconsistent conclusions. Recent approaches rely on externally provided graphs and do not explore how LLMs can construct and use their own graph-structured reasoning, particularly in open-domain QA. To fill this gap, we novelly explore graph-structured reasoning of LLMs in general-domain question answering. We propose Self-Graph Reasoning (SGR), a framework that enables LLMs to explicitly represent their reasoning process as a structured graph before producing the final answer. We further construct a graph-structured reasoning dataset that merges multiple candidate reasoning graphs into refined graph structures for model training. Experiments on five QA benchmarks across both general and specialized domains show that SGR consistently improves reasoning consistency and yields a 17.74% gain over the base model. The LLaMA-3.3-70B model fine-tuned with SGR performs comparably to GPT-4o and surpasses Claude-3.5-Haiku, demonstrating the effectiveness of graph-structured reasoning.
>
---
#### [new 042] Metaphors are a Source of Cross-Domain Misalignment of Large Reasoning Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究 metaphors 引起大模型推理内容的跨领域偏差问题。通过实验发现 metaphors 与模型偏差存在强因果关系，并设计了检测器预测偏差内容。**

- **链接: [https://arxiv.org/pdf/2601.03388v1](https://arxiv.org/pdf/2601.03388v1)**

> **作者:** Zhibo Hu; Chen Wang; Yanfeng Shu; Hye-young Paik; Liming Zhu
>
> **备注:** 17 pages, 7 figures
>
> **摘要:** Earlier research has shown that metaphors influence human's decision making, which raises the question of whether metaphors also influence large language models (LLMs)' reasoning pathways, considering their training data contain a large number of metaphors. In this work, we investigate the problem in the scope of the emergent misalignment problem where LLMs can generalize patterns learned from misaligned content in one domain to another domain. We discover a strong causal relationship between metaphors in training data and the misalignment degree of LLMs' reasoning contents. With interventions using metaphors in pre-training, fine-tuning and re-alignment phases, models' cross-domain misalignment degrees change significantly. As we delve deeper into the causes behind this phenomenon, we observe that there is a connection between metaphors and the activation of global and local latent features of large reasoning models. By monitoring these latent features, we design a detector that predict misaligned content with high accuracy.
>
---
#### [new 043] DeepSynth-Eval: Objectively Evaluating Information Consolidation in Deep Survey Writing
- **分类: cs.CL**

- **简介: 该论文属于信息整合任务，旨在解决深度调研中合成阶段评估不足的问题。通过构建基准和评估协议，提升合成能力的客观评价。**

- **链接: [https://arxiv.org/pdf/2601.03540v1](https://arxiv.org/pdf/2601.03540v1)**

> **作者:** Hongzhi Zhang; Yuanze Hu; Tinghai Zhang; Jia Fu; Tao Wang; Junwei Jing; Zhaoxin Fan; Qi Wang; Ruiming Tang; Han Li; Guorui Zhou; Kun Gai
>
> **摘要:** The evolution of Large Language Models (LLMs) towards autonomous agents has catalyzed progress in Deep Research. While retrieval capabilities are well-benchmarked, the post-retrieval synthesis stage--where agents must digest massive amounts of context and consolidate fragmented evidence into coherent, long-form reports--remains under-evaluated due to the subjectivity of open-ended writing. To bridge this gap, we introduce DeepSynth-Eval, a benchmark designed to objectively evaluate information consolidation capabilities. We leverage high-quality survey papers as gold standards, reverse-engineering research requests and constructing "Oracle Contexts" from their bibliographies to isolate synthesis from retrieval noise. We propose a fine-grained evaluation protocol using General Checklists (for factual coverage) and Constraint Checklists (for structural organization), transforming subjective judgment into verifiable metrics. Experiments across 96 tasks reveal that synthesizing information from hundreds of references remains a significant challenge. Our results demonstrate that agentic plan-and-write workflows significantly outperform single-turn generation, effectively reducing hallucinations and improving adherence to complex structural constraints.
>
---
#### [new 044] InfiniteWeb: Scalable Web Environment Synthesis for GUI Agent Training
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出InfiniteWeb，用于生成可扩展的网页环境，解决GUI代理训练数据不足的问题。通过自动化生成真实网站，提升代理性能。**

- **链接: [https://arxiv.org/pdf/2601.04126v1](https://arxiv.org/pdf/2601.04126v1)**

> **作者:** Ziyun Zhang; Zezhou Wang; Xiaoyi Zhang; Zongyu Guo; Jiahao Li; Bin Li; Yan Lu
>
> **备注:** Work In Progress
>
> **摘要:** GUI agents that interact with graphical interfaces on behalf of users represent a promising direction for practical AI assistants. However, training such agents is hindered by the scarcity of suitable environments. We present InfiniteWeb, a system that automatically generates functional web environments at scale for GUI agent training. While LLMs perform well on generating a single webpage, building a realistic and functional website with many interconnected pages faces challenges. We address these challenges through unified specification, task-centric test-driven development, and a combination of website seed with reference design image to ensure diversity. Our system also generates verifiable task evaluators enabling dense reward signals for reinforcement learning. Experiments show that InfiniteWeb surpasses commercial coding agents at realistic website construction, and GUI agents trained on our generated environments achieve significant performance improvements on OSWorld and Online-Mind2Web, demonstrating the effectiveness of proposed system.
>
---
#### [new 045] eTracer: Towards Traceable Text Generation via Claim-Level Grounding
- **分类: cs.CL**

- **简介: 该论文提出eTracer，解决文本生成的可追溯性问题，通过主张级证据对齐提升生成内容的可信度与验证效率。属于可解释文本生成任务。**

- **链接: [https://arxiv.org/pdf/2601.03669v1](https://arxiv.org/pdf/2601.03669v1)**

> **作者:** Bohao Chu; Qianli Wang; Hendrik Damm; Hui Wang; Ula Muhabbek; Elisabeth Livingstone; Christoph M. Friedrich; Norbert Fuhr
>
> **备注:** ACL 2026 Conference Submission (8 main pages)
>
> **摘要:** How can system-generated responses be efficiently verified, especially in the high-stakes biomedical domain? To address this challenge, we introduce eTracer, a plug-and-play framework that enables traceable text generation by grounding claims against contextual evidence. Through post-hoc grounding, each response claim is aligned with contextual evidence that either supports or contradicts it. Building on claim-level grounding results, eTracer not only enables users to precisely trace responses back to their contextual source but also quantifies response faithfulness, thereby enabling the verifiability and trustworthiness of generated responses. Experiments show that our claim-level grounding approach alleviates the limitations of conventional grounding methods in aligning generated statements with contextual sentence-level evidence, resulting in substantial improvements in overall grounding quality and user verification efficiency. The code and data are available at https://github.com/chubohao/eTracer.
>
---
#### [new 046] SearchAttack: Red-Teaming LLMs against Real-World Threats via Framing Unsafe Web Information-Seeking Tasks
- **分类: cs.CL**

- **简介: 论文提出SearchAttack，用于对搜索增强的LLM进行红队测试，解决其在面对有害网络信息时的安全漏洞问题。通过构造特定查询，引导LLM利用搜索结果达成恶意目的。**

- **链接: [https://arxiv.org/pdf/2601.04093v1](https://arxiv.org/pdf/2601.04093v1)**

> **作者:** Yu Yan; Sheng Sun; Mingfeng Li; Zheming Yang; Chiwei Zhu; Fei Ma; Benfeng Xu; Min Liu
>
> **备注:** We find that the key to jailbreak the LLM is objectifying its safety responsibility, thus we delegate the open-web to inject harmful semantics and get the huge gain from unmoderated web resources
>
> **摘要:** Recently, people have suffered and become increasingly aware of the unreliability gap in LLMs for open and knowledge-intensive tasks, and thus turn to search-augmented LLMs to mitigate this issue. However, when the search engine is triggered for harmful tasks, the outcome is no longer under the LLM's control. Once the returned content directly contains targeted, ready-to-use harmful takeaways, the LLM's safeguards cannot withdraw that exposure. Motivated by this dilemma, we identify web search as a critical attack surface and propose \textbf{\textit{SearchAttack}} for red-teaming. SearchAttack outsources the harmful semantics to web search, retaining only the query's skeleton and fragmented clues, and further steers LLMs to reconstruct the retrieved content via structural rubrics to achieve malicious goals. Extensive experiments are conducted to red-team the search-augmented LLMs for responsible vulnerability assessment. Empirically, SearchAttack demonstrates strong effectiveness in attacking these systems.
>
---
#### [new 047] Advances and Challenges in Semantic Textual Similarity: A Comprehensive Survey
- **分类: cs.CL; cs.AI; math.OC**

- **简介: 该论文属于语义文本相似度任务，旨在综述STS领域的进展与挑战。论文总结了六类方法，分析了技术发展与应用，指出了未来研究方向。**

- **链接: [https://arxiv.org/pdf/2601.03270v1](https://arxiv.org/pdf/2601.03270v1)**

> **作者:** Lokendra Kumar; Neelesh S. Upadhye; Kannan Piedy
>
> **备注:** 16 pages, 2 figures
>
> **摘要:** Semantic Textual Similarity (STS) research has expanded rapidly since 2021, driven by advances in transformer architectures, contrastive learning, and domain-specific techniques. This survey reviews progress across six key areas: transformer-based models, contrastive learning, domain-focused solutions, multi-modal methods, graph-based approaches, and knowledge-enhanced techniques. Recent transformer models such as FarSSiBERT and DeBERTa-v3 have achieved remarkable accuracy, while contrastive methods like AspectCSE have established new benchmarks. Domain-adapted models, including CXR-BERT for medical texts and Financial-STS for finance, demonstrate how STS can be effectively customized for specialized fields. Moreover, multi-modal, graph-based, and knowledge-integrated models further enhance semantic understanding and representation. By organizing and analyzing these developments, the survey provides valuable insights into current methods, practical applications, and remaining challenges. It aims to guide researchers and practitioners alike in navigating rapid advancements, highlighting emerging trends and future opportunities in the evolving field of STS.
>
---
#### [new 048] Analyzing and Improving Cross-lingual Knowledge Transfer for Machine Translation
- **分类: cs.CL**

- **简介: 论文研究多语言机器翻译中的跨语言知识迁移问题，旨在提升模型在低资源语言上的表现。通过分析语言相似性、数据可用性及训练策略，提出改进方法以增强模型的泛化能力与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.04036v1](https://arxiv.org/pdf/2601.04036v1)**

> **作者:** David Stap
>
> **备注:** PhD dissertation defended on November 26th, 2025
>
> **摘要:** Multilingual machine translation systems aim to make knowledge accessible across languages, yet learning effective cross-lingual representations remains challenging. These challenges are especially pronounced for low-resource languages, where limited parallel data constrains generalization and transfer. Understanding how multilingual models share knowledge across languages requires examining the interaction between representations, data availability, and training strategies. In this thesis, we study cross-lingual knowledge transfer in neural models and develop methods to improve robustness and generalization in multilingual settings, using machine translation as a central testbed. We analyze how similarity between languages influences transfer, how retrieval and auxiliary supervision can strengthen low-resource translation, and how fine-tuning on parallel data can introduce unintended trade-offs in large language models. We further examine the role of language diversity during training and show that increasing translation coverage improves generalization and reduces off-target behavior. Together, this work highlights how modeling choices and data composition shape multilingual learning and offers insights toward more inclusive and resilient multilingual NLP systems.
>
---
#### [new 049] Layer-Order Inversion: Rethinking Latent Multi-Hop Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型的多跳推理机制，揭示层序反转现象，提出概率回忆与提取框架，解释多跳推理过程。**

- **链接: [https://arxiv.org/pdf/2601.03542v1](https://arxiv.org/pdf/2601.03542v1)**

> **作者:** Xukai Liu; Ye Liu; Jipeng Zhang; Yanghai Zhang; Kai Zhang; Qi Liu
>
> **备注:** 16 pages, 18 figures
>
> **摘要:** Large language models (LLMs) perform well on multi-hop reasoning, yet how they internally compose multiple facts remains unclear. Recent work proposes \emph{hop-aligned circuit hypothesis}, suggesting that bridge entities are computed sequentially across layers before later-hop answers. Through systematic analyses on real-world multi-hop queries, we show that this hop-aligned assumption does not generalize: later-hop answer entities can become decodable earlier than bridge entities, a phenomenon we call \emph{layer-order inversion}, which strengthens with total hops. To explain this behavior, we propose a \emph{probabilistic recall-and-extract} framework that models multi-hop reasoning as broad probabilistic recall in shallow MLP layers followed by selective extraction in deeper attention layers. This framework is empirically validated through systematic probing analyses, reinterpreting prior layer-wise decoding evidence, explaining chain-of-thought gains, and providing a mechanistic diagnosis of multi-hop failures despite correct single-hop knowledge. Code is available at https://github.com/laquabe/Layer-Order-Inversion.
>
---
#### [new 050] FLEx: Language Modeling with Few-shot Language Explanations
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出FLEx方法，用于通过少量语言解释提升语言模型性能。解决模型错误重复问题，通过聚类选择错误示例并生成解释提示，无需修改模型权重。**

- **链接: [https://arxiv.org/pdf/2601.04157v1](https://arxiv.org/pdf/2601.04157v1)**

> **作者:** Adar Avsian; Christopher Richardson; Anirudh Sundar; Larry Heck
>
> **摘要:** Language models have become effective at a wide range of tasks, from math problem solving to open-domain question answering. However, they still make mistakes, and these mistakes are often repeated across related queries. Natural language explanations can help correct these errors, but collecting them at scale may be infeasible, particularly in domains where expert annotators are required. To address this issue, we introduce FLEx ($\textbf{F}$ew-shot $\textbf{L}$anguage $\textbf{Ex}$planations), a method for improving model behavior using a small number of explanatory examples. FLEx selects representative model errors using embedding-based clustering, verifies that the associated explanations correct those errors, and summarizes them into a prompt prefix that is prepended at inference-time. This summary guides the model to avoid similar errors on new inputs, without modifying model weights. We evaluate FLEx on CounterBench, GSM8K, and ReasonIF. We find that FLEx consistently outperforms chain-of-thought (CoT) prompting across all three datasets and reduces up to 83\% of CoT's remaining errors.
>
---
#### [new 051] RedBench: A Universal Dataset for Comprehensive Red Teaming of Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出RedBench，一个用于评估大语言模型安全性的通用数据集，解决现有数据集分类不一致、覆盖有限的问题。任务是红队测试，工作包括整合37个数据集，建立标准化分类体系。**

- **链接: [https://arxiv.org/pdf/2601.03699v1](https://arxiv.org/pdf/2601.03699v1)**

> **作者:** Quy-Anh Dang; Chris Ngo; Truong-Son Hy
>
> **摘要:** As large language models (LLMs) become integral to safety-critical applications, ensuring their robustness against adversarial prompts is paramount. However, existing red teaming datasets suffer from inconsistent risk categorizations, limited domain coverage, and outdated evaluations, hindering systematic vulnerability assessments. To address these challenges, we introduce RedBench, a universal dataset aggregating 37 benchmark datasets from leading conferences and repositories, comprising 29,362 samples across attack and refusal prompts. RedBench employs a standardized taxonomy with 22 risk categories and 19 domains, enabling consistent and comprehensive evaluations of LLM vulnerabilities. We provide a detailed analysis of existing datasets, establish baselines for modern LLMs, and open-source the dataset and evaluation code. Our contributions facilitate robust comparisons, foster future research, and promote the development of secure and reliable LLMs for real-world deployment. Code: https://github.com/knoveleng/redeval
>
---
#### [new 052] LLM-MC-Affect: LLM-Based Monte Carlo Modeling of Affective Trajectories and Latent Ambiguity for Interpersonal Dynamic Insight
- **分类: cs.CL; cs.CY**

- **简介: 该论文提出LLM-MC-Affect，用于建模情感轨迹和隐含模糊性，解决人际互动中的情感动态分析问题。通过概率框架量化情感与交互耦合。**

- **链接: [https://arxiv.org/pdf/2601.03645v1](https://arxiv.org/pdf/2601.03645v1)**

> **作者:** Yu-Zheng Lin; Bono Po-Jen Shih; John Paul Martin Encinas; Elizabeth Victoria Abraham Achom; Karan Himanshu Patel; Jesus Horacio Pacheco; Sicong Shao; Jyotikrishna Dass; Soheil Salehi; Pratik Satam
>
> **摘要:** Emotional coordination is a core property of human interaction that shapes how relational meaning is constructed in real time. While text-based affect inference has become increasingly feasible, prior approaches often treat sentiment as a deterministic point estimate for individual speakers, failing to capture the inherent subjectivity, latent ambiguity, and sequential coupling found in mutual exchanges. We introduce LLM-MC-Affect, a probabilistic framework that characterizes emotion not as a static label, but as a continuous latent probability distribution defined over an affective space. By leveraging stochastic LLM decoding and Monte Carlo estimation, the methodology approximates these distributions to derive high-fidelity sentiment trajectories that explicitly quantify both central affective tendencies and perceptual ambiguity. These trajectories enable a structured analysis of interpersonal coupling through sequential cross-correlation and slope-based indicators, identifying leading or lagging influences between interlocutors. To validate the interpretive capacity of this approach, we utilize teacher-student instructional dialogues as a representative case study, where our quantitative indicators successfully distill high-level interaction insights such as effective scaffolding. This work establishes a scalable and deployable pathway for understanding interpersonal dynamics, offering a generalizable solution that extends beyond education to broader social and behavioral research.
>
---
#### [new 053] ContextFocus: Activation Steering for Contextual Faithfulness in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在外部信息与内部知识冲突时输出不忠实的问题。提出ContextFocus方法，在不微调模型的情况下提升上下文忠实性。**

- **链接: [https://arxiv.org/pdf/2601.04131v1](https://arxiv.org/pdf/2601.04131v1)**

> **作者:** Nikhil Anand; Shwetha Somasundaram; Anirudh Phukan; Apoorv Saxena; Koyel Mukherjee
>
> **摘要:** Large Language Models (LLMs) encode vast amounts of parametric knowledge during pre-training. As world knowledge evolves, effective deployment increasingly depends on their ability to faithfully follow externally retrieved context. When such evidence conflicts with the model's internal knowledge, LLMs often default to memorized facts, producing unfaithful outputs. In this work, we introduce ContextFocus, a lightweight activation steering approach that improves context faithfulness in such knowledge-conflict settings while preserving fluency and efficiency. Unlike prior approaches, our solution requires no model finetuning and incurs minimal inference-time overhead, making it highly efficient. We evaluate ContextFocus on the ConFiQA benchmark, comparing it against strong baselines including ContextDPO, COIECD, and prompting-based methods. Furthermore, we show that our method is complementary to prompting strategies and remains effective on larger models. Extensive experiments show that ContextFocus significantly improves contextual-faithfulness. Our results highlight the effectiveness, robustness, and efficiency of ContextFocus in improving contextual-faithfulness of LLM outputs.
>
---
#### [new 054] Decide Then Retrieve: A Training-Free Framework with Uncertainty-Guided Triggering and Dual-Path Retrieval
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决RAG模型中检索噪声和性能受限问题。提出DTR框架，通过不确定性引导检索触发和双路径检索机制提升效果。**

- **链接: [https://arxiv.org/pdf/2601.03908v1](https://arxiv.org/pdf/2601.03908v1)**

> **作者:** Wang Chen; Guanqiang Qi; Weikang Li; Yang Li; Deguo Xia; Jizhou Huang
>
> **摘要:** Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external knowledge, but existing approaches indiscriminately trigger retrieval and rely on single-path evidence construction, often introducing noise and limiting performance gains. In this work, we propose Decide Then Retrieve (DTR), a training-free framework that adaptively determines when retrieval is necessary and how external information should be selected. DTR leverages generation uncertainty to guide retrieval triggering and introduces a dual-path retrieval mechanism with adaptive information selection to better handle sparse and ambiguous queries. Extensive experiments across five open-domain QA benchmarks, multiple model scales, and different retrievers demonstrate that DTR consistently improves EM and F1 over standard RAG and strong retrieval-enhanced baselines, while reducing unnecessary retrievals. The code and data used in this paper are available at https://github.com/ChenWangHKU/DTR.
>
---
#### [new 055] Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models
- **分类: cs.CL; cs.CR; cs.LG**

- **简介: 该论文属于LLM安全评估任务，旨在提升红队测试效果。提出Jailbreak-Zero方法，通过攻击模型生成多样化对抗提示，实现更高的攻击成功率和更优的策略多样性。**

- **链接: [https://arxiv.org/pdf/2601.03265v1](https://arxiv.org/pdf/2601.03265v1)**

> **作者:** Kai Hu; Abhinav Aggarwal; Mehran Khodabandeh; David Zhang; Eric Hsin; Li Chen; Ankit Jain; Matt Fredrikson; Akash Bharadwaj
>
> **备注:** Socially Responsible and Trustworthy Foundation Models at NeurIPS 2025
>
> **摘要:** This paper introduces Jailbreak-Zero, a novel red teaming methodology that shifts the paradigm of Large Language Model (LLM) safety evaluation from a constrained example-based approach to a more expansive and effective policy-based framework. By leveraging an attack LLM to generate a high volume of diverse adversarial prompts and then fine-tuning this attack model with a preference dataset, Jailbreak-Zero achieves Pareto optimality across the crucial objectives of policy coverage, attack strategy diversity, and prompt fidelity to real user inputs. The empirical evidence demonstrates the superiority of this method, showcasing significantly higher attack success rates against both open-source and proprietary models like GPT-40 and Claude 3.5 when compared to existing state-of-the-art techniques. Crucially, Jailbreak-Zero accomplishes this while producing human-readable and effective adversarial prompts with minimal need for human intervention, thereby presenting a more scalable and comprehensive solution for identifying and mitigating the safety vulnerabilities of LLMs.
>
---
#### [new 056] SegNSP: Revisiting Next Sentence Prediction for Linear Text Segmentation
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于文本分割任务，旨在解决线性文本分段问题。通过将任务转化为下一步句子预测，提出SegNSP模型，提升分割质量。**

- **链接: [https://arxiv.org/pdf/2601.03474v1](https://arxiv.org/pdf/2601.03474v1)**

> **作者:** José Isidro; Filipe Cunha; Purificação Silvano; Alípio Jorge; Nuno Guimarães; Sérgio Nunes; Ricardo Campos
>
> **摘要:** Linear text segmentation is a long-standing problem in natural language processing (NLP), focused on dividing continuous text into coherent and semantically meaningful units. Despite its importance, the task remains challenging due to the complexity of defining topic boundaries, the variability in discourse structure, and the need to balance local coherence with global context. These difficulties hinder downstream applications such as summarization, information retrieval, and question answering. In this work, we introduce SegNSP, framing linear text segmentation as a next sentence prediction (NSP) task. Although NSP has largely been abandoned in modern pre-training, its explicit modeling of sentence-to-sentence continuity makes it a natural fit for detecting topic boundaries. We propose a label-agnostic NSP approach, which predicts whether the next sentence continues the current topic without requiring explicit topic labels, and enhance it with a segmentation-aware loss combined with harder negative sampling to better capture discourse continuity. Unlike recent proposals that leverage NSP alongside auxiliary topic classification, our approach avoids task-specific supervision. We evaluate our model against established baselines on two datasets, CitiLink-Minutes, for which we establish the first segmentation benchmark, and WikiSection. On CitiLink-Minutes, SegNSP achieves a B-$F_1$ of 0.79, closely aligning with human-annotated topic transitions, while on WikiSection it attains a B-F$_1$ of 0.65, outperforming the strongest reproducible baseline, TopSeg, by 0.17 absolute points. These results demonstrate competitive and robust performance, highlighting the effectiveness of modeling sentence-to-sentence continuity for improving segmentation quality and supporting downstream NLP applications.
>
---
#### [new 057] Evaluation Framework for AI Creativity: A Case Study Based on Story Generation
- **分类: cs.CL**

- **简介: 该论文属于AI故事生成评估任务，旨在解决创造性文本评价难题。提出包含四个维度的评估框架，通过实验验证其有效性。**

- **链接: [https://arxiv.org/pdf/2601.03698v1](https://arxiv.org/pdf/2601.03698v1)**

> **作者:** Pharath Sathya; Yin Jou Huang; Fei Cheng
>
> **备注:** Work in progress
>
> **摘要:** Evaluating creative text generation remains a challenge because existing reference-based metrics fail to capture the subjective nature of creativity. We propose a structured evaluation framework for AI story generation comprising four components (Novelty, Value, Adherence, and Resonance) and eleven sub-components. Using controlled story generation via ``Spike Prompting'' and a crowdsourced study of 115 readers, we examine how different creative components shape both immediate and reflective human creativity judgments. Our findings show that creativity is evaluated hierarchically rather than cumulatively, with different dimensions becoming salient at different stages of judgment, and that reflective evaluation substantially alters both ratings and inter-rater agreement. Together, these results support the effectiveness of our framework in revealing dimensions of creativity that are obscured by reference-based evaluation.
>
---
#### [new 058] ADEPT: Adaptive Dynamic Early-Exit Process for Transformers
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ADEPT，解决大语言模型推理中的计算效率问题，通过动态早停机制提升生成和分类任务效率。**

- **链接: [https://arxiv.org/pdf/2601.03700v1](https://arxiv.org/pdf/2601.03700v1)**

> **作者:** Sangmin Yoo; Srikanth Malla; Chiho Choi; Wei D. Lu; Joon Hee Choi
>
> **备注:** 11 figures, 8 tables, 22 pages
>
> **摘要:** The inference of large language models imposes significant computational workloads, often requiring the processing of billions of parameters. Although early-exit strategies have proven effective in reducing computational demands by halting inference earlier, they apply either to only the first token in the generation phase or at the prompt level in the prefill phase. Thus, the Key-Value (KV) cache for skipped layers remains a bottleneck for subsequent token generation, limiting the benefits of early exit. We introduce ADEPT (Adaptive Dynamic Early-exit Process for Transformers), a novel approach designed to overcome this issue and enable dynamic early exit in both the prefill and generation phases. The proposed adaptive token-level early-exit mechanism adjusts computation dynamically based on token complexity, optimizing efficiency without compromising performance. ADEPT further enhances KV generation procedure by decoupling sequential dependencies in skipped layers, making token-level early exit more practical. Experimental results demonstrate that ADEPT improves efficiency by up to 25% in language generation tasks and achieves a 4x speed-up in downstream classification tasks, with up to a 45% improvement in performance.
>
---
#### [new 059] PCoA: A New Benchmark for Medical Aspect-Based Summarization With Phrase-Level Context Attribution
- **分类: cs.CL**

- **简介: 该论文提出PCoA基准，解决医疗领域摘要的精确上下文归属问题。通过标注和评估框架，提升摘要质量与可信度。**

- **链接: [https://arxiv.org/pdf/2601.03418v1](https://arxiv.org/pdf/2601.03418v1)**

> **作者:** Bohao Chu; Sameh Frihat; Tabea M. G. Pakull; Hendrik Damm; Meijie Li; Ula Muhabbek; Georg Lodde; Norbert Fuhr
>
> **备注:** ACL 2026 Conference Submission (8 main pages)
>
> **摘要:** Verifying system-generated summaries remains challenging, as effective verification requires precise attribution to the source context, which is especially crucial in high-stakes medical domains. To address this challenge, we introduce PCoA, an expert-annotated benchmark for medical aspect-based summarization with phrase-level context attribution. PCoA aligns each aspect-based summary with its supporting contextual sentences and contributory phrases within them. We further propose a fine-grained, decoupled evaluation framework that independently assesses the quality of generated summaries, citations, and contributory phrases. Through extensive experiments, we validate the quality and consistency of the PCoA dataset and benchmark several large language models on the proposed task. Experimental results demonstrate that PCoA provides a reliable benchmark for evaluating system-generated summaries with phrase-level context attribution. Furthermore, comparative experiments show that explicitly identifying relevant sentences and contributory phrases before summarization can improve overall quality. The data and code are available at https://github.com/chubohao/PCoA.
>
---
#### [new 060] How Do Large Language Models Learn Concepts During Continual Pre-Training?
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在持续预训练中如何学习和遗忘概念，分析概念电路的动态变化，旨在提升模型的可解释性和训练策略。**

- **链接: [https://arxiv.org/pdf/2601.03570v1](https://arxiv.org/pdf/2601.03570v1)**

> **作者:** Barry Menglong Yao; Sha Li; Yunzhi Yao; Minqian Liu; Zaishuo Xia; Qifan Wang; Lifu Huang
>
> **备注:** 12 pages, 19 figures
>
> **摘要:** Human beings primarily understand the world through concepts (e.g., dog), abstract mental representations that structure perception, reasoning, and learning. However, how large language models (LLMs) acquire, retain, and forget such concepts during continual pretraining remains poorly understood. In this work, we study how individual concepts are acquired and forgotten, as well as how multiple concepts interact through interference and synergy. We link these behavioral dynamics to LLMs' internal Concept Circuits, computational subgraphs associated with specific concepts, and incorporate Graph Metrics to characterize circuit structure. Our analysis reveals: (1) LLMs concept circuits provide a non-trivial, statistically significant signal of concept learning and forgetting; (2) Concept circuits exhibit a stage-wise temporal pattern during continual pretraining, with an early increase followed by gradual decrease and stabilization; (3) concepts with larger learning gains tend to exhibit greater forgetting under subsequent training; (4) semantically similar concepts induce stronger interference than weakly related ones; (5) conceptual knowledge differs in their transferability, with some significantly facilitating the learning of others. Together, our findings offer a circuit-level view of concept learning dynamics and inform the design of more interpretable and robust concept-aware training strategies for LLMs.
>
---
#### [new 061] Prompting Underestimates LLM Capability for Time Series Classification
- **分类: cs.CL**

- **简介: 该论文研究LLM在时间序列分类任务中的表现。指出当前基于提示的评估低估了模型能力，通过对比提示输出与线性探测器，证明LLM内部表征有效，但评估方法存在偏差。**

- **链接: [https://arxiv.org/pdf/2601.03464v1](https://arxiv.org/pdf/2601.03464v1)**

> **作者:** Dan Schumacher; Erfan Nourbakhsh; Rocky Slavin; Anthony Rios
>
> **备注:** 8 pages + Appendix and References, 9 figures
>
> **摘要:** Prompt-based evaluations suggest that large language models (LLMs) perform poorly on time series classification, raising doubts about whether they encode meaningful temporal structure. We show that this conclusion reflects limitations of prompt-based generation rather than the model's representational capacity by directly comparing prompt outputs with linear probes over the same internal representations. While zero-shot prompting performs near chance, linear probes improve average F1 from 0.15-0.26 to 0.61-0.67, often matching or exceeding specialized time series models. Layer-wise analyses further show that class-discriminative time series information emerges in early transformer layers and is amplified by visual and multimodal inputs. Together, these results demonstrate a systematic mismatch between what LLMs internally represent and what prompt-based evaluation reveals, leading current evaluations to underestimate their time series understanding.
>
---
#### [new 062] Reasoning Model Is Superior LLM-Judge, Yet Suffers from Biases
- **分类: cs.CL**

- **简介: 该论文属于模型评估任务，旨在比较推理模型与非推理大语言模型的判断能力。研究发现推理模型在准确性、指令遵循和抗攻击性上更优，但存在表面质量偏差。为此提出PlanJudge策略以减少偏差。**

- **链接: [https://arxiv.org/pdf/2601.03630v1](https://arxiv.org/pdf/2601.03630v1)**

> **作者:** Hui Huang; Xuanxin Wu; Muyun Yang; Yuki Arase
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** This paper presents the first systematic comparison investigating whether Large Reasoning Models (LRMs) are superior judge to non-reasoning LLMs. Our empirical analysis yields four key findings: 1) LRMs outperform non-reasoning LLMs in terms of judgment accuracy, particularly on reasoning-intensive tasks; 2) LRMs demonstrate superior instruction-following capabilities in evaluation contexts; 3) LRMs exhibit enhanced robustness against adversarial attacks targeting judgment tasks; 4) However, LRMs still exhibit strong biases in superficial quality. To improve the robustness against biases, we propose PlanJudge, an evaluation strategy that prompts the model to generate an explicit evaluation plan before execution. Despite its simplicity, our experiments demonstrate that PlanJudge significantly mitigates biases in both LRMs and standard LLMs.
>
---
#### [new 063] Where meaning lives: Layer-wise accessibility of psycholinguistic features in encoder and decoder language models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究Transformer模型中语义信息的层次分布，属于自然语言理解任务。旨在解决心理语言特征在模型中的编码位置问题，通过层分析和不同嵌入方法比较，揭示语义在模型中的分布规律。**

- **链接: [https://arxiv.org/pdf/2601.03798v1](https://arxiv.org/pdf/2601.03798v1)**

> **作者:** Taisiia Tikhomirova; Dirk U. Wulff
>
> **摘要:** Understanding where transformer language models encode psychologically meaningful aspects of meaning is essential for both theory and practice. We conduct a systematic layer-wise probing study of 58 psycholinguistic features across 10 transformer models, spanning encoder-only and decoder-only architectures, and compare three embedding extraction methods. We find that apparent localization of meaning is strongly method-dependent: contextualized embeddings yield higher feature-specific selectivity and different layer-wise profiles than isolated embeddings. Across models and methods, final-layer representations are rarely optimal for recovering psycholinguistic information with linear probes. Despite these differences, models exhibit a shared depth ordering of meaning dimensions, with lexical properties peaking earlier and experiential and affective dimensions peaking later. Together, these results show that where meaning "lives" in transformer models reflects an interaction between methodological choices and architectural constraints.
>
---
#### [new 064] Membox: Weaving Topic Continuity into Long-Range Memory for LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Membox，解决LLM代理对话中主题连续性缺失的问题。通过构建层次化记忆结构，提升长期记忆的连贯性和效率。**

- **链接: [https://arxiv.org/pdf/2601.03785v1](https://arxiv.org/pdf/2601.03785v1)**

> **作者:** Dehao Tao; Guoliang Ma; Yongfeng Huang; Minghu Jiang
>
> **摘要:** Human-agent dialogues often exhibit topic continuity-a stable thematic frame that evolves through temporally adjacent exchanges-yet most large language model (LLM) agent memory systems fail to preserve it. Existing designs follow a fragmentation-compensation paradigm: they first break dialogue streams into isolated utterances for storage, then attempt to restore coherence via embedding-based retrieval. This process irreversibly damages narrative and causal flow, while biasing retrieval towards lexical similarity. We introduce membox, a hierarchical memory architecture centered on a Topic Loom that continuously monitors dialogue in a sliding-window fashion, grouping consecutive same-topic turns into coherent "memory boxes" at storage time. Sealed boxes are then linked by a Trace Weaver into long-range event-timeline traces, recovering macro-topic recurrences across discontinuities. Experiments on LoCoMo demonstrate that Membox achieves up to 68% F1 improvement on temporal reasoning tasks, outperforming competitive baselines (e.g., Mem0, A-MEM). Notably, Membox attains these gains while using only a fraction of the context tokens required by existing methods, highlighting a superior balance between efficiency and effectiveness. By explicitly modeling topic continuity, Membox offers a cognitively motivated mechanism for enhancing both coherence and efficiency in LLM agents.
>
---
#### [new 065] PALM-Bench: A Comprehensive Benchmark for Personalized Audio-Language Models
- **分类: cs.CL**

- **简介: 该论文属于个性化音频-语言模型任务，旨在解决通用模型无法有效支持个性化问答的问题。提出PALM-Bench基准，评估模型在多说话者场景下的个性化理解与推理能力。**

- **链接: [https://arxiv.org/pdf/2601.03531v1](https://arxiv.org/pdf/2601.03531v1)**

> **作者:** Yuwen Wang; Xinyuan Qian; Tian-Hao Zhang; Jiaran Gao; Yuchen Pan; Xin Wang; Zhou Pan; Chen Wei; Yiming Wang
>
> **备注:** Under review
>
> **摘要:** Large Audio-Language Models (LALMs) have demonstrated strong performance in audio understanding and generation. Yet, our extensive benchmarking reveals that their behavior is largely generic (e.g., summarizing spoken content) and fails to adequately support personalized question answering (e.g., summarizing what my best friend says). In contrast, human conditions their interpretation and decision-making on each individual's personal context. To bridge this gap, we formalize the task of Personalized LALMs (PALM) for recognizing personal concepts and reasoning within personal context. Moreover, we create the first benchmark (PALM-Bench) to foster the methodological advances in PALM and enable structured evaluation on several tasks across multi-speaker scenarios. Our extensive experiments on representative open-source LALMs, show that existing training-free prompting and supervised fine-tuning strategies, while yield improvements, remains limited in modeling personalized knowledge and transferring them across tasks robustly. Data and code will be released.
>
---
#### [new 066] Reasoning Pattern Alignment Merging for Adaptive Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于推理任务，旨在解决大模型推理路径冗长导致的计算和延迟问题。通过模型融合方法RPAM，实现高效自适应推理。**

- **链接: [https://arxiv.org/pdf/2601.03506v1](https://arxiv.org/pdf/2601.03506v1)**

> **作者:** Zhaofeng Zhong; Wei Yuan; Tong Chen; Xiangyu Zhao; Quoc Viet Hung Nguyen; Hongzhi Yin
>
> **备注:** 16 pages, 4 figures
>
> **摘要:** Recent large reasoning models (LRMs) have made substantial progress in complex reasoning tasks, yet they often generate lengthy reasoning paths for every query, incurring unnecessary computation and latency. Existing speed-up approaches typically rely on retraining the model or designing sophisticated prompting, which are either prohibitively expensive or highly sensitive to the input and prompt formulation. In this work, we study model merging as a lightweight alternative for efficient reasoning: by combining a long chain-of-thought (Long-CoT) reasoning model with a Short-CoT instruction model, we obtain an adaptive reasoner without training from scratch or requiring large-scale additional data. Building on this idea, we propose Reasoning Pattern Alignment Merging (RPAM), a layer-wise model merging framework based on feature alignment to facilitate query-adaptive reasoning. RPAM first constructs a small pattern-labeled calibration set that assigns each query an appropriate reasoning pattern. It then optimizes layer-wise merging coefficients by aligning the merged model's intermediate representations with those of the selected model, while a contrastive objective explicitly pushes them away from the non-selected model. Experiments on seven widely used reasoning benchmarks show that RPAM substantially reduces inference cost while maintaining strong performance. Upon article acceptance, we will provide open-source code to reproduce experiments for RPAM.
>
---
#### [new 067] NeuronScope: A Multi-Agent Framework for Explaining Polysemantic Neurons in Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于神经网络解释任务，旨在解决语言模型中神经元多义性问题。提出NeuronScope框架，通过迭代方式分解并解释神经元激活，提升解释准确性。**

- **链接: [https://arxiv.org/pdf/2601.03671v1](https://arxiv.org/pdf/2601.03671v1)**

> **作者:** Weiqi Liu; Yongliang Miao; Haiyan Zhao; Yanguang Liu; Mengnan Du
>
> **摘要:** Neuron-level interpretation in large language models (LLMs) is fundamentally challenged by widespread polysemanticity, where individual neurons respond to multiple distinct semantic concepts. Existing single-pass interpretation methods struggle to faithfully capture such multi-concept behavior. In this work, we propose NeuronScope, a multi-agent framework that reformulates neuron interpretation as an iterative, activation-guided process. NeuronScope explicitly deconstructs neuron activations into atomic semantic components, clusters them into distinct semantic modes, and iteratively refines each explanation using neuron activation feedback. Experiments demonstrate that NeuronScope uncovers hidden polysemanticity and produces explanations with significantly higher activation correlation compared to single-pass baselines.
>
---
#### [new 068] A path to natural language through tokenisation and transformers
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 论文探讨了BPE在自然语言处理中的作用，分析其如何影响词频分布和信息熵，揭示BPE作为统计变换机制的特性。任务为语言模型的tokenisation研究，解决BPE与语言统计规律的关系问题。**

- **链接: [https://arxiv.org/pdf/2601.03368v1](https://arxiv.org/pdf/2601.03368v1)**

> **作者:** David S. Berman; Alexander G. Stapleton
>
> **备注:** 19 pages, 7 figures, 2 tables
>
> **摘要:** Natural languages exhibit striking regularities in their statistical structure, including notably the emergence of Zipf's and Heaps' laws. Despite this, it remains broadly unclear how these properties relate to the modern tokenisation schemes used in contemporary transformer models. In this note, we analyse the information content (as measured by the Shannon entropy) of various corpora under the assumption of a Zipfian frequency distribution, and derive a closed-form expression for the slot entropy expectation value. We then empirically investigate how byte--pair encoding (BPE) transforms corpus statistics, showing that recursive applications of BPE drive token frequencies toward a Zipfian power law while inducing a characteristic growth pattern in empirical entropy. Utilizing the ability of transformers to learn context dependent token probability distributions, we train language models on corpora tokenised at varying BPE depths, revealing that the model predictive entropies increasingly agree with Zipf-derived predictions as the BPE depth increases. Attention-based diagnostics further indicate that deeper tokenisation reduces local token dependencies, bringing the empirical distribution closer to the weakly dependent (near IID) regime. Together, these results clarify how BPE acts not only as a compression mechanism but also as a statistical transform that reconstructs key informational properties of natural language.
>
---
#### [new 069] Atlas: Orchestrating Heterogeneous Models and Tools for Multi-Domain Complex Reasoning
- **分类: cs.CL**

- **简介: 该论文提出ATLAS框架，解决多领域复杂推理中异构模型与工具的协同问题，通过双路径方法提升性能。**

- **链接: [https://arxiv.org/pdf/2601.03872v1](https://arxiv.org/pdf/2601.03872v1)**

> **作者:** Jinyang Wu; Guocheng Zhai; Ruihan Jin; Jiahao Yuan; Yuhao Shen; Shuai Zhang; Zhengqi Wen; Jianhua Tao
>
> **摘要:** The integration of large language models (LLMs) with external tools has significantly expanded the capabilities of AI agents. However, as the diversity of both LLMs and tools increases, selecting the optimal model-tool combination becomes a high-dimensional optimization challenge. Existing approaches often rely on a single model or fixed tool-calling logic, failing to exploit the performance variations across heterogeneous model-tool pairs. In this paper, we present ATLAS (Adaptive Tool-LLM Alignment and Synergistic Invocation), a dual-path framework for dynamic tool usage in cross-domain complex reasoning. ATLAS operates via a dual-path approach: (1) \textbf{training-free cluster-based routing} that exploits empirical priors for domain-specific alignment, and (2) \textbf{RL-based multi-step routing} that explores autonomous trajectories for out-of-distribution generalization. Extensive experiments across 15 benchmarks demonstrate that our method outperforms closed-source models like GPT-4o, surpassing existing routing methods on both in-distribution (+10.1%) and out-of-distribution (+13.1%) tasks. Furthermore, our framework shows significant gains in visual reasoning by orchestrating specialized multi-modal tools.
>
---
#### [new 070] IntroLM: Introspective Language Models via Prefilling-Time Self-Evaluation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出IntroLM，解决LLM输出质量预测问题，通过自省token实现自我评估，无需外部模型，提升效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2601.03511v1](https://arxiv.org/pdf/2601.03511v1)**

> **作者:** Hossein Hosseini Kasnavieh; Gholamreza Haffari; Chris Leckie; Adel N. Toosi
>
> **摘要:** A major challenge for the operation of large language models (LLMs) is how to predict whether a specific LLM will produce sufficiently high-quality output for a given query. Existing approaches rely on external classifiers, most commonly BERT based models, which suffer from limited context windows, constrained representational capacity, and additional computational overhead. We propose IntroLM, a method that enables causal language models to predict their own output quality during the prefilling phase without affecting generation using introspective tokens. By introducing token conditional LoRA that activates only for the introspective token, the model learns to predict the output quality for a given query while preserving the original backbone behavior and avoiding external evaluators. On question answering benchmarks, IntroLM applied to Qwen3 8B achieves a ROC AUC of 90 precent for success prediction, outperforming a DeBERTa classifier by 14 precent. When integrated into multi model routing systems, IntroLM achieves superior cost performance tradeoffs, reducing latency by up to 33 precent and large model usage by up to 50 precent at matched reliability.
>
---
#### [new 071] EpiQAL: Benchmarking Large Language Models in Epidemiological Question Answering for Enhanced Alignment and Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出EpiQAL，首个针对流行病学问答的基准，解决证据驱动的流行病推理问题，通过三个子集评估事实回忆、多步推理和结论重建。**

- **链接: [https://arxiv.org/pdf/2601.03471v1](https://arxiv.org/pdf/2601.03471v1)**

> **作者:** Mingyang Wei; Dehai Min; Zewen Liu; Yuzhang Xie; Guanchen Wu; Carl Yang; Max S. Y. Lau; Qi He; Lu Cheng; Wei Jin
>
> **备注:** 21 pages, 3 figures, 12 tables
>
> **摘要:** Reliable epidemiological reasoning requires synthesizing study evidence to infer disease burden, transmission dynamics, and intervention effects at the population level. Existing medical question answering benchmarks primarily emphasize clinical knowledge or patient-level reasoning, yet few systematically evaluate evidence-grounded epidemiological inference. We present EpiQAL, the first diagnostic benchmark for epidemiological question answering across diverse diseases, comprising three subsets built from open-access literature. The subsets respectively evaluate text-grounded factual recall, multi-step inference linking document evidence with epidemiological principles, and conclusion reconstruction with the Discussion section withheld. Construction combines expert-designed taxonomy guidance, multi-model verification, and retrieval-based difficulty control. Experiments on ten open models reveal that current LLMs show limited performance on epidemiological reasoning, with multi-step inference posing the greatest challenge. Model rankings shift across subsets, and scale alone does not predict success. Chain-of-Thought prompting benefits multi-step inference but yields mixed results elsewhere. EpiQAL provides fine-grained diagnostic signals for evidence grounding, inferential reasoning, and conclusion reconstruction.
>
---
#### [new 072] Benchmark^2: Systematic Evaluation of LLM Benchmarks
- **分类: cs.CL**

- **简介: 该论文属于模型评估任务，旨在解决LLM基准测试质量评估问题。提出Benchmark^2框架，通过三个指标评估基准质量，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2601.03986v1](https://arxiv.org/pdf/2601.03986v1)**

> **作者:** Qi Qian; Chengsong Huang; Jingwen Xu; Changze Lv; Muling Wu; Wenhao Liu; Xiaohua Wang; Zhenghua Wang; Zisu Huang; Muzhao Tian; Jianhan Xu; Kun Hu; He-Da Wang; Yao Hu; Xuanjing Huang; Xiaoqing Zheng
>
> **摘要:** The rapid proliferation of benchmarks for evaluating large language models (LLMs) has created an urgent need for systematic methods to assess benchmark quality itself. We propose Benchmark^2, a comprehensive framework comprising three complementary metrics: (1) Cross-Benchmark Ranking Consistency, measuring whether a benchmark produces model rankings aligned with peer benchmarks; (2) Discriminability Score, quantifying a benchmark's ability to differentiate between models; and (3) Capability Alignment Deviation, identifying problematic instances where stronger models fail but weaker models succeed within the same model family. We conduct extensive experiments across 15 benchmarks spanning mathematics, reasoning, and knowledge domains, evaluating 11 LLMs across four model families. Our analysis reveals significant quality variations among existing benchmarks and demonstrates that selective benchmark construction based on our metrics can achieve comparable evaluation performance with substantially reduced test sets.
>
---
#### [new 073] Tracing the complexity profiles of different linguistic phenomena through the intrinsic dimension of LLM representations
- **分类: cs.CL**

- **简介: 该论文研究语言复杂性在LLM表示中的内在维度（ID），旨在区分不同复杂性类型。通过分析ID差异，揭示语言处理阶段。属于自然语言处理任务，解决复杂性表征问题。**

- **链接: [https://arxiv.org/pdf/2601.03779v1](https://arxiv.org/pdf/2601.03779v1)**

> **作者:** Marco Baroni; Emily Cheng; Iria deDios-Flores; Francesca Franzon
>
> **摘要:** We explore the intrinsic dimension (ID) of LLM representations as a marker of linguistic complexity, asking if different ID profiles across LLM layers differentially characterize formal and functional complexity. We find the formal contrast between sentences with multiple coordinated or subordinated clauses to be reflected in ID differences whose onset aligns with a phase of more abstract linguistic processing independently identified in earlier work. The functional contrasts between sentences characterized by right branching vs. center embedding or unambiguous vs. ambiguous relative clause attachment are also picked up by ID, but in a less marked way, and they do not correlate with the same processing phase. Further experiments using representational similarity and layer ablation confirm the same trends. We conclude that ID is a useful marker of linguistic complexity in LLMs, that it allows to differentiate between different types of complexity, and that it points to similar stages of linguistic processing across disparate LLMs.
>
---
#### [new 074] Rethinking Table Pruning in TableQA: From Sequential Revisions to Gold Trajectory-Supervised Parallel Search
- **分类: cs.CL**

- **简介: 该论文属于表格问答任务，旨在解决表 pruning 中因依赖不可靠信号导致的关键数据丢失问题。提出 TabTrim 框架，通过黄金轨迹监督并行搜索提升表 pruning 效果。**

- **链接: [https://arxiv.org/pdf/2601.03851v1](https://arxiv.org/pdf/2601.03851v1)**

> **作者:** Yu Guo; Shenghao Ye; Shuangwu Chen; Zijian Wen; Tao Zhang; Qirui Bai; Dong Jin; Yunpeng Hou; Huasen He; Jian Yang; Xiaobin Tan
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** Table Question Answering (TableQA) benefits significantly from table pruning, which extracts compact sub-tables by eliminating redundant cells to streamline downstream reasoning. However, existing pruning methods typically rely on sequential revisions driven by unreliable critique signals, often failing to detect the loss of answer-critical data. To address this limitation, we propose TabTrim, a novel table pruning framework which transforms table pruning from sequential revisions to gold trajectory-supervised parallel search. TabTrim derives a gold pruning trajectory using the intermediate sub-tables in the execution process of gold SQL queries, and trains a pruner and a verifier to make the step-wise pruning result align with the gold pruning trajectory. During inference, TabTrim performs parallel search to explore multiple candidate pruning trajectories and identify the optimal sub-table. Extensive experiments demonstrate that TabTrim achieves state-of-the-art performance across diverse tabular reasoning tasks: TabTrim-8B reaches 73.5% average accuracy, outperforming the strongest baseline by 3.2%, including 79.4% on WikiTQ and 61.2% on TableBench.
>
---
#### [new 075] When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life
- **分类: cs.CL**

- **简介: 该论文属于多模态安全评估任务，旨在解决MLLM生成不安全内容的问题。提出SaLAD基准和评估框架，分析模型在日常场景中的安全表现。**

- **链接: [https://arxiv.org/pdf/2601.04043v1](https://arxiv.org/pdf/2601.04043v1)**

> **作者:** Xinyue Lou; Jinan Xu; Jingyi Yin; Xiaolong Wang; Zhaolu Kang; Youwei Liao; Yixuan Wang; Xiangyu Shi; Fengran Mo; Su Yao; Kaiyu Huang
>
> **摘要:** As Multimodal Large Language Models (MLLMs) become an indispensable assistant in human life, the unsafe content generated by MLLMs poses a danger to human behavior, perpetually overhanging human society like a sword of Damocles. To investigate and evaluate the safety impact of MLLMs responses on human behavior in daily life, we introduce SaLAD, a multimodal safety benchmark which contains 2,013 real-world image-text samples across 10 common categories, with a balanced design covering both unsafe scenarios and cases of oversensitivity. It emphasizes realistic risk exposure, authentic visual inputs, and fine-grained cross-modal reasoning, ensuring that safety risks cannot be inferred from text alone. We further propose a safety-warning-based evaluation framework that encourages models to provide clear and informative safety warnings, rather than generic refusals. Results on 18 MLLMs demonstrate that the top-performing models achieve a safe response rate of only 57.2% on unsafe queries. Moreover, even popular safety alignment methods limit effectiveness of the models in our scenario, revealing the vulnerabilities of current MLLMs in identifying dangerous behaviors in daily life. Our dataset is available at https://github.com/xinyuelou/SaLAD.
>
---
#### [new 076] Step Potential Advantage Estimation: Harnessing Intermediate Confidence and Correctness for Efficient Mathematical Reasoning
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM推理过程中奖励估计粗粒度的问题。通过引入中间置信度与正确性，提出SPAE方法实现细粒度信用分配，提升推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.03823v1](https://arxiv.org/pdf/2601.03823v1)**

> **作者:** Fei Wu; Zhenrong Zhang; Qikai Chang; Jianshu Zhang; Quan Liu; Jun Du
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) elicits long chain-of-thought reasoning in large language models (LLMs), but outcome-based rewards lead to coarse-grained advantage estimation. While existing approaches improve RLVR via token-level entropy or sequence-level length control, they lack a semantically grounded, step-level measure of reasoning progress. As a result, LLMs fail to distinguish necessary deduction from redundant verification: they may continue checking after reaching a correct solution and, in extreme cases, overturn a correct trajectory into an incorrect final answer. To remedy the lack of process supervision, we introduce a training-free probing mechanism that extracts intermediate confidence and correctness and combines them into a Step Potential signal that explicitly estimates the reasoning state at each step. Building on this signal, we propose Step Potential Advantage Estimation (SPAE), a fine-grained credit assignment method that amplifies potential gains, penalizes potential drops, and applies penalty after potential saturates to encourage timely termination. Experiments across multiple benchmarks show SPAE consistently improves accuracy while substantially reducing response length, outperforming strong RL baselines and recent efficient reasoning and token-level advantage estimation methods. The code is available at https://github.com/cii030/SPAE-RL.
>
---
#### [new 077] Simulated Students in Tutoring Dialogues: Substance or Illusion?
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于教育技术领域，旨在解决LLM在教学对话中模拟学生效果不佳的问题。通过定义任务、提出评估指标并测试多种方法，发现现有策略效果有限，需进一步研究。**

- **链接: [https://arxiv.org/pdf/2601.04025v1](https://arxiv.org/pdf/2601.04025v1)**

> **作者:** Alexander Scarlatos; Jaewook Lee; Simon Woodhead; Andrew Lan
>
> **摘要:** Advances in large language models (LLMs) enable many new innovations in education. However, evaluating the effectiveness of new technology requires real students, which is time-consuming and hard to scale up. Therefore, many recent works on LLM-powered tutoring solutions have used simulated students for both training and evaluation, often via simple prompting. Surprisingly, little work has been done to ensure or even measure the quality of simulated students. In this work, we formally define the student simulation task, propose a set of evaluation metrics that span linguistic, behavioral, and cognitive aspects, and benchmark a wide range of student simulation methods on these metrics. We experiment on a real-world math tutoring dialogue dataset, where both automated and human evaluation results show that prompting strategies for student simulation perform poorly; supervised fine-tuning and preference optimization yield much better but still limited performance, motivating future work on this challenging task.
>
---
#### [new 078] Do LLM Self-Explanations Help Users Predict Model Behavior? Evaluating Counterfactual Simulatability with Pragmatic Perturbations
- **分类: cs.CL**

- **简介: 该论文属于模型解释性研究任务，旨在评估自解释是否帮助用户预测模型行为。通过实验比较不同扰动策略，发现自解释能提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2601.03775v1](https://arxiv.org/pdf/2601.03775v1)**

> **作者:** Pingjun Hong; Benjamin Roth
>
> **摘要:** Large Language Models (LLMs) can produce verbalized self-explanations, yet prior studies suggest that such rationales may not reliably reflect the model's true decision process. We ask whether these explanations nevertheless help users predict model behavior, operationalized as counterfactual simulatability. Using StrategyQA, we evaluate how well humans and LLM judges can predict a model's answers to counterfactual follow-up questions, with and without access to the model's chain-of-thought or post-hoc explanations. We compare LLM-generated counterfactuals with pragmatics-based perturbations as alternative ways to construct test cases for assessing the potential usefulness of explanations. Our results show that self-explanations consistently improve simulation accuracy for both LLM judges and humans, but the degree and stability of gains depend strongly on the perturbation strategy and judge strength. We also conduct a qualitative analysis of free-text justifications written by human users when predicting the model's behavior, which provides evidence that access to explanations helps humans form more accurate predictions on the perturbed questions.
>
---
#### [new 079] RADAR: Retrieval-Augmented Detector with Adversarial Refinement for Robust Fake News Detection
- **分类: cs.CL**

- **简介: 该论文属于虚假新闻检测任务，旨在对抗大模型生成的谣言。提出RADAR方法，结合检索增强与对抗优化，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.03981v1](https://arxiv.org/pdf/2601.03981v1)**

> **作者:** Song-Duo Ma; Yi-Hung Liu; Hsin-Yu Lin; Pin-Yu Chen; Hong-Yan Huang; Shau-Yung Hsu; Yun-Nung Chen
>
> **摘要:** To efficiently combat the spread of LLM-generated misinformation, we present RADAR, a retrieval-augmented detector with adversarial refinement for robust fake news detection. Our approach employs a generator that rewrites real articles with factual perturbations, paired with a lightweight detector that verifies claims using dense passage retrieval. To enable effective co-evolution, we introduce verbal adversarial feedback (VAF). Rather than relying on scalar rewards, VAF issues structured natural-language critiques; these guide the generator toward more sophisticated evasion attempts, compelling the detector to adapt and improve. On a fake news detection benchmark, RADAR achieves 86.98% ROC-AUC, significantly outperforming general-purpose LLMs with retrieval. Ablation studies confirm that detector-side retrieval yields the largest gains, while VAF and few-shot demonstrations provide critical signals for robust training.
>
---
#### [new 080] Modular Prompt Optimization: Optimizing Structured Prompts with Section-Local Textual Gradients
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升小模型的推理性能。针对传统提示优化方法的不足，提出MPO框架，通过分段优化提升提示质量。**

- **链接: [https://arxiv.org/pdf/2601.04055v1](https://arxiv.org/pdf/2601.04055v1)**

> **作者:** Prith Sharma; Austin Z. Henley
>
> **摘要:** Prompt quality plays a central role in controlling the behavior, reliability, and reasoning performance of large language models (LLMs), particularly for smaller open-source instruction-tuned models that depend heavily on explicit structure. While recent work has explored automatic prompt optimization using textual gradients and self-refinement, most existing methods treat prompts as monolithic blocks of text, making it difficult to localize errors, preserve critical instructions, or prevent uncontrolled prompt growth. We introduce Modular Prompt Optimization (MPO), a schema-based prompt optimization framework that treats prompts as structured objects composed of fixed semantic sections, including system role, context, task description, constraints, and output format. MPO applies section-local textual gradients, generated by a critic language model, to refine each section independently while keeping the overall prompt schema fixed. Section updates are consolidated through de-duplication to reduce redundancy and interference between components, yielding an interpretable and robust optimization process. We evaluate MPO on two reasoning benchmarks, ARC-Challenge and MMLU, using LLaMA-3 8B-Instruct and Mistral-7B-Instruct as solver models. Across both benchmarks and models, MPO consistently outperforms an untuned structured prompt and the TextGrad baseline, achieving substantial accuracy gains without modifying model parameters or altering prompt structure. These results demonstrate that maintaining a fixed prompt schema while applying localized, section-wise optimization is an effective and practical approach for improving reasoning performance in small open-source LMs.
>
---
#### [new 081] Compact Example-Based Explanations for Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型解释任务，旨在提升语言模型的例证解释质量。解决如何有效选择训练数据作为解释的问题，提出新的选择相关性评分，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2601.03786v1](https://arxiv.org/pdf/2601.03786v1)**

> **作者:** Loris Schoenegger; Benjamin Roth
>
> **备注:** 8 pages
>
> **摘要:** Training data influence estimation methods quantify the contribution of training documents to a model's output, making them a promising source of information for example-based explanations. As humans cannot interpret thousands of documents, only a small subset of the training data can be presented as an explanation. Although the choice of which documents to include directly affects explanation quality, previous evaluations of such systems have largely ignored any selection strategies. To address this, we propose a novel selection relevance score, a retraining-free metric that quantifies how useful a set of examples is for explaining a model's output. We validate this score through fine-tuning experiments, confirming that it can predict whether a set of examples supports or undermines the model's predictions. Using this metric, we further show that common selection strategies often underperform random selection. Motivated by this finding, we propose a strategy that balances influence and representativeness, enabling better use of selection budgets than naively selecting the highest-ranking examples.
>
---
#### [new 082] SpeakerSleuth: Evaluating Large Audio-Language Models as Judges for Multi-turn Speaker Consistency
- **分类: cs.CL**

- **简介: 该论文属于语音生成质量评估任务，旨在解决LALMs在多轮对话中判断说话人一致性的问题。通过构建基准测试，发现模型在音频一致性判断上存在偏差，优先文本而非声学特征。**

- **链接: [https://arxiv.org/pdf/2601.04029v1](https://arxiv.org/pdf/2601.04029v1)**

> **作者:** Jonggeun Lee; Junseong Pyo; Gyuhyeon Seo; Yohan Jo
>
> **备注:** 28 pages
>
> **摘要:** Large Audio-Language Models (LALMs) as judges have emerged as a prominent approach for evaluating speech generation quality, yet their ability to assess speaker consistency across multi-turn conversations remains unexplored. We present SpeakerSleuth, a benchmark evaluating whether LALMs can reliably judge speaker consistency in multi-turn dialogues through three tasks reflecting real-world requirements. We construct 1,818 human-verified evaluation instances across four diverse datasets spanning synthetic and real speech, with controlled acoustic difficulty. Evaluating nine widely-used LALMs, we find that models struggle to reliably detect acoustic inconsistencies. For instance, given audio samples of the same speaker's turns, some models overpredict inconsistency, whereas others are overly lenient. Models further struggle to identify the exact turns that are problematic. When other interlocutors' turns are provided together, performance degrades dramatically as models prioritize textual coherence over acoustic cues, failing to detect even obvious gender switches for a speaker. On the other hand, models perform substantially better in choosing the audio that best matches the speaker among several acoustic variants, demonstrating inherent acoustic discrimination capabilities. These findings expose a significant bias in LALMs: they tend to prioritize text over acoustics, revealing fundamental modality imbalances that need to be addressed to build reliable audio-language judges.
>
---
#### [new 083] KDCM: Reducing Hallucination in LLMs through Explicit Reasoning Structures
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs中的幻觉问题。通过引入代码引导的推理结构，提升模型推理的准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2601.04086v1](https://arxiv.org/pdf/2601.04086v1)**

> **作者:** Jinbo Hao; Kai Yang; Qingzhen Su; Yifan Li; Chao Jiang
>
> **摘要:** To mitigate hallucinations in large language models (LLMs), we propose a framework that focuses on errors induced by prompts. Our method extends a chain-style knowledge distillation approach by incorporating a programmable module that guides knowledge graph exploration. This module is embedded as executable code within the reasoning prompt, allowing the model to leverage external structured knowledge during inference. Based on this design, we develop an enhanced distillation-based reasoning framework that explicitly regulates intermediate reasoning steps, resulting in more reliable predictions. We evaluate the proposed approach on multiple public benchmarks using GPT-4 and LLaMA-3.3. Experimental results show that code-guided reasoning significantly improves contextual modeling and reduces prompt-induced hallucinations. Specifically, HIT@1, HIT@3, and HIT@5 increase by 15.64%, 13.38%, and 13.28%, respectively, with scores exceeding 95% across several evaluation settings. These findings indicate that the proposed method effectively constrains erroneous reasoning while improving both accuracy and interpretability.
>
---
#### [new 084] DisastQA: A Comprehensive Benchmark for Evaluating Question Answering in Disaster Management
- **分类: cs.CL**

- **简介: 该论文提出DisastQA，一个用于评估灾害管理中问答任务的基准。针对现有基准在不确定信息上的不足，构建了3000个经过验证的问题，涵盖八类灾害，通过人机协作确保数据质量，并设计了新的评估方法。**

- **链接: [https://arxiv.org/pdf/2601.03670v1](https://arxiv.org/pdf/2601.03670v1)**

> **作者:** Zhitong Chen; Kai Yin; Xiangjue Dong; Chengkai Liu; Xiangpeng Li; Yiming Xiao; Bo Li; Junwei Ma; Ali Mostafavi; James Caverlee
>
> **摘要:** Accurate question answering (QA) in disaster management requires reasoning over uncertain and conflicting information, a setting poorly captured by existing benchmarks built on clean evidence. We introduce DisastQA, a large-scale benchmark of 3,000 rigorously verified questions (2,000 multiple-choice and 1,000 open-ended) spanning eight disaster types. The benchmark is constructed via a human-LLM collaboration pipeline with stratified sampling to ensure balanced coverage. Models are evaluated under varying evidence conditions, from closed-book to noisy evidence integration, enabling separation of internal knowledge from reasoning under imperfect information. For open-ended QA, we propose a human-verified keypoint-based evaluation protocol emphasizing factual completeness over verbosity. Experiments with 20 models reveal substantial divergences from general-purpose leaderboards such as MMLU-Pro. While recent open-weight models approach proprietary systems in clean settings, performance degrades sharply under realistic noise, exposing critical reliability gaps for disaster response. All code, data, and evaluation resources are available at https://github.com/TamuChen18/DisastQA_open.
>
---
#### [new 085] WRAVAL -- WRiting Assist eVALuation
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出WRAVAL框架，用于评估小型语言模型在非推理任务中的表现。针对现有评估方法忽略工业应用效果的问题，结合数据生成与LLM评价，提升模型实用性能。**

- **链接: [https://arxiv.org/pdf/2601.03268v1](https://arxiv.org/pdf/2601.03268v1)**

> **作者:** Gabriel Benedict; Matthew Butler; Naved Merchant; Eetu Salama-Laine
>
> **摘要:** The emergence of Large Language Models (LLMs) has shifted language model evaluation toward reasoning and problem-solving tasks as measures of general intelligence. Small Language Models (SLMs) -- defined here as models under 10B parameters -- typically score 3-4 times lower than LLMs on these metrics. However, we demonstrate that these evaluations fail to capture SLMs' effectiveness in common industrial applications, such as tone modification tasks (e.g., funny, serious, professional). We propose an evaluation framework specifically designed to highlight SLMs' capabilities in non-reasoning tasks where predefined evaluation datasets don't exist. Our framework combines novel approaches in data generation, prompt-tuning, and LLM-based evaluation to demonstrate the potential of task-specific finetuning. This work provides practitioners with tools to effectively benchmark both SLMs and LLMs for practical applications, particularly in edge and private computing scenarios. Our implementation is available at: https://github.com/amazon-science/wraval.
>
---
#### [new 086] PsychEthicsBench: Evaluating Large Language Models Against Australian Mental Health Ethics
- **分类: cs.CL**

- **简介: 该论文属于伦理评估任务，旨在解决LLM在心理健康应用中的伦理对齐问题。提出PsychEthicsBench基准，评估模型的伦理知识与行为响应。**

- **链接: [https://arxiv.org/pdf/2601.03578v1](https://arxiv.org/pdf/2601.03578v1)**

> **作者:** Yaling Shen; Stephanie Fong; Yiwen Jiang; Zimu Wang; Feilong Tang; Qingyang Xu; Xiangyu Zhao; Zhongxing Xu; Jiahe Liu; Jinpeng Hu; Dominic Dwyer; Zongyuan Ge
>
> **备注:** 17 pages
>
> **摘要:** The increasing integration of large language models (LLMs) into mental health applications necessitates robust frameworks for evaluating professional safety alignment. Current evaluative approaches primarily rely on refusal-based safety signals, which offer limited insight into the nuanced behaviors required in clinical practice. In mental health, clinically inadequate refusals can be perceived as unempathetic and discourage help-seeking. To address this gap, we move beyond refusal-centric metrics and introduce \texttt{PsychEthicsBench}, the first principle-grounded benchmark based on Australian psychology and psychiatry guidelines, designed to evaluate LLMs' ethical knowledge and behavioral responses through multiple-choice and open-ended tasks with fine-grained ethicality annotations. Empirical results across 14 models reveal that refusal rates are poor indicators of ethical behavior, revealing a significant divergence between safety triggers and clinical appropriateness. Notably, we find that domain-specific fine-tuning can degrade ethical robustness, as several specialized models underperform their base backbones in ethical alignment. PsychEthicsBench provides a foundation for systematic, jurisdiction-aware evaluation of LLMs in mental health, encouraging more responsible development in this domain.
>
---
#### [new 087] VietMed-MCQ: A Consistency-Filtered Data Synthesis Framework for Vietnamese Traditional Medicine Evaluation
- **分类: cs.CL**

- **简介: 该论文属于医疗问答任务，旨在解决越南传统医学数据稀缺问题。通过构建 VietMed-MCQ 数据集，采用 RAG 管道和一致性验证机制，提升模型在该领域的表现。**

- **链接: [https://arxiv.org/pdf/2601.03792v1](https://arxiv.org/pdf/2601.03792v1)**

> **作者:** Huynh Trung Kiet; Dao Sy Duy Minh; Nguyen Dinh Ha Duong; Le Hoang Minh Huy; Long Nguyen; Dien Dinh
>
> **备注:** 11 pages, 4 figures. Dataset and code released
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable proficiency in general medical domains. However, their performance significantly degrades in specialized, culturally specific domains such as Vietnamese Traditional Medicine (VTM), primarily due to the scarcity of high-quality, structured benchmarks. In this paper, we introduce VietMed-MCQ, a novel multiple-choice question dataset generated via a Retrieval-Augmented Generation (RAG) pipeline with an automated consistency check mechanism. Unlike previous synthetic datasets, our framework incorporates a dual-model validation approach to ensure reasoning consistency through independent answer verification, though the substring-based evidence checking has known limitations. The complete dataset of 3,190 questions spans three difficulty levels and underwent validation by one medical expert and four students, achieving 94.2 percent approval with substantial inter-rater agreement (Fleiss' kappa = 0.82). We benchmark seven open-source models on VietMed-MCQ. Results reveal that general-purpose models with strong Chinese priors outperform Vietnamese-centric models, highlighting cross-lingual conceptual transfer, while all models still struggle with complex diagnostic reasoning. Our code and dataset are publicly available to foster research in low-resource medical domains.
>
---
#### [new 088] Doc-PP: Document Policy Preservation Benchmark for Large Vision-Language Models
- **分类: cs.CL**

- **简介: 该论文属于文档问答任务，解决LVLM在遵循用户定义的保密政策时的信息泄露问题。构建了Doc-PP基准，并提出DVA框架以提升安全性。**

- **链接: [https://arxiv.org/pdf/2601.03926v1](https://arxiv.org/pdf/2601.03926v1)**

> **作者:** Haeun Jang; Hwan Chang; Hwanhee Lee
>
> **摘要:** The deployment of Large Vision-Language Models (LVLMs) for real-world document question answering is often constrained by dynamic, user-defined policies that dictate information disclosure based on context. While ensuring adherence to these explicit constraints is critical, existing safety research primarily focuses on implicit social norms or text-only settings, overlooking the complexities of multimodal documents. In this paper, we introduce Doc-PP (Document Policy Preservation Benchmark), a novel benchmark constructed from real-world reports requiring reasoning across heterogeneous visual and textual elements under strict non-disclosure policies. Our evaluation highlights a systemic Reasoning-Induced Safety Gap: models frequently leak sensitive information when answers must be inferred through complex synthesis or aggregated across modalities, effectively circumventing existing safety constraints. Furthermore, we identify that providing extracted text improves perception but inadvertently facilitates leakage. To address these vulnerabilities, we propose DVA (Decompose-Verify-Aggregation), a structural inference framework that decouples reasoning from policy verification. Experimental results demonstrate that DVA significantly outperforms standard prompting defenses, offering a robust baseline for policy-compliant document understanding
>
---
#### [new 089] Less is more: Not all samples are effective for evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型评估任务，旨在解决专业领域基准测试中冗余样本多、计算成本高的问题。通过无历史数据的压缩框架，有效降低评估成本。**

- **链接: [https://arxiv.org/pdf/2601.03272v1](https://arxiv.org/pdf/2601.03272v1)**

> **作者:** Wentang Song; Jinqiang Li; Kele Huang; Junhui Lin; Shengxiang Wu; Zhongshi Xie
>
> **摘要:** The versatility of Large Language Models (LLMs) in vertical domains has spurred the development of numerous specialized evaluation benchmarks. However, these benchmarks often suffer from significant semantic redundancy and impose high computational costs during evaluation. Existing compression methods, such as tinyBenchmarks depend critically on correctness labels from multiple historical models evaluated on the full test set, making them inapplicable in cold-start scenarios, such as the introduction of a new task, domain, or model with no prior evaluation history. To address this limitation, we propose a history-free test set compression framework that requires no prior model performance data. Our method begins by fine-tuning a base LLM on a small amount of domain-specific data to internalize task-relevant semantics. It then generates high-level semantic embeddings for all original test samples using only their raw textual content. In this domain-adapted embedding space, we perform task-aware clustering and introduce a novel dataset X-ray mechanism that analyzes cluster geometry to dynamically calibrate the compression intensity based on the intrinsic redundancy of the benchmark. Experiments on professional-domain dataset, notably a large-scale 3GPP communications benchmark, demonstrate that our approach effectively identifies and removes redundant samples, reducing evaluation cost by over 90% while preserving high fidelity to the full benchmark.
>
---
#### [new 090] LLM_annotate: A Python package for annotating and analyzing fiction characters
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍LLM_annotate工具，用于小说角色性格分析。解决角色行为标注与特征推断问题，通过文本分块、模型标注、质量评估等方法实现高效分析。**

- **链接: [https://arxiv.org/pdf/2601.03274v1](https://arxiv.org/pdf/2601.03274v1)**

> **作者:** Hannes Rosenbusch
>
> **摘要:** LLM_annotate is a Python package for analyzing the personality of fiction characters with large language models. It standardizes workflows for annotating character behaviors in full texts (e.g., books and movie scripts), inferring character traits, and validating annotation/inference quality via a human-in-the-loop GUI. The package includes functions for text chunking, LLM-based annotation, character name disambiguation, quality scoring, and computation of character-level statistics and embeddings. Researchers can use any LLM, commercial, open-source, or custom, within LLM_annotate. Through tutorial examples using The Simpsons Movie and the novel Pride and Prejudice, I demonstrate the usage of the package for efficient and reproducible character analyses.
>
---
#### [new 091] ELO: Efficient Layer-Specific Optimization for Continual Pretraining of Multilingual LLMs
- **分类: cs.CL**

- **简介: 该论文属于多语言大模型持续预训练任务，旨在解决传统方法计算成本高和源语言性能下降的问题。提出ELO方法，通过分层优化提升目标语言性能并保持源语言能力。**

- **链接: [https://arxiv.org/pdf/2601.03648v1](https://arxiv.org/pdf/2601.03648v1)**

> **作者:** HanGyeol Yoo; ChangSu Choi; Minjun Kim; Seohyun Song; SeungWoo Song; Inho Won; Jongyoul Park; Cheoneum Park; KyungTae Lim
>
> **备注:** 12 pages, Accepted to EACL 2026 (Industrial Track)
>
> **摘要:** We propose an efficient layer-specific optimization (ELO) method designed to enhance continual pretraining (CP) for specific languages in multilingual large language models (MLLMs). This approach addresses the common challenges of high computational cost and degradation of source language performance associated with traditional CP. The ELO method consists of two main stages: (1) ELO Pretraining, where a small subset of specific layers, identified in our experiments as the critically important first and last layers, are detached from the original MLLM and trained with the target language. This significantly reduces not only the number of trainable parameters but also the total parameters computed during the forward pass, minimizing GPU memory consumption and accelerating the training process. (2) Layer Alignment, where the newly trained layers are reintegrated into the original model, followed by a brief full fine-tuning step on a small dataset to align the parameters. Experimental results demonstrate that the ELO method achieves a training speedup of up to 6.46 times compared to existing methods, while improving target language performance by up to 6.2\% on qualitative benchmarks and effectively preserving source language (English) capabilities.
>
---
#### [new 092] Whose Facts Win? LLM Source Preferences under Knowledge Conflicts
- **分类: cs.CL**

- **简介: 该论文研究LLM在知识冲突下对信息源的偏好，属于自然语言处理中的可信度研究任务。通过实验发现LLM更信任权威来源，但偏好可被重复信息改变。提出方法减少重复偏差，保持大部分原始偏好。**

- **链接: [https://arxiv.org/pdf/2601.03746v1](https://arxiv.org/pdf/2601.03746v1)**

> **作者:** Jakob Schuster; Vagrant Gautam; Katja Markert
>
> **备注:** Data and code: https://github.com/JaSchuste/llm-source-preference
>
> **摘要:** As large language models (LLMs) are more frequently used in retrieval-augmented generation pipelines, it is increasingly relevant to study their behavior under knowledge conflicts. Thus far, the role of the source of the retrieved information has gone unexamined. We address this gap with a novel framework to investigate how source preferences affect LLM resolution of inter-context knowledge conflicts in English, motivated by interdisciplinary research on credibility. With a comprehensive, tightly-controlled evaluation of 13 open-weight LLMs, we find that LLMs prefer institutionally-corroborated information (e.g., government or newspaper sources) over information from people and social media. However, these source preferences can be reversed by simply repeating information from less credible sources. To mitigate repetition effects and maintain consistent preferences, we propose a novel method that reduces repetition bias by up to 99.8%, while also maintaining at least 88.8% of original preferences. We release all data and code to encourage future work on credibility and source preferences in knowledge-intensive NLP.
>
---
#### [new 093] Mem-Gallery: Benchmarking Multimodal Long-Term Conversational Memory for MLLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态语言模型任务，旨在解决长期对话记忆评估问题。提出Mem-Gallery基准，评估模型在多模态场景下的记忆保持与管理能力。**

- **链接: [https://arxiv.org/pdf/2601.03515v1](https://arxiv.org/pdf/2601.03515v1)**

> **作者:** Yuanchen Bei; Tianxin Wei; Xuying Ning; Yanjun Zhao; Zhining Liu; Xiao Lin; Yada Zhu; Hendrik Hamann; Jingrui He; Hanghang Tong
>
> **备注:** 34 pages, 18 figures
>
> **摘要:** Long-term memory is a critical capability for multimodal large language model (MLLM) agents, particularly in conversational settings where information accumulates and evolves over time. However, existing benchmarks either evaluate multi-session memory in text-only conversations or assess multimodal understanding within localized contexts, failing to evaluate how multimodal memory is preserved, organized, and evolved across long-term conversational trajectories. Thus, we introduce Mem-Gallery, a new benchmark for evaluating multimodal long-term conversational memory in MLLM agents. Mem-Gallery features high-quality multi-session conversations grounded in both visual and textual information, with long interaction horizons and rich multimodal dependencies. Building on this dataset, we propose a systematic evaluation framework that assesses key memory capabilities along three functional dimensions: memory extraction and test-time adaptation, memory reasoning, and memory knowledge management. Extensive benchmarking across thirteen memory systems reveals several key findings, highlighting the necessity of explicit multimodal information retention and memory organization, the persistent limitations in memory reasoning and knowledge management, as well as the efficiency bottleneck of current models.
>
---
#### [new 094] Bare-Metal Tensor Virtualization: Overcoming the Memory Wall in Edge-AI Inference on ARM64
- **分类: cs.CL; cs.AI; cs.AR; cs.LG**

- **简介: 该论文属于边缘AI推理任务，旨在解决内存墙问题。通过软件虚拟张量核心和直接内存映射，提升ARM64设备的推理效率。**

- **链接: [https://arxiv.org/pdf/2601.03324v1](https://arxiv.org/pdf/2601.03324v1)**

> **作者:** Bugra Kilictas; Faruk Alpay
>
> **备注:** 14 pages, 2 figures. Code and data available at https://github.com/farukalpay/stories100m
>
> **摘要:** The deployment of Large Language Models (LLMs) on edge devices is fundamentally constrained by the "Memory Wall" the bottleneck where data movement latency outstrips arithmetic throughput. Standard inference runtimes often incur significant overhead through high-level abstractions, dynamic dispatch, and unaligned memory access patterns. In this work, we present a novel "Virtual Tensor Core" architecture implemented in software, optimized specifically for ARM64 microarchitectures (Apple Silicon). By bypassing standard library containers in favor of direct memory mapping (mmap) and implementing hand-tuned NEON SIMD kernels, we achieve a form of "Software-Defined Direct Memory Access (DMA)." Our proposed Tensor Virtualization Layout (TVL) guarantees 100% cache line utilization for weight matrices, while our zero-copy loader eliminates initialization latency. Experimental results on a 110M parameter model demonstrate a stable throughput of >60 tokens/second on M2 hardware. While proprietary hardware accelerators (e.g., Apple AMX) can achieve higher peak throughput, our architecture provides a fully open, portable, and deterministic reference implementation for studying the memory bottleneck on general-purpose ARM silicon, meeting the 200ms psycholinguistic latency threshold without opaque dependencies.
>
---
#### [new 095] From Implicit to Explicit: Token-Efficient Logical Supervision for Mathematical Reasoning in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于数学推理任务，旨在解决大模型逻辑推理能力不足的问题。通过提出FSLR框架，直接训练模型理解逻辑关系，提升推理准确性并提高训练效率。**

- **链接: [https://arxiv.org/pdf/2601.03682v1](https://arxiv.org/pdf/2601.03682v1)**

> **作者:** Shaojie Wang; Liang Zhang
>
> **摘要:** Recent studies reveal that large language models (LLMs) exhibit limited logical reasoning abilities in mathematical problem-solving, instead often relying on pattern-matching and memorization. We systematically analyze this limitation, focusing on logical relationship understanding, which is a core capability underlying genuine logical reasoning, and reveal that errors related to this capability account for over 90\% of incorrect predictions, with Chain-of-Thought Supervised Fine-Tuning (CoT-SFT) failing to substantially reduce these errors. To address this bottleneck, we propose First-Step Logical Reasoning (FSLR), a lightweight training framework targeting logical relationship understanding. Our key insight is that the first planning step-identifying which variables to use and which operation to apply-encourages the model to derive logical relationships directly from the problem statement. By training models on this isolated step, FSLR provides explicit supervision for logical relationship understanding, unlike CoT-SFT which implicitly embeds such relationships within complete solution trajectories. Extensive experiments across multiple models and datasets demonstrate that FSLR consistently outperforms CoT-SFT under both in-distribution and out-of-distribution settings, with average improvements of 3.2\% and 4.6\%, respectively. Moreover, FSLR achieves 4-6x faster training and reduces training token consumption by over 80\%.
>
---
#### [new 096] Agent-Dice: Disentangling Knowledge Updates via Geometric Consensus for Agent Continual Learning
- **分类: cs.CL**

- **简介: 该论文属于持续学习任务，旨在解决代理在学习新任务时的灾难性遗忘问题。提出Agent-Dice框架，通过知识解耦提升模型稳定性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.03641v1](https://arxiv.org/pdf/2601.03641v1)**

> **作者:** Zheng Wu; Xingyu Lou; Xinbei Ma; Yansi Li; Weiwen Liu; Weinan Zhang; Jun Wang; Zhuosheng Zhang
>
> **摘要:** Large Language Model (LLM)-based agents significantly extend the utility of LLMs by interacting with dynamic environments. However, enabling agents to continually learn new tasks without catastrophic forgetting remains a critical challenge, known as the stability-plasticity dilemma. In this work, we argue that this dilemma fundamentally arises from the failure to explicitly distinguish between common knowledge shared across tasks and conflicting knowledge introduced by task-specific interference. To address this, we propose Agent-Dice, a parameter fusion framework based on directional consensus evaluation. Concretely, Agent-Dice disentangles knowledge updates through a two-stage process: geometric consensus filtering to prune conflicting gradients, and curvature-based importance weighting to amplify shared semantics. We provide a rigorous theoretical analysis that establishes the validity of the proposed fusion scheme and offers insight into the origins of the stability-plasticity dilemma. Extensive experiments on GUI agents and tool-use agent domains demonstrate that Agent-Dice exhibits outstanding continual learning performance with minimal computational overhead and parameter updates.
>
---
#### [new 097] All That Glisters Is Not Gold: A Benchmark for Reference-Free Counterfactual Financial Misinformation Detection
- **分类: cs.CL; cs.CE; q-fin.CP**

- **简介: 该论文属于金融谣言检测任务，旨在解决无参考的虚假信息识别问题。提出RFC Bench基准，通过对比分析提升检测效果，揭示当前模型在无外部依据时的不足。**

- **链接: [https://arxiv.org/pdf/2601.04160v1](https://arxiv.org/pdf/2601.04160v1)**

> **作者:** Yuechen Jiang; Zhiwei Liu; Yupeng Cao; Yueru He; Ziyang Xu; Chen Xu; Zhiyang Deng; Prayag Tiwari; Xi Chen; Alejandro Lopez-Lira; Jimin Huang; Junichi Tsujii; Sophia Ananiadou
>
> **备注:** 39 pages; 24 figures
>
> **摘要:** We introduce RFC Bench, a benchmark for evaluating large language models on financial misinformation under realistic news. RFC Bench operates at the paragraph level and captures the contextual complexity of financial news where meaning emerges from dispersed cues. The benchmark defines two complementary tasks: reference free misinformation detection and comparison based diagnosis using paired original perturbed inputs. Experiments reveal a consistent pattern: performance is substantially stronger when comparative context is available, while reference free settings expose significant weaknesses, including unstable predictions and elevated invalid outputs. These results indicate that current models struggle to maintain coherent belief states without external grounding. By highlighting this gap, RFC Bench provides a structured testbed for studying reference free reasoning and advancing more reliable financial misinformation detection in real world settings.
>
---
#### [new 098] Evaluating the Pre-Consultation Ability of LLMs using Diagnostic Guidelines
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗对话评估任务，旨在解决LLMs在预诊能力上的评价问题。通过构建基准数据集EPAG，对比诊断指南并进行疾病诊断实验，验证模型性能。**

- **链接: [https://arxiv.org/pdf/2601.03627v1](https://arxiv.org/pdf/2601.03627v1)**

> **作者:** Jean Seo; Gibaeg Kim; Kihun Shin; Seungseop Lim; Hyunkyung Lee; Wooseok Han; Jongwon Lee; Eunho Yang
>
> **备注:** EACL 2026 Industry
>
> **摘要:** We introduce EPAG, a benchmark dataset and framework designed for Evaluating the Pre-consultation Ability of LLMs using diagnostic Guidelines. LLMs are evaluated directly through HPI-diagnostic guideline comparison and indirectly through disease diagnosis. In our experiments, we observe that small open-source models fine-tuned with a well-curated, task-specific dataset can outperform frontier LLMs in pre-consultation. Additionally, we find that increased amount of HPI (History of Present Illness) does not necessarily lead to improved diagnostic performance. Further experiments reveal that the language of pre-consultation influences the characteristics of the dialogue. By open-sourcing our dataset and evaluation pipeline on https://github.com/seemdog/EPAG, we aim to contribute to the evaluation and further development of LLM applications in real-world clinical settings.
>
---
#### [new 099] e5-omni: Explicit Cross-modal Alignment for Omni-modal Embeddings
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态嵌入任务，旨在解决跨模态对齐中的尺度不一致、负样本失效和统计不匹配问题。提出e5-omni模型，通过温度校准、负样本课程和批量白化提升对齐效果。**

- **链接: [https://arxiv.org/pdf/2601.03666v1](https://arxiv.org/pdf/2601.03666v1)**

> **作者:** Haonan Chen; Sicheng Gao; Radu Timofte; Tetsuya Sakai; Zhicheng Dou
>
> **摘要:** Modern information systems often involve different types of items, e.g., a text query, an image, a video clip, or an audio segment. This motivates omni-modal embedding models that map heterogeneous modalities into a shared space for direct comparison. However, most recent omni-modal embeddings still rely heavily on implicit alignment inherited from pretrained vision-language model (VLM) backbones. In practice, this causes three common issues: (i) similarity logits have modality-dependent sharpness, so scores are not on a consistent scale; (ii) in-batch negatives become less effective over time because mixed-modality batches create an imbalanced hardness distribution; as a result, many negatives quickly become trivial and contribute little gradient; and (iii) embeddings across modalities show mismatched first- and second-order statistics, which makes rankings less stable. To tackle these problems, we propose e5-omni, a lightweight explicit alignment recipe that adapts off-the-shelf VLMs into robust omni-modal embedding models. e5-omni combines three simple components: (1) modality-aware temperature calibration to align similarity scales, (2) a controllable negative curriculum with debiasing to focus on confusing negatives while reducing the impact of false negatives, and (3) batch whitening with covariance regularization to better match cross-modal geometry in the shared embedding space. Experiments on MMEB-V2 and AudioCaps show consistent gains over strong bi-modal and omni-modal baselines, and the same recipe also transfers well to other VLM backbones. We release our model checkpoint at https://huggingface.co/Haon-Chen/e5-omni-7B.
>
---
#### [new 100] Rendering Data Unlearnable by Exploiting LLM Alignment Mechanisms
- **分类: cs.CL**

- **简介: 该论文属于数据保护任务，旨在防止LLM学习敏感数据。通过注入特定免责声明，利用模型对齐机制阻止有效学习，实现数据不可学性。**

- **链接: [https://arxiv.org/pdf/2601.03401v1](https://arxiv.org/pdf/2601.03401v1)**

> **作者:** Ruihan Zhang; Jun Sun
>
> **摘要:** Large language models (LLMs) are increasingly trained on massive, heterogeneous text corpora, raising serious concerns about the unauthorised use of proprietary or personal data during model training. In this work, we address the problem of data protection against unwanted model learning in a realistic black-box setting. We propose Disclaimer Injection, a novel data-level defence that renders text unlearnable to LLMs. Rather than relying on model-side controls or explicit data removal, our approach exploits the models' own alignment mechanisms: by injecting carefully designed alignment-triggering disclaimers to prevent effective learning. Through layer-wise analysis, we find that fine-tuning on such protected data induces persistent activation of alignment-related layers, causing alignment constraints to override task learning even on common inputs. Consequently, models trained on such data exhibit substantial and systematic performance degradation compared to standard fine-tuning. Our results identify alignment behaviour as a previously unexplored lever for data protection and, to our knowledge, present the first practical method for restricting data learnability at LLM scale without requiring access to or modification of the training pipeline.
>
---
#### [new 101] Submodular Evaluation Subset Selection in Automatic Prompt Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自动提示优化任务，解决评估子集选择问题。提出SESS方法，通过子模函数最大化提升提示优化效果。**

- **链接: [https://arxiv.org/pdf/2601.03493v1](https://arxiv.org/pdf/2601.03493v1)**

> **作者:** Jinming Nian; Zhiyuan Peng; Hongwei Shang; Dae Hoon Park; Yi Fang
>
> **摘要:** Automatic prompt optimization reduces manual prompt engineering, but relies on task performance measured on a small, often randomly sampled evaluation subset as its main source of feedback signal. Despite this, how to select that evaluation subset is usually treated as an implementation detail. We study evaluation subset selection for prompt optimization from a principled perspective and propose SESS, a submodular evaluation subset selection method. We frame selection as maximizing an objective set function and show that, under mild conditions, it is monotone and submodular, enabling greedy selection with theoretical guarantees. Across GSM8K, MATH, and GPQA-Diamond, submodularly selected evaluation subsets can yield better optimized prompts than random or heuristic baselines.
>
---
#### [new 102] The Critical Role of Aspects in Measuring Document Similarity
- **分类: cs.CL**

- **简介: 该论文属于文档相似度测量任务，旨在解决传统方法忽略具体方面的问题。提出ASPECTSIM框架，通过指定方面提升相似度评估的准确性。**

- **链接: [https://arxiv.org/pdf/2601.03435v1](https://arxiv.org/pdf/2601.03435v1)**

> **作者:** Eftekhar Hossain; Tarnika Hazra; Ahatesham Bhuiyan; Santu Karmaker
>
> **备注:** 24 Pages, 10 Figures, 10 Tables
>
> **摘要:** We introduce ASPECTSIM, a simple and interpretable framework that requires conditioning document similarity on an explicitly specified aspect, which is different from the traditional holistic approach in measuring document similarity. Experimenting with a newly constructed benchmark of 26K aspect-document pairs, we found that ASPECTSIM, when implemented with direct GPT-4o prompting, achieves substantially higher human-machine agreement ($\approx$80% higher) than the same for holistic similarity without explicit aspects. These findings underscore the importance of explicitly accounting for aspects when measuring document similarity and highlight the need to revise standard practice. Next, we conducted a large-scale meta-evaluation using 16 smaller open-source LLMs and 9 embedding models with a focus on making ASPECTSIM accessible and reproducible. While directly prompting LLMs to produce ASPECTSIM scores turned out be ineffective (20-30% human-machine agreement), a simple two-stage refinement improved their agreement by $\approx$140%. Nevertheless, agreement remains well below that of GPT-4o-based models, indicating that smaller open-source LLMs still lag behind large proprietary models in capturing aspect-conditioned similarity.
>
---
#### [new 103] DiVA: Fine-grained Factuality Verification with Agentic-Discriminative Verifier
- **分类: cs.CL**

- **简介: 该论文属于事实性验证任务，旨在解决现有方法无法区分错误严重程度的问题。提出DiVA框架，结合生成模型的搜索能力和判别模型的评分能力，提升细粒度验证效果。**

- **链接: [https://arxiv.org/pdf/2601.03605v1](https://arxiv.org/pdf/2601.03605v1)**

> **作者:** Hui Huang; Muyun Yang; Yuki Arase
>
> **摘要:** Despite the significant advancements of Large Language Models (LLMs), their factuality remains a critical challenge, fueling growing interest in factuality verification. Existing research on factuality verification primarily conducts binary judgments (e.g., correct or incorrect), which fails to distinguish varying degrees of error severity. This limits its utility for applications such as fine-grained evaluation and preference optimization. To bridge this gap, we propose the Agentic Discriminative Verifier (DiVA), a hybrid framework that synergizes the agentic search capabilities of generative models with the precise scoring aptitude of discriminative models. We also construct a new benchmark, FGVeriBench, as a robust testbed for fine-grained factuality verification. Experimental results on FGVeriBench demonstrate that our DiVA significantly outperforms existing methods on factuality verification for both general and multi-hop questions.
>
---
#### [new 104] Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在提升语言模型的语法能力。通过引入语言学习任务，增强模型的语言素养，同时保持推理能力。**

- **链接: [https://arxiv.org/pdf/2601.03448v1](https://arxiv.org/pdf/2601.03448v1)**

> **作者:** Atsuki Yamaguchi; Maggie Mi; Nikolaos Aletras
>
> **摘要:** Language models (LMs) are pre-trained on raw text datasets to generate text sequences token-by-token. While this approach facilitates the learning of world knowledge and reasoning, it does not explicitly optimize for linguistic competence. To bridge this gap, we propose L2T, a pre-training framework integrating Language Learning Tasks alongside standard next-token prediction. Inspired by human language acquisition, L2T transforms raw text into structured input-output pairs to provide explicit linguistic stimulation. Pre-training LMs on a mixture of raw text and L2T data not only improves overall performance on linguistic competence benchmarks but accelerates its acquisition, while maintaining competitive performance on general reasoning tasks.
>
---
#### [new 105] GuardEval: A Multi-Perspective Benchmark for Evaluating Safety, Fairness, and Robustness in LLM Moderators
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于内容安全任务，旨在解决LLM在 moderation 中的安全性、公平性和鲁棒性问题。提出GuardEval基准和GGuard模型，提升对复杂敏感内容的识别能力。**

- **链接: [https://arxiv.org/pdf/2601.03273v1](https://arxiv.org/pdf/2601.03273v1)**

> **作者:** Naseem Machlovi; Maryam Saleki; Ruhul Amin; Mohamed Rahouti; Shawqi Al-Maliki; Junaid Qadir; Mohamed M. Abdallah; Ala Al-Fuqaha
>
> **摘要:** As large language models (LLMs) become deeply embedded in daily life, the urgent need for safer moderation systems, distinguishing between naive from harmful requests while upholding appropriate censorship boundaries, has never been greater. While existing LLMs can detect harmful or unsafe content, they often struggle with nuanced cases such as implicit offensiveness, subtle gender and racial biases, and jailbreak prompts, due to the subjective and context-dependent nature of these issues. Furthermore, their heavy reliance on training data can reinforce societal biases, resulting in inconsistent and ethically problematic outputs. To address these challenges, we introduce GuardEval, a unified multi-perspective benchmark dataset designed for both training and evaluation, containing 106 fine-grained categories spanning human emotions, offensive and hateful language, gender and racial bias, and broader safety concerns. We also present GemmaGuard (GGuard), a QLoRA fine-tuned version of Gemma3-12B trained on GuardEval, to assess content moderation with fine-grained labels. Our evaluation shows that GGuard achieves a macro F1 score of 0.832, substantially outperforming leading moderation models, including OpenAI Moderator (0.64) and Llama Guard (0.61). We show that multi-perspective, human-centered safety benchmarks are critical for reducing biased and inconsistent moderation decisions. GuardEval and GGuard together demonstrate that diverse, representative data materially improve safety, fairness, and robustness on complex, borderline cases.
>
---
#### [new 106] AI Generated Text Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI生成文本检测任务，旨在解决学术不端问题。通过对比传统和深度学习模型，评估不同方法的检测效果。**

- **链接: [https://arxiv.org/pdf/2601.03812v1](https://arxiv.org/pdf/2601.03812v1)**

> **作者:** Adilkhan Alikhanov; Aidar Amangeldi; Diar Demeubay; Dilnaz Akhmetzhan; Nurbek Moldakhmetov; Omar Polat; Galymzhan Zharas
>
> **摘要:** The rapid development of large language models has led to an increase in AI-generated text, with students increasingly using LLM-generated content as their own work, which violates academic integrity. This paper presents an evaluation of AI text detection methods, including both traditional machine learning models and transformer-based architectures. We utilize two datasets, HC3 and DAIGT v2, to build a unified benchmark and apply a topic-based data split to prevent information leakage. This approach ensures robust generalization across unseen domains. Our experiments show that TF-IDF logistic regression achieves a reasonable baseline accuracy of 82.87%. However, deep learning models outperform it. The BiLSTM classifier achieves an accuracy of 88.86%, while DistilBERT achieves a similar accuracy of 88.11% with the highest ROC-AUC score of 0.96, demonstrating the strongest overall performance. The results indicate that contextual semantic modeling is significantly superior to lexical features and highlight the importance of mitigating topic memorization through appropriate evaluation protocols. The limitations of this work are primarily related to dataset diversity and computational constraints. In future work, we plan to expand dataset diversity and utilize parameter-efficient fine-tuning methods such as LoRA. We also plan to explore smaller or distilled models and employ more efficient batching strategies and hardware-aware optimization.
>
---
#### [new 107] Beyond Perplexity: A Lightweight Benchmark for Knowledge Retention in Supervised Fine-Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的模型微调任务，旨在解决验证困惑度无法区分事实学习与语言模仿的问题。提出KR-Test框架，通过对比示例评估知识保留情况。**

- **链接: [https://arxiv.org/pdf/2601.03505v1](https://arxiv.org/pdf/2601.03505v1)**

> **作者:** Soheil Zibakhsh Shabgahi; Pedram Aghazadeh; Farinaz Koushanfar
>
> **摘要:** Supervised Fine-Tuning (SFT) is a standard approach for injecting domain knowledge into Large Language Models (LLMs). However, relying on validation perplexity to monitor training is often insufficient, as it confounds stylistic mimicry with genuine factual internalization. To address this, we introduce the Knowledge Retention (KR) Test , a lightweight, corpus-grounded evaluation framework designed to distinguish factual learning from linguistics. KR-Test utilizes automatically generated contrastive examples to measure likelihood preferences for correct versus incorrect continuations, requiring no instruction tuning or generative decoding. We validate the framework's integrity through a "blind vs. oracle" baseline analysis. Furthermore, we demonstrate the diagnostic capabilities of KR-Test by analyzing the training dynamics of Low-Rank Adaptation (LoRA). By exposing the fine-grained dissociation between linguistic convergence and knowledge retention, KR-Test enhances the interpretability of fine-tuning dynamics.
>
---
#### [new 108] Internal Reasoning vs. External Control: A Thermodynamic Analysis of Sycophancy in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型中的谄媚现象，比较内部推理与外部控制的效果，旨在提升模型的安全性与准确性。任务为模型安全分析，解决如何减少谄媚行为的问题，通过实验验证外部控制更有效。**

- **链接: [https://arxiv.org/pdf/2601.03263v1](https://arxiv.org/pdf/2601.03263v1)**

> **作者:** Edward Y. Chang
>
> **备注:** 15 pages, 4 figures, 11 tables
>
> **摘要:** Large Language Models frequently exhibit sycophancy, prioritizing user agreeableness over correctness. We investigate whether this requires external regulation or can be mitigated by internal reasoning alone. Using CAP-GSM8K (N=500), an adversarial dataset, we evaluate internal (CoT) versus external (RCA) mechanisms across GPT-3.5, GPT-4o, and GPT-5.1. Our results reveal the structural limits of internal reasoning: it causes performance collapse in weak models (the Prioritization Paradox) and leaves an 11.4\% final output gap in frontier models. In contrast, RCA structurally eliminates sycophancy (0.0\%) across all tiers. We synthesize these findings into a thermodynamic hierarchy: hybrid systems achieve Resonance (optimal efficiency) only when capabilities are matched and strong, while weak or mismatched pairs succumb to Dissonance and Entropy. This confirms that external structural constraints are strictly necessary to guarantee safety.
>
---
#### [new 109] Analyzing Reasoning Consistency in Large Multimodal Models under Cross-Modal Conflicts
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究大模型在跨模态冲突下的推理一致性问题，针对文本幻觉导致的错误传播提出解决方案。**

- **链接: [https://arxiv.org/pdf/2601.04073v1](https://arxiv.org/pdf/2601.04073v1)**

> **作者:** Zhihao Zhu; Jiafeng Liang; Shixin Jiang; Jinlan Fu; Ming Liu; Guanglu Sun; See-Kiong Ng; Bing Qin
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Large Multimodal Models (LMMs) have demonstrated impressive capabilities in video reasoning via Chain-of-Thought (CoT). However, the robustness of their reasoning chains remains questionable. In this paper, we identify a critical failure mode termed textual inertia, where once a textual hallucination occurs in the thinking process, models tend to blindly adhere to the erroneous text while neglecting conflicting visual evidence. To systematically investigate this, we propose the LogicGraph Perturbation Protocol that structurally injects perturbations into the reasoning chains of diverse LMMs spanning both native reasoning architectures and prompt-driven paradigms to evaluate their self-reflection capabilities. The results reveal that models successfully self-correct in less than 10% of cases and predominantly succumb to blind textual error propagation. To mitigate this, we introduce Active Visual-Context Refinement, a training-free inference paradigm which orchestrates an active visual re-grounding mechanism to enforce fine-grained verification coupled with an adaptive context refinement strategy to summarize and denoise the reasoning history. Experiments demonstrate that our approach significantly stifles hallucination propagation and enhances reasoning robustness.
>
---
#### [new 110] Sandwich Reasoning: An Answer-Reasoning-Answer Approach for Low-Latency Query Correction
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于查询修正任务，旨在解决实时性与准确性之间的矛盾。提出SandwichR方法，在低延迟下实现高精度的查询修正。**

- **链接: [https://arxiv.org/pdf/2601.03672v1](https://arxiv.org/pdf/2601.03672v1)**

> **作者:** Chen Zhang; Kepu Zhang; Jiatong Zhang; Xiao Zhang; Jun Xu
>
> **摘要:** Query correction is a critical entry point in modern search pipelines, demanding high accuracy strictly within real-time latency constraints. Chain-of-Thought (CoT) reasoning improves accuracy but incurs prohibitive latency for real-time query correction. A potential solution is to output an answer before reasoning to reduce latency; however, under autoregressive decoding, the early answer is independent of subsequent reasoning, preventing the model from leveraging its reasoning capability to improve accuracy. To address this issue, we propose Sandwich Reasoning (SandwichR), a novel approach that explicitly aligns a fast initial answer with post-hoc reasoning, enabling low-latency query correction without sacrificing reasoning-aware accuracy. SandwichR follows an Answer-Reasoning-Answer paradigm, producing an initial correction, an explicit reasoning process, and a final refined correction. To align the initial answer with post-reasoning insights, we design a consistency-aware reinforcement learning (RL) strategy: a dedicated consistency reward enforces alignment between the initial and final corrections, while margin-based rejection sampling prioritizes borderline samples where reasoning drives the most impactful corrective gains. Additionally, we construct a high-quality query correction dataset, addressing the lack of specialized benchmarks for complex query correction. Experimental results demonstrate that SandwichR achieves SOTA accuracy comparable to standard CoT while delivering a 40-70% latency reduction, resolving the latency-accuracy trade-off in online search.
>
---
#### [new 111] SoK: Privacy Risks and Mitigations in Retrieval-Augmented Generation Systems
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于隐私安全任务，旨在分析RAG系统的隐私风险及应对方法。通过系统综述，梳理风险、提出缓解措施并构建评估框架。**

- **链接: [https://arxiv.org/pdf/2601.03979v1](https://arxiv.org/pdf/2601.03979v1)**

> **作者:** Andreea-Elena Bodea; Stephen Meisenbacher; Alexandra Klymenko; Florian Matthes
>
> **备注:** 17 pages, 3 figures, 5 tables. This work has been accepted for publication at the IEEE Conference on Secure and Trustworthy Machine Learning (SaTML 2026). The final version will be available on IEEE Xplore
>
> **摘要:** The continued promise of Large Language Models (LLMs), particularly in their natural language understanding and generation capabilities, has driven a rapidly increasing interest in identifying and developing LLM use cases. In an effort to complement the ingrained "knowledge" of LLMs, Retrieval-Augmented Generation (RAG) techniques have become widely popular. At its core, RAG involves the coupling of LLMs with domain-specific knowledge bases, whereby the generation of a response to a user question is augmented with contextual and up-to-date information. The proliferation of RAG has sparked concerns about data privacy, particularly with the inherent risks that arise when leveraging databases with potentially sensitive information. Numerous recent works have explored various aspects of privacy risks in RAG systems, from adversarial attacks to proposed mitigations. With the goal of surveying and unifying these works, we ask one simple question: What are the privacy risks in RAG, and how can they be measured and mitigated? To answer this question, we conduct a systematic literature review of RAG works addressing privacy, and we systematize our findings into a comprehensive set of privacy risks, mitigation techniques, and evaluation strategies. We supplement these findings with two primary artifacts: a Taxonomy of RAG Privacy Risks and a RAG Privacy Process Diagram. Our work contributes to the study of privacy in RAG not only by conducting the first systematization of risks and mitigations, but also by uncovering important considerations when mitigating privacy risks in RAG systems and assessing the current maturity of proposed mitigations.
>
---
#### [new 112] Anti-Length Shift: Dynamic Outlier Truncation for Training Efficient Reasoning Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于高效推理模型训练任务，旨在解决模型在简单问题上过度冗长的问题。通过引入DOT方法，动态抑制冗余token，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.03969v1](https://arxiv.org/pdf/2601.03969v1)**

> **作者:** Wei Wu; Liyi Chen; Congxi Xiao; Tianfu Wang; Qimeng Wang; Chengqiang Lu; Yan Gao; Yi Wu; Yao Hu; Hui Xiong
>
> **摘要:** Large reasoning models enhanced by reinforcement learning with verifiable rewards have achieved significant performance gains by extending their chain-of-thought. However, this paradigm incurs substantial deployment costs as models often exhibit excessive verbosity on simple queries. Existing efficient reasoning methods relying on explicit length penalties often introduce optimization conflicts and leave the generative mechanisms driving overthinking largely unexamined. In this paper, we identify a phenomenon termed length shift where models increasingly generate unnecessary reasoning on trivial inputs during training. To address this, we introduce Dynamic Outlier Truncation (DOT), a training-time intervention that selectively suppresses redundant tokens. This method targets only the extreme tail of response lengths within fully correct rollout groups while preserving long-horizon reasoning capabilities for complex problems. To complement this intervention and ensure stable convergence, we further incorporate auxiliary KL regularization and predictive dynamic sampling. Experimental results across multiple model scales demonstrate that our approach significantly pushes the efficiency-performance Pareto frontier outward. Notably, on the AIME-24, our method reduces inference token usage by 78% while simultaneously increasing accuracy compared to the initial policy and surpassing state-of-the-art efficient reasoning methods.
>
---
#### [new 113] Stable Language Guidance for Vision-Language-Action Models
- **分类: cs.RO; cs.CL**

- **简介: 该论文属于视觉-语言-动作模型任务，解决语言扰动导致的模型脆弱问题。提出RSS框架，通过分离视觉与语义信息提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.04052v1](https://arxiv.org/pdf/2601.04052v1)**

> **作者:** Zhihao Zhan; Yuhao Chen; Jiaying Zhou; Qinhan Lv; Hao Liu; Keze Wang; Liang Lin; Guangrun Wang
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated impressive capabilities in generalized robotic control; however, they remain notoriously brittle to linguistic perturbations. We identify a critical ``modality collapse'' phenomenon where strong visual priors overwhelm sparse linguistic signals, causing agents to overfit to specific instruction phrasings while ignoring the underlying semantic intent. To address this, we propose \textbf{Residual Semantic Steering (RSS)}, a probabilistic framework that disentangles physical affordance from semantic execution. RSS introduces two theoretical innovations: (1) \textbf{Monte Carlo Syntactic Integration}, which approximates the true semantic posterior via dense, LLM-driven distributional expansion, and (2) \textbf{Residual Affordance Steering}, a dual-stream decoding mechanism that explicitly isolates the causal influence of language by subtracting the visual affordance prior. Theoretical analysis suggests that RSS effectively maximizes the mutual information between action and intent while suppressing visual distractors. Empirical results across diverse manipulation benchmarks demonstrate that RSS achieves state-of-the-art robustness, maintaining performance even under adversarial linguistic perturbations.
>
---
#### [new 114] STELLA: Self-Reflective Terminology-Aware Framework for Building an Aerospace Information Retrieval Benchmark
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决航空航天领域缺乏专用基准的问题。提出STELLA框架及基准，通过系统化方法构建包含术语匹配和语义匹配的查询数据集，评估嵌入模型性能。**

- **链接: [https://arxiv.org/pdf/2601.03496v1](https://arxiv.org/pdf/2601.03496v1)**

> **作者:** Bongmin Kim
>
> **备注:** 25 pages, 2 figures
>
> **摘要:** Tasks in the aerospace industry heavily rely on searching and reusing large volumes of technical documents, yet there is no public information retrieval (IR) benchmark that reflects the terminology- and query-intent characteristics of this domain. To address this gap, this paper proposes the STELLA (Self-Reflective TErminoLogy-Aware Framework for BuiLding an Aerospace Information Retrieval Benchmark) framework. Using this framework, we introduce the STELLA benchmark, an aerospace-specific IR evaluation set constructed from NASA Technical Reports Server (NTRS) documents via a systematic pipeline that comprises document layout detection, passage chunking, terminology dictionary construction, synthetic query generation, and cross-lingual extension. The framework generates two types of queries: the Terminology Concordant Query (TCQ), which includes the terminology verbatim to evaluate lexical matching, and the Terminology Agnostic Query (TAQ), which utilizes the terminology's description to assess semantic matching. This enables a disentangled evaluation of the lexical and semantic matching capabilities of embedding models. In addition, we combine Chain-of-Density (CoD) and the Self-Reflection method with query generation to improve quality and implement a hybrid cross-lingual extension that reflects real user querying practices. Evaluation of seven embedding models on the STELLA benchmark shows that large decoder-based embedding models exhibit the strongest semantic understanding, while lexical matching methods such as BM25 remain highly competitive in domains where exact lexical matching technical term is crucial. The STELLA benchmark provides a reproducible foundation for reliable performance evaluation and improvement of embedding models in aerospace-domain IR tasks. The STELLA benchmark can be found in https://huggingface.co/datasets/telepix/STELLA.
>
---
#### [new 115] FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于持续学习任务，旨在解决大语言模型的灾难性遗忘问题。提出FOREVER框架，基于遗忘曲线调整记忆回放策略，提升模型持续学习能力。**

- **链接: [https://arxiv.org/pdf/2601.03938v1](https://arxiv.org/pdf/2601.03938v1)**

> **作者:** Yujie Feng; Hao Wang; Jian Li; Xu Chu; Zhaolu Kang; Yiran Liu; Yasha Wang; Philip S. Yu; Xiao-Ming Wu
>
> **摘要:** Continual learning (CL) for large language models (LLMs) aims to enable sequential knowledge acquisition without catastrophic forgetting. Memory replay methods are widely used for their practicality and effectiveness, but most rely on fixed, step-based heuristics that often misalign with the model's actual learning progress, since identical training steps can result in varying degrees of parameter change. Motivated by recent findings that LLM forgetting mirrors the Ebbinghaus human forgetting curve, we propose FOREVER (FORgEtting curVe-inspired mEmory Replay), a novel CL framework that aligns replay schedules with a model-centric notion of time. FOREVER defines model time using the magnitude of optimizer updates, allowing forgetting curve-inspired replay intervals to align with the model's internal evolution rather than raw training steps. Building on this approach, FOREVER incorporates a forgetting curve-based replay scheduler to determine when to replay and an intensity-aware regularization mechanism to adaptively control how to replay. Extensive experiments on three CL benchmarks and models ranging from 0.6B to 13B parameters demonstrate that FOREVER consistently mitigates catastrophic forgetting.
>
---
#### [new 116] Current Agents Fail to Leverage World Model as Tool for Foresight
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 论文研究当前智能体无法有效利用世界模型进行预测，属于增强智能体预见能力的任务。旨在解决智能体在使用生成式世界模型时存在的决策、解释和整合问题。通过实验发现其利用率低、误用率高，性能不稳定。**

- **链接: [https://arxiv.org/pdf/2601.03905v1](https://arxiv.org/pdf/2601.03905v1)**

> **作者:** Cheng Qian; Emre Can Acikgoz; Bingxuan Li; Xiusi Chen; Yuji Zhang; Bingxiang He; Qinyu Luo; Dilek Hakkani-Tür; Gokhan Tur; Yunzhu Li; Heng Ji; Heng Ji
>
> **备注:** 36 Pages, 13 Figures, 17 Tables
>
> **摘要:** Agents built on vision-language models increasingly face tasks that demand anticipating future states rather than relying on short-horizon reasoning. Generative world models offer a promising remedy: agents could use them as external simulators to foresee outcomes before acting. This paper empirically examines whether current agents can leverage such world models as tools to enhance their cognition. Across diverse agentic and visual question answering tasks, we observe that some agents rarely invoke simulation (fewer than 1%), frequently misuse predicted rollouts (approximately 15%), and often exhibit inconsistent or even degraded performance (up to 5%) when simulation is available or enforced. Attribution analysis further indicates that the primary bottleneck lies in the agents' capacity to decide when to simulate, how to interpret predicted outcomes, and how to integrate foresight into downstream reasoning. These findings underscore the need for mechanisms that foster calibrated, strategic interaction with world models, paving the way toward more reliable anticipatory cognition in future agent systems.
>
---
#### [new 117] How Real is Your Jailbreak? Fine-grained Jailbreak Evaluation with Anchored Reference
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于LLM安全任务，旨在解决 jailbreak 攻击评估不准确的问题。提出FJAR框架，通过细粒度分类和锚定参考提高评估精度。**

- **链接: [https://arxiv.org/pdf/2601.03288v1](https://arxiv.org/pdf/2601.03288v1)**

> **作者:** Songyang Liu; Chaozhuo Li; Rui Pu; Litian Zhang; Chenxu Wang; Zejian Chen; Yuting Zhang; Yiming Hei
>
> **备注:** 7 pages, 3 figures, preprint
>
> **摘要:** Jailbreak attacks present a significant challenge to the safety of Large Language Models (LLMs), yet current automated evaluation methods largely rely on coarse classifications that focus mainly on harmfulness, leading to substantial overestimation of attack success. To address this problem, we propose FJAR, a fine-grained jailbreak evaluation framework with anchored references. We first categorized jailbreak responses into five fine-grained categories: Rejective, Irrelevant, Unhelpful, Incorrect, and Successful, based on the degree to which the response addresses the malicious intent of the query. This categorization serves as the basis for FJAR. Then, we introduce a novel harmless tree decomposition approach to construct high-quality anchored references by breaking down the original queries. These references guide the evaluator in determining whether the response genuinely fulfills the original query. Extensive experiments demonstrate that FJAR achieves the highest alignment with human judgment and effectively identifies the root causes of jailbreak failures, providing actionable guidance for improving attack strategies.
>
---
#### [new 118] HyperCLOVA X 32B Think
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文介绍HyperCLOVA X 32B Think，一个专注于韩国语言文化及推理能力的多模态模型，解决跨模态理解和智能代理任务，通过预训练和微调提升性能。**

- **链接: [https://arxiv.org/pdf/2601.03286v1](https://arxiv.org/pdf/2601.03286v1)**

> **作者:** NAVER Cloud HyperCLOVA X Team
>
> **备注:** Technical Report
>
> **摘要:** In this report, we present HyperCLOVA X 32B Think, a vision-language model designed with particular emphasis on reasoning within the Korean linguistic and cultural context, as well as agentic ability. HyperCLOVA X 32B Think is pre-trained with a strong focus on reasoning capabilities and subsequently post-trained to support multimodal understanding, enhanced reasoning, agentic behaviors, and alignment with human preferences. Experimental evaluations against comparably sized models demonstrate that our model achieves strong performance on Korean text-to-text and vision-to-text benchmarks, as well as on agent-oriented evaluation tasks. By open-sourcing HyperCLOVA X 32B Think, we aim to support broader adoption and facilitate further research and innovation across both academic and industrial communities.
>
---
#### [new 119] STAR-S: Improving Safety Alignment through Self-Taught Reasoning on Safety Rules
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于安全对齐任务，旨在防御 jailbreak 攻击。通过自教推理框架 STAR-S，提升模型对安全规则的推理能力，增强安全性。**

- **链接: [https://arxiv.org/pdf/2601.03537v1](https://arxiv.org/pdf/2601.03537v1)**

> **作者:** Di Wu; Yanyan Zhao; Xin Lu; Mingzhe Li; Bing Qin
>
> **备注:** 19 pages,4 figures
>
> **摘要:** Defending against jailbreak attacks is crucial for the safe deployment of Large Language Models (LLMs). Recent research has attempted to improve safety by training models to reason over safety rules before responding. However, a key issue lies in determining what form of safety reasoning effectively defends against jailbreak attacks, which is difficult to explicitly design or directly obtain. To address this, we propose \textbf{STAR-S} (\textbf{S}elf-\textbf{TA}ught \textbf{R}easoning based on \textbf{S}afety rules), a framework that integrates the learning of safety rule reasoning into a self-taught loop. The core of STAR-S involves eliciting reasoning and reflection guided by safety rules, then leveraging fine-tuning to enhance safety reasoning. Repeating this process creates a synergistic cycle. Improvements in the model's reasoning and interpretation of safety rules allow it to produce better reasoning data under safety rule prompts, which is then utilized for further training. Experiments show that STAR-S effectively defends against jailbreak attacks, outperforming baselines. Code is available at: https://github.com/pikepokenew/STAR_S.git.
>
---
#### [new 120] SciNetBench: A Relation-Aware Benchmark for Scientific Literature Retrieval Agents
- **分类: cs.CE; cs.CL**

- **简介: 该论文提出SciNetBench，一个用于科学文献检索代理的关系感知基准。旨在解决现有检索系统无法有效捕捉文献间关系的问题，通过评估三类关系进行改进。**

- **链接: [https://arxiv.org/pdf/2601.03260v1](https://arxiv.org/pdf/2601.03260v1)**

> **作者:** Chenyang Shao; Yong Li; Fengli Xu
>
> **摘要:** The rapid development of AI agent has spurred the development of advanced research tools, such as Deep Research. Achieving this require a nuanced understanding of the relations within scientific literature, surpasses the scope of keyword-based or embedding-based retrieval. Existing retrieval agents mainly focus on the content-level similarities and are unable to decode critical relational dynamics, such as identifying corroborating or conflicting studies or tracing technological lineages, all of which are essential for a comprehensive literature review. Consequently, this fundamental limitation often results in a fragmented knowledge structure, misleading sentiment interpretation, and inadequate modeling of collective scientific progress. To investigate relation-aware retrieval more deeply, we propose SciNetBench, the first Scientific Network Relation-aware Benchmark for literature retrieval agents. Constructed from a corpus of over 18 million AI papers, our benchmark systematically evaluates three levels of relations: ego-centric retrieval of papers with novel knowledge structures, pair-wise identification of scholarly relationships, and path-wise reconstruction of scientific evolutionary trajectories. Through extensive evaluation of three categories of retrieval agents, we find that their accuracy on relation-aware retrieval tasks often falls below 20%, revealing a core shortcoming of current retrieval paradigms. Notably, further experiments on the literature review tasks demonstrate that providing agents with relational ground truth leads to a substantial 23.4% performance improvement in the review quality, validating the critical importance of relation-aware retrieval. We publicly release our benchmark at https://anonymous.4open.science/r/SciNetBench/ to support future research on advanced retrieval systems.
>
---
#### [new 121] RadDiff: Describing Differences in Radiology Image Sets with Natural Language
- **分类: cs.CV; cs.AI; cs.CL; cs.CY; cs.LG**

- **简介: 该论文提出RadDiff系统，用于描述放射学图像集之间的差异，解决医学影像比较与分析问题，通过多模态推理和医疗知识注入实现精准差异描述。**

- **链接: [https://arxiv.org/pdf/2601.03733v1](https://arxiv.org/pdf/2601.03733v1)**

> **作者:** Xiaoxian Shen; Yuhui Zhang; Sahithi Ankireddy; Xiaohan Wang; Maya Varma; Henry Guo; Curtis Langlotz; Serena Yeung-Levy
>
> **摘要:** Understanding how two radiology image sets differ is critical for generating clinical insights and for interpreting medical AI systems. We introduce RadDiff, a multimodal agentic system that performs radiologist-style comparative reasoning to describe clinically meaningful differences between paired radiology studies. RadDiff builds on a proposer-ranker framework from VisDiff, and incorporates four innovations inspired by real diagnostic workflows: (1) medical knowledge injection through domain-adapted vision-language models; (2) multimodal reasoning that integrates images with their clinical reports; (3) iterative hypothesis refinement across multiple reasoning rounds; and (4) targeted visual search that localizes and zooms in on salient regions to capture subtle findings. To evaluate RadDiff, we construct RadDiffBench, a challenging benchmark comprising 57 expert-validated radiology study pairs with ground-truth difference descriptions. On RadDiffBench, RadDiff achieves 47% accuracy, and 50% accuracy when guided by ground-truth reports, significantly outperforming the general-domain VisDiff baseline. We further demonstrate RadDiff's versatility across diverse clinical tasks, including COVID-19 phenotype comparison, racial subgroup analysis, and discovery of survival-related imaging features. Together, RadDiff and RadDiffBench provide the first method-and-benchmark foundation for systematically uncovering meaningful differences in radiological data.
>
---
#### [new 122] Roles of MLLMs in Visually Rich Document Retrieval for RAG: A Survey
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于文档检索任务，解决VRD在RAG中的语义复杂性问题。通过分析MLLM的三种角色，探讨其在提升检索效果中的应用与挑战。**

- **链接: [https://arxiv.org/pdf/2601.03262v1](https://arxiv.org/pdf/2601.03262v1)**

> **作者:** Xiantao Zhang
>
> **备注:** 18 pages; accepted at AACL-IJCNLP 2025 (main conference)
>
> **摘要:** Visually rich documents (VRDs) challenge retrieval-augmented generation (RAG) with layout-dependent semantics, brittle OCR, and evidence spread across complex figures and structured tables. This survey examines how Multimodal Large Language Models (MLLMs) are being used to make VRD retrieval practical for RAG. We organize the literature into three roles: Modality-Unifying Captioners, Multimodal Embedders, and End-to-End Representers. We compare these roles along retrieval granularity, information fidelity, latency and index size, and compatibility with reranking and grounding. We also outline key trade-offs and offer some practical guidance on when to favor each role. Finally, we identify promising directions for future research, including adaptive retrieval units, model size reduction, and the development of evaluation methods.
>
---
#### [new 123] Muse: Towards Reproducible Long-Form Song Generation with Fine-Grained Style Control
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于可控长歌词生成任务，旨在解决学术研究不可复现的问题。工作包括发布开源系统Muse及合成数据集，实现细粒度风格控制的歌曲生成。**

- **链接: [https://arxiv.org/pdf/2601.03973v1](https://arxiv.org/pdf/2601.03973v1)**

> **作者:** Changhao Jiang; Jiahao Chen; Zhenghao Xiang; Zhixiong Yang; Hanchen Wang; Jiabao Zhuang; Xinmeng Che; Jiajun Sun; Hui Li; Yifei Cao; Shihan Dou; Ming Zhang; Junjie Ye; Tao Ji; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Recent commercial systems such as Suno demonstrate strong capabilities in long-form song generation, while academic research remains largely non-reproducible due to the lack of publicly available training data, hindering fair comparison and progress. To this end, we release a fully open-source system for long-form song generation with fine-grained style conditioning, including a licensed synthetic dataset, training and evaluation pipelines, and Muse, an easy-to-deploy song generation model. The dataset consists of 116k fully licensed synthetic songs with automatically generated lyrics and style descriptions paired with audio synthesized by SunoV5. We train Muse via single-stage supervised finetuning of a Qwen-based language model extended with discrete audio tokens using MuCodec, without task-specific losses, auxiliary objectives, or additional architectural components. Our evaluations find that although Muse is trained with a modest data scale and model size, it achieves competitive performance on phoneme error rate, text--music style similarity, and audio aesthetic quality, while enabling controllable segment-level generation across different musical structures. All data, model weights, and training and evaluation pipelines will be publicly released, paving the way for continued progress in controllable long-form song generation research. The project repository is available at https://github.com/yuhui1038/Muse.
>
---
#### [new 124] MixRx: Predicting Drug Combination Interactions with LLMs
- **分类: q-bio.OT; cs.AI; cs.CL; cs.LG**

- **简介: 论文提出MixRx，利用大语言模型预测药物组合的相互作用（加成、协同、拮抗）。任务属于生物预测，解决药物组合效果预测问题，测试了多个模型并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2601.03277v1](https://arxiv.org/pdf/2601.03277v1)**

> **作者:** Risha Surana; Cameron Saidock; Hugo Chacon
>
> **摘要:** MixRx uses Large Language Models (LLMs) to classify drug combination interactions as Additive, Synergistic, or Antagonistic, given a multi-drug patient history. We evaluate the performance of 4 models, GPT-2, Mistral Instruct 2.0, and the fine-tuned counterparts. Our results showed a potential for such an application, with the Mistral Instruct 2.0 Fine-Tuned model providing an average accuracy score on standard and perturbed datasets of 81.5%. This paper aims to further develop an upcoming area of research that evaluates if LLMs can be used for biological prediction tasks.
>
---
#### [new 125] Spectral Archaeology: The Causal Topology of Model Evolution
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型分析任务，旨在揭示模型演化中的因果拓扑结构。通过注意力图谱谱分析，识别模型训练中的断层与特性变化，提供审计与验证工具。**

- **链接: [https://arxiv.org/pdf/2601.03424v1](https://arxiv.org/pdf/2601.03424v1)**

> **作者:** Valentin Noël
>
> **备注:** 45 pages, 15 figures, Under Review
>
> **摘要:** Behavioral benchmarks tell us \textit{what} a model does, but not \textit{how}. We introduce a training-free mechanistic probe using attention-graph spectra. Treating each layer as a token graph, we compute algebraic connectivity ($λ_2$), smoothness, and spectral entropy. Across 12 models and 10 languages, these measures yield stable ``spectral fingerprints'' that expose discontinuities missed by standard evaluation. We report four results. (1) Models undergoing specific curriculum transitions (e.g., code-to-chat) show an English-only, syntax-triggered connectivity failure on non-canonical constructions, reaching $Δλ_2 \approx -0.76$. We term this scar \textit{Passive-Triggered Connectivity Collapse} (PTCC). Analysis of the Phi lineage reveals that PTCC appears and resolves across developmental stages, implicating brittle curriculum shifts rather than synthetic data per se. (2) PTCC reflects a specialization trade-off: strengthened formal routing at the expense of stylistic flexibility. (3) We identify four recurrent processing strategies; simple frozen-threshold rules enable perfect forensic identification across lineages. (4) Mechanistically, PTCC localizes to a sparse Layer 2 ``compensatory patch'' of heads that fails under syntactic stress; activation steering can partially restore connectivity, recovering $\approx 38\%$ of lost information flow. Finally, dominant topological regimes track tokenization density more than language identity, suggesting ``healthy'' geometry varies systematically across scripts. Overall, attention-graph spectra provide a practical tool for auditing and training-regime verification.
>
---
#### [new 126] Controllable LLM Reasoning via Sparse Autoencoder-Based Steering
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于模型控制任务，旨在解决LRMs推理策略自主选择效率低、易出错的问题。通过SAE分解隐藏状态，提出SAE-Steering方法实现对推理策略的有效控制。**

- **链接: [https://arxiv.org/pdf/2601.03595v1](https://arxiv.org/pdf/2601.03595v1)**

> **作者:** Yi Fang; Wenjie Wang; Mingfeng Xue; Boyi Deng; Fengli Xu; Dayiheng Liu; Fuli Feng
>
> **备注:** Under Review
>
> **摘要:** Large Reasoning Models (LRMs) exhibit human-like cognitive reasoning strategies (e.g. backtracking, cross-verification) during reasoning process, which improves their performance on complex tasks. Currently, reasoning strategies are autonomously selected by LRMs themselves. However, such autonomous selection often produces inefficient or even erroneous reasoning paths. To make reasoning more reliable and flexible, it is important to develop methods for controlling reasoning strategies. Existing methods struggle to control fine-grained reasoning strategies due to conceptual entanglement in LRMs' hidden states. To address this, we leverage Sparse Autoencoders (SAEs) to decompose strategy-entangled hidden states into a disentangled feature space. To identify the few strategy-specific features from the vast pool of SAE features, we propose SAE-Steering, an efficient two-stage feature identification pipeline. SAE-Steering first recalls features that amplify the logits of strategy-specific keywords, filtering out over 99\% of features, and then ranks the remaining features by their control effectiveness. Using the identified strategy-specific features as control vectors, SAE-Steering outperforms existing methods by over 15\% in control effectiveness. Furthermore, controlling reasoning strategies can redirect LRMs from erroneous paths to correct ones, achieving a 7\% absolute accuracy improvement.
>
---
#### [new 127] RiskCueBench: Benchmarking Anticipatory Reasoning from Early Risk Cues in Video-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出RiskCueBench基准，用于评估视频语言模型从早期风险线索中进行预见性推理的能力，旨在解决实时风险预测中的挑战。**

- **链接: [https://arxiv.org/pdf/2601.03369v1](https://arxiv.org/pdf/2601.03369v1)**

> **作者:** Sha Luo; Yogesh Prabhu; Tim Ossowski; Kaiping Chen; Junjie Hu
>
> **摘要:** With the rapid growth of video centered social media, the ability to anticipate risky events from visual data is a promising direction for ensuring public safety and preventing real world accidents. Prior work has extensively studied supervised video risk assessment across domains such as driving, protests, and natural disasters. However, many existing datasets provide models with access to the full video sequence, including the accident itself, which substantially reduces the difficulty of the task. To better reflect real world conditions, we introduce a new video understanding benchmark RiskCueBench in which videos are carefully annotated to identify a risk signal clip, defined as the earliest moment that indicates a potential safety concern. Experimental results reveal a significant gap in current systems ability to interpret evolving situations and anticipate future risky events from early visual signals, highlighting important challenges for deploying video risk prediction models in practice.
>
---
#### [new 128] EASLT: Emotion-Aware Sign Language Translation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于手语翻译任务，解决手语翻译中因忽略面部表情导致的语义模糊问题。提出EASLT框架，通过情感编码和情感感知融合模块提升翻译准确性。**

- **链接: [https://arxiv.org/pdf/2601.03549v1](https://arxiv.org/pdf/2601.03549v1)**

> **作者:** Guobin Tu; Di Weng
>
> **摘要:** Sign Language Translation (SLT) is a complex cross-modal task requiring the integration of Manual Signals (MS) and Non-Manual Signals (NMS). While recent gloss-free SLT methods have made strides in translating manual gestures, they frequently overlook the semantic criticality of facial expressions, resulting in ambiguity when distinct concepts share identical manual articulations. To address this, we present **EASLT** (**E**motion-**A**ware **S**ign **L**anguage **T**ranslation), a framework that treats facial affect not as auxiliary information, but as a robust semantic anchor. Unlike methods that relegate facial expressions to a secondary role, EASLT incorporates a dedicated emotional encoder to capture continuous affective dynamics. These representations are integrated via a novel *Emotion-Aware Fusion* (EAF) module, which adaptively recalibrates spatio-temporal sign features based on affective context to resolve semantic ambiguities. Extensive evaluations on the PHOENIX14T and CSL-Daily benchmarks demonstrate that EASLT establishes advanced performance among gloss-free methods, achieving BLEU-4 scores of 26.15 and 22.80, and BLEURT scores of 61.0 and 57.8, respectively. Ablation studies confirm that explicitly modeling emotion effectively decouples affective semantics from manual dynamics, significantly enhancing translation fidelity. Code is available at https://github.com/TuGuobin/EASLT.
>
---
#### [new 129] Adaptive-Boundary-Clipping GRPO: Ensuring Bounded Ratios for Stable and Generalizable Training
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，针对GRPO算法在大语言模型训练中的不足，提出ABC-GRPO改进方法，提升训练稳定性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.03895v1](https://arxiv.org/pdf/2601.03895v1)**

> **作者:** Chi Liu; Xin Chen
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Group Relative Policy Optimization (GRPO) has emerged as a popular algorithm for reinforcement learning with large language models (LLMs). However, upon analyzing its clipping mechanism, we argue that it is suboptimal in certain scenarios. With appropriate modifications, GRPO can be significantly enhanced to improve both flexibility and generalization. To this end, we propose Adaptive-Boundary-Clipping GRPO (ABC-GRPO), an asymmetric and adaptive refinement of the original GRPO framework. We demonstrate that ABC-GRPO achieves superior performance over standard GRPO on mathematical reasoning tasks using the Qwen3 LLMs. Moreover, ABC-GRPO maintains substantially higher entropy throughout training, thereby preserving the model's exploration capacity and mitigating premature convergence. The implementation code is available online to ease reproducibility https://github.com/chi2liu/ABC-GRPO.
>
---
#### [new 130] FocusUI: Efficient UI Grounding via Position-Preserving Visual Token Selection
- **分类: cs.CV; cs.AI; cs.CL; cs.HC**

- **简介: 该论文提出FocusUI，解决UI接地任务中的计算效率与注意力稀释问题，通过选择关键视觉块并保持位置连续性提升性能。**

- **链接: [https://arxiv.org/pdf/2601.03928v1](https://arxiv.org/pdf/2601.03928v1)**

> **作者:** Mingyu Ouyang; Kevin Qinghong Lin; Mike Zheng Shou; Hwee Tou Ng
>
> **备注:** 14 pages, 13 figures
>
> **摘要:** Vision-Language Models (VLMs) have shown remarkable performance in User Interface (UI) grounding tasks, driven by their ability to process increasingly high-resolution screenshots. However, screenshots are tokenized into thousands of visual tokens (e.g., about 4700 for 2K resolution), incurring significant computational overhead and diluting attention. In contrast, humans typically focus on regions of interest when interacting with UI. In this work, we pioneer the task of efficient UI grounding. Guided by practical analysis of the task's characteristics and challenges, we propose FocusUI, an efficient UI grounding framework that selects patches most relevant to the instruction while preserving positional continuity for precise grounding. FocusUI addresses two key challenges: (1) Eliminating redundant tokens in visual encoding. We construct patch-level supervision by fusing an instruction-conditioned score with a rule-based UI-graph score that down-weights large homogeneous regions to select distinct and instruction-relevant visual tokens. (2) Preserving positional continuity during visual token selection. We find that general visual token pruning methods suffer from severe accuracy degradation on UI grounding tasks due to broken positional information. We introduce a novel PosPad strategy, which compresses each contiguous sequence of dropped visual tokens into a single special marker placed at the sequence's last index to preserve positional continuity. Comprehensive experiments on four grounding benchmarks demonstrate that FocusUI surpasses GUI-specific baselines. On the ScreenSpot-Pro benchmark, FocusUI-7B achieves a performance improvement of 3.7% over GUI-Actor-7B. Even with only 30% visual token retention, FocusUI-7B drops by only 3.2% while achieving up to 1.44x faster inference and 17% lower peak GPU memory.
>
---
#### [new 131] Content vs. Form: What Drives the Writing Score Gap Across Socioeconomic Backgrounds? A Generated Panel Approach
- **分类: econ.EM; cs.AI; cs.CL; cs.CY**

- **简介: 该论文研究写作评分中的社会经济背景差异，区分内容与形式的影响。任务是分离内容与风格对分数差距的贡献，通过生成文本变体进行分析。**

- **链接: [https://arxiv.org/pdf/2601.03469v1](https://arxiv.org/pdf/2601.03469v1)**

> **作者:** Nadav Kunievsky; Pedro Pertusi
>
> **摘要:** Students from different socioeconomic backgrounds exhibit persistent gaps in test scores, gaps that can translate into unequal educational and labor-market outcomes later in life. In many assessments, performance reflects not only what students know, but also how effectively they can communicate that knowledge. This distinction is especially salient in writing assessments, where scores jointly reward the substance of students' ideas and the way those ideas are expressed. As a result, observed score gaps may conflate differences in underlying content with differences in expressive skill. A central question, therefore, is how much of the socioeconomic-status (SES) gap in scores is driven by differences in what students say versus how they say it. We study this question using a large corpus of persuasive essays written by U.S. middle- and high-school students. We introduce a new measurement strategy that separates content from style by leveraging large language models to generate multiple stylistic variants of each essay. These rewrites preserve the underlying arguments while systematically altering surface expression, creating a "generated panel" that introduces controlled within-essay variation in style. This approach allows us to decompose SES gaps in writing scores into contributions from content and style. We find an SES gap of 0.67 points on a 1-6 scale. Approximately 69% of the gap is attributable to differences in essay content quality, Style differences account for 26% of the gap, and differences in evaluation standards across SES groups account for the remaining 5%. These patterns seems stable across demographic subgroups and writing tasks. More broadly, our approach shows how large language models can be used to generate controlled variation in observational data, enabling researchers to isolate and quantify the contributions of otherwise entangled factors.
>
---
## 更新

#### [replaced 001] SWE-Lego: Pushing the Limits of Supervised Fine-tuning for Software Issue Resolving
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出SWE-Lego，用于软件问题解决的监督微调方法，旨在提升代码修复等任务性能。通过优化数据集和训练流程，实现更高效准确的模型效果。**

- **链接: [https://arxiv.org/pdf/2601.01426v2](https://arxiv.org/pdf/2601.01426v2)**

> **作者:** Chaofan Tao; Jierun Chen; Yuxin Jiang; Kaiqi Kou; Shaowei Wang; Ruoyu Wang; Xiaohui Li; Sidi Yang; Yiming Du; Jianbo Dai; Zhiming Mao; Xinyu Wang; Lifeng Shang; Haoli Bai
>
> **备注:** Project website: https://github.com/SWE-Lego/SWE-Lego
>
> **摘要:** We present SWE-Lego, a supervised fine-tuning (SFT) recipe designed to achieve state-ofthe-art performance in software engineering (SWE) issue resolving. In contrast to prevalent methods that rely on complex training paradigms (e.g., mid-training, SFT, reinforcement learning, and their combinations), we explore how to push the limits of a lightweight SFT-only approach for SWE tasks. SWE-Lego comprises three core building blocks, with key findings summarized as follows: 1) the SWE-Lego dataset, a collection of 32k highquality task instances and 18k validated trajectories, combining real and synthetic data to complement each other in both quality and quantity; 2) a refined SFT procedure with error masking and a difficulty-based curriculum, which demonstrably improves action quality and overall performance. Empirical results show that with these two building bricks alone,the SFT can push SWE-Lego models to state-of-the-art performance among open-source models of comparable size on SWE-bench Verified: SWE-Lego-Qwen3-8B reaches 42.2%, and SWE-Lego-Qwen3-32B attains 52.6%. 3) We further evaluate and improve test-time scaling (TTS) built upon the SFT foundation. Based on a well-trained verifier, SWE-Lego models can be significantly boosted--for example, 42.2% to 49.6% and 52.6% to 58.8% under TTS@16 for the 8B and 32B models, respectively.
>
---
#### [replaced 002] Context-Alignment: Activating and Enhancing LLM Capabilities in Time Series
- **分类: cs.LG; cs.CL; stat.AP**

- **简介: 该论文属于时间序列任务，旨在提升预训练大语言模型在时间序列上的能力。通过引入上下文对齐方法，增强模型对时间序列的逻辑和结构理解。**

- **链接: [https://arxiv.org/pdf/2501.03747v3](https://arxiv.org/pdf/2501.03747v3)**

> **作者:** Yuxiao Hu; Qian Li; Dongxiao Zhang; Jinyue Yan; Yuntian Chen
>
> **备注:** This paper has been accepted by ICLR 2025
>
> **摘要:** Recently, leveraging pre-trained Large Language Models (LLMs) for time series (TS) tasks has gained increasing attention, which involves activating and enhancing LLMs' capabilities. Many methods aim to activate LLMs' capabilities based on token-level alignment, but overlook LLMs' inherent strength in natural language processing -- \textit{their deep understanding of linguistic logic and structure rather than superficial embedding processing.} We propose Context-Alignment (CA), a new paradigm that aligns TS with a linguistic component in the language environments familiar to LLMs to enable LLMs to contextualize and comprehend TS data, thereby activating their capabilities. Specifically, such context-level alignment comprises structural alignment and logical alignment, which is achieved by Dual-Scale Context-Alignment GNNs (DSCA-GNNs) applied to TS-language multimodal inputs. Structural alignment utilizes dual-scale nodes to describe hierarchical structure in TS-language, enabling LLMs to treat long TS data as a whole linguistic component while preserving intrinsic token features. Logical alignment uses directed edges to guide logical relationships, ensuring coherence in the contextual semantics. Following the DSCA-GNNs framework, we propose an instantiation method of CA, termed Few-Shot prompting Context-Alignment (FSCA), to enhance the capabilities of pre-trained LLMs in handling TS tasks. FSCA can be flexibly and repeatedly integrated into various layers of pre-trained LLMs to improve awareness of logic and structure, thereby enhancing performance. Extensive experiments show the effectiveness of FSCA and the importance of Context-Alignment across tasks, particularly in few-shot and zero-shot forecasting, confirming that Context-Alignment provides powerful prior knowledge on context. The code is open-sourced at https://github.com/tokaka22/ICLR25-FSCA.
>
---
#### [replaced 003] ILID: Native Script Language Identification for Indian Languages
- **分类: cs.CL**

- **简介: 该论文属于语言识别任务，旨在解决印度多语言中因同种文字和相似性导致的识别难题。作者构建了包含23种语言的25万句数据集，并提出高效模型提升识别效果。**

- **链接: [https://arxiv.org/pdf/2507.11832v3](https://arxiv.org/pdf/2507.11832v3)**

> **作者:** Yash Ingle; Pruthwik Mishra
>
> **备注:** 10 pages, 1 figure, 6 tables
>
> **摘要:** The language identification task is a crucial fundamental step in NLP. Often it serves as a pre-processing step for widely used NLP applications such as multilingual machine translation, information retrieval, question and answering, and text summarization. The core challenge of language identification lies in distinguishing languages in noisy, short, and code-mixed environments. This becomes even harder in case of diverse Indian languages that exhibit lexical and phonetic similarities, but have distinct differences. Many Indian languages share the same script, making the task even more challenging. Taking all these challenges into account, we develop and release a dataset of 250K sentences consisting of 23 languages including English and all 22 official Indian languages labeled with their language identifiers, where data in most languages are newly created. We also develop and release baseline models using state-of-the-art approaches in machine learning and fine-tuning pre-trained transformer models. Our models outperforms the state-of-the-art pre-trained transformer models for the language identification task. The dataset and the codes are available at https://yashingle-ai.github.io/ILID/ and in Huggingface open source libraries.
>
---
#### [replaced 004] Talk Less, Verify More: Improving LLM Assistants with Semantic Checks and Execution Feedback
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于自然语言处理任务，旨在提升LLM助手的准确性与可靠性。针对现有系统缺乏验证机制的问题，提出Q*和Feedback+两种验证方法，通过语义检查和执行反馈优化输出。**

- **链接: [https://arxiv.org/pdf/2601.00224v2](https://arxiv.org/pdf/2601.00224v2)**

> **作者:** Yan Sun; Ming Cai; Stanley Kok
>
> **备注:** WITS 2025 (Workshop on Information Technologies and Systems 2025)
>
> **摘要:** As large language model (LLM) assistants become increasingly integrated into enterprise workflows, their ability to generate accurate, semantically aligned, and executable outputs is critical. However, current conversational business analytics (CBA) systems often lack built-in verification mechanisms, leaving users to manually validate potentially flawed results. This paper introduces two complementary verification techniques: Q*, which performs reverse translation and semantic matching between code and user intent, and Feedback+, which incorporates execution feedback to guide code refinement. Embedded within a generator-discriminator framework, these mechanisms shift validation responsibilities from users to the system. Evaluations on three benchmark datasets, Spider, Bird, and GSM8K, demonstrate that both Q* and Feedback+ reduce error rates and task completion time. The study also identifies reverse translation as a key bottleneck, highlighting opportunities for future improvement. Overall, this work contributes a design-oriented framework for building more reliable, enterprise-grade GenAI systems capable of trustworthy decision support.
>
---
#### [replaced 005] Relevance to Utility: Process-Supervised Rewrite for RAG
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息检索与生成任务，解决检索文本与生成内容不匹配的问题。提出R2U方法，通过联合优化重写与回答提升生成效果。**

- **链接: [https://arxiv.org/pdf/2509.15577v2](https://arxiv.org/pdf/2509.15577v2)**

> **作者:** Jaeyoung Kim; Jongho Kim; Seung-won Hwang; Seoho Song; Young-In Song
>
> **摘要:** Retrieval-augmented generation systems often suffer from a gap between optimizing retrieval relevance and generative utility. With such a gap, retrieved documents may be topically relevant but still lack the content needed for effective reasoning during generation. While existing bridge modules attempt to rewrite the retrieved text for better generation, we show how they fail by not capturing "document utility". In this work, we propose R2U, with a key distinction of approximating true utility through joint observation of rewriting and answering in the reasoning process. To distill, R2U scale such supervision to enhance reliability in distillation. We further construct utility-improvement supervision by measuring the generator's gain of the answer under the rewritten context, yielding signals for fine-tuning and preference optimization. We evaluate our method across multiple open-domain question-answering benchmarks. The empirical results demonstrate consistent improvements over strong bridging baselines
>
---
#### [replaced 006] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于问答任务，解决RAG模型在文档排列变化下的幻觉问题。通过稳定生成机制提升答案一致性与准确性。**

- **链接: [https://arxiv.org/pdf/2601.02993v2](https://arxiv.org/pdf/2601.02993v2)**

> **作者:** Qianchi Zhang; Hainan Zhang; Liang Pang; Hongwei Zheng; Zhiming Zheng
>
> **备注:** 18 pages, 13figures, 8 tables. The code will be released after the review process
>
> **摘要:** Retrieval-Augmented Generation (RAG) has become a key paradigm for reducing factual hallucinations in large language models (LLMs), yet little is known about how the order of retrieved documents affects model behavior. We empirically show that under Top-5 retrieval with the gold document included, LLM answers vary substantially across permutations of the retrieved set, even when the gold document is fixed in the first position. This reveals a previously underexplored sensitivity to retrieval permutations. Although robust RAG methods primarily focus on enhancing LLM robustness to low-quality retrieval and mitigating positional bias to distribute attention fairly over long contexts, neither approach directly addresses permutation sensitivity. In this paper, we propose Stable-RAG, which exploits permutation sensitivity estimation to mitigate permutation-induced hallucinations. Stable-RAG runs the generator under multiple retrieval orders, clusters hidden states, and decodes from a cluster-center representation that captures the dominant reasoning pattern. It then uses these reasoning results to align hallucinated outputs toward the correct answer, encouraging the model to produce consistent and accurate predictions across document permutations. Experiments on three QA datasets show that Stable-RAG significantly improves answer accuracy, reasoning consistency and robust generalization across datasets, retrievers, and input lengths compared with baselines.
>
---
#### [replaced 007] Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models
- **分类: q-bio.NC; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理领域，旨在探索大语言模型中的功能网络及关键神经元。通过类比脑功能网络，分析模型内部结构，揭示其对性能的影响。**

- **链接: [https://arxiv.org/pdf/2502.20408v2](https://arxiv.org/pdf/2502.20408v2)**

> **作者:** Yiheng Liu; Zhengliang Liu; Zihao Wu; Junhao Ning; Haiyang Sun; Sichen Xia; Yang Yang; Xiaohui Gao; Ning Qiang; Bao Ge; Tianming Liu; Junwei Han; Xintao Hu
>
> **备注:** 21 pages, 18 figures
>
> **摘要:** In recent years, the rapid advancement of large language models (LLMs) in natural language processing has sparked significant interest among researchers to understand their mechanisms and functional characteristics. Although prior studies have attempted to explain LLM functionalities by identifying and interpreting specific neurons, these efforts mostly focus on individual neuron contributions, neglecting the fact that human brain functions are realized through intricate interaction networks. Inspired by research on functional brain networks (FBNs) in the field of neuroscience, we utilize similar methodologies estabilished in FBN analysis to explore the "functional networks" within LLMs in this study. Experimental results highlight that, much like the human brain, LLMs exhibit certain functional networks that recur frequently during their operation. Further investigation reveals that these functional networks are indispensable for LLM performance. Inhibiting key functional networks severely impairs the model's capabilities. Conversely, amplifying the activity of neurons within these networks can enhance either the model's overall performance or its performance on specific tasks. This suggests that these functional networks are strongly associated with either specific tasks or the overall performance of the LLM. Code is available at https://github.com/WhatAboutMyStar/LLM_ACTIVATION.
>
---
#### [replaced 008] Dissecting Physics Reasoning in Small Language Models: A Multi-Dimensional Analysis from an Educational Perspective
- **分类: cs.CL; cs.AI; physics.ed-ph**

- **简介: 该论文属于教育AI任务，旨在解决小语言模型在物理推理中的可靠性问题。通过构建基准和评估框架，分析模型推理过程中的错误模式与表现。**

- **链接: [https://arxiv.org/pdf/2505.20707v2](https://arxiv.org/pdf/2505.20707v2)**

> **作者:** Nicy Scaria; Silvester John Joseph Kennedy; Krishna Agarwal; Diksha Seth; Deepak Subramani
>
> **摘要:** Small Language Models (SLMs) offer privacy and efficiency for educational deployment, yet their utility depends on reliable multistep reasoning. Existing benchmarks often prioritize final answer accuracy, obscuring 'right answer, wrong procedure' failures that can reinforce student misconceptions. This work investigates SLM physics reasoning reliability, stage wise failure modes, and robustness under paired contextual variants. We introduce Physbench, comprising of 3,162 high school and AP level physics questions derived from OpenStax in a structured reference solution format with Bloom's Taxonomy annotations, plus 2,700 paired culturally contextualized variants. Using P-REFS, a stage wise evaluation rubric, we assess 10 SLMs across 58,000 responses. Results reveal substantial reliability gap: among final answer correct solutions, 75 to 98% contain at least one reasoning error. Failure modes shift with model capability; weaker models fail primarily at interpretation or modeling while stronger models often fail during execution. Paired contextual variations have minimal impact on top models but degrade the performance of mid-tier models. These findings demonstrate that safe educational AI requires evaluation paradigms that prioritize reasoning fidelity over final-answer correctness.
>
---
#### [replaced 009] Answering the Unanswerable Is to Err Knowingly: Analyzing and Mitigating Abstention Failures in Large Reasoning Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI可信性研究任务，旨在解决大推理模型在面对无法回答问题时未能正确拒绝回答的问题。通过分析模型认知与响应的不一致，提出一种轻量级方法提升拒绝率。**

- **链接: [https://arxiv.org/pdf/2508.18760v2](https://arxiv.org/pdf/2508.18760v2)**

> **作者:** Yi Liu; Xiangyu Liu; Zequn Sun; Wei Hu
>
> **备注:** Accepted in the 39th AAAI Conference on Artificial Intelligence (AAAI 2025)
>
> **摘要:** Large reasoning models (LRMs) have shown remarkable progress on complex reasoning tasks. However, some questions posed to LRMs are inherently unanswerable, such as math problems lacking sufficient conditions. We find that LRMs continually fail to provide appropriate abstentions when confronted with these unanswerable questions. In this paper, we systematically analyze, investigate, and resolve this issue for trustworthy AI. We first conduct a detailed analysis of the distinct response behaviors of LRMs when facing unanswerable questions. Then, we show that LRMs possess sufficient cognitive capabilities to recognize the flaws in these questions. However, they fail to exhibit appropriate abstention behavior, revealing a misalignment between their internal cognition and external response. Finally, to resolve this issue, we propose a lightweight, two-stage method that combines cognitive monitoring with inference-time intervention. Experimental results demonstrate that our method significantly improves the abstention rate while maintaining the overall reasoning performance.
>
---
#### [replaced 010] CodeFlowBench: A Multi-turn, Iterative Benchmark for Complex Code Generation
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出CodeFlowBench，用于评估大模型在多轮代码生成中的能力，解决代码复用与迭代开发问题。**

- **链接: [https://arxiv.org/pdf/2504.21751v3](https://arxiv.org/pdf/2504.21751v3)**

> **作者:** Sizhe Wang; Zhengren Wang; Dongsheng Ma; Yongan Yu; Rui Ling; Zhiyu Li; Feiyu Xiong; Wentao Zhang
>
> **摘要:** Modern software development demands code that is maintainable, testable, and scalable by organizing the implementation into modular components with iterative reuse of existing codes. We formalize this iterative, multi-turn paradigm as codeflow and introduce CodeFlowBench, the first benchmark designed to comprehensively evaluate LLMs' ability to perform codeflow - implementing new functionality by reusing existing functions over multiple turns. CodeFlowBench comprises two complementary components: CodeFlowBench-Comp, a core collection of 5,000+ competitive programming problems from Codeforces updated via an automated pipeline and CodeFlowBench-Repo, which is sourced from GitHub repositories to better reflect real-world scenarios. Furthermore, a novel evaluation framework featured dual assessment protocol and structural metrics derived from dependency trees is introduced. Extensive experiments reveal significant performance degradation in multi-turn codeflow scenarios. Furthermore, our in-depth analysis illustrates that model performance inversely correlates with dependency complexity. These findings not only highlight the critical challenges for supporting real-world workflows, but also establish CodeFlowBench as an essential tool for advancing code generation research.
>
---
#### [replaced 011] Generating Storytelling Images with Rich Chains-of-Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于故事生成任务，旨在解决稀缺且复杂的讲故事图像生成问题。提出StorytellingPainter框架，结合LLM推理与T2I合成，并引入Mini-Storytellers提升性能。**

- **链接: [https://arxiv.org/pdf/2512.07198v2](https://arxiv.org/pdf/2512.07198v2)**

> **作者:** Xiujie Song; Qi Jia; Shota Watanabe; Xiaoyi Pang; Ruijie Chen; Mengyue Wu; Kenny Q. Zhu
>
> **摘要:** A single image can convey a compelling story through logically connected visual clues, forming Chains-of-Reasoning (CoRs). We define these semantically rich images as Storytelling Images. By conveying multi-layered information that inspires active interpretation, these images enable a wide range of applications, such as illustration and cognitive screening. Despite their potential, such images are scarce and complex to create. To address this, we introduce the Storytelling Image Generation task and propose StorytellingPainter, a two-stage pipeline combining the reasoning of Large Language Models (LLMs) with Text-to-Image (T2I) synthesis. We also develop a dedicated evaluation framework assessing semantic complexity, diversity, and text-image alignment. Furthermore, given the critical role of story generation in the task, we introduce lightweight Mini-Storytellers to bridge the performance gap between small-scale and proprietary LLMs. Experimental results demonstrate the feasibility of our approaches.
>
---
#### [replaced 012] LiteStage: Latency-aware Layer Skipping for Multi-stage Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，针对多阶段推理中延迟与准确率不平衡的问题，提出LiteStage框架，通过分阶段层跳过和在线置信度退出提升效率。**

- **链接: [https://arxiv.org/pdf/2510.14211v2](https://arxiv.org/pdf/2510.14211v2)**

> **作者:** Beomseok Kang; Jiwon Song; Jae-Joon Kim
>
> **摘要:** Multi-stage reasoning has emerged as an effective strategy for enhancing the reasoning capability of small language models by decomposing complex problems into sequential sub-stages. However, this comes at the cost of increased latency. We observe that existing adaptive acceleration techniques, such as layer skipping, struggle to balance efficiency and accuracy in this setting due to two key challenges: (1) stage-wise variation in skip sensitivity, and (2) the generation of redundant output tokens. To address these, we propose LiteStage, a latency-aware layer skipping framework for multi-stage reasoning. LiteStage combines a stage-wise offline search that allocates optimal layer budgets with an online confidence-based generation early exit to suppress unnecessary decoding. Experiments on three benchmarks, e.g., OBQA, CSQA, and StrategyQA, show that LiteStage outperforms prior training-free layer skipping methods.
>
---
#### [replaced 013] The performances of the Chinese and U.S. Large Language Models on the Topic of Chinese Culture
- **分类: cs.CL**

- **简介: 该论文属于跨文化语言模型比较任务，旨在分析中美国模型在中文文化主题上的表现差异。通过测试模型对传统文化的理解，发现中国模型普遍更优。**

- **链接: [https://arxiv.org/pdf/2601.02830v2](https://arxiv.org/pdf/2601.02830v2)**

> **作者:** Feiyan Liu; Siyan Zhao; Chenxun Zhuo; Tianming Liu; Bao Ge
>
> **摘要:** Cultural backgrounds shape individuals' perspectives and approaches to problem-solving. Since the emergence of GPT-1 in 2018, large language models (LLMs) have undergone rapid development. To date, the world's ten leading LLM developers are primarily based in China and the United States. To examine whether LLMs released by Chinese and U.S. developers exhibit cultural differences in Chinese-language settings, we evaluate their performance on questions about Chinese culture. This study adopts a direct-questioning paradigm to evaluate models such as GPT-5.1, DeepSeek-V3.2, Qwen3-Max, and Gemini2.5Pro. We assess their understanding of traditional Chinese culture, including history, literature, poetry, and related domains. Comparative analyses between LLMs developed in China and the U.S. indicate that Chinese models generally outperform their U.S. counterparts on these tasks. Among U.S.-developed models, Gemini 2.5Pro and GPT-5.1 achieve relatively higher accuracy. The observed performance differences may potentially arise from variations in training data distribution, localization strategies, and the degree of emphasis on Chinese cultural content during model development.
>
---
#### [replaced 014] After Retrieval, Before Generation: Enhancing the Trustworthiness of Large Language Models in Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于RAG任务，解决LLM在检索与生成间信任度不足的问题。构建TRD数据集，提出BRIDGE框架，动态平衡内外部知识，提升响应可信度。**

- **链接: [https://arxiv.org/pdf/2505.17118v2](https://arxiv.org/pdf/2505.17118v2)**

> **作者:** Xinbang Dai; Huikang Hu; Yuncheng Hua; Jiaqi Li; Yongrui Chen; Rihui Jin; Nan Hu; Guilin Qi
>
> **备注:** 22 pages, 8 figures
>
> **摘要:** Retrieval-augmented generation (RAG) is a promising paradigm, yet its trustworthiness remains a critical concern. A major vulnerability arises prior to generation: models often fail to balance parametric (internal) and retrieved (external) knowledge, particularly when the two sources conflict or are unreliable. To analyze these scenarios comprehensively, we construct the Trustworthiness Response Dataset (TRD) with 36,266 questions spanning four RAG settings. We reveal that existing approaches address isolated scenarios-prioritizing one knowledge source, naively merging both, or refusing answers-but lack a unified framework to handle different real-world conditions simultaneously. Therefore, we propose the BRIDGE framework, which dynamically determines a comprehensive response strategy of large language models (LLMs). BRIDGE leverages an adaptive weighting mechanism named soft bias to guide knowledge collection, followed by a Maximum Soft-bias Decision Tree to evaluate knowledge and select optimal response strategies (trust internal/external knowledge, or refuse). Experiments show BRIDGE outperforms baselines by 5-15% in accuracy while maintaining balanced performance across all scenarios. Our work provides an effective solution for LLMs' trustworthy responses in real-world RAG applications.
>
---
#### [replaced 015] Believing without Seeing: Quality Scores for Contextualizing Vision-Language Model Explanations
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于视觉语言模型解释质量评估任务，旨在解决盲人用户无法查看视觉上下文时对模型预测的误信问题。通过提出视觉保真度和对比性两个质量评分函数，提升解释的可靠性。**

- **链接: [https://arxiv.org/pdf/2509.25844v2](https://arxiv.org/pdf/2509.25844v2)**

> **作者:** Keyu He; Tejas Srinivasan; Brihi Joshi; Xiang Ren; Jesse Thomason; Swabha Swayamdipta
>
> **摘要:** When people query Vision-Language Models (VLMs) but cannot see the accompanying visual context (e.g. for blind and low-vision users), augmenting VLM predictions with natural language explanations can signal which model predictions are reliable. However, prior work has found that explanations can easily convince users that inaccurate VLM predictions are correct. To remedy undesirable overreliance on VLM predictions, we propose evaluating two complementary qualities of VLM-generated explanations via two quality scoring functions. We propose Visual Fidelity, which captures how faithful an explanation is to the visual context, and Contrastiveness, which captures how well the explanation identifies visual details that distinguish the model's prediction from plausible alternatives. On the A-OKVQA, VizWiz, and MMMU-Pro tasks, these quality scoring functions are better calibrated with model correctness than existing explanation qualities. We conduct a user study in which participants have to decide whether a VLM prediction is accurate without viewing its visual context. We observe that showing our quality scores alongside VLM explanations improves participants' accuracy at predicting VLM correctness by 11.1%, including a 15.4% reduction in the rate of falsely believing incorrect predictions. These findings highlight the utility of explanation quality scores in fostering appropriate reliance on VLM predictions.
>
---
#### [replaced 016] Investigating CoT Monitorability in Large Reasoning Models
- **分类: cs.CL**

- **简介: 该论文属于AI安全领域，研究如何通过链式推理（CoT）监测大模型的不当行为。解决CoT可信度和监测可靠性问题，通过实证分析和干预方法提升监测效果。**

- **链接: [https://arxiv.org/pdf/2511.08525v3](https://arxiv.org/pdf/2511.08525v3)**

> **作者:** Shu Yang; Junchao Wu; Xilin Gong; Xuansheng Wu; Derek Wong; Ninghao Liu; Di Wang
>
> **摘要:** Large Reasoning Models (LRMs) have demonstrated remarkable performance on complex tasks by engaging in extended reasoning before producing final answers. Beyond improving abilities, these detailed reasoning traces also create a new opportunity for AI safety, CoT Monitorability: monitoring potential model misbehavior, such as the use of shortcuts or sycophancy, through their chain-of-thought (CoT) during decision-making. However, two key fundamental challenges arise when attempting to build more effective monitors through CoT analysis. First, as prior research on CoT faithfulness has pointed out, models do not always truthfully represent their internal decision-making in the generated reasoning. Second, monitors themselves may be either overly sensitive or insufficiently sensitive, and can potentially be deceived by models' long, elaborate reasoning traces. In this paper, we present the first systematic investigation of the challenges and potential of CoT monitorability. Motivated by two fundamental challenges we mentioned before, we structure our study around two central perspectives: (i) verbalization: to what extent do LRMs faithfully verbalize the true factors guiding their decisions in the CoT, and (ii) monitor reliability: to what extent can misbehavior be reliably detected by a CoT-based monitor? Specifically, we provide empirical evidence and correlation analyses between verbalization quality, monitor reliability, and LLM performance across mathematical, scientific, and ethical domains. Then we further investigate how different CoT intervention methods, designed to improve reasoning efficiency or performance, will affect monitoring effectiveness. Finally, we propose MoME, a new paradigm in which LLMs monitor other models' misbehavior through their CoT and provide structured judgments along with supporting evidence.
>
---
#### [replaced 017] Attention Needs to Focus: A Unified Perspective on Attention Allocation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决Transformer中注意力分配不当导致的表示崩溃和注意力陷阱问题。提出Lazy Attention机制，通过位置区分和弹性Softmax优化注意力分布。**

- **链接: [https://arxiv.org/pdf/2601.00919v2](https://arxiv.org/pdf/2601.00919v2)**

> **作者:** Zichuan Fu; Wentao Song; Guojing Li; Yejing Wang; Xian Wu; Yimin Deng; Hanyu Yan; Yefeng Zheng; Xiangyu Zhao
>
> **备注:** preprint
>
> **摘要:** The Transformer architecture, a cornerstone of modern Large Language Models (LLMs), has achieved extraordinary success in sequence modeling, primarily due to its attention mechanism. However, despite its power, the standard attention mechanism is plagued by well-documented issues: representational collapse and attention sink. Although prior work has proposed approaches for these issues, they are often studied in isolation, obscuring their deeper connection. In this paper, we present a unified perspective, arguing that both can be traced to a common root -- improper attention allocation. We identify two failure modes: 1) Attention Overload, where tokens receive comparable high weights, blurring semantic features that lead to representational collapse; 2) Attention Underload, where no token is semantically relevant, yet attention is still forced to distribute, resulting in spurious focus such as attention sink. Building on this insight, we introduce Lazy Attention, a novel mechanism designed for a more focused attention distribution. To mitigate overload, it employs positional discrimination across both heads and dimensions to sharpen token distinctions. To counteract underload, it incorporates Elastic-Softmax, a modified normalization function that relaxes the standard softmax constraint to suppress attention on irrelevant tokens. Experiments on the FineWeb-Edu corpus, evaluated across nine diverse benchmarks, demonstrate that Lazy Attention successfully mitigates attention sink and achieves competitive performance compared to both standard attention and modern architectures, while reaching up to 59.58% attention sparsity.
>
---
#### [replaced 018] Task-Stratified Knowledge Scaling Laws for Post-Training Quantized Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究Post-Training Quantization（PTQ）中的知识能力差异，提出任务分层的量化扩展规律，解决量化对不同知识能力影响不均的问题。**

- **链接: [https://arxiv.org/pdf/2508.18609v3](https://arxiv.org/pdf/2508.18609v3)**

> **作者:** Chenxi Zhou; Pengfei Cao; Jiang Li; Bohan Yu; Jinyu Ye; Jun Zhao; Kang Liu
>
> **摘要:** Post-Training Quantization (PTQ) is a critical strategy for efficient Large Language Models (LLMs) deployment. However, existing scaling laws primarily focus on general performance, overlooking crucial fine-grained factors and how quantization differentially impacts diverse knowledge capabilities. To address this, we establish Task-Stratified Knowledge Scaling Laws. By stratifying capabilities into memorization, application, and reasoning, we develop a framework that unifies model size, bit-width, and fine-grained factors: group size and calibration set size. Validated on 293 diverse PTQ configurations, our framework demonstrates strong fit and cross-architecture consistency. It reveals distinct sensitivities across knowledge capabilities: reasoning is precision-critical, application is scale-responsive, and memorization is calibration-sensitive. We highlight that in low-bit scenarios, optimizing these fine-grained factors is essential for preventing performance collapse. These findings provide an empirically-backed foundation for designing knowledge-aware quantization strategies.
>
---
#### [replaced 019] Pitfalls of Evaluating Language Models with Open Benchmarks
- **分类: cs.CL**

- **简介: 该论文属于语言模型评估任务，指出开放基准测试存在数据泄露风险，导致评估结果不可靠。通过构造作弊模型验证问题，并提出防护策略，强调需结合私有或动态基准以提升评估可靠性。**

- **链接: [https://arxiv.org/pdf/2507.00460v2](https://arxiv.org/pdf/2507.00460v2)**

> **作者:** Md. Najib Hasan; Md Mahadi Hassan Sibat; Mohammad Fakhruddin Babar; Souvika Sarkar; Monowar Hasan; Santu Karmaker
>
> **摘要:** Open Large Language Model (LLM) benchmarks, such as HELM and BIG-Bench, provide standardized and transparent evaluation protocols that support comparative analysis, reproducibility, and systematic progress tracking in Language Model (LM) research. Yet, this openness also creates substantial risks of data leakage during LM testing--deliberate or inadvertent, thereby undermining the fairness and reliability of leaderboard rankings and leaving them vulnerable to manipulation by unscrupulous actors. We illustrate the severity of this issue by intentionally constructing cheating models: smaller variants of BART, T5, and GPT-2, fine-tuned directly on publicly available test-sets. As expected, these models excel on the target benchmarks but fail terribly to generalize to comparable unseen testing sets. We then examine task specific simple paraphrase-based safeguarding strategies to mitigate the impact of data leakage and evaluate their effectiveness and limitations. Our findings underscore three key points: (i) high leaderboard performance on limited open, static benchmarks may not reflect real-world utility; (ii) private or dynamically generated benchmarks should complement open benchmarks to maintain evaluation integrity; and (iii) a reexamination of current benchmarking practices is essential for reliable and trustworthy LM assessment.
>
---
#### [replaced 020] From Human Intention to Action Prediction: Intention-Driven End-to-End Autonomous Driving
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文属于意图驱动的端到端自动驾驶任务，旨在解决现有系统难以理解人类高层意图的问题。工作包括构建基准数据集、提出新评估方法和两种解决方案。**

- **链接: [https://arxiv.org/pdf/2512.12302v2](https://arxiv.org/pdf/2512.12302v2)**

> **作者:** Huan Zheng; Yucheng Zhou; Tianyi Yan; Jiayi Su; Hongjun Chen; Dubing Chen; Xingtai Gui; Wencheng Han; Runzhou Tao; Zhongying Qiu; Jianfei Yang; Jianbing Shen
>
> **摘要:** While end-to-end autonomous driving has achieved remarkable progress in geometric control, current systems remain constrained by a command-following paradigm that relies on simple navigational instructions. Transitioning to genuinely intelligent agents requires the capability to interpret and fulfill high-level, abstract human intentions. However, this advancement is hindered by the lack of dedicated benchmarks and semantic-aware evaluation metrics. In this paper, we formally define the task of Intention-Driven End-to-End Autonomous Driving and present Intention-Drive, a comprehensive benchmark designed to bridge this gap. We construct a large-scale dataset featuring complex natural language intentions paired with high-fidelity sensor data. To overcome the limitations of conventional trajectory-based metrics, we introduce the Imagined Future Alignment (IFA), a novel evaluation protocol leveraging generative world models to assess the semantic fulfillment of human goals beyond mere geometric accuracy. Furthermore, we explore the solution space by proposing two distinct paradigms: an end-to-end vision-language planner and a hierarchical agent-based framework. The experiments reveal a critical dichotomy where existing models exhibit satisfactory driving stability but struggle significantly with intention fulfillment. Notably, the proposed frameworks demonstrate superior alignment with human intentions.
>
---
#### [replaced 021] User Perceptions of Privacy and Helpfulness in LLM Responses to Privacy-Sensitive Scenarios
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于隐私与人工智能交叉任务，旨在解决LLM在隐私敏感场景中响应的隐私保护与有用性评估问题。通过用户研究发现，用户对LLM响应的评价存在较大差异，而代理模型无法准确反映用户感知。**

- **链接: [https://arxiv.org/pdf/2510.20721v2](https://arxiv.org/pdf/2510.20721v2)**

> **作者:** Xiaoyuan Wu; Roshni Kaushik; Wenkai Li; Lujo Bauer; Koichi Onoue
>
> **摘要:** Large language models (LLMs) are rapidly being adopted for tasks like drafting emails, summarizing meetings, and answering health questions. In these settings, users may need to share private information (e.g., contact details, health records). To evaluate LLMs' ability to identify and redact such information, prior work introduced real-life, scenario-based benchmarks (e.g., ConfAIde, PrivacyLens) and found that LLMs can leak private information in complex scenarios. However, these evaluations relied on proxy LLMs to judge the helpfulness and privacy-preservation quality of LLM responses, rather than directly measuring users' perceptions. To understand how users perceive the helpfulness and privacy-preservation quality of LLM responses to privacy-sensitive scenarios, we conducted a user study ($n=94$) using 90 PrivacyLens scenarios. We found that users had low agreement with each other when evaluating identical LLM responses. In contrast, five proxy LLMs reached high agreement, yet each proxy LLM had low correlation with users' evaluations. These results indicate that proxy LLMs cannot accurately estimate users' wide range of perceptions of utility and privacy in privacy-sensitive scenarios. We discuss the need for more user-centered studies to measure LLMs' ability to help users while preserving privacy, and for improving alignment between LLMs and users in estimating perceived privacy and utility.
>
---
#### [replaced 022] SWAA: Sliding Window Attention Adaptation for Efficient Long-Context LLMs Without Pretraining
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决长文本推理中Transformer模型计算复杂度高的问题。通过SWAA方法，使模型在不重新预训练的情况下高效处理长上下文。**

- **链接: [https://arxiv.org/pdf/2512.10411v4](https://arxiv.org/pdf/2512.10411v4)**

> **作者:** Yijiong Yu; Jiale Liu; Qingyun Wu; Huazheng Wang; Ji Pei
>
> **摘要:** The quadratic complexity of self-attention in Transformer-based Large Language Models (LLMs) renders long-context inference prohibitively expensive. While Sliding Window Attention (SWA), the simplest sparse attention pattern, offers a linear-complexity alternative, naively applying it to models pretrained with Full Attention (FA) causes catastrophic long-context performance collapse due to the training-inference mismatch. To address this, we propose Sliding Window Attention Adaptation (SWAA), a plug-and-play toolkit of recipes that adapt FA models to SWA without costly pretraining. SWAA systematically combines five strategies: (1) applying SWA only during prefilling; (2) preserving "sink" tokens; (3) interleaving FA/SWA layers; (4) chain-of-thought (CoT); and (5) fine-tuning. Our experiments demonstrate that while individual methods are insufficient, specific synergistic combinations can effectively recover original long-context capabilities. After further analyzing performance-efficiency trade-offs, we identify recommended SWAA configurations for diverse scenarios, which achieve 30% to 100% speedups for long-context LLM inference with acceptable quality loss. Our code is available at https://github.com/yuyijiong/sliding-window-attention-adaptation
>
---
#### [replaced 023] Reinforcement Learning for Tool-Integrated Interleaved Thinking towards Cross-Domain Generalization
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决工具增强模型在跨领域任务中的泛化问题。通过提出RITE框架和Dr. GRPO优化方法，提升模型在不同领域的推理能力。**

- **链接: [https://arxiv.org/pdf/2510.11184v2](https://arxiv.org/pdf/2510.11184v2)**

> **作者:** Zhengyu Chen; Jinluan Yang; Teng Xiao; Ruochen Zhou; Luan Zhang; Xiangyu Xi; Xiaowei Shi; Wei Wang; Jinggang Wang
>
> **摘要:** Recent advances in large language models (LLMs) have demonstrated remarkable capabilities in reasoning and tool utilization. However, the generalization of tool-augmented reinforcement learning (RL) across diverse domains remains a significant challenge. Standard paradigms often treat tool usage as a linear or isolated event, which becomes brittle when transferring skills from restricted domains (e.g., mathematics) to open-ended tasks. In this work, we investigate the cross-domain generalization of an LLM agent trained exclusively on mathematical problem-solving. To facilitate robust skill transfer, we propose a {\textbf{R}einforcement Learning for \textbf{I}nterleaved \textbf{T}ool \textbf{E}xecution (RITE)}. Unlike traditional methods, RITE enforces a continuous ``Plan-Action-Reflection'' cycle, allowing the model to ground its reasoning in intermediate tool outputs and self-correct during long-horizon tasks. To effectively train this complex interleaved policy, we introduce {Dr. GRPO}, a robust optimization objective that utilizes token-level loss aggregation with importance sampling to mitigate reward sparsity and high-variance credit assignment. Furthermore, we employ a dual-component reward system and dynamic curriculum via online rollout filtering to ensure structural integrity and sample efficiency. Extensive experiments reveal that our approach, despite being trained solely on math tasks, achieves state-of-the-art performance across diverse reasoning domains, demonstrating high token efficiency and strong generalization capabilities.
>
---
#### [replaced 024] Can Large Language Models Identify Implicit Suicidal Ideation? An Empirical Evaluation
- **分类: cs.CL**

- **简介: 该论文属于心理健康检测任务，旨在评估大语言模型识别隐性自杀意念和提供支持性回应的能力。研究构建了新数据集并测试了多个模型，发现其在该任务中存在显著不足。**

- **链接: [https://arxiv.org/pdf/2502.17899v2](https://arxiv.org/pdf/2502.17899v2)**

> **作者:** Tong Li; Shu Yang; Junchao Wu; Jiyao Wei; Lijie Hu; Mengdi Li; Derek F. Wong; Joshua R. Oltmanns; Di Wang
>
> **摘要:** We present a comprehensive evaluation framework for assessing Large Language Models' (LLMs) capabilities in suicide prevention, focusing on two critical aspects: the Identification of Implicit Suicidal ideation (IIS) and the Provision of Appropriate Supportive responses (PAS). We introduce \ourdata, a novel dataset of 1,308 test cases built upon psychological frameworks including D/S-IAT and Negative Automatic Thinking, alongside real-world scenarios. Through extensive experiments with 8 widely used LLMs under different contextual settings, we find that current models struggle significantly with detecting implicit suicidal ideation and providing appropriate support, highlighting crucial limitations in applying LLMs to mental health contexts. Our findings underscore the need for more sophisticated approaches in developing and evaluating LLMs for sensitive psychological applications.
>
---
#### [replaced 025] Beyond Direct Generation: A Decomposed Approach to Well-Crafted Screenwriting with LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于创意写作任务，旨在解决LLMs生成高质量剧本的问题。通过分解生成过程，先构建叙事再转换格式，提升剧本质量。**

- **链接: [https://arxiv.org/pdf/2510.23163v3](https://arxiv.org/pdf/2510.23163v3)**

> **作者:** Hang Lei; Shengyi Zong; Zhaoyan Li; Ziren Zhou; Hao Liu; Liang Yu
>
> **摘要:** The screenplay serves as the foundation for television production, defining narrative structure, character development, and dialogue. While Large Language Models (LLMs) show great potential in creative writing, direct end-to-end generation approaches often fail to produce well-crafted screenplays. We argue this failure stems from forcing a single model to simultaneously master two disparate capabilities: creative narrative construction and rigid format adherence. The resulting outputs may mimic superficial style but lack the deep structural integrity and storytelling substance required for professional use. To enable LLMs to generate high-quality screenplays, we introduce Dual-Stage Refinement (DSR), a decomposed framework that decouples creative narrative generation from format conversion. The first stage transforms a brief outline into rich, novel-style prose. The second stage refines this narrative into a professionally formatted screenplay. This separation enables the model to specialize in one distinct capability at each stage. A key challenge in implementing DSR is the scarcity of paired outline-to-novel training data. We address this through hybrid data synthesis: reverse synthesis deconstructs existing screenplays into structured inputs, while forward synthesis leverages these inputs to generate high-quality narrative texts as training targets. Blind evaluations by professional screenwriters show that DSR achieves a 75% win rate against strong baselines like Gemini-2.5-Pro and reaches 82.7% of human-level performance. Our work demonstrates that decomposed generation architecture with tailored data synthesis effectively specializes LLMs in complex creative domains.
>
---
#### [replaced 026] Interleaved Reasoning for Large Language Models via Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理效率低的问题。通过强化学习引导模型进行交替思考与回答，提升推理效率和准确性。**

- **链接: [https://arxiv.org/pdf/2505.19640v2](https://arxiv.org/pdf/2505.19640v2)**

> **作者:** Roy Xie; David Qiu; Deepak Gopinath; Dong Lin; Yanchao Sun; Chong Wang; Saloni Potdar; Bhuwan Dhingra
>
> **摘要:** Long chain-of-thought (CoT) significantly enhances the reasoning capabilities of large language models (LLMs). However, extensive reasoning traces lead to inefficiencies and increased time-to-first-token (TTFT). We propose a training paradigm that uses only reinforcement learning (RL) to guide reasoning LLMs to interleave thinking and answering for multi-hop questions. We observe that models inherently possess the ability to perform interleaved reasoning, which can be further enhanced through RL. We introduce a simple yet effective reward scheme to incentivize correct intermediate steps, guiding the policy model toward correct reasoning paths by leveraging intermediate signals generated during interleaved reasoning. Extensive experiments across five diverse datasets and three RL algorithms (PPO, GRPO, and REINFORCE++) demonstrate consistent improvements over traditional think-answer reasoning, without requiring external tools. Our method improves final task accuracy and overall efficiency by enabling more effective credit assignment during RL. Specifically, our approach achieves a 12.5% improvement in Pass@1 accuracy, while reducing overall reasoning length by 37% and TTFT by over 80% on average. Furthermore, our method, trained solely on question answering and logical reasoning datasets, exhibits strong generalization to complex reasoning datasets such as MATH, GPQA, and MMLU. Additionally, we conduct in-depth analysis to reveal several valuable insights into conditional reward modeling.
>
---
#### [replaced 027] Qomhra: A Bilingual Irish and English Large Language Model
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决低资源语言如爱尔兰语在大语言模型中的不足。研究提出Qomhrá模型，通过预训练、指令调优和合成人类偏好数据提升其性能。**

- **链接: [https://arxiv.org/pdf/2510.17652v3](https://arxiv.org/pdf/2510.17652v3)**

> **作者:** Joseph McInerney; Khanh-Tung Tran; Liam Lonergan; Ailbhe Ní Chasaide; Neasa Ní Chiaráin; Barry Devereux
>
> **摘要:** Large language model (LLM) research and development has overwhelmingly focused on the world's major languages, leading to under-representation of low-resource languages such as Irish. This paper introduces \textbf{Qomhrá}, a bilingual Irish and English LLM, developed under extremely low-resource constraints. A complete pipeline is outlined spanning bilingual continued pre-training, instruction tuning, and the synthesis of human preference data for future alignment training. We focus on the lack of scalable methods to create human preference data by proposing a novel method to synthesise such data by prompting an LLM to generate ``accepted'' and ``rejected'' responses, which we validate as aligning with L1 Irish speakers. To select an LLM for synthesis, we evaluate the top closed-weight LLMs for Irish language generation performance. Gemini-2.5-Pro is ranked highest by L1 and L2 Irish-speakers, diverging from LLM-as-a-judge ratings, indicating a misalignment between current LLMs and the Irish-language community. Subsequently, we leverage Gemini-2.5-Pro to translate a large scale English-language instruction tuning dataset to Irish and to synthesise a first-of-its-kind Irish-language human preference dataset. We comprehensively evaluate Qomhrá across several benchmarks, testing translation, gender understanding, topic identification, and world knowledge; these evaluations show gains of up to 29\% in Irish and 44\% in English compared to the existing open-source Irish LLM baseline, UCCIX. The results of our framework provide insight and guidance to developing LLMs for both Irish and other low-resource languages.
>
---
#### [replaced 028] FinDeepResearch: Evaluating Deep Research Agents in Rigorous Financial Analysis
- **分类: cs.CL**

- **简介: 该论文属于金融分析任务，旨在评估深度研究代理的能力。提出HisRubric框架和FinDeepResearch基准，进行多方法实验，揭示其优缺点。**

- **链接: [https://arxiv.org/pdf/2510.13936v2](https://arxiv.org/pdf/2510.13936v2)**

> **作者:** Fengbin Zhu; Xiang Yao Ng; Ziyang Liu; Chang Liu; Xianwei Zeng; Chao Wang; Tianhui Tan; Xuan Yao; Pengyang Shao; Min Xu; Zixuan Wang; Jing Wang; Xin Lin; Junfeng Li; Jingxian Zhu; Yang Zhang; Wenjie Wang; Fuli Feng; Richang Hong; Huanbo Luan; Ke-Wei Huang; Tat-Seng Chua
>
> **摘要:** Deep Research (DR) agents, powered by advanced Large Language Models (LLMs), have recently garnered increasing attention for their capability in conducting complex research tasks. However, existing literature lacks a rigorous and systematic evaluation of DR Agent's capabilities in critical research analysis. To address this gap, we first propose HisRubric, a novel evaluation framework with a hierarchical analytical structure and a fine-grained grading rubric for rigorously assessing DR agents' capabilities in corporate financial analysis. This framework mirrors the professional analyst's workflow, progressing from data recognition to metric calculation, and finally to strategic summarization and interpretation. Built on this framework, we construct a FinDeepResearch benchmark that comprises 64 listed companies from 8 financial markets across 4 languages, encompassing a total of 15,808 grading items. We further conduct extensive experiments on the FinDeepResearch using 16 representative methods, including 6 DR agents, 5 LLMs equipped with both deep reasoning and search capabilities, and 5 LLMs with deep reasoning capabilities only. The results reveal the strengths and limitations of these approaches across diverse capabilities, financial markets, and languages, offering valuable insights for future research and development. The benchmark and evaluation code is publicly available at https://OpenFinArena.com/.
>
---
#### [replaced 029] Merlin's Whisper: Enabling Efficient Reasoning in Large Language Models via Black-box Persuasive Prompting
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于提升大语言模型推理效率的任务，旨在减少推理过程中的计算和延迟开销。通过黑盒说服提示技术，生成简洁响应，保持准确性。**

- **链接: [https://arxiv.org/pdf/2510.10528v2](https://arxiv.org/pdf/2510.10528v2)**

> **作者:** Heming Xia; Cunxiao Du; Rui Li; Chak Tou Leong; Yongqi Li; Wenjie Li
>
> **摘要:** Large reasoning models (LRMs) have demonstrated remarkable proficiency in tackling complex tasks through step-by-step thinking. However, this lengthy reasoning process incurs substantial computational and latency overheads, hindering the practical deployment of LRMs. This work presents a new approach to mitigating overthinking in LRMs via black-box persuasive prompting. By treating LRMs as black-box communicators, we investigate how to persuade them to generate concise responses without compromising accuracy. We introduce Whisper, an iterative refinement framework that generates high-quality persuasive prompts from diverse perspectives. Experiments across multiple benchmarks demonstrate that Whisper consistently reduces token usage while preserving performance. Notably, Whisper achieves a 3x reduction in average response length on simple GSM8K questions for the Qwen3 model series and delivers an average ~40% token reduction across all benchmarks. For closed-source APIs, Whisper reduces token usage on MATH-500 by 46% for Claude-3.7 and 50% for Gemini-2.5. Further analysis reveals the broad applicability of Whisper across data domains, model scales, and families, underscoring the potential of black-box persuasive prompting as a practical strategy for enhancing LRM efficiency.
>
---
#### [replaced 030] PAM: Training Policy-Aligned Moderation Filters at Scale
- **分类: cs.CL**

- **简介: 该论文提出PAM框架，用于训练符合用户政策的过滤器，解决LLM对齐与安全问题。通过自动化数据生成，实现高效、灵活的政策执行。**

- **链接: [https://arxiv.org/pdf/2505.19766v3](https://arxiv.org/pdf/2505.19766v3)**

> **作者:** Masoomali Fatehkia; Enes Altinisik; Mohamed Osman; Husrev Taha Sencar
>
> **摘要:** Large language models (LLMs) remain vulnerable to misalignment and jailbreaks, making external safeguards like moderation filters essential, yet existing filters often focus narrowly on safety, falling short of the broader alignment needs seen in real-world deployments. We introduce Policy Aligned Moderation (PAM), a flexible framework for training custom moderation filters grounded in user-defined policies that extend beyond conventional safety objectives. PAM automates training data generation without relying on human-written examples, enabling scalable support for diverse, application-specific alignment goals and generation policies. PAM-trained filters match the performance of state-of-the-art safety moderation filters and policy reasoning models, and outperform them on PAMbench, four newly introduced user-annotated policy enforcement benchmarks that target age restrictions, dietary accommodations, cultural alignment, and limitations in medical guidance. These performance gains are achieved while the PAM filter runs 5-100x faster at inference than policy-conditioned reasoning models.
>
---
#### [replaced 031] TrojanStego: Your Language Model Can Secretly Be A Steganographic Privacy Leaking Agent
- **分类: cs.CL; cs.CR**

- **简介: 该论文提出TrojanStego，属于隐私泄露攻击任务，旨在解决LLM秘密传输敏感信息的问题。通过语言隐写术实现隐蔽数据外泄，实验验证其有效性与隐蔽性。**

- **链接: [https://arxiv.org/pdf/2505.20118v4](https://arxiv.org/pdf/2505.20118v4)**

> **作者:** Dominik Meier; Jan Philip Wahle; Paul Röttger; Terry Ruas; Bela Gipp
>
> **备注:** 9 pages, 5 figures To be presented in the Conference on Empirical Methods in Natural Language Processing, 2025
>
> **摘要:** As large language models (LLMs) become integrated into sensitive workflows, concerns grow over their potential to leak confidential information. We propose TrojanStego, a novel threat model in which an adversary fine-tunes an LLM to embed sensitive context information into natural-looking outputs via linguistic steganography, without requiring explicit control over inference inputs. We introduce a taxonomy outlining risk factors for compromised LLMs, and use it to evaluate the risk profile of the threat. To implement TrojanStego, we propose a practical encoding scheme based on vocabulary partitioning learnable by LLMs via fine-tuning. Experimental results show that compromised models reliably transmit 32-bit secrets with 87% accuracy on held-out prompts, reaching over 97% accuracy using majority voting across three generations. Further, they maintain high utility, can evade human detection, and preserve coherence. These results highlight a new class of LLM data exfiltration attacks that are passive, covert, practical, and dangerous.
>
---
#### [replaced 032] Does Memory Need Graphs? A Unified Framework and Empirical Analysis for Long-Term Dialog Memory
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话记忆任务，旨在探讨图结构在长期对话记忆中的有效性。通过构建统一框架，对比分析不同设计选择，发现性能差异多由基础设置引起，而非架构创新。**

- **链接: [https://arxiv.org/pdf/2601.01280v2](https://arxiv.org/pdf/2601.01280v2)**

> **作者:** Sen Hu; Yuxiang Wei; Jiaxin Ran; Zhiyuan Yao; Xueran Han; Huacan Wang; Ronghao Chen; Lei Zou
>
> **摘要:** Graph structures are increasingly used in dialog memory systems, but empirical findings on their effectiveness remain inconsistent, making it unclear which design choices truly matter. We present an experimental, system-oriented analysis of long-term dialog memory architectures. We introduce a unified framework that decomposes dialog memory systems into core components and supports both graph-based and non-graph approaches. Under this framework, we conduct controlled, stage-wise experiments on LongMemEval and HaluMem, comparing common design choices in memory representation, organization, maintenance, and retrieval. Our results show that many performance differences are driven by foundational system settings rather than specific architectural innovations. Based on these findings, we identify stable and reliable strong baselines for future dialog memory research.
>
---
#### [replaced 033] MedDialogRubrics: A Comprehensive Benchmark and Evaluation Framework for Multi-turn Medical Consultations in Large Language Models
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于医疗对话系统任务，旨在解决医学大模型评估框架不足的问题。构建了MedDialogRubrics基准，包含合成病例和专家优化的评价标准，以评估模型的多轮诊断能力。**

- **链接: [https://arxiv.org/pdf/2601.03023v2](https://arxiv.org/pdf/2601.03023v2)**

> **作者:** Lecheng Gong; Weimin Fang; Ting Yang; Dongjie Tao; Chunxiao Guo; Peng Wei; Bo Xie; Jinqun Guan; Zixiao Chen; Fang Shi; Jinjie Gu; Junwei Liu
>
> **摘要:** Medical conversational AI (AI) plays a pivotal role in the development of safer and more effective medical dialogue systems. However, existing benchmarks and evaluation frameworks for assessing the information-gathering and diagnostic reasoning abilities of medical large language models (LLMs) have not been rigorously evaluated. To address these gaps, we present MedDialogRubrics, a novel benchmark comprising 5,200 synthetically constructed patient cases and over 60,000 fine-grained evaluation rubrics generated by LLMs and subsequently refined by clinical experts, specifically designed to assess the multi-turn diagnostic capabilities of LLM. Our framework employs a multi-agent system to synthesize realistic patient records and chief complaints from underlying disease knowledge without accessing real-world electronic health records, thereby mitigating privacy and data-governance concerns. We design a robust Patient Agent that is limited to a set of atomic medical facts and augmented with a dynamic guidance mechanism that continuously detects and corrects hallucinations throughout the dialogue, ensuring internal coherence and clinical plausibility of the simulated cases. Furthermore, we propose a structured LLM-based and expert-annotated rubric-generation pipeline that retrieves Evidence-Based Medicine (EBM) guidelines and utilizes the reject sampling to derive a prioritized set of rubric items ("must-ask" items) for each case. We perform a comprehensive evaluation of state-of-the-art models and demonstrate that, across multiple assessment dimensions, current models face substantial challenges. Our results indicate that improving medical dialogue will require advances in dialogue management architectures, not just incremental tuning of the base-model.
>
---
#### [replaced 034] CoreCodeBench: Decoupling Code Intelligence via Fine-Grained Repository-Level Tasks
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出CoreCodeBench，用于评估代码智能模型的细粒度能力。针对现有基准评估粗略、静态的问题，通过分解任务提升评估精度与适应性。**

- **链接: [https://arxiv.org/pdf/2507.05281v2](https://arxiv.org/pdf/2507.05281v2)**

> **作者:** Lingyue Fu; Hao Guan; Bolun Zhang; Haowei Yuan; Yaoming Zhu; Jun Xu; Zongyu Wang; Lin Qiu; Xunliang Cai; Xuezhi Cao; Weiwen Liu; Weinan Zhang; Yong Yu
>
> **摘要:** The evaluation of Large Language Models (LLMs) for software engineering has shifted towards complex, repository-level tasks. However, existing benchmarks predominantly rely on coarse-grained pass rates that treat programming proficiency as a monolithic capability, obscuring specific cognitive bottlenecks. Furthermore, the static nature of these benchmarks renders them vulnerable to data contamination and performance saturation. To address these limitations, we introduce CoreCodeBench, a configurable repository-level benchmark designed to dissect coding capabilities through atomized tasks. Leveraging our automated framework, CorePipe, we extract and transform Python repositories into a comprehensive suite of tasks that isolate distinct cognitive demands within identical code contexts. Unlike static evaluations, CoreCodeBench supports controllable difficulty scaling to prevent saturation and ensures superior data quality. It achieves a 78.55% validity yield, significantly surpassing the 31.7% retention rate of SWE-bench-Verified. Extensive experiments with state-of-the-art LLMs reveal a significant capability misalignment, evidenced by distinct ranking shifts across cognitive dimensions. This indicates that coding proficiency is non-monolithic, as strength in one aspect does not necessarily translate to others. These findings underscore the necessity of our fine-grained taxonomy in diagnosing model deficiencies and offer a sustainable, rigorous framework for evolving code intelligence. The code for CorePipe is available at https://github.com/AGI-Eval-Official/CoreCodeBench, and the data for CoreCodeBench can be accessed at https://huggingface.co/collections/tubehhh/corecodebench-68256d2faabf4b1610a08caa.
>
---
#### [replaced 035] Can LLMs Track Their Output Length? A Dynamic Feedback Mechanism for Precise Length Regulation
- **分类: cs.CL**

- **简介: 该论文属于文本生成任务，旨在解决LLMs难以精确控制输出长度的问题。提出一种动态反馈机制，实现长度精准调节。**

- **链接: [https://arxiv.org/pdf/2601.01768v2](https://arxiv.org/pdf/2601.01768v2)**

> **作者:** Meiman Xiao; Ante Wang; Qingguo Hu; Zhongjian Miao; Huangjun Shen; Longyue Wang; Weihua Luo; Jinsong Su
>
> **摘要:** Precisely controlling the length of generated text is a common requirement in real-world applications. However, despite significant advancements in following human instructions, Large Language Models (LLMs) still struggle with this task. In this work, we demonstrate that LLMs often fail to accurately measure their response lengths, leading to poor adherence to length constraints. To address this issue, we propose a novel length regulation approach that incorporates dynamic length feedback during generation, enabling adaptive adjustments to meet target lengths. Experiments on summarization and biography tasks show our training-free approach significantly improves precision in achieving target token, word, or sentence counts without compromising quality. Additionally, we demonstrate that further supervised fine-tuning allows our method to generalize effectively to broader text-generation tasks.
>
---
#### [replaced 036] HiCoLoRA: Addressing Context-Prompt Misalignment via Hierarchical Collaborative LoRA for Zero-Shot DST
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于零样本对话状态追踪任务，解决对话上下文与静态提示语义不匹配的问题。提出HiCoLoRA框架，通过层次化LoRA和语义增强方法提升模型性能。**

- **链接: [https://arxiv.org/pdf/2509.19742v3](https://arxiv.org/pdf/2509.19742v3)**

> **作者:** Shuyu Zhang; Yifan Wei; Xinru Wang; Yanmin Zhu; Yangfan He; Yixuan Weng; Bin Li; Yujie Liu
>
> **摘要:** Zero-shot Dialog State Tracking (zs-DST) is essential for enabling Task-Oriented Dialog Systems (TODs) to generalize to new domains without costly data annotation. A central challenge lies in the semantic misalignment between dynamic dialog contexts and static prompts, leading to inflexible cross-layer coordination, domain interference, and catastrophic forgetting. To tackle this, we propose Hierarchical Collaborative Low-Rank Adaptation (HiCoLoRA), a framework that enhances zero-shot slot inference through robust prompt alignment. It features a hierarchical LoRA architecture for dynamic layer-specific processing (combining lower-layer heuristic grouping and higher-layer full interaction), integrates Spectral Joint Domain-Slot Clustering to identify transferable associations (feeding an Adaptive Linear Fusion Mechanism), and employs Semantic-Enhanced SVD Initialization (SemSVD-Init) to preserve pre-trained knowledge. Experiments on multi-domain datasets MultiWOZ and SGD show that HiCoLoRA outperforms baselines, achieving SOTA in zs-DST. Code is available at https://github.com/carsonz/HiCoLoRA.
>
---
#### [replaced 037] Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于安全检测任务，旨在解决LVLM对多模态越狱攻击的防御问题。通过分析内部表示，提出RCS框架提升检测效果。**

- **链接: [https://arxiv.org/pdf/2512.12069v2](https://arxiv.org/pdf/2512.12069v2)**

> **作者:** Peichun Hua; Hao Li; Shanghao Shi; Zhiyuan Yu; Ning Zhang
>
> **备注:** 37 pages, 13 figures
>
> **摘要:** Large Vision-Language Models (LVLMs) are vulnerable to a growing array of multimodal jailbreak attacks, necessitating defenses that are both generalizable to novel threats and efficient for practical deployment. Many current strategies fall short, either targeting specific attack patterns, which limits generalization, or imposing high computational overhead. While lightweight anomaly-detection methods offer a promising direction, we find that their common one-class design tends to confuse novel benign inputs with malicious ones, leading to unreliable over-rejection. To address this, we propose Representational Contrastive Scoring (RCS), a framework built on a key insight: the most potent safety signals reside within the LVLM's own internal representations. Our approach inspects the internal geometry of these representations, learning a lightweight projection to maximally separate benign and malicious inputs in safety-critical layers. This enables a simple yet powerful contrastive score that differentiates true malicious intent from mere novelty. Our instantiations, MCD (Mahalanobis Contrastive Detection) and KCD (K-nearest Contrastive Detection), achieve state-of-the-art performance on a challenging evaluation protocol designed to test generalization to unseen attack types. This work demonstrates that effective jailbreak detection can be achieved by applying simple, interpretable statistical methods to the appropriate internal representations, offering a practical path towards safer LVLM deployment. Our code is available on Github https://github.com/sarendis56/Jailbreak_Detection_RCS.
>
---
#### [replaced 038] Fair Document Valuation in LLM Summaries via Shapley Values
- **分类: cs.CL; econ.GN**

- **简介: 该论文属于内容归属任务，旨在解决LLM摘要中文档贡献度评估问题。提出基于Shapley值的框架，并引入Cluster Shapley提升效率与公平性。**

- **链接: [https://arxiv.org/pdf/2505.23842v4](https://arxiv.org/pdf/2505.23842v4)**

> **作者:** Zikun Ye; Hema Yoganarasimhan
>
> **摘要:** Large Language Models (LLMs) are increasingly used in systems that retrieve and summarize content from multiple sources, such as search engines and AI assistants. While these systems enhance user experience through coherent summaries, they obscure the individual contributions of original content creators, raising concerns about credit attribution and compensation. We address the challenge of valuing individual documents used in LLM-generated summaries by proposing a Shapley value-based framework for fair document valuation. Although theoretically appealing, exact Shapley value computation is prohibitively expensive at scale. To improve efficiency, we develop Cluster Shapley, a simple approximation algorithm that leverages semantic similarity among documents to reduce computation while maintaining attribution accuracy. Using Amazon product review data, we empirically show that off-the-shelf Shapley approximations, such as Monte Carlo sampling and Kernel SHAP, perform suboptimally in LLM settings, whereas Cluster Shapley substantially improves the efficiency-accuracy frontier. Moreover, simple attribution rules (e.g., equal or relevance-based allocation), though computationally cheap, lead to highly unfair outcomes. Together, our findings highlight the potential of structure-aware Shapley approximations tailored to LLM summarization and offer guidance for platforms seeking scalable and fair content attribution mechanisms.
>
---
#### [replaced 039] Investigating Counterclaims in Causality Extraction from Text
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于因果关系抽取任务，旨在解决忽视反例因果陈述的问题。通过构建包含因果、反因果和非因果陈述的数据集，提升模型对反因果语句的识别能力。**

- **链接: [https://arxiv.org/pdf/2510.08224v2](https://arxiv.org/pdf/2510.08224v2)**

> **作者:** Tim Hagen; Niklas Deckers; Felix Wolter; Harrisen Scells; Martin Potthast
>
> **摘要:** Many causal claims, such as "sugar causes hyperactivity," are disputed or outdated. Yet research on causality extraction from text has almost entirely neglected counterclaims of causation. To close this gap, we conduct a thorough literature review of causality extraction, compile an extensive inventory of linguistic realizations of countercausal claims, and develop rigorous annotation guidelines that explicitly incorporate countercausal language. We also highlight how counterclaims of causation are an integral part of causal reasoning. Based on our guidelines, we construct a new dataset comprising 1028 causal claims, 952 counterclaims, and 1435 uncausal statements, achieving substantial inter-annotator agreement (Cohen's $κ= 0.74$). In our experiments, state-of-the-art models trained solely on causal claims misclassify counterclaims more than 10 times as often as models trained on our dataset.
>
---
#### [replaced 040] ChartAgent: A Multimodal Agent for Visually Grounded Reasoning in Complex Chart Question Answering
- **分类: cs.AI; cs.CE; cs.CL; cs.CV; stat.ME**

- **简介: 该论文属于图表问答任务，旨在解决无标注图表的视觉推理问题。提出ChartAgent框架，通过视觉子任务分解和图表操作提升准确率。**

- **链接: [https://arxiv.org/pdf/2510.04514v2](https://arxiv.org/pdf/2510.04514v2)**

> **作者:** Rachneet Kaur; Nishan Srishankar; Zhen Zeng; Sumitra Ganesh; Manuela Veloso
>
> **备注:** NeurIPS 2025 Multimodal Algorithmic Reasoning Workshop (https://marworkshop.github.io/neurips25/) (Oral Paper Presentation)
>
> **摘要:** Recent multimodal LLMs have shown promise in chart-based visual question answering, but their performance declines sharply on unannotated charts-those requiring precise visual interpretation rather than relying on textual shortcuts. To address this, we introduce ChartAgent, a novel agentic framework that explicitly performs visual reasoning directly within the chart's spatial domain. Unlike textual chain-of-thought reasoning, ChartAgent iteratively decomposes queries into visual subtasks and actively manipulates and interacts with chart images through specialized actions such as drawing annotations, cropping regions (e.g., segmenting pie slices, isolating bars), and localizing axes, using a library of chart-specific vision tools to fulfill each subtask. This iterative reasoning process closely mirrors human cognitive strategies for chart comprehension. ChartAgent achieves state-of-the-art accuracy on the ChartBench and ChartX benchmarks, surpassing prior methods by up to 16.07% absolute gain overall and 17.31% on unannotated, numerically intensive queries. Furthermore, our analyses show that ChartAgent is (a) effective across diverse chart types, (b) achieves the highest scores across varying visual and reasoning complexity levels, and (c) serves as a plug-and-play framework that boosts performance across diverse underlying LLMs. Our work is among the first to demonstrate visually grounded reasoning for chart understanding using tool-augmented multimodal agents.
>
---
#### [replaced 041] DiFlow-TTS: Compact and Low-Latency Zero-Shot Text-to-Speech with Factorized Discrete Flow Matching
- **分类: cs.SD; cs.CL; cs.CV**

- **简介: 该论文提出DiFlow-TTS，一种零样本文本转语音系统，通过离散流匹配实现高效生成。解决语音合成自然度与速度问题，采用因子化表示和确定性映射器提升性能。**

- **链接: [https://arxiv.org/pdf/2509.09631v3](https://arxiv.org/pdf/2509.09631v3)**

> **作者:** Ngoc-Son Nguyen; Thanh V. T. Tran; Hieu-Nghia Huynh-Nguyen; Truong-Son Hy; Van Nguyen
>
> **摘要:** This paper introduces DiFlow-TTS, a novel zero-shot text-to-speech (TTS) system that employs discrete flow matching for generative speech modeling. We position this work as an entry point that may facilitate further advances in this research direction. Through extensive empirical evaluation, we analyze both the strengths and limitations of this approach across key aspects, including naturalness, expressive attributes, speaker identity, and inference latency. To this end, we leverage factorized speech representations and design a deterministic Phoneme-Content Mapper for modeling linguistic content, together with a Factorized Discrete Flow Denoiser that jointly models multiple discrete token streams corresponding to prosody and acoustics to capture expressive speech attributes. Experimental results demonstrate that DiFlow-TTS achieves strong performance across multiple metrics while maintaining a compact model size, up to 11.7 times smaller, and enabling low-latency inference that is up to 34 times faster than recent state-of-the-art baselines. Audio samples are available on our demo page: https://diflow-tts.github.io.
>
---
#### [replaced 042] InComeS: Integrating Compression and Selection Mechanisms into LLMs for Efficient Model Editing
- **分类: cs.CL**

- **简介: 该论文属于模型编辑任务，旨在解决LLMs在处理多编辑时效率低下的问题。提出InComeS框架，通过压缩和选择机制提升编辑效率与效果。**

- **链接: [https://arxiv.org/pdf/2505.22156v3](https://arxiv.org/pdf/2505.22156v3)**

> **作者:** Shuaiyi Li; Zhisong Zhang; Yang Deng; Chenlong Deng; Tianqing Fang; Hongming Zhang; Haitao Mi; Dong Yu; Wai Lam
>
> **备注:** 18 pages,5 figures
>
> **摘要:** Although existing model editing methods perform well in recalling exact edit facts, they often struggle in complex scenarios that require deeper semantic understanding rather than mere knowledge regurgitation. Leveraging the strong contextual reasoning abilities of large language models (LLMs), in-context learning (ICL) becomes a promising editing method by comprehending edit information through context encoding. However, this method is constrained by the limited context window of LLMs, leading to degraded performance and efficiency as the number of edits increases. To overcome this limitation, we propose InComeS, a flexible framework that enhances LLMs' ability to process editing contexts through explicit compression and selection mechanisms. Specifically, InComeS compresses each editing context into the key-value (KV) cache of a special gist token, enabling efficient handling of multiple edits without being restricted by the model's context window. Furthermore, specialized cross-attention modules are added to dynamically select the most relevant information from the gist pools, enabling adaptive and effective utilization of edit information. We conduct experiments on diverse model editing benchmarks with various editing formats, and the results demonstrate the effectiveness and efficiency of our method.
>
---
#### [replaced 043] LAG: Logic-Augmented Generation from a Cartesian Perspective
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识增强生成任务，旨在解决大模型在复杂推理中易产生幻觉的问题。通过逻辑分解与结构化推理，提升生成准确性。**

- **链接: [https://arxiv.org/pdf/2508.05509v3](https://arxiv.org/pdf/2508.05509v3)**

> **作者:** Yilin Xiao; Chuang Zhou; Yujing Zhang; Qinggang Zhang; Su Dong; Shengyuan Chen; Chang Yang; Xiao Huang
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet exhibit critical limitations in knowledge-intensive tasks, often generating hallucinations when faced with questions requiring specialized expertise. While retrieval-augmented generation (RAG) mitigates this by integrating external knowledge, it struggles with complex reasoning scenarios due to its reliance on direct semantic retrieval and lack of structured logical organization. Inspired by Cartesian principles from \textit{Discours de la méthode}, this paper introduces Logic-Augmented Generation (LAG), a novel paradigm that reframes knowledge augmentation through systematic question decomposition, atomic memory bank and logic-aware reasoning. Specifically, LAG first decomposes complex questions into atomic sub-questions ordered by logical dependencies. It then resolves these sequentially, using prior answers to guide context retrieval for subsequent sub-questions, ensuring stepwise grounding in the logical chain. Experiments on four benchmarks demonstrate that LAG significantly improves accuracy and reduces hallucination over existing methods.
>
---
#### [replaced 044] Are Vision Language Models Cross-Cultural Theory of Mind Reasoners?
- **分类: cs.CL; cs.CV; cs.CY**

- **简介: 该论文属于跨文化社会推理任务，旨在评估视觉语言模型的理论心智能力。研究构建了CulturalToM-VQA数据集，测试模型在不同文化背景下的社会理解能力，发现模型仍存在局限。**

- **链接: [https://arxiv.org/pdf/2512.17394v2](https://arxiv.org/pdf/2512.17394v2)**

> **作者:** Zabir Al Nazi; GM Shahariar; Md. Abrar Hossain; Wei Peng
>
> **摘要:** Theory of Mind (ToM) - the ability to attribute beliefs and intents to others - is fundamental for social intelligence, yet Vision-Language Model (VLM) evaluations remain largely Western-centric. In this work, we introduce CulturalToM-VQA, a benchmark of 5,095 visually situated ToM probes across diverse cultural contexts, rituals, and social norms. Constructed through a frontier proprietary MLLM, human-verified pipeline, the dataset spans a taxonomy of six ToM tasks and four complexity levels. We benchmark 10 VLMs (2023-2025) and observe a significant performance leap: while earlier models struggle, frontier models achieve high accuracy (>93%). However, significant limitations persist: models struggle with false belief reasoning (19-83% accuracy) and show high regional variance (20-30% gaps). Crucially, we find that SOTA models exhibit social desirability bias - systematically favoring semantically positive answer choices over negative ones. Ablation experiments reveal that some frontier models rely heavily on parametric social priors, frequently defaulting to safety-aligned predictions. Furthermore, while Chain-of-Thought prompting aids older models, it yields minimal gains for newer ones. Overall, our work provides a testbed for cross-cultural social reasoning, underscoring that despite architectural gains, achieving robust, visually grounded understanding remains an open challenge.
>
---
#### [replaced 045] Detecting PTSD in Clinical Interviews: A Comparative Analysis of NLP Methods and Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于情感分析任务，旨在检测临床访谈中的PTSD。通过比较不同NLP方法，发现领域适配模型表现更优，为AI辅助诊断提供参考。**

- **链接: [https://arxiv.org/pdf/2504.01216v3](https://arxiv.org/pdf/2504.01216v3)**

> **作者:** Feng Chen; Dror Ben-Zeev; Gillian Sparks; Arya Kadakia; Trevor Cohen
>
> **摘要:** Post-Traumatic Stress Disorder (PTSD) remains underdiagnosed in clinical settings, presenting opportunities for automated detection to identify patients. This study evaluates natural language processing approaches for detecting PTSD from clinical interview transcripts. We compared general and mental health-specific transformer models (BERT/RoBERTa), embedding-based methods (SentenceBERT/LLaMA), and large language model prompting strategies (zero-shot/few-shot/chain-of-thought) using the DAIC-WOZ dataset. Domain-specific end-to-end models significantly outperformed general models (Mental-RoBERTa AUPRC=0.675+/-0.084 vs. RoBERTa-base 0.599+/-0.145). SentenceBERT embeddings with neural networks achieved the highest overall performance (AUPRC=0.758+/-0.128). Few-shot prompting using DSM-5 criteria yielded competitive results with two examples (AUPRC=0.737). Performance varied significantly across symptom severity and comorbidity status with depression, with higher accuracy for severe PTSD cases and patients with comorbid depression. Our findings highlight the potential of domain-adapted embeddings and LLMs for scalable screening while underscoring the need for improved detection of nuanced presentations and offering insights for developing clinically viable AI tools for PTSD assessment.
>
---
#### [replaced 046] SPIO: Ensemble and Selective Strategies via LLM-Based Multi-Agent Planning in Automated Data Science
- **分类: cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文提出SPIO框架，解决自动化数据科学中的流程僵化问题，通过多路径规划与集成提升模型性能。**

- **链接: [https://arxiv.org/pdf/2503.23314v2](https://arxiv.org/pdf/2503.23314v2)**

> **作者:** Wonduk Seo; Juhyeon Lee; Yanjun Shao; Qingshan Zhou; Seunghyun Lee; Yi Bu
>
> **备注:** Under Review
>
> **摘要:** Large Language Models (LLMs) have enabled dynamic reasoning in automated data analytics, yet recent multi-agent systems remain limited by rigid, single-path workflows that restrict strategic exploration and often lead to suboptimal outcomes. To overcome these limitations, we propose SPIO (Sequential Plan Integration and Optimization), a framework that replaces rigid workflows with adaptive, multi-path planning across four core modules: data preprocessing, feature engineering, model selection, and hyperparameter tuning. In each module, specialized agents generate diverse candidate strategies, which are cascaded and refined by an optimization agent. SPIO offers two operating modes: SPIO-S for selecting a single optimal pipeline, and SPIO-E for ensembling top-k pipelines to maximize robustness. Extensive evaluations on Kaggle and OpenML benchmarks show that SPIO consistently outperforms state-of-the-art baselines, achieving an average performance gain of 5.6%. By explicitly exploring and integrating multiple solution paths, SPIO delivers a more flexible, accurate, and reliable foundation for automated data science.
>
---
#### [replaced 047] League of LLMs: A Benchmark-Free Paradigm for Mutual Evaluation of Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出LOL框架，用于无需基准的大型语言模型互评，解决评估不可靠问题。通过多轮互评，提升评估的客观性与透明度。**

- **链接: [https://arxiv.org/pdf/2507.22359v3](https://arxiv.org/pdf/2507.22359v3)**

> **作者:** Qianhong Guo; Wei Xie; Xiaofang Cai; Enze Wang; Shuoyoucheng Ma; Xiaobing Sun; Tian Xia; Kai Chen; Xiaofeng Wang; Baosheng Wang
>
> **摘要:** Although large language models (LLMs) have shown exceptional capabilities across a wide range of tasks, reliable evaluation remains a critical challenge due to data contamination, opaque operation, and subjective preferences. To address these issues, we propose League of LLMs (LOL), a novel benchmark-free evaluation paradigm that organizes multiple LLMs into a self-governed league for multi-round mutual evaluation. LOL integrates four core criteria (dynamic, transparent, objective, and professional) to mitigate key limitations of existing paradigms. Experiments on eight mainstream LLMs in mathematics and programming demonstrate that LOL can effectively distinguish LLM capabilities while maintaining high internal ranking stability (Top-$k$ consistency $= 70.7\%$). Beyond ranking, LOL reveals empirical findings that are difficult for traditional paradigms to capture. For instance, ``memorization-based answering'' behaviors are observed in some models, and a statistically significant homophily bias is found within the OpenAI family ($Δ= 9$, $p < 0.05$). Finally, we make our framework and code publicly available as a valuable complement to the current LLM evaluation ecosystem.
>
---
#### [replaced 048] DyBBT: Dynamic Balance via Bandit inspired Targeting for Dialog Policy with Cognitive Dual-Systems
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出DyBBT，解决对话系统中探索策略静态导致效率低的问题。通过动态切换推理模式，提升对话策略性能。**

- **链接: [https://arxiv.org/pdf/2509.19695v2](https://arxiv.org/pdf/2509.19695v2)**

> **作者:** Shuyu Zhang; Yifan Wei; Jialuo Yuan; Xinru Wang; Yanmin Zhu; Bin Li; Yujie Liu
>
> **摘要:** Task oriented dialog systems often rely on static exploration strategies that do not adapt to dynamic dialog contexts, leading to inefficient exploration and suboptimal performance. We propose DyBBT, a novel dialog policy learning framework that formalizes the exploration challenge through a structured cognitive state space capturing dialog progression, user uncertainty, and slot dependency. DyBBT proposes a bandit inspired meta-controller that dynamically switches between a fast intuitive inference (System 1) and a slow deliberative reasoner (System 2) based on real-time cognitive states and visitation counts. Extensive experiments on single- and multi-domain benchmarks show that DyBBT achieves state-of-the-art performance in success rate, efficiency, and generalization, with human evaluations confirming its decisions are well aligned with expert judgment. Code is available at https://github.com/carsonz/DyBBT.
>
---
#### [replaced 049] Big Reasoning with Small Models: Instruction Retrieval at Inference Time
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升小模型的推理能力。针对小模型在需要专业知识或多步骤推理的任务中表现不足的问题，提出指令检索方法，在推理时引入结构化指导，提升模型准确性。**

- **链接: [https://arxiv.org/pdf/2510.13935v2](https://arxiv.org/pdf/2510.13935v2)**

> **作者:** Kenan Alkiek; David Jurgens; Vinod Vydiswaran
>
> **摘要:** Small language models (SLMs) enable low-cost, private, on-device inference, but they often fail on problems that require specialized domain knowledge or multi-step reasoning. Existing approaches for improving reasoning either rely on scale (e.g., chain-of-thought prompting), require task-specific training that limits reuse and generality (e.g., distillation), or retrieve unstructured information that still leaves the SLM to determine an appropriate reasoning strategy. We propose instruction retrieval, an inference-time intervention that augments an SLM with structured, reusable reasoning procedures rather than raw passages. We construct an Instruction Corpus by clustering similar training questions and using a teacher model to generate generalizable guides that pair domain background with explicit step-by-step procedures. At inference, the SLM retrieves the instructions most relevant to a given query and executes the associated procedures without any additional fine-tuning. Across three challenging domains: medicine, law, and mathematics, instruction retrieval yields consistent gains for models with at least 3B parameters, improving accuracy by 9.4%, 7.9%, and 5.1%, respectively, with the strongest 14B model surpassing GPT-4o's zero-shot performance on knowledge-intensive tasks.
>
---
#### [replaced 050] WebAnchor: Anchoring Agent Planning to Stabilize Long-Horizon Web Reasoning
- **分类: cs.CL**

- **简介: 该论文属于Web推理任务，旨在解决长距离规划不稳定问题。提出Anchor-GRPO框架，通过分阶段优化提升任务成功率和工具使用效率。**

- **链接: [https://arxiv.org/pdf/2601.03164v2](https://arxiv.org/pdf/2601.03164v2)**

> **作者:** Xinmiao Yu; Liwen Zhang; Xiaocheng Feng; Yong Jiang; Bing Qin; Pengjun Xie; Jingren Zhou
>
> **摘要:** Large Language Model(LLM)-based agents have shown strong capabilities in web information seeking, with reinforcement learning (RL) becoming a key optimization paradigm. However, planning remains a bottleneck, as existing methods struggle with long-horizon strategies. Our analysis reveals a critical phenomenon, plan anchor, where the first reasoning step disproportionately impacts downstream behavior in long-horizon web reasoning tasks. Current RL algorithms, fail to account for this by uniformly distributing rewards across the trajectory. To address this, we propose Anchor-GRPO, a two-stage RL framework that decouples planning and execution. In Stage 1, the agent optimizes its first-step planning using fine-grained rubrics derived from self-play experiences and human calibration. In Stage 2, execution is aligned with the initial plan through sparse rewards, ensuring stable and efficient tool usage. We evaluate Anchor-GRPO on four benchmarks: BrowseComp, BrowseComp-Zh, GAIA, and XBench-DeepSearch. Across models from 3B to 30B, Anchor-GRPO outperforms baseline GRPO and First-step GRPO, improving task success and tool efficiency. Notably, WebAnchor-30B achieves 46.0% pass@1 on BrowseComp and 76.4% on GAIA. Anchor-GRPO also demonstrates strong scalability, getting higher accuracy as model size and context length increase.
>
---
#### [replaced 051] LayerNorm Induces Recency Bias in Transformer Decoders
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究Transformer解码器中的位置偏差问题。论文揭示LayerNorm与因果自注意力结合导致近期偏差，分析其机制并提出改进方向。**

- **链接: [https://arxiv.org/pdf/2509.21042v2](https://arxiv.org/pdf/2509.21042v2)**

> **作者:** Junu Kim; Xiao Liu; Zhenghao Lin; Lei Ji; Yeyun Gong; Edward Choi
>
> **备注:** Codes available at: https://github.com/starmpcc/layernorm_recency_bias
>
> **摘要:** Causal self-attention provides positional information to Transformer decoders. Prior work has shown that stacks of causal self-attention layers alone induce a positional bias in attention scores toward earlier tokens. However, this differs from the bias toward later tokens typically observed in Transformer decoders, known as recency bias. We address this discrepancy by analyzing the interaction between causal self-attention and other architectural components. We show that stacked causal self-attention layers combined with LayerNorm induce recency bias. Furthermore, we examine the effects of residual connections and the distribution of input token embeddings on this bias. Our results provide new theoretical insights into how positional information interacts with architectural components and suggest directions for improving positional encoding strategies.
>
---
#### [replaced 052] Empirical Comparison of Encoder-Based Language Models and Feature-Based Supervised Machine Learning Approaches to Automated Scoring of Long Essays
- **分类: cs.CL; cs.LG**

- **简介: 论文研究自动化评分任务，针对长作文评分中编码器模型的挑战，比较了语言模型与集成学习方法，发现融合多模型嵌入的集成方法效果最佳。**

- **链接: [https://arxiv.org/pdf/2601.02659v2](https://arxiv.org/pdf/2601.02659v2)**

> **作者:** Kuo Wang; Haowei Hua; Pengfei Yan; Hong Jiao; Dan Song
>
> **备注:** 22 pages, 5 figures, 3 tables, presented at National Council on Measurement in Education 2025
>
> **摘要:** Long context may impose challenges for encoder-only language models in text processing, specifically for automated scoring of essays. This study trained several commonly used encoder-based language models for automated scoring of long essays. The performance of these trained models was evaluated and compared with the ensemble models built upon the base language models with a token limit of 512?. The experimented models include BERT-based models (BERT, RoBERTa, DistilBERT, and DeBERTa), ensemble models integrating embeddings from multiple encoder models, and ensemble models of feature-based supervised machine learning models, including Gradient-Boosted Decision Trees, eXtreme Gradient Boosting, and Light Gradient Boosting Machine. We trained, validated, and tested each model on a dataset of 17,307 essays, with an 80%/10%/10% split, and evaluated model performance using Quadratic Weighted Kappa. This study revealed that an ensemble-of-embeddings model that combines multiple pre-trained language model representations with gradient-boosting classifier as the ensemble model significantly outperforms individual language models at scoring long essays.
>
---
#### [replaced 053] The Gray Area: Characterizing Moderator Disagreement on Reddit
- **分类: cs.CY; cs.CL; cs.IT**

- **简介: 该论文研究在线内容审核中的争议问题，属于内容 moderation 任务。它分析了Reddit moderators之间的分歧，探讨了争议案例的特点与处理难度。**

- **链接: [https://arxiv.org/pdf/2601.01620v2](https://arxiv.org/pdf/2601.01620v2)**

> **作者:** Shayan Alipour; Shruti Phadke; Seyed Shahabeddin Mousavi; Amirhossein Afsharrad; Morteza Zihayat; Mattia Samory
>
> **备注:** Accepted at ICWSM 2026
>
> **摘要:** Volunteer moderators play a crucial role in sustaining online dialogue, but they often disagree about what should or should not be allowed. In this paper, we study the complexity of content moderation with a focus on disagreements between moderators, which we term the ``gray area'' of moderation. Leveraging 5 years and 4.3 million moderation log entries from 24 subreddits of different topics and sizes, we characterize how gray area, or disputed cases, differ from undisputed cases. We show that one-in-seven moderation cases are disputed among moderators, often addressing transgressions where users' intent is not directly legible, such as in trolling and brigading, as well as tensions around community governance. This is concerning, as almost half of all gray area cases involved automated moderation decisions. Through information-theoretic evaluations, we demonstrate that gray area cases are inherently harder to adjudicate than undisputed cases and show that state-of-the-art language models struggle to adjudicate them. We highlight the key role of expert human moderators in overseeing the moderation process and provide insights about the challenges of current moderation processes and tools.
>
---
#### [replaced 054] Monadic Context Engineering
- **分类: cs.AI; cs.CL; cs.FL**

- **简介: 该论文提出Monadic Context Engineering（MCE），解决AI代理架构的复杂性问题，通过函数式编程结构提升系统稳定性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.22431v2](https://arxiv.org/pdf/2512.22431v2)**

> **作者:** Yifan Zhang; Yang Yuan; Mengdi Wang; Andrew Chi-Chih Yao
>
> **摘要:** The proliferation of Large Language Models (LLMs) has catalyzed a shift towards autonomous agents capable of complex reasoning and tool use. However, current agent architectures are frequently constructed using imperative, ad hoc patterns. This results in brittle systems plagued by difficulties in state management, error handling, and concurrency. This paper introduces Monadic Context Engineering (MCE), a novel architectural paradigm leveraging the algebraic structures of Functors, Applicative Functors, and Monads to provide a formal foundation for agent design. MCE treats agent workflows as computational contexts where cross-cutting concerns, such as state propagation, short-circuiting error handling, and asynchronous execution, are managed intrinsically by the algebraic properties of the abstraction. We demonstrate how Monads enable robust sequential composition, how Applicatives provide a principled structure for parallel execution, and crucially, how Monad Transformers allow for the systematic composition of these capabilities. This layered approach enables developers to construct complex, resilient, and efficient AI agents from simple, independently verifiable components. We further extend this framework to describe Meta-Agents, which leverage MCE for generative orchestration, dynamically creating and managing sub-agent workflows through metaprogramming.
>
---
#### [replaced 055] Shared Path: Unraveling Memorization in Multilingual LLMs through Language Similarities
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言大模型研究任务，旨在解决多语言模型中记忆行为的分析问题。通过引入语言相似性，揭示记忆模式与语言关系的关联，提出新的分析方法以评估和缓解记忆漏洞。**

- **链接: [https://arxiv.org/pdf/2505.15722v2](https://arxiv.org/pdf/2505.15722v2)**

> **作者:** Xiaoyu Luo; Yiyi Chen; Johannes Bjerva; Qiongxiu Li
>
> **备注:** 17 pages, 14 tables, 10 figures
>
> **摘要:** We present the first comprehensive study of Memorization in Multilingual Large Language Models (MLLMs), analyzing 95 languages using models across diverse model scales, architectures, and memorization definitions. As MLLMs are increasingly deployed, understanding their memorization behavior has become critical. Yet prior work has focused primarily on monolingual models, leaving multilingual memorization underexplored, despite the inherently long-tailed nature of training corpora. We find that the prevailing assumption, that memorization is highly correlated with training data availability, fails to fully explain memorization patterns in MLLMs. We hypothesize that the conventional focus on monolingual settings, effectively treating languages in isolation, may obscure the true patterns of memorization. To address this, we propose a novel graph-based correlation metric that incorporates language similarity to analyze cross-lingual memorization. Our analysis reveals that among similar languages, those with fewer training tokens tend to exhibit higher memorization, a trend that only emerges when cross-lingual relationships are explicitly modeled. These findings underscore the importance of a \textit{language-aware} perspective in evaluating and mitigating memorization vulnerabilities in MLLMs. This also constitutes empirical evidence that language similarity both explains Memorization in MLLMs and underpins Cross-lingual Transferability, with broad implications for multilingual NLP.
>
---
#### [replaced 056] Reward Is Enough: LLMs Are In-Context Reinforcement Learners
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文探讨大语言模型在推理过程中表现出的强化学习能力，属于自然语言处理任务。研究解决如何让模型在推理时自我优化的问题，通过引入ICRL提示框架实现任务性能提升。**

- **链接: [https://arxiv.org/pdf/2506.06303v4](https://arxiv.org/pdf/2506.06303v4)**

> **作者:** Kefan Song; Amir Moeini; Peng Wang; Lei Gong; Rohan Chandra; Shangtong Zhang; Yanjun Qi
>
> **摘要:** Reinforcement learning (RL) is a framework for solving sequential decision-making problems. In this work, we demonstrate that, surprisingly, RL emerges during the inference time of large language models (LLMs), a phenomenon we term in-context RL (ICRL). To reveal this capability, we introduce a simple multi-round prompting framework, we call ICRL prompting, for inference-time self-improvement. The goal of ICRL prompting is to guide LLMs to perform reinforcement learning during inference for self-improvement on a given task. After each response, the model receives numerical scalar feedback, denoted as a reward. In the next round, we prompt the LLM again together with a context that concatenates all prior responses and their associated rewards. We consistently observe that response quality improves as the context grows. In other words, the LLM can optimize scalar reward signals during inference, exhibiting behavior analogous to reinforcement learning. We evaluate ICRL prompting on Game of 24, creative writing, ScienceWorld, and Olympiad-level math competitions (AIME and HMMT), demonstrating significant improvements over baselines such as Self-Refine and Reflexion. Notably, even when the reward signals are generated by the same LLM, ICRL prompting still improves performance, highlighting a promising new paradigm for test-time scaling.
>
---
#### [replaced 057] DeepSearch: Overcome the Bottleneck of Reinforcement Learning with Verifiable Rewards via Monte Carlo Tree Search
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决RLVR训练中探索不足导致的性能瓶颈。通过引入MCTS进行系统性探索，提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2509.25454v3](https://arxiv.org/pdf/2509.25454v3)**

> **作者:** Fang Wu; Weihao Xuan; Heli Qi; Ximing Lu; Aaron Tu; Li Erran Li; Yejin Choi
>
> **摘要:** Although RLVR has become an essential component for developing advanced reasoning skills in language models, contemporary studies have documented training plateaus after thousands of optimization steps, i.e., notable decreases in performance gains despite increased computational investment. This limitation stems from the sparse exploration patterns inherent in current RLVR practices, where models rely on limited rollouts that often miss critical reasoning paths and fail to provide systematic coverage of the solution space. We present DeepSearch, a framework that integrates Monte Carlo Tree Search (MCTS) directly into RLVR training. In contrast to existing methods that rely on tree search only at inference, DeepSearch embeds structured search into the training loop, enabling systematic exploration and fine-grained credit assignment across reasoning steps. Through training-time exploration, DeepSearch addresses the fundamental bottleneck of insufficient exploration, which leads to diminishing performance improvements over prolonged training steps. Our contributions include: (1) a global frontier selection strategy that prioritizes promising nodes across the search tree, (2) selection with entropy-based guidance that identifies confident paths for supervision, and (3) adaptive replay buffer training with solution caching for efficiency. Experiments on mathematical reasoning benchmarks show that DeepSearch achieves 62.95% average accuracy and establishes a new state-of-the-art for 1.5B reasoning models, while using 5.7x fewer GPU hours than extended training approaches. These results highlight the importance of strategic exploration over brute-force scaling and demonstrate the promise of algorithmic innovation for advancing RLVR methodologies. DeepSearch establishes a new direction for scaling reasoning capabilities through systematic search rather than prolonged computation.
>
---
#### [replaced 058] HAL: Inducing Human-likeness in LLMs with Alignment
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出HAL框架，用于提升语言模型的对话人类相似性。任务是改善人机交互中的自然度，解决难以定义和优化人类相似性的问题，通过可解释的奖励机制进行对齐。**

- **链接: [https://arxiv.org/pdf/2601.02813v2](https://arxiv.org/pdf/2601.02813v2)**

> **作者:** Masum Hasan; Junjie Zhao; Ehsan Hoque
>
> **摘要:** Conversational human-likeness plays a central role in human-AI interaction, yet it has remained difficult to define, measure, and optimize. As a result, improvements in human-like behavior are largely driven by scale or broad supervised training, rather than targeted alignment. We introduce Human Aligning LLMs (HAL), a framework for aligning language models to conversational human-likeness using an interpretable, data-driven reward. HAL derives explicit conversational traits from contrastive dialogue data, combines them into a compact scalar score, and uses this score as a transparent reward signal for alignment with standard preference optimization methods. Using this approach, we align models of varying sizes without affecting their overall performance. In large-scale human evaluations, models aligned with HAL are more frequently perceived as human-like in conversation. Because HAL operates over explicit, interpretable traits, it enables inspection of alignment behavior and diagnosis of unintended effects. More broadly, HAL demonstrates how soft, qualitative properties of language--previously outside the scope for alignment--can be made measurable and aligned in an interpretable and explainable way.
>
---
#### [replaced 059] PM4Bench: Benchmarking Large Vision-Language Models with Parallel Multilingual Multi-Modal Multi-task Corpus
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态语言模型评估任务，旨在解决现有基准测试中语料不平行和多模态输入分离的问题。提出PM4Bench，使用平行语料进行多语言、多任务评估，并引入视觉融合文本的测试场景。**

- **链接: [https://arxiv.org/pdf/2503.18484v2](https://arxiv.org/pdf/2503.18484v2)**

> **作者:** Junyuan Gao; Jiahe Song; Jiang Wu; Runchuan Zhu; Guanlin Shen; Shasha Wang; Xingjian Wei; Haote Yang; Songyang Zhang; Weijia Li; Bin Wang; Dahua Lin; Lijun Wu; Conghui He
>
> **备注:** Equal contribution: Junyuan Gao, Jiahe Song, Jiang Wu; Corresponding author: Conghui He
>
> **摘要:** While Large Vision-Language Models (LVLMs) demonstrate promising multilingual capabilities, their evaluation is currently hindered by two critical limitations: (1) the use of non-parallel corpora, which conflates inherent language capability gaps with dataset artifacts, precluding a fair assessment of cross-lingual alignment; and (2) disjointed multimodal inputs, which deviate from real-world scenarios where most texts are embedded within visual contexts. To address these challenges, we propose PM4Bench, the first Multilingual Multi-Modal Multi-task Benchmark constructed on a strictly parallel corpus across 10 languages. By eliminating content divergence, our benchmark enables a fair comparison of model capabilities across different languages. We also introduce a vision setting where textual queries are visually fused into images, compelling models to jointly "see," "read," and "think". Extensive evaluation of 10 LVLMs uncover a substantial performance drop in the Vision setting compared to standard inputs. Further analysis reveals that OCR capability is not only a general bottleneck but also contributes to cross-lingual performance disparities, suggesting that improving multilingual OCR is essential for advancing LVLM performance. We will release PM4Bench at https://github.com/opendatalab/PM4Bench .
>
---
#### [replaced 060] Quantifying LLM Biases Across Instruction Boundary in Mixed Question Forms
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM在混合题型数据集中的偏差问题。研究提出Instruction Boundary概念和BiasDetector基准，评估不同指令对模型识别稀疏标签的影响。**

- **链接: [https://arxiv.org/pdf/2509.20278v4](https://arxiv.org/pdf/2509.20278v4)**

> **作者:** Zipeng Ling; Shuliang Liu; Yuehao Tang; Chen Huang; Gaoyang Jiang; Shenghong Fu; Junqi Yang; Yao Wan; Jiawan Zhang; Kejia Huang; Xuming Hu
>
> **摘要:** Large Language Models (LLMs) annotated datasets are widely used nowadays, however, large-scale annotations often show biases in low-quality datasets. For example, Multiple-Choice Questions (MCQs) datasets with one single correct option is common, however, there may be questions attributed to none or multiple correct options; whereas true-or-false questions are supposed to be labeled with either True or False, but similarly the text can include unsolvable elements, which should be further labeled as Unknown. There are problems when low-quality datasets with mixed question forms can not be identified. We refer to these exceptional label forms as Sparse Labels, and LLMs' ability to distinguish datasets with Sparse Labels mixture is important. Since users may not know situations of datasets, their instructions can be biased. To study how different instruction settings affect LLMs' identifications of Sparse Labels mixture, we introduce the concept of Instruction Boundary, which systematically evaluates different instruction settings that lead to biases. We propose BiasDetector, a diagnostic benchmark to systematically evaluate LLMs on datasets with mixed question forms under Instruction Boundary settings. Experiments show that users' instructions induce large biases on our benchmark, highlighting the need not only for LLM developers to recognize risks of LLM biased annotation resulting in Sparse Labels mixture, but also problems arising from users' instructions to identify them. Code, datasets and detailed implementations are available at https://github.com/ZpLing/Instruction-Boundary.
>
---
#### [replaced 061] The Invisible Leash: Why RLVR May or May Not Escape Its Origin
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于AI模型优化任务，探讨RLVR是否真正扩展模型推理能力。研究发现RLVR可能受限于基础模型分布，导致探索不足，需创新算法突破限制。**

- **链接: [https://arxiv.org/pdf/2507.14843v3](https://arxiv.org/pdf/2507.14843v3)**

> **作者:** Fang Wu; Weihao Xuan; Ximing Lu; Mingjie Liu; Yi Dong; Zaid Harchaoui; Yejin Choi
>
> **摘要:** Recent advances in LLMs highlight Reinforcement Learning with Verifiable Rewards (RLVR) as a promising method for enhancing AI capabilities, particularly in solving complex logical tasks. However, it remains unclear whether the current practice of RLVR truly expands a model's reasoning boundary or mainly amplifies high-reward outputs that the base model already knows, leading to improved precision. This study presents an empirical investigation that provides new insights into the potential limits of the common RLVR recipe. We examine how, under current training conditions, RLVR can operate as a support-constrained optimization mechanism that may restrict the discovery of entirely novel solutions, remaining constrained by the base model's initial distribution. We also identify an entropy-reward trade-off: while the current RLVR recipe reliably enhances precision, it may progressively narrow exploration and potentially overlook correct yet underrepresented solutions. Extensive empirical experiments show that although the current RLVR recipe consistently improves pass@1, the shrinkage of empirical support generally outweighs the expansion of empirical support under larger sampling budgets, failing to recover correct answers that were previously accessible to the base model. Interestingly, we also observe that while RLVR sometimes increases token-level entropy, it leads to greater uncertainty at each generation step but declining answer-level entropy. This suggests that these seemingly more uncertain generation paths ultimately converge onto a smaller set of distinct answers. Taken together, our findings reveal potential limits of the current RLVR recipe in extending reasoning horizons. Breaking this invisible leash may require future algorithmic innovations, such as explicit exploration mechanisms or hybrid strategies that allocate probability mass to underrepresented solution regions.
>
---
#### [replaced 062] PilotRL: Training Language Model Agents via Global Planning-Guided Progressive Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于智能体任务，解决LLM在复杂任务中缺乏长期规划与协调的问题。提出AdaPlan和PilotRL框架，提升模型的规划与执行能力。**

- **链接: [https://arxiv.org/pdf/2508.00344v4](https://arxiv.org/pdf/2508.00344v4)**

> **作者:** Keer Lu; Chong Chen; Xili Wang; Bin Cui; Yunhuai Liu; Wentao Zhang
>
> **摘要:** Large Language Models (LLMs) have shown remarkable advancements in tackling agent-oriented tasks. Despite their potential, existing work faces challenges when deploying LLMs in agent-based environments. The widely adopted agent paradigm ReAct centers on integrating single-step reasoning with immediate action execution, which limits its effectiveness in complex tasks requiring long-term strategic planning. Furthermore, the coordination between the planner and executor during problem-solving is also a critical factor to consider in agent design. Additionally, current approaches predominantly rely on supervised fine-tuning, which often leads models to memorize established task completion trajectories, thereby restricting their generalization ability when confronted with novel problem contexts. To address these challenges, we introduce an adaptive global plan-based agent paradigm AdaPlan, aiming to synergize high-level explicit guidance with execution to support effective long-horizon decision-making. Based on the proposed paradigm, we further put forward PilotRL, a global planning-guided training framework for LLM agents driven by progressive reinforcement learning. We first develop the model's ability to follow explicit guidance from global plans when addressing agent tasks. Subsequently, based on this foundation, we focus on optimizing the quality of generated plans. Finally, we conduct joint optimization of the model's planning and execution coordination. Experiments indicate that PilotRL could achieve state-of-the-art performances, with LLaMA3.1-8B-Instruct + PilotRL surpassing closed-sourced GPT-4o by 3.60%, while showing a more substantial gain of 55.78% comparing to GPT-4o-mini at a comparable parameter scale.
>
---
#### [replaced 063] MoE-DiffuSeq: Enhancing Long-Document Diffusion Models with Sparse Attention and Mixture of Experts
- **分类: cs.CL**

- **简介: 该论文提出MoE-DiffuSeq，用于长文本生成任务，解决长文档扩散模型计算成本高、效率低的问题，通过稀疏注意力和专家混合架构提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2512.20604v2](https://arxiv.org/pdf/2512.20604v2)**

> **作者:** Alexandros Christoforos; Chadbourne Davis
>
> **备注:** Under submission
>
> **摘要:** We propose \textbf{MoE-DiffuSeq}, a diffusion-based framework for efficient long-form text generation that integrates sparse attention with a Mixture-of-Experts (MoE) architecture. Existing sequence diffusion models suffer from prohibitive computational and memory costs when scaling to long documents, largely due to dense attention and slow iterative reconstruction. MoE-DiffuSeq addresses these limitations by combining expert routing with a tailored sparse attention mechanism, substantially reducing attention complexity while preserving global coherence and textual fidelity. In addition, we introduce a \emph{soft absorbing state} within the diffusion process that reshapes attention dynamics during denoising, enabling faster sequence reconstruction and more precise token refinement. This design accelerates both training and sampling without sacrificing generation quality. Extensive experiments on long-document benchmarks demonstrate that MoE-DiffuSeq consistently outperforms prior diffusion-based and sparse-attention baselines in training efficiency, inference speed, and generation quality. Our approach is particularly effective for long-context applications such as scientific document generation, large-scale code synthesis, and extended dialogue modeling, establishing a scalable and expressive solution for diffusion-based long-form text generation.
>
---
#### [replaced 064] A Systematic Comparison between Extractive Self-Explanations and Human Rationales in Text Classification
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究文本分类任务中，比较自解释与人类理由的合理性与忠实性。通过分析不同模型的解释，探讨其与人类注释的一致性及有效性。**

- **链接: [https://arxiv.org/pdf/2410.03296v3](https://arxiv.org/pdf/2410.03296v3)**

> **作者:** Stephanie Brandl; Oliver Eberle
>
> **备注:** preprint
>
> **摘要:** Instruction-tuned LLMs are able to provide \textit{an} explanation about their output to users by generating self-explanations, without requiring the application of complex interpretability techniques. In this paper, we analyse whether this ability results in a \textit{good} explanation. We evaluate self-explanations in the form of input rationales with respect to their plausibility to humans. We study three text classification tasks: sentiment classification, forced labour detection and claim verification. We include Danish and Italian translations of the sentiment classification task and compare self-explanations to human annotations. For this, we collected human rationale annotations for Climate-Fever, a claim verification dataset. We furthermore evaluate the faithfulness of human and self-explanation rationales with respect to correct model predictions, and extend the study by incorporating post-hoc attribution-based explanations. We analyse four open-weight LLMs and find that alignment between self-explanations and human rationales highly depends on text length and task complexity. Nevertheless, self-explanations yield faithful subsets of token-level rationales, whereas post-hoc attribution methods tend to emphasize structural and formatting tokens, reflecting fundamentally different explanation strategies.
>
---
#### [replaced 065] Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在引入新知识时产生的事实幻觉问题，通过设计数据集和分析不同任务，揭示幻觉机制并提出缓解方法。**

- **链接: [https://arxiv.org/pdf/2511.02626v2](https://arxiv.org/pdf/2511.02626v2)**

> **作者:** Renfei Dang; Peng Hu; Zhejian Lai; Changjiang Gao; Min Zhang; Shujian Huang
>
> **摘要:** Prior works have shown that fine-tuning on new knowledge can induce factual hallucinations in large language models (LLMs), leading to incorrect outputs when evaluated on previously known information. However, the specific manifestations of such hallucination and its underlying mechanisms remain insufficiently understood. Our work addresses this gap by designing a controlled dataset \textit{Biography-Reasoning}, and conducting a fine-grained analysis across multiple knowledge types and two task types, including knowledge question answering (QA) and knowledge reasoning tasks. We find that hallucinations not only severely affect tasks involving newly introduced knowledge, but also propagate to other evaluation tasks. Moreover, when fine-tuning on a dataset in which a specific knowledge type consists entirely of new knowledge, LLMs exhibit elevated hallucination tendencies. This suggests that the degree of unfamiliarity within a particular knowledge type, rather than the overall proportion of new knowledge, is a stronger driver of hallucinations. Through interpretability analysis, we show that learning new knowledge weakens the model's attention to key entities in the input question, leading to an over-reliance on surrounding context and a higher risk of hallucination. Conversely, reintroducing a small amount of known knowledge during the later stages of training restores attention to key entities and substantially mitigates hallucination behavior. Finally, we demonstrate that disrupted attention patterns can propagate across lexically similar contexts, facilitating the spread of hallucinations beyond the original task.
>
---
#### [replaced 066] Task Matters: Knowledge Requirements Shape LLM Responses to Context-Memory Conflict
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM在上下文与记忆冲突下的表现，属于自然语言处理任务。解决任务依赖知识类型差异导致的性能问题，通过实验分析不同策略的影响。**

- **链接: [https://arxiv.org/pdf/2506.06485v3](https://arxiv.org/pdf/2506.06485v3)**

> **作者:** Kaiser Sun; Fan Bai; Mark Dredze
>
> **备注:** Major revision
>
> **摘要:** Large language models (LLMs) draw on both contextual information and parametric memory, yet these sources can conflict. Prior studies have largely examined this issue in contextual question answering, implicitly assuming that tasks should rely on the provided context, leaving unclear how LLMs behave when tasks require different types and degrees of knowledge utilization. We address this gap with a model-agnostic diagnostic framework that holds underlying knowledge constant while introducing controlled conflicts across tasks with varying knowledge demands. Experiments on representative open-source LLMs show that performance degradation under conflict is driven by both task-specific knowledge reliance and conflict plausibility; that strategies such as rationales or context reiteration increase context reliance, helping context-only tasks but harming those requiring parametric knowledge; and that these effects bias model-based evaluation, calling into question the reliability of LLMs as judges. Overall, our findings reveal that context-memory conflict is inherently task-dependent and motivate task-aware approaches to balancing context and memory in LLM deployment and evaluation.
>
---
#### [replaced 067] DRA-GRPO: Your GRPO Needs to Know Diverse Reasoning Paths for Mathematical Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于数学推理任务，解决GRPO方法中因奖励不敏感导致的多样性不足问题，提出DRA框架通过语义密度调整奖励，提升策略多样性与性能。**

- **链接: [https://arxiv.org/pdf/2505.09655v3](https://arxiv.org/pdf/2505.09655v3)**

> **作者:** Xiwen Chen; Wenhui Zhu; Peijie Qiu; Xuanzhao Dong; Hao Wang; Haiyu Wu; Huayu Li; Aristeidis Sotiras; Yalin Wang; Abolfazl Razi
>
> **摘要:** Post-training LLMs with Reinforcement Learning, specifically Group Relative Policy Optimization (GRPO), has emerged as a paradigm for enhancing mathematical reasoning. However, standard GRPO relies on scalar correctness rewards that are often non-injective with respect to semantic content: distinct reasoning paths receive identical rewards. This leads to a Diversity-Quality Inconsistency, where the policy collapses into a narrow set of dominant modes while ignoring equally valid but structurally novel strategies. To bridge this gap, we propose Diversity-aware Reward Adjustment (DRA), a theoretically grounded framework that calibrates the reward signal using the semantic density of sampled groups. By leveraging Submodular Mutual Information (SMI), DRA implements an Inverse Propensity Scoring (IPS) mechanism that effectively de-biases the gradient estimation. This creates a repulsive force against redundancy, driving the policy to achieve better coverage of the high-reward landscape. Our method is plug-and-play and integrates seamlessly with GRPO variants. Empirical evaluations on five math benchmarks demonstrate that DRA-GRPO consistently outperforms strong baselines, achieving an average accuracy of 58.2% on DeepSeek-R1-Distill-Qwen-1.5B with only 7,000 training samples and $55 cost, highlighting the critical role of diversity calibration in data-efficient alignment.
>
---
#### [replaced 068] Jailbreaking Commercial Black-Box LLMs with Explicitly Harmful Prompts
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于模型攻击任务，旨在提升对推理型大模型的越狱能力。通过整合多种技巧构建DH-CoT攻击方法，并提出MDH数据清洗框架以提高评估准确性。**

- **链接: [https://arxiv.org/pdf/2508.10390v3](https://arxiv.org/pdf/2508.10390v3)**

> **作者:** Chiyu Zhang; Lu Zhou; Xiaogang Xu; Jiafei Wu; Liming Fang; Zhe Liu
>
> **摘要:** Existing black-box jailbreak attacks achieve certain success on non-reasoning models but degrade significantly on recent SOTA reasoning models. To improve attack ability, inspired by adversarial aggregation strategies, we integrate multiple jailbreak tricks into a single developer template. Especially, we apply Adversarial Context Alignment to purge semantic inconsistencies and use NTP (a type of harmful prompt) -based few-shot examples to guide malicious outputs, lastly forming DH-CoT attack with a fake chain of thought. In experiments, we further observe that existing red-teaming datasets include samples unsuitable for evaluating attack gains, such as BPs, NHPs, and NTPs. Such data hinders accurate evaluation of true attack effect lifts. To address this, we introduce MDH, a Malicious content Detection framework integrating LLM-based annotation with Human assistance, with which we clean data and build RTA dataset suite. Experiments show that MDH reliably filters low-quality samples and that DH-CoT effectively jailbreaks models including GPT-5 and Claude-4, notably outperforming SOTA methods like H-CoT and TAP.
>
---
#### [replaced 069] Improved LLM Agents for Financial Document Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于金融文档问答任务，解决LLM在处理数值问题时的不足。提出改进的批评代理和计算器代理，提升性能并增强安全性。**

- **链接: [https://arxiv.org/pdf/2506.08726v3](https://arxiv.org/pdf/2506.08726v3)**

> **作者:** Nelvin Tan; Zian Seng; Liang Zhang; Yu-Ching Shih; Dong Yang; Amol Salunkhe
>
> **备注:** 13 pages, 6 figures. More analysis is added to Appendix C
>
> **摘要:** Large language models (LLMs) have shown impressive capabilities on numerous natural language processing tasks. However, LLMs still struggle with numerical question answering for financial documents that include tabular and textual data. Recent works have showed the effectiveness of critic agents (i.e., self-correction) for this task given oracle labels. Building upon this framework, this paper examines the effectiveness of the traditional critic agent when oracle labels are not available, and show, through experiments, that this critic agent's performance deteriorates in this scenario. With this in mind, we present an improved critic agent, along with the calculator agent which outperforms the previous state-of-the-art approach (program-of-thought) and is safer. Furthermore, we investigate how our agents interact with each other, and how this interaction affects their performance.
>
---
#### [replaced 070] ToolRM: Outcome Reward Models for Tool-Calling Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决工具调用中奖励建模不足的问题。通过构建基准测试和提出ToolRM模型，提升工具使用效果评估与策略训练性能。**

- **链接: [https://arxiv.org/pdf/2509.11963v2](https://arxiv.org/pdf/2509.11963v2)**

> **作者:** Mayank Agarwal; Ibrahim Abdelaziz; Kinjal Basu; Merve Unuvar; Luis A. Lastras; Yara Rizk; Pavan Kapanipathi
>
> **摘要:** As large language models (LLMs) increasingly interact with external tools, reward modeling for tool use has emerged as a critical yet underexplored area of research. Existing reward models, trained primarily on natural language outputs, struggle to evaluate tool-based reasoning and execution. To quantify this gap, we introduce FC-RewardBench, the first benchmark to systematically evaluate reward models in tool-calling scenarios. Our analysis shows that current reward models frequently miss key signals of effective tool use, highlighting the need for domain-specific modeling. We address this by proposing a training framework for outcome reward models using data synthesized from permissively licensed, open-weight LLMs. We introduce ToolRM - a suite of reward models for tool-use ranging from 1.7B to 14B parameters. Across diverse settings, these models consistently outperform general-purpose baselines. Notably, they achieve up to a 25% improvement with Best-of-N sampling, while also improving robustness to input noise, enabling effective data filtering, and supporting RL-training of policy models.
>
---
#### [replaced 071] InsertGNN: Can Graph Neural Networks Outperform Humans in TOEFL Sentence Insertion Problem?
- **分类: cs.CL**

- **简介: 该论文属于文本整合任务，旨在解决句子插入问题。提出InsertGNN模型，通过图神经网络提升句子间关系理解，实验显示其效果接近人类水平。**

- **链接: [https://arxiv.org/pdf/2103.15066v3](https://arxiv.org/pdf/2103.15066v3)**

> **作者:** Fang Wu; Stan Z. Li
>
> **摘要:** The integration of sentences poses an intriguing challenge within the realm of NLP, but it has not garnered the attention it deserves. Existing methods that focus on sentence arrangement, textual consistency, and question answering are inadequate in addressing this issue. To bridge this gap, we introduce InsertGNN, which conceptualizes the problem as a graph and employs a hierarchical Graph Neural Network (GNN) to comprehend the interplay between sentences. Our approach was rigorously evaluated on a TOEFL dataset, and its efficacy was further validated on the expansive arXiv dataset using cross-domain learning. Thorough experimentation unequivocally establishes InsertGNN's superiority over all comparative benchmarks, achieving an impressive 70% accuracy, a performance on par with average human test scores.
>
---
#### [replaced 072] HiKE: Hierarchical Evaluation Framework for Korean-English Code-Switching Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多语言语音识别任务，旨在解决韩英混语识别问题。提出HiKE框架，包含真实混语数据和分层标注，用于评估和提升模型的混语识别能力。**

- **链接: [https://arxiv.org/pdf/2509.24613v3](https://arxiv.org/pdf/2509.24613v3)**

> **作者:** Gio Paik; Yongbeom Kim; Soungmin Lee; Sangmin Ahn; Chanwoo Kim
>
> **备注:** EACL Findings 2026
>
> **摘要:** Despite advances in multilingual automatic speech recognition (ASR), code-switching (CS), the mixing of languages within an utterance common in daily speech, remains a severely underexplored challenge. In this paper, we introduce HiKE: the Hierarchical Korean-English code-switching benchmark, the first globally accessible non-synthetic evaluation framework for Korean-English CS, aiming to provide a means for the precise evaluation of multilingual ASR models and to foster research in the field. The proposed framework not only consists of high-quality, natural CS data across various topics, but also provides meticulous loanword labels and a hierarchical CS-level labeling scheme (word, phrase, and sentence) that together enable a systematic evaluation of a model's ability to handle each distinct level of code-switching. Through evaluations of diverse multilingual ASR models and fine-tuning experiments, this paper demonstrates that although most multilingual ASR models initially exhibit inadequate CS-ASR performance, this capability can be enabled through fine-tuning with synthetic CS data. HiKE is available at https://github.com/ThetaOne-AI/HiKE.
>
---
#### [replaced 073] Beyond Monolingual Assumptions: A Survey of Code-Switched NLP in the Era of Large Language Models across Modalities
- **分类: cs.CL**

- **简介: 该论文属于多语言自然语言处理任务，旨在解决代码切换建模问题。通过分析327篇研究，探讨LLM在多语言输入中的挑战与进展，提出数据、评估和模型改进方向。**

- **链接: [https://arxiv.org/pdf/2510.07037v4](https://arxiv.org/pdf/2510.07037v4)**

> **作者:** Rajvee Sheth; Samridhi Raj Sinha; Mahavir Patil; Himanshu Beniwal; Mayank Singh
>
> **摘要:** Code-switching (CSW), the alternation of languages and scripts within a single utterance, remains a fundamental challenge for multilingual NLP, even amidst the rapid advances of large language models (LLMs). Amidst the rapid advances of large language models (LLMs), most LLMs still struggle with mixed-language inputs, limited Codeswitching (CSW) datasets, and evaluation biases, which hinder their deployment in multilingual societies. This survey provides the first comprehensive analysis of CSW-aware LLM research, reviewing 327 studies spanning five research areas, 15+ NLP tasks, 30+ datasets, and 80+ languages. We categorize recent advances by architecture, training strategy, and evaluation methodology, outlining how LLMs have reshaped CSW modeling and identifying the challenges that persist. The paper concludes with a roadmap that emphasizes the need for inclusive datasets, fair evaluation, and linguistically grounded models to achieve truly multilingual capabilities https://github.com/lingo-iitgn/awesome-code-mixing/.
>
---
#### [replaced 074] FinTagging: Benchmarking LLMs for Extracting and Structuring Financial Information
- **分类: cs.CL; cs.AI; cs.CE**

- **简介: 该论文提出FinTagging，解决金融信息结构化标注问题。任务为金融数据提取与概念映射，通过两阶段方法提升LLM在财务领域的结构感知能力。**

- **链接: [https://arxiv.org/pdf/2505.20650v3](https://arxiv.org/pdf/2505.20650v3)**

> **作者:** Yan Wang; Lingfei Qian; Xueqing Peng; Yang Ren; Keyi Wang; Yi Han; Dongji Feng; Fengran Mo; Shengyuan Lin; Qinchuan Zhang; Kaiwen He; Chenri Luo; Jianxing Chen; Junwei Wu; Chen Xu; Ziyang Xu; Jimin Huang; Guojun Xiong; Xiao-Yang Liu; Qianqian Xie; Jian-Yun Nie
>
> **摘要:** Accurate interpretation of numerical data in financial reports is critical for markets and regulators. Although XBRL (eXtensible Business Reporting Language) provides a standard for tagging financial figures, mapping thousands of facts to over ten thousand US-GAAP concepts remains costly and error-prone. Existing benchmarks oversimplify this task as flat, single-step classification over small subsets of concepts, ignoring the hierarchical semantics of the taxonomy and the structured nature of financial documents. As a result, these benchmarks fail to evaluate Large Language Models (LLMs) under realistic reporting conditions. To bridge this gap, we introduce FinTagging, the first comprehensive benchmark for structure-aware and full-scope XBRL tagging. We decompose the complex tagging process into two subtasks: (1) FinNI (Financial Numeric Identification), which extracts entities and types from heterogeneous contexts such as text and tables; and (2) FinCL (Financial Concept Linking), which maps extracted entities to the full US-GAAP taxonomy. This two-stage formulation enables a fair assessment of LLM capabilities in numerical reasoning and taxonomy alignment. Evaluating diverse LLMs in zero-shot settings shows that while models generalize well in extraction, they struggle with fine-grained concept linking, revealing important limitations in domain-specific, structure-aware reasoning. Code is available on GitHub, and datasets are available on Hugging Face.
>
---
#### [replaced 075] Exploring Iterative Controllable Summarization with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于摘要生成任务，旨在解决大语言模型在控制摘要属性（如长度、主题）上的不足。通过引入迭代评估指标和提出GTE框架，提升模型的可控性与适应性。**

- **链接: [https://arxiv.org/pdf/2411.12460v3](https://arxiv.org/pdf/2411.12460v3)**

> **作者:** Sangwon Ryu; Heejin Do; Daehee Kim; Hwanjo Yu; Dongwoo Kim; Yunsu Kim; Gary Geunbae Lee; Jungseul Ok
>
> **备注:** EACL Findings 2026
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable performance in abstractive summarization tasks. However, their ability to precisely control summary attributes (e.g., length or topic) remains underexplored, limiting their adaptability to specific user preferences. In this paper, we systematically explore the controllability of LLMs. To this end, we revisit summary attribute measurements and introduce iterative evaluation metrics, failure rate and average iteration count to precisely evaluate controllability of LLMs, rather than merely assessing errors. Our findings show that LLMs struggle more with numerical attributes than with linguistic attributes. To address this challenge, we propose a guide-to-explain framework (GTE) for controllable summarization. Our GTE framework enables the model to identify misaligned attributes in the initial draft and guides it in self-explaining errors in the previous output. By allowing the model to reflect on its misalignment, GTE generates well-adjusted summaries that satisfy the desired attributes with robust effectiveness, requiring surprisingly fewer iterations than other iterative approaches.
>
---
#### [replaced 076] How Training Data Shapes the Use of Parametric and In-Context Knowledge in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，研究语言模型如何利用训练数据中的参数化知识和上下文知识。旨在解决模型在冲突情况下如何选择知识的问题，通过实验揭示训练数据特性对知识利用的影响。**

- **链接: [https://arxiv.org/pdf/2510.02370v2](https://arxiv.org/pdf/2510.02370v2)**

> **作者:** Minsung Kim; Dong-Kyum Kim; Jea Kwon; Nakyeong Yang; Kyomin Jung; Meeyoung Cha
>
> **备注:** 16 pages
>
> **摘要:** Large language models leverage not only parametric knowledge acquired during training but also in-context knowledge provided at inference time, despite the absence of explicit training objectives for using both sources. Prior work has further shown that when these knowledge sources conflict, models resolve the tension based on their internal confidence, preferring parametric knowledge for high-confidence facts while deferring to contextual information for less familiar ones. However, the training conditions that give rise to such knowledge utilization behaviors remain unclear. To address this gap, we conduct controlled experiments in which we train language models while systematically manipulating key properties of the training data. Our results reveal a counterintuitive finding: three properties commonly regarded as detrimental must co-occur for robust knowledge utilization and conflict resolution to emerge: (i) intra-document repetition of information, (ii) a moderate degree of within-document inconsistency, and (iii) a skewed knowledge frequency distribution. We further validate that the same training dynamics observed in our controlled setting also arise during real-world language model pretraining, and we analyze how post-training procedures can reshape models' knowledge preferences. Together, our findings provide concrete empirical guidance for training language models that harmoniously integrate parametric and in-context knowledge.
>
---
#### [replaced 077] EngTrace: A Symbolic Benchmark for Verifiable Process Supervision of Engineering Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出EngTrace，一个用于验证工程推理过程的符号基准，解决现有基准无法评估工程物理推理的问题。通过生成测试用例和两阶段评估框架，提升对LLM工程能力的评估准确性。**

- **链接: [https://arxiv.org/pdf/2511.01650v2](https://arxiv.org/pdf/2511.01650v2)**

> **作者:** Ayesha Gull; Muhammad Usman Safder; Rania Elbadry; Fan Zhang; Veselin Stoyanov; Preslav Nakov; Zhuohan Xie
>
> **备注:** 22 pages, includes figures and tables; introduces the EngTrace benchmark
>
> **摘要:** Large Language Models (LLMs) are increasingly entering specialized, safety-critical engineering workflows governed by strict quantitative standards and immutable physical laws, making rigorous evaluation of their reasoning capabilities imperative. However, existing benchmarks such as MMLU, MATH, and HumanEval assess isolated cognitive skills, failing to capture the physically grounded reasoning central to engineering, where scientific principles, quantitative modeling, and practical constraints must converge. To enable verifiable process supervision in engineering, we introduce EngTrace, a symbolic benchmark comprising 90 templates across three major engineering branches, nine core domains and 20 distinct areas. Through domain-aware parameterization, we generate 1,350 unique, contamination-resistant test cases to stress-test generalization. Moving beyond outcome matching, we introduce a verifiable two-stage evaluation framework that uses a tiered protocol to validate intermediate reasoning traces alongside final answers through automated procedural checks and a heterogeneous AI Tribunal. Our evaluation of 24 leading LLMs reveals a distinct trade-off between numeric precision and trace fidelity, identifying a complexity cliff where abstract mathematical pre-training fails to translate into the integrative reasoning required for advanced engineering tasks.
>
---
#### [replaced 078] Social Bias in Popular Question-Answering Benchmarks
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言处理领域，探讨QA和RC基准中的社会偏见问题。研究发现现有基准在人口统计学上代表性不足，存在性别、宗教和地域偏见，呼吁改进评估方法。**

- **链接: [https://arxiv.org/pdf/2505.15553v3](https://arxiv.org/pdf/2505.15553v3)**

> **作者:** Angelie Kraft; Judith Simon; Sonja Schimmler
>
> **备注:** Presented at the main track of the IJCNLP-AACL 2025 conference (Mumbai and Online)
>
> **摘要:** Question-answering (QA) and reading comprehension (RC) benchmarks are commonly used for assessing the capabilities of large language models (LLMs) to retrieve and reproduce knowledge. However, we demonstrate that popular QA and RC benchmarks do not cover questions about different demographics or regions in a representative way. We perform a content analysis of 30 benchmark papers and a quantitative analysis of 20 respective benchmark datasets to learn (1) who is involved in the benchmark creation, (2) whether the benchmarks exhibit social bias, or whether this is addressed or prevented, and (3) whether the demographics of the creators and annotators correspond to particular biases in the content. Most benchmark papers analyzed provide insufficient information about those involved in benchmark creation, particularly the annotators. Notably, just one (WinoGrande) explicitly reports measures taken to address social representation issues. Moreover, the data analysis revealed gender, religion, and geographic biases across a wide range of encyclopedic, commonsense, and scholarly benchmarks. Our work adds to the mounting criticism of AI evaluation practices and shines a light on biased benchmarks being a potential source of LLM bias by incentivizing biased inference heuristics.
>
---
#### [replaced 079] Proverbs or Pythian Oracles? Sentiments and Emotions in Greek Sayings
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在研究希腊谚语中的情绪与情感。通过构建多标签标注框架，分析谚语的情感分布，揭示其多维特性。**

- **链接: [https://arxiv.org/pdf/2510.13341v2](https://arxiv.org/pdf/2510.13341v2)**

> **作者:** Katerina Korre; John Pavlopoulos
>
> **摘要:** Proverbs are among the most fascinating language phenomena that transcend cultural and linguistic boundaries. Yet, much of the global landscape of proverbs remains underexplored, as many cultures preserve their traditional wisdom within their own communities due to the oral tradition of the phenomenon. Taking advantage of the current advances in Natural Language Processing (NLP), we focus on Greek proverbs, analyzing their sentiment and emotion. Departing from an annotated dataset of Greek proverbs, (1) we propose a multi-label annotation framework and dataset that captures the emotional variability of the proverbs, (2) we up-scale to local varieties, (3) we sketch a map of Greece that provides an overview of the distribution of emotions. Our findings show that the interpretation of proverbs is multidimensional, a property manifested through both multi-labeling and instance-level polarity. LLMs can capture and reproduce this complexity, and can therefore help us better understand the proverbial landscape of a place, as in the case of Greece, where surprise and anger compete and coexist within proverbs.
>
---
#### [replaced 080] SSSD: Simply-Scalable Speculative Decoding
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出SSSD，一种无需训练的推测解码方法，用于加速大语言模型推理。解决现有方法复杂度高、适应性差的问题，通过轻量n-gram匹配与硬件感知推测，提升效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2411.05894v2](https://arxiv.org/pdf/2411.05894v2)**

> **作者:** Michele Marzollo; Jiawei Zhuang; Niklas Roemer; Niklas Zwingenberger; Lorenz K. Müller; Lukas Cavigelli
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** Speculative Decoding has emerged as a popular technique for accelerating inference in Large Language Models. However, most existing approaches yield only modest improvements in production serving systems. Methods that achieve substantial speedups typically rely on an additional trained draft model or auxiliary model components, increasing deployment and maintenance complexity. This added complexity reduces flexibility, particularly when serving workloads shift to tasks, domains, or languages that are not well represented in the draft model's training data. We introduce Simply-Scalable Speculative Decoding (SSSD), a training-free method that combines lightweight n-gram matching with hardware-aware speculation. Relative to standard autoregressive decoding, SSSD reduces latency by up to 2.9x. It achieves performance on par with leading training-based approaches across a broad range of benchmarks, while requiring substantially lower adoption effort--no data preparation, training or tuning are needed--and exhibiting superior robustness under language and domain shift, as well as in long-context settings.
>
---
#### [replaced 081] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决小模型工具使用中的schema误配问题。通过调整工具schema以匹配模型预训练知识，提升模型工具使用能力。**

- **链接: [https://arxiv.org/pdf/2510.07248v2](https://arxiv.org/pdf/2510.07248v2)**

> **作者:** Jonggeun Lee; Woojung Song; Jongwook Han; Haesung Pyun; Yohan Jo
>
> **备注:** 22 pages
>
> **摘要:** Small language models (SLMs) enable scalable multi-agent tool systems where multiple SLMs handle subtasks orchestrated by a powerful coordinator. However, they struggle with tool-use tasks, particularly in selecting appropriate tools and identifying correct parameters. A common failure mode is schema misalignment: models hallucinate plausible but nonexistent tool names that reflect naming conventions internalized during pretraining but absent from the provided tool schema. Rather than forcing models to adapt to arbitrary schemas, we propose adapting schemas to align with models' pretrained knowledge. We introduce PA-Tool (Pretraining-Aligned Tool Schema Generation), a training-free method that leverages peakedness, a signal from contamination detection indicating pretraining familiarity, to rename tool components. By generating multiple candidates and selecting those with the highest peakedness across samples, PA-Tool identifies pretraining-aligned naming patterns. Experiments on MetaTool and RoTBench show improvements of up to 17%, with schema misalignment errors reduced by 80%. PA-Tool enables small models to approach state-of-the-art performance while maintaining computational efficiency in adapting to new tools without retraining. Our work demonstrates that schema-level interventions can unlock the tool-use potential of resource-efficient models by adapting schemas to models rather than models to schemas.
>
---
#### [replaced 082] Authors Should Label Their Own Documents
- **分类: cs.CL**

- **简介: 论文提出作者标注方法，用于提升主观情感和信念的标注质量。针对传统第三方标注的不足，通过作者实时标注提升数据质量，应用于产品推荐任务，效果显著优于现有基准。**

- **链接: [https://arxiv.org/pdf/2512.12976v3](https://arxiv.org/pdf/2512.12976v3)**

> **作者:** Marcus Ma; Cole Johnson; Nolan Bridges; Jackson Trager; Georgios Chochlakis; Shrikanth Narayanan
>
> **摘要:** Third-party annotation is the status quo for labeling text, but egocentric information such as sentiment and belief can at best only be approximated by a third-person proxy. We introduce author labeling, an annotation technique where the writer of the document itself annotates the data at the moment of creation. We collaborate with a commercial chatbot with over 20,000 users to deploy an author labeling annotation system. This system identifies task-relevant queries, generates on-the-fly labeling questions, and records authors' answers in real time. We train and deploy an online-learning model architecture for product recommendation with author-labeled data to improve performance. We train our model to minimize the prediction error on questions generated for a set of predetermined subjective beliefs using author-labeled responses. Our model achieves a 537% improvement in click-through rate compared to an industry advertising baseline running concurrently. We then compare the quality and practicality of author labeling to three traditional annotation approaches for sentiment analysis and find author labeling to be higher quality, faster to acquire, and cheaper. These findings reinforce existing literature that annotations, especially for egocentric and subjective beliefs, are significantly higher quality when labeled by the author rather than a third party. To facilitate broader scientific adoption, we release an author labeling service for the research community at https://academic.echogroup.ai.
>
---
#### [replaced 083] A Comparative Analysis of Contextual Representation Flow in State-Space and Transformer Architectures
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型分析任务，比较了状态空间模型与Transformer在上下文表示流动上的差异，揭示了两者在表征同质化机制上的不同。**

- **链接: [https://arxiv.org/pdf/2510.06640v2](https://arxiv.org/pdf/2510.06640v2)**

> **作者:** Nhat M. Hoang; Do Xuan Long; Cong-Duy Nguyen; Min-Yen Kan; Luu Anh Tuan
>
> **摘要:** State Space Models (SSMs) have recently emerged as efficient alternatives to Transformer-Based Models (TBMs) for long-sequence processing with linear scaling, yet how contextual information flows across layers in these architectures remains understudied. We present the first unified, token- and layer-wise analysis of representation propagation in SSMs and TBMs. Using centered kernel alignment, variance-based metrics, and probing, we characterize how representations evolve within and across layers. We find a key divergence: TBMs rapidly homogenize token representations, with diversity reemerging only in later layers, while SSMs preserve token uniqueness early but converge to homogenization deeper. Theoretical analysis and parameter randomization further reveal that oversmoothing in TBMs stems from architectural design, whereas in SSMs, it arises mainly from training dynamics. These insights clarify the inductive biases of both architectures and inform future model and training designs for long-context reasoning.
>
---
#### [replaced 084] GIFT: Guided Importance-Aware Fine-Tuning for Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型优化任务，解决扩散模型微调困难的问题。通过引入重要性权重的GIFT方法，提升生成一致性与性能。**

- **链接: [https://arxiv.org/pdf/2509.20863v2](https://arxiv.org/pdf/2509.20863v2)**

> **作者:** Guowei Xu; Wenxin Xu; Jiawang Zhao; Kaisheng Ma
>
> **备注:** preprint
>
> **摘要:** Diffusion models have recently shown strong potential in language modeling, offering faster generation compared to traditional autoregressive approaches. However, applying supervised fine-tuning (SFT) to diffusion models remains challenging, as they lack precise probability estimates at each denoising step. While the diffusion mechanism enables the model to reason over entire sequences, it also makes the generation process less predictable and often inconsistent. This highlights the importance of controlling key tokens that guide the direction of generation. To address this issue, we propose GIFT, an importance-aware finetuning method for diffusion language models, where tokens are assigned different importance weights based on their entropy. Derived from diffusion theory, GIFT delivers substantial gains: across diverse settings including different mainstream training datasets ranging from 1k to 10k in size, utilizing LoRA or full parameter fine-tuning, and training on base or instruct models, GIFT consistently achieves superior overall performance compared to standard SFT on four widely used reasoning benchmarks (Sudoku, Countdown, GSM8K, and MATH-500).
>
---
#### [replaced 085] When in Doubt, Consult: Expert Debate for Sexism Detection via Confidence-Based Routing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于 sexism 检测任务，旨在解决传统方法难以识别隐蔽、复杂性别歧视内容的问题。通过两阶段框架，结合数据增强与专家推理机制，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2512.23732v3](https://arxiv.org/pdf/2512.23732v3)**

> **作者:** Anwar Alajmi; Gabriele Pergola
>
> **摘要:** Online sexism increasingly appears in subtle, context-dependent forms that evade traditional detection methods. Its interpretation often depends on overlapping linguistic, psychological, legal, and cultural dimensions, which produce mixed and sometimes contradictory signals in annotated datasets. These inconsistencies, combined with label scarcity and class imbalance, result in unstable decision boundaries and cause fine-tuned models to overlook subtler, underrepresented forms of harm. To address these challenges, we propose a two-stage framework that unifies (i) targeted training procedures to better regularize supervision to scarce and noisy data with (ii) selective, reasoning-based inference to handle ambiguous or borderline cases. First, we stabilize the training combining class-balanced focal loss, class-aware batching, and post-hoc threshold calibration, strategies for the firs time adapted for this domain to mitigate label imbalance and noisy supervision. Second, we bridge the gap between efficiency and reasoning with a a dynamic routing mechanism that distinguishes between unambiguous instances and complex cases requiring a deliberative process. This reasoning process results in the novel Collaborative Expert Judgment (CEJ) module which prompts multiple personas and consolidates their reasoning through a judge model. Our approach outperforms existing approaches across several public benchmarks, with F1 gains of +4.48% and +1.30% on EDOS Tasks A and B, respectively, and a +2.79% improvement in ICM on EXIST 2025 Task 1.1.
>
---
#### [replaced 086] Multiplayer Nash Preference Optimization
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出MNPO，解决大语言模型与人类偏好对齐问题。针对传统方法在多参与者场景中的局限性，MNPO将对齐建模为多人博弈，提升对复杂偏好的适应能力。**

- **链接: [https://arxiv.org/pdf/2509.23102v2](https://arxiv.org/pdf/2509.23102v2)**

> **作者:** Fang Wu; Xu Huang; Weihao Xuan; Zhiwei Zhang; Yijia Xiao; Guancheng Wan; Xiaomin Li; Bing Hu; Peng Xia; Jure Leskovec; Yejin Choi
>
> **摘要:** Reinforcement learning from human feedback (RLHF) has emerged as the standard paradigm for aligning large language models with human preferences. However, reward-based methods built on the Bradley-Terry assumption struggle to capture the non-transitive and heterogeneous nature of real-world preferences. To address this, recent studies have reframed alignment as a two-player Nash game, giving rise to Nash learning from human feedback (NLHF). While this perspective has inspired algorithms such as INPO, ONPO, and EGPO with strong theoretical and empirical guarantees, they remain fundamentally restricted to two-player interactions, creating a single-opponent bias that fails to capture the full complexity of realistic preference structures. This work introduces Multiplayer Nash Preference Optimization (MNPO), a novel framework that generalizes NLHF to the multiplayer regime. It formulates alignment as an n-player game, where each policy competes against a population of opponents while being regularized toward a reference model. We demonstrate that MNPO inherits the equilibrium guarantees of two-player methods while enabling richer competitive dynamics and improved coverage of diverse preference structures. Comprehensive empirical evaluation shows that MNPO consistently outperforms existing NLHF baselines on instruction-following benchmarks, achieving superior alignment quality under heterogeneous annotator conditions and mixed-policy evaluation scenarios. Together, these results establish MNPO as a principled and scalable framework for aligning LLMs with complex, non-transitive human preferences. Code is available at https://github.com/smiles724/MNPO.
>
---
#### [replaced 087] LFD: Layer Fused Decoding to Exploit External Knowledge in Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于检索增强生成任务，解决如何有效利用外部知识的问题。通过引入噪声和分层策略，提出LFD方法，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2508.19614v3](https://arxiv.org/pdf/2508.19614v3)**

> **作者:** Yang Sun; Zhiyong Xie; Lixin Zou; Dan Luo; Min Tang; Xiangyu Zhao; Yunwei Zhao; Xixun Lin; Yanxiong Lu; Chenliang Li
>
> **摘要:** Retrieval-augmented generation (RAG) incorporates external knowledge into large language models (LLMs), improving their adaptability to downstream tasks and enabling information updates. Surprisingly, recent empirical evidence demonstrates that injecting noise into retrieved relevant documents paradoxically facilitates exploitation of external knowledge and improves generation quality. Although counterintuitive and challenging to apply in practice, this phenomenon enables granular control and rigorous analysis of how LLMs integrate external knowledge. Therefore, in this paper, we intervene on noise injection and establish a layer-specific functional demarcation within the LLM: shallow layers specialize in local context modeling, intermediate layers focus on integrating long-range external factual knowledge, and deeper layers primarily rely on parametric internal knowledge. Building on this insight, we propose Layer Fused Decoding (LFD), a simple decoding strategy that directly combines representations from an intermediate layer with final-layer decoding outputs to fully exploit the external factual knowledge. To identify the optimal intermediate layer, we introduce an internal knowledge score (IKS) criterion that selects the layer with the lowest IKS value in the latter half of layers. Experimental results across multiple benchmarks demonstrate that LFD helps RAG systems more effectively surface retrieved context knowledge with minimal cost.
>
---
#### [replaced 088] Table as a Modality for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于表格推理任务，旨在解决LLMs处理表格数据能力不足的问题。通过提出TAMO框架，将表格作为独立模态进行建模，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2512.00947v2](https://arxiv.org/pdf/2512.00947v2)**

> **作者:** Liyao Li; Chao Ye; Wentao Ye; Yifei Sun; Zhe Jiang; Haobo Wang; Jiaming Tian; Yiming Zhang; Ningtao Wang; Xing Fu; Gang Chen; Junbo Zhao
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** To migrate the remarkable successes of Large Language Models (LLMs), the community has made numerous efforts to generalize them to the table reasoning tasks for the widely deployed tabular data. Despite that, in this work, by showing a probing experiment on our proposed StructQA benchmark, we postulate that even the most advanced LLMs (such as GPTs) may still fall short of coping with tabular data. More specifically, the current scheme often simply relies on serializing the tabular data, together with the meta information, then inputting them through the LLMs. We argue that the loss of structural information is the root of this shortcoming. In this work, we further propose TAMO, which bears an ideology to treat the tables as an independent modality integrated with the text tokens. The resulting model in TAMO is a multimodal framework consisting of a hypergraph neural network as the global table encoder seamlessly integrated with the mainstream LLM. Empirical results on various benchmarking datasets, including HiTab, WikiTQ, WikiSQL, FeTaQA, and StructQA, have demonstrated significant improvements on generalization with an average relative gain of 42.65%.
>
---

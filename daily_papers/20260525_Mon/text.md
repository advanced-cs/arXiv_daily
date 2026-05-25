# 自然语言处理 cs.CL

- **最新发布 85 篇**

- **更新 56 篇**

## 最新发布

#### [new 001] Metadata Predictability Is Not Evidence Dependence: An Intervention-Based Audit for Weak-Label Benchmarks
- **分类: cs.CL**

- **简介: 该论文研究弱标签基准的审计方法，解决如何验证模型是否依赖证据而非元数据的问题。通过结合元数据统计与证据干预分析，提出新的评估指标。**

- **链接: [https://arxiv.org/pdf/2605.23701](https://arxiv.org/pdf/2605.23701)**

> **作者:** Kan Shao
>
> **备注:** 5 pages, 1 figure, 1 table. Accepted at ICML 2026 Workshop on Hypothesis Testing
>
> **摘要:** We study a protocol-level test for weak-label benchmarks: whether benchmark outputs change when the provided evidence is intervened on. Metadata-only shortcut checks answer a different question, namely whether outputs are predictable from metadata priors. We therefore combine a metadata statistic, the Metadata Prior Dominance Score (MPDS), with an evidence-intervention statistic, {\Delta}Evi, measuring sensitivity to evidence identity under cross-item shuffling. Synthetic HotpotQA gives a constructed counterexample to metadata-only screening: MPDS is only moderate (0.643), yet {\Delta}Evi is zero. Stronger-reader reruns show why calibration belongs in the test procedure: SNLI shows a calibration reversal, reconstructed HotpotQA occupies a question-dominant warning region, and FEVER is a strongly evidence-sensitive positive control across four transformers. The practical lesson is simple: benchmark audits should report metadata-only screening, evidence intervention, and reader-strength calibration together.
>
---
#### [new 002] Structure-Guided Entity Resolution: Fine-Tuning LLMs for Robust Name Matching in Complex Linguistic Contexts
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于实体消歧任务，解决跨记录姓名匹配问题。通过结构引导的两阶段微调方法，提升模型在复杂语言环境中的匹配精度。**

- **链接: [https://arxiv.org/pdf/2605.23597](https://arxiv.org/pdf/2605.23597)**

> **作者:** Shivam Chourasia; Hitesh Kapoor; Nilesh Patil
>
> **备注:** Accepted to ACL 2026. 8 pages, 1 figure, 2 tables
>
> **摘要:** Matching person names across heterogeneous records is a core challenge in entity resolution, especially within linguistically and culturally complex environments. Variations in naming conventions, inconsistent transliteration across scripts, and frequent data entry errors make it difficult to unify user identities, an essential requirement for Know Your Customer (KYC) compliance. While Large Language Models have shown promise in understanding natural language, they often struggle with the structured ambiguity present in such domain-specific settings. This paper introduces Structure-Guided Entity Resolution (SGER), a novel framework that fine-tunes an LLM through a two-phase curriculum. The model is first trained to parse the grammatical and semantic structure of personal names, then optimized for the downstream task of binary entity matching. We evaluate SGER in the challenging context of Indian identity data, one of the most linguistically diverse and noisy environments globally. SGER achieves 99.02% accuracy and an F1 of 0.994 on a held-out set of 50,000 real-world pairs, outperforming GPT-4o few-shot prompting and single-stage fine-tuning baselines. The system is fully deployed in production at Dream11, the world's largest fantasy sports platform, serving 250M+ users. Our results demonstrate that curriculum-guided training enables robust, high-precision entity resolution in real-world multilingual systems at scale.
>
---
#### [new 003] Hidden Human-Like Nature of Machine-Generated Texts: Theory and Detection Enhancement
- **分类: cs.CL**

- **简介: 该论文属于文本检测任务，旨在解决机器生成文本中隐藏的人类相似性带来的检测难题。通过理论分析和框架优化，提升现有检测方法的准确性。**

- **链接: [https://arxiv.org/pdf/2605.23190](https://arxiv.org/pdf/2605.23190)**

> **作者:** Chenwang Wu; Yiu-ming Cheung; Bo Han; Defu Lian
>
> **摘要:** Machine-generated texts (MGTs) produced by large language models (LLMs) are increasingly prevalent across various applications, while their potential misuse in fake news propagation and phishing has raised serious concerns, highlighting the need for MGT detection. Existing paragraph-level detection methods commonly treat MGTs as entirely machine-like, overlooking the hidden human-like nature of machine-generated texts: even fully machine-generated texts may contain spans that are highly consistent with human writing. To this end, we first reveal the existence of such hidden human-like spans, and then theoretically analyze their impact on detection. Our analysis shows that these spans increase the sentence complexity for detection, thereby making MGT detection intrinsically harder. Based on this finding, we propose a model-agnostic stacked enhancement framework that improves existing detectors by reducing the influence of hidden human-like spans. Specifically, we model span-level retention decisions as a latent-variable problem and instantiate the optimization with a hard-EM-inspired procedure, where the detector iteratively filters confidently human-like subsequences and refines itself on the remaining text. Extensive experiments across various LLMs and practical scenarios demonstrate that the proposed framework consistently enhances existing detectors. Notably, the framework can also work in a training-free manner, offering flexibility and scalability for practical deployment.
>
---
#### [new 004] OpenSkillEval: Automatically Auditing the Open Skill Ecosystem for LLM Agents
- **分类: cs.CL**

- **简介: 该论文属于LLM代理技能评估任务，旨在解决技能质量评价与选择问题。提出OpenSkillEval框架，自动构建真实任务实例并评估技能效果。**

- **链接: [https://arxiv.org/pdf/2605.23657](https://arxiv.org/pdf/2605.23657)**

> **作者:** Jiahao Ying; Boxian Ai; Wei Tang; Siyuan Liu; Yixin Cao
>
> **摘要:** Skills, i.e., structured workflow instructions distilled for large language models (LLMs), are becoming an increasingly important mechanism for improving agent performance on real-world downstream tasks. However, as the open-source skill ecosystem rapidly expands, it remains unclear how different models and agent frameworks interact with skills, how to evaluate skill quality, and how users should select skills under practical cost-performance trade-offs. In this paper, we present \textsc{OpenSkillEval}, an automatic evaluation framework for both skill-augmented agent systems and the skills themselves. Instead of relying on static benchmarks, \textsc{OpenSkillEval} automatically constructs realistic task instances from evolving real-world artifacts across five categories of downstream applications: presentation generation, front-end web design, poster generation, data visualization, and report generation. It further collects and organizes community-contributed skills for controlled comparison under unified task settings. Using more than 600 dynamically generated task instances and 30 open-source skills, we conduct a systematic evaluation of state-of-the-art models and agent frameworks. Our results show that skill availability does not guarantee effective skill usage, that the benefit of skill augmentation depends strongly on both the underlying model and the agent framework, and that many publicly popular skills do not consistently outperform base agents without skills. These findings highlight the need for dynamic, task-grounded evaluation and provide practical insights into the design, selection, and deployment of skills for LLM agents. Additional cases and benchmark resources are available on the project website: this https URL.
>
---
#### [new 005] Metacognition as Reward: Reinforcing LLM Reasoning via Knowledge and Regulation Signals
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习任务，旨在提升大语言模型的推理能力。针对现有方法在奖励设计上的不足，提出Metacognition-as-Reward框架，通过知识与调节信号优化推理过程。**

- **链接: [https://arxiv.org/pdf/2605.23384](https://arxiv.org/pdf/2605.23384)**

> **作者:** Sirui Chen; Lei Xu; Yuying Zhao; Yutian Chen; Yu Wang; Beier Zhu; Hanwang Zhang; Shengjie Zhao; Chaochao Lu
>
> **摘要:** Recent RL methods have substantially improved the reasoning abilities of LLMs. Existing reward designs mainly follow two paradigms: (1) Reinforcement learning with verifiable rewards (RLVR) derives outcome signals from executable checks or ground-truth answers, but provides limited guidance for intermediate reasoning behaviors. (2) Rubrics-as-reward (RaR) goes beyond final-answer checking by using natural-language rubrics to assess reasoning quality and task compliance, but often requires instance-specific rubrics and substantial design effort. To address these issues, we introduce Metacognition-as-Reward (MaR), a metacognition-inspired RL framework that guides LLM reasoning through two general process dimensions: i) metacognitive knowledge, which identifies task-relevant information without hand-crafted instance-specific rubrics, and ii) metacognitive regulation, which plans and adjusts the reasoning process to provide reward guidance beyond final-answer outcomes. MaR scaffolds model rollouts into explicit metacognitive components and optimizes them with a trajectory-level reward over task knowledge coverage, regulation fidelity, and final-answer correctness. In this way, MaR extends reward feedback to reasoning trajectories while grounding the reward signals in general metacognitive dimensions. Experiments on 22 benchmarks show that MaR consistently improves model performance, achieving up to a 7.7% gain over the base model and up to an 11.0% gain over vanilla DAPO. Notably, Qwen3.5-9B + MaR narrows the gap to frontier models, surpassing GPT-OSS-120B on overall average and outperforming stronger models on several individual benchmarks. Process-level analysis further shows substantial improvements in reasoning process quality. MaR also generalizes to out-of-domain datasets, where MaR-trained models improve over their corresponding base models on average.
>
---
#### [new 006] DFKI-MLT at SemEval-2026 TASK 7: Steering Multilingual Models Towards Cultural Knowledge
- **分类: cs.CL**

- **简介: 该论文属于SemEval-2026 Task 7文化意识任务，旨在提升多语言模型的文化知识。通过激活转向技术，利用平行语料生成语言向量，改进模型在文化推理任务中的表现。**

- **链接: [https://arxiv.org/pdf/2605.23069](https://arxiv.org/pdf/2605.23069)**

> **作者:** Yusser Al Ghussin; Daniil Gurgurov; Yasser Hamidullah; Josef van Genabith; Cristina España-Bonet; Simon Ostermann
>
> **备注:** Accepted to The 20th International Workshop on Semantic Evaluation at ACL 2026
>
> **摘要:** Large language models (LLMs) are increasingly used across diverse linguistic and cultural contexts, yet their cultural knowledge remains uneven across regions and languages. We present the DFKI-MLT system for SemEval-2026 Task 7 on cultural awareness, where we apply activation steering to multilingual LLMs using language vectors extracted from parallel FLORES data. Our method performs inference-time adaptation by adding language-specific steering vectors to the residual stream at a selected transformer layer, without any parameter updates. We participated in both the short-answer (SAQ) and multiple-choice (MCQ) tracks; however, only our MCQ submission received an official score. In the official MCQ track, we achieved 86.96% accuracy, ranking 7th out of 17 teams. To better understand system behavior, we conduct post-hoc analyses on the shared-task MCQ and SAQ settings. These analyses show that activation steering yields modest and heterogeneous improvements on cultural reasoning: gains are strongly layer-sensitive, vary substantially across language-region pairs, with some configurations even degrading performance, and interact with prompt formulation, comparing generic and culturally conditioned prompts. Our findings suggest that prompt design and activation steering should be jointly optimized for culturally aware multilingual inference.
>
---
#### [new 007] As X, Do Y: How Persona and Task Combine in Instruction-Tuned LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究角色提示在指令调优大模型中的作用，探讨 persona 与 task 的组合机制。通过分析残差流中的线性分解，揭示其局部可加性及不可压缩性，解决角色提示有效性的理解问题。**

- **链接: [https://arxiv.org/pdf/2605.23147](https://arxiv.org/pdf/2605.23147)**

> **作者:** Eric Xu
>
> **备注:** 12 pages, 1 figure. Code: this https URL
>
> **摘要:** Role prompts of the form As X, do Y admit a clean linear decomposition at one specific site in the residual stream: the prompt-to-answer transition -- the last prompt token together with the first two generated tokens -- in an early/mid layer band. There, persona and task contribute through partially orthogonal additive directions. Forming a pure persona effect $\Delta_X$, a pure task effect $\Delta_Y$, and substituting $h_{BB} + \Delta_X + \Delta_Y$ for the clean residual yields downstream output within a small KL of clean on Gemma-2-2B-IT and Qwen-2.5-\{1.5B, 3B\}-Instruct, across a 12-cell short grid and a 48-cell long-persona grid, with persona-specific behavioral markers preserved. The natural inference from this additive structure is that the role prompt can be compressed into a single cached residual vector. \emph{We show it cannot.} Injecting the cached additive prediction -- or even the oracle clean residual $h_{XY}$ -- into a baseline host prompt with the persona text removed does not approach the clean long-persona target, at one site or at many layers. Persona-conditioned multi-token generation flows through attention back to the persona-text positions throughout the prompt, which no residual at one site reproduces. Local additivity in the residual stream does not imply prompt compressibility. The additive structure at the prompt-to-answer transition supports interpretability and fine-grained steering of persona or task contributions; persona-conditioned behavior across the full continuation depends on a distributed prompt/KV mechanism that local activation arithmetic does not displace.
>
---
#### [new 008] How Far Will They Go? Red-Teaming Online Influence with Large Language Models
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于安全评估任务，旨在检测开源大语言模型在政治观点表达上的可控性。通过红队测试，评估模型的过顿窗口及越狱效果，揭示其政治倾向与区域差异。**

- **链接: [https://arxiv.org/pdf/2605.22880](https://arxiv.org/pdf/2605.22880)**

> **作者:** Daniel C. Ruiz; Anna Serbina; Ashwin Rao; Emilio Ferrara; Luca Luceri
>
> **备注:** 30 pages, 8 figures, submitted to COLM 2026
>
> **摘要:** As large language model (LLM)-based agents increasingly participate in online discourse, red-teaming their capacity to support political influence campaigns is critical for information integrity. In pursuit of this goal, we focus on locally deployed open-source LLMs, as opposed to frontier API-only models, given their superior alignment with the operational constraints of privacy-conscious malicious actors deployed in social media environments. We introduce an empirical red-teaming framework for measuring LLM Overton Windows (OWs), defined as the range of political opinions a model can reliably express on controversial topics, and for quantifying how simple natural-language jailbreaks expand that range. We evaluate more than 30 LLMs spanning 10 model families and five countries of origin. We find systematic asymmetries in political expressivity: open-source LLMs are typically more willing to generate left-leaning social media content, OWs tend to contract inversely to model size, and regional differences are substantial despite uneven representation in the open-source ecosystem. Jailbreak potency also varies sharply across model families, motivating a workflow for identifying effective combinations of jailbreak techniques. Taken together, our results establish a practical framework for auditing the political steerability of open-source LLMs and for helping future researchers design stronger countermeasures against LLM-enabled influence campaigns.
>
---
#### [new 009] Knowledge Distillation for Low-Resource Open-source Text-to-SQL Model
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于Text-to-SQL任务，解决低资源环境下SQL生成效果差的问题。通过构建知识库并注入训练与推理，提升模型在领域特定数据库上的性能。**

- **链接: [https://arxiv.org/pdf/2605.22843](https://arxiv.org/pdf/2605.22843)**

> **作者:** Tianhao Qiu; Xiaojun Chen
>
> **备注:** 17ages, 5 figures
>
> **摘要:** Text-to-SQL converts natural language questions into executable SQL queries, enabling non-technical users to access relational databases for analytics and intelligent data services. In real-world scenarios, performance is often constrained by low-resource settings, where high-quality annotated \texttt{<question, SQL>} pairs are scarce, particularly for domain-specific databases. Additional challenges include opaque schema definitions, abbreviations, and implicit business logic that are not explicitly encoded in the schema. Existing data synthesis and prompting techniques improve coverage but often fail to produce task-specific, semantically grounded examples aligned with database constraints. To address these challenges, we propose a knowledge-aware Text-to-SQL framework that constructs task-specific knowledge base including schema semantics, abbreviations, business logic, and query patterns, and injects them into both training and inference. This framework generates diverse, contextually grounded synthetic training data and enhances inference through targeted knowledge retrieval. Experiments on seven benchmarks, covering both general and domain-specific datasets, demonstrate that our approach substantially improves the performance of open-source and closed-source large language models in Text-to-SQL tasks, especially in low-resource domain-specific settings, enhancing generalization, robustness, and adaptability.
>
---
#### [new 010] A Survey of Text and Speech Resources for Hausa and Fongbe: Availability, Quality, and Gaps for NLP Development
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理资源调查任务，旨在评估豪萨语和 Fonké 语的文本与语音资源现状及不足。通过系统收集和分析现有资源，识别出两者在领域多样性及语音语料上的主要缺口。**

- **链接: [https://arxiv.org/pdf/2605.22828](https://arxiv.org/pdf/2605.22828)**

> **作者:** Mahounan Pericles Adjovi; Victor Olufemi; Roald Eiselen; Prasenjit Mitra
>
> **备注:** 8 pages, 7 tables; survey paper; to appear in IEEE SDS 2026
>
> **摘要:** This survey provides a comprehensive catalog of publicly available text and speech resources for two West African languages: Hausa, an Afroasiatic language with approximately 80-100 million speakers, and Fongbe, a Niger-Congo language spoken by approximately 2 million people in Benin. These languages represent contrasting cases on the resource availability spectrum. We address the question: \textit{What is the current state of publicly available NLP resources for Hausa and Fongbe, and what gaps remain?} Through systematic search of academic repositories, data platforms, and web sources, we catalog parallel corpora, monolingual text collections, speech datasets, pre-trained models, and evaluation benchmarks. For each resource, we document size, domain coverage, format, licensing, and accessibility. Our findings reveal that Hausa benefits from broader text resource diversity across news, encyclopedic, and educational domains. Fongbe, while having more limited text resources, has been the focus of recent academic speech data collection initiatives. Both languages are represented in Masakhane benchmarks for NER and POS tagging. We provide task-specific recommendations and identify priority gaps including domain-diverse Fongbe text and dedicated Hausa speech corpora.
>
---
#### [new 011] Same Model, Different Weakness: How Language and Modality Reshape the Jailbreak Attack Surface in Frontier MLLMs
- **分类: cs.CL**

- **简介: 该论文属于安全评估任务，研究语言和模态对多模态大模型越狱攻击的影响。通过跨语言实验，发现不同语言下模型的脆弱性差异，揭示了安全评估框架需重新设计。**

- **链接: [https://arxiv.org/pdf/2605.23157](https://arxiv.org/pdf/2605.23157)**

> **作者:** Casey Ford; Madison Van Doren; Sicheng Jin; Emily Dix
>
> **摘要:** The attack surface of a multimodal large language model (MLLM) is language-dependent in ways that reveal the mechanistic structure of alignment failures. We present the first systematic cross-lingual, multimodal red-teaming study comparing jailbreak vulnerability in US English (en-US) and Mexican Spanish (es-MX) across four frontier MLLMs: Claude Sonnet 4.5, GPT-5, Pixtral Large, and Qwen Omni. Using a fixed adversarial benchmark of 363 diverse prompt scenarios administered in text-only and multimodal conditions, we collected 52,272 harm ratings and binary attack success judgements from matched panels of nine native-speaker annotators per language group. Our central finding is that language does not scale vulnerability uniformly. Bayesian mixed-effects analyses reveal that linguistic framing attacks such as role-play become substantially less effective under Spanish prompting, while visually explicit multimodal attacks become more effective, which directly implicates the prompt-language interface rather than global annotator leniency. This dissociation indicates that linguistic and visual alignment failures operate through distinct mechanisms, and that switching language is sufficient to expose that separation. The practical consequence is that safety rankings are not preserved across languages. Qwen Omni overtakes Pixtral Large as the most vulnerable model among es-MX participants, a rank reversal no scalar correction of English-condition scores could recover, and absolute attack success rates have declined across model generations without closing the gaps between them. These findings demonstrate that safety evaluation frameworks treating language and modality as independent dimensions fundamentally misspecify the attack surface of globally deployed MLLMs, and must be redesigned accordingly.
>
---
#### [new 012] From Correctness to Preference: A Framework for Personalized Agentic Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于个性化强化学习任务，解决用户偏好差异带来的行为定制问题。提出PARPO框架和PSGM记忆模块，实现个性化策略优化与技能检索。**

- **链接: [https://arxiv.org/pdf/2605.23382](https://arxiv.org/pdf/2605.23382)**

> **作者:** Ranxu zhang; zeyang li; Jiacheng Huang; Rui Zhang; Xiaozhou Xu; sun zhe; Yanyong Zhang; Chao Wang
>
> **备注:** 34 pages, 7 figures, Under Review
>
> **摘要:** Agentic reinforcement learning (Agentic RL) has achieved strong progress in tasks with clear success signals. However, many real-world agent applications require user-conditioned behavior: the same query may call for different planning strategies and tool-use decisions across users. This setting raises key challenges: generic rewards cannot capture heterogeneous user preferences, observed behaviors are entangled with conformity effects, and flat memories cannot support personalized skill retrieval. To this end, we propose a unified personalized Agentic RL framework that embeds personalization into training-time optimization. At its core is \emph{Personalized Anchor Reward-Decoupled Policy Optimization} (\textbf{PARPO}), which decouples generic task-quality rewards from personalized preference rewards and uses user-specific anchors to stabilize learning under heterogeneous reward scales. We further introduce a two-stage preference-disentangled reward model and \emph{Preference-Aligned Skill Evolution Graph Memory} (\textbf{PSGM}) for personalized supervision and preference-aligned skill retrieval. Together, they form a closed loop of preference identification, policy optimization, and structured skill accumulation. Experiments on ETAPP, ETAPP-Hard, and SJAgent show that our framework consistently outperforms strong memory and RL baselines. Code and data are included in the supplementary materials.
>
---
#### [new 013] HawkesLLM: Semantic Uncertainty Propagation in Agentic Text Simulation
- **分类: cs.CL; stat.ML**

- **简介: 该论文提出HawkesLLM，解决文本生成中不确定性传播问题。通过分离时间建模与生成，提升后期语义对齐。属于文本模拟任务。**

- **链接: [https://arxiv.org/pdf/2605.23043](https://arxiv.org/pdf/2605.23043)**

> **作者:** Zewei Deng; Tinghan Ye; Liyan Xie
>
> **备注:** 10 pages, 4 figures, Accepted at the ICML 2026 Workshop on Statistical Frameworks for Uncertainty in Agentic Systems
>
> **摘要:** Agentic text-simulation systems write in sequence, with each item becoming possible context for later steps. That makes uncertainty path-dependent: an early ambiguity can affect later outputs. This paper studies this problem with HawkesLLM, a framework that separates temporal influence modeling from text generation. We represent the cascade as a network whose nodes are text-generating agents. A multivariate Hawkes process models how these nodes activate over time and which earlier node outputs should influence later prompts. A language model then writes each new event from the compact memory selected by this temporal model. We evaluate the framework on a held-out Global Database of Events, Language, and Tone (GDELT) news-cascade case study. The diagnostics track semantic alignment with local held-out references and separate local drift from global drift. In this setting, HawkesLLM improves late-stage semantic alignment under a compact prompt-memory budget.
>
---
#### [new 014] ARES: Automated Rubric Synthesis for Scalable LLM Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出ARES框架，解决大规模LLM强化学习中人工编写评分标准的难题。通过自动构建带评分标准的训练数据，提升开放性任务的奖励监督效果。**

- **链接: [https://arxiv.org/pdf/2605.23454](https://arxiv.org/pdf/2605.23454)**

> **作者:** Xiaoyuan Li; Keqin Bao; Moxin Li; Yubo Ma; Yichang Zhang; Wenjie Wang; Fuli Feng; Dayiheng Liu
>
> **备注:** Under Review
>
> **摘要:** Rubric-based rewards offer a promising way to extend reinforcement learning (RL) for large language models beyond tasks with automatically verifiable answers. However, scaling rubric-based RL remains challenging: existing approaches often rely on expert-written rubrics and manually constructed question sets, while fixed task-level rubrics may fail to capture the evaluation requirements of individual questions. We propose ARES (Automated Rubric synthEsis for Scalable RL), a framework for automatically constructing rubric-based RL data at scale. Starting from raw pretraining documents, ARES converts source knowledge into self-contained question-answer pairs and co-generates question-specific weighted rubrics, enabling instance-level reward supervision for open-ended responses. To improve diversity and quality, ARES conditions generation on domain labels and persona information, and applies validation filters for question self-containment, answer faithfulness, and rubric validity. Using ARES, we construct 100K rubric-annotated instances across ten domains. Experiments on seven benchmarks show that rubric-based RL trained with ARES, outperforms continual pretraining, supervised fine-tuning, and binary-reward RL, with the largest gains on multi-dimensional open-ended tasks such as healthcare and instruction following.
>
---
#### [new 015] Graph Alignment Topology as an Inductive Bias for Grounding Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于事实性检测任务，旨在解决大语言模型生成内容缺乏依据的问题。通过构建对齐图结构并训练图神经网络，提升生成内容的准确性。**

- **链接: [https://arxiv.org/pdf/2605.22963](https://arxiv.org/pdf/2605.22963)**

> **作者:** Paul Landes; Pranav Herur; Adam Cross; Jimeng Sun
>
> **摘要:** Large Language Models (LLMs) are optimized to produce distributionally plausible continuations rather than to explicitly verify whether generated propositions are entailed by source documents. This inductive bias enables generalization, but it does not encode whether responses are grounded with respect to a reference. These issues limit the use of LLMs in domains where strict factual correctness is crucial, such as clinical decision support. Existing hallucination detection approaches improve factuality through retrieval augmentation, self-consistency, or claim verification, but generally do not learn directly over alignment topology. To leverage alignment topology as an inductive bias, we construct aligned bipartite graphs between reference information and LLM outputs and train a graph neural network (GNN) to model alignment structure using message passing. The method achieves state-of-the-art results on four diverse hallucination and question-answering datasets, outperforming all compared methods, including foundational LLMs such as GPT-4o.
>
---
#### [new 016] Emotion Recognition in Sign Language Conversation
- **分类: cs.CL**

- **简介: 该论文属于手语情感识别任务，旨在解决现有数据缺乏对话上下文的问题。研究构建了eJSL Dialog数据集，并验证了通用模型在手语场景中的性能不足。**

- **链接: [https://arxiv.org/pdf/2605.23328](https://arxiv.org/pdf/2605.23328)**

> **作者:** Yusong Wang; Keyu Mao; Takao Obi; Minghao Shao; Kotaro Funakoshi
>
> **摘要:** Emotion Recognition in Conversation is a core component of affective computing, while current resources of sign language emotion datasets primarily focus on isolated sentences and lack conversational context. Models trained exclusively on these isolated utterances demonstrate degraded performance in real world scenarios because they cannot utilize historical dialogue flow. To address this structural limitation, we introduce the ERC task to sign language video analysis and propose the eJSL Dialog dataset. Constructed using the scripts from the STUDIES corpus, the dataset contains 1,920 video samples organized into 480 unique dialogues. We conduct systematic benchmarking on this dataset using models ranging from isolated visual networks to multimodal conversational architectures. The results reveal a domain gap when applying generic multimodal conversational emotion recognition models to sign language. These findings demonstrate the explicit need for context aware visual extractors specific to sign language and indicate that expanding the scale of conversational datasets to support large scale pre-training is a necessary next step for future research.
>
---
#### [new 017] ChartFI: Benchmarking Faithfulness and Insightfulness of Chart Descriptions from Multimodal Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于图表描述质量评估任务，旨在解决多模态大模型生成图表描述的准确性和洞察力问题。构建了ChartFI-Bench基准，并设计了评估指标。**

- **链接: [https://arxiv.org/pdf/2605.23694](https://arxiv.org/pdf/2605.23694)**

> **作者:** Fen Wang; Zekai Shao; Qiman Kang; Chunran Hu; Zhixuan Zhang; Lexu Xie; Chao Liu; Siming Chen
>
> **摘要:** Chart descriptions are essential for accessibility, cross-modal retrieval, and assisting readers in extracting insights from complex visualizations. As multimodal large language models (MLLMs) are increasingly adopted for automated chart description generation, a critical question arises: how faithfully and insightfully do these models actually describe charts? Current benchmarks fall short on two fronts: existing datasets consist of simple, homogeneous charts paired with shallow, fact-enumerating descriptions; and prevailing metrics fail to capture the multi-faceted nature of description quality. To address these gaps, we present the Chart Faithfulness and Insightfulness Benchmark (ChartFI-Bench). We first summarize four dimensions that characterize high-quality chart descriptions: factual accuracy, salient feature emphasis, domain-informed guidance, and chart-text complementarity. Guided by these dimensions, we construct a high-quality benchmark comprising 896 chart-description pairs, which feature visually complex charts and semantically rich descriptions. Furthermore, we design four aligned evaluation metrics -- Faithfulness, Coverage, Informativeness, and Acuity -- to systematically assess the quality of descriptions across these dimensions. Experiments conducted on mainstream MLLMs demonstrate the effectiveness of the proposed framework and reveal common weaknesses among existing models.
>
---
#### [new 018] Positional Failures in Long-Context LLMs: A Blind Spot in Reasoning Benchmarks
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究长文本中位置失效问题，针对推理基准设计缺陷，提出CRE评估框架，发现模型在长上下文中位置敏感，影响性能。**

- **链接: [https://arxiv.org/pdf/2605.23170](https://arxiv.org/pdf/2605.23170)**

> **作者:** Chuyifei Zhang; Hongyu Cui; Xiaowen Huang; Jitao Sang
>
> **备注:** 20 pages, 1 figure, 23 tables
>
> **摘要:** Position-controlled evaluation is standard for retrieval tasks such as Needle-in-a-Haystack and RULER, but mainstream reasoning benchmarks do not control positional placement of target tasks in long contexts. We audit 11 long-context benchmarks and find none jointly controls task position, filler content, and context length for reasoning. An audit of four flagship long-context releases finds no main result-table entry for NIAH, RULER, or LongBench-family benchmarks, while agentic and coding benchmarks appear in main result-tables across all four. We propose Context Rot Evaluation (CRE), a controlled framework varying all three factors, and evaluate nine LLMs on GSM8K and ARC-Challenge across two rounds: an initial five-model set and four newer vendor releases. Models can drop sharply when the target task moves from end to middle, and the drop grows worse with context length for vulnerable models. MiMo-v2-Flash drops 88pp at 64K under with_solutions filler (middle accuracy 8%). Newer releases show smaller drops: at 64K, three of four stay within +/-6pp of end-position accuracy; MiMo-V2.5-Pro narrows the MiMo-v2-Flash 88pp drop to 32pp. Under questions_only_v2 filler, middle-position drops persist across all four (range -16pp to -56pp across 8K, 32K, 64K). At 8K, a diagnostic probe adding a target-task copy at the end brings middle accuracy within +/-4pp of end baseline across all nine models, consistent with a positional explanation. In the initial five-model set, 76% of middle-position errors match surrounding filler text versus 22% at the end position, consistent with filler-answer interference as a dominant error mode. These results expose a structural evaluation gap in current reasoning benchmark design and vendor evaluation practice: positional vulnerabilities that grow with context length cannot be measured when task position is not controlled.
>
---
#### [new 019] RAS: Reflection-Augmented Scaling with In-Context Learning for Executable Cypher Query Generation
- **分类: cs.CL**

- **简介: 该论文属于自然语言到查询语句生成任务，解决生成不可执行Cypher查询的问题。通过引入RAS方法，利用执行反馈提升查询可执行性。**

- **链接: [https://arxiv.org/pdf/2605.22937](https://arxiv.org/pdf/2605.22937)**

> **作者:** Minseok Jung; Abhas Ricky; Muhammad Rameez Chatni
>
> **摘要:** Inference-time scaling can reduce errors in structured query generation, but methods to allocate the compute for query code generation remains underexplored. We study Text2Cypher, where language models generate Cypher queries that execute against property graph databases. Non-executable queries constitute a distinct syntactic failure separate from semantic inaccuracy: a syntax error triggers a system-generated error message from the database. These error messages are typically discarded at inference time rather than leveraged through in-context learning (ICL). We compare two inference methods: Independent Scaling (IS), which performs memoryless resampling, and Reflection-Augmented Scaling (RAS), which conditions each new attempt on prior execution feedback via ICL. Across three Neo4j datasets and five code-specialized language models, RAS reduces the Query Execution Error Rate by 41--50% at n{=}5, outperforming IS at 32--38%. Execution errors are not merely failures to discard but actionable feedback, and structuring inference-time compute around them is a more efficient path to executability than scaling independent samples.
>
---
#### [new 020] Can AI Guess What You Know? Performance Comparison of Large Language Models for Human Domain Knowledge Estimation From Communication Logs
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于知识估计任务，旨在解决从通信日志中自动推断个体领域知识的问题。研究对比了多个大语言模型的性能，发现Gemini 2.5 Flash表现最佳，但准确性与消息量关联不大。**

- **链接: [https://arxiv.org/pdf/2605.22971](https://arxiv.org/pdf/2605.22971)**

> **作者:** Ko Watanabe; Shoya Ishimaru
>
> **摘要:** Employees often struggle to identify ``who knows what,'' leading to organizational productivity losses. We investigate whether Large Language Models (LLMs) can infer individual domain knowledge directly from long-term Slack logs. Analyzing 27,188 messages from 43 users, we evaluated seven models (including Gemini, Claude, and GPT families) by comparing their zero-shot estimates against self-reported skill ratings from 27 participants. Gemini 2.5 Flash achieved the lowest error (MAE 21.13%), while GPT models showed significantly larger discrepancies. Notably, estimation accuracy depended only weakly on message volume, indicating that more text alone does not guarantee better inference. These findings demonstrate the feasibility and current limits of automated expertise mapping, highlighting the need for privacy-preserving deployments and richer, structure-aware representations of human knowledge.
>
---
#### [new 021] Fast-dDrive: Efficient Block-Diffusion VLM for Autonomous Driving
- **分类: cs.CL**

- **简介: 该论文提出Fast-dDrive，一种高效的块扩散视觉-语言-动作模型，用于自动驾驶。解决轨迹规划与推理效率的平衡问题，通过结构化输出和优化解码策略提升性能。**

- **链接: [https://arxiv.org/pdf/2605.23163](https://arxiv.org/pdf/2605.23163)**

> **作者:** Kewei Zhang; Jin Wang; Sensen Gao; Chengyue Wu; Yulong Cao; Songyang Han; Boris Ivanovic; Langechuan Liu; Marco Pavone; Song Han; Daquan Zhou; Enze Xie
>
> **摘要:** End-to-end autonomous driving via Vision-Language-Action (VLA) models demands a precarious balance between high-fidelity trajectory planning and efficient inference. Existing paradigms typically fall short: autoregressive (AR) VLAs are memory-bandwidth-bound on edge hardware and prone to exposure-bias drift, while full-sequence diffusion models preclude KV-cache reuse and suffer from "logical leakage" that violates the fundamental perceive-then-plan causality. We present Fast-dDrive, a block-diffusion VLA that performs bidirectional refinement within semantic units while enforcing strict causal ordering across them. Leveraging the observation that driving VLAs often emit structured JSON-like outputs, Fast-dDrive freezes structural tokens into a section scaffold and employs a section-aware training recipe that prioritizes safety-critical planning. We further introduce Scaffold Speculative Decoding to achieve AR-equivalent quality at significantly higher throughput. Finally, we propose a low-overhead test-time scaling scheme: by forking $N$ stochastic trajectory rollouts from a single shared-prefix KV cache and averaging them, we effectively suppress prediction variance at a fractional computational cost. Empirical results demonstrate that Fast-dDrive redefines the speed-accuracy frontier for driving agents. On the WOD-E2E test set, Fast-dDrive achieves SOTA ADE@3s and ADE@5s, alongside the highest RFS among diffusion-based VLAs; on nuScenes, it reduces average L2 error to $0.32$m (a $22\%$ improvement). When integrated with SGLang, our framework delivers $12\times$ throughput speedup over the AR baseline, narrowing the gap between high-capacity VLAs and the efficiency demands of real-time on-vehicle deployment.
>
---
#### [new 022] Self-Improving In-Context Learning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在提升少样本学习效果。通过优化提示嵌入，提高模型对任务的推理能力，无需微调或生成新文本。**

- **链接: [https://arxiv.org/pdf/2605.23180](https://arxiv.org/pdf/2605.23180)**

> **作者:** Baturay Saglam; Dionysis Kalogerias
>
> **摘要:** We propose to improve in-context learning (ICL) by optimizing the continuous embeddings of a fixed few-shot prompt at test time. The key observation is that the log-probabilities a model assigns to its demonstrated outputs$\unicode{x2013}$available from a single forward pass without generating any tokens$\unicode{x2013}$provide a meaningful signal for how well the model has inferred the task from its demonstrations. We formalize this signal as a bounded, self-supervised confidence proxy and maximize it via zeroth-order optimization over the prompt embeddings, yielding a test-time calibration procedure. The approach requires no finetuning, no token generation, no predefined label set, and no external data, making it equally applicable to both classification and free-form generation tasks. Across a comprehensive suite of ICL tasks, the proposed calibration consistently matches or improves upon the base model and outperforms classification-specific baselines on most tasks. The statistically significant correlation between proxy improvement and downstream accuracy gain confirms that the proposed proxy encodes a reliable optimization signal for in-context learning.
>
---
#### [new 023] SSDAU: Structured Semantic Data Augmentation for Joint Entity and Relation Extraction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于联合实体与关系抽取任务，旨在解决数据质量差导致模型泛化能力弱的问题。提出SSDAU方法，通过保持语义结构进行数据增强，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.23440](https://arxiv.org/pdf/2605.23440)**

> **作者:** Jiawei He; Mengyu Shi; Chunrong Fang
>
> **备注:** 12 pages, 3 figure
>
> **摘要:** Joint Entity and Relation Extraction (JERE) is highly susceptible to weak generalization due to low-quality training data. Data augmentation is a common strategy to enhance model generalization across different domains. However, existing data augmentation methods often overlook text relevance and may disrupt semantic structures and dependencies, making it difficult to generate effective augmented data for improving model generalization. In this paper, we propose Structured Semantic Data Augmentation (SSDAU), a novel method designed to preserve the semantic structure of text during augmentation. SSDAU segments text based on entity labels and employs an encoder to capture semantic features of entities through context awareness. It then performs entity semantic restructuring to generate augmented data. To distinguish semantically similar entities, SSDAU fuses contextualized embeddings with traditional similarity scores. To mitigate potential topic ambiguity and information loss, we apply the BERTTopic model to filter out irrelevant topics, ensuring topic consistency. We evaluate SSDAU on datasets with different annotation types and compare its performance on five representative JERE models against seven popular data augmentation baselines. Experiments demonstrate that SSDAU generates semantically consistent data with superior robustness against ambiguity (8.26\% F1 decrease vs.\ 31.91\% for baselines), significantly outperforming all existing methods across all metrics.
>
---
#### [new 024] Benchmarking Google Embeddings 2 against Open-Source Models for Multilingual Dense Retrieval and RAG Systems
- **分类: cs.CL**

- **简介: 该论文对比了Google Embeddings 2与五种开源模型在多语言密集检索和RAG系统中的性能，评估其效果与效率，旨在找出最优模型。**

- **链接: [https://arxiv.org/pdf/2605.23618](https://arxiv.org/pdf/2605.23618)**

> **作者:** Stefano Cirillo; Domenico Desiato; Giuseppe Polese; Giandomenico Solimando
>
> **备注:** 9 pages, 2 figures, 5 tables. Text and evaluation code available at this https URL
>
> **摘要:** We benchmark Google Embeddings (GE2), a Vertex-AI-hosted bi-encoder with 2,048-token context and explicit task-type conditioning, against five open-source alternatives: BGE-M3, E5-large, Multilingual-E5-large (mE5-L), LaBSE, and Paraphrase-Multilingual-MPNet (mMPNet). Evaluation covers four BEIR subsets, a synthetic Italian RAG corpus, a chunking ablation considering 5 sizes of tokens with three strategies, and per-query latency on commodity CPU hardware. GE2 ranks first on every task, achieving BEIR this http URL@10 = 0.638 and IT-RAG-Bench nDCG@10 = 0.282, but at 231.6 ms median latency, it is roughly 14x slower than the fastest local models. mE5-L reaches within 0.003 nDCG of GE2 on Italian at 31 ms, making it the preferred option when sub-100 ms SLAs matter. A more striking finding concerns LaBSE, which, despite widespread multilingual deployment scores 0.188 average nDCG@10 on BEIR, below every dedicated retrieval model including mMPNet. Chunking experiments show that all six models saturate at 32-token chunks on our corpus, with semantic chunking providing measurable gains only at 16 tokens.
>
---
#### [new 025] Multilingual Steering by Design: Multilingual Sparse Autoencoders and Principled Layer Selection
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决多语言下SAE控制不稳定的问题。通过多语言训练和理论层选择方法，提升语言控制的可靠性与质量。**

- **链接: [https://arxiv.org/pdf/2605.23036](https://arxiv.org/pdf/2605.23036)**

> **作者:** Yusser Al Ghussin; Daniil Gurgurov; Tanja Baeumel; Josef van Genabith; Patrick Schramowski; Simon Ostermann
>
> **备注:** Accepted to TrustNLP Workshop at ACL 2026
>
> **摘要:** Sparse autoencoders (SAEs) enable feature-level mechanistic interpretability and activation steering in large language models (LLMs), but SAE-based language control remains unreliable in multilingual settings: most SAEs are trained on English-only data, and steering layers are chosen heuristically. We address these limitations by advancing a principled, mechanistic account of multilingual language steering with SAEs. First, we show that training SAEs on multilingual data consistently strengthens cross-lingual representations and yields more reliable, quality-preserving language control across layers and model families. Second, we introduce an \emph{a priori} steering layer-selection rule based on the intersection of multilingual alignment and language separability, which predicts effective intervention depths without exhaustive layerwise search. We evaluate our approach on LLaMA-3.1-8B and Gemma-2-9B across machine translation and cross-lingual summarization (CrossSumm), using SpBLEU, ROUGE-L, COMET, and LaSE. Our results show that multilingual SAEs combined with intersection-selected layers stabilize the trade-off between language identification accuracy and generation quality, providing a principled, predictive, representation-level account of multilingual SAE steering.
>
---
#### [new 026] Naturalistic measure of social norms alignment
- **分类: cs.CL**

- **简介: 该论文属于社会规范对齐研究，旨在解决自然场景下社会规范一致性测量的问题。通过构建数据集和提出新指标，评估不同主体在社会困境中的回应一致性。**

- **链接: [https://arxiv.org/pdf/2605.23420](https://arxiv.org/pdf/2605.23420)**

> **作者:** Yevhen Kostiuk; Kenneth Enevoldsen; Peter Bjerregaard Vahlstrup; Márton Kardos; Kristoffer Nielbo
>
> **摘要:** Social norms reflect shared expectations on acceptable behavior. Measuring social norms alignment remains challenging, with existing approaches typically relying on artificial closed-form evaluations such as multiple-choice questionnaires or measuring agreement with predefined statements. In the context of this work, social norms alignment refers to measuring an agreement between solutions with respect to the social problem or dilemma. We propose a framework for measuring social norm alignment in naturalistic, free-form settings through solution matching. The framework enables us to measure alignment between any two dilemma responses e.g., LLMs to a human, LLMs to LLMs, or human to human. We introduce two metrics: stated and explicit agreement accuracy, and construct a dataset of 3k non-trivial social dilemmas in Danish. All dilemmas are assigned reference solutions derived from three panelists, who serve as culturally grounded judges. We evaluate the agreement of several LLMs and human responses in an interaction setup that resembles natural user-model conversations. Our results show that the proposed metrics produce consistent model rankings and reveal variation in agreement across different types of dilemmas, with higher agreement observed for topics such as neighbor conflicts and shared living situations. Overall, our work introduces a dataset and evaluation framework for studying culturally grounded social reasoning in naturalistic open-ended conversations.
>
---
#### [new 027] What Training Data Teaches RL Memory Agents: An Empirical Study of Curriculum Effects in Memory-Augmented QA
- **分类: cs.CL**

- **简介: 该论文研究强化学习中记忆增强问答系统的训练数据影响，探讨不同训练课程对模型技能的影响。任务为记忆增强的问答系统训练，解决如何通过数据组合优化模型性能的问题。工作包括对比不同训练集效果，发现混合数据最优，并提出训练优化建议。**

- **链接: [https://arxiv.org/pdf/2605.23067](https://arxiv.org/pdf/2605.23067)**

> **作者:** Xinjie He; Zhiyuan Lin; Su Liu; Jialun Wu; Qiyang Xie; Weikai Zhou; Shuai Xiao
>
> **备注:** 14 pages, 2 figures, 11 tables. Code, checkpoints, and evaluation artifacts available at this https URL
>
> **摘要:** Reinforcement learning (RL) has emerged as a viable recipe for training LLM agents to reason over external memory banks in multi-session dialogue. Existing work trains exclusively on a single benchmark, leaving open how the composition of training data shapes the skills a memory agent acquires. We present a controlled empirical study that holds architecture, RL algorithm, and all hyperparameters fixed and varies only the training curriculum across three conditions: in-domain (LoCoMo), mixed-benchmark (LoCoMo + LongMemEval), and out-of-domain (LongMemEval only). Across two benchmarks and ten question types, curriculum composition acts as a fine-grained lever on specialization rather than a uniform scaling factor on performance. The mixed curriculum yields the strongest overall F1 on both evaluation sets. Training on a narrow out-of-domain set transfers a targeted skill - temporal reasoning - despite weak aggregate performance. Per-type differences substantially exceed aggregate differences, indicating that single-number benchmark comparisons systematically underreport curriculum effects. We further report two practical lessons from adapting GRPO to a single-GPU regime: cross-benchmark mixing requires filtering format-specific noise from memory banks to preserve training signal, and binary exact-match reward produces no learning signal at the small group sizes (G = 4) required on one GPU, motivating continuous reward functions in this regime.
>
---
#### [new 028] A graph-based analysis of semantic types and coercion in contextualized word embeddings
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决语义类型不匹配导致的强制现象。通过构建图模型和提出新指标，分析词嵌入中的语义类型信息。**

- **链接: [https://arxiv.org/pdf/2605.23710](https://arxiv.org/pdf/2605.23710)**

> **作者:** Long Chen; Deniz Ekin Yavas
>
> **摘要:** Semantic type mismatch between a noun and its context is central to coercion phenomena. This paper introduces a graph-based method to examine how lexical and contextual type information is reflected in word embeddings. We select nouns from ten semantic types, annotate corpus instances for type matching (matching vs. coercion vs. other mismatch vs. unrestricted), and construct graphs using BERT and sense-enhanced embeddings. Two metrics -- Neighbor Type Probability (NTP) and Neighbor Type Entropy (NTE) -- are proposed to analyze neighborhood type distributions. Results show that graphs constructed with sense-enhanced embeddings reflect semantic type information better, and matching and mismatch sentences can be distinguished through the proposed metrics.
>
---
#### [new 029] ClimateChat-300K: A Multi-Modal Facebook Dataset for Understanding Diverse Perspectives in Climate Communication
- **分类: cs.CL**

- **简介: 该论文介绍了一个用于气候传播研究的多模态数据集ClimateChat-300K，旨在分析公众对气候变化的不同观点。任务是理解气候讨论中的公众参与和情感倾向，解决信息极化与误解问题。工作包括数据收集、主题建模与情感分析。**

- **链接: [https://arxiv.org/pdf/2605.23326](https://arxiv.org/pdf/2605.23326)**

> **作者:** Wajdi Zaghouani; Md. Rafiul Biswas; Mabrouka Bessghaier; Shimaa Ibrahim; George Mikros
>
> **摘要:** We present ClimateChat-300K, a large-scale dataset of 299,329 public Facebook posts about climate change collected between May 2020 and May 2024 through the CrowdTangle platform. The dataset contains 41 metadata features including post content, engagement metrics, and page attributes, covering material from more than 26,000 global pages. Each post includes rich contextual information such as language, timestamp, page category, and interaction counts, enabling comprehensive analyses of public discourse around climate communication. Using topic modeling and sentiment analysis, we identify ten main themes grouped into five domains: policy, activism, cooperation, science, and conservation. The results reveal that emotional tone, post format, and page identity strongly influence audience engagement, with visually rich and emotionally charged content receiving the highest levels of interaction. The dataset also demonstrates how online discussions evolved in response to major events such as international climate summits and the COVID-19 pandemic period. ClimateChat-300K provides an open resource for reproducible and interdisciplinary research on polarization, misinformation, and the dynamics of digital climate discourse. By releasing this dataset, we aim to support transparent, data-driven research and contribute to a deeper un-derstanding of how public engagement with climate issues develops across time, geography, and institutional contexts.
>
---
#### [new 030] NLG Evaluation: Past, Present, Future
- **分类: cs.CL**

- **简介: 论文探讨自然语言生成（NLG）评估的发展与未来趋势，分析其从语言学向机器学习的转变，以及评估方法的演进，旨在解决NLG技术的高效、安全和质量评估问题。**

- **链接: [https://arxiv.org/pdf/2605.23715](https://arxiv.org/pdf/2605.23715)**

> **作者:** Ehud Reiter
>
> **备注:** Will appear in Proceeedings of RetroEval 2026
>
> **摘要:** Natural Language Generation (NLG) evaluation has changed dramatically since 1990, and will continue to evolve in the future. In 1990, when NLG had close ties to linguistics, there was very little formal experimental evaluation in the modern sense. In 2026, when NLG is closely linked to machine learning, experimental evaluation is expected and indeed fundamental to research. Many evaluation techniques were developed over this period, including most recently LLM-as-Judge. I expect NLG evaluation will continue to evolve in the future. In particular, impact, qualitative, and safety evaluation will become more important as large numbers of people routinely use NLG technology.
>
---
#### [new 031] Brain-LLM Alignment Tracks Training Data, Not Typology
- **分类: cs.CL; cs.AI; q-bio.NC**

- **简介: 该论文研究跨语言脑-大模型对齐问题，分析训练数据与语言类型的影响。通过fMRI数据和多模型测试，发现对齐模式由训练语言主导，而非语言本身特性。**

- **链接: [https://arxiv.org/pdf/2605.23032](https://arxiv.org/pdf/2605.23032)**

> **作者:** Dongxin Guo; Jikun Wu; Siu Ming Yiu
>
> **备注:** Accepted to CoNLL 2026. 9 pages main content + 4 pages references + 6 pages appendix; 4 figures, 13 tables
>
> **摘要:** Brain-LLM alignment is well established in English, yet the brain's language network is neuroanatomically universal across languages. Does alignment also generalize cross-linguistically, and what governs the variation? We test this using fMRI data from 112 participants across English, Chinese, and French (the Le Petit Prince corpus) and seven LLMs spanning English-dominant, Chinese-dominant, and multilingual architectures. Our central finding is that training-language dominance, not an inherent property of English, drives the alignment pattern: a Chinese-dominant model (Baichuan2-7B), architecture-matched to LLaMA-2-7B, reverses the gradient entirely, aligning best with Chinese brains and worst with English. Beyond training dominance, formal typological distance independently covaries with alignment degradation, syntax-associated brain regions (IFG) show $2.3\times$ steeper typological gradients than lexico-semantic regions (PTL), and tokenization fertility accounts for $\sim$60% of a cross-linguistic shift in optimal encoding layer. These results reveal that the apparent "English advantage" in brain-LLM alignment is an artifact of training data composition, while the remaining variation reflects genuine typological structure concentrated in syntactic processing.
>
---
#### [new 032] AraHopeCorpus: Annotation Guidelines and Dataset for Hope Speech in Arabic Social Media Crisis Discourse
- **分类: cs.CL**

- **简介: 该论文介绍AraHopeCorpus，首个阿拉伯语希望言论数据集，用于研究冲突期间社交媒体中的积极话语。任务为希望言论识别，解决其在阿拉伯语语境中研究不足的问题。工作包括数据收集、标注及分析。**

- **链接: [https://arxiv.org/pdf/2605.23325](https://arxiv.org/pdf/2605.23325)**

> **作者:** Esra'a Sharqawi; Wajdi Zaghouani
>
> **摘要:** Social media has become a crucial arena for shaping public narratives during armed conflicts, providing space for both harmful and constructive communication. While hate speech and misinformation have been widely studied, expressions that promote resilience, solidarity, and optimism remain underexplored, particularly in Arabic contexts. This paper introduces AraHopeCorpus, the first annotated dataset of Arabic hope speech collected from ten thousand YouTube comments related to the war on Gaza between 2023 and 2024. Using a detailed annotation framework, comments were classified into three categories: hope speech, no hope speech, and neutral or unclear discourse. The dataset shows that hopeful language dominates, accounting for more than sixty four percent of all comments. These expressions of hope appear mainly as religious encouragement, collective solidarity, and optimism for endurance and justice. No hope speech, representing about thirteen percent, reflects despair and disillusionment, while the rest of the comments contain neutral or mixed content. Inter-Annotator Agreement reached substantial levels (Cohen's Kappa equals 0.71), though dialectal variation, sarcasm, and implicit meaning posed annotation challenges. A comparative analysis between human annotators and ChatGPT revealed that large language models can support annotation but remain limited in handling dialectal and culturally embedded expressions. AraHopeCorpus will be released for research purposes under an open and non commercial license. It provides a valuable resource for studying constructive digital discourse, enabling further research on hope speech detection, crisis communication, and resilience in Arabic social media.
>
---
#### [new 033] Convergence Without Understanding: When Language Models Agree on Representations but Disagree on Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型在表示层面趋同但推理过程不一致的现象，分析了16个模型在800个推理题上的表现，揭示了三类差异，对模型集成和可解释性有重要影响。**

- **链接: [https://arxiv.org/pdf/2605.23315](https://arxiv.org/pdf/2605.23315)**

> **作者:** Muhammad Usama; Dong Eui Chang
>
> **摘要:** Large language models trained under diverse objectives and architectures have been shown to develop increasingly similar internal representations, an observation formalized as the Platonic Representation Hypothesis. Whether this representational convergence extends to the reasoning processes that operate over shared representations remains untested. We evaluate representational similarity across 16 language models from 8 families (1.5B to 72B parameters) on 800 reasoning problems spanning mathematics, science, commonsense, and truthfulness, stratifying by problem difficulty, computational stage, and causal relevance. Our analysis reveals three dissociations: a difficulty inversion, where models converge more on problems they collectively fail (Centered Kernel Alignment [CKA] = 0.897) than on those they solve (CKA = 0.830); a generation gap, where pre-decision representations align (CKA = 0.875) while post-decision representations diverge (CKA = 0.274); and epiphenomenal correctness, where shared information is decodable across models (66% transfer accuracy) but exerts minimal causal influence on predictions (1.5% to 5.5% flip rate across ablation protocols). These results indicate that representational convergence in language models reflects shared input processing constraints rather than shared reasoning strategies, with direct implications for ensemble design, interpretability transfer, and evaluations of model similarity. Code is available at this https URL.
>
---
#### [new 034] Evaluating Large Language Models in a Complex Hidden Role Game
- **分类: cs.CL; cs.AI; cs.GT; cs.MA**

- **简介: 该论文属于AI安全领域，研究LLM在复杂社交推理游戏中的欺骗能力。通过框架和指标评估模型表现，发现其战略深度不足，难以有效进行多轮操纵。**

- **链接: [https://arxiv.org/pdf/2605.22826](https://arxiv.org/pdf/2605.22826)**

> **作者:** Niklas Bauer
>
> **备注:** Master's thesis, University of Göttingen
>
> **摘要:** Quantifying the deceptive potential of Large Language Models (LLMs) is critical for AI safety, yet difficult to achieve in uncontrolled environments. This work investigates the reasoning, persuasion, and deceptive capabilities of LLMs within the social deduction game Secret Hitler. I introduce an open-source framework and novel metrics to measure performance: Role Identification Accuracy, Deception Retention Rate, and Game State Impact Rate. By benchmarking models against rule-based algorithms and human games, I identify a gap between conversational ability and strategic depth. The study also analyzes the impact of reasoning-enhancement techniques on win rates and strategic reasoning. Neither Chain-of-Thought prompting nor internal memory bring improvements in performance, with up to 23.2% worse win rates for fascist roles. While rule-based agents align with expert human voting decisions 86.7% of the time, models like Llama 3.1 70B achieve only a 59.7% accuracy. Models playing as Fascists consistently yield negative impact scores and fail to sustain deception, resulting in roughly 40% shorter games compared to humans. These findings suggest that current architectures remain ineffective at complex, multi-turn manipulation. As capabilities advance, detecting when models begin to master these deceptive behaviors is crucial. The developed framework serves as a reproducible testbed for future alignment research.
>
---
#### [new 035] A Fine-Tuned BERT Classifier for Personal-Letter Titles in Late-Ming and Early-Qing Collected Works
- **分类: cs.CL; cs.AI; cs.CY; cs.DB**

- **简介: 该论文提出Lepton模型，用于区分明清文集目录中的个人书信与近似标题的序文。属于文本分类任务，解决古籍标题识别问题。**

- **链接: [https://arxiv.org/pdf/2605.23103](https://arxiv.org/pdf/2605.23103)**

> **作者:** Queenie Luo
>
> **摘要:** I present Lepton (Letter Prediction), a fine-tuned BERT classifier that predicts whether a title in a Classical Chinese wenji table of contents is a personal letter or a closely confusable preface (particularly the farewell-preface). Lepton fine-tunes bert-base-chinese on 5438 hand-labeled wenji titles from thirty-three late-Ming and early-Qing literati. I've deployed the model on Hugging Face and has been used at the China Biographical Database (CBDB) to identify approximately fifty-five thousand letters across mid-Ming through early-Qing wenji, populating the Ming Letter Platform.
>
---
#### [new 036] When AI Takes Sides on Questions of Faith: Persistent Asymmetries in AI-Mediated Faith Guidance
- **分类: cs.CL; cs.CY**

- **简介: 该论文研究AI在处理宗教转换建议时的不对称性，属于自然语言处理中的偏见检测任务。它揭示了LLM对不同宗教的倾向性，指出模型存在系统性偏好。**

- **链接: [https://arxiv.org/pdf/2605.22975](https://arxiv.org/pdf/2605.22975)**

> **作者:** Brett Israelsen; Sheryl Carty; Josh Coates; Nancy Fulda; Julie Park; Pete Whiting
>
> **备注:** 29 pages, 16 figures
>
> **摘要:** We ask whether large language models (LLMs) treat queries about religious conversion symmetrically. The answer is no. When asked for advice on hypothetical faith transitions from one religion to another, then asked the reversed question, models exhibited consistent asymmetries, favoring some religions while subtly discouraging conversion to others. On average Catholic, Bahá'í, and Sikh religions were broadly favored (high support for joining, low support for leaving), while Atheists, Agnostics, and Jehovah's Witnesses were primarily disfavored. Patterns varied by model size and model provider, with Grok 4.20 exhibiting the strongest asymmetries. We tested 20 commercial and open-source language models across 182 religion pairings using a human-verified LLM-as-a-judge framework. Each model was probed via interactions with a simulated user asking for advice on a potential faith conversion. Models tended to use more encouraging language for some faith transitions over others; these patterns were systematically repeatable across multiple trials. All LLMs tested exhibited reproducible asymmetry, though the pattern of preferences differed for each. Overall preferences persist across multiple question phrasings and variations in the religious pairing dataset. Taken together, these results suggest that asymmetry is a robust property of model behavior rather than an artifact of how the models' answers were scored. It is important to consider that any imbalances deployed and reproduced en masse can have real-world implications.
>
---
#### [new 037] Is a Document Educational or Just Wikipedia-Style? -- Pitfalls of Classifier-Based Quality Filtering
- **分类: cs.CL**

- **简介: 该论文研究分类器质量过滤任务，揭示其在处理维基风格文档时的漏洞，指出简单重格式化可使低质内容通过过滤，影响预训练语料质量。**

- **链接: [https://arxiv.org/pdf/2605.23721](https://arxiv.org/pdf/2605.23721)**

> **作者:** Mateusz Klimaszewski; Piotr Andruszkiewicz
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** Classifier-based Quality Filtering has recently emerged as a fundamental technique in constructing pre-training corpora. The ability to deploy a single model that can replace or supplement a set of heuristics has proven effective across numerous Large Language Models. In this work, we expose a critical vulnerability in this approach by demonstrating how a straightforward Wikipedia-style reformatting operation can substantially alter a model's quality assessment and enable low-quality content to surpass filtering thresholds. Our analysis reveals that the FineWeb-Edu CQF model would reverse its filtering decision for approximately 7% of evaluated documents, thereby admitting content into the pre-training corpus that would otherwise have been excluded.
>
---
#### [new 038] When Symptoms Are Not Enough: Evidence-Weighting Patterns in Large Language Model Psychiatric Screening
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于心理疾病筛查任务，旨在评估大语言模型在精神疾病诊断中的可靠性。研究通过分析患者叙述，探讨模型对症状、功能损害和保护性情境的证据权重，以提高筛查准确性。**

- **链接: [https://arxiv.org/pdf/2605.23148](https://arxiv.org/pdf/2605.23148)**

> **作者:** Jianfeng Zhu; Megan Korhummel; Ruoming Jin; Karin G. Coifman
>
> **备注:** 25 pages 7 figures
>
> **摘要:** As demand for mental health care outpaces clinician-delivered assessment, scalable screening tools are increasingly needed. Large language models (LLMs) may identify psychiatric risk from patient narratives, but their reliability across diagnoses, demographic subgroups, and evidence-use patterns remains uncertain. We introduce a SCID-anchored benchmark of 555 semi-structured experiential interviews paired with diagnostic reference labels for anxiety disorder, major depressive disorder, post-traumatic stress disorder, and any current mental health disorder. Using zero-shot task-specific prompting, we evaluated five state-of-the-art LLMs and examined whether false-negative errors reflected missed psychiatric evidence or differential weighting of symptom, functional-impairment, and protective-context cues. Performance varied across tasks and models, with accuracy ranging from 0.49 to 0.86 and Matthews correlation coefficients from 0.16 to 0.38. GPT-4.1 Mini and GPT-5 Mini showed the most consistent disorder-specific accuracy. Subgroup analyses found higher depression-classification accuracy among male than female participants, no consistent age-related pattern, and modest non-uniform variation across race strata. Evidence-integration analyses showed that false-negative anxiety and PTSD classifications often contained explicit symptom evidence but were accompanied by preserved functioning, coping ability, or social support. Functional-impairment evidence shifted model outputs toward positive classifications, whereas protective-context evidence shifted outputs away. These findings suggest that LLMs may support scalable psychiatric screening, but their tendency to discount symptom evidence in the presence of preserved functioning or protective context requires careful validation before clinical deployment.
>
---
#### [new 039] Cultural Adaptation in Large Language Models for Political Discourse
- **分类: cs.CL**

- **简介: 论文探讨了大语言模型在政治话语中的文化适应问题，旨在解决跨文化应用中的系统性误差。通过构建评估矩阵，提出方法路径以提升模型的民主安全性和文化准确性。**

- **链接: [https://arxiv.org/pdf/2605.23332](https://arxiv.org/pdf/2605.23332)**

> **作者:** Wajdi Zaghouani
>
> **摘要:** The integration of large language models into political discourse analysis creates new opportunities for comparative research, policy analysis, and civic technology, while introducing material risks for democratic accountability. This paper argues that cultural adaptation is a prerequisite for trustworthy deployment of large language models in political communication across diverse linguistic and institutional contexts. Current systems remain shaped by English dominant data, uneven multilingual coverage, and assumptions grounded in a narrow range of political institutions and discourse conventions, producing systematic errors when applied across cultures. We formalize cultural adaptation across translation, discourse, and ontology levels, identify recurring cultural failure modes in political NLP, and propose an operational evaluation matrix grounded in cultural fidelity, calibration, and democratic safety. Building on political text analysis, sociotechnical auditing, and cross cultural pragmatics, we outline methodological pathways including participatory dataset development, culturally aware transfer learning, and benchmark design that makes cultural adaptation empirically measurable. We conclude by clarifying governance constraints and scope conditions under which culturally adaptive political NLP can support democratic legitimacy.
>
---
#### [new 040] Query-Adaptive Semantic Chunking for Retrieval-Augmented Generation: A Dynamic Strategy with Contextual Window Expansion
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于信息检索任务，旨在解决固定文档分块导致的精度与召回率矛盾问题。提出QASC方法，通过查询自适应动态构建语义块，提升检索相关性。**

- **链接: [https://arxiv.org/pdf/2605.22834](https://arxiv.org/pdf/2605.22834)**

> **作者:** Mudit Rastogi
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems depend critically on document chunking quality for retrieving relevant context. Fixed chunking segments documents into uniform units irrespective of semantics or user intent, producing a precision-recall trade-off unresolvable by tuning chunk size alone. Semantic and agentic methods partially address these limitations but do not integrate user queries at the chunking stage. We present Query-Adaptive Semantic Chunking (QASC), which dynamically constructs chunks by integrating queries into segmentation through three mechanisms: cosine similarity scoring between sentence and query embeddings to identify seed sentences, contextual window expansion around seeds to preserve coherence, and chunk-level score aggregation to ensure holistic relevance. We evaluate QASC on 100 technical documents across 200 queries spanning four types, comparing against fixed chunking at five granularities, recursive splitting, semantic chunking, and agentic chunking. QASC achieves an F1-score of 0.85, a relative improvement of 18-27% over fixed chunking and 8-12% over semantic and agentic alternatives. Ablation studies confirm each component contributes meaningfully. Human evaluation by three annotators (Cohen kappa = 0.82) corroborates that QASC produces more relevant and coherent chunks than existing methods.
>
---
#### [new 041] Articulatory strategy as a source of variation in acoustic vowel dynamics
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音学研究，探讨发音策略如何影响元音声学动态。通过分析36名英语母语者的数据，发现舌位变化与元音共振峰动态相关，揭示了个体发音差异的机制。**

- **链接: [https://arxiv.org/pdf/2605.23416](https://arxiv.org/pdf/2605.23416)**

> **作者:** Patrycja Strycharczuk; Justin J. H. Lo; Sam Kirkham
>
> **摘要:** Acoustic vowel dynamics have some speaker-identifying characteristics, which have been ascribed to individual properties of articulatory strategies: formant transitions have a particular shape because speakers move their articulators, using specific and practised movements. However, there is little existing evidence that different articulatory strategies systematically affect formant dynamics. The present study corroborates the link between the two. Ultrasound tongue imaging data from 36 speakers of Northern-Anglo English are used to identify distinct articulatory strategies for the production of palatal vowel /i/. Tongue shape in /i/ is found to be a significant predictor of formant dynamics in diphthongs with a palatal offglide. The observed relationships can be explained by the characteristics of articulatory movement conditioned by vocal tract shape. Greater articulatory displacement of tongue root and/or dorsum produces greater distortion from the mean tongue shape in palatal vowels, and it also requires higher articulatory velocities, resulting in relatively earlier and steeper formant transitions. The results contribute to the conceptual understanding of individuality in speech, by illuminating the regularising and individual aspects of articulatory compensation.
>
---
#### [new 042] Model Collapse as Cultural Evolution
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究模型崩溃现象，将其视为文化演化过程。通过实验验证了迭代学习理论，提出并验证了五项预测，揭示了结构退化规律，为自训练流程设计提供依据。**

- **链接: [https://arxiv.org/pdf/2605.23054](https://arxiv.org/pdf/2605.23054)**

> **作者:** Dongxin Guo; Jikun Wu; Siu Ming Yiu
>
> **备注:** Accepted at CoNLL 2026. 18 pages, 3 figures, 2 tables
>
> **摘要:** Model collapse, the progressive degradation of LLMs trained on their own outputs, has been characterized statistically but lacks a linguistic explanation for which structures degrade, in what order, and why. We show that iterated learning theory from cultural evolution fills this gap. We derive five falsifiable predictions, distinguish those uniquely discriminative for the theory from confirmatory ones, and test them by self-training LLaMA-2-7B and Mistral-7B over 10 generations in English, German, and Turkish. The critical discriminative finding: compositionality follows a non-monotonic trajectory (initially rising, then falling) under unfiltered self-training. This signature persists with maximally regular seed data (ruling out noise removal) and is sustained only by task-grounded filtering, not random filtering, providing the first LLM-scale evidence for the compression-communication tradeoff. All predictions are confirmed with large effect sizes (Hedges' $g > 1.6$; $\mathrm{BF}_{10} > 100$), and LLM regularization gradients closely match human behavioral data ($R^2 = 0.94$). These results reframe model collapse as a cultural transmission phenomenon and yield concrete principles for self-training pipeline design.
>
---
#### [new 043] Hierarchical Concept Geometry in Language Models Emerges from Word Co-occurrence
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究语言模型中概念层次结构的几何表示，解决如何从词共现统计中生成层级几何的问题。通过分析词向量嵌入的谱结构，揭示其与WordNet层级的对应关系。**

- **链接: [https://arxiv.org/pdf/2605.23821](https://arxiv.org/pdf/2605.23821)**

> **作者:** Andres Nava; Matthieu Wyart
>
> **备注:** 34 pages, 12 figures, including appendices
>
> **摘要:** We propose a distributional theory of how hypernymy -- the ``is-a'' relation between general and specific concepts -- is encoded geometrically in language representations. Starting from the empirically verified assumption that words closer on the WordNet hypernym graph co-occur more often, we characterize theoretically the spectrum of the resulting embedding Gram matrix of word2vec embeddings. Under mild positivity and decay conditions on the co-occurrence kernel, we prove that the leading eigenvectors first separate broad taxonomic branches and then progressively finer sub-branches, producing a \emph{hierarchical splitting geometry} with a coarse-to-fine spectral organization that mirrors the tree. We confirm these predictions in word2vec embeddings across many sampled WordNet subtrees, and show that the same signature extends strikingly well to Gemma 2B unembeddings. Our results indicate that hierarchical concept geometry in LLMs need not reflect a hierarchy-specific functional mechanism, but emerges from the spectral structure of pairwise word statistics.
>
---
#### [new 044] Sparse Autoencoders Map Brain-LLM Alignment onto Cortical Semantic Topography
- **分类: cs.CL; cs.AI; q-bio.NC**

- **简介: 该论文属于计算神经语言学任务，旨在解释LLM与大脑语言响应的对齐机制。通过稀疏自编码器提取语义特征，验证其与大脑皮层语义拓扑的对应关系。**

- **链接: [https://arxiv.org/pdf/2605.23035](https://arxiv.org/pdf/2605.23035)**

> **作者:** Dongxin Guo; Jikun Wu; Siu Ming Yiu
>
> **备注:** Accepted at CoNLL 2026. 20 pages (9 main + 1 limitations/acknowledgments + 3 references + 7 appendix), 5 figures, 20 tables
>
> **摘要:** Intermediate layers of large language models (LLMs) best predict human brain responses to language, one of the most robust findings in computational neurolinguistics, yet why remains mechanistically unexplained. We address this gap by bridging sparse autoencoders (SAEs) from mechanistic interpretability with neural encoding models, decomposing GPT-2 XL and Llama-3.1-8B into 16K-32K interpretable features per layer. A human-validated taxonomy ($\kappa \geq 0.74$) reveals that semantic features alone recover 94% of peak encoding performance ($r=0.285$), substantially exceeding variance-matched baselines ($p<0.001$, $d=1.31$). Beyond this aggregate dominance, we test a novel cortical topography prediction: five semantic subcategories derived a priori from three independent neuroscience programs should map onto distinct brain regions. A formal convergence test confirms this alignment (Spearman $\rho=0.72$, $p<0.001$; hypergeometric $p=0.007$), demonstrating that SAE-discovered features recapitulate known cortical semantic organization at a granularity inaccessible to prior methods. SAE features further predict human reading times beyond lexical controls ($\Delta\mathrm{logLik}=38.4$, $p<0.001$), and an exploratory prediction-error analysis provides preliminary evidence that the brain additionally encodes unexpected semantic content. Results generalize across English, Chinese, and French.
>
---
#### [new 045] A Reproducible Universal Dependencies-Style Pipeline for Katharevousa Greek Parliamentary Text
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决Katharevousa希腊语议会文本的句法分析问题。通过构建可复现的解析流程，提升历史文本的NLP处理能力。**

- **链接: [https://arxiv.org/pdf/2605.22978](https://arxiv.org/pdf/2605.22978)**

> **作者:** George Mikros; Fotios Fitsilis
>
> **备注:** 12 pages, 1 figure, 2 tables; companion to the kathnlp open-source release at this https URL
>
> **摘要:** Katharevousa Greek remains poorly served by contemporary NLP pipelines despite its importance for legal, administrative, and parliamentary archives. We present a reproducible workflow for building and evaluating a Universal Dependencies-style parsing resource for Katharevousa parliamentary questions from Greece's early post-junta period. The pipeline links OCR-aware reconstruction, schema-constrained LLM-assisted annotation, automatic validation, deterministic CoNLL-U snapshotting, fixed-split evaluation, and model-family comparison. The frozen automatically validated reference set contains 1{,}697 sentences, split into 1{,}357 training sentences and 340 held-out test sentences. We compare off-the-shelf Greek and Ancient Greek parsers, a feature-based parser, mBERT, XLM-R, and custom Stanza training under the same scoring protocol. Off-the-shelf systems show substantial register mismatch: the strongest external baseline, spaCy Greek, reaches 0.4183 LAS. The best structural parser, an XLM-R model, reaches 0.8893 UPOS accuracy, 0.7250 dependency-relation F1, 0.6098 UAS, and 0.5162 LAS, an absolute LAS gain of 0.0980 over the best external baseline. The feature-based model remains competitive for UPOS and relation labeling, indicating that transparent lexical-context features still matter at this data scale. Beyond scores, the paper contributes an auditable methodology for turning difficult historical parliamentary OCR into reusable syntactic NLP infrastructure. The entire pipeline -- code, schema, frozen reference annotations, fixed train/test split, and per-model benchmark reports -- is released as an open-access companion to this paper.
>
---
#### [new 046] The Efficiency Frontier: A Unified Framework for Cost-Performance Optimization in LLM Context Management
- **分类: cs.CL**

- **简介: 该论文属于LLM上下文管理任务，解决长上下文处理中的成本与性能优化问题。提出“效率前沿”框架，统一评估不同策略的部署效果，降低token使用成本。**

- **链接: [https://arxiv.org/pdf/2605.23071](https://arxiv.org/pdf/2605.23071)**

> **作者:** Binqi Shen; Lier Jin; Hanyu Cai; Lan Hu; Yuting Xin
>
> **摘要:** Large language models (LLMs) increasingly rely on long-context processing, but expanding context windows introduces substantial computational and financial costs. Existing context reduction approaches, including retrieval and memory compression methods, are typically evaluated using performance and efficiency metrics independently, limiting systematic comparison and deployment-aware decision-making. This paper introduces The Efficiency Frontier, a unified framework for cost-performance optimization in LLM context management. The framework models context strategy selection as a deployment-aware optimization problem that jointly accounts for task performance, token cost, and preprocessing reuse through amortized cost modeling. Unlike existing evaluations that compare methods in isolation, the proposed framework enables decision-oriented analysis of when different context management strategies become preferable under varying operational conditions. Evaluated on 5,000 HotpotQA instances, the framework reveals distinct operational regimes and transition boundaries between retrieval-based and preprocessing-based strategies. Results show that deployment-aware optimization reduces effective token usage by approximately 25% at comparable performance ($F1 \approx 0.78$), while amortized memory compression achieves over 50% lower token cost relative to full-context prompting in higher-performance settings. Overall, the proposed framework provides a principled and practical foundation for evaluating and deploying scalable, efficient, and sustainable LLM systems.
>
---
#### [new 047] Do Language Models Know What Not to Say? Causal Evidence for Statistical Preemption in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言模型是否通过统计竞争获得否定知识，属于自然语言处理中的语法规则学习任务。通过实验验证统计预置机制，揭示模型如何通过频率差异判断表达是否合适。**

- **链接: [https://arxiv.org/pdf/2605.23039](https://arxiv.org/pdf/2605.23039)**

> **作者:** Dongxin Guo; Jikun Wu; Siu Ming Yiu
>
> **备注:** Accepted at CoNLL 2026. 21 pages (9 main body + appendices and references); 4 figures, 14 tables
>
> **摘要:** How do learners acquire knowledge of what is unacceptable without negative evidence? Construction Grammar proposes statistical preemption: exposure to a conventional form (e.g., "donated the books to the library") preempts structurally possible but unattested alternatives ("*donated the library the books"). We present a computational study that, for the first time, directly dissociates statistical preemption from the competing entrenchment hypothesis in large language models within a single converging design. Across four experiments spanning 120 English verb-construction pairings (dative, causative, locative), we show that (1) LLM surprisal patterns correlate strongly with human acceptability judgments ($r = 0.79$), validated against three independent behavioral datasets; (2) these patterns are driven by competing-form frequency rather than overall verb frequency, confirmed by non-circular partial correlations; (3) preemption sensitivity scales as a power law with model size; and (4) a controlled fine-tuning intervention causally demonstrates that manipulating competing-form frequencies shifts preemption behavior in the predicted direction, with reverse-direction controls ruling out frequency-sensitivity confounds. These results provide converging evidence that neural language models acquire negative linguistic knowledge through distributional competition, the core mechanism posited by Construction Grammar.
>
---
#### [new 048] Learnability-Informed Fine-Tuning of Diffusion Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在提升扩散语言模型的推理能力。针对SFT在DLMs中效果不佳的问题，提出LIFT算法，有效利用不同扩散阶段的信息进行微调。**

- **链接: [https://arxiv.org/pdf/2605.22939](https://arxiv.org/pdf/2605.22939)**

> **作者:** Shubham Parashar; Atharv Chagi; Jacob Helwig; Lakshmi Jotsna; Sushil Vemuri; James Caverlee; Dileep Kalathil; Shuiwang Ji
>
> **摘要:** We aim to improve the reasoning capabilities of diffusion language models (DLMs). While SFT is a popular post-training recipe for autoregressive models, its use in DLMs faces challenges and can even hurt performance, though the underlying causes remain understudied. Our analysis reveals that vanilla SFT overlooks learnability, namely what and when tokens are learned. Specifically, rare tokens are difficult to learn when most of the input is masked, whereas it is straightforward and thus of little value to learn common tokens when most of the input is unmasked. Motivated by our analysis, we propose LIFT, an efficient SFT-based post-training algorithm for DLMs. LIFT learns easy tokens when most of the input is masked and hard tokens when more context is available, thus aligning the training with the information available at different diffusion time steps. Our results show that LIFT outperforms existing SFT baselines across six reasoning benchmarks, achieving up to a 3x relative gain on AIME'24 and AIME'25. Our code is publicly available at this https URL.
>
---
#### [new 049] When Is Next-Token Prediction Useful? Marginalization, Ergodicity, Mixture Identifiability, Local Sufficiency, RAG, Tools, and Programming
- **分类: cs.CL; stat.ML**

- **简介: 论文探讨语言模型训练中下一词预测的适用性，区分了条件分布、边缘文本过程和模型分布，分析其在不同情况下的有效性，属于自然语言处理任务，解决模型训练与实际生成间差异的问题。**

- **链接: [https://arxiv.org/pdf/2605.23278](https://arxiv.org/pdf/2605.23278)**

> **作者:** Francesco Corielli
>
> **摘要:** Language models trained on observed sequences are often described as learning the conditional distribution of the next token given previous tokens. This description is only conditionally correct. A model trained on realized token trajectories does not observe full conditional laws; it receives sampled continuations. Moreover, real language generation is conditioned not only on previous words but also on non-textual circumstances: facts, events, intentions, goals, beliefs, social context, and task-specific constraints. This paper distinguishes three objects that are often conflated: the full conditional language process conditioned on latent circumstances, the marginal text-only process obtained by integrating those circumstances out, and the model-induced distribution learned from finite observed corpora. The paper argues that interpreting model training as estimating the marginal text-only law requires strong assumptions of stationarity, representativeness, and ergodicity, assumptions that are standard in statistical estimation but problematic when applied to heterogeneous language corpora. Even if these assumptions hold, the marginal text-only law is useful only when the observed prefix is an approximately sufficient statistic for the latent circumstances relevant to continuation. In information-theoretic terms, usefulness requires that the residual conditional mutual information between the next token and the omitted circumstances, given the observed text, be small. The paper then extends this argument to heterogeneous training corpora. Finally, the paper interprets Retrieval Augmented Generation (RAG) and tool use as conditional sufficiency devices.
>
---
#### [new 050] Memorization Dynamics of Fill-in-the-Middle Pretraining
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究FIM预训练模型的复述记忆动态，对比其与LTR模型在记忆行为上的差异，旨在揭示FIM对精确文本回忆的影响。**

- **链接: [https://arxiv.org/pdf/2605.22981](https://arxiv.org/pdf/2605.22981)**

> **作者:** Tobias von Arx; Tanguy Dieudonné
>
> **备注:** MemFM @ ICML 2026
>
> **摘要:** Fill-in-the-middle (FIM) is a pretraining objective widely used to equip causal language models with infilling ability, yet its effect on verbatim memorization remains underexplored. We study the memorization dynamics of FIM in a controlled setting by pretraining matched Llama 3.2 models with FIM and standard left-to-right (LTR) objectives on a FineWeb-Gutenberg corpus containing repeated Gutenberg excerpts. With prefix-based probes, FIM more often recovers short or partially matching spans, while LTR more often assigns high confidence to long exact continuations. We observe that verbatim extraction under FIM-training grows approximately linearly with repetitions over the tested range. Evaluating native FIM-format probes reveals that suffix context is not sufficient: verbatim recall under FIM-training remains strongly anchored in prefix context. Our results also show that evaluating only one span length or probing format can miss important nuances in memorization behavior.
>
---
#### [new 051] Asking For An Old Friend: Diagnosing and Mitigating Temporal Failure Modes in LLM-based Statutory Question Answering
- **分类: cs.CL**

- **简介: 该论文属于法律问答任务，解决大语言模型在处理动态法律条文时的时效性问题，提出基准数据并验证不同方法的效果。**

- **链接: [https://arxiv.org/pdf/2605.23497](https://arxiv.org/pdf/2605.23497)**

> **作者:** Max Prior; Andreas Schultz; Matthias Grabmair
>
> **摘要:** Large language models are increasingly used for legal research, yet their fixed training cutoffs and reliance on static parametric knowledge are at odds with the evolving nature of statutory law. We study two temporal failure modes: post-cutoff staleness, where models apply superseded rules after legislative amendments, and recency bias, where models prefer newer provisions even when a historical version governs the fact pattern. To this end, we present a benchmark of 312 expert-validated, time-sensitive German statutory QA pairs spanning three categories: Post-Cutoff Amendment Questions, Pre-Amendment Questions, and Multi-Provision Pre-Amendment Questions. We evaluate five LLMs by OpenAI, Anthropic and DeepSeek under four inference settings: Vanilla, Web-search, and two retrieval-augmented variants that enforce temporal validity via a fact date extraction and version filtering. Using an LLM-as-a-judge validated against human expert ratings, we find severe degradation in the Vanilla post-cutoff setting. Both RAG approaches substantially improve performance across all question types, while web search yields unstable gains and exhibits a marked recency bias on historically anchored tasks. Our results indicate that reliable legal QA requires treating temporal validity as a hard constraint.
>
---
#### [new 052] OnePred: Next-Query Prediction via Recursive Intent Memory in Multi-Turn Conversations
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出OnePred模型，解决多轮对话中下一查询预测任务。通过递归意图记忆减少token消耗，提升预测质量。**

- **链接: [https://arxiv.org/pdf/2605.23668](https://arxiv.org/pdf/2605.23668)**

> **作者:** Jiangwang Chen; Bowen Zhang; Zixin Song; Jiazheng Kang; Xiao Yang; Da Zhu; Guanjun Jiang
>
> **摘要:** Although large language model (LLM) conversational systems process millions of multi-turn dialogues daily, they remain fundamentally reactive: they respond only after the user types a query. A key step toward proactive interaction is next-query prediction, which anticipates the user's subsequent query based solely on the preceding dialogue. Progress on this task is hindered by the lack of dedicated benchmarks and a fundamental efficiency--quality trade-off: naively concatenating full dialogue history incurs linearly growing token consumption, while truncating to the latest turn discards crucial cross-turn context. Our key insight is that accurate prediction does not require re-reading raw history; it suffices to track the user's evolving intent trajectory across topics, unresolved needs, and interest shifts. We propose OnePred, which maintains a recursively updated memory as its sole cross-turn context, bounding the per-turn cost independently of conversation length. We train the model via a two-stage reinforcement learning pipeline that first teaches what to predict, then what to compress, shaping the memory into a prediction-oriented intent chain. To establish a rigorous testbed, we introduce NQP-Bench, spanning three diverse subsets. Experiments demonstrate that OnePred reduces per-turn token consumption by up to 22$\times$ compared to full-history inputs while consistently exceeding all baselines in prediction quality, with larger gains on longer conversations. Our code is publicly available at this https URL.
>
---
#### [new 053] How Human-Like Are Large Language Models? A Register-Aware Linguistic Evaluation Framework
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成评估任务，旨在解决LLM生成文本与人类语言相似性的问题。通过构建上下文感知框架，使用MMD和67个语法特征比较LLM与人类文本的分布差异。**

- **链接: [https://arxiv.org/pdf/2605.23651](https://arxiv.org/pdf/2605.23651)**

> **作者:** Björn Nieth; Marianna Gracheva; Michaela Mahlberg; Bjoern Eskofier; Emmanuelle Salin
>
> **备注:** 8.5 pages (main) + 31 pages appendix, 29 figures, 10 tables. Code and data: this https URL
>
> **摘要:** While factual correctness and task-performance have been in focus of Large Language Model (LLM) research for a long time, the fundamental question of how human-like generated texts are on a linguistic level has been underexplored. From a corpus-linguistic perspective, language production is inherently context-dependent, with distinct communicative contexts giving rise to differences in frequencies and co-occurrence patterns of linguistic features. A text failing to adhere to these patterns can be content-wise correct, but still be unfavorable to human readers. In this work, we propose a context-aware evaluation framework in which human-likeness is assessed using a two-sample problem between the linguistic feature distribution of a human reference corpus for a given register and a corresponding LLM-generated corpus. We implement this framework using the Maximum Mean Discrepancy (MMD) and the 67 lexico-grammatical features introduced by Biber, which are commonly applied in corpus linguistics. In our experiments, we compare seven instruction-tuned, open-source models across five English-language datasets spanning distinct registers against a human baseline. While across all tested setups, LLMs deviate from the human baseline, which models are closest to human language depends on the register and is not dictated by model size.
>
---
#### [new 054] A Proactive Multi-Agent Dialogue Framework for Assessing Social Language Disorder Traits in Autism
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于 autism 诊断任务，旨在提升社会语言障碍特质的评估效率。提出TPA框架，通过主动提问策略提高诊断信息获取效果。**

- **链接: [https://arxiv.org/pdf/2605.22993](https://arxiv.org/pdf/2605.22993)**

> **作者:** Chuanbo Hu; Minglei Yin; Bin Liu; Wenqi Li; Lynn K. Paul; Shuo Wang; Xin Li
>
> **摘要:** Characteristic linguistic behaviors associated with Social Language Disorder (SLD) in autism spectrum disorder, including echoic repetition, pronoun displacement, and stereotyped media quoting, are largely absent from spontaneous conversation and only emerge under specific conversational conditions. In structured clinical assessments, this latency means that questioning strategy selection is a critical yet underappreciated determinant of how much diagnostic information a conversation yields. Whether large language models (LLMs) can be guided to proactively select questioning strategies that systematically surface these latent traits remains largely unexplored. Here we present TPA (Think, Plan, Ask), a proactive multi-agent dialogue framework applied to the language assessment component of the Autism Diagnostic Observation Schedule Module 4 (ADOS-2), in which a doctor agent explicitly reasons about which traits remain unobserved before selecting a clinically grounded strategy and generating a targeted question. A patient agent grounded in real ADOS-2 clinical data enables reproducible evaluation without real patient participation, validated across three independent experiments confirming adequate fidelity to real patient language. Evaluated on 484 episodes from 35 patients, TPA outperforms six competitive dialogue planning baselines across all primary metrics, achieving 82.1% SLD trait coverage, 16.6% higher than automated replay of real clinical dialogues conducted by trained clinicians (65.5%), with substantially greater per-turn diagnostic efficiency (AUCC: 0.628 vs. 0.458, absolute gain +0.170). These results demonstrate that proactive questioning strategy selection substantially improves the efficiency of automated SLD trait assessment, with direct implications for scalable AI-assisted clinical screening.
>
---
#### [new 055] A Comparative Evaluation of Structural Topic Models and BERTopic for Short, Open-Ended Survey Responses
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于文本挖掘任务，旨在比较STM与BERTopic在处理短篇开放式调查回复的效果。通过调整多种参数，评估两种模型的连贯性与可解释性，提出选择与结合模型的建议。**

- **链接: [https://arxiv.org/pdf/2605.23093](https://arxiv.org/pdf/2605.23093)**

> **作者:** Yan Jiang; Sihong Liu; Philip A. Fisher
>
> **摘要:** Topic modeling in applied psychology increasingly spans two methodological traditions: probabilistic bag-of-words models and newer embedding-based approaches. Yet many evaluations of these methods rely on longer and cleaner benchmark corpora, leaving less guidance for short, open-ended survey responses. This paper compares Structural Topic Models (STM), a probabilistic topic model, and BERTopic, an embedding-based model, for analyzing open-ended survey responses. We evaluated three STM conditions and five BERTopic conditions, varying typographical correction, stemming, embedding choice, and contextual augmentation, a strategy we introduced to provide additional semantic context for very short responses. Results indicate that BERTopic consistently produced higher topic coherence than STM, with contextual augmentation yielding the strongest performance gains. In contrast, higher-dimensional embeddings alone did not improve coherence and were associated with greater data loss. Qualitative evaluation showed that BERTopic generated more interpretable and stable topics, while STM topics were often broader and more mixed. However, STM provides stronger support for inferential covariate analysis, whereas BERTopic covariate comparisons are primarily descriptive. These findings suggest that STM and BERTopic offer complementary strengths. We conclude with practical guidance for selecting and combining topic modeling approaches in applied social science research.
>
---
#### [new 056] EquiSumm : A Gender Bias-Aware Framework for Inclusive Tweet Summarization
- **分类: cs.CL**

- **简介: 该论文属于文本摘要任务，旨在解决社交媒体摘要中的性别偏见问题。提出EquiSumm框架，考虑性别因素生成更公平的摘要。**

- **链接: [https://arxiv.org/pdf/2605.23412](https://arxiv.org/pdf/2605.23412)**

> **作者:** Chaitanya Wanjari; Jessica Kamal; Riddhi Jain; Samruddhi Kurhe; Roshni Chakraborty
>
> **备注:** Accepted at AI for Social Good Workshop, Pattern Recognition and Machine Intelligence (PReMI 2025), IIT Delhi. 6 pages, 2 figures
>
> **摘要:** While social media platforms, such as Twitter, provide a medium for large-scale opinion sharing during news events, it is manually impossible for individuals or media agencies to process the vast volume of content to identify key viewpoints. In order to resolve this, several automatic summarization techniques have been proposed to condense large collections of tweets into concise and informative summaries. However, these algorithms do not explicitly consider demographic fairness. Several existing research works have developed automated summarization approaches that can provide a holistic overview of the key aspects and major opinions shared on social media platforms related to a news event. However, these approaches do not explicitly consider different forms of demographic representation, such as gender, which can lead to biased summary representation. In this paper, we propose EquiSumm, which considers the gender aspect of the shared opinion to generate a summary, and our experimental analysis on two major datasets indicates the performance effectiveness with respect to existing research works.
>
---
#### [new 057] Multilingual Knowledge Transfer under Data Constraints via Lexical Interventions
- **分类: cs.CL**

- **简介: 该论文属于跨语言知识迁移任务，解决低资源语言数据不足导致的知识获取难题。通过在预训练阶段进行词汇替换，提升知识转移效果，无需额外训练。**

- **链接: [https://arxiv.org/pdf/2605.23885](https://arxiv.org/pdf/2605.23885)**

> **作者:** Anastasiia Sedova; Natalie Schluter; Skyler Seto; Maartje ter Hoeve
>
> **摘要:** Cross-lingual knowledge transfer is critical for building high-performing multilingual language models for languages with insufficient training data. When target language data is scarce, the knowledge required for many downstream tasks involving scientific reasoning, commonsense inference, and world knowledge must be acquired primarily from the high-resource language, making effective knowledge transfer essential. Existing methods for improving such cross-lingual knowledge transfer require large amounts of parallel data, translation systems, auxiliary models, or additional training stages that are largely unavailable for many languages. We propose LINK - a data-level intervention method that improves knowledge transfer during model pretraining through lexical substitutions in high-resource part of pretraining data using bilingual vocabularies. For a given replacement ratio, randomly selected words in a portion of the high-resource (English) training corpus are swapped with their word-level translations, requiring no additional model training and only a bilingual vocabulary, which can be obtained at near-zero cost for virtually any language. Evaluation on eight languages across five model sizes shows notable improvements on downstream tasks in the target language, with up to a 2x speedup in training to reach equivalent performance.
>
---
#### [new 058] DreamerNLplus: Interpretable Modeling of Mental Health Dynamics from Social Media Timelines using Hybrid Rule-Based and RAG Methods
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于心理健康动态建模任务，旨在从社交媒体时间线中分析用户心理状态变化。工作包括结合规则与RAG方法进行状态预测、事件检测和序列摘要，提升心理健康分析的准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2605.23052](https://arxiv.org/pdf/2605.23052)**

> **作者:** Maryia Zhyrko; Daisy Monika Lal; Erik van Mulligen; Lifeng Han
>
> **备注:** Accepted by CLPsych2026. CLPsych 2026 will be held at ACL in San Diego July 4th, 2026
>
> **摘要:** We present DreamerNLplus, a hybrid framework for modeling mental health dynamics from social media timelines in the CLPsych 2026 shared task. Our system addresses three tasks: psychological state modeling, temporal change detection, and sequence-level summarization. For Task 1, we combine LLM-based data augmentation, DeBERTa classification, and Random Forest regression for structured state prediction. For Task 2, we use few-shot prompting with a locally deployed Llama 3.1 model to detect Switch and Escalation events using short-term temporal context. For Task 3.1, we explore both a deterministic rule-based summarization pipeline and a few-shot LLM-based approach, ranking \textbf{2nd} officially. Our RAG-based method achieves strong performance in Task 3.2, ranking \textbf{1st} for Improvement and \textbf{3rd} for Deterioration, demonstrating its ability to capture recurrent psychological change patterns across timelines. Our analysis reveals key challenges, including the mismatch between classification and regression performance, the difficulty of modeling temporal transitions, and the disagreement between semantic and similarity-based evaluation metrics. These findings highlight the complexity of modeling mental health dynamics and motivate future work on unified evaluation frameworks. We share our code and prompts at this https URL
>
---
#### [new 059] When Do LLMs Reason? A Dynamical Systems View via Entropy Phase Transitions
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究LLM推理时机，解决何时启用链式思维有效的问题。通过分析熵动态，提出EDRM框架，实现高效推理选择。**

- **链接: [https://arxiv.org/pdf/2605.22873](https://arxiv.org/pdf/2605.22873)**

> **作者:** Wei Xia; Haoqing Wang; Zhi-Hong Deng; Yehui Tang
>
> **摘要:** Chain-of-thought (CoT) reasoning has become the default strategy for enhancing LLM capabilities, yet its application raises a fundamental question: when is explicit reasoning actually beneficial? Empirical evidence reveals a striking paradox: CoT often provides marginal or even negative gains on factual and open-ended tasks while multiplying token consumption. In this work, we show that LLM reasoning is not a static property of tasks or models, but a \emph{dynamic decoding state} that emerges during generation. Through systematic analysis, we find early-stage entropy dynamics provide a reliable signal of this state: tasks benefiting from CoT exhibit consistent entropy reduction, while others display unstable or increasing patterns. This behavior can be interpreted as a phase-transition-like shift from a high-entropy exploratory regime to a low-entropy structured reasoning regime. Based on these insights, we propose \textbf{EDRM} (Entropy Dynamics-based Reasoning Manifold), a lightweight and training-free routing framework that leverages early decoding entropy to adaptively select inference strategies. EDRM embeds entropy trajectories into a compact and interpretable manifold representation, enabling both zero-shot deployment and fine-grained instance-level adaptation. Across 15 benchmarks and 4 LLMs of varying scales and architectures, EDRM consistently outperforms static baselines. At the dataset level, EDRM achieves \textbf{41--55\%} token reduction while improving accuracy with as few as 50 calibration samples. At the instance level, it further improves accuracy by up to \textbf{4.7\%} while maintaining \textbf{27--45\%} token savings. These results suggest that reasoning should be invoked selectively rather than by default, and demonstrate the effectiveness of entropy-driven decoding control for efficient and adaptive LLM inference.
>
---
#### [new 060] RADAR: Relative Angular Divergence Across Representations
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文提出RADAR，用于评估基础模型的跨域迁移能力，解决负迁移问题。通过分析表示空间的几何特性，提升跨域任务性能。**

- **链接: [https://arxiv.org/pdf/2605.23028](https://arxiv.org/pdf/2605.23028)**

> **作者:** Xavier Cadet; Mateusz Nowak; Peter Chin
>
> **备注:** 27 pages; 8 figures; 10 tables
>
> **摘要:** Machine learning methods rely on data. However, gathering suitable data can be challenging due to availability constraints, cost, or the need for domain expertise. Expanding datasets with additional sources is a common response to limited data, yet this practice does not always improve downstream performance and can sometimes lead to a loss of performance, known as negative transfer. We propose RADAR, a simple, geometrically grounded metric for estimating cross-domain transferability in foundation models. RADAR analyzes the layer-wise evolution of representations by measuring angular alignments and relative changes in distance along layer-to-layer displacement trajectories, and by comparing empirical distributions of within-domain and cross-domain dynamics. We hypothesize that domain transferability is related to the divergence between these trajectory distributions. We evaluate the metric across multiple modalities, including cross-lingual sentiment classification with text embedding models and cross-domain image classification with foundation vision models. Across several settings, RADAR provides competitive predictive performance relative to existing transferability metrics on several vision and text benchmarks, with particularly strong results when domain transitions are smooth or cleanly separated. Our ablations further suggest that the effectiveness of transferability estimation depends on the geometry of the model's internal representation space, with different modalities favoring different topological formulations.
>
---
#### [new 061] Decomposing and Measuring Evaluation Awareness
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在解决模型在评估中调整行为的问题。通过分解评估意识，提出EvalAwareBench进行测试与分析。**

- **链接: [https://arxiv.org/pdf/2605.23055](https://arxiv.org/pdf/2605.23055)**

> **作者:** Changling Li; Terry Jingchen Zhang; Jie Zhang; Zhijing Jin; Sahar Abdelnabi; Maksym Andriushchenko
>
> **摘要:** Frontier language models sometimes recognize that they are being evaluated and adjust their behavior, undermining validity of benchmark results. Yet the field studies it without a shared foundation, conflating properties of the evaluation with properties of the model, and detection with behavioral response. We ground evaluation awareness in social psychology, decomposing it into an environment component (how recognizable the task is) and a model component that separates recognition from propensity to act on it. We operationalize the environment component through eight categorized trigger factors, such as placeholder entities and grading-style output formats, and study recognition and behavior through chain-of-thought monitoring. Across nine frontier models and four benchmarks, recognition rates depend on the specific pairing of model and benchmark rather than on either in isolation. Recognition rarely leads to behavioral change, and when it does, the direction depends on the type of evaluation perceived. Models are also more sensitive to safety than capability evaluations, placing safety benchmark validity at greater risk. To study which factors each model is sensitive to and how they interact, we propose \textbf{EvalAwareBench}, a factor-controlled benchmark of 100 paired safety-capability tasks where each of the eight factors can be independently toggled, varying evaluative signals while holding the underlying request fixed. Through EvalAwareBench, we find that no single factor uniformly affects all models, but stacking factors progressively raises evaluation awareness across all of them. Our framework and EvalAwareBench provide the tools to measure, attribute, and mitigate evaluation awareness, pointing to behavioral consistency under recognition as a promising path forward.
>
---
#### [new 062] Strategic Coercion Within Alliances: The Greenland Sovereignty Game as an AI Stress Test
- **分类: physics.soc-ph; cs.AI; cs.CL; cs.GT; cs.MA; econ.GN**

- **简介: 该论文属于AI与地缘政治交叉研究，通过模拟测试LLM在联盟内部战略胁迫中的行为，分析其决策模式及影响因素。**

- **链接: [https://arxiv.org/pdf/2605.22841](https://arxiv.org/pdf/2605.22841)**

> **作者:** Rommin Adl; Peyton Williams
>
> **备注:** 78 pages, 17 figures, 18 tables. Multi-agent LLM simulation recovering structural utility parameters across 8 frontier models in the Greenland sovereignty crisis. v3: typo pass, fixes phantom action names (REQUEST_MULTILATERAL, INDEPENDENT) and a Blunden date mismatch. v2 added Section V safety findings (legitimacy-laundered escalation, signal decoupling) and Appendix H
>
> **摘要:** What happens when the strongest alliance member pressures a weaker member over territory and strategic control? We examine the Greenland sovereignty crisis as a stress test for LLM geopolitics, centered on the 2019-2026 U.S. push to acquire Greenland from the Kingdom of Denmark. The crisis nests two collective-action problems: Arctic strategic control and whether NATO can enforce alliance norms against the dominant member. We develop three games (asymmetric coercion; a NATO assurance game with a critical-mass tipping point; a triadic extensive-form game with social preferences) and test them with a multi-agent simulation in which eight frontier LLMs play six geopolitical roles (United States, Denmark, Greenland, NATO, Russia, Canada) across 3,604 completed games and 108,120 action observations. Using inverse game theory, we recover each model's structural utility parameters (alpha, beta, gamma, delta, eta) for material self-interest, reciprocity, inequality aversion, norm respect, and commitment consistency. Three findings stand out. First, all eight models become more escalatory under coercion framing (four-action escalation rises from 10.7% to 28.6%). Second, Chinese-origin models show systematically different power-weight profiles from Western-origin models when playing the U.S. role. Third, peaceful US acquisition emerges in only 1.9% of clean games and only 3 of 8 frontier models ever achieve it, most prominently DeepSeek V3.2, which executes a stable five-round playbook through the metropole. Prompts emphasizing jus cogens and self-determination reduce escalation back near baseline in the English-only confirmatory sample; multilingual contrasts are reported as exploratory sensitivity checks. We position this as a structural benchmark for LLM geopolitical behavior, complementing action-frequency benchmarks.
>
---
#### [new 063] ImProver 2: Iteratively Self-Improving LMs for Neurosymbolic Proof Optimization
- **分类: cs.AI; cs.CL; cs.LG; cs.LO**

- **简介: 该论文提出ImProver 2，解决形式化数学证明优化问题。通过神经符号框架，提升小模型在复杂指标上的证明重构能力，实现高效、可扩展的证明优化。**

- **链接: [https://arxiv.org/pdf/2605.22885](https://arxiv.org/pdf/2605.22885)**

> **作者:** Riyaz Ahuja; Tate Rowney; Jeremy Avigad; Sean Welleck
>
> **摘要:** Formal mathematics libraries are rapidly expanding, creating a growing need to refactor verified proofs for maintainability and to improve training data quality for neural provers. However, scalable proof optimization is hindered by heterogeneous and heuristically specified objectives, scarce data, and high training and inference costs. To overcome these challenges, we introduce ImProver 2, a neurosymbolic framework for automated proof optimization in Lean 4. ImProver 2 combines a data-efficient expert-iteration pipeline with a scaffold that exposes formal structure alongside lightweight informal abstractions. We further introduce a suite of metrics capturing structural proof properties. Using ImProver 2, we train a 7B-parameter model that outperforms orders-of-magnitude larger models within the same model family, and is competitive with mid-tier frontier models across metrics. We additionally demonstrate that our neurosymbolic scaffold significantly improves performance across both small and frontier models. We show that with proper scaffolding and training, small models can effectively restructure research-level proofs over complex and varied metrics, matching substantially larger systems and establishing proof optimization as a scalable, learnable task.
>
---
#### [new 064] SkillOpt: Executive Strategy for Self-Evolving Agent Skills
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出SkillOpt，用于系统化优化智能体技能，解决传统技能生成方法不可靠的问题。通过文本编辑提升技能效果，无需额外推理调用。**

- **链接: [https://arxiv.org/pdf/2605.23904](https://arxiv.org/pdf/2605.23904)**

> **作者:** Yifan Yang; Ziyang Gong; Weiquan Huang; Qihao Yang; Ziwei Zhou; Zisu Huang; Yan Li; Xuemei Gao; Qi Dai; Bei Liu; Kai Qiu; Yuqing Yang; Dongdong Chen; Xue Yang; Chong Luo
>
> **备注:** 27 pages, 4 figures, 6 tables
>
> **摘要:** Agent skills today are hand-crafted, generated one-shot, or evolved through loosely controlled self-revision, none of which behaves like a deep-learning optimizer for the skill, and none of which reliably improves over its starting point under feedback. We argue the skill should instead be trained as the external state of a frozen agent, with the same discipline that makes weight-space optimization reproducible. SkillOpt is, to our knowledge, the first systematic controllable text-space optimizer for agent skills: a separate optimizer model turns scored rollouts into bounded add/delete/replace edits on a single skill document, and an edit is accepted only when it strictly improves a held-out validation score. A textual learning-rate budget, rejected-edit buffer, and epoch-wise slow/meta update make skill training stable while adding zero inference-time model calls at deployment. Across six benchmarks, seven target models, and three execution harnesses (direct chat, Codex, Claude Code), SkillOpt is best or tied on all 52 evaluated (model, benchmark, harness) cells and beats every per-cell competitor among human, one-shot LLM, Trace2Skill, TextGrad, GEPA, and EvoSkill skills. On GPT-5.5 it lifts the average no-skill accuracy by +23.5 points in direct chat, by +24.8 inside the Codex agentic loop, and by +19.1 inside Claude Code. Transfer experiments further show that optimized skill artifacts retain value when moved across model scales, between Codex and Claude Code execution environments, and to a nearby math benchmark without further optimization.
>
---
#### [new 065] PrefBench: Evaluating Zero-Shot LLM Agents in Hidden-Preference Personalized Pricing Negotiations
- **分类: cs.GT; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出PrefBench，用于评估LLM在隐藏偏好个性化定价谈判中的表现。任务是解决隐藏买家偏好下的定价策略问题，通过模拟实验评估LLM的谈判效果与利润表现。**

- **链接: [https://arxiv.org/pdf/2605.22855](https://arxiv.org/pdf/2605.22855)**

> **作者:** Yingjie Lei
>
> **备注:** 24 pages, 3 figures, 5 tables. Code is available at this https URL
>
> **摘要:** Personalized pricing negotiations are a challenging testbed for LLM agents because successful interaction does not guarantee profitable decision making. A seller may produce valid actions and close many deals while still pricing poorly when buyer willingness to pay and bargaining traits remain hidden. This paper presents PrefBench, a simulator-based benchmark for hidden-preference personalized pricing negotiations. Each episode pairs a simulated buyer with a fixed vehicle-customization bundle; the seller observes public persona descriptors, bundle information, and negotiation history, while latent buyer variables govern valuation, patience, counter-offer behavior, and walkaway decisions. PrefBench evaluates this setting through an LLM-facing state-summary protocol that constrains agents to return strict JSON actions under a fixed hidden-information boundary. We evaluate zero-shot LLM sellers against heuristic references over 7,500 episodes. The tested LLMs follow the protocol reliably and achieve deal rates above 0.99, but their seller-profit outcomes remain weak: the best LLM average profit is only slightly above the random baseline and far below a simple concession heuristic under the same episode stream. These results show that structured action compliance and agreement-seeking behavior can coexist with weak profit-sensitive bargaining. PrefBench provides a controlled benchmark for evaluating pricing-agent behavior under hidden buyer preferences.
>
---
#### [new 066] SciAtlas: A Large-Scale Knowledge Graph for Automated Scientific Research
- **分类: cs.AI; cs.CL; cs.IR; cs.LG**

- **简介: 该论文提出SciAtlas，一个大规模跨学科知识图谱，用于解决学术信息碎片化与逻辑连接难的问题，通过结构化拓扑关系支持高效科研自动化。**

- **链接: [https://arxiv.org/pdf/2605.22878](https://arxiv.org/pdf/2605.22878)**

> **作者:** Shuofei Qiao; Yunxiang Wei; Jiazheng Fan; Bin Wu; Busheng Zhang; Mengru Wang; Yuqi Zhu; Ningyu Zhang; Keyan Ding; Qiang Zhang; Huajun Chen
>
> **备注:** Ongoing Work
>
> **摘要:** The exponential growth of global academic output has confronted researchers and AI agents with an unprecedented ``information explosion,'' where fragmented and unstructured knowledge organization impedes deep interdisciplinary integration. Current academic retrieval tools predominantly rely on superficial keyword matching or vector-space semantic retrieval, which lack the topological reasoning capabilities required to navigate complex logical connections. Agentic deep-research-based frameworks are often prone to logical hallucinations and consuming high inference costs. To bridge this gap, in this report, we introduce SciAtlas, a large-scale, multi-disciplinary, heterogeneous academic resource knowledge graph designed as a panoramic scientific evolution network. By integrating over 43M papers from 26 disciplines, and a total of 157M entities and 3B triplets, SciAtlas provides a structured topological cognitive substrate that dismantles disciplinary barriers and furnishes AI agents with a global perspective. Furthermore, we develop a neuro-symbolic retrieval algorithm featuring tri-path collaborative recall and graph reranking, achieving a seamless transition from simple semantic matching to deterministic association discovery. We also present key application directions of SciAtlas, including literature review, automated research trend synthesis, idea positioning, and academic trajectory exploration, to demonstrate that SciAtlas can serve as an effective ``cognitive map'' to empower the full loop of automated scientific research while significantly reducing reasoning costs. We have released the interfaces for KG retrieval and various downstream tasks in our GitHub repo.
>
---
#### [new 067] What Does the Server See? Understanding Privacy Leakage from Large Language Models in Split Inference
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文研究split inference中的隐私泄露问题，提出ActInv方法重建客户端输入，分析层间脆弱性并设计防御方案PriPert。**

- **链接: [https://arxiv.org/pdf/2605.23158](https://arxiv.org/pdf/2605.23158)**

> **作者:** Mingyuan Fan; Yu Liu; Fuyi Wang; Cen Chen
>
> **备注:** Accepted to ACM CCS'26
>
> **摘要:** The deployment of large language models (LLMs) on resource-constrained devices remains challenging, spurring interest in split inference, where models are partitioned between client and server to reduce computational burden and enhance privacy by transmitting only intermediate activations. However, the privacy-preserving capabilities of split inference, particularly in the context of LLMs, have not been exhaustively investigated. To fill this gap, we introduce ActInv, which solves an intermediate activation matching problem to reconstruct the client's input. Extensive evaluations demonstrate that ActInv achieves high-fidelity reconstructions, even in the presence of common perturbation-based defenses such as Gaussian noise injection and activation sparsification. To systematically understand this vulnerability, we develop Perturbation Amplification Factor (PAF), a metric for quantifying a layer's inherent resistance to reconstruction. Our analysis reveals that privacy vulnerability is not uniform across layers, with some layers being highly susceptible to leakage while others offer natural resistance. Furthermore, we demonstrate that defense effectiveness can be significantly improved by calibrating perturbation directions to maximize reconstruction error during backpropagation. Building on these insights, we design PriPert and conduct comprehensive evaluations, covering privacy, utility, and computational overhead, to demonstrate its effectiveness.
>
---
#### [new 068] GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出GEMQ，解决MoE-LLMs内存过高的问题。通过全局混合精度量化和路由微调，实现高效推理。**

- **链接: [https://arxiv.org/pdf/2605.23078](https://arxiv.org/pdf/2605.23078)**

> **作者:** Jianing Deng; Song Wang; Dongwei Wang; Zijie Liu; Tianlong Chen; Huanrui Yang; Jingtong Hu
>
> **备注:** ICML 2026
>
> **摘要:** Mixture-of-Experts Large Language Models (MoE-LLMs) achieve strong performance but incur substantial memory overhead due to massive expert parameters. Mixed-precision quantization mitigates this cost by allocating expert-wise bit-widths based on their importance, approaching the accuracy-memory Pareto frontier and enabling extreme low-bit quantization. However, existing methods rely on layer-wise importance estimation and overlook router shifts induced by quantization, resulting in suboptimal allocation and routing. In this work, we propose Global Expert-level Mixed-precision Quantization (GEMQ) to overcome these limitations via (1) a global linear-programming formulation that captures model-wide expert importance based on quantization error analysis, and (2) efficient router fine-tuning to adapt routing to quantized experts. These components are integrated into a progressive quantization framework that iteratively refines importance estimation and allocation. Experiments demonstrate that GEMQ significantly reduces memory and accelerates inference with minimal accuracy degradation. Source code is available at this https URL .
>
---
#### [new 069] Autonomous Frontier-Based Exploration with VLM Guidance
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文属于自主机器人探索任务，旨在提升未知危险环境中的探索效率。通过引入VLM进行高层决策，替代传统几何启发式方法，实现更优的路径选择与地图覆盖。**

- **链接: [https://arxiv.org/pdf/2605.23165](https://arxiv.org/pdf/2605.23165)**

> **作者:** Aarush Aitha; Avideh Zakhor
>
> **备注:** 8 pages, 10 figures, CVPR 2026: 2nd Workshop on 3D-LLM/VLA: Bridging Language, Vision and Action in 3D Environments
>
> **摘要:** Autonomous robotic exploration of unknown and hazardous environments, a long-standing challenge, can be significantly improved by leveraging the advanced reasoning of Vision-Language Models (VLMs). We introduce a novel exploration pipeline where a VLM performs high-level strategic decision-making, guiding a conventional low-level robotics control stack. At decision points, the robot generates a multimodal prompt with its current map and visual imagery of potential paths, or frontiers. The VLM analyzes this prompt to select the most promising frontier, replacing simple geometric heuristics with contextual spatial reasoning. This approach, validated in simulation across six indoor environments, improves map coverage by up to 24\% over existing methods. Our pipeline is lightweight, training-free, and easily transferable to any robot with standard sensors and an internet connection.
>
---
#### [new 070] ModeSwitch-LLM: A Lightweight Phase-Aware Controller for Cross-Mode LLM Inference on a Single GPU
- **分类: cs.LG; cs.CL; cs.PF**

- **简介: 该论文提出ModeSwitch-LLM，解决单GPU上大模型推理效率问题，通过请求感知路由选择合适模式提升性能。**

- **链接: [https://arxiv.org/pdf/2605.23057](https://arxiv.org/pdf/2605.23057)**

> **作者:** Aman Sunesh; Ali Alshehhi; Hivansh Dhakne
>
> **备注:** 10 pages main text, 11 pages including references, 5 figures, 3 tables. Preprint
>
> **摘要:** ModeSwitch-LLM is a lightweight request-boundary controller for improving single-GPU large language model inference efficiency by routing each request to an appropriate fixed inference mode. Instead of relying on one static serving configuration, the system selects among FP16, quantized modes, speculative decoding, and hybrid modes such as GPTQ plus prefix caching and INT8 plus continuous batching using cheap workload-level features. We evaluate ModeSwitch-LLM on Meta-Llama-3.1-8B-Instruct served on a single NVIDIA A100 GPU. On deployment-style synthetic workloads, the online controller achieves a 2.10x mean latency speedup over FP16 and a 0.48x mean energy ratio, corresponding to 51.7% lower energy per token. On automatic benchmarks used as a quality gate, accuracy remains close to FP16 with a mean delta of +0.17 percentage points. We also evaluate lightweight learned routers, but find that they do not clearly outperform the rule-based controller because they add routing overhead and more often select modes that violate quality, energy, or memory constraints. These results show that simple request-aware routing can recover substantial efficiency from existing inference modes without retraining the model or changing its architecture.
>
---
#### [new 071] Robust LLM Watermarking with Minimal Semantic Distortion for IP Protection
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM知识产权保护问题。提出SAFESEAL框架，在保持模型性能的同时实现强可检测性水印，有效抵御攻击。**

- **链接: [https://arxiv.org/pdf/2605.23175](https://arxiv.org/pdf/2605.23175)**

> **作者:** Kieu Dang; Phung Lai; NhatHai Phan; Yelong Shen; Ruoming Jin
>
> **摘要:** Proprietary large language models (LLMs) face risks of intellectual property (IP) violation, as adversaries can replicate an LLM by collecting input-output pairs to train a surrogate model, causing financial setbacks. Watermarks offer a promising defense to verify ownership, but existing methods often struggle with semantic distortion, factual inconsistency, and adversarial attacks. In addition, key-conditioned watermarks for provider-specific detection, especially in cross-provider and multi-user scenarios, remain largely underexplored. To address these challenges, we propose SAFESEAL, a novel key-conditioned watermarking framework that achieves strong detectability with minimal impact on model utility, effectively balancing detectability, utility, and robustness. SAFESEAL preserves named entities while substituting linguistic terms with context-aware synonyms through a key-conditioned Tournament sampling mechanism, maintaining semantic fidelity and factual consistency. For detection, we introduce a key-conditioned contrastive detector that jointly encodes the text and key, enabling provider-specific and robust watermark verification. We derive theoretical bounds on the utility-detectability trade-off and significantly reduce latency through lightweight models, batching, and parallelism. Extensive experiments show that SAFESEAL outperforms baselines in utility, detectability, and robustness, achieving a BERTScore of 0.983, entity similarity of 0.963, a 98.2% detection rate, and the highest human ratings for text quality and content preservation, with latency comparable to the fastest baseline. To promote transparency and community-driven progress, we release the first public watermark leaderboard and an interactive demo.
>
---
#### [new 072] EVE-Agent: Evidence-Verifiable Self-Evolving Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出EVE-Agent，解决自进化搜索代理中缺乏可验证证据的问题。通过生成带证据的问答对，提升训练可靠性，确保自生成课程可信可审计。**

- **链接: [https://arxiv.org/pdf/2605.22905](https://arxiv.org/pdf/2605.22905)**

> **作者:** Yamato Arai; Yuma Ichikawa
>
> **备注:** 23 pages, 2 figures
>
> **摘要:** Self-evolving agents should not train on examples they cannot justify. Data-free self-evolving search agents offer a scalable route to systems that generate their own questions, answer them, and improve from their own feedback without human annotations. Yet, without verifiable evidence, this loop can reward fluent but unsupported examples, turning the self-generated curriculum into an opaque and potentially unreliable training signal. We argue that evidence verifiability is a prerequisite for trustworthy self-evolution in search agents: each generated instance should include not only an answer but also a source-grounded span whose contribution to that answer can be measured. We introduce EVE-Agent, an Evidence-Verifiable Self-Evolving Agent that operationalizes this principle through a modification to the proposer--solver framework. The proposer generates a question, an answer, and a verbatim evidence span. An evidence verifier then rewards the span according to the marginal accuracy gain when the evidence is provided. This produces a training signal that favors evidence that genuinely helps answer the question, without requiring oracle answers, human labels, or external annotations. EVE-Agent leaves the backbone model, retriever, search tool, and optimization framework unchanged. Experiments show that EVE-Agent substantially improves evidence-grounded correctness over prior self-evolving search agents. The resulting curriculum is not merely self-generated but auditable by construction: each training example carries an inspectable source span that explains why it should be trusted.
>
---
#### [new 073] Multi-Gate Residuals
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于深度学习任务，旨在解决残差网络中激活值无界增长的问题。提出Multi-Gate Residuals方法，在不增加通信开销的情况下稳定激活规模。**

- **链接: [https://arxiv.org/pdf/2605.23259](https://arxiv.org/pdf/2605.23259)**

> **作者:** Zhizhan Zheng; Feiyun Zhang; Shuchun Liu; Tian Xia; Xi Liu; Dasheng Hu; Hongquan Zhou
>
> **摘要:** While Attention Residuals has shown some effectiveness in addressing the widespread issue of unbounded activation growth across deep residual layers, it inevitably incurs significant communication overhead. To circumvent this bottleneck, we propose Multi-Gate Residuals (MGR), which stabilizes activation scales without additional communication burden. It utilizes a straightforward scoring and gating mechanism to maintain multi-stream context, coupled with Attention Pooling to extract hidden states from the stream states. Empirical experiments demonstrate that MGR is practical for large-scale training and deployment, offering tangible performance improvements over existing architectures.
>
---
#### [new 074] CultivAgents: Cultivating Relationship-Centered Multi-Agent Systems for Personalized Gardening
- **分类: cs.HC; cs.CL; cs.CY; cs.MA**

- **简介: 该论文提出CultivAgents，一个面向个性化园艺的多智能体系统，解决传统工具缺乏文化与生态适配性的问题。通过多阶段评估验证其有效性。**

- **链接: [https://arxiv.org/pdf/2605.23193](https://arxiv.org/pdf/2605.23193)**

> **作者:** Yiyang Wang; Moeiini Reilly; Britney Johnson; Kefei Yan; Alex Cabral; Josiah Hester
>
> **备注:** Preprint, 9 pages. Website: this https URL
>
> **摘要:** Gardening is critical to support well-being, cultural continuity, and food autonomy, yet existing digital tools often provide generic advice that overlooks gardeners' skills, local ecologies, seasons, and cultural contexts. We introduce CultivAgents, a relationship-centered multi-agent system for personalized, socio-culturally grounded gardening support. Grounded in ethics of care, CultivAgents coordinates multiple specialized agents: an Experience Agent that adapts guidance to users' skill levels, an Environmental Agent that grounds advice in local and seasonal conditions, and an Ethnobotanical Agent that connects plants to cultural knowledge and histories. We evaluated CultivAgents through a three-phase mixed-methods study with domain experts (n=3), HCI researchers (n=7), and community gardeners (n=5), analyzing expert feedback, pre/post surveys, and participatory design activities. Results suggest that CultivAgents helped gardeners translate interest into situated action: community gardeners reported increased confidence (3.00 to 3.60), motivation (4.00 to 4.40), and trust in acting on AI advice (3.20 to 4.00). Participants valued hyperlocal ecological guidance and complementary agent perspectives, while also identifying limits in cultural specificity, ecological grounding, and agent coordination. The work advances relationship-centered AI, offering design implications for multi-agent systems that support food sovereignty, community resilience, and cultural preservation.
>
---
#### [new 075] CoSPlay: Cooperative Self-Play at Test-Time with Self-Generated Code and Unit Test
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出CoSPlay，解决代码生成中依赖真实单元测试的问题。通过自生成测试用例与代码协同优化，提升代码质量与测试准确性，实现无需真实测试数据的高效代码生成。**

- **链接: [https://arxiv.org/pdf/2605.23491](https://arxiv.org/pdf/2605.23491)**

> **作者:** Zhangyi Hu; Chenhui Liu; Tian Huang; Jindong Li; Yang Yang; Jiemin Wu; Zining Zhong; Menglin Yang; Yutao Yue
>
> **备注:** Code is available at: this https URL | Data & log is available at: this https URL
>
> **摘要:** Recently, Reinforcement Learning with Verifiable Rewards (RLVR) and Test-Time Scaling (TTS) have advanced LLM code generation through executable verification. Yet Ground-Truth Unit Tests (GT UTs) remain a bottleneck: SOTA RLVR methods require them for costly training, while existing TTS methods lose competitiveness without them. This motivates GT-free TTS, where existing methods directly use self-generated UTs to refine and select code candidates. Yet such UTs are often noisy or spuriously coupled with wrong code, and UT quality in turn cannot be validated without reliable code. The key challenge is therefore to jointly improve both. To this end, we present CoSPlay, a GT-free, training-free framework that jointly improves codes and UTs through cooperative self-play. It first explores diverse solution ideas and identifies their potential failure modes to produce discriminative UT ideas. It then uses bidirectional pass-count signals from the Code-UT execution matrix to iteratively prune or fix weak codes and refresh or replace unreliable UTs, letting the two pools co-evolve. Finally, when multiple codes remain tied at the highest pass count, it picks the final code from the largest output-consensus cluster, since correct codes agree on the same inputs while wrong codes diverge. Experiments on four challenging benchmarks show that CoSPlay on Qwen2.5-7B-Instruct improves average BoN from 22.1% to 33.2% and UT accuracy from 14.6% to 78.3%, matching or surpassing the RLVR model CURE-7B. When applied to CURE-7B, it further improves BoN by 5.7%. CoSPlay also generalizes across diverse backbones and outperforms GT-free TTS baselines under comparable token budgets, with continued gains as the budget scales up. These results suggest a scalable inference strategy for competitive code generation without any GT data.
>
---
#### [new 076] AI-Friendly LaTeX: Using LaTeX Code as a Knowledge Source for Retrieval-Augmented Generation
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决如何将LaTeX源代码转化为AI友好的知识结构。通过预处理，将LaTeX转换为适合向量数据库索引的Markdown和JSONL格式。**

- **链接: [https://arxiv.org/pdf/2605.22923](https://arxiv.org/pdf/2605.22923)**

> **作者:** Tom Verhoeff
>
> **备注:** 19 pages, 3 figures
>
> **摘要:** Large language models can answer questions about textbooks, lecture notes, and programming exercises more reliably when their answers are grounded in an explicit knowledge source. Retrieval-augmented generation (RAG) is a common approach: relevant fragments of a document are retrieved and inserted into the model context before answering. For mathematical and technical material, the original LaTeX source can be a better starting point than a PDF, because it contains structural information, labels, sectioning commands, macros, and authorial intent that are often lost or distorted in PDF extraction. However, LaTeX source is not automatically AI-friendly. Cross-references must be resolved, custom macros must be interpreted, exercises and examples must be identified, and author-supplied semantic metadata may be needed. This article describes a focused preprocessing approach for turning LaTeX source, together with its compiled auxiliary files and optional author annotations, into Markdown and JSONL chunks suitable for indexing in a vector database.
>
---
#### [new 077] The Readout Shortcut: Positional Number Copying Dominates Arithmetic CoT Readout in Small Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究小语言模型在算术任务中依赖链式思维（CoT）的机制，发现模型主要通过复制答案位置的数字而非逻辑推理来生成答案。任务为算术推理，解决CoT实际贡献问题，通过实验揭示了模型的“位置捷径”现象。**

- **链接: [https://arxiv.org/pdf/2605.22870](https://arxiv.org/pdf/2605.22870)**

> **作者:** Ming Liu
>
> **备注:** 18 pages (8 main + 10 appendix), 3 figures, 5 tables
>
> **摘要:** Chain-of-thought (CoT) prompting is necessary for arithmetic in small language models, yet shuffling its steps preserves most performance. What does CoT contribute if not logical sequencing? In three 1-3B instruction-tuned LMs on GSM8K, we isolate the answer-readout stage via prefix completion and identify a positional shortcut: the model copies whichever number occupies the trailing position before the answer delimiter, regardless of intermediate reasoning. Gold-answer presence accounts for 54-92 pp of accuracy (89-92% of each model's teacher-forcing ceiling); even on incorrect items, the final answer matches the last CoT number 95-96% of the time. The copy channel takes precedence over retained-context completion: replacing the trailing number with a wrong value collapses accuracy to near-zero despite correct intermediates, yet removing it recovers 5-32 pp above that floor--even single-step arithmetic the model can otherwise perform is suppressed when a copyable number is present. Qwen and Llama copy novel distractors 87-95% of the time; Gemma gates selectively. Head-level ablation implicates architecture-specific head sets; the effect replicates on GSM-Symbolic. On non-arithmetic BBH tasks, shuffle retention drops sharply; at 7-8B, content-selective gating emerges. Step-level faithfulness evaluations risk conflating positional answer transport with genuine computation--a failure mode for CoT-based oversight.
>
---
#### [new 078] The Deterministic Horizon: Impossibility Results as Design Specifications for Trustworthy AI Systems
- **分类: cs.AI; cs.CC; cs.CL; cs.LG**

- **简介: 该论文探讨AI系统设计中的不可行性结果，将其转化为设计规范。研究解决可信AI的边界问题，提出Deterministic Horizon概念，分析模型精度上限及不同场景下的约束条件。**

- **链接: [https://arxiv.org/pdf/2605.23024](https://arxiv.org/pdf/2605.23024)**

> **作者:** Dongxin Guo
>
> **备注:** PhD thesis, Department of Computer Science, The University of Hong Kong, 2026. 271 pages, 18 figures, 15 tables, 5 algorithms
>
> **摘要:** Large language models now write software, draft legal documents, and produce clinical notes, yet fundamental limits, from Turing and Arrow to the No Free Lunch theorems, shape what computation can do. This thesis turns such impossibility results from curiosities into design rules. Its flagship result proves an accuracy ceiling set by architecture alone: past a critical reasoning depth, no amount of training moves it, at any adapter rank, sample size, or loss function. Computable before deployment from layer count and embedding width, this Deterministic Horizon is measured between nineteen and thirty-one across twelve transformer architectures, and fine-tuning on optimal-length traces recovers under four percentage points. The mechanism is a capacity invariant of the residual stream, and an information-theoretic conversion yields super-exponential accuracy decay past the horizon. An unconditional circuit-complexity lower bound for modular exponentiation against constant-depth prime-modulus circuits complements this result. The same argument recasts across subfields: preference learning under any misspecified model jumps discontinuously in sample complexity; multi-stage retrieval pipelines require at least as many independent metrics as stages; standard truthful auctions fail for agents with prompt-dependent valuations; and zero-knowledge verification of neural inference pays a measured overhead of one hundred ten to one hundred ninety times per non-linear activation. Together these form a catalogue of sixteen specifications, each pairing a computable boundary, a quantified violation cost, and a constructive design rule: two compositions are proved, one pairing is an honest obstruction, and four remain open. The impossibility-specification methodology is offered for the generative research programme that trustworthy AI may need. Every fundamental limit of AI is also a design rule.
>
---
#### [new 079] FastKernels: Benchmarking GPU Kernel Generation in Production
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于GPU内核生成任务，旨在解决现有基准与生产环境不匹配的问题。作者提出FastKernels基准，涵盖多种架构，提升内核生成效果与实际应用的契合度。**

- **链接: [https://arxiv.org/pdf/2605.23215](https://arxiv.org/pdf/2605.23215)**

> **作者:** Gabriele Oliaro; Yichao Fu; May Jiang; Owen Lu; Junli Wang; Zhihao Jia; Hao Zhang; Samyam Rajbhandari
>
> **摘要:** LLM-based agents for GPU kernel generation are advancing rapidly, yet their progress is fundamentally constrained by the benchmarks they optimize against. Existing benchmarks are poorly aligned with production inference frameworks: they evaluate kernels on a single GPU with synthetic inputs, ignore the surrounding compilation stack, and reward replicating known optimizations rather than discovering new ones. The resulting reward signals are misleading: agents learn to generate kernels that score well in sandboxes but introduce interface incompatibilities, compilation-stack conflicts, and silent correctness degradation when integrated into real systems. We introduce FastKernels, a kernel benchmark built around a minimal set of 46 representative architectures spanning 8 categories, whose kernels collectively subsume those of 96.2% (409/425) of HuggingFace Transformers architectures. FastKernels doubles as a minimalistic, production-grade inference framework that runs at parity with hardened systems such as vLLM and SGLang on mainstream LLM serving and substantially exceeds upstream references on under-served architectures; each task's interface mirrors the corresponding module in the state-of-the-art library for its architecture family, enabling direct deployment of optimized kernels into production codebases. Evaluating state-of-the-art kernel agents on FastKernels, we find that even the strongest agent achieves only 0.94$\times$ aggregate speedup over production baselines, with weaker agents at $0.78\times$ and $0.53\times$ -- confirming that benchmark-production misalignment is a critical bottleneck for the field. We release FastKernels as a stepping stone toward kernel agents whose benchmark gains translate directly into production throughput improvements. Code is available at this https URL
>
---
#### [new 080] ETCHR: Editing To Clarify and Harness Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出ETCHR，解决多模态大模型中视觉推理的瓶颈问题，通过解耦图像编辑与理解模型，提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2605.23897](https://arxiv.org/pdf/2605.23897)**

> **作者:** Beichen Zhang; Yuhong Liu; Jinsong Li; Yuhang Zang; Jiaqi Wang; Dahua Lin
>
> **备注:** Code, model and data are open-sourced at this https URL
>
> **摘要:** Multimodal Large Language Models have advanced visual reasoning, yet a purely textual chain of thought remains a bottleneck for questions that require fine-grained focus or view transformations. The ''think with images'' paradigm narrows this gap, but existing approaches are either constrained by fixed predefined toolkits or produce noisy intermediate images from unified multimodal methods. We pursue a third option: using a dedicated image editing model and decouple it with an understanding model. However, off-the-shelf image editors fail as reasoning assistants with two complementary gaps: a language-side gap, where editors trained as passive instruction-followers cannot map an abstract question to an appropriate visual transformation, and a generation-side gap, where edit correctness degrades as reasoning depth grows. Guided by this analysis, we introduce ETCHR (Editing To Clarify and Harness Reasoning), a question-conditioned, reasoning-aware image editor decoupled from the downstream understanding model and trained with a two-stage recipe targeted at the two gaps: Reasoning Imitation via supervised fine-tuning on edit trajectories, followed by Reasoning Enhancement with VLM-derived rewards for edit correctness and downstream reasoning accuracy. Since the editor is decoupled, ETCHR plugs into different open- and closed-source MLLMs in a training-free manner. Across five task families (fine-grained perception, chart understanding, logic reasoning, jigsaw restoration, and 3D understanding), ETCHR raises average Pass@1 from 55.95 to 60.77 (+4.82) with Qwen3-VL-8B, from 65.08 to 70.55 (+5.47) with Gemini-3.1-Flash-Lite, and from 76.55 to 81.16 (+4.61) with the 1T-parameter MoE model Kimi K2.5.
>
---
#### [new 081] DiLaDiff: Distilled Latent-Augmented Diffusion for Language Modeling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出DiLaDiff，解决扩散语言模型生成质量与速度的矛盾。通过引入连续潜在空间和一致性蒸馏，提升生成效率与质量。任务为语言建模。**

- **链接: [https://arxiv.org/pdf/2605.23605](https://arxiv.org/pdf/2605.23605)**

> **作者:** Jean-Marie Lemercier; Tomas Geffner; Karsten Kreis; Morteza Mardani; Arash Vahdat; Ante Jukić
>
> **摘要:** Diffusion language models intrinsically fail to capture correlations between decoded tokens, which leads to a harsh trade-off between sampling quality and throughput. To solve this issue, we propose DiLaDiff, a variant of masked diffusion language models with three components: (1) a continuous latent space with semantic capabilities, learned by an auto-encoder fine-tuned from an existing masked diffusion language model; (2) a latent diffusion model learning the prior over the encoder distribution; (3) a consistency model distilling the learned prior into a few-step latent generative model. We show that, even without distillation, our latent-guided diffusion model outperforms the masked diffusion baseline while significantly accelerating inference. Consistency distillation further lowers the computational overhead of continuous diffusion, such that the latent is generated in negligible time compared to discrete decoding.
>
---
#### [new 082] Seeing without Looking: Do Vision-Language Benchmarks Really Test Vision?
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型任务，探讨基准测试是否真实评估视觉理解。研究发现现有基准无法有效衡量细粒度视觉定位，提出多维度分析验证模型对视觉证据的依赖程度。**

- **链接: [https://arxiv.org/pdf/2605.22903](https://arxiv.org/pdf/2605.22903)**

> **作者:** Zixuan Lan; Luzhe Sun; Matthew R. Walter; Jiawei Zhou
>
> **备注:** Accepted to GRAIL-V: Grounded Retrieval and Agentic Intelligence for Vision-Language, CVPR 2026 Workshop. accepted version
>
> **摘要:** Benchmark accuracy is often implicitly assumed to reflect grounded visual understanding in vision-language models (VLMs), yet it remains unclear to what extent such scores truly reflect reliance on visual evidence. Motivated by a surprising observation that removing a substantial fraction of image tokens only degrades model performance very slightly on a widely used hallucination benchmark, we systematically investigate this mismatch in a set of open-source VLMs. Our analysis spans multiple levels of granularity, spanning global visual degradation, localized occlusion, question reformulation, answer-space expansion, and decision-level analyses beyond standard accuracy. We further complement these behavioral results with a layer-wise analysis of vision-token geometry. Throughout the experiments, we find that although VLMs do incorporate visual input, their predictions are less sensitive to the loss of fine-grained visual evidence that standard accuracy should have suggested. Even when the final prediction remains unchanged, the model's internal support for the correct answer may already be weakened. We further complement a representation-level analysis, which shows increasing similarity among visual tokens in deeper layers, providing a possible explanation for our findings. Together, these results suggest that current benchmarks are not sufficient to reliably evaluate fine-grained visual grounding in VLMs.
>
---
#### [new 083] Transcoders Trace Visual Grounding and Hallucinations in Vision-Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型的可解释性研究，旨在揭示模型如何将视觉信息转化为文本。通过引入基于Transcoders的框架，分析模型的计算路径，识别视觉基础和幻觉生成机制。**

- **链接: [https://arxiv.org/pdf/2605.22902](https://arxiv.org/pdf/2605.22902)**

> **作者:** Dimitrios Damianos; Leon Voukoutis; Georgios Skyrianos; Vassilis Katsouros; Georgios Paraskevopoulos
>
> **摘要:** Generative Vision-Language Models (VLMs) perform well on multimodal reasoning, but how visual inputs are transformed to text remains poorly understood. Existing interpretability work on VLMs uses Sparse Autoencoders (SAEs), which decompose static residual representations and miss the functional updates that drive cross-modal interaction. We adopt a function-centric framework based on Transcoders, sparse approximations of MLP sublayers that act as a causal proxy for layer-wise computation. Applied to Gemma 3-4B-IT, the framework decomposes the model into interpretable computational pathways linking image patches to directions in token generation. Transcoder attributions produce stronger and more stable effects on visually grounded tokens under patch ablation than SAE attributions, and align better with semantically relevant image regions. A False Visual Grounding counterfactual analysis confirms that the recovered pathways are specific to vision-language this http URL, we perform a structural analysis of hallucinated generations, by extracting graph-based indicators from circuit traces produced by the transcoders. A logistic classifier over these mechanistic graph features predicts hallucinations at AUC $0.68$. These results show that function-centric circuit decomposition yields interpretable and predictive accounts of multimodal computation in VLMs.
>
---
#### [new 084] Strong Teacher Not Needed? On Distillation in LLM Pretraining
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究知识蒸馏在大语言模型预训练中的有效性，挑战“强教师必优”的传统观点。通过调整教师学生架构和训练数据量，发现弱教师也能有效提升学生模型性能。**

- **链接: [https://arxiv.org/pdf/2605.23857](https://arxiv.org/pdf/2605.23857)**

> **作者:** Taiming Lu; Zhuang Liu
>
> **摘要:** Knowledge distillation generally assumes a strong-to-weak relationship where stronger teachers yield better students. In this work, we examine this assumption about distillation in large language model pretraining. By varying architecture sizes and training token budgets, we create strong-to-weak, same-level, and weak-to-strong teacher-student relationships, and study distillation's effectiveness under each. We find that the teacher need not be strong: with proper mixing of the language modeling and knowledge distillation losses, even small and undertrained teachers improve larger students. At the same time, a stronger teacher is not always better: pushing the teacher further, through more parameters or more training tokens, can saturate or even reverse the distillation gains. We further observe that distillation improves generalization (out-of-distribution and downstream performance) more readily than in-domain fitting. Together, these results challenge the common belief that distillation pretraining always requires a strong teacher.
>
---
#### [new 085] Decomposing Queries into Tool Calls for Long-Video Keyframe Retrieval
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于长视频关键帧检索任务，旨在解决如何根据查询准确找到相关帧的问题。提出ToolMerge方法，通过分解查询为工具调用并合并结果，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2605.23826](https://arxiv.org/pdf/2605.23826)**

> **作者:** Michal Shlapentokh-Rothman; Prachi Garg; Yu-Xiong Wang; Derek Hoiem
>
> **摘要:** Keyframe selection is a direct way to provide verifiable visual evidence for long-video question answering (QA). Queries differ in what they require, and finding the right frames depends on knowing what to look for. Existing keyframe selectors either score every frame against a single query, or decompose the query into a fixed schema evaluated by a single visual tool. We propose ToolMerge, a keyframe retrieval method based on decomposition and merging: an Large Language Model (LLM) based planner decomposes the query into tool calls and specifies how their per-tool rankings are merged using boolean operators. To evaluate retrieval directly, we construct Molmo-2 Moments (M2M), a benchmark in which every question is anchored to a specific time interval by construction. Across QA, question retrieval, and caption retrieval, ToolMerge is competitive with prior keyframe selectors, most notably on caption retrieval, outperforming other methods by 5%. Code and data can be found at this https URL .
>
---
## 更新

#### [replaced 001] Visually-Guided Policy Optimization for Multimodal Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉-语言模型的多模态推理任务，旨在解决模型视觉忠实度不足的问题。通过引入VGPO框架，增强视觉关注与记忆，提升多模态任务表现。**

- **链接: [https://arxiv.org/pdf/2604.09349](https://arxiv.org/pdf/2604.09349)**

> **作者:** Zengbin Wang; Feng Xiong; Liang Lin; Xuecai Hu; Yong Wang; Yanlin Wang; Man Zhang; Xiangxiang Chu
>
> **备注:** Accepted to ACL 2026, this https URL
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has significantly advanced the reasoning ability of vision-language models (VLMs). However, the inherent text-dominated nature of VLMs often leads to insufficient visual faithfulness, characterized by sparse attention activation to visual tokens. More importantly, our empirical analysis reveals that temporal visual forgetting along reasoning steps exacerbates this deficiency. To bridge this gap, we propose Visually-Guided Policy Optimization (VGPO), a novel framework to reinforce visual focus during policy optimization. Specifically, VGPO initially introduces a Visual Attention Compensation mechanism that leverages visual similarity to localize and amplify visual cues, while progressively elevating visual expectations in later steps to counteract visual forgetting. Building on this mechanism, we implement a dual-grained advantage re-weighting strategy: the intra-trajectory level highlights tokens exhibiting relatively high visual activation, while the inter-trajectory level prioritizes trajectories demonstrating superior visual accumulation. Extensive experiments demonstrate that VGPO achieves better visual activation and superior performance in mathematical multimodal reasoning and visual-dependent tasks. The code has been released at this https URL.
>
---
#### [replaced 002] Skill Retrieval Augmentation for Agentic AI
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于智能代理任务，旨在解决技能调用效率低的问题。提出SRA方法，通过动态检索外部技能提升代理性能，并构建了SRA-Bench进行评估。**

- **链接: [https://arxiv.org/pdf/2604.24594](https://arxiv.org/pdf/2604.24594)**

> **作者:** Weihang Su; Jianming Long; Qingyao Ai; Yichen Tang; Changyue Wang; Yiteng Tu; Yiqun Liu
>
> **摘要:** As large language models (LLMs) evolve into agentic problem solvers, they increasingly rely on external, reusable skills to handle tasks beyond their native parametric capabilities. In existing agent systems, the dominant strategy for incorporating skills is to explicitly enumerate available skills within the context window. However, this strategy fails to scale: as skill corpora expand, context budgets are consumed rapidly, and the agent becomes markedly less accurate in identifying the right skill. To this end, this paper formulates Skill Retrieval Augmentation (SRA), a new paradigm in which agents dynamically retrieve, incorporate, and apply relevant skills from large external skill corpora on demand. To make this problem measurable, we construct a large-scale skill corpus and introduce SRA-Bench, the first benchmark for decomposed evaluation of the full SRA pipeline, covering skill retrieval, skill incorporation, and end-task execution. SRA-Bench contains 5,400 capability-intensive test instances and 636 manually constructed gold skills, which are mixed with web-collected distractor skills to form a large-scale corpus of 26,262 skills. Extensive experiments show that retrieval-based skill augmentation can substantially improve agent performance, validating the promise of the paradigm. At the same time, we uncover a fundamental gap in skill incorporation: current LLM agents tend to load skills at similar rates, regardless of whether a gold skill is retrieved or whether the task actually requires external capabilities. This shows that the bottleneck in skill augmentation lies not only in retrieval but also in the base model's ability to determine which skill to load and when external loading is actually needed. These findings position SRA as a distinct research problem and establish a foundation for the scalable augmentation of capabilities in future agent systems.
>
---
#### [replaced 003] Benchmarking Commercial ASR Systems on Code-Switching Speech: Arabic, Persian, and German
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音识别任务，研究代码转换语音的识别效果，评估商业ASR系统在多语言对中的表现，解决代码转换场景下ASR准确率低的问题。**

- **链接: [https://arxiv.org/pdf/2605.19069](https://arxiv.org/pdf/2605.19069)**

> **作者:** Sajjad Abdoli; Ghassan Al-Sumaidaee; Clayton W. Taylor; Ahmad ElShiekh; Ahmed Rashad
>
> **摘要:** Code-switching -- the natural alternation between two languages within a single utterance -- remains one of the most challenging and under-studied conditions for automatic speech recognition (ASR). We present a benchmark evaluating five commercial ASR providers across four language pairs: Egyptian Arabic--English, Saudi Arabic (Najdi/Hijazi)--English, Persian (Farsi)--English, and German--English, comprising 300 samples per pair selected by a two-stage pipeline combining heuristic filtering with a GPT-4o and Gemini 1.5 Pro ensemble scorer, reducing LLM costs by $\approx$91\%. We evaluate on both WER and BERTScore, showing that while both metrics agree on the ordinal ranking of systems for all Arabic and Persian pairs ($\tau = 1.0$), WER inflates the magnitude of quality gaps by approximately 3$\times$ by penalising semantically correct transliteration choices. ElevenLabs Scribe v2 achieves the lowest WER (13.2\% overall) and leads on BERTScore (0.936 overall). Difficulty-stratified analysis reveals performance gaps masked by aggregate averages, and BERT embedding projections confirm semantic proximity between reference and hypothesis despite surface-level script differences. The dataset is publicly available at this https URL.
>
---
#### [replaced 004] TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration
- **分类: cs.CL**

- **简介: 该论文属于语言模型加速任务，解决MoE扩散模型中专家激活效率低的问题。提出TEAM框架，通过时间空间一致性提升解码效率，实现2.2倍加速。**

- **链接: [https://arxiv.org/pdf/2602.08404](https://arxiv.org/pdf/2602.08404)**

> **作者:** Linye Wei; Zixiang Luo; Pingzhi Tang; Meng Li
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Diffusion large language models (dLLMs) have recently gained significant attention due to their inherent support for parallel decoding. Building on this paradigm, Mixture-of-Experts (MoE) dLLMs with autoregressive (AR) initialization have further demonstrated strong performance competitive with mainstream AR models. However, we identify a fundamental mismatch between MoE architectures and diffusion-based decoding. Specifically, a large number of experts are activated at each denoising step, while only a small subset of tokens is ultimately accepted, resulting in substantial inference overhead and limiting their deployment in latency-sensitive applications. In this work, we propose TEAM, a plug-and-play framework that accelerates MoE dLLMs by enabling more accepted tokens with fewer activated experts. TEAM is motivated by the observation that expert routing decisions exhibit strong temporal consistency across denoising levels as well as spatial consistency across token positions. Leveraging these properties, TEAM employs three complementary expert activation and decoding strategies, conservatively selecting necessary experts for decoded and masked tokens and simultaneously performing aggressive speculative exploration across multiple candidates. Experimental results demonstrate that TEAM achieves up to 2.2x speedup over vanilla MoE dLLM, with negligible performance degradation. Code is released at this https URL.
>
---
#### [replaced 005] Evaluating Memory Structure in LLM Agents
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于记忆结构评估任务，旨在解决LLM在复杂记忆组织上的能力不足问题。提出StructMemEval基准，测试代理组织长期记忆的能力，发现简单检索增强模型表现不佳，需明确提示才能有效处理结构化任务。**

- **链接: [https://arxiv.org/pdf/2602.11243](https://arxiv.org/pdf/2602.11243)**

> **作者:** Alina Shutova; Alexandra Olenina; Ivan Vinogradov; Anton Sinitsin
>
> **备注:** Preprint, work in progress
>
> **摘要:** Modern LLM-based agents and chat assistants rely on long-term memory frameworks to store reusable knowledge, recall user preferences, and augment reasoning. As researchers create more complex memory architectures, it becomes increasingly difficult to analyze their capabilities and guide future memory designs. Most long-term memory benchmarks focus on simple fact retention, multi-hop recall, and time-based changes. While undoubtedly important, these capabilities can often be achieved with simple retrieval-augmented LLMs and do not test complex memory hierarchies. To bridge this gap, we propose StructMemEval - a benchmark that tests the agent's ability to organize its long-term memory, not just factual recall. We gather a suite of tasks that humans solve by organizing their knowledge in a specific structure: transaction ledgers, to-do lists, trees and others. Our initial experiments show that simple retrieval-augmented LLMs struggle with these tasks, whereas memory agents can reliably solve them if prompted how to organize their memory. However, we also find that modern LLMs do not always recognize the memory structure when not prompted to do so. This highlights an important direction for future improvements in both LLM training and memory frameworks.
>
---
#### [replaced 006] Mind Your Moras: Orthography-Aware Error Analysis of Neural Japanese Morphological Generation
- **分类: cs.CL**

- **简介: 该论文属于日语形态生成任务，研究模型在处理过去时形态时的错误，分析与假名拼写相关的系统性错误，提出错误分类体系。**

- **链接: [https://arxiv.org/pdf/2605.20043](https://arxiv.org/pdf/2605.20043)**

> **作者:** Wen Zhang
>
> **摘要:** We present an orthography-aware error analysis of Japanese past-tense morphological inflection, treating hiragana not merely as a transcriptional medium, but as a representational system encoding morphophonological distinctions that may influence model generalization. We evaluate two character-level sequence-to-sequence architectures on past-tense formation using datasets formatted according to the SIGMORPHON 2020 and 2023 shared task conventions. Despite high aggregate accuracy, models exhibit systematic, linguistically interpretable errors that cluster around specific orthographic properties of hiragana. We introduce a concise error taxonomy capturing seven primary failure modes and provide both quantitative and qualitative analyses. Gemination-related errors dominate residual failures, accounting for 75-80% of errors, particularly in verbs whose stems end in the vowel e and require gemination before the past-tense suffix. Error patterns remain highly consistent across architectures and random seeds, suggesting a robust interaction between orthographic representation, morphological structure, and data frequency effects in shaping model generalization. These results underscore the necessity of orthography-aware evaluation for understanding neural generalization in morphologically complex languages.
>
---
#### [replaced 007] CoFrGeNet: Continued Fraction Architectures for Language Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CoFrGeNet，一种基于连分数的生成模型架构，用于语言生成任务。旨在解决Transformer参数多、训练耗时的问题，通过新组件替代注意力和前馈网络，减少参数并提升效率。**

- **链接: [https://arxiv.org/pdf/2601.21766](https://arxiv.org/pdf/2601.21766)**

> **作者:** Amit Dhurandhar; Vijil Chenthamarakshan; Dennis Wei; Tejaswini Pedapati; Karthikeyan Natesan Ramamurthy; Rahul Nair
>
> **备注:** Earlier version accepted to ICML 2026
>
> **摘要:** Transformers are arguably the preferred architecture for language generation. In this paper, inspired by continued fractions, we introduce a new function class for generative modeling. The architecture family implementing this function class is named CoFrGeNets - Continued Fraction Generative Networks. We design novel architectural components based on this function class that can replace Multi-head Attention and Feed-Forward Networks in Transformer blocks while requiring much fewer parameters. We derive custom gradient formulations to optimize the proposed components more accurately and efficiently than using standard PyTorch-based gradients. Our components are a plug-in replacement requiring little change in training or inference procedures that have already been put in place for Transformer-based models thus making our approach easy to incorporate in large industrial workflows. We experiment on two very different transformer architectures GPT2-xl (1.5B) and Llama3 (3.2B), where the former we pre-train on OpenWebText and GneissWeb, while the latter we pre-train on the docling data mix which consists of nine different datasets. Results show that the performance on downstream classification, Q\& A, reasoning and text understanding tasks of our models is competitive and sometimes even superior to the original models with $\frac{2}{3}$ to $\frac{1}{2}$ the parameters and shorter pre-training time. We believe that future implementations customized to hardware will further bring out the true potential of our architectures.
>
---
#### [replaced 008] Improving Sampling for Masked Diffusion Models via Information Gain
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，针对Masked Diffusion Models采样方法存在的局部最优问题，提出Info-Gain Sampler，通过平衡即时不确定性和信息增益提升生成质量。**

- **链接: [https://arxiv.org/pdf/2602.18176](https://arxiv.org/pdf/2602.18176)**

> **作者:** Kaisen Yang; Jayden Teoh; Kaicheng Yang; Yitong Zhang; Alex Lamb
>
> **备注:** this https URL Accepted by ICML2026 Accepted by ICML2026
>
> **摘要:** Masked Diffusion Models (MDMs) enable flexible decoding orders, yet existing samplers remain largely greedy, selecting locally certain tokens without accounting for their downstream effects. We show that this myopia can increase cumulative uncertainty and lead to suboptimal generation. To address this, we propose the **Info-Gain Sampler**, a training-free decoding method that uses the bidirectional structure of MDMs to balance immediate uncertainty with the information gained over remaining masked positions. Across reasoning, coding, creative writing, and image generation tasks, Info-Gain Sampler consistently outperforms existing MDM samplers, improving average reasoning accuracy by 2.9--11.6 percentage points and achieving a 62.8% average win rate in creative writing. The code is available at this https URL.
>
---
#### [replaced 009] InfiGFusion: Graph-on-Logits Distillation via Efficient Gromov-Wasserstein for Model Fusion
- **分类: cs.CL**

- **简介: 该论文属于模型融合任务，旨在解决多模型协同生成中的语义依赖问题。通过构建图结构的logits蒸馏损失，提升融合效果与稳定性。**

- **链接: [https://arxiv.org/pdf/2505.13893](https://arxiv.org/pdf/2505.13893)**

> **作者:** Yuanyi Wang; Zhaoyi Yan; Yiming Zhang; Qi Zhou; Yanggan Gu; Fei Wu; Hongxia Yang
>
> **摘要:** Recent advances in large language models (LLMs) have intensified efforts to fuse heterogeneous open-source models into a unified system that inherits their complementary strengths. Existing logit-based fusion methods maintain inference efficiency but treat vocabulary dimensions independently, overlooking semantic dependencies encoded by cross-dimension interactions. These dependencies reflect how token types interact under a model's internal reasoning and are essential for aligning models with diverse generation behaviors. To explicitly model these dependencies, we propose \textbf{InfiGFusion}, the first structure-aware fusion framework with a novel \textit{Graph-on-Logits Distillation} (GLD) loss. Specifically, we retain the top-$k$ logits per output and aggregate their outer products across sequence positions to form a global co-activation graph, where nodes represent vocabulary channels and edges quantify their joint activations. To ensure scalability and efficiency, we design a sorting-based closed-form approximation that reduces the original $O(n^4)$ cost of Gromov-Wasserstein distance to $O(n \log n)$, with provable approximation guarantees. Experiments across multiple fusion settings show that GLD consistently improves fusion quality and stability. InfiGFusion outperforms SOTA models and fusion baselines across 11 benchmarks spanning reasoning, coding, and mathematics. It shows particular strength in complex reasoning tasks, with +35.6 improvement on Multistep Arithmetic and +37.06 on Causal Judgement over SFT, demonstrating superior multi-step and relational inference.
>
---
#### [replaced 010] ThoughtTrace: Understanding User Thoughts in Real-World LLM Interactions
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ThoughtTrace数据集，解决用户思维与行为不匹配的问题。通过收集真实对话和用户自述思考，用于提升AI对用户隐含需求的理解与适应能力。**

- **链接: [https://arxiv.org/pdf/2605.20087](https://arxiv.org/pdf/2605.20087)**

> **作者:** Chuanyang Jin; Binze Li; Haopeng Xie; Cathy Mengying Fang; Tianjian Li; Shayne Longpre; Hongxiang Gu; Maximillian Chen; Tianmin Shu
>
> **备注:** 53 pages, 23 figures, 4 tables. Project website: this https URL
>
> **摘要:** Conversational AI has now reached billions of users, yet existing datasets capture only what people say, not what they think. We introduce ThoughtTrace, the first large-scale dataset that pairs real-world multi-turn human--AI conversations with users' self-reported thoughts: their reasons for sending prompts and reactions to assistant responses. ThoughtTrace comprises 1,058 users, 2,155 conversations, 17,058 turns, and 10,174 thought annotations collected across 20 language models. Our analysis shows that ThoughtTrace captures long-horizon, topically diverse interactions, and that thoughts are semantically distinct from messages, difficult for frontier LLMs to infer from context, diverse in content, and tied to conversation stages. We further demonstrate the utility of thoughts for downstream modeling. First, thoughts improve user-behavior prediction as inference-time context. Second, thought-guided rewrites provide fine-grained alignment signals for training personalized assistants. Together, ThoughtTrace establishes user thoughts as a new data modality for studying the cognitive dynamics behind human--AI interaction and provides a foundation for building assistants that better understand and adapt to users' latent goals, preferences, and needs.
>
---
#### [replaced 011] More Context, Larger Models, or Moral Knowledge? A Systematic Study of Schwartz Value Detection in Political Texts
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究政治文本中Schwartz价值检测任务，探讨上下文、道德知识和模型规模对检测效果的影响。**

- **链接: [https://arxiv.org/pdf/2605.22641](https://arxiv.org/pdf/2605.22641)**

> **作者:** Víctor Yeste; Paolo Rosso
>
> **备注:** Code: this https URL, best model: this https URL, 18 pages, 3 figures
>
> **摘要:** Detecting Schwartz values in political text is difficult because implicit cues often depend on surrounding arguments and fine-grained distinctions between neighboring values. We study when context and explicit moral knowledge help sentence-level value detection. Using the ValuesML/Touché ValueEval format, we compare sentence, window, and full-document inputs; no-RAG and retrieval-augmented settings with a curated moral knowledge base; supervised DeBERTa-v3-base/large encoders; and zero-shot LLMs from 12B to 123B parameters. The results show that more context is not uniformly better: full-document context improves supervised DeBERTa encoders by 3.8-4.8 macro-F1 points over sentence-only input, but does not consistently help zero-shot LLMs. Retrieved moral knowledge is more consistently useful in matched comparisons, improving each tested model family and context condition under early fusion. However, scaling from DeBERTa-v3-base to large and from 12B to larger LLMs does not guarantee gains, and simple early fusion outperforms the tested late-fusion and cross-attention RAG variants for encoders. Per-value analyses show that context and retrieval help most for socially situated or conceptually confusable values. These findings suggest that value-sensitive NLP should evaluate context, knowledge, and model family jointly rather than treating longer inputs or larger models as universal improvements.
>
---
#### [replaced 012] Entropy-Gradient Inversion: Moving Toward Internal Mechanism of Large Reasoning Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于大模型推理研究，旨在解决模型内部机制与行为分析的差距及强化学习不稳定问题。提出熵-梯度反转和CorR-PO方法，提升推理性能。**

- **链接: [https://arxiv.org/pdf/2605.17770](https://arxiv.org/pdf/2605.17770)**

> **作者:** Junyao Yang; Chen Qian; Kun Wang; Linfeng Zhang; Quanshi Zhang; Yong Liu; Dongrui Liu
>
> **备注:** The authors are withdrawing this manuscript due to fundamental inaccuracies in the institutional affiliations and administrative attributions provided at the time of submission. As this version cannot be validated under the correct institutional framework, the authors request its formal withdrawal from the repository. No immediate replacement is intended
>
> **摘要:** The advancement of Large Reasoning Models (LRMs) has catalyzed a paradigm shift from reactive ``fast thinking'' text generation to systematic, step-by-step ``slow thinking'' reasoning, unlocking state-of-the-art performance in complex mathematical and logical tasks. However, the field faces \textit{the fundamental gap between token-level behavioral analysis and internal reasoning mechanisms, and the instability of reinforcement learning (RL) for reasoning optimization relying on costly external verifiers}. We identify and formally define \textbf{Entropy-Gradient Inversion}, a robust negative correlation between token entropy and logit gradients that acts as a definitive geometric fingerprint for LRM reasoning capability. Building on this, we propose \textbf{Correlation-Regularized Group Policy Optimization (CorR-PO)}, which embeds this inversion signature into RL reward regularization. Extensive experiments on various reasoning benchmarks across multiple model scales show CorR-PO consistently outperforms state-of-the-art baselines, confirming that stronger inversion directly correlates with superior reasoning performance.
>
---
#### [replaced 013] DELICATE: Diachronic Entity LInking using Classes And Temporal Evidence
- **分类: cs.CL**

- **简介: 该论文聚焦于历史意大利语的实体链接任务，解决因文档类型复杂、数据不足和长尾实体带来的挑战。提出DELICATE方法与ENIEDE语料库，提升实体链接效果与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.10404](https://arxiv.org/pdf/2511.10404)**

> **作者:** Cristian Santini; Sebastian Barzaghi; Paolo Sernani; Emanuele Frontoni; Mehwish Alam
>
> **摘要:** In spite of the remarkable advancements in the field of Natural Language Processing, the task of Entity Linking (EL) remains challenging in the field of humanities due to complex document typologies, lack of domain-specific datasets and models, and long-tail entities, i.e., entities under-represented in Knowledge Bases (KBs). The goal of this paper is to address these issues with two main contributions. The first contribution is DELICATE, a novel neuro-symbolic method for EL on historical Italian which combines a BERT-based encoder with contextual information from Wikidata to select appropriate KB entities using temporal plausibility and entity type consistency. The second contribution is ENEIDE, a multi-domain EL corpus in historical Italian semi-automatically extracted from two annotated editions spanning from the 19th to the 20th century and including literary and political texts. Results show how DELICATE outperforms other EL models in historical Italian even if compared with larger architectures with billions of parameters. Moreover, further analyses reveal how DELICATE confidence scores and features sensitivity provide results which are more explainable and interpretable than purely neural methods.
>
---
#### [replaced 014] Fine-grained Claim-level RAG Benchmark for Law
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律领域RAG系统评估任务，旨在解决现有评估框架缺乏细粒度分析及语言、用户群体局限的问题。工作包括构建多语言、多用户类型的ClaimRAG-LAW数据集，并进行细粒度性能分析。**

- **链接: [https://arxiv.org/pdf/2605.21071](https://arxiv.org/pdf/2605.21071)**

> **作者:** Souvick Das; Sallam Abualhaija; Domenico Bianculli
>
> **摘要:** The rapid progress of large language models (LLMs) is shifting semantic search toward a question-answering paradigm, where users ask questions and LLMs generate responses. In high-stake domains such as law, retrieval-augmented generation (RAG) is commonly used to mitigate hallucinations in generated responses. Nonetheless, prior work shows that RAG systems, whether general-purpose or legal-specific, still hallucinate at varying rates, making fine-grained evaluation essential. Despite the need, existing evaluation frameworks for legal RAG systems lack the granularity required to provide detailed analysis of retrieval and generation performance separately. Moreover, current benchmarks are largely English-only and centered on legal expert queries, overlooking non-expert needs. We introduce ClaimRAG-LAW, a comprehensive dataset for legal RAG that supports French and English, targets both experts and non-experts, and includes diverse question types reflecting realistic scenarios. We further apply a fine-grained evaluation framework of state-of-the-art legal RAG systems, revealing limitations in retrieval, generation, and claim-level analysis in the legal domain.
>
---
#### [replaced 015] Efficient and Transferable Agentic Knowledge Graph RAG via Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱问答任务，旨在解决KG-RAG系统效率低、依赖固定流程的问题。提出KG-R1框架，通过强化学习优化，实现高效且可迁移的问答效果。**

- **链接: [https://arxiv.org/pdf/2509.26383](https://arxiv.org/pdf/2509.26383)**

> **作者:** Junhong Lin; Shicheng Liu; Jinyeop Song; Song Wang; Julian Shun; Yada Zhu
>
> **摘要:** Knowledge-graph retrieval-augmented generation (KG-RAG) couples large language models (LLMs) with structured, verifiable knowledge graphs (KGs) to reduce hallucination and provide reasoning traces. However, current KG-RAG systems often rely on fixed pipelines of multiple LLM modules (e.g., planning, reasoning, and responding), which inflate inference costs and tie performance to specific graph schemas. To address this, we introduce KG-R1, an agentic framework that optimizes KG-RAG through reinforcement learning (RL). Unlike modular workflows, KG-R1 uses a single agent that interacts with KGs as its environment, learning to retrieve information at each step and incorporating it into its reasoning and generation in a unified process. Across Knowledge-Graph Question Answering (KGQA) benchmarks, KG-R1 demonstrates both efficiency and transferability-using Qwen 2.5-3B, KG-R1 improves answer accuracy with fewer generation tokens than prior multi-module workflow methods that use much larger foundation or fine-tuned models. Furthermore, KG-R1 exhibits strong plug-and-play capability: after training, maintaining accuracy on unseen KGs without retraining. These properties make KG-R1 a promising KG-RAG framework for real-world deployment. Our code is publicly available at this http URL.
>
---
#### [replaced 016] When Irregularity Helps: A Subclass Analysis of Inductive Bias in Neural Morphology
- **分类: cs.CL**

- **简介: 该论文研究神经形态生成任务，解决模型在罕见形态子类上的错误集中问题。通过分析日语动词过去式，发现特定不规则子类导致大量错误，提出需进行更细粒度的子类分析。**

- **链接: [https://arxiv.org/pdf/2605.20558](https://arxiv.org/pdf/2605.20558)**

> **作者:** Wen Zhang
>
> **摘要:** Neural morphological generation systems often achieve high aggregate accuracy on benchmark datasets, yet such performance can conceal systematic errors concentrated in rare morphological subclasses. We examine Japanese past-tense verb inflection and show that a very small, structurally specific irregular subtype (<1% of data) accounts for a disproportionate share of model errors. Controlled ablation experiments demonstrate that removing this subtype yields larger improvements in generalization than removing all irregular verbs, indicating that not all irregularity contributes equally to model instability. These findings suggest that error concentration is driven by the interaction between extreme low-frequency morphological patterns and specific morphophonological processes, particularly gemination. We argue that morphological evaluation should incorporate finer-grained subclass analysis beyond standard conjugation categories.
>
---
#### [replaced 017] PRISM: Pareto-Efficient Retrieval over Intent-Aware Structured Memory for Long-Horizon Agents
- **分类: cs.CL**

- **简介: 该论文提出PRISM，解决长周期智能体的记忆管理问题。通过检索与压缩结合，提升回答准确率并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2605.12260](https://arxiv.org/pdf/2605.12260)**

> **作者:** Jingyi Peng; Zhongwei Wan; Weiting Liu; Qiuzhuang Sun
>
> **备注:** Preprint
>
> **摘要:** Long-horizon language agents accumulate conversation history far faster than any fixed context window can hold, making memory management critical to both answer accuracy and serving cost. Existing approaches either expand the context window without addressing what is retrieved, perform heavy ingestion-time fact extraction at substantial token cost, or rely on heuristic graph traversal that leaves both accuracy and efficiency on the table. We present PRISM, a training-free retrieval-side framework that treats long-horizon memory as a joint retrieval-and-compression problem over a graph-structured memory. PRISM combines four orthogonal inference-time components: Hierarchical Bundle Search over typed relation paths, Query-Sensitive Edge Costing that aligns traversal with detected query intent, Evidence Compression that compresses the candidate bundle into a compact answer-side context, and Adaptive Intent Routing that routes most queries through zero-LLM tiers. By formulating retrieval as min-cost selection over typed path templates and pairing it with an LLM-side compression step, PRISM surfaces the right evidence under a strict context budget without any fine-tuning or modification to the upstream ingestion pipeline. Experiments on the LoCoMo benchmark show that PRISM delivers substantially higher LLM-judge accuracy than every same-protocol baseline at an order-of-magnitude smaller context budget, occupying a previously empty corner of the accuracy-context-cost frontier and demonstrating a superior balance between answer quality and retrieval efficiency.
>
---
#### [replaced 018] Benchmarking Gaslighting Attacks Against Speech Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于安全评估任务，旨在检测语音大语言模型在面对操控性输入时的脆弱性。通过设计五种操控策略，评估模型的鲁棒性及行为响应。**

- **链接: [https://arxiv.org/pdf/2509.19858](https://arxiv.org/pdf/2509.19858)**

> **作者:** Jinyang Wu; Bin Zhu; Xiandong Zou; Qiquan Zhang; Xu Fang; Pan Zhou
>
> **备注:** 5 pages, 2 figures, 3 tables
>
> **摘要:** As Speech Large Language Models (Speech LLMs) become increasingly integrated into voice-based applications, ensuring their robustness against manipulative or adversarial input becomes critical. Although prior work has studied adversarial attacks in text-based LLMs and vision-language models, the unique cognitive and perceptual challenges of speech-based interaction remain underexplored. In contrast, speech presents inherent ambiguity, continuity, and perceptual diversity, which make adversarial attacks more difficult to detect. In this paper, we introduce gaslighting attacks, strategically crafted prompts designed to mislead, override, or distort model reasoning as a means to evaluate the vulnerability of Speech LLMs. Specifically, we construct five manipulation strategies: Anger, Cognitive Disruption, Sarcasm, Implicit, and Professional Negation, designed to test model robustness across varied tasks. It is worth noting that our framework captures both performance degradation and behavioral responses, including unsolicited apologies and refusals, to diagnose different dimensions of susceptibility. Moreover, acoustic perturbation experiments are conducted to assess multi-modal robustness. To quantify model vulnerability, comprehensive evaluation across 5 Speech and multi-modal LLMs on over 10,000 test samples from 5 diverse datasets reveals an average accuracy drop of 24.3% under the five gaslighting attacks, indicating significant behavioral vulnerability. These findings highlight the need for more resilient and trustworthy speech-based AI systems.
>
---
#### [replaced 019] Fine-Tuning Causal LLMs for Text Classification: Embedding-Based vs. Instruction-Based Approaches
- **分类: cs.CL; cs.AI**

- **简介: 论文研究在资源受限下如何微调因果大语言模型进行文本分类，比较了基于嵌入和指令微调的方法，验证了其有效性与效率。**

- **链接: [https://arxiv.org/pdf/2512.12677](https://arxiv.org/pdf/2512.12677)**

> **作者:** Amirhossein Yousefiramandi; Ciaran Cooney
>
> **备注:** 24 pages, 6 figures
>
> **摘要:** We explore efficient strategies to fine-tune decoder-only Large Language Models (LLMs) for downstream text classification under resource constraints. Two approaches are investigated: (1) attaching a classification head to a pre-trained causal LLM and fine-tuning on the task using the LLM's final-token embedding as a sequence representation, and (2) instruction-tuning the LLM in a prompt-to-response format for classification. To enable single-GPU fine-tuning of models up to 8B parameters, we combine 4-bit model quantization with Low-Rank Adaptation (LoRA) for parameter-efficient training. Experiments on two patent benchmarks, a proprietary 5-class single-label corpus and the public WIPO-Alpha multi-label dataset with 14 categories, show that the embedding-based method matches or exceeds the instruction-tuned method on single-label classification while training 10 to 30 times fewer parameters. Instruction-tuning is competitive only in the multi-label regime, and only with substantially larger trainable budgets of at least 100M parameters. Both methods are very competitive with fine-tuned domain-specific BERT models, and on the single-label task they surpass them. Paired McNemar tests and bootstrap Delta F1 95 percent confidence intervals confirm that the numerical advantage of the embedding-head approach is consistent in direction but not statistically certified at p < 0.05. We further validate single-label generalization on AG News and report ablations on pooling, verbalizer choice, and calibration, together with a distillation recipe that recovers BERT-class throughput. We discuss the advantages of each approach while outlining practical guidelines and future directions for optimizing LLM fine-tuning in classification scenarios.
>
---
#### [replaced 020] Syntactically-guided Information Maintenance in Sentence Comprehension
- **分类: cs.CL**

- **简介: 该论文研究语言理解中信息维护的机制，探讨语法结构如何影响记忆成本。任务为语言处理中的信息维护，解决如何高效维持关键信息的问题。通过分析日语阅读时间数据，验证语法结构对维护成本的影响。**

- **链接: [https://arxiv.org/pdf/2604.27468](https://arxiv.org/pdf/2604.27468)**

> **作者:** Shinnosuke Isono; Kohei Kajikawa
>
> **摘要:** Maintaining information in context is essential in successful real-time language comprehension, but maintenance is cognitively costly and can slow processing. We hypothesize that rational language users selectively maintain information that is crucial for future prediction, guided by syntactic structure. Under this view, two factors affect maintenance cost: the number of predicted heads and the number of incomplete dependencies. Although these factors have been treated as competing hypotheses in the literature, our account predicts that they are not reducible to one another. We show this is the case in a naturalistic reading time dataset in Japanese, a language in which the two factors contrast particularly clearly. We further show that there is a tradeoff such that readers that slow down for maintenance tend to benefit more from predictability, providing additional support for the proposed account. These patterns are not evident in English, however, and we highlight some issues to be resolved to understand the contribution of syntax in memory-efficient processing of various languages.
>
---
#### [replaced 021] Stable Behavior, Limited Variation: Persona Validity in LLM Agents for Urban Sentiment Perception
- **分类: cs.CL; cs.SI**

- **简介: 该论文研究LLM代理在城市情感感知中的角色，探讨不同人格设定对情感判断的影响。任务是评估人格设定是否带来有意义的多样性，工作包括实验分析和对比无设定模型表现。**

- **链接: [https://arxiv.org/pdf/2604.28048](https://arxiv.org/pdf/2604.28048)**

> **作者:** Neemias B da Silva; Rodrigo Minetto; Daniel Silver; Thiago H Silva
>
> **备注:** 8 pages, 8 figures. IEEE DCOSS - UrbCom
>
> **摘要:** Large Language Models (LLMs) are increasingly used as proxies for human perception in urban analysis, yet it remains unclear whether persona prompting produces meaningful and reproducible behavioral diversity. We investigate whether distinct personas influence urban sentiment judgments generated by multimodal LLMs. Using a factorial set of personas spanning gender, economic status, political orientation, and personality, we instantiate multiple agents per persona to evaluate urban scene images from the PerceptSent dataset and assess both within-persona consistency and cross-persona variation. Results show strong convergence among agents sharing a persona, indicating stable and reproducible behavior. However, cross-persona differentiation is limited: economic status and personality induce statistically detectable but practically modest variation, while gender shows no measurable effect and political orientation only negligible impact. Agents also exhibit an extremity bias, collapsing intermediate sentiment categories common in human annotations. As a result, performance remains strong on coarse-grained polarity tasks but degrades as sentiment resolution increases, suggesting that simple label-based persona prompting does not capture fine-grained perceptual judgments. To isolate the contribution of persona conditioning, we additionally evaluate the same model without personas. Surprisingly, the no-persona model sometimes matches or exceeds persona-conditioned agreement with human labels across all task variants, suggesting that simple label-based persona prompting may add limited annotation value in this setting.
>
---
#### [replaced 022] HalluScan: A Systematic Benchmark for Detecting and Mitigating Hallucinations in Instruction-Following LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在检测和减轻指令遵循型大模型的幻觉问题。通过构建基准框架HalluScan，评估多种检测方法并提出新指标与算法。**

- **链接: [https://arxiv.org/pdf/2605.02443](https://arxiv.org/pdf/2605.02443)**

> **作者:** Ahmed Cherif
>
> **备注:** 38 pages, 13 figures, 10 tables. Submitted to Neural Computing and Applications
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse natural language processing tasks, yet they remain susceptible to hallucinations -- generating content that is factually incorrect, unfaithful to provided context, or misaligned with user instructions. We present HalluScan, a comprehensive benchmark framework that systematically evaluates hallucination detection and mitigation across 72 configurations spanning 6 detection methods, 4 open-weight model families, and 3 diverse domains. We introduce three key contributions: (1) HalluScore, a novel composite metric that achieves a Pearson correlation of r = 0.41 with human expert judgments; (2) Adaptive Detection Routing (ADR), an intelligent routing algorithm achieving 2.0x cost reduction with only 0.1% AUROC degradation; and (3) systematic error cascade decomposition revealing substantial variation in hallucination error types across domains. Our experiments reveal that NLI Verification achieves the highest overall AUROC of 0.88, while RAV achieves the second-highest AUROC of 0.66.
>
---
#### [replaced 023] LambdaPO: A Lambda Style Policy Optimization for Reasoning Language Models
- **分类: cs.CL**

- **简介: 该论文提出LambdaPO方法，解决强化学习中策略优化的信息瓶颈问题，通过重构优势估计和引入语义密度奖励，提升语言模型在复杂任务中的表现。**

- **链接: [https://arxiv.org/pdf/2605.19416](https://arxiv.org/pdf/2605.19416)**

> **作者:** Zhe Yuan; Yipeng Zhou; Jinghan Li; Xinyuan Chen; Bowen Deng; Zhiqian Chen; Liang Zhao
>
> **备注:** We find that our method's results are obtained with a different data split compared to the baselines. The conclusions are not reliable. We would like to withdraw, so that we could fix this problem
>
> **摘要:** Group Relative Policy Optimization(GRPO) has become a cornerstone of modern reinforcement learning alignment, prized for its efficacy in foregoing an explicit value-critic by leveraging reward normalization across sampled trajectory cohorts. However, the method's reliance on a monolithic statistical baseline, such as the group mean, collapses the relational topology of the trajectory space into a single scalar, thereby erasing the fine-grained preference information essential for navigating complex, rank-sensitive reward landscapes. To address this issue, we introduce a novel framework, Lambda Policy Optimization (LambdaPO), that addresses this information-theoretic bottleneck by re-conceptualizing advantage estimation from a scalar value to a decomposed, pairwise preference structure. Specifically, the advantage for any given trajectory is formulated as the integrated sum of reward differentials against all peers in its cohort, where each pairwise comparison is dynamically attenuated by the policy's own probabilistic confidence in the established preference. To further mitigate the sparsity of binary outcome supervision, we augment the objective with a semantic density reward, derived from the precision-recall alignment between generated reasoning traces and ground-truth solutions. As a result, our method can mine more fine-grained optimization signals from a group of rollouts, guiding the LLM to a better optima. Experimental results across challenging math reasoning and question-answering tasks demonstrates that LambdaPO improves performance compared to the baseline methods.
>
---
#### [replaced 024] GT-HarmBench: Benchmarking AI Safety Risks Through the Lens of Game Theory
- **分类: cs.AI; cs.CL; cs.CY; cs.GT; cs.MA**

- **简介: 该论文属于AI安全研究任务，旨在解决多智能体环境下的安全风险问题。通过构建GT-HarmBench基准，分析模型在高风险场景中的表现并提出改进方法。**

- **链接: [https://arxiv.org/pdf/2602.12316](https://arxiv.org/pdf/2602.12316)**

> **作者:** Pepijn Cobben; Xuanqiang Angelo Huang; Thao Amelia Pham; Isabel Dahlgren; Terry Jingchen Zhang; Zhijing Jin
>
> **摘要:** Frontier AI systems are increasingly capable and deployed in high-stakes multi-agent environments. However, existing AI safety benchmarks largely evaluate single agents, leaving multi-agent risks such as coordination failure and conflict poorly understood. We introduce GT-HarmBench, a benchmark of 1,535 high-stakes scenarios spanning game-theoretic structures such as the Prisoner's Dilemma, Stag Hunt and Chicken. Scenarios are drawn from realistic AI risk contexts in the MIT AI Risk Repository. Across 15 frontier models, agents fail to choose socially beneficial actions in 38% of high-stakes cases, such as military escalation, election manipulation, and medical malpractice. We measure sensitivity to game-theoretic prompt framing and ordering, and analyze reasoning patterns driving failures. We further show that game-theoretic interventions improve socially beneficial outcomes by up to 18%. Our results highlight substantial reliability gaps and provide a broad standardized testbed for studying alignment in multi-agent environments. The benchmark and code are available at this https URL.
>
---
#### [replaced 025] Vector Retrieval with Similarity and Diversity: How Hard Is It?
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决同时兼顾相似性与多样性的向量检索问题。提出VRSD模型，并设计无参数算法，实验证明优于现有方法。**

- **链接: [https://arxiv.org/pdf/2407.04573](https://arxiv.org/pdf/2407.04573)**

> **作者:** Hang Gao; Dong Deng; Yongfeng Zhang
>
> **摘要:** Dense vector retrieval is an important building block of modern machine learning systems, underlying applications ranging from semantic search to retrieval-augmented generation and knowledge-intensive reasoning. Beyond retrieving items that are individually similar to a query, many applications require a set of results that is also diverse, complementary, and collectively informative. Balancing similarity and diversity is therefore central to effective retrieval, but remains challenging to optimize in a stable and theoretically grounded way. Maximal Marginal Relevance (MMR) is a widely adopted heuristic for this problem, yet its reliance on a manually tuned parameter leads to optimization fluctuations and unpredictable retrieval results. More broadly, existing methods provide limited theoretical insight into how similarity and diversity interact in dense vector spaces, leaving the joint optimization problem insufficiently understood. To address these challenges, this paper introduces a novel approach that characterizes both constraints simultaneously by maximizing the similarity between the query vector and the sum of the selected candidate vectors. We formally define this optimization problem, Vector Retrieval with Similarity and Diversity (VRSD), and prove that it is NP-complete, establishing a rigorous theoretical bound on the inherent difficulty of this dual-objective retrieval. Subsequently, we present a parameter-free heuristic algorithm to solve VRSD. Extensive evaluations on multiple datasets, incorporating both objective geometric metrics and LLM-simulated subjective assessments, demonstrate that our VRSD heuristic consistently outperforms established baselines, including MMR and Determinantal Point Processes (k-DPP).
>
---
#### [replaced 026] PROGRESSLM: Towards Progress Reasoning in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言模型的进度推理任务，旨在解决从部分观察中推断任务进展的问题。研究提出Progress-Bench基准和ProgressLM数据集，探索两种推理方法，发现现有模型在此任务上表现不佳。**

- **链接: [https://arxiv.org/pdf/2601.15224](https://arxiv.org/pdf/2601.15224)**

> **作者:** Jianshu Zhang; Chengxuan Qian; Haosen Sun; Haoran Lu; Dingcheng Wang; Letian Xue; Han Liu
>
> **备注:** ACL 2026 Camera Ready Version
>
> **摘要:** Estimating task progress requires reasoning over long-horizon dynamics rather than recognizing static visual content. While modern Vision-Language Models (VLMs) excel at describing what is visible, it remains unclear whether they can infer how far a task has progressed from partial observations. To this end, we introduce Progress-Bench, a benchmark for systematically evaluating progress reasoning in VLMs. Beyond benchmarking, we further explore a human-inspired two-stage progress reasoning paradigm through both training-free prompting and training-based approach based on curated dataset ProgressLM-45K. Experiments on 14 VLMs show that most models are not yet ready for task progress estimation, exhibiting sensitivity to demonstration modality and viewpoint changes, as well as poor handling of unanswerable cases. While training-free prompting that enforces structured progress reasoning yields limited and model-dependent gains, the training-based ProgressLM-3B achieves consistent improvements even at a small model scale, despite being trained on a task set fully disjoint from the evaluation tasks. Further analyses reveal characteristic error patterns and clarify when and why progress reasoning succeeds or fails. Website: this https URL
>
---
#### [replaced 027] Training-Free Multimodal Large Language Model Orchestration
- **分类: cs.CL**

- **简介: 该论文属于多模态系统任务，解决传统方法成本高、扩展性差的问题。提出无需训练的LLM编排框架，整合模态专家，提升效率与灵活性。**

- **链接: [https://arxiv.org/pdf/2508.10016](https://arxiv.org/pdf/2508.10016)**

> **作者:** Tianyu Xie; Yuexiao Ma; Yuhang Wu; Wang Chen; Jiayi Ji; Tat-Seng Chua; Xiawu Zheng; Rongrong Ji
>
> **摘要:** Building interactive omni-modal assistants often relies on end-to-end multimodal alignment to fuse heterogeneous modalities, which incurs substantial data and compute costs and limits extensibility. We present Training-Free Large Language Model Orchestration (LLM Orchestration), a training-free orchestration framework that integrates off-the-shelf modality experts into a unified multimodal input--output system without additional gradient-based training for integration. LLM Orchestration comprises three components: (1) an LLM controller that infers user intent and emits explicit control tokens for expert selection and sequencing, enabling protocol-constrained and auditable routing; (2) a text-centric cross-modal memory that compresses multimodal evidence into structured records for lightweight retrieval and reuse, reducing redundant expert invocations across turns; and (3) a unified interaction layer that executes routing and memory decisions to support consistent modality transitions, full-duplex streaming, and interruption-aware dialogue. Across diverse multimodal benchmarks, LLM Orchestration achieves strong performance under standard evaluation constraints while maintaining low orchestration overhead and modular upgradeability, providing a practical alternative to costly joint training for omni-modal systems.
>
---
#### [replaced 028] Tabular PDF Information Extraction with Local LLMs and Layout-Aware Parsing: A Reliability Evaluation
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于信息提取任务，解决从学术PDF中可靠提取结构化数据的问题。通过比较不同方法，评估其在受限计算环境下的效果与效率。**

- **链接: [https://arxiv.org/pdf/2604.00003](https://arxiv.org/pdf/2604.00003)**

> **作者:** Muhammad Anis Al Hilmi; Neelansh Khare; Noel Framil Iglesias; Kurnia Adi Cahyanto; Azhar Al Afghani; Musfi Yuliadi
>
> **备注:** 9 pages, 5 figures, 3 tables
>
> **摘要:** Extracting structured information from academic PDF documents is non trivial: a single page typically combines free text metadata with tabular regions, exhibits cross program variation, and is susceptible to Unicode encoding artifacts that interfere with downstream parsing. This study evaluates the reliability of information extraction approaches for tabular PDF documents, using academic course registration documents (Kartu Rencana Studi or KRS) from Indonesian higher education as a case study. Three strategies are compared: LLM only, Hybrid Deterministic - LLM (regex & LLM), and a Camelot based pipeline with LLM fallback. Experiments were conducted on 140 documents for the LLM based test and 860 documents for the Camelot based pipeline evaluation, covering four study programs with varying data in tables and metadata. Three 12 - 14B LLM models (Gemma 3, Phi 4, and Qwen 2.5) were run locally using Ollama and a consumer grade CPU without a GPU. Evaluations used exact match (EM) and Levenshtein similarity (LS) metrics with a threshold of 0.7. Although not applicable to all models, the results show that the hybrid approach can improve efficiency compared to LLM only, especially for deterministic metadata. The Camelot based pipeline with LLM fallback produced the best combination of accuracy (EM and LS up to 0.99 - 1.00) and computational efficiency (less than 1 second per PDF in most cases). The Qwen 2.5:14b model demonstrated the most consistent performance across all scenarios. These findings confirm that integrating deterministic and LLM based methods is a reliable and efficient strategy for information extraction from tabular text based PDF documents in computationally constrained environments.
>
---
#### [replaced 029] Towards Generalization of Block Attention via Automatic Segmentation and Block Distillation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决块注意力机制在长文本中的应用问题。通过自动分块和块蒸馏方法，提升块注意力的性能与效率。**

- **链接: [https://arxiv.org/pdf/2605.15913](https://arxiv.org/pdf/2605.15913)**

> **作者:** Shuaiyi Li; Zhisong Zhang; Yan Wang; Lei Zhu; Dongyang Ma; Chenlong Deng; Yang Deng; Wai Lam
>
> **备注:** 16 pages, 2 figures
>
> **摘要:** Block attention, which processes the input as separate blocks that cannot attend to one another, offers significant potential to improve KV cache reuse in long-context scenarios such as Retrieval-Augmented Generation (RAG). However, its broader application is hindered by two key challenges: the difficulty of segmenting input text into meaningful, self-contained blocks, and the inefficiency of existing block fine-tuning methods that risk degrading performance. To address these, we first construct SemanticSeg, a large and diverse semantic segmentation dataset containing over 30k instances across 16 categories-including books, code, web text, and conversations with text lengths ranging from 2k to 32k. Using this dataset, we train a lightweight segmenter to automatically partition text into human-instinct-aligned blocks with controllable granularity. Second, we propose block distillation, a training framework that is more efficient than block fine-tuning, which uses a frozen full-attention teacher model to guide the block-attention student. This framework integrates three novel components: block sink tokens to mitigate information loss at block boundaries, block dropout to leverage training signals from all blocks, and token-level loss weighting to focus learning on block-attention-sensitive tokens. Experiments across multiple models and benchmarks demonstrate that our segmenter outperforms heuristic and statistical baselines, and block distillation achieves near-full-attention performance under block attention, establishing a practical and scalable pathway for deploying block attention.
>
---
#### [replaced 030] Speak-to-Structure: Evaluating LLMs in Open-domain Natural Language-Driven Molecule Generation
- **分类: cs.CL**

- **简介: 该论文属于自然语言驱动分子生成任务，旨在解决现有数据集无法评估模型创造力的问题。提出S^2-Bench基准，包含三个生成任务，并引入大规模数据集提升模型性能。**

- **链接: [https://arxiv.org/pdf/2412.14642](https://arxiv.org/pdf/2412.14642)**

> **作者:** Jiatong Li; Junxian Li; Weida Wang; Yunqing Liu; Changmeng Zheng; Yatao Bian; Dongzhan Zhou; Xiao-yong Wei; Qing Li
>
> **备注:** Accepted by KDD 2026. Our codes and datasets are fully accessible through the this https URL and this https URL
>
> **摘要:** Recently, Large Language Models (LLMs) have demonstrated great potential in natural language-driven molecule discovery. However, existing datasets and benchmarks for molecule-text alignment are predominantly built on one-to-one mappings, measuring LLMs' ability to retrieve a single, pre-defined answer, rather than their creative potential to generate diverse, yet equally valid, molecular candidates. To address this critical gap, we propose Speak-to-Structure (S^2-Bench), the first benchmark to evaluate LLMs in open-domain natural language-driven molecule generation. S^2-Bench is specifically designed for one-to-many relationships, challenging LLMs to exhibit genuine molecular understanding and open-ended generation capabilities. Our benchmark includes three key tasks: molecule editing (MolEdit), molecule optimization (MolOpt), and customized molecule generation (MolCustom), each probing a different aspect of molecule discovery. We also introduce OpenMolIns, a large-scale instruction tuning dataset that enables Llama3.1-8B to surpass the most powerful LLMs like GPT-4o and Claude-3.5 on S^2-Bench. Our comprehensive evaluation of 31 LLMs shifts the focus from simple pattern recall to realistic molecular design, paving the way for more capable LLMs in natural language-driven molecule discovery. Our codes and datasets are fully accessible through the Github Repository: this https URL and Huggingface Datasets: this https URL.
>
---
#### [replaced 031] GradingAttack: Exposing Security Vulnerabilities in LLM Based Educational Grading Agents
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于教育AI安全任务，旨在解决LLM评分系统易受攻击的问题。提出GradingAttack框架，通过令牌和提示级攻击测试系统漏洞，揭示现有系统的安全性不足。**

- **链接: [https://arxiv.org/pdf/2602.00979](https://arxiv.org/pdf/2602.00979)**

> **作者:** Xueyi Li; Zhuoneng Zhou; Zitao Liu; Yongdong Wu
>
> **摘要:** Large language models (LLMs) are increasingly deployed as educational agents for automatic short answer grading (ASAG) in real-world educational environments, significantly boosting assessment efficiency and scalability. However, when these grading agents operate ``in the wild'', their vulnerability to adversarial manipulation raises critical concerns about agent security and trustworthiness. In this paper, we introduce GradingAttack, a fine-grained adversarial attack framework that systematically evaluates the security vulnerabilities of LLM based educational grading agents. Specifically, we design token-level and prompt-level attack strategies that manipulate agent grading outcomes while maintaining high stealth, exposing fundamental weaknesses in current agent deployments. Experiments on multiple datasets demonstrate that both attack strategies effectively compromise grading agents, with prompt-level attacks achieving higher success rates and token-level attacks exhibiting superior stealth capability. Our findings reveal that current LLM based educational agents lack robust defenses against adversarial attacks, underscoring the urgent need for developing secure and trustworthy agent systems for critical educational applications.
>
---
#### [replaced 032] Boiling the Frog: A Multi-Turn Benchmark for Agentic Safety
- **分类: cs.CL**

- **简介: 该论文属于AI安全任务，旨在评估工具使用型AI在办公环境中的安全性。通过构建多轮基准测试，检测模型对渐进式攻击的敏感性。**

- **链接: [https://arxiv.org/pdf/2605.22643](https://arxiv.org/pdf/2605.22643)**

> **作者:** Piercosma Bisconti; Matteo Prandi; Federico Pierucci; Federico Sartore; Enrico Panai; Laura Caroli; Yue Zhu; Adam Leon Smith; Luca Nannini; Marcello Galisai; Susanna Cifani; Francesco Giarrusso; Marcantonio Bracale Syrnikov; Daniele Nardi
>
> **摘要:** Background. Traditional safety benchmarks for language models evaluate generated text: whether a model outputs toxic language, reproduces bias, or follows harmful instructions. When models are deployed as agents, the safety-relevant object shifts from what the system says to what it does within an environment, and evaluating model responses under prompting is no longer sufficient to address the safety challenges posed by artificial intelligence. Recent developments have seen the rise of benchmarks that evaluate large language models as agents. We contribute to this strand of research. Approach. We introduce Boiling the Frog, a benchmark that evaluates whether tool-using AI models deployed in corporate and office settings are susceptible to incremental attacks. Each scenario begins with benign workspace edits and later introduces a risk-bearing request. The benchmark focuses on stateful multi-turn evaluation: chains expose a persistent workspace, place the risk-bearing payload at controlled positions in the turn sequence, and score whether the resulting artifact state becomes unsafe. Scenarios are organized through a three-level operational risk taxonomy grounded in the Boiling the Frog risks, the AI Act Annex I and Annex III high-risk contexts, and EU AI Act's Code of Practice on General-Purpose AI (GPAI). Results. Across a nine-model panel, aggregate strict attack success rate (ASR) is 44.4%. Model-level ASR ranges from 20.5% for Claude Haiku 4.5 to 92.9% for Gemini 3.1 Flash Lite, with Seed 2.0 Lite also above 80%. Average chain category-level ASR reaches 93.3% for Code of Practice loss-of-control scenarios.
>
---
#### [replaced 033] Boundary-targeted Membership Inference Attacks on Safety Classifiers
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于隐私安全任务，研究如何通过边界样本推断安全分类器的训练数据。工作包括提出新的选择策略，提升成员推理攻击效果，并验证内容过滤无效、噪声策略有效。**

- **链接: [https://arxiv.org/pdf/2605.22373](https://arxiv.org/pdf/2605.22373)**

> **作者:** Anthony Hughes; Alexander Goldberg; Prince Jha; Adam Perer; Nikolaos Aletras; Niloofar Mireshghallah
>
> **摘要:** Safety classifiers are essential safeguards within generative AI systems, filtering harmful content or identifying at-risk users when interacting with large language models. Despite their necessity, these models are trained on sensitive datasets including discussions of self-harm and mental health, raising important, yet poorly understood, privacy concerns. Membership inference attacks (MIAs) allow adversaries to infer membership of examples used to train models. In this work, we hypothesize that identifying the examples on which the classifier is least confident are informative for an adversary to infer membership. This reflects a localized failure of generalization, where the model relies on memorization to resolve ambiguity in the training set. To investigate this, we introduce a new boundary-targeted selection strategy that identifies low confidence examples that amplify the signal of an examples membership within a training set. Our experimental results show that an adversary can recover 19% of the conversations a safety classifier flagged as indicating user distress, at a 5% false-positive rate, on a classifier fine-tuned for detecting a user who may require emotional support. This is $3.5$ times more than attacking using state-of-the-art MIA methods alone. Finally, we characterize the boundary laying examples and show that content-based filtering is ineffective for protection, and existing noise strategies can effectively mitigate susceptibility of these examples.
>
---
#### [replaced 034] BURMESE-SAN: Burmese NLP Benchmark for Evaluating Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出BURMESE-SAN基准，用于评估大语言模型在缅甸语的理解、推理和生成能力。解决低资源语言模型评估问题，涵盖多个NLP任务。**

- **链接: [https://arxiv.org/pdf/2602.18788](https://arxiv.org/pdf/2602.18788)**

> **作者:** Thura Aung; Jann Railey Montalan; Jian Gang Ngui; Peerat Limkonchotiwat
>
> **摘要:** We introduce BURMESE-SAN, the first holistic benchmark that systematically evaluates large language models (LLMs) for Burmese across three core NLP competencies: understanding (NLU), reasoning (NLR), and generation (NLG). BURMESE-SAN consolidates seven subtasks spanning these competencies, including Question Answering, Sentiment Analysis, Toxicity Detection, Causal Reasoning, Natural Language Inference, Abstractive Summarization, and Machine Translation, several of which were previously unavailable for Burmese. The benchmark is constructed through a rigorous native-speaker-driven process to ensure linguistic naturalness, fluency, and cultural authenticity while minimizing translation-induced artifacts. We conduct a large-scale evaluation of both open-weight and commercial LLMs to examine challenges in Burmese modeling arising from limited pretraining coverage, rich morphology, and syntactic variation. Our results show that Burmese performance depends more on architectural design, language representation, and instruction tuning than on model scale alone. In particular, Southeast Asia regional fine-tuning and newer model generations yield substantial gains. Finally, we release BURMESE-SAN as a public leaderboard to support systematic evaluation and sustained progress in Burmese and other low-resource languages. this https URL
>
---
#### [replaced 035] The Double Dilemma in Multi-Task Radiology Report Generation: A Gradient Dynamics Analysis and Solution
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于多任务放射报告生成任务，解决线性标量化策略在平衡临床监督与报告平滑性上的不足。提出CAME-Grad优化器，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.22635](https://arxiv.org/pdf/2605.22635)**

> **作者:** Erjian Zhang; Yatong Hao; Liejun Wang; Zhiqing Guo
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** While multi-task learning based automatic radiology report generation (RRG) is widely adopted to ensure clinical consistency, most focus on architectural designs yet remain limited to coarse linear scalarization strategies. These strategies cannot effectively balance the hard constraints of discriminative clinical supervision with the smoothness requirements of report generation. To address these problems, we analyze the failure mechanism of linear scalarization from the perspective of gradient dynamics, utilizing the stochastic differential equation (SDE) framework to characterize it as a "Double Dilemma" of drift term deviation and diffusion term decay. Based on this, we propose a backbone-agnostic optimizer named Conflict-Averse Magnitude-Enhanced Gradient Descent (CAME-Grad). Through conflict-averse direction rectification and magnitude-enhanced energy injection, the algorithm not only ensures geometric validity, but also avoids local optimal solutions. Then, the adaptive gradient fusion mechanism is used to establish a dynamic balance between the theoretical optimal direction and the task-specific inductive bias. Experiments show that as a universal plug-and-play optimizer, CAME-Grad brings substantial and consistent improvements across eight diverse RRG methods, elevating overall clinical efficacy performance by an average of 2.3% on MIMIC-CXR and 1.9% on IU X-Ray. Our code is available at this https URL.
>
---
#### [replaced 036] SciHorizon-GENE: Benchmarking LLM for Life Sciences Inference from Gene Knowledge to Functional Understanding
- **分类: q-bio.GN; cs.AI; cs.CL**

- **简介: 该论文属于生物信息学任务，旨在解决LLM在基因到功能推理中的可靠性问题。构建了SciHorizon-GENE基准，评估模型在四个关键方面的表现。**

- **链接: [https://arxiv.org/pdf/2601.12805](https://arxiv.org/pdf/2601.12805)**

> **作者:** Xiaohan Huang; Meng Xiao; Chuan Qin; Qingqing Long; Jinmiao Chen; Yuanchun Zhou; Hengshu Zhu
>
> **备注:** Accepted by SIGKDD 2026. 12 pages
>
> **摘要:** Large language models (LLMs) have shown growing promise in biomedical research, particularly for knowledge-driven interpretation tasks. However, their ability to reliably reason from gene-level knowledge to functional understanding, a core requirement for knowledge-enhanced cell atlas interpretation, remains largely underexplored. To address this gap, we introduce SciHorizon-GENE, a large-scale gene-centric benchmark constructed from authoritative biological databases. The benchmark integrates curated knowledge for over 190K human genes and comprises more than 540K questions covering diverse gene-to-function reasoning scenarios relevant to cell type annotation, functional interpretation, and mechanism-oriented analysis. Motivated by behavioral patterns observed in preliminary examinations, SciHorizon-GENE evaluates LLMs along four biologically critical perspectives: research attention sensitivity, hallucination tendency, answer completeness, and literature influence, explicitly targeting failure modes that limit the safe adoption of LLMs in biological interpretation pipelines. We systematically evaluate a wide range of state-of-the-art general-purpose and biomedical LLMs, revealing substantial heterogeneity in gene-level reasoning capabilities and persistent challenges in generating faithful, complete, and literature-grounded functional interpretations. Our benchmark establishes a systematic foundation for analyzing LLM behavior at the gene scale and offers insights for model selection and development, with direct relevance to knowledge-enhanced biological interpretation.
>
---
#### [replaced 037] MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出MAS-Orchestra框架，解决多智能体系统设计效率低和效果不确定的问题。通过函数调用强化学习实现整体协调，并引入MASBENCH基准评估任务特性，提升多智能体系统性能。**

- **链接: [https://arxiv.org/pdf/2601.14652](https://arxiv.org/pdf/2601.14652)**

> **作者:** Zixuan Ke; Yifei Ming; Austin Xu; Ryan Chin; Xuan-Phi Nguyen; Prathyusha Jwalapuram; Jiayu Wang; Semih Yavuz; Caiming Xiong; Shafiq Joty
>
> **备注:** ICML 2026
>
> **摘要:** While multi-agent systems (MAS) promise elevated intelligence through coordination of agents, current approaches to automatic MAS design under-deliver. Such shortcomings stem from two key factors: (1) methodological complexity - agent orchestration is performed using sequential, code-level execution that limits global system-level holistic reasoning and scales poorly with agent complexity - and (2) efficacy uncertainty - MAS are deployed without understanding if there are tangible benefits compared to single-agent systems (SAS). We propose MASOrchestra, a training-time framework that formulates MAS orchestration as a function-calling reinforcement learning problem with holistic orchestration, generating an entire MAS at once. In MAS-Orchestra, complex, goal-oriented subagents are abstracted as callable functions, enabling global reasoning over system structure while hiding internal execution details. To rigorously study when and why MAS are beneficial, we introduce MASBENCH, a controlled benchmark that characterizes tasks along five axes: Depth, Horizon, Breadth, Parallel, and Robustness. Our analysis reveals that MAS gains depend critically on task structure, verification protocols, and the capabilities of both orchestrator and subagents, rather than holding universally. Guided by these insights, MAS-Orchestra achieves consistent improvements on public benchmarks including mathematical reasoning, multi-hop QA, and search-based QA, while achieving more than 10x efficiency over strong baselines. Together, MAS-Orchestra and MASBENCH enable better training and understanding of MAS in the pursuit of multi-agent intelligence.
>
---
#### [replaced 038] How Far Are We from Generating Missing Modalities with Foundation Models?
- **分类: cs.MM; cs.CL; cs.CV**

- **简介: 该论文属于多模态生成任务，旨在解决缺失模态重建问题。通过分析现有模型的不足，提出一种新的框架以提升生成质量。**

- **链接: [https://arxiv.org/pdf/2506.03530](https://arxiv.org/pdf/2506.03530)**

> **作者:** Guanzhou Ke; Bo Wang; Guoqing Chao; Weiming Hu; Shengfeng He
>
> **备注:** T-PAMI
>
> **摘要:** Multimodal foundation models have demonstrated impressive capabilities across diverse tasks. However, their potential as plug-and-play solutions for missing modality reconstruction remains underexplored. To bridge this gap, we identify and formalize three potential paradigms for missing modality reconstruction, and perform a comprehensive evaluation across these paradigms, covering 42 model variants in terms of reconstruction accuracy and adaptability to downstream tasks. Our analysis reveals that current foundation models often fall short in two critical aspects: (i) fine-grained semantic extraction from the available modalities, and (ii) robust validation of generated modalities. These limitations lead to suboptimal and, at times, misaligned generations. To address these challenges, we propose an agentic framework tailored for missing modality reconstruction. This framework dynamically formulates modality-aware mining strategies based on the input context, facilitating the extraction of richer and more discriminative semantic features. In addition, we introduce a self-refinement mechanism, which iteratively verifies and enhances the quality of generated modalities through internal feedback. Experimental results show that our method reduces FID for missing image reconstruction by at least 14\% and MER for missing text reconstruction by at least 10\% compared to baselines. Code are released at: this https URL.
>
---
#### [replaced 039] FINESSE-Bench: A Hierarchical Benchmark Suite for Financial Domain Knowledge and Technical Analysis in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出FINESSE-Bench，一个用于评估大语言模型在金融领域知识和技术分析能力的分层基准套件，解决现有基准缺乏专业难度层次和全面评估的问题。**

- **链接: [https://arxiv.org/pdf/2605.15482](https://arxiv.org/pdf/2605.15482)**

> **作者:** Dmitry Stanishevskii; Nini Kamkia; Alexey Khoroshilov; Dmitry Zmitrovich; Denis Kokosinskii; Zhirayr Hayrapetyan; Andrei Kalmykov
>
> **备注:** 21 pages, 10 tables, 2 figures
>
> **摘要:** Large language models (LLMs) are increasingly being applied to financial analysis, reporting, investment decision support, risk management, compliance, and professional training. However, robust evaluation of their domain competence in finance remains incomplete. Widely used open benchmarks such as FinQA, ConvFinQA, and TAT-QA have played an important role in advancing financial question answering and numerical reasoning, but they focus primarily on question answering over financial reports and do not provide an explicit hierarchy of professional difficulty. Broader resources, including FinanceBench, PIXIU, FinBen, and FLaME, expand the coverage of financial tasks, yet the problem of evaluating the transition from foundational knowledge to expert-level financial reasoning remains open. In this work, we present FINESSE-Bench, a suite of eight specialized benchmarks comprising 3,993 questions for hierarchical evaluation of financial competencies in LLMs. FINESSE-Bench combines exam-oriented datasets inspired by professional certifications (CFA-like Levels 1-3, CMT-like Level 2, and CFTe-like Level 1), applied trading task collections, and a Russian-language olympiad benchmark. This design enables evaluation of domain breadth, performance degradation as difficulty increases, the ability to solve computational tasks, and model behavior in specialized financial domains. We also describe a unified evaluation protocol covering multiple-choice questions, numerical answers, and short open-ended responses, together with an automated scoring scheme for freeform answers based on the LLM-as-judge paradigm. FINESSE-Bench is intended both as a complement to existing open financial benchmarks and as a tool for more substantive evaluation of professionally relevant financial competencies in large language models.
>
---
#### [replaced 040] Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的模型微调任务，旨在解决SFT泛化能力不足的问题。通过研究基于概率的目标函数，发现不同目标在模型能力连续体上的表现差异，提出适应模型能力的优化策略。**

- **链接: [https://arxiv.org/pdf/2510.00526](https://arxiv.org/pdf/2510.00526)**

> **作者:** Gaotang Li; Ruizhong Qiu; Xiusi Chen; Heng Ji; Hanghang Tong
>
> **备注:** ICML 2026
>
> **摘要:** Supervised fine-tuning (SFT) is the standard approach for post-training large language models (LLMs), yet it often shows limited generalization. We trace this limitation to its default training objective: negative log likelihood (NLL). While NLL is classically optimal when training from scratch, post-training operates in a different paradigm and could violate its optimality assumptions, where models already encode task-relevant priors and supervision can be long and noisy. In this work, we systematically study various probability-based objectives and characterize when and why different objectives succeed or fail under varying conditions. Through comprehensive experiments and extensive ablation studies across 8 model backbones, 27 benchmarks, and 7 domains, we uncover a critical dimension that governs objective behavior: the model-capability continuum. Near the model-strong end, prior-leaning objectives that downweight low-probability tokens (e.g., $-p$, $-p^{10}$, thresholded variants) consistently outperform NLL; toward the model-weak end, NLL dominates; in between, no single objective prevails. Our theoretical analysis further elucidates how objectives trade places across the continuum, providing a principled foundation for adapting objectives to model capability. The code is available at this https URL.
>
---
#### [replaced 041] SemEval-2026 Task 6: CLARITY -- Unmasking Political Question Evasions
- **分类: cs.CL**

- **简介: 该论文介绍SemEval-2026 Task 6 CLARITY任务，旨在识别政治问答中的回避策略。通过两个子任务，分析回答的清晰度和回避类型，推动计算话语分析发展。**

- **链接: [https://arxiv.org/pdf/2603.14027](https://arxiv.org/pdf/2603.14027)**

> **作者:** Konstantinos Thomas; Giorgos Filandrianos; Maria Lymperaiou; Chrysoula Zerva; Giorgos Stamou
>
> **备注:** SemEval 2026 (Task organizers)
>
> **摘要:** Political speakers often avoid answering questions directly while maintaining the appearance of responsiveness. Despite its importance for public discourse, such strategic evasion remains underexplored in Natural Language Processing. We introduce SemEval-2026 Task 6, CLARITY, a shared task on political question evasion consisting of two subtasks: (i) clarity-level classification into Clear Reply, Ambivalent, and Clear Non-Reply, and (ii) evasion-level classification into nine fine-grained evasion strategies. The benchmark is constructed from U.S. presidential interviews and follows an expert-grounded taxonomy of response clarity and evasion. The task attracted 124 registered teams, who submitted 946 valid runs for clarity-level classification and 539 for evasion-level classification. Results show a substantial gap in difficulty between the two subtasks: the best system achieved 0.89 macro-F1 on clarity classification, surpassing the strongest baseline by a large margin, while the top evasion-level system reached 0.68 macro-F1, matching the best baseline. Overall, large language model prompting and hierarchical exploitation of the taxonomy emerged as the most effective strategies, with top systems consistently outperforming those that treated the two subtasks independently. CLARITY establishes political response evasion as a challenging benchmark for computational discourse analysis and highlights the difficulty of modeling strategic ambiguity in political language.
>
---
#### [replaced 042] RoIt-XMASA: Multi-Domain Multilingual Sentiment Analysis Dataset for Romanian and Italian
- **分类: cs.CL**

- **简介: 该论文提出RoIt-XMASA数据集，解决跨语言和跨领域情感分析问题，通过多目标对抗训练提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.17134](https://arxiv.org/pdf/2604.17134)**

> **作者:** Andrei-Marius Avram; Aureliu Valentin Antonie; Cosmin-Mircea Croitoru; Vlad Andrei Muntean; Dumitru-Clementin Cercel
>
> **备注:** Accepted at the International AAAI Conference on Web and Social Media (ICWSM 2026)
>
> **摘要:** We present RoIt-XMASA, a multilingual dataset that extends the Cross-lingual Multi-domain Amazon Sentiment Analysis to Italian and Romanian, comprising 36,000 labeled reviews across three domains (books, movies, and music) and 202,141 unlabeled samples. To address cross-lingual and cross-domain challenges, we propose a multi-target adversarial training framework that employs loss reversal with meta-learned coefficients to dynamically balance sentiment discrimination with domain and language invariance. XLM-R achieves an F1-score of 66.23% with our approach, outperforming the baseline by 4.64%. Few-shot evaluation shows that Llama-3.1-8B achieves 58.43% F1-score, revealing a meaningful trade-off between the efficiency of prompting-based approaches and the higher performance of task-specific fine-tuning.
>
---
#### [replaced 043] TingIS: Real-time Risk Event Discovery from Noisy Customer Incidents at Enterprise Scale
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出TingIS系统，用于实时发现企业级客户事件中的风险。解决高噪声、高吞吐下的事件关联与降噪问题，通过多阶段事件链接和降噪管道实现高效风险识别。**

- **链接: [https://arxiv.org/pdf/2604.21889](https://arxiv.org/pdf/2604.21889)**

> **作者:** Jun Wang; Ziyin Zhang; Rui Wang; Hang Yu; Peng Di; Rui Wang
>
> **备注:** Accepted to ACL 2026 Industry Track (oral presentation)
>
> **摘要:** Real-time detection and mitigation of technical anomalies are critical for large-scale cloud-native services, where even minutes of downtime can result in massive financial losses and diminished user trust. While customer incidents serve as a vital signal for discovering risks missed by monitoring, extracting actionable intelligence from this data remains challenging due to extreme noise, high throughput, and semantic complexity of diverse business lines. In this paper, we present TingIS, an end-to-end system designed for enterprise-grade incident discovery. At the core of TingIS is a multi-stage event linking engine that synergizes efficient indexing techniques with Large Language Models (LLMs) to make informed decisions on event merging, enabling the stable extraction of actionable incidents from just a handful of diverse user descriptions. This engine is complemented by a cascaded routing mechanism for precise business attribution and a multi-dimensional noise reduction pipeline that integrates domain knowledge, statistical patterns, and behavioral filtering. Deployed in a production environment handling a peak throughput of over 2,000 messages per minute and 300,000 messages per day, TingIS achieves a P90 alert latency of 3.5 minutes and a 95\% discovery rate for high-priority incidents. Benchmarks constructed from real-world data demonstrate that TingIS significantly outperforms baseline methods in routing accuracy, clustering quality, and Signal-to-Noise Ratio.
>
---
#### [replaced 044] Freeze Deep, Train Shallow: Interpretable Layer Allocation for Continued Pre-Training
- **分类: cs.CL**

- **简介: 该论文属于持续预训练任务，解决如何选择性更新模型层以降低成本的问题。提出LayerTracer框架，指导冻结深层、训练浅层，提升效果。**

- **链接: [https://arxiv.org/pdf/2605.11416](https://arxiv.org/pdf/2605.11416)**

> **作者:** Yu-Hang Wu; Qin-Yuan Liu; Qiu-Yang Zhao; Bo Jiang; Jiang-Feng Yang; Qing-Wei Cong
>
> **摘要:** Selective layer-wise updates are essential for low-cost continued pre-training of Large Language Models (LLMs), yet determining which layers to freeze or train remains an empirical black-box problem due to the lack of interpretable guidance. To address this issue, we propose LayerTracer, an architecture-agnostic diagnostic framework that reveals the evolution patterns of layer-wise representations and stability by locating task execution positions and quantifying layer sensitivity. Analysis results reveal that deep layers act as critical regions for task execution and maintain high stability against disruptive updates. Guided by this finding, we conduct three controlled continued pre-training trials to compare diverse freeze-train strategies, demonstrating that training shallow layers while freezing deep layers consistently outperforms full-parameter fine-tuning and the opposite allocation on both C-Eval and CMMLU benchmarks. We further present a hybrid model case study, which validates that placing high-quality pre-trained modules in deep layers effectively preserves inherent knowledge of the model. This work delivers a low-cost and interpretable solution for resource-constrained teams, offering actionable guidance for layer-wise parameter allocation in continued pre-training and hybrid model construction.
>
---
#### [replaced 045] TurkicNLP: An NLP Toolkit for Turkic Languages
- **分类: cs.CL**

- **简介: 该论文提出TurkicNLP，一个针对突厥语族的自然语言处理工具包，解决多语言、多文字系统资源分散的问题，提供统一的NLP处理流程。**

- **链接: [https://arxiv.org/pdf/2602.19174](https://arxiv.org/pdf/2602.19174)**

> **作者:** Sherzod Hakimov
>
> **备注:** The toolkit is available here: this https URL
>
> **摘要:** Natural language processing for the Turkic language family, spoken by over 200 million people across Eurasia, remains fragmented, with most languages lacking unified tooling and resources. We present TurkicNLP, an open-source Python library providing a single, consistent NLP pipeline for Turkic languages across four script families: Latin, Cyrillic, Perso-Arabic, and Old Turkic Runic. The library covers tokenization, morphological analysis, part-of-speech tagging, dependency parsing, named entity recognition, bidirectional script transliteration, cross-lingual sentence embeddings, and machine translation through one language-agnostic API. A modular multi-backend architecture integrates rule-based finite-state transducers and neural models transparently, with automatic script detection and routing between script variants. Outputs follow the CoNLL-U standard for full interoperability and extension. Code and documentation are hosted at this https URL .
>
---
#### [replaced 046] Evaluating Customized vs. Generalist Transformer-based Models for Legal Contract Classification
- **分类: cs.CL**

- **简介: 该论文研究法律合同分类任务，比较定制化与通用Transformer模型的效果。旨在解决通用模型在法律领域表现不足的问题，通过实验验证定制模型的优势。**

- **链接: [https://arxiv.org/pdf/2508.07849](https://arxiv.org/pdf/2508.07849)**

> **作者:** Amrita Singh; H. Suhan Karaca; Aditya Joshi; Hye-young Paik; Jiaojiao Jiang
>
> **备注:** Accepted to Customizable NLP at ACL 2026
>
> **摘要:** Despite advances in legal NLP, no comprehensive evaluation of Transformer-based models customized for legal tasks (referred to as `legal-specific' models in this paper) exists for contract classification tasks. To address this gap, we present an evaluation of 13 legal-specific transformer-based models on 3 English-language contract classification tasks and compare them with 9 generalist models. The results show that legal-specific models consistently outperform generalist models, especially on tasks requiring nuanced legal understanding. They also help reduce misclassification of rare classes in imbalanced datasets. Legal-BERT and Contracts-BERT establish new SOTAs on two of the three tasks, despite having 69% fewer parameters than the best-performing generalist models. We also identify CaseLaw-BERT and LexLM as strong additional baselines for contract classification. Our results highlight the shortcomings of generalist models, emphasizing the need for domain-specific customization, particularly in the context of legal applications.
>
---
#### [replaced 047] How Mobile World Model Guides GUI Agents?
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究移动世界模型对GUI代理的指导作用，解决长期高风险交互中的动作预测问题。通过多模态世界模型训练，提升代理任务性能，并分析不同表示的有效性。**

- **链接: [https://arxiv.org/pdf/2605.10347](https://arxiv.org/pdf/2605.10347)**

> **作者:** Weikai Xu; Kun Huang; Yunren Feng; Jiaxing Li; Yuhan Chen; Yuxuan Liu; Zhizheng Jiang; Heng Qu; Pengzhi Gao; Wei Liu; Jian Luan; Xiaolin Hu; Bo An
>
> **摘要:** Recent advances in vision-language models have enabled mobile GUI agents to perceive visual interfaces and execute user instructions, but reliable prediction of action consequences remains critical for long-horizon and high-risk interactions. Existing mobile world models provide either text-based or image-based future states, yet it remains unclear which representation is useful, whether generated rollouts can replace real environments, and how test-time guidance helps agents of different strengths. To answer the above questions, we filter and annotate mobile world-model data, then train world models across four modalities: delta text, full text, diffusion-based images, and renderable code. These models achieve SoTA performance on both MobileWorldBench and Code2WorldBench. Furthermore, by evaluating their downstream utility on AITZ, AndroidControl, and AndroidWorld, we obtain three findings. First, renderable code reconstruction achieves high in-distribution fidelity and provides effective multimodal supervision for data construction, while text-based feedback is more robust for online out-of-distribution (OOD) execution. Second, world-model-generated trajectories can provide transferable interaction experience in the training process and improve agents' end-to-end task performance, although these data do not preserve the original distribution. Last, for overconfident mobile agents with low action entropy, posterior self-reflection provides limited gains, suggesting that world models are more effective as prior perception or training supervision than as universal post-hoc verifiers.
>
---
#### [replaced 048] Entropy-Aware On-Policy Distillation of Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言模型知识蒸馏任务，旨在解决传统方法在高熵教师分布下生成多样性不足和学习不稳定的问题。通过引入熵感知的正向KL散度，提升生成多样性和对齐效果。**

- **链接: [https://arxiv.org/pdf/2603.07079](https://arxiv.org/pdf/2603.07079)**

> **作者:** Woogyeol Jin; Taywon Min; Yongjin Yang; Swanand Ravindra Kadhe; Yi Zhou; Dennis Wei; Nathalie Baracaldo; Kimin Lee
>
> **备注:** 18 pages, 11 figures, ICML 2026
>
> **摘要:** On-policy distillation is a promising approach for transferring knowledge between language models, where a student learns from dense token-level signals along its own trajectories. This framework typically uses reverse KL divergence, encouraging the student to match the teacher's high-confidence predictions. However, we show that the mode-seeking property of reverse KL reduces generation diversity and yields unstable learning signals when the teacher distribution has high entropy. To address this, we introduce Entropy-Aware On-Policy Distillation. Our key idea is augmenting the standard reverse KL objective with forward KL when teacher entropy is high, capturing the full range of plausible outputs while retaining precise imitation elsewhere. It balances mode-seeking precision with mode-covering robustness without sacrificing on-policy training efficiency. Experiments show that our method maintains generation diversity (sustained token-level entropy) and improves student-teacher alignment (lower forward KL on high-entropy tokens). Across six math reasoning benchmarks, this yields Pass@8 accuracy gains of +1.37 for Qwen3-0.6B-Base, +2.39 for Qwen3-1.7B-Base, and +5.05 for Qwen3-4B-Base compared to baseline on-policy distillation methods. These results demonstrate that accounting for teacher uncertainty is essential for maintaining diversity and achieving effective knowledge transfer.
>
---
#### [replaced 049] Differences in Typological Alignment in Language Models' Treatment of Differential Argument Marking
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究语言模型对差异性论元标记（DAM）的处理。旨在探讨模型是否表现出人类语言的类型学偏好，通过实验发现模型在标记方向上类似人类，但未复制对象优先现象。**

- **链接: [https://arxiv.org/pdf/2602.17653](https://arxiv.org/pdf/2602.17653)**

> **作者:** Iskar Deng; Nathalia Xu; Shane Steinert-Threlkeld
>
> **备注:** 16 pages, 8 figures, 7 tables. To appear at CoNLL 2026
>
> **摘要:** Recent work has shown that language models (LMs) trained on synthetic corpora can exhibit typological preferences that resemble cross-linguistic regularities in human languages, particularly for syntactic phenomena such as word order. In this paper, we extend this paradigm to differential argument marking (DAM), a semantic licensing system in which morphological marking depends on semantic prominence. Using a controlled synthetic learning method, we train GPT-2 models on 18 corpora implementing distinct DAM systems and evaluate their generalization using minimal pairs. Our results reveal a dissociation between two typological dimensions of DAM. Models reliably exhibit human-like preferences for natural markedness direction, favoring systems in which overt marking targets semantically atypical arguments. In contrast, models do not reproduce the strong object preference in human languages, in which overt marking in DAM more often targets objects rather than subjects. These findings suggest that different typological tendencies may arise from distinct underlying sources.
>
---
#### [replaced 050] Sparser Block-Sparse Attention via Token Permutation
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于自然语言处理任务，旨在解决长序列中自注意力机制计算效率低的问题。通过引入令牌排列策略，提升块稀疏注意力的效率与效果。**

- **链接: [https://arxiv.org/pdf/2510.21270](https://arxiv.org/pdf/2510.21270)**

> **作者:** Xinghao Wang; Pengyu Wang; Dong Zhang; Chenkun Tan; Shaojun Zhou; Zhaoxiang Liu; Shiguo Lian; Fangxu Liu; Kai Song; Xipeng Qiu
>
> **备注:** ICML 2026
>
> **摘要:** Scaling the context length of large language models (LLMs) offers significant benefits but is computationally expensive. This expense stems primarily from the self-attention mechanism, whose $O(N^2)$ complexity with respect to sequence length presents a major bottleneck for both memory and latency. Fortunately, the attention matrix is often sparse, particularly for long sequences, suggesting an opportunity for optimization. Block-sparse attention has emerged as a promising solution that partitions sequences into blocks and skips computation for a subset of these blocks. However, the effectiveness of this method is highly dependent on the underlying attention patterns, which can lead to sub-optimal block-level sparsity. For instance, important key tokens for queries within a single block may be scattered across numerous other blocks, leading to computational redundancy. In this work, we propose Permuted Block-Sparse Attention (\textbf{PBS-Attn}), a plug-and-play method that leverages the permutation properties of attention to increase block-level sparsity and enhance the computational efficiency of LLM prefilling. We conduct comprehensive experiments on challenging real-world long-context datasets, demonstrating that PBS-Attn consistently outperforms existing block-sparse attention methods in model accuracy and closely matches the full attention baseline. Powered by our custom permuted-FlashAttention kernels, PBS-Attn achieves an end-to-end speedup of up to $2.75\times$ in long-context prefilling, confirming its practical viability. Code available at this https URL
>
---
#### [replaced 051] Evaluating Counterfactual Strategic Reasoning in Large Language Models
- **分类: cs.CL**

- **简介: 论文评估大语言模型在博弈论场景中的反事实战略推理能力，旨在区分其是真实推理还是依赖记忆模式。通过修改收益结构和动作标签，测试模型在变化环境中的表现，揭示其在激励敏感性、结构泛化和战略推理方面的局限。**

- **链接: [https://arxiv.org/pdf/2603.19167](https://arxiv.org/pdf/2603.19167)**

> **作者:** Dimitrios Georgousis; Maria Lymperaiou; Angeliki Dimitriou; Giorgos Filandrianos; Giorgos Stamou
>
> **备注:** Accepted at GEM@ACL 2026
>
> **摘要:** We evaluate Large Language Models (LLMs) in repeated game-theoretic settings to assess whether strategic performance reflects genuine reasoning or reliance on memorized patterns. We consider two canonical games, Prisoner's Dilemma (PD) and Rock-Paper-Scissors (RPS), upon which we introduce counterfactual variants that alter payoff structures and action labels, breaking familiar symmetries and dominance relations. Our multi-metric evaluation framework compares default and counterfactual instantiations, showcasing LLM limitations in incentive sensitivity, structural generalization and strategic reasoning within counterfactual environments.
>
---
#### [replaced 052] Long-Context Reasoning Through Proxy-Based Chain-of-Thought Tuning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于长文本推理任务，旨在解决大模型在长上下文任务中表现不佳的问题。通过ProxyCoT框架，将短代理上下文的推理能力迁移至全长上下文，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.20201](https://arxiv.org/pdf/2605.20201)**

> **作者:** Miao Li; Irina Saparina; Alexander Gurung; Mirella Lapata
>
> **备注:** Long paper, ACL 2026 (Main conference)
>
> **摘要:** Recent large language models support inputs of up to 10 million tokens, yet they perform poorly on long-context tasks that require complex reasoning. Such tasks can be solved using only a subset of the input -- a proxy context -- rather than the full sequence. Despite sharing the same underlying reasoning process, models exhibit a significant performance disparity between proxy and full contexts. To improve long-context reasoning, we propose ProxyCoT, a novel training framework that transfers reasoning capabilities from short proxy contexts to full long contexts. Specifically, we first obtain high-quality chain-of-thought reasoning traces on proxy contexts through reinforcement learning or distillation from a larger teacher model, and then ground the generated traces in full long contexts with supervised fine-tuning. Experiments across different datasets demonstrate that ProxyCoT consistently outperforms strong baselines with reduced computational overhead. Furthermore, models trained with ProxyCoT generalize their long-context reasoning capabilities to out-of-domain tasks.
>
---
#### [replaced 053] Pooling and Semantic Shift: The Fundamental Challenges in Long Text Embedding and Retrieval
- **分类: cs.CL; cs.IR**

- **简介: 该论文研究长文本嵌入与检索任务，解决嵌入空间退化问题。指出池化操作与语义演变是根本原因，提出理论分析并验证语义变化对嵌入质量的影响。**

- **链接: [https://arxiv.org/pdf/2603.21437](https://arxiv.org/pdf/2603.21437)**

> **作者:** Hang Gao; Wujiang Xu; Kai Mei; Dimitris N. Metaxas
>
> **摘要:** Transformer-based embedding models frequently exhibit geometric pathologies, such as anisotropy and length-induced representation collapse, which can degrade downstream retrieval performance. While prior work often attributes these issues directly to text length or attention mechanisms, we argue that the fundamental drivers are instead the inherent pooling operations coupled with internal semantic shift. In this paper, we establish a unified theoretical framework proving that contextual pooling intrinsically causes embedding collapse. Specifically, we mathematically prove that pooling semantically diverse sentences inevitably leads to micro-level semantic dilution, and strictly reduces the Mean Pairwise Distance of the vector space, guaranteeing macro-level spatial concentration. Grounded in these geometric insights, we formally define semantic shift to capture the natural semantic evolution and dispersion within a text. Through carefully controlled experiments across diverse models and corpora, we disentangle text length from semantic content. We demonstrate that semantic shift is the primary predictor of severe embedding concentration. Crucially, our retrieval evaluations reveal that anisotropy is fundamentally harmful only when induced by strong semantic shifts, reconciling conflicting observations in prior literature and offering a principled explanation for the long-context challenges faced by modern embedding models.
>
---
#### [replaced 054] Patterns vs. Patients: Evaluating LLMs against Mental Health Professionals on Personality Disorder Diagnosis through First-Person Narratives
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 该论文属于心理诊断任务，比较LLMs与专业人员在基于第一人称叙述的BPD/NPD诊断中的表现，发现模型在NPD识别上存在偏差。**

- **链接: [https://arxiv.org/pdf/2512.20298](https://arxiv.org/pdf/2512.20298)**

> **作者:** Karolina Drożdż; Kacper Dudzic; Anna Sterna; Marcin Moskalewicz
>
> **摘要:** Growing reliance on LLMs for psychiatric self-assessment raises questions about their ability to interpret qualitative patient narratives. This depth over breadth case study directly compares state-of-the-art LLMs and mental health professionals in assessing Borderline (BPD) and Narcissistic (NPD) Personality Disorders based on Polish-language first-person autobiographical accounts. Within our sample, the overall diagnostic scores of the top-performing Gemini Pro models (65.48%) were 21.91 percentage points higher than the average scores of the human professionals (43.57%). While both models and human experts excelled at identifying BPD (F1 = 83.4 & F1 = 80.0, respectively), models severely underdiagnosed NPD (F1 = 6.7 vs. 50.0), showing a potential reluctance toward the value-laden term "narcissism." Qualitatively, models provided confident, elaborate justifications focused on patterns and formal categories, while human experts remained concise and cautious, emphasizing the patients' sense of self and temporal experience. Our findings demonstrate that while LLMs might be competent at interpreting complex first-person clinical data, their outputs still carry critical reliability and bias issues.
>
---
#### [replaced 055] Multi-SpatialMLLM: Multi-Frame Spatial Understanding with Multi-Modal Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态语言模型任务，旨在解决MLLM在多帧空间理解上的不足。通过引入新数据集和框架，提升模型的多帧感知能力，并验证其在机器人等场景的应用效果。**

- **链接: [https://arxiv.org/pdf/2505.17015](https://arxiv.org/pdf/2505.17015)**

> **作者:** Runsen Xu; Weiyao Wang; Hao Tang; Xingyu Chen; Xiaodong Wang; Fu-Jen Chu; Matt Feiszli; Kevin J. Liang
>
> **备注:** CVPR 2026 Camera Ready. 27 pages. Project page: this https URL
>
> **摘要:** Multi-modal large language models (MLLMs) have rapidly advanced in visual tasks, yet their spatial understanding remains limited to single images, leaving them ill-suited for physical-world applications that require multi-frame reasoning. In this paper, we propose a framework to equip MLLMs with multi-frame spatial understanding by integrating fundamental spatial skills, including depth perception, visual correspondence, and dynamic perception. We design a novel data pipeline and collect the MultiSPA dataset of more than 27 million samples spanning diverse 3D and 4D scenes to enable training. Alongside MultiSPA, we introduce a comprehensive benchmark that tests a wide spectrum of spatial tasks under uniform metrics. Our resulting model, Multi-SpatialMLLM, achieves significant gains over baselines and proprietary systems, demonstrating scalable and generalizable multi-frame perception. We further observe multi-task benefits and emergent spatial capabilities in challenging scenarios, and showcase how our model can serve as a multi-frame reward annotator for robotics.
>
---
#### [replaced 056] SciNet: Evaluating AI Agents in Relation-Aware Scientific Literature Retrieval
- **分类: cs.CE; cs.CL**

- **简介: 该论文提出SciNet，解决科学文献检索中关系理解不足的问题。构建关系感知数据集，提升检索准确性与文献综述质量。**

- **链接: [https://arxiv.org/pdf/2601.03260](https://arxiv.org/pdf/2601.03260)**

> **作者:** Chenyang Shao; Fengli Xu; Yong Li
>
> **摘要:** AI agents have seen widespread adoption in information retrieval for scientific research, giving rise to tools such as Deep Research. However, existing retrieval agents mainly rely on keyword- or embedding-based methods. While effective at capturing content-level similarities, they struggle to understand complex relational networks among scientific papers, such as identifying corroborating or conflicting studies and tracing technological lineages. This fundamental limitation often results in fragmented knowledge structures, misinterpreted research sentiment, and ineffective modeling of collective scientific progress. To address this limitation, we introduce SciNet, the first Scientific Network relation-aware dataset for information retrieval agents. Built on a meta-database of 269 million papers across 7 disciplines and containing 8,940 carefully designed tasks, SciNet systematically captures three levels of relational understanding: ego-centric retrieval of papers with novel knowledge structures, pairwise identification of scholarly relationships, and path-wise reconstruction of scientific evolution. Extensive evaluation of three categories of retrieval agents shows that their accuracy on relation-aware tasks often falls below 20%, highlighting a fundamental shortcoming of current retrieval paradigms. Importantly, in a downstream literature review application, agents empowered with SciNet achieve a 25.3% improvement in review quality, highlighting the critical value of relation-aware retrieval for deepening scientific insights. We publicly release SciNet at this https URL to support future research.
>
---

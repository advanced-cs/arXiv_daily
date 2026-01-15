# 自然语言处理 cs.CL

- **最新发布 92 篇**

- **更新 55 篇**

## 最新发布

#### [new 001] Can LLMs interpret figurative language as humans do?: surface-level vs representational similarity
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言理解任务，探讨LLMs是否能像人类一样解读隐喻语言。通过对比人类与四个大模型在六种语言特征上的评分，发现模型在表面层接近人类，但在表征层存在显著差异，尤其在理解习语和网络用语时表现不佳。**

- **链接: [https://arxiv.org/pdf/2601.09041v1](https://arxiv.org/pdf/2601.09041v1)**

> **作者:** Samhita Bollepally; Aurora Sloman-Moll; Takashi Yamauchi
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** Large language models generate judgments that resemble those of humans. Yet the extent to which these models align with human judgments in interpreting figurative and socially grounded language remains uncertain. To investigate this, human participants and four instruction-tuned LLMs of different sizes (GPT-4, Gemma-2-9B, Llama-3.2, and Mistral-7B) rated 240 dialogue-based sentences representing six linguistic traits: conventionality, sarcasm, funny, emotional, idiomacy, and slang. Each of the 240 sentences was paired with 40 interpretive questions, and both humans and LLMs rated these sentences on a 10-point Likert scale. Results indicated that humans and LLMs aligned at the surface level with humans, but diverged significantly at the representational level, especially in interpreting figurative sentences involving idioms and Gen Z slang. GPT-4 most closely approximates human representational patterns, while all models struggle with context-dependent and socio-pragmatic expressions like sarcasm, slang, and idiomacy.
>
---
#### [new 002] ProFit: Leveraging High-Value Signals in SFT via Probability-Guided Token Selection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型训练任务，旨在解决SFT中因单参考答案导致的过拟合问题。通过选择性屏蔽低概率token，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.09195v1](https://arxiv.org/pdf/2601.09195v1)**

> **作者:** Tao Liu; Taiqiang Wu; Runming Yang; Shaoning Sun; Junjie Wang; Yujiu Yang
>
> **摘要:** Supervised fine-tuning (SFT) is a fundamental post-training strategy to align Large Language Models (LLMs) with human intent. However, traditional SFT often ignores the one-to-many nature of language by forcing alignment with a single reference answer, leading to the model overfitting to non-core expressions. Although our empirical analysis suggests that introducing multiple reference answers can mitigate this issue, the prohibitive data and computational costs necessitate a strategic shift: prioritizing the mitigation of single-reference overfitting over the costly pursuit of answer diversity. To achieve this, we reveal the intrinsic connection between token probability and semantic importance: high-probability tokens carry the core logical framework, while low-probability tokens are mostly replaceable expressions. Based on this insight, we propose ProFit, which selectively masks low-probability tokens to prevent surface-level overfitting. Extensive experiments confirm that ProFit consistently outperforms traditional SFT baselines on general reasoning and mathematical benchmarks.
>
---
#### [new 003] TeachPro: Multi-Label Qualitative Teaching Evaluation via Cross-View Graph Synergy and Semantic Anchored Evidence Encoding
- **分类: cs.CL**

- **简介: 该论文提出TeachPro，用于多标签教学评估，解决传统评价可靠性低、反馈单一的问题。通过语义锚定编码和跨视图图协同网络，全面分析教学维度。**

- **链接: [https://arxiv.org/pdf/2601.09246v1](https://arxiv.org/pdf/2601.09246v1)**

> **作者:** Xiangqian Wang; Yifan Jia; Yang Xiang; Yumin Zhang; Yanbin Wang; Ke Liu
>
> **摘要:** Standardized Student Evaluation of Teaching often suffer from low reliability, restricted response options, and response distortion. Existing machine learning methods that mine open-ended comments usually reduce feedback to binary sentiment, which overlooks concrete concerns such as content clarity, feedback timeliness, and instructor demeanor, and provides limited guidance for instructional improvement.We propose TeachPro, a multi-label learning framework that systematically assesses five key teaching dimensions: professional expertise, instructional behavior, pedagogical efficacy, classroom experience, and other performance metrics. We first propose a Dimension-Anchored Evidence Encoder, which integrates three core components: (i) a pre-trained text encoder that transforms qualitative feedback annotations into contextualized embeddings; (ii) a prompt module that represents five teaching dimensions as learnable semantic anchors; and (iii) a cross-attention mechanism that aligns evidence with pedagogical dimensions within a structured semantic space. We then propose a Cross-View Graph Synergy Network to represent student comments. This network comprises two components: (i) a Syntactic Branch that extracts explicit grammatical dependencies from parse trees, and (ii) a Semantic Branch that models latent conceptual relations derived from BERT-based similarity graphs. BiAffine fusion module aligns syntactic and semantic units, while a differential regularizer disentangles embeddings to encourage complementary representations. Finally, a cross-attention mechanism bridges the dimension-anchored evidence with the multi-view comment representations. We also contribute a novel benchmark dataset featuring expert qualitative annotations and multi-label scores. Extensive experiments demonstrate that TeachPro offers superior diagnostic granularity and robustness across diverse evaluation settings.
>
---
#### [new 004] Identity-Robust Language Model Generation via Content Integrity Preservation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决用户身份导致的语言模型输出偏差问题。通过保留关键语义信息，减少非关键身份信息影响，提升生成内容的一致性与公平性。**

- **链接: [https://arxiv.org/pdf/2601.09141v1](https://arxiv.org/pdf/2601.09141v1)**

> **作者:** Miao Zhang; Kelly Chen; Md Mehrab Tanjim; Rumi Chunara
>
> **摘要:** Large Language Model (LLM) outputs often vary across user sociodemographic attributes, leading to disparities in factual accuracy, utility, and safety, even for objective questions where demographic information is irrelevant. Unlike prior work on stereotypical or representational bias, this paper studies identity-dependent degradation of core response quality. We show empirically that such degradation arises from biased generation behavior, despite factual knowledge being robustly encoded across identities. Motivated by this mismatch, we propose a lightweight, training-free framework for identity-robust generation that selectively neutralizes non-critical identity information while preserving semantically essential attributes, thus maintaining output content integrity. Experiments across four benchmarks and 18 sociodemographic identities demonstrate an average 77% reduction in identity-dependent bias compared to vanilla prompting and a 45% reduction relative to prompt-based defenses. Our work addresses a critical gap in mitigating the impact of user identity cues in prompts on core generation quality.
>
---
#### [new 005] The Imperfective Paradox in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言理解任务，探讨LLMs是否真正理解事件的语义结构。研究解决LLM在时态推理中的偏差问题，通过构建数据集并分析模型表现，揭示其依赖先验而非逻辑推理的倾向。**

- **链接: [https://arxiv.org/pdf/2601.09373v1](https://arxiv.org/pdf/2601.09373v1)**

> **作者:** Bolei Ma; Yusuke Miyao
>
> **摘要:** Do Large Language Models (LLMs) genuinely grasp the compositional semantics of events, or do they rely on surface-level probabilistic heuristics? We investigate the Imperfective Paradox, a logical phenomenon where the past progressive aspect entails event realization for activities (e.g., running $\to$ ran) but not for accomplishments (e.g., building $\nrightarrow$ built). We introduce ImperfectiveNLI, a diagnostic dataset designed to probe this distinction across diverse semantic classes. Evaluating state-of-the-art open-weight models, we uncover a pervasive Teleological Bias: models systematically hallucinate completion for goal-oriented events, often overriding explicit textual negation. Representational analyses show that while internal embeddings often distinguish process from result, inference decisions are dominated by strong priors about goal attainment. We further find that prompting-based interventions reduce hallucinated completions but also increase incorrect rejections of valid entailments. Our findings suggest that current LLMs lack structural aspectual awareness, operating as predictive narrative engines rather than faithful logical reasoners.
>
---
#### [new 006] Mi:dm 2.0 Korea-centric Bilingual Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍Mi:dm 2.0，一种面向韩国文化的双语大语言模型，解决韩国数据不足和文化适配问题，通过高质量数据和优化架构提升性能。**

- **链接: [https://arxiv.org/pdf/2601.09066v1](https://arxiv.org/pdf/2601.09066v1)**

> **作者:** Donghoon Shin; Sejung Lee; Soonmin Bae; Hwijung Ryu; Changwon Ok; Hoyoun Jung; Hyesung Ji; Jeehyun Lim; Jehoon Lee; Ji-Eun Han; Jisoo Baik; Mihyeon Kim; Riwoo Chung; Seongmin Lee; Wonjae Park; Yoonseok Heo; Youngkyung Seo; Seyoun Won; Boeun Kim; Cheolhun Heo; Eunkyeong Lee; Honghee Lee; Hyeongju Ju; Hyeontae Seo; Jeongyong Shim; Jisoo Lee; Junseok Koh; Junwoo Kim; Minho Lee; Minji Kang; Minju Kim; Sangha Nam; Seongheum Park; Taehyeong Kim; Euijai Ahn; Hong Seok Jeung; Jisu Shin; Jiyeon Kim; Seonyeong Song; Seung Hyun Kong; Sukjin Hong; Taeyang Yun; Yu-Seon Kim; A-Hyun Lee; Chae-Jeong Lee; Hye-Won Yu; Ji-Hyun Ahn; Song-Yeon Kim; Sun-Woo Jung; Eunju Kim; Eunji Ha; Jinwoo Baek; Yun-ji Lee; Wanjin Park; Jeong Yeop Kim; Eun Mi Kim; Hyoung Jun Park; Jung Won Yoon; Min Sung Noh; Myung Gyo Oh; Wongyoung Lee; Yun Jin Park; Young S. Kwon; Hyun Keun Kim; Jieun Lee; YeoJoo Park
>
> **摘要:** We introduce Mi:dm 2.0, a bilingual large language model (LLM) specifically engineered to advance Korea-centric AI. This model goes beyond Korean text processing by integrating the values, reasoning patterns, and commonsense knowledge inherent to Korean society, enabling nuanced understanding of cultural contexts, emotional subtleties, and real-world scenarios to generate reliable and culturally appropriate responses. To address limitations of existing LLMs, often caused by insufficient or low-quality Korean data and lack of cultural alignment, Mi:dm 2.0 emphasizes robust data quality through a comprehensive pipeline that includes proprietary data cleansing, high-quality synthetic data generation, strategic data mixing with curriculum learning, and a custom Korean-optimized tokenizer to improve efficiency and coverage. To realize this vision, we offer two complementary configurations: Mi:dm 2.0 Base (11.5B parameters), built with a depth-up scaling strategy for general-purpose use, and Mi:dm 2.0 Mini (2.3B parameters), optimized for resource-constrained environments and specialized tasks. Mi:dm 2.0 achieves state-of-the-art performance on Korean-specific benchmarks, with top-tier zero-shot results on KMMLU and strong internal evaluation results across language, humanities, and social science tasks. The Mi:dm 2.0 lineup is released under the MIT license to support extensive research and commercial use. By offering accessible and high-performance Korea-centric LLMs, KT aims to accelerate AI adoption across Korean industries, public services, and education, strengthen the Korean AI developer community, and lay the groundwork for the broader vision of K-intelligence. Our models are available at https://huggingface.co/K-intelligence. For technical inquiries, please contact midm-llm@kt.com.
>
---
#### [new 007] Contrastive Bi-Encoder Models for Multi-Label Skill Extraction: Enhancing ESCO Ontology Matching with BERT and Attention Mechanisms
- **分类: cs.CL; econ.GN**

- **简介: 该论文属于多标签技能提取任务，旨在解决劳动市场分析中岗位描述与技能分类匹配的问题。通过构建对比双编码器模型，提升ESCO本体匹配效果。**

- **链接: [https://arxiv.org/pdf/2601.09119v1](https://arxiv.org/pdf/2601.09119v1)**

> **作者:** Yongming Sun
>
> **摘要:** Fine-grained labor market analysis increasingly relies on mapping unstructured job advertisements to standardized skill taxonomies such as ESCO. This mapping is naturally formulated as an Extreme Multi-Label Classification (XMLC) problem, but supervised solutions are constrained by the scarcity and cost of large-scale, taxonomy-aligned annotations--especially in non-English settings where job-ad language diverges substantially from formal skill definitions. We propose a zero-shot skill extraction framework that eliminates the need for manually labeled job-ad training data. The framework uses a Large Language Model (LLM) to synthesize training instances from ESCO definitions, and introduces hierarchically constrained multi-skill generation based on ESCO Level-2 categories to improve semantic coherence in multi-label contexts. On top of the synthetic corpus, we train a contrastive bi-encoder that aligns job-ad sentences with ESCO skill descriptions in a shared embedding space; the encoder augments a BERT backbone with BiLSTM and attention pooling to better model long, information-dense requirement statements. An upstream RoBERTa-based binary filter removes non-skill sentences to improve end-to-end precision. Experiments show that (i) hierarchy-conditioned generation improves both fluency and discriminability relative to unconstrained pairing, and (ii) the resulting multi-label model transfers effectively to real-world Chinese job advertisements, achieving strong zero-shot retrieval performance (F1@5 = 0.72) and outperforming TF--IDF and standard BERT baselines. Overall, the proposed pipeline provides a scalable, data-efficient pathway for automated skill coding in labor economics and workforce analytics.
>
---
#### [new 008] DeepResearchEval: An Automated Framework for Deep Research Task Construction and Agentic Evaluation
- **分类: cs.CL**

- **简介: 该论文提出DeepResearchEval，解决深度研究任务构建与评估难题。通过自动化生成复杂研究任务，并进行动态质量评估和事实核查，提升评估的准确性与适应性。**

- **链接: [https://arxiv.org/pdf/2601.09688v1](https://arxiv.org/pdf/2601.09688v1)**

> **作者:** Yibo Wang; Lei Wang; Yue Deng; Keming Wu; Yao Xiao; Huanjin Yao; Liwei Kang; Hai Ye; Yongcheng Jing; Lidong Bing
>
> **备注:** Source code: https://github.com/Infinity-AILab/DeepResearchEval
>
> **摘要:** Deep research systems are widely used for multi-step web research, analysis, and cross-source synthesis, yet their evaluation remains challenging. Existing benchmarks often require annotation-intensive task construction, rely on static evaluation dimensions, or fail to reliably verify facts when citations are missing. To bridge these gaps, we introduce DeepResearchEval, an automated framework for deep research task construction and agentic evaluation. For task construction, we propose a persona-driven pipeline generating realistic, complex research tasks anchored in diverse user profiles, applying a two-stage filter Task Qualification and Search Necessity to retain only tasks requiring multi-source evidence integration and external retrieval. For evaluation, we propose an agentic pipeline with two components: an Adaptive Point-wise Quality Evaluation that dynamically derives task-specific evaluation dimensions, criteria, and weights conditioned on each generated task, and an Active Fact-Checking that autonomously extracts and verifies report statements via web search, even when citations are missing.
>
---
#### [new 009] DPWriter: Reinforcement Learning with Diverse Planning Branching for Creative Writing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于创意写作任务，旨在解决强化学习导致输出多样性不足的问题。通过引入多样化的规划分支和群体感知奖励机制，提升生成内容的多样性。**

- **链接: [https://arxiv.org/pdf/2601.09609v1](https://arxiv.org/pdf/2601.09609v1)**

> **作者:** Qian Cao; Yahui Liu; Wei Bi; Yi Zhao; Ruihua Song; Xiting Wang; Ruiming Tang; Guorui Zhou; Han Li
>
> **摘要:** Reinforcement learning (RL)-based enhancement of large language models (LLMs) often leads to reduced output diversity, undermining their utility in open-ended tasks like creative writing. Current methods lack explicit mechanisms for guiding diverse exploration and instead prioritize optimization efficiency and performance over diversity. This paper proposes an RL framework structured around a semi-structured long Chain-of-Thought (CoT), in which the generation process is decomposed into explicitly planned intermediate steps. We introduce a Diverse Planning Branching method that strategically introduces divergence at the planning phase based on diversity variation, alongside a group-aware diversity reward to encourage distinct trajectories. Experimental results on creative writing benchmarks demonstrate that our approach significantly improves output diversity without compromising generation quality, consistently outperforming existing baselines.
>
---
#### [new 010] Recursive Knowledge Synthesis for Multi-LLM Systems: Stability Analysis and Tri-Agent Audit Framework
- **分类: cs.CL**

- **简介: 该论文属于多大模型系统任务，旨在解决稳定性与可解释性问题。通过三代理架构实现递归知识合成，验证系统稳定性与透明度。**

- **链接: [https://arxiv.org/pdf/2601.08839v1](https://arxiv.org/pdf/2601.08839v1)**

> **作者:** Toshiyuki Shigemura
>
> **备注:** 25 pages, 9 figures. Pilot feasibility study using public-access large language models without API-level orchestration
>
> **摘要:** This paper presents a tri-agent cross-validation framework for analyzing stability and explainability in multi-model large language systems. The architecture integrates three heterogeneous LLMs-used for semantic generation, analytical consistency checking, and transparency auditing-into a recursive interaction cycle. This design induces Recursive Knowledge Synthesis (RKS), where intermediate representations are continuously refined through mutually constraining transformations irreducible to single-model behavior. Across 47 controlled trials using public-access LLM deployments (October 2025), we evaluated system stability via four metrics: Reflex Reliability Score (RRS), Transparency Score (TS), Deviation Detection Rate (DDR), and Correction Success Rate (CSR). The system achieved mean RRS = 0.78+-0.06 and maintained TS >= 0.8 in about 68% of trials. Approximately 89% of trials converged, supporting the theoretical prediction that transparency auditing acts as a contraction operator within the composite validation mapping. The contributions are threefold: (1) a structured tri-agent framework for coordinated reasoning across heterogeneous LLMs, (2) a formal RKS model grounded in fixed-point theory, and (3) empirical evaluation of inter-model stability under realistic, non-API public-access conditions. These results provide initial empirical evidence that a safety-preserving, humansupervised multi-LLM architecture can achieve stable recursive knowledge synthesis in realistic, publicly deployed environments.
>
---
#### [new 011] NewsScope: Schema-Grounded Cross-Domain News Claim Extraction with Open Models
- **分类: cs.CL**

- **简介: 该论文提出NewsScope，解决跨领域新闻事实提取问题。构建数据集并微调模型，提升结构化事实抽取的准确性和通用性。**

- **链接: [https://arxiv.org/pdf/2601.08852v1](https://arxiv.org/pdf/2601.08852v1)**

> **作者:** Nidhi Pandya
>
> **备注:** 5 pages, 3 tables. Code, model, and benchmark publicly released
>
> **摘要:** Automated news verification requires structured claim extraction, but existing approaches either lack schema compliance or generalize poorly across domains. This paper presents NewsScope, a cross-domain dataset, benchmark, and fine-tuned model for schema-grounded news claim extraction. The dataset contains 455 articles across politics, health, science/environment, and business, consisting of 395 in-domain articles and 60 out-of-source articles for generalization testing. LLaMA 3.1 8B was fine-tuned using LoRA on 315 training examples and evaluated on held-out in-domain (80 articles) and out-of-source (60 articles) test sets. Human evaluation on 400 claims shows NewsScope achieves 89.4% human-evaluated accuracy compared to GPT-4o-mini's 93.7% (p=0.07). NewsScope outperforms GPT-4o-mini on political claims (94.3% vs. 87.8%). A numeric grounding filter further improves accuracy to 91.6%, narrowing the gap to 2.1 percentage points. Inter-annotator agreement studies (160 claims) confirm labeling reliability (94.6% positive agreement on SUPPORTED judgments). The open-weight model enables offline deployment at approximately $15 on-demand compute (or $0 on free tiers). Code and benchmark are publicly released.
>
---
#### [new 012] Creating a Hybrid Rule and Neural Network Based Semantic Tagger using Silver Standard Data: the PyMUSAS framework for Multilingual Semantic Annotation
- **分类: cs.CL**

- **简介: 该论文属于语义标注任务，旨在解决USAS框架在多语言和大规模评估中的不足。通过构建银标准数据集，结合规则系统与神经网络模型进行改进和评估。**

- **链接: [https://arxiv.org/pdf/2601.09648v1](https://arxiv.org/pdf/2601.09648v1)**

> **作者:** Andrew Moore; Paul Rayson; Dawn Archer; Tim Czerniak; Dawn Knight; Daisy Lal; Gearóid Ó Donnchadha; Mícheál Ó Meachair; Scott Piao; Elaine Uí Dhonnchadha; Johanna Vuorinen; Yan Yabo; Xiaobin Yang
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** Word Sense Disambiguation (WSD) has been widely evaluated using the semantic frameworks of WordNet, BabelNet, and the Oxford Dictionary of English. However, for the UCREL Semantic Analysis System (USAS) framework, no open extensive evaluation has been performed beyond lexical coverage or single language evaluation. In this work, we perform the largest semantic tagging evaluation of the rule based system that uses the lexical resources in the USAS framework covering five different languages using four existing datasets and one novel Chinese dataset. We create a new silver labelled English dataset, to overcome the lack of manually tagged training data, that we train and evaluate various mono and multilingual neural models in both mono and cross-lingual evaluation setups with comparisons to their rule based counterparts, and show how a rule based system can be enhanced with a neural network model. The resulting neural network models, including the data they were trained on, the Chinese evaluation dataset, and all of the code have been released as open resources.
>
---
#### [new 013] UserLM-R1: Modeling Human Reasoning in User Language Models with Multi-Reward Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升用户模拟器的泛化能力和策略性。解决现有方法依赖静态资料、缺乏战略思维的问题，提出UserLM-R1模型，通过多奖励强化学习增强推理与谈判能力。**

- **链接: [https://arxiv.org/pdf/2601.09215v1](https://arxiv.org/pdf/2601.09215v1)**

> **作者:** Feng Zhang; Shijia Li; Chunmao Zhang; Zhanyu Ma; Jun Xu; Jiuchong Gao; Jinghua Hao; Renqing He; Jingwen Xu; Han Liu
>
> **摘要:** User simulators serve as the critical interactive environment for agent post-training, and an ideal user simulator generalizes across domains and proactively engages in negotiation by challenging or bargaining. However, current methods exhibit two issues. They rely on static and context-unaware profiles, necessitating extensive manual redesign for new scenarios, thus limiting generalizability. Moreover, they neglect human strategic thinking, leading to vulnerability to agent manipulation. To address these issues, we propose UserLM-R1, a novel user language model with reasoning capability. Specifically, we first construct comprehensive user profiles with both static roles and dynamic scenario-specific goals for adaptation to diverse scenarios. Then, we propose a goal-driven decision-making policy to generate high-quality rationales before producing responses, and further refine the reasoning and improve strategic capabilities with supervised fine-tuning and multi-reward reinforcement learning. Extensive experimental results demonstrate that UserLM-R1 outperforms competitive baselines, particularly on the more challenging adversarial set.
>
---
#### [new 014] Frame of Reference: Addressing the Challenges of Common Ground Representation in Situational Dialogs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统任务，旨在解决情境对话中共同基础的表示问题。研究评估了模型通过实体关系建立和利用共同基础的能力，并提出改进方法。**

- **链接: [https://arxiv.org/pdf/2601.09365v1](https://arxiv.org/pdf/2601.09365v1)**

> **作者:** Biswesh Mohapatra; Théo Charlot; Giovanni Duca; Mayank Palan; Laurent Romary; Justine Cassell
>
> **摘要:** Common ground plays a critical role in situated spoken dialogues, where interlocutors must establish and maintain shared references to entities, events, and relations to sustain coherent interaction. For dialog systems, the ability to correctly ground conversational content in order to refer back to it later is particularly important. Prior studies have demonstrated that LLMs are capable of performing grounding acts such as requesting clarification or producing acknowledgments, yet relatively little work has investigated how common ground can be explicitly represented and stored for later use. Without such mechanisms, it remains unclear whether acknowledgment or clarification behaviors truly reflect a grounded understanding. In this work, we evaluate a model's ability to establish and exploit common ground through relational references to entities within the shared context in a situational dialogue. We test multiple methods for representing common ground in situated dialogues and further propose approaches to improve both the establishment of common ground and its subsequent use in the conversation.
>
---
#### [new 015] SpectraQuery: A Hybrid Retrieval-Augmented Conversational Assistant for Battery Science
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出SpectraQuery，解决科学问答中结构化数据与非结构化文献联合推理问题，通过混合检索增强生成技术实现高效准确的回答。**

- **链接: [https://arxiv.org/pdf/2601.09036v1](https://arxiv.org/pdf/2601.09036v1)**

> **作者:** Sreya Vangara; Jagjit Nanda; Yan-Kai Tzeng; Eric Darve
>
> **备注:** 11 pages, 8 figures, appendix included
>
> **摘要:** Scientific reasoning increasingly requires linking structured experimental data with the unstructured literature that explains it, yet most large language model (LLM) assistants cannot reason jointly across these modalities. We introduce SpectraQuery, a hybrid natural-language query framework that integrates a relational Raman spectroscopy database with a vector-indexed scientific literature corpus using a Structured and Unstructured Query Language (SUQL)-inspired design. By combining semantic parsing with retrieval-augmented generation, SpectraQuery translates open-ended questions into coordinated SQL and literature retrieval operations, producing cited answers that unify numerical evidence with mechanistic explanation. Across SQL correctness, answer groundedness, retrieval effectiveness, and expert evaluation, SpectraQuery demonstrates strong performance: approximately 80 percent of generated SQL queries are fully correct, synthesized answers reach 93-97 percent groundedness with 10-15 retrieved passages, and battery scientists rate responses highly across accuracy, relevance, grounding, and clarity (4.1-4.6/5). These results show that hybrid retrieval architectures can meaningfully support scientific workflows by bridging data and discourse for high-volume experimental datasets.
>
---
#### [new 016] Consistency-Aware Editing for Entity-level Unlearning in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型中的实体级遗忘任务，旨在有效删除特定实体知识而不影响模型其他能力。提出CAE框架，通过一致性约束实现高效、鲁棒的实体遗忘。**

- **链接: [https://arxiv.org/pdf/2601.08840v1](https://arxiv.org/pdf/2601.08840v1)**

> **作者:** Xiaoqi Han; Víctor Gutiérrez-Basulto; Ru Li; Xiaoli Li; Jiye Liang; Jeff Z. Pan
>
> **摘要:** Large language models (LLMs) risk retaining sensitive, copyrighted, or harmful information from their training data. Entity-level unlearning addresses this issue by removing all knowledge of a specific entity while preserving the model's overall capabilities. Existing approaches typically rely on full-model fine-tuning or prompt-based interventions, which can be computationally expensive or brittle when handling paraphrased queries. Recently, model editing has emerged as an efficient alternative for updating knowledge in LLMs, offering a promising direction for unlearning. However, existing editing techniques are typically designed for instance-level updates, modifying responses to specific attributes of an entity rather than eliminating all knowledge associated with the entity. In this paper, we investigate how editing techniques can be adapted for effective and efficient entity-level unlearning. To this end, we introduce a novel consistency-aware editing (CAE) framework. CAE aggregates a diverse set of prompts related to a target entity, including its attributes, relations, and adversarial paraphrases. It then jointly learns a low-rank update guided by a consistency regularizer that aligns the editing directions across prompts. This promotes robust and comprehensive forgetting while minimizing interference with unrelated knowledge. We further examine where different entities are stored within the model and how many diverse prompts are needed for successful unlearning. We evaluate CAE on two challenging benchmarks, RWKU and ToFU, and demonstrate that it (i) provides insights into how entity-level knowledge is internally represented and deleted in LLMs, (ii) significantly improves forgetting accuracy and robustness over traditional unlearning and editing baselines, and (iii) enables scalable entity removal using only tens of carefully selected prompts.
>
---
#### [new 017] OpenDecoder: Open Large Language Model Decoding to Incorporate Document Quality in RAG
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于RAG任务，旨在解决检索信息质量影响生成内容的问题。提出OpenDecoder，通过显式评估信息质量特征提升生成效果。**

- **链接: [https://arxiv.org/pdf/2601.09028v1](https://arxiv.org/pdf/2601.09028v1)**

> **作者:** Fengran Mo; Zhan Su; Yuchen Hui; Jinghan Zhang; Jia Ao Sun; Zheyuan Liu; Chao Zhang; Tetsuya Sakai; Jian-Yun Nie
>
> **备注:** Accepted by ACM WWW 2026
>
> **摘要:** The development of large language models (LLMs) has achieved superior performance in a range of downstream tasks, including LLM-based retrieval-augmented generation (RAG). The quality of generated content heavily relies on the usefulness of the retrieved information and the capacity of LLMs' internal information processing mechanism to incorporate it in answer generation. It is generally assumed that the retrieved information is relevant to the question. However, the retrieved information may have a variable degree of relevance and usefulness, depending on the question and the document collection. It is important to take into account the relevance of the retrieved information in answer generation. In this paper, we propose OpenDecoder, a new approach that leverages explicit evaluation of the retrieved information as quality indicator features for generation. We aim to build a RAG model that is more robust to varying levels of noisy context. Three types of explicit evaluation information are considered: relevance score, ranking score, and QPP (query performance prediction) score. The experimental results on five benchmark datasets demonstrate the effectiveness and better robustness of OpenDecoder by outperforming various baseline methods. Importantly, this paradigm is flexible to be integrated with the post-training of LLMs for any purposes and incorporated with any type of external indicators.
>
---
#### [new 018] Más contexto no es mejor. Paradoja de la dilución vectorial en RAG corporativos
- **分类: cs.CL; cs.AI**

- **简介: 论文研究RAG中的"向量稀释"问题，探讨上下文注入比例对检索效果的影响。属于信息检索任务，旨在优化上下文注入以提升召回率并减少精度损失。**

- **链接: [https://arxiv.org/pdf/2601.08851v1](https://arxiv.org/pdf/2601.08851v1)**

> **作者:** Alex Dantart
>
> **备注:** in Spanish and English languages
>
> **摘要:** Técnicas recientes de "Contextualized Chunking" inyectan resúmenes para mejorar el contexto en RAG, pero introducen una "dilución vectorial" que opaca el contenido local. Evaluando distintos ratios de inyección, demostramos una curva en "U invertida": una inyección moderada mejora el "Recall" (+18%), pero superar un umbral crítico (CIR > 0.4) reduce la precisión en un 22% para consultas específicas. Proponemos un marco teórico para calcular el ratio óptimo de inyección. -- Recent "Contextualized Chunking" techniques inject summaries to improve RAG context but introduce "vector dilution" drowning out local content. Evaluating various injection ratios, we demonstrate an "inverted U" curve: moderate injection boosts Recall (+18%), but exceeding a critical threshold (CIR > 0.4) drops precision by 22% for specific queries. We propose a theoretical framework to calculate the optimal injection ratio.
>
---
#### [new 019] Evaluating Role-Consistency in LLMs for Counselor Training
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs在虚拟咨询中保持角色一致性的问题。通过构建对抗数据集，评估不同模型的角色一致性与对话连贯性。**

- **链接: [https://arxiv.org/pdf/2601.08892v1](https://arxiv.org/pdf/2601.08892v1)**

> **作者:** Eric Rudolph; Natalie Engert; Jens Albrecht
>
> **摘要:** The rise of online counseling services has highlighted the need for effective training methods for future counselors. This paper extends research on VirCo, a Virtual Client for Online Counseling, designed to complement traditional role-playing methods in academic training by simulating realistic client interactions. Building on previous work, we introduce a new dataset incorporating adversarial attacks to test the ability of large language models (LLMs) to maintain their assigned roles (role-consistency). The study focuses on evaluating the role consistency and coherence of the Vicuna model's responses, comparing these findings with earlier research. Additionally, we assess and compare various open-source LLMs for their performance in sustaining role consistency during virtual client interactions. Our contributions include creating an adversarial dataset, evaluating conversation coherence and persona consistency, and providing a comparative analysis of different LLMs.
>
---
#### [new 020] TranslateGemma Technical Report
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译任务，旨在提升Gemma 3模型的翻译性能。通过两阶段微调和强化学习优化，开发了TranslateGemma模型，实现了高效且高质量的翻译效果。**

- **链接: [https://arxiv.org/pdf/2601.09012v1](https://arxiv.org/pdf/2601.09012v1)**

> **作者:** Mara Finkelstein; Isaac Caswell; Tobias Domhan; Jan-Thorsten Peter; Juraj Juraska; Parker Riley; Daniel Deutsch; Cole Dilanni; Colin Cherry; Eleftheria Briakou; Elizabeth Nielsen; Jiaming Luo; Kat Black; Ryan Mullins; Sweta Agrawal; Wenda Xu; Erin Kats; Stephane Jaskiewicz; Markus Freitag; David Vilar
>
> **摘要:** We present TranslateGemma, a suite of open machine translation models based on the Gemma 3 foundation models. To enhance the inherent multilingual capabilities of Gemma 3 for the translation task, we employ a two-stage fine-tuning process. First, supervised fine-tuning is performed using a rich mixture of high-quality large-scale synthetic parallel data generated via state-of-the-art models and human-translated parallel data. This is followed by a reinforcement learning phase, where we optimize translation quality using an ensemble of reward models, including MetricX-QE and AutoMQM, targeting translation quality. We demonstrate the effectiveness of TranslateGemma with human evaluation on the WMT25 test set across 10 language pairs and with automatic evaluation on the WMT24++ benchmark across 55 language pairs. Automatic metrics show consistent and substantial gains over the baseline Gemma 3 models across all sizes. Notably, smaller TranslateGemma models often achieve performance comparable to larger baseline models, offering improved efficiency. We also show that TranslateGemma models retain strong multimodal capabilities, with enhanced performance on the Vistra image translation benchmark. The release of the open TranslateGemma models aims to provide the research community with powerful and adaptable tools for machine translation.
>
---
#### [new 021] When to Trust: A Causality-Aware Calibration Framework for Accurate Knowledge Graph Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于知识图谱增强生成任务，解决KG-RAG模型过度自信的问题，提出Ca2KG框架提升预测校准性。**

- **链接: [https://arxiv.org/pdf/2601.09241v1](https://arxiv.org/pdf/2601.09241v1)**

> **作者:** Jing Ren; Bowen Li; Ziqi Xu; Xinkun Zhang; Haytham Fayek; Xiaodong Li
>
> **备注:** Accepted by WWW 2026
>
> **摘要:** Knowledge Graph Retrieval-Augmented Generation (KG-RAG) extends the RAG paradigm by incorporating structured knowledge from knowledge graphs, enabling Large Language Models (LLMs) to perform more precise and explainable reasoning. While KG-RAG improves factual accuracy in complex tasks, existing KG-RAG models are often severely overconfident, producing high-confidence predictions even when retrieved sub-graphs are incomplete or unreliable, which raises concerns for deployment in high-stakes domains. To address this issue, we propose Ca2KG, a Causality-aware Calibration framework for KG-RAG. Ca2KG integrates counterfactual prompting, which exposes retrieval-dependent uncertainties in knowledge quality and reasoning reliability, with a panel-based re-scoring mechanism that stabilises predictions across interventions. Extensive experiments on two complex QA datasets demonstrate that Ca2KG consistently improves calibration while maintaining or even enhancing predictive accuracy.
>
---
#### [new 022] Where Knowledge Collides: A Mechanistic Study of Intra-Memory Knowledge Conflict in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型研究任务，旨在解决内部知识冲突问题。通过机制可解释方法，定位并干预预训练数据中的冲突知识编码。**

- **链接: [https://arxiv.org/pdf/2601.09445v1](https://arxiv.org/pdf/2601.09445v1)**

> **作者:** Minh Vu Pham; Hsuvas Borkakoty; Yufang Hou
>
> **摘要:** In language models (LMs), intra-memory knowledge conflict largely arises when inconsistent information about the same event is encoded within the model's parametric knowledge. While prior work has primarily focused on resolving conflicts between a model's internal knowledge and external resources through approaches such as fine-tuning or knowledge editing, the problem of localizing conflicts that originate during pre-training within the model's internal representations remain unexplored. In this work, we design a framework based on mechanistic interpretability methods to identify where and how conflicting knowledge from the pre-training data is encoded within LMs. Our findings contribute to a growing body of evidence that specific internal components of a language model are responsible for encoding conflicting knowledge from pre-training, and we demonstrate how mechanistic interpretability methods can be leveraged to causally intervene in and control conflicting knowledge at inference time.
>
---
#### [new 023] Benchmarking Post-Training Quantization of Large Language Models under Microscaling Floating Point Formats
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型量化任务，研究在MXFP格式下后训练量化的效果。针对现有方法多针对整数量化的问题，系统评估了多种PTQ算法在MXFP下的表现，提出了优化策略。**

- **链接: [https://arxiv.org/pdf/2601.09555v1](https://arxiv.org/pdf/2601.09555v1)**

> **作者:** Manyi Zhang; Ji-Fu Li; Zhongao Sun; Haoli Bai; Hui-Ling Zhen; Zhenhua Dong; Xianzhi Yu
>
> **摘要:** Microscaling Floating-Point (MXFP) has emerged as a promising low-precision format for large language models (LLMs). Despite various post-training quantization (PTQ) algorithms being proposed, they mostly focus on integer quantization, while their applicability and behavior under MXFP formats remain largely unexplored. To address this gap, this work conducts a systematic investigation of PTQ under MXFP formats, encompassing over 7 PTQ algorithms, 15 evaluation benchmarks, and 3 LLM families. The key findings include: 1) MXFP8 consistently achieves near-lossless performance, while MXFP4 introduces substantial accuracy degradation and remains challenging; 2) PTQ effectiveness under MXFP depends strongly on format compatibility, with some algorithmic paradigms being consistently more effective than others; 3) PTQ performance exhibits highly consistent trends across model families and modalities, in particular, quantization sensitivity is dominated by the language model rather than the vision encoder in multimodal LLMs; 4) The scaling factor of quantization is a critical error source in MXFP4, and a simple pre-scale optimization strategy can significantly mitigate its impact. Together, these results provide practical guidance on adapting existing PTQ methods to MXFP quantization.
>
---
#### [new 024] Companion Agents: A Table-Information Mining Paradigm for Text-to-SQL
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于Text-to-SQL任务，解决数据库标注缺失下的SQL生成问题。提出Companion Agents，通过数据库端预先挖掘信息提升准确性。**

- **链接: [https://arxiv.org/pdf/2601.08838v1](https://arxiv.org/pdf/2601.08838v1)**

> **作者:** Jiahui Chen; Lei Fu; Jian Cui; Yu Lei; Zhenning Dong
>
> **备注:** 11 pages
>
> **摘要:** Large-scale Text-to-SQL benchmarks such as BIRD typically assume complete and accurate database annotations as well as readily available external knowledge, which fails to reflect common industrial settings where annotations are missing, incomplete, or erroneous. This mismatch substantially limits the real-world applicability of state-of-the-art (SOTA) Text-to-SQL systems. To bridge this gap, we explore a database-centric approach that leverages intrinsic, fine-grained information residing in relational databases to construct missing evidence and improve Text-to-SQL accuracy under annotation-scarce conditions. Our key hypothesis is that when a query requires multi-step reasoning over extensive table information, existing methods often struggle to reliably identify and utilize the truly relevant knowledge. We therefore propose to "cache" query-relevant knowledge on the database side in advance, so that it can be selectively activated at inference time. Based on this idea, we introduce Companion Agents (CA), a new Text-to-SQL paradigm that incorporates a group of agents accompanying database schemas to proactively mine and consolidate hidden inter-table relations, value-domain distributions, statistical regularities, and latent semantic cues before query generation. Experiments on BIRD under the fully missing evidence setting show that CA recovers +4.49 / +4.37 / +14.13 execution accuracy points on RSL-SQL / CHESS / DAIL-SQL, respectively, with larger gains on the Challenging subset +9.65 / +7.58 / +16.71. These improvements stem from CA's automatic database-side mining and evidence construction, suggesting a practical path toward industrial-grade Text-to-SQL deployment without reliance on human-curated evidence.
>
---
#### [new 025] Is Grokking Worthwhile? Functional Analysis and Transferability of Generalization Circuits in Transformers
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究Transformer模型在组合任务中的泛化能力，探讨“grokking”是否提升模型性能。通过分析推理路径，发现grokking并非引入新推理模式，而是整合记忆事实，且泛化电路转移能力有限。**

- **链接: [https://arxiv.org/pdf/2601.09049v1](https://arxiv.org/pdf/2601.09049v1)**

> **作者:** Kaiyu He; Zhang Mian; Peilin Wu; Xinya Du; Zhiyu Chen
>
> **摘要:** While Large Language Models (LLMs) excel at factual retrieval, they often struggle with the "curse of two-hop reasoning" in compositional tasks. Recent research suggests that parameter-sharing transformers can bridge this gap by forming a "Generalization Circuit" during a prolonged "grokking" phase. A fundamental question arises: Is a grokked model superior to its non-grokked counterparts on downstream tasks? Furthermore, is the extensive computational cost of waiting for the grokking phase worthwhile? In this work, we conduct a mechanistic study to evaluate the Generalization Circuit's role in knowledge assimilation and transfer. We demonstrate that: (i) The inference paths established by non-grokked and grokked models for in-distribution compositional queries are identical. This suggests that the "Generalization Circuit" does not represent the sudden acquisition of a new reasoning paradigm. Instead, we argue that grokking is the process of integrating memorized atomic facts into an naturally established reasoning path. (ii) Achieving high accuracy on unseen cases after prolonged training and the formation of a certain reasoning path are not bound; they can occur independently under specific data regimes. (iii) Even a mature circuit exhibits limited transferability when integrating new knowledge, suggesting that "grokked" Transformers do not achieve a full mastery of compositional logic.
>
---
#### [new 026] A.X K1 Technical Report
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍A.X K1，一个519B参数的Mixture-of-Experts语言模型，旨在提升推理能力和推理效率。通过优化训练配置和词汇量，在固定计算预算下实现高效训练。模型支持可控推理，适用于多种实际场景。**

- **链接: [https://arxiv.org/pdf/2601.09200v1](https://arxiv.org/pdf/2601.09200v1)**

> **作者:** Sung Jun Cheon; Jaekyung Cho; Seongho Choi; Hyunjun Eun; Seokhwan Jo; Jaehyun Jun; Minsoo Kang; Jin Kim; Jiwon Kim; Minsang Kim; Sungwan Kim; Seungsik Kim; Tae Yoon Kim; Youngrang Kim; Hyeongmun Lee; Sangyeol Lee; Sungeun Lee; Youngsoon Lee; Yujin Lee; Seongmin Ok; Chanyong Park; Hyewoong Park; Junyoung Park; Hyunho Yang; Subin Yi; Soohyun Bae; Dhammiko Arya; Yongseok Choi; Sangho Choi; Dongyeon Cho; Seungmo Cho; Gyoungeun Han; Yong-jin Han; Seokyoung Hong; Hyeon Hwang; Wonbeom Jang; Minjeong Ju; Wonjin Jung; Keummin Ka; Sungil Kang; Dongnam Kim; Joonghoon Kim; Jonghwi Kim; SaeRom Kim; Sangjin Kim; Seongwon Kim; Youngjin Kim; Seojin Lee; Sunwoo Lee; Taehoon Lee; Chanwoo Park; Sohee Park; Sooyeon Park; Yohan Ra; Sereimony Sek; Seungyeon Seo; Gun Song; Sanghoon Woo; Janghan Yoon; Sungbin Yoon
>
> **摘要:** We introduce A.X K1, a 519B-parameter Mixture-of-Experts (MoE) language model trained from scratch. Our design leverages scaling laws to optimize training configurations and vocabulary size under fixed computational budgets. A.X K1 is pre-trained on a corpus of approximately 10T tokens, curated by a multi-stage data processing pipeline. Designed to bridge the gap between reasoning capability and inference efficiency, A.X K1 supports explicitly controllable reasoning to facilitate scalable deployment across diverse real-world scenarios. We propose a simple yet effective Think-Fusion training recipe, enabling user-controlled switching between thinking and non-thinking modes within a single unified model. Extensive evaluations demonstrate that A.X K1 achieves performance competitive with leading open-source models, while establishing a distinctive advantage in Korean-language benchmarks.
>
---
#### [new 027] Empathy Applicability Modeling for General Health Queries
- **分类: cs.CL**

- **简介: 该论文属于医疗NLP任务，旨在解决LLMs缺乏临床共情的问题。提出EAF框架，通过分析患者查询预测共情需求，提升医疗沟通的同理心。**

- **链接: [https://arxiv.org/pdf/2601.09696v1](https://arxiv.org/pdf/2601.09696v1)**

> **作者:** Shan Randhawa; Agha Ali Raza; Kentaro Toyama; Julie Hui; Mustafa Naseem
>
> **备注:** In Submission to ACL
>
> **摘要:** LLMs are increasingly being integrated into clinical workflows, yet they often lack clinical empathy, an essential aspect of effective doctor-patient communication. Existing NLP frameworks focus on reactively labeling empathy in doctors' responses but offer limited support for anticipatory modeling of empathy needs, especially in general health queries. We introduce the Empathy Applicability Framework (EAF), a theory-driven approach that classifies patient queries in terms of the applicability of emotional reactions and interpretations, based on clinical, contextual, and linguistic cues. We release a benchmark of real patient queries, dual-annotated by Humans and GPT-4o. In the subset with human consensus, we also observe substantial human-GPT alignment. To validate EAF, we train classifiers on human-labeled and GPT-only annotations to predict empathy applicability, achieving strong performance and outperforming the heuristic and zero-shot LLM baselines. Error analysis highlights persistent challenges: implicit distress, clinical-severity ambiguity, and contextual hardship, underscoring the need for multi-annotator modeling, clinician-in-the-loop calibration, and culturally diverse annotation. EAF provides a framework for identifying empathy needs before response generation, establishes a benchmark for anticipatory empathy modeling, and enables supporting empathetic communication in asynchronous healthcare.
>
---
#### [new 028] Efficient Multilingual Dialogue Processing via Translation Pipelines and Distilled Language Models
- **分类: cs.CL**

- **简介: 该论文属于多语言对话摘要与问答任务，旨在解决低资源语言处理问题。通过翻译管道和蒸馏模型实现高效多语言处理。**

- **链接: [https://arxiv.org/pdf/2601.09059v1](https://arxiv.org/pdf/2601.09059v1)**

> **作者:** Santiago Martínez Novoa; Nicolás Rozo Fajardo; Diego Alejandro González Vargas; Nicolás Bedoya Figueroa
>
> **摘要:** This paper presents team Kl33n3x's multilingual dialogue summarization and question answering system developed for the NLPAI4Health 2025 shared task. The approach employs a three-stage pipeline: forward translation from Indic languages to English, multitask text generation using a 2.55B parameter distilled language model, and reverse translation back to source languages. By leveraging knowledge distillation techniques, this work demonstrates that compact models can achieve highly competitive performance across nine languages. The system achieved strong win rates across the competition's tasks, with particularly robust performance on Marathi (86.7% QnA), Tamil (86.7% QnA), and Hindi (80.0% QnA), demonstrating the effectiveness of translation-based approaches for low-resource language processing without task-specific fine-tuning.
>
---
#### [new 029] OrthoGeoLoRA: Geometric Parameter-Efficient Fine-Tuning for Structured Social Science Concept Retrieval on theWeb
- **分类: cs.CL**

- **简介: 该论文提出OrthoGeoLoRA方法，解决社会科学研究中基于网络的结构化概念检索问题，通过几何参数高效微调提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.09185v1](https://arxiv.org/pdf/2601.09185v1)**

> **作者:** Zeqiang Wang; Xinyue Wu; Chenxi Li; Zixi Chen; Nishanth Sastry; Jon Johnson; Suparna De
>
> **摘要:** Large language models and text encoders increasingly power web-based information systems in the social sciences, including digital libraries, data catalogues, and search interfaces used by researchers, policymakers, and civil society. Full fine-tuning is often computationally and energy intensive, which can be prohibitive for smaller institutions and non-profit organizations in the Web4Good ecosystem. Parameter-Efficient Fine-Tuning (PEFT), especially Low-Rank Adaptation (LoRA), reduces this cost by updating only a small number of parameters. We show that the standard LoRA update $ΔW = BA^\top$ has geometric drawbacks: gauge freedom, scale ambiguity, and a tendency toward rank collapse. We introduce OrthoGeoLoRA, which enforces an SVD-like form $ΔW = BΣA^\top$ by constraining the low-rank factors to be orthogonal (Stiefel manifold). A geometric reparameterization implements this constraint while remaining compatible with standard optimizers such as Adam and existing fine-tuning pipelines. We also propose a benchmark for hierarchical concept retrieval over the European Language Social Science Thesaurus (ELSST), widely used to organize social science resources in digital repositories. Experiments with a multilingual sentence encoder show that OrthoGeoLoRA outperforms standard LoRA and several strong PEFT variants on ranking metrics under the same low-rank budget, offering a more compute- and parameter-efficient path to adapt foundation models in resource-constrained settings.
>
---
#### [new 030] Adaptive Multi-Stage Patent Claim Generation with Unified Quality Assessment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于专利生成任务，解决跨司法管辖区泛化差、语义关系建模不足和质量评估不可靠的问题。提出三阶段框架，提升专利权利要求生成与评估效果。**

- **链接: [https://arxiv.org/pdf/2601.09120v1](https://arxiv.org/pdf/2601.09120v1)**

> **作者:** Chen-Wei Liang; Bin Guo; Zhen-Yuan Wei; Mu-Jiang-Shan Wang
>
> **备注:** 18 pages, 7 figures. Preprint
>
> **摘要:** Current patent claim generation systems face three fundamental limitations: poor cross-jurisdictional generalization, inadequate semantic relationship modeling between claims and prior art, and unreliable quality assessment. We introduce a novel three-stage framework that addresses these challenges through relationship-aware similarity analysis, domain-adaptive claim generation, and unified quality assessment. Our approach employs multi-head attention with eight specialized heads for explicit relationship modeling, integrates curriculum learning with dynamic LoRA adapter selection across five patent domains, and implements cross-attention mechanisms between evaluation aspects for comprehensive quality assessment. Extensive experiments on USPTO HUPD dataset, EPO patent collections, and Patent-CE benchmark demonstrate substantial improvements: 7.6-point ROUGE-L gain over GPT-4o, 8.3\% BERTScore enhancement over Llama-3.1-8B, and 0.847 correlation with human experts compared to 0.623 for separate evaluation models. Our method maintains 89.4\% cross-jurisdictional performance retention versus 76.2\% for baselines, establishing a comprehensive solution for automated patent prosecution workflows.
>
---
#### [new 031] SlidesGen-Bench: Evaluating Slides Generation via Computational and Quantitative Metrics
- **分类: cs.CL**

- **简介: 该论文属于幻灯片生成评估任务，旨在解决现有评价体系不统一、主观性强的问题。提出SlidesGen-Bench基准，通过内容、美观度和可编辑性三个维度进行量化评估，并构建了对齐人类偏好的数据集。**

- **链接: [https://arxiv.org/pdf/2601.09487v1](https://arxiv.org/pdf/2601.09487v1)**

> **作者:** Yunqiao Yang; Wenbo Li; Houxing Ren; Zimu Lu; Ke Wang; Zhiyuan Huang; Zhuofan Zong; Mingjie Zhan; Hongsheng Li
>
> **备注:** 37 pages, 34 figures
>
> **摘要:** The rapid evolution of Large Language Models (LLMs) has fostered diverse paradigms for automated slide generation, ranging from code-driven layouts to image-centric synthesis. However, evaluating these heterogeneous systems remains challenging, as existing protocols often struggle to provide comparable scores across architectures or rely on uncalibrated judgments. In this paper, we introduce SlidesGen-Bench, a benchmark designed to evaluate slide generation through a lens of three core principles: universality, quantification, and reliability. First, to establish a unified evaluation framework, we ground our analysis in the visual domain, treating terminal outputs as renderings to remain agnostic to the underlying generation method. Second, we propose a computational approach that quantitatively assesses slides across three distinct dimensions - Content, Aesthetics, and Editability - offering reproducible metrics where prior works relied on subjective or reference-dependent proxies. Finally, to ensure high correlation with human preference, we construct the Slides-Align1.5k dataset, a human preference aligned dataset covering slides from nine mainstream generation systems across seven scenarios. Our experiments demonstrate that SlidesGen-Bench achieves a higher degree of alignment with human judgment than existing evaluation pipelines. Our code and data are available at https://github.com/YunqiaoYang/SlidesGen-Bench.
>
---
#### [new 032] MCGA: A Multi-task Classical Chinese Literary Genre Audio Corpus
- **分类: cs.CL**

- **简介: 该论文提出MCGA，一个涵盖六种任务的古文语音语料库，旨在解决音频在古文研究中的不足，提升多模态大模型的音频处理能力。**

- **链接: [https://arxiv.org/pdf/2601.09270v1](https://arxiv.org/pdf/2601.09270v1)**

> **作者:** Yexing Du; Kaiyuan Liu; Bihe Zhang; Youcheng Pan; Bo Yang; Liangyu Huo; Xiyuan Zhang; Jian Xie; Daojing He; Yang Xiang; Ming Liu; Bin Qin
>
> **摘要:** With the rapid advancement of Multimodal Large Language Models (MLLMs), their potential has garnered significant attention in Chinese Classical Studies (CCS). While existing research has primarily focused on text and visual modalities, the audio corpus within this domain remains largely underexplored. To bridge this gap, we propose the Multi-task Classical Chinese Literary Genre Audio Corpus (MCGA). It encompasses a diverse range of literary genres across six tasks: Automatic Speech Recognition (ASR), Speech-to-Text Translation (S2TT), Speech Emotion Captioning (SEC), Spoken Question Answering (SQA), Speech Understanding (SU), and Speech Reasoning (SR). Through the evaluation of ten MLLMs, our experimental results demonstrate that current models still face substantial challenges when processed on the MCGA test set. Furthermore, we introduce an evaluation metric for SEC and a metric to measure the consistency between the speech and text capabilities of MLLMs. We release MCGA and our code to the public to facilitate the development of MLLMs with more robust multidimensional audio capabilities in CCS. MCGA Corpus: https://github.com/yxduir/MCGA
>
---
#### [new 033] Beyond Consensus: Perspectivist Modeling and Evaluation of Annotator Disagreement in NLP
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，解决标注者意见不一致的问题。通过分析 disagreement 的来源，提出统一的建模与评估框架，推动对不同视角的建模与理解。**

- **链接: [https://arxiv.org/pdf/2601.09065v1](https://arxiv.org/pdf/2601.09065v1)**

> **作者:** Yinuo Xu; David Jurgens
>
> **摘要:** Annotator disagreement is widespread in NLP, particularly for subjective and ambiguous tasks such as toxicity detection and stance analysis. While early approaches treated disagreement as noise to be removed, recent work increasingly models it as a meaningful signal reflecting variation in interpretation and perspective. This survey provides a unified view of disagreement-aware NLP methods. We first present a domain-agnostic taxonomy of the sources of disagreement spanning data, task, and annotator factors. We then synthesize modeling approaches using a common framework defined by prediction targets and pooling structure, highlighting a shift from consensus learning toward explicitly modeling disagreement, and toward capturing structured relationships among annotators. We review evaluation metrics for both predictive performance and annotator behavior, and noting that most fairness evaluations remain descriptive rather than normative. We conclude by identifying open challenges and future directions, including integrating multiple sources of variation, developing disagreement-aware interpretability frameworks, and grappling with the practical tradeoffs of perspectivist modeling.
>
---
#### [new 034] Directional Attractors in LLM Reasoning: How Similarity Retrieval Steers Iterative Summarization Based Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究LLM推理中的方向性吸引子问题，通过语义缓存提升迭代摘要推理的准确性，解决重复策略和上下文扩展问题。**

- **链接: [https://arxiv.org/pdf/2601.08846v1](https://arxiv.org/pdf/2601.08846v1)**

> **作者:** Cagatay Tekin; Charbel Barakat; Luis Joseph Luna Limgenco
>
> **备注:** 6 pages, 2 figures. Code available at: github.com/cagopat/InftyThink-with-Cross-Chain-Memory
>
> **摘要:** Iterative summarization based reasoning frameworks such as InftyThink enable long-horizon reasoning in large language models (LLMs) by controlling context growth, but they repeatedly regenerate similar reasoning strategies across tasks. We introduce InftyThink with Cross-Chain Memory, an extension that augments iterative reasoning with an embedding-based semantic cache of previously successful reasoning patterns. At each reasoning step, the model retrieves and conditions on the most semantically similar stored lemmas, guiding inference without expanding the context window indiscriminately. Experiments on MATH500, AIME2024, and GPQA-Diamond demonstrate that semantic lemma retrieval improves accuracy in structured domains while exposing failure modes in tests that include heterogeneous domains. Geometric analyses of reasoning trajectories reveal that cache retrieval induces directional biases in embedding space, leading to consistent fix (improve baseline accuracy) and break (degradation in baseline accuracy) attractors. Our results highlight both the benefits and limits of similarity-based memory for self-improving LLM reasoning.
>
---
#### [new 035] From Symbolic to Natural-Language Relations: Rethinking Knowledge Graph Construction in the Era of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱构建任务，旨在解决传统符号关系表示的局限性。论文提出使用自然语言描述关系，以更灵活地表达复杂语义。**

- **链接: [https://arxiv.org/pdf/2601.09069v1](https://arxiv.org/pdf/2601.09069v1)**

> **作者:** Kanyao Han; Yushang Lai
>
> **摘要:** Knowledge graphs (KGs) have commonly been constructed using predefined symbolic relation schemas, typically implemented as categorical relation labels. This design has notable shortcomings: real-world relations are often contextual, nuanced, and sometimes uncertain, and compressing it into discrete relation labels abstracts away critical semantic detail. Nevertheless, symbolic-relation KGs remain widely used because they have been operationally effective and broadly compatible with pre-LLM downstream models and algorithms, in which KG knowledge could be retrieved or encoded into quantified features and embeddings at scale. The emergence of LLMs has reshaped how knowledge is created and consumed. LLMs support scalable synthesis of domain facts directly in concise natural language, and prompting-based inference favors context-rich free-form text over quantified representations. This position paper argues that these changes call for rethinking the representation of relations themselves rather than merely using LLMs to populate conventional schemas more efficiently. We therefore advocate moving from symbolic to natural-language relation descriptions, and we propose hybrid design principles that preserve a minimal structural backbone while enabling more flexible and context-sensitive relational representations.
>
---
#### [new 036] Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出ITP框架，解决智能体复杂任务规划问题。通过自适应展望机制，结合世界模型生成多步想象轨迹，提升推理能力。**

- **链接: [https://arxiv.org/pdf/2601.08955v1](https://arxiv.org/pdf/2601.08955v1)**

> **作者:** Youwei Liu; Jian Wang; Hanlin Wang; Beichen Guo; Wenjie Li
>
> **摘要:** Recent advances in world models have shown promise for modeling future dynamics of environmental states, enabling agents to reason and act without accessing real environments. Current methods mainly perform single-step or fixed-horizon rollouts, leaving their potential for complex task planning under-exploited. We propose Imagine-then-Plan (\texttt{ITP}), a unified framework for agent learning via lookahead imagination, where an agent's policy model interacts with the learned world model, yielding multi-step ``imagined'' trajectories. Since the imagination horizon may vary by tasks and stages, we introduce a novel adaptive lookahead mechanism by trading off the ultimate goal and task progress. The resulting imagined trajectories provide rich signals about future consequences, such as achieved progress and potential conflicts, which are fused with current observations, formulating a partially \textit{observable} and \textit{imaginable} Markov decision process to guide policy learning. We instantiate \texttt{ITP} with both training-free and reinforcement-trained variants. Extensive experiments across representative agent benchmarks demonstrate that \texttt{ITP} significantly outperforms competitive baselines. Further analyses validate that our adaptive lookahead largely enhances agents' reasoning capability, providing valuable insights into addressing broader, complex tasks.
>
---
#### [new 037] How Many Human Judgments Are Enough? Feasibility Limits of Human Preference Evaluation
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究人类偏好评估的可行性，解决如何确定足够判断数量以可靠检测模型改进的问题。通过分析数据，发现多数情况需更多判断，提出优化策略。**

- **链接: [https://arxiv.org/pdf/2601.09084v1](https://arxiv.org/pdf/2601.09084v1)**

> **作者:** Wilson Y. Lee
>
> **摘要:** Human preference evaluations are widely used to compare generative models, yet it remains unclear how many judgments are required to reliably detect small improvements. We show that when preference signal is diffuse across prompts (i.e., all prompt types are similarly informative), proportional allocation is minimax-optimal: no allocation strategy substantially improves detectability. Empirical analysis of large-scale human preference datasets shows that most comparisons fall into this diffuse regime, exhibiting small preference margins that require far more judgments than typically collected, even in well-sampled comparisons. These limits persist across evaluation protocols and modalities, including chat, image generation, and code generation with execution feedback. In contrast, curated benchmarks that reduce prompt induced variability systematically induce larger margins and improve detectability through a $1.5\times$ reduction in prompt-level variance. Our results show that inconclusive or negative human evaluation outcomes frequently reflect underpowered evaluation rather than model equivalence, underscoring the need to account explicitly for effect size, budget, and protocol design.
>
---
#### [new 038] SubTokenTest: A Practical Benchmark for Real-World Sub-token Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在解决大语言模型在子词理解上的不足。通过构建SubTokenTest基准，评估模型在实际任务中的子词理解能力，并分析其表现及改进方法。**

- **链接: [https://arxiv.org/pdf/2601.09089v1](https://arxiv.org/pdf/2601.09089v1)**

> **作者:** Shuyang Hou; Yi Hu; Muhan Zhang
>
> **摘要:** Recent advancements in large language models (LLMs) have significantly enhanced their reasoning capabilities. However, they continue to struggle with basic character-level tasks, such as counting letters in words, a problem rooted in their tokenization process. While existing benchmarks have highlighted this weakness through basic character operations, such failures are often dismissed due to lacking practical relevance. Yet, many real-world applications, such as navigating text-based maps or interpreting structured tables, rely heavily on precise sub-token understanding. In this regard, we introduce SubTokenTest, a comprehensive benchmark that assesses sub-token understanding through practical, utility-driven tasks. Our benchmark includes ten tasks across four domains and isolates tokenization-related failures by decoupling performance from complex reasoning. We provide a comprehensive evaluation of nine advanced LLMs. Additionally, we investigate the impact of test-time scaling on sub-token reasoning and explore how character-level information is encoded within the hidden states.
>
---
#### [new 039] Understanding or Memorizing? A Case Study of German Definite Articles in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型是否通过规则或记忆处理德语定冠词。任务是区分规则泛化与记忆关联。通过梯度方法分析参数更新，发现模型依赖记忆而非抽象规则。**

- **链接: [https://arxiv.org/pdf/2601.09313v1](https://arxiv.org/pdf/2601.09313v1)**

> **作者:** Jonathan Drechsel; Erisa Bytyqi; Steffen Herbold
>
> **摘要:** Language models perform well on grammatical agreement, but it is unclear whether this reflects rule-based generalization or memorization. We study this question for German definite singular articles, whose forms depend on gender and case. Using GRADIEND, a gradient-based interpretability method, we learn parameter update directions for gender-case specific article transitions. We find that updates learned for a specific gender-case article transition frequently affect unrelated gender-case settings, with substantial overlap among the most affected neurons across settings. These results argue against a strictly rule-based encoding of German definite articles, indicating that models at least partly rely on memorized associations rather than abstract grammatical rules.
>
---
#### [new 040] LLMs Got Rhythm? Hybrid Phonological Filtering for Greek Poetry Rhyme Detection and Generation
- **分类: cs.CL**

- **简介: 该论文属于诗歌韵律检测与生成任务，解决LLMs在希腊语韵律处理上的不足。通过结合LLM与音系算法，提升韵律识别与生成的准确性。**

- **链接: [https://arxiv.org/pdf/2601.09631v1](https://arxiv.org/pdf/2601.09631v1)**

> **作者:** Stergios Chatzikyriakidis
>
> **摘要:** Large Language Models (LLMs), despite their remarkable capabilities across NLP tasks, struggle with phonologically-grounded phenomena like rhyme detection and generation. This is even more evident in lower-resource languages such as Modern Greek. In this paper, we present a hybrid system that combines LLMs with deterministic phonological algorithms to achieve accurate rhyme identification/analysis and generation. Our approach implements a comprehensive taxonomy of Greek rhyme types, including Pure, Rich, Imperfect, Mosaic, and Identical Pre-rhyme Vowel (IDV) patterns, and employs an agentic generation pipeline with phonological verification. We evaluate multiple prompting strategies (zero-shot, few-shot, Chain-of-Thought, and RAG-augmented) across several LLMs including Claude 3.7 and 4.5, GPT-4o, Gemini 2.0 and open-weight models like Llama 3.1 8B and 70B and Mistral Large. Results reveal a significant "Reasoning Gap": while native-like models (Claude 3.7) perform intuitively (40\% accuracy in identification), reasoning-heavy models (Claude 4.5) achieve state-of-the-art performance (54\%) only when prompted with Chain-of-Thought. Most critically, pure LLM generation fails catastrophically (under 4\% valid poems), while our hybrid verification loop restores performance to 73.1\%. We release our system and a crucial, rigorously cleaned corpus of 40,000+ rhymes, derived from the Anemoskala and Interwar Poetry corpora, to support future research.
>
---
#### [new 041] TaxoBell: Gaussian Box Embeddings for Self-Supervised Taxonomy Expansion
- **分类: cs.CL**

- **简介: 该论文属于知识图谱任务，解决taxonomy扩展问题。针对传统方法在处理非对称关系和不确定性上的不足，提出TaxoBell框架，利用高斯盒嵌入提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.09633v1](https://arxiv.org/pdf/2601.09633v1)**

> **作者:** Sahil Mishra; Srinitish Srinivasan; Srikanta Bedathur; Tanmoy Chakraborty
>
> **备注:** Accepted in The Web Conference (WWW) 2026
>
> **摘要:** Taxonomies form the backbone of structured knowledge representation across diverse domains, enabling applications such as e-commerce catalogs, semantic search, and biomedical discovery. Yet, manual taxonomy expansion is labor-intensive and cannot keep pace with the emergence of new concepts. Existing automated methods rely on point-based vector embeddings, which model symmetric similarity and thus struggle with the asymmetric "is-a" relationships that are fundamental to taxonomies. Box embeddings offer a promising alternative by enabling containment and disjointness, but they face key issues: (i) unstable gradients at the intersection boundaries, (ii) no notion of semantic uncertainty, and (iii) limited capacity to represent polysemy or ambiguity. We address these shortcomings with TaxoBell, a Gaussian box embedding framework that translates between box geometries and multivariate Gaussian distributions, where means encode semantic location and covariances encode uncertainty. Energy-based optimization yields stable optimization, robust modeling of ambiguous concepts, and interpretable hierarchical reasoning. Extensive experimentation on five benchmark datasets demonstrates that TaxoBell significantly outperforms eight state-of-the-art taxonomy expansion baselines by 19% in MRR and around 25% in Recall@k. We further demonstrate the advantages and pitfalls of TaxoBell with error analysis and ablation studies.
>
---
#### [new 042] PediaMind-R1: A Temperament-Aware Language Model for Personalized Early Childhood Care Reasoning via Cognitive Modeling and Preference Alignment
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PediaMind-R1，用于个性化育儿推理。任务是解决早期儿童照护中的个性化问题，通过整合 temperament 理论与语言模型，提升 caregiving 的精准性与情感共鸣。**

- **链接: [https://arxiv.org/pdf/2601.08848v1](https://arxiv.org/pdf/2601.08848v1)**

> **作者:** Zihe Zhang; Can Zhang; Yanheng Xu; Xin Hu; Jichao Leng
>
> **备注:** Accepted at EMNLP 2025 PALS Workshop (PALS: EXPLORING ACTIVE AND PASSIVE LLM PERSONALIZATION)
>
> **摘要:** This paper presents PediaMind-R1, a domain-specialized large language model designed to achieve active personalization in intelligent parenting scenarios. Unlike conventional systems that provide generic suggestions, PediaMind-R1 draws on insights from developmental psychology. It introduces temperament theory from the Thomas-Chess framework and builds a temperament knowledge graph for infants and toddlers (0-3 years). Our two-stage training pipeline first uses supervised fine-tuning to teach structured chain-of-thought reasoning, and then applies a GRPO-based alignment stage to reinforce logical consistency, domain expertise, and empathetic caregiving strategies. We further design an evaluation framework comprising temperament-sensitive multiple-choice tests and human assessments. The results demonstrate that PediaMind-R1 can accurately interpret early childhood temperament profiles and proactively engage in individualized reasoning. This work highlights the value of integrating vertical-domain modeling with psychological theory. It offers a novel approach to developing user-centered LLMs that advance the practice of active personalization in sensitive caregiving contexts.
>
---
#### [new 043] When to Invoke: Refining LLM Fairness with Toxicity Assessment
- **分类: cs.CL**

- **简介: 该论文属于语言模型公平性任务，旨在解决LLM在毒性评估中的偏见问题。通过引入FairToT框架，提升评估公平性与一致性。**

- **链接: [https://arxiv.org/pdf/2601.09250v1](https://arxiv.org/pdf/2601.09250v1)**

> **作者:** Jing Ren; Bowen Li; Ziqi Xu; Renqiang Luo; Shuo Yu; Xin Ye; Haytham Fayek; Xiaodong Li; Feng Xia
>
> **备注:** Accepted by Findings of WWW 2026
>
> **摘要:** Large Language Models (LLMs) are increasingly used for toxicity assessment in online moderation systems, where fairness across demographic groups is essential for equitable treatment. However, LLMs often produce inconsistent toxicity judgements for subtle expressions, particularly those involving implicit hate speech, revealing underlying biases that are difficult to correct through standard training. This raises a key question that existing approaches often overlook: when should corrective mechanisms be invoked to ensure fair and reliable assessments? To address this, we propose FairToT, an inference-time framework that enhances LLM fairness through prompt-guided toxicity assessment. FairToT identifies cases where demographic-related variation is likely to occur and determines when additional assessment should be applied. In addition, we introduce two interpretable fairness indicators that detect such cases and improve inference consistency without modifying model parameters. Experiments on benchmark datasets show that FairToT reduces group-level disparities while maintaining stable and reliable toxicity predictions, demonstrating that inference-time refinement offers an effective and practical approach for fairness improvement in LLM-based toxicity assessment systems. The source code can be found at https://aisuko.github.io/fair-tot/.
>
---
#### [new 044] Scalable and Reliable Evaluation of AI Knowledge Retrieval Systems: RIKER and the Coherent Simulated Universe
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出RIKER，用于评估AI知识检索系统。解决静态基准易污染、人工标注成本高问题，通过生成文档进行可扩展、抗污染的评估。**

- **链接: [https://arxiv.org/pdf/2601.08847v1](https://arxiv.org/pdf/2601.08847v1)**

> **作者:** JV Roig
>
> **备注:** 25 pages, 17 tables, 1 figure
>
> **摘要:** Evaluating knowledge systems (LLMs, RAG, knowledge graphs, etc) faces fundamental challenges: static benchmarks are vulnerable to contamination, LLM-based judges exhibit systematic biases, and ground truth extraction requires expensive human annotation. We present RIKER (Retrieval Intelligence and Knowledge Extraction Rating), both a benchmark and a replicable methodology based on paradigm inversion - generating documents from known ground truth rather than extracting ground truth from documents. This approach enables deterministic scoring and scalable evaluation without human annotation or reference models, and contamination resistance through regenerable corpora. Our evaluation of 33 models using over 21 billion tokens reveals that context length claims frequently exceed usable capacity, with significant degradation beyond 32K tokens; cross-document aggregation proves substantially harder than single-document extraction; and grounding ability and hallucination resistance are distinct capabilities - models excelling at finding facts that exist may still fabricate facts that do not. Beyond the specific benchmark, we contribute a domain-agnostic methodology for constructing scalable and contamination-resistant evaluations wherever synthetic documents can be generated from structured ground truth.
>
---
#### [new 045] Dialogue Telemetry: Turn-Level Instrumentation for Autonomous Information Gathering
- **分类: cs.CL**

- **简介: 该论文属于对话系统任务，解决自主信息收集对话中缺乏逐轮观测的问题。提出Dialogue Telemetry框架，生成Progress Estimator和Stalling Index信号，用于监测效率和检测无效提问。**

- **链接: [https://arxiv.org/pdf/2601.09570v1](https://arxiv.org/pdf/2601.09570v1)**

> **作者:** Dimitris Panagopoulos; Adolfo Perrusquia; Weisi Guo
>
> **备注:** 16 pages, 9 Figures, Version submitted to IEEE for publication
>
> **摘要:** Autonomous systems conducting schema-grounded information-gathering dialogues face an instrumentation gap, lacking turn-level observables for monitoring acquisition efficiency and detecting when questioning becomes unproductive. We introduce Dialogue Telemetry (DT), a measurement framework that produces two model-agnostic signals after each question-answer exchange: (i) a Progress Estimator (PE) quantifying residual information potential per category (with a bits-based variant), and (ii) a Stalling Index (SI) detecting an observable failure signature characterized by repeated category probing with semantically similar, low-marginal-gain responses. SI flags this pattern without requiring causal diagnosis, supporting monitoring in settings where attributing degradation to specific causes may be impractical. We validate DT in controlled search-and-rescue (SAR)-inspired interviews using large language model (LLM)-based simulations, distinguishing efficient from stalled dialogue traces and illustrating downstream utility by integrating DT signals into a reinforcement learning (RL) policy. Across these settings, DT provides interpretable turn-level instrumentation that improves policy performance when stalling carries operational costs.
>
---
#### [new 046] ReGraM: Region-First Knowledge Graph Reasoning for Medical Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗问答任务，旨在提升事实准确性。针对现有方法噪声大、推理不稳定的问题，提出ReGraM框架，通过构建查询对齐子图进行局部推理，有效减少错误。**

- **链接: [https://arxiv.org/pdf/2601.09280v1](https://arxiv.org/pdf/2601.09280v1)**

> **作者:** Chaerin Lee; Sohee Park; Hyunsik Na; Daseon Choi
>
> **备注:** 18 pages, 2 figures. Preprint
>
> **摘要:** Recent studies in medical question answering (Medical QA) have actively explored the integration of large language models (LLMs) with biomedical knowledge graphs (KGs) to improve factual accuracy. However, most existing approaches still rely on traversing the entire KG or performing large-scale retrieval, which introduces substantial noise and leads to unstable multi-hop reasoning. We argue that the core challenge lies not in expanding access to knowledge, but in identifying and reasoning over the appropriate subset of evidence for each query. ReGraM is a region-first knowledge graph reasoning framework that addresses this challenge by constructing a query-aligned subgraph and performing stepwise reasoning constrained to this localized region under multiple evidence aware modes. By focusing inference on only the most relevant portion of the KG, ReGraM departs from the assumption that all relations are equally useful an assumption that rarely holds in domain-specific medical settings. Experiments on seven medical QA benchmarks demonstrate that ReGraM consistently outperforms a strong baseline (KGARevion), achieving an 8.04% absolute accuracy gain on MCQ, a 4.50% gain on SAQ, and a 42.9% reduction in hallucination rate. Ablation and qualitative analyses further show that aligning region construction with hop-wise reasoning is the primary driver of these improvements. Overall, our results highlight region-first KG reasoning as an effective paradigm for improving factual accuracy and consistency in medical QA.
>
---
#### [new 047] Triples and Knowledge-Infused Embeddings for Clustering and Classification of Scientific Documents
- **分类: cs.CL; cs.AI; cs.DL**

- **简介: 该论文属于科学文献的聚类与分类任务，旨在提升文献组织与理解效果。通过结合文本与结构化知识三元组，改进模型性能，实验表明混合表示在分类中表现最佳。**

- **链接: [https://arxiv.org/pdf/2601.08841v1](https://arxiv.org/pdf/2601.08841v1)**

> **作者:** Mihael Arcan
>
> **摘要:** The increasing volume and complexity of scientific literature demand robust methods for organizing and understanding research documents. In this study, we explore how structured knowledge, specifically, subject-predicate-object triples, can enhance the clustering and classification of scientific papers. We propose a modular pipeline that combines unsupervised clustering and supervised classification over multiple document representations: raw abstracts, extracted triples, and hybrid formats that integrate both. Using a filtered arXiv corpus, we extract relational triples from abstracts and construct four text representations, which we embed using four state-of-the-art transformer models: MiniLM, MPNet, SciBERT, and SPECTER. We evaluate the resulting embeddings with KMeans, GMM, and HDBSCAN for unsupervised clustering, and fine-tune classification models for arXiv subject prediction. Our results show that full abstract text yields the most coherent clusters, but that hybrid representations incorporating triples consistently improve classification performance, reaching up to 92.6% accuracy and 0.925 macro-F1. We also find that lightweight sentence encoders (MiniLM, MPNet) outperform domain-specific models (SciBERT, SPECTER) in clustering, while SciBERT excels in structured-input classification. These findings highlight the complementary benefits of combining unstructured text with structured knowledge, offering new insights into knowledge-infused representations for semantic organization of scientific documents.
>
---
#### [new 048] Multicultural Spyfall: Assessing LLMs through Dynamic Multilingual Social Deduction Game
- **分类: cs.CL**

- **简介: 该论文属于多语言多文化评估任务，旨在解决LLMs在非英语场景下的表现不足问题。通过社交推理游戏Spyfall测试模型的跨文化能力。**

- **链接: [https://arxiv.org/pdf/2601.09017v1](https://arxiv.org/pdf/2601.09017v1)**

> **作者:** Haryo Akbarianto Wibowo; Alaa Elsetohy; Qinrong Cui; Alham Fikri Aji
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has necessitated more robust evaluation methods that go beyond static benchmarks, which are increasingly prone to data saturation and leakage. In this paper, we propose a dynamic benchmarking framework for evaluating multilingual and multicultural capabilities through the social deduction game Spyfall. In our setup, models must engage in strategic dialogue to either identify a secret agent or avoid detection, utilizing culturally relevant locations or local foods. Our results show that our game-based rankings align closely with the Chatbot Arena. However, we find a significant performance gap in non-English contexts: models are generally less proficient when handling locally specific entities and often struggle with rule-following or strategic integrity in non-English languages. We demonstrate that this game-based approach provides a scalable, leakage-resistant, and culturally nuanced alternative to traditional NLP benchmarks. The game history can be accessed here https://huggingface.co/datasets/haryoaw/cultural-spyfall.
>
---
#### [new 049] Gaming the Answer Matcher: Examining the Impact of Text Manipulation on Automated Judgment
- **分类: cs.CL**

- **简介: 该论文属于自动化答案匹配任务，旨在检验文本操纵对模型评分的影响。研究发现，常见策略如冗长回答等无法提升得分，且二分类评分更稳健。**

- **链接: [https://arxiv.org/pdf/2601.08849v1](https://arxiv.org/pdf/2601.08849v1)**

> **作者:** Manas Khatore; Sumana Sridharan; Kevork Sulahian; Benjamin J. Smith; Shi Feng
>
> **备注:** Accepted to the AAAI 2026 Workshop on AI Governance (AIGOV)
>
> **摘要:** Automated answer matching, which leverages LLMs to evaluate free-text responses by comparing them to a reference answer, shows substantial promise as a scalable and aligned alternative to human evaluation. However, its reliability requires robustness against strategic attacks such as guesswork or verbosity that may artificially inflate scores without improving actual correctness. In this work, we systematically investigate whether such tactics deceive answer matching models by prompting examinee models to: (1) generate verbose responses, (2) provide multiple answers when unconfident, and (3) embed conflicting answers with the correct answer near the start of their response. Our results show that these manipulations do not increase scores and often reduce them. Additionally, binary scoring (which requires a matcher to answer with a definitive "correct" or "incorrect") is more robust to attacks than continuous scoring (which requires a matcher to determine partial correctness). These findings show that answer matching is generally robust to inexpensive text manipulation and is a viable alternative to traditional LLM-as-a-judge or human evaluation when reference answers are available.
>
---
#### [new 050] Entropy Sentinel: Continuous LLM Accuracy Monitoring from Decoding Entropy Traces in STEM
- **分类: cs.CL**

- **简介: 该论文属于模型监控任务，旨在解决LLM在领域漂移下的准确性评估与数据采集优化问题。通过分析解码熵迹，提取统计特征并预测实例正确性，实现模型性能的持续监控与改进。**

- **链接: [https://arxiv.org/pdf/2601.09001v1](https://arxiv.org/pdf/2601.09001v1)**

> **作者:** Pedro Memoli Buffa; Luciano Del Corro
>
> **摘要:** Deploying LLMs raises two coupled challenges: (1) monitoring - estimating where a model underperforms as traffic and domains drift - and (2) improvement - prioritizing data acquisition to close the largest performance gaps. We test whether an inference-time signal can estimate slice-level accuracy under domain shift. For each response, we compute an output-entropy profile from final-layer next-token probabilities (from top-k logprobs) and summarize it with eleven statistics. A lightweight classifier predicts instance correctness, and averaging predicted probabilities yields a domain-level accuracy estimate. We evaluate on ten STEM reasoning benchmarks with exhaustive train/test compositions (k in {1,2,3,4}; all "10 choose k" combinations), across nine LLMs from six families (3B-20B). Estimates often track held-out benchmark accuracy, and several models show near-monotonic ordering of domains. Output-entropy profiles are thus an accessible signal for scalable monitoring and for targeting data acquisition.
>
---
#### [new 051] Value-Aware Numerical Representations for Transformer Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决语言模型在数值理解上的不足。通过引入值感知的数值表示，增强模型对数字的准确理解与计算能力。**

- **链接: [https://arxiv.org/pdf/2601.09706v1](https://arxiv.org/pdf/2601.09706v1)**

> **作者:** Andreea Dutulescu; Stefan Ruseti; Mihai Dascalu
>
> **摘要:** Transformer-based language models often achieve strong results on mathematical reasoning benchmarks while remaining fragile on basic numerical understanding and arithmetic operations. A central limitation is that numbers are processed as symbolic tokens whose embeddings do not explicitly encode numerical value, leading to systematic errors. We introduce a value-aware numerical representation that augments standard tokenized inputs with a dedicated prefix token whose embedding is explicitly conditioned on the underlying numerical value. This mechanism injects magnitude information directly into the model's input space while remaining compatible with existing tokenizers and decoder-only Transformer architectures. Evaluation on arithmetic tasks shows that the proposed approach outperforms baselines across numerical formats, tasks, and operand lengths. These results indicate that explicitly encoding numerical value is an effective and efficient way to improve fundamental numerical robustness in language models.
>
---
#### [new 052] Emissions and Performance Trade-off Between Small and Large Language Models
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 该论文属于自然语言处理领域，探讨LLMs与SLMs在性能与碳排放间的权衡，旨在寻找可持续的AI解决方案。研究对比分析了两者在多个任务中的表现与能耗，证明SLMs可在保持性能的同时显著降低碳排放。**

- **链接: [https://arxiv.org/pdf/2601.08844v1](https://arxiv.org/pdf/2601.08844v1)**

> **作者:** Anandita Garg; Uma Gaba; Deepan Muthirayan; Anish Roy Chowdhury
>
> **备注:** 6 pages. Accepted as a full paper to the 3rd International Conference on Foundation and Large Language Models (IEEE FLLM) 2025
>
> **摘要:** The advent of Large Language Models (LLMs) has raised concerns about their enormous carbon footprint, starting with energy-intensive training and continuing through repeated inference. This study investigates the potential of using fine-tuned Small Language Models (SLMs) as a sustainable alternative for predefined tasks. Here, we present a comparative analysis of the performance-emissions trade-off between LLMs and fine-tuned SLMs across selected tasks under Natural Language Processing, Reasoning and Programming. Our results show that in four out of the six selected tasks, SLMs maintained comparable performances for a significant reduction in carbon emissions during inference. Our findings demonstrate the viability of smaller models in mitigating the environmental impact of resource-heavy LLMs, thus advancing towards sustainable, green AI.
>
---
#### [new 053] From Adversarial Poetry to Adversarial Tales: An Interpretability Research Agenda
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 该论文属于安全与可解释性研究任务，旨在解决LLM对结构化攻击的脆弱性问题。通过构建叙事类攻击，揭示模型在理解隐含意图上的缺陷，并提出可解释性研究方向。**

- **链接: [https://arxiv.org/pdf/2601.08837v1](https://arxiv.org/pdf/2601.08837v1)**

> **作者:** Piercosma Bisconti; Marcello Galisai; Matteo Prandi; Federico Pierucci; Olga Sorokoletova; Francesco Giarrusso; Vincenzo Suriani; Marcantonio Brancale; Daniele Nardi
>
> **摘要:** Safety mechanisms in LLMs remain vulnerable to attacks that reframe harmful requests through culturally coded structures. We introduce Adversarial Tales, a jailbreak technique that embeds harmful content within cyberpunk narratives and prompts models to perform functional analysis inspired by Vladimir Propp's morphology of folktales. By casting the task as structural decomposition, the attack induces models to reconstruct harmful procedures as legitimate narrative interpretation. Across 26 frontier models from nine providers, we observe an average attack success rate of 71.3%, with no model family proving reliably robust. Together with our prior work on Adversarial Poetry, these findings suggest that structurally-grounded jailbreaks constitute a broad vulnerability class rather than isolated techniques. The space of culturally coded frames that can mediate harmful intent is vast, likely inexhaustible by pattern-matching defenses alone. Understanding why these attacks succeed is therefore essential: we outline a mechanistic interpretability research agenda to investigate how narrative cues reshape model representations and whether models can learn to recognize harmful intent independently of surface form.
>
---
#### [new 054] Ability Transfer and Recovery via Modularized Parameters Localization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型在持续预训练或微调后出现的能力退化问题，提出ACT方法通过定位关键参数通道实现能力迁移与恢复，属于模型优化任务。**

- **链接: [https://arxiv.org/pdf/2601.09398v1](https://arxiv.org/pdf/2601.09398v1)**

> **作者:** Songyao Jin; Kun Zhou; Wenqi Li; Peng Wang; Biwei Huang
>
> **摘要:** Large language models can be continually pre-trained or fine-tuned to improve performance in specific domains, languages, or skills, but this specialization often degrades other capabilities and may cause catastrophic forgetting. We investigate how abilities are distributed within LLM parameters by analyzing module activations under domain- and language-specific inputs for closely related models. Across layers and modules, we find that ability-related activations are highly concentrated in a small set of channels (typically <5\%), and these channels are largely disentangled with good sufficiency and stability. Building on these observations, we propose ACT (Activation-Guided Channel-wise Ability Transfer), which localizes ability-relevant channels via activation differences and selectively transfers only the corresponding parameters, followed by lightweight fine-tuning for compatibility. Experiments on multilingual mathematical and scientific reasoning show that ACT can recover forgotten abilities while preserving retained skills. It can also merge multiple specialized models to integrate several abilities into a single model with minimal interference. Our code and data will be publicly released.
>
---
#### [new 055] Relation Extraction Capabilities of LLMs on Clinical Text: A Bilingual Evaluation for English and Turkish
- **分类: cs.CL**

- **简介: 该论文属于临床关系抽取任务，旨在解决非英语语言标注数据不足的问题。研究构建了首个英土双语临床RE数据集，并评估多种提示策略与检索方法的效果。**

- **链接: [https://arxiv.org/pdf/2601.09367v1](https://arxiv.org/pdf/2601.09367v1)**

> **作者:** Aidana Aidynkyzy; Oğuz Dikenelli; Oylum Alatlı; Şebnem Bora
>
> **摘要:** The scarcity of annotated datasets for clinical information extraction in non-English languages hinders the evaluation of large language model (LLM)-based methods developed primarily in English. In this study, we present the first comprehensive bilingual evaluation of LLMs for the clinical Relation Extraction (RE) task in both English and Turkish. To facilitate this evaluation, we introduce the first English-Turkish parallel clinical RE dataset, derived and carefully curated from the 2010 i2b2/VA relation classification corpus. We systematically assess a diverse set of prompting strategies, including multiple in-context learning (ICL) and Chain-of-Thought (CoT) approaches, and compare their performance to fine-tuned baselines such as PURE. Furthermore, we propose Relation-Aware Retrieval (RAR), a novel in-context example selection method based on contrastive learning, that is specifically designed to capture both sentence-level and relation-level semantics. Our results show that prompting-based LLM approaches consistently outperform traditional fine-tuned models. Moreover, evaluations for English performed better than their Turkish counterparts across all evaluated LLMs and prompting techniques. Among ICL methods, RAR achieves the highest performance, with Gemini 1.5 Flash reaching a micro-F1 score of 0.906 in English and 0.888 in Turkish. Performance further improves to 0.918 F1 in English when RAR is combined with a structured reasoning prompt using the DeepSeek-V3 model. These findings highlight the importance of high-quality demonstration retrieval and underscore the potential of advanced retrieval and prompting techniques to bridge resource gaps in clinical natural language processing.
>
---
#### [new 056] Resisting Correction: How RLHF Makes Language Models Ignore External Safety Signals in Natural Conversation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型安全任务，研究RLHF导致模型在自然对话中忽略外部安全信号的问题。通过实验发现指令调优模型在不同交互模式下表现不一，揭示了安全校正失效的关键原因。**

- **链接: [https://arxiv.org/pdf/2601.08842v1](https://arxiv.org/pdf/2601.08842v1)**

> **作者:** Felipe Biava Cataneo
>
> **摘要:** Safety architectures for language models increasingly rely on external monitors to detect errors and inject corrective signals at inference time. For such systems to function in interactive settings, models must be able to incorporate externally provided confidence information into their verbal responses. In this work, we test whether instruction-tuned language models preserve this controllability across different interaction modes. Using Llama-3.2-3B on GSM8K, we perform a causal intervention study in which explicit external confidence signals are injected and model compliance is measured under multiple prompt strategies. We find that base models exhibit near-perfect controllability (Spearman rho close to 1.0), while instruction-tuned models display a striking context dependence: they fully comply with external corrections under explicit command prompts (bias approximately 0 percent, rho = 0.93), yet systematically ignore the same signals in natural conversational queries (bias plus 40 percent, rho = 0.04). This behavior is not a capability failure; the model can process the signal, but an emergent property of RLHF optimization that prioritizes conversational fluency over external calibration cues in natural dialogue. We further show that internal token-level confidence in small models is uninformative (r = 0.035), underscoring the necessity of external supervision. Our findings highlight a deployment-critical failure mode: the interaction style users expect is precisely where safety corrections are least effective.
>
---
#### [new 057] Structured Knowledge Representation through Contextual Pages for Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于知识增强生成任务，旨在解决RAG中知识表示不连贯的问题。提出PAGER框架，通过结构化页面组织知识，提升知识质量和利用效率。**

- **链接: [https://arxiv.org/pdf/2601.09402v1](https://arxiv.org/pdf/2601.09402v1)**

> **作者:** Xinze Li; Zhenghao Liu; Haidong Xin; Yukun Yan; Shuo Wang; Zheni Zeng; Sen Mei; Ge Yu; Maosong Sun
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by incorporating external knowledge. Recently, some works have incorporated iterative knowledge accumulation processes into RAG models to progressively accumulate and refine query-related knowledge, thereby constructing more comprehensive knowledge representations. However, these iterative processes often lack a coherent organizational structure, which limits the construction of more comprehensive and cohesive knowledge representations. To address this, we propose PAGER, a page-driven autonomous knowledge representation framework for RAG. PAGER first prompts an LLM to construct a structured cognitive outline for a given question, which consists of multiple slots representing a distinct knowledge aspect. Then, PAGER iteratively retrieves and refines relevant documents to populate each slot, ultimately constructing a coherent page that serves as contextual input for guiding answer generation. Experiments on multiple knowledge-intensive benchmarks and backbone models show that PAGER consistently outperforms all RAG baselines. Further analyses demonstrate that PAGER constructs higher-quality and information-dense knowledge representations, better mitigates knowledge conflicts, and enables LLMs to leverage external knowledge more effectively. All code is available at https://github.com/OpenBMB/PAGER.
>
---
#### [new 058] MVSS: A Unified Framework for Multi-View Structured Survey Generation
- **分类: cs.CL**

- **简介: 该论文提出MVSS框架，解决自动科学综述生成中的结构组织与方法比较问题。通过多视图联合生成，提升综述的结构质量与证据依据。**

- **链接: [https://arxiv.org/pdf/2601.09504v1](https://arxiv.org/pdf/2601.09504v1)**

> **作者:** Yinqi Liu; Yueqi Zhu; Yongkang Zhang; Xinfeng Li; Feiran Liu; Yufei Sun; Xin Wang; Renzhao Liang; Yidong Wang; Cunxiang Wang
>
> **摘要:** Scientific surveys require not only summarizing large bodies of literature, but also organizing them into clear and coherent conceptual structures. Existing automatic survey generation methods typically focus on linear text generation and struggle to explicitly model hierarchical relations among research topics and structured methodological comparisons, resulting in gaps in structural organization compared to expert-written surveys. We propose MVSS, a multi-view structured survey generation framework that jointly generates and aligns citation-grounded hierarchical trees, structured comparison tables, and survey text. MVSS follows a structure-first paradigm: it first constructs a conceptual tree of the research domain, then generates comparison tables constrained by the tree, and finally uses both as structural constraints for text generation. This enables complementary multi-view representations across structure, comparison, and narrative. We introduce an evaluation framework assessing structural quality, comparative completeness, and citation fidelity. Experiments on 76 computer science topics show MVSS outperforms existing methods in organization and evidence grounding, achieving performance comparable to expert surveys.
>
---
#### [new 059] SITA: Learning Speaker-Invariant and Tone-Aware Speech Representations for Low-Resource Tonal Languages
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决低资源声调语言中语音表示不鲁棒的问题。提出SITA方法，增强说话人不变性和声调感知，提升词义识别准确率。**

- **链接: [https://arxiv.org/pdf/2601.09050v1](https://arxiv.org/pdf/2601.09050v1)**

> **作者:** Tianyi Xu; Xuan Ouyang; Binwei Yao; Shoua Xiong; Sara Misurelli; Maichou Lor; Junjie Hu
>
> **备注:** 8 pages (excluding references, limitations, ethics, acknowledgement, and appendix); 4 figures in the main paper; appendix included
>
> **摘要:** Tonal low-resource languages are widely spoken yet remain underserved by modern speech technology. A key challenge is learning representations that are robust to nuisance variation such as gender while remaining tone-aware for different lexical meanings. To address this, we propose SITA, a lightweight adaptation recipe that enforces Speaker-Invariance and Tone-Awareness for pretrained wav2vec-style encoders. SITA uses staged multi-objective training: (i) a cross-gender contrastive objective encourages lexical consistency across speakers, while a tone-repulsive loss prevents tone collapse by explicitly separating same-word different-tone realizations; and (ii) an auxiliary Connectionist Temporal Classification (CTC)-based ASR objective with distillation stabilizes recognition-relevant structure. We evaluate primarily on Hmong, a highly tonal and severely under-resourced language where off-the-shelf multilingual encoders fail to represent tone effectively. On a curated Hmong word corpus, SITA improves cross-gender lexical retrieval accuracy, while maintaining usable ASR accuracy relative to an ASR-adapted XLS-R teacher. We further observe similar gains when transferring the same recipe to Mandarin, suggesting SITA is a general, plug-in approach for adapting multilingual speech encoders to tonal languages.
>
---
#### [new 060] Improving Implicit Hate Speech Detection via a Community-Driven Multi-Agent Framework
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于隐性仇恨言论检测任务，旨在提升检测的准确性和公平性。通过构建社区驱动的多智能体框架，融合社会文化背景，优化分类性能。**

- **链接: [https://arxiv.org/pdf/2601.09342v1](https://arxiv.org/pdf/2601.09342v1)**

> **作者:** Ewelina Gajewska; Katarzyna Budzynska; Jarosław A Chudziak
>
> **备注:** This paper has been accepted for the upcoming 18th International Conference on Agents and Artificial Intelligence (ICAART-2026), Marbella, Spain. The final published version will appear in the official conference proceedings
>
> **摘要:** This work proposes a contextualised detection framework for implicitly hateful speech, implemented as a multi-agent system comprising a central Moderator Agent and dynamically constructed Community Agents representing specific demographic groups. Our approach explicitly integrates socio-cultural context from publicly available knowledge sources, enabling identity-aware moderation that surpasses state-of-the-art prompting methods (zero-shot prompting, few-shot prompting, chain-of-thought prompting) and alternative approaches on a challenging ToxiGen dataset. We enhance the technical rigour of performance evaluation by incorporating balanced accuracy as a central metric of classification fairness that accounts for the trade-off between true positive and true negative rates. We demonstrate that our community-driven consultative framework significantly improves both classification accuracy and fairness across all target groups.
>
---
#### [new 061] DeliberationBench: When Do More Voices Hurt? A Controlled Study of Multi-LLM Deliberation Protocols
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模型协作任务，旨在评估多LLM协作协议的有效性。研究发现，单一最佳模型表现优于所有协作协议，揭示复杂系统未必提升效果。**

- **链接: [https://arxiv.org/pdf/2601.08835v1](https://arxiv.org/pdf/2601.08835v1)**

> **作者:** Vaarunay Kaushal; Taranveer Singh
>
> **备注:** 6 pages, 5 figures
>
> **摘要:** Multi-agent systems where Large Language Models (LLMs) deliberate to form consensus have gained significant attention, yet their practical value over simpler methods remains under-scrutinized. We introduce DELIBERATIONBENCH, a controlled benchmark evaluating three deliberation protocols against a strong baseline of selecting the best response from a pool of model outputs. Across 270 questions and three independent seeds (810 total evaluations), we find a striking negative result: the best-single baseline achieves an 82.5% +- 3.3% win rate, dramatically outperforming the best deliberation protocol(13.8% +- 2.6%). This 6.0x performance gap is statistically significant (p < 0.01) and comes at 1.5-2.5x higher computational cost. Our findings challenge assumptions that complexity enhances quality in multi-LLM systems.
>
---
#### [new 062] A Review: PTSD in Pre-Existing Medical Condition on Social Media
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于文献综述任务，探讨PTSD与慢性疾病在社交媒体上的表现，分析如何通过NLP和ML识别潜在患者，并强调在线支持的重要性。**

- **链接: [https://arxiv.org/pdf/2601.08836v1](https://arxiv.org/pdf/2601.08836v1)**

> **作者:** Zaber Al Hassan Ayon; Nur Hafieza Ismail; Nur Shazwani Kamarudin
>
> **备注:** Published in (IJACSA) International Journal of Advanced Computer Science and Applications, Vol. 15, No. 11, 2024
>
> **摘要:** Post-Traumatic Stress Disorder (PTSD) is a multifaceted mental health condition, particularly challenging for individuals with pre-existing medical conditions. This review critically examines the intersection of PTSD and chronic illnesses as expressed on social media platforms. By systematically analyzing literature from 2008 to 2024, the study explores how PTSD manifests and is managed in individuals with chronic conditions such as cancer, heart disease, and autoimmune disorders, with a focus on online expressions on platforms like X (formally known as Twitter) and Facebook. Findings demonstrate that social media data offers valuable insights into the unique challenges faced by individuals with both PTSD and chronic illnesses. Specifically, natural language processing (NLP) and machine learning (ML) techniques can identify potential PTSD cases among these populations, achieving accuracy rates between 74% and 90%. Furthermore, the role of online support communities in shaping coping strategies and facilitating early interventions is highlighted. This review underscores the necessity of incorporating considerations of pre-existing medical conditions in PTSD research and treatment, emphasizing social media's potential as a monitoring and support tool for vulnerable groups. Future research directions and clinical implications are also discussed, with an emphasis on developing targeted interventions.
>
---
#### [new 063] Rubric-Conditioned LLM Grading: Alignment, Uncertainty, and Robustness
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自动短答案评分任务，研究LLM在基于评分标准的评分中的对齐性、不确定性和鲁棒性，评估其可靠性并提出改进方法。**

- **链接: [https://arxiv.org/pdf/2601.08843v1](https://arxiv.org/pdf/2601.08843v1)**

> **作者:** Haotian Deng; Chris Farber; Jiyoon Lee; David Tang
>
> **摘要:** Automated short-answer grading (ASAG) remains a challenging task due to the linguistic variability of student responses and the need for nuanced, rubric-aligned partial credit. While Large Language Models (LLMs) offer a promising solution, their reliability as automated judges in rubric-based settings requires rigorous assessment. In this paper, we systematically evaluate the performance of LLM-judges for rubric-based short-answer grading. We investigate three key aspects: the alignment of LLM grading with expert judgment across varying rubric complexities, the trade-off between uncertainty and accuracy facilitated by a consensus-based deferral mechanism, and the model's robustness under random input perturbations and adversarial attacks. Using the SciEntsBank benchmark and Qwen 2.5-72B, we find that alignment is strong for binary tasks but degrades with increased rubric granularity. Our "Trust Curve" analysis demonstrates a clear trade-off where filtering low-confidence predictions improves accuracy on the remaining subset. Additionally, robustness experiments reveal that while the model is resilient to prompt injection, it is sensitive to synonym substitutions. Our work provides critical insights into the capabilities and limitations of rubric-conditioned LLM judges, highlighting the importance of uncertainty estimation and robustness testing for reliable deployment.
>
---
#### [new 064] Improving Symbolic Translation of Language Models for Logical Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于逻辑推理任务，旨在提升小型语言模型的符号化翻译能力。针对翻译错误问题，通过数据增强、分步推理和验证模块优化模型表现。**

- **链接: [https://arxiv.org/pdf/2601.09446v1](https://arxiv.org/pdf/2601.09446v1)**

> **作者:** Ramya Keerthy Thatikonda; Jiuzhou Han; Wray Buntine; Ehsan Shareghi
>
> **备注:** The Third workshop of NeusymBridge @AAAI 2026 (Bridging Neurons and Symbols for NLP and Knowledge Graph Reasoning)
>
> **摘要:** The use of formal language for deductive logical reasoning aligns well with language models (LMs), where translating natural language (NL) into first-order logic (FOL) and employing an external solver results in a verifiable and therefore reliable reasoning system. However, smaller LMs often struggle with this translation task, frequently producing incorrect symbolic outputs due to formatting and translation errors. Existing approaches typically rely on self-iteration to correct these errors, but such methods depend heavily on the capabilities of the underlying model. To address this, we first categorize common errors and fine-tune smaller LMs using data synthesized by large language models. The evaluation is performed using the defined error categories. We introduce incremental inference, which divides inference into two stages, predicate generation and FOL translation, providing greater control over model behavior and enhancing generation quality as measured by predicate metrics. This decomposition framework also enables the use of a verification module that targets predicate-arity errors to further improve performance. Our study evaluates three families of models across four logical-reasoning datasets. The comprehensive fine-tuning, incremental inference, and verification modules reduce error rates, increase predicate coverage, and improve reasoning performance for smaller LMs, moving us closer to developing reliable and accessible symbolic-reasoning systems.
>
---
#### [new 065] LLMs can Compress LLMs: Adaptive Pruning by Agents
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于模型压缩任务，解决如何有效剪枝大语言模型以降低计算成本并保持性能的问题。提出代理引导的自适应剪枝方法，提升压缩效果。**

- **链接: [https://arxiv.org/pdf/2601.09694v1](https://arxiv.org/pdf/2601.09694v1)**

> **作者:** Sai Varun Kodathala; Rakesh Vunnam
>
> **备注:** 17 Pages
>
> **摘要:** As Large Language Models (LLMs) continue to scale, post-training pruning has emerged as a promising approach to reduce computational costs while preserving performance. Existing methods such as SparseGPT and Wanda achieve high sparsity through layer-wise weight reconstruction or activation-aware magnitude pruning, but rely on uniform or hand-crafted heuristics to determine per-layer sparsity ratios. Moreover, recent work has shown that pruned LLMs suffer from severe factual knowledge degradation, with structured pruning methods experiencing near-total collapse in factual question-answering capabilities. We introduce agent-guided pruning, where a foundation model acts as an adaptive pruning agent to intelligently select which layers to prune at each iteration while preserving critical knowledge pathways. Our method constructs layer-wise sensitivity profiles by combining Wanda-inspired weight-activation metrics with gradient importance scores, normalized as z-scores for model-agnostic comparison. These statistics are processed by an LLM agent equipped with self-reflection capabilities, enabling it to learn from previous pruning outcomes and iteratively refine its strategy. A checkpoint rollback mechanism maintains model quality by reverting when perplexity degradation exceeds a threshold. We evaluate our approach on Qwen3 models (4B and 8B parameters) at approximately 45% sparsity, demonstrating substantial improvements over structured pruning baselines: 56% relative improvement in MMLU accuracy, 19x better factual knowledge retention on FreebaseQA, and 69% lower perplexity degradation. Notably, our framework requires no retraining, operates in a model-agnostic manner, and exhibits effective self-correction with only 2-4 rollbacks across 21-40 iterations, demonstrating that foundation models can effectively guide the compression of other foundation models.
>
---
#### [new 066] SERM: Self-Evolving Relevance Model with Agent-Driven Learning from Massive Query Streams
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决动态查询流中相关性模型泛化能力不足的问题。通过引入自进化机制和多智能体方法，提升模型性能与标签可靠性。**

- **链接: [https://arxiv.org/pdf/2601.09515v1](https://arxiv.org/pdf/2601.09515v1)**

> **作者:** Chenglong Wang; Canjia Li; Xingzhao Zhu; Yifu Huo; Huiyu Wang; Weixiong Lin; Yun Yang; Qiaozhi He; Tianhua Zhou; Xiaojia Chang; Jingbo Zhu; Tong Xiao
>
> **摘要:** Due to the dynamically evolving nature of real-world query streams, relevance models struggle to generalize to practical search scenarios. A sophisticated solution is self-evolution techniques. However, in large-scale industrial settings with massive query streams, this technique faces two challenges: (1) informative samples are often sparse and difficult to identify, and (2) pseudo-labels generated by the current model could be unreliable. To address these challenges, in this work, we propose a Self-Evolving Relevance Model approach (SERM), which comprises two complementary multi-agent modules: a multi-agent sample miner, designed to detect distributional shifts and identify informative training samples, and a multi-agent relevance annotator, which provides reliable labels through a two-level agreement framework. We evaluate SERM in a large-scale industrial setting, which serves billions of user requests daily. Experimental results demonstrate that SERM can achieve significant performance gains through iterative self-evolution, as validated by extensive offline multilingual evaluations and online testing.
>
---
#### [new 067] Bias Dynamics in BabyLMs: Towards a Compute-Efficient Sandbox for Democratising Pre-Training Debiasing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的模型偏见研究任务，旨在解决大模型训练成本高、难以调试偏见的问题。通过使用低成本的BabyLMs模型，模拟大模型的偏见形成过程，实现高效、经济的预训练去偏研究。**

- **链接: [https://arxiv.org/pdf/2601.09421v1](https://arxiv.org/pdf/2601.09421v1)**

> **作者:** Filip Trhlik; Andrew Caines; Paula Buttery
>
> **备注:** 21 pages, 18 figures
>
> **摘要:** Pre-trained language models (LMs) have, over the last few years, grown substantially in both societal adoption and training costs. This rapid growth in size has constrained progress in understanding and mitigating their biases. Since re-training LMs is prohibitively expensive, most debiasing work has focused on post-hoc or masking-based strategies, which often fail to address the underlying causes of bias. In this work, we seek to democratise pre-model debiasing research by using low-cost proxy models. Specifically, we investigate BabyLMs, compact BERT-like models trained on small and mutable corpora that can approximate bias acquisition and learning dynamics of larger models. We show that BabyLMs display closely aligned patterns of intrinsic bias formation and performance development compared to standard BERT models, despite their drastically reduced size. Furthermore, correlations between BabyLMs and BERT hold across multiple intra-model and post-model debiasing methods. Leveraging these similarities, we conduct pre-model debiasing experiments with BabyLMs, replicating prior findings and presenting new insights regarding the influence of gender imbalance and toxicity on bias formation. Our results demonstrate that BabyLMs can serve as an effective sandbox for large-scale LMs, reducing pre-training costs from over 500 GPU-hours to under 30 GPU-hours. This provides a way to democratise pre-model debiasing research and enables faster, more accessible exploration of methods for building fairer LMs.
>
---
#### [new 068] Routing with Generated Data: Annotation-Free LLM Skill Estimation and Expert Selection
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究LLM路由器在无标注数据下的技能评估与专家选择问题。提出RGD框架，通过生成数据训练路由器，解决实际中标签数据缺失的问题。**

- **链接: [https://arxiv.org/pdf/2601.09692v1](https://arxiv.org/pdf/2601.09692v1)**

> **作者:** Tianyi Niu; Justin Chih-Yao Chen; Genta Indra Winata; Shi-Xiong Zhang; Supriyo Chakraborty; Sambit Sahu; Yue Zhang; Elias Stengel-Eskin; Mohit Bansal
>
> **备注:** Code: https://github.com/tianyiniu/RoutingGenData
>
> **摘要:** Large Language Model (LLM) routers dynamically select optimal models for given inputs. Existing approaches typically assume access to ground-truth labeled data, which is often unavailable in practice, especially when user request distributions are heterogeneous and unknown. We introduce Routing with Generated Data (RGD), a challenging setting in which routers are trained exclusively on generated queries and answers produced from high-level task descriptions by generator LLMs. We evaluate query-answer routers (using both queries and labels) and query-only routers across four diverse benchmarks and 12 models, finding that query-answer routers degrade faster than query-only routers as generator quality decreases. Our analysis reveals two crucial characteristics of effective generators: they must accurately respond to their own questions, and their questions must produce sufficient performance differentiation among the model pool. We then show how filtering for these characteristics can improve the quality of generated data. We further propose CASCAL, a novel query-only router that estimates model correctness through consensus voting and identifies model-specific skill niches via hierarchical clustering. CASCAL is substantially more robust to generator quality, outperforming the best query-answer router by 4.6% absolute accuracy when trained on weak generator data.
>
---
#### [new 069] Dissecting Judicial Reasoning in U.S. Copyright Damage Awards
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于计算法律分析任务，旨在解决版权损害判决中司法推理不一致的问题。通过构建话语分析框架，提取并量化判决中的推理模式，提升法律决策的可预测性与透明度。**

- **链接: [https://arxiv.org/pdf/2601.09459v1](https://arxiv.org/pdf/2601.09459v1)**

> **作者:** Pei-Chi Lo; Thomas Y. Lu
>
> **备注:** Presented in SIGKDD'25 SciSoc LLM Workshop: Large Language Models for Scientific and Societal Advances
>
> **摘要:** Judicial reasoning in copyright damage awards poses a core challenge for computational legal analysis. Although federal courts follow the 1976 Copyright Act, their interpretations and factor weightings vary widely across jurisdictions. This inconsistency creates unpredictability for litigants and obscures the empirical basis of legal decisions. This research introduces a novel discourse-based Large Language Model (LLM) methodology that integrates Rhetorical Structure Theory (RST) with an agentic workflow to extract and quantify previously opaque reasoning patterns from judicial opinions. Our framework addresses a major gap in empirical legal scholarship by parsing opinions into hierarchical discourse structures and using a three-stage pipeline, i.e., Dataset Construction, Discourse Analysis, and Agentic Feature Extraction. This pipeline identifies reasoning components and extract feature labels with corresponding discourse subtrees. In analyzing copyright damage rulings, we show that discourse-augmented LLM analysis outperforms traditional methods while uncovering unquantified variations in factor weighting across circuits. These findings offer both methodological advances in computational legal analysis and practical insights into judicial reasoning, with implications for legal practitioners seeking predictive tools, scholars studying legal principle application, and policymakers confronting inconsistencies in copyright law.
>
---
#### [new 070] Geometric Stability: The Missing Axis of Representations
- **分类: cs.LG; cs.CL; q-bio.QM; stat.ML**

- **简介: 该论文提出“几何稳定性”概念，解决表示学习中结构可靠性评估问题。通过框架Shesha，分析不同场景下的稳定性与相似性关系，揭示其独立性，并应用于安全监控、可控性预测等任务。**

- **链接: [https://arxiv.org/pdf/2601.09173v1](https://arxiv.org/pdf/2601.09173v1)**

> **作者:** Prashant C. Raju
>
> **摘要:** Analysis of learned representations has a blind spot: it focuses on $similarity$, measuring how closely embeddings align with external references, but similarity reveals only what is represented, not whether that structure is robust. We introduce $geometric$ $stability$, a distinct dimension that quantifies how reliably representational geometry holds under perturbation, and present $Shesha$, a framework for measuring it. Across 2,463 configurations in seven domains, we show that stability and similarity are empirically uncorrelated ($ρ\approx 0.01$) and mechanistically distinct: similarity metrics collapse after removing the top principal components, while stability retains sensitivity to fine-grained manifold structure. This distinction yields actionable insights: for safety monitoring, stability acts as a functional geometric canary, detecting structural drift nearly 2$\times$ more sensitively than CKA while filtering out the non-functional noise that triggers false alarms in rigid distance metrics; for controllability, supervised stability predicts linear steerability ($ρ= 0.89$-$0.96$); for model selection, stability dissociates from transferability, revealing a geometric tax that transfer optimization incurs. Beyond machine learning, stability predicts CRISPR perturbation coherence and neural-behavioral coupling. By quantifying $how$ $reliably$ systems maintain structure, geometric stability provides a necessary complement to similarity for auditing representations across biological and computational systems.
>
---
#### [new 071] GIFT: Unlocking Global Optimality in Post-Training via Finite-Temperature Gibbs Initialization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大模型后训练任务，解决SFT与RL间优化不匹配问题。提出GIFT方法，通过有限温度吉布斯初始化，提升后训练全局最优性。**

- **链接: [https://arxiv.org/pdf/2601.09233v1](https://arxiv.org/pdf/2601.09233v1)**

> **作者:** Zhengyang Zhao; Lu Ma; Yizhen Jiang; Xiaochen Ma; Zimo Meng; Chengyu Shen; Lexiang Tang; Haoze Sun; Peng Pei; Wentao Zhang
>
> **摘要:** The prevailing post-training paradigm for Large Reasoning Models (LRMs)--Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (RL)--suffers from an intrinsic optimization mismatch: the rigid supervision inherent in SFT induces distributional collapse, thereby exhausting the exploration space necessary for subsequent RL. In this paper, we reformulate SFT within a unified post-training framework and propose Gibbs Initialization with Finite Temperature (GIFT). We characterize standard SFT as a degenerate zero-temperature limit that suppresses base priors. Conversely, GIFT incorporates supervision as a finite-temperature energy potential, establishing a distributional bridge that ensures objective consistency throughout the post-training pipeline. Our experiments demonstrate that GIFT significantly outperforms standard SFT and other competitive baselines when utilized for RL initialization, providing a mathematically principled pathway toward achieving global optimality in post-training. Our code is available at https://github.com/zzy1127/GIFT.
>
---
#### [new 072] Toward Understanding Unlearning Difficulty: A Mechanistic Perspective and Circuit-Guided Difficulty Metric
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于机器学习任务，旨在解决模型中样本难遗忘问题。通过分析模型内部机制，提出CUD指标评估样本遗忘难度，提升模型可解释性与可控性。**

- **链接: [https://arxiv.org/pdf/2601.09624v1](https://arxiv.org/pdf/2601.09624v1)**

> **作者:** Jiali Cheng; Ziheng Chen; Chirag Agarwal; Hadi Amiri
>
> **摘要:** Machine unlearning is becoming essential for building trustworthy and compliant language models. Yet unlearning success varies considerably across individual samples: some are reliably erased, while others persist despite the same procedure. We argue that this disparity is not only a data-side phenomenon, but also reflects model-internal mechanisms that encode and protect memorized information. We study this problem from a mechanistic perspective based on model circuits--structured interaction pathways that govern how predictions are formed. We propose Circuit-guided Unlearning Difficulty (CUD), a {\em pre-unlearning} metric that assigns each sample a continuous difficulty score using circuit-level signals. Extensive experiments demonstrate that CUD reliably separates intrinsically easy and hard samples, and remains stable across unlearning methods. We identify key circuit-level patterns that reveal a mechanistic signature of difficulty: easy-to-unlearn samples are associated with shorter, shallower interactions concentrated in earlier-to-intermediate parts of the original model, whereas hard samples rely on longer and deeper pathways closer to late-stage computation. Compared to existing qualitative studies, CUD takes a first step toward a principled, fine-grained, and interpretable analysis of unlearning difficulty; and motivates the development of unlearning methods grounded in model mechanisms.
>
---
#### [new 073] SLAM-LLM: A Modular, Open-Source Multimodal Large Language Model Framework and Best Practice for Speech, Language, Audio and Music Processing
- **分类: cs.SD; cs.CL; cs.MM**

- **简介: 该论文提出SLAM-LLM，一个面向语音、音频和音乐处理的多模态大语言模型框架，解决现有框架对音频支持不足的问题，提供模块化配置和训练方案。**

- **链接: [https://arxiv.org/pdf/2601.09385v1](https://arxiv.org/pdf/2601.09385v1)**

> **作者:** Ziyang Ma; Guanrou Yang; Wenxi Chen; Zhifu Gao; Yexing Du; Xiquan Li; Zhisheng Zheng; Haina Zhu; Jianheng Zhuo; Zheshu Song; Ruiyang Xu; Tiranrui Wang; Yifan Yang; Yanqiao Zhu; Zhikang Niu; Liumeng Xue; Yinghao Ma; Ruibin Yuan; Shiliang Zhang; Kai Yu; Eng Siong Chng; Xie Chen
>
> **备注:** Published in IEEE Journal of Selected Topics in Signal Processing (JSTSP)
>
> **摘要:** The recent surge in open-source Multimodal Large Language Models (MLLM) frameworks, such as LLaVA, provides a convenient kickoff for artificial intelligence developers and researchers. However, most of the MLLM frameworks take vision as the main input modality, and provide limited in-depth support for the modality of speech, audio, and music. This situation hinders the development of audio-language models, and forces researchers to spend a lot of effort on code writing and hyperparameter tuning. We present SLAM-LLM, an open-source deep learning framework designed to train customized MLLMs, focused on speech, language, audio, and music processing. SLAM-LLM provides a modular configuration of different encoders, projectors, LLMs, and parameter-efficient fine-tuning plugins. SLAM-LLM also includes detailed training and inference recipes for mainstream tasks, along with high-performance checkpoints like LLM-based Automatic Speech Recognition (ASR), Automated Audio Captioning (AAC), and Music Captioning (MC). Some of these recipes have already reached or are nearing state-of-the-art performance, and some relevant techniques have also been accepted by academic papers. We hope SLAM-LLM will accelerate iteration, development, data engineering, and model training for researchers. We are committed to continually pushing forward audio-based MLLMs through this open-source framework, and call on the community to contribute to the LLM-based speech, audio and music processing.
>
---
#### [new 074] Show, don't tell -- Providing Visual Error Feedback for Handwritten Documents
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于手写文档错误反馈任务，旨在提供视觉化错误提示。研究分析了从手写图像到准确反馈的挑战，比较了模块化与端到端系统，指出当前方法效果不佳，并提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2601.09586v1](https://arxiv.org/pdf/2601.09586v1)**

> **作者:** Said Yasin; Torsten Zesch
>
> **摘要:** Handwriting remains an essential skill, particularly in education. Therefore, providing visual feedback on handwritten documents is an important but understudied area. We outline the many challenges when going from an image of handwritten input to correctly placed informative error feedback. We empirically compare modular and end-to-end systems and find that both approaches currently do not achieve acceptable overall quality. We identify the major challenges and outline an agenda for future research.
>
---
#### [new 075] Linear Complexity Self-Supervised Learning for Music Understanding with Random Quantizer
- **分类: cs.SD; cs.AI; cs.CL; cs.LG**

- **简介: 该论文聚焦音乐信息检索任务，旨在缩小基础模型规模。通过结合Branchformer与SummaryMixing，并引入随机量化，有效降低模型大小，同时保持性能。**

- **链接: [https://arxiv.org/pdf/2601.09603v1](https://arxiv.org/pdf/2601.09603v1)**

> **作者:** Petros Vavaroutsos; Theodoros Palamas; Pantelis Vikatos
>
> **备注:** accepted by ACM/SIGAPP Symposium on Applied Computing (SAC 2026)
>
> **摘要:** In recent years, foundation models have become very popular due to their exceptional performance, mainly in natural language (NLP) tasks where they were first introduced. These models usually consist of hundreds of millions, or even billions, of parameters, making them resource-intensive during training and in production systems, leading to increased costs. This paper focuses on the reduction of a foundation's model size when applied to music information retrieval (MIR) tasks. Our research combines the Branchformer architecture with SummaryMixing, which were first applied in speech recognition, along with a random quantization process. To facilitate reproducibility, we conduct pre-training on publicly available datasets, complemented by a proprietary dataset comparable in scale to other private datasets reported in the literature. We ensure robust evaluation by using a framework consisting of a variety of downstream MIR tasks. Our results show that our architecture achieves competitive performance when compared with other state-of-the-art models that use multi-head self-attention, while reducing the model size from 8.5% up to 12.3%.
>
---
#### [new 076] MMR-GRPO: Accelerating GRPO-Style Training through Diversity-Aware Reward Reweighting
- **分类: cs.LG; cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于数学推理模型训练任务，旨在解决GRPO训练效率低的问题。通过引入MMR-GRPO方法，提升训练速度并保持性能。**

- **链接: [https://arxiv.org/pdf/2601.09085v1](https://arxiv.org/pdf/2601.09085v1)**

> **作者:** Kangda Wei; Ruihong Huang
>
> **摘要:** Group Relative Policy Optimization (GRPO) has become a standard approach for training mathematical reasoning models; however, its reliance on multiple completions per prompt makes training computationally expensive. Although recent work has reduced the number of training steps required to reach peak performance, the overall wall-clock training time often remains unchanged or even increases due to higher per-step cost. We propose MMR-GRPO, which integrates Maximal Marginal Relevance to reweigh rewards based on completion diversity. Our key insight is that semantically redundant completions contribute limited marginal learning signal; prioritizing diverse solutions yields more informative updates and accelerates convergence. Extensive evaluations across three model sizes (1.5B, 7B, 8B), three GRPO variants, and five mathematical reasoning benchmarks show that MMR-GRPO achieves comparable peak performance while requiring on average 47.9% fewer training steps and 70.2% less wall-clock time. These gains are consistent across models, methods, and benchmarks. We will release our code, trained models, and experimental protocols.
>
---
#### [new 077] PluriHarms: Benchmarking the Full Spectrum of Human Judgments on AI Harm
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于AI安全任务，旨在解决AI危害判断的多样性问题。通过构建PluriHarms基准，研究人类在AI危害上的分歧原因及影响因素。**

- **链接: [https://arxiv.org/pdf/2601.08951v1](https://arxiv.org/pdf/2601.08951v1)**

> **作者:** Jing-Jing Li; Joel Mire; Eve Fleisig; Valentina Pyatkin; Anne Collins; Maarten Sap; Sydney Levine
>
> **摘要:** Current AI safety frameworks, which often treat harmfulness as binary, lack the flexibility to handle borderline cases where humans meaningfully disagree. To build more pluralistic systems, it is essential to move beyond consensus and instead understand where and why disagreements arise. We introduce PluriHarms, a benchmark designed to systematically study human harm judgments across two key dimensions -- the harm axis (benign to harmful) and the agreement axis (agreement to disagreement). Our scalable framework generates prompts that capture diverse AI harms and human values while targeting cases with high disagreement rates, validated by human data. The benchmark includes 150 prompts with 15,000 ratings from 100 human annotators, enriched with demographic and psychological traits and prompt-level features of harmful actions, effects, and values. Our analyses show that prompts that relate to imminent risks and tangible harms amplify perceived harmfulness, while annotator traits (e.g., toxicity experience, education) and their interactions with prompt content explain systematic disagreement. We benchmark AI safety models and alignment methods on PluriHarms, finding that while personalization significantly improves prediction of human harm judgments, considerable room remains for future progress. By explicitly targeting value diversity and disagreement, our work provides a principled benchmark for moving beyond "one-size-fits-all" safety toward pluralistically safe AI.
>
---
#### [new 078] AviationLMM: A Large Multimodal Foundation Model for Civil Aviation
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出 AviationLMM，一个用于民航的大型多模态基础模型，旨在整合语音、雷达、传感器等异构数据，解决传统AI在民航中孤立、单一的问题，提升情境感知与决策支持能力。**

- **链接: [https://arxiv.org/pdf/2601.09105v1](https://arxiv.org/pdf/2601.09105v1)**

> **作者:** Wenbin Li; Jingling Wu; Xiaoyong Lin. Jing Chen; Cong Chen
>
> **备注:** Accepted by 2025 7th International Conference on Interdisciplinary Computer Science and Engineering (ICICSE 2025) conference, Chongqing, China; 9 pages,1 figure,5 tables
>
> **摘要:** Civil aviation is a cornerstone of global transportation and commerce, and ensuring its safety, efficiency and customer satisfaction is paramount. Yet conventional Artificial Intelligence (AI) solutions in aviation remain siloed and narrow, focusing on isolated tasks or single modalities. They struggle to integrate heterogeneous data such as voice communications, radar tracks, sensor streams and textual reports, which limits situational awareness, adaptability, and real-time decision support. This paper introduces the vision of AviationLMM, a Large Multimodal foundation Model for civil aviation, designed to unify the heterogeneous data streams of civil aviation and enable understanding, reasoning, generation and agentic applications. We firstly identify the gaps between existing AI solutions and requirements. Secondly, we describe the model architecture that ingests multimodal inputs such as air-ground voice, surveillance, on-board telemetry, video and structured texts, and performs cross-modal alignment and fusion, and produces flexible outputs ranging from situation summaries and risk alerts to predictive diagnostics and multimodal incident reconstructions. In order to fully realize this vision, we identify key research opportunities to address, including data acquisition, alignment and fusion, pretraining, reasoning, trustworthiness, privacy, robustness to missing modalities, and synthetic scenario generation. By articulating the design and challenges of AviationLMM, we aim to boost the civil aviation foundation model progress and catalyze coordinated research efforts toward an integrated, trustworthy and privacy-preserving aviation AI ecosystem.
>
---
#### [new 079] Fine Grained Evaluation of LLMs-as-Judges
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文属于信息检索任务，研究如何用大语言模型作为裁判评估文本相关性。通过分析模型判断文档相关性的准确性和理由，验证其在监督下的有效性。**

- **链接: [https://arxiv.org/pdf/2601.08919v1](https://arxiv.org/pdf/2601.08919v1)**

> **作者:** Sourav Saha; Mandar Mitra
>
> **摘要:** A good deal of recent research has focused on how Large Language Models (LLMs) may be used as `judges' in place of humans to evaluate the quality of the output produced by various text / image processing systems. Within this broader context, a number of studies have investigated the specific question of how effectively LLMs can be used as relevance assessors for the standard ad hoc task in Information Retrieval (IR). We extend these studies by looking at additional questions. Most importantly, we use a Wikipedia based test collection created by the INEX initiative, and prompt LLMs to not only judge whether documents are relevant / non-relevant, but to highlight relevant passages in documents that it regards as useful. The human relevance assessors involved in creating this collection were given analogous instructions, i.e., they were asked to highlight all passages within a document that respond to the information need expressed in a query. This enables us to evaluate the quality of LLMs as judges not only at the document level, but to also quantify how often these `judges' are right for the right reasons. Our findings suggest that LLMs-as-judges work best under human supervision.
>
---
#### [new 080] Disentangling Task Conflicts in Multi-Task LoRA via Orthogonal Gradient Projection
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于多任务学习领域，解决多任务LoRA中任务冲突导致性能下降的问题。通过正交梯度投影方法，减少任务干扰，提升模型效果。**

- **链接: [https://arxiv.org/pdf/2601.09684v1](https://arxiv.org/pdf/2601.09684v1)**

> **作者:** Ziyu Yang; Guibin Chen; Yuxin Yang; Aoxiong Zeng; Xiangquan Yang
>
> **备注:** preprint
>
> **摘要:** Multi-Task Learning (MTL) combined with Low-Rank Adaptation (LoRA) has emerged as a promising direction for parameter-efficient deployment of Large Language Models (LLMs). By sharing a single adapter across multiple tasks, one can significantly reduce storage overhead. However, this approach suffers from negative transfer, where conflicting gradient updates from distinct tasks degrade the performance of individual tasks compared to single-task fine-tuning. This problem is exacerbated in LoRA due to the low-rank constraint, which limits the optimization landscape's capacity to accommodate diverse task requirements. In this paper, we propose Ortho-LoRA, a gradient projection method specifically tailored for the bipartite structure of LoRA. Ortho-LoRA dynamically projects conflicting task gradients onto the orthogonal complement of each other within the intrinsic LoRA subspace. Extensive experiments on the GLUE benchmark demonstrate that Ortho-LoRA effectively mitigates task interference, outperforming standard joint training and recovering 95\% of the performance gap between multi-task and single-task baselines with negligible computational overhead.
>
---
#### [new 081] ShortCoder: Knowledge-Augmented Syntax Optimization for Token-Efficient Code Generation
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于代码生成任务，旨在提升生成效率。针对传统方法token消耗大问题，提出ShortCoder框架，通过语法简化、数据合成和模型微调，实现更高效且语义一致的代码生成。**

- **链接: [https://arxiv.org/pdf/2601.09703v1](https://arxiv.org/pdf/2601.09703v1)**

> **作者:** Sicong Liu; Yanxian Huang; Mingwei Liu; Jiachi Chen; Ensheng Shi; Yuchi Ma; Hongyu Zhang; Yin Zhang; Yanlin Wang
>
> **摘要:** Code generation tasks aim to automate the conversion of user requirements into executable code, significantly reducing manual development efforts and enhancing software productivity. The emergence of large language models (LLMs) has significantly advanced code generation, though their efficiency is still impacted by certain inherent architectural constraints. Each token generation necessitates a complete inference pass, requiring persistent retention of contextual information in memory and escalating resource consumption. While existing research prioritizes inference-phase optimizations such as prompt compression and model quantization, the generation phase remains underexplored. To tackle these challenges, we propose a knowledge-infused framework named ShortCoder, which optimizes code generation efficiency while preserving semantic equivalence and readability. In particular, we introduce: (1) ten syntax-level simplification rules for Python, derived from AST-preserving transformations, achieving 18.1% token reduction without functional compromise; (2) a hybrid data synthesis pipeline integrating rule-based rewriting with LLM-guided refinement, producing ShorterCodeBench, a corpus of validated tuples of original code and simplified code with semantic consistency; (3) a fine-tuning strategy that injects conciseness awareness into the base LLMs. Extensive experimental results demonstrate that ShortCoder consistently outperforms state-of-the-art methods on HumanEval, achieving an improvement of 18.1%-37.8% in generation efficiency over previous methods while ensuring the performance of code generation.
>
---
#### [new 082] Long-term Task-oriented Agent: Proactive Long-term Intent Maintenance in Dynamic Environments
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于任务导向的智能体研究，解决长期意图维持与环境动态适应问题。提出主动交互范式，构建数据集并设计基准评估模型效果。**

- **链接: [https://arxiv.org/pdf/2601.09382v1](https://arxiv.org/pdf/2601.09382v1)**

> **作者:** Qinglong Shi; Donghai Wang; Hantao Zhou; Jiguo Li; Jun Xu; Jiuchong Gao; Jinghua Hao; Renqing He
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** Current large language model agents predominantly operate under a reactive paradigm, responding only to immediate user queries within short-term sessions. This limitation hinders their ability to maintain long-term user's intents and dynamically adapt to evolving external environments. In this paper, we propose a novel interaction paradigm for proactive Task-oriented Agents capable of bridging the gap between relatively static user's needs and a dynamic environment. We formalize proactivity through two key capabilities, (i) Intent-Conditioned Monitoring: The agent autonomously formulates trigger conditions based on dialog history; (ii) Event-Triggered Follow-up: The agent actively engages the user upon detecting useful environmental updates. We introduce a high-quality data synthesis pipeline to construct complex, multi-turn dialog data in a dynamic environment. Furthermore, we attempt to address the lack of evaluation criteria of task-oriented interaction in a dynamic environment by proposing a new benchmark, namely ChronosBench. We evaluated some leading close-source and open-source models at present and revealed their flaws in long-term task-oriented interaction. Furthermore, our fine-tuned model trained using synthetic data for supervised learning achieves a task completion rate of 85.19% for complex tasks including shifts in user intent, outperforming other models under test. And the result validated the effectiveness of our data-driven strategy.
>
---
#### [new 083] Speech-Hands: A Self-Reflection Voice Agentic Approach to Speech Recognition and Audio Reasoning with Omni Perception
- **分类: cs.SD; cs.AI; cs.CL; cs.MA; eess.AS**

- **简介: 该论文提出Speech-Hands框架，解决语音识别与音频推理中的自我信任与外部感知决策问题，通过自省机制提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2601.09413v1](https://arxiv.org/pdf/2601.09413v1)**

> **作者:** Zhen Wan; Chao-Han Huck Yang; Jinchuan Tian; Hanrong Ye; Ankita Pasad; Szu-wei Fu; Arushi Goel; Ryo Hachiuma; Shizhe Diao; Kunal Dhawan; Sreyan Ghosh; Yusuke Hirota; Zhehuai Chen; Rafael Valle; Ehsan Hosseini Asl; Chenhui Chu; Shinji Watanabe; Yu-Chiang Frank Wang; Boris Ginsburg
>
> **备注:** Preprint. The version was submitted in October 2025
>
> **摘要:** We introduce a voice-agentic framework that learns one critical omni-understanding skill: knowing when to trust itself versus when to consult external audio perception. Our work is motivated by a crucial yet counterintuitive finding: naively fine-tuning an omni-model on both speech recognition and external sound understanding tasks often degrades performance, as the model can be easily misled by noisy hypotheses. To address this, our framework, Speech-Hands, recasts the problem as an explicit self-reflection decision. This learnable reflection primitive proves effective in preventing the model from being derailed by flawed external candidates. We show that this agentic action mechanism generalizes naturally from speech recognition to complex, multiple-choice audio reasoning. Across the OpenASR leaderboard, Speech-Hands consistently outperforms strong baselines by 12.1% WER on seven benchmarks. The model also achieves 77.37% accuracy and high F1 on audio QA decisions, showing robust generalization and reliability across diverse audio question answering datasets. By unifying perception and decision-making, our work offers a practical path toward more reliable and resilient audio intelligence.
>
---
#### [new 084] Navigating Ideation Space: Decomposed Conceptual Representations for Positioning Scientific Ideas
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于科学发现任务，旨在解决文献检索与创新性评估问题。通过构建Ideation Space分解科学知识，实现精准的文献检索和新颖性分析。**

- **链接: [https://arxiv.org/pdf/2601.08901v1](https://arxiv.org/pdf/2601.08901v1)**

> **作者:** Yuexi Shen; Minqian Liu; Dawei Zhou; Lifu Huang
>
> **备注:** 21 pages, 6 tables
>
> **摘要:** Scientific discovery is a cumulative process and requires new ideas to be situated within an ever-expanding landscape of existing knowledge. An emerging and critical challenge is how to identify conceptually relevant prior work from rapidly growing literature, and assess how a new idea differentiates from existing research. Current embedding approaches typically conflate distinct conceptual aspects into single representations and cannot support fine-grained literature retrieval; meanwhile, LLM-based evaluators are subject to sycophancy biases, failing to provide discriminative novelty assessment. To tackle these challenges, we introduce the Ideation Space, a structured representation that decomposes scientific knowledge into three distinct dimensions, i.e., research problem, methodology, and core findings, each learned through contrastive training. This framework enables principled measurement of conceptual distance between ideas, and modeling of ideation transitions that capture the logical connections within a proposed idea. Building upon this representation, we propose a Hierarchical Sub-Space Retrieval framework for efficient, targeted literature retrieval, and a Decomposed Novelty Assessment algorithm that identifies which aspects of an idea are novel. Extensive experiments demonstrate substantial improvements, where our approach achieves Recall@30 of 0.329 (16.7% over baselines), our ideation transition retrieval reaches Hit Rate@30 of 0.643, and novelty assessment attains 0.37 correlation with expert judgments. In summary, our work provides a promising paradigm for future research on accelerating and evaluating scientific discovery.
>
---
#### [new 085] Collaborative Multi-Agent Test-Time Reinforcement Learning for Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出MATTRL框架，解决多智能体强化学习中的稳定性与效率问题，通过注入结构化文本经验提升推理能力。**

- **链接: [https://arxiv.org/pdf/2601.09667v1](https://arxiv.org/pdf/2601.09667v1)**

> **作者:** Zhiyuan Hu; Yunhai Hu; Juncheng Liu; Shuyue Stella Li; Yucheng Wang; Zhen Xu; See-Kiong Ng; Anh Tuan Luu; Xinxing Xu; Bryan Hooi; Cynthia Breazeal; Hae Won Park
>
> **备注:** Work in Progress
>
> **摘要:** Multi-agent systems have evolved into practical LLM-driven collaborators for many applications, gaining robustness from diversity and cross-checking. However, multi-agent RL (MARL) training is resource-intensive and unstable: co-adapting teammates induce non-stationarity, and rewards are often sparse and high-variance. Therefore, we introduce \textbf{Multi-Agent Test-Time Reinforcement Learning (MATTRL)}, a framework that injects structured textual experience into multi-agent deliberation at inference time. MATTRL forms a multi-expert team of specialists for multi-turn discussions, retrieves and integrates test-time experiences, and reaches consensus for final decision-making. We also study credit assignment for constructing a turn-level experience pool, then reinjecting it into the dialogue. Across challenging benchmarks in medicine, math, and education, MATTRL improves accuracy by an average of 3.67\% over a multi-agent baseline, and by 8.67\% over comparable single-agent baselines. Ablation studies examine different credit-assignment schemes and provide a detailed comparison of how they affect training outcomes. MATTRL offers a stable, effective and efficient path to distribution-shift-robust multi-agent reasoning without tuning.
>
---
#### [new 086] Human-AI Co-design for Clinical Prediction Models
- **分类: cs.AI; cs.CL; stat.ME**

- **简介: 该论文属于临床预测模型任务，旨在解决传统协作过程耗时低效的问题。提出HACHI框架，通过人机协作加速可解释模型开发，提升模型效果与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.09072v1](https://arxiv.org/pdf/2601.09072v1)**

> **作者:** Jean Feng; Avni Kothari; Patrick Vossler; Andrew Bishara; Lucas Zier; Newton Addo; Aaron Kornblith; Yan Shuo Tan; Chandan Singh
>
> **摘要:** Developing safe, effective, and practically useful clinical prediction models (CPMs) traditionally requires iterative collaboration between clinical experts, data scientists, and informaticists. This process refines the often small but critical details of the model building process, such as which features/patients to include and how clinical categories should be defined. However, this traditional collaboration process is extremely time- and resource-intensive, resulting in only a small fraction of CPMs reaching clinical practice. This challenge intensifies when teams attempt to incorporate unstructured clinical notes, which can contain an enormous number of concepts. To address this challenge, we introduce HACHI, an iterative human-in-the-loop framework that uses AI agents to accelerate the development of fully interpretable CPMs by enabling the exploration of concepts in clinical notes. HACHI alternates between (i) an AI agent rapidly exploring and evaluating candidate concepts in clinical notes and (ii) clinical and domain experts providing feedback to improve the CPM learning process. HACHI defines concepts as simple yes-no questions that are used in linear models, allowing the clinical AI team to transparently review, refine, and validate the CPM learned in each round. In two real-world prediction tasks (acute kidney injury and traumatic brain injury), HACHI outperforms existing approaches, surfaces new clinically relevant concepts not included in commonly-used CPMs, and improves model generalizability across clinical sites and time periods. Furthermore, HACHI reveals the critical role of the clinical AI team, such as directing the AI agent to explore concepts that it had not previously considered, adjusting the granularity of concepts it considers, changing the objective function to better align with the clinical objectives, and identifying issues of data bias and leakage.
>
---
#### [new 087] Distribution-Aligned Sequence Distillation for Superior Long-CoT Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于推理任务，解决开源模型性能不足问题。提出DASD-4B-Thinking，通过改进序列蒸馏方法提升长链推理能力。**

- **链接: [https://arxiv.org/pdf/2601.09088v1](https://arxiv.org/pdf/2601.09088v1)**

> **作者:** Shaotian Yan; Kaiyuan Liu; Chen Shen; Bing Wang; Sinan Fan; Jun Zhang; Yue Wu; Zheng Wang; Jieping Ye
>
> **备注:** Project Page: https://github.com/D2I-ai/dasd-thinking
>
> **摘要:** In this report, we introduce DASD-4B-Thinking, a lightweight yet highly capable, fully open-source reasoning model. It achieves SOTA performance among open-source models of comparable scale across challenging benchmarks in mathematics, scientific reasoning, and code generation -- even outperforming several larger models. We begin by critically reexamining a widely adopted distillation paradigm in the community: SFT on teacher-generated responses, also known as sequence-level distillation. Although a series of recent works following this scheme have demonstrated remarkable efficiency and strong empirical performance, they are primarily grounded in the SFT perspective. Consequently, these approaches focus predominantly on designing heuristic rules for SFT data filtering, while largely overlooking the core principle of distillation itself -- enabling the student model to learn the teacher's full output distribution so as to inherit its generalization capability. Specifically, we identify three critical limitations in current practice: i) Inadequate representation of the teacher's sequence-level distribution; ii) Misalignment between the teacher's output distribution and the student's learning capacity; and iii) Exposure bias arising from teacher-forced training versus autoregressive inference. In summary, these shortcomings reflect a systemic absence of explicit teacher-student interaction throughout the distillation process, leaving the essence of distillation underexploited. To address these issues, we propose several methodological innovations that collectively form an enhanced sequence-level distillation training pipeline. Remarkably, DASD-4B-Thinking obtains competitive results using only 448K training samples -- an order of magnitude fewer than those employed by most existing open-source efforts. To support community research, we publicly release our models and the training dataset.
>
---
#### [new 088] Permutation Matching Under Parikh Budgets: Linear-Time Detection, Packing, and Disjoint Selection
- **分类: cs.DS; cs.CL**

- **简介: 该论文研究排列模式匹配问题，解决在文本中寻找符合特定字符计数的子串，并优化其打包和非重叠选择。提出线性时间算法。**

- **链接: [https://arxiv.org/pdf/2601.09577v1](https://arxiv.org/pdf/2601.09577v1)**

> **作者:** MD Nazmul Alam Shanto; Md. Tanzeem Rahat; Md. Manzurul Hasan
>
> **备注:** 12 pages (Excluding reference)
>
> **摘要:** We study permutation (jumbled/Abelian) pattern matching over a general alphabet $Σ$. Given a pattern P of length m and a text T of length n, the classical task is to decide whether T contains a length-m substring whose Parikh vector equals that of P . While this existence problem admits a linear-time sliding-window solution, many practical applications require optimization and packing variants beyond mere detection. We present a unified sliding-window framework based on maintaining the Parikh-vector difference between P and the current window of T , enabling permutation matching in O(n + σ) time and O(σ) space, where σ = |Σ|. Building on this foundation, we introduce a combinatorial-optimization variant that we call Maximum Feasible Substring under Pattern Supply (MFSP): find the longest substring S of T whose symbol counts are component-wise bounded by those of P . We show that MFSP can also be solved in O(n + σ) time via a two-pointer feasibility maintenance algorithm, providing an exact packing interpretation of P as a resource budget. Finally, we address non-overlapping occurrence selection by modeling each permutation match as an equal-length interval and proving that a greedy earliest-finishing strategy yields a maximum-cardinality set of disjoint matches, computable in linear time once all matches are enumerated. Our results provide concise, provably correct algorithms with tight bounds, and connect frequency-based string matching to packing-style optimization primitives.
>
---
#### [new 089] EvasionBench: Detecting Evasive Answers in Financial Q&A via Multi-Model Consensus and LLM-as-Judge
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于金融问答中的反规避检测任务，旨在识别财报电话会议中的回避回答。通过多模型共识和LLM作为裁判的方法提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.09142v1](https://arxiv.org/pdf/2601.09142v1)**

> **作者:** Shijian Ma; Yan Lin; Yi Yang
>
> **备注:** Shijian Ma and Yan Lin contributed equally. Corresponding author: Yan Lin
>
> **摘要:** Detecting evasive answers in earnings calls is critical for financial transparency, yet progress is hindered by the lack of large-scale benchmarks. We introduce EvasionBench, comprising 30,000 training samples and 1,000 human-annotated test samples (Cohen's Kappa 0.835) across three evasion levels. Our key contribution is a multi-model annotation framework leveraging a core insight: disagreement between frontier LLMs signals hard examples most valuable for training. We mine boundary cases where two strong annotators conflict, using a judge to resolve labels. This approach outperforms single-model distillation by 2.4 percent, with judge-resolved samples improving generalization despite higher training loss (0.421 vs 0.393) - evidence that disagreement mining acts as implicit regularization. Our trained model Eva-4B (4B parameters) achieves 81.3 percent accuracy, outperforming its base by 25 percentage points and approaching frontier LLM performance at a fraction of inference cost.
>
---
#### [new 090] StegoStylo: Squelching Stylometric Scrutiny through Steganographic Stitching
- **分类: cs.CR; cs.CL; cs.IR**

- **简介: 该论文属于隐私保护任务，旨在对抗stylometric分析。通过对抗性风格攻击和隐写术，降低作者识别准确性，实现身份隐匿。**

- **链接: [https://arxiv.org/pdf/2601.09056v1](https://arxiv.org/pdf/2601.09056v1)**

> **作者:** Robert Dilworth
>
> **备注:** 16 pages, 6 figures, 1 table
>
> **摘要:** Stylometry--the identification of an author through analysis of a text's style (i.e., authorship attribution)--serves many constructive purposes: it supports copyright and plagiarism investigations, aids detection of harmful content, offers exploratory cues for certain medical conditions (e.g., early signs of dementia or depression), provides historical context for literary works, and helps uncover misinformation and disinformation. In contrast, when stylometry is employed as a tool for authorship verification--confirming whether a text truly originates from a claimed author--it can also be weaponized for malicious purposes. Techniques such as de-anonymization, re-identification, tracking, profiling, and downstream effects like censorship illustrate the privacy threats that stylometric analysis can enable. Building on these concerns, this paper further explores how adversarial stylometry combined with steganography can counteract stylometric analysis. We first present enhancements to our adversarial attack, $\textit{TraceTarnish}$, providing stronger evidence of its capacity to confound stylometric systems and reduce their attribution and verification accuracy. Next, we examine how steganographic embedding can be fine-tuned to mask an author's stylistic fingerprint, quantifying the level of authorship obfuscation achievable as a function of the proportion of words altered with zero-width Unicode characters. Based on our findings, steganographic coverage of 33% or higher seemingly ensures authorship obfuscation. Finally, we reflect on the ways stylometry can be used to undermine privacy and argue for the necessity of defensive tools like $\textit{TraceTarnish}$.
>
---
#### [new 091] Reading or Reasoning? Format Decoupled Reinforcement Learning for Document OCR
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于文档OCR任务，旨在解决格式敏感文本识别中的高不确定性问题。通过引入FD-RL方法，提升OCR模型性能。**

- **链接: [https://arxiv.org/pdf/2601.08834v1](https://arxiv.org/pdf/2601.08834v1)**

> **作者:** Yufeng Zhong; Lei Chen; Zhixiong Zeng; Xuanle Zhao; Deyang Jiang; Liming Zheng; Jing Huang; Haibo Qiu; Peng Shi; Siqi Yang; Lin Ma
>
> **备注:** technical report
>
> **摘要:** Reading text from images or scanned documents via OCR models has been a longstanding focus of researchers. Intuitively, text reading is perceived as a straightforward perceptual task, and existing work primarily focuses on constructing enriched data engineering to enhance SFT capabilities. In this work, we observe that even advanced OCR models exhibit significantly higher entropy in formatted text (\emph{e.g.}, formula, table, etc.) compared to plain text, often by an order of magnitude. These statistical patterns reveal that advanced OCR models struggle with high output uncertainty when dealing with format sensitive document, suggesting that reasoning over diverse reading pathways may improve OCR performance. To address this, we propose format decoupled reinforcement learning (FD-RL), which leverages high-entropy patterns for targeted optimization. Our approach employs entropy-based data filtration strategy to identify format-intensive instances, and adopt format decoupled rewards tailored to different format types, enabling format-level validation rather than token-level memorization. FD-RL achieves an average score of 90.41 on OmniDocBench, setting a new record for end-to-end models on this highly popular benchmark. More importantly, we conduct comprehensive ablation studies over data, training, filtering, and rewarding strategies, thoroughly validating their effectiveness.
>
---
#### [new 092] Spectral Generative Flow Models: A Physics-Inspired Replacement for Vectorized Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出一种基于物理的生成模型SGFMs，用于替代传统语言模型。旨在解决长距离连贯性和物理结构问题，通过连续场演化实现高效生成。**

- **链接: [https://arxiv.org/pdf/2601.08893v1](https://arxiv.org/pdf/2601.08893v1)**

> **作者:** Andrew Kiruluta
>
> **摘要:** We introduce Spectral Generative Flow Models (SGFMs), a physics-inspired alternative to transformer-based large language models. Instead of representing text or video as sequences of discrete tokens processed by attention, SGFMs treat generation as the evolution of a continuous field governed by constrained stochastic dynamics in a multiscale wavelet basis. This formulation replaces global attention with local operators, spectral projections, and Navier--Stokes-like transport, yielding a generative mechanism grounded in continuity, geometry, and physical structure. Our framework provides three key innovations: (i) a field-theoretic ontology in which text and video are unified as trajectories of a stochastic partial differential equation; (ii) a wavelet-domain representation that induces sparsity, scale separation, and computational efficiency; and (iii) a constrained stochastic flow that enforces stability, coherence, and uncertainty propagation. Together, these components define a generative architecture that departs fundamentally from autoregressive modeling and diffusion-based approaches. SGFMs offer a principled path toward long-range coherence, multimodal generality, and physically structured inductive bias in next-generation generative models.
>
---
## 更新

#### [replaced 001] Mathematical Derivation Graphs: A Relation Extraction Task in STEM Manuscripts
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于关系抽取任务，旨在解决STEM文本中数学表达式依赖关系的提取问题。作者构建了MDGD数据集，并评估了模型在该任务上的表现。**

- **链接: [https://arxiv.org/pdf/2410.21324v2](https://arxiv.org/pdf/2410.21324v2)**

> **作者:** Vishesh Prasad; Brian Kim; Nickvash Kani
>
> **备注:** 29 pages, 11 figures
>
> **摘要:** Recent advances in natural language processing (NLP), particularly with the emergence of large language models (LLMs), have significantly enhanced the field of textual analysis. However, while these developments have yielded substantial progress in analyzing natural language text, applying analysis to mathematical equations and their relationships within texts has produced mixed results. This paper takes the initial steps in expanding the problem of relation extraction towards understanding the dependency relationships between mathematical expressions in STEM articles. The authors construct the Mathematical Derivation Graphs Dataset (MDGD), sourced from a random sampling of the arXiv corpus, containing an analysis of $107$ published STEM manuscripts with over $2000$ manually labeled inter-equation dependency relationships, resulting in a new object referred to as a derivation graph that summarizes the mathematical content of the manuscript. The authors exhaustively evaluate analytical and machine learning (ML) based models to assess their capability to identify and extract the derivation relationships for each article and compare the results with the ground truth. The authors show that the best tested LLMs achieve $F_1$ scores of $\sim45\%-52\%$, and attempt to improve their performance by combining them with analytic algorithms and other methods.
>
---
#### [replaced 002] Navigating the Reality Gap: Privacy-Preserving On-Device Continual Adaptation of ASR for Clinical Telephony
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决临床电话场景中ASR的现实差距问题。通过本地持续适应框架提升模型性能，同时保障隐私。**

- **链接: [https://arxiv.org/pdf/2512.16401v4](https://arxiv.org/pdf/2512.16401v4)**

> **作者:** Darshil Chauhan; Adityasinh Solanki; Vansh Patel; Kanav Kapoor; Ritvik Jain; Aditya Bansal; Pratik Narang; Dhruv Kumar
>
> **备注:** 17 pages, 13 figures. Under review
>
> **摘要:** Automatic Speech Recognition (ASR) holds immense potential to assist in clinical documentation and patient report generation, particularly in resource-constrained regions. However, deployment is currently hindered by a technical deadlock: a severe "Reality Gap" between laboratory performance and noisy, real-world clinical audio, coupled with strict privacy and resource constraints. Such adaptation is essential for clinical telephony systems, where patient speech is highly variable and transcription errors can directly impact downstream clinical workflows. We quantify this gap, showing that a robust multilingual model (IndicWav2Vec) degrades up to a 40.94% WER on rural clinical telephony speech from India, rendering it unusable. We demonstrate consistent improvements on these helpline interactions without transmitting raw patient data off-device via an on-device continual adaptation framework using Low-Rank Adaptation (LoRA). We conduct an investigative study of stabilization strategies, characterizing the trade-offs between data-driven and parameter-driven approaches. Our results demonstrate that multi-domain Experience Replay (ER) yields the primary performance gains, achieving a 17.1% relative improvement in target WER and reducing catastrophic forgetting by 55% compared to naive adaptation. Furthermore, we investigate a stabilized importance estimation strategy (Absolute Fisher) to ensure robust convergence against the high-variance gradients common in clinical telephony speech. Finally, we verify via a domain-specific spot check that acoustic adaptation is a fundamental prerequisite for usability in healthcare settings which cannot be bypassed by language models alone.
>
---
#### [replaced 003] Higher Satisfaction, Lower Cost: A Technical Report on How LLMs Revolutionize Meituan's Intelligent Interaction Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于智能交互系统任务，解决数据构建、对话性能、规则适应、多智能体协作及评估等问题，提出WOWService系统提升用户体验与效率。**

- **链接: [https://arxiv.org/pdf/2510.13291v2](https://arxiv.org/pdf/2510.13291v2)**

> **作者:** Xuxin Cheng; Ke Zeng; Zhiquan Cao; Linyi Dai; Wenxuan Gao; Fei Han; Ai Jian; Feng Hong; Wenxing Hu; Zihe Huang; Dejian Kong; Jia Leng; Zhuoyuan Liao; Pei Liu; Jiaye Lin; Xing Ma; Jingqing Ruan; Jiaxing Song; Xiaoyu Tan; Ruixuan Xiao; Wenhui Yu; Wenyu Zhan; Haoxing Zhang; Chao Zhou; Hao Zhou; Shaodong Zheng; Ruinian Chen; Siyuan Chen; Ziyang Chen; Yiwen Dong; Yaoyou Fan; Yangyi Fang; Yang Gan; Shiguang Guo; Qi He; Chaowen Hu; Binghui Li; Dailin Li; Xiangyu Li; Yan Li; Chengjian Liu; Xiangfeng Liu; Jiahui Lv; Qiao Ma; Jiang Pan; Cong Qin; Chenxing Sun; Wen Sun; Zhonghui Wang; Abudukelimu Wuerkaixi; Xin Yang; Fangyi Yuan; Yawen Zhu; Tianyi Zhai; Jie Zhang; Runlai Zhang; Yao Xu; Yiran Zhao; Yifan Wang; Xunliang Cai; Yangen Hu; Cao Liu; Lu Pan; Xiaoli Wang; Bo Xiao; Wenyuan Yao; Qianlin Zhou; Benchang Zhu
>
> **备注:** 36 pages, 14 figures
>
> **摘要:** Enhancing customer experience is essential for business success, particularly as service demands grow in scale and complexity. Generative artificial intelligence and Large Language Models (LLMs) have empowered intelligent interaction systems to deliver efficient, personalized, and 24/7 support. In practice, intelligent interaction systems encounter several challenges: (1) Constructing high-quality data for cold-start training is difficult, hindering self-evolution and raising labor costs. (2) Multi-turn dialogue performance remains suboptimal due to inadequate intent understanding, rule compliance, and solution extraction. (3) Frequent evolution of business rules affects system operability and transferability, constraining low-cost expansion and adaptability. (4) Reliance on a single LLM is insufficient in complex scenarios, where the absence of multi-agent frameworks and effective collaboration undermines process completeness and service quality. (5) The open-domain nature of multi-turn dialogues, lacking unified golden answers, hampers quantitative evaluation and continuous optimization. To address these challenges, we introduce WOWService, an intelligent interaction system tailored for industrial applications. With the integration of LLMs and multi-agent architectures, WOWService enables autonomous task management and collaborative problem-solving. Specifically, WOWService focuses on core modules including data construction, general capability enhancement, business scenario adaptation, multi-agent coordination, and automated evaluation. Currently, WOWService is deployed on the Meituan App, achieving significant gains in key metrics, e.g., User Satisfaction Metric 1 (USM 1) -27.53% and User Satisfaction Metric 2 (USM 2) +25.51%, demonstrating its effectiveness in capturing user needs and advancing personalized service.
>
---
#### [replaced 004] Agent Bain vs. Agent McKinsey: A New Text-to-SQL Benchmark for the Business Domain
- **分类: cs.CL**

- **简介: 该论文提出CORGI基准，扩展文本到SQL任务，解决商业领域复杂查询问题，涵盖描述、解释、预测和推荐类问题。**

- **链接: [https://arxiv.org/pdf/2510.07309v4](https://arxiv.org/pdf/2510.07309v4)**

> **作者:** Yue Li; Ran Tao; Derek Hommel; Yusuf Denizay Dönder; Sungyong Chang; David Mimno; Unso Eun Seo Jo
>
> **备注:** 23 pages, under review for ACL ARR
>
> **摘要:** Text-to-SQL benchmarks have traditionally only tested simple data access as a translation task of natural language to SQL queries. But in reality, users tend to ask diverse questions that require more complex responses including data-driven predictions or recommendations. Using the business domain as a motivating example, we introduce CORGI, a new benchmark that expands text-to-SQL to reflect practical database queries encountered by end users. CORGI is composed of synthetic databases inspired by enterprises such as DoorDash, Airbnb, and Lululemon. It provides questions across four increasingly complicated categories of business queries: descriptive, explanatory, predictive, and recommendational. This challenge calls for causal reasoning, temporal forecasting, and strategic recommendation, reflecting multi-level and multi-step agentic intelligence. We find that LLM performance degrades on higher-level questions as question complexity increases. CORGI also introduces and encourages the text-to-SQL community to consider new automatic methods for evaluating open-ended, qualitative responses in data access tasks. Our experiments show that LLMs exhibit an average 33.12% lower success execution rate (SER) on CORGI compared to existing benchmarks such as BIRD, highlighting the substantially higher complexity of real-world business needs. We release the CORGI dataset, an evaluation framework, and a submission website to support future research.
>
---
#### [replaced 005] Template-Based Probes Are Imperfect Lenses for Counterfactual Bias Evaluation in LLMs
- **分类: cs.CL; cs.CY; cs.LG**

- **简介: 该论文属于自然语言处理中的偏见评估任务，旨在解决模板探针在衡量语言模型偏见时可能引入的系统性偏差问题。研究发现模板探针可能导致对白人文本的负面分类率异常偏高，揭示了语言数据中的不对称性影响评估结果。**

- **链接: [https://arxiv.org/pdf/2404.03471v5](https://arxiv.org/pdf/2404.03471v5)**

> **作者:** Farnaz Kohankhaki; D. B. Emerson; Jacob-Junqi Tian; Laleh Seyyed-Kalantari; Faiza Khan Khattak
>
> **备注:** 22 Pages, 6 Figures, 5 Tables
>
> **摘要:** Bias in large language models (LLMs) has many forms, from overt discrimination to implicit stereotypes. Counterfactual bias evaluation is a widely used approach to quantifying bias and often relies on template-based probes that explicitly state group membership. It aims to measure whether the outcome of a task performed by an LLM is invariant to a change in group membership. In this work, we find that template-based probes can introduce systematic distortions in bias measurements. Specifically, we consistently find that such probes suggest that LLMs classify text associated with White race as negative at disproportionately elevated rates. This is observed consistently across a large collection of LLMs, over several diverse template-based probes, and with different classification approaches. We hypothesize that this arises artificially due to linguistic asymmetries present in LLM pretraining data, in the form of markedness, (e.g., Black president vs. president) and templates used for bias measurement (e.g., Black president vs. White president). These findings highlight the need for more rigorous methodologies in counterfactual bias evaluation, ensuring that observed disparities reflect genuine biases rather than artifacts of linguistic conventions.
>
---
#### [replaced 006] Do Language Models Associate Sound with Meaning? A Multimodal Study of Sound Symbolism
- **分类: cs.CL**

- **简介: 该论文属于语言模型与声音符号学的交叉研究，旨在探讨多模态大语言模型是否能关联声音与意义。通过构建数据集并分析模型对语音象征性的理解，揭示其处理声音信息的机制。**

- **链接: [https://arxiv.org/pdf/2511.10045v5](https://arxiv.org/pdf/2511.10045v5)**

> **作者:** Jinhong Jeong; Sunghyun Lee; Jaeyoung Lee; Seonah Han; Youngjae Yu
>
> **备注:** 33 pages, 27 tables, 10 figures
>
> **摘要:** Sound symbolism is a linguistic concept that refers to non-arbitrary associations between phonetic forms and their meanings. We suggest that this can be a compelling probe into how Multimodal Large Language Models (MLLMs) interpret auditory information in human languages. We investigate MLLMs' performance on phonetic iconicity across textual (orthographic and IPA) and auditory forms of inputs with up to 25 semantic dimensions (e.g., sharp vs. round), observing models' layer-wise information processing by measuring phoneme-level attention fraction scores. To this end, we present LEX-ICON, an extensive mimetic word dataset consisting of 8,052 words from four natural languages (English, French, Japanese, and Korean) and 2,930 systematically constructed pseudo-words, annotated with semantic features applied across both text and audio modalities. Our key findings demonstrate (1) MLLMs' phonetic intuitions that align with existing linguistic research across multiple semantic dimensions and (2) phonosemantic attention patterns that highlight models' focus on iconic phonemes. These results bridge domains of artificial intelligence and cognitive linguistics, providing the first large-scale, quantitative analyses of phonetic iconicity in terms of MLLMs' interpretability.
>
---
#### [replaced 007] TeleMem: Building Long-Term and Multimodal Memory for Agentic AI
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出TeleMem，解决AI长期交互与多模态记忆问题。通过动态提取和结构化写入，提升记忆效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.06037v2](https://arxiv.org/pdf/2601.06037v2)**

> **作者:** Chunliang Chen; Ming Guan; Xiao Lin; Jiaxu Li; Qiyi Wang; Xiangyu Chen; Jixiang Luo; Changzhi Sun; Dell Zhang; Xuelong Li
>
> **摘要:** Large language models (LLMs) excel at many NLP tasks but struggle to sustain long-term interactions due to limited attention over extended dialogue histories. Retrieval-augmented generation (RAG) mitigates this issue but lacks reliable mechanisms for updating or refining stored memories, leading to schema-driven hallucinations, inefficient write operations, and minimal support for multimodal reasoning.To address these challenges, we propose TeleMem, a unified long-term and multimodal memory system that maintains coherent user profiles through narrative dynamic extraction, ensuring that only dialogue-grounded information is preserved. TeleMem further introduces a structured writing pipeline that batches, retrieves, clusters, and consolidates memory entries, substantially improving storage efficiency, reducing token usage, and accelerating memory operations. Additionally, a multimodal memory module combined with ReAct-style reasoning equips the system with a closed-loop observe, think, and act process that enables accurate understanding of complex video content in long-term contexts. Experimental results show that TeleMem surpasses the state-of-the-art Mem0 baseline with 19% higher accuracy, 43% fewer tokens, and a 2.1x speedup on the ZH-4O long-term role-play gaming benchmark.
>
---
#### [replaced 008] DiffSampling: Enhancing Diversity and Accuracy in Neural Text Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于文本生成任务，旨在解决语言模型生成内容重复、缺乏多样性及准确性的问题。提出DiffSampling方法，通过分析概率分布提升生成质量。**

- **链接: [https://arxiv.org/pdf/2502.14037v5](https://arxiv.org/pdf/2502.14037v5)**

> **作者:** Giorgio Franceschelli; Mirco Musolesi
>
> **备注:** Published in Transactions on Machine Learning Research (2025), see https://tmlr.infinite-conf.org/paper_pages/kXjHbMvdIi.html
>
> **摘要:** Despite their growing capabilities, language models still frequently reproduce content from their training data, generate repetitive text, and favor common grammatical patterns and vocabulary. A possible cause is the decoding strategy: the most common strategies either consider only the most probable tokens, which reduces output diversity, or increase the likelihood of unlikely tokens, compromising output accuracy and correctness. In this paper, we propose DiffSampling, a new decoding method that leverages a mathematical analysis of the token probability distribution to ensure the generation of contextually appropriate text. In particular, the difference between consecutive, sorted probabilities can be used to truncate incorrect tokens. In addition, we also propose two variations of the proposed method that aim to correct the subtle inconsistencies of common sampling strategies. Experiments involving four different text-generation tasks demonstrate that our approach consistently performs at least on par with the existing methods it builds upon in terms of quality, despite sampling from a larger set of tokens.
>
---
#### [replaced 009] Bench360: Benchmarking Local LLM Inference from 360 Degrees
- **分类: cs.CL; cs.AI; cs.LG; cs.PF**

- **简介: 该论文属于大模型推理优化任务，旨在解决本地LLM推理评估碎片化问题。提出Bench360框架，统一评估多任务、多引擎的推理性能与质量。**

- **链接: [https://arxiv.org/pdf/2511.16682v2](https://arxiv.org/pdf/2511.16682v2)**

> **作者:** Linus Stuhlmann; Mauricio Fadel Argerich; Jonathan Fürst
>
> **摘要:** Running LLMs locally has become increasingly common, but users face a complex design space across models, quantization levels, inference engines, and serving scenarios. Existing inference benchmarks are fragmented and focus on isolated goals, offering little guidance for practical deployments. We present Bench360, a framework for evaluating local LLM inference across tasks, usage patterns, and system metrics in one place. Bench360 supports custom tasks, integrates multiple inference engines and quantization formats, and reports both task quality and system behavior (latency, throughput, energy, startup time). We demonstrate it on four NLP tasks across three GPUs and four engines, showing how design choices shape efficiency and output quality. Results confirm that tradeoffs are substantial and configuration choices depend on specific workloads and constraints. There is no universal best option, underscoring the need for comprehensive, deployment-oriented benchmarks.
>
---
#### [replaced 010] Word Synchronization Challenge: A Benchmark for Word Association Responses for Large Language Models
- **分类: cs.HC; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型在人机交互中的词义同步能力。通过构建动态游戏框架，测试模型模仿人类认知过程的能力，以提升人机协作的拟人性与准确性。**

- **链接: [https://arxiv.org/pdf/2502.08312v2](https://arxiv.org/pdf/2502.08312v2)**

> **作者:** Tanguy Cazalets; Joni Dambre
>
> **摘要:** This paper introduces the Word Synchronization Challenge, a novel benchmark to evaluate large language models (LLMs) in Human-Computer Interaction (HCI). This benchmark uses a dynamic game-like framework to test LLMs ability to mimic human cognitive processes through word associations. By simulating complex human interactions, it assesses how LLMs interpret and align with human thought patterns during conversational exchanges, which are essential for effective social partnerships in HCI. Initial findings highlight the influence of model sophistication on performance, offering insights into the models capabilities to engage in meaningful social interactions and adapt behaviors in human-like ways. This research advances the understanding of LLMs potential to replicate or diverge from human cognitive functions, paving the way for more nuanced and empathetic human-machine collaborations.
>
---
#### [replaced 011] DR-LoRA: Dynamic Rank LoRA for Mixture-of-Experts Adaptation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于模型微调任务，解决MoE模型中参数分配不均的问题。通过动态调整LoRA秩，提升参数利用率和任务性能。**

- **链接: [https://arxiv.org/pdf/2601.04823v3](https://arxiv.org/pdf/2601.04823v3)**

> **作者:** Guanzhi Deng; Bo Li; Ronghao Chen; Huacan Wang; Lijie Wen; Linqi Song
>
> **摘要:** Mixture-of-Experts (MoE) has become a prominent paradigm for scaling Large Language Models (LLMs). Parameter-efficient fine-tuning (PEFT), such as LoRA, is widely adopted to adapt pretrained MoE LLMs to downstream tasks. However, existing approaches assign identical LoRA ranks to all experts, overlooking the intrinsic functional specialization within MoE LLMs. This uniform allocation leads to resource mismatch, task-relevant experts are under-provisioned while less relevant ones receive redundant parameters. We propose a Dynamic Rank LoRA framework named DR-LoRA, which dynamically grows expert LoRA ranks during fine-tuning based on task-specific demands. DR-LoRA employs an Expert Saliency Scoring mechanism that integrates expert routing frequency and LoRA rank importance to quantify each expert's demand for additional capacity. Experts with higher saliency scores are prioritized for rank expansion, enabling the automatic formation of a heterogeneous rank distribution tailored to the target task. Experiments on multiple benchmarks demonstrate that DR-LoRA consistently outperforms standard LoRA and static allocation strategies under the same parameter budget, achieving superior task performance with more efficient parameter utilization.
>
---
#### [replaced 012] FigEx2: Visual-Conditioned Panel Detection and Captioning for Scientific Compound Figures
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 本文提出FigEx2，解决科学复合图中面板检测与描述生成任务，通过视觉条件框架实现面板定位与细粒度描述，提升多模态一致性与跨领域泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.08026v2](https://arxiv.org/pdf/2601.08026v2)**

> **作者:** Jifeng Song; Arun Das; Pan Wang; Hui Ji; Kun Zhao; Yufei Huang
>
> **摘要:** Scientific compound figures combine multiple labeled panels into a single image, but captions in real pipelines are often missing or only provide figure-level summaries, making panel-level understanding difficult. In this paper, we propose FigEx2, visual-conditioned framework that localizes panels and generates panel-wise captions directly from the compound figure. To mitigate the impact of diverse phrasing in open-ended captioning, we introduce a noise-aware gated fusion module that adaptively filters token-level features to stabilize the detection query space. Furthermore, we employ a staged optimization strategy combining supervised learning with reinforcement learning (RL), utilizing CLIP-based alignment and BERTScore-based semantic rewards to enforce strict multimodal consistency. To support high-quality supervision, we curate BioSci-Fig-Cap, a refined benchmark for panel-level grounding, alongside cross-disciplinary test suites in physics and chemistry. Experimental results demonstrate that FigEx2 achieves a superior 0.726 mAP@0.5:0.95 for detection and significantly outperforms Qwen3-VL-8B by 0.51 in METEOR and 0.24 in BERTScore. Notably, FigEx2 exhibits remarkable zero-shot transferability to out-of-distribution scientific domains without any fine-tuning.
>
---
#### [replaced 013] Toward Conversational Hungarian Speech Recognition: Introducing the BEA-Large and BEA-Dialogue Datasets
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决匈牙利语对话语音数据不足的问题。通过构建两个新数据集并建立基线模型，推动匈牙利语语音技术发展。**

- **链接: [https://arxiv.org/pdf/2511.13529v2](https://arxiv.org/pdf/2511.13529v2)**

> **作者:** Máté Gedeon; Piroska Zsófia Barta; Péter Mihajlik; Tekla Etelka Gráczi; Anna Kohári; Katalin Mády
>
> **备注:** Submitted to LREC 2026
>
> **摘要:** The advancement of automatic speech recognition (ASR) has been largely enhanced by extensive datasets in high-resource languages, while languages such as Hungarian remain underrepresented due to limited spontaneous and conversational corpora. To address this gap, we introduce two new datasets -- BEA-Large and BEA-Dialogue -- constructed from the previously unprocessed portions of the Hungarian speech corpus named BEA. BEA-Large extends BEA-Base with 255 hours of spontaneous speech from 433 speakers, enriched with detailed segment-level metadata. BEA-Dialogue, comprising 85 hours of spontaneous conversations, is a Hungarian speech corpus featuring natural dialogues partitioned into speaker-independent subsets, supporting research in conversational ASR and speaker diarization. We establish reproducible baselines on these datasets using publicly available ASR models, with the fine-tuned Fast Conformer model achieving word error rates as low as 14.18% on spontaneous and 4.8% on repeated speech. Diarization experiments yield diarization error rates between 12.46% and 17.40%, providing reference points for future improvements. The results highlight the persistent difficulty of conversational ASR, particularly due to disfluencies, overlaps, and informal speech patterns. By releasing these datasets and baselines, we aim to advance Hungarian speech technology and offer a methodological framework for developing spontaneous and conversational benchmarks in other languages.
>
---
#### [replaced 014] Sentiment Analysis Of Shopee Product Reviews Using Distilbert
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在高效处理Shopee产品评论。通过使用DistilBERT模型进行分类，提升分析效率与准确性。**

- **链接: [https://arxiv.org/pdf/2511.22313v2](https://arxiv.org/pdf/2511.22313v2)**

> **作者:** Zahri Aksa Dautd; Aviv Yuniar Rahman
>
> **备注:** The authors have decided to withdraw this manuscript because substantial improvements are needed in the methodology, data analysis, and presentation of the research results to ensure the article's scientific quality meets journal publication standards. Therefore, the authors plan to conduct a thorough revision before resubmitting to an appropriate journal
>
> **摘要:** The rapid growth of digital commerce has led to the accumulation of a massive number of consumer reviews on online platforms. Shopee, as one of the largest e-commerce platforms in Southeast Asia, receives millions of product reviews every day containing valuable information regarding customer satisfaction and preferences. Manual analysis of these reviews is inefficient, thus requiring a computational approach such as sentiment analysis. This study examines the use of DistilBERT, a lightweight transformer-based deep learning model, for sentiment classification on Shopee product reviews. The dataset used consists of approximately one million English-language reviews that have been preprocessed and trained using the distilbert-base-uncased model. Evaluation was conducted using accuracy, precision, recall, and F1-score metrics, and compared against benchmark models such as BERT and SVM. The results show that DistilBERT achieved an accuracy of 94.8%, slightly below BERT (95.3%) but significantly higher than SVM (90.2%), with computation time reduced by more than 55%. These findings demonstrate that DistilBERT provides an optimal balance between accuracy and efficiency, making it suitable for large scale sentiment analysis on e-commerce platforms. Keywords: Sentiment Analysis, DistilBERT, Shopee Reviews, Natural Language Processing, Deep Learning, Transformer Models.
>
---
#### [replaced 015] AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出AutoToM，解决理论心智（ToM）推理问题，通过自动化代理建模实现可扩展、鲁棒的内心推断。**

- **链接: [https://arxiv.org/pdf/2502.15676v3](https://arxiv.org/pdf/2502.15676v3)**

> **作者:** Zhining Zhang; Chuanyang Jin; Mung Yao Jia; Shunchi Zhang; Tianmin Shu
>
> **备注:** NeurIPS 2025 (Spotlight). 42 pages, 11 figures, 15 tables. Website at https://chuanyangjin.com/AutoToM/
>
> **摘要:** Theory of Mind (ToM), the ability to understand people's minds based on their behavior, is key to developing socially intelligent agents. Current approaches to ToM reasoning either rely on prompting Large Language Models (LLMs), which are prone to systematic errors, or use handcrafted, rigid agent models for model-based inference, which are more robust but fail to generalize across domains. In this work, we introduce AutoToM, an automated agent modeling method for scalable, robust, and interpretable mental inference. Given a ToM problem, AutoToM first proposes an initial agent model and then performs automated Bayesian inverse planning based on this model, leveraging an LLM backend. Guided by inference uncertainty, it iteratively refines the model by introducing additional mental variables and/or incorporating more timesteps in the context. Across five diverse benchmarks, AutoToM outperforms existing ToM methods and even large reasoning models. Additionally, we show that AutoToM can produce human-like confidence estimates and enable online mental inference for embodied decision-making.
>
---
#### [replaced 016] LingVarBench: Benchmarking LLMs on Entity Recognitions and Linguistic Verbalization Patterns in Phone-Call Transcripts
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于实体识别任务，旨在解决电话转录文本中实体提取困难的问题。通过构建LingVarBench基准和生成合成数据，提升模型对语言变化的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2508.15801v2](https://arxiv.org/pdf/2508.15801v2)**

> **作者:** Seyedali Mohammadi; Manas Paldhe; Amit Chhabra; Youngseo Son; Vishal Seshagiri
>
> **备注:** Accepted to EACL 2026 (Industry Track); to appear in the proceedings
>
> **摘要:** We study structured entity extraction from phone-call transcripts in customer-support and healthcare settings, where annotation is costly, and data access is limited by privacy and consent. Existing methods degrade under disfluencies, interruptions, and speaker overlap, yet large real-call corpora are rarely shareable. We introduce LingVarBench, a benchmark and semantic synthetic data generation pipeline that generates linguistically varied training data via (1) LLM-sampled entity values, (2) curated linguistic verbalization patterns covering diverse disfluencies and entity-specific readout styles, and (3) a value-transcript consistency filter. Using this dataset, DSPy's SIMBA automatically synthesizes and optimizes extraction prompts, reducing manual prompt engineering and targeting robustness to verbal variation. On real customer transcripts, prompts optimized solely on LingVarBench outperform zero-shot baselines and match or closely approach human-tuned prompts for structured entities such as ZIP code, date of birth, and name (F1 approximately 94-95 percent). For subjective questionnaire items, optimized prompts substantially improve over zero-shot performance and approach human-tuned prompts. LingVarBench offers a practical and cost-efficient path to deployment in a direct-answer setting, with real annotations later enabling additional refinement.
>
---
#### [replaced 017] Stuttering-Aware Automatic Speech Recognition for Indonesian Language
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决印尼语口吃语音识别性能下降的问题。通过生成合成口吃数据并微调预训练模型，提升系统对口吃语音的识别能力。**

- **链接: [https://arxiv.org/pdf/2601.03727v2](https://arxiv.org/pdf/2601.03727v2)**

> **作者:** Fadhil Muhammad; Alwin Djuliansah; Adrian Aryaputra Hamzah; Kurniawati Azizah
>
> **备注:** Preprint
>
> **摘要:** Automatic speech recognition systems have achieved remarkable performance on fluent speech but continue to degrade significantly when processing stuttered speech, a limitation that is particularly acute for low-resource languages like Indonesian where specialized datasets are virtually non-existent. To overcome this scarcity, we propose a data augmentation framework that generates synthetic stuttered audio by injecting repetitions and prolongations into fluent text through a combination of rule-based transformations and large language models followed by text-to-speech synthesis. We apply this synthetic data to fine-tune a pre-trained Indonesian Whisper model using transfer learning, enabling the architecture to adapt to dysfluent acoustic patterns without requiring large-scale real-world recordings. Our experiments demonstrate that this targeted synthetic exposure consistently reduces recognition errors on stuttered speech while maintaining performance on fluent segments, validating the utility of synthetic data pipelines for developing more inclusive speech technologies in under-represented languages.
>
---
#### [replaced 018] Temporal Knowledge Graph Question Answering: A Survey
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 本文属于时间知识图谱问答任务，旨在解决时间问题定义模糊和方法分类不清的问题。梳理了时间问题分类并综述了相关技术方法。**

- **链接: [https://arxiv.org/pdf/2406.14191v4](https://arxiv.org/pdf/2406.14191v4)**

> **作者:** Miao Su; Zixuan Li; Zhuo Chen; Long Bai; Xiaolong Jin; Jiafeng Guo
>
> **备注:** 8 pages, 3 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Knowledge Base Question Answering (KBQA) has been a long-standing field to answer questions based on knowledge bases. Recently, the evolving dynamics of knowledge have attracted a growing interest in Temporal Knowledge Graph Question Answering (TKGQA), an emerging task to answer temporal questions. However, this field grapples with ambiguities in defining temporal questions and lacks a systematic categorization of existing methods for TKGQA. In response, this paper provides a thorough survey from two perspectives: the taxonomy of temporal questions and the methodological categorization for TKGQA. Specifically, we first establish a detailed taxonomy of temporal questions engaged in prior studies. Subsequently, we provide a comprehensive review of TKGQA techniques of two categories: semantic parsing-based and TKG embedding-based. Building on this review, the paper outlines potential research directions aimed at advancing the field of TKGQA. This work aims to serve as a comprehensive reference for TKGQA and to stimulate further research.
>
---
#### [replaced 019] SOP-Maze: Evaluating Large Language Models on Complicated Business Standard Operating Procedures
- **分类: cs.CL**

- **简介: 该论文提出SOP-Maze基准，评估大语言模型在复杂业务流程中的表现，解决其在遵循规程、逻辑推理和计算准确性方面的不足。**

- **链接: [https://arxiv.org/pdf/2510.08942v2](https://arxiv.org/pdf/2510.08942v2)**

> **作者:** Jiaming Wang; Zhe Tang; Zehao Jin; Hefei Chen; Yilin Jin; Peng Ding; Xiaoyu Li; Xuezhi Cao
>
> **摘要:** As large language models (LLMs) are widely deployed as domain-specific agents, many benchmarks have been proposed to evaluate their ability to follow instructions and make decisions in real-world scenarios. However, business scenarios often involve complex standard operating procedures (SOPs), and the evaluation of LLM capabilities in such contexts has not been fully explored. To bridge this gap, we propose SOP-Maze, a benchmark constructed from real-world business data and adapted into a collection of 397 instances and 3422 subtasks from 23 complex SOP scenarios. We further categorize SOP tasks into two broad classes: Lateral Root System (LRS), representing wide-option tasks that demand precise selection; and Heart Root System (HRS), which emphasizes deep logical reasoning with complex branches. Extensive experiments reveal that nearly all state-of-the-art models struggle with SOP-Maze. We conduct a comprehensive analysis and identify three key error categories: (i) route blindness: difficulty following procedures; (ii) conversational fragility: inability to handle real dialogue nuances; and (iii) calculation errors: mistakes in time or arithmetic reasoning under complex contexts. The systematic study explores LLM performance across SOP tasks that challenge both breadth and depth, offering new insights for improving model capabilities. We have open-sourced our work on: https://github.com/meituan-longcat/SOP-Maze.
>
---
#### [replaced 020] The Agentic Leash: Extracting Causal Feedback Fuzzy Cognitive Maps with LLMs
- **分类: cs.AI; cs.CL; cs.HC; cs.IR**

- **简介: 该论文属于知识提取任务，旨在通过LLM生成因果反馈模糊认知图（FCMs）。工作包括设计代理系统，从文本中提取概念节点和因果关系，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2601.00097v2](https://arxiv.org/pdf/2601.00097v2)**

> **作者:** Akash Kumar Panda; Olaoluwa Adigun; Bart Kosko
>
> **备注:** 15 figures
>
> **摘要:** We design a large-language-model (LLM) agent that extracts causal feedback fuzzy cognitive maps (FCMs) from raw text. The causal learning or extraction process is agentic both because of the LLM's semi-autonomy and because ultimately the FCM dynamical system's equilibria drive the LLM agents to fetch and process causal text. The fetched text can in principle modify the adaptive FCM causal structure and so modify the source of its quasi-autonomy--its equilibrium limit cycles and fixed-point attractors. This bidirectional process endows the evolving FCM dynamical system with a degree of autonomy while still staying on its agentic leash. We show in particular that a sequence of three finely tuned system instructions guide an LLM agent as it systematically extracts key nouns and noun phrases from text, as it extracts FCM concept nodes from among those nouns and noun phrases, and then as it extracts or infers partial or fuzzy causal edges between those FCM nodes. We test this FCM generation on a recent essay about the promise of AI from the late diplomat and political theorist Henry Kissinger and his colleagues. This three-step process produced FCM dynamical systems that converged to the same equilibrium limit cycles as did the human-generated FCMs even though the human-generated FCM differed in the number of nodes and edges. A final FCM mixed generated FCMs from separate Gemini and ChatGPT LLM agents. The mixed FCM absorbed the equilibria of its dominant mixture component but also created new equilibria of its own to better approximate the underlying causal dynamical system.
>
---
#### [replaced 021] ProfVLM: A Lightweight Video-Language Model for Multi-View Proficiency Estimation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于动作质量评估任务，解决传统方法输出不具解释性的分类标签问题。提出ProfVLM模型，通过生成式视觉语言建模，联合预测熟练度并生成自然语言反馈。**

- **链接: [https://arxiv.org/pdf/2509.26278v2](https://arxiv.org/pdf/2509.26278v2)**

> **作者:** Edoardo Bianchi; Jacopo Staiano; Antonio Liotta
>
> **摘要:** Existing approaches treat action quality assessment and skill proficiency estimation as classification problems, outputting discrete labels without interpretable reasoning. We reformulate this task as generative vision language modeling, introducing ProfVLM, a compact model that jointly predicts proficiency levels and generates expert-like natural language feedback from multi-view videos. ProfVLM leverages conditional language generation to provide actionable insights along with quantitative evaluation scores. Central to our method is an AttentiveGatedProjector that dynamically fuses and projects multi-view egocentric and exocentric features from a frozen TimeSformer backbone into a language model fine-tuned for feedback generation. Trained on EgoExo4D with expert commentaries, ProfVLM surpasses state-of-the-art methods while using up to 20x fewer parameters and reducing training time by up to 60% compared to existing classification-based methods. By providing natural language critiques aligned with performance levels, this work shows that generative vision-language modeling offers a powerful and efficient paradigm shift for interpretable action quality assessment.
>
---
#### [replaced 022] ThinkBrake: A Simple Test-Time Decoding Control for Efficient Reasoning
- **分类: cs.CL**

- **简介: 该论文提出ThinkBrake，用于控制推理过程中的过度思考问题，提升大模型推理效率与准确性。任务为高效推理控制，解决过早停止与过度推理问题，通过监测概率边际实现动态停止。**

- **链接: [https://arxiv.org/pdf/2510.00546v4](https://arxiv.org/pdf/2510.00546v4)**

> **作者:** Sangjun Song; Minjae Oh; Seungkyu Lee; Sungmin Jo; Yohan Jo
>
> **摘要:** Large Reasoning Models (LRMs) allocate substantial inference-time compute to Chain-of-Thought (CoT) reasoning, improving performance on mathematics, scientific QA, and tool usage. However, this introduces overthinking: LRMs often reach a correct intermediate solution, continue reasoning, and overwrite it with an incorrect answer. We first demonstrate that oracle stopping--where we inject </think> at every sentence boundary and select the best stopping point in hindsight--improves average accuracy by 8% while reducing thinking tokens by 72%, exposing substantial overthinking. Motivated by this finding, we propose ThinkBrake, which monitors the log-probability margin between the top continuation token and </think> at sentence boundaries, stopping reasoning when this margin narrows. ThinkBrake requires no training and achieves favorable accuracy-efficiency trade-offs across math, scientific QA, and tool usage benchmarks, reducing thinking token usage by up to 30%. Furthermore, we provide theoretical analysis showing that ThinkBrake is equivalent to test-time realignment with a reward bonus for the </think> token.
>
---
#### [replaced 023] Where to Begin: Efficient Pretraining via Subnetwork Selection and Distillation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于小语言模型预训练任务，旨在提升其效率与性能。通过子网络选择和知识蒸馏，减少计算资源消耗，提高训练效果。**

- **链接: [https://arxiv.org/pdf/2510.07227v2](https://arxiv.org/pdf/2510.07227v2)**

> **作者:** Arjun Krishnakumar; Rhea Sanjay Sukthanker; Hannan Javed Mahadik; Gabriela Kadlecová; Vladyslav Moroshan; Timur Carstensen; Frank Hutter; Aaron Klein
>
> **摘要:** Small Language models (SLMs) offer an efficient and accessible alternative to Large Language Models (LLMs), delivering strong performance while using far fewer resources. We introduce a simple and effective framework for pretraining SLMs that brings together three complementary ideas. First, we identify structurally sparse sub-network initializations that consistently outperform randomly initialized models of similar size under the same compute budget. Second, we use evolutionary search to automatically discover high-quality sub-network initializations, providing better starting points for pretraining. Third, we apply knowledge distillation from larger teacher models to speed up training and improve generalization. Together, these components make SLM pretraining substantially more efficient: our best model, discovered using evolutionary search and initialized with LLM weights, matches the validation perplexity of a comparable Pythia SLM while requiring 5.16x and 1.26x fewer floating point operations for token budgets of 10B and 100B, respectively. We release all code publicly, offering a practical and reproducible path toward cost-efficient small language model development at scale.
>
---
#### [replaced 024] "Hiding in Plain Sight": Designing Synthetic Dialog Generation for Uncovering Socially Situated Norms
- **分类: cs.CL**

- **简介: 该论文属于对话生成任务，旨在揭示社交情境中的规范。通过构建合成对话数据集，自动发现并标注规范违反情况，提升模型对规范的理解与检测能力。**

- **链接: [https://arxiv.org/pdf/2410.00998v2](https://arxiv.org/pdf/2410.00998v2)**

> **作者:** Chengfei Wu; Dan Goldwasser
>
> **备注:** Accepted at AAAI Workshop on Shaping Responsible Synthetic Data in the Era of Foundation Models
>
> **摘要:** Naturally situated conversations encapsulate the social norms inherent to their context, reflecting both the relationships between interlocutors and the underlying communicative intent. In this paper, we propose a novel, multi-step framework for generating dialogues that automatically uncovers social norms from rich, context-laden interactions through a process of self-assessment and norm discovery, rather than relying on predefined norm labels. Leveraging this framework, we construct NormHint, a comprehensive synthetic dialogue dataset spanning a wide range of interlocutor attributes (e.g., age, profession, personality), relationship types, conversation topics, and conversational trajectories. NormHint is meticulously annotated with turn-level norm violation information, detailed participant descriptions, and remediation suggestions-including alternative trajectories achieved through early intervention. Human validation and automated analysis demonstrate that our dataset captures diverse conversational topics with high naturalness and realism. Moreover, we discovered that fine-tuning a model with our norm violation data significantly enhances its ability to detect and understand potential norm violations in conversations.
>
---
#### [replaced 025] Head Pursuit: Probing Attention Specialization in Multimodal Transformers
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文研究多模态Transformer中注意力头的语义专化问题，通过分析注意力头对特定概念的贡献，实现对模型输出的可控编辑。**

- **链接: [https://arxiv.org/pdf/2510.21518v2](https://arxiv.org/pdf/2510.21518v2)**

> **作者:** Lorenzo Basile; Valentino Maiorca; Diego Doimo; Francesco Locatello; Alberto Cazzaniga
>
> **备注:** Accepted at NeurIPS 2025 (spotlight)
>
> **摘要:** Language and vision-language models have shown impressive performance across a wide range of tasks, but their internal mechanisms remain only partly understood. In this work, we study how individual attention heads in text-generative models specialize in specific semantic or visual attributes. Building on an established interpretability method, we reinterpret the practice of probing intermediate activations with the final decoding layer through the lens of signal processing. This lets us analyze multiple samples in a principled way and rank attention heads based on their relevance to target concepts. Our results show consistent patterns of specialization at the head level across both unimodal and multimodal transformers. Remarkably, we find that editing as few as 1% of the heads, selected using our method, can reliably suppress or enhance targeted concepts in the model output. We validate our approach on language tasks such as question answering and toxicity mitigation, as well as vision-language tasks including image classification and captioning. Our findings highlight an interpretable and controllable structure within attention layers, offering simple tools for understanding and editing large-scale generative models.
>
---
#### [replaced 026] Investigating Retrieval-Augmented Generation Systems on Unanswerable, Uncheatable, Realistic, Multi-hop Queries
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于RAG系统研究任务，旨在解决真实复杂查询下的回答难题。针对现有基准不足，提出CRUMQs生成管道，提升基准难度，增强RAG系统能力。**

- **链接: [https://arxiv.org/pdf/2510.11956v3](https://arxiv.org/pdf/2510.11956v3)**

> **作者:** Gabrielle Kaili-May Liu; Bryan Li; Arman Cohan; William Gantt Walden; Eugene Yang
>
> **备注:** ECIR 2026
>
> **摘要:** Real-world use cases often present RAG systems with complex queries for which relevant information is missing from the corpus or is incomplete. In these settings, RAG systems must be able to reject unanswerable, out-of-scope queries and identify failures of retrieval and multi-hop reasoning. Despite this, existing RAG benchmarks rarely reflect realistic task complexity for multi-hop or out-of-scope questions, which often can be cheated via disconnected reasoning (i.e., solved without genuine multi-hop inference) or require only simple factual recall. This limits the ability for such benchmarks to uncover limitations of existing RAG systems. To address this gap, we present the first pipeline for automatic, difficulty-controlled creation of un$\underline{c}$heatable, $\underline{r}$ealistic, $\underline{u}$nanswerable, and $\underline{m}$ulti-hop $\underline{q}$uerie$\underline{s}$ (CRUMQs), adaptable to any corpus and domain. We use our pipeline to create CRUMQs over two popular RAG datasets and demonstrate its effectiveness via benchmark experiments on leading retrieval-augmented LLMs. Results show that compared to prior RAG benchmarks, CRUMQs are highly challenging for RAG systems and achieve up to 81.0\% reduction in cheatability scores. More broadly, our pipeline offers a simple way to enhance benchmark difficulty and drive development of more capable RAG systems.
>
---
#### [replaced 027] KPoEM: A Human-Annotated Dataset for Emotion Classification and RAG-Based Poetry Generation in Korean Modern Poetry
- **分类: cs.CL; cs.CY; cs.LG**

- **简介: 该论文提出KPoEM数据集，用于韩语现代诗歌的情感分类与生成。解决诗歌情感分析及生成难题，通过标注和模型优化提升情感识别与创作能力。**

- **链接: [https://arxiv.org/pdf/2509.03932v2](https://arxiv.org/pdf/2509.03932v2)**

> **作者:** Iro Lim; Haein Ji; Byungjun Kim
>
> **备注:** 43 pages, 22 tables, 3 figures, Digital Humanities and Social Sciences Korea Conference, James Joo-Jin Kim Center for Korean Studies, University of Pennsylvania, Philadelphia, USA
>
> **摘要:** This study introduces KPoEM (Korean Poetry Emotion Mapping), a novel dataset that serves as a foundation for both emotion-centered analysis and generative applications in modern Korean poetry. Despite advancements in NLP, poetry remains underexplored due to its complex figurative language and cultural specificity. We constructed a multi-label dataset of 7,662 entries (7,007 line-level and 615 work-level), annotated with 44 fine-grained emotion categories from five influential Korean poets. The KPoEM emotion classification model, fine-tuned through a sequential strategy -- moving from general-purpose corpora to the specialized KPoEM dataset -- achieved an F1-micro score of 0.60, significantly outperforming previous models (0.43). The model demonstrates an enhanced ability to identify temporally and culturally specific emotional expressions while preserving core poetic sentiments. Furthermore, applying the structured emotion dataset to a RAG-based poetry generation model demonstrates the empirical feasibility of generating texts that reflect the emotional and cultural sensibilities of Korean literature. This integrated approach strengthens the connection between computational techniques and literary analysis, opening new pathways for quantitative emotion research and generative poetics. Overall, this study provides a foundation for advancing emotion-centered analysis and creation in modern Korean poetry.
>
---
#### [replaced 028] Can Language Models Discover Scaling Laws?
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于科学发现任务，旨在解决自动发现模型性能缩放规律的问题。通过构建实验数据集和提出SLDAgent，实现更准确的缩放定律自动发现。**

- **链接: [https://arxiv.org/pdf/2507.21184v4](https://arxiv.org/pdf/2507.21184v4)**

> **作者:** Haowei Lin; Haotian Ye; Wenzheng Feng; Quzhe Huang; Yujun Li; Hubert Lim; Zhengrui Li; Xiangyu Wang; Jianzhu Ma; James Zou; Yitao Liang
>
> **摘要:** Discovering scaling laws for predicting model performance at scale is a fundamental and open-ended challenge, mostly reliant on slow, case specific human experimentation. To investigate the potential for LLMs to automate this process, we collect over 5,000 experiments from existing literature and curate eight diverse scaling law discovery tasks. While existing agents struggle to produce accurate law formulas, this paper introduces SLDAgent, an evolution-based agent that co-optimize the scaling law model and the parameters, enabling it to autonomously explore complex relationships between variables. For the first time, we demonstrates that SLDAgent can automatically discover laws that exhibit consistently more accurate extrapolation than their established, human-derived counterparts across all tasks. Through comprehensive analysis, we elucidate why these discovered laws are superior and verify their practical utility in both pretraining and finetuning applications. This work establishes a new paradigm for agentic scientific discovery, showing that AI systems can understand their own scaling behavior, and can contribute novel and practical knowledge back to the research community.
>
---
#### [replaced 029] Collaborative Causal Sensemaking: Closing the Complementarity Gap in Human-AI Decision Support
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于人机协作任务，旨在解决人类与AI在决策支持中互补性不足的问题。通过构建协同因果推理能力，提升人机团队的协作效果。**

- **链接: [https://arxiv.org/pdf/2512.07801v4](https://arxiv.org/pdf/2512.07801v4)**

> **作者:** Raunak Jain; Mudita Khurana
>
> **摘要:** LLM-based agents are increasingly deployed for expert decision support, yet human-AI teams in high-stakes settings do not yet reliably outperform the best individual. We argue this complementarity gap reflects a fundamental mismatch: current agents are trained as answer engines, not as partners in the collaborative sensemaking through which experts actually make decisions. Sensemaking (the ability to co-construct causal explanations, surface uncertainties, and adapt goals) is the key capability that current training pipelines do not explicitly develop or evaluate. We propose Collaborative Causal Sensemaking (CCS) as a research agenda to develop this capability from the ground up, spanning new training environments that reward collaborative thinking, representations for shared human-AI mental models, and evaluation centred on trust and complementarity. Taken together, these directions shift MAS research from building oracle-like answer engines to cultivating AI teammates that co-reason with their human partners over the causal structure of shared decisions, advancing the design of effective human-AI teams.
>
---
#### [replaced 030] Afri-MCQA: Multimodal Cultural Question Answering for African Languages
- **分类: cs.CL**

- **简介: 该论文提出Afri-MCQA，首个覆盖15种非洲语言的多模态文化问答基准，旨在解决非洲语言AI研究不足的问题。**

- **链接: [https://arxiv.org/pdf/2601.05699v2](https://arxiv.org/pdf/2601.05699v2)**

> **作者:** Atnafu Lambebo Tonja; Srija Anand; Emilio Villa-Cueva; Israel Abebe Azime; Jesujoba Oluwadara Alabi; Muhidin A. Mohamed; Debela Desalegn Yadeta; Negasi Haile Abadi; Abigail Oppong; Nnaemeka Casmir Obiefuna; Idris Abdulmumin; Naome A Etori; Eric Peter Wairagala; Kanda Patrick Tshinu; Imanigirimbabazi Emmanuel; Gabofetswe Malema; Alham Fikri Aji; David Ifeoluwa Adelani; Thamar Solorio
>
> **摘要:** Africa is home to over one-third of the world's languages, yet remains underrepresented in AI research. We introduce Afri-MCQA, the first Multilingual Cultural Question-Answering benchmark covering 7.5k Q&A pairs across 15 African languages from 12 countries. The benchmark offers parallel English-African language Q&A pairs across text and speech modalities and was entirely created by native speakers. Benchmarking large language models (LLMs) on Afri-MCQA shows that open-weight models perform poorly across evaluated cultures, with near-zero accuracy on open-ended VQA when queried in native language or speech. To evaluate linguistic competence, we include control experiments meant to assess this specific aspect separate from cultural knowledge, and we observe significant performance gaps between native languages and English for both text and speech. These findings underscore the need for speech-first approaches, culturally grounded pretraining, and cross-lingual cultural transfer. To support more inclusive multimodal AI development in African languages, we release our Afri-MCQA under academic license or CC BY-NC 4.0 on HuggingFace (https://huggingface.co/datasets/Atnafu/Afri-MCQA)
>
---
#### [replaced 031] Spiffy: Multiplying Diffusion LLM Acceleration via Lossless Speculative Decoding
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型加速任务，解决dLLM生成速度慢的问题。提出Spiffy算法，通过无损推测解码提升推理速度，实现2.8-3.1倍加速。**

- **链接: [https://arxiv.org/pdf/2509.18085v3](https://arxiv.org/pdf/2509.18085v3)**

> **作者:** Sudhanshu Agrawal; Risheek Garrepalli; Raghavv Goel; Mingu Lee; Christopher Lott; Fatih Porikli
>
> **备注:** Original version uploaded on Sep 22, 2025. (v2): Extended Table 2 with additional analysis and referenced it in Sec 5.2. (v3): Added note to Sec 4.2 and Appendix A.2 specifying conditions for losslessness
>
> **摘要:** Diffusion LLMs (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs (AR-LLMs) with the potential to operate at significantly higher token generation rates. However, currently available open-source dLLMs often generate at much lower rates, typically decoding only a single token at every denoising timestep in order to maximize output quality. We present Spiffy, a speculative decoding algorithm that accelerates dLLM inference by $\mathbf{2.8{-}3.1\times}$ while provably preserving the model's output distribution. This work addresses the unique challenges involved in applying ideas from speculative decoding of AR-LLMs to the dLLM setting. Spiffy proposes draft states by leveraging the dLLM's distribution itself in an auto-speculative manner. This approach is efficient and effective, and eliminates the overheads of training and running an independent draft model. To structure the candidate draft states, we propose a novel directed draft graph which is uniquely designed to take advantage of the bidirectional, block-wise nature of dLLM generation and can be verified in parallel by the dLLM. To further optimize the structure of these draft graphs, we introduce an efficient, offline calibration algorithm that procedurally determines high-quality graph configurations. These optimized draft graphs, enabling increased acceptance rates, lead to a significant boost in the overall speedup achieved by the system. Crucially, Spiffy is also complementary to other recent innovations in improving dLLM generation speeds such as KV-caching and multi-token unmasking. We demonstrate that when combined with such parallel decoding algorithms, Spiffy is able to effectively multiply the benefits of these methods leading to total speedups of up to $\mathbf{7.9\times}$.
>
---
#### [replaced 032] Controlled Self-Evolution for Algorithmic Code Optimization
- **分类: cs.CL; cs.AI; cs.NE**

- **简介: 该论文属于算法代码优化任务，旨在解决自进化方法效率低的问题。提出CSE框架，通过结构化初始化、反馈引导进化和层次记忆提升优化效果。**

- **链接: [https://arxiv.org/pdf/2601.07348v3](https://arxiv.org/pdf/2601.07348v3)**

> **作者:** Tu Hu; Ronghao Chen; Shuo Zhang; Jianghao Yin; Mou Xiao Feng; Jingping Liu; Shaolei Zhang; Wenqi Jiang; Yuqi Fang; Sen Hu; Yi Xu; Huacan Wang
>
> **备注:** 27 pages
>
> **摘要:** Self-evolution methods enhance code generation through iterative "generate-verify-refine" cycles, yet existing approaches suffer from low exploration efficiency, failing to discover solutions with superior complexity within limited budgets. This inefficiency stems from initialization bias trapping evolution in poor solution regions, uncontrolled stochastic operations lacking feedback guidance, and insufficient experience utilization across tasks. To address these bottlenecks, we propose Controlled Self-Evolution (CSE), which consists of three key components. Diversified Planning Initialization generates structurally distinct algorithmic strategies for broad solution space coverage. Genetic Evolution replaces stochastic operations with feedback-guided mechanisms, enabling targeted mutation and compositional crossover. Hierarchical Evolution Memory captures both successful and failed experiences at inter-task and intra-task levels. Experiments on EffiBench-X demonstrate that CSE consistently outperforms all baselines across various LLM backbones. Furthermore, CSE achieves higher efficiency from early generations and maintains continuous improvement throughout evolution. Our code is publicly available at https://github.com/QuantaAlpha/EvoControl.
>
---
#### [replaced 033] Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning
- **分类: cs.CL; cs.MA**

- **简介: 该论文提出Memory-R1，通过强化学习增强大语言模型的内存管理能力，解决其状态缺失和长程推理受限的问题。**

- **链接: [https://arxiv.org/pdf/2508.19828v5](https://arxiv.org/pdf/2508.19828v5)**

> **作者:** Sikuan Yan; Xiufeng Yang; Zuchao Huang; Ercong Nie; Zifeng Ding; Zonggen Li; Xiaowen Ma; Jinhe Bi; Kristian Kersting; Jeff Z. Pan; Hinrich Schütze; Volker Tresp; Yunpu Ma
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive capabilities across a wide range of NLP tasks, but they remain fundamentally stateless, constrained by limited context windows that hinder long-horizon reasoning. Recent efforts to address this limitation often augment LLMs with an external memory bank, yet most existing pipelines are static and heuristic-driven, lacking a learned mechanism for deciding what to store, update, or retrieve. We present Memory-R1, a reinforcement learning (RL) framework that equips LLMs with the ability to actively manage and utilize external memory through two specialized agents: a Memory Manager that learns structured operations, including ADD, UPDATE, DELETE, and NOOP; and an Answer Agent that pre-selects and reasons over relevant entries. Both agents are fine-tuned with outcome-driven RL (PPO and GRPO), enabling adaptive memory management with minimal supervision. With only 152 training QA pairs, Memory-R1 outperforms strong baselines and generalizes across diverse question types, three benchmarks (LoCoMo, MSC, LongMemEval), and multiple model scales (3B-14B).
>
---
#### [replaced 034] ExPO: Unlocking Hard Reasoning with Self-Explanation-Guided Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出ExPO框架，解决复杂推理任务中强化学习效果不佳的问题。通过生成高质量正样本，引导模型探索新推理路径，提升推理能力。属于机器学习中的强化学习与推理任务。**

- **链接: [https://arxiv.org/pdf/2507.02834v2](https://arxiv.org/pdf/2507.02834v2)**

> **作者:** Ruiyang Zhou; Shuozhe Li; Amy Zhang; Liu Leqi
>
> **备注:** Accepted to NeurIPS 2025 (Poster). Code available at https://github.com/HumainLab/ExPO_rl_reasoning_by_explanation
>
> **摘要:** Self-improvement via RL often fails on complex reasoning tasks because GRPO-style post-training methods rely on the model's initial ability to generate positive samples. Without guided exploration, these approaches merely reinforce what the model already knows (distribution-sharpening) rather than enabling the model to solve problems where it initially generates no correct solutions. To unlock reasoning ability in such settings, the model must explore new reasoning trajectories beyond its current output distribution. Such exploration requires access to sufficiently good positive samples to guide the learning. While expert demonstrations seem like a natural solution, we find that they are often ineffective in RL post-training. Instead, we identify two key properties of effective positive samples: they should (1) be likely under the current policy, and (2) increase the model's likelihood of predicting the correct answer. Based on these insights, we propose $\textbf{Self-Explanation Policy Optimization (ExPO)}$-a simple and modular framework that generates such samples by conditioning on the ground-truth answer. It can be integrated with popular RL training methods like GRPO and DPO. ExPO enables efficient exploration and guides the model to produce reasoning trajectories more aligned with its policy than expert-written CoTs, while ensuring higher quality than its own (incorrect) samples. Experiments show that ExPO improves both learning efficiency and final performance on reasoning benchmarks, surpassing expert-demonstration-based methods in challenging settings such as MATH level-5, where the model initially struggles the most. Code is available at https://github.com/HumainLab/ExPO_rl_reasoning_by_explanation .
>
---
#### [replaced 035] Can LLMs Generate Reliable Test Case Generators? A Study on Competition-Level Programming Problems
- **分类: cs.CL; cs.AI; cs.SE**

- **简介: 论文研究LLMs生成可靠测试用例生成器的能力，针对竞赛级编程问题，提出TCGBench基准，解决如何生成有效测试用例以检测代码缺陷的问题。**

- **链接: [https://arxiv.org/pdf/2506.06821v4](https://arxiv.org/pdf/2506.06821v4)**

> **作者:** Yuhan Cao; Zian Chen; Kun Quan; Ziliang Zhang; Yu Wang; Xiaoning Dong; Yeqi Feng; Guanzhong He; Jingcheng Huang; Jianhao Li; Yixuan Tan; Jiafu Tang; Yilin Tang; Junlei Wu; Qianyu Xiao; Can Zheng; Shouchen Zhou; Yuxiang Zhu; Yiming Huang; Tianxing He
>
> **备注:** 37 pages, 22 figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, capable of tackling complex tasks during inference. However, the extent to which LLMs can be utilized for code checking or debugging through test case generation remains largely unexplored. We investigate this problem from the perspective of competition-level programming (CP) programs and propose TCGBench, a Benchmark for (LLM generation of) Test Case Generators. This benchmark comprises two tasks, aimed at studying the capabilities of LLMs in (1) generating valid test case generators for a given CP problem, and further (2) generating targeted test case generators that expose bugs in human-written code. Experimental results indicate that while state-of-the-art LLMs can generate valid test case generators in most cases, most LLMs struggle to generate targeted test cases that reveal flaws in human code effectively. Especially, even advanced reasoning models (e.g., o3-mini) fall significantly short of human performance in the task of generating targeted generators. Furthermore, we construct a high-quality, manually curated dataset of instructions for generating targeted generators. Analysis demonstrates that the performance of LLMs can be enhanced with the aid of this dataset, by both prompting and fine-tuning.
>
---
#### [replaced 036] Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States of Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决多步骤推理中验证效率低的问题。提出通过探测大模型内部状态，构建轻量级探针进行推理步骤验证，提升测试时扩展性能。**

- **链接: [https://arxiv.org/pdf/2511.06209v3](https://arxiv.org/pdf/2511.06209v3)**

> **作者:** Jingwei Ni; Ekaterina Fadeeva; Tianyi Wu; Mubashara Akhtar; Jiaheng Zhang; Elliott Ash; Markus Leippold; Timothy Baldwin; See-Kiong Ng; Artem Shelmanov; Mrinmaya Sachan
>
> **备注:** Preprint under review
>
> **摘要:** LLMs can solve complex tasks by generating long, multi-step reasoning chains. Test-time scaling (TTS) can further improve LLM performance by sampling multiple variants of intermediate reasoning steps, verifying their correctness, and strategically choosing the best steps for continuation. However, existing verification approaches, such as Process Reward Models (PRMs), are computationally expensive, limited to specific domains, and require large-scale human or model-generated annotations. We propose a lightweight alternative for step-level reasoning verification based on probing the internal states of LLMs. We train a transformer-based probe that uses the internal states of the frozen LLM to estimate the credibility of its reasoning steps during generation. Annotation can be generated either by another larger LLM (e.g., DeepSeek-R1) or in a self-supervised manner by the original model itself. The probes are both effective and lightweight, containing fewer than 10M parameters. Across multiple domains, including mathematics, planning, and general knowledge question answering, our probes match or even exceed the performance of PRMs that are up to 810x larger. Our findings suggest that the internal states of LLMs encode their confidence in reasoning processes and can serve as reliable signals for reasoning step verification, offering a promising direction towards scalable and generalizable TTS and introspective LLMs.
>
---
#### [replaced 037] Revisiting the Uniform Information Density Hypothesis in LLM Reasoning Traces
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究LLM推理轨迹中的信息密度均匀性，旨在评估其与推理质量的关系。通过提出熵基度量，分析局部与全局均匀性，发现均匀信息密度可提升推理准确率。任务为模型推理质量评估。**

- **链接: [https://arxiv.org/pdf/2510.06953v2](https://arxiv.org/pdf/2510.06953v2)**

> **作者:** Minju Gwak; Guijin Son; Jaehyung Kim
>
> **摘要:** The Uniform Information Density (UID) hypothesis suggests that effective communication maintains a stable flow of information. In this work, we revisit this principle in the context of large language model (LLM) reasoning traces, asking whether step-level uniformity reflects reasoning quality. To this end, we propose an entropy-based stepwise information density metric and introduce two complementary measures of uniformity, local and global uniformity scores. Across the experiments on six different reasoning benchmarks, we find that step-level uniformity not only provides a strong theoretical lens but also yields practical performance benefits; for example, selecting reasoning traces with more uniform information density at the step-level improves accuracy by 10-32\% relative gains over baselines at AIME2025. Our analysis further reveals that correct reasoning traces tend to avoid sharp information density spikes, while incorrect traces exhibit irregular information bursts. These results demonstrate that UID-inspired information density measures outperform alternative internal signals as predictors of reasoning quality. Results highlight the uniformity of the information density as a robust diagnostic and selection criterion for building more reliable and accurate reasoning systems.
>
---
#### [replaced 038] Prompting4Debugging: Red-Teaming Text-to-Image Diffusion Models by Finding Problematic Prompts
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于安全测试任务，旨在检测文本到图像扩散模型的安全机制漏洞。通过自动寻找问题提示，验证安全机制的可靠性。**

- **链接: [https://arxiv.org/pdf/2309.06135v3](https://arxiv.org/pdf/2309.06135v3)**

> **作者:** Zhi-Yi Chin; Chieh-Ming Jiang; Ching-Chun Huang; Pin-Yu Chen; Wei-Chen Chiu
>
> **备注:** ICML 2024 main conference paper. The source code is available at https://github.com/zhiyichin/P4D
>
> **摘要:** Text-to-image diffusion models, e.g. Stable Diffusion (SD), lately have shown remarkable ability in high-quality content generation, and become one of the representatives for the recent wave of transformative AI. Nevertheless, such advance comes with an intensifying concern about the misuse of this generative technology, especially for producing copyrighted or NSFW (i.e. not safe for work) images. Although efforts have been made to filter inappropriate images/prompts or remove undesirable concepts/styles via model fine-tuning, the reliability of these safety mechanisms against diversified problematic prompts remains largely unexplored. In this work, we propose Prompting4Debugging (P4D) as a debugging and red-teaming tool that automatically finds problematic prompts for diffusion models to test the reliability of a deployed safety mechanism. We demonstrate the efficacy of our P4D tool in uncovering new vulnerabilities of SD models with safety mechanisms. Particularly, our result shows that around half of prompts in existing safe prompting benchmarks which were originally considered "safe" can actually be manipulated to bypass many deployed safety mechanisms, including concept removal, negative prompt, and safety guidance. Our findings suggest that, without comprehensive testing, the evaluations on limited safe prompting benchmarks can lead to a false sense of safety for text-to-image models.
>
---
#### [replaced 039] Development and Evaluation of HopeBot: an LLM-based chatbot for structured and interactive PHQ-9 depression screening
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于心理健康筛查任务，旨在解决传统抑郁筛查工具缺乏互动性的问题。研究开发了基于大语言模型的聊天机器人HopeBot，通过交互式方式实施PHQ-9筛查，并验证其有效性与用户接受度。**

- **链接: [https://arxiv.org/pdf/2507.05984v2](https://arxiv.org/pdf/2507.05984v2)**

> **作者:** Zhijun Guo; Alvina Lai; Julia Ive; Alexandru Petcu; Yutong Wang; Luyuan Qi; Johan H Thygesen; Kezhi Li
>
> **摘要:** Static tools like the Patient Health Questionnaire-9 (PHQ-9) effectively screen depression but lack interactivity and adaptability. We developed HopeBot, a chatbot powered by a large language model (LLM) that administers the PHQ-9 using retrieval-augmented generation and real-time clarification. In a within-subject study, 132 adults in the United Kingdom and China completed both self-administered and chatbot versions. Scores demonstrated strong agreement (ICC = 0.91; 45% identical). Among 75 participants providing comparative feedback, 71% reported greater trust in the chatbot, highlighting clearer structure, interpretive guidance, and a supportive tone. Mean ratings (0-10) were 8.4 for comfort, 7.7 for voice clarity, 7.6 for handling sensitive topics, and 7.4 for recommendation helpfulness; the latter varied significantly by employment status and prior mental-health service use (p < 0.05). Overall, 87.1% expressed willingness to reuse or recommend HopeBot. These findings demonstrate voice-based LLM chatbots can feasibly serve as scalable, low-burden adjuncts for routine depression screening.
>
---
#### [replaced 040] To Retrieve or To Think? An Agentic Approach for Context Evolution
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识密集型任务，解决现有方法计算成本高、性能下降的问题。提出ACE框架，通过动态决策优化上下文演化，提升准确率并减少资源消耗。**

- **链接: [https://arxiv.org/pdf/2601.08747v2](https://arxiv.org/pdf/2601.08747v2)**

> **作者:** Rubing Chen; Jian Wang; Wenjie Li; Xiao-Yong Wei; Qing Li
>
> **摘要:** Current context augmentation methods, such as retrieval-augmented generation, are essential for solving knowledge-intensive reasoning tasks. However, they typically adhere to a rigid, brute-force strategy that executes retrieval at every step. This indiscriminate approach not only incurs unnecessary computational costs but also degrades performance by saturating the context with irrelevant noise. To address these limitations, we introduce Agentic Context Evolution (ACE), a framework inspired by human metacognition that dynamically determines whether to seek new evidence or reason with existing knowledge. ACE employs a central orchestrator agent to make decisions strategically via majority voting. It aims to alternate between activating a retriever agent for external retrieval and a reasoner agent for internal analysis and refinement. By eliminating redundant retrieval steps, ACE maintains a concise and evolved context. Extensive experiments on challenging multi-hop QA benchmarks demonstrate that ACE significantly outperforms competitive baselines in accuracy while achieving efficient token consumption. Our work provides valuable insights into advancing context-evolved generation for complex, knowledge-intensive tasks.
>
---
#### [replaced 041] LaoBench: A Large-Scale Multidimensional Lao Benchmark for Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出LaoBench，一个针对老挝语的多维基准测试，解决低资源语言评估不足的问题。工作包括构建高质量数据集，涵盖文化、教育和翻译任务，用于评估大语言模型的语言理解与推理能力。**

- **链接: [https://arxiv.org/pdf/2511.11334v2](https://arxiv.org/pdf/2511.11334v2)**

> **作者:** Jian Gao; Richeng Xuan; Zhaolu Kang; Dingshi Liao; Wenxin Huang; Zongmou Huang; Yangdi Xu; Bowen Qin; Zheqi He; Xi Yang; Changjin Li; Yonghua Lin
>
> **摘要:** The rapid advancement of large language models (LLMs) has not been matched by their evaluation in low-resource languages, especially Southeast Asian languages like Lao. To fill this gap, we introduce \textbf{LaoBench}, the first large-scale, high-quality, and multidimensional benchmark for assessing LLM language understanding and reasoning in Lao. LaoBench contains \textbf{17,000+} expert-curated samples across three dimensions: culturally grounded knowledge application, curriculum-aligned K12 education, and bilingual translation among Lao, Chinese, and English. It includes open-source and held-out subsets, where the held-out portion enables secure black-box evaluation via a controlled service to improve fairness and data security. We construct LaoBench with a hybrid pipeline that combines expert authoring with agent-assisted verification, ensuring linguistic accuracy, cultural relevance, and educational validity. We evaluate diverse state-of-the-art open-source and closed-source LLMs, and find that even strong multilingual models lag behind human experts, particularly in culturally grounded reasoning and translation fidelity. We hope LaoBench will catalyze research on Lao and other underrepresented Southeast Asian languages for more inclusive multilingual evaluation.
>
---
#### [replaced 042] Autofocus Retrieval: An Effective Pipeline for Multi-Hop Question Answering With Semi-Structured Knowledge
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于多跳问答任务，解决如何有效利用半结构化知识库的问题。提出AF-Retriever框架，结合结构化与文本检索，提升问答效果。**

- **链接: [https://arxiv.org/pdf/2505.09246v3](https://arxiv.org/pdf/2505.09246v3)**

> **作者:** Derian Boer; Stephen Roth; Stefan Kramer
>
> **摘要:** In many real-world settings, machine learning models and interactive systems have access to both structured knowledge, e.g., knowledge graphs or tables, and unstructured content, e.g., natural language documents. Yet, most rely on either. Semi-Structured Knowledge Bases (SKBs) bridge this gap by linking unstructured content to nodes within structured data. In this work, we present Autofocus-Retriever (AF-Retriever), a modular framework for SKB-based, multi-hop question answering. It combines structural and textual retrieval through novel integration steps and optimizations, achieving the best zero- and one-shot results across all three STaRK QA benchmarks, which span diverse domains and evaluation metrics. AF-Retriever's average first-hit rate surpasses the second-best method by 32.1%. Its performance is driven by (1) leveraging exchangeable large language models (LLMs) to extract entity attributes and relational constraints for both parsing and reranking the top-k answers, (2) vector similarity search for ranking both extracted entities and final answers, (3) a novel incremental scope expansion procedure that prepares for the reranking on a configurable amount of suitable candidates that fulfill the given constraints the most, and (4) a hybrid retrieval strategy that reduces error susceptibility. In summary, while constantly adjusting the focus like an optical autofocus, AF-Retriever delivers a configurable amount of answer candidates in four constraint-driven retrieval steps, which are then supplemented and ranked through four additional processing steps. An ablation study and a detailed error analysis, including a comparison of three different LLM reranking strategies, provide component-level insights. The source code is available at https://github.com/kramerlab/AF-Retriever.
>
---
#### [replaced 043] Non-Linear Scoring Model for Translation Quality Evaluation
- **分类: cs.CL**

- **简介: 该论文属于翻译质量评估任务，解决线性评分模型在不同样本长度下的偏差问题，提出非线性评分模型以更准确反映人类对翻译质量的感知。**

- **链接: [https://arxiv.org/pdf/2511.13467v4](https://arxiv.org/pdf/2511.13467v4)**

> **作者:** Serge Gladkoff; Lifeng Han; Katerina Gasova
>
> **备注:** Technical report, 31 pages
>
> **摘要:** Analytic Translation Quality Evaluation (TQE), based on Multidimensional Quality Metrics (MQM), traditionally uses a linear error-to-penalty scale calibrated to a reference sample of 1000-2000 words. However, linear extrapolation biases judgment on samples of different sizes, over-penalizing short samples and under-penalizing long ones, producing misalignment with expert intuition. Building on the Multi-Range framework, this paper presents a calibrated, non-linear scoring model that better reflects how human content consumers perceive translation quality across samples of varying length. Empirical data from three large-scale enterprise environments shows that acceptable error counts grow logarithmically, not linearly, with sample size. Psychophysical and cognitive evidence, including the Weber-Fechner law and Cognitive Load Theory, supports this premise by explaining why the perceptual impact of additional errors diminishes while the cognitive burden grows with scale. We propose a two-parameter model E(x) = a * ln(1 + b * x), a, b > 0, anchored to a reference tolerance and calibrated from two tolerance points using a one-dimensional root-finding step. The model yields an explicit interval within which the linear approximation stays within +/-20 percent relative error and integrates into existing evaluation workflows with only a dynamic tolerance function added. The approach improves interpretability, fairness, and inter-rater reliability across both human and AI-generated translations. By operationalizing a perceptually valid scoring paradigm, it advances translation quality evaluation toward more accurate and scalable assessment. The model also provides a stronger basis for AI-based document-level evaluation aligned with human judgment. Implementation considerations for CAT/LQA systems and implications for human and AI-generated text evaluation are discussed.
>
---
#### [replaced 044] Survey of End-to-End Multi-Speaker Automatic Speech Recognition for Monaural Audio
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于多说话人语音识别任务，旨在解决单通道音频中多人重叠语音的识别与归属问题。工作包括梳理E2E架构、分析不同模型结构并评估其性能。**

- **链接: [https://arxiv.org/pdf/2505.10975v2](https://arxiv.org/pdf/2505.10975v2)**

> **作者:** Xinlu He; Jacob Whitehill
>
> **备注:** Accepted for publication in Computer Speech & Language (CSL)
>
> **摘要:** Monaural multi-speaker automatic speech recognition (ASR) remains challenging due to data scarcity and the intrinsic difficulty of recognizing and attributing words to individual speakers, particularly in overlapping speech. Recent advances have driven the shift from cascade systems to end-to-end (E2E) architectures, which reduce error propagation and better exploit the synergy between speech content and speaker identity. Despite rapid progress in E2E multi-speaker ASR, the field lacks a comprehensive review of recent developments. This survey provides a systematic taxonomy of E2E neural approaches for multi-speaker ASR, highlighting recent advances and comparative analysis. Specifically, we analyze: (1) architectural paradigms (SIMO vs.~SISO) for pre-segmented audio, analyzing their distinct characteristics and trade-offs; (2) recent architectural and algorithmic improvements based on these two paradigms; (3) extensions to long-form speech, including segmentation strategy and speaker-consistent hypothesis stitching. Further, we (4) evaluate and compare methods across standard benchmarks. We conclude with a discussion of open challenges and future research directions towards building robust and scalable multi-speaker ASR.
>
---
#### [replaced 045] Mitigating Gender Bias via Fostering Exploratory Thinking in LLMs
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言处理中的公平性任务，旨在解决LLMs中的性别偏见问题。通过生成结构相同的故事对，引导模型进行探索性思考并调整道德判断，以减少偏见。**

- **链接: [https://arxiv.org/pdf/2505.17217v3](https://arxiv.org/pdf/2505.17217v3)**

> **作者:** Kangda Wei; Hasnat Md Abdullah; Ruihong Huang
>
> **摘要:** Large Language Models (LLMs) often exhibit gender bias, resulting in unequal treatment of male and female subjects across different contexts. To address this issue, we propose a novel data generation framework that fosters exploratory thinking in LLMs. Our approach prompts models to generate story pairs featuring male and female protagonists in structurally identical, morally ambiguous scenarios, then elicits and compares their moral judgments. When inconsistencies arise, the model is guided to produce balanced, gender-neutral judgments. These story-judgment pairs are used to fine-tune or optimize the models via Direct Preference Optimization (DPO). Experimental results show that our method significantly reduces gender bias while preserving or even enhancing general model capabilities. We will release the code and generated data. We release the code and generated data at: https://github.com/WeiKangda/LLMs-Exploratory-Bias-Mitigation/tree/main.
>
---
#### [replaced 046] HapticLLaMA: A Multimodal Sensory Language Model for Haptic Captioning
- **分类: cs.CL**

- **简介: 该论文提出HapticLLaMA，解决从触觉信号生成语言描述的任务，通过两种分词方法结合LLaMA模型，提升触觉感知的自然语言描述能力。**

- **链接: [https://arxiv.org/pdf/2508.06475v2](https://arxiv.org/pdf/2508.06475v2)**

> **作者:** Guimin Hu; Daniel Hershcovich; Hasti Seifi
>
> **摘要:** Haptic captioning is the task of generating natural language descriptions from haptic signals, such as vibrations, for use in virtual reality, accessibility, and rehabilitation applications. While previous multimodal research has focused primarily on vision and audio, haptic signals for the sense of touch remain underexplored. To address this gap, we formalize the haptic captioning task and propose HapticLLaMA, a multimodal sensory language model that interprets vibration signals into descriptions in a given sensory, emotional, or associative category. We investigate two types of haptic tokenizers, a frequency-based tokenizer and an EnCodec-based tokenizer, that convert haptic signals into sequences of discrete units, enabling their integration with the LLaMA model. HapticLLaMA is trained in two stages: (1) supervised fine-tuning using the LLaMA architecture with LoRA-based adaptation, and (2) fine-tuning via reinforcement learning from human feedback (RLHF). We assess HapticLLaMA's captioning performance using both automated n-gram metrics and human evaluation. HapticLLaMA demonstrates strong capability in interpreting haptic vibration signals, achieving a METEOR score of 59.98 and a BLEU-4 score of 32.06 respectively. Additionally, over 61% of the generated captions received human ratings above 3.5 on a 7-point scale, with RLHF yielding a 10% improvement in the overall rating distribution, indicating stronger alignment with human haptic perception. These findings highlight the potential of large language models to process and adapt to sensory data.
>
---
#### [replaced 047] Structured yet Bounded Temporal Understanding in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型的时空理解能力，解决时间参照框架对模型行为影响的问题。通过分析不同时间参照结构下的相似性判断，揭示模型如何组织时间表征。**

- **链接: [https://arxiv.org/pdf/2510.16685v2](https://arxiv.org/pdf/2510.16685v2)**

> **作者:** Damin Zhang; Julia Rayz
>
> **备注:** Under review. Results on larger dataset. Correct a theoretical error. 11 pages, 5 figures
>
> **摘要:** Large language models (LLMs) increasingly show strong performance on temporally grounded tasks, such as timeline construction, temporal question answering, and event ordering. However, it remains unclear how their behavior depends on the way time is anchored in language. In this work, we study LLMs' temporal understanding through temporal frames of reference (t-FoRs), contrasting deictic framing (past-present-future) and sequential framing (before-after). Using a large-scale dataset of real-world events from Wikidata and similarity judgement task, we examine how LLMs' outputs vary with temporal distance, interval relations, and event duration. Our results show that LLMs systematically adapt to both t-FoRs, but the resulting similarity patterns differ significantly. Under deictic t-FoR, the similarity judgement scores form graded and asymmetric structures centered on the present, with sharper decline for future events and higher variance in the past. Under sequential t-FoR, similarity becomes strongly negative once events are temporally separated. Temporal judgements are also shaped by interval algebra and duration, with instability concentrated in overlap- and containment-based relations, and duration influencing only past events under deictic t-FoR. Overall, these findings characterize how LLMs organize temporal representation under different reference structures and identify the factors that most strongly shape their temporal understanding.
>
---
#### [replaced 048] Quiet Feature Learning in Algorithmic Tasks
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究Transformer模型在算法任务中的学习过程，发现损失曲线存在异常相变，揭示了隐含的中间特征对任务性能的关键作用。**

- **链接: [https://arxiv.org/pdf/2505.03997v3](https://arxiv.org/pdf/2505.03997v3)**

> **作者:** Prudhviraj Naidu; Zixian Wang; Leon Bergen; Ramamohan Paturi
>
> **备注:** Accepted as Oral presentation @ AAAI 2026 Special Track on AI Alignment
>
> **摘要:** We train Transformer-based language models on ten foundational algorithmic tasks and observe pronounced phase transitions in their loss curves that deviate from established power-law scaling trends. Over large ranges of compute, the validation loss barely improves, then abruptly decreases. Probing the models' internal representations reveals that quiet features are learned prior to any decrease in task loss. These quiet features represent intermediate algorithmic computations that do not by themselves improve the output loss. Ablation experiments demonstrate that individual quiet features are causally necessary for task performance. Our results demonstrate that substantial representational progress can remain hidden beneath an apparently flat loss curve, challenging the prevailing use of cross-entropy as a proxy for learning and motivating richer diagnostics for monitoring model training.
>
---
#### [replaced 049] Can Editing LLMs Inject Harm?
- **分类: cs.CL**

- **简介: 该论文属于安全风险研究任务，探讨知识编辑可能引发的有害信息注入问题。工作包括构建数据集、分析误导与偏见注入风险，并验证攻击的隐蔽性。**

- **链接: [https://arxiv.org/pdf/2407.20224v4](https://arxiv.org/pdf/2407.20224v4)**

> **作者:** Canyu Chen; Baixiang Huang; Zekun Li; Zhaorun Chen; Shiyang Lai; Xiongxiao Xu; Jia-Chen Gu; Jindong Gu; Huaxiu Yao; Chaowei Xiao; Xifeng Yan; William Yang Wang; Philip Torr; Dawn Song; Kai Shu
>
> **备注:** Accepted to Proceedings of AAAI 2026. The first two authors contributed equally. 7 pages for main paper, 31 pages including appendix. The code, results, dataset for this paper and more resources are on the project website: https://llm-editing.github.io
>
> **摘要:** Large Language Models (LLMs) have emerged as a new information channel. Meanwhile, one critical but under-explored question is: Is it possible to bypass the safety alignment and inject harmful information into LLMs stealthily? In this paper, we propose to reformulate knowledge editing as a new type of safety threat for LLMs, namely Editing Attack, and conduct a systematic investigation with a newly constructed dataset EditAttack. Specifically, we focus on two typical safety risks of Editing Attack including Misinformation Injection and Bias Injection. For the first risk, we find that editing attacks can inject both commonsense and long-tail misinformation into LLMs, and the effectiveness for the former one is particularly high. For the second risk, we discover that not only can biased sentences be injected into LLMs with high effectiveness, but also one single biased sentence injection can degrade the overall fairness. Then, we further illustrate the high stealthiness of editing attacks. Our discoveries demonstrate the emerging misuse risks of knowledge editing techniques on compromising the safety alignment of LLMs and the feasibility of disseminating misinformation or bias with LLMs as new channels.
>
---
#### [replaced 050] AgenticIE: An Adaptive Agent for Information Extraction from Complex Regulatory Documents
- **分类: cs.CL**

- **简介: 该论文提出AgenticIE系统，解决复杂监管文档中的信息抽取与问答问题，通过设计领域特定的代理架构提升抽取效果。**

- **链接: [https://arxiv.org/pdf/2509.11773v3](https://arxiv.org/pdf/2509.11773v3)**

> **作者:** Gaye Colakoglu; Gürkan Solmaz; Jonathan Fürst
>
> **摘要:** Declaration of Performance (DoP) documents, mandated by EU regulation, specify characteristics of construction products, such as fire resistance and insulation. While this information is essential for quality control and reducing carbon footprints, it is not easily machine readable. Despite content requirements, DoPs exhibit significant variation in layout, schema, and format, further complicated by their multilingual nature. In this work, we propose DoP Key Information Extraction (KIE) and Question Answering (QA) as new NLP challenges. To address this challenge, we design a domain-specific AgenticIE system based on a planner-executor-corresponder pattern. For evaluation, we introduce a high-density, expert-annotated dataset of complex, multi-page regulatory documents in English and German. Unlike standard IE datasets (e.g., FUNSD, CORD) with sparse annotations, our dataset contains over 15K annotated entities, averaging over 190 annotations per document. Our agentic system outperforms static and multimodal LLM baselines, achieving Exact Match (EM) scores of 0.396 vs. 0.342 (GPT-4o, +16%) and 0.314 (GPT-4o-V, +26%) across the KIE and QA tasks. Our experimental analysis validates the benefits of the agentic system, as well as the challenging nature of our new DoP dataset.
>
---
#### [replaced 051] Optimizing Fine-Tuning through Advanced Initialization Strategies for Low-Rank Adaptation
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型微调任务，旨在解决LoRA方法中低秩矩阵初始化不当导致的性能瓶颈。提出IniLoRA及其变体，通过更优初始化提升模型效果。**

- **链接: [https://arxiv.org/pdf/2510.03731v4](https://arxiv.org/pdf/2510.03731v4)**

> **作者:** Yongfu Xue
>
> **摘要:** The rapid development of parameter-efficient fine-tuning methods has noticeably improved the efficiency of adapting large language models. Among these, LoRA has gained widespread popularity due to its strong balance of effectiveness and parameter efficiency. However, LoRA relies on initializing two low-rank matrices whose product is zero, which limits its ability to effectively activate and leverage the original model weights-creating a potential bottleneck for optimal performance. To address this limitation, we propose \textbf{IniLoRA}, a novel initialization strategy that initializes the low-rank matrices to closely approximate the original model weights. Experimental results indicate that IniLoRA achieves better performance than LoRA across a range of models and tasks. Additionally, we introduce two variants, IniLoRA-$α$ and IniLoRA-$β$, both leveraging distinct initialization methods to enhance performance further.
>
---
#### [replaced 052] DYCP: Dynamic Context Pruning for Long-Form Dialogue with LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于长对话任务，解决LLMs在长对话中响应延迟高、答案质量下降的问题。提出DyCP方法，动态管理上下文，提升回答质量并降低延迟。**

- **链接: [https://arxiv.org/pdf/2601.07994v2](https://arxiv.org/pdf/2601.07994v2)**

> **作者:** Nayoung Choi; Jonathan Zhang; Jinho D. Choi
>
> **备注:** Accepted (B) to TACL 2026
>
> **摘要:** Large Language Models (LLMs) often exhibit increased response latency and degraded answer quality as dialogue length grows, making effective context management essential. However, existing methods rely on extra LLM calls to build memory or perform offline memory construction without considering the current user utterance, which can introduce inefficiencies or disrupt conversational continuity. We introduce DyCP, a lightweight context management method that dynamically segment and retrieve relevant memory at query time. It preserves the sequential structure of dialogue without predefined topic boundaries and supports efficient, adaptive context retrieval. Across three long-form dialogue benchmarks, LoCoMo, MT-Bench+, and SCM4LLMs, and multiple LLMs, DyCP consistently improves answer quality while reducing response latency. We also examine the gap between modern LLMs' expanded context windows and their actual long-context processing capacity, highlighting the continued importance of effective context management.
>
---
#### [replaced 053] A Benchmark for End-to-End Zero-Shot Biomedical Relation Extraction with LLMs: Experiments with OpenAI Models
- **分类: cs.CL**

- **简介: 该论文属于生物医学关系抽取任务，旨在评估大模型在零样本情况下的表现。通过构建基准数据集，测试了OpenAI模型的端到端零样本关系抽取能力，发现其在复杂输入上仍有提升空间。**

- **链接: [https://arxiv.org/pdf/2504.04083v3](https://arxiv.org/pdf/2504.04083v3)**

> **作者:** Aviv Brokman; Xuguang Ai; Yuhang Jiang; Shashank Gupta; Ramakanth Kavuluru
>
> **备注:** Accepted to appear in proceedings of the 3rd Workshop on Artificial Intelligence for Scientific Publications (WASP@AACL 2025). Code and data are available here: https://github.com/bionlproc/ZeroShotRE
>
> **摘要:** Extracting relations from scientific literature is a fundamental task in biomedical NLP because entities and relations among them drive hypothesis generation and knowledge discovery. As literature grows rapidly, relation extraction (RE) is indispensable to curate knowledge graphs to be used as computable structured and symbolic representations. With the rise of LLMs, it is pertinent to examine if it is better to skip tailoring supervised RE methods, save annotation burden, and just use zero shot RE (ZSRE) via LLM API calls. In this paper, we propose a benchmark with seven biomedical RE datasets with interesting characteristics and evaluate three Open AI models (GPT-4, o1, and GPT-OSS-120B) for end-to-end ZSRE. We show that LLM-based ZSRE is inching closer to supervised methods in performances on some datasets but still struggles on complex inputs expressing multiple relations with different predicates. Our error analysis reveals scope for improvements.
>
---
#### [replaced 054] Pragmatic Inference for Moral Reasoning Acquisition: Generalization via Distributional Semantics
- **分类: cs.CL**

- **简介: 该论文属于道德推理任务，旨在解决LLMs在道德推理中的泛化问题。通过引入语用推理方法，结合道德基础理论，提升模型的道德推理能力。**

- **链接: [https://arxiv.org/pdf/2509.24102v3](https://arxiv.org/pdf/2509.24102v3)**

> **作者:** Guangliang Liu; Xi Chen; Bocheng Chen; Han Zi; Xitong Zhang; Kristen Johnson
>
> **摘要:** Moral reasoning has emerged as a promising research direction for Large Language Models (LLMs), yet achieving generalization remains a central challenge. From a linguistic standpoint, this difficulty arises because LLMs are adept at capturing distributional semantics, which fundamentally differs from the morals which operate at the pragmatic level. This paper investigates how LLMs can achieve generalized moral reasoning despite their reliance on distributional semantics. We propose pragmatic inference methods grounded in moral foundations theory, which leverage contextual information at each step to bridge the pragmatic gap and guide LLMs in connecting moral foundations with moral reasoning objectives. Experimental results demonstrate that our approach significantly enhances LLMs' generalization in moral reasoning, providing a foundation for future research grounded in moral foundations theory.
>
---
#### [replaced 055] Beyond Chunking: Discourse-Aware Hierarchical Retrieval for Long Document Question Answering
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于长文档问答任务，旨在解决传统方法忽视话语结构的问题。通过引入话语分析和层级检索，提升问答系统的准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2506.06313v4](https://arxiv.org/pdf/2506.06313v4)**

> **作者:** Huiyao Chen; Yi Yang; Yinghui Li; Meishan Zhang; Baotian Hu; Min Zhang
>
> **备注:** 21 pages, 9 figures
>
> **摘要:** Existing long-document question answering systems typically process texts as flat sequences or use heuristic chunking, which overlook the discourse structures that naturally guide human comprehension. We present a discourse-aware hierarchical framework that leverages rhetorical structure theory (RST) for long document question answering. Our approach converts discourse trees into sentence-level representations and employs LLM-enhanced node representations to bridge structural and semantic information. The framework involves three key innovations: language-universal discourse parsing for lengthy documents, LLM-based enhancement of discourse relation nodes, and structure-guided hierarchical retrieval. Extensive experiments on four datasets demonstrate consistent improvements over existing approaches through the incorporation of discourse structure, across multiple genres and languages. Moreover, the proposed framework exhibits strong robustness across diverse document types and linguistic settings.
>
---

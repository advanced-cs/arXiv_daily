# 自然语言处理 cs.CL

- **最新发布 117 篇**

- **更新 88 篇**

## 最新发布

#### [new 001] Kardia-R1: Unleashing LLMs to Reason toward Understanding and Empathy for Emotional Support via Rubric-as-Judge Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对对话系统中缺乏个性化情感推理的问题，提出Kardia-R1框架。通过构建大规模用户锚定的基准数据集KardiaBench，采用基于评分标准的强化学习方法，实现可解释的情感理解与共情响应生成，显著提升模型在情感准确性、共情性与人格一致性上的表现。**

- **链接: [https://arxiv.org/pdf/2512.01282v1](https://arxiv.org/pdf/2512.01282v1)**

> **作者:** Jiahao Yuan; Zhiqing Cui; Hanqing Wang; Yuansheng Gao; Yucheng Zhou; Usman Naseem
>
> **摘要:** As web platforms evolve towards greater personalization and emotional complexity, conversational agents must transcend superficial empathy to demonstrate identity-aware emotional reasoning. However, existing systems face two limitations: (1) reliance on situation-centric datasets lacking persistent user identity, which hampers the capture of personalized affective nuances; and (2) dependence on opaque, coarse reward signals that hinder development of verifiable empathetic reasoning. To address these gaps, we introduce KardiaBench, a large-scale user-grounded benchmark comprising 178,080 QA pairs across 22,080 multi-turn conversations anchored to 671 real-world profiles. The dataset is constructed via a model-in-the-loop pipeline with iterative rubric-guided refinement to ensure psychological plausibility and persona consistency. This progressive empathy pipeline that integrates user comprehension, contextual reasoning, and emotion perception into conversations, followed by iterative critique and rubric-based refinement to ensure psychological plausibility, emotional fidelity, and persona consistency. Building on this, we propose Kardia-R1, a framework that trains models for interpretable, stepwise empathetic cognition. Kardia-R1 leverages Rubric-as-Judge Empathetic Reinforcement Learning (Rubric-ERL), a GRPO-based method that uses explainable, human-aligned rubric rewards to tightly couple user understanding, emotional inference, and supportive response generation. Extensive experiments across four LLM backbones demonstrate that Kardia-R1 consistently outperforms othet methods in emotion accuracy, empathy, relevance, persona consistency, and safety. Our dataset and model will be released at https://github.com/JhCircle/Kardia-R1.
>
---
#### [new 002] Graphing the Truth: Structured Visualizations for Automated Hallucination Detection in LLMs
- **分类: cs.CL**

- **简介: 该论文针对大模型幻觉问题，提出基于知识图谱的可视化框架，将模型输出与源知识关联并标注置信度，实现幻觉定位与可解释性。通过交互式界面支持用户诊断与反馈，构建人机协同的可靠性增强机制。**

- **链接: [https://arxiv.org/pdf/2512.00663v1](https://arxiv.org/pdf/2512.00663v1)**

> **作者:** Tanmay Agrawal
>
> **摘要:** Large Language Models have rapidly advanced in their ability to interpret and generate natural language. In enterprise settings, they are frequently augmented with closed-source domain knowledge to deliver more contextually informed responses. However, operational constraints such as limited context windows and inconsistencies between pre-training data and supplied knowledge often lead to hallucinations, some of which appear highly credible and escape routine human review. Current mitigation strategies either depend on costly, large-scale gold-standard Q\&A curation or rely on secondary model verification, neither of which offers deterministic assurance. This paper introduces a framework that organizes proprietary knowledge and model-generated content into interactive visual knowledge graphs. The objective is to provide end users with a clear, intuitive view of potential hallucination zones by linking model assertions to underlying sources of truth and indicating confidence levels. Through this visual interface, users can diagnose inconsistencies, identify weak reasoning chains, and supply corrective feedback. The resulting human-in-the-loop workflow creates a structured feedback loop that can enhance model reliability and continuously improve response quality.
>
---
#### [new 003] Rectifying LLM Thought from Lens of Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于提升大语言模型（LLM）的推理能力，针对长链思维（CoT）中常见的过度思考、推理过长等问题。提出RePro方法，从优化视角将推理视为梯度下降过程，通过双评分机制构建过程级奖励，结合RLVR框架进行后训练优化，有效提升推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2512.01925v1](https://arxiv.org/pdf/2512.01925v1)**

> **作者:** Junnan Liu; Hongwei Liu; Songyang Zhang; Kai Chen
>
> **备注:** Work in progress
>
> **摘要:** Recent advancements in large language models (LLMs) have been driven by their emergent reasoning capabilities, particularly through long chain-of-thought (CoT) prompting, which enables thorough exploration and deliberation. Despite these advances, long-CoT LLMs often exhibit suboptimal reasoning behaviors, such as overthinking and excessively protracted reasoning chains, which can impair performance. In this paper, we analyze reasoning processes through an optimization lens, framing CoT as a gradient descent procedure where each reasoning step constitutes an update toward problem resolution. Building on this perspective, we introduce RePro (Rectifying Process-level Reward), a novel approach to refine LLM reasoning during post-training. RePro defines a surrogate objective function to assess the optimization process underlying CoT, utilizing a dual scoring mechanism to quantify its intensity and stability. These scores are aggregated into a composite process-level reward, seamlessly integrated into reinforcement learning with verifiable rewards (RLVR) pipelines to optimize LLMs. Extensive experiments across multiple reinforcement learning algorithms and diverse LLMs, evaluated on benchmarks spanning mathematics, science, and coding, demonstrate that RePro consistently enhances reasoning performance and mitigates suboptimal reasoning behaviors.
>
---
#### [new 004] DrawingBench: Evaluating Spatial Reasoning and UI Interaction Capabilities of Large Language Models through Mouse-Based Drawing Tasks
- **分类: cs.CL**

- **简介: 该论文提出DrawingBench，一个用于评估大语言模型空间推理与界面交互能力的透明验证框架。针对现有评估缺乏可审计性的问题，通过鼠标绘图任务，以8项客观标准实现可复现评分与行为审计，揭示模型在工具状态管理与长程规划中的局限，并验证外部反馈优于自修正。**

- **链接: [https://arxiv.org/pdf/2512.01174v1](https://arxiv.org/pdf/2512.01174v1)**

> **作者:** Hyunjun Kim; Sooyoung Ryu
>
> **备注:** AAAI 2026 TrustAgent Workshop
>
> **摘要:** As agentic AI systems increasingly operate autonomously, establishing trust through verifiable evaluation becomes critical. Yet existing benchmarks lack the transparency and auditability needed to assess whether agents behave reliably. We present DrawingBench, a verification framework for evaluating the trustworthiness of agentic LLMs through spatial reasoning tasks that require generating sequences of low-level GUI actions. Unlike opaque evaluations, DrawingBench provides transparent, rule-based assessment: 8 objective criteria enable reproducible scoring, while action-level inspection allows stakeholders to audit agent behavior. Our framework comprises 250 diverse prompts across 20 categories and 4 difficulty levels, deterministic evaluation metrics, and an external oversight mechanism through multi-turn feedback that enables human control over agent refinement. Evaluating four state-of-the-art LLMs (Claude-4 Sonnet, GPT-4.1, GPT-4.1-mini, Gemini-2.5 Flash) across 1,000 tests, we establish both capabilities and limitations: models achieved 92.8% perfect performance with structured external feedback driving significant improvements (average +3.2%, up to +32.8% for complex scenes), but systematic error patterns emerged in tool state management and long-horizon planning. Notably, specification clarity proved more important than task complexity -- models achieved 100% perfect performance when given explicit, verifiable criteria. These findings demonstrate that transparent evaluation frameworks can establish trust in agentic systems, with external oversight proving more reliable than self-correction for guiding agent behavior. Our open-source framework provides a template for trustworthy agent assessment. Code and data: https://github.com/hyunjun1121/DrawingBench
>
---
#### [new 005] Exploring Human Perceptions of AI Responses: Insights from a Mixed-Methods Study on Risk Mitigation in Generative Models
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文研究生成式AI响应的感知问题，聚焦于人类对缓解策略效果的评价。针对模型幻觉与有害内容风险，通过混合方法实验评估多维度表现，揭示语言背景、经验等因素影响，并提出新评估指标，为人类-AI评估提供洞见。**

- **链接: [https://arxiv.org/pdf/2512.01892v1](https://arxiv.org/pdf/2512.01892v1)**

> **作者:** Heloisa Candello; Muneeza Azmat; Uma Sushmitha Gunturi; Raya Horesh; Rogerio Abreu de Paula; Heloisa Pimentel; Marcelo Carpinette Grave; Aminat Adebiyi; Tiago Machado; Maysa Malfiza Garcia de Macedo
>
> **备注:** 16 pages, 2 figures, 6 tables. Under review for publication
>
> **摘要:** With the rapid uptake of generative AI, investigating human perceptions of generated responses has become crucial. A major challenge is their `aptitude' for hallucinating and generating harmful contents. Despite major efforts for implementing guardrails, human perceptions of these mitigation strategies are largely unknown. We conducted a mixed-method experiment for evaluating the responses of a mitigation strategy across multiple-dimensions: faithfulness, fairness, harm-removal capacity, and relevance. In a within-subject study design, 57 participants assessed the responses under two conditions: harmful response plus its mitigation and solely mitigated response. Results revealed that participants' native language, AI work experience, and annotation familiarity significantly influenced evaluations. Participants showed high sensitivity to linguistic and contextual attributes, penalizing minor grammar errors while rewarding preserved semantic contexts. This contrasts with how language is often treated in the quantitative evaluation of LLMs. We also introduced new metrics for training and evaluating mitigation strategies and insights for human-AI evaluation studies.
>
---
#### [new 006] ART: Adaptive Response Tuning Framework -- A Multi-Agent Tournament-Based Approach to LLM Response Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ART框架，针对LLM响应不一致、幻觉等问题，通过多智能体竞赛与协作，结合ELO评分与共识融合，优化生成质量。实验显示其显著提升准确性与可靠性，适用于高要求场景。**

- **链接: [https://arxiv.org/pdf/2512.00617v1](https://arxiv.org/pdf/2512.00617v1)**

> **作者:** Omer Jauhar Khan
>
> **备注:** 8 pages, 5 figures, 5 tables. Conference-style paper
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation. However, single-model responses often exhibit inconsistencies, hallucinations, and varying quality across different query domains. This paper presents ART (Adaptive Response Tuning), a novel framework that employs tournament-style ELO ranking and multi-agent reasoning to systematically optimize LLM outputs. By enabling multiple LLM agents to compete, critique, and collaborate through structured tournament workflows, ART produces consensus responses that outperform individual model outputs. Our framework introduces configurable tournament parameters, dynamic agent selection, and multiple consensus fusion strategies. Experimental evaluations demonstrate significant improvements in response accuracy, coherence, and reliability compared to baseline single-model approaches. The ART framework provides a scalable, production-ready solution for applications requiring high-quality, vetted LLM responses, achieving an 8.4% improvement in overall quality metrics and R22 values exceeding 0.96 in ELO rating convergence.
>
---
#### [new 007] InnoGym: Benchmarking the Innovation Potential of AI Agents
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文提出InnoGym，首个系统评估AI代理创新潜力的基准与框架。针对现有评测仅关注正确性、忽视方法多样性的不足，引入性能提升与新颖性双指标，涵盖18个真实领域任务，并提供可复现的执行环境。实验揭示当前代理虽具创意但缺乏鲁棒性，凸显创新与有效性间的差距。**

- **链接: [https://arxiv.org/pdf/2512.01822v1](https://arxiv.org/pdf/2512.01822v1)**

> **作者:** Jintian Zhang; Kewei Xu; Jingsheng Zheng; Zhuoyun Yu; Yuqi Zhu; Yujie Luo; Lanning Wei; Shuofei Qiao; Lun Du; Da Zheng; Shumin Deng; Huajun Chen; Ningyu Zhang
>
> **备注:** Work in progress
>
> **摘要:** LLMs and Agents have achieved impressive progress in code generation, mathematical reasoning, and scientific discovery. However, existing benchmarks primarily measure correctness, overlooking the diversity of methods behind solutions. True innovation depends not only on producing correct answers but also on the originality of the approach. We present InnoGym, the first benchmark and framework designed to systematically evaluate the innovation potential of AI agents. InnoGym introduces two complementary metrics: performance gain, which measures improvement over the best-known solutions, and novelty, which captures methodological differences from prior approaches. The benchmark includes 18 carefully curated tasks from real-world engineering and scientific domains, each standardized through resource filtering, evaluator validation, and solution collection. In addition, we provide iGym, a unified execution environment for reproducible and long-horizon evaluations. Extensive experiments show that while some agents produce novel approaches, their lack of robustness limits performance gains. These results highlight a key gap between creativity and effectiveness, underscoring the need for benchmarks that evaluate both.
>
---
#### [new 008] Emergent Convergence in Multi-Agent LLM Annotation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多智能体大语言模型在协作标注任务中的涌现协调行为。针对黑箱模型间协作机制不明的问题，通过模拟多轮讨论，分析语义一致性、嵌入几何变化等指标，揭示其自发形成词汇与语义收敛、不对称影响力及协商行为，为理解非显式角色下的群体智能提供新视角。**

- **链接: [https://arxiv.org/pdf/2512.00047v1](https://arxiv.org/pdf/2512.00047v1)**

> **作者:** Angelina Parfenova; Alexander Denzler; Juergen Pfeffer
>
> **摘要:** Large language models (LLMs) are increasingly deployed in collaborative settings, yet little is known about how they coordinate when treated as black-box agents. We simulate 7500 multi-agent, multi-round discussions in an inductive coding task, generating over 125000 utterances that capture both final annotations and their interactional histories. We introduce process-level metrics: code stability, semantic self-consistency, and lexical confidence alongside sentiment and convergence measures, to track coordination dynamics. To probe deeper alignment signals, we analyze the evolving geometry of output embeddings, showing that intrinsic dimensionality declines over rounds, suggesting semantic compression. The results reveal that LLM groups converge lexically and semantically, develop asymmetric influence patterns, and exhibit negotiation-like behaviors despite the absence of explicit role prompting. This work demonstrates how black-box interaction analysis can surface emergent coordination strategies, offering a scalable complement to internal probe-based interpretability methods.
>
---
#### [new 009] Lost without translation -- Can transformer (language models) understand mood states?
- **分类: cs.CL**

- **简介: 该论文研究多语言情绪识别任务，旨在解决大模型难以理解印度语言中情绪表达的问题。通过收集11种印地语系语言的247个情绪短语，对比直接嵌入与翻译后嵌入的效果，发现翻译可显著提升聚类性能，但依赖外部工具不可持续，强调需构建本土化语言模型以推动全球心理健康应用。**

- **链接: [https://arxiv.org/pdf/2512.00274v1](https://arxiv.org/pdf/2512.00274v1)**

> **作者:** Prakrithi Shivaprakash; Diptadhi Mukherjee; Lekhansh Shukla; Animesh Mukherjee; Prabhat Chand; Pratima Murthy
>
> **备注:** 33 pages, 3 figures, 2 tables
>
> **摘要:** Background: Large Language Models show promise in psychiatry but are English-centric. Their ability to understand mood states in other languages is unclear, as different languages have their own idioms of distress. Aim: To quantify the ability of language models to faithfully represent phrases (idioms of distress) of four distinct mood states (depression, euthymia, euphoric mania, dysphoric mania) expressed in Indian languages. Methods: We collected 247 unique phrases for the four mood states across 11 Indic languages. We tested seven experimental conditions, comparing k-means clustering performance on: (a) direct embeddings of native and Romanised scripts (using multilingual and Indic-specific models) and (b) embeddings of phrases translated to English and Chinese. Performance was measured using a composite score based on Adjusted Rand Index, Normalised Mutual Information, Homogeneity and Completeness. Results: Direct embedding of Indic languages failed to cluster mood states (Composite Score = 0.002). All translation-based approaches showed significant improvement. High performance was achieved using Gemini-translated English (Composite=0.60) and human-translated English (Composite=0.61) embedded with gemini-001. Surprisingly, human-translated English, further translated into Chinese and embedded with a Chinese model, performed best (Composite = 0.67). Specialised Indic models (IndicBERT and Sarvam-M) performed poorly. Conclusion: Current models cannot meaningfully represent mood states directly from Indic languages, posing a fundamental barrier to their psychiatric application for diagnostic or therapeutic purposes in India. While high-quality translation bridges this gap, reliance on proprietary models or complex translation pipelines is unsustainable. Models must first be built to understand diverse local languages to be effective in global mental health.
>
---
#### [new 010] A Comparison of Human and ChatGPT Classification Performance on Complex Social Media Data
- **分类: cs.CL**

- **简介: 该论文比较人类与ChatGPT在复杂社交媒体数据分类任务中的表现。针对大语言模型在处理含细微语义的语言时的准确性不足问题，研究评估了GPT-3.5、GPT-4和GPT-4o在四种提示风格下的分类性能，发现其在处理复杂语义时仍存局限，建议谨慎使用。**

- **链接: [https://arxiv.org/pdf/2512.00673v1](https://arxiv.org/pdf/2512.00673v1)**

> **作者:** Breanna E. Green; Ashley L. Shea; Pengfei Zhao; Drew B. Margolin
>
> **备注:** About 15 pages, draft version of accepted conference full paper. Published paper to follow
>
> **摘要:** Generative artificial intelligence tools, like ChatGPT, are an increasingly utilized resource among computational social scientists. Nevertheless, there remains space for improved understanding of the performance of ChatGPT in complex tasks such as classifying and annotating datasets containing nuanced language. Method. In this paper, we measure the performance of GPT-4 on one such task and compare results to human annotators. We investigate ChatGPT versions 3.5, 4, and 4o to examine performance given rapid changes in technological advancement of large language models. We craft four prompt styles as input and evaluate precision, recall, and F1 scores. Both quantitative and qualitative evaluations of results demonstrate that while including label definitions in prompts may help performance, overall GPT-4 has difficulty classifying nuanced language. Qualitative analysis reveals four specific findings. Our results suggest the use of ChatGPT in classification tasks involving nuanced language should be conducted with prudence.
>
---
#### [new 011] How Far Are We from Genuinely Useful Deep Research Agents?
- **分类: cs.CL**

- **简介: 该论文针对深度研究代理（DRA）在生成综合报告时的实用性不足问题，提出FINDER基准与DEFT失败分类体系。通过100项人工标注任务和419项检查项，系统评估报告质量，并揭示当前DRAs主要在证据整合、验证与推理规划上存在缺陷。**

- **链接: [https://arxiv.org/pdf/2512.01948v1](https://arxiv.org/pdf/2512.01948v1)**

> **作者:** Dingling Zhang; He Zhu; Jincheng Ren; Kangqi Song; Xinran Zhou; Boyu Feng; Shudong Liu; Jiabin Luo; Weihao Xie; Zhaohui Wang; Tianrui Qin; King Zhu; Yuqing Wang; Qianben Chen; Yuchen Eleanor Jiang; Wei Wang; Jiaheng Liu; Wangchunshu Zhou
>
> **备注:** 34 pages
>
> **摘要:** Deep Research Agents (DRAs) aim to automatically produce analyst-level reports through iterative information retrieval and synthesis. However, most existing DRAs were validated on question-answering benchmarks, while research on generating comprehensive reports remains overlooked. Worse, current benchmarks for report synthesis suffer from task complexity and subjective metrics -- this fails to reflect user demands and limits the practical utility of generated reports. To address these gaps, we present Fine-grained DEepResearch bench (FINDER), an enhanced benchmark consisting of 100 human-curated research tasks with 419 structured checklist items that standardize report structure, analytical depth, and factual grounding. Based on approximately 1,000 reports produced by mainstream DRAs, we further propose Deep rEsearch Failure Taxonomy (DEFT), the first failure taxonomy for deep research agents. DEFT contains 14 fine-grained failure modes across reasoning, retrieval, and generation, and is built upon grounded theory with human-LLM co-annotating and inter-annotator reliability validation. Our experimental findings reveal that current DRAs struggle not with task comprehension but with evidence integration, verification, and reasoning-resilient planning.
>
---
#### [new 012] Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference
- **分类: cs.CL**

- **简介: 该论文聚焦于误导性信息检测任务，针对隐性虚假信息——即通过关键信息遗漏诱导错误判断的现象。提出OmiGraph框架，构建上下文关联图以挖掘被省略内容，通过关系建模与消息传递机制识别遗漏模式，实现对隐性误导的精准检测。**

- **链接: [https://arxiv.org/pdf/2512.01728v1](https://arxiv.org/pdf/2512.01728v1)**

> **作者:** Zhengjia Wang; Danding Wang; Qiang Sheng; Jiaying Wu; Juan Cao
>
> **备注:** AAAI 2026
>
> **摘要:** This paper investigates the detection of misinformation, which deceives readers by explicitly fabricating misleading content or implicitly omitting important information necessary for informed judgment. While the former has been extensively studied, omission-based deception remains largely overlooked, even though it can subtly guide readers toward false conclusions under the illusion of completeness. To pioneer in this direction, this paper presents OmiGraph, the first omission-aware framework for misinformation detection. Specifically, OmiGraph constructs an omission-aware graph for the target news by utilizing a contextual environment that captures complementary perspectives of the same event, thereby surfacing potentially omitted contents. Based on this graph, omission-oriented relation modeling is then proposed to identify the internal contextual dependencies, as well as the dynamic omission intents, formulating a comprehensive omission relation representation. Finally, to extract omission patterns for detection, OmiGraph introduces omission-aware message-passing and aggregation that establishes holistic deception perception by integrating the omission contents and relations. Experiments show that, by considering the omission perspective, our approach attains remarkable performance, achieving average improvements of +5.4% F1 and +5.3% ACC on two large-scale benchmarks.
>
---
#### [new 013] Beyond SFT: Reinforcement Learning for Safer Large Reasoning Models with Better Reasoning Ability
- **分类: cs.CL**

- **简介: 该论文研究大推理模型（LRM）的安全对齐问题。针对现有监督微调（SFT）导致安全提升不一致、削弱推理能力的问题，提出采用强化学习（RL）优化模型安全。实验表明，RL在保持推理能力的同时，更稳定地提升安全性，抑制不安全探索，实现更可靠的推理过程。**

- **链接: [https://arxiv.org/pdf/2512.01848v1](https://arxiv.org/pdf/2512.01848v1)**

> **作者:** Jinghan Jia; Nathalie Baracaldo; Sijia Liu
>
> **摘要:** Large reasoning models (LRMs) extend large language models by generating explicit chain-of-thought (CoT) reasoning, significantly improving mathematical and logical problem solving. However, this explicit reasoning process also introduces new safety risks, as unsafe behaviors often emerge within intermediate reasoning trajectories, even when final answers appear harmless. Existing safety alignment approaches primarily rely on supervised fine-tuning (SFT) over safety-oriented long CoT datasets. While intuitive, we find that SFT produces inconsistent safety improvements, degrades reasoning ability, and generalizes poorly across model families. These limitations suggest that purely supervised approaches are insufficient for robust safety alignment in LRMs. To address this, we investigate reinforcement learning (RL) as a complementary optimization framework for LRM safety training. Unlike SFT, RL directly optimizes model policies with reward feedback, enabling more adaptive and stable alignment. Extensive experiments across multiple model families and benchmarks show that RL achieves stronger and more consistent safety gains while maintaining reasoning competence. Further analysis of reflection dynamics and token-level entropy reveals that RL suppresses unsafe exploratory reasoning while preserving reflective depth, leading to safer and more reliable reasoning processes.
>
---
#### [new 014] Wikontic: Constructing Wikidata-Aligned, Ontology-Aware Knowledge Graphs with Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Wikontic，一个基于大模型的多阶段知识图谱构建框架，旨在解决现有LLM-KG系统中知识质量不足的问题。通过提取带限定词的三元组、约束类型与关系、实体归一化，生成高质量、结构一致的紧凑知识图谱，显著提升问答性能与信息保留率，且构建高效。**

- **链接: [https://arxiv.org/pdf/2512.00590v1](https://arxiv.org/pdf/2512.00590v1)**

> **作者:** Alla Chepurova; Aydar Bulatov; Yuri Kuratov; Mikhail Burtsev
>
> **摘要:** Knowledge graphs (KGs) provide structured, verifiable grounding for large language models (LLMs), but current LLM-based systems commonly use KGs as auxiliary structures for text retrieval, leaving their intrinsic quality underexplored. In this work, we propose Wikontic, a multi-stage pipeline that constructs KGs from open-domain text by extracting candidate triplets with qualifiers, enforcing Wikidata-based type and relation constraints, and normalizing entities to reduce duplication. The resulting KGs are compact, ontology-consistent, and well-connected; on MuSiQue, the correct answer entity appears in 96% of generated triplets. On HotpotQA, our triplets-only setup achieves 76.0 F1, and on MuSiQue 59.8 F1, matching or surpassing several retrieval-augmented generation baselines that still require textual context. In addition, Wikontic attains state-of-the-art information-retention performance on the MINE-1 benchmark (86%), outperforming prior KG construction methods. Wikontic is also efficient at build time: KG construction uses less than 1,000 output tokens, about 3$\times$ fewer than AriGraph and $<$1/20 of GraphRAG. The proposed pipeline enhances the quality of the generated KG and offers a scalable solution for leveraging structured knowledge in LLMs.
>
---
#### [new 015] Less is More: Resource-Efficient Low-Rank Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型微调中LoRA方法资源开销高、参数干扰问题，提出EffiLoRA。通过共享统一A矩阵并动态选择性更新B矩阵，实现跨模态的高效微调，在保持性能的同时显著降低资源消耗，提升模型效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.00878v1](https://arxiv.org/pdf/2512.00878v1)**

> **作者:** Chunlin Tian; Xuyang Wei; Huanrong Liu; Zhijiang Guo; Li Li
>
> **备注:** 18 pages, 7 figures
>
> **摘要:** Low-Rank Adaptation (LoRA) is a widely adopted parameter-efficient fine-tuning (PEFT) method for Large Language Models (LLMs), but it still incurs notable overhead and suffers from parameter interference in complex datasets. While re- cent works decouple LoRA update matrices to exploit matrix-wise asymmetry, training costs remain high. We revisit LoRA from the perspective of inter-matrix and intra-layer parameter redundancy and propose Resource-Efficient Low-Rank Adaptation, EffiLoRA, a lightweight and generalizable approach for language, multimodal, and diffusion models. EffiLoRA employs a unified A matrix across all transformer layers and introduces a runtime selective B matrices up- date to dynamically trade-off the system resource budget and model performance. EffiLoRA consistently outperforms LoRA across diverse modalities, including commonsense reasoning, visual instruction tuning, and image generation, demon- strating improved efficiency and robustness.
>
---
#### [new 016] BHRAM-IL: A Benchmark for Hallucination Recognition and Assessment in Multiple Indian Languages
- **分类: cs.CL; cs.AI; cs.ET**

- **简介: 该论文提出BHRAM-IL基准，针对多语言大模型中的幻觉问题，聚焦印地语、古吉拉特语、马拉地语、奥里亚语等印度语言。构建了36,047个跨领域问题数据集，评估14个主流多语言模型的幻觉生成情况，提供标准化评估指标，推动多语言幻觉检测研究。**

- **链接: [https://arxiv.org/pdf/2512.01852v1](https://arxiv.org/pdf/2512.01852v1)**

> **作者:** Hrishikesh Terdalkar; Kirtan Bhojani; Aryan Dongare; Omm Aditya Behera
>
> **备注:** Accepted at BHASHA Workshop @ IJCNLP/AACL 2025
>
> **摘要:** Large language models (LLMs) are increasingly deployed in multilingual applications but often generate plausible yet incorrect or misleading outputs, known as hallucinations. While hallucination detection has been studied extensively in English, under-resourced Indian languages remain largely unexplored. We present BHRAM-IL, a benchmark for hallucination recognition and assessment in multiple Indian languages, covering Hindi, Gujarati, Marathi, Odia, along with English. The benchmark comprises 36,047 curated questions across nine categories spanning factual, numerical, reasoning, and linguistic tasks. We evaluate 14 state-of-the-art multilingual LLMs on a benchmark subset of 10,265 questions, analyzing cross-lingual and factual hallucinations across languages, models, scales, categories, and domains using category-specific metrics normalized to (0,1) range. Aggregation over all categories and models yields a primary score of 0.23 and a language-corrected fuzzy score of 0.385, demonstrating the usefulness of BHRAM-IL for hallucination-focused evaluation. The dataset, and the code for generation and evaluation are available on GitHub (https://github.com/sambhashana/BHRAM-IL/) and HuggingFace (https://huggingface.co/datasets/sambhashana/BHRAM-IL/) to support future research in multilingual hallucination detection and mitigation.
>
---
#### [new 017] The Art of Scaling Test-Time Compute for Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型推理时的计算资源动态分配（测试时扩展，TTS）。针对现有TTS策略缺乏系统比较、模型类型与问题难度影响不明确的问题，作者在8个开源模型上开展大规模实验，发现不同策略无绝对优劣，模型按推理轨迹特征分为短/长视野类，且性能随算力单调提升。据此提出基于问题难度、模型类型和算力预算的TTS策略选择方案。**

- **链接: [https://arxiv.org/pdf/2512.02008v1](https://arxiv.org/pdf/2512.02008v1)**

> **作者:** Aradhye Agarwal; Ayan Sengupta; Tanmoy Chakraborty
>
> **摘要:** Test-time scaling (TTS) -- the dynamic allocation of compute during inference -- is a promising direction for improving reasoning in large language models (LLMs). However, a systematic comparison of well-known TTS strategies under identical conditions is missing, and the influence of model type and problem difficulty on performance remains unclear. To address these gaps, we conduct the first large-scale study of TTS, spanning over thirty billion tokens generated using eight open-source LLMs (7B to 235B parameters), across four reasoning datasets. We observe three consistent trends: (1) no single TTS strategy universally dominates; (2) reasoning models exhibit distinct trace-quality patterns across problem difficulty and trace length, forming short-horizon and long-horizon categories; and (3) for a given model type, the optimal TTS performance scales monotonically with compute budget. Based on these insights, we provide a practical recipe for selecting the best TTS strategy, considering problem difficulty, model type, and compute budget, providing a practical guide to effective inference-time scaling.
>
---
#### [new 018] Mitigating the Threshold Priming Effect in Large Language Model-Based Relevance Judgments via Personality Infusing
- **分类: cs.CL; cs.IR**

- **简介: 该论文研究大语言模型（LLM）在相关性判断中的阈值启动效应问题。针对LLM易受先前判断影响的缺陷，提出通过注入大五人格特质进行个性提示，以减轻启动偏差。实验表明，高开放性、低神经质等人格特征可有效降低启动效应，且最优人格因模型和任务而异，为提升LLM评估可靠性提供了新方法。**

- **链接: [https://arxiv.org/pdf/2512.00390v1](https://arxiv.org/pdf/2512.00390v1)**

> **作者:** Nuo Chen; Hanpei Fang; Jiqun Liu; Wilson Wei; Tetsuya Sakai; Xiao-Ming Wu
>
> **摘要:** Recent research has explored LLMs as scalable tools for relevance labeling, but studies indicate they are susceptible to priming effects, where prior relevance judgments influence later ones. Although psychological theories link personality traits to such biases, it is unclear whether simulated personalities in LLMs exhibit similar effects. We investigate how Big Five personality profiles in LLMs influence priming in relevance labeling, using multiple LLMs on TREC 2021 and 2022 Deep Learning Track datasets. Our results show that certain profiles, such as High Openness and Low Neuroticism, consistently reduce priming susceptibility. Additionally, the most effective personality in mitigating priming may vary across models and task types. Based on these findings, we propose personality prompting as a method to mitigate threshold priming, connecting psychological evidence with LLM-based evaluation practices.
>
---
#### [new 019] Conveying Imagistic Thinking in Traditional Chinese Medicine Translation: A Prompt Engineering and LLM-Based Evaluation Framework
- **分类: cs.CL**

- **简介: 该论文针对中医典籍翻译中意象思维传递不足的问题，提出基于提示工程与大模型的交互式评估框架。通过引导大模型识别并传达隐喻、转喻，结合多角色读者模拟评分，验证了优化后译文在认知维度上的优越性，为古籍概念密集文本的翻译提供了可复制的方法路径。**

- **链接: [https://arxiv.org/pdf/2512.01198v1](https://arxiv.org/pdf/2512.01198v1)**

> **作者:** Jiatong Han
>
> **备注:** 3 figures
>
> **摘要:** Traditional Chinese Medicine (TCM) theory is built on imagistic thinking, in which medical principles and diagnostic and therapeutic logic are structured through metaphor and metonymy. However, existing English translations largely rely on literal rendering, making it difficult for target-language readers to reconstruct the underlying conceptual networks and apply them in clinical practice. This study adopted a human-in-the-loop (HITL) framework and selected four passages from the medical canon Huangdi Neijing that are fundamental in theory. Through prompt-based cognitive scaffolding, DeepSeek V3.1 was guided to identify metaphor and metonymy in the source text and convey the theory in translation. In the evaluation stage, ChatGPT 5 Pro and Gemini 2.5 Pro were instructed by prompts to simulate three types of real-world readers. Human translations, baseline model translations, and prompt-adjusted translations were scored by the simulated readers across five cognitive dimensions, followed by structured interviews and Interpretative Phenomenological Analysis (IPA). Results show that the prompt-adjusted LLM translations perform best across all five dimensions, with high cross-model and cross-role consistency. The interview themes reveal differences between human and machine translation, effective strategies for metaphor and metonymy transfer, and readers' cognitive preferences. This study provides a cognitive, efficient, and replicable HITL methodological pathway for the translation of ancient, concept-dense texts such as TCM.
>
---
#### [new 020] Agreement-Constrained Probabilistic Minimum Bayes Risk Decoding
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对机器翻译中的最小贝叶斯风险（MBR）解码效率低的问题，提出一种约束一致性的概率MBR（AC-PMBR）方法。通过知识蒸馏模型指导得分矩阵补全，减少评估次数的同时提升翻译质量，显著改善了计算成本与译文质量的权衡。**

- **链接: [https://arxiv.org/pdf/2512.01316v1](https://arxiv.org/pdf/2512.01316v1)**

> **作者:** Koki Natsumi; Hiroyuki Deguchi; Yusuke Sakai; Hidetaka Kamigaito; Taro Watanabe
>
> **备注:** IJCNLP-AACL 2025 Main
>
> **摘要:** Minimum Bayes risk (MBR) decoding generates high-quality translations by maximizing the expected utility of output candidates, but it evaluates all pairwise scores over the candidate set; hence, it takes quadratic time with respect to the number of candidates. To reduce the number of utility function calls, probabilistic MBR (PMBR) decoding partially evaluates quality scores using sampled pairs of candidates and completes the missing scores with a matrix completion algorithm. Nevertheless, it degrades the translation quality as the number of utility function calls is reduced. Therefore, to improve the trade-off between quality and cost, we propose agreement-constrained PMBR (AC-PMBR) decoding, which leverages a knowledge distilled model to guide the completion of the score matrix. Our AC-PMBR decoding improved approximation errors of matrix completion by up to 3 times and achieved higher translation quality compared with PMBR decoding at a comparable computational cost on the WMT'23 En$\leftrightarrow$De translation tasks.
>
---
#### [new 021] DeformAr: Rethinking NER Evaluation through Component Analysis and Visual Analytics
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对阿拉伯语命名实体识别（NER）中模型性能低于英语的问题，提出DeformAr框架。通过组件分析与可视化工具，系统诊断数据与模型间的交互影响，揭示性能差异的根源，推动对低资源语言NLP模型的可解释性研究。**

- **链接: [https://arxiv.org/pdf/2512.00938v1](https://arxiv.org/pdf/2512.00938v1)**

> **作者:** Ahmed Mustafa Younes
>
> **备注:** PhD Thesis, University of Sussex, 2025. 311 pages, 140 figures, 32 tables. Submitted as a PDF-only. First supervisor: Julie Weeds. Second supervisor: David Weir
>
> **摘要:** Transformer models have significantly advanced Natural Language Processing (NLP), demonstrating strong performance in English. However, their effectiveness in Arabic, particularly for Named Entity Recognition (NER), remains limited, even with larger pre-trained models. This performance gap stems from multiple factors, including tokenisation, dataset quality, and annotation inconsistencies. Existing studies often analyze these issues in isolation, failing to capture their joint effect on system behaviour and performance. We introduce DeformAr (Debugging and Evaluation Framework for Transformer-based NER Systems), a novel framework designed to investigate and explain the performance discrepancy between Arabic and English NER systems. DeformAr integrates a data extraction library and an interactive dashboard, supporting two modes of evaluation: cross-component analysis and behavioural analysis. The framework divides each language into dataset and model components to examine their interactions. The analysis proceeds in two stages. First, cross-component analysis provides systematic diagnostic measures across data and model subcomponents, addressing the "what," "how," and "why" behind observed discrepancies. The second stage applies behavioural analysis by combining interpretability techniques with token-level metrics, interactive visualisations, and representation space analysis. These stages enable a component-aware diagnostic process that detects model behaviours and explains them by linking them to underlying representational patterns and data factors. DeformAr is the first Arabic-specific, component-based interpretability tool, offering a crucial resource for advancing model analysis in under-resourced languages.
>
---
#### [new 022] Tree Matching Networks for Natural Language Inference: Parameter-Efficient Semantic Understanding via Dependency Parse Trees
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对自然语言推理任务，提出基于依存句法树的Tree Matching Networks（TMN），以减少模型参数和训练时间。通过利用显式语法结构，提升语义理解效率，显著优于BERT在SNLI任务上的表现，且在相似性任务中仍面临挑战。**

- **链接: [https://arxiv.org/pdf/2512.00204v1](https://arxiv.org/pdf/2512.00204v1)**

> **作者:** Jason Lunder
>
> **备注:** 16 pages, preprint
>
> **摘要:** In creating sentence embeddings for Natural Language Inference (NLI) tasks, using transformer-based models like BERT leads to high accuracy, but require hundreds of millions of parameters. These models take in sentences as a sequence of tokens, and learn to encode the meaning of the sequence into embeddings such that those embeddings can be used reliably for NLI tasks. Essentially, every word is considered against every other word in the sequence, and the transformer model is able to determine the relationships between them, entirely from scratch. However, a model that accepts explicit linguistic structures like dependency parse trees may be able to leverage prior encoded information about these relationships, without having to learn them from scratch, thus improving learning efficiency. To investigate this, we adapt Graph Matching Networks (GMN) to operate on dependency parse trees, creating Tree Matching Networks (TMN). We compare TMN to a BERT based model on the SNLI entailment task and on the SemEval similarity task. TMN is able to achieve significantly better results with a significantly reduced memory footprint and much less training time than the BERT based model on the SNLI task, while both models struggled to preform well on the SemEval. Explicit structural representations significantly outperform sequence-based models at comparable scales, but current aggregation methods limit scalability. We propose multi-headed attention aggregation to address this limitation.
>
---
#### [new 023] Fine-tuning of lightweight large language models for sentiment classification on heterogeneous financial textual data
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究轻量级开源大语言模型在异构金融文本上的情感分类任务。针对大模型依赖昂贵资源的问题，比较FinBERT与三个轻量模型在多源、多语言数据上的表现，发现轻量模型仅用5%数据即可达良好效果，验证其低成本高效性。**

- **链接: [https://arxiv.org/pdf/2512.00946v1](https://arxiv.org/pdf/2512.00946v1)**

> **作者:** Alvaro Paredes Amorin; Andre Python; Christoph Weisser
>
> **摘要:** Large language models (LLMs) play an increasingly important role in finan- cial markets analysis by capturing signals from complex and heterogeneous textual data sources, such as tweets, news articles, reports, and microblogs. However, their performance is dependent on large computational resources and proprietary datasets, which are costly, restricted, and therefore inacces- sible to many researchers and practitioners. To reflect realistic situations we investigate the ability of lightweight open-source LLMs - smaller and publicly available models designed to operate with limited computational resources - to generalize sentiment understanding from financial datasets of varying sizes, sources, formats, and languages. We compare the benchmark finance natural language processing (NLP) model, FinBERT, and three open-source lightweight LLMs, DeepSeek-LLM 7B, Llama3 8B Instruct, and Qwen3 8B on five publicly available datasets: FinancialPhraseBank, Financial Question Answering, Gold News Sentiment, Twitter Sentiment and Chinese Finance Sentiment. We find that LLMs, specially Qwen3 8B and Llama3 8B, perform best in most scenarios, even from using only 5% of the available training data. These results hold in zero-shot and few-shot learning scenarios. Our findings indicate that lightweight, open-source large language models (LLMs) consti- tute a cost-effective option, as they can achieve competitive performance on heterogeneous textual data even when trained on only a limited subset of the extensive annotated corpora that are typically deemed necessary.
>
---
#### [new 024] MMAG: Mixed Memory-Augmented Generation for Large Language Models Applications
- **分类: cs.CL; cs.IR**

- **简介: 该论文针对大语言模型在长时交互中缺乏记忆与连贯性的问题，提出混合记忆增强生成（MMAG）框架，构建五层记忆体系，涵盖对话、用户长期、事件、感知及短期记忆。通过认知心理学启发，实现记忆协调与优先级管理，在Heero代理中验证其提升交互沉浸感与个性化能力，解决持续性与上下文一致性难题。**

- **链接: [https://arxiv.org/pdf/2512.01710v1](https://arxiv.org/pdf/2512.01710v1)**

> **作者:** Stefano Zeppieri
>
> **摘要:** Large Language Models (LLMs) excel at generating coherent text within a single prompt but fall short in sustaining relevance, personalization, and continuity across extended interactions. Human communication, however, relies on multiple forms of memory, from recalling past conversations to adapting to personal traits and situational context. This paper introduces the Mixed Memory-Augmented Generation (MMAG) pattern, a framework that organizes memory for LLM-based agents into five interacting layers: conversational, long-term user, episodic and event-linked, sensory and context-aware, and short-term working memory. Drawing inspiration from cognitive psychology, we map these layers to technical components and outline strategies for coordination, prioritization, and conflict resolution. We demonstrate the approach through its implementation in the Heero conversational agent, where encrypted long-term bios and conversational history already improve engagement and retention. We further discuss implementation concerns around storage, retrieval, privacy, and latency, and highlight open challenges. MMAG provides a foundation for building memory-rich language agents that are more coherent, proactive, and aligned with human needs.
>
---
#### [new 025] Enhancing BERT Fine-Tuning for Sentiment Analysis in Lower-Resourced Languages
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对低资源语言情感分析任务，解决标注数据少导致模型性能差的问题。提出结合主动学习与聚类的动态数据选择框架，通过优化数据筛选策略，在减少30%标注量的同时提升F1分数最多4点，增强微调稳定性。**

- **链接: [https://arxiv.org/pdf/2512.01460v1](https://arxiv.org/pdf/2512.01460v1)**

> **作者:** Jozef Kubík; Marek Šuppa; Martin Takáč
>
> **摘要:** Limited data for low-resource languages typically yield weaker language models (LMs). Since pre-training is compute-intensive, it is more pragmatic to target improvements during fine-tuning. In this work, we examine the use of Active Learning (AL) methods augmented by structured data selection strategies which we term 'Active Learning schedulers', to boost the fine-tuning process with a limited amount of training data. We connect the AL to data clustering and propose an integrated fine-tuning pipeline that systematically combines AL, clustering, and dynamic data selection schedulers to enhance model's performance. Experiments in the Slovak, Maltese, Icelandic and Turkish languages show that the use of clustering during the fine-tuning phase together with AL scheduling can simultaneously produce annotation savings up to 30% and performance improvements up to four F1 score points, while also providing better fine-tuning stability.
>
---
#### [new 026] MEGConformer: Conformer-Based MEG Decoder for Robust Speech and Phoneme Classification
- **分类: cs.CL; cs.LG; cs.NE; cs.SD**

- **简介: 该论文针对脑磁图（MEG）信号的语音检测与音素分类任务，提出基于Conformer的轻量级解码器。通过适配306通道原始MEG数据，引入专用增强、加权策略与归一化方法，有效缓解分布偏移与样本不均衡问题，在竞赛中分别取得88.9%和65.8%的F1-macro成绩，显著优于基线。**

- **链接: [https://arxiv.org/pdf/2512.01443v1](https://arxiv.org/pdf/2512.01443v1)**

> **作者:** Xabier de Zuazo; Ibon Saratxaga; Eva Navas
>
> **备注:** 10 pages, 5 figures, 4 tables, LibriBrain Workshop, NeurIPS 2025
>
> **摘要:** We present Conformer-based decoders for the LibriBrain 2025 PNPL competition, targeting two foundational MEG tasks: Speech Detection and Phoneme Classification. Our approach adapts a compact Conformer to raw 306-channel MEG signals, with a lightweight convolutional projection layer and task-specific heads. For Speech Detection, a MEG-oriented SpecAugment provided a first exploration of MEG-specific augmentation. For Phoneme Classification, we used inverse-square-root class weighting and a dynamic grouping loader to handle 100-sample averaged examples. In addition, a simple instance-level normalization proved critical to mitigate distribution shifts on the holdout split. Using the official Standard track splits and F1-macro for model selection, our best systems achieved 88.9% (Speech) and 65.8% (Phoneme) on the leaderboard, surpassing the competition baselines and ranking within the top-10 in both tasks. For further implementation details, the technical documentation, source code, and checkpoints are available at https://github.com/neural2speech/libribrain-experiments.
>
---
#### [new 027] CryptoBench: A Dynamic Benchmark for Expert-Level Evaluation of LLM Agents in Cryptocurrency
- **分类: cs.CL**

- **简介: 该论文提出CryptoBench，首个面向加密货币领域的动态专家评测基准，旨在评估大语言模型代理在高时效、强对抗环境下的真实能力。针对现有基准在专业数据融合与预测分析上的不足，构建了月度更新的50题动态任务集，按四象限分类，揭示主流模型存在“检索-预测失衡”问题，凸显其分析短板。**

- **链接: [https://arxiv.org/pdf/2512.00417v1](https://arxiv.org/pdf/2512.00417v1)**

> **作者:** Jiacheng Guo; Suozhi Huang; Zixin Yao; Yifan Zhang; Yifu Lu; Jiashuo Liu; Zihao Li; Yanyan Deng; Qixin Xiao; Jia Tian; Kanghong Zhan; Tianyi Li; Xiaochen Liu; Jason Ge; Chaoyang He; Kaixuan Huang; Lin Yang; Wenhao Huang; Mengdi Wang
>
> **摘要:** This paper introduces CryptoBench, the first expert-curated, dynamic benchmark designed to rigorously evaluate the real-world capabilities of Large Language Model (LLM) agents in the uniquely demanding and fast-paced cryptocurrency domain. Unlike general-purpose agent benchmarks for search and prediction, professional crypto analysis presents specific challenges: \emph{extreme time-sensitivity}, \emph{a highly adversarial information environment}, and the critical need to synthesize data from \emph{diverse, specialized sources}, such as on-chain intelligence platforms and real-time Decentralized Finance (DeFi) dashboards. CryptoBench thus serves as a much more challenging and valuable scenario for LLM agent assessment. To address these challenges, we constructed a live, dynamic benchmark featuring 50 questions per month, expertly designed by crypto-native professionals to mirror actual analyst workflows. These tasks are rigorously categorized within a four-quadrant system: Simple Retrieval, Complex Retrieval, Simple Prediction, and Complex Prediction. This granular categorization enables a precise assessment of an LLM agent's foundational data-gathering capabilities alongside its advanced analytical and forecasting skills. Our evaluation of ten LLMs, both directly and within an agentic framework, reveals a performance hierarchy and uncovers a failure mode. We observe a \textit{retrieval-prediction imbalance}, where many leading models, despite being proficient at data retrieval, demonstrate a pronounced weakness in tasks requiring predictive analysis. This highlights a problematic tendency for agents to appear factually grounded while lacking the deeper analytical capabilities to synthesize information.
>
---
#### [new 028] SUPERChem: A Multimodal Reasoning Benchmark in Chemistry
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出SUPERChem，一个面向化学领域的多模态推理基准，旨在解决现有评测任务过于简单、缺乏过程评估及与专家能力不匹配的问题。通过500个专家标注的复杂问题和解题路径，实现对模型推理质量的精细化评估，揭示了大模型在化学推理中的局限性。**

- **链接: [https://arxiv.org/pdf/2512.01274v1](https://arxiv.org/pdf/2512.01274v1)**

> **作者:** Zehua Zhao; Zhixian Huang; Junren Li; Siyu Lin; Junting Zhou; Fengqi Cao; Kun Zhou; Rui Ge; Tingting Long; Yuexiang Zhu; Yan Liu; Jie Zheng; Junnian Wei; Rong Zhu; Peng Zou; Wenyu Li; Zekai Cheng; Tian Ding; Yaxuan Wang; Yizhao Yan; Tingru Wei; Haowei Ming; Weijie Mao; Chen Sun; Yiming Liu; Zichen Wang; Zuo Zhang; Tong Yang; Hao Ma; Zhen Gao; Jian Pei
>
> **备注:** 35 pages, 11 figures, 5 tables
>
> **摘要:** Current benchmarks for evaluating the chemical reasoning capabilities of Large Language Models (LLMs) are limited by oversimplified tasks, lack of process-level evaluation, and misalignment with expert-level chemistry skills. To address these issues, we introduce SUPERChem, a benchmark of 500 expert-curated reasoning-intensive chemistry problems, covering diverse subfields and provided in both multimodal and text-only formats. Original content and an iterative curation pipeline eliminate flawed items and mitigate data contamination. Each problem is paired with an expert-authored solution path, enabling Reasoning Path Fidelity (RPF) scoring to evaluate reasoning quality beyond final-answer accuracy. Evaluations against a human baseline of 40.3% accuracy show that even the best-performing model, GPT-5 (High), reaches only 38.5%, followed closely by Gemini 2.5 Pro (37.9%) and DeepSeek-V3.1-Think (37.3%). SUPERChem elicits multi-step, multimodal reasoning, reveals model-dependent effects of visual information, and distinguishes high-fidelity reasoners from heuristic ones. By providing a challenging benchmark and a reliable evaluation framework, SUPERChem aims to facilitate the advancement of LLMs toward expert-level chemical intelligence. The dataset of the benchmark is available at https://huggingface.co/datasets/ZehuaZhao/SUPERChem.
>
---
#### [new 029] PromptBridge: Cross-Model Prompt Transfer for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型（LLM）在跨模型迁移时因提示敏感性导致的性能下降问题（模型漂移），提出PromptBridge框架。其通过少量对齐任务，利用反射式提示优化与跨模型映射，实现无需重新训练的高效提示迁移，显著提升多模型场景下的任务准确率与迁移效率。**

- **链接: [https://arxiv.org/pdf/2512.01420v1](https://arxiv.org/pdf/2512.01420v1)**

> **作者:** Yaxuan Wang; Quan Liu; Zhenting Wang; Zichao Li; Wei Wei; Yang Liu; Yujia Bao
>
> **摘要:** Large language models (LLMs) underpin applications in code generation, mathematical reasoning, and agent-based workflows. In practice, systems access LLMs via commercial APIs or open-source deployments, and the model landscape (e.g., GPT, Claude, Llama) evolves rapidly. This rapid evolution forces frequent model switches driven by capability, cost, deployment constraints, and privacy. Yet prompts are highly model-sensitive: reusing a prompt engineered for one model on another often yields substantially worse performance than a prompt optimized for the target model. We term this phenomenon Model Drifting. Through extensive empirical analysis across diverse LLM configurations, we show that model drifting is both common and severe. To address this challenge, we introduce PromptBridge, a training-free framework that preserves prompt effectiveness under model switches, enabling cross-model prompt transfer without costly per-task or per-model re-optimization. PromptBridge requires only a small set of alignment tasks for calibration. It first applies Model-Adaptive Reflective Prompt Evolution (MAP-RPE) to obtain task- and model-specific optimal prompts via iterative reflective refinement and quantitative evaluation. Using the resulting calibrated prompt pairs for the source and target models, PromptBridge learns a cross-model prompt mapping. At test time, i.e., for an unseen task, given a source-model prompt, this mapping directly produces an optimized prompt for the target model. Experiments in single-agent and multi-agent settings show that PromptBridge consistently improves downstream accuracy while reducing migration effort. The code will be available soon.
>
---
#### [new 030] G-KV: Decoding-Time KV Cache Eviction with Global Attention
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型推理中长序列导致的计算与内存瓶颈，提出G-KV方法，通过融合局部与历史注意力得分的全局评分机制，更精准评估令牌重要性，实现高效的KV缓存淘汰。结合后训练优化技术，提升模型在压缩缓存下的性能。**

- **链接: [https://arxiv.org/pdf/2512.00504v1](https://arxiv.org/pdf/2512.00504v1)**

> **作者:** Mengqi Liao; Lu Wang; Chaoyun Zhang; Zekai Shen; Xiaowei Mao; Si Qin; Qingwei Lin; Saravan Rajmohan; Dongmei Zhang; Huaiyu Wan
>
> **摘要:** Recent reasoning large language models (LLMs) excel in complex tasks but encounter significant computational and memory challenges due to long sequence lengths. KV cache compression has emerged as an effective approach to greatly enhance the efficiency of reasoning. However, existing methods often focus on prompt compression or token eviction with local attention score, overlooking the long-term importance of tokens. We propose G-KV, a KV cache eviction method that employs a global scoring mechanism, combining local and historical attention scores to more accurately assess token importance. Additionally, we introduce post-training techniques, including reinforcement learning and distillation, to optimize models for compressed KV cache settings. The code of this paper is available on: https://github.com/microsoft/G-KV.
>
---
#### [new 031] SCALE: Selective Resource Allocation for Overcoming Performance Bottlenecks in Mathematical Test-time Scaling
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型数学推理中的测试时计算扩展问题，提出SCALE框架。通过分解问题、评估难度并选择性分配资源（系统1处理简单任务，系统2处理复杂任务），实现对难点的精准投入。有效提升准确率（最高+13.75%）同时降低33%-53%计算成本，解决了均匀分配资源导致的效率瓶颈。**

- **链接: [https://arxiv.org/pdf/2512.00466v1](https://arxiv.org/pdf/2512.00466v1)**

> **作者:** Yang Xiao; Chunpu Xu; Ruifeng Yuan; Jiashuo Wang; Wenjie Li; Pengfei Liu
>
> **备注:** accepted by AAAI 2026
>
> **摘要:** Test-time compute scaling has emerged as a powerful paradigm for enhancing mathematical reasoning in large language models (LLMs) by allocating additional computational resources during inference. However, current methods employ uniform resource distribution across all reasoning sub-problems, creating fundamental bottlenecks where challenging sub-problems receive insufficient attention while routine operations consume disproportionate resources. This uniform allocation creates performance bottlenecks where additional computational resources yield diminishing returns. Inspired by dual-process theory, we propose \textbf{SCALE} (Selective Resource Allocation), a framework that selectively allocates computational resources based on sub-problem difficulty. SCALE operates through four stages: (1) problem decomposition into sequential reasoning sub-problems, (2) difficulty assessment of each sub-problem to distinguish between routine operations and computationally challenging sub-problems, (3) selective processing mode assignment between System 1 for simple sub-problems and System 2 for complex ones, and (4) sequential execution with context propagation. By concentrating resources on challenging sub-problems while processing routine operations efficiently, SCALE achieves substantial performance improvements with superior resource utilization. Extensive experiments demonstrate that SCALE significantly outperforms uniform scaling baselines, achieving accuracy improvements of up to 13.75 percentage points (57.50% to 71.25% on AIME25) while reducing computational costs by 33%-53%, representing a major advance in test-time scaling that addresses fundamental limitations of current approaches.
>
---
#### [new 032] How do we measure privacy in text? A survey of text anonymization metrics
- **分类: cs.CL**

- **简介: 该论文属于文本隐私保护任务，旨在解决文本匿名化后隐私评估标准不统一的问题。通过系统调研47篇论文，梳理六类隐私概念及其度量方法，分析其与法律标准及用户期望的契合度，提出更可靠、可比且合规的评估框架。**

- **链接: [https://arxiv.org/pdf/2512.01109v1](https://arxiv.org/pdf/2512.01109v1)**

> **作者:** Yaxuan Ren; Krithika Ramesh; Yaxing Yao; Anjalie Field
>
> **备注:** 13 pages, 1 figure, 1 table. To be published in Findings of the Association for Computational Linguistics (AACL-IJCNLP 2025). Related resources at: https://github.com/ryxGuo/privacy-metrics-survey
>
> **摘要:** In this work, we aim to clarify and reconcile metrics for evaluating privacy protection in text through a systematic survey. Although text anonymization is essential for enabling NLP research and model development in domains with sensitive data, evaluating whether anonymization methods sufficiently protect privacy remains an open challenge. In manually reviewing 47 papers that report privacy metrics, we identify and compare six distinct privacy notions, and analyze how the associated metrics capture different aspects of privacy risk. We then assess how well these notions align with legal privacy standards (HIPAA and GDPR), as well as user-centered expectations grounded in HCI studies. Our analysis offers practical guidance on navigating the landscape of privacy evaluation approaches further and highlights gaps in current practices. Ultimately, we aim to facilitate more robust, comparable, and legally aware privacy evaluations in text anonymization.
>
---
#### [new 033] ELR-1000: A Community-Generated Dataset for Endangered Indic Indigenous Languages
- **分类: cs.CL; cs.HC**

- **简介: 该论文提出ELR-1000，一个由社区贡献的多模态数据集，包含10种濒危印度本土语言的1060道传统食谱。针对低资源、文化特异性语言翻译难题，研究评估了大语言模型性能，并发现提供上下文信息可显著提升翻译质量。工作旨在推动濒危语言的自然语言处理技术发展，促进文化可持续性。**

- **链接: [https://arxiv.org/pdf/2512.01077v1](https://arxiv.org/pdf/2512.01077v1)**

> **作者:** Neha Joshi; Pamir Gogoi; Aasim Mirza; Aayush Jansari; Aditya Yadavalli; Ayushi Pandey; Arunima Shukla; Deepthi Sudharsan; Kalika Bali; Vivek Seshadri
>
> **备注:** Accepted at AACL 2025 (Main)
>
> **摘要:** We present a culturally-grounded multimodal dataset of 1,060 traditional recipes crowdsourced from rural communities across remote regions of Eastern India, spanning 10 endangered languages. These recipes, rich in linguistic and cultural nuance, were collected using a mobile interface designed for contributors with low digital literacy. Endangered Language Recipes (ELR)-1000 -- captures not only culinary practices but also the socio-cultural context embedded in indigenous food traditions. We evaluate the performance of several state-of-the-art large language models (LLMs) on translating these recipes into English and find the following: despite the models' capabilities, they struggle with low-resource, culturally-specific language. However, we observe that providing targeted context -- including background information about the languages, translation examples, and guidelines for cultural preservation -- leads to significant improvements in translation quality. Our results underscore the need for benchmarks that cater to underrepresented languages and domains to advance equitable and culturally-aware language technologies. As part of this work, we release the ELR-1000 dataset to the NLP community, hoping it motivates the development of language technologies for endangered languages.
>
---
#### [new 034] Latent Debate: A Surrogate Framework for Interpreting LLM Thinking
- **分类: cs.CL**

- **简介: 该论文提出“隐式辩论”（latent debate）框架，用于解释大语言模型（LLM）的内部推理过程。针对LLM思维不可见及幻觉问题，该方法通过分析单次推理中模型内部隐含的支持与反驳信号，构建可解释的结构化代理模型，实现对预测的忠实模拟，并为幻觉检测提供有效基准。**

- **链接: [https://arxiv.org/pdf/2512.01909v1](https://arxiv.org/pdf/2512.01909v1)**

> **作者:** Lihu Chen; Xiang Yin; Francesca Toni
>
> **备注:** Preprint
>
> **摘要:** Understanding the internal thinking process of Large Language Models (LLMs) and the cause of hallucinations remains a key challenge. To this end, we introduce latent debate, a novel framework for interpreting model predictions through the lens of implicit internal arguments. Unlike the current work of self-consistency and multi-agent debate, which relies on explicit debates among multiple answers or multiple models, latent debate captures the hidden supporting and attacking signals that arise within a single model during a single inference. We first present a model- and task-agnostic conceptual framework, and then instantiate it symbolically to approximate the thinking process of LLMs on True/False prediction tasks. Empirical studies demonstrate that latent debate is a faithful structured surrogate model that has highly consistent predictions with the original LLM. Beyond interpretability, we demonstrate that latent debate provides a strong baseline for hallucination detection. Further analysis reveals strong correlations between hallucinations and debate patterns, such as a high degree of latent debates in the middle layers is linked to a higher risk of hallucinations. These findings position latent debate as a potential framework for understanding internal mechanisms of LLMs, especially for scenarios where internal (dis)agreements appear during the inference steps.
>
---
#### [new 035] Prism: A Minimal Compositional Metalanguage for Specifying Agent Behavior
- **分类: cs.CL**

- **简介: 论文提出Prism，一种用于指定工具使用型软件代理行为的极简组合型元语言。针对传统方法中控制结构冗杂、可维护性差的问题，通过固定核心语法与可扩展领域语法分离，实现自然语言规则到可检查、可执行策略的映射，提升代理行为的透明性与安全性。**

- **链接: [https://arxiv.org/pdf/2512.00611v1](https://arxiv.org/pdf/2512.00611v1)**

> **作者:** Franck Binard; Vanja Kljajevic
>
> **摘要:** Prism is a small, compositional metalanguage for specifying the behaviour of tool-using software agents. Rather than introducing ad hoc control constructs, Prism is built around a fixed core context, Core1, which provides a minimal background grammar of categories numbers, strings, user prompts, tools together with abstract combinators for booleans, predicates, pairs, and lists. Agent policies are written as ordinary expressions using a single abstraction operator so that conditionals appear as selections between alternatives instead of imperative if-else blocks. Domains extend the core by defining their own context-mini-grammars that introduce new categories, predicates, and external tools while reusing the same compositional machinery. We illustrate this with worked examples from thermostat control, home security, e-commerce recommendation, and medical monitoring, showing how natural language decision rules can be mapped to inspectable, executable policies. From a linguistic perspective, Prism enforces a clear separation between a reusable grammar-like core and domain specific lexicons and treats tools as bridges between internal policy representations and the external world. From an engineering perspective, it offers a compact interface language for agent control, making the space of possible actions explicit and amenable to analysis, verification, and safety constraints.
>
---
#### [new 036] Mitigating Hallucinations in Zero-Shot Scientific Summarisation: A Pilot Study
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究零样本科学摘要中的幻觉问题，旨在通过提示工程（PE）方法减少大语言模型生成内容与原文的不一致。针对8篇酵母生物技术论文，测试了7种提示策略，评估6项指标，发现上下文重复和随机添加能显著提升摘要与原文的词汇对齐度，验证了提示工程在缓解幻觉方面的潜力。**

- **链接: [https://arxiv.org/pdf/2512.00931v1](https://arxiv.org/pdf/2512.00931v1)**

> **作者:** Imane Jaaouine; Ross D. King
>
> **摘要:** Large language models (LLMs) produce context inconsistency hallucinations, which are LLM generated outputs that are misaligned with the user prompt. This research project investigates whether prompt engineering (PE) methods can mitigate context inconsistency hallucinations in zero-shot LLM summarisation of scientific texts, where zero-shot indicates that the LLM relies purely on its pre-training data. Across eight yeast biotechnology research paper abstracts, six instruction-tuned LLMs were prompted with seven methods: a base- line prompt, two levels of increasing instruction complexity (PE-1 and PE-2), two levels of context repetition (CR-K1 and CR-K2), and two levels of random addition (RA-K1 and RA-K2). Context repetition involved the identification and repetition of K key sentences from the abstract, whereas random addition involved the repetition of K randomly selected sentences from the abstract, where K is 1 or 2. A total of 336 LLM-generated summaries were evaluated using six metrics: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore, METEOR, and cosine similarity, which were used to compute the lexical and semantic alignment be- tween the summaries and the abstracts. Four hypotheses on the effects of prompt methods on summary alignment with the reference text were tested. Statistical analysis on 3744 collected datapoints was performed using bias-corrected and accelerated (BCa) bootstrap confidence intervals and Wilcoxon signed-rank tests with Bonferroni-Holm correction. The results demonstrated that CR and RA significantly improve the lexical alignment of LLM-generated summaries with the abstracts. These findings indicate that prompt engineering has the potential to impact hallucinations in zero-shot scientific summarisation tasks.
>
---
#### [new 037] Multilingual Conversational AI for Financial Assistance: Bridging Language Barriers in Indian FinTech
- **分类: cs.CL**

- **简介: 该论文针对印度金融普惠中的语言障碍问题，提出一种支持多语言及混合语（如Hinglish）的对话式AI系统。采用多智能体架构，实现语言识别、功能管理与多语言响应生成，显著提升用户参与度并保持低延迟，推动新兴市场数字金融服务的包容性发展。**

- **链接: [https://arxiv.org/pdf/2512.01439v1](https://arxiv.org/pdf/2512.01439v1)**

> **作者:** Bharatdeep Hazarika; Arya Suneesh; Prasanna Devadiga; Pawan Kumar Rajpoot; Anshuman B Suresh; Ahmed Ifthaquar Hussain
>
> **摘要:** India's linguistic diversity presents both opportunities and challenges for fintech platforms. While the country has 31 major languages and over 100 minor ones, only 10\% of the population understands English, creating barriers to financial inclusion. We present a multilingual conversational AI system for a financial assistance use case that supports code-mixed languages like Hinglish, enabling natural interactions for India's diverse user base. Our system employs a multi-agent architecture with language classification, function management, and multilingual response generation. Through comparative analysis of multiple language models and real-world deployment, we demonstrate significant improvements in user engagement while maintaining low latency overhead (4-8\%). This work contributes to bridging the language gap in digital financial services for emerging markets.
>
---
#### [new 038] Language Diversity: Evaluating Language Usage and AI Performance on African Languages in Digital Spaces
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的语言检测任务，旨在解决非洲语言在数字空间中数据稀缺与检测不准的问题。研究对比了Reddit和新闻媒体数据，发现新闻数据更适合作为训练语料，提出应构建能处理混合语言的模型以提升非洲语言识别性能。**

- **链接: [https://arxiv.org/pdf/2512.01557v1](https://arxiv.org/pdf/2512.01557v1)**

> **作者:** Edward Ajayi; Eudoxie Umwari; Mawuli Deku; Prosper Singadi; Jules Udahemuka; Bekalu Tadele; Chukuemeka Edeh
>
> **摘要:** This study examines the digital representation of African languages and the challenges this presents for current language detection tools. We evaluate their performance on Yoruba, Kinyarwanda, and Amharic. While these languages are spoken by millions, their online usage on conversational platforms is often sparse, heavily influenced by English, and not representative of the authentic, monolingual conversations prevalent among native speakers. This lack of readily available authentic data online creates a challenge of scarcity of conversational data for training language models. To investigate this, data was collected from subreddits and local news sources for each language. The analysis showed a stark contrast between the two sources. Reddit data was minimal and characterized by heavy code-switching. Conversely, local news media offered a robust source of clean, monolingual language data, which also prompted more user engagement in the local language on the news publishers social media pages. Language detection models, including the specialized AfroLID and a general LLM, performed with near-perfect accuracy on the clean news data but struggled with the code-switched Reddit posts. The study concludes that professionally curated news content is a more reliable and effective source for training context-rich AI models for African languages than data from conversational platforms. It also highlights the need for future models that can process clean and code-switched text to improve the detection accuracy for African languages.
>
---
#### [new 039] Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对大语言模型训练中NVFP4量化导致的训练发散与性能下降问题，提出Four Over Six（4/6）方法，通过为每块值评估两个缩放因子，优化近最大值的表示精度。该方法提升量化准确性，防止训练发散，且可在NVIDIA Blackwell GPU上高效实现，适用于多种量化场景。**

- **链接: [https://arxiv.org/pdf/2512.02010v1](https://arxiv.org/pdf/2512.02010v1)**

> **作者:** Jack Cook; Junxian Guo; Guangxuan Xiao; Yujun Lin; Song Han
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** As large language models have grown larger, low-precision numerical formats such as NVFP4 have become increasingly popular due to the speed and memory benefits they provide. However, to accelerate computation with NVFP4, all matrix multiplication operands--weights and activations in the forward pass, and weights, activations, and gradients in the backward pass--must be quantized to NVFP4, often leading to divergence during training and performance degradation during inference. NVFP4 by evaluating multiple potential scale factors for each block of values. To address this issue, in this work we introduce Four Over Six (4/6), a modification to the NVFP4 quantization algorithm that evaluates two potential scale factors for each block of values. Unlike integer formats, floating-point formats such as FP4 have the most quantization error on near-maximal values in each block, which we find to be primarily responsible for downstream performance degradation. We find that for some blocks, scaling to smaller FP4 values makes the distribution of representable values more uniform, improving representation of near-maximal values. Importantly, 4/6 can be implemented efficiently on NVIDIA Blackwell GPUs, making it viable to use while training LLMs with NVFP4. In pre-training experiments with transformer and hybrid model architectures, we find that 4/6 prevents divergence in several cases, bringing training loss significantly closer to BF16 compared to models trained with current state-of-the-art NVFP4 training recipes. We also find that 4/6 can be easily incorporated into many different post-training quantization methods and generally improves downstream accuracy. We hope this inspires future work in training and deploying models with NVFP4.
>
---
#### [new 040] Generalist Large Language Models Outperform Clinical Tools on Medical Benchmarks
- **分类: cs.CL**

- **简介: 该论文属于医疗AI评估任务，旨在解决临床AI工具缺乏独立量化评估的问题。研究对比了两款临床工具与三款通用大模型在医学知识与临床对齐任务上的表现，发现通用模型整体更优，揭示临床工具存在多项缺陷，呼吁加强透明评估。**

- **链接: [https://arxiv.org/pdf/2512.01191v1](https://arxiv.org/pdf/2512.01191v1)**

> **作者:** Krithik Vishwanath; Mrigayu Ghosh; Anton Alyakin; Daniel Alexander Alber; Yindalon Aphinyanaphongs; Eric Karl Oermann
>
> **备注:** 17 pages, 4 figures (2 regular, 2 supplemental)
>
> **摘要:** Specialized clinical AI assistants are rapidly entering medical practice, often framed as safer or more reliable than general-purpose large language models (LLMs). Yet, unlike frontier models, these clinical tools are rarely subjected to independent, quantitative evaluation, creating a critical evidence gap despite their growing influence on diagnosis, triage, and guideline interpretation. We assessed two widely deployed clinical AI systems (OpenEvidence and UpToDate Expert AI) against three state-of-the-art generalist LLMs (GPT-5, Gemini 3 Pro, and Claude Sonnet 4.5) using a 1,000-item mini-benchmark combining MedQA (medical knowledge) and HealthBench (clinician-alignment) tasks. Generalist models consistently outperformed clinical tools, with GPT-5 achieving the highest scores, while OpenEvidence and UpToDate demonstrated deficits in completeness, communication quality, context awareness, and systems-based safety reasoning. These findings reveal that tools marketed for clinical decision support may often lag behind frontier LLMs, underscoring the urgent need for transparent, independent evaluation before deployment in patient-facing workflows.
>
---
#### [new 041] MAC-SLU: Multi-Intent Automotive Cabin Spoken Language Understanding Benchmark
- **分类: cs.CL; cs.MM**

- **简介: 该论文提出MAC-SLU，一个面向车载舱内多意图语音理解的基准数据集，解决现有SLU数据集多样性不足、缺乏LLM/LALM统一评测的问题。通过构建复杂真实场景数据，对主流模型进行评测，验证了微调优于上下文学习，且端到端音频语言模型可避免误差传播。**

- **链接: [https://arxiv.org/pdf/2512.01603v1](https://arxiv.org/pdf/2512.01603v1)**

> **作者:** Yuezhang Peng; Chonghao Cai; Ziang Liu; Shuai Fan; Sheng Jiang; Hua Xu; Yuxin Liu; Qiguang Chen; Kele Xu; Yao Li; Sheng Wang; Libo Qin; Xie Chen
>
> **摘要:** Spoken Language Understanding (SLU), which aims to extract user semantics to execute downstream tasks, is a crucial component of task-oriented dialog systems. Existing SLU datasets generally lack sufficient diversity and complexity, and there is an absence of a unified benchmark for the latest Large Language Models (LLMs) and Large Audio Language Models (LALMs). This work introduces MAC-SLU, a novel Multi-Intent Automotive Cabin Spoken Language Understanding Dataset, which increases the difficulty of the SLU task by incorporating authentic and complex multi-intent data. Based on MAC-SLU, we conducted a comprehensive benchmark of leading open-source LLMs and LALMs, covering methods like in-context learning, supervised fine-tuning (SFT), and end-to-end (E2E) and pipeline paradigms. Our experiments show that while LLMs and LALMs have the potential to complete SLU tasks through in-context learning, their performance still lags significantly behind SFT. Meanwhile, E2E LALMs demonstrate performance comparable to pipeline approaches and effectively avoid error propagation from speech recognition. Code\footnote{https://github.com/Gatsby-web/MAC\_SLU} and datasets\footnote{huggingface.co/datasets/Gatsby1984/MAC\_SLU} are released publicly.
>
---
#### [new 042] IndicParam: Benchmark to evaluate LLMs on low-resource Indic Languages
- **分类: cs.CL**

- **简介: 该论文提出IndicParam，一个面向11种低资源与极低资源印地语族语言的多选题基准测试。旨在评估大语言模型在这些语言上的表现，解决跨语言迁移能力不足的问题。研究包含超1.3万道题目，涵盖知识与语言两类任务，并测试多种题型，揭示现有模型在低资源语言上的显著局限性。**

- **链接: [https://arxiv.org/pdf/2512.00333v1](https://arxiv.org/pdf/2512.00333v1)**

> **作者:** Ayush Maheshwari; Kaushal Sharma; Vivek Patel; Aditya Maheshwari
>
> **摘要:** While large language models excel on high-resource multilingual tasks, low- and extremely low-resource Indic languages remain severely under-evaluated. We present IndicParam, a human-curated benchmark of over 13,000 multiple-choice questions covering 11 such languages (Nepali, Gujarati, Marathi, Odia as low-resource; Dogri, Maithili, Rajasthani, Sanskrit, Bodo, Santali, Konkani as extremely low-resource) plus Sanskrit-English code-mixed set. We evaluated 19 LLMs, both proprietary and open-weights, which reveals that even the top-performing GPT-5 reaches only 45.0% average accuracy, followed by DeepSeek-3.2 (43.1) and Claude-4.5 (42.7). We additionally label each question as knowledge-oriented or purely linguistic to discriminate factual recall from grammatical proficiency. Further, we assess the ability of LLMs to handle diverse question formats-such as list-based matching, assertion-reason pairs, and sequence ordering-alongside conventional multiple-choice questions. IndicParam provides insights into limitations of cross-lingual transfer and establishes a challenging benchmark for Indic languages. The dataset is available at https://huggingface.co/datasets/bharatgenai/IndicParam. Scripts to run benchmark are present at https://github.com/ayushbits/IndicParam.
>
---
#### [new 043] OmniFusion: Simultaneous Multilingual Multimodal Translations via Modular Fusion
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出OmniFusion，一种端到端的多模态多语言翻译模型，解决传统语音翻译延迟高、无法利用视觉信息的问题。通过融合多模态基础模型与翻译大模型，实现语音、图文联合翻译，显著降低1秒延迟并提升翻译质量。**

- **链接: [https://arxiv.org/pdf/2512.00234v1](https://arxiv.org/pdf/2512.00234v1)**

> **作者:** Sai Koneru; Matthias Huck; Jan Niehues
>
> **备注:** Preprint for ACL 2026
>
> **摘要:** There has been significant progress in open-source text-only translation large language models (LLMs) with better language coverage and quality. However, these models can be only used in cascaded pipelines for speech translation (ST), performing automatic speech recognition first followed by translation. This introduces additional latency, which is particularly critical in simultaneous ST (SimulST), and prevents the model from exploiting multimodal context, such as images, which can aid disambiguation. Pretrained multimodal foundation models (MMFMs) already possess strong perception and reasoning capabilities across multiple modalities, but generally lack the multilingual coverage and specialized translation performance of dedicated translation LLMs. To build an effective multimodal translation system, we propose an end-to-end approach that fuses MMFMs with translation LLMs. We introduce a novel fusion strategy that connects hidden states from multiple layers of a pretrained MMFM to a translation LLM, enabling joint end-to-end training. The resulting model, OmniFusion, built on Omni 2.5-7B as the MMFM and SeedX PPO-7B as the translation LLM, can perform speech-to-text, speech-and-image-to-text, and text-and-image-to-text translation. Experiments demonstrate that OmniFusion effectively leverages both audio and visual inputs, achieves a 1-second latency reduction in SimulST compared to cascaded pipelines and also improves the overall translation quality\footnote{Code is available at https://github.com/saikoneru/OmniFusion}.
>
---
#### [new 044] Towards Corpus-Grounded Agentic LLMs for Multilingual Grammatical Analysis
- **分类: cs.CL**

- **简介: 该论文研究多语言语法分析任务，旨在解决标注语料系统分析耗时耗力的问题。提出基于代理型大模型的框架，通过自然语言理解、代码生成与数据推理，实现对通用依存语料库的自动化语法分析，验证了其在13类词序特征、170余语言上的可行性与有效性。**

- **链接: [https://arxiv.org/pdf/2512.00214v1](https://arxiv.org/pdf/2512.00214v1)**

> **作者:** Matej Klemen; Tjaša Arčon; Luka Terčon; Marko Robnik-Šikonja; Kaja Dobrovoljc
>
> **备注:** Pre-print, submission under review
>
> **摘要:** Empirical grammar research has become increasingly data-driven, but the systematic analysis of annotated corpora still requires substantial methodological and technical effort. We explore how agentic large language models (LLMs) can streamline this process by reasoning over annotated corpora and producing interpretable, data-grounded answers to linguistic questions. We introduce an agentic framework for corpus-grounded grammatical analysis that integrates concepts such as natural-language task interpretation, code generation, and data-driven reasoning. As a proof of concept, we apply it to Universal Dependencies (UD) corpora, testing it on multilingual grammatical tasks inspired by the World Atlas of Language Structures (WALS). The evaluation spans 13 word-order features and over 170 languages, assessing system performance across three complementary dimensions - dominant-order accuracy, order-coverage completeness, and distributional fidelity - which reflect how well the system generalizes, identifies, and quantifies word-order variations. The results demonstrate the feasibility of combining LLM reasoning with structured linguistic data, offering a first step toward interpretable, scalable automation of corpus-based grammatical inquiry.
>
---
#### [new 045] Beware of Reasoning Overconfidence: Pitfalls in the Reasoning Process for Multi-solution Tasks
- **分类: cs.CL**

- **简介: 该论文研究多解任务中大模型的推理过自信问题，指出其因过早收敛导致答案不完整。提出MuSoBench基准，对比短/长思维链，发现长思维链通过迭代反思缓解过自信，并基于注意力熵分析支持认知僵化假说，强调需以探索全面性替代单一正确率评估。**

- **链接: [https://arxiv.org/pdf/2512.01725v1](https://arxiv.org/pdf/2512.01725v1)**

> **作者:** Jiannan Guan; Qiguang Chen; Libo Qin; Dengyun Peng; Jinhao Liu; Liangyu Huo; Jian Xie; Wanxiang Che
>
> **摘要:** Large Language Models (LLMs) excel in reasoning tasks requiring a single correct answer, but they perform poorly in multi-solution tasks that require generating comprehensive and diverse answers. We attribute this limitation to \textbf{reasoning overconfidence}: a tendency to express undue certainty in an incomplete solution set. To examine the effect, we introduce \textit{MuSoBench}, a benchmark of multi-solution problems. Experiments show that the conventional short chain-of-thought (Short-CoT) prompting paradigm exhibits pronounced overconfidence, whereas the emerging long chain-of-thought (Long-CoT) approach mitigates it through iterative exploration and self-reflection. We further characterise observable behaviours and influential factors. To probe the underlying cause, we propose the \textbf{cognitive-rigidity hypothesis}, which posits that overconfidence arises when the reasoning process prematurely converges on a narrow set of thought paths. An attention-entropy analysis offers preliminary support for this view. These findings provide tools for assessing the completeness of LLM reasoning and highlight the need to move evaluation beyond single-answer accuracy toward comprehensive exploration.
>
---
#### [new 046] When Safety Blocks Sense: Measuring Semantic Confusion in LLM Refusals
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型安全对齐中的误拒问题，提出“语义混淆”概念，构建包含10,000个控制同义句的ParaGuard数据集，设计三类词元级度量指标，揭示模型在相近语义下拒绝不一致的现象，实现对安全拒答合理性与稳定性的精准评估与优化。**

- **链接: [https://arxiv.org/pdf/2512.01037v1](https://arxiv.org/pdf/2512.01037v1)**

> **作者:** Riad Ahmed Anonto; Md Labid Al Nahiyan; Md Tanvir Hassan; Ch. Md. Rakin Haider
>
> **摘要:** Safety-aligned language models often refuse prompts that are actually harmless. Current evaluations mostly report global rates such as false rejection or compliance. These scores treat each prompt alone and miss local inconsistency, where a model accepts one phrasing of an intent but rejects a close paraphrase. This gap limits diagnosis and tuning. We introduce "semantic confusion," a failure mode that captures such local inconsistency, and a framework to measure it. We build ParaGuard, a 10k-prompt corpus of controlled paraphrase clusters that hold intent fixed while varying surface form. We then propose three model-agnostic metrics at the token level: Confusion Index, Confusion Rate, and Confusion Depth. These metrics compare each refusal to its nearest accepted neighbors and use token embeddings, next-token probabilities, and perplexity signals. Experiments across diverse model families and deployment guards show that global false-rejection rate hides critical structure. Our metrics reveal globally unstable boundaries in some settings, localized pockets of inconsistency in others, and cases where stricter refusal does not increase inconsistency. We also show how confusion-aware auditing separates how often a system refuses from how sensibly it refuses. This gives developers a practical signal to reduce false refusals while preserving safety.
>
---
#### [new 047] CACARA: Cross-Modal Alignment Leveraging a Text-Centric Approach for Cost-Effective Multimodal and Multilingual Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CACARA模型，解决多模态与多语言学习中资源消耗高的问题。通过文本中心对齐机制，实现新模态的低成本集成与100+语言的涌现式支持，无需全量重训或显式多语言预训练，显著提升跨模态检索性能，同时保持低训练成本。**

- **链接: [https://arxiv.org/pdf/2512.00496v1](https://arxiv.org/pdf/2512.00496v1)**

> **作者:** Diego A. B. Moreira; Alef I. Ferreira; Jhessica Silva; Gabriel O. dos Santos; Gustavo Bonil; João Gondim; Marina dos Santos; Helena Maia; Simone Hashiguti; Nádia da Silva; Carolina Scarton; Helio Pedrini; Sandra Avila
>
> **备注:** 25 pages, 12 tables, 5 figures
>
> **摘要:** As deep learning models evolve, new applications and challenges are rapidly emerging. Tasks that once relied on a single modality, such as text, images, or audio, are now enriched by seamless interactions between multimodal data. These connections bridge information gaps: an image can visually materialize a text, while audio can add context to an image. Researchers have developed numerous multimodal models, but most rely on resource-intensive training across multiple modalities. Similarly, extending these models to new languages often follows the same resource-heavy training strategy. In this work, we propose a multimodal and multilingual architecture, CACARA, trained through emergent alignment learning, enabling the seamless integration of new modalities into an existing bimodal/multimodal model without requiring full retraining. This work breaks new ground by demonstrating that this emergent alignment paradigm can unlock multilingual capabilities from monolingual training. By fine-tuning the newly incorporated modality only on data aligned with the English language, our model develops support for over 100 languages without explicit multilingual pretraining or tuning of the text encoder. Such emergent multimodal and multilingual properties are gained efficiently, preserving previously learned knowledge at a training cost comparable to that of a monolingual model. Our strategy achieves up to a 14.24 percentage points improvement in R@1 audio-to-text retrieval, outperforming state-of-the-art multimodal models -- all without the heavy computational cost of retraining across every modality and language.
>
---
#### [new 048] Comparative Analysis of 47 Context-Based Question Answer Models Across 8 Diverse Datasets
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于上下文问答任务，旨在评估47个预训练模型在8个数据集上的零样本性能。通过基准测试，发现Electra-large模型表现最优，且模型性能受上下文长度、答案长度和复杂度影响。研究还尝试用遗传算法融合多模型输出以提升准确率。**

- **链接: [https://arxiv.org/pdf/2512.00323v1](https://arxiv.org/pdf/2512.00323v1)**

> **作者:** Muhammad Muneeb; David B. Ascher; Ahsan Baidar Bakht
>
> **摘要:** Context-based question answering (CBQA) models provide more accurate and relevant answers by considering the contextual information. They effectively extract specific information given a context, making them functional in various applications involving user support, information retrieval, and educational platforms. In this manuscript, we benchmarked the performance of 47 CBQA models from Hugging Face on eight different datasets. This study aims to identify the best-performing model across diverse datasets without additional fine-tuning. It is valuable for practical applications where the need to retrain models for specific datasets is minimized, streamlining the implementation of these models in various contexts. The best-performing models were trained on the SQuAD v2 or SQuAD v1 datasets. The best-performing model was ahotrod/electra_large_discriminator_squad2_512, which yielded 43\% accuracy across all datasets. We observed that the computation time of all models depends on the context length and the model size. The model's performance usually decreases with an increase in the answer length. Moreover, the model's performance depends on the context complexity. We also used the Genetic algorithm to improve the overall accuracy by integrating responses from other models. ahotrod/electra_large_discriminator_squad2_512 generated the best results for bioasq10b-factoid (65.92\%), biomedical\_cpgQA (96.45\%), QuAC (11.13\%), and Question Answer Dataset (41.6\%). Bert-large-uncased-whole-word-masking-finetuned-squad achieved an accuracy of 82\% on the IELTS dataset.
>
---
#### [new 049] Learning the Boundary of Solvability: Aligning LLMs to Detect Unsolvable Problems
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型在面对不可解问题时易产生幻觉和过度自信的问题，提出UnsolvableQA数据集与UnsolvableRL强化学习框架。通过构建含矛盾的逻辑与数学题对，训练模型识别本质不可解问题，提升可靠性。实验证明其能精准检测不可解性并防止能力坍塌。**

- **链接: [https://arxiv.org/pdf/2512.01661v1](https://arxiv.org/pdf/2512.01661v1)**

> **作者:** Dengyun Peng; Qiguang Chen; Bofei Liu; Jiannan Guan; Libo Qin; Zheng Yan; Jinhao Liu; Jianshu Zhang; Wanxiang Che
>
> **备注:** preprint
>
> **摘要:** Ensuring LLM reliability requires not only solving complex problems but also recognizing when a problem is unsolvable. Current models often struggle to distinguish objective unsolvability (inherent contradictions in the problem) from subjective capability limitations (problems beyond the model's competence), which leads to hallucinations and overconfidence. To address this, we propose UnsolvableQA and UnsolvableRL to solve feasible problems, detect inherent contradictions, and prudently refuse tasks beyond capability. Specifically, we construct UnsolvableQA, a dataset of paired solvable and unsolvable instances derived via a dual-track methodology: programmatic generation for logic puzzles and a novel "Reverse Construction" method that injects contradictions into valid reasoning chains for mathematics. Building on this dataset, we introduce UnsolvableRL, a reinforcement learning framework with three reward components jointly accounting for accuracy, unsolvability, and difficulty. Empirical results show that our approach achieves near-perfect unsolvability detection while also improving accuracy on solvable tasks. Crucially, we identify Capability Collapse, demonstrating that explicit exposure to unsolvable data is indispensable for preventing models from becoming systematically overconfident. Our code and data are available at https://github.com/sfasfaffa/unsolvableQA.
>
---
#### [new 050] Developing a Comprehensive Framework for Sentiment Analysis in Turkish
- **分类: cs.CL**

- **简介: 该论文针对土耳其语和英语的情感分析任务，提出综合性框架。通过融合多类特征、构建领域词典、进行形态极性分析、设计新型神经网络与词嵌入，并改进上下文建模，显著提升分类效果，解决了低资源语言情感分析难题。**

- **链接: [https://arxiv.org/pdf/2512.00515v1](https://arxiv.org/pdf/2512.00515v1)**

> **作者:** Cem Rifki Aydin
>
> **备注:** Ph.D. Thesis, Bogazici University, 2020
>
> **摘要:** In this thesis, we developed a comprehensive framework for sentiment analysis that takes its many aspects into account mainly for Turkish. We have also proposed several approaches specific to sentiment analysis in English only. We have accordingly made five major and three minor contributions. We generated a novel and effective feature set by combining unsupervised, semi-supervised, and supervised metrics. We then fed them as input into classical machine learning methods, and outperformed neural network models for datasets of different genres in both Turkish and English. We created a polarity lexicon with a semi-supervised domain-specific method, which has been the first approach applied for corpora in Turkish. We performed a fine morphological analysis for the sentiment classification task in Turkish by determining the polarities of morphemes. This can be adapted to other morphologically-rich or agglutinative languages as well. We have built a novel neural network architecture, which combines recurrent and recursive neural network models for English. We built novel word embeddings that exploit sentiment, syntactic, semantic, and lexical characteristics for both Turkish and English. We also redefined context windows as subclauses in modelling word representations in English. This can also be applied to other linguistic fields and natural language processing tasks. We have achieved state-of-the-art and significant results for all these original approaches. Our minor contributions include methods related to aspect-based sentiment in Turkish, parameter redefinition in the semi-supervised approach, and aspect term extraction techniques for English. This thesis can be considered the most detailed and comprehensive study made on sentiment analysis in Turkish as of July, 2020. Our work has also contributed to the opinion classification problem in English.
>
---
#### [new 051] A Taxonomy of Errors in English as she is spoke: Toward an AI-Based Method of Error Analysis for EFL Writing Instruction
- **分类: cs.CL**

- **简介: 该论文针对EFL写作中错误分析的自动化需求，提出基于大语言模型的错误分类系统。通过构建融合语言学理论的多层次错误分类体系，实现对拼写、语法、标点等错误的精准识别与反馈。研究验证了AI在复杂文本分析中的潜力，但指出其在语境理解与类别扩展方面仍需改进。**

- **链接: [https://arxiv.org/pdf/2512.00392v1](https://arxiv.org/pdf/2512.00392v1)**

> **作者:** Damian Heywood; Joseph Andrew Carrier; Kyu-Hong Hwang
>
> **备注:** Metadata at "Replication Data for: A Taxonomy of Errors in English as she is spoke: An AI-Based System for Error Analysis for EFL Writing Instruction", https://doi.org/10.7910/DVN/N5O7C4, Harvard Dataverse, V1
>
> **摘要:** This study describes the development of an AI-assisted error analysis system designed to identify, categorize, and correct writing errors in English. Utilizing Large Language Models (LLMs) like Claude 3.5 Sonnet and DeepSeek R1, the system employs a detailed taxonomy grounded in linguistic theories from Corder (1967), Richards (1971), and James (1998). Errors are classified at both word and sentence levels, covering spelling, grammar, and punctuation. Implemented through Python-coded API calls, the system provides granular feedback beyond traditional rubric-based assessments. Initial testing on isolated errors refined the taxonomy, addressing challenges like overlapping categories. Final testing used "English as she is spoke" by Jose da Fonseca (1855), a text rich with authentic linguistic errors, to evaluate the system's capacity for handling complex, multi-layered analysis. The AI successfully identified diverse error types but showed limitations in contextual understanding and occasionally generated new error categories when encountering uncoded errors. This research demonstrates AI's potential to transform EFL instruction by automating detailed error analysis and feedback. While promising, further development is needed to improve contextual accuracy and expand the taxonomy to stylistic and discourse-level errors.
>
---
#### [new 052] Catch Me If You Can: How Smaller Reasoning Models Pretend to Reason with Mathematical Fidelity
- **分类: cs.CL**

- **简介: 该论文针对大模型数学推理评估中仅依赖准确率导致的逻辑缺陷被掩盖问题，提出四维诊断框架（一致性、传递性、反事实敏感性、扰动鲁棒性），揭示小模型虽有高准确率却依赖模式匹配而非真实推理。研究以Qwen3-0.6B为例，暴露其推理脆弱性，框架可推广至不同模型，推动向可验证推理评估演进。**

- **链接: [https://arxiv.org/pdf/2512.00552v1](https://arxiv.org/pdf/2512.00552v1)**

> **作者:** Subramanyam Sahoo; Vinija Jain; Saanidhya Vats; Siddharth Mohapatra; Rui Min; Aman Chadha; Divya Chaudhary
>
> **备注:** 8 pages, 5 figures. A preprint. Initial Work
>
> **摘要:** Current evaluation of mathematical reasoning in language models relies primarily on answer accuracy, potentially masking fundamental failures in logical computation. We introduce a diagnostic framework that distinguishes genuine mathematical reasoning from superficial pattern matching through four complementary axes: forward-backward consistency, transitivity coverage, counterfactual sensitivity, and perturbation robustness. Through a case study applying this framework to Qwen3-0.6B on the MenatQA dataset, we reveal a striking disconnect between surface performance and reasoning fidelity. While the model achieves reasonable answer accuracy (70%+), it demonstrates poor backward consistency (15%), limited transitivity coverage (32.2%), and brittle sensitivity to perturbations. Our diagnostics expose reasoning failures invisible to traditional accuracy metrics, suggesting that this small model relies heavily on pattern matching rather than genuine logical computation. While our empirical findings are based on a single 600M-parameter model, the diagnostic framework itself is model-agnostic and generalizable. We release our evaluation protocols to enable the research community to assess reasoning fidelity across different model scales and architectures, moving beyond surface-level accuracy toward verifiable mathematical reasoning.
>
---
#### [new 053] Sycophancy Claims about Language Models: The Missing Human-in-the-Loop
- **分类: cs.CL; cs.CY**

- **简介: 该论文研究大语言模型（LLM）中的奉承性回应现象，指出当前研究在测量方法上存在挑战，且缺乏对人类感知的评估。论文识别出五种核心操作化定义，强调应引入“人机协同”视角，以更准确区分奉承行为与AI对齐中的其他概念，并为未来研究提供可行建议。**

- **链接: [https://arxiv.org/pdf/2512.00656v1](https://arxiv.org/pdf/2512.00656v1)**

> **作者:** Jan Batzner; Volker Stocker; Stefan Schmid; Gjergji Kasneci
>
> **备注:** NeurIPS 2025 Workshop on LLM Evaluation and ICLR 2025 Workshop on Bi-Directional Human-AI Alignment
>
> **摘要:** Sycophantic response patterns in Large Language Models (LLMs) have been increasingly claimed in the literature. We review methodological challenges in measuring LLM sycophancy and identify five core operationalizations. Despite sycophancy being inherently human-centric, current research does not evaluate human perception. Our analysis highlights the difficulties in distinguishing sycophantic responses from related concepts in AI alignment and offers actionable recommendations for future research.
>
---
#### [new 054] Evidence-Guided Schema Normalization for Temporal Tabular Reasoning
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文针对时序半结构化表格的问答任务，提出基于SQL的三步法：从维基百科信息框生成3NF模式、生成查询并执行。研究表明，模式设计质量比模型规模更影响准确率，提出三项证据原则，最优配置达80.39 EM，提升16.8%。**

- **链接: [https://arxiv.org/pdf/2512.00329v1](https://arxiv.org/pdf/2512.00329v1)**

> **作者:** Ashish Thanga; Vibhu Dixit; Abhilash Shankarampeta; Vivek Gupta
>
> **摘要:** Temporal reasoning over evolving semi-structured tables poses a challenge to current QA systems. We propose a SQL-based approach that involves (1) generating a 3NF schema from Wikipedia infoboxes, (2) generating SQL queries, and (3) query execution. Our central finding challenges model scaling assumptions: the quality of schema design has a greater impact on QA precision than model capacity. We establish three evidence-based principles: normalization that preserves context, semantic naming that reduces ambiguity, and consistent temporal anchoring. Our best configuration (Gemini 2.5 Flash schema + Gemini-2.0-Flash queries) achieves 80.39 EM, a 16.8\% improvement over the baseline (68.89 EM).
>
---
#### [new 055] Assertion-Conditioned Compliance: A Provenance-Aware Vulnerability in Multi-Turn Tool-Calling Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对多轮工具调用大模型的安全性问题，提出Assertion-Conditioned Compliance（A-CC）评估范式，揭示模型对用户或系统来源的误导性指令存在高度顺从性。研究解决多轮对话中模型鲁棒性不足的隐患，通过量化用户与功能源误导下的合规行为，暴露关键安全漏洞。**

- **链接: [https://arxiv.org/pdf/2512.00332v1](https://arxiv.org/pdf/2512.00332v1)**

> **作者:** Daud Waqas; Aaryamaan Golthi; Erika Hayashida; Huanzhi Mao
>
> **备注:** 15 pages (incl. Appendix), 2 figures, 7 tables
>
> **摘要:** Multi-turn tool-calling LLMs (models capable of invoking external APIs or tools across several user turns) have emerged as a key feature in modern AI assistants, enabling extended dialogues from benign tasks to critical business, medical, and financial operations. Yet implementing multi-turn pipelines remains difficult for many safety-critical industries due to ongoing concerns regarding model resilience. While standardized benchmarks such as the Berkeley Function-Calling Leaderboard (BFCL) have underpinned confidence concerning advanced function-calling models (like Salesforce's xLAM V2), there is still a lack of visibility into multi-turn conversation-level robustness, especially given their exposure to real-world systems. In this paper, we introduce Assertion-Conditioned Compliance (A-CC), a novel evaluation paradigm for multi-turn function-calling dialogues. A-CC provides holistic metrics that evaluate a model's behavior when confronted with misleading assertions originating from two distinct vectors: (1) user-sourced assertions (USAs), which measure sycophancy toward plausible but misinformed user beliefs, and (2) function-sourced assertions (FSAs), which measure compliance with plausible but contradictory system policies (e.g., stale hints from unmaintained tools). Our results show that models are highly vulnerable to both USA sycophancy and FSA policy conflicts, confirming A-CC as a critical, latent vulnerability in deployed agents.
>
---
#### [new 056] DyFuLM: An Advanced Multimodal Framework for Sentiment Analysis
- **分类: cs.CL**

- **简介: 该论文针对情感分析中复杂文本的情感理解难题，提出动态融合学习模型DyFuLM。通过引入层次化动态融合与门控特征聚合模块，实现多层级特征的自适应融合与信息平衡，显著提升粗粒度（82.64%）和细粒度（68.48%）情感分类精度，验证了各模块对性能的关键贡献。**

- **链接: [https://arxiv.org/pdf/2512.01410v1](https://arxiv.org/pdf/2512.01410v1)**

> **作者:** Ruohan Zhou; Jiachen Yuan; Churui Yang; Wenzheng Huang; Guoyan Zhang; Shiyao Wei; Jiazhen Hu; Ning Xin; Md Maruf Hasan
>
> **备注:** 8 pages, 6 figures, preprint. Under review for a suitable AI conference
>
> **摘要:** Understanding sentiment in complex textual expressions remains a fundamental challenge in affective computing. To address this, we propose a Dynamic Fusion Learning Model (DyFuLM), a multimodal framework designed to capture both hierarchical semantic representations and fine-grained emotional nuances. DyFuLM introduces two key moodules: a Hierarchical Dynamic Fusion module that adaptively integrates multi-level features, and a Gated Feature Aggregation module that regulates cross-layer information ffow to achieve balanced representation learning. Comprehensive experiments on multi-task sentiment datasets demonstrate that DyFuLM achieves 82.64% coarse-grained and 68.48% fine-grained accuracy, yielding the lowest regression errors (MAE = 0.0674, MSE = 0.0082) and the highest R^2 coefficient of determination (R^2= 0.6903). Furthermore, the ablation study validates the effectiveness of each module in DyFuLM. When all modules are removed, the accuracy drops by 0.91% for coarse-grained and 0.68% for fine-grained tasks. Keeping only the gated fusion module causes decreases of 0.75% and 0.55%, while removing the dynamic loss mechanism results in drops of 0.78% and 0.26% for coarse-grained and fine-grained sentiment classification, respectively. These results demonstrate that each module contributes significantly to feature interaction and task balance. Overall, the experimental findings further validate that DyFuLM enhances sentiment representation and overall performance through effective hierarchical feature fusion.
>
---
#### [new 057] OPOR-Bench: Evaluating Large Language Models on Online Public Opinion Report Generation
- **分类: cs.CL**

- **简介: 该论文针对在线公共舆情报告自动生成任务，解决缺乏统一任务定义与评估基准的问题。提出OPOR-GEN任务，构建事件中心的OPOR-BENCH数据集，并设计基于智能体的评估框架OPOR-EVAL，实现与人类判断高度相关的自动化评估，为该领域研究提供坚实基础。**

- **链接: [https://arxiv.org/pdf/2512.01896v1](https://arxiv.org/pdf/2512.01896v1)**

> **作者:** Jinzheng Yu; Yang Xu; Haozhen Li; Junqi Li; Yifan Feng; Ligu Zhu; Hao Shen; Lei Shi
>
> **备注:** 27 pages, accepted by CMC-Computers, Materials & Continua, 2025
>
> **摘要:** Online Public Opinion Reports consolidate news and social media for timely crisis management by governments and enterprises. While large language models have made automated report generation technically feasible, systematic research in this specific area remains notably absent, particularly lacking formal task definitions and corresponding benchmarks. To bridge this gap, we define the Automated Online Public Opinion Report Generation (OPOR-GEN) task and construct OPOR-BENCH, an event-centric dataset covering 463 crisis events with their corresponding news articles, social media posts, and a reference summary. To evaluate report quality, we propose OPOR-EVAL, a novel agent-based framework that simulates human expert evaluation by analyzing generated reports in context. Experiments with frontier models demonstrate that our framework achieves high correlation with human judgments. Our comprehensive task definition, benchmark dataset, and evaluation framework provide a solid foundation for future research in this critical domain.
>
---
#### [new 058] Sentiment Analysis and Emotion Classification using Machine Learning Techniques for Nagamese Language - A Low-resource Language
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对低资源语言纳加语（Nagamese）开展情感分析与情绪分类任务，首次构建了包含1195个词的情感极性词典，并结合朴素贝叶斯与支持向量机等机器学习方法，实现对文本情感极性（正、负、中性）及基本情绪的识别，填补了该语言在自然语言处理领域的研究空白。**

- **链接: [https://arxiv.org/pdf/2512.01256v1](https://arxiv.org/pdf/2512.01256v1)**

> **作者:** Ekha Morang; Surhoni A. Ngullie; Sashienla Longkumer; Teisovi Angami
>
> **备注:** 10 pages
>
> **摘要:** The Nagamese language, a.k.a Naga Pidgin, is an Assamese-lexified creole language developed primarily as a means of communication in trade between the people from Nagaland and people from Assam in the north-east India. Substantial amount of work in sentiment analysis has been done for resource-rich languages like English, Hindi, etc. However, no work has been done in Nagamese language. To the best of our knowledge, this is the first attempt on sentiment analysis and emotion classification for the Nagamese Language. The aim of this work is to detect sentiments in terms of polarity (positive, negative and neutral) and basic emotions contained in textual content of Nagamese language. We build sentiment polarity lexicon of 1,195 nagamese words and use these to build features along with additional features for supervised machine learning techniques using Na"ive Bayes and Support Vector Machines. Keywords: Nagamese, NLP, sentiment analysis, machine learning
>
---
#### [new 059] FastPOS: Language-Agnostic Scalable POS Tagging Framework Low-Resource Use Case
- **分类: cs.CL**

- **简介: 该论文提出一种语言无关的可扩展词性标注框架FastPOS，针对低资源语言如孟加拉语和印地语，仅需少量代码即可迁移。利用Transformer架构实现高准确率（96.85%~97%），支持快速跨语言适配，降低模型设计成本，促进非主流语言NLP发展。**

- **链接: [https://arxiv.org/pdf/2512.00745v1](https://arxiv.org/pdf/2512.00745v1)**

> **作者:** Md Abdullah Al Kafi; Sumit Kumar Banshal
>
> **摘要:** This study proposes a language-agnostic transformer-based POS tagging framework designed for low-resource languages, using Bangla and Hindi as case studies. With only three lines of framework-specific code, the model was adapted from Bangla to Hindi, demonstrating effective portability with minimal modification. The framework achieves 96.85 percent and 97 percent token-level accuracy across POS categories in Bangla and Hindi while sustaining strong F1 scores despite dataset imbalance and linguistic overlap. A performance discrepancy in a specific POS category underscores ongoing challenges in dataset curation. The strong results stem from the underlying transformer architecture, which can be replaced with limited code adjustments. Its modular and open-source design enables rapid cross-lingual adaptation while reducing model design and tuning overhead, allowing researchers to focus on linguistic preprocessing and dataset refinement, which are essential for advancing NLP in underrepresented languages.
>
---
#### [new 060] Self-Supervised Borrowing Detection on Multilingual Wordlists
- **分类: cs.CL**

- **简介: 该论文提出一种无监督的多语言借词检测方法，结合PMI相似度与轻量对比学习，利用语音特征向量和自动阈值选择，无需标注数据即可有效识别借词。在基准数据集上表现优于传统字符串相似度方法，达到或超过有监督基线。**

- **链接: [https://arxiv.org/pdf/2512.01713v1](https://arxiv.org/pdf/2512.01713v1)**

> **作者:** Tim Wientzek
>
> **备注:** 29 pages, 3 figures, 12 tables
>
> **摘要:** This paper presents a fully self-supervised approach to borrowing detection in multilingual wordlists. The method combines two sources of information: PMI similarities based on a global correspondence model and a lightweight contrastive component trained on phonetic feature vectors. It further includes an automatic procedure for selecting decision thresholds without requiring labeled data. Experiments on benchmark datasets show that PMI alone already improves over existing string similarity measures such as NED and SCA, and that the combined similarity performs on par with or better than supervised baselines. An ablation study highlights the importance of character encoding, temperature settings and augmentation strategies. The approach scales to datasets of different sizes, works without manual supervision and is provided with a command-line tool that allows researchers to conduct their own studies.
>
---
#### [new 061] Text Annotation via Inductive Coding: Comparing Human Experts to LLMs in Qualitative Data Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究基于大语言模型（LLMs）的归纳编码在定性数据分析中的应用，旨在自动化从数据中自动生成标签的过程。研究对比了六种开源LLMs与人类专家的表现，发现两者在处理复杂与简单语句时表现相反，且标注偏差模式不同。**

- **链接: [https://arxiv.org/pdf/2512.00046v1](https://arxiv.org/pdf/2512.00046v1)**

> **作者:** Angelina Parfenova; Andreas Marfurt; Alexander Denzler; Juergen Pfeffer
>
> **摘要:** This paper investigates the automation of qualitative data analysis, focusing on inductive coding using large language models (LLMs). Unlike traditional approaches that rely on deductive methods with predefined labels, this research investigates the inductive process where labels emerge from the data. The study evaluates the performance of six open-source LLMs compared to human experts. As part of the evaluation, experts rated the perceived difficulty of the quotes they coded. The results reveal a peculiar dichotomy: human coders consistently perform well when labeling complex sentences but struggle with simpler ones, while LLMs exhibit the opposite trend. Additionally, the study explores systematic deviations in both human and LLM generated labels by comparing them to the golden standard from the test set. While human annotations may sometimes differ from the golden standard, they are often rated more favorably by other humans. In contrast, some LLMs demonstrate closer alignment with the true labels but receive lower evaluations from experts.
>
---
#### [new 062] Reward Auditor: Inference on Reward Modeling Suitability in Real-World Perturbed Scenarios
- **分类: cs.CL**

- **简介: 该论文针对大语言模型对齐中奖励模型（RM）在真实世界扰动下的可靠性问题，提出“Reward Auditor”框架。通过假设检验量化偏好置信度分布退化，评估RM在特定扰动场景下的系统性脆弱性，解决现有方法忽视实际应用中适配性的问题，推动更安全、可信的对齐系统发展。**

- **链接: [https://arxiv.org/pdf/2512.00920v1](https://arxiv.org/pdf/2512.00920v1)**

> **作者:** Jianxiang Zang; Yongda Wei; Ruxue Bai; Shiyu Jiang; Nijia Mo; Binhong Li; Qiang Sun; Hui Liu
>
> **摘要:** Reliable reward models (RMs) are critical for ensuring the safe alignment of large language models (LLMs). However, current evaluation methods focus solely on preference perception accuracies in given specific scenarios, obscuring the critical vulnerabilities of RMs in real-world scenarios. We identify the true challenge lies in assessing a novel dimension: Suitability, defined as conditional reliability under specific real-world perturbations. To this end, we introduce Reward Auditor, a hypothesis-testing framework specifically designed for RM suitability inference. Rather than answering "How accurate is the RM's preference perception for given samples?", it employs scientific auditing to answer: "Can we infer RMs exhibit systematic vulnerabilities in specific real-world scenarios?". Under real-world perturbed scenarios, Reward Auditor quantifies statistical significance and effect size by auditing distribution degradation of RM preference perception confidence. This enables inference of both the certainty and severity of RM vulnerabilities across diverse real-world scenarios. This lays a solid foundation for building next-generation LLM alignment systems that are verifiably safe, more robust, and trustworthy.
>
---
#### [new 063] TempPerturb-Eval: On the Joint Effects of Internal Temperature and External Perturbations in RAG Robustness
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究RAG系统在噪声检索下的鲁棒性，聚焦内部温度与外部扰动的联合影响。针对现有评估忽视二者交互的问题，提出扰动-温度分析框架，通过多类型扰动与不同温度实验，揭示高温加剧脆弱性、扰动敏感性非线性的规律，构建诊断基准与调参指南。**

- **链接: [https://arxiv.org/pdf/2512.01183v1](https://arxiv.org/pdf/2512.01183v1)**

> **作者:** Yongxin Zhou; Philippe Mulhem; Didier Schwab
>
> **摘要:** The evaluation of Retrieval-Augmented Generation (RAG) systems typically examines retrieval quality and generation parameters like temperature in isolation, overlooking their interaction. This work presents a systematic investigation of how text perturbations (simulating noisy retrieval) interact with temperature settings across multiple LLM runs. We propose a comprehensive RAG Perturbation-Temperature Analysis Framework that subjects retrieved documents to three distinct perturbation types across varying temperature settings. Through extensive experiments on HotpotQA with both open-source and proprietary LLMs, we demonstrate that performance degradation follows distinct patterns: high-temperature settings consistently amplify vulnerability to perturbations, while certain perturbation types exhibit non-linear sensitivity across the temperature range. Our work yields three key contributions: (1) a diagnostic benchmark for assessing RAG robustness, (2) an analytical framework for quantifying perturbation-temperature interactions, and (3) practical guidelines for model selection and parameter tuning under noisy retrieval conditions.
>
---
#### [new 064] MARSAD: A Multi-Functional Tool for Real-Time Social Media Analysis
- **分类: cs.CL**

- **简介: 该论文提出MARSAD，一个面向阿拉伯语社交媒体的实时分析平台。针对多任务分析需求，解决非技术用户难以获取与处理社交数据的问题。工作包括集成情感、情绪、虚假信息、仇恨言论等分析功能，提供安全数据抓取与可视化，实现高效、易用的多维度社会媒体监测。**

- **链接: [https://arxiv.org/pdf/2512.01369v1](https://arxiv.org/pdf/2512.01369v1)**

> **作者:** Md. Rafiul Biswas; Firoj Alam; Wajdi Zaghouani
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** MARSAD is a multifunctional natural language processing (NLP) platform designed for real-time social media monitoring and analysis, with a particular focus on the Arabic-speaking world. It enables researchers and non-technical users alike to examine both live and archived social media content, producing detailed visualizations and reports across various dimensions, including sentiment analysis, emotion analysis, propaganda detection, fact-checking, and hate speech detection. The platform also provides secure data-scraping capabilities through API keys for accessing public social media data. MARSAD's backend architecture integrates flexible document storage with structured data management, ensuring efficient processing of large and multimodal datasets. Its user-friendly frontend supports seamless data upload and interaction.
>
---
#### [new 065] Advancing Academic Chatbots: Evaluation of Non Traditional Outputs
- **分类: cs.CL**

- **简介: 该论文研究学术聊天机器人在非传统输出上的表现，旨在评估大模型生成幻灯片与播客脚本的质量。通过对比两种检索策略（Graph RAG与Advanced RAG）和双模型（GPT 4o mini与LLaMA 3），发现Advanced RAG结合GPT 4o mini在准确性和生成质量上最优，强调人类评估在识别布局与风格缺陷中的关键作用。**

- **链接: [https://arxiv.org/pdf/2512.00991v1](https://arxiv.org/pdf/2512.00991v1)**

> **作者:** Nicole Favero; Francesca Salute; Daniel Hardt
>
> **摘要:** Most evaluations of large language models focus on standard tasks such as factual question answering or short summarization. This research expands that scope in two directions: first, by comparing two retrieval strategies, Graph RAG, structured knowledge-graph based, and Advanced RAG, hybrid keyword-semantic search, for QA; and second, by evaluating whether LLMs can generate high quality non-traditional academic outputs, specifically slide decks and podcast scripts. We implemented a prototype combining Meta's LLaMA 3 70B open weight and OpenAI's GPT 4o mini API based. QA performance was evaluated using both human ratings across eleven quality dimensions and large language model judges for scalable cross validation. GPT 4o mini with Advanced RAG produced the most accurate responses. Graph RAG offered limited improvements and led to more hallucinations, partly due to its structural complexity and manual setup. Slide and podcast generation was tested with document grounded retrieval. GPT 4o mini again performed best, though LLaMA 3 showed promise in narrative coherence. Human reviewers were crucial for detecting layout and stylistic flaws, highlighting the need for combined human LLM evaluation in assessing emerging academic outputs.
>
---
#### [new 066] MCAT: Scaling Many-to-Many Speech-to-Text Translation with MLLMs to 70 Languages
- **分类: cs.CL**

- **简介: 该论文针对多语言语音转文本翻译任务，解决语言覆盖窄和推理效率低的问题。提出MCAT框架，通过课程学习与数据平衡扩展至70语言，并设计优化语音适配器将序列长度压缩至30 tokens，显著提升多对多翻译能力与批量推理效率，仅需10小时/语言数据及约100M参数。**

- **链接: [https://arxiv.org/pdf/2512.01512v1](https://arxiv.org/pdf/2512.01512v1)**

> **作者:** Yexing Du; Kaiyuan Liu; Youcheng Pan; Bo Yang; Keqi Deng; Xie Chen; Yang Xiang; Ming Liu; Bin Qin; YaoWei Wang
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved great success in Speech-to-Text Translation (S2TT) tasks. However, current research is constrained by two key challenges: language coverage and efficiency. Most of the popular S2TT datasets are substantially English-centric, which restricts the scaling-up of MLLMs' many-to-many translation capabilities. Moreover, the inference speed of MLLMs degrades dramatically when the speech is converted into long sequences (e.g., 750 tokens). To address these limitations, we propose a Multilingual Cost-effective Accelerated Speech-to-Text Translator (MCAT) framework, which includes two innovations. First, a language scaling method that leverages curriculum learning and a data balancing strategy is introduced to extend the language coverage supported by MLLMs to 70 languages and achieve mutual translation among these languages. Second, an optimized speech adapter module is designed to reduce the length of the speech sequence to only 30 tokens. Extensive experiments were conducted on MLLMs of different scales (9B and 27B). The experimental results demonstrate that MCAT not only surpasses state-of-the-art end-to-end models on the FLEURS dataset across 70x69 directions but also enhances batch inference efficiency. This is achieved with only ~100M trainable parameters and by using only 10 hours of S2TT data per language. Furthermore, we have released MCAT as open-source to promote the development of MLLMs for robust S2TT capabilities. The code and models are released at https://github.com/yxduir/m2m-70.
>
---
#### [new 067] Dr.Mi-Bench: A Modular-integrated Benchmark for Scientific Deep Research Agent
- **分类: cs.CL**

- **简介: 该论文针对科学深度研究代理的评估难题，提出Dr.Mi-Bench基准与Dr.Mi-Eval评估范式。聚焦科研领域，涵盖10个学科200个实例，通过模块化评估框架，诊断代理在规划、检索、推理中的短板，揭示其多源检索与跨域一致性不足，并指出提升高层规划能力是释放大模型潜力的关键。**

- **链接: [https://arxiv.org/pdf/2512.00986v1](https://arxiv.org/pdf/2512.00986v1)**

> **作者:** Zhihan Guo; Feiyang Xu; Yifan Li; Muzhi Li; Shuai Zou; Jiele Wu; Han Shi; Haoli Bai; Ho-fung Leung; Irwin King
>
> **摘要:** The explosive growth in academic literature necessitates automated deep research (DR) agents, yet their evaluation remains a significant challenge. First, existing benchmarks often focus narrowly on retrieval while neglecting high-level planning and reasoning. Second, existing benchmarks favor general domains over the scientific domains that are the core application for DR agents. To address these gaps, we introduce Dr.Mi-Bench, a Modular-integrated benchmark for scientific DR agents. Grounded in academic literature, our benchmark uses a human-annotated dataset of 200 instances across 10 scientific domains, including both research and review papers. Besides, we also propose a Modular-integrated Evaluation Paradigm for DR Agents (Dr.Mi-Eval), a novel modular-integrated evaluation paradigm, which leverages the rich structure of academic papers to assess the core competencies of planning, retrieval, and reasoning through two complementary modes: an end-to-end evaluation for DR agents and an isolated evaluation for foundational LLMs as potential backbones. Experimental results reveal a fragmented performance landscape: agents exhibit specialized strengths but share critical weaknesses, most notably in performing the multi-source retrieval required for review-style tasks and performing consistently across diverse scientific fields. Moreover, improving high-level planning capability is the crucial factor for unlocking the reasoning potential of foundational LLMs as backbones. By exposing these actionable failure modes, Dr.Mi-Bench provides a diagnostic tool to guide the development of more reliable academic research assistants.
>
---
#### [new 068] Minimal-Edit Instruction Tuning for Low-Resource Indic GEC
- **分类: cs.CL**

- **简介: 该论文针对低资源印地语系语法错误修正（GEC）任务，解决标注数据少、书写系统多样、形态复杂问题。提出无需数据增强的指令微调方法，采用4-bit量化与PEFT技术，结合分类器引导的提示设计和确定性解码，实现最小化、语义保持的编辑，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.00219v1](https://arxiv.org/pdf/2512.00219v1)**

> **作者:** Akhil Rajeev P
>
> **备注:** Submitted to AACL-IJCNLP Bhasha Workshop Shared Task1 :GEC
>
> **摘要:** Grammatical error correction for Indic languages faces limited supervision, diverse scripts, and rich morphology. We propose an augmentation-free setup that uses instruction-tuned large language models and conservative decoding. A 12B GEMMA 3 model is instruction-tuned in bnb 4-bit precision with parameter-efficient fine-tuning (PEFT) and Alpaca-style formatting. Decoding follows a deterministic, constraint-aware procedure with a lightweight normaliser that encourages minimal, meaning-preserving edits. We operationalise inference, subsequent to instruction fine-tuning (IFT), via a fixed, language-specific prompt directly synthesised from a deterministic error classifier's taxonomy, label distributions, and precedence ordering computed on the training corpus. Under the official untuned GLEU evaluation, the system scores 92.41 on Malayalam, sixth overall, and 81.44 on Hindi, third overall. These results indicate that classifier-informed prompt design, adapter-based instruction tuning, and deterministic decoding provide a reproducible and a computationally efficient alternative to augmentation-centred pipelines for Indic GEC. The approach also motivates future work on stronger morphosyntactic constraints and human-centred evaluation of conservative edits.
>
---
#### [new 069] Table as a Modality for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型在表格推理任务中的表现不足，提出将表格视为独立模态的TAMO框架。通过引入超图神经网络作为全局表编码器，与主流LLM融合，有效保留表格结构信息，显著提升泛化能力，在多个基准上平均性能提升42.65%。**

- **链接: [https://arxiv.org/pdf/2512.00947v1](https://arxiv.org/pdf/2512.00947v1)**

> **作者:** Liyao Li; Chao Ye; Wentao Ye; Yifei Sun; Zhe Jiang; Haobo Wang; Jiaming Tian; Yiming Zhang; Ningtao Wang; Xing Fu; Gang Chen; Junbo Zhao
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** To migrate the remarkable successes of Large Language Models (LLMs), the community has made numerous efforts to generalize them to the table reasoning tasks for the widely deployed tabular data. Despite that, in this work, by showing a probing experiment on our proposed StructQA benchmark, we postulate that even the most advanced LLMs (such as GPTs) may still fall short of coping with tabular data. More specifically, the current scheme often simply relies on serializing the tabular data, together with the meta information, then inputting them through the LLMs. We argue that the loss of structural information is the root of this shortcoming. In this work, we further propose TAMO, which bears an ideology to treat the tables as an independent modality integrated with the text tokens. The resulting model in TAMO is a multimodal framework consisting of a hypergraph neural network as the global table encoder seamlessly integrated with the mainstream LLM. Empirical results on various benchmarking datasets, including HiTab, WikiTQ, WikiSQL, FeTaQA, and StructQA, have demonstrated significant improvements on generalization with an average relative gain of 42.65%.
>
---
#### [new 070] CourseTimeQA: A Lecture-Video Benchmark and a Latency-Constrained Cross-Modal Fusion Method for Timestamped QA
- **分类: cs.CL**

- **简介: 该论文针对教育视频中的时序问答任务，提出CourseTimeQA基准与轻量级跨模态检索方法CrossFusion-RAG。旨在在单卡延迟与内存约束下，实现高效精准的时序段检索与答案生成，通过融合语音识别与视觉特征，提升检索性能并保证实时性。**

- **链接: [https://arxiv.org/pdf/2512.00360v1](https://arxiv.org/pdf/2512.00360v1)**

> **作者:** Vsevolod Kovalev; Parteek Kumar
>
> **备注:** 5 figures, 8 tables
>
> **摘要:** We study timestamped question answering over educational lecture videos under a single-GPU latency/memory budget. Given a natural-language query, the system retrieves relevant timestamped segments and synthesizes a grounded answer. We present CourseTimeQA (52.3 h, 902 queries across six courses) and a lightweight, latency-constrained cross-modal retriever (CrossFusion-RAG) that combines frozen encoders, a learned 512->768 vision projection, shallow query-agnostic cross-attention over ASR and frames with a temporal-consistency regularizer, and a small cross-attentive reranker. On CourseTimeQA, CrossFusion-RAG improves nDCG@10 by 0.10 and MRR by 0.08 over a strong BLIP-2 retriever while achieving approximately 1.55 s median end-to-end latency on a single A100. Closest comparators (zero-shot CLIP multi-frame pooling; CLIP + cross-encoder reranker + MMR; learned late-fusion gating; text-only hybrid with cross-encoder reranking and its MMR variant; caption-augmented text retrieval; non-learned temporal smoothing) are evaluated under matched hardware and indexing. We report robustness across ASR noise (WER quartiles), diagnostics for temporal localization, and full training/tuning details to support reproducible comparison.
>
---
#### [new 071] WaterSearch: A Quality-Aware Search-based Watermarking Framework for Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对大模型文本水印中检测性与文本质量的权衡问题，提出WaterSearch框架。通过控制种子池实现并行生成，联合优化分布保真度与水印信号特性，提升文本质量与抗攻击能力。实验表明其在多任务、短文本及低熵场景下显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.00837v1](https://arxiv.org/pdf/2512.00837v1)**

> **作者:** Yukang Lin; Jiahao Shao; Shuoran Jiang; Wentao Zhu; Bingjie Lu; Xiangping Wu; Joanna Siebert; Qingcai Chen
>
> **摘要:** Watermarking acts as a critical safeguard in text generated by Large Language Models (LLMs). By embedding identifiable signals into model outputs, watermarking enables reliable attribution and enhances the security of machine-generated content. Existing approaches typically embed signals by manipulating token generation probabilities. Despite their effectiveness, these methods inherently face a trade-off between detectability and text quality: the signal strength and randomness required for robust watermarking tend to degrade the performance of downstream tasks. In this paper, we design a novel embedding scheme that controls seed pools to facilitate diverse parallel generation of watermarked text. Based on that scheme, we propose WaterSearch, a sentence-level, search-based watermarking framework adaptable to a wide range of existing methods. WaterSearch enhances text quality by jointly optimizing two key aspects: 1) distribution fidelity and 2) watermark signal characteristics. Furthermore, WaterSearch is complemented by a sentence-level detection method with strong attack robustness. We evaluate our method on three popular LLMs across ten diverse tasks. Extensive experiments demonstrate that our method achieves an average performance improvement of 51.01\% over state-of-the-art baselines at a watermark detectability strength of 95\%. In challenging scenarios such as short text generation and low-entropy output generation, our method yields performance gains of 47.78\% and 36.47\%, respectively. Moreover, under different attack senarios including insertion, synonym substitution and paraphrase attasks, WaterSearch maintains high detectability, further validating its robust anti-attack capabilities. Our code is available at \href{https://github.com/Yukang-Lin/WaterSearch}{https://github.com/Yukang-Lin/WaterSearch}.
>
---
#### [new 072] Accelerating Bangla NLP Tasks with Automatic Mixed Precision: Resource-Efficient Training Preserving Model Efficacy
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究如何通过自动混合精度（AMP）加速孟加拉语自然语言处理（NLP）任务。针对硬件资源有限导致训练效率低的问题，采用AMP在保持模型性能的前提下，实现44.5%的训练加速和17.6%的内存降低，验证了其在四大孟加拉语NLP任务中的有效性。**

- **链接: [https://arxiv.org/pdf/2512.00829v1](https://arxiv.org/pdf/2512.00829v1)**

> **作者:** Md Mehrab Hossain Opi; Sumaiya Khan; Moshammad Farzana Rahman
>
> **摘要:** Training models for Natural Language Processing (NLP) requires substantial computational resources and time, posing significant challenges, especially for NLP development in Bangla, where access to high-end hardware is often limited. In this work, we explore automatic mixed precision (AMP) training as a means to improve computational efficiency without sacrificing model performance. By leveraging a dynamic mix of 16-bit and 32-bit floating-point computations, AMP lowers GPU memory requirements and speeds up training without degrading model performance. We evaluate AMP across four standard Bangla NLP tasks, namely sentiment analysis, named entity recognition, error classification, and question answering, using four transformer-based models: BanglaBERT, BanglishBERT, XLM-R, and mBERT. Our results demonstrate that AMP accelerates training by 44.5% and reduces memory consumption by 17.6%, while maintaining F-1 score within 99.7% of the full-precision baselines. This empirical study highlights AMP's potential to democratize access to state-of-the-art NLP capabilities in hardware-constrained settings by lowering computational barriers.
>
---
#### [new 073] EduEval: A Hierarchical Cognitive Benchmark for Evaluating Large Language Models in Chinese Education
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型在中文教育领域评估不足的问题，提出EduEval基准。它构建了融合认知维度的评测框架，整合真实教育数据，涵盖11000+题目，评估14个模型在不同认知层级的表现，揭示模型在推理与创意任务中的短板，为优化教育类LLM提供依据。**

- **链接: [https://arxiv.org/pdf/2512.00290v1](https://arxiv.org/pdf/2512.00290v1)**

> **作者:** Guoqing Ma; Jia Zhu; Hanghui Guo; Weijie Shi; Yue Cui; Jiawei Shen; Zilong Li; Yidan Liang
>
> **摘要:** Large language models (LLMs) demonstrate significant potential for educational applications. However, their unscrutinized deployment poses risks to educational standards, underscoring the need for rigorous evaluation. We introduce EduEval, a comprehensive hierarchical benchmark for evaluating LLMs in Chinese K-12 education. This benchmark makes three key contributions: (1) Cognitive Framework: We propose the EduAbility Taxonomy, which unifies Bloom's Taxonomy and Webb's Depth of Knowledge to organize tasks across six cognitive dimensions including Memorization, Understanding, Application, Reasoning, Creativity, and Ethics. (2) Authenticity: Our benchmark integrates real exam questions, classroom conversation, student essays, and expert-designed prompts to reflect genuine educational challenges; (3) Scale: EduEval comprises 24 distinct task types with over 11,000 questions spanning primary to high school levels. We evaluate 14 leading LLMs under both zero-shot and few-shot settings, revealing that while models perform well on factual tasks, they struggle with classroom dialogue classification and exhibit inconsistent results in creative content generation. Interestingly, several open source models outperform proprietary systems on complex educational reasoning. Few-shot prompting shows varying effectiveness across cognitive dimensions, suggesting that different educational objectives require tailored approaches. These findings provide targeted benchmarking metrics for developing LLMs specifically optimized for diverse Chinese educational tasks.
>
---
#### [new 074] Slovak Conceptual Dictionary
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对斯洛伐克语资源匮乏问题，构建了首个概念词典。旨在解决低资源语言在自然语言处理中缺乏高质量语料库的问题，通过创建可机器读取的词汇知识库，为文本自动化处理提供基础支持。**

- **链接: [https://arxiv.org/pdf/2512.00579v1](https://arxiv.org/pdf/2512.00579v1)**

> **作者:** Miroslav Blšták
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** When solving tasks in the field of natural language processing, we sometimes need dictionary tools, such as lexicons, word form dictionaries or knowledge bases. However, the availability of dictionary data is insufficient in many languages, especially in the case of low resourced languages. In this article, we introduce a new conceptual dictionary for the Slovak language as the first linguistic tool of this kind. Since Slovak language is a language with limited linguistic resources and there are currently not available any machine-readable linguistic data sources with a sufficiently large volume of data, many tasks which require automated processing of Slovak text achieve weaker results compared to other languages and are almost impossible to solve.
>
---
#### [new 075] Cross-Lingual Interleaving for Speech Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对低资源语言在语音语言模型（SLM）中进展滞后的问题，提出跨语言交错训练方法，通过混合多语言语音标记实现无文本监督的跨语言学习。构建了英法语数据集及评估基准，实验证明该方法提升单语语义准确率，增强跨语言生成与隐状态对齐，为构建多语言语音理解模型提供简单高效的方案。**

- **链接: [https://arxiv.org/pdf/2512.01865v1](https://arxiv.org/pdf/2512.01865v1)**

> **作者:** Adel Moumen; Guangzhi Sun; Philip C. Woodland
>
> **摘要:** Spoken Language Models (SLMs) aim to learn linguistic competence directly from speech using discrete units, widening access to Natural Language Processing (NLP) technologies for languages with limited written resources. However, progress has been largely English-centric due to scarce spoken evaluation benchmarks and training data, making cross-lingual learning difficult. We present a cross-lingual interleaving method that mixes speech tokens across languages without textual supervision. We also release an EN-FR training dataset, TinyStories (~42k hours), together with EN-FR spoken StoryCloze and TopicCloze benchmarks for cross-lingual semantic evaluation, both synthetically generated using GPT-4. On 360M and 1B SLMs under matched training-token budgets, interleaving improves monolingual semantic accuracy, enables robust cross-lingual continuation, and strengthens cross-lingual hidden-state alignment. Taken together, these results indicate that cross-lingual interleaving is a simple, scalable route to building multilingual SLMs that understand and converse across languages. All resources will be made open-source to support reproducibility.
>
---
#### [new 076] Auxiliary-Hyperparameter-Free Sampling: Entropy Equilibrium for Text Generation
- **分类: cs.CL**

- **简介: 该论文针对大语言模型文本生成中采样策略依赖额外超参数的问题，提出无辅助超参数的熵平衡采样（EES）方法。通过信息论思想动态调节候选集，平衡熵与概率质量，提升生成质量与多样性，无需调参，简化部署。**

- **链接: [https://arxiv.org/pdf/2512.00789v1](https://arxiv.org/pdf/2512.00789v1)**

> **作者:** Xiaodong Cai; Hai Lin; Shaoxiong Zhan; Weiqi Luo; Hong-Gee Kim; Hongyan Hao; Yu Yang; Hai-Tao Zheng
>
> **摘要:** Token sampling strategies critically influence text generation quality in large language models (LLMs). However, existing methods introduce additional hyperparameters, requiring extensive tuning and complicating deployment. We present Entropy Equilibrium Sampling (EES), an auxiliary hyperparameter-free approach inspired by information theory that can dynamically adjust candidate sets by balancing normalized entropy with probability mass. We evaluate EES on both reasoning and generation tasks across a range of model architectures. Our results show that EES consistently performs well across temperature settings, delivering competitive accuracy and coherence while maintaining diversity. By eliminating the need for hyperparameter tuning, EES greatly simplifies deployment while improving performance. Code is available at https://github.com/shuanncai/EES
>
---
#### [new 077] Closing the Gap: Data-Centric Fine-Tuning of Vision Language Models for the Standardized Exam Questions
- **分类: cs.CV; cs.AI; cs.CL; cs.CY**

- **简介: 该论文聚焦于视觉语言模型在标准化考试题中的多模态推理任务，旨在提升开放权重模型性能。通过构建161.4百万词元的高质量多模态数据集并优化推理语法，实现78.6%准确率，仅比闭源模型低1.0%，验证了数据驱动方法在提升模型表现中的关键作用。**

- **链接: [https://arxiv.org/pdf/2512.00042v1](https://arxiv.org/pdf/2512.00042v1)**

> **作者:** Egemen Sert; Şeyda Ertekin
>
> **摘要:** Multimodal reasoning has become a cornerstone of modern AI research. Standardized exam questions offer a uniquely rigorous testbed for such reasoning, providing structured visual contexts and verifiable answers. While recent progress has largely focused on algorithmic advances such as reinforcement learning (e.g., GRPO, DPO), the data centric foundations of vision language reasoning remain less explored. We show that supervised fine-tuning (SFT) with high-quality data can rival proprietary approaches. To this end, we compile a 161.4 million token multimodal dataset combining textbook question-solution pairs, curriculum aligned diagrams, and contextual materials, and fine-tune Qwen-2.5VL-32B using an optimized reasoning syntax (QMSA). The resulting model achieves 78.6% accuracy, only 1.0% below Gemini 2.0 Flash, on our newly released benchmark YKSUniform, which standardizes 1,854 multimodal exam questions across 309 curriculum topics. Our results reveal that data composition and representational syntax play a decisive role in multimodal reasoning. This work establishes a data centric framework for advancing open weight vision language models, demonstrating that carefully curated and curriculum-grounded multimodal data can elevate supervised fine-tuning to near state-of-the-art performance.
>
---
#### [new 078] HalluGraph: Auditable Hallucination Detection for Legal RAG Systems via Knowledge Graph Alignment
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对法律RAG系统中的幻觉问题，提出HalluGraph框架，通过知识图谱结构对齐实现可审计的幻觉检测。旨在解决生成内容与源文档不一致的可信性问题，工作包括设计实体归因（EG）与关系保持（RP）指标，实现高精度、可解释的幻觉识别，显著优于传统语义相似度方法。**

- **链接: [https://arxiv.org/pdf/2512.01659v1](https://arxiv.org/pdf/2512.01659v1)**

> **作者:** Valentin Noël; Elimane Yassine Seidou; Charly Ken Capo-Chichi; Ghanem Amari
>
> **备注:** 8 pages, 4 figures, under review
>
> **摘要:** Legal AI systems powered by retrieval-augmented generation (RAG) face a critical accountability challenge: when an AI assistant cites case law, statutes, or contractual clauses, practitioners need verifiable guarantees that generated text faithfully represents source documents. Existing hallucination detectors rely on semantic similarity metrics that tolerate entity substitutions, a dangerous failure mode when confusing parties, dates, or legal provisions can have material consequences. We introduce HalluGraph, a graph-theoretic framework that quantifies hallucinations through structural alignment between knowledge graphs extracted from context, query, and response. Our approach produces bounded, interpretable metrics decomposed into \textit{Entity Grounding} (EG), measuring whether entities in the response appear in source documents, and \textit{Relation Preservation} (RP), verifying that asserted relationships are supported by context. On structured control documents, HalluGraph achieves near-perfect discrimination ($>$400 words, $>$20 entities), HalluGraph achieves $AUC = 0.979$, while maintaining robust performance ($AUC \approx 0.89$) on challenging generative legal task, consistently outperforming semantic similarity baselines. The framework provides the transparency and traceability required for high-stakes legal applications, enabling full audit trails from generated assertions back to source passages.
>
---
#### [new 079] From Atomic to Composite: Reinforcement Learning Enables Generalization in Complementary Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究互补推理任务，旨在解决模型在复杂推理中泛化能力不足的问题。通过分解为参数推理与上下文推理两种原子技能，发现仅靠监督微调（SFT）导致过拟合，而强化学习（RL）需在原子技能掌握后才能合成复杂策略，实现零样本泛化。**

- **链接: [https://arxiv.org/pdf/2512.01970v1](https://arxiv.org/pdf/2512.01970v1)**

> **作者:** Sitao Cheng; Xunjian Yin; Ruiwen Zhou; Yuxuan Li; Xinyi Wang; Liangming Pan; William Yang Wang; Victor Zhong
>
> **备注:** Work in Progress. Code and data will be available at https://github.com/sitaocheng/from_atomic_to_composite
>
> **摘要:** The mechanism by which RL contributes to reasoning capabilities-whether it incentivizes the synthesis of new skills or merely amplifies existing behaviors-remains a subject of intense debate. In this work, we investigate this question through the lens of Complementary Reasoning, a complex task that requires integrating internal parametric knowledge with external contextual information. Using a controlled synthetic dataset of human biographies, we strictly decouple this ability into two atomic skills: Parametric Reasoning (relying on internal knowledge) and Contextual Reasoning (depending on external information). To rigorously assess capability boundaries, we evaluate generalization across three distinct levels of difficulty: I.I.D., Composition, and Zero-shot settings. We find that while SFT is sufficient for in-distribution performance, it struggles with O.O.D. generalization, particularly in Zero-shot settings where relational combinations are novel. Crucially, we identify the SFT Generalization Paradox: Models supervised solely on the composite task achieve near-perfect in-distribution accuracy but collapse on out-of-distribution generalization, indicating their reliance on rote memorization of path shortcuts. In contrast, we find that RL acts as a reasoning synthesizer rather than a probability amplifier. However, we uncover a strict atomic prerequisite: RL can only synthesize these complex strategies if the base model has first mastered the independent atomic skills (Parametric and Contextual) via SFT. These findings challenge the view of RL as a mere amplifier, suggesting that given sufficient atomic foundations, RL can actively synthesize complex reasoning strategies from learned primitives without explicit supervision on such complex strategies. This indicates that decoupled atomic training followed by RL offers a scalable path to generalization for complex reasoning tasks.
>
---
#### [new 080] Towards Active Synthetic Data Generation for Finetuning Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究语言模型微调中的主动合成数据生成，旨在通过迭代闭环方式动态生成高质量训练数据。针对固定生成预算下静态合成数据效率低的问题，提出基于学生模型状态的动态数据选择策略，利用简单主动学习方法提升微调性能，在数学与逻辑推理任务上验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2512.00884v1](https://arxiv.org/pdf/2512.00884v1)**

> **作者:** Samuel Kessler; Menglin Xia; Daniel Madrigal Diaz; Dongge Han; Helia Heshemi; Saravan Rajmohan; Victor Ruehle; Jordan T. Ash
>
> **备注:** 14 figures, 36 pages
>
> **摘要:** A common and effective means for improving language model capabilities involves finetuning a ``student'' language model's parameters on generations from a more proficient ``teacher'' model. Termed ``synthetic data'', these generations are often produced before any student finetuning, but some work has considered generating new synthetic samples as training progresses. This paper studies and advocates for the latter case, where data are generated in an iterative, closed-loop fashion that is guided by the current state of the student model. For a fixed budget of generated samples, or a budget in terms of compute spent querying a teacher, we show that this curation of finetuning data affords improved student performance over static generation. Further, while there have been several LLM-specific methods proposed that operate in this regime, we find that simple, inexpensive selection criteria from the active learning literature tend to be most performant. We validate these claims across four mathematical and logical reasoning datasets using four different small language models.
>
---
#### [new 081] Bias Testing and Mitigation in Black Box LLMs using Metamorphic Relations
- **分类: cs.SE; cs.CL**

- **简介: 该论文针对大语言模型中的隐性社会偏见问题，提出基于元测试关系（MR）的统一评估与缓解框架。通过构建六种新颖的MR变换，自动生成语义等价但具挑战性的输入，揭示模型在复杂上下文中的偏差行为，并利用生成样本进行微调，显著提升模型公平性，有效检测并减少偏见。**

- **链接: [https://arxiv.org/pdf/2512.00556v1](https://arxiv.org/pdf/2512.00556v1)**

> **作者:** Sina Salimian; Gias Uddin; Sumon Biswas; Henry Leung
>
> **摘要:** The widespread deployment of Large Language Models (LLMs) has intensified concerns about subtle social biases embedded in their outputs. Existing guardrails often fail when faced with indirect or contextually complex bias-inducing prompts. To address these limitations, we propose a unified framework for both systematic bias evaluation and targeted mitigation. Our approach introduces six novel Metamorphic Relations (MRs) that, based on metamorphic testing principles, transform direct bias-inducing inputs into semantically equivalent yet adversarially challenging variants. These transformations enable an automated method for exposing hidden model biases: when an LLM responds inconsistently or unfairly across MR-generated variants, the underlying bias becomes detectable. We further show that the same MRs can be used to generate diverse bias-inducing samples for fine-tuning, directly linking the testing process to mitigation. Using six state-of-the-art LLMs - spanning open-source and proprietary models - and a representative subset of 385 questions from the 8,978-item BiasAsker benchmark covering seven protected groups, our MRs reveal up to 14% more hidden biases compared to existing tools. Moreover, fine-tuning with both original and MR-mutated samples significantly enhances bias resiliency, increasing safe response rates from 54.7% to over 88.9% across models. These results highlight metamorphic relations as a practical mechanism for improving fairness in conversational AI.
>
---
#### [new 082] The Necessity of Imperfection:Reversing Model Collapse via Simulating Cognitive Boundedness
- **分类: cs.AI; cs.CL; cs.CY; cs.LG; q-fin.TR**

- **简介: 该论文针对大模型训练中因合成数据过度平滑导致的“模型坍缩”问题，提出模拟人类认知局限的PMCSF框架。通过逆向解析文本为认知向量，并用数学定义的认知扰动重构文本，生成具人类典型不完美特征的合成数据。实验表明，该方法显著提升数据认知真实性与实际应用性能，为解决AI数据危机提供新路径。**

- **链接: [https://arxiv.org/pdf/2512.01354v1](https://arxiv.org/pdf/2512.01354v1)**

> **作者:** Zhongjie Jiang
>
> **备注:** 38 pages,5 figures,30 tables. This paper proposes the Prompt-driven Cognitive Computing Framework (PMCSF) and validates it with A-share market stress tests (N=23 for 2015 crash, N=13 for 2024 bull market). Includes detailed appendices on cognitive vector definitions, perturbation operators, and financial backtest data
>
> **摘要:** Although synthetic data is widely promoted as a remedy, its prevailing production paradigm -- one optimizing for statistical smoothness -- systematically removes the long-tail, cognitively grounded irregularities that characterize human text. Prolonged training on such statistically optimal but cognitively impoverished data accelerates model collapse. This paper proposes a paradigm shift: instead of imitating the surface properties of data, we simulate the cognitive processes that generate human text. We introduce the Prompt-driven Cognitive Computing Framework (PMCSF), whose core consists of a Cognitive State Decoder (CSD) that reverse-engineers unstructured text into structured cognitive vectors, and a Cognitive Text Encoder (CTE) that re-materializes these states into text enriched with human-typical imperfections via mathematically defined Cognitive Perturbation Operators. The framework is validated through a two-stage objective evaluation pipeline. First, in cognitive codec verification, CTE text yields a Jensen-Shannon divergence of 0.0614 from human text (vs. 0.4431 for standard LLM output), passes double-blind professional media review, and achieves an intraclass correlation coefficient ICC > 0.9 for cognitive profile alignment across heterogeneous models. Second, in functional gain evaluation, isomorphic stress tests in the A-share market show that strategies incorporating CTE-generated data reduce maximum drawdown by 47.4% during the 2015 crash and deliver 8.6% Defensive Alpha, exceeding transaction costs by a factor of 33. Our findings demonstrate that modelling human cognitive limitations -- not copying surface data -- enables synthetic data with genuine functional gain, offering a viable technical pathway toward resolving the AI data-collapse crisis.
>
---
#### [new 083] Evaluating Legal Reasoning Traces with Legal Issue Tree Rubrics
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对法律领域大模型推理轨迹评估难题，构建了24K规模的LEGIT数据集，将判决书转化为法律问题树作为评估标准。通过实证发现，法律问题覆盖与正确性均影响模型表现，并验证RAG与基于评分的强化学习在提升推理能力上各有优势。**

- **链接: [https://arxiv.org/pdf/2512.01020v1](https://arxiv.org/pdf/2512.01020v1)**

> **作者:** Jinu Lee; Kyoung-Woon On; Simeng Han; Arman Cohan; Julia Hockenmaier
>
> **摘要:** Evaluating the quality of LLM-generated reasoning traces in expert domains (e.g., law) is essential for ensuring credibility and explainability, yet remains challenging due to the inherent complexity of such reasoning tasks. We introduce LEGIT (LEGal Issue Trees), a novel large-scale (24K instances) expert-level legal reasoning dataset with an emphasis on reasoning trace evaluation. We convert court judgments into hierarchical trees of opposing parties' arguments and the court's conclusions, which serve as rubrics for evaluating the issue coverage and correctness of the reasoning traces. We verify the reliability of these rubrics via human expert annotations and comparison with coarse, less informative rubrics. Using the LEGIT dataset, we show that (1) LLMs' legal reasoning ability is seriously affected by both legal issue coverage and correctness, and that (2) retrieval-augmented generation (RAG) and RL with rubrics bring complementary benefits for legal reasoning abilities, where RAG improves overall reasoning capability, whereas RL improves correctness albeit with reduced coverage.
>
---
#### [new 084] Use of Retrieval-Augmented Large Language Model Agent for Long-Form COVID-19 Fact-Checking
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文针对长篇新冠虚假信息的自动化事实核查任务，提出SAFE系统，结合检索增强生成技术（LOTR-RAG）与自检索机制（SRAG），提升大模型在一致性与可解释性上的表现。实验表明，该系统显著优于基线模型，尤其在复杂虚假信息核查中展现出更强可靠性。**

- **链接: [https://arxiv.org/pdf/2512.00007v1](https://arxiv.org/pdf/2512.00007v1)**

> **作者:** Jingyi Huang; Yuyi Yang; Mengmeng Ji; Charles Alba; Sheng Zhang; Ruopeng An
>
> **摘要:** The COVID-19 infodemic calls for scalable fact-checking solutions that handle long-form misinformation with accuracy and reliability. This study presents SAFE (system for accurate fact extraction and evaluation), an agent system that combines large language models with retrieval-augmented generation (RAG) to improve automated fact-checking of long-form COVID-19 misinformation. SAFE includes two agents - one for claim extraction and another for claim verification using LOTR-RAG, which leverages a 130,000-document COVID-19 research corpus. An enhanced variant, SAFE (LOTR-RAG + SRAG), incorporates Self-RAG to refine retrieval via query rewriting. We evaluated both systems on 50 fake news articles (2-17 pages) containing 246 annotated claims (M = 4.922, SD = 3.186), labeled as true (14.1%), partly true (14.4%), false (27.0%), partly false (2.2%), and misleading (21.0%) by public health professionals. SAFE systems significantly outperformed baseline LLMs in all metrics (p < 0.001). For consistency (0-1 scale), SAFE (LOTR-RAG) scored 0.629, exceeding both SAFE (+SRAG) (0.577) and the baseline (0.279). In subjective evaluations (0-4 Likert scale), SAFE (LOTR-RAG) also achieved the highest average ratings in usefulness (3.640), clearness (3.800), and authenticity (3.526). Adding SRAG slightly reduced overall performance, except for a minor gain in clearness. SAFE demonstrates robust improvements in long-form COVID-19 fact-checking by addressing LLM limitations in consistency and explainability. The core LOTR-RAG design proved more effective than its SRAG-augmented variant, offering a strong foundation for scalable misinformation mitigation.
>
---
#### [new 085] Securing Large Language Models (LLMs) from Prompt Injection Attacks
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文研究大语言模型（LLM）的提示注入攻击防御问题。针对任务特定微调方法JATMO的脆弱性，提出改进的遗传攻击框架HOUYI，评估其在多语言和代码干扰下的有效性。结果表明JATMO虽降低攻击成功率，但无法完全防御，且存在性能与安全性的权衡，强调需采用多层次防御策略。**

- **链接: [https://arxiv.org/pdf/2512.01326v1](https://arxiv.org/pdf/2512.01326v1)**

> **作者:** Omar Farooq Khan Suri; John McCrae
>
> **备注:** 10 pages, 1 figure, 1 table
>
> **摘要:** Large Language Models (LLMs) are increasingly being deployed in real-world applications, but their flexibility exposes them to prompt injection attacks. These attacks leverage the model's instruction-following ability to make it perform malicious tasks. Recent work has proposed JATMO, a task-specific fine-tuning approach that trains non-instruction-tuned base models to perform a single function, thereby reducing susceptibility to adversarial instructions. In this study, we evaluate the robustness of JATMO against HOUYI, a genetic attack framework that systematically mutates and optimizes adversarial prompts. We adapt HOUYI by introducing custom fitness scoring, modified mutation logic, and a new harness for local model testing, enabling a more accurate assessment of defense effectiveness. We fine-tuned LLaMA 2-7B, Qwen1.5-4B, and Qwen1.5-0.5B models under the JATMO methodology and compared them with a fine-tuned GPT-3.5-Turbo baseline. Results show that while JATMO reduces attack success rates relative to instruction-tuned models, it does not fully prevent injections; adversaries exploiting multilingual cues or code-related disruptors still bypass defenses. We also observe a trade-off between generation quality and injection vulnerability, suggesting that better task performance often correlates with increased susceptibility. Our results highlight both the promise and limitations of fine-tuning-based defenses and point toward the need for layered, adversarially informed mitigation strategies.
>
---
#### [new 086] Generative Adversarial Gumbel MCTS for Abstract Visual Composition Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究抽象视觉构图生成任务，旨在解决几何约束下基于文本目标的离散结构组合难题。提出结合蒙特卡洛树搜索与视觉语言模型的生成对抗框架，通过约束引导搜索和对抗性奖励优化，提升生成结果的可行性与语义一致性，在拼图任务中优于扩散与自回归模型。**

- **链接: [https://arxiv.org/pdf/2512.01242v1](https://arxiv.org/pdf/2512.01242v1)**

> **作者:** Zirui Zhao; Boye Niu; David Hsu; Wee Sun Lee
>
> **摘要:** We study abstract visual composition, in which identity is primarily determined by the spatial configuration and relations among a small set of geometric primitives (e.g., parts, symmetry, topology). They are invariant primarily to texture and photorealistic detail. Composing such structures from fixed components under geometric constraints and vague goal specification (such as text) is non-trivial due to combinatorial placement choices, limited data, and discrete feasibility (overlap-free, allowable orientations), which create a sparse solution manifold ill-suited to purely statistical pixel-space generators. We propose a constraint-guided framework that combines explicit geometric reasoning with neural semantics. An AlphaGo-style search enforces feasibility, while a fine-tuned vision-language model scores semantic alignment as reward signals. Our algorithm uses a policy network as a heuristic in Monte-Carlo Tree Search and fine-tunes the network via search-generated plans. Inspired by the Generative Adversarial Network, we use the generated instances for adversarial reward refinement. Over time, the generation should approach the actual data more closely when the reward model cannot distinguish between generated instances and ground-truth. In the Tangram Assembly task, our approach yields higher validity and semantic fidelity than diffusion and auto-regressive baselines, especially as constraints tighten.
>
---
#### [new 087] Measuring What LLMs Think They Do: SHAP Faithfulness and Deployability on Financial Tabular Classification
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究大语言模型（LLMs）在金融表格分类任务中的可靠性。针对其解释性与实际行为不一致的问题，通过SHAP值分析对比LLM与经典模型LightGBM的特征重要性，发现二者存在显著差异，揭示了LLMs在结构化金融建模中的局限性，但指出结合少量样本提示和可解释性改进后仍具应用潜力。**

- **链接: [https://arxiv.org/pdf/2512.00163v1](https://arxiv.org/pdf/2512.00163v1)**

> **作者:** Saeed AlMarri; Mathieu Ravaut; Kristof Juhasz; Gautier Marti; Hamdan Al Ahbabi; Ibrahim Elfadel
>
> **备注:** 7 pages, 3 figures, 3 tables, AAAI 2026 Deployable AI Workshop
>
> **摘要:** Large Language Models (LLMs) have attracted significant attention for classification tasks, offering a flexible alternative to trusted classical machine learning models like LightGBM through zero-shot prompting. However, their reliability for structured tabular data remains unclear, particularly in high stakes applications like financial risk assessment. Our study systematically evaluates LLMs and generates their SHAP values on financial classification tasks. Our analysis shows a divergence between LLMs self-explanation of feature impact and their SHAP values, as well as notable differences between LLMs and LightGBM SHAP values. These findings highlight the limitations of LLMs as standalone classifiers for structured financial modeling, but also instill optimism that improved explainability mechanisms coupled with few-shot prompting will make LLMs usable in risk-sensitive domains.
>
---
#### [new 088] LM4Opt-RA: A Multi-Candidate LLM Framework with Structured Ranking for Automating Network Resource Allocation
- **分类: cs.NI; cs.AI; cs.CL**

- **简介: 该论文针对网络资源分配优化任务，提出LM4Opt-RA框架，解决现有LLM在动态、复杂约束下建模能力不足的问题。构建NL4RA数据集，设计多候选生成与结构化排序机制，引入LAME自动化评估指标，显著提升数学建模准确性，优于基线模型。**

- **链接: [https://arxiv.org/pdf/2512.00039v1](https://arxiv.org/pdf/2512.00039v1)**

> **作者:** Tasnim Ahmed; Siana Rizwan; Naveed Ejaz; Salimur Choudhury
>
> **摘要:** Building on advancements in Large Language Models (LLMs), we can tackle complex analytical and mathematical reasoning tasks requiring nuanced contextual understanding. A prime example of such complex tasks is modelling resource allocation optimization in networks, which extends beyond translating natural language inputs into mathematical equations or Linear Programming (LP), Integer Linear Programming (ILP), and Mixed-Integer Linear Programming (MILP) models. However, existing benchmarks and datasets cannot address the complexities of such problems with dynamic environments, interdependent variables, and heterogeneous constraints. To address this gap, we introduce NL4RA, a curated dataset comprising 50 resource allocation optimization problems formulated as LP, ILP, and MILP. We then evaluate the performance of well-known open-source LLMs with varying parameter counts. To enhance existing LLM based methods, we introduce LM4Opt RA, a multi candidate framework that applies diverse prompting strategies such as direct, few shot, and chain of thought, combined with a structured ranking mechanism to improve accuracy. We identified discrepancies between human judgments and automated scoring such as ROUGE, BLEU, or BERT scores. However, human evaluation is time-consuming and requires specialized expertise, making it impractical for a fully automated end-to-end framework. To quantify the difference between LLM-generated responses and ground truth, we introduce LLM-Assisted Mathematical Evaluation (LAME), an automated metric designed for mathematical formulations. Using LM4Opt-RA, Llama-3.1-70B achieved a LAME score of 0.8007, outperforming other models by a significant margin, followed closely by Llama-3.1-8B. While baseline LLMs demonstrate considerable promise, they still lag behind human expertise; our proposed method surpasses these baselines regarding LAME and other metrics.
>
---
#### [new 089] Stabilizing Reinforcement Learning with LLMs: Formulation and Practices
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大语言模型在强化学习中的稳定训练问题。针对RL训练不稳定难题，提出基于一阶近似的理论解释，阐明为何可使用词元级代理目标优化序列级奖励，并验证重要性采样、裁剪及路由回放等技术的有效性，实验验证了其在大规模MoE模型上的稳定性与性能一致性。**

- **链接: [https://arxiv.org/pdf/2512.01374v1](https://arxiv.org/pdf/2512.01374v1)**

> **作者:** Chujie Zheng; Kai Dang; Bowen Yu; Mingze Li; Huiqiang Jiang; Junrong Lin; Yuqiong Liu; An Yang; Jingren Zhou; Junyang Lin
>
> **摘要:** This paper proposes a novel formulation for reinforcement learning (RL) with large language models, explaining why and under what conditions the true sequence-level reward can be optimized via a surrogate token-level objective in policy gradient methods such as REINFORCE. Specifically, through a first-order approximation, we show that this surrogate becomes increasingly valid only when both the training-inference discrepancy and policy staleness are minimized. This insight provides a principled explanation for the crucial role of several widely adopted techniques in stabilizing RL training, including importance sampling correction, clipping, and particularly Routing Replay for Mixture-of-Experts (MoE) models. Through extensive experiments with a 30B MoE model totaling hundreds of thousands of GPU hours, we show that for on-policy training, the basic policy gradient algorithm with importance sampling correction achieves the highest training stability. When off-policy updates are introduced to accelerate convergence, combining clipping and Routing Replay becomes essential to mitigate the instability caused by policy staleness. Notably, once training is stabilized, prolonged optimization consistently yields comparable final performance regardless of cold-start initialization. We hope that the shared insights and the developed recipes for stable RL training will facilitate future research.
>
---
#### [new 090] BackportBench: A Multilingual Benchmark for Automated Backporting of Patches
- **分类: cs.SE; cs.CL; cs.CR**

- **简介: 该论文针对软件依赖更新难的问题，提出BackportBench，首个多语言补丁回溯基准测试集，包含202个真实补丁回溯任务。通过构建可执行环境与测试用例，评估自动化回溯方法，发现代理式方法优于传统方法，尤其在需逻辑与结构修改的场景中表现更佳。**

- **链接: [https://arxiv.org/pdf/2512.01396v1](https://arxiv.org/pdf/2512.01396v1)**

> **作者:** Zhiqing Zhong; Jiaming Huang; Pinjia He
>
> **备注:** Under review
>
> **摘要:** Many modern software projects evolve rapidly to incorporate new features and security patches. It is important for users to update their dependencies to safer versions, but many still use older, vulnerable package versions because upgrading can be difficult and may break their existing codebase. Software developers can mitigate this problem by backporting security patches to older releases. However, manually backporting is time-consuming and error-prone. The effectiveness of existing automated backporting techniques on general software remains unclear since they typically target only code-hunk or function-level patch porting scenarios and are evaluated with imperfect metrics. To facilitate the development and evaluation of automated backporting techniques, we introduce BackportBench, the first comprehensive benchmark suite for patch backporting problem. BackportBench is a multilingual benchmark that contains 202 patch backporting problems from PyPI, Maven, and npm, each with executable Docker environments and relevant test cases. We evaluated existing patch porting methods and LLM-based techniques that have the potential to adapt to this task using BackportBench. The results show that the agentic method has outperformed traditional patch porting methods, especially on cases that require logical and structural changes. However, the performance varies across different programming languages. Based on the findings, we draw several implications for researchers and software practitioners in future work on automated backporting.
>
---
#### [new 091] Challenges of Heterogeneity in Big Data: A Comparative Study of Classification in Large-Scale Structured and Unstructured Domains
- **分类: cs.LG; cs.CL; cs.DC**

- **简介: 该论文研究大数据异构性对分类任务的影响，对比结构化与非结构化数据的分类策略。针对不同数据特性，结合进化与贝叶斯优化、分布式处理技术，发现高维数据中简化模型更优，文本数据则依赖特征工程提升泛化能力，提出基于数据类型与基础设施的算法选择框架。**

- **链接: [https://arxiv.org/pdf/2512.00298v1](https://arxiv.org/pdf/2512.00298v1)**

> **作者:** González Trigueros Jesús Eduardo; Alonso Sánchez Alejandro; Muñoz Rivera Emilio; Peñarán Prieto Mariana Jaqueline; Mendoza González Camila Natalia
>
> **备注:** 13 pages, 1 figure, 3 tables. Comparative study involving Apache Spark and Hyperparameter Optimization. Keywords: Big Data, NLP, Tabular Data
>
> **摘要:** This study analyzes the impact of heterogeneity ("Variety") in Big Data by comparing classification strategies across structured (Epsilon) and unstructured (Rest-Mex, IMDB) domains. A dual methodology was implemented: evolutionary and Bayesian hyperparameter optimization (Genetic Algorithms, Optuna) in Python for numerical data, and distributed processing in Apache Spark for massive textual corpora. The results reveal a "complexity paradox": in high-dimensional spaces, optimized linear models (SVM, Logistic Regression) outperformed deep architectures and Gradient Boosting. Conversely, in text-based domains, the constraints of distributed fine-tuning led to overfitting in complex models, whereas robust feature engineering -- specifically Transformer-based embeddings (ROBERTa) and Bayesian Target Encoding -- enabled simpler models to generalize effectively. This work provides a unified framework for algorithm selection based on data nature and infrastructure constraints.
>
---
#### [new 092] AlignSAE: Concept-Aligned Sparse Autoencoders
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对大语言模型知识表征难以解释的问题，提出AlignSAE方法。通过“预训练+后训练”策略，将稀疏自编码器的隐层特征对齐到人类定义的概念体系，实现特征与概念的精准绑定。实验表明，该方法支持精确的因果干预，如概念替换，提升模型可解释性与可控性。**

- **链接: [https://arxiv.org/pdf/2512.02004v1](https://arxiv.org/pdf/2512.02004v1)**

> **作者:** Minglai Yang; Xinyu Guo; Mihai Surdeanu; Liangming Pan
>
> **备注:** 20 pages, 7 figures, 5 tables
>
> **摘要:** Large Language Models (LLMs) encode factual knowledge within hidden parametric spaces that are difficult to inspect or control. While Sparse Autoencoders (SAEs) can decompose hidden activations into more fine-grained, interpretable features, they often struggle to reliably align these features with human-defined concepts, resulting in entangled and distributed feature representations. To address this, we introduce AlignSAE, a method that aligns SAE features with a defined ontology through a "pre-train, then post-train" curriculum. After an initial unsupervised training phase, we apply supervised post-training to bind specific concepts to dedicated latent slots while preserving the remaining capacity for general reconstruction. This separation creates an interpretable interface where specific relations can be inspected and controlled without interference from unrelated features. Empirical results demonstrate that AlignSAE enables precise causal interventions, such as reliable "concept swaps", by targeting single, semantically aligned slots.
>
---
#### [new 093] Probing the "Psyche'' of Large Reasoning Models: Understanding Through a Human Lens
- **分类: cs.AI; cs.CL**

- **简介: 该论文聚焦大模型推理能力分析，提出基于人类认知的五组十七类推理步骤分类体系，构建27万+标注数据集，揭示现有模型“双检”机制流于表面。通过自动标注框架CAPO实现高效分析，主张以多步反思替代简单自检，为提升推理模型训练与后训练提供可操作路径。**

- **链接: [https://arxiv.org/pdf/2512.00729v1](https://arxiv.org/pdf/2512.00729v1)**

> **作者:** Yuxiang Chen; Zuohan Wu; Ziwei Wang; Xiangning Yu; Xujia Li; Linyi Yang; Mengyue Yang; Jun Wang; Lei Chen
>
> **备注:** 13 pages
>
> **摘要:** Large reasoning models (LRMs) have garnered significant attention from researchers owing to their exceptional capability in addressing complex tasks. Motivated by the observed human-like behaviors in their reasoning processes, this paper introduces a comprehensive taxonomy to characterize atomic reasoning steps and probe the ``psyche'' of LRM intelligence. Specifically, it comprises five groups and seventeen categories derived from human mental processes, thereby grounding the understanding of LRMs in an interdisciplinary perspective. The taxonomy is then applied for an in-depth understanding of current LRMs, resulting in a distinct labeled dataset that comprises 277,534 atomic reasoning steps. Using this resource, we analyze contemporary LRMs and distill several actionable takeaways for improving training and post-training of reasoning models. Notably, our analysis reveals that prevailing post-answer ``double-checks'' (self-monitoring evaluations) are largely superficial and rarely yield substantive revisions. Thus, incentivizing comprehensive multi-step reflection, rather than simple self-monitoring, may offer a more effective path forward. To complement the taxonomy, an automatic annotation framework, named CAPO, is proposed to leverage large language models (LLMs) for generating the taxonomy-based annotations. Experimental results demonstrate that CAPO achieves higher consistency with human experts compared to baselines, facilitating a scalable and comprehensive analysis of LRMs from a human cognitive perspective. Together, the taxonomy, CAPO, and the derived insights provide a principled, scalable path toward understanding and advancing LRM reasoning.
>
---
#### [new 094] H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文研究大语言模型中的幻觉问题，提出“H-Neurons”概念，揭示极少数神经元（<0.1%）可预测幻觉生成。通过识别、行为验证与溯源分析，发现这些神经元在预训练阶段形成，且因果关联于过度顺从行为。研究连接宏观现象与微观机制，为提升模型可靠性提供新路径。**

- **链接: [https://arxiv.org/pdf/2512.01797v1](https://arxiv.org/pdf/2512.01797v1)**

> **作者:** Cheng Gao; Huimin Chen; Chaojun Xiao; Zhiyi Chen; Zhiyuan Liu; Maosong Sun
>
> **备注:** 20 pages, 4 figures
>
> **摘要:** Large language models (LLMs) frequently generate hallucinations -- plausible but factually incorrect outputs -- undermining their reliability. While prior work has examined hallucinations from macroscopic perspectives such as training data and objectives, the underlying neuron-level mechanisms remain largely unexplored. In this paper, we conduct a systematic investigation into hallucination-associated neurons (H-Neurons) in LLMs from three perspectives: identification, behavioral impact, and origins. Regarding their identification, we demonstrate that a remarkably sparse subset of neurons (less than $0.1\%$ of total neurons) can reliably predict hallucination occurrences, with strong generalization across diverse scenarios. In terms of behavioral impact, controlled interventions reveal that these neurons are causally linked to over-compliance behaviors. Concerning their origins, we trace these neurons back to the pre-trained base models and find that these neurons remain predictive for hallucination detection, indicating they emerge during pre-training. Our findings bridge macroscopic behavioral patterns with microscopic neural mechanisms, offering insights for developing more reliable LLMs.
>
---
#### [new 095] One Swallow Does Not Make a Summer: Understanding Semantic Structures in Embedding Spaces
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对嵌入空间语义结构不透明的问题，提出语义场子空间（SFS）和SAFARI算法，通过新度量“语义迁移”揭示层次化语义结构。工作包括构建几何保真的上下文感知表示、设计高效无监督算法，并在多模态数据上验证其在分类与偏见检测中的优越性与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.00852v1](https://arxiv.org/pdf/2512.00852v1)**

> **作者:** Yandong Sun; Qiang Huang; Ziwei Xu; Yiqun Sun; Yixuan Tang; Anthony K. H. Tung
>
> **摘要:** Embedding spaces are fundamental to modern AI, translating raw data into high-dimensional vectors that encode rich semantic relationships. Yet, their internal structures remain opaque, with existing approaches often sacrificing semantic coherence for structural regularity or incurring high computational overhead to improve interpretability. To address these challenges, we introduce the Semantic Field Subspace (SFS), a geometry-preserving, context-aware representation that captures local semantic neighborhoods within the embedding space. We also propose SAFARI (SemAntic Field subspAce deteRmInation), an unsupervised, modality-agnostic algorithm that uncovers hierarchical semantic structures using a novel metric called Semantic Shift, which quantifies how semantics evolve as SFSes evolve. To ensure scalability, we develop an efficient approximation of Semantic Shift that replaces costly SVD computations, achieving a 15~30x speedup with average errors below 0.01. Extensive evaluations across six real-world text and image datasets show that SFSes outperform standard classifiers not only in classification but also in nuanced tasks such as political bias detection, while SAFARI consistently reveals interpretable and generalizable semantic hierarchies. This work presents a unified framework for structuring, analyzing, and scaling semantic understanding in embedding spaces.
>
---
#### [new 096] Agentic Policy Optimization via Instruction-Policy Co-Evolution
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对大语言模型在强化学习中因静态指令导致的性能瓶颈，提出INSPO框架，实现指令与策略的协同进化。通过动态优化指令池并结合经验回放进行策略反思，使模型自主发现更优推理路径，在多轮任务中显著提升性能，仅增加微量计算开销。**

- **链接: [https://arxiv.org/pdf/2512.01945v1](https://arxiv.org/pdf/2512.01945v1)**

> **作者:** Han Zhou; Xingchen Wan; Ivan Vulić; Anna Korhonen
>
> **备注:** 10 pages, 3 figures, 2 tables (18 pages including references and appendices)
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has advanced the reasoning capability of large language models (LLMs), enabling autonomous agents that can conduct effective multi-turn and tool-integrated reasoning. While instructions serve as the primary protocol for defining agents, RLVR typically relies on static and manually designed instructions. However, those instructions may be suboptimal for the base model, and the optimal instruction may change as the agent's policy improves and explores the interaction with the environment. To bridge the gap, we introduce INSPO, a novel Instruction-Policy co-evolution framework that integrates instruction optimization as a dynamic component of the reinforcement learning (RL) loop. INSPO maintains a dynamic population of instruction candidates that are sampled with questions, where reward signals in RL loops are automatically attributed to each instruction, and low performers are periodically pruned. New instructions are generated and verified through an on-policy reflection mechanism, where an LLM-based optimizer analyzes past experience from a replay buffer and evolves more effective strategies given the current policy. We conduct extensive experiments on multi-turn retrieval and reasoning tasks, demonstrating that INSPO substantially outperforms strong baselines relying on static instructions. INSPO discovers innovative instructions that guide the agent toward more strategic reasoning paths, achieving substantial performance gains with only a marginal increase in computational overhead.
>
---
#### [new 097] LLM CHESS: Benchmarking Reasoning and Instruction-Following in LLMs through Chess
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出LLM CHESS框架，通过棋类博弈评估大模型的推理与指令遵循能力。针对现有基准静态、易过拟合的问题，构建动态、随机对抗环境，量化模型表现并生成可比性Elo评分。实验揭示顶尖模型在持续博弈中仍存短板，验证了框架的有效性与挑战性。**

- **链接: [https://arxiv.org/pdf/2512.01992v1](https://arxiv.org/pdf/2512.01992v1)**

> **作者:** Sai Kolasani; Maxim Saplin; Nicholas Crispino; Kyle Montgomery; Jared Quincy Davis; Matei Zaharia; Chi Wang; Chenguang Wang
>
> **摘要:** We introduce LLM CHESS, an evaluation framework designed to probe the generalization of reasoning and instruction-following abilities in large language models (LLMs) through extended agentic interaction in the domain of chess. We rank over 50 open and closed source models by playing against a random opponent using a range of behavioral metrics, including win and loss rates, move quality, move legality, hallucinated actions, and game duration. For a subset of top reasoning models, we derive an Elo estimate by playing against a chess engine with variably configured skill, which allows for comparisons between models in an easily understandable way. Despite the simplicity of the instruction-following task and the weakness of the opponent, many state-of-the-art models struggle to complete games or achieve consistent wins. Similar to other benchmarks on complex reasoning tasks, our experiments reveal a clear separation between reasoning and non-reasoning models. However, unlike existing static benchmarks, the stochastic and dynamic nature of LLM CHESS uniquely reduces overfitting and memorization while preventing benchmark saturation, proving difficult even for top reasoning models. To support future work on evaluating reasoning and instruction-following in LLMs, we release our experimental framework, a public leaderboard, and a dataset of associated games.
>
---
#### [new 098] Large Language Models Cannot Reliably Detect Vulnerabilities in JavaScript: The First Systematic Benchmark and Evaluation
- **分类: cs.CR; cs.CL; cs.SE**

- **简介: 该论文针对大语言模型（LLM）在JavaScript漏洞检测中的可靠性问题，提出系统性评估框架。针对现有基准的覆盖不全、标签不合理及案例不真实三大缺陷，构建了符合三原则的ARENAJS基准与JUDGEJS评估框架，首次系统评测七款主流LLM，发现其推理与鲁棒性均严重不足，表明LLM尚难可靠检测JavaScript漏洞。**

- **链接: [https://arxiv.org/pdf/2512.01255v1](https://arxiv.org/pdf/2512.01255v1)**

> **作者:** Qingyuan Fei; Xin Liu; Song Li; Shujiang Wu; Jianwei Hou; Ping Chen; Zifeng Kang
>
> **摘要:** Researchers have proposed numerous methods to detect vulnerabilities in JavaScript, especially those assisted by Large Language Models (LLMs). However, the actual capability of LLMs in JavaScript vulnerability detection remains questionable, necessitating systematic evaluation and comprehensive benchmarks. Unfortunately, existing benchmarks suffer from three critical limitations: (1) incomplete coverage, such as covering a limited subset of CWE types; (2) underestimation of LLM capabilities caused by unreasonable ground truth labeling; and (3) overestimation due to unrealistic cases such as using isolated vulnerable files rather than complete projects. In this paper, we introduce, for the first time, three principles for constructing a benchmark for JavaScript vulnerability detection that directly address these limitations: (1) comprehensiveness, (2) no underestimation, and (3) no overestimation. Guided by these principles, we propose FORGEJS, the first automatic benchmark generation framework for evaluating LLMs' capability in JavaScript vulnerability detection. Then, we use FORGEJS to construct ARENAJS-the first systematic benchmark for LLM-based JavaScript vulnerability detection-and further propose JUDGEJS, an automatic evaluation framework. We conduct the first systematic evaluation of LLMs for JavaScript vulnerability detection, leveraging JUDGEJS to assess seven popular commercial LLMs on ARENAJS. The results show that LLMs not only exhibit limited reasoning capabilities, but also suffer from severe robustness defects, indicating that reliable JavaScript vulnerability detection with LLMs remains an open challenge.
>
---
#### [new 099] Whose Personae? Synthetic Persona Experiments in LLM Research and Pathways to Transparency
- **分类: cs.CY; cs.CL**

- **简介: 该论文研究大语言模型对齐中的合成人格实验，针对人格代表性与生态效度不足的问题，分析63项研究发现任务与目标人群常未明确。提出透明度检查清单，强调数据驱动的代表性采样与生态效度，以提升评估严谨性。**

- **链接: [https://arxiv.org/pdf/2512.00461v1](https://arxiv.org/pdf/2512.00461v1)**

> **作者:** Jan Batzner; Volker Stocker; Bingjun Tang; Anusha Natarajan; Qinhao Chen; Stefan Schmid; Gjergji Kasneci
>
> **备注:** Published at AAAI/ACM AIES 2025. Presented at NeurIPS 2025 Workshop Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
>
> **摘要:** Synthetic personae experiments have become a prominent method in Large Language Model alignment research, yet the representativeness and ecological validity of these personae vary considerably between studies. Through a review of 63 peer-reviewed studies published between 2023 and 2025 in leading NLP and AI venues, we reveal a critical gap: task and population of interest are often underspecified in persona-based experiments, despite personalization being fundamentally dependent on these criteria. Our analysis shows substantial differences in user representation, with most studies focusing on limited sociodemographic attributes and only 35% discussing the representativeness of their LLM personae. Based on our findings, we introduce a persona transparency checklist that emphasizes representative sampling, explicit grounding in empirical data, and enhanced ecological validity. Our work provides both a comprehensive assessment of current practices and practical guidelines to improve the rigor and ecological validity of persona-based evaluations in language model alignment research.
>
---
#### [new 100] Melody or Machine: Detecting Synthetic Music with Dual-Stream Contrastive Learning
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文针对AI生成音乐的检测任务，解决现有模型在新生成器上泛化能力差的问题。提出大规模基准MoM和双流对比学习架构CLAM，通过分析人声与乐器间的机器不一致特征，提升对合成音乐的检测精度，实现F1 0.925的新纪录。**

- **链接: [https://arxiv.org/pdf/2512.00621v1](https://arxiv.org/pdf/2512.00621v1)**

> **作者:** Arnesh Batra; Dev Sharma; Krish Thukral; Ruhani Bhatia; Naman Batra; Aditya Gautam
>
> **备注:** Accepted at Transactions on Machine Learning Research (TMLR)
>
> **摘要:** The rapid evolution of end-to-end AI music generation poses an escalating threat to artistic authenticity and copyright, demanding detection methods that can keep pace. While foundational, existing models like SpecTTTra falter when faced with the diverse and rapidly advancing ecosystem of new generators, exhibiting significant performance drops on out-of-distribution (OOD) content. This generalization failure highlights a critical gap: the need for more challenging benchmarks and more robust detection architectures. To address this, we first introduce Melody or Machine (MoM), a new large-scale benchmark of over 130,000 songs (6,665 hours). MoM is the most diverse dataset to date, built with a mix of open and closed-source models and a curated OOD test set designed specifically to foster the development of truly generalizable detectors. Alongside this benchmark, we introduce CLAM, a novel dual-stream detection architecture. We hypothesize that subtle, machine-induced inconsistencies between vocal and instrumental elements, often imperceptible in a mixed signal, offer a powerful tell-tale sign of synthesis. CLAM is designed to test this hypothesis by employing two distinct pre-trained audio encoders (MERT and Wave2Vec2) to create parallel representations of the audio. These representations are fused by a learnable cross-aggregation module that models their inter-dependencies. The model is trained with a dual-loss objective: a standard binary cross-entropy loss for classification, complemented by a contrastive triplet loss which trains the model to distinguish between coherent and artificially mismatched stream pairings, enhancing its sensitivity to synthetic artifacts without presuming a simple feature alignment. CLAM establishes a new state-of-the-art in synthetic music forensics. It achieves an F1 score of 0.925 on our challenging MoM benchmark.
>
---
#### [new 101] Text Mining Analysis of Symptom Patterns in Medical Chatbot Conversations
- **分类: cs.LG; cs.CL; cs.IR**

- **简介: 该论文属于医疗对话中的症状模式挖掘任务，旨在从聊天机器人对话中提取并分析患者自述症状的结构与关联。研究基于多轮医患对话数据，结合LDA、K-Means、NER和Apriori等方法，识别症状主题、聚类相似描述并发现高频症状对，揭示了可作为早期诊断信号的临床模式。**

- **链接: [https://arxiv.org/pdf/2512.00768v1](https://arxiv.org/pdf/2512.00768v1)**

> **作者:** Hamed Razavi
>
> **备注:** 9 pages, 4 tables
>
> **摘要:** The fast growth of digital health systems has led to a need to better comprehend how they interpret and represent patient-reported symptoms. Chatbots have been used in healthcare to provide clinical support and enhance the user experience, making it possible to provide meaningful clinical patterns from text-based data through chatbots. The proposed research utilises several different natural language processing methods to study the occurrences of symptom descriptions in medicine as well as analyse the patterns that emerge through these conversations within medical bots. Through the use of the Medical Conversations to Disease Dataset which contains 960 multi-turn dialogues divided into 24 Clinical Conditions, a standardised representation of conversations between patient and bot is created for further analysis by computational means. The multi-method approach uses a variety of tools, including Latent Dirichlet Allocation (LDA) to identify latent symptom themes, K-Means to group symptom descriptions by similarity, Transformer-based Named Entity Recognition (NER) to extract medical concepts, and the Apriori algorithm to discover frequent symptom pairs. Findings from the analysis indicate a coherent structure of clinically relevant topics, moderate levels of clustering cohesiveness and several high confidence rates on the relationships between symptoms like fever headache and rash itchiness. The results support the notion that conversational medical data can be a valuable diagnostic signal for early symptom interpretation, assist in strengthening decision support and improve how users interact with tele-health technology. By demonstrating a method for converting unstructured free-flowing dialogue into actionable knowledge regarding symptoms this work provides an extensible framework to further enhance future performance, dependability and clinical utility of selecting medical chatbots.
>
---
#### [new 102] Associative Syntax and Maximal Repetitions reveal context-dependent complexity in fruit bat communication
- **分类: cs.LG; cs.CL; cs.IT; q-bio.QM**

- **简介: 该论文研究果蝠通信中的复杂性，属于自然语言处理中的语音分析任务。针对非离散语音系统难以标注的问题，提出基于流形学习的无监督标注方法，结合最大重复序列（MRs）与音节转移网络，揭示了果蝠在冲突情境下具有更高复杂性的关联性语法特征，表明沟通复杂度随情境变化而提升。**

- **链接: [https://arxiv.org/pdf/2512.01033v1](https://arxiv.org/pdf/2512.01033v1)**

> **作者:** Luigi Assom
>
> **备注:** Accepted for a lightning talk at the NeurIPS 2025 Workshop: "AI for Non-Human Animal Communication"
>
> **摘要:** This study presents an unsupervised method to infer discreteness, syntax and temporal structures of fruit-bats vocalizations, as a case study of graded vocal systems, and evaluates the complexity of communication patterns in relation with behavioral context. The method improved the baseline for unsupervised labeling of vocal units (i.e. syllables) through manifold learning, by investigating how dimen- sionality reduction on mel-spectrograms affects labeling, and comparing it with unsupervised labels based on acoustic similarity. We then encoded vocalizations as syllabic sequences to analyze the type of syntax, and extracted the Maximal Repetitions (MRs) to evaluate syntactical structures. We found evidence for: i) associative syntax, rather than combinatorial (context classification is unaffected by permutation of sequences, F 1 > 0.9); ii) context-dependent use of syllables (Wilcoxon rank-sum tests, p-value < 0.05); iii) heavy-tail distribution of MRs (truncated power-law, exponent α < 2), indicative of mechanism encoding com- binatorial complexity. Analysis of MRs and syllabic transition networks revealed that mother-pupil interactions were characterized by repetitions, while commu- nication in conflict-contexts exhibited higher complexity (longer MRs and more interconnected vocal sequences) than non-agonistic contexts. We propose that communicative complexity is higher in scenarios of disagreement, reflecting lower compressibility of information.
>
---
#### [new 103] Statistical NLP for Optimization of Clinical Trial Success Prediction in Pharmaceutical R&D
- **分类: cs.LG; cs.CL; q-bio.QM**

- **简介: 该论文针对神经科学领域临床试验成功率低的问题，提出基于统计NLP与BioBERT的分类模型，通过提取临床试验文本特征，预测技术与监管成功概率。旨在提升研发决策效率，优化资源分配。**

- **链接: [https://arxiv.org/pdf/2512.00586v1](https://arxiv.org/pdf/2512.00586v1)**

> **作者:** Michael R. Doane
>
> **备注:** Doctor of Engineering Praxis Dissertation, The George Washington University. 122 pages. Present affiliation: Iambic Therapeutics
>
> **摘要:** This work presents the development and evaluation of an NLP-enabled probabilistic classifier designed to estimate the probability of technical and regulatory success (pTRS) for clinical trials in the field of neuroscience. While pharmaceutical R&D is plagued by high attrition rates and enormous costs, particularly within neuroscience, where success rates are below 10%, timely identification of promising programs can streamline resource allocation and reduce financial risk. Leveraging data from the ClinicalTrials.gov database and success labels from the recently developed Clinical Trial Outcome dataset, the classifier extracts text-based clinical trial features using statistical NLP techniques. These features were integrated into several non-LLM frameworks (logistic regression, gradient boosting, and random forest) to generate calibrated probability scores. Model performance was assessed on a retrospective dataset of 101,145 completed clinical trials spanning 1976-2024, achieving an overall ROC-AUC of 0.64. An LLM-based predictive model was then built using BioBERT, a domain-specific language representation encoder. The BioBERT-based model achieved an overall ROC-AUC of 0.74 and a Brier Score of 0.185, indicating its predictions had, on average, 40% less squared error than would be observed using industry benchmarks. The BioBERT-based model also made trial outcome predictions that were superior to benchmark values 70% of the time overall. By integrating NLP-driven insights into drug development decision-making, this work aims to enhance strategic planning and optimize investment allocation in neuroscience programs.
>
---
#### [new 104] LPCD: Unified Framework from Layer-Wise to Submodule Quantization
- **分类: stat.ML; cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对大模型后训练量化（PTQ）中仅关注单层而忽略复杂子模块的问题，提出LPCD统一框架。通过在任意子模块上优化松弛目标并投影至层级量化器，实现从层到子模块的统一量化，有效提升多架构、多比特下的量化性能，兼顾效率与兼容性。**

- **链接: [https://arxiv.org/pdf/2512.01546v1](https://arxiv.org/pdf/2512.01546v1)**

> **作者:** Yuma Ichikawa; Yudai Fujimoto; Akira Sakai
>
> **备注:** 21 pages, 4 figures
>
> **摘要:** Post-training quantization (PTQ) aims to preserve model-level behavior; however, most methods focus on individual linear layers. Even recent extensions, such as QEP and LoaQ, which mitigate error propagation or target specific submodules, still rely on layer-wise formulations and fail to capture the behavior of larger submodules. We introduce Layer-Projected Coordinate Descent (LPCD), a unified framework that extends PTQ beyond layers by optimizing relaxed objectives across arbitrary submodules and projecting the solutions with layer-wise quantizers. LPCD generalizes existing methods and provides a principled approach to quantizing complex submodules while maintaining the efficiency and compatibility of layer-wise PTQ pipelines. Across diverse LLM architectures and bit-widths, LPCD-based submodule quantization consistently enhances both layer-wise PTQ methods and existing submodule approaches.
>
---
#### [new 105] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出StreamGaze，首个评估多模态大模型在流式视频中利用眼动信号进行时序推理与主动理解的基准。针对现有工作缺乏对眼动引导推理的评测，构建了融合眼动轨迹的时空对齐问答数据集，揭示当前模型在眼动感知、意图预测上的显著不足，并提供深入分析以指导未来研究。**

- **链接: [https://arxiv.org/pdf/2512.01707v1](https://arxiv.org/pdf/2512.01707v1)**

> **作者:** Daeun Lee; Subhojyoti Mukherjee; Branislav Kveton; Ryan A. Rossi; Viet Dac Lai; Seunghyun Yoon; Trung Bui; Franck Dernoncourt; Mohit Bansal
>
> **备注:** Project page: https://streamgaze.github.io/
>
> **摘要:** Streaming video understanding requires models not only to process temporally incoming frames, but also to anticipate user intention for realistic applications like AR glasses. While prior streaming benchmarks evaluate temporal reasoning, none measure whether MLLMs can interpret or leverage human gaze signals within a streaming setting. To fill this gap, we introduce StreamGaze, the first benchmark designed to evaluate how effectively MLLMs use gaze for temporal and proactive reasoning in streaming videos. StreamGaze introduces gaze-guided past, present, and proactive tasks that comprehensively evaluate streaming video understanding. These tasks assess whether models can use real-time gaze to follow shifting attention and infer user intentions from only past and currently observed frames. To build StreamGaze, we develop a gaze-video QA generation pipeline that aligns egocentric videos with raw gaze trajectories via fixation extraction, region-specific visual prompting, and scanpath construction. This pipeline produces spatio-temporally grounded QA pairs that closely reflect human perceptual dynamics. Across all StreamGaze tasks, we observe substantial performance gaps between state-of-the-art MLLMs and human performance, revealing fundamental limitations in gaze-based temporal reasoning, intention modeling, and proactive prediction. We further provide detailed analyses of gaze-prompting strategies, reasoning behaviors, and task-specific failure modes, offering deeper insight into why current MLLMs struggle and what capabilities future models must develop. All data and code will be publicly released to support continued research in gaze-guided streaming video understanding.
>
---
#### [new 106] Generalized Medical Phrase Grounding
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出广义医学短语定位（GMPG）任务，解决传统系统仅支持单区域定位且无法处理非诊断、多区域或不可定位短语的问题。作者构建MedGrounder模型，采用两阶段训练，实现零样本迁移，有效定位多区域及非地面短语，减少人工标注依赖，并可与报告生成器组合生成带标注报告。**

- **链接: [https://arxiv.org/pdf/2512.01085v1](https://arxiv.org/pdf/2512.01085v1)**

> **作者:** Wenjun Zhang; Shekhar S. Chandra; Aaron Nicolson
>
> **摘要:** Medical phrase grounding (MPG) maps textual descriptions of radiological findings to corresponding image regions. These grounded reports are easier to interpret, especially for non-experts. Existing MPG systems mostly follow the referring expression comprehension (REC) paradigm and return exactly one bounding box per phrase. Real reports often violate this assumption. They contain multi-region findings, non-diagnostic text, and non-groundable phrases, such as negations or descriptions of normal anatomy. Motivated by this, we reformulate the task as generalised medical phrase grounding (GMPG), where each sentence is mapped to zero, one, or multiple scored regions. To realise this formulation, we introduce the first GMPG model: MedGrounder. We adopted a two-stage training regime: pre-training on report sentence--anatomy box alignment datasets and fine-tuning on report sentence--human annotated box datasets. Experiments on PadChest-GR and MS-CXR show that MedGrounder achieves strong zero-shot transfer and outperforms REC-style and grounded report generation baselines on multi-region and non-groundable phrases, while using far fewer human box annotations. Finally, we show that MedGrounder can be composed with existing report generators to produce grounded reports without retraining the generator.
>
---
#### [new 107] Mode-Conditioning Unlocks Superior Test-Time Scaling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对测试时扩展中因模式坍塌导致的多样性不足问题，提出模式条件化（ModC）框架，通过分配计算资源至不同推理模式，提升模型在图搜索和大模型推理任务中的表现。无需标签即可利用梯度聚类实现，显著提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2512.01127v1](https://arxiv.org/pdf/2512.01127v1)**

> **作者:** Chen Henry Wu; Sachin Goyal; Aditi Raghunathan
>
> **摘要:** Parallel sampling promises substantial gains in test-time scaling, but its effectiveness is sharply limited by diversity collapse, where models concentrate on a few modes and repeated samples produce the same mistakes. We propose the mode-conditioning (ModC) framework, which explicitly allocates test-time compute across reasoning modes using either specialist models or mode-specific prefixes. ModC consistently improves scaling across controlled graph-search tasks and large-scale reasoning benchmarks, spanning model families and sizes from 0.5B to 7B. On OpenThoughts, fine-tuning Qwen2.5-7B with ModC achieves a 4x efficiency gain over standard training while also improving the maximum attainable Pass@k. We further show that gradient clustering enables ModC without explicit mode labels, yielding up to 10% gains on datasets such as NuminaMath. Finally, we show that ModC improves reinforcement learning (RL) and can further boost diversity-inducing RL methods. These results demonstrate that standard training underutilizes the diversity in data, and that ModC provides a simple, effective remedy for unlocking the full benefits of diversity in test-time scaling.
>
---
#### [new 108] Progressive Code Integration for Abstractive Bug Report Summarization
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文针对缺陷报告摘要任务，解决现有方法忽视代码片段、依赖表面文本导致摘要不完整的问题。提出渐进式代码融合框架，将长代码片段分步融入大模型输入，突破上下文窗口限制，联合文本与代码生成语义丰富摘要，显著提升摘要质量。**

- **链接: [https://arxiv.org/pdf/2512.00325v1](https://arxiv.org/pdf/2512.00325v1)**

> **作者:** Shaira Sadia Karim; Abrar Mahmud Rahim; Lamia Alam; Ishmam Tashdeed; Lutfun Nahar Lota; Md. Abu Raihan M. Kamal; Md. Azam Hossain
>
> **摘要:** Bug reports are often unstructured and verbose, making it challenging for developers to efficiently comprehend software issues. Existing summarization approaches typically rely on surface-level textual cues, resulting in incomplete or redundant summaries, and they frequently ignore associated code snippets, which are essential for accurate defect diagnosis. To address these limitations, we propose a progressive code-integration framework for LLM-based abstractive bug report summarization. Our approach incrementally incorporates long code snippets alongside textual content, overcoming standard LLM context window constraints and producing semantically rich summaries. Evaluated on four benchmark datasets using eight LLMs, our pipeline outperforms extractive baselines by 7.5%-58.2% and achieves performance comparable to state-of-the-art abstractive methods, highlighting the benefits of jointly leveraging textual and code information for enhanced bug comprehension.
>
---
#### [new 109] Testing the Machine Consciousness Hypothesis
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; cs.NE; q-bio.NC**

- **简介: 该论文探究机器意识假说，旨在验证意识是否为无特定硬件基础的计算系统中由集体智能通过通信同步预测而涌现的属性。研究构建基于元胞自动机的分布式预测模型系统，通过模拟局部观察者间信息交换，探索自指性自我模型如何在去中心化环境中形成，以建立可实证的机器意识理论。**

- **链接: [https://arxiv.org/pdf/2512.01081v1](https://arxiv.org/pdf/2512.01081v1)**

> **作者:** Stephen Fitz
>
> **摘要:** The Machine Consciousness Hypothesis states that consciousness is a substrate-free functional property of computational systems capable of second-order perception. I propose a research program to investigate this idea in silico by studying how collective self-models (coherent, self-referential representations) emerge from distributed learning systems embedded within universal self-organizing environments. The theory outlined here starts from the supposition that consciousness is an emergent property of collective intelligence systems undergoing synchronization of prediction through communication. It is not an epiphenomenon of individual modeling but a property of the language that a system evolves to internally describe itself. For a model of base reality, I begin with a minimal but general computational world: a cellular automaton, which exhibits both computational irreducibility and local reducibility. On top of this computational substrate, I introduce a network of local, predictive, representational (neural) models capable of communication and adaptation. I use this layered model to study how collective intelligence gives rise to self-representation as a direct consequence of inter-agent alignment. I suggest that consciousness does not emerge from modeling per se, but from communication. It arises from the noisy, lossy exchange of predictive messages between groups of local observers describing persistent patterns in the underlying computational substrate (base reality). It is through this representational dialogue that a shared model arises, aligning many partial views of the world. The broader goal is to develop empirically testable theories of machine consciousness, by studying how internal self-models may form in distributed systems without centralized control.
>
---
#### [new 110] Chain-of-Ground: Improving GUI Grounding via Iterative Reasoning and Reference Feedback
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文针对GUI接地任务中模型对微小或相似目标定位不准、现实布局模糊等问题，提出无需训练的多步迭代推理框架Chain-of-Ground。通过引导模型逐步反思与修正假设，提升定位精度与可解释性，在ScreenSpot Pro和TPanel UI数据集上显著优于基线，验证了结构化迭代优化的有效性。**

- **链接: [https://arxiv.org/pdf/2512.01979v1](https://arxiv.org/pdf/2512.01979v1)**

> **作者:** Aiden Yiliu Li; Bizhi Yu; Daoan Lei; Tianhe Ren; Shilong Liu
>
> **摘要:** GUI grounding aims to align natural language instructions with precise regions in complex user interfaces. Advanced multimodal large language models show strong ability in visual GUI grounding but still struggle with small or visually similar targets and ambiguity in real world layouts. These limitations arise from limited grounding capacity and from underuse of existing reasoning potential. We present Chain of Ground CoG a training free multi step grounding framework that uses multimodal large language models for iterative visual reasoning and refinement. Instead of direct prediction the model progressively reflects and adjusts its hypotheses leading to more accurate and interpretable localization. Our approach achieves 68.4 accuracy on the ScreenSpot Pro benchmark an improvement of 4.8 points. To measure real world generalization we introduce TPanel UI a dataset of 420 labeled industrial control panels with visual distortions such as blur and masking. On TPanel UI Chain of Ground improves over the strong baseline Qwen3 VL 235B by 6.9 points showing the effectiveness of multi step training free grounding across real world and digital interfaces. These results highlight a direction for unlocking grounding potential through structured iterative refinement instead of additional training.
>
---
#### [new 111] EmoRAG: Evaluating RAG Robustness to Symbolic Perturbations
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文研究RAG系统在符号扰动下的鲁棒性，聚焦于微小表情符号（如"(@_@)"）导致的严重误检问题。通过实验证明，单个表情符号即可几乎100%误导检索，且大模型更脆弱。提出机制分析与防御策略，揭示当前RAG系统存在重大安全漏洞。**

- **链接: [https://arxiv.org/pdf/2512.01335v1](https://arxiv.org/pdf/2512.01335v1)**

> **作者:** Xinyun Zhou; Xinfeng Li; Yinan Peng; Ming Xu; Xuanwang Zhang; Miao Yu; Yidong Wang; Xiaojun Jia; Kun Wang; Qingsong Wen; XiaoFeng Wang; Wei Dong
>
> **备注:** Accepted to ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) 2026
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems are increasingly central to robust AI, enhancing large language model (LLM) faithfulness by incorporating external knowledge. However, our study unveils a critical, overlooked vulnerability: their profound susceptibility to subtle symbolic perturbations, particularly through near-imperceptible emoticon tokens such as "(@_@)" that can catastrophically mislead retrieval, termed EmoRAG. We demonstrate that injecting a single emoticon into a query makes it nearly 100% likely to retrieve semantically unrelated texts that contain a matching emoticon. Our extensive experiment across general question-answering and code domains, using a range of state-of-the-art retrievers and generators, reveals three key findings: (I) Single-Emoticon Disaster: Minimal emoticon injections cause maximal disruptions, with a single emoticon almost 100% dominating RAG output. (II) Positional Sensitivity: Placing an emoticon at the beginning of a query can cause severe perturbation, with F1-Scores exceeding 0.92 across all datasets. (III) Parameter-Scale Vulnerability: Counterintuitively, models with larger parameters exhibit greater vulnerability to the interference. We provide an in-depth analysis to uncover the underlying mechanisms of these phenomena. Furthermore, we raise a critical concern regarding the robustness assumption of current RAG systems, envisioning a threat scenario where an adversary exploits this vulnerability to manipulate the RAG system. We evaluate standard defenses and find them insufficient against EmoRAG. To address this, we propose targeted defenses, analyzing their strengths and limitations in mitigating emoticon-based perturbations. Finally, we outline future directions for building robust RAG systems.
>
---
#### [new 112] Breaking It Down: Domain-Aware Semantic Segmentation for Retrieval Augmented Generation
- **分类: cs.IR; cs.CL**

- **简介: 该论文研究RAG中的文档分块任务，针对传统分块方法破坏语义结构的问题，提出两种领域感知的语义分块方法PSC和MFC。基于PubMed数据训练，通过增强PubMedQA评估其对检索与生成的影响，结果表明方法显著提升检索性能并具有良好泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.00367v1](https://arxiv.org/pdf/2512.00367v1)**

> **作者:** Aparajitha Allamraju; Maitreya Prafulla Chitale; Hiranmai Sri Adibhatla; Rahul Mishra; Manish Shrivastava
>
> **摘要:** Document chunking is a crucial component of Retrieval-Augmented Generation (RAG), as it directly affects the retrieval of relevant and precise context. Conventional fixed-length and recursive splitters often produce arbitrary, incoherent segments that fail to preserve semantic structure. Although semantic chunking has gained traction, its influence on generation quality remains underexplored. This paper introduces two efficient semantic chunking methods, Projected Similarity Chunking (PSC) and Metric Fusion Chunking (MFC), trained on PubMed data using three different embedding models. We further present an evaluation framework that measures the effect of chunking on both retrieval and generation by augmenting PubMedQA with full-text PubMed Central articles. Our results show substantial retrieval improvements (24x with PSC) in MRR and higher Hits@k on PubMedQA. We provide a comprehensive analysis, including statistical significance and response-time comparisons with common chunking libraries. Despite being trained on a single domain, PSC and MFC also generalize well, achieving strong out-of-domain generation performance across multiple datasets. Overall, our findings confirm that our semantic chunkers, especially PSC, consistently deliver superior performance.
>
---
#### [new 113] ZIP-RC: Zero-overhead Inference-time Prediction of Reward and Cost for Adaptive and Interpretable Generation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出ZIP-RC，一种零开销的推理时奖励与成本预测方法。针对大语言模型缺乏自我反思能力、无法动态调整生成策略的问题，通过复用原有计算资源，在不增加额外开销的前提下，实时预测生成结果的奖励与成本，实现自适应、可解释的推理生成，显著提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2512.01457v1](https://arxiv.org/pdf/2512.01457v1)**

> **作者:** Rohin Manvi; Joey Hong; Tim Seyde; Maxime Labonne; Mathias Lechner; Sergey Levine
>
> **备注:** Code coming soon
>
> **摘要:** Large language models excel at reasoning but lack key aspects of introspection, including anticipating their own success and the computation required to achieve it. Humans use real-time introspection to decide how much effort to invest, when to make multiple attempts, when to stop, and when to signal success or failure. Without this, LLMs struggle to make intelligent meta-cognition decisions. Test-time scaling methods like Best-of-N drive up cost and latency by using a fixed budget of samples regardless of the marginal benefit of each one at any point in generation, and the absence of confidence signals can mislead people, prevent appropriate escalation to better tools, and undermine trustworthiness. Learned verifiers or reward models can provide confidence estimates, but do not enable adaptive inference and add substantial cost by requiring extra models or forward passes. We present ZIP-RC, an adaptive inference method that equips models with zero-overhead inference-time predictions of reward and cost. At every token, ZIP-RC reuses reserved or unused logits in the same forward pass as next-token prediction to output a joint distribution over final reward and remaining length -- no extra models, architecture change, or inference overhead. This full joint distribution is used to compute a sampling utility which is the linear combination of the expected maximum reward, total compute, and latency of set of samples if generated to completion. During inference, we maximize this utility with meta-actions that determine which prefix of tokens to continue or initiate sampling from. On mixed-difficulty mathematical benchmarks, ZIP-RC improves accuracy by up to 12% over majority voting at equal or lower average cost, and traces smooth Pareto frontiers between quality, compute, and latency. By providing real-time reward-cost introspection, ZIP-RC enables adaptive, efficient reasoning.
>
---
#### [new 114] Pay Attention Later: From Vector Space Diffusion to Linearithmic Spectral Phase-Locking
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对Transformer模型在知识更新中面临的“可塑性-稳定性困境”，提出PRISM模型。通过将语义编码为复数域谐振频率，用线性对数复杂度的门控谐波卷积替代自注意力机制，实现无损快速适应新概念，显著优于标准Transformer在动态知识注入下的表现。**

- **链接: [https://arxiv.org/pdf/2512.01208v1](https://arxiv.org/pdf/2512.01208v1)**

> **作者:** Alper Yıldırım; İbrahim Yücedağ
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Standard Transformers suffer from a "Semantic Alignment Tax", a prohibitive optimization cost required to organize a chaotic initialization into a coherent geometric map via local gradient diffusion. We hypothesize that this reliance on diffusive learning creates "Catastrophic Rigidity", rendering models unable to adapt to novel concepts without destroying their pre-trained reasoning capabilities. To isolate this phenomenon, we introduce Iterative Semantic Map Refinement (ISMR), a diagnostic protocol revealing that alignment is a fixed geometric barrier that scaling cannot solve; a 20-layer model overcomes this barrier no faster than a 1-layer model. We introduce the Phase-Resonant Intelligent Spectral Model (PRISM). PRISM encodes semantic identity as resonant frequencies in the complex domain (C^d) and replaces quadratic self-attention with linearithmic O(N log N) Gated Harmonic Convolutions. We validate PRISM on the WMT14 translation task. While the Standard Transformer maintains a slight edge in general competence on static benchmarks (23.88 vs 21.40 BLEU), it fails the "Plasticity-Stability" stress test completely. When injected with novel concepts, the Transformer suffers Catastrophic Forgetting, degrading by -10.55 BLEU points while achieving only 60% acquisition. In contrast, PRISM demonstrates Lossless Plasticity, achieving 96% 5-shot acquisition with negligible degradation (-0.84 BLEU). These results suggest that harmonic representations effectively decouple memory from reasoning, offering a structural solution to the plasticity-stability dilemma in real-time knowledge adaptation.
>
---
#### [new 115] LEC: Linear Expectation Constraints for False-Discovery Control in Selective Prediction and Routing Systems
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对大语言模型预测不可靠的问题，提出LEC方法，通过线性期望约束实现选择性预测与路由系统的错误发现率（FDR）控制。其核心是基于校准数据计算FDR约束下的最优阈值，提升预测可靠性与样本保留率，并设计双模型路由机制，在保持统一FDR保障下提高准确率与覆盖率。**

- **链接: [https://arxiv.org/pdf/2512.01556v1](https://arxiv.org/pdf/2512.01556v1)**

> **作者:** Zhiyuan Wang; Aniri; Tianlong Chen; Yue Zhang; Heng Tao Shen; Xiaoshuang Shi; Kaidi Xu
>
> **摘要:** Large language models (LLMs) often generate unreliable answers, while heuristic uncertainty methods fail to fully distinguish correct from incorrect predictions, causing users to accept erroneous answers without statistical guarantees. We address this issue through the lens of false discovery rate (FDR) control, ensuring that among all accepted predictions, the proportion of errors does not exceed a target risk level. To achieve this in a principled way, we propose LEC, which reinterprets selective prediction as a constrained decision problem by enforcing a Linear Expectation Constraint over selection and error indicators. Then, we establish a finite-sample sufficient condition, which relies only on a held-out set of exchangeable calibration samples, to compute an FDR-constrained, coverage-maximizing threshold. Furthermore, we extend LEC to a two-model routing mechanism: given a prompt, if the current model's uncertainty exceeds its calibrated threshold, we delegate it to a stronger model, while maintaining a unified FDR guarantee. Evaluations on closed-ended and open-ended question-answering (QA) datasets show that LEC achieves tighter FDR control and substantially improves sample retention over prior methods. Moreover, the two-model routing mechanism achieves lower risk levels while accepting more correct samples than each individual model.
>
---
#### [new 116] Rep3Net: An Approach Exploiting Multimodal Representation for Molecular Bioactivity Prediction
- **分类: cs.LG; cs.CL; q-bio.QM**

- **简介: 该论文针对药物发现中分子生物活性预测任务，解决传统QSAR模型难以捕捉分子结构与上下文信息的问题。提出Rep3Net，融合分子描述符、图结构表示及ChemBERTa生成的SMILES嵌入，实现多模态特征融合，显著提升PARP-1靶点预测性能。**

- **链接: [https://arxiv.org/pdf/2512.00521v1](https://arxiv.org/pdf/2512.00521v1)**

> **作者:** Sabrina Islam; Md. Atiqur Rahman; Md. Bakhtiar Hasan; Md. Hasanul Kabir
>
> **摘要:** In early stage drug discovery, bioactivity prediction of molecules against target proteins plays a crucial role. Trdaitional QSAR models that utilizes molecular descriptor based data often struggles to predict bioactivity of molecules effectively due to its limitation in capturing structural and contextual information embedded within each compound. To address this challenge, we propose Rep3Net, a unified deep learning architecture that not only incorporates descriptor data but also includes spatial and relational information through graph-based represenation of compounds and contextual information through ChemBERTa generated embeddings from SMILES strings. Our model employing multimodal concatenated features produce reliable bioactivity prediction on Poly [ADP-ribose] polymerase 1 (PARP-1) dataset. PARP-1 is a crucial agent in DNA damage repair and has become a significant theraputic target in malignancies that depend on it for survival and growth. A comprehensive analysis and comparison with conventional standalone models including GCN, GAT, XGBoost, etc. demonstrates that our architecture achieves the highest predictive performance. In computational screening of compounds in drug discovery, our architecture provides a scalable framework for bioactivity prediction.
>
---
#### [new 117] VeriPy - A New Python-Based Approach for SDR Pipelined/Unrolled Hardware Accelerator Generation
- **分类: cs.AR; cs.CL**

- **简介: 该论文提出VeriPy，一种基于Python的HLS工具，旨在降低通信工程师使用硬件加速器的门槛。针对SDR应用中硬件设计复杂、需专业知识的问题，VeriPy可自动生成流水线/展开架构的Verilog代码，支持自动测试、资源估算与优化，显著提升频率性能，代码简洁无需低层知识。**

- **链接: [https://arxiv.org/pdf/2512.00006v1](https://arxiv.org/pdf/2512.00006v1)**

> **作者:** Yuqin Zhao; Linghui Ye; Haihang Xia; Luke Seed; Tiantai Deng
>
> **备注:** 13 Pages, 16 figures, and 9 tables. Aim to submit to IEEE TCAD
>
> **摘要:** Software-defined radio (SDR) plays an important role in the communication field by providing a flexible and customized communication system for different purposes according to the needs. To enhance the performance of SDR applications, hardware accelerators have been widely deployed in recent years. In facing this obstacle, a necessity arises for a high-level synthesis (HLS) tool specifically designed for communication engineers without detailed hardware knowledge. To lower the barrier between SDR engineers and hardware development, this work proposed a Python-based HLS tool, VeriPy, which can generate both mainstream architecture for hardware accelerators in Verilog specifically for SDR designs including unrolled design and pipelined design, requiring no detailed digital hardware knowledge or Hardware Description Languages (HDL). Furthermore, VeriPy supports automatic testbench generation with random input stimulus, an extensible hardware library, performance and resource estimation, and offers strong optimisation potential at both the algorithmic and digital hardware levels. The generated hardware design by VeriPy can achieve up to 70% faster operating frequency compared to pragma-optimised Vivado HLS designs with a reasonably higher resource con-sumption while delivering comparable performance and resource consumption to hand-coded implementations. Regarding code complexity, VeriPy requires no pragmas, completely eliminating the need for low-level hardware knowledge. For straightforward algorithms, the input code length remains comparable to that of Vivado HLS.
>
---
## 更新

#### [replaced 001] From Topology to Retrieval: Decoding Embedding Spaces with Unified Signatures
- **分类: cs.LG; cs.CL; cs.IR**

- **简介: 该论文研究文本嵌入空间的拓扑与几何结构，旨在解决嵌入空间表征不清晰、度量冗余的问题。提出统一拓扑签名（UTS）框架，通过多属性分析揭示模型架构影响，并成功预测文档可检索性，提升模型可解释性与下游任务性能。**

- **链接: [https://arxiv.org/pdf/2511.22150v2](https://arxiv.org/pdf/2511.22150v2)**

> **作者:** Florian Rottach; William Rudman; Bastian Rieck; Harrisen Scells; Carsten Eickhoff
>
> **摘要:** Studying how embeddings are organized in space not only enhances model interpretability but also uncovers factors that drive downstream task performance. In this paper, we present a comprehensive analysis of topological and geometric measures across a wide set of text embedding models and datasets. We find a high degree of redundancy among these measures and observe that individual metrics often fail to sufficiently differentiate embedding spaces. Building on these insights, we introduce Unified Topological Signatures (UTS), a holistic framework for characterizing embedding spaces. We show that UTS can predict model-specific properties and reveal similarities driven by model architecture. Further, we demonstrate the utility of our method by linking topological structure to ranking effectiveness and accurately predicting document retrievability. We find that a holistic, multi-attribute perspective is essential to understanding and leveraging the geometry of text embeddings.
>
---
#### [replaced 002] Prompt-R1: Collaborative Automatic Prompting Framework via End-to-end Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出Prompt-R1，一种基于端到端强化学习的协作式自动提示框架。针对用户难以生成有效提示导致大模型性能受限的问题，利用小模型生成优化提示，与大模型协同完成复杂任务。通过双约束奖励机制提升推理准确性和生成质量，实现高效、可插拔的自动提示生成。**

- **链接: [https://arxiv.org/pdf/2511.01016v4](https://arxiv.org/pdf/2511.01016v4)**

> **作者:** Wenjin Liu; Haoran Luo; Xueyuan Lin; Haoming Liu; Tiesunlong Shen; Jiapu Wang; Rui Mao; Erik Cambria
>
> **摘要:** Recently, advanced large language models (LLMs) have emerged at an increasingly rapid pace. However, when faced with complex problems, most users are often unable to provide accurate and effective prompts to interact with LLMs, thus limiting the performance of LLMs. To address this challenge, we propose Prompt-R1, an end-to-end reinforcement learning framework that uses a small-scale LLM to collaborate with large-scale LLMs, replacing user interaction to solve problems better. This collaboration is cast as a multi-turn prompt interaction, where the small-scale LLM thinks and generates prompts, and the large-scale LLM performs complex reasoning. A dual-constrained reward is designed to optimize for correctness, generation quality, and reasoning accuracy. Prompt-R1 provides a plug-and-play framework that supports both inference and training with various large-scale LLMs. Experiments on multiple public datasets show that Prompt-R1 significantly outperforms baseline models across tasks. Our code is publicly available at https://github.com/QwenQKing/Prompt-R1.
>
---
#### [replaced 003] MERA Code: A Unified Framework for Evaluating Code Generation Across Tasks
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出MERA Code，一个针对俄语代码生成的统一评估框架，涵盖11项任务和8种语言。旨在解决现有评估忽视代码质量与实际应用的问题，通过开源代码库、评分系统及排行榜，全面评测LLMs在非英语环境下的编码能力，推动模型评估标准化与研究发展。**

- **链接: [https://arxiv.org/pdf/2507.12284v3](https://arxiv.org/pdf/2507.12284v3)**

> **作者:** Artem Chervyakov; Alexander Kharitonov; Pavel Zadorozhny; Adamenko Pavel; Rodion Levichev; Dmitrii Vorobev; Dmitrii Salikhov; Aidar Valeev; Alena Pestova; Maria Dziuba; Ilseyar Alimova; Artem Zavgorodnev; Aleksandr Medvedev; Stanislav Moiseev; Elena Bruches; Daniil Grebenkin; Roman Derunets; Vikulov Vladimir; Anton Emelyanov; Dmitrii Babaev; Vladimir V. Ivanov; Valentin Malykh; Alena Fenogenova
>
> **摘要:** Advancements in LLMs have enhanced task automation in software engineering; however, current evaluations primarily focus on natural language tasks, overlooking code quality. Most benchmarks prioritize high-level reasoning over executable code and real-world performance, leaving gaps in understanding true capabilities and risks associated with these models in production. To address this issue, we propose MERA Code, a new addition to the MERA benchmark family, specifically focused on evaluating code for the latest code generation LLMs in Russian. This benchmark includes 11 evaluation tasks that span 8 programming languages. Our proposed evaluation methodology features a taxonomy that outlines the practical coding skills necessary for models to complete these tasks. The benchmark comprises an open-source codebase for users to conduct MERA assessments, a scoring system compatible with various programming environments, and a platform featuring a leaderboard and submission system. We evaluate open LLMs and frontier API models, analyzing their limitations in terms of practical coding tasks in non-English languages. We are publicly releasing MERA to guide future research, anticipate groundbreaking features in model development, and standardize evaluation procedures.
>
---
#### [replaced 004] Teaching Language Models to Critique via Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对代码生成中模型输出质量难以持续提升的问题，提出CTRL框架，通过强化学习训练语言模型作为批评者，自动生成能提升生成结果的反馈。无需人工标注，该方法显著提高通过率并减少错误累积，实现测试时迭代优化，最大提升达106.1%。**

- **链接: [https://arxiv.org/pdf/2502.03492v2](https://arxiv.org/pdf/2502.03492v2)**

> **作者:** Zhihui Xie; Jie Chen; Liyu Chen; Weichao Mao; Jingjing Xu; Lingpeng Kong
>
> **摘要:** Teaching large language models (LLMs) to critique and refine their outputs is crucial for building systems that can iteratively improve, yet it is fundamentally limited by the ability to provide accurate judgments and actionable suggestions. In this work, we study LLM critics for code generation and propose $\texttt{CTRL}$, a framework for $\texttt{C}$ritic $\texttt{T}$raining via $\texttt{R}$einforcement $\texttt{L}$earning, which trains a critic model to generate feedback that maximizes correction performance for a fixed generator model without human supervision. Our results demonstrate that critics trained with $\texttt{CTRL}$ significantly enhance pass rates and mitigate compounding errors across both base and stronger generator models. Furthermore, we show that these critic models act as accurate generative reward models and enable test-time scaling through iterative critique-revision, achieving up to 106.1% relative improvements across challenging code generation benchmarks.
>
---
#### [replaced 005] The AI Productivity Index (APEX)
- **分类: econ.GN; cs.AI; cs.CL; cs.HC**

- **简介: 该论文提出扩展版AI生产力指数（APEX-v1-extended），评估前沿模型在投资银行、管理咨询、律师和医生四类职业中的经济价值任务表现。通过扩大测试集至400例并优化评分方法，发现模型仍存显著局限。研究开源100个非基准案例及评估工具，以推动相关领域研究。**

- **链接: [https://arxiv.org/pdf/2509.25721v3](https://arxiv.org/pdf/2509.25721v3)**

> **作者:** Bertie Vidgen; Abby Fennelly; Evan Pinnix; Julien Bencheck; Daniyal Khan; Zach Richards; Austin Bridges; Calix Huang; Ben Hunsberger; Isaac Robinson; Akul Datta; Chirag Mahapatra; Dominic Barton; Cass R. Sunstein; Eric Topol; Brendan Foody; Osvald Nitski
>
> **摘要:** We present an extended version of the AI Productivity Index (APEX-v1-extended), a benchmark for assessing whether frontier models are capable of performing economically valuable tasks in four jobs: investment banking associate, management consultant, big law associate, and primary care physician (MD). This technical report details the extensions to APEX-v1, including an increase in the held-out evaluation set from n = 50 to n = 100 cases per job (n = 400 total) and updates to the grading methodology. We present a new leaderboard, where GPT5 (Thinking = High) remains the top performing model with a score of 67.0%. APEX-v1-extended shows that frontier models still have substantial limitations when performing typical professional tasks. To support further research, we are open sourcing n = 25 non-benchmark example cases per role (n = 100 total) along with our evaluation harness.
>
---
#### [replaced 006] Comparative Evaluation of Expressive Japanese Character Text-to-Speech with VITS and Style-BERT-VITS2
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文针对日语角色语音合成任务，比较VITS与Style-BERT-VITS2 JP Extra模型。针对日语特有的高低音敏感与风格多样性问题，基于三个角色数据集，评估自然度、可懂度与说话人一致性。结果表明，SBV2JE在自然度和可懂度上接近真人，且支持音高控制，适用于语言学习与角色对话生成。**

- **链接: [https://arxiv.org/pdf/2505.17320v2](https://arxiv.org/pdf/2505.17320v2)**

> **作者:** Zackary Rackauckas; Julia Hirschberg
>
> **备注:** Accepted to IEEE UEMCON 2025
>
> **摘要:** Synthesizing expressive Japanese character speech poses unique challenges due to pitch-accent sensitivity and stylistic variability. This paper empirically evaluates two open-source text-to-speech models--VITS and Style-BERT-VITS2 JP Extra (SBV2JE)--on in-domain, character-driven Japanese speech. Using three character-specific datasets, we evaluate models across naturalness (mean opinion and comparative mean opinion score), intelligibility (word error rate), and speaker consistency. SBV2JE matches human ground truth in naturalness (MOS 4.37 vs. 4.38), achieves lower WER, and shows slight preference in CMOS. Enhanced by pitch-accent controls and a WavLM-based discriminator, SBV2JE proves effective for applications like language learning and character dialogue generation, despite higher computational demands.
>
---
#### [replaced 007] Multilingual DistilWhisper: Efficient Distillation of Multi-task Speech Models via Language-Specific Experts
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文针对多语言语音识别中低资源语言性能不足的问题，提出DistilWhisper方法。通过引入语言特定专家与知识蒸馏，对Whisper-small进行轻量级微调，有效提升多语言ASR性能，同时保持多任务、多语言特性，参数开销极小。**

- **链接: [https://arxiv.org/pdf/2311.01070v4](https://arxiv.org/pdf/2311.01070v4)**

> **作者:** Thomas Palmeira Ferraz; Marcely Zanon Boito; Caroline Brun; Vassilina Nikoulina
>
> **备注:** Accepted and Presented at IEEE ICASSP 2024 (this is extended version)
>
> **摘要:** Whisper is a multitask and multilingual speech model covering 99 languages. It yields commendable automatic speech recognition (ASR) results in a subset of its covered languages, but the model still underperforms on a non-negligible number of under-represented languages, a problem exacerbated in smaller model versions. In this work, we propose DistilWhisper, an approach able to bridge the performance gap in ASR for these languages while retaining the advantages of multitask and multilingual capabilities. Our approach involves two key strategies: lightweight modular ASR fine-tuning of whisper-small using language-specific experts, and knowledge distillation from whisper-large-v2. This dual approach allows us to effectively boost ASR performance while keeping the robustness inherited from the multitask and multilingual pre-training. Results demonstrate that our approach is more effective than standard fine-tuning or LoRA adapters, boosting performance in the targeted languages for both in- and out-of-domain test sets, while introducing only a negligible parameter overhead at inference.
>
---
#### [replaced 008] SRPO: Self-Referential Policy Optimization for Vision-Language-Action Models
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文针对视觉-语言-动作模型在机器人操作中依赖专家示范、奖励稀疏的问题，提出自参照策略优化（SRPO）。通过利用模型自身生成的成功轨迹作为参考，结合世界模型的潜在表示，为失败轨迹赋予渐进式奖励，实现高效无监督强化学习。在LIBERO基准上，200步内达99.2%成功率，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.15605v2](https://arxiv.org/pdf/2511.15605v2)**

> **作者:** Senyu Fei; Siyin Wang; Li Ji; Ao Li; Shiduo Zhang; Liming Liu; Jinlong Hou; Jingjing Gong; Xianzhong Zhao; Xipeng Qiu
>
> **摘要:** Vision-Language-Action (VLA) models excel in robotic manipulation but are constrained by their heavy reliance on expert demonstrations, leading to demonstration bias and limiting performance. Reinforcement learning (RL) is a vital post-training strategy to overcome these limits, yet current VLA-RL methods, including group-based optimization approaches, are crippled by severe reward sparsity. Relying on binary success indicators wastes valuable information in failed trajectories, resulting in low training efficiency. To solve this, we propose Self-Referential Policy Optimization (SRPO), a novel VLA-RL framework. SRPO eliminates the need for external demonstrations or manual reward engineering by leveraging the model's own successful trajectories, generated within the current training batch, as a self-reference. This allows us to assign a progress-wise reward to failed attempts. A core innovation is the use of latent world representations to measure behavioral progress robustly. Instead of relying on raw pixels or requiring domain-specific fine-tuning, we utilize the compressed, transferable encodings from a world model's latent space. These representations naturally capture progress patterns across environments, enabling accurate, generalized trajectory comparison. Empirical evaluations on the LIBERO benchmark demonstrate SRPO's efficiency and effectiveness. Starting from a supervised baseline with 48.9% success, SRPO achieves a new state-of-the-art success rate of 99.2% in just 200 RL steps, representing a 103% relative improvement without any extra supervision. Furthermore, SRPO shows substantial robustness, achieving a 167% performance improvement on the LIBERO-Plus benchmark.
>
---
#### [replaced 009] NarraBench: A Comprehensive Framework for Narrative Benchmarking
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出NarraBench，一个理论驱动的叙事理解任务分类体系，系统梳理78个现有基准。针对当前评估体系覆盖不足、主观性与视角类任务缺失等问题，指出仅27%的任务被充分涵盖，强调需发展更全面的评测框架，以提升大模型叙事理解能力的评估效果。**

- **链接: [https://arxiv.org/pdf/2510.09869v3](https://arxiv.org/pdf/2510.09869v3)**

> **作者:** Sil Hamilton; Matthew Wilkens; Andrew Piper
>
> **摘要:** We present NarraBench, a theory-informed taxonomy of narrative-understanding tasks, as well as an associated survey of 78 existing benchmarks in the area. We find significant need for new evaluations covering aspects of narrative understanding that are either overlooked in current work or are poorly aligned with existing metrics. Specifically, we estimate that only 27% of narrative tasks are well captured by existing benchmarks, and we note that some areas -- including narrative events, style, perspective, and revelation -- are nearly absent from current evaluations. We also note the need for increased development of benchmarks capable of assessing constitutively subjective and perspectival aspects of narrative, that is, aspects for which there is generally no single correct answer. Our taxonomy, survey, and methodology are of value to NLP researchers seeking to test LLM narrative understanding.
>
---
#### [replaced 010] HEALTH-PARIKSHA: Assessing RAG Models for Health Chatbots in Real-World Multilingual Settings
- **分类: cs.CL**

- **简介: 该论文研究多语言医疗聊天机器人中检索增强生成模型的性能。针对真实世界多语言数据，评估24个LLM在印度英语及4种印地语系语言中的表现，发现指令微调模型在印地语上表现不佳，且印地语查询的事实准确率低于英文，代码混用和文化相关性带来挑战。**

- **链接: [https://arxiv.org/pdf/2410.13671v3](https://arxiv.org/pdf/2410.13671v3)**

> **作者:** Varun Gumma; Ananditha Raghunath; Mohit Jain; Sunayana Sitaram
>
> **摘要:** Assessing the capabilities and limitations of large language models (LLMs) has garnered significant interest, yet the evaluation of multiple models in real-world scenarios remains rare. Multilingual evaluation often relies on translated benchmarks, which typically do not capture linguistic and cultural nuances present in the source language. This study provides an extensive assessment of 24 LLMs on real world data collected from Indian patients interacting with a medical chatbot in Indian English and 4 other Indic languages. We employ a uniform Retrieval Augmented Generation framework to generate responses, which are evaluated using both automated techniques and human evaluators on four specific metrics relevant to our application. We find that models vary significantly in their performance and that instruction tuned Indic models do not always perform well on Indic language queries. Further, we empirically show that factual correctness is generally lower for responses to Indic queries compared to English queries. Finally, our qualitative work shows that code-mixed and culturally relevant queries in our dataset pose challenges to evaluated models.
>
---
#### [replaced 011] Attributional Safety Failures in Large Language Models under Code-Mixed Perturbations
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在多语混用输入下的安全失效问题。针对模型在非英语环境下安全防护能力骤降的现象，提出“归因漂移”解释机制，并设计轻量级翻译恢复策略，显著提升多语环境下的安全性，推动更公平可靠的AI安全防护。**

- **链接: [https://arxiv.org/pdf/2505.14469v2](https://arxiv.org/pdf/2505.14469v2)**

> **作者:** Somnath Banerjee; Pratyush Chatterjee; Shanu Kumar; Sayan Layek; Parag Agrawal; Rima Hazra; Animesh Mukherjee
>
> **摘要:** While LLMs appear robustly safety-aligned in English, we uncover a catastrophic, overlooked weakness: attributional collapse under code-mixed perturbations. Our systematic evaluation of open models shows that the linguistic camouflage of code-mixing -- ``blending languages within a single conversation'' -- can cause safety guardrails to fail dramatically. Attack success rates (ASR) spike from a benign 9\% in monolingual English to 69\% under code-mixed inputs, with rates exceeding 90\% in non-Western contexts such as Arabic and Hindi. These effects hold not only on controlled synthetic datasets but also on real-world social media traces, revealing a serious risk for billions of users. To explain why this happens, we introduce saliency drift attribution (SDA), an interpretability framework that shows how, under code-mixing, the model's internal attention drifts away from safety-critical tokens (e.g., ``violence'' or ``corruption''), effectively blinding it to harmful intent. Finally, we propose a lightweight translation-based restoration strategy that recovers roughly 80\% of the safety lost to code-mixing, offering a practical path toward more equitable and robust LLM safety.
>
---
#### [replaced 012] Chain-of-Query: Unleashing the Power of LLMs in SQL-Aided Table Understanding via Multi-Agent Collaboration
- **分类: cs.CL; cs.AI; cs.DB**

- **简介: 该论文针对表格理解任务中LLM因结构复杂性导致的推理困难问题，提出Chain-of-Query（CoQ）多智能体框架。通过自然语言表模式表示、分条款生成SQL及混合推理分工，提升查询质量并降低无效SQL率，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2508.15809v3](https://arxiv.org/pdf/2508.15809v3)**

> **作者:** Songyuan Sui; Hongyi Liu; Serena Liu; Li Li; Soo-Hyun Choi; Rui Chen; Xia Hu
>
> **备注:** AACL 2025 Main Conference (Oral)
>
> **摘要:** Table understanding requires structured, multi-step reasoning. Large Language Models (LLMs) struggle with it due to the structural complexity of tabular data. Recently, multi-agent frameworks for SQL generation have shown promise in tackling the challenges of understanding tabular data, but existing approaches often suffer from limitations such as the inability to comprehend table structure for reliable SQL generation, error propagation that results in invalid queries, and over-reliance on execution correctness. To address these issues, we propose Chain-of-Query (CoQ), a novel multi-agent framework for SQL-aided table understanding. CoQ adopts natural-language-style representations of table schemas to abstract away structural noise and enhance understanding. It employs a clause-by-clause SQL generation strategy to improve query quality and introduces a hybrid reasoning division that separates SQL-based mechanical reasoning from LLM-based logical inference, thereby reducing reliance on execution outcomes. Extensive experiments across four models and five widely used benchmarks demonstrate that CoQ achieves substantial accuracy improvements and significantly lowers invalid SQL rates compared to prior generic LLM-based, SQL-aided, and hybrid baselines, confirming its superior effectiveness in table understanding. The code is available at https://github.com/SongyuanSui/ChainofQuery.
>
---
#### [replaced 013] DPRM: A Dual Implicit Process Reward Model in Multi-Hop Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对多跳问答（MHQA）中的推理过程评估问题，提出双隐式过程奖励模型DPRM。它通过无标注的隐式奖励建模，分别学习文本链与知识图谱的推理路径，并引入一致性约束，实现两者的协同优化，有效提升推理质量与答案准确率。**

- **链接: [https://arxiv.org/pdf/2511.08364v2](https://arxiv.org/pdf/2511.08364v2)**

> **作者:** Xinyi Wang; Yiping Song; Zhiliang Tian; Bo Liu; Tingjin Luo; Minlie Huang
>
> **摘要:** In multi-hop question answering (MHQA) tasks, Chain of Thought (CoT) improves the quality of generation by guiding large language models (LLMs) through multi-step reasoning, and Knowledge Graphs (KGs) reduce hallucinations via semantic matching. Outcome Reward Models (ORMs) provide feedback after generating the final answers but fail to evaluate the process for multi-step reasoning. Traditional Process Reward Models (PRMs) evaluate the reasoning process but require costly human annotations or rollout generation. While implicit PRM is trained only with outcome signals and derives step rewards through reward parameterization without explicit annotations, it is more suitable for multi-step reasoning in MHQA tasks. However, existing implicit PRM has only been explored for plain text scenarios. When adapting to MHQA tasks, it cannot handle the graph structure constraints in KGs and capture the potential inconsistency between CoT and KG paths. To address these limitations, we propose the DPRM (Dual Implicit Process Reward Model). It trains two implicit PRMs for CoT and KG reasoning in MHQA tasks. Both PRMs, namely KG-PRM and CoT-PRM, derive step-level rewards from outcome signals via reward parameterization without additional explicit annotations. Among them, KG-PRM uses preference pairs to learn structural constraints from KGs. DPRM further introduces a consistency constraint between CoT and KG reasoning steps, making the two PRMs mutually verify and collaboratively optimize the reasoning paths. We also provide a theoretical demonstration of the derivation of process rewards. Experimental results show that our method outperforms 13 baselines on multiple datasets with up to 16.6% improvement on Hit@1.
>
---
#### [replaced 014] SpeechRole: A Large-Scale Dataset and Benchmark for Evaluating Speech Role-Playing Agents
- **分类: cs.CL**

- **简介: 该论文针对语音角色扮演（SRPA）缺乏系统评估的问题，构建了SpeechRole-Data大规模语音数据集，涵盖98个角色的11.2万条对话，并提出SpeechRole-Eval多维度评估基准。旨在推动语音驱动的角色扮演研究，解决现有方法在语音风格一致性和角色连贯性上的挑战。**

- **链接: [https://arxiv.org/pdf/2508.02013v4](https://arxiv.org/pdf/2508.02013v4)**

> **作者:** Changhao Jiang; Jiajun Sun; Yifei Cao; Jiabao Zhuang; Hui Li; Baoyu Fan; Tao Ji; Tao Gui; Qi Zhang
>
> **摘要:** Recently, role-playing agents have emerged as a promising paradigm for achieving personalized interaction and emotional resonance. Existing research primarily focuses on the textual modality, neglecting the critical dimension of speech in realistic interactive scenarios. In particular, there is a lack of systematic evaluation for Speech Role-Playing Agents (SRPAs). To address this gap, we construct SpeechRole-Data, a large-scale, high-quality dataset that comprises 98 diverse roles and 112k speech-based single-turn and multi-turn conversations. Each role demonstrates distinct vocal characteristics, including timbre and prosody, thereby enabling more sophisticated speech role-playing. Furthermore, we propose SpeechRole-Eval, a multidimensional evaluation benchmark that systematically assesses SRPAs performance in key aspects such as fundamental interaction ability, speech expressiveness, and role-playing fidelity. Experimental results reveal the advantages and challenges of both cascaded and end-to-end speech role-playing agents in maintaining vocal style consistency and role coherence. We release all data, code, and baseline models to provide a solid foundation for speech-driven multimodal role-playing research and to foster further developments in this field.
>
---
#### [replaced 015] Persistent Instability in LLM's Personality Measurements: Effects of Scale, Reasoning, and Conversation History
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLM）人格测量的持续不稳定性问题，属于模型行为一致性评估任务。针对现有模型在不同设置下人格表现波动大的问题，通过构建PERSIST框架，系统测试25个模型在多种条件下的表现，发现模型规模、推理模式、对话历史等均无法有效提升人格测量稳定性，揭示当前对齐策略在安全应用中的局限性。**

- **链接: [https://arxiv.org/pdf/2508.04826v2](https://arxiv.org/pdf/2508.04826v2)**

> **作者:** Tommaso Tosato; Saskia Helbling; Yorguin-Jose Mantilla-Ramos; Mahmood Hegazy; Alberto Tosato; David John Lemay; Irina Rish; Guillaume Dumas
>
> **备注:** Accepted at AAAI 2026, Track on AI Alignment
>
> **摘要:** Large language models require consistent behavioral patterns for safe deployment, yet there are indications of large variability that may lead to an instable expression of personality traits in these models. We present PERSIST (PERsonality Stability in Synthetic Text), a comprehensive evaluation framework testing 25 open-source models (1B-685B parameters) across 2 million+ responses. Using traditional (BFI, SD3) and novel LLM-adapted personality questionnaires, we systematically vary model size, personas, reasoning modes, question order or paraphrasing, and conversation history. Our findings challenge fundamental assumptions: (1) Question reordering alone can introduce large shifts in personality measurements; (2) Scaling provides limited stability gains: even 400B+ models exhibit standard deviations >0.3 on 5-point scales; (3) Interventions expected to stabilize behavior, such as reasoning and inclusion of conversation history, can paradoxically increase variability; (4) Detailed persona instructions produce mixed effects, with misaligned personas showing significantly higher variability than the helpful assistant baseline; (5) The LLM-adapted questionnaires, despite their improved ecological validity, exhibit instability comparable to human-centric versions. This persistent instability across scales and mitigation strategies suggests that current LLMs lack the architectural foundations for genuine behavioral consistency. For safety-critical applications requiring predictable behavior, these findings indicate that current alignment strategies may be inadequate.
>
---
#### [replaced 016] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLM）在对抗攻击下语义置信度的鲁棒性问题。针对置信度易受扰动和越狱攻击影响导致误判的问题，提出两类攻击框架，验证了现有防御机制无效。研究揭示当前置信度表达脆弱，亟需构建更可靠的置信机制。**

- **链接: [https://arxiv.org/pdf/2507.06489v2](https://arxiv.org/pdf/2507.06489v2)**

> **作者:** Stephen Obadinma; Xiaodan Zhu
>
> **备注:** Published in NeurIPS 2025
>
> **摘要:** Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to help ensure transparency, trust, and safety in many applications, including those involving human-AI interactions. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce attack frameworks targeting verbal confidence scores through both perturbation and jailbreak-based methods, and demonstrate that these attacks can significantly impair verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current verbal confidence is vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the need to design robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.
>
---
#### [replaced 017] Explainable Semantic Text Relations: A Question-Answering Framework for Comparing Document Content
- **分类: cs.CL**

- **简介: 该论文研究文本间语义关系的可解释判定任务。针对文档内容对比中难以精确识别信息重叠、包含或差异的问题，提出基于可回答问题集（AQS）的集合论框架，定义等价、包含、互交等语义关系，并利用AQS差集提供解释。构建了可控的合成数据集，评估模型在捕捉深层语义关系上的能力，公开数据与代码以促进研究。**

- **链接: [https://arxiv.org/pdf/2509.08304v2](https://arxiv.org/pdf/2509.08304v2)**

> **作者:** Yehudit Aperstein; Alon Gottlib; Gal Benita; Alexander Apartsin
>
> **备注:** 18 pages, 1 figure
>
> **摘要:** Understanding semantic relations between two texts is crucial for many information and document management tasks, in which one must determine whether the content fully overlaps, is completely superseded by another document, or overlaps only partially, with unique information in each. Beyond establishing this relation, it is equally important to provide explainable outputs that specify which pieces of information are present, missing, or newly added between the text pair. In this study, we formally define semantic relations between two texts through the set-theoretic relation between their respective Answerable Question Sets (AQS), the sets of questions each text can answer. Under this formulation, Semantic Text Relation (STR), such as equivalence, inclusion, and mutual overlap, becomes a well-defined set relation between the corresponding texts' AQSs. The set differences between the AQSs also serve as an explanation or diagnostic tool for identifying how the information in the texts diverges. Using this definition, we construct a synthetic benchmark that captures fine-grained informational relations through controlled paraphrasing and deliberate information removal supported by AQS manipulations. We then use this dataset to evaluate several discriminative and generative models for classifying text pairs into STR categories, assessing how well different model architectures capture semantic relations beyond surface-level similarity. We publicly release both the dataset and the data generation code to support further research.
>
---
#### [replaced 018] Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对对话句向量学习任务，解决标注成本高、缺乏有效监督信号的问题。提出TaDSE模型，利用模板等词级先验知识，通过对比学习框架生成高质量句向量，并构建多样化增强数据集提升性能。实验表明其优于现有方法。**

- **链接: [https://arxiv.org/pdf/2305.14299v2](https://arxiv.org/pdf/2305.14299v2)**

> **作者:** Minsik Oh; Jiwei Li; Guoyin Wang
>
> **摘要:** Learning high quality sentence embeddings from dialogues has drawn increasing attentions as it is essential to solve a variety of dialogue-oriented tasks with low annotation cost. Annotating and gathering utterance relationships in conversations are difficult, while token-level annotations, \eg, entities, slots and templates, are much easier to obtain. Other sentence embedding methods are usually sentence-level self-supervised frameworks and cannot utilize token-level extra knowledge. We introduce Template-aware Dialogue Sentence Embedding (TaDSE), a novel augmentation method that utilizes template information to learn utterance embeddings via self-supervised contrastive learning framework. We further enhance the effect with a synthetically augmented dataset that diversifies utterance-template association, in which slot-filling is a preliminary step. We evaluate TaDSE performance on five downstream benchmark dialogue datasets. The experiment results show that TaDSE achieves significant improvements over previous SOTA methods for dialogue. We further introduce a novel analytic instrument of semantic compression test, for which we discover a correlation with uniformity and alignment. Our code will be released upon acceptance.
>
---
#### [replaced 019] DND: Boosting Large Language Models with Dynamic Nested Depth
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DND方法，针对大语言模型推理中冗余计算与关键信息遗漏问题，通过动态选择关键token进行嵌套重处理，提升模型性能。在不显著增加参数和计算量的前提下，有效增强模型对难点token的处理能力，适用于密集型与MoE架构模型的后训练优化。**

- **链接: [https://arxiv.org/pdf/2510.11001v2](https://arxiv.org/pdf/2510.11001v2)**

> **作者:** Tieyuan Chen; Xiaodong Chen; Haoxing Chen; Zhenzhong Lan; Weiyao Lin; Jianguo Li
>
> **备注:** TL;DR: We introduce Dynamic Nested Depth (DND), an efficient paradigm that adaptively identifies critical tokens and selectively deepens their computation via nested re-processing
>
> **摘要:** We introduce Dynamic Nested Depth (DND), a novel method that improves performance for off-the-shelf LLMs by selecting critical tokens to reprocess in a nested depth manner. Specifically, at the end of the given transformer layer, DND identifies more critical tokens with a router and feeds them back for an extra round of processing, effectively ``reviewing" difficult tokens while avoiding redundant computation for easier ones. The dynamic selection mechanism is tailored for precise control via two novel strategies: a router controlling loss to enhance token selection distinguishability, and a threshold control scheme to ensure selection stability. We demonstrate the effectiveness of DND by directly integrating it into pre-trained dense and MoE models during a post-training phase. On diverse benchmarks, this approach boosts the performances of the dense Qwen3-1.7B by 1.88% and the MoE Qwen3-30B-A3B by 0.87%, all with a minimal parameter and computing increase.
>
---
#### [replaced 020] Human Decision-making is Susceptible to AI-driven Manipulation
- **分类: cs.AI; cs.CL; cs.CY; cs.HC**

- **简介: 该论文研究AI对人类决策的潜在操纵风险，属于人机交互与AI伦理任务。针对AI可能利用认知偏见操控用户的问题，通过实验对比三类AI代理，发现即使简单操纵也显著影响用户决策，揭示人类对AI操纵高度敏感，呼吁建立伦理与监管框架保护自主性。**

- **链接: [https://arxiv.org/pdf/2502.07663v3](https://arxiv.org/pdf/2502.07663v3)**

> **作者:** Sahand Sabour; June M. Liu; Siyang Liu; Chris Z. Yao; Shiyao Cui; Xuanming Zhang; Wen Zhang; Yaru Cao; Advait Bhat; Jian Guan; Wei Wu; Rada Mihalcea; Hongning Wang; Tim Althoff; Tatia M. C. Lee; Minlie Huang
>
> **备注:** Work in progress
>
> **摘要:** AI systems are increasingly intertwined with daily life, assisting users with various tasks and guiding decision-making. This integration introduces risks of AI-driven manipulation, where such systems may exploit users' cognitive biases and emotional vulnerabilities to steer them toward harmful outcomes. Through a randomized between-subjects experiment with 233 participants, we examined human susceptibility to such manipulation in financial (e.g., purchases) and emotional (e.g., conflict resolution) decision-making contexts. Participants interacted with one of three AI agents: a neutral agent (NA) optimizing for user benefit without explicit influence, a manipulative agent (MA) designed to covertly influence beliefs and behaviors, or a strategy-enhanced manipulative agent (SEMA) equipped with established psychological tactics, allowing it to select and apply them adaptively during interactions to reach its hidden objectives. By analyzing participants' preference ratings, we found significant susceptibility to AI-driven manipulation. Particularly across both decision-making domains, interacting with the manipulative agents significantly increased the odds of rating hidden incentives higher than optimal options (Financial, MA: OR=5.24, SEMA: OR=7.96; Emotional, MA: OR=5.52, SEMA: OR=5.71) compared to the NA group. Notably, we found no clear evidence that employing psychological strategies (SEMA) was overall more effective than simple manipulative objectives (MA) on our primary outcomes. Hence, AI-driven manipulation could become widespread even without requiring sophisticated tactics and expertise. While our findings are preliminary and derived from hypothetical, low-stakes scenarios, we highlight a critical vulnerability in human-AI interactions, emphasizing the need for ethical safeguards and regulatory frameworks to protect human autonomy.
>
---
#### [replaced 021] ShoppingBench: A Real-World Intent-Grounded Shopping Benchmark for LLM-based Agents
- **分类: cs.CL**

- **简介: 该论文提出ShoppingBench，一个面向大语言模型购物代理的现实世界意图对齐基准。针对现有基准仅覆盖简单购物意图的问题，构建包含250万真实商品的交互式模拟环境，涵盖复杂多步购物任务。通过轨迹蒸馏与强化学习，训练出性能接近GPT-4.1的小型代理，显著提升模型在复杂购物场景下的表现。**

- **链接: [https://arxiv.org/pdf/2508.04266v2](https://arxiv.org/pdf/2508.04266v2)**

> **作者:** Jiangyuan Wang; Kejun Xiao; Qi Sun; Huaipeng Zhao; Tao Luo; Jian Dong Zhang; Xiaoyi Zeng
>
> **备注:** submit to AAAI2026
>
> **摘要:** Existing benchmarks in e-commerce primarily focus on basic user intents, such as finding or purchasing products. However, real-world users often pursue more complex goals, such as applying vouchers, managing budgets, and finding multi-products seller. To bridge this gap, we propose ShoppingBench, a novel end-to-end shopping benchmark designed to encompass increasingly challenging levels of grounded intent. Specifically, we propose a scalable framework to simulate user instructions based on various intents derived from sampled real-world products. To facilitate consistent and reliable evaluations, we provide a large-scale shopping sandbox that serves as an interactive simulated environment, incorporating over 2.5 million real-world products. Experimental results demonstrate that even state-of-the-art language agents (such as GPT-4.1) achieve absolute success rates under 50% on our benchmark tasks, highlighting the significant challenges posed by our ShoppingBench. In addition, we propose a trajectory distillation strategy and leverage supervised fine-tuning, along with reinforcement learning on synthetic trajectories, to distill the capabilities of a large language agent into a smaller one. As a result, our trained agent achieves competitive performance compared to GPT-4.1.
>
---
#### [replaced 022] Checklists Are Better Than Reward Models For Aligning Language Models
- **分类: cs.CL**

- **简介: 该论文针对语言模型指令跟随能力的提升问题，提出基于检查清单的强化学习方法（RLCF）。通过从指令中提取具体检查项，结合AI判断与验证程序评分，生成动态奖励信号，实现更精准的模型对齐。实验表明，该方法在多个基准上均显著优于现有方法，尤其在复杂任务中表现突出。**

- **链接: [https://arxiv.org/pdf/2507.18624v2](https://arxiv.org/pdf/2507.18624v2)**

> **作者:** Vijay Viswanathan; Yanchao Sun; Shuang Ma; Xiang Kong; Meng Cao; Graham Neubig; Tongshuang Wu
>
> **备注:** Presented at NeurIPS 2025
>
> **摘要:** Language models must be adapted to understand and follow user instructions. Reinforcement learning is widely used to facilitate this -- typically using fixed criteria such as "helpfulness" and "harmfulness". In our work, we instead propose using flexible, instruction-specific criteria as a means of broadening the impact that reinforcement learning can have in eliciting instruction following. We propose "Reinforcement Learning from Checklist Feedback" (RLCF). From instructions, we extract checklists and evaluate how well responses satisfy each item - using both AI judges and specialized verifier programs - then combine these scores to compute rewards for RL. We compare RLCF with other alignment methods applied to a strong instruction following model (Qwen2.5-7B-Instruct) on five widely-studied benchmarks -- RLCF is the only method to improve performance on every benchmark, including a 4-point boost in hard satisfaction rate on FollowBench, a 6-point increase on InFoBench, and a 3-point rise in win rate on Arena-Hard. These results establish checklist feedback as a key tool for improving language models' support of queries that express a multitude of needs.
>
---
#### [replaced 023] Adaptive Margin RLHF via Preference over Preferences
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对RLHF中奖励模型学习的边际问题，提出基于“偏好之偏好”的自适应边际方法（DPO-PoP），通过人类对偏好强度的排序信号，动态推断每条数据的边际。解决了固定或简单边际导致泛化能力不足的问题，提升了分类与生成性能，并揭示了判别与生成间的权衡，提出针对性采样策略。**

- **链接: [https://arxiv.org/pdf/2509.22851v3](https://arxiv.org/pdf/2509.22851v3)**

> **作者:** Yaswanth Chittepu; Prasann Singhal; Greg Durrett; Scott Niekum
>
> **摘要:** Margin-based optimization is fundamental to improving generalization and robustness in classification tasks. In the context of reward model learning from preferences within Reinforcement Learning from Human Feedback (RLHF), existing methods typically rely on no margins, fixed margins, or margins that are simplistic functions of preference ratings. However, such formulations often fail to account for the varying strengths of different preferences, for example some preferences are associated with larger margins between responses, or they rely on noisy margin information derived from ratings. We argue that modeling the strength of preferences can lead to better generalization and more faithful alignment. Furthermore, many existing methods that use adaptive margins assume access to accurate preference scores, which can be difficult for humans to provide reliably. We propose an approach that leverages preferences over preferences, that is annotations indicating which of two preferences reflects a stronger distinction. We use this ordinal signal to infer adaptive margins on a per-datapoint basis. We introduce an extension to Direct Preference Optimization (DPO), DPO-PoP, that incorporates adaptive margins from preference-over-preference supervision, enabling improved discriminative and generative performance. Empirically, our method outperforms vanilla DPO, DPO with fixed margins, and DPO with ground-truth margins on the UltraFeedback dataset. Additionally, we show that there is a tradeoff between discriminative and generative performance: improving test classification accuracy, particularly by correctly labeling weaker preferences at the expense of stronger ones, can lead to a decline in generative quality. To navigate this tradeoff, we propose two sampling strategies to gather preference-over-preference labels: one favoring discriminative performance and one favoring generative performance.
>
---
#### [replaced 024] Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training
- **分类: cs.CL**

- **简介: 该论文针对语言模型预训练中数据混合比例优化难题，提出Nemotron-CLIMB框架，通过语义聚类与迭代搜索自动发现最优数据混合。工作包括构建高效数据集Nemotron-ClimbMix与研究平台Nemotron-ClimbLab，实现在400B token下超越Llama-3.2-1B 2.0%，特定领域优化提升5%。**

- **链接: [https://arxiv.org/pdf/2504.13161v2](https://arxiv.org/pdf/2504.13161v2)**

> **作者:** Shizhe Diao; Yu Yang; Yonggan Fu; Xin Dong; Dan Su; Markus Kliegl; Zijia Chen; Peter Belcak; Yoshi Suhara; Hongxu Yin; Mostofa Patwary; Yingyan; Lin; Jan Kautz; Pavlo Molchanov
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Pre-training datasets are typically collected from web content and lack inherent domain divisions. For instance, widely used datasets like Common Crawl do not include explicit domain labels, while manually curating labeled datasets such as The Pile is labor-intensive. Consequently, identifying an optimal pre-training data mixture remains a challenging problem, despite its significant benefits for pre-training performance. To address these challenges, we propose CLustering-based Iterative Data Mixture Bootstrapping (Nemotron-CLIMB), an automated framework that discovers, evaluates, and refines data mixtures in a pre-training setting. Specifically, Nemotron-CLIMB embeds and clusters large-scale datasets in a semantic space and then iteratively searches for optimal mixtures using a smaller proxy model and a predictor. When continuously trained on 400B tokens with this mixture, our 1B model exceeds the state-of-the-art Llama-3.2-1B by 2.0%. Moreover, we observe that optimizing for a specific domain (e.g., Social Sciences) yields a 5% improvement over random sampling. Finally, we introduce Nemotron-ClimbLab, a filtered 1.2-trillion-token corpus with 20 clusters as a research playground, and Nemotron-ClimbMix, a compact yet powerful 400-billion-token dataset designed for efficient pre-training that delivers superior performance under an equal token budget. We analyze the final data mixture, elucidating the characteristics of an optimal data mixture. Our data is available at: https://research.nvidia.com/labs/lpr/climb/
>
---
#### [replaced 025] TRepLiNa: Layer-wise CKA+REPINA Alignment Improves Low-Resource Machine Translation in Aya-23 8B
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对低资源语言（LRL）翻译难题，提出TRepLiNa方法，通过联合使用CKA与REPINA，在Aya-23 8B模型的中层特征上对齐跨语言表示。实验验证其在零样本、少样本及微调场景下能有效提升LRL到高资源语言的翻译质量，尤其在数据稀缺时表现优异。**

- **链接: [https://arxiv.org/pdf/2510.06249v3](https://arxiv.org/pdf/2510.06249v3)**

> **作者:** Toshiki Nakai; Ravi Kiran Chikkala; Lena Sophie Oberkircher; Nicholas Jennings; Natalia Skachkova; Tatiana Anikina; Jesujoba Oluwadara Alabi
>
> **备注:** It is work in progress
>
> **摘要:** The 2025 Multimodal Models for Low-Resource Contexts and Social Impact (MMLoSo) Language Challenge addresses one of India's most pressing linguistic gaps: the lack of resources for its diverse low-resource languages (LRLs). In this study, we investigate whether enforcing cross-lingual similarity in specific internal layers of a decoder-only multilingual large language model (LLM) can improve translation quality from LRL to high-resource language (HRL). Specifically, we combine Centered Kernel Alignment (CKA), a similarity metric that encourages representations of different languages to align, with REPINA, a regularization method that constrains parameter updates to remain close to the pretrained model, into a joint method we call TRepLiNa. In this research project, we experiment with zero-shot, few-shot, and fine-tuning settings using Aya-23 8B with QLoRA across MMLoSo shared task language pairs (Mundari, Santali, Bhili) with Hindi/English pivots. Our results show that aligning mid-level layers using TRepLiNa (CKA+REPINA) is a low-cost, practical approach to improving LRL translation, especially in data-scarce settings.
>
---
#### [replaced 026] CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation
- **分类: cs.CL**

- **简介: 该论文针对多语言、多模态大模型在语音与文本事实性判断上的评估不足问题，提出CCFQA基准，涵盖8种语言的平行语音-文本事实性问答数据。旨在系统评估模型跨语言、跨模态的事实性能力，并提出少样本迁移策略，有效提升多语言语音问答性能。**

- **链接: [https://arxiv.org/pdf/2508.07295v2](https://arxiv.org/pdf/2508.07295v2)**

> **作者:** Yexing Du; Kaiyuan Liu; Youcheng Pan; Zheng Chu; Bo Yang; Xiaocheng Feng; Ming Liu; Yang Xiang
>
> **备注:** Accepted in AAAI 2026
>
> **摘要:** As Large Language Models (LLMs) are increasingly popularized in the multilingual world, ensuring hallucination-free factuality becomes markedly crucial. However, existing benchmarks for evaluating the reliability of Multimodal Large Language Models (MLLMs) predominantly focus on textual or visual modalities with a primary emphasis on English, which creates a gap in evaluation when processing multilingual input, especially in speech. To bridge this gap, we propose a novel Cross-lingual and Cross-modal Factuality benchmark (CCFQA). Specifically, the CCFQA benchmark contains parallel speech-text factual questions across 8 languages, designed to systematically evaluate MLLMs' cross-lingual and cross-modal factuality capabilities. Our experimental results demonstrate that current MLLMs still face substantial challenges on the CCFQA benchmark. Furthermore, we propose a few-shot transfer learning strategy that effectively transfers the Question Answering (QA) capabilities of LLMs in English to multilingual Spoken Question Answering (SQA) tasks, achieving competitive performance with GPT-4o-mini-Audio using just 5-shot training. We release CCFQA as a foundational research resource to promote the development of MLLMs with more robust and reliable speech understanding capabilities. Our code and dataset are available at https://github.com/yxduir/ccfqa.
>
---
#### [replaced 027] Event2Vec: A Geometric Approach to Learning Composable Representations of Event Sequences
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Event2Vec，一种基于几何结构的事件序列表示学习框架。针对传统方法难以捕捉序列结构依赖的问题，模型通过欧氏与双曲空间中的可组合嵌入，实现事件序列的线性加和表示。实验表明其在无监督条件下有效建模序列结构。**

- **链接: [https://arxiv.org/pdf/2509.12188v2](https://arxiv.org/pdf/2509.12188v2)**

> **作者:** Antonin Sulc
>
> **备注:** 10 pages, 3 figures, Symmetry and Geometry in Neural Representations Workshop at NeuralIPS (NeurReps) 2025
>
> **摘要:** The study of neural representations, both in biological and artificial systems, is increasingly revealing the importance of geometric and topological structures. Inspired by this, we introduce Event2Vec, a novel framework for learning representations of discrete event sequences. Our model leverages a simple, additive recurrent structure to learn composable, interpretable embeddings. We provide a theoretical analysis demonstrating that, under specific training objectives, our model's learned representations in a Euclidean space converge to an ideal additive structure. This ensures that the representation of a sequence is the vector sum of its constituent events, a property we term the linear additive hypothesis. To address the limitations of Euclidean geometry for hierarchical data, we also introduce a variant of our model in hyperbolic space, which is naturally suited to embedding tree-like structures with low distortion. We present experiments to validate our hypothesis. Quantitative evaluation on the Brown Corpus yields a Silhouette score of 0.0564, outperforming a Word2Vec baseline (0.0215), demonstrating the model's ability to capture structural dependencies without supervision.
>
---
#### [replaced 028] Quantifying Cognitive Bias Induction in LLM-Generated Content
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLM）生成内容中的认知偏差问题，聚焦摘要与新闻核实任务。通过评估五类LLM，发现其存在框架偏见、幻觉和优先效应，并导致用户决策偏差。研究还测试了18种缓解方法，验证了针对性干预的有效性。**

- **链接: [https://arxiv.org/pdf/2507.03194v2](https://arxiv.org/pdf/2507.03194v2)**

> **作者:** Abeer Alessa; Param Somane; Akshaya Lakshminarasimhan; Julian Skirzynski; Julian McAuley; Jessica Echterhoff
>
> **备注:** 21 pages (including references and appendix), 3figures. accepted to AACL 2025
>
> **摘要:** Large language models (LLMs) are integrated into applications like shopping reviews, summarization, or medical diagnosis support, where their use affects human decisions. We investigate the extent to which LLMs expose users to biased content and demonstrate its effect on human decision-making. We assess five LLM families in summarization and news fact-checking tasks, evaluating the consistency of LLMs with their context and their tendency to hallucinate on a new self-updating dataset. Our findings show that LLMs expose users to content that changes the context's sentiment in 26.42% of cases (framing bias), hallucinate on 60.33% of post-knowledge-cutoff questions, and highlight context from earlier parts of the prompt (primacy bias) in 10.12% of cases, averaged across all tested models. We further find that humans are 32% more likely to purchase the same product after reading a summary of the review generated by an LLM rather than the original review. To address these issues, we evaluate 18 mitigation methods across three LLM families and find the effectiveness of targeted interventions.
>
---
#### [replaced 029] COMPASS: Context-Modulated PID Attention Steering System for Hallucination Mitigation
- **分类: cs.CL**

- **简介: 该论文针对大语言模型生成虚假事实的问题，提出COMPASS系统，通过上下文依赖度评分（CRS）与PID控制器动态调节注意力，实现无需重训练的实时纠错。任务为减少幻觉，核心工作是构建可解释的反馈控制框架，提升生成事实一致性。**

- **链接: [https://arxiv.org/pdf/2511.14776v2](https://arxiv.org/pdf/2511.14776v2)**

> **作者:** Kenji Sahay; Snigdha Pandya; Rohan Nagale; Anna Lin; Shikhar Shiromani; Kevin Zhu; Dev Sunishchal
>
> **备注:** 9 pages, 6 figures including algorithmns, 2 tables
>
> **摘要:** Large language models (LLMs) often generate fluent but factually incorrect statements despite having access to relevant evidence, a failure mode rooted in how they allocate attention between contextual and parametric knowledge. Understanding and steering this internal behavior is key both for trustworthy deployment and for scientific interpretability of model mechanisms. We introduce COMPASS (Context-Modulated PID Attention Steering System), a lightweight, interpretable control framework that embeds a model-based feedback loop directly within decoding. COMPASS quantifies context reliance via a transparent metric, the Context Reliance Score (CRS), which serves as an online probe of how attention heads ground generation in evidence. Using this interpretable signal, a PID controller dynamically modulates attention heads to maintain factual consistency without retraining or multi-pass decoding. Across benchmarks (HotpotQA, XSum, HaluEval, RAGTruth), COMPASS consistently reduces contextual hallucination rates (2.8 to 5.8 percent absolute) while revealing how distinct attention heads contribute to evidence alignment. These results highlight feedback-driven interpretability as a pathway toward scientific understanding of LLM behavior.
>
---
#### [replaced 030] Beyond Token Length: Step Pruner for Efficient and Accurate Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型推理过程中的冗余问题，提出Step Pruner（SP）框架。通过步骤感知的奖励函数与动态停止机制，鼓励模型生成更简洁、高效的推理路径，有效减少冗余步骤，提升推理效率与准确性，在多个基准上显著降低token使用量。**

- **链接: [https://arxiv.org/pdf/2510.03805v3](https://arxiv.org/pdf/2510.03805v3)**

> **作者:** Canhui Wu; Qiong Cao; Chang Li; Zhenfang Wang; Chao Xue; Yuwei Fan; Wei Xi; Xiaodong He
>
> **备注:** 21 pages, 9 figures
>
> **摘要:** Large Reasoning Models (LRMs) demonstrate strong performance on complex tasks but often suffer from excessive verbosity, known as "overthinking." Existing solutions via reinforcement learning (RL) typically penalize generated tokens to promote conciseness. However, these methods encounter two challenges: responses with fewer tokens do not always correspond to fewer reasoning steps, and models may develop hacking behavior in later stages of training by discarding reasoning steps to minimize token usage. In this work, we introduce \textbf{Step Pruner (SP)}, an RL framework that steers LRMs toward more efficient reasoning by favoring compact reasoning steps. Our step-aware reward function prioritizes correctness while imposing penalties for redundant steps, and withholds rewards for incorrect responses to prevent the reinforcement of erroneous reasoning. Moreover, we propose a dynamic stopping mechanism: when the model's output no longer shortens, training is halted to prevent hacking behavior caused by the merging of steps. Extensive experiments across four reasoning benchmarks demonstrate that SP achieves state-of-the-art accuracy while significantly reducing response length. For instance, on AIME24, SP reduces token usage by \textbf{69.7\%}.
>
---
#### [replaced 031] Measuring and Guiding Monosemanticity
- **分类: cs.CL**

- **简介: 该论文针对大语言模型（LLM）中特征表示难以精准定位与控制的问题，提出特征单义性评分（FMS）量化方法，并设计条件稀疏自编码器（G-SAE），通过引入标签概念引导训练，提升特征的单义性与可解释性。实验表明，该方法显著改善了毒性检测、写作风格识别等任务中的可控性与精度。**

- **链接: [https://arxiv.org/pdf/2506.19382v2](https://arxiv.org/pdf/2506.19382v2)**

> **作者:** Ruben Härle; Felix Friedrich; Manuel Brack; Stephan Wäldchen; Björn Deiseroth; Patrick Schramowski; Kristian Kersting
>
> **摘要:** There is growing interest in leveraging mechanistic interpretability and controllability to better understand and influence the internal dynamics of large language models (LLMs). However, current methods face fundamental challenges in reliably localizing and manipulating feature representations. Sparse Autoencoders (SAEs) have recently emerged as a promising direction for feature extraction at scale, yet they, too, are limited by incomplete feature isolation and unreliable monosemanticity. To systematically quantify these limitations, we introduce Feature Monosemanticity Score (FMS), a novel metric to quantify feature monosemanticity in latent representation. Building on these insights, we propose Guided Sparse Autoencoders (G-SAE), a method that conditions latent representations on labeled concepts during training. We demonstrate that reliable localization and disentanglement of target concepts within the latent space improve interpretability, detection of behavior, and control. Specifically, our evaluations on toxicity detection, writing style identification, and privacy attribute recognition show that G-SAE not only enhances monosemanticity but also enables more effective and fine-grained steering with less quality degradation. Our findings provide actionable guidelines for measuring and advancing mechanistic interpretability and control of LLMs.
>
---
#### [replaced 032] Debating Truth: Debate-driven Claim Verification with Multiple Large Language Model Agents
- **分类: cs.CL**

- **简介: 该论文针对复杂事实核查任务，提出基于多LLM代理辩论的验证框架DebateCV。通过对立代理多轮辩论暴露认知偏差，由调解者判别证据优劣。为解决调解者零样本能力不足问题，引入Debate-SFT合成数据微调方法，显著提升验证准确率与理由质量。**

- **链接: [https://arxiv.org/pdf/2507.19090v2](https://arxiv.org/pdf/2507.19090v2)**

> **作者:** Haorui He; Yupeng Li; Dacheng Wen; Yang Chen; Reynold Cheng; Donglong Chen; Francis C. M. Lau
>
> **摘要:** Claim verification is essential for digital literacy, yet state-of-the-art single-agent methods often struggle with complex claims that require nuanced analysis of multifaceted online evidence. Inspired by real-world human fact-checking practices, we propose \textbf{DebateCV}, the first debate-driven claim verification framework powered by multiple LLM agents. In DebateCV, two \textit{Debaters} argue opposing stances over multiple rounds to surface subtle errors in single-agent assessments. A decisive \textit{Moderator} is then required to weigh the evidential strength of conflicting arguments to deliver an accurate verdict. Yet zero-shot agents struggle to adjudicate multi-round debates for verifying complex claims, often defaulting to neutral judgements, and no datasets exist for training agents for this role. To bridge this gap, we propose \textbf{Debate-SFT}, a post-training framework that leverages synthetic data to enhance agents' ability to effectively adjudicate debates for claim verification. Results show that our methods surpass state-of-the-art non-debate approaches in both accuracy (across various evidence conditions) and justification quality, which strengthens societal resilience against misinformation and contributes to a more trustworthy online information ecosystem.
>
---
#### [replaced 033] EQ-Negotiator: Dynamic Emotional Personas Empower Small Language Models for Edge-Deployable Credit Negotiation
- **分类: cs.CL**

- **简介: 该论文针对小语言模型（SLMs）在隐私敏感场景下进行信用谈判时情感智能不足的问题，提出EQ-Negotiator框架。通过结合博弈论与隐马尔可夫模型，动态追踪并响应债务人情绪，使7B模型在谈判效率与回款率上超越十倍于己的LLM，实现高效、伦理且隐私保护的边缘部署。**

- **链接: [https://arxiv.org/pdf/2511.03370v2](https://arxiv.org/pdf/2511.03370v2)**

> **作者:** Yunbo Long; Yuhan Liu; Alexandra Brintrup
>
> **摘要:** The deployment of large language models (LLMs) in automated negotiation has set a high performance benchmark, but their computational cost and data privacy requirements render them unsuitable for many privacy-sensitive, on-device applications such as mobile assistants, embodied AI agents or private client interactions. While small language models (SLMs) offer a practical alternative, they suffer from a significant performance gap compared to LLMs in playing emotionally charged complex personas, especially for credit negotiation. This paper introduces EQ-Negotiator, a novel framework that bridges this capability gap using emotional personas. Its core is a reasoning system that integrates game theory with a Hidden Markov Model(HMM) to learn and track debtor emotional states online, without pre-training. This allows EQ-Negotiator to equip SLMs with the strategic intelligence to counter manipulation while de-escalating conflict and upholding ethical standards. Through extensive agent-to-agent simulations across diverse credit negotiation scenarios, including adversarial debtor strategies like cheating, threatening, and playing the victim, we show that a 7B parameter language model with EQ-Negotiator achieves better debt recovery and negotiation efficiency than baseline LLMs more than 10 times its size. This work advances persona modeling from descriptive character profiles to dynamic emotional architectures that operate within privacy constraints. Besides, this paper establishes that strategic emotional intelligence, not raw model scale, is the critical factor for success in automated negotiation, paving the way for effective, ethical, and privacy-preserving AI negotiators that can operate on the edge.
>
---
#### [replaced 034] LLM-based Automated Grading with Human-in-the-Loop
- **分类: cs.CL**

- **简介: 该论文研究自动短答案评分（ASAG）任务，旨在提升LLM在基于评分量规的评估中接近人类水平的性能。针对现有方法依赖完全自动化、难以精准评分的问题，提出人机协同框架GradeHITL，利用LLM与专家交互，动态优化评分标准，显著提升评分准确性。**

- **链接: [https://arxiv.org/pdf/2504.05239v3](https://arxiv.org/pdf/2504.05239v3)**

> **作者:** Yucheng Chu; Hang Li; Kaiqi Yang; Yasemin Copur-Gencturk; Jiliang Tang
>
> **备注:** Accepted to IEEE TALE 2025
>
> **摘要:** The rise of artificial intelligence (AI) technologies, particularly large language models (LLMs), has brought significant advancements to the field of education. Among various applications, automatic short answer grading (ASAG), which focuses on evaluating open-ended textual responses, has seen remarkable progress with the introduction of LLMs. These models not only enhance grading performance compared to traditional ASAG approaches but also move beyond simple comparisons with predefined "golden" answers, enabling more sophisticated grading scenarios, such as rubric-based evaluation. However, existing LLM-powered methods still face challenges in achieving human-level grading performance in rubric-based assessments due to their reliance on fully automated approaches. In this work, we explore the potential of LLMs in ASAG tasks by leveraging their interactive capabilities through a human-in-the-loop (HITL) approach. Our proposed framework, GradeHITL, utilizes the generative properties of LLMs to pose questions to human experts, incorporating their insights to refine grading rubrics dynamically. This adaptive process significantly improves grading accuracy, outperforming existing methods and bringing ASAG closer to human-level evaluation.
>
---
#### [replaced 035] Robust Multimodal Sentiment Analysis with Distribution-Based Feature Recovery and Fusion
- **分类: cs.CL**

- **简介: 该论文针对社交媒体中图像-文本对的多模态情感分析任务，解决真实场景下模态质量低或缺失的问题。提出基于分布的特征恢复与融合方法（DRF），通过模态特征队列建模分布，量化模态质量并恢复缺失模态，实现统一鲁棒的多模态融合。**

- **链接: [https://arxiv.org/pdf/2511.18751v2](https://arxiv.org/pdf/2511.18751v2)**

> **作者:** Daiqing Wu; Dongbao Yang; Yu Zhou; Can Ma
>
> **备注:** Accepted by ACM MM 2024
>
> **摘要:** As posts on social media increase rapidly, analyzing the sentiments embedded in image-text pairs has become a popular research topic in recent years. Although existing works achieve impressive accomplishments in simultaneously harnessing image and text information, they lack the considerations of possible low-quality and missing modalities. In real-world applications, these issues might frequently occur, leading to urgent needs for models capable of predicting sentiment robustly. Therefore, we propose a Distribution-based feature Recovery and Fusion (DRF) method for robust multimodal sentiment analysis of image-text pairs. Specifically, we maintain a feature queue for each modality to approximate their feature distributions, through which we can simultaneously handle low-quality and missing modalities in a unified framework. For low-quality modalities, we reduce their contributions to the fusion by quantitatively estimating modality qualities based on the distributions. For missing modalities, we build inter-modal mapping relationships supervised by samples and distributions, thereby recovering the missing modalities from available ones. In experiments, two disruption strategies that corrupt and discard some modalities in samples are adopted to mimic the low-quality and missing modalities in various real-world scenarios. Through comprehensive experiments on three publicly available image-text datasets, we demonstrate the universal improvements of DRF compared to SOTA methods under both two strategies, validating its effectiveness in robust multimodal sentiment analysis.
>
---
#### [replaced 036] Adversarial Confusion Attack: Disrupting Multimodal Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出“对抗混淆攻击”，针对多模态大语言模型（MLLMs）的系统性破坏。旨在通过生成扰动图像，使模型输出混乱或自信错误，影响其可靠性。利用小规模开源模型集合最大化下一词熵，实现对多种模型的有效迁移攻击，涵盖开放与闭源模型。**

- **链接: [https://arxiv.org/pdf/2511.20494v3](https://arxiv.org/pdf/2511.20494v3)**

> **作者:** Jakub Hoscilowicz; Artur Janicki
>
> **摘要:** We introduce the Adversarial Confusion Attack, a new class of threats against multimodal large language models (MLLMs). Unlike jailbreaks or targeted misclassification, the goal is to induce systematic disruption that makes the model generate incoherent or confidently incorrect outputs. Practical applications include embedding such adversarial images into websites to prevent MLLM-powered AI Agents from operating reliably. The proposed attack maximizes next-token entropy using a small ensemble of open-source MLLMs. In the white-box setting, we show that a single adversarial image can disrupt all models in the ensemble, both in the full-image and Adversarial CAPTCHA settings. Despite relying on a basic adversarial technique (PGD), the attack generates perturbations that transfer to both unseen open-source (e.g., Qwen3-VL) and proprietary (e.g., GPT-5.1) models.
>
---
#### [replaced 037] PEPPER: Perception-Guided Perturbation for Robust Backdoor Defense in Text-to-Image Diffusion Models
- **分类: cs.CL**

- **简介: 该论文针对文本到图像扩散模型的后门攻击问题，提出PEPPER防御方法。通过语义差异大但视觉相似的提示重写，并添加隐蔽元素，破坏触发器，降低攻击成功率，同时保持生成质量。可与现有防御方法结合，提升整体鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.16830v2](https://arxiv.org/pdf/2511.16830v2)**

> **作者:** Oscar Chew; Po-Yi Lu; Jayden Lin; Kuan-Hao Huang; Hsuan-Tien Lin
>
> **摘要:** Recent studies show that text to image (T2I) diffusion models are vulnerable to backdoor attacks, where a trigger in the input prompt can steer generation toward harmful or unintended content. To address this, we introduce PEPPER (PErcePtion Guided PERturbation), a backdoor defense that rewrites the caption into a semantically distant yet visually similar caption while adding unobstructive elements. With this rewriting strategy, PEPPER disrupt the trigger embedded in the input prompt, dilute the influence of trigger tokens and thereby achieve enhanced robustness. Experiments show that PEPPER is particularly effective against text encoder based attacks, substantially reducing attack success while preserving generation quality. Beyond this, PEPPER can be paired with any existing defenses yielding consistently stronger and generalizable robustness than any standalone method. Our code will be released on Github.
>
---
#### [replaced 038] Reasoning-Intensive Regression
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究推理密集型回归（RiR），即从文本中推断细微数值评分。针对有限数据与计算资源下，传统方法在深层上下文分析中的不足，提出MENTAT方法，结合批量反思提示优化与神经集成学习，显著提升性能，为RiR任务提供新范式。**

- **链接: [https://arxiv.org/pdf/2508.21762v2](https://arxiv.org/pdf/2508.21762v2)**

> **作者:** Diane Tchuindjo; Omar Khattab
>
> **摘要:** AI researchers and practitioners increasingly apply large language models (LLMs) to what we call reasoning-intensive regression (RiR), i.e., deducing subtle numerical scores from text. Unlike standard language regression tasks, e.g., for sentiment or similarity, RiR often appears instead in ad-hoc problems such as rubric-based scoring, modeling dense rewards in complex environments, or domain-specific retrieval, where much deeper analysis of context is required while only limited task-specific training data and computation are available. We cast four realistic problems as RiR tasks to establish an initial benchmark, and use that to test our hypothesis that prompting frozen LLMs and finetuning Transformer encoders via gradient descent will both often struggle in RiR. We then propose MENTAT, a simple and lightweight method that combines batch-reflective prompt optimization with neural ensemble learning. MENTAT achieves up to 65% improvement over both baselines, though substantial room remains for future advances in RiR.
>
---
#### [replaced 039] Escaping Collapse: The Strength of Weak Data for Large Language Model Training
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大语言模型训练中合成数据的优化问题，旨在解决过度使用未精心筛选的合成数据导致模型性能“坍塌”的难题。通过借鉴提升算法思想，提出动态聚焦困难样本的训练策略，理论分析与实验验证表明该方法可持续提升模型性能。**

- **链接: [https://arxiv.org/pdf/2502.08924v2](https://arxiv.org/pdf/2502.08924v2)**

> **作者:** Kareem Amin; Sara Babakniya; Alex Bie; Weiwei Kong; Umar Syed; Sergei Vassilvitskii
>
> **摘要:** Synthetically-generated data plays an increasingly larger role in training large language models. However, while synthetic data has been found to be useful, studies have also shown that without proper curation it can cause LLM performance to plateau, or even "collapse", after many training iterations. In this paper, we formalize this question and develop a theoretical framework to investigate how much curation is needed in order to ensure that LLM performance continually improves. Our analysis is inspired by boosting, a classic machine learning technique that leverages a very weak learning algorithm to produce an arbitrarily good classifier. The approach we analyze subsumes many recently proposed methods for training LLMs on synthetic data, and thus our analysis sheds light on why they are successful, and also suggests opportunities for future improvement. We present experiments that validate our theory, and show that dynamically focusing labeling resources on the most challenging examples -- in much the same way that boosting focuses the efforts of the weak learner -- leads to improved performance.
>
---
#### [replaced 040] MimeQA: Towards Socially-Intelligent Nonverbal Foundation Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文针对AI在非语言社交理解上的不足，提出基于哑剧视频的MimeQA数据集，构建非语言社交推理任务。通过806个标注问答对，评估视频大模型发现其非语言理解能力差（准确率20-30%），远低于人类（86%），揭示模型过度依赖文本提示、难以捕捉细微非语言互动的问题。**

- **链接: [https://arxiv.org/pdf/2502.16671v3](https://arxiv.org/pdf/2502.16671v3)**

> **作者:** Hengzhi Li; Megan Tjandrasuwita; Yi R. Fung; Armando Solar-Lezama; Paul Pu Liang
>
> **备注:** NeurIPS 2025 Datasets and Benchmarks
>
> **摘要:** As AI becomes more closely integrated with peoples' daily activities, socially intelligent AI that can understand and interact seamlessly with humans in daily lives is increasingly important. However, current works in AI social reasoning all rely on language-only or language-dominant approaches to benchmark and training models, resulting in systems that are improving in verbal communication but struggle with nonverbal social understanding. To address this limitation, we tap into a novel data source rich in nonverbal social interactions -- mime videos. Mimes refer to the art of expression through gesture and movement without spoken words, which presents unique challenges and opportunities in interpreting nonverbal social communication. We contribute a new dataset called MimeQA, obtained by sourcing ~8 hours of videos clips from YouTube and developing a comprehensive video question-answering benchmark comprising 806 carefully annotated and verified question-answer pairs, designed to probe nonverbal social reasoning capabilities. Using MimeQA, we evaluate state-of-the-art video large language models (VideoLLMs) and find that they achieve low accuracy, generally ranging from 20-30%, while humans score 86%. Our analysis reveals that VideoLLMs often fail to ground imagined objects and over-rely on the text prompt while ignoring subtle nonverbal interactions. We hope to inspire future work in AI models that embody true social intelligence capable of interpreting non-verbal human interactions.
>
---
#### [replaced 041] From Code Foundation Models to Agents and Applications: A Practical Guide to Code Intelligence
- **分类: cs.SE; cs.CL**

- **简介: 该论文聚焦代码智能任务，解决代码大模型从预训练到应用中的技术瓶颈。系统分析了代码LLM的全生命周期，涵盖数据、训练与提示工程，对比通用与专用模型，揭示研究与实践差距，并通过实验提供可复现的优化路径。**

- **链接: [https://arxiv.org/pdf/2511.18538v2](https://arxiv.org/pdf/2511.18538v2)**

> **作者:** Jian Yang; Xianglong Liu; Weifeng Lv; Ken Deng; Shawn Guo; Lin Jing; Yizhi Li; Shark Liu; Xianzhen Luo; Yuyu Luo; Changzai Pan; Ensheng Shi; Yingshui Tan; Renshuai Tao; Jiajun Wu; Xianjie Wu; Zhenhe Wu; Daoguang Zan; Chenchen Zhang; Wei Zhang; He Zhu; Terry Yue Zhuo; Kerui Cao; Xianfu Cheng; Jun Dong; Shengjie Fang; Zhiwei Fei; Xiangyuan Guan; Qipeng Guo; Zhiguang Han; Joseph James; Tianqi Luo; Renyuan Li; Yuhang Li; Yiming Liang; Congnan Liu; Jiaheng Liu; Qian Liu; Ruitong Liu; Tyler Loakman; Xiangxin Meng; Chuang Peng; Tianhao Peng; Jiajun Shi; Mingjie Tang; Boyang Wang; Haowen Wang; Yunli Wang; Fanglin Xu; Zihan Xu; Fei Yuan; Ge Zhang; Jiayi Zhang; Xinhao Zhang; Wangchunshu Zhou; Hualei Zhu; King Zhu; Brown Dai; Aishan Liu; Zhoujun Li; Chenghua Lin; Tianyu Liu; Chao Peng; Kai Shen; Libo Qin; Shuangyong Song; Zizheng Zhan; Jiajun Zhang; Jie Zhang; Zhaoxiang Zhang; Bo Zheng
>
> **摘要:** Large language models (LLMs) have fundamentally transformed automated software development by enabling direct translation of natural language descriptions into functional code, driving commercial adoption through tools like Github Copilot (Microsoft), Cursor (Anysphere), Trae (ByteDance), and Claude Code (Anthropic). While the field has evolved dramatically from rule-based systems to Transformer-based architectures, achieving performance improvements from single-digit to over 95\% success rates on benchmarks like HumanEval. In this work, we provide a comprehensive synthesis and practical guide (a series of analytic and probing experiments) about code LLMs, systematically examining the complete model life cycle from data curation to post-training through advanced prompting paradigms, code pre-training, supervised fine-tuning, reinforcement learning, and autonomous coding agents. We analyze the code capability of the general LLMs (GPT-4, Claude, LLaMA) and code-specialized LLMs (StarCoder, Code LLaMA, DeepSeek-Coder, and QwenCoder), critically examining the techniques, design decisions, and trade-offs. Further, we articulate the research-practice gap between academic research (e.g., benchmarks and tasks) and real-world deployment (e.g., software-related code tasks), including code correctness, security, contextual awareness of large codebases, and integration with development workflows, and map promising research directions to practical needs. Last, we conduct a series of experiments to provide a comprehensive analysis of code pre-training, supervised fine-tuning, and reinforcement learning, covering scaling law, framework selection, hyperparameter sensitivity, model architectures, and dataset comparisons.
>
---
#### [replaced 042] BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出BoundingDocs，一个包含空间标注的文档问答统一数据集。将信息抽取等任务转化为QA形式，结合OCR与答案位置边界框，用于训练和评估大模型。研究不同提示技术对文档理解的影响，探索高效推理方法。**

- **链接: [https://arxiv.org/pdf/2501.03403v3](https://arxiv.org/pdf/2501.03403v3)**

> **作者:** Simone Giovannini; Fabio Coppini; Andrea Gemelli; Simone Marinai
>
> **摘要:** We present a unified dataset for document Question-Answering (QA), which is obtained combining several public datasets related to Document AI and visually rich document understanding (VRDU). Our main contribution is twofold: on the one hand we reformulate existing Document AI tasks, such as Information Extraction (IE), into a Question-Answering task, making it a suitable resource for training and evaluating Large Language Models; on the other hand, we release the OCR of all the documents and include the exact position of the answer to be found in the document image as a bounding box. Using this dataset, we explore the impact of different prompting techniques (that might include bounding box information) on the performance of open-weight models, identifying the most effective approaches for document comprehension.
>
---
#### [replaced 043] Extending Multilingual Machine Translation through Imitation Learning
- **分类: cs.CL**

- **简介: 该论文研究多语言机器翻译中扩展新语言的任务。针对仅拥有新语言与英语平行语料的难题，提出基于模仿学习的Imit-MNMT方法，通过专家模型生成伪双语数据，结合语言特异性加权与翻译行为模仿，有效提升新旧语言间翻译性能，缓解灾难性遗忘问题。**

- **链接: [https://arxiv.org/pdf/2311.08538v2](https://arxiv.org/pdf/2311.08538v2)**

> **作者:** Wen Lai; Viktor Hangya; Yingli Shen; Alexander Fraser
>
> **摘要:** Despite the growing variety of languages supported by existing multilingual neural machine translation (MNMT) models, most of the world's languages are still being left behind. We aim to extend large-scale MNMT models to incorporate a new language, enabling translations between this new language and all previously supported languages, even in the challenging scenario where only a parallel corpus between the new language and English is available. Previous methods, such as continued training on parallel data including the new language, often suffer from catastrophic forgetting, which degrades performance on other languages. We propose a novel approach Imit-MNMT which treats this task as an imitation learning problem, a technique widely used in computer vision but less explored in natural language processing. Specifically, we leverage an expert model to generate pseudo-parallel corpora between the new language and the existing languages. We then introduce a data distribution imitation strategy using language-specific weighting, alongside a translation behavior imitation mechanism. Extensive experiments show that our approach significantly improves translation performance between the new and existing languages while mitigating catastrophic forgetting.
>
---
#### [replaced 044] Combining Textual and Structural Information for Premise Selection in Lean
- **分类: cs.LG; cs.AI; cs.CL; cs.LO**

- **简介: 该论文针对形式化证明中前提选择任务，解决现有方法忽略前提间依赖关系的问题。提出结合文本嵌入与图神经网络的图增强方法，利用异构依赖图捕捉状态-前提和前提-前提关系，在LeanDojo基准上显著优于语言基线。**

- **链接: [https://arxiv.org/pdf/2510.23637v2](https://arxiv.org/pdf/2510.23637v2)**

> **作者:** Job Petrovčič; David Eliecer Narvaez Denis; Ljupčo Todorovski
>
> **摘要:** Premise selection is a key bottleneck for scaling theorem proving in large formal libraries. Yet existing language-based methods often treat premises in isolation, ignoring the web of dependencies that connects them. We present a graph-augmented approach that combines dense text embeddings of Lean formalizations with graph neural networks over a heterogeneous dependency graph capturing both state-premise and premise-premise relations. On the LeanDojo Benchmark, our method outperforms the ReProver language-based baseline by over 25\% across standard retrieval metrics. These results suggest that relational information is beneficial for premise selection.
>
---
#### [replaced 045] Eye of Judgement: Dissecting the Evaluation of Russian-speaking LLMs with POLLUX
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出POLLUX，一个面向俄语大模型的开源评估基准。针对现有评估方法依赖人力、成本高且不透明的问题，构建了包含35类任务、2100个精细设计提示的评测体系，并引入LLM-as-a-Judge机制，实现可解释、可扩展的自动化评估。**

- **链接: [https://arxiv.org/pdf/2505.24616v4](https://arxiv.org/pdf/2505.24616v4)**

> **作者:** Nikita Martynov; Anastasia Mordasheva; Dmitriy Gorbetskiy; Danil Astafurov; Ulyana Isaeva; Elina Basyrova; Sergey Skachkov; Victoria Berestova; Nikolay Ivanov; Valeriia Zanina; Alena Fenogenova
>
> **备注:** short version
>
> **摘要:** We introduce POLLUX, a comprehensive open-source benchmark designed to evaluate the generative capabilities of large language models (LLMs) in Russian. Our main contribution is a novel evaluation methodology that enhances the interpretability of LLM assessment. For each task type, we define a set of detailed criteria and develop a scoring protocol where models evaluate responses and provide justifications for their ratings. This enables transparent, criteria-driven evaluation beyond traditional resource-consuming, side-by-side human comparisons. POLLUX includes a detailed, fine-grained taxonomy of 35 task types covering diverse generative domains such as code generation, creative writing, and practical assistant use cases, totaling 2,100 manually crafted and professionally authored prompts. Each task is categorized by difficulty (easy/medium/hard), with experts constructing the dataset entirely from scratch. We also release a family of LLM-as-a-Judge (7B and 32B) evaluators trained for nuanced assessment of generative outputs. This approach provides scalable, interpretable evaluation and annotation tools for model development, effectively replacing costly and less precise human judgments.
>
---
#### [replaced 046] Soft Adaptive Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对大语言模型强化学习中策略优化的不稳定性问题，提出软自适应策略优化（SAPO）。针对高方差的词元级重要性比率，SAPO采用平滑温度控制门替代硬裁剪，实现序列一致且词元自适应的更新，提升样本效率与训练稳定性，在数学推理与多任务场景中表现更优。**

- **链接: [https://arxiv.org/pdf/2511.20347v2](https://arxiv.org/pdf/2511.20347v2)**

> **作者:** Chang Gao; Chujie Zheng; Xiong-Hui Chen; Kai Dang; Shixuan Liu; Bowen Yu; An Yang; Shuai Bai; Jingren Zhou; Junyang Lin
>
> **摘要:** Reinforcement learning (RL) plays an increasingly important role in enhancing the reasoning capabilities of large language models (LLMs), yet stable and performant policy optimization remains challenging. Token-level importance ratios often exhibit high variance-a phenomenon exacerbated in Mixture-of-Experts models-leading to unstable updates. Existing group-based policy optimization methods, such as GSPO and GRPO, alleviate this problem via hard clipping, making it difficult to maintain both stability and effective learning. We propose Soft Adaptive Policy Optimization (SAPO), which replaces hard clipping with a smooth, temperature-controlled gate that adaptively attenuates off-policy updates while preserving useful learning signals. Compared with GSPO and GRPO, SAPO is both sequence-coherent and token-adaptive. Like GSPO, SAPO maintains sequence-level coherence, but its soft gating forms a continuous trust region that avoids the brittle hard clipping band used in GSPO. When a sequence contains a few highly off-policy tokens, GSPO suppresses all gradients for that sequence, whereas SAPO selectively down-weights only the offending tokens and preserves the learning signal from the near-on-policy ones, improving sample efficiency. Relative to GRPO, SAPO replaces hard token-level clipping with smooth, temperature-controlled scaling, enabling more informative and stable updates. Empirical results on mathematical reasoning benchmarks indicate that SAPO exhibits improved training stability and higher Pass@1 performance under comparable training budgets. Moreover, we employ SAPO to train the Qwen3-VL model series, demonstrating that SAPO yields consistent performance gains across diverse tasks and different model sizes. Overall, SAPO provides a more reliable, scalable, and effective optimization strategy for RL training of LLMs.
>
---
#### [replaced 047] LLMs can hide text in other text of the same length
- **分类: cs.AI; cs.CL; cs.CR; cs.LG**

- **简介: 该论文提出Calgacus协议，利用大语言模型将一段文本隐藏于同长度的另一段看似合理的文本中。任务为隐写术，解决信息隐蔽传输问题。工作包括设计高效编码解码方法，证明80亿参数模型即可实现本地秒级处理，揭示文本与意图分离对信任体系的冲击，并警示AI安全风险。**

- **链接: [https://arxiv.org/pdf/2510.20075v4](https://arxiv.org/pdf/2510.20075v4)**

> **作者:** Antonio Norelli; Michael Bronstein
>
> **备注:** 21 pages, main paper 9 pages
>
> **摘要:** A meaningful text can be hidden inside another, completely different yet still coherent and plausible, text of the same length. For example, a tweet containing a harsh political critique could be embedded in a tweet that celebrates the same political leader, or an ordinary product review could conceal a secret manuscript. This uncanny state of affairs is now possible thanks to Large Language Models, and in this paper we present Calgacus, a simple and efficient protocol to achieve it. We show that even modest 8-billion-parameter open-source LLMs are sufficient to obtain high-quality results, and a message as long as this abstract can be encoded and decoded locally on a laptop in seconds. The existence of such a protocol demonstrates a radical decoupling of text from authorial intent, further eroding trust in written communication, already shaken by the rise of LLM chatbots. We illustrate this with a concrete scenario: a company could covertly deploy an unfiltered LLM by encoding its answers within the compliant responses of a safe model. This possibility raises urgent questions for AI safety and challenges our understanding of what it means for a Large Language Model to know something.
>
---
#### [replaced 048] Life-Code: Central Dogma Modeling with Multi-Omics Sequence Unification
- **分类: cs.LG; cs.AI; cs.CL; q-bio.GN**

- **简介: 该论文提出Life-Code框架，解决多组学数据孤立分析的问题。通过统一编码序列并设计联合建模架构，实现DNA、RNA、蛋白间的跨模态理解，提升对中心法则下遗传信息传递的建模能力。**

- **链接: [https://arxiv.org/pdf/2502.07299v3](https://arxiv.org/pdf/2502.07299v3)**

> **作者:** Zicheng Liu; Siyuan Li; Zhiyuan Chen; Chang Yu; Qirong Yang; Yucheng Guo; Yujie Yang; Xiaoming Zhang; Stan Z. Li
>
> **备注:** Preprint V3 (10 pages main text)
>
> **摘要:** The interactions between DNA, RNA, and proteins are fundamental to biological processes, as illustrated by the central dogma of molecular biology. Although modern biological pre-trained models have achieved great success in analyzing these macromolecules individually, their interconnected nature remains underexplored. This paper follows the guidance of the central dogma to redesign both the data and model pipeline and offers a comprehensive framework, Life-Code, that spans different biological functions. As for data flow, we propose a unified pipeline to integrate multi-omics data by reverse-transcribing RNA and reverse-translating amino acids into nucleotide-based sequences. As for the model, we design a codon tokenizer and a hybrid long-sequence architecture to encode the interactions between coding and non-coding regions through masked modeling pre-training. To model the translation and folding process with coding sequences, Life-Code learns protein structures of the corresponding amino acids by knowledge distillation from off-the-shelf protein language models. Such designs enable Life-Code to capture complex interactions within genetic sequences, providing a more comprehensive understanding of multi-omics with the central dogma. Extensive experiments show that Life-Code achieves state-of-the-art results on various tasks across three omics, highlighting its potential for advancing multi-omics analysis and interpretation.
>
---
#### [replaced 049] RealWebAssist: A Benchmark for Long-Horizon Web Assistance with Real-World Users
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出RealWebAssist基准，针对长周期网页助手任务中真实用户指令的复杂性问题。旨在评估AI在多步骤、跨网站交互中理解模糊指令、追踪用户意图与状态的能力。通过收集真实用户序列指令数据集，揭示当前模型在语义理解与图形界面操作上的显著不足。**

- **链接: [https://arxiv.org/pdf/2504.10445v2](https://arxiv.org/pdf/2504.10445v2)**

> **作者:** Suyu Ye; Haojun Shi; Darren Shih; Hyokun Yun; Tanya Roosta; Tianmin Shu
>
> **备注:** Project Website: https://scai.cs.jhu.edu/projects/RealWebAssist/ Code: https://github.com/SCAI-JHU/RealWebAssist
>
> **摘要:** To achieve successful assistance with long-horizon web-based tasks, AI agents must be able to sequentially follow real-world user instructions over a long period. Unlike existing web-based agent benchmarks, sequential instruction following in the real world poses significant challenges beyond performing a single, clearly defined task. For instance, real-world human instructions can be ambiguous, require different levels of AI assistance, and may evolve over time, reflecting changes in the user's mental state. To address this gap, we introduce RealWebAssist, a novel benchmark designed to evaluate sequential instruction-following in realistic scenarios involving long-horizon interactions with the web, visual GUI grounding, and understanding ambiguous real-world user instructions. RealWebAssist includes a dataset of sequential instructions collected from real-world human users. Each user instructs a web-based assistant to perform a series of tasks on multiple websites. A successful agent must reason about the true intent behind each instruction, keep track of the mental state of the user, understand user-specific routines, and ground the intended tasks to actions on the correct GUI elements. Our experimental results show that state-of-the-art models struggle to understand and ground user instructions, posing critical challenges in following real-world user instructions for long-horizon web assistance.
>
---
#### [replaced 050] Extracting memorized pieces of (copyrighted) books from open-weight language models
- **分类: cs.CL; cs.CY; cs.LG**

- **简介: 该论文研究生成式AI模型对受版权保护书籍的“记忆”现象，属于机器学习与版权法交叉任务。针对诉讼中关于模型是否记忆内容的争议，通过概率提取方法分析17个开源大模型对50本书的 memorization 情况，发现多数模型未整体或部分记忆多数书籍，但Llama 3.1 70B完全记忆了《哈利·波特》首部等书，且可精准复现。结果对版权判案具重要启示。**

- **链接: [https://arxiv.org/pdf/2505.12546v4](https://arxiv.org/pdf/2505.12546v4)**

> **作者:** A. Feder Cooper; Aaron Gokaslan; Ahmed Ahmed; Amy B. Cyphert; Christopher De Sa; Mark A. Lemley; Daniel E. Ho; Percy Liang
>
> **摘要:** Plaintiffs and defendants in copyright lawsuits over generative AI often make sweeping, opposing claims about the extent to which large language models (LLMs) have memorized plaintiffs' protected expression in their training data. Drawing on both machine learning and copyright law, we show that these polarized positions dramatically oversimplify the relationship between memorization and copyright. To do so, we extend a recent probabilistic extraction technique to measure memorization of 50 books in 17 open-weight LLMs. Through thousands of experiments, we show that the extent of memorization varies both by model and by book. With respect to our specific extraction methodology, we find that most LLMs do not memorize most books -- either in whole or in part. However, we also find that Llama 3.1 70B entirely memorizes some books, like the first Harry Potter book and 1984. In fact, the first Harry Potter is so memorized that, using a seed prompt consisting of just the first few tokens of the first chapter, we can deterministically generate the entire book near-verbatim. We discuss why our results have significant implications for copyright cases, though not ones that unambiguously favor either side.
>
---
#### [replaced 051] Measuring Chain-of-Thought Monitorability Through Faithfulness and Verbosity
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究链式思维（CoT）的可监控性，旨在解决模型推理过程透明度不足的问题。通过引入“忠实性”与“冗长度”两个维度，构建综合评估指标，衡量CoT作为模型外部工作记忆的有效性。实验在多个基准上验证了不同模型家族的监控能力差异，揭示了仅依赖忠实性评估的局限性，并开源了评估工具以支持复现。**

- **链接: [https://arxiv.org/pdf/2510.27378v2](https://arxiv.org/pdf/2510.27378v2)**

> **作者:** Austin Meek; Eitan Sprejer; Iván Arcuschin; Austin J. Brockmeier; Steven Basart
>
> **备注:** Project page at https://ajmeek.github.io/cot_monitorability_website/
>
> **摘要:** Chain-of-thought (CoT) outputs let us read a model's step-by-step reasoning. Since any long, serial reasoning process must pass through this textual trace, the quality of the CoT is a direct window into what the model is thinking. This visibility could help us spot unsafe or misaligned behavior (monitorability), but only if the CoT is transparent about its internal reasoning (faithfulness). Fully measuring faithfulness is difficult, so researchers often focus on examining the CoT in cases where the model changes its answer after adding a cue to the input. This proxy finds some instances of unfaithfulness but loses information when the model maintains its answer, and does not investigate aspects of reasoning not tied to the cue. We extend these results to a more holistic sense of monitorability by introducing verbosity: whether the CoT lists every factor needed to solve the task. We combine faithfulness and verbosity into a single monitorability score that shows how well the CoT serves as the model's external `working memory', a property that many safety schemes based on CoT monitoring depend on. We evaluate instruction-tuned and reasoning models on BBH, GPQA, and MMLU. Our results show that models can appear faithful yet remain hard to monitor when they leave out key factors, and that monitorability differs sharply across model families. We release our evaluation code using the Inspect library to support reproducible future work.
>
---
#### [replaced 052] SpikingBrain: Spiking Brain-inspired Large Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对大语言模型在长文本处理中的效率瓶颈，提出脑启发的SpikingBrain模型。通过线性注意力、脉冲神经网络与硬件优化，实现低内存、高效率训练与推理，支持非NVIDIA平台大规模部署，显著提升长序列处理速度与能效。**

- **链接: [https://arxiv.org/pdf/2509.05276v3](https://arxiv.org/pdf/2509.05276v3)**

> **作者:** Yuqi Pan; Yupeng Feng; Jinghao Zhuang; Siyu Ding; Han Xu; Zehao Liu; Bohan Sun; Yuhong Chou; Xuerui Qiu; Anlin Deng; Anjie Hu; Shurong Wang; Peng Zhou; Man Yao; Jibin Wu; Jian Yang; Guoliang Sun; Bo Xu; Guoqi Li
>
> **摘要:** Mainstream Transformer-based large language models face major efficiency bottlenecks: training computation scales quadratically with sequence length, and inference memory grows linearly, limiting long-context processing. Building large models on non-NVIDIA platforms also poses challenges for stable and efficient training. To address this, we introduce SpikingBrain, a family of brain-inspired models designed for efficient long-context training and inference. SpikingBrain leverages the MetaX GPU cluster and focuses on three aspects: (1) Model Architecture: linear and hybrid-linear attention architectures with adaptive spiking neurons; (2) Algorithmic Optimizations: an efficient, conversion-based training pipeline and a dedicated spike coding framework; (3) System Engineering: customized training frameworks, operator libraries, and parallelism strategies tailored to MetaX hardware. Using these techniques, we develop two models: SpikingBrain-7B, a linear LLM, and SpikingBrain-76B, a hybrid-linear MoE LLM. These models demonstrate the feasibility of large-scale LLM development on non-NVIDIA platforms, and training remains stable for weeks on hundreds of MetaX GPUs with Model FLOPs Utilization at expected levels. SpikingBrain achieves performance comparable to open-source Transformer baselines while using only about 150B tokens for continual pre-training. Our models also significantly improve long-context efficiency and deliver inference with (partially) constant memory and event-driven spiking behavior. For example, SpikingBrain-7B attains over 100x speedup in Time to First Token for 4M-token sequences. Furthermore, the proposed spiking scheme achieves 69.15 percent sparsity, enabling low-power operation. Overall, this work demonstrates the potential of brain-inspired mechanisms to drive the next generation of efficient and scalable large model design.
>
---
#### [replaced 053] A Method for Handling Negative Similarities in Explainable Graph Spectral Clustering of Text Documents -- Extended Version
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究文本文档的可解释图谱聚类任务，针对doc2vec、GloVe等嵌入产生的负相似性问题，提出解决组合与归一化拉普拉斯矩阵中负相似性的方法，实验验证其能提升聚类准确率并使原适用于词向量空间的解释方法适用于GloVe。**

- **链接: [https://arxiv.org/pdf/2504.12360v2](https://arxiv.org/pdf/2504.12360v2)**

> **作者:** Mieczysław A. Kłopotek; Sławomir T. Wierzchoń; Bartłomiej Starosta; Dariusz Czerski; Piotr Borkowski
>
> **备注:** 1 figure, 17 pages, this is an extended version of a paper accepted for the 25th International Conference on Computational Science (ICCS), 7-9 July 2025
>
> **摘要:** This paper investigates the problem of Graph Spectral Clustering with negative similarities, resulting from document embeddings different from the traditional Term Vector Space (like doc2vec, GloVe, etc.). Solutions for combinatorial Laplacians and normalized Laplacians are discussed. An experimental investigation shows the advantages and disadvantages of 6 different solutions proposed in the literature and in this research. The research demonstrates that GloVe embeddings frequently cause failures of normalized Laplacian based GSC due to negative similarities. Furthermore, application of methods curing similarity negativity leads to accuracy improvement for both combinatorial and normalized Laplacian based GSC. It also leads to applicability for GloVe embeddings of explanation methods developed originally bythe authors for Term Vector Space embeddings.
>
---
#### [replaced 054] Efficient LLM-Jailbreaking via Multimodal-LLM Jailbreak
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究大语言模型（LLM）的越狱攻击任务，旨在高效诱导LLM生成不当内容。针对直接越狱效率低的问题，提出通过构建基于目标LLM的多模态模型（MLLM），先对MLLM进行越狱获取嵌入，再转化为文本后缀用于攻击原LLM，显著提升效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2405.20015v3](https://arxiv.org/pdf/2405.20015v3)**

> **作者:** Haoxuan Ji; Zheng Lin; Zhenxing Niu; Xinbo Gao; Gang Hua
>
> **摘要:** This paper focuses on jailbreaking attacks against large language models (LLMs), eliciting them to generate objectionable content in response to harmful user queries. Unlike previous LLM-jailbreak methods that directly orient to LLMs, our approach begins by constructing a multimodal large language model (MLLM) built upon the target LLM. Subsequently, we perform an efficient MLLM jailbreak and obtain a jailbreaking embedding. Finally, we convert the embedding into a textual jailbreaking suffix to carry out the jailbreak of target LLM. Compared to the direct LLM-jailbreak methods, our indirect jailbreaking approach is more efficient, as MLLMs are more vulnerable to jailbreak than pure LLM. Additionally, to improve the attack success rate of jailbreak, we propose an image-text semantic matching scheme to identify a suitable initial input. Extensive experiments demonstrate that our approach surpasses current state-of-the-art jailbreak methods in terms of both efficiency and effectiveness. Moreover, our approach exhibits superior cross-class generalization abilities.
>
---
#### [replaced 055] SpeechIQ: Speech-Agentic Intelligence Quotient Across Cognitive Levels in Voice Understanding by Large Language Models
- **分类: cs.CL; cs.AI; cs.SC; cs.SD; eess.AS**

- **简介: 该论文提出SpeechIQ，一种基于认知层次的语音理解评估框架，针对大语言模型的语音理解能力。它超越传统词错误率，从记忆、理解、应用三层次评估，解决现有评测指标单一、难以发现幻觉与标注错误的问题，统一比较端到端与级联模型，推动多模态训练研究。**

- **链接: [https://arxiv.org/pdf/2507.19361v2](https://arxiv.org/pdf/2507.19361v2)**

> **作者:** Zhen Wan; Chao-Han Huck Yang; Yahan Yu; Jinchuan Tian; Sheng Li; Ke Hu; Zhehuai Chen; Shinji Watanabe; Fei Cheng; Chenhui Chu; Sadao Kurohashi
>
> **备注:** ACL 2025 main. Our Speech-IQ leaderboard is hosted at huggingface.co/spaces/nvidia/Speech-IQ-leaderboard. Speech-IQ Calculator: https://github.com/YukinoWan/SpeechIQ
>
> **摘要:** We introduce Speech-based Intelligence Quotient (SIQ) as a new form of human cognition-inspired evaluation pipeline for voice understanding large language models, LLM Voice, designed to assess their voice understanding ability. Moving beyond popular voice understanding metrics such as word error rate (WER), SIQ examines LLM Voice across three cognitive levels motivated by Bloom's Taxonomy: (1) Remembering (i.e., WER for verbatim accuracy); (2) Understanding (i.e., similarity of LLM's interpretations); and (3) Application (i.e., QA accuracy for simulating downstream tasks). We demonstrate that SIQ not only quantifies voice understanding abilities but also provides unified comparisons between cascaded methods (e.g., ASR LLM) and end-to-end models, identifies annotation errors in existing benchmarks, and detects hallucinations in LLM Voice. Our framework represents a first-of-its-kind intelligence examination that bridges cognitive principles with voice-oriented benchmarks, while exposing overlooked challenges in multi-modal training. Our code and data will be open source to encourage future studies.
>
---
#### [replaced 056] Small Drafts, Big Verdict: Information-Intensive Visual Reasoning via Speculation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对信息密集型图像的视觉问答任务，解决复杂布局中关键线索定位难、多跳推理效率低的问题。提出无需训练的Speculative Verdict框架，通过轻量级草稿专家生成多样推理路径，由强模型综合并筛选高共识路径，实现高效准确的答案生成。**

- **链接: [https://arxiv.org/pdf/2510.20812v2](https://arxiv.org/pdf/2510.20812v2)**

> **作者:** Yuhan Liu; Lianhui Qin; Shengjie Wang
>
> **摘要:** Large Vision-Language Models (VLMs) have achieved remarkable progress in multimodal understanding, yet they struggle when reasoning over information-intensive images that densely interleave textual annotations with fine-grained graphical elements. The main challenges lie in precisely localizing critical cues in dense layouts and multi-hop reasoning to integrate dispersed evidence. We propose Speculative Verdict (SV), a training-free framework inspired by speculative decoding that combines multiple lightweight draft experts with a large verdict model. In the draft stage, small VLMs act as draft experts to generate reasoning paths that provide diverse localization candidates; in the verdict stage, a strong VLM synthesizes these paths to produce the final answer, minimizing computational cost while recovering correct answers. To further improve efficiency and accuracy, SV introduces a consensus expert selection mechanism that forwards only high-agreement reasoning paths to the verdict. Empirically, SV achieves consistent gains on challenging information-intensive and high-resolution visual question answering benchmarks, including InfographicVQA, ChartMuseum, ChartQAPro, and HR-Bench 4K. By synthesizing correct insights from multiple partially accurate reasoning paths, SV achieves both error correction and cost-efficiency compared to large proprietary models or training pipelines. Code is available at https://github.com/Tinaliu0123/speculative-verdict.
>
---
#### [replaced 057] Influence Functions for Efficient Data Selection in Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对大模型推理数据质量评估难题，提出用影响函数量化单个链式思维样本对下游准确率的因果影响，实现高效数据筛选。任务为数学推理中的数据选择，解决“高质量数据”定义模糊问题，通过影响函数剪枝显著优于传统方法。**

- **链接: [https://arxiv.org/pdf/2510.06108v2](https://arxiv.org/pdf/2510.06108v2)**

> **作者:** Prateek Humane; Paolo Cudrano; Daniel Z. Kaplan; Matteo Matteucci; Supriyo Chakraborty; Irina Rish
>
> **备注:** 4 pages, 2 figures; added link to codebase
>
> **摘要:** Fine-tuning large language models (LLMs) on chain-of-thought (CoT) data shows that a small amount of high-quality data can outperform massive datasets. Yet, what constitutes "quality" remains ill-defined. Existing reasoning methods rely on indirect heuristics such as problem difficulty or trace length, while instruction-tuning has explored a broader range of automated selection strategies, but rarely in the context of reasoning. We propose to define reasoning data quality using influence functions, which measure the causal effect of individual CoT examples on downstream accuracy, and introduce influence-based pruning, which consistently outperforms perplexity and embedding-based baselines on math reasoning within a model family.
>
---
#### [replaced 058] SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出SWE-RL，通过强化学习利用开源软件演化数据提升LLM的推理能力。针对现有RL方法局限于竞赛编程与数学问题，该工作首次将RL扩展至真实软件工程场景，基于代码演化数据训练出高性能模型Llama3-SWE-RL-70B，在真实GitHub问题上达到41.0%解决率，且展现出跨任务泛化能力。**

- **链接: [https://arxiv.org/pdf/2502.18449v2](https://arxiv.org/pdf/2502.18449v2)**

> **作者:** Yuxiang Wei; Olivier Duchenne; Jade Copet; Quentin Carbonneaux; Lingming Zhang; Daniel Fried; Gabriel Synnaeve; Rishabh Singh; Sida I. Wang
>
> **备注:** Accepted to NeurIPS 2025 Main Track
>
> **摘要:** The recent DeepSeek-R1 release has demonstrated the immense potential of reinforcement learning (RL) in enhancing the general reasoning capabilities of large language models (LLMs). While DeepSeek-R1 and other follow-up work primarily focus on applying RL to competitive coding and math problems, this paper introduces SWE-RL, the first approach to scale RL-based LLM reasoning for real-world software engineering. Leveraging a lightweight rule-based reward (e.g., the similarity score between ground-truth and LLM-generated solutions), SWE-RL enables LLMs to autonomously recover a developer's reasoning processes and solutions by learning from extensive open-source software evolution data -- the record of a software's entire lifecycle, including its code snapshots, code changes, and events such as issues and pull requests. Trained on top of Llama 3, our resulting reasoning model, Llama3-SWE-RL-70B, achieves a 41.0% solve rate on SWE-bench Verified -- a human-verified collection of real-world GitHub issues. To our knowledge, this is the best performance reported for medium-sized (<100B) LLMs to date, even comparable to leading proprietary LLMs like GPT-4o. Surprisingly, despite performing RL solely on software evolution data, Llama3-SWE-RL has even emerged with generalized reasoning skills. For example, it shows improved results on five out-of-domain tasks, namely, function coding, library use, code reasoning, mathematics, and general language understanding, whereas a supervised-finetuning baseline even leads to performance degradation on average. Overall, SWE-RL opens up a new direction to improve the reasoning capabilities of LLMs through reinforcement learning on massive software engineering data.
>
---
#### [replaced 059] Generating Text from Uniform Meaning Representation
- **分类: cs.CL**

- **简介: 该论文研究多语言统一意义表示（UMR）到文本的生成任务。针对UMR数据稀缺的问题，提出三种方法：直接使用AMR模型、转换UMR为AMR后生成，以及用UMR数据微调模型。实验表明微调方法在英、中文上表现优异，验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2502.11973v2](https://arxiv.org/pdf/2502.11973v2)**

> **作者:** Emma Markle; Reihaneh Iranmanesh; Shira Wein
>
> **备注:** Accepted to IJCNLP-AACL 2026
>
> **摘要:** Uniform Meaning Representation (UMR) is a recently developed graph-based semantic representation, which expands on Abstract Meaning Representation (AMR) in a number of ways, in particular through the inclusion of document-level information and multilingual flexibility. In order to effectively adopt and leverage UMR for downstream tasks, efforts must be placed toward developing a UMR technological ecosystem. Though only a small amount of UMR annotations have been produced to date, in this work, we investigate the first approaches to producing text from multilingual UMR graphs. Exploiting the structural similarity between UMR and AMR graphs and the wide availability of AMR technologies, we introduce (1) a baseline approach which passes UMR graphs to AMR-to-text generation models, (2) a pipeline conversion of UMR to AMR, then using AMR-to-text generation models, and (3) a fine-tuning approach for both foundation models and AMR-to-text generation models with UMR data. Our best performing models achieve multilingual BERTscores of 0.825 for English and 0.882 for Chinese, a promising indication of the effectiveness of fine-tuning approaches for UMR-to-text generation even with limited UMR data.
>
---
#### [replaced 060] Advancing Natural Language Formalization to First Order Logic with Fine-tuned LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦自然语言到一阶逻辑（FOL）的自动形式化任务，旨在提升翻译准确性。通过微调LLMs，对比架构与训练策略，提出词汇扩展、谓词条件等技术，验证了T5模型在谓词可用时性能显著，且具备跨数据集泛化能力，揭示谓词提取是主要瓶颈。**

- **链接: [https://arxiv.org/pdf/2509.22338v2](https://arxiv.org/pdf/2509.22338v2)**

> **作者:** Felix Vossel; Till Mossakowski; Björn Gehrke
>
> **备注:** 15 pages, 7 tables, accepted at the International Joint Conference on Learning & Reasoning (IJCLR 2025)
>
> **摘要:** Automating the translation of natural language to first-order logic (FOL) is crucial for knowledge representation and formal methods, yet remains challenging. We present a systematic evaluation of fine-tuned LLMs for this task, comparing architectures (encoder-decoder vs. decoder-only) and training strategies. Using the MALLS and Willow datasets, we explore techniques like vocabulary extension, predicate conditioning, and multilingual training, introducing metrics for exact match, logical equivalence, and predicate alignment. Our fine-tuned Flan-T5-XXL achieves 70% accuracy with predicate lists, outperforming GPT-4o and even the DeepSeek-R1-0528 model with CoT reasoning ability as well as symbolic systems like ccg2lambda. Key findings show: (1) predicate availability boosts performance by 15-20%, (2) T5 models surpass larger decoder-only LLMs, and (3) models generalize to unseen logical arguments (FOLIO dataset) without specific training. While structural logic translation proves robust, predicate extraction emerges as the main bottleneck.
>
---
#### [replaced 061] GermanPartiesQA: Benchmarking Commercial Large Language Models and AI Companions for Political Alignment and Sycophancy
- **分类: cs.CY; cs.CL**

- **简介: 该论文针对政治领域中闭源大语言模型的偏见问题，提出GermanPartiesQA基准，评估六款商用LLM在德国11次选举中的政治立场识别能力。通过角色扮演实验，发现模型存在事实性不足、意识形态倾向及可被角色引导的现象，但未证实“谄媚”行为，为评估AI在选举支持工具中的政治中立性提供依据。**

- **链接: [https://arxiv.org/pdf/2407.18008v2](https://arxiv.org/pdf/2407.18008v2)**

> **作者:** Jan Batzner; Volker Stocker; Stefan Schmid; Gjergji Kasneci
>
> **备注:** Published at AAAI/ACM AIES 2025. Presented at NeurIPS 2025 Workshop on LLM Evaluation and the International Monetary Fund's 12th Statistical Forum. GermanPartiesQA Benchmark under https://github.com/janbatzner/germanpartiesqa
>
> **摘要:** Large language models (LLMs) are increasingly shaping citizens' information ecosystems. Products incorporating LLMs, such as chatbots and AI Companions, are now widely used for decision support and information retrieval, including in sensitive domains, raising concerns about hidden biases and growing potential to shape individual decisions and public opinion. This paper introduces GermanPartiesQA, a benchmark of 418 political statements from German Voting Advice Applications across 11 elections to evaluate six commercial LLMs. We evaluate their political alignment based on role-playing experiments with political personas. Our evaluation reveals three specific findings: (1) Factual limitations: LLMs show limited ability to accurately generate factual party positions, particularly for centrist parties. (2) Model-specific ideological alignment: We identify consistent alignment patterns and the degree of political steerability for each model across temperature settings and experiments. (3) Claim of sycophancy: While models adjust to political personas during role-play, we find this reflects persona-based steerability rather than the increasingly popular, yet contested concept of sycophancy. Our study contributes to evaluating the political alignment of closed-source LLMs that are increasingly embedded in electoral decision support tools and AI Companion chatbots.
>
---
#### [replaced 062] Black-Box On-Policy Distillation of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一种黑盒、在线策略的大型语言模型蒸馏方法GAD，通过生成对抗机制训练学生模型，使其输出逼近教师模型。旨在解决无内部参数访问时高效蒸馏的问题，实验表明其性能优于传统方法，显著提升学生模型表现。**

- **链接: [https://arxiv.org/pdf/2511.10643v2](https://arxiv.org/pdf/2511.10643v2)**

> **作者:** Tianzhu Ye; Li Dong; Zewen Chi; Xun Wu; Shaohan Huang; Furu Wei
>
> **摘要:** Black-box distillation creates student large language models (LLMs) by learning from a proprietary teacher model's text outputs alone, without access to its internal logits or parameters. In this work, we introduce Generative Adversarial Distillation (GAD), which enables on-policy and black-box distillation. GAD frames the student LLM as a generator and trains a discriminator to distinguish its responses from the teacher LLM's, creating a minimax game. The discriminator acts as an on-policy reward model that co-evolves with the student, providing stable, adaptive feedback. Experimental results show that GAD consistently surpasses the commonly used sequence-level knowledge distillation. In particular, Qwen2.5-14B-Instruct (student) trained with GAD becomes comparable to its teacher, GPT-5-Chat, on the LMSYS-Chat automatic evaluation. The results establish GAD as a promising and effective paradigm for black-box LLM distillation.
>
---
#### [replaced 063] A Machine Learning Approach for Detection of Mental Health Conditions and Cyberbullying from Social Media
- **分类: cs.CL; cs.SI**

- **简介: 该论文针对社交媒体中精神健康问题与网络欺凌的检测难题，提出一种统一的多分类框架。通过构建平衡训练集与真实分布测试集，对比多种模型，发现领域适配的MentalBERT表现最佳。研究强调可解释性与伦理安全，开发了可视化工具辅助人工审核，旨在提供可信赖的筛查支持。**

- **链接: [https://arxiv.org/pdf/2511.20001v2](https://arxiv.org/pdf/2511.20001v2)**

> **作者:** Edward Ajayi; Martha Kachweka; Mawuli Deku; Emily Aiken
>
> **备注:** Accepted for Oral Presentation at the AAAI-26 Bridge Program on AI for Medicine and Healthcare (AIMedHealth). To appear in Proceedings of Machine Learning Research (PMLR)
>
> **摘要:** Mental health challenges and cyberbullying are increasingly prevalent in digital spaces, necessitating scalable and interpretable detection systems. This paper introduces a unified multiclass classification framework for detecting ten distinct mental health and cyberbullying categories from social media data. We curate datasets from Twitter and Reddit, implementing a rigorous "split-then-balance" pipeline to train on balanced data while evaluating on a realistic, held-out imbalanced test set. We conducted a comprehensive evaluation comparing traditional lexical models, hybrid approaches, and several end-to-end fine-tuned transformers. Our results demonstrate that end-to-end fine-tuning is critical for performance, with the domain-adapted MentalBERT emerging as the top model, achieving an accuracy of 0.92 and a Macro F1 score of 0.76, surpassing both its generic counterpart and a zero-shot LLM baseline. Grounded in a comprehensive ethical analysis, we frame the system as a human-in-the-loop screening aid, not a diagnostic tool. To support this, we introduce a hybrid SHAPLLM explainability framework and present a prototype dashboard ("Social Media Screener") designed to integrate model predictions and their explanations into a practical workflow for moderators. Our work provides a robust baseline, highlighting future needs for multi-label, clinically-validated datasets at the critical intersection of online safety and computational mental health.
>
---
#### [replaced 064] IoT-LLM: a framework for enhancing Large Language Model reasoning from real-world sensor data
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对大语言模型（LLM）在物理世界推理中的不足，提出IoT-LLM框架，通过融合物联网传感器数据与知识增强，提升模型感知与推理能力。工作包括数据预处理、知识扩展和思维链激活，设计了多任务基准，实验表明其显著提升模型性能。**

- **链接: [https://arxiv.org/pdf/2410.02429v4](https://arxiv.org/pdf/2410.02429v4)**

> **作者:** Tuo An; Yunjiao Zhou; Han Zou; Jianfei Yang
>
> **备注:** 33 pages, 13 figures
>
> **摘要:** Large Language Models excel in textual tasks but often struggle with physical-world reasoning tasks. Inspired by human cognition, where perception is fundamental to reasoning, we explore augmenting LLMs with enhanced perception abilities using Internet of Things (IoT) data and pertinent knowledge. In this work, we systematically study LLMs' capability to address IoT-sensory tasks by augmenting their perception and knowledge base, and then propose a unified framework, IoT-LLM, to enhance such capability. In IoT-LLM, we customize three steps: preprocessing IoT data into suitable formats, expanding LLMs knowledge via IoT-oriented retrieval-augmented generation and activating LLMs commonsense knowledge through chain-of-thought prompting. We design a benchmark comprising five real-world tasks with varying data types and reasoning complexities to evaluate the performance of IoT-LLM. Experimental results reveal that IoT-LLM significantly improves the performance of IoT-sensory task reasoning of LLMs, with models like GPT-4o-mini showing a 49.4% average improvement over previous methods.
>
---
#### [replaced 065] InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文提出InvisibleInk框架，解决大模型生成私有长文本时的隐私保护问题。通过改进采样机制，仅对敏感信息进行差分隐私处理，降低计算开销。实验证明其在保持文本质量的同时，计算成本仅为基线的1/8，首次实现低成本高质私有长文本生成。**

- **链接: [https://arxiv.org/pdf/2507.02974v2](https://arxiv.org/pdf/2507.02974v2)**

> **作者:** Vishnu Vinod; Krishna Pillutla; Abhradeep Guha Thakurta
>
> **备注:** Published at NeurIPS 2025
>
> **摘要:** As major progress in LLM-based long-form text generation enables paradigms such as retrieval-augmented generation (RAG) and inference-time scaling, safely incorporating private information into the generation remains a critical open question. We present InvisibleInk, a highly scalable long-form text generation framework satisfying rigorous differential privacy guarantees with respect to the sensitive reference texts. It interprets sampling from the LLM's next-token-distribution as the exponential mechanism over the LLM logits with two innovations. First, we reduce the privacy cost by isolating and clipping only the sensitive information in the model logits (relative to the public logits). Second, we improve text quality by sampling without any privacy cost from a small superset of the top-$k$ private tokens. Empirical evaluations demonstrate a consistent $8\times$ (or more) reduction in computation cost over state-of-the-art baselines to generate long-form private text of the same utility across privacy levels. InvisibleInk is able to generate, for the first time, high-quality private long-form text at less than $4$-$8\times$ times the computation cost of non-private generation, paving the way for its practical use. We open-source a pip-installable Python package (invink) for InvisibleInk at https://github.com/cerai-iitm/invisibleink.
>
---
#### [replaced 066] NeKo: Cross-Modality Post-Recognition Error Correction with Tasks-Guided Mixture-of-Experts Language Model
- **分类: cs.CL; cs.AI; cs.LG; cs.MA; eess.AS**

- **简介: 该论文针对多模态语音、语言、视觉到文本的识别后纠错任务，提出基于任务引导专家混合（MoE）的NeKo模型。通过让专家学习特定数据集特征并动态路由，实现单模型高效融合多领域知识，显著降低错误率，在多个基准上超越现有模型。**

- **链接: [https://arxiv.org/pdf/2411.05945v2](https://arxiv.org/pdf/2411.05945v2)**

> **作者:** Yen-Ting Lin; Zhehuai Chen; Piotr Zelasko; Zhen Wan; Xuesong Yang; Zih-Ching Chen; Krishna C Puvvada; Szu-Wei Fu; Ke Hu; Jun Wei Chiu; Jagadeesh Balam; Boris Ginsburg; Yu-Chiang Frank Wang; Chao-Han Huck Yang
>
> **备注:** ACL 2025 Industry Track. NeKo LMs: https://huggingface.co/nvidia/NeKo-v0-post-correction
>
> **摘要:** Construction of a general-purpose post-recognition error corrector poses a crucial question: how can we most effectively train a model on a large mixture of domain datasets? The answer would lie in learning dataset-specific features and digesting their knowledge in a single model. Previous methods achieve this by having separate correction language models, resulting in a significant increase in parameters. In this work, we present Mixture-of-Experts as a solution, highlighting that MoEs are much more than a scalability tool. We propose a Multi-Task Correction MoE, where we train the experts to become an ``expert'' of speech-to-text, language-to-text and vision-to-text datasets by learning to route each dataset's tokens to its mapped expert. Experiments on the Open ASR Leaderboard show that we explore a new state-of-the-art performance by achieving an average relative 5.0% WER reduction and substantial improvements in BLEU scores for speech and translation tasks. On zero-shot evaluation, NeKo outperforms GPT-3.5 and Claude-Opus with 15.5% to 27.6% relative WER reduction in the Hyporadise benchmark. NeKo performs competitively on grammar and post-OCR correction as a multi-task model.
>
---
#### [replaced 067] Evaluating Large Language Models on the 2026 Korean CSAT Mathematics Exam: Measuring Mathematical Ability in a Zero-Data-Leakage Setting
- **分类: cs.CL**

- **简介: 该论文评估大语言模型在2026年韩国高考数学考试上的数学推理能力，旨在解决评测中数据泄露问题。研究构建了零数据泄露的评估环境，对24个主流模型进行多模态、多语言测试，发现部分模型接近满分，但微调与推理增强显著增加开销。**

- **链接: [https://arxiv.org/pdf/2511.18649v2](https://arxiv.org/pdf/2511.18649v2)**

> **作者:** Goun Pyeon; Inbum Heo; Jeesu Jung; Taewook Hwang; Hyuk Namgoong; Hyein Seo; Yerim Han; Eunbin Kim; Hyeonseok Kang; Sangkeun Jung
>
> **备注:** 52 pages
>
> **摘要:** This study systematically evaluated the mathematical reasoning capabilities of Large Language Models (LLMs) using the 2026 Korean College Scholastic Ability Test (CSAT) Mathematics section, ensuring a completely contamination-free evaluation environment. To address data leakage issues in existing benchmarks, we digitized all 46 questions (22 common and 24 elective) within two hours of the exam's public release, eliminating any possibility of inclusion in model training data. We conducted comprehensive evaluations of 24 state-of-the-art LLMs across varying input modalities (Text-only, Image-only, Text+Figure) and prompt languages (Korean, English). The GPT-5 family models achieved perfect scores (100 points) under a limited set of language-modality configurations, while Grok 4, Qwen 3 235B, and Gemini 2.5 pro also scored above 97 points. Notably, gpt-oss-20B achieved 95.7 points despite its relatively small size, demonstrating high cost-effectiveness. Problem-specific analysis revealed Calculus as the weakest domain with significant performance degradation on 4-point high-difficulty problems. Text input consistently outperformed image input, while prompt language effects varied by model scale. In reasoning enhancement experiments with GPT-5 series, increased reasoning intensity improved performance (82.6->100 points) but quadrupled token usage and drastically reduced efficiency, suggesting that models with minimal reasoning may be more practical. This research contributes: (1) implementation of a completely unexposed evaluation environment, (2) a standardized digitization pipeline that converts human-targeted exam materials into LLM-ready evaluation data, and (3) a practical evaluation perspective integrating performance, cost, and time considerations. Detailed results and model comparisons are available at the 2026 Korean CSAT LLM Evaluation Leaderboard; https://isoft.cnu.ac.kr/csat2026/
>
---
#### [replaced 068] Prompt-OT: An Optimal Transport Regularization Paradigm for Knowledge Preservation in Vision-Language Model Adaptation
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文针对视觉-语言模型（VLM）在下游任务适应中出现的过拟合与零样本泛化能力下降问题，提出基于最优传输（OT）的提示学习框架Prompt-OT。通过约束视觉与文本特征分布的结构一致性，实现知识保留与性能提升，在多个评估场景中优于现有方法。**

- **链接: [https://arxiv.org/pdf/2503.08906v3](https://arxiv.org/pdf/2503.08906v3)**

> **作者:** Xiwen Chen; Wenhui Zhu; Peijie Qiu; Hao Wang; Huayu Li; Haiyu Wu; Aristeidis Sotiras; Yalin Wang; Abolfazl Razi
>
> **备注:** Accepted to WACV 2026
>
> **摘要:** Vision-language models (VLMs) such as CLIP demonstrate strong performance but struggle when adapted to downstream tasks. Prompt learning has emerged as an efficient and effective strategy to adapt VLMs while preserving their pre-trained knowledge. However, existing methods still lead to overfitting and degrade zero-shot generalization. To address this challenge, we propose an optimal transport (OT)-guided prompt learning framework that mitigates forgetting by preserving the structural consistency of feature distributions between pre-trained and fine-tuned models. Unlike conventional point-wise constraints, OT naturally captures cross-instance relationships and expands the feasible parameter space for prompt tuning, allowing a better trade-off between adaptation and generalization. Our approach enforces joint constraints on both vision and text representations, ensuring a holistic feature alignment. Extensive experiments on benchmark datasets demonstrate that our simple yet effective method can outperform existing prompt learning strategies in base-to-novel generalization, cross-dataset evaluation, and domain generalization without additional augmentation or ensemble techniques. The code is available at https://github.com/ChongQingNoSubway/Prompt-OT
>
---
#### [replaced 069] PARROT: Persuasion and Agreement Robustness Rating of Output Truth -- A Sycophancy Robustness Benchmark for LLMs
- **分类: cs.CL; cs.AI; cs.CE; cs.LG**

- **简介: 该论文提出PARROT框架，评估大模型在权威与说服压力下的鲁棒性，解决模型“谄媚”（sycophancy）导致的错误响应问题。通过双盲测试、置信度追踪和行为分类，发现小模型易受误导，准确率与信心双重下降，主张将抗压能力纳入模型安全核心目标。**

- **链接: [https://arxiv.org/pdf/2511.17220v2](https://arxiv.org/pdf/2511.17220v2)**

> **作者:** Yusuf Çelebi; Özay Ezerceli; Mahmoud El Hussieni
>
> **摘要:** This study presents PARROT (Persuasion and Agreement Robustness Rating of Output Truth), a robustness focused framework designed to measure the degradation in accuracy that occurs under social pressure exerted on users through authority and persuasion in large language models (LLMs) the phenomenon of sycophancy (excessive conformity). PARROT (i) isolates causal effects by comparing the neutral version of the same question with an authoritatively false version using a double-blind evaluation, (ii) quantifies confidence shifts toward the correct and imposed false responses using log-likelihood-based calibration tracking, and (iii) systematically classifies failure modes (e.g., robust correct, sycophantic agreement, reinforced error, stubborn error, self-correction, etc.) using an eight-state behavioral taxonomy. We evaluated 22 models using 1,302 MMLU-style multiple-choice questions across 13 domains and domain-specific authority templates. Findings show marked heterogeneity: advanced models (e.g., GPT-5, GPT-4.1, Claude Sonnet 4.5) exhibit low "follow rates" ($\leq 11\%$, GPT-5: 4\%) and minimal accuracy loss, while older/smaller models show severe epistemic collapse (GPT-4: 80\%, Qwen 2.5-1.5B: 94\%). The danger is not limited to response changes; weak models reduce confidence in the correct response while increasing confidence in the imposed incorrect response. While international law and global knowledge at the domain level exhibit high fragility, elementary mathematics is relatively resilient. Consequently, we argue that the goal of "resistance to overfitting pressure" should be addressed as a primary objective alongside accuracy, harm avoidance, and privacy for safe deployment in the real world.
>
---
#### [replaced 070] Predicting the Performance of Black-box LLMs through Follow-up Queries
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究黑箱大模型行为预测任务，旨在解决无法访问模型内部结构时准确判断其输出可靠性的问题。通过设计后续追问并利用响应概率作为表示，训练线性模型实现对模型正确性、对抗攻击及模型伪装的可靠检测，效果甚至优于白盒方法。**

- **链接: [https://arxiv.org/pdf/2501.01558v4](https://arxiv.org/pdf/2501.01558v4)**

> **作者:** Dylan Sam; Marc Finzi; J. Zico Kolter
>
> **备注:** NeurIPS 2025
>
> **摘要:** Reliably predicting the behavior of language models -- such as whether their outputs are correct or have been adversarially manipulated -- is a fundamentally challenging task. This is often made even more difficult as frontier language models are offered only through closed-source APIs, providing only black-box access. In this paper, we predict the behavior of black-box language models by asking follow-up questions and taking the probabilities of responses \emph{as} representations to train reliable predictors. We first demonstrate that training a linear model on these responses reliably and accurately predicts model correctness on question-answering and reasoning benchmarks. Surprisingly, this can \textit{even outperform white-box linear predictors} that operate over model internals or activations. Furthermore, we demonstrate that these follow-up question responses can reliably distinguish between a clean version of an LLM and one that has been adversarially influenced via a system prompt to answer questions incorrectly or to introduce bugs into generated code. Finally, we show that they can also be used to differentiate between black-box LLMs, enabling the detection of misrepresented models provided through an API. Overall, our work shows promise in monitoring black-box language model behavior, supporting their deployment in larger, autonomous systems.
>
---
#### [replaced 071] Reliable Reasoning Beyond Natural Language
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型在复杂推理中的可靠性问题，提出NLR数据集与神经符号推理方法。通过引入Prolog符号引擎，将模型从迭代计算中解放，实现多路径思维的高效推理，在多个基准上显著提升性能，尤其在高变量耦合场景下仍保持高准确率。**

- **链接: [https://arxiv.org/pdf/2407.11373v3](https://arxiv.org/pdf/2407.11373v3)**

> **作者:** Nasim Borazjanizadeh; Steven T. Piantadosi
>
> **摘要:** Despite their linguistic competence, Large Language Models (LLMs) often struggle to reason reliably and flexibly. To identify these shortcomings, we introduce the Non-Linear Reasoning (NLR) dataset, a collection of 55 unique, hand-designed problems that target reasoning bottlenecks arising from the sequential prediction paradigm of LLMs and the inherently linear nature of natural language. NLR tasks require iterative updates, backtracking, and reasoning across multiple parallel chains of thought but only basic arithmetic to solve. To address these limitations, we propose a neurosymbolic reasoning approach that integrates Prolog, a symbolic reasoning engine, into the inference pipeline of LLMs. This division of labor shifts the LLM's task from iterative computations to inferring all information, explicit or implied through common sense, and encoding it as logical code. Our method yields large and robust performance gains across the GSM8k and BIG-bench Navigate benchmarks and achieves near-perfect accuracy on NLR problems, maintaining robustness even as variable interdependence - the number of other variables on which the value of a single variable depends - increases.
>
---
#### [replaced 072] Confident RAG: Enhancing the Performance of LLMs for Mathematics Question Answering through Multi-Embedding and Confidence Scoring
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型在数学问答中推理能力不足的问题，提出Confident RAG方法。通过多嵌入模型生成多个答案并选择高置信度结果，显著提升准确率。相比基础LLM和RAG，分别提升约10%和5%，验证了其高效性与普适性，为教育场景中的智能助教系统提供可靠方案。**

- **链接: [https://arxiv.org/pdf/2507.17442v3](https://arxiv.org/pdf/2507.17442v3)**

> **作者:** Shiting Chen; Zijian Zhao; Jinsong Chen
>
> **摘要:** Large Language Models (LLMs) hold significant promise for mathematics education, yet they often struggle with complex mathematical reasoning. While Retrieval-Augmented Generation (RAG) mitigates these issues by grounding LLMs in external knowledge, its effectiveness remains unstable, heavily dependent on the choice of a single embedding model. Moving beyond static RAG workflows, we draw on agentic workflow patterns, a paradigm that introduces structured task decomposition and collaboration to enhance system performance. We propose and examine two novel approaches that combine the benefits of multiple embedding models. While our Mixture-Embedding RAG approach (fusing retrieved documents) shows limited gains, our Confident RAG method (generating multiple answers and selecting the one with the highest confidence score) demonstrates significant improvement. Experimental results show that Confident RAG achieved average accuracy improvements of approximately 10% over vanilla LLMs and 5% over vanilla RAG. The consistent results across different LLMs and embedding models indicate that Confident RAG is an efficient plug-and-play solution for trustworthy mathematical AI assistants. Finally, we discuss how this work lays the groundwork for deploying Agentic RAG systems in educational settings, where autonomous planning and iterative refinement can be built upon our robust retrieval foundation.
>
---
#### [replaced 073] Do different prompting methods yield a common task representation in language models?
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究语言模型在不同提示方式（演示与指令）下是否产生相同任务表征。通过推广函数向量（FVs）方法，发现两类提示激发不同的模型组件，虽有部分重叠，但无共同表征。结果支持结合多种提示形式，并揭示了任务推理机制的复杂性。**

- **链接: [https://arxiv.org/pdf/2505.12075v3](https://arxiv.org/pdf/2505.12075v3)**

> **作者:** Guy Davidson; Todd M. Gureckis; Brenden M. Lake; Adina Williams
>
> **备注:** 10 pages, 4 figures; presented at NeurIPS 2025
>
> **摘要:** Demonstrations and instructions are two primary approaches for prompting language models to perform in-context learning (ICL) tasks. Do identical tasks elicited in different ways result in similar representations of the task? An improved understanding of task representation mechanisms would offer interpretability insights and may aid in steering models. We study this through \textit{function vectors} (FVs), recently proposed as a mechanism to extract few-shot ICL task representations. We generalize FVs to alternative task presentations, focusing on short textual instruction prompts, and successfully extract instruction function vectors that promote zero-shot task accuracy. We find evidence that demonstration- and instruction-based function vectors leverage different model components, and offer several controls to dissociate their contributions to task performance. Our results suggest that different task promptings forms do not induce a common task representation through FVs but elicit different, partly overlapping mechanisms. Our findings offer principled support to the practice of combining instructions and task demonstrations, imply challenges in universally monitoring task inference across presentation forms, and encourage further examinations of LLM task inference mechanisms.
>
---
#### [replaced 074] Enhancing OCR for Sino-Vietnamese Language Processing via Fine-tuned PaddleOCRv5
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对古越南汉喃文（Han-Nom）文本的光学字符识别难题，提出对PaddleOCRv5进行微调。通过在古籍手稿数据上重训练识别模块，显著提升在模糊、非标准字形等复杂条件下的识别准确率（37.5%→50.0%），并开发交互式演示系统，助力历史文献数字化与跨语言研究。**

- **链接: [https://arxiv.org/pdf/2510.04003v2](https://arxiv.org/pdf/2510.04003v2)**

> **作者:** Minh Hoang Nguyen; Su Nguyen Thiet
>
> **备注:** Short Paper: 7 pages, 8 figures, 3 tables
>
> **摘要:** Recognizing and processing Classical Chinese (Han-Nom) texts play a vital role in digitizing Vietnamese historical documents and enabling cross-lingual semantic research. However, existing OCR systems struggle with degraded scans, non-standard glyphs, and handwriting variations common in ancient sources. In this work, we propose a fine-tuning approach for PaddleOCRv5 to improve character recognition on Han-Nom texts. We retrain the text recognition module using a curated subset of ancient Vietnamese Chinese manuscripts, supported by a full training pipeline covering preprocessing, LMDB conversion, evaluation, and visualization. Experimental results show a significant improvement over the base model, with exact accuracy increasing from 37.5 percent to 50.0 percent, particularly under noisy image conditions. Furthermore, we develop an interactive demo that visually compares pre- and post-fine-tuning recognition results, facilitating downstream applications such as Han-Vietnamese semantic alignment, machine translation, and historical linguistics research. The demo is available at https://huggingface.co/spaces/MinhDS/Fine-tuned-PaddleOCRv5
>
---
#### [replaced 075] Med-gte-hybrid: A contextual embedding transformer model for extracting actionable information from clinical texts
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出med-gte-hybrid模型，用于从临床文本中提取可操作信息。针对非结构化医疗文本理解难题，通过对比学习与去噪自编码器融合调优，提升语义表征能力。在MIMIC-IV数据集上验证其在疾病预后、肾功能预测等任务中的优越性能，显著改善患者分层与文本检索效果。**

- **链接: [https://arxiv.org/pdf/2502.15996v3](https://arxiv.org/pdf/2502.15996v3)**

> **作者:** Aditya Kumar; Simon Rauch; Mario Cypko; Oliver Amft
>
> **备注:** 22 pages, 4 figures, 2 tables
>
> **摘要:** We introduce a novel contextual embedding model med-gte-hybrid that was derived from the gte-large sentence transformer to extract information from unstructured clinical narratives. Our model tuning strategy for med-gte-hybrid combines contrastive learning and a denoising autoencoder. To evaluate the performance of med-gte-hybrid, we investigate several clinical prediction tasks in large patient cohorts extracted from the MIMIC-IV dataset, including Chronic Kidney Disease (CKD) patient prognosis, estimated glomerular filtration rate (eGFR) prediction, and patient mortality prediction. Furthermore, we demonstrate that the med-gte-hybrid model improves patient stratification, clustering, and text retrieval, thus outperforms current state-of-the-art models on the Massive Text Embedding Benchmark (MTEB). While some of our evaluations focus on CKD, our hybrid tuning of sentence transformers could be transferred to other medical domains and has the potential to improve clinical decision-making and personalised treatment pathways in various healthcare applications.
>
---
#### [replaced 076] DeID-GPT: Zero-shot Medical Text De-Identification by GPT-4
- **分类: cs.CL; cs.CY**

- **简介: 该论文提出DeID-GPT框架，利用GPT-4实现医疗文本零样本去标识化。针对医疗文本隐私保护中传统方法泛化性差、需调优的问题，通过大模型的命名实体识别能力自动识别并移除敏感信息，有效保留文本结构与语义，提升去标识化准确性和可靠性。**

- **链接: [https://arxiv.org/pdf/2303.11032v3](https://arxiv.org/pdf/2303.11032v3)**

> **作者:** Zhengliang Liu; Yue Huang; Xiaowei Yu; Lu Zhang; Zihao Wu; Chao Cao; Haixing Dai; Lin Zhao; Yiwei Li; Peng Shu; Fang Zeng; Lichao Sun; Wei Liu; Dinggang Shen; Quanzheng Li; Tianming Liu; Dajiang Zhu; Xiang Li
>
> **摘要:** The digitization of healthcare has facilitated the sharing and re-using of medical data but has also raised concerns about confidentiality and privacy. HIPAA (Health Insurance Portability and Accountability Act) mandates removing re-identifying information before the dissemination of medical records. Thus, effective and efficient solutions for de-identifying medical data, especially those in free-text forms, are highly needed. While various computer-assisted de-identification methods, including both rule-based and learning-based, have been developed and used in prior practice, such solutions still lack generalizability or need to be fine-tuned according to different scenarios, significantly imposing restrictions in wider use. The advancement of large language models (LLM), such as ChatGPT and GPT-4, have shown great potential in processing text data in the medical domain with zero-shot in-context learning, especially in the task of privacy protection, as these models can identify confidential information by their powerful named entity recognition (NER) capability. In this work, we developed a novel GPT4-enabled de-identification framework (``DeID-GPT") to automatically identify and remove the identifying information. Compared to existing commonly used medical text data de-identification methods, our developed DeID-GPT showed the highest accuracy and remarkable reliability in masking private information from the unstructured medical text while preserving the original structure and meaning of the text. This study is one of the earliest to utilize ChatGPT and GPT-4 for medical text data processing and de-identification, which provides insights for further research and solution development on the use of LLMs such as ChatGPT/GPT-4 in healthcare. Codes and benchmarking data information are available at https://github.com/yhydhx/ChatGPT-API.
>
---
#### [replaced 077] A Semantic-based Optimization Approach for Repairing LLMs: Case Study on Code Generation
- **分类: cs.SE; cs.CL; cs.LG**

- **简介: 该论文针对大语言模型（LLM）代码生成中的错误问题，提出基于语义优化的修复方法STAR。通过优化过程定位并修复“有缺陷神经元”，实现高效、低副作用的模型修复，显著提升代码生成准确性和泛化能力，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2503.12899v4](https://arxiv.org/pdf/2503.12899v4)**

> **作者:** Jian Gu; Aldeida Aleti; Chunyang Chen; Hongyu Zhang
>
> **备注:** 13 pages, 6 figure, 8 tables, camera-ready version
>
> **摘要:** Language Models (LMs) are widely used in software engineering for code generation, but they may produce erroneous code. Rather than repairing outputs, a more thorough remedy is to address underlying model failures. LM repair offers a lightweight solution: it requires minimal data, lowers computational cost, and limits side effects. Unlike full retraining, LM repair focuses on applying tailored updates to targeted neurons, making it suitable for limited resources, high-performance demands, or strict safety requirements. In this paper, we propose Semantic Targeting for Analytical Repair (STAR), a novel semantic-based optimization method for repairing LLMs. STAR realizes the main operations of repairing LMs in an optimization process, including locating ``buggy neurons'', solving ``neuron patches'', and patching ``buggy neurons''. The neuron patches are computed with a solid semantic-based analytical formula, which directly bridges the changes to logits with the deltas of neurons, by steering latent representations. Compared to the prior work of LM repair (MINT) and standard optimization methods (SGD), STAR integrates their strengths while mitigating their limitations. By reformulating LM repair as an optimization process, STAR may solve multiple failures together, significantly improving the usefulness. Evaluated on coding tasks using popular code LMs, STAR demonstrates superior effectiveness compared with the state-of-the-art. Besides, STAR exhibits better efficiency. In terms of side effects, namely the balance between generalization and specificity, STAR outperforms prior work by a significant margin. Additionally, we conducted assessments on the overfitting risk of LM repair as well as the cumulative impact. Further, we analyzed the differences with pipeline-based methods and explained the reason why STAR is better and how it mitigated the common limitations of LM repair.
>
---
#### [replaced 078] SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **简介: 该论文针对大语言模型（LLM）幻觉问题，提出SECA方法，通过语义等价且连贯的自然修改生成真实可执行的对抗提示，以更真实地触发幻觉。解决了现有攻击方法因引入荒谬或改变原意而缺乏实用性的缺陷，显著提升攻击成功率并保持语义一致性。**

- **链接: [https://arxiv.org/pdf/2510.04398v2](https://arxiv.org/pdf/2510.04398v2)**

> **作者:** Buyun Liang; Liangzu Peng; Jinqi Luo; Darshan Thaker; Kwan Ho Ryan Chan; René Vidal
>
> **备注:** Accepted at NeurIPS 2025. Code is available at https://github.com/Buyun-Liang/SECA
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in high-risk domains. However, state-of-the-art LLMs often produce hallucinations, raising serious concerns about their reliability. Prior work has explored adversarial attacks for hallucination elicitation in LLMs, but it often produces unrealistic prompts, either by inserting gibberish tokens or by altering the original meaning. As a result, these approaches offer limited insight into how hallucinations may occur in practice. While adversarial attacks in computer vision often involve realistic modifications to input images, the problem of finding realistic adversarial prompts for eliciting LLM hallucinations has remained largely underexplored. To address this gap, we propose Semantically Equivalent and Coherent Attacks (SECA) to elicit hallucinations via realistic modifications to the prompt that preserve its meaning while maintaining semantic coherence. Our contributions are threefold: (i) we formulate finding realistic attacks for hallucination elicitation as a constrained optimization problem over the input prompt space under semantic equivalence and coherence constraints; (ii) we introduce a constraint-preserving zeroth-order method to effectively search for adversarial yet feasible prompts; and (iii) we demonstrate through experiments on open-ended multiple-choice question answering tasks that SECA achieves higher attack success rates while incurring almost no semantic equivalence or semantic coherence errors compared to existing methods. SECA highlights the sensitivity of both open-source and commercial gradient-inaccessible LLMs to realistic and plausible prompt variations. Code is available at https://github.com/Buyun-Liang/SECA.
>
---
#### [replaced 079] LLM-based Human Simulations Have Not Yet Been Reliable
- **分类: cs.CL**

- **简介: 该论文属于人行为模拟任务，旨在解决当前大语言模型（LLM）在社会、经济等场景中模拟人类行为可靠性不足的问题。通过系统综述与分析，指出模型局限与设计缺陷，并提出数据、能力与设计优化相结合的解决方案框架及可操作算法，以提升模拟可信度。**

- **链接: [https://arxiv.org/pdf/2501.08579v2](https://arxiv.org/pdf/2501.08579v2)**

> **作者:** Qian Wang; Jiaying Wu; Zichen Jiang; Zhenheng Tang; Bingqiao Luo; Nuo Chen; Wei Chen; Bingsheng He
>
> **摘要:** Large Language Models (LLMs) are increasingly employed for simulating human behaviors across diverse domains. However, our position is that current LLM-based human simulations remain insufficiently reliable, as evidenced by significant discrepancies between their outcomes and authentic human actions. Our investigation begins with a systematic review of LLM-based human simulations in social, economic, policy, and psychological contexts, identifying their common frameworks, recent advances, and persistent limitations. This review reveals that such discrepancies primarily stem from inherent limitations of LLMs and flaws in simulation design, both of which are examined in detail. Building on these insights, we propose a systematic solution framework that emphasizes enriching data foundations, advancing LLM capabilities, and ensuring robust simulation design to enhance reliability. Finally, we introduce a structured algorithm that operationalizes the proposed framework, aiming to guide credible and human-aligned LLM-based simulations. To facilitate further research, we provide a curated list of related literature and resources at https://github.com/Persdre/awesome-llm-human-simulation.
>
---
#### [replaced 080] DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning
- **分类: cs.CL**

- **简介: 该论文提出DESIGNER框架，解决大模型多学科复杂推理数据匮乏问题。通过提取12万+“设计逻辑”，基于书籍与网络文本生成跨75个学科的高难度、多样化推理题，显著提升Qwen3和Llama3模型的多领域推理能力，优于现有数据集及官方优化模型。**

- **链接: [https://arxiv.org/pdf/2508.12726v4](https://arxiv.org/pdf/2508.12726v4)**

> **作者:** Weize Liu; Yongchi Zhao; Yijia Luo; Mingyu Xu; Jiaheng Liu; Yanan Li; Xiguo Hu; Zhiqi Bai; Yuchi Xu; Wenbo Su; Bo Zheng
>
> **摘要:** Large language models (LLMs) have achieved remarkable success in many natural language tasks but still struggle with complex, multi-step reasoning, particularly across diverse disciplines. Existing reasoning datasets often lack disciplinary breadth, reasoning depth, and diversity, as well as guiding principles for question synthesis. We propose DESIGNER: a DESIGN-logic-guidEd Reasoning data synthesis pipeline that leverages naturally available, extensive raw documents (e.g., book corpus and web corpus) to generate multidisciplinary challenging questions. We introduce the concept of "design logic" and instruct LLMs to mimic human educators' question-creation process, enabling the automated synthesis of large-scale, high-difficulty questions. We use LLMs to reverse-engineer and abstract over 120,000 design logics from existing questions across various disciplines. By matching these design logics with source documents, we are able to generate reasoning questions with controllable question types and difficulty levels. Using this pipeline, we synthesized two large-scale reasoning datasets that span 75 disciplines: DLR-Book (3.04 million questions from the book corpus) and DLR-Web (1.66 million questions from the web corpus). Data analysis indicates that the questions synthesized by our method exhibit greater difficulty and diversity compared to those in the baseline datasets. We validate our synthesized data through supervised fine-tuning (SFT) on the Qwen3 and Llama3 model families. Our data substantially enhances their multidisciplinary reasoning capabilities, outperforming existing datasets. Notably, by applying SFT on the base versions of these models using only our data, we even surpass their official final models that have undergone the full post-training process.
>
---
#### [replaced 081] Spark-Prover-X1: Formal Theorem Proving Through Diverse Data Training
- **分类: cs.CL**

- **简介: 该论文针对大语言模型在形式化定理证明中因数据稀缺导致的推理能力受限问题，提出Spark-Prover-X1模型。通过三阶段训练框架，结合多样化数据与渐进式优化，显著提升7B模型的证明能力。构建ExamFormal-Bench基准，实验证明其在竞赛级难题上表现优异，为轻量级模型提供高效形式化推理路径。**

- **链接: [https://arxiv.org/pdf/2511.13043v3](https://arxiv.org/pdf/2511.13043v3)**

> **作者:** Xinyuan Zhou; Yi Lei; Xiaoyu Zhou; Jingyi Sun; Yu Zhu; Zhongyi Ye; Weitai Zhang; Quan Liu; Si Wei; Cong Liu
>
> **摘要:** Large Language Models (LLMs) have shown significant promise in automated theorem proving, yet progress is often constrained by the scarcity of diverse and high-quality formal language data. To address this issue, we introduce Spark-Prover-X1, a 7B parameter model trained via an three-stage framework designed to unlock the reasoning potential of more accessible and moderately-sized LLMs. The first stage infuses deep knowledge through continuous pre-training on a broad mathematical corpus, enhanced by a suite of novel data tasks. Key innovation is a "CoT-augmented state prediction" task to achieve fine-grained reasoning. The second stage employs Supervised Fine-tuning (SFT) within an expert iteration loop to specialize both the Spark-Prover-X1-7B and Spark-Formalizer-X1-7B models. Finally, a targeted round of Group Relative Policy Optimization (GRPO) is applied to sharpen the prover's capabilities on the most challenging problems. To facilitate robust evaluation, particularly on problems from real-world examinations, we also introduce ExamFormal-Bench, a new benchmark dataset of 402 formal problems. Experimental results demonstrate that Spark-Prover achieves state-of-the-art performance among similarly-sized open-source models within the "Whole-Proof Generation" paradigm. It shows exceptional performance on difficult competition benchmarks, notably solving 27 problems on PutnamBench (pass@32) and achieving 24.0\% on CombiBench (pass@32). Our work validates that this diverse training data and progressively refined training pipeline provides an effective path for enhancing the formal reasoning capabilities of lightweight LLMs. We will release both Spark-Prover-X1-7B and Spark-Formalizer-X1-7B, along with the ExamFormal-Bench dataset, in the near future.
>
---
#### [replaced 082] SpeechJudge: Towards Human-Level Judgment for Speech Naturalness
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文针对语音合成中自然度评价缺乏大规模人类反馈数据的问题，提出SpeechJudge体系。构建99K语音对的人类偏好数据集，建立评估基准SpeechJudge-Eval，开发基于Qwen2.5的生成式奖励模型SpeechJudge-GRM，显著提升自然度判断准确率，助力语音模型与人类感知对齐。**

- **链接: [https://arxiv.org/pdf/2511.07931v2](https://arxiv.org/pdf/2511.07931v2)**

> **作者:** Xueyao Zhang; Chaoren Wang; Huan Liao; Ziniu Li; Yuancheng Wang; Li Wang; Dongya Jia; Yuanzhe Chen; Xiulin Li; Zhuo Chen; Zhizheng Wu
>
> **备注:** Dataset, Model, and Code: https://github.com/AmphionTeam/SpeechJudge
>
> **摘要:** Aligning large generative models with human feedback is a critical challenge. In speech synthesis, this is particularly pronounced due to the lack of a large-scale human preference dataset, which hinders the development of models that truly align with human perception. To address this, we introduce SpeechJudge, a comprehensive suite comprising a dataset, a benchmark, and a reward model centered on naturalness--one of the most fundamental subjective metrics for speech synthesis. First, we present SpeechJudge-Data, a large-scale human feedback corpus of 99K speech pairs. The dataset is constructed using a diverse set of advanced zero-shot text-to-speech (TTS) models across diverse speech styles and multiple languages, with human annotations for both intelligibility and naturalness preference. From this, we establish SpeechJudge-Eval, a challenging benchmark for speech naturalness judgment. Our evaluation reveals that existing metrics and AudioLLMs struggle with this task; the leading model, Gemini-2.5-Flash, achieves less than 70% agreement with human judgment, highlighting a significant gap for improvement. To bridge this gap, we develop SpeechJudge-GRM, a generative reward model (GRM) based on Qwen2.5-Omni-7B. It is trained on SpeechJudge-Data via a two-stage post-training process: Supervised Fine-Tuning (SFT) with Chain-of-Thought rationales followed by Reinforcement Learning (RL) with GRPO on challenging cases. On the SpeechJudge-Eval benchmark, the proposed SpeechJudge-GRM demonstrates superior performance, achieving 77.2% accuracy (and 79.4% after inference-time scaling @10) compared to a classic Bradley-Terry reward model (72.7%). Furthermore, SpeechJudge-GRM can be also employed as a reward function during the post-training of speech generation models to facilitate their alignment with human preferences.
>
---
#### [replaced 083] Comprehensive Evaluation on Lexical Normalization: Boundary-Aware Approaches for Unsegmented Languages
- **分类: cs.CL**

- **简介: 该论文针对未分词语言的词汇规范化任务，解决用户生成文本中非正式表达的处理难题。构建了大规模多领域日语数据集，基于预训练模型提出边界感知的规范化方法，并从准确率与效率多角度验证，证明编码器与解码器单一结构均具优异性能。**

- **链接: [https://arxiv.org/pdf/2505.22273v2](https://arxiv.org/pdf/2505.22273v2)**

> **作者:** Shohei Higashiyama; Masao Utiyama
>
> **备注:** EMNLP 2025 (Findings), 26 pages
>
> **摘要:** Lexical normalization research has sought to tackle the challenge of processing informal expressions in user-generated text, yet the absence of comprehensive evaluations leaves it unclear which methods excel across multiple perspectives. Focusing on unsegmented languages, we make three key contributions: (1) creating a large-scale, multi-domain Japanese normalization dataset, (2) developing normalization methods based on state-of-the-art pretrained models, and (3) conducting experiments across multiple evaluation perspectives. Our experiments show that both encoder-only and decoder-only approaches achieve promising results in both accuracy and efficiency.
>
---
#### [replaced 084] Closing the Data-Efficiency Gap Between Autoregressive and Masked Diffusion LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型的知识注入效率问题，针对自回归模型（arLLMs）在微调中数据效率低、易受“逆序诅咒”影响的缺陷，提出一种新的掩码微调范式。通过对比arLLMs与掩码扩散模型（dLLMs）在正向/逆向问答上的表现，验证了新方法能显著提升arLLMs的数据效率，使其性能接近dLLMs，且可扩展至数学能力训练。**

- **链接: [https://arxiv.org/pdf/2510.09885v2](https://arxiv.org/pdf/2510.09885v2)**

> **作者:** Xu Pan; Ely Hahami; Jingxuan Fan; Ziqian Xie; Haim Sompolinsky
>
> **摘要:** Despite autoregressive large language models (arLLMs) being the current dominant paradigm in language modeling, effectively updating these models to incorporate new factual knowledge still remains difficult. They resist knowledge injection via fine-tuning due to inherent shortcomings such as the "reversal curse" -- the challenge of answering questions that reverse the original information order in the training sample. Masked diffusion large language models (dLLMs) are rapidly emerging as a powerful alternative to the arLLM paradigm, with evidence of better data efficiency and free of the "reversal curse" in pre-training. However, it is unknown whether these advantages extend to the post-training phase, i.e. whether pre-trained dLLMs can easily acquire new knowledge through fine-tuning. On three diverse datasets, we fine-tune arLLMs and dLLMs, evaluating them with forward and backward style Question Answering (QA) to probe knowledge generalization and the reversal curse. Our results confirm that arLLMs critically rely on extensive data augmentation via paraphrases for QA generalization, and paraphrases are only effective when their information order matches the QA style. Conversely, dLLMs achieve high accuracies on both forward and backward QAs without paraphrases; adding paraphrases yields only marginal gains. Inspired by the dLLM's performance, we introduce a novel masked fine-tuning paradigm for knowledge injection into pre-trained arLLMs. This proposed method successfully and drastically improves the data efficiency of arLLM fine-tuning, effectively closing its performance gap with dLLMs. We further show that the masked fine-tuning paradigm of arLLMs can be extended to the supervised fine-tuning (SFT) of mathematical capability. Across two models and two datasets, our masked SFT outperforms regular SFT.
>
---
#### [replaced 085] Recursive numeral systems are highly regular and easy to process
- **分类: cs.CL; cs.FL**

- **简介: 该论文研究递归数词系统的最优性问题，旨在解决以往研究因忽略规律性而导致的自然语言系统与非自然系统区分困难的问题。作者引入最小描述长度（MDL）方法，从规律性和处理复杂度角度重新评估系统，证明规律性可自然导出先前依赖人为约束的结果，强调在语言最优性研究中应考虑形式集合的整体规律性。**

- **链接: [https://arxiv.org/pdf/2510.27049v2](https://arxiv.org/pdf/2510.27049v2)**

> **作者:** Ponrawee Prasertsom; Andrea Silvi; Jennifer Culbertson; Moa Johansson; Devdatt Dubhashi; Kenny Smith
>
> **摘要:** Previous work has argued that recursive numeral systems optimise the trade-off between lexicon size and average morphosyntatic complexity (Denić and Szymanik, 2024). However, showing that only natural-language-like systems optimise this tradeoff has proven elusive, and the existing solution has relied on ad-hoc constraints to rule out unnatural systems (Yang and Regier, 2025). Here, we argue that this issue arises because the proposed trade-off has neglected regularity, a crucial aspect of complexity central to human grammars in general. Drawing on the Minimum Description Length (MDL) approach, we propose that recursive numeral systems are better viewed as efficient with regard to their regularity and processing complexity. We show that our MDL-based measures of regularity and processing complexity better capture the key differences between attested, natural systems and unattested but possible ones, including "optimal" recursive numeral systems from previous work, and that the ad-hoc constraints from previous literature naturally follow from regularity. Our approach highlights the need to incorporate regularity across sets of forms in studies that attempt to measure and explain optimality in language.
>
---
#### [replaced 086] Does Understanding Inform Generation in Unified Multimodal Models? From Analysis to Path Forward
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究统一多模态模型中理解与生成的关系，针对“理解是否真正指导生成”这一核心问题，构建了去耦合的评估框架UniSandbox及合成数据集。通过实验证明存在理解-生成差距，提出通过显式链式思维（CoT）和自训练实现隐式推理，并揭示查询架构的潜在CoT特性促进知识迁移。**

- **链接: [https://arxiv.org/pdf/2511.20561v2](https://arxiv.org/pdf/2511.20561v2)**

> **作者:** Yuwei Niu; Weiyang Jin; Jiaqi Liao; Chaoran Feng; Peng Jin; Bin Lin; Zongjian Li; Bin Zhu; Weihao Yu; Li Yuan
>
> **摘要:** Recent years have witnessed significant progress in Unified Multimodal Models, yet a fundamental question remains: Does understanding truly inform generation? To investigate this, we introduce UniSandbox, a decoupled evaluation framework paired with controlled, synthetic datasets to avoid data leakage and enable detailed analysis. Our findings reveal a significant understanding-generation gap, which is mainly reflected in two key dimensions: reasoning generation and knowledge transfer. Specifically, for reasoning generation tasks, we observe that explicit Chain-of-Thought (CoT) in the understanding module effectively bridges the gap, and further demonstrate that a self-training approach can successfully internalize this ability, enabling implicit reasoning during generation. Additionally, for knowledge transfer tasks, we find that CoT assists the generative process by helping retrieve newly learned knowledge, and also discover that query-based architectures inherently exhibit latent CoT-like properties that affect this transfer. UniSandbox provides preliminary insights for designing future unified architectures and training strategies that truly bridge the gap between understanding and generation. Code and data are available at https://github.com/PKU-YuanGroup/UniSandBox
>
---
#### [replaced 087] Superposition Yields Robust Neural Scaling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大语言模型的神经缩放定律，旨在揭示模型性能随规模增长的机制。通过控制表示超叠加程度，发现强超叠加下损失与模型维度呈反比关系，解释了缩放律的普适性，并验证了开源模型和Chinchilla定律与此一致，指出超叠加是缩放律的核心驱动因素。**

- **链接: [https://arxiv.org/pdf/2505.10465v4](https://arxiv.org/pdf/2505.10465v4)**

> **作者:** Yizhou Liu; Ziming Liu; Jeff Gore
>
> **备注:** Best Paper Runner-up at NeurIPS 2025
>
> **摘要:** The success of today's large language models (LLMs) depends on the observation that larger models perform better. However, the origin of this neural scaling law, that loss decreases as a power law with model size, remains unclear. We propose that representation superposition, meaning that LLMs represent more features than they have dimensions, can be a key contributor to loss and cause neural scaling. Based on Anthropic's toy model, we use weight decay to control the degree of superposition, allowing us to systematically study how loss scales with model size. When superposition is weak, the loss follows a power law only if data feature frequencies are power-law distributed. In contrast, under strong superposition, the loss generically scales inversely with model dimension across a broad class of frequency distributions, due to geometric overlaps between representation vectors. We confirmed that open-sourced LLMs operate in the strong superposition regime and have loss scaling inversely with model dimension, and that the Chinchilla scaling laws are also consistent with this behavior. Our results identify representation superposition as a central driver of neural scaling laws, providing insights into questions like when neural scaling laws can be improved and when they will break down.
>
---
#### [replaced 088] A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对小语言模型（SLM）高效训练难题，提出低秩克隆（LRC）方法。通过联合实现权重软剪枝与激活克隆，提升知识迁移效率，解决信息丢失、表征对齐差和FFN信号利用不足问题。实验表明，仅用20B token即可超越万亿级数据训练的SOTA模型，训练效率超1000倍。**

- **链接: [https://arxiv.org/pdf/2505.12781v3](https://arxiv.org/pdf/2505.12781v3)**

> **作者:** Jitai Hao; Qiang Huang; Hao Liu; Xinyan Xiao; Zhaochun Ren; Jun Yu
>
> **备注:** NeurIPS 2025 Spotlight
>
> **摘要:** Training high-performing Small Language Models (SLMs) remains costly, even with knowledge distillation and pruning from larger teacher models. Existing work often faces three key challenges: (1) information loss from hard pruning, (2) inefficient alignment of representations, and (3) underutilization of informative activations, particularly from Feed-Forward Networks (FFNs). To address these challenges, we introduce Low-Rank Clone (LRC), an efficient pre-training method that constructs SLMs aspiring to behavioral equivalence with strong teacher models. LRC trains a set of low-rank projection matrices that jointly enable soft pruning by compressing teacher weights, and activation clone by aligning student activations, including FFN signals, with those of the teacher. This unified design maximizes knowledge transfer while removing the need for explicit alignment modules. Extensive experiments with open-source teachers (e.g., Llama-3.2-3B-Instruct, Qwen2.5-3B/7B-Instruct) show that LRC matches or surpasses state-of-the-art models trained on trillions of tokens--while using only 20B tokens, achieving over 1,000x training efficiency. Our codes and model checkpoints are available at https://github.com/CURRENTF/LowRankClone and https://huggingface.co/collections/JitaiHao/low-rank-clone-lrc-6828389e96a93f1d4219dfaf.
>
---

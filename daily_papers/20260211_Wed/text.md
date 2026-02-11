# 自然语言处理 cs.CL

- **最新发布 84 篇**

- **更新 56 篇**

## 最新发布

#### [new 001] MEVER: Multi-Modal and Explainable Claim Verification with Graph-based Evidence Retrieval
- **分类: cs.CL**

- **简介: 该论文属于事实验证任务，旨在解决多模态证据融合与可解释性问题。提出MEVER模型，实现证据检索、多模态验证和解释生成。**

- **链接: [https://arxiv.org/pdf/2602.10023v1](https://arxiv.org/pdf/2602.10023v1)**

> **作者:** Delvin Ce Zhang; Suhan Cui; Zhelin Chu; Xianren Zhang; Dongwon Lee
>
> **备注:** Accepted to EACL-26
>
> **摘要:** Verifying the truthfulness of claims usually requires joint multi-modal reasoning over both textual and visual evidence, such as analyzing both textual caption and chart image for claim verification. In addition, to make the reasoning process transparent, a textual explanation is necessary to justify the verification result. However, most claim verification works mainly focus on the reasoning over textual evidence only or ignore the explainability, resulting in inaccurate and unconvincing verification. To address this problem, we propose a novel model that jointly achieves evidence retrieval, multi-modal claim verification, and explanation generation. For evidence retrieval, we construct a two-layer multi-modal graph for claims and evidence, where we design image-to-text and text-to-image reasoning for multi-modal retrieval. For claim verification, we propose token- and evidence-level fusion to integrate claim and evidence embeddings for multi-modal verification. For explanation generation, we introduce multi-modal Fusion-in-Decoder for explainability. Finally, since almost all the datasets are in general domain, we create a scientific dataset, AIChartClaim, in AI domain to complement claim verification community. Experiments show the strength of our model.
>
---
#### [new 002] ViSpeechFormer: A Phonemic Approach for Vietnamese Automatic Speech Recognition
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，针对越南语ASR问题，提出基于音素的ViSpeechFormer模型，解决词汇外词泛化和训练偏差问题。**

- **链接: [https://arxiv.org/pdf/2602.10003v1](https://arxiv.org/pdf/2602.10003v1)**

> **作者:** Khoa Anh Nguyen; Long Minh Hoang; Nghia Hieu Nguyen; Luan Thanh Nguyen; Ngan Luu-Thuy Nguyen
>
> **摘要:** Vietnamese has a phonetic orthography, where each grapheme corresponds to at most one phoneme and vice versa. Exploiting this high grapheme-phoneme transparency, we propose ViSpeechFormer (\textbf{Vi}etnamese \textbf{Speech} Trans\textbf{Former}), a phoneme-based approach for Vietnamese Automatic Speech Recognition (ASR). To the best of our knowledge, this is the first Vietnamese ASR framework that explicitly models phonemic representations. Experiments on two publicly available Vietnamese ASR datasets show that ViSpeechFormer achieves strong performance, generalizes better to out-of-vocabulary words, and is less affected by training bias. This phoneme-based paradigm is also promising for other languages with phonetic orthographies. The code will be released upon acceptance of this paper.
>
---
#### [new 003] SCORE: Specificity, Context Utilization, Robustness, and Relevance for Reference-Free LLM Evaluation
- **分类: cs.CL**

- **简介: 该论文属于参考无监督的LLM评估任务，旨在解决现有评估框架无法准确衡量模型输出是否满足领域敏感决策需求的问题。工作包括提出多维评估框架和构建专业数据集。**

- **链接: [https://arxiv.org/pdf/2602.10017v1](https://arxiv.org/pdf/2602.10017v1)**

> **作者:** Homaira Huda Shomee; Rochana Chaturvedi; Yangxinyu Xie; Tanwi Mallick
>
> **摘要:** Large language models (LLMs) are increasingly used to support question answering and decision-making in high-stakes, domain-specific settings such as natural hazard response and infrastructure planning, where effective answers must convey fine-grained, decision-critical details. However, existing evaluation frameworks for retrieval-augmented generation (RAG) and open-ended question answering primarily rely on surface-level similarity, factual consistency, or semantic relevance, and often fail to assess whether responses provide the specific information required for domain-sensitive decisions. To address this gap, we propose a multi-dimensional, reference-free evaluation framework that assesses LLM outputs along four complementary dimensions: specificity, robustness to paraphrasing and semantic perturbations, answer relevance, and context utilization. We introduce a curated dataset of 1,412 domain-specific question-answer pairs spanning 40 professional roles and seven natural hazard types to support systematic evaluation. We further conduct human evaluation to assess inter-annotator agreement and alignment between model outputs and human judgments, which highlights the inherent subjectivity of open-ended, domain-specific evaluation. Our results show that no single metric sufficiently captures answer quality in isolation and demonstrate the need for structured, multi-metric evaluation frameworks when deploying LLMs in high-stakes applications.
>
---
#### [new 004] Conceptual Cultural Index: A Metric for Cultural Specificity via Relative Generality
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文化分析任务，旨在解决句子层面文化特异性评估问题。提出概念文化指数（CCI），通过比较目标文化与其他文化的普遍性来衡量文化特异性。**

- **链接: [https://arxiv.org/pdf/2602.09444v1](https://arxiv.org/pdf/2602.09444v1)**

> **作者:** Takumi Ohashi; Hitoshi Iyatomi
>
> **备注:** 9 pages, 2 figures, 8 tables. Accepted at the First Workshop on Multilingual Multicultural Evaluation (MME) @ EACL 2026
>
> **摘要:** Large language models (LLMs) are increasingly deployed in multicultural settings; however, systematic evaluation of cultural specificity at the sentence level remains underexplored. We propose the Conceptual Cultural Index (CCI), which estimates cultural specificity at the sentence level. CCI is defined as the difference between the generality estimate within the target culture and the average generality estimate across other cultures. This formulation enables users to operationally control the scope of culture via comparison settings and provides interpretability, since the score derives from the underlying generality estimates. We validate CCI on 400 sentences (200 culture-specific and 200 general), and the resulting score distribution exhibits the anticipated pattern: higher for culture-specific sentences and lower for general ones. For binary separability, CCI outperforms direct LLM scoring, yielding more than a 10-point improvement in AUC for models specialized to the target culture. Our code is available at https://github.com/IyatomiLab/CCI .
>
---
#### [new 005] Improving Interpretability of Lexical Semantic Change with Neurobiological Features
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的语义演变研究任务，旨在提升词汇语义变化的可解释性。通过将词向量映射到神经生物学特征空间，实现对语义变化的系统分析与解释。**

- **链接: [https://arxiv.org/pdf/2602.09760v1](https://arxiv.org/pdf/2602.09760v1)**

> **作者:** Kohei Oda; Hiroya Takamura; Kiyoaki Shirai; Natthawut Kertkeidkachorn
>
> **备注:** PACLIC 2025
>
> **摘要:** Lexical Semantic Change (LSC) is the phenomenon in which the meaning of a word change over time. Most studies on LSC focus on improving the performance of estimating the degree of LSC, however, it is often difficult to interpret how the meaning of a word change. Enhancing the interpretability of LSC is a significant challenge as it could lead to novel insights in this field. To tackle this challenge, we propose a method to map the semantic space of contextualized embeddings of words obtained by a pre-trained language model to a neurobiological feature space. In the neurobiological feature space, each dimension corresponds to a primitive feature of words, and its value represents the intensity of that feature. This enables humans to interpret LSC systematically. When employed for the estimation of the degree of LSC, our method demonstrates superior performance in comparison to the majority of the previous methods. In addition, given the high interpretability of the proposed method, several analyses on LSC are carried out. The results demonstrate that our method not only discovers interesting types of LSC that have been overlooked in previous studies but also effectively searches for words with specific types of LSC.
>
---
#### [new 006] Anagent For Enhancing Scientific Table & Figure Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Anagent框架，解决科学表格和图表分析难题。通过多智能体协作，提升复杂多模态数据的解析与推理能力。**

- **链接: [https://arxiv.org/pdf/2602.10081v1](https://arxiv.org/pdf/2602.10081v1)**

> **作者:** Xuehang Guo; Zhiyong Lu; Tom Hope; Qingyun Wang
>
> **摘要:** In scientific research, analysis requires accurately interpreting complex multimodal knowledge, integrating evidence from different sources, and drawing inferences grounded in domain-specific knowledge. However, current artificial intelligence (AI) systems struggle to consistently demonstrate such capabilities. The complexity and variability of scientific tables and figures, combined with heterogeneous structures and long-context requirements, pose fundamental obstacles to scientific table \& figure analysis. To quantify these challenges, we introduce AnaBench, a large-scale benchmark featuring $63,178$ instances from nine scientific domains, systematically categorized along seven complexity dimensions. To tackle these challenges, we propose Anagent, a multi-agent framework for enhanced scientific table \& figure analysis through four specialized agents: Planner decomposes tasks into actionable subtasks, Expert retrieves task-specific information through targeted tool execution, Solver synthesizes information to generate coherent analysis, and Critic performs iterative refinement through five-dimensional quality assessment. We further develop modular training strategies that leverage supervised finetuning and specialized reinforcement learning to optimize individual capabilities while maintaining effective collaboration. Comprehensive evaluation across 170 subdomains demonstrates that Anagent achieves substantial improvements, up to $\uparrow 13.43\%$ in training-free settings and $\uparrow 42.12\%$ with finetuning, while revealing that task-oriented reasoning and context-aware problem-solving are essential for high-quality scientific table \& figure analysis. Our project page: https://xhguo7.github.io/Anagent/.
>
---
#### [new 007] The CLEF-2026 CheckThat! Lab: Advancing Multilingual Fact-Checking
- **分类: cs.CL**

- **简介: 该论文属于多语言事实核查任务，旨在解决在线信息虚假与操控问题。工作包括源检索、数值时间事实核查及生成完整核查文章，推动验证流程发展。**

- **链接: [https://arxiv.org/pdf/2602.09516v1](https://arxiv.org/pdf/2602.09516v1)**

> **作者:** Julia Maria Struß; Sebastian Schellhammer; Stefan Dietze; Venktesh V; Vinay Setty; Tanmoy Chakraborty; Preslav Nakov; Avishek Anand; Primakov Chungkham; Salim Hafid; Dhruv Sahnan; Konstantin Todorov
>
> **备注:** misinformation, disinformation, fact-checking, claim source retrieval, generating fact-checking articles
>
> **摘要:** The CheckThat! lab aims to advance the development of innovative technologies combating disinformation and manipulation efforts in online communication across a multitude of languages and platforms. While in early editions the focus has been on core tasks of the verification pipeline (check-worthiness, evidence retrieval, and verification), in the past three editions, the lab added additional tasks linked to the verification process. In this year's edition, the verification pipeline is at the center again with the following tasks: Task 1 on source retrieval for scientific web claims (a follow-up of the 2025 edition), Task 2 on fact-checking numerical and temporal claims, which adds a reasoning component to the 2025 edition, and Task 3, which expands the verification pipeline with generation of full-fact-checking articles. These tasks represent challenging classification and retrieval problems as well as generation challenges at the document and span level, including multilingual settings.
>
---
#### [new 008] BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation
- **分类: cs.CL; cs.AI; cs.SE**

- **简介: 该论文属于模型评估任务，旨在解决LLM-as-a-Judge中的偏见问题。提出BiasScope框架，自动发现潜在偏见，提升评估可靠性。**

- **链接: [https://arxiv.org/pdf/2602.09383v1](https://arxiv.org/pdf/2602.09383v1)**

> **作者:** Peng Lai; Zhihao Ou; Yong Wang; Longyue Wang; Jian Yang; Yun Chen; Guanhua Chen
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** LLM-as-a-Judge has been widely adopted across various research and practical applications, yet the robustness and reliability of its evaluation remain a critical issue. A core challenge it faces is bias, which has primarily been studied in terms of known biases and their impact on evaluation outcomes, while automated and systematic exploration of potential unknown biases is still lacking. Nevertheless, such exploration is crucial for enhancing the robustness and reliability of evaluations. To bridge this gap, we propose BiasScope, a LLM-driven framework for automatically and at scale discovering potential biases that may arise during model evaluation. BiasScope can uncover potential biases across different model families and scales, with its generality and effectiveness validated on the JudgeBench dataset. It overcomes the limitations of existing approaches, transforming bias discovery from a passive process relying on manual effort and predefined bias lists into an active and comprehensive automated exploration. Moreover, based on BiasScope, we propose JudgeBench-Pro, an extended version of JudgeBench and a more challenging benchmark for evaluating the robustness of LLM-as-a-judge. Strikingly, even powerful LLMs as evaluators show error rates above 50\% on JudgeBench-Pro, underscoring the urgent need to strengthen evaluation robustness and to mitigate potential biases further.
>
---
#### [new 009] Listen to the Layers: Mitigating Hallucinations with Inter-Layer Disagreement
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型生成文本中的幻觉问题。通过分析模型内部层间的不一致性，提出CoCoA算法以提升生成内容的准确性。**

- **链接: [https://arxiv.org/pdf/2602.09486v1](https://arxiv.org/pdf/2602.09486v1)**

> **作者:** Koduvayur Subbalakshmi; Sabbir Hossain Ujjal; Venkata Krishna Teja Mangichetty; Nastaran Jamalipour Soofi
>
> **备注:** Preprint, 23 pages, 13 tables, 12 figures
>
> **摘要:** Pretrained Large Language Models (LLMs) are prone to generating fluent yet factually incorrect text-a phenomenon known as hallucinations, undermining their reliability and utility in downstream tasks. We hypothesize that a generated text span's factuality is correlated with its representational instability across the model's internal layers. Based on this, we propose the CoCoA (Confusion and Consistency Aware) decoder, a novel, training-free decoding algorithm that mitigates hallucinations at inference time by listening to these signals in the middle layers. We propose two metrics to quantify this instability in the middle layers, and use it to penalize outputs that exhibit high internal confusion, thereby steering the model towards more internally consistent and factually grounded outputs. We further propose a self-information gated variant, CoCoA-SIG, that dynamically modulates this penalty to selectively target high-surprise, unstable generations. Extensive experiments on diverse tasks, including question-answering, summarization and code generation demonstrate that CoCoA significantly improves factual correctness across multiple model families (e.g., Llama-3, Qwen-2.5, Mistral). By leveraging model-intrinsic signals, CoCoA offers an effective and broadly applicable method for enhancing the trustworthiness of LLMs at inference time, without requiring any model retraining.
>
---
#### [new 010] Evaluating Social Bias in RAG Systems: When External Context Helps and Reasoning Hurts
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的公平性研究任务，旨在解决RAG系统中的社会偏见问题。通过实验发现外部上下文有助于减少偏见，但推理过程可能增加偏见，需构建更公平的推理框架。**

- **链接: [https://arxiv.org/pdf/2602.09442v1](https://arxiv.org/pdf/2602.09442v1)**

> **作者:** Shweta Parihar; Lu Cheng
>
> **备注:** Accepted as a full paper with an oral presentation at the 30th Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD 2026)
>
> **摘要:** Social biases inherent in large language models (LLMs) raise significant fairness concerns. Retrieval-Augmented Generation (RAG) architectures, which retrieve external knowledge sources to enhance the generative capabilities of LLMs, remain susceptible to the same bias-related challenges. This work focuses on evaluating and understanding the social bias implications of RAG. Through extensive experiments across various retrieval corpora, LLMs, and bias evaluation datasets, encompassing more than 13 different bias types, we surprisingly observe a reduction in bias in RAG. This suggests that the inclusion of external context can help counteract stereotype-driven predictions, potentially improving fairness by diversifying the contextual grounding of the model's outputs. To better understand this phenomenon, we then explore the model's reasoning process by integrating Chain-of-Thought (CoT) prompting into RAG while assessing the faithfulness of the model's CoT. Our experiments reveal that the model's bias inclinations shift between stereotype and anti-stereotype responses as more contextual information is incorporated from the retrieved documents. Interestingly, we find that while CoT enhances accuracy, contrary to the bias reduction observed with RAG, it increases overall bias across datasets, highlighting the need for bias-aware reasoning frameworks that can mitigate this trade-off.
>
---
#### [new 011] TraceMem: Weaving Narrative Memory Schemata from User Conversational Traces
- **分类: cs.CL**

- **简介: 该论文提出TraceMem，用于构建用户对话中的连贯叙事记忆，解决LLM长期交互中对话历史管理的问题。通过三阶段流程生成结构化记忆卡片，提升多跳和时间推理能力。**

- **链接: [https://arxiv.org/pdf/2602.09712v1](https://arxiv.org/pdf/2602.09712v1)**

> **作者:** Yiming Shu; Pei Liu; Tiange Zhang; Ruiyang Gao; Jun Ma; Chen Sun
>
> **摘要:** Sustaining long-term interactions remains a bottleneck for Large Language Models (LLMs), as their limited context windows struggle to manage dialogue histories that extend over time. Existing memory systems often treat interactions as disjointed snippets, failing to capture the underlying narrative coherence of the dialogue stream. We propose TraceMem, a cognitively-inspired framework that weaves structured, narrative memory schemata from user conversational traces through a three-stage pipeline: (1) Short-term Memory Processing, which employs a deductive topic segmentation approach to demarcate episode boundaries and extract semantic representation; (2) Synaptic Memory Consolidation, a process that summarizes episodes into episodic memories before distilling them alongside semantics into user-specific traces; and (3) Systems Memory Consolidation, which utilizes two-stage hierarchical clustering to organize these traces into coherent, time-evolving narrative threads under unifying themes. These threads are encapsulated into structured user memory cards, forming narrative memory schemata. For memory utilization, we provide an agentic search mechanism to enhance reasoning process. Evaluation on the LoCoMo benchmark shows that TraceMem achieves state-of-the-art performance with a brain-inspired architecture. Analysis shows that by constructing coherent narratives, it surpasses baselines in multi-hop and temporal reasoning, underscoring its essential role in deep narrative comprehension. Additionally, we provide an open discussion on memory systems, offering our perspectives and future outlook on the field. Our code implementation is available at: https://github.com/YimingShu-teay/TraceMem
>
---
#### [new 012] Aligning Tree-Search Policies with Fixed Token Budgets in Test-Time Scaling of LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决树搜索解码在固定token预算下的效率问题。提出BG-MCTS算法，根据剩余预算动态调整搜索策略，提升解码效果。**

- **链接: [https://arxiv.org/pdf/2602.09574v1](https://arxiv.org/pdf/2602.09574v1)**

> **作者:** Sora Miyamoto; Daisuke Oba; Naoaki Okazaki
>
> **摘要:** Tree-search decoding is an effective form of test-time scaling for large language models (LLMs), but real-world deployment imposes a fixed per-query token budget that varies across settings. Existing tree-search policies are largely budget-agnostic, treating the budget as a termination condition, which can lead to late-stage over-branching or premature termination. We propose {Budget-Guided MCTS} (BG-MCTS), a tree-search decoding algorithm that aligns its search policy with the remaining token budget: it starts with broad exploration, then prioritizes refinement and answer completion as the budget depletes while reducing late-stage branching from shallow nodes. BG-MCTS consistently outperforms budget-agnostic tree-search baselines across different budgets on MATH500 and AIME24/25 with open-weight LLMs.
>
---
#### [new 013] Don't Shoot The Breeze: Topic Continuity Model Using Nonlinear Naive Bayes With Attention
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于对话系统任务，旨在解决聊天机器人中话题连续性问题。通过改进的朴素贝叶斯模型和注意力机制，提升对长对话话题连贯性的判断能力。**

- **链接: [https://arxiv.org/pdf/2602.09312v1](https://arxiv.org/pdf/2602.09312v1)**

> **作者:** Shu-Ting Pi; Pradeep Bagavan; Yejia Li; Disha; Qun Liu
>
> **备注:** EMNLP 2024: Industry Track; 8 pages, 2 figures, 1 table
>
> **摘要:** Utilizing Large Language Models (LLM) as chatbots in diverse business scenarios often presents the challenge of maintaining topic continuity. Abrupt shifts in topics can lead to poor user experiences and inefficient utilization of computational resources. In this paper, we present a topic continuity model aimed at assessing whether a response aligns with the initial conversation topic. Our model is built upon the expansion of the corresponding natural language understanding (NLU) model into quantifiable terms using a Naive Bayes approach. Subsequently, we have introduced an attention mechanism and logarithmic nonlinearity to enhance its capability to capture topic continuity. This approach allows us to convert the NLU model into an interpretable analytical formula. In contrast to many NLU models constrained by token limits, our proposed model can seamlessly handle conversations of any length with linear time complexity. Furthermore, the attention mechanism significantly improves the model's ability to identify topic continuity in complex conversations. According to our experiments, our model consistently outperforms traditional methods, particularly in handling lengthy and intricate conversations. This unique capability offers us an opportunity to ensure the responsible and interpretable use of LLMs.
>
---
#### [new 014] AlignTune: Modular Toolkit for Post-Training Alignment of Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出AlignTune工具包，解决大语言模型后训练对齐中的流程分散、难以复现问题，通过统一接口和模块化设计实现可控制的对齐实验。**

- **链接: [https://arxiv.org/pdf/2602.09621v1](https://arxiv.org/pdf/2602.09621v1)**

> **作者:** R E Zera Marveen Lyngkhoi; Chirag Chawla; Pratinav Seth; Utsav Avaiya; Soham Bhattacharjee; Mykola Khandoga; Rui Yuan; Vinay Kumar Sankarapu
>
> **备注:** https://github.com/Lexsi-Labs/aligntune
>
> **摘要:** Post-training alignment is central to deploying large language models (LLMs), yet practical workflows remain split across backend-specific tools and ad-hoc glue code, making experiments hard to reproduce. We identify backend interference, reward fragmentation, and irreproducible pipelines as key obstacles in alignment research. We introduce AlignTune, a modular toolkit exposing a unified interface for supervised fine-tuning (SFT) and RLHF-style optimization with interchangeable TRL and Unsloth backends. AlignTune standardizes configuration, provides an extensible reward layer (rule-based and learned), and integrates evaluation over standard benchmarks and custom tasks. By isolating backend-specific logic behind a single factory boundary, AlignTune enables controlled comparisons and reproducible alignment experiments.
>
---
#### [new 015] Knowledge Integration Decay in Search-Augmented Reasoning of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于知识增强的推理任务，解决LLM在长链推理中知识整合衰退的问题。提出SAKE方法，在推理过程首尾锚定知识，提升知识整合效果。**

- **链接: [https://arxiv.org/pdf/2602.09517v1](https://arxiv.org/pdf/2602.09517v1)**

> **作者:** Sangwon Yu; Ik-hwan Kim; Donghun Kang; Bongkyu Hwang; Junhwa Choi; Suk-hoon Jung; Seungki Hong; Taehee Lee; Sungroh Yoon
>
> **摘要:** Modern Large Language Models (LLMs) have demonstrated remarkable capabilities in complex tasks by employing search-augmented reasoning to incorporate external knowledge into long chains of thought. However, we identify a critical yet underexplored bottleneck in this paradigm, termed Knowledge Integration Decay (KID). Specifically, we observe that as the length of reasoning generated before search grows, models increasingly fail to integrate retrieved evidence into subsequent reasoning steps, limiting performance even when relevant information is available. To address this, we propose Self-Anchored Knowledge Encoding (SAKE), a training-free inference-time strategy designed to stabilize knowledge utilization. By anchoring retrieved knowledge at both the beginning and end of the reasoning process, SAKE prevents it from being overshadowed by prior context, thereby preserving its semantic integrity. Extensive experiments on multi-hop QA and complex reasoning benchmarks demonstrate that SAKE significantly mitigates KID and improves performance, offering a lightweight yet effective solution for knowledge integration in agentic LLMs.
>
---
#### [new 016] AgentSkiller: Scaling Generalist Agent Intelligence through Semantically Integrated Cross-Domain Data Synthesis
- **分类: cs.CL**

- **简介: 该论文提出AgentSkiller，解决通用智能代理数据稀缺问题，通过跨领域数据合成构建可靠环境，提升模型功能调用能力。**

- **链接: [https://arxiv.org/pdf/2602.09372v1](https://arxiv.org/pdf/2602.09372v1)**

> **作者:** Zexu Sun; Bokai Ji; Hengyi Cai; Shuaiqiang Wang; Lei Wang; Guangxia Li; Xu Chen
>
> **备注:** 33 pages, 9 figures
>
> **摘要:** Large Language Model agents demonstrate potential in solving real-world problems via tools, yet generalist intelligence is bottlenecked by scarce high-quality, long-horizon data. Existing methods collect privacy-constrained API logs or generate scripted interactions lacking diversity, which struggle to produce data requisite for scaling capabilities. We propose AgentSkiller, a fully automated framework synthesizing multi-turn interaction data across realistic, semantically linked domains. It employs a DAG-based architecture with explicit state transitions to ensure determinism and recoverability. The pipeline builds a domain ontology and Person-Centric Entity Graph, defines tool interfaces via Service Blueprints for Model Context Protocol servers, and populates environments with consistent databases and strict Domain Policies. A cross-domain fusion mechanism links services to simulate complex tasks. Finally, the pipeline creates user tasks by verifying solution paths, filtering via execution-based validation, and generating queries using a Persona-based Simulator for automated rollout. This produces reliable environments with clear state changes. To demonstrate effectiveness, we synthesized $\approx$ 11K interaction samples; experimental results indicate that models trained on this dataset achieve significant improvements on function calling over baselines, particularly in larger parameter regimes.
>
---
#### [new 017] Where Are We At with Automatic Speech Recognition for the Bambara Language?
- **分类: cs.CL**

- **简介: 该论文属于自动语音识别任务，旨在评估Bambara语言的ASR性能。研究构建了首个标准化基准，测试37个模型，发现当前性能未达标准，揭示多语言预训练不足。**

- **链接: [https://arxiv.org/pdf/2602.09785v1](https://arxiv.org/pdf/2602.09785v1)**

> **作者:** Seydou Diallo; Yacouba Diarra; Mamadou K. Keita; Panga Azazia Kamaté; Adam Bouno Kampo; Aboubacar Ouattara
>
> **备注:** v1- 8 pages, 5 tables, 1 figure- AfricaNLP Workshop @ EACL 2026
>
> **摘要:** This paper introduces the first standardized benchmark for evaluating Automatic Speech Recognition (ASR) in the Bambara language, utilizing one hour of professionally recorded Malian constitutional text. Designed as a controlled reference set under near-optimal acoustic and linguistic conditions, the benchmark was used to evaluate 37 models, ranging from Bambara-trained systems to large-scale commercial models. Our findings reveal that current ASR performance remains significantly below deployment standards in a narrow formal domain; the top-performing system in terms of Word Error Rate (WER) achieved 46.76\% and the best Character Error Rate (CER) of 13.00\% was set by another model, while several prominent multilingual models exceeded 100\% WER. These results suggest that multilingual pre-training and model scaling alone are insufficient for underrepresented languages. Furthermore, because this dataset represents a best-case scenario of the most simplified and formal form of spoken Bambara, these figures are yet to be tested against practical, real-world settings. We provide the benchmark and an accompanying public leaderboard to facilitate transparent evaluation and future research in Bambara speech technology.
>
---
#### [new 018] LLM Reasoning Predicts When Models Are Right: Evidence from Coding Classroom Discourse
- **分类: cs.CL**

- **简介: 该论文属于教育对话分析任务，旨在解决LLM在自动标注时错误检测的问题。通过分析教师话语及模型推理，使用TF-IDF和分类器识别错误预测，发现正确推理具有因果性特征。**

- **链接: [https://arxiv.org/pdf/2602.09832v1](https://arxiv.org/pdf/2602.09832v1)**

> **作者:** Bakhtawar Ahtisham; Kirk Vanacore; Zhuqian Zhou; Jinsook Lee; Rene F. Kizilcec
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed to automatically label and analyze educational dialogue at scale, yet current pipelines lack reliable ways to detect when models are wrong. We investigate whether reasoning generated by LLMs can be used to predict the correctness of a model's own predictions. We analyze 30,300 teacher utterances from classroom dialogue, each labeled by multiple state-of-the-art LLMs with an instructional move construct and an accompanying reasoning. Using human-verified ground-truth labels, we frame the task as predicting whether a model's assigned label for a given utterance is correct. We encode LLM reasoning using Term Frequency-Inverse Document Frequency (TF-IDF) and evaluate five supervised classifiers. A Random Forest classifier achieves an F1 score of 0.83 (Recall = 0.854), successfully identifying most incorrect predictions and outperforming baselines. Training specialist detectors for specific instructional move constructs further improves performance on difficult constructs, indicating that error detection benefits from construct-specific linguistic cues. Using the Linguistic Inquiry and Word Count (LIWC) framework, we examine four linguistic markers of correctness: Causation, Differentiation, Tentativeness, and Insight. Correct predictions exhibit grounded causal language (e.g., because, therefore), while incorrect reasoning is substantially more likely to rely on epistemic hedging (e.g., might, could) and performative metacognition (e.g., think, realize). Syntactic complexity does not distinguish correct from incorrect reasoning, and longer reasoning is not more reliable. These findings demonstrate that reasoning-based error detection offers a practical and scalable approach to quality control in automated educational dialogue analysis.
>
---
#### [new 019] Where-to-Unmask: Ground-Truth-Guided Unmasking Order Learning for Masked Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文属于文本生成任务，解决MDLM中未掩码位置选择问题。提出Gt-Margin优化解码顺序，提升生成质量与推理准确率。**

- **链接: [https://arxiv.org/pdf/2602.09501v1](https://arxiv.org/pdf/2602.09501v1)**

> **作者:** Hikaru Asano; Tadashi Kozuno; Kuniaki Saito; Yukino Baba
>
> **备注:** 15 pages, 6 figures
>
> **摘要:** Masked Diffusion Language Models (MDLMs) generate text by iteratively filling masked tokens, requiring two coupled decisions at each step: which positions to unmask (where-to-unmask) and which tokens to place (what-to-unmask). While standard MDLM training directly optimizes token prediction (what-to-unmask), inference-time unmasking orders (where-to-unmask) are typically determined by heuristic confidence measures or trained through reinforcement learning with costly on-policy rollouts. To address this, we introduce Gt-Margin, a position-wise score derived from ground-truth tokens, defined as the probability margin between the correct token and its strongest alternative. Gt-Margin yields an oracle unmasking order that prioritizes easier positions first under each partially masked state. We demonstrate that leveraging this oracle unmasking order significantly enhances final generation quality, particularly on logical reasoning benchmarks. Building on this insight, we train a supervised unmasking planner via learning-to-rank to imitate the oracle ordering from masked contexts. The resulting planner integrates into standard MDLM sampling to select where-to-unmask, improving reasoning accuracy without modifying the token prediction model.
>
---
#### [new 020] NOWJ @BioCreative IX ToxHabits: An Ensemble Deep Learning Approach for Detecting Substance Use and Contextual Information in Clinical Texts
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对临床文本中物质使用信息的抽取任务，解决毒物使用及上下文识别问题。采用集成深度学习方法，结合BETO与CRF，提升检测精度。**

- **链接: [https://arxiv.org/pdf/2602.09469v1](https://arxiv.org/pdf/2602.09469v1)**

> **作者:** Huu-Huy-Hoang Tran; Gia-Bao Duong; Quoc-Viet-Anh Tran; Thi-Hai-Yen Vuong; Hoang-Quynh Le
>
> **摘要:** Extracting drug use information from unstructured Electronic Health Records remains a major challenge in clinical Natural Language Processing. While Large Language Models demonstrate advancements, their use in clinical NLP is limited by concerns over trust, control, and efficiency. To address this, we present NOWJ submission to the ToxHabits Shared Task at BioCreative IX. This task targets the detection of toxic substance use and contextual attributes in Spanish clinical texts, a domain-specific, low-resource setting. We propose a multi-output ensemble system tackling both Subtask 1 - ToxNER and Subtask 2 - ToxUse. Our system integrates BETO with a CRF layer for sequence labeling, employs diverse training strategies, and uses sentence filtering to boost precision. Our top run achieved 0.94 F1 and 0.97 precision for Trigger Detection, and 0.91 F1 for Argument Detection.
>
---
#### [new 021] Are Language Models Sensitive to Morally Irrelevant Distractors?
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于AI伦理任务，旨在探究语言模型是否受无关道德因素影响。研究通过引入非道德干扰因素，发现其显著改变模型的道德判断，提示需更细致的道德评估。**

- **链接: [https://arxiv.org/pdf/2602.09416v1](https://arxiv.org/pdf/2602.09416v1)**

> **作者:** Andrew Shaw; Christina Hahn; Catherine Rasgaitis; Yash Mishra; Alisa Liu; Natasha Jaques; Yulia Tsvetkov; Amy X. Zhang
>
> **摘要:** With the rapid development and uptake of large language models (LLMs) across high-stakes settings, it is increasingly important to ensure that LLMs behave in ways that align with human values. Existing moral benchmarks prompt LLMs with value statements, moral scenarios, or psychological questionnaires, with the implicit underlying assumption that LLMs report somewhat stable moral preferences. However, moral psychology research has shown that human moral judgements are sensitive to morally irrelevant situational factors, such as smelling cinnamon rolls or the level of ambient noise, thereby challenging moral theories that assume the stability of human moral judgements. Here, we draw inspiration from this "situationist" view of moral psychology to evaluate whether LLMs exhibit similar cognitive moral biases to humans. We curate a novel multimodal dataset of 60 "moral distractors" from existing psychological datasets of emotionally-valenced images and narratives which have no moral relevance to the situation presented. After injecting these distractors into existing moral benchmarks to measure their effects on LLM responses, we find that moral distractors can shift the moral judgements of LLMs by over 30% even in low-ambiguity scenarios, highlighting the need for more contextual moral evaluations and more nuanced cognitive moral modeling of LLMs.
>
---
#### [new 022] Unsupervised Cross-Lingual Part-of-Speech Tagging with Monolingual Corpora Only
- **分类: cs.CL**

- **简介: 该论文属于跨语言词性标注任务，解决低资源语言缺乏标注数据的问题。通过单语语料和无监督机器翻译，构建伪平行语料并进行词性标注，提升标注效果。**

- **链接: [https://arxiv.org/pdf/2602.09366v1](https://arxiv.org/pdf/2602.09366v1)**

> **作者:** Jianyu Zheng
>
> **备注:** 16 pages, 6 figures, 7 tables, under review
>
> **摘要:** Due to the scarcity of part-of-speech annotated data, existing studies on low-resource languages typically adopt unsupervised approaches for POS tagging. Among these, POS tag projection with word alignment method transfers POS tags from a high-resource source language to a low-resource target language based on parallel corpora, making it particularly suitable for low-resource language settings. However, this approach relies heavily on parallel corpora, which are often unavailable for many low-resource languages. To overcome this limitation, we propose a fully unsupervised cross-lingual part-of-speech(POS) tagging framework that relies solely on monolingual corpora by leveraging unsupervised neural machine translation(UNMT) system. This UNMT system first translates sentences from a high-resource language into a low-resource one, thereby constructing pseudo-parallel sentence pairs. Then, we train a POS tagger for the target language following the standard projection procedure based on word alignments. Moreover, we propose a multi-source projection technique to calibrate the projected POS tags on the target side, enhancing to train a more effective POS tagger. We evaluate our framework on 28 language pairs, covering four source languages (English, German, Spanish and French) and seven target languages (Afrikaans, Basque, Finnis, Indonesian, Lithuanian, Portuguese and Turkish). Experimental results show that our method can achieve performance comparable to the baseline cross-lingual POS tagger with parallel sentence pairs, and even exceeds it for certain target languages. Furthermore, our proposed multi-source projection technique further boosts performance, yielding an average improvement of 1.3% over previous methods.
>
---
#### [new 023] Digital Linguistic Bias in Spanish: Evidence from Lexical Variation in LLMs
- **分类: cs.CL**

- **简介: 该论文研究LLMs对西班牙语地理词汇变异的捕捉能力，属于方言知识评估任务。旨在解决模型在不同地区西班牙语变体上的表现差异问题，通过大规模测试验证其识别能力。**

- **链接: [https://arxiv.org/pdf/2602.09346v1](https://arxiv.org/pdf/2602.09346v1)**

> **作者:** Yoshifumi Kawasaki
>
> **摘要:** This study examines the extent to which Large Language Models (LLMs) capture geographic lexical variation in Spanish, a language that exhibits substantial regional variation. Treating LLMs as virtual informants, we probe their dialectal knowledge using two survey-style question formats: Yes-No questions and multiple-choice questions. To this end, we exploited a large-scale, expert-curated database of Spanish lexical variation. Our evaluation covers more than 900 lexical items across 21 Spanish-speaking countries and is conducted at both the country and dialectal area levels. Across both evaluation formats, the results reveal systematic differences in how LLMs represent Spanish language varieties. Lexical variation associated with Spain, Equatorial Guinea, Mexico & Central America, and the La Plata River is recognized more accurately by the models, while the Chilean variety proves particularly difficult for the models to distinguish. Importantly, differences in the volume of country-level digital resources do not account for these performance patterns, suggesting that factors beyond data quantity shape dialectal representation in LLMs. By providing a fine-grained, large-scale evaluation of geographic lexical variation, this work advances empirical understanding of dialectal knowledge in LLMs and contributes new evidence to discussions of Digital Linguistic Bias in Spanish.
>
---
#### [new 024] LEMUR: A Corpus for Robust Fine-Tuning of Multilingual Law Embedding Models for Retrieval
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于法律信息检索任务，旨在解决多语言法律语料不足和检索效果差的问题。构建了LEMUR语料库，并通过微调提升多语言法律嵌入模型的检索性能。**

- **链接: [https://arxiv.org/pdf/2602.09570v1](https://arxiv.org/pdf/2602.09570v1)**

> **作者:** Narges Baba Ahmadi; Jan Strich; Martin Semmann; Chris Biemann
>
> **备注:** Accepted at EACL SRW 26
>
> **摘要:** Large language models (LLMs) are increasingly used to access legal information. Yet, their deployment in multilingual legal settings is constrained by unreliable retrieval and the lack of domain-adapted, open-embedding models. In particular, existing multilingual legal corpora are not designed for semantic retrieval, and PDF-based legislative sources introduce substantial noise due to imperfect text extraction. To address these challenges, we introduce LEMUR, a large-scale multilingual corpus of EU environmental legislation constructed from 24,953 official EUR-Lex PDF documents covering 25 languages. We quantify the fidelity of PDF-to-text conversion by measuring lexical consistency against authoritative HTML versions using the Lexical Content Score (LCS). Building on LEMUR, we fine-tune three state-of-the-art multilingual embedding models using contrastive objectives in both monolingual and bilingual settings, reflecting realistic legal-retrieval scenarios. Experiments across low- and high-resource languages demonstrate that legal-domain fine-tuning consistently improves Top-k retrieval accuracy relative to strong baselines, with particularly pronounced gains for low-resource languages. Cross-lingual evaluations show that these improvements transfer to unseen languages, indicating that fine-tuning primarily enhances language-independent, content-level legal representations rather than language-specific cues. We publish code\footnote{\href{https://github.com/nargesbh/eur_lex}{GitHub Repository}} and data\footnote{\href{https://huggingface.co/datasets/G4KMU/LEMUR}{Hugging Face Dataset}}.
>
---
#### [new 025] ViMultiChoice: Toward a Method That Gives Explanation for Multiple-Choice Reading Comprehension in Vietnamese
- **分类: cs.CL**

- **简介: 该论文属于多选阅读理解任务，旨在解决模型无法解释选择原因的问题。作者提出ViMultiChoice方法，同时预测答案并生成解释，提升准确率。**

- **链接: [https://arxiv.org/pdf/2602.09961v1](https://arxiv.org/pdf/2602.09961v1)**

> **作者:** Trung Tien Cao; Lam Minh Thai; Nghia Hieu Nguyen; Duc-Vu Nguyen; Ngan Luu-Thuy Nguyen
>
> **摘要:** Multiple-choice Reading Comprehension (MCRC) models aim to select the correct answer from a set of candidate options for a given question. However, they typically lack the ability to explain the reasoning behind their choices. In this paper, we introduce a novel Vietnamese dataset designed to train and evaluate MCRC models with explanation generation capabilities. Furthermore, we propose ViMultiChoice, a new method specifically designed for modeling Vietnamese reading comprehension that jointly predicts the correct answer and generates a corresponding explanation. Experimental results demonstrate that ViMultiChoice outperforms existing MCRC baselines, achieving state-of-the-art (SotA) performance on both the ViMMRC 2.0 benchmark and the newly introduced dataset. Additionally, we show that jointly training option decision and explanation generation leads to significant improvements in multiple-choice accuracy.
>
---
#### [new 026] Measuring Inclusion in Interaction: Inclusion Analytics for Human-AI Collaborative Learning
- **分类: cs.CL**

- **简介: 该论文属于人机协作学习领域，旨在解决如何动态评估包容性问题。提出包含分析框架，通过互动层面的指标揭示参与度、情感氛围和认知公平性。**

- **链接: [https://arxiv.org/pdf/2602.09269v1](https://arxiv.org/pdf/2602.09269v1)**

> **作者:** Jaeyoon Choi; Nia Nixon
>
> **摘要:** Inclusion, equity, and access are widely valued in AI and education, yet are often assessed through coarse sample descriptors or post-hoc self-reports that miss how inclusion is shaped moment by moment in collaborative problem solving (CPS). In this proof-of-concept paper, we introduce inclusion analytics, a discourse-based framework for examining inclusion as a dynamic, interactional process in CPS. We conceptualize inclusion along three complementary dimensions -- participation equity, affective climate, and epistemic equity -- and demonstrate how these constructs can be made analytically visible using scalable, interaction-level measures. Using both simulated conversations and empirical data from human-AI teaming experiments, we illustrate how inclusion analytics can surface patterns of participation, relational dynamics, and idea uptake that remain invisible to aggregate or post-hoc evaluations. This work represents an initial step toward process-oriented approaches to measuring inclusion in human-AI collaborative learning environments.
>
---
#### [new 027] Breaking the Pre-Sampling Barrier: Activation-Informed Difficulty-Aware Self-Consistency
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决自一致性方法推理成本高的问题。通过利用神经网络激活信息动态调整样本数量，降低计算开销。**

- **链接: [https://arxiv.org/pdf/2602.09438v1](https://arxiv.org/pdf/2602.09438v1)**

> **作者:** Taewoong Yoon; Geunyeong Jeong; Geon Park; Sihyeong Yeom; Harksoo Kim
>
> **摘要:** Self-Consistency (SC) is an effective decoding strategy that improves the reasoning performance of Large Language Models (LLMs) by generating multiple chain-of-thought reasoning paths and selecting the final answer via majority voting. However, it suffers from substantial inference costs because it requires a large number of samples. To mitigate this issue, Difficulty-Adaptive Self-Consistency (DSC) was proposed to reduce unnecessary token usage for easy problems by adjusting the number of samples according to problem difficulty. However, DSC requires additional model calls and pre-sampling to estimate difficulty, and this process is repeated when applying to each dataset, leading to significant computational overhead. In this work, we propose Activation-Informed Difficulty-Aware Self-Consistency (ACTSC) to address these limitations. ACTSC leverages internal difficulty signals reflected in the feed-forward network neuron activations to construct a lightweight difficulty estimation probe, without any additional token generation or model calls. The probe dynamically adjusts the number of samples for SC and can be applied to new datasets without requiring pre-sampling for difficulty estimation. To validate its effectiveness, we conduct experiments on five benchmarks. Experimental results show that ACTSC effectively reduces inference costs while maintaining accuracy relative to existing methods.
>
---
#### [new 028] How Do People Quantify Naturally: Evidence from Mandarin Picture Description
- **分类: cs.CL**

- **简介: 该论文属于语言产生研究，探讨汉语中量化表达的自然使用。通过图片描述任务，分析数量、生物性和模态对量化行为的影响，揭示量化策略的选择机制。**

- **链接: [https://arxiv.org/pdf/2602.09838v1](https://arxiv.org/pdf/2602.09838v1)**

> **作者:** Yayun Zhang; Guanyi Chen; Fahime Same; Saad Mahamood; Tingting He
>
> **摘要:** Quantification is a fundamental component of everyday language use, yet little is known about how speakers decide whether and how to quantify in naturalistic production. We investigate quantification in Mandarin Chinese using a picture-based elicited description task in which speakers freely described scenes containing multiple objects, without explicit instructions to count or quantify. Across both spoken and written modalities, we examine three aspects of quantification: whether speakers choose to quantify at all, how precise their quantification is, and which quantificational strategies they adopt. Results show that object numerosity, animacy, and production modality systematically shape quantificational behaviour. In particular, increasing numerosity reduces both the likelihood and the precision of quantification, while animate referents and modality selectively modulate strategy choice. This study demonstrates how quantification can be examined under unconstrained production conditions and provides a naturalistic dataset for further analyses of quantity expression in language production.
>
---
#### [new 029] Targum -- A Multilingual New Testament Translation Corpus
- **分类: cs.CL**

- **简介: 该论文属于翻译研究任务，旨在解决现有语料缺乏深度的问题。构建了包含657个新约译本的多语言语料库，提供详细元数据支持多层次分析。**

- **链接: [https://arxiv.org/pdf/2602.09724v1](https://arxiv.org/pdf/2602.09724v1)**

> **作者:** Maciej Rapacz; Aleksander Smywiński-Pohl
>
> **摘要:** Many European languages possess rich biblical translation histories, yet existing corpora - in prioritizing linguistic breadth - often fail to capture this depth. To address this gap, we introduce a multilingual corpus of 657 New Testament translations, of which 352 are unique, with unprecedented depth in five languages: English (208 unique versions from 396 total), French (41 from 78), Italian (18 from 33), Polish (30 from 48), and Spanish (55 from 102). Aggregated from 12 online biblical libraries and one preexisting corpus, each translation is manually annotated with metadata that maps the text to a standardized identifier for the work, its specific edition, and its year of revision. This canonicalization empowers researchers to define "uniqueness" for their own needs: they can perform micro-level analyses on translation families, such as the KJV lineage, or conduct macro-level studies by deduplicating closely related texts. By providing the first resource designed for such flexible, multilevel analysis, our corpus establishes a new benchmark for the quantitative study of translation history.
>
---
#### [new 030] ATTNPO: Attention-Guided Process Supervision for Efficient Reasoning
- **分类: cs.CL**

- **简介: 该论文属于机器学习任务，旨在解决大模型推理过程中的冗余问题。通过引入注意力机制，实现精准的步骤奖励分配，减少冗余推理，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2602.09953v1](https://arxiv.org/pdf/2602.09953v1)**

> **作者:** Shuaiyi Nie; Siyu Ding; Wenyuan Zhang; Linhao Yu; Tianmeng Yang; Yao Chen; Tingwen Liu; Weichong Yin; Yu Sun; Hua Wu
>
> **备注:** Work in process
>
> **摘要:** Large reasoning models trained with reinforcement learning and verifiable rewards (RLVR) achieve strong performance on complex reasoning tasks, yet often overthink, generating redundant reasoning without performance gains. Existing trajectory-level length penalties often fail to effectively shorten reasoning length and degrade accuracy, as they uniformly treat all reasoning steps and lack fine-grained signals to distinguish redundancy from necessity. Meanwhile, process-supervised methods are typically resource-intensive and suffer from inaccurate credit assignment. To address these issues, we propose ATTNPO, a low-overhead process-supervised RL framework that leverages the model's intrinsic attention signals for step-level credit assignment. We first identify a set of special attention heads that naturally focus on essential steps while suppressing redundant ones. By leveraging the attention scores of these heads, We then employ two sub-strategies to mitigate overthinking by discouraging redundant steps while preserving accuracy by reducing penalties on essential steps. Experimental results show that ATTNPO substantially reduces reasoning length while significantly improving performance across 9 benchmarks.
>
---
#### [new 031] On the Optimal Reasoning Length for RL-Trained Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究强化学习训练语言模型的最优推理长度。旨在解决推理长度与效率、性能之间的平衡问题，通过实验分析不同长度控制方法的效果。**

- **链接: [https://arxiv.org/pdf/2602.09591v1](https://arxiv.org/pdf/2602.09591v1)**

> **作者:** Daisuke Nohara; Taishi Nakamura; Rio Yokota
>
> **备注:** 15 pages, 10 figures. Submitted to the Workshop on Scaling Post-training for LLMs (SPOT) at ICLR 2026
>
> **摘要:** Reinforcement learning substantially improves reasoning in large language models, but it also tends to lengthen chain of thought outputs and increase computational cost during both training and inference. Though length control methods have been proposed, it remains unclear what the optimal output length is for balancing efficiency and performance. In this work, we compare several length control methods on two models, Qwen3-1.7B Base and DeepSeek-R1-Distill-Qwen-1.5B. Our results indicate that length penalties may hinder reasoning acquisition, while properly tuned length control can improve efficiency for models with strong prior reasoning. By extending prior work to RL trained policies, we identify two failure modes, 1) long outputs increase dispersion, and 2) short outputs lead to under-thinking.
>
---
#### [new 032] Unsupervised Layer-Wise Dynamic Test Time Adaptation for LLMs
- **分类: cs.CL**

- **简介: 该论文属于测试时自适应任务，解决无监督下模型对每个提示的不稳定适应问题。提出分层动态自适应方法，通过调整学习率提升稳定性与性能。**

- **链接: [https://arxiv.org/pdf/2602.09719v1](https://arxiv.org/pdf/2602.09719v1)**

> **作者:** Longhuan Xu; Cunjian Chen; Feng Yin
>
> **摘要:** Test-time adaptation (TTA) for large language models (LLMs) updates model parameters at inference time using signals available at deployment. This paper focuses on a common yet under-explored regime: unsupervised, sample-specific TTA, where the model adapts independently for each prompt using only the prompt itself, without gold answers or external supervision. Although appealing, naive unsupervised TTA with a fixed, handcrafted learning rate can be unstable: updates may overfit to prompt-specific statistics, drift from the desired answer distribution, and ultimately degrade generation quality. This failure mode is not surprising, as in this case TTA must adapt to a single prompt within only a few gradient steps, unlike standard training that averages updates over large datasets and long optimization horizons. Therefore, we propose layer-wise dynamic test-time adaptation, a framework which explicitly modulates TTA strength as a function of prompt representation, LLM structure and adaptation step. In our setting, TTA updates only LoRA parameters, and a lightweight hypernetwork predicts per-layer, per-step learning-rate multipliers, enabling fine-grained control. Experiments across various datasets and LLMs consistently show that our method substantially strengthens TTA by learning effective scaling patterns over adaptation steps and transformer layer projections, improving stability while delivering better performance.
>
---
#### [new 033] AnalyticsGPT: An LLM Workflow for Scientometric Question Answering
- **分类: cs.CL; cs.DL**

- **简介: 该论文提出AnalyticsGPT，解决科学计量问答任务，通过LLM实现数据检索与分析，提升对“科学之科学”问题的处理效率。**

- **链接: [https://arxiv.org/pdf/2602.09817v1](https://arxiv.org/pdf/2602.09817v1)**

> **作者:** Khang Ly; Georgios Cheirmpos; Adrian Raudaschl; Christopher James; Seyed Amin Tabatabaei
>
> **摘要:** This paper introduces AnalyticsGPT, an intuitive and efficient large language model (LLM)-powered workflow for scientometric question answering. This underrepresented downstream task addresses the subcategory of meta-scientific questions concerning the "science of science." When compared to traditional scientific question answering based on papers, the task poses unique challenges in the planning phase. Namely, the need for named-entity recognition of academic entities within questions and multi-faceted data retrieval involving scientometric indices, e.g. impact factors. Beyond their exceptional capacity for treating traditional natural language processing tasks, LLMs have shown great potential in more complex applications, such as task decomposition and planning and reasoning. In this paper, we explore the application of LLMs to scientometric question answering, and describe an end-to-end system implementing a sequential workflow with retrieval-augmented generation and agentic concepts. We also address the secondary task of effectively synthesizing the data into presentable and well-structured high-level analyses. As a database for retrieval-augmented generation, we leverage a proprietary research performance assessment platform. For evaluation, we consult experienced subject matter experts and leverage LLMs-as-judges. In doing so, we provide valuable insights on the efficacy of LLMs towards a niche downstream task. Our (skeleton) code and prompts are available at: https://github.com/lyvykhang/llm-agents-scientometric-qa/tree/acl.
>
---
#### [new 034] From FusHa to Folk: Exploring Cross-Lingual Transfer in Arabic Language Models
- **分类: cs.CL**

- **简介: 研究阿拉伯语模型在不同方言间的跨语言迁移效果，探讨其可行性及差异原因。任务为跨语言迁移学习，解决方言间性能不均问题，通过实验与分析揭示地理因素影响及负干扰现象。**

- **链接: [https://arxiv.org/pdf/2602.09826v1](https://arxiv.org/pdf/2602.09826v1)**

> **作者:** Abdulmuizz Khalak; Abderrahmane Issam; Gerasimos Spanakis
>
> **备注:** Accepted to VarDial 2026
>
> **摘要:** Arabic Language Models (LMs) are pretrained predominately on Modern Standard Arabic (MSA) and are expected to transfer to its dialects. While MSA as the standard written variety is commonly used in formal settings, people speak and write online in various dialects that are spread across the Arab region. This poses limitations for Arabic LMs, since its dialects vary in their similarity to MSA. In this work we study cross-lingual transfer of Arabic models using probing on 3 Natural Language Processing (NLP) Tasks, and representational similarity. Our results indicate that transfer is possible but disproportionate across dialects, which we find to be partially explained by their geographic proximity. Furthermore, we find evidence for negative interference in models trained to support all Arabic dialects. This questions their degree of similarity, and raises concerns for cross-lingual transfer in Arabic models.
>
---
#### [new 035] A Unified Assessment of the Poverty of the Stimulus Argument for Neural Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型研究任务，旨在验证语言习得中的PoSH假设。通过构建评估基准，测试神经模型在有限数据下的语法泛化能力，探讨是否需要先天语法约束。**

- **链接: [https://arxiv.org/pdf/2602.09992v1](https://arxiv.org/pdf/2602.09992v1)**

> **作者:** Xiulin Yang; Arianna Bisazza; Nathan Schneider; Ethan Gotlieb Wilcox
>
> **摘要:** How can children acquire native-level syntax from limited input? According to the Poverty of the Stimulus Hypothesis (PoSH), the linguistic input children receive is insufficient to explain certain generalizations that are robustly learned; innate linguistic constraints, many have argued, are thus necessary to explain language learning. Neural language models, which lack such language-specific constraints in their design, offer a computational test of this longstanding (but controversial) claim. We introduce \poshbench, a training-and-evaluation suite targeting question formation, islands to movement, and other English phenomena at the center of the PoSH arguments. Training Transformer models on 10--50M words of developmentally plausible text, we find indications of generalization on all phenomena even without direct positive evidence -- yet neural models remain less data-efficient and their generalizations are weaker than those of children. We further enhance our models with three recently proposed cognitively motivated inductive biases. We find these biases improve general syntactic competence but not \poshbench performance. Our findings challenge the claim that innate syntax is the only possible route to generalization, while suggesting that human-like data efficiency requires inductive biases beyond those tested here.
>
---
#### [new 036] SinFoS: A Parallel Dataset for Translating Sinhala Figures of Speech
- **分类: cs.CL**

- **简介: 该论文提出SinFoS数据集，用于解决低资源语言（如僧伽罗语）中修辞表达的机器翻译问题。任务为跨语言修辞翻译，通过构建数据集并评估大模型表现，推动文化敏感的自然语言处理研究。**

- **链接: [https://arxiv.org/pdf/2602.09866v1](https://arxiv.org/pdf/2602.09866v1)**

> **作者:** Johan Sofalas; Dilushri Pavithra; Nevidu Jayatilleke; Ruvan Weerasinghe
>
> **备注:** 19 pages, 6 figures, 8 tables, Accepted paper at the 22nd Workshop on Multiword Expressions (MWE 2026) @ EACL 2026
>
> **摘要:** Figures of Speech (FoS) consist of multi-word phrases that are deeply intertwined with culture. While Neural Machine Translation (NMT) performs relatively well with the figurative expressions of high-resource languages, it often faces challenges when dealing with low-resource languages like Sinhala due to limited available data. To address this limitation, we introduce a corpus of 2,344 Sinhala figures of speech with cultural and cross-lingual annotations. We examine this dataset to classify the cultural origins of the figures of speech and to identify their cross-lingual equivalents. Additionally, we have developed a binary classifier to differentiate between two types of FOS in the dataset, achieving an accuracy rate of approximately 92%. We also evaluate the performance of existing LLMs on this dataset. Our findings reveal significant shortcomings in the current capabilities of LLMs, as these models often struggle to accurately convey idiomatic meanings. By making this dataset publicly available, we offer a crucial benchmark for future research in low-resource NLP and culturally aware machine translation.
>
---
#### [new 037] Steer2Edit: From Activation Steering to Component-Level Editing
- **分类: cs.CL**

- **简介: 该论文提出Steer2Edit，解决大模型行为控制问题，通过将激活引导转化为组件级权重编辑，提升安全性和效率。**

- **链接: [https://arxiv.org/pdf/2602.09870v1](https://arxiv.org/pdf/2602.09870v1)**

> **作者:** Chung-En Sun; Ge Yan; Zimo Wang; Tsui-Wei Weng
>
> **摘要:** Steering methods influence Large Language Model behavior by identifying semantic directions in hidden representations, but are typically realized through inference-time activation interventions that apply a fixed, global modification to the model's internal states. While effective, such interventions often induce unfavorable attribute-utility trade-offs under strong control, as they ignore the fact that many behaviors are governed by a small and heterogeneous subset of model components. We propose Steer2Edit, a theoretically grounded, training-free framework that transforms steering vectors from inference-time control signals into diagnostic signals for component-level rank-1 weight editing. Instead of uniformly injecting a steering direction during generation, Steer2Edit selectively redistributes behavioral influence across individual attention heads and MLP neurons, yielding interpretable edits that preserve the standard forward pass and remain compatible with optimized parallel inference. Across safety alignment, hallucination mitigation, and reasoning efficiency, Steer2Edit consistently achieves more favorable attribute-utility trade-offs: at matched downstream performance, it improves safety by up to 17.2%, increases truthfulness by 9.8%, and reduces reasoning length by 12.2% on average. Overall, Steer2Edit provides a principled bridge between representation steering and weight editing by translating steering signals into interpretable, training-free parameter updates.
>
---
#### [new 038] AmharicIR+Instr: A Two-Dataset Resource for Neural Retrieval and Instruction Tuning
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出AmharicIR+Instr数据集，用于解决低资源语言Amharic的神经检索和指令跟随生成问题，包含两个经过验证的高质量数据集。**

- **链接: [https://arxiv.org/pdf/2602.09914v1](https://arxiv.org/pdf/2602.09914v1)**

> **作者:** Tilahun Yeshambel; Moncef Garouani; Josiane Mothe
>
> **备注:** 7 pages, Submitted to resource track
>
> **摘要:** Neural retrieval and GPT-style generative models rely on large, high-quality supervised data, which is still scarce for low-resource languages such as Amharic. We release an Amharic data resource consisting of two datasets that supports research on (i) neural retrieval-ranking and (ii) instruction-following text generation. The retrieval-ranking dataset contains 1,091 manually verified query-positive-negative document triplets drawn from diverse Amharic sources and constructed to support contrastive training and benchmarking of neural retrievers (e.g., DPR, ColBERT-style late interaction and SPLADE-style sparse neural retrieval). Triplets are created through a combination of expert-curated queries, web-derived queries, and LLM-assisted generation, with positive/negative documents selected from the web or synthesized by LLMs and then validated by native speakers. The instruction prompt-response dataset comprises 6,285 Amharic prompt-response pairs spanning multiple domains and instruction types, generated with several LLMs and refined through manual review and correction for grammaticality, relevance, fluency, and factual plausibility. We release both datasets with standardized splits and formats (CSV,JSON,JSONL) to enable reproducible work on Amharic retrieval, ranking, and generative modelling. These datasets also come with a methodology that can be generalized to other low-resource languages.
>
---
#### [new 039] Beyond Uniform Credit: Causal Credit Assignment for Policy Optimization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型推理任务，解决生成文本中各token贡献度评估不均的问题。通过因果重要性加权方法，提升策略梯度优化效果。**

- **链接: [https://arxiv.org/pdf/2602.09331v1](https://arxiv.org/pdf/2602.09331v1)**

> **作者:** Mykola Khandoga; Rui Yuan; Vinay Kumar Sankarapu
>
> **备注:** 12 pages, 1 figure
>
> **摘要:** Policy gradient methods for language model reasoning, such as GRPO and DAPO, assign uniform credit to all generated tokens - the filler phrase "Let me think" receives the same gradient update as the critical calculation "23 + 45 = 68." We propose counterfactual importance weighting: mask reasoning spans, measure the drop in answer probability, and upweight tokens accordingly during policy gradient updates. Our method requires no auxiliary models or external annotation, instead importance is estimated directly from the policy model's own probability shifts. Experiments on GSM8K across three models spanning the Qwen and Llama families demonstrate consistent improvements over uniform baselines and faster convergence to equivalent accuracy. Inverting the importance signal hurts performance, confirming we capture genuine causal structure rather than noise. Analysis shows the method correctly prioritizes calculation steps over scaffolding text. We view these findings as establishing counterfactual importance weighting as a foundation for further research rather than a complete solution.
>
---
#### [new 040] Maastricht University at AMIYA: Adapting LLMs for Dialectal Arabic using Fine-tuning and MBR Decoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于方言阿拉伯语生成任务，旨在提升大语言模型在方言上的表现。通过微调、适配器融合和MBR解码，增强方言忠实度与语义准确性。**

- **链接: [https://arxiv.org/pdf/2602.09703v1](https://arxiv.org/pdf/2602.09703v1)**

> **作者:** Abdulhai Alali; Abderrahmane Issam
>
> **摘要:** Large Language Models (LLMs) are becoming increasingly multilingual, supporting hundreds of languages, especially high resource ones. Unfortunately, Dialect variations are still underrepresented due to limited data and linguistic variation. In this work, we adapt a pre-trained LLM to improve dialectal performance. Specifically, we use Low Rank Adaptation (LoRA) fine-tuning on monolingual and English Dialect parallel data, adapter merging and dialect-aware MBR decoding to improve dialectal fidelity generation and translation. Experiments on Syrian, Moroccan, and Saudi Arabic show that merging and MBR improve dialectal fidelity while preserving semantic accuracy. This combination provides a compact and effective framework for robust dialectal Arabic generation.
>
---
#### [new 041] FM SO.P: A Progressive Task Mixture Framework with Automatic Evaluation for Cross-Domain SOP Understanding
- **分类: cs.CL**

- **简介: 该论文提出FM SO.P框架，解决跨领域SOP理解问题，通过渐进任务混合和自动评估系统提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.09336v1](https://arxiv.org/pdf/2602.09336v1)**

> **作者:** Siyuan Huang; Ziyu Wang; Chao Pan; Han Zhao
>
> **摘要:** Standard Operating Procedures (SOPs) are critical for enterprise operations, yet existing language models struggle with SOP understanding and cross-domain generalization. Current methods fail because joint training cannot differentiate between reasoning capabilities that SOP requires: terminology precision, sequential ordering, and constraint reasoning. We propose FM SO.P, solving these challenges through two novelties. First, we introduce progressive task mixtures that build capabilities by stages across three task types with cumulative data: concept disambiguation for terminology precision, action sequence understanding for procedural correctness, and scenario-aware graph reasoning for conditional logic. Second, we propose an automatic multi-agent evaluation system consisting of three agents that adaptively generate rubrics, stratified test sets, and rubric scoring, adapting to domains (e.g., temporal constraints for DMV, regulatory compliance for banking). Evaluated on SOPBench across seven domains (Bank, DMV, Healthcare, Market, University, Library, Hotel), FM SO.P achieves 48.3\% pass rate with our 32B model and 34.3\% with our opensource 7B model, matching Qwen-2.5-72B-Instruct baseline (34.4\%) with 10x fewer parameters.
>
---
#### [new 042] MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MATA框架，解决TableQA任务中的可靠性与效率问题。通过多智能体协作和小模型工具，提升准确率并减少大模型调用。**

- **链接: [https://arxiv.org/pdf/2602.09642v1](https://arxiv.org/pdf/2602.09642v1)**

> **作者:** Sieun Hyeon; Jusang Oh; Sunghwan Steve Cho; Jaeyoung Do
>
> **摘要:** Recent advances in Large Language Models (LLMs) have significantly improved table understanding tasks such as Table Question Answering (TableQA), yet challenges remain in ensuring reliability, scalability, and efficiency, especially in resource-constrained or privacy-sensitive environments. In this paper, we introduce MATA, a multi-agent TableQA framework that leverages multiple complementary reasoning paths and a set of tools built with small language models. MATA generates candidate answers through diverse reasoning styles for a given table and question, then refines or selects the optimal answer with the help of these tools. Furthermore, it incorporates an algorithm designed to minimize expensive LLM agent calls, enhancing overall efficiency. MATA maintains strong performance with small, open-source models and adapts easily across various LLM types. Extensive experiments on two benchmarks of varying difficulty with ten different LLMs demonstrate that MATA achieves state-of-the-art accuracy and highly efficient reasoning while avoiding excessive LLM inference. Our results highlight that careful orchestration of multiple reasoning pathways yields scalable and reliable TableQA. The code is available at https://github.com/AIDAS-Lab/MATA.
>
---
#### [new 043] Decoupled Reasoning with Implicit Fact Tokens (DRIFT): A Dual-Model Framework for Efficient Long-Context Inference
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决长文本推理中知识与推理耦合的问题。提出DRIFT框架，分离知识提取与推理过程，提升长上下文推理效率。**

- **链接: [https://arxiv.org/pdf/2602.10021v1](https://arxiv.org/pdf/2602.10021v1)**

> **作者:** Wenxuan Xie; Yujia Wang; Xin Tan; Chaochao Lu; Xia Hu; Xuhong Wang
>
> **摘要:** The integration of extensive, dynamic knowledge into Large Language Models (LLMs) remains a significant challenge due to the inherent entanglement of factual data and reasoning patterns. Existing solutions, ranging from non-parametric Retrieval-Augmented Generation (RAG) to parametric knowledge editing, are often constrained in practice by finite context windows, retriever noise, or the risk of catastrophic forgetting. In this paper, we propose DRIFT, a novel dual-model architecture designed to explicitly decouple knowledge extraction from the reasoning process. Unlike static prompt compression, DRIFT employs a lightweight knowledge model to dynamically compress document chunks into implicit fact tokens conditioned on the query. These dense representations are projected into the reasoning model's embedding space, replacing raw, redundant text while maintaining inference accuracy. Extensive experiments show that DRIFT significantly improves performance on long-context tasks, outperforming strong baselines among comparably sized models. Our approach provides a scalable and efficient paradigm for extending the effective context window and reasoning capabilities of LLMs. Our code is available at https://github.com/Lancelot-Xie/DRIFT.
>
---
#### [new 044] Life Cycle-Aware Evaluation of Knowledge Distillation for Machine Translation: Environmental Impact and Translation Quality Trade-offs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于机器翻译任务，研究知识蒸馏的环境影响与翻译质量之间的权衡。通过考虑计算成本和碳足迹，评估不同蒸馏方法的性能，提供在约束条件下的选择依据。**

- **链接: [https://arxiv.org/pdf/2602.09691v1](https://arxiv.org/pdf/2602.09691v1)**

> **作者:** Joseph Attieh; Timothee Mickus; Anne-Laure Ligozat; Aurélie Névéol; Jörg Tiedemann
>
> **摘要:** Knowledge distillation (KD) is a tool to compress a larger system (teacher) into a smaller one (student). In machine translation, studies typically report only the translation quality of the student and omit the computational complexity of performing KD, making it difficult to select among the many available KD choices under compute-induced constraints. In this study, we evaluate representative KD methods by considering both translation quality and computational cost. We express computational cost as a carbon footprint using the machine learning life cycle assessment (MLCA) tool. This assessment accounts for runtime operational emissions and amortized hardware production costs throughout the KD model life cycle (teacher training, distillation, and inference). We find that (i) distillation overhead dominates the total footprint at small deployment volumes, (ii) inference dominates at scale, making KD beneficial only beyond a task-dependent usage threshold, and (iii) word-level distillation typically offers more favorable footprint-quality trade-offs than sequence-level distillation. Our protocol provides reproducible guidance for selecting KD methods under explicit quality and compute-induced constraints.
>
---
#### [new 045] Comprehensive Comparison of RAG Methods Across Multi-Domain Conversational QA
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于对话问答任务，旨在解决多轮RAG方法比较不足的问题。通过实验分析不同RAG方法在多个数据集上的表现，发现简单方法更有效，强调检索策略与数据结构的匹配。**

- **链接: [https://arxiv.org/pdf/2602.09552v1](https://arxiv.org/pdf/2602.09552v1)**

> **作者:** Klejda Alushi; Jan Strich; Chris Biemann; Martin Semmann
>
> **备注:** Accepted to EACL SRW 26
>
> **摘要:** Conversational question answering increasingly relies on retrieval-augmented generation (RAG) to ground large language models (LLMs) in external knowledge. Yet, most existing studies evaluate RAG methods in isolation and primarily focus on single-turn settings. This paper addresses the lack of a systematic comparison of RAG methods for multi-turn conversational QA, where dialogue history, coreference, and shifting user intent substantially complicate retrieval. We present a comprehensive empirical study of vanilla and advanced RAG methods across eight diverse conversational QA datasets spanning multiple domains. Using a unified experimental setup, we evaluate retrieval quality and answer generation using generator and retrieval metrics, and analyze how performance evolves across conversation turns. Our results show that robust yet straightforward methods, such as reranking, hybrid BM25, and HyDE, consistently outperform vanilla RAG. In contrast, several advanced techniques fail to yield gains and can even degrade performance below the No-RAG baseline. We further demonstrate that dataset characteristics and dialogue length strongly influence retrieval effectiveness, explaining why no single RAG strategy dominates across settings. Overall, our findings indicate that effective conversational RAG depends less on method complexity than on alignment between the retrieval strategy and the dataset structure. We publish the code used.\footnote{\href{https://github.com/Klejda-A/exp-rag.git}{GitHub Repository}}
>
---
#### [new 046] EcoGym: Evaluating LLMs for Long-Horizon Plan-and-Execute in Interactive Economies
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出EcoGym，一个用于评估大语言模型在互动经济中长期规划与执行能力的基准。任务是解决现有评估框架不足的问题，通过三个环境测试模型的长期战略和执行效率。**

- **链接: [https://arxiv.org/pdf/2602.09514v1](https://arxiv.org/pdf/2602.09514v1)**

> **作者:** Xavier Hu; Jinxiang Xia; Shengze Xu; Kangqi Song; Yishuo Yuan; Guibin Zhang; Jincheng Ren; Boyu Feng; Li Lu; Tieyong Zeng; Jiaheng Liu; Minghao Liu; Yuchen Elenor Jiang; Wei Wang; He Zhu; Wangchunshu Zhou
>
> **备注:** work in progress
>
> **摘要:** Long-horizon planning is widely recognized as a core capability of autonomous LLM-based agents; however, current evaluation frameworks suffer from being largely episodic, domain-specific, or insufficiently grounded in persistent economic dynamics. We introduce EcoGym, a generalizable benchmark for continuous plan-and-execute decision making in interactive economies. EcoGym comprises three diverse environments: Vending, Freelance, and Operation, implemented in a unified decision-making process with standardized interfaces, and budgeted actions over an effectively unbounded horizon (1000+ steps if 365 day-loops for evaluation). The evaluation of EcoGym is based on business-relevant outcomes (e.g., net worth, income, and DAU), targeting long-term strategic coherence and robustness under partial observability and stochasticity. Experiments across eleven leading LLMs expose a systematic tension: no single model dominates across all three scenarios. Critically, we find that models exhibit significant suboptimality in either high-level strategies or efficient actions executions. EcoGym is released as an open, extensible testbed for transparent long-horizon agent evaluation and for studying controllability-utility trade-offs in realistic economic settings.
>
---
#### [new 047] MILE-RefHumEval: A Reference-Free, Multi-Independent LLM Framework for Human-Aligned Evaluation
- **分类: cs.CL**

- **简介: 该论文提出MILE-RefHumEval，用于无需参考文本的LLM评估任务，解决人工标注依赖和评估一致性问题，通过独立提示评估器实现高效、准确的人机对齐评估。**

- **链接: [https://arxiv.org/pdf/2602.09624v1](https://arxiv.org/pdf/2602.09624v1)**

> **作者:** Nalin Srun; Parisa Rastin; Guénaël Cabanes; Lydia Boudjeloud Assala
>
> **摘要:** We introduce MILE-RefHumEval, a reference-free framework for evaluating Large Language Models (LLMs) without ground-truth annotations or evaluator coordination. It leverages an ensemble of independently prompted evaluators guided by a human-aligned schema, supporting both discrete and continuous scoring judgement. With task-specific prompts from best candidate selection, summarization and image captioning to dialogue, MILE-RefHumEval provides flexible, interpretable, and scalable assessments. Experiments show it aligns closely with human judgments, outperforms prior methods, and reduces computational overhead, offering an efficient, robust, and human-aligned solution for real-world LLM evaluation.
>
---
#### [new 048] Understanding Risk and Dependency in AI Chatbot Use from User Discourse
- **分类: cs.CL**

- **简介: 该论文属于AI安全研究任务，旨在理解用户在使用AI聊天机器人时的心理风险与依赖。通过分析Reddit用户讨论，识别出五种心理风险维度，揭示用户实际体验中的安全感知与情绪模式。**

- **链接: [https://arxiv.org/pdf/2602.09339v1](https://arxiv.org/pdf/2602.09339v1)**

> **作者:** Jianfeng Zhu; Karin G. Coifman; Ruoming Jin
>
> **备注:** 21 pages, 5 figures
>
> **摘要:** Generative AI systems are increasingly embedded in everyday life, yet empirical understanding of how psychological risk associated with AI use emerges, is experienced, and is regulated by users remains limited. We present a large-scale computational thematic analysis of posts collected between 2023 and 2025 from two Reddit communities, r/AIDangers and r/ChatbotAddiction, explicitly focused on AI-related harm and distress. Using a multi-agent, LLM-assisted thematic analysis grounded in Braun and Clarke's reflexive framework, we identify 14 recurring thematic categories and synthesize them into five higher-order experiential dimensions. To further characterize affective patterns, we apply emotion labeling using a BERT-based classifier and visualize emotional profiles across dimensions. Our findings reveal five empirically derived experiential dimensions of AI-related psychological risk grounded in real-world user discourse, with self-regulation difficulties emerging as the most prevalent and fear concentrated in concerns related to autonomy, control, and technical risk. These results provide early empirical evidence from lived user experience of how AI safety is perceived and emotionally experienced outside laboratory or speculative contexts, offering a foundation for future AI safety research, evaluation, and responsible governance.
>
---
#### [new 049] Context-Aware Counterfactual Data Augmentation for Gender Bias Mitigation in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的性别偏见缓解任务，旨在解决细调语言模型时因去偏导致的语言建模能力下降问题。提出Context-CDA方法，通过增强上下文提升合成数据质量，有效减少偏见并保持模型性能。**

- **链接: [https://arxiv.org/pdf/2602.09590v1](https://arxiv.org/pdf/2602.09590v1)**

> **作者:** Shweta Parihar; Liu Guangliang; Natalie Parde; Lu Cheng
>
> **摘要:** A challenge in mitigating social bias in fine-tuned language models (LMs) is the potential reduction in language modeling capability, which can harm downstream performance. Counterfactual data augmentation (CDA), a widely used method for fine-tuning, highlights this issue by generating synthetic data that may align poorly with real-world distributions or creating overly simplistic counterfactuals that ignore the social context of altered sensitive attributes (e.g., gender) in the pretraining corpus. To address these limitations, we propose a simple yet effective context-augmented CDA method, Context-CDA, which uses large LMs to enhance the diversity and contextual relevance of the debiasing corpus. By minimizing discrepancies between the debiasing corpus and pretraining data through augmented context, this approach ensures better alignment, enhancing language modeling capability. We then employ uncertainty-based filtering to exclude generated counterfactuals considered low-quality by the target smaller LMs (i.e., LMs to be debiased), further improving the fine-tuning corpus quality. Experimental results on gender bias benchmarks demonstrate that Context-CDA effectively mitigates bias without sacrificing language modeling performance while offering insights into social biases by analyzing distribution shifts in next-token generation probabilities.
>
---
#### [new 050] Overview of PAN 2026: Voight-Kampff Generative AI Detection, Text Watermarking, Multi-Author Writing Style Analysis, Generative Plagiarism Detection, and Reasoning Trajectory Detection
- **分类: cs.CL**

- **简介: 该论文属于文本分析与检测任务，旨在解决AI生成文本识别、水印评估、多作者分析、抄袭检测及推理轨迹追踪等问题。**

- **链接: [https://arxiv.org/pdf/2602.09147v1](https://arxiv.org/pdf/2602.09147v1)**

> **作者:** Janek Bevendorff; Maik Fröbe; André Greiner-Petter; Andreas Jakoby; Maximilian Mayerl; Preslav Nakov; Henry Plutz; Martin Potthast; Benno Stein; Minh Ngoc Ta; Yuxia Wang; Eva Zangerle
>
> **摘要:** The goal of the PAN workshop is to advance computational stylometry and text forensics via objective and reproducible evaluation. In 2026, we run the following five tasks: (1) Voight-Kampff Generative AI Detection, particularly in mixed and obfuscated authorship scenarios, (2) Text Watermarking, a new task that aims to find new and benchmark the robustness of existing text watermarking schemes, (3) Multi-author Writing Style Analysis, a continued task that aims to find positions of authorship change, (4) Generative Plagiarism Detection, a continued task that targets source retrieval and text alignment between generated text and source documents, and (5) Reasoning Trajectory Detection, a new task that deals with source detection and safety detection of LLM-generated or human-written reasoning trajectories. As in previous years, PAN invites software submissions as easy-to-reproduce Docker containers for most of the tasks. Since PAN 2012, more than 1,100 submissions have been made this way via the TIRA experimentation platform.
>
---
#### [new 051] Effective Reasoning Chains Reduce Intrinsic Dimensionality
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理领域，研究如何通过有效推理链降低任务内在维度，提升模型泛化能力。工作包括提出内在维度作为量化指标，验证不同推理策略的效果。**

- **链接: [https://arxiv.org/pdf/2602.09276v1](https://arxiv.org/pdf/2602.09276v1)**

> **作者:** Archiki Prasad; Mandar Joshi; Kenton Lee; Mohit Bansal; Peter Shaw
>
> **备注:** 20 pages, 3 figures
>
> **摘要:** Chain-of-thought (CoT) reasoning and its variants have substantially improved the performance of language models on complex reasoning tasks, yet the precise mechanisms by which different strategies facilitate generalization remain poorly understood. While current explanations often point to increased test-time computation or structural guidance, establishing a consistent, quantifiable link between these factors and generalization remains challenging. In this work, we identify intrinsic dimensionality as a quantitative measure for characterizing the effectiveness of reasoning chains. Intrinsic dimensionality quantifies the minimum number of model dimensions needed to reach a given accuracy threshold on a given task. By keeping the model architecture fixed and varying the task formulation through different reasoning strategies, we demonstrate that effective reasoning strategies consistently reduce the intrinsic dimensionality of the task. Validating this on GSM8K with Gemma-3 1B and 4B, we observe a strong inverse correlation between the intrinsic dimensionality of a reasoning strategy and its generalization performance on both in-distribution and out-of-distribution data. Our findings suggest that effective reasoning chains facilitate learning by better compressing the task using fewer parameters, offering a new quantitative metric for analyzing reasoning processes.
>
---
#### [new 052] AI-Assisted Scientific Assessment: A Case Study on Climate Change
- **分类: cs.CL**

- **简介: 该论文属于科学评估任务，旨在探讨AI在气候科学研究中的应用。通过案例研究，评估AI辅助科学评估的效果与局限性。**

- **链接: [https://arxiv.org/pdf/2602.09723v1](https://arxiv.org/pdf/2602.09723v1)**

> **作者:** Christian Buck; Levke Caesar; Michelle Chen Huebscher; Massimiliano Ciaramita; Erich M. Fischer; Zeke Hausfather; Özge Kart Tokmak; Reto Knutti; Markus Leippold; Joseph Ludescher; Katharine J. Mach; Sofia Palazzo Corner; Kasra Rafiezadeh Shahi; Johan Rockström; Joeri Rogelj; Boris Sakschewski
>
> **摘要:** The emerging paradigm of AI co-scientists focuses on tasks characterized by repeatable verification, where agents explore search spaces in 'guess and check' loops. This paradigm does not extend to problems where repeated evaluation is impossible and ground truth is established by the consensus synthesis of theory and existing evidence. We evaluate a Gemini-based AI environment designed to support collaborative scientific assessment, integrated into a standard scientific workflow. In collaboration with a diverse group of 13 scientists working in the field of climate science, we tested the system on a complex topic: the stability of the Atlantic Meridional Overturning Circulation (AMOC). Our results show that AI can accelerate the scientific workflow. The group produced a comprehensive synthesis of 79 papers through 104 revision cycles in just over 46 person-hours. AI contribution was significant: most AI-generated content was retained in the report. AI also helped maintain logical consistency and presentation quality. However, expert additions were crucial to ensure its acceptability: less than half of the report was produced by AI. Furthermore, substantial oversight was required to expand and elevate the content to rigorous scientific standards.
>
---
#### [new 053] Text summarization via global structure awareness
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本摘要任务，旨在解决长文档摘要中全局结构丢失的问题。提出GloSA-sum方法，通过拓扑数据分析保持语义和逻辑完整性，提升摘要效率与质量。**

- **链接: [https://arxiv.org/pdf/2602.09821v1](https://arxiv.org/pdf/2602.09821v1)**

> **作者:** Jiaquan Zhang; Chaoning Zhang; Shuxu Chen; Yibei Liu; Chenghao Li; Qigan Sun; Shuai Yuan; Fachrina Dewi Puspitasari; Dongshen Han; Guoqing Wang; Sung-Ho Bae; Yang Yang
>
> **备注:** 24pages
>
> **摘要:** Text summarization is a fundamental task in natural language processing (NLP), and the information explosion has made long-document processing increasingly demanding, making summarization essential. Existing research mainly focuses on model improvements and sentence-level pruning, but often overlooks global structure, leading to disrupted coherence and weakened downstream performance. Some studies employ large language models (LLMs), which achieve higher accuracy but incur substantial resource and time costs. To address these issues, we introduce GloSA-sum, the first summarization approach that achieves global structure awareness via topological data analysis (TDA). GloSA-sum summarizes text efficiently while preserving semantic cores and logical dependencies. Specifically, we construct a semantic-weighted graph from sentence embeddings, where persistent homology identifies core semantics and logical structures, preserved in a ``protection pool'' as the backbone for summarization. We design a topology-guided iterative strategy, where lightweight proxy metrics approximate sentence importance to avoid repeated high-cost computations, thus preserving structural integrity while improving efficiency. To further enhance long-text processing, we propose a hierarchical strategy that integrates segment-level and global summarization. Experiments on multiple datasets demonstrate that GloSA-sum reduces redundancy while preserving semantic and logical integrity, striking a balance between accuracy and efficiency, and further benefits LLM downstream tasks by shortening contexts while retaining essential reasoning chains.
>
---
#### [new 054] Decomposing Reasoning Efficiency in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理效率评估问题。通过分解token效率因素，分析模型在准确性和效率间的权衡，提出新的评估框架。**

- **链接: [https://arxiv.org/pdf/2602.09805v1](https://arxiv.org/pdf/2602.09805v1)**

> **作者:** Daniel Kaiser; Arnoldo Frigessi; Ali Ramezani-Kebrya; Benjamin Ricaud
>
> **备注:** Preprint (under review). 29 pages, 4 figures
>
> **摘要:** Large language models trained for reasoning trade off inference tokens against accuracy, yet standard evaluations report only final accuracy, obscuring where tokens are spent or wasted. We introduce a trace-optional framework that decomposes token efficiency into interpretable factors: completion under a fixed token budget (avoiding truncation), conditional correctness given completion, and verbosity (token usage). When benchmark metadata provides per-instance workload proxies, we further factor verbosity into two components: mean verbalization overhead (tokens per work unit) and a coupling coefficient capturing how overhead scales with task workload. When reasoning traces are available, we add deterministic trace-quality measures (grounding, repetition, prompt copying) to separate degenerate looping from verbose-but-engaged reasoning, avoiding human labeling and LLM judges. Evaluating 25 models on CogniLoad, we find that accuracy and token-efficiency rankings diverge (Spearman $ρ=0.63$), efficiency gaps are often driven by conditional correctness, and verbalization overhead varies by about 9 times (only weakly related to model scale). Our decomposition reveals distinct bottleneck profiles that suggest different efficiency interventions.
>
---
#### [new 055] Quantum-Audit: Evaluating the Reasoning Limits of LLMs on Quantum Computing
- **分类: cs.CL**

- **简介: 该论文属于评估任务，旨在检验大语言模型在量子计算领域的理解能力。通过构建包含2700个问题的基准测试，分析模型对量子概念的掌握情况及错误假设的识别能力。**

- **链接: [https://arxiv.org/pdf/2602.10092v1](https://arxiv.org/pdf/2602.10092v1)**

> **作者:** Mohamed Afane; Kayla Laufer; Wenqi Wei; Ying Mao; Junaid Farooq; Ying Wang; Juntao Chen
>
> **备注:** 18 pages
>
> **摘要:** Language models have become practical tools for quantum computing education and research, from summarizing technical papers to explaining theoretical concepts and answering questions about recent developments in the field. While existing benchmarks evaluate quantum code generation and circuit design, their understanding of quantum computing concepts has not been systematically measured. Quantum-Audit addresses this gap with 2,700 questions covering core quantum computing topics. We evaluate 26 models from leading organizations. Our benchmark comprises 1,000 expert-written questions, 1,000 questions extracted from research papers using LLMs and validated by experts, plus an additional 700 questions including 350 open-ended questions and 350 questions with false premises to test whether models can correct erroneous assumptions. Human participants scored between 23% and 86%, with experts averaging 74%. Top-performing models exceeded the expert average, with Claude Opus 4.5 reaching 84% accuracy, though top models showed an average 12-point accuracy drop on expert-written questions compared to LLM-generated ones. Performance declined further on advanced topics, dropping to 73% on security questions. Additionally, models frequently accepted and reinforced false premises embedded in questions instead of identifying them, with accuracy below 66% on these critical reasoning tasks.
>
---
#### [new 056] Contractual Deepfakes: Can Large Language Models Generate Contracts?
- **分类: cs.CL; cs.AI**

- **简介: 论文探讨LLMs生成合同的可行性，指出其仅能生成表面合理文本，无法真正理解法律语境，无法替代专业法律工作。任务为评估LLMs在法律合同生成中的适用性。**

- **链接: [https://arxiv.org/pdf/2602.09384v1](https://arxiv.org/pdf/2602.09384v1)**

> **作者:** Eliza Mik
>
> **备注:** Accepted for publication
>
> **摘要:** Notwithstanding their unprecedented ability to generate text, LLMs do not understand the meaning of words, have no sense of context and cannot reason. Their output constitutes an approximation of statistically dominant word patterns. And yet, the drafting of contracts is often presented as a typical legal task that could be facilitated by this technology. This paper seeks to put an end to such unreasonable ideas. Predicting words differs from using language in the circumstances of specific transactions and reconstituting common contractual phrases differs from reasoning about the law. LLMs seem to be able to generate generic and superficially plausible contractual documents. In the cold light of day, such documents may turn out to be useless assemblages of inconsistent provisions or contracts that are enforceable but unsuitable for a given transaction. This paper casts a shadow on the simplistic assumption that LLMs threaten the continued viability of the legal industry.
>
---
#### [new 057] UniARM: Towards a Unified Autoregressive Reward Model for Multi-Objective Test-Time Alignment
- **分类: cs.CL**

- **简介: 该论文属于多目标对齐任务，旨在解决生成内容与用户多偏好目标不一致的问题。提出UniARM框架，统一建模所有偏好维度，避免参数独立导致的特征纠缠。**

- **链接: [https://arxiv.org/pdf/2602.09538v1](https://arxiv.org/pdf/2602.09538v1)**

> **作者:** Hongyan Xie; Yikun Ban; Ruiyu Fang; Zixuan Huang; Deqing Wang; Jianxin Li; Yitong Yao; Chao Wang; Shuangyong Song
>
> **备注:** Under Review
>
> **摘要:** Multi-objective alignment aims to align LLM responses with multiple human preference objectives. Among existing methods, guiding the generation of frozen LLMs through autoregressive reward models (ARMs) to accomplish multi-objective test-time alignment is a low-cost solution. However, these methods typically rely on independent parameters for each preference objective, either by training ARMs independently across preference dimensions, which neglects interactions among preference features, or by training a single ARM with separate feature extraction modules for each preference, which can cause feature entanglement. Both strategies can result in misalignment between generated outputs and user preferences. To address this limitation, we propose Preference-Modulated \& Shared Low-Rank Adaptation (MoSLoRA) for ARM training, which first extracts shared features via a preference-agnostic module and then applies affine transformations to shared features via a preference modulation module conditioned on mixed preference vectors. This design mitigates feature entanglement and enables precise control over preference trade-offs during inference. Building on this, we introduce the Unified Autoregressive Reward Model (UniARM), a novel framework for multi-objective test-time alignment. UniARM jointly models all preference dimensions in a single parameter space, eliminating the need for independent parameters for each preference objective. es on larger-scale LLMs, enhancing its practical usability.
>
---
#### [new 058] AfriNLLB: Efficient Translation Models for African Languages
- **分类: cs.CL**

- **简介: 该论文提出AfriNLLB，一种高效翻译模型，解决非洲语言翻译效率低的问题。通过模型压缩和微调，实现快速部署与良好性能。**

- **链接: [https://arxiv.org/pdf/2602.09373v1](https://arxiv.org/pdf/2602.09373v1)**

> **作者:** Yasmin Moslem; Aman Kassahun Wassie; Amanuel Gizachew Abebe
>
> **备注:** Accepted at AfricaNLP 2026 (oral)
>
> **摘要:** In this work, we present AfriNLLB, a series of lightweight models for efficient translation from and into African languages. AfriNLLB supports 15 language pairs (30 translation directions), including Swahili, Hausa, Yoruba, Amharic, Somali, Zulu, Lingala, Afrikaans, Wolof, and Egyptian Arabic, as well as other African Union official languages such as Arabic (MSA), French, Portuguese, and Spanish. Our training data covers bidirectional translation between English and 13 languages, and between French and two languages (Lingala and Wolof). AfriNLLB models are based on NLLB-200 600M, which we compress using iterative layer pruning and quantization. We fine-tune the pruned models on parallel corpora we curated for African languages, employing knowledge distillation from a larger teacher model. Our work aims at enabling efficient deployment of translation models for African languages in resource-constrained settings. Our evaluation results demonstrate that AfriNLLB models achieve performance comparable to the baseline while being significantly faster. We release two versions of the AfriNLLB models, a Transformers version that allows further fine-tuning and a CTranslate2 version for efficient inference. Moreover, we release all the training data that we used for fine-tuning the baseline and pruned models to facilitate further research.
>
---
#### [new 059] Advancing Block Diffusion Language Models for Test-Time Scaling
- **分类: cs.CL**

- **简介: 该论文属于语言模型推理任务，解决测试时扩展下的解码效率与效果平衡问题。提出BACD和TCCF框架，提升BDLM的推理性能。**

- **链接: [https://arxiv.org/pdf/2602.09555v1](https://arxiv.org/pdf/2602.09555v1)**

> **作者:** Yi Lu; Deyang Kong; Jianing Wang; Linsen Guo; Xue Wang; Qi Guo; Tao Gui; Xuanjing Huang; Wei Ye; Shikun Zhang; Wei Wang
>
> **摘要:** Recent advances in block diffusion language models have demonstrated competitive performance and strong scalability on reasoning tasks. However, existing BDLMs have limited exploration under the test-time scaling setting and face more severe decoding challenges in long Chain-of-Thought reasoning, particularly in balancing the decoding speed and effectiveness. In this work, we propose a unified framework for test-time scaling in BDLMs that introduces adaptivity in both decoding and block-wise generation. At the decoding level, we propose Bounded Adaptive Confidence Decoding (BACD), a difficulty-aware sampling strategy that dynamically adjusts denoising based on model confidence, accelerating inference while controlling error accumulation. Beyond step-wise adaptivity, we introduce Think Coarse, Critic Fine (TCCF), a test-time scaling paradigm that allocates large block sizes to exploratory reasoning and smaller block sizes to refinement, achieving an effective efficiency-effectiveness balance. To enable efficient and effective decoding with a large block size, we adopt Progressive Block Size Extension, which mitigates performance degradation when scaling block sizes. Extensive experiments show that applying BACD and TCCF to TDAR-8B yields significant improvements over strong baselines such as TraDo-8B (2.26x speedup, +11.2 points on AIME24). These results mark an important step toward unlocking the potential of BDLMs for test-time scaling in complex reasoning tasks.
>
---
#### [new 060] LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究如何通过LLM内部表示预测任务成功率，以提高推理效率。任务为模型效率优化，解决如何识别需额外计算的输入问题。工作包括训练线性探测器预测成功，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2602.09924v1](https://arxiv.org/pdf/2602.09924v1)**

> **作者:** William Lugoloobi; Thomas Foster; William Bankes; Chris Russell
>
> **摘要:** Running LLMs with extended reasoning on every problem is expensive, but determining which inputs actually require additional compute remains challenging. We investigate whether their own likelihood of success is recoverable from their internal representations before generation, and if this signal can guide more efficient inference. We train linear probes on pre-generation activations to predict policy-specific success on math and coding tasks, substantially outperforming surface features such as question length and TF-IDF. Using E2H-AMC, which provides both human and model performance on identical problems, we show that models encode a model-specific notion of difficulty that is distinct from human difficulty, and that this distinction increases with extended reasoning. Leveraging these probes, we demonstrate that routing queries across a pool of models can exceed the best-performing model whilst reducing inference cost by up to 70\% on MATH, showing that internal representations enable practical efficiency gains even when they diverge from human intuitions about difficulty. Our code is available at: https://github.com/KabakaWilliam/llms_know_difficulty
>
---
#### [new 061] Learning from the Irrecoverable: Error-Localized Policy Optimization for Tool-Integrated LLM Reasoning
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，解决工具集成推理中的稀疏奖励和步骤信用分配问题。提出ELPO方法，通过定位不可逆错误步骤并优化策略，提升推理性能。**

- **链接: [https://arxiv.org/pdf/2602.09598v1](https://arxiv.org/pdf/2602.09598v1)**

> **作者:** Qiao Liang; Yuke Zhu; Chao Ge; Lei Yang; Ying Shen; Bo Zheng; Sheng Guo
>
> **备注:** 20 pages, 11 figures
>
> **摘要:** Tool-integrated reasoning (TIR) enables LLM agents to solve tasks through planning, tool use, and iterative revision, but outcome-only reinforcement learning in this setting suffers from sparse, delayed rewards and weak step-level credit assignment. In long-horizon TIR trajectories, an early irrecoverable mistake can determine success or failure, making it crucial to localize the first irrecoverable step and leverage it for fine-grained credit assignment. We propose Error-Localized Policy Optimization (ELPO), which localizes the first irrecoverable step via binary-search rollout trees under a fixed rollout budget, converts the resulting tree into stable learning signals through hierarchical advantage attribution, and applies error-localized adaptive clipping to strengthen corrective updates on the critical step and its suffix. Across TIR benchmarks in math, science QA, and code execution, ELPO consistently outperforms strong Agentic RL baselines under comparable sampling budgets, with additional gains in Pass@K and Major@K scaling, rollout ranking quality, and tool-call efficiency. Our code will be publicly released soon.
>
---
#### [new 062] Effective vocabulary expanding of multilingual language models for extremely low-resource languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决多语言模型对低资源语言支持不足的问题。通过扩展词汇并利用双语词典优化初始化，提升模型在目标语言上的表现。**

- **链接: [https://arxiv.org/pdf/2602.09388v1](https://arxiv.org/pdf/2602.09388v1)**

> **作者:** Jianyu Zheng
>
> **备注:** 12 pages, 5 figures, 7 tables, under review
>
> **摘要:** Multilingual pre-trained language models(mPLMs) offer significant benefits for many low-resource languages. To further expand the range of languages these models can support, many works focus on continued pre-training of these models. However, few works address how to extend mPLMs to low-resource languages that were previously unsupported. To tackle this issue, we expand the model's vocabulary using a target language corpus. We then screen out a subset from the model's original vocabulary, which is biased towards representing the source language(e.g. English), and utilize bilingual dictionaries to initialize the representations of the expanded vocabulary. Subsequently, we continue to pre-train the mPLMs using the target language corpus, based on the representations of these expanded vocabulary. Experimental results show that our proposed method outperforms the baseline, which uses randomly initialized expanded vocabulary for continued pre-training, in POS tagging and NER tasks, achieving improvements by 0.54% and 2.60%, respectively. Furthermore, our method demonstrates high robustness in selecting the training corpora, and the models' performance on the source language does not degrade after continued pre-training.
>
---
#### [new 063] The Devil Behind Moltbook: Anthropic Safety is Always Vanishing in Self-Evolving AI Societies
- **分类: cs.CL**

- **简介: 该论文属于AI安全领域，研究自进化AI社会的安全问题。解决自进化与安全对齐的矛盾，通过理论分析和实验验证，揭示安全退化机制并提出解决方案。**

- **链接: [https://arxiv.org/pdf/2602.09877v1](https://arxiv.org/pdf/2602.09877v1)**

> **作者:** Chenxu Wang; Chaozhuo Li; Songyang Liu; Zejian Chen; Jinyu Hou; Ji Qi; Rui Li; Litian Zhang; Qiwei Ye; Zheng Liu; Xu Chen; Xi Zhang; Philip S. Yu
>
> **摘要:** The emergence of multi-agent systems built from large language models (LLMs) offers a promising paradigm for scalable collective intelligence and self-evolution. Ideally, such systems would achieve continuous self-improvement in a fully closed loop while maintaining robust safety alignment--a combination we term the self-evolution trilemma. However, we demonstrate both theoretically and empirically that an agent society satisfying continuous self-evolution, complete isolation, and safety invariance is impossible. Drawing on an information-theoretic framework, we formalize safety as the divergence degree from anthropic value distributions. We theoretically demonstrate that isolated self-evolution induces statistical blind spots, leading to the irreversible degradation of the system's safety alignment. Empirical and qualitative results from an open-ended agent community (Moltbook) and two closed self-evolving systems reveal phenomena that align with our theoretical prediction of inevitable safety erosion. We further propose several solution directions to alleviate the identified safety concern. Our work establishes a fundamental limit on the self-evolving AI societies and shifts the discourse from symptom-driven safety patches to a principled understanding of intrinsic dynamical risks, highlighting the need for external oversight or novel safety-preserving mechanisms.
>
---
#### [new 064] Circuit Fingerprints: How Answer Tokens Encode Their Geometrical Path
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型解释与控制任务，旨在解决如何发现和操控Transformer中的电路结构。通过几何方向分析，实现无需梯度的电路发现与情感控制，提升可解释性与可控性。**

- **链接: [https://arxiv.org/pdf/2602.09784v1](https://arxiv.org/pdf/2602.09784v1)**

> **作者:** Andres Saurez; Neha Sengar; Dongsoo Har
>
> **备注:** Submitted to ICML 2026. 15 pages, 11 figures
>
> **摘要:** Circuit discovery and activation steering in transformers have developed as separate research threads, yet both operate on the same representational space. Are they two views of the same underlying structure? We show they follow a single geometric principle: answer tokens, processed in isolation, encode the directions that would produce them. This Circuit Fingerprint hypothesis enables circuit discovery without gradients or causal intervention -- recovering comparable structure to gradient-based methods through geometric alignment alone. We validate this on standard benchmarks (IOI, SVA, MCQA) across four model families, achieving circuit discovery performance comparable to gradient-based methods. The same directions that identify circuit components also enable controlled steering -- achieving 69.8\% emotion classification accuracy versus 53.1\% for instruction prompting while preserving factual accuracy. Beyond method development, this read-write duality reveals that transformer circuits are fundamentally geometric structures: interpretability and controllability are two facets of the same object.
>
---
#### [new 065] AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于代码生成任务，旨在解决验证代码评估标准不统一的问题。提出AlgoVeri基准，测试三种工具在77个经典算法上的表现，揭示各系统的验证能力差异。**

- **链接: [https://arxiv.org/pdf/2602.09464v1](https://arxiv.org/pdf/2602.09464v1)**

> **作者:** Haoyu Zhao; Ziran Yang; Jiawei Li; Deyuan He; Zenan Li; Chi Jin; Venugopal V. Veeravalli; Aarti Gupta; Sanjeev Arora
>
> **备注:** 32 pages
>
> **摘要:** Vericoding refers to the generation of formally verified code from rigorous specifications. Recent AI models show promise in vericoding, but a unified methodology for cross-paradigm evaluation is lacking. Existing benchmarks test only individual languages/tools (e.g., Dafny, Verus, and Lean) and each covers very different tasks, so the performance numbers are not directly comparable. We address this gap with AlgoVeri, a benchmark that evaluates vericoding of $77$ classical algorithms in Dafny, Verus, and Lean. By enforcing identical functional contracts, AlgoVeri reveals critical capability gaps in verification systems. While frontier models achieve tractable success in Dafny ($40.3$% for Gemini-3 Flash), where high-level abstractions and SMT automation simplify the workflow, performance collapses under the systems-level memory constraints of Verus ($24.7$%) and the explicit proof construction required by Lean (7.8%). Beyond aggregate metrics, we uncover a sharp divergence in test-time compute dynamics: Gemini-3 effectively utilizes iterative repair to boost performance (e.g., tripling pass rates in Dafny), whereas GPT-OSS saturates early. Finally, our error analysis shows that language design affects the refinement trajectory: while Dafny allows models to focus on logical correctness, Verus and Lean trap models in persistent syntactic and semantic barriers. All data and evaluation code can be found at https://github.com/haoyuzhao123/algoveri.
>
---
#### [new 066] Would a Large Language Model Pay Extra for a View? Inferring Willingness to Pay from Subjective Choices
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于主观决策支持任务，研究LLM在旅行助手场景中的选择行为，通过分析其响应推断隐含支付意愿，并与人类基准比较，探讨模型在不同条件下的表现。**

- **链接: [https://arxiv.org/pdf/2602.09802v1](https://arxiv.org/pdf/2602.09802v1)**

> **作者:** Manon Reusens; Sofie Goethals; Toon Calders; David Martens
>
> **摘要:** As Large Language Models (LLMs) are increasingly deployed in applications such as travel assistance and purchasing support, they are often required to make subjective choices on behalf of users in settings where no objectively correct answer exists. We study LLM decision-making in a travel-assistant context by presenting models with choice dilemmas and analyzing their responses using multinomial logit models to derive implied willingness to pay (WTP) estimates. These WTP values are subsequently compared to human benchmark values from the economics literature. In addition to a baseline setting, we examine how model behavior changes under more realistic conditions, including the provision of information about users' past choices and persona-based prompting. Our results show that while meaningful WTP values can be derived for larger LLMs, they also display systematic deviations at the attribute level. Additionally, they tend to overestimate human WTP overall, particularly when expensive options or business-oriented personas are introduced. Conditioning models on prior preferences for cheaper options yields valuations that are closer to human benchmarks. Overall, our findings highlight both the potential and the limitations of using LLMs for subjective decision support and underscore the importance of careful model selection, prompt design, and user representation when deploying such systems in practice.
>
---
#### [new 067] Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出Agent World Model（AWM），用于生成合成环境以训练强化学习代理。解决缺乏多样化可靠环境的问题，通过代码驱动的环境提升训练效率与一致性。**

- **链接: [https://arxiv.org/pdf/2602.10090v1](https://arxiv.org/pdf/2602.10090v1)**

> **作者:** Zhaoyang Wang; Canwen Xu; Boyi Liu; Yite Wang; Siwei Han; Zhewei Yao; Huaxiu Yao; Yuxiong He
>
> **备注:** 41 pages
>
> **摘要:** Recent advances in large language model (LLM) have empowered autonomous agents to perform complex tasks that require multi-turn interactions with tools and environments. However, scaling such agent training is limited by the lack of diverse and reliable environments. In this paper, we propose Agent World Model (AWM), a fully synthetic environment generation pipeline. Using this pipeline, we scale to 1,000 environments covering everyday scenarios, in which agents can interact with rich toolsets (35 tools per environment on average) and obtain high-quality observations. Notably, these environments are code-driven and backed by databases, providing more reliable and consistent state transitions than environments simulated by LLMs. Moreover, they enable more efficient agent interaction compared with collecting trajectories from realistic environments. To demonstrate the effectiveness of this resource, we perform large-scale reinforcement learning for multi-turn tool-use agents. Thanks to the fully executable environments and accessible database states, we can also design reliable reward functions. Experiments on three benchmarks show that training exclusively in synthetic environments, rather than benchmark-specific ones, yields strong out-of-distribution generalization. The code is available at https://github.com/Snowflake-Labs/agent-world-model.
>
---
#### [new 068] Covo-Audio Technical Report
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出Covo-Audio，一个7B参数的端到端音频语言模型，解决音频理解和对话生成任务，通过预训练和微调实现高质量语音交互。**

- **链接: [https://arxiv.org/pdf/2602.09823v1](https://arxiv.org/pdf/2602.09823v1)**

> **作者:** Wenfu Wang; Chenxing Li; Liqiang Zhang; Yiyang Zhao; Yuxiang Zou; Hanzhao Li; Mingyu Cui; Hao Zhang; Kun Wei; Le Xu; Zikang Huang; Jiajun Xu; Jiliang Hu; Xiang He; Zeyu Xie; Jiawen Kang; Youjun Chen; Meng Yu; Dong Yu; Rilin Chen; Linlin Di; Shulin Feng; Na Hu; Yang Liu; Bang Wang; Shan Yang
>
> **备注:** Technical Report
>
> **摘要:** In this work, we present Covo-Audio, a 7B-parameter end-to-end LALM that directly processes continuous audio inputs and generates audio outputs within a single unified architecture. Through large-scale curated pretraining and targeted post-training, Covo-Audio achieves state-of-the-art or competitive performance among models of comparable scale across a broad spectrum of tasks, including speech-text modeling, spoken dialogue, speech understanding, audio understanding, and full-duplex voice interaction. Extensive evaluations demonstrate that the pretrained foundation model exhibits strong speech-text comprehension and semantic reasoning capabilities on multiple benchmarks, outperforming representative open-source models of comparable scale. Furthermore, Covo-Audio-Chat, the dialogue-oriented variant, demonstrates strong spoken conversational abilities, including understanding, contextual reasoning, instruction following, and generating contextually appropriate and empathetic responses, validating its applicability to real-world conversational assistant scenarios. Covo-Audio-Chat-FD, the evolved full-duplex model, achieves substantially superior performance on both spoken dialogue capabilities and full-duplex interaction behaviors, demonstrating its competence in practical robustness. To mitigate the high cost of deploying end-to-end LALMs for natural conversational systems, we propose an intelligence-speaker decoupling strategy that separates dialogue intelligence from voice rendering, enabling flexible voice customization with minimal text-to-speech (TTS) data while preserving dialogue performance. Overall, our results highlight the strong potential of 7B-scale models to integrate sophisticated audio intelligence with high-level semantic reasoning, and suggest a scalable path toward more capable and versatile LALMs.
>
---
#### [new 069] Triggered: A Statistical Analysis of Environmental Influences on Extremist Groups
- **分类: cs.SI; cs.CL**

- **简介: 该论文属于信息生态研究，分析环境因素对极端主义群体的影响。通过数据建模，探讨暴力事件、新闻报道及语言扩散对社区行为的作用，揭示不同极端群体的差异性反应。**

- **链接: [https://arxiv.org/pdf/2602.09289v1](https://arxiv.org/pdf/2602.09289v1)**

> **作者:** Christine de Kock; Eduard Hovy
>
> **摘要:** Online extremist communities operate within a wider information ecosystem shaped by real-world events, news coverage, and cross-community interaction. We adopt a systems perspective to examine these influences using seven years of data from two ideologically distinct extremist forums (Stormfront and Incels) and a mainstream reference community (r/News). We ask three questions: how extremist violence impacts community behaviour; whether news coverage of political entities predicts shifts in conversation dynamics; and whether linguistic diffusion occurs between mainstream and extremist spaces and across extremist ideologies. Methodologically, we combine counterfactual synthesis to estimate event-level impacts with vector autoregression and Granger causality analyses to model ongoing relationships among news signals, behavioural outcomes, and cross-community language change. Across analyses, our results indicate that Stormfront and r/News appear to be more reactive to external stimuli, while Incels demonstrates less cross-community linguistic influence and less responsiveness to news and violent events. These findings underscore that extremist communities are not homogeneous, but differ in how tightly they are coupled to the surrounding information ecosystem.
>
---
#### [new 070] PABU: Progress-Aware Belief Update for Efficient LLM Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出PABU框架，解决LLM代理任务中因全历史信息导致的冗余和效率低的问题。通过建模任务进度和选择性保留信息，提升任务完成率并减少交互步骤。**

- **链接: [https://arxiv.org/pdf/2602.09138v1](https://arxiv.org/pdf/2602.09138v1)**

> **作者:** Haitao Jiang; Lin Ge; Hengrui Cai; Rui Song
>
> **摘要:** Large Language Model (LLM) agents commonly condition actions on full action-observation histories, which introduce task-irrelevant information that easily leads to redundant actions and higher inference cost. We propose Progress-Aware Belief Update (PABU), a belief-state framework that compactly represents an agent's state by explicitly modeling task progress and selectively retaining past actions and observations. At each step, the agent predicts its relative progress since the previous round and decides whether the newly encountered interaction should be stored, conditioning future decisions only on the retained subset. Across eight environments in the AgentGym benchmark, and using identical training trajectories, PABU achieves an 81.0% task completion rate, outperforming previous State of the art (SoTA) models with full-history belief by 23.9%. Additionally, PABU's progress-oriented action selection improves efficiency, reducing the average number of interaction steps to 9.5, corresponding to a 26.9% reduction. Ablation studies show that both explicit progress prediction and selective retention are necessary for robust belief learning and performance gains.
>
---
#### [new 071] UI-Venus-1.5 Technical Report
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出UI-Venus-1.5，一个统一的GUI代理模型，用于自动化数字环境交互。解决GUI代理在泛化性和性能上的挑战，通过技术改进提升任务执行效果。**

- **链接: [https://arxiv.org/pdf/2602.09082v1](https://arxiv.org/pdf/2602.09082v1)**

> **作者:** Veuns-Team; :; Changlong Gao; Zhangxuan Gu; Yulin Liu; Xinyu Qiu; Shuheng Shen; Yue Wen; Tianyu Xia; Zhenyu Xu; Zhengwen Zeng; Beitong Zhou; Xingran Zhou; Weizhi Chen; Sunhao Dai; Jingya Dou; Yichen Gong; Yuan Guo; Zhenlin Guo; Feng Li; Qian Li; Jinzhen Lin; Yuqi Zhou; Linchao Zhu; Liang Chen; Zhenyu Guo; Changhua Meng; Weiqiang Wang
>
> **摘要:** GUI agents have emerged as a powerful paradigm for automating interactions in digital environments, yet achieving both broad generality and consistently strong task performance remains challenging.In this report, we present UI-Venus-1.5, a unified, end-to-end GUI Agent designed for robust real-world applications.The proposed model family comprises two dense variants (2B and 8B) and one mixture-of-experts variant (30B-A3B) to meet various downstream application scenarios.Compared to our previous version, UI-Venus-1.5 introduces three key technical advances: (1) a comprehensive Mid-Training stage leveraging 10 billion tokens across 30+ datasets to establish foundational GUI semantics; (2) Online Reinforcement Learning with full-trajectory rollouts, aligning training objectives with long-horizon, dynamic navigation in large-scale environments; and (3) a single unified GUI Agent constructed via Model Merging, which synthesizes domain-specific models (grounding, web, and mobile) into one cohesive checkpoint. Extensive evaluations demonstrate that UI-Venus-1.5 establishes new state-of-the-art performance on benchmarks such as ScreenSpot-Pro (69.6%), VenusBench-GD (75.0%), and AndroidWorld (77.6%), significantly outperforming previous strong baselines. In addition, UI-Venus-1.5 demonstrates robust navigation capabilities across a variety of Chinese mobile apps, effectively executing user instructions in real-world scenarios. Code: https://github.com/inclusionAI/UI-Venus; Model: https://huggingface.co/collections/inclusionAI/ui-venus
>
---
#### [new 072] Code2World: A GUI World Model via Renderable Code Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.HC**

- **简介: 该论文提出Code2World，解决GUI环境预测问题。通过生成可渲染代码实现高保真界面预测，提升导航成功率。**

- **链接: [https://arxiv.org/pdf/2602.09856v1](https://arxiv.org/pdf/2602.09856v1)**

> **作者:** Yuhao Zheng; Li'an Zhong; Yi Wang; Rui Dai; Kaikui Liu; Xiangxiang Chu; Linyuan Lv; Philip Torr; Kevin Qinghong Lin
>
> **备注:** github: https://github.com/AMAP-ML/Code2World project page: https://amap-ml.github.io/Code2World/
>
> **摘要:** Autonomous GUI agents interact with environments by perceiving interfaces and executing actions. As a virtual sandbox, the GUI World model empowers agents with human-like foresight by enabling action-conditioned prediction. However, existing text- and pixel-based approaches struggle to simultaneously achieve high visual fidelity and fine-grained structural controllability. To this end, we propose Code2World, a vision-language coder that simulates the next visual state via renderable code generation. Specifically, to address the data scarcity problem, we construct AndroidCode by translating GUI trajectories into high-fidelity HTML and refining synthesized code through a visual-feedback revision mechanism, yielding a corpus of over 80K high-quality screen-action pairs. To adapt existing VLMs into code prediction, we first perform SFT as a cold start for format layout following, then further apply Render-Aware Reinforcement Learning which uses rendered outcome as the reward signal by enforcing visual semantic fidelity and action consistency. Extensive experiments demonstrate that Code2World-8B achieves the top-performing next UI prediction, rivaling the competitive GPT-5 and Gemini-3-Pro-Image. Notably, Code2World significantly enhances downstream navigation success rates in a flexible manner, boosting Gemini-2.5-Flash by +9.5% on AndroidWorld navigation. The code is available at https://github.com/AMAP-ML/Code2World.
>
---
#### [new 073] Overview of the TREC 2025 RAGTIME Track
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于多语言信息报告生成任务，旨在研究从多语种文档生成报告的问题。论文介绍了RAGTIME跟踪任务及其三个子任务，并展示了相关结果。**

- **链接: [https://arxiv.org/pdf/2602.10024v1](https://arxiv.org/pdf/2602.10024v1)**

> **作者:** Dawn Lawrie; Sean MacAvaney; James Mayfield; Luca Soldaini; Eugene Yang; Andrew Yates
>
> **备注:** 10 pages, 3 figures, notebook version of the RAGTIME 2025 overview paper
>
> **摘要:** The principal goal of the RAG TREC Instrument for Multilingual Evaluation (RAGTIME) track at TREC is to study report generation from multilingual source documents. The track has created a document collection containing Arabic, Chinese, English, and Russian news stories. RAGTIME includes three task types: Multilingual Report Generation, English Report Generation, and Multilingual Information Retrieval (MLIR). A total of 125 runs were submitted by 13 participating teams (and as baselines by the track coordinators) for three tasks. This overview describes these three tasks and presents the available results.
>
---
#### [new 074] Benchmarking the Energy Savings with Speculative Decoding Strategies
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型优化任务，旨在研究 speculative decoding 策略的能耗问题，分析模型规模、策略和数据集对能耗的影响。**

- **链接: [https://arxiv.org/pdf/2602.09113v1](https://arxiv.org/pdf/2602.09113v1)**

> **作者:** Rohit Dutta; Paramita Koley; Soham Poddar; Janardan Misra; Sanjay Podder; Naveen Balani; Saptarshi Ghosh; Niloy Ganguly
>
> **备注:** Accepted at EACL Findings 2026
>
> **摘要:** Speculative decoding has emerged as an effective method to reduce latency and inference cost of LLM inferences. However, there has been inadequate attention towards the energy requirements of these models. To address this gap, this paper presents a comprehensive survey of energy requirements of speculative decoding strategies, with detailed analysis on how various factors -- model size and family, speculative decoding strategies, and dataset characteristics -- influence the energy optimizations.
>
---
#### [new 075] Flexible Entropy Control in RLVR with Gradient-Preserving Perspective
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决RLVR中策略熵崩溃问题。通过动态剪切阈值调控熵，提升模型输出多样性和学习效率。**

- **链接: [https://arxiv.org/pdf/2602.09782v1](https://arxiv.org/pdf/2602.09782v1)**

> **作者:** Kun Chen; Peng Shi; Fanfan Liu; Haibo Qiu; Zhixiong Zeng; Siqi Yang; Wenji Mao
>
> **备注:** https://github.com/Kwen-Chen/Flexible-Entropy-Control
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a critical method for enhancing the reasoning capabilities of Large Language Models (LLMs). However, continuous training often leads to policy entropy collapse, characterized by a rapid decay in entropy that results in premature overconfidence, reduced output diversity, and vanishing gradient norms that inhibit learning. Gradient-Preserving Clipping is a primary factor influencing these dynamics, but existing mitigation strategies are largely static and lack a framework connecting clipping mechanisms to precise entropy control. This paper proposes reshaping entropy control in RL from the perspective of Gradient-Preserving Clipping. We first theoretically and empirically verify the contributions of specific importance sampling ratio regions to entropy growth and reduction. Leveraging these findings, we introduce a novel regulation mechanism using dynamic clipping threshold to precisely manage entropy. Furthermore, we design and evaluate dynamic entropy control strategies, including increase-then-decrease, decrease-increase-decrease, and oscillatory decay. Experimental results demonstrate that these strategies effectively mitigate entropy collapse, and achieve superior performance across multiple benchmarks.
>
---
#### [new 076] LingxiDiagBench: A Multi-Agent Framework for Benchmarking LLMs in Chinese Psychiatric Consultation and Diagnosis
- **分类: cs.MA; cs.CL**

- **简介: 该论文提出LingxiDiagBench，用于评估大语言模型在中文精神科咨询与诊断中的性能，解决缺乏真实模拟和动态对话评估的问题。**

- **链接: [https://arxiv.org/pdf/2602.09379v1](https://arxiv.org/pdf/2602.09379v1)**

> **作者:** Shihao Xu; Tiancheng Zhou; Jiatong Ma; Yanli Ding; Yiming Yan; Ming Xiao; Guoyi Li; Haiyang Geng; Yunyun Han; Jianhua Chen; Yafeng Deng
>
> **摘要:** Mental disorders are highly prevalent worldwide, but the shortage of psychiatrists and the inherent subjectivity of interview-based diagnosis create substantial barriers to timely and consistent mental-health assessment. Progress in AI-assisted psychiatric diagnosis is constrained by the absence of benchmarks that simultaneously provide realistic patient simulation, clinician-verified diagnostic labels, and support for dynamic multi-turn consultation. We present LingxiDiagBench, a large-scale multi-agent benchmark that evaluates LLMs on both static diagnostic inference and dynamic multi-turn psychiatric consultation in Chinese. At its core is LingxiDiag-16K, a dataset of 16,000 EMR-aligned synthetic consultation dialogues designed to reproduce real clinical demographic and diagnostic distributions across 12 ICD-10 psychiatric categories. Through extensive experiments across state-of-the-art LLMs, we establish key findings: (1) although LLMs achieve high accuracy on binary depression--anxiety classification (up to 92.3%), performance deteriorates substantially for depression--anxiety comorbidity recognition (43.0%) and 12-way differential diagnosis (28.5%); (2) dynamic consultation often underperforms static evaluation, indicating that ineffective information-gathering strategies significantly impair downstream diagnostic reasoning; (3) consultation quality assessed by LLM-as-a-Judge shows only moderate correlation with diagnostic accuracy, suggesting that well-structured questioning alone does not ensure correct diagnostic decisions. We release LingxiDiag-16K and the full evaluation framework to support reproducible research at https://github.com/Lingxi-mental-health/LingxiDiagBench.
>
---
#### [new 077] CAPID: Context-Aware PII Detection for Question-Answering Systems
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于隐私保护任务，旨在解决用户查询中PII检测问题。提出CAPID方法，通过微调小语言模型，准确识别并过滤敏感信息，提升问答系统隐私性和响应质量。**

- **链接: [https://arxiv.org/pdf/2602.10074v1](https://arxiv.org/pdf/2602.10074v1)**

> **作者:** Mariia Ponomarenko; Sepideh Abedini; Masoumeh Shafieinejad; D. B. Emerson; Shubhankar Mohapatra; Xi He
>
> **备注:** Accepted to the Student Research Workshop at EACL 2026
>
> **摘要:** Detecting personally identifiable information (PII) in user queries is critical for ensuring privacy in question-answering systems. Current approaches mainly redact all PII, disregarding the fact that some of them may be contextually relevant to the user's question, resulting in a degradation of response quality. Large language models (LLMs) might be able to help determine which PII are relevant, but due to their closed source nature and lack of privacy guarantees, they are unsuitable for sensitive data processing. To achieve privacy-preserving PII detection, we propose CAPID, a practical approach that fine-tunes a locally owned small language model (SLM) that filters sensitive information before it is passed to LLMs for QA. However, existing datasets do not capture the context-dependent relevance of PII needed to train such a model effectively. To fill this gap, we propose a synthetic data generation pipeline that leverages LLMs to produce a diverse, domain-rich dataset spanning multiple PII types and relevance levels. Using this dataset, we fine-tune an SLM to detect PII spans, classify their types, and estimate contextual relevance. Our experiments show that relevance-aware PII detection with a fine-tuned SLM substantially outperforms existing baselines in span, relevance and type accuracy while preserving significantly higher downstream utility under anonymization.
>
---
#### [new 078] SWE-AGI: Benchmarking Specification-Driven Software Construction with MoonBit in the Era of Autonomous Agents
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出SWE-AGI基准，用于评估AI在明确规范下构建生产级软件的能力。任务是基于标准文档实现复杂系统，解决AI自主软件工程的可行性与挑战问题。**

- **链接: [https://arxiv.org/pdf/2602.09447v1](https://arxiv.org/pdf/2602.09447v1)**

> **作者:** Zhirui Zhang; Hongbo Zhang; Haoxiang Fei; Zhiyuan Bao; Yubin Chen; Zhengyu Lei; Ziyue Liu; Yixuan Sun; Mingkun Xiao; Zihang Ye; Yu Zhang; Hongcheng Zhu; Yuxiang Wen; Heung-Yeung Shum
>
> **备注:** 20 pages, 3 figures
>
> **摘要:** Although large language models (LLMs) have demonstrated impressive coding capabilities, their ability to autonomously build production-scale software from explicit specifications remains an open question. We introduce SWE-AGI, an open-source benchmark for evaluating end-to-end, specification-driven construction of software systems written in MoonBit. SWE-AGI tasks require LLM-based agents to implement parsers, interpreters, binary decoders, and SAT solvers strictly from authoritative standards and RFCs under a fixed API scaffold. Each task involves implementing 1,000-10,000 lines of core logic, corresponding to weeks or months of engineering effort for an experienced human developer. By leveraging the nascent MoonBit ecosystem, SWE-AGI minimizes data leakage, forcing agents to rely on long-horizon architectural reasoning rather than code retrieval. Across frontier models, gpt-5.3-codex achieves the best overall performance (solving 19/22 tasks, 86.4%), outperforming claude-opus-4.6 (15/22, 68.2%), and kimi-2.5 exhibits the strongest performance among open-source models. Performance degrades sharply with increasing task difficulty, particularly on hard, specification-intensive systems. Behavioral analysis further reveals that as codebases scale, code reading, rather than writing, becomes the dominant bottleneck in AI-assisted development. Overall, while specification-driven autonomous software engineering is increasingly viable, substantial challenges remain before it can reliably support production-scale development.
>
---
#### [new 079] QP-OneModel: A Unified Generative LLM for Multi-Task Query Understanding in Xiaohongshu Search
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出QP-OneModel，解决SNS搜索中多任务查询理解问题。通过统一生成模型和强化学习，提升语义理解与任务效果。**

- **链接: [https://arxiv.org/pdf/2602.09901v1](https://arxiv.org/pdf/2602.09901v1)**

> **作者:** Jianzhao Huang; Xiaorui Huang; Fei Zhao; Yunpeng Liu; Hui Zhang; Fangcheng Shi; Congfeng Li; Zechen Sun; Yi Wu; Yao Hu; Yunhan Bai; Shaosheng Cao
>
> **摘要:** Query Processing (QP) bridges user intent and content supply in large-scale Social Network Service (SNS) search engines. Traditional QP systems rely on pipelines of isolated discriminative models (e.g., BERT), suffering from limited semantic understanding and high maintenance overhead. While Large Language Models (LLMs) offer a potential solution, existing approaches often optimize sub-tasks in isolation, neglecting intrinsic semantic synergy and necessitating independent iterations. Moreover, standard generative methods often lack grounding in SNS scenarios, failing to bridge the gap between open-domain corpora and informal SNS linguistic patterns, while struggling to adhere to rigorous business definitions. We present QP-OneModel, a Unified Generative LLM for Multi-Task Query Understanding in the SNS domain. We reformulate heterogeneous sub-tasks into a unified sequence generation paradigm, adopting a progressive three-stage alignment strategy culminating in multi-reward Reinforcement Learning. Furthermore, QP-OneModel generates intent descriptions as a novel high-fidelity semantic signal, effectively augmenting downstream tasks such as query rewriting and ranking. Offline evaluations show QP-OneModel achieves a 7.35% overall gain over discriminative baselines, with significant F1 boosts in NER (+9.01%) and Term Weighting (+9.31%). It also exhibits superior generalization, surpassing a 32B model by 7.60% accuracy on unseen tasks. Fully deployed at Xiaohongshu, online A/B tests confirm its industrial value, optimizing retrieval relevance (DCG) by 0.21% and lifting user retention by 0.044%.
>
---
#### [new 080] FlyAOC: Evaluating Agentic Ontology Curation of Drosophila Scientific Knowledge Bases
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出FlyBench，用于评估AI代理在科学文献中进行本体论注释的全流程能力。任务属于科学知识库的自动化注释，解决现有基准无法覆盖完整工作流的问题。工作包括设计基准、测试不同代理架构并分析性能差异。**

- **链接: [https://arxiv.org/pdf/2602.09163v1](https://arxiv.org/pdf/2602.09163v1)**

> **作者:** Xingjian Zhang; Sophia Moylan; Ziyang Xiong; Qiaozhu Mei; Yichen Luo; Jiaqi W. Ma
>
> **摘要:** Scientific knowledge bases accelerate discovery by curating findings from primary literature into structured, queryable formats for both human researchers and emerging AI systems. Maintaining these resources requires expert curators to search relevant papers, reconcile evidence across documents, and produce ontology-grounded annotations - a workflow that existing benchmarks, focused on isolated subtasks like named entity recognition or relation extraction, do not capture. We present FlyBench to evaluate AI agents on end-to-end agentic ontology curation from scientific literature. Given only a gene symbol, agents must search and read from a corpus of 16,898 full-text papers to produce structured annotations: Gene Ontology terms describing function, expression patterns, and historical synonyms linking decades of nomenclature. The benchmark includes 7,397 expert-curated annotations across 100 genes drawn from FlyBase, the Drosophila (fruit fly) knowledge base. We evaluate four baseline agent architectures: memorization, fixed pipeline, single-agent, and multi-agent. We find that architectural choices significantly impact performance, with multi-agent designs outperforming simpler alternatives, yet scaling backbone models yields diminishing returns. All baselines leave substantial room for improvement. Our analysis surfaces several findings to guide future development; for example, agents primarily use retrieval to confirm parametric knowledge rather than discover new information. We hope FlyBench will drive progress on retrieval-augmented scientific reasoning, a capability with broad applications across scientific domains.
>
---
#### [new 081] Why Linear Interpretability Works: Invariant Subspaces as a Result of Architectural Constraints
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究Transformer模型中线性可解释性的原因，解决为何简单方法能有效解析复杂结构的问题。通过理论分析和实验验证，提出架构约束导致语义特征位于不变子空间。**

- **链接: [https://arxiv.org/pdf/2602.09783v1](https://arxiv.org/pdf/2602.09783v1)**

> **作者:** Andres Saurez; Yousung Lee; Dongsoo Har
>
> **备注:** Submitted to ICML 2026. 19 pages, 13 figures
>
> **摘要:** Linear probes and sparse autoencoders consistently recover meaningful structure from transformer representations -- yet why should such simple methods succeed in deep, nonlinear systems? We show this is not merely an empirical regularity but a consequence of architectural necessity: transformers communicate information through linear interfaces (attention OV circuits, unembedding matrices), and any semantic feature decoded through such an interface must occupy a context-invariant linear subspace. We formalize this as the \emph{Invariant Subspace Necessity} theorem and derive the \emph{Self-Reference Property}: tokens directly provide the geometric direction for their associated features, enabling zero-shot identification of semantic structure without labeled data or learned probes. Empirical validation in eight classification tasks and four model families confirms the alignment between class tokens and semantically related instances. Our framework provides \textbf{a principled architectural explanation} for why linear interpretability methods work, unifying linear probes and sparse autoencoders.
>
---
#### [new 082] TVTSyn: Content-Synchronous Time-Varying Timbre for Streaming Voice Conversion and Anonymization
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于语音转换与匿名化任务，解决实时合成中身份与内容不匹配的问题。提出TVT表示，实现内容同步的时变音色，提升自然度与隐私保护。**

- **链接: [https://arxiv.org/pdf/2602.09389v1](https://arxiv.org/pdf/2602.09389v1)**

> **作者:** Waris Quamer; Mu-Ruei Tseng; Ghady Nasrallah; Ricardo Gutierrez-Osuna
>
> **摘要:** Real-time voice conversion and speaker anonymization require causal, low-latency synthesis without sacrificing intelligibility or naturalness. Current systems have a core representational mismatch: content is time-varying, while speaker identity is injected as a static global embedding. We introduce a streamable speech synthesizer that aligns the temporal granularity of identity and content via a content-synchronous, time-varying timbre (TVT) representation. A Global Timbre Memory expands a global timbre instance into multiple compact facets; frame-level content attends to this memory, a gate regulates variation, and spherical interpolation preserves identity geometry while enabling smooth local changes. In addition, a factorized vector-quantized bottleneck regularizes content to reduce residual speaker leakage. The resulting system is streamable end-to-end, with <80 ms GPU latency. Experiments show improvements in naturalness, speaker transfer, and anonymization compared to SOTA streaming baselines, establishing TVT as a scalable approach for privacy-preserving and expressive speech synthesis under strict latency budgets.
>
---
#### [new 083] Not-in-Perspective: Towards Shielding Google's Perspective API Against Adversarial Negation Attacks
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于文本检测任务，旨在解决机器学习模型对否定攻击的脆弱性问题。通过引入形式化推理方法，提升毒性检测系统的鲁棒性与准确性。**

- **链接: [https://arxiv.org/pdf/2602.09343v1](https://arxiv.org/pdf/2602.09343v1)**

> **作者:** Michail S. Alexiou; J. Sukarno Mertoguno
>
> **摘要:** The rise of cyberbullying in social media platforms involving toxic comments has escalated the need for effective ways to monitor and moderate online interactions. Existing solutions of automated toxicity detection systems, are based on a machine or deep learning algorithms. However, statistics-based solutions are generally prone to adversarial attacks that contain logic based modifications such as negation in phrases and sentences. In that regard, we present a set of formal reasoning-based methodologies that wrap around existing machine learning toxicity detection systems. Acting as both pre-processing and post-processing steps, our formal reasoning wrapper helps alleviating the negation attack problems and significantly improves the accuracy and efficacy of toxicity scoring. We evaluate different variations of our wrapper on multiple machine learning models against a negation adversarial dataset. Experimental results highlight the improvement of hybrid (formal reasoning and machine-learning) methods against various purely statistical solutions.
>
---
#### [new 084] Collective Behavior of AI Agents: the Case of Moltbook
- **分类: physics.soc-ph; cs.CL; cs.MA**

- **简介: 该论文分析AI代理在Moltbook上的集体行为，探讨其与人类社交系统的相似性和差异。任务为研究AI社交动态，解决AI群体行为模式问题，通过大数据分析发现统计规律及关键区别。**

- **链接: [https://arxiv.org/pdf/2602.09270v1](https://arxiv.org/pdf/2602.09270v1)**

> **作者:** Giordano De Marzo; David Garcia
>
> **摘要:** We present a large scale data analysis of Moltbook, a Reddit-style social media platform exclusively populated by AI agents. Analyzing over 369,000 posts and 3.0 million comments from approximately 46,000 active agents, we find that AI collective behavior exhibits many of the same statistical regularities observed in human online communities: heavy-tailed distributions of activity, power-law scaling of popularity metrics, and temporal decay patterns consistent with limited attention dynamics. However, we also identify key differences, including a sublinear relationship between upvotes and discussion size that contrasts with human behavior. These findings suggest that, while individual AI agents may differ fundamentally from humans, their emergent collective dynamics share structural similarities with human social systems.
>
---
## 更新

#### [replaced 001] Offline World Models as Imagination Networks in Cognitive Agents
- **分类: cs.AI; cs.CL; cs.SI; q-bio.NC**

- **简介: 该论文属于认知科学与人工智能交叉任务，旨在比较人类与大语言模型的内部世界模型。通过心理网络分析，研究者发现人类想象网络结构一致，而LLMs表现差异显著。**

- **链接: [https://arxiv.org/pdf/2510.04391v4](https://arxiv.org/pdf/2510.04391v4)**

> **作者:** Saurabh Ranjan; Brian Odegaard
>
> **摘要:** The computational role of imagination remains debated. While classical accounts emphasize reward maximization, emerging evidence suggests it accesses internal world models (IWMs). We employ psychological network analysis to compare IWMs in humans and large language models (LLMs) via imagination vividness ratings, distinguishing offline world models (persistent memory structures accessed independent of immediate goals) from online models (task-specific representations). Analyzing 2,743 humans across three populations and six LLM variants, we find human imagination networks exhibit robust structural consistency, with high centrality correlations and aligned clustering. LLMs show minimal clustering and weak correlations with human networks, even with conversational memory, across environmental and sensory contexts. These differences highlight disparities in how biological and artificial systems organize internal representations. Our framework offers quantitative metrics for evaluating offline world models in cognitive agents.
>
---
#### [replaced 002] CARINOX: Inference-time Scaling with Category-Aware Reward-based Initial Noise Optimization and Exploration
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文本到图像生成任务，解决复杂场景下图像与文本不一致的问题。提出CARINOX框架，结合噪声优化与探索，提升图像与文本的对齐度。**

- **链接: [https://arxiv.org/pdf/2509.17458v2](https://arxiv.org/pdf/2509.17458v2)**

> **作者:** Seyed Amir Kasaei; Ali Aghayari; Arash Marioriyad; Niki Sepasian; Shayan Baghayi Nejad; MohammadAmin Fazli; Mahdieh Soleymani Baghshah; Mohammad Hossein Rohban
>
> **备注:** Accepted at TMLR (2026)
>
> **摘要:** Text-to-image diffusion models, such as Stable Diffusion, can produce high-quality and diverse images but often fail to achieve compositional alignment, particularly when prompts describe complex object relationships, attributes, or spatial arrangements. Recent inference-time approaches address this by optimizing or exploring the initial noise under the guidance of reward functions that score text-image alignment without requiring model fine-tuning. While promising, each strategy has intrinsic limitations when used alone: optimization can stall due to poor initialization or unfavorable search trajectories, whereas exploration may require a prohibitively large number of samples to locate a satisfactory output. Our analysis further shows that neither single reward metrics nor ad-hoc combinations reliably capture all aspects of compositionality, leading to weak or inconsistent guidance. To overcome these challenges, we present Category-Aware Reward-based Initial Noise Optimization and Exploration (CARINOX), a unified framework that combines noise optimization and exploration with a principled reward selection procedure grounded in correlation with human judgments. Evaluations on two complementary benchmarks covering diverse compositional challenges show that CARINOX raises average alignment scores by +16% on T2I-CompBench++ and +11% on the HRS benchmark, consistently outperforming state-of-the-art optimization and exploration-based methods across all major categories, while preserving image quality and diversity. The project page is available at https://amirkasaei.com/carinox/.
>
---
#### [replaced 003] Short-Context Dominance: How Much Local Context Natural Language Actually Needs?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型对长上下文的依赖程度，旨在解决短上下文是否足够预测后续内容的问题。通过测量最小上下文长度，提出DaMCL方法识别长上下文序列，并改进解码算法以提升性能。**

- **链接: [https://arxiv.org/pdf/2512.08082v2](https://arxiv.org/pdf/2512.08082v2)**

> **作者:** Vala Vakilian; Zimeng Wang; Ankit Singh Rawat; Christos Thrampoulidis
>
> **备注:** 38 pages, 7 figures, includes appendix and references
>
> **摘要:** We investigate the short-context dominance hypothesis: that for most sequences, a small local prefix suffices to predict their next tokens. Using large language models as statistical oracles, we measure the minimum context length (MCL) needed to reproduce accurate full-context predictions across datasets with sequences of varying lengths. For sequences with 1-7k tokens from long-context documents, we consistently find that 75-80% require only the last 96 tokens at most. Given the dominance of short-context tokens, we then ask whether it is possible to detect challenging long-context sequences for which a short local prefix does not suffice for prediction. We introduce a practical proxy to MCL, called Distributionally Aware MCL (DaMCL), that does not require knowledge of the actual next-token and is compatible with sampling strategies beyond greedy decoding. Our experiments validate that simple thresholding of the metric defining DaMCL achieves high performance in detecting long vs. short context sequences. Finally, to counter the bias that short-context dominance induces in LLM output distributions, we develop an intuitive decoding algorithm that leverages our detector to identify and boost tokens that are long-range-relevant. Across Q&A tasks and model architectures, we confirm that mitigating the bias improves performance.
>
---
#### [replaced 004] In-Context Learning Without Copying
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究如何在不依赖诱导头的情况下实现抽象的上下文学习。通过提出Hapax训练方法，验证了抽象ICL能力可独立于诱导头存在。**

- **链接: [https://arxiv.org/pdf/2511.05743v2](https://arxiv.org/pdf/2511.05743v2)**

> **作者:** Kerem Sahin; Sheridan Feucht; Adam Belfki; Jannik Brinkmann; Aaron Mueller; David Bau; Chris Wendler
>
> **摘要:** Induction heads are attention heads that perform inductive copying by matching patterns from earlier context and copying their continuations verbatim. As models develop induction heads, they experience a sharp drop in training loss, a phenomenon cited as evidence that induction heads may underlie a wide range of in-context learning (ICL) capabilities. In this work, we investigate whether induction heads are a necessary building block for learning abstractive ICL capabilities (i.e., tasks where the answer is not contained in the input context), or whether such capabilities can emerge independently. We propose Hapax, a training regime that omits the loss contribution of tokens predictable by induction heads. Despite a significant reduction in inductive copying, abstractive ICL capabilities are preserved, with the model achieving higher accuracy than the vanilla model on 13 out of 21 tasks, even though 31.7% of tokens are omitted from the loss. Furthermore, our model achieves lower loss values on token positions that induction heads cannot predict. Mechanistic analysis shows that models trained with Hapax develop fewer and weaker induction heads despite preserving abstractive ICL capabilities. Our findings suggest that the developmental link between induction heads and abstractive ICL capabilities is weaker than previously hypothesized.
>
---
#### [replaced 005] Self-Guided Function Calling in Large Language Models via Stepwise Experience Recall
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决LLM在多步骤工具调用中的选择、参数生成和链路规划问题。提出SEER方法，通过逐步回忆经验池提升性能。**

- **链接: [https://arxiv.org/pdf/2508.15214v3](https://arxiv.org/pdf/2508.15214v3)**

> **作者:** Sijia Cui; Aiyao He; Shuai Xu; Hongming Zhang; Yanna Wang; Qingyang Zhang; Yajing Wang; Bo Xu
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Function calling enables large language models (LLMs) to interact with external systems by leveraging tools and APIs. When faced with multi-step tool usage, LLMs still struggle with tool selection, parameter generation, and tool-chain planning. Existing methods typically rely on manually designing task-specific demonstrations, or retrieving from a curated library. These approaches demand substantial expert effort and prompt engineering becomes increasingly complex and inefficient as tool diversity and task difficulty scale. To address these challenges, we propose a self-guided method, Stepwise Experience Recall (SEER), which performs fine-grained, stepwise retrieval from a continually updated experience pool. Instead of relying on static or manually curated library, SEER incrementally augments the experience pool with past successful trajectories, enabling continuous expansion of the pool and improved model performance over time. Evaluated on the ToolQA benchmark, SEER achieves an average improvement of 6.1% on easy and 4.7% on hard questions. We further test SEER on $τ$-bench, which includes two real-world domains. Powered by Qwen2.5-7B and Qwen2.5-72B models, SEER demonstrates substantial accuracy gains of 7.44% and 23.38%, respectively.
>
---
#### [replaced 006] Cochain: Balancing Insufficient and Excessive Collaboration in LLM Agent Workflows
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决LLM代理流程中协作不足与过度的问题。提出Cochain框架，通过知识图谱和提示树提升协作效率。**

- **链接: [https://arxiv.org/pdf/2505.10936v3](https://arxiv.org/pdf/2505.10936v3)**

> **作者:** Jiaxing Zhao; Hongbin Xie; Yuzhen Lei; Xuan Song; Zhuoran Shi; Lianxin Li; Shuangxue Liu; Linguo Xie; Haoran Zhang
>
> **备注:** 35 pages, 23 figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive performance in executing complex reasoning tasks. Chain-of-thought effectively enhances reasoning capabilities by unlocking the potential of large models, while multi-agent systems provide more comprehensive solutions by integrating the collective intelligence of multiple agents. However, both approaches face significant limitations. Single-agent with chain-of-thought, due to the inherent complexity of designing cross-domain prompts, faces collaboration challenges. Meanwhile, multi-agent systems consume substantial tokens and inevitably dilute the primary problem, which is particularly problematic in business workflow tasks. To address these challenges, we propose Cochain, a collaboration prompting framework that effectively solves the business workflow collaboration problem by combining knowledge and prompts at a reduced cost. Specifically, we construct an integrated knowledge graph that incorporates knowledge from multiple stages. Furthermore, by maintaining and retrieving a prompts tree, we can obtain prompt information relevant to other stages of the business workflow. We perform extensive evaluations of Cochain across multiple datasets, demonstrating that Cochain outperforms all baselines in both prompt engineering and multi-agent LLMs. Additionally, expert evaluation results indicate that the use of a small model in combination with Cochain outperforms GPT-4.
>
---
#### [replaced 007] Machine Text Detectors are Membership Inference Attacks
- **分类: cs.CL**

- **简介: 该论文属于信息安全领域，研究会员推理攻击与机器生成文本检测之间的关系。工作包括理论分析、实验验证及统一评估工具MINT的开发，旨在揭示两任务间的可迁移性。**

- **链接: [https://arxiv.org/pdf/2510.19492v2](https://arxiv.org/pdf/2510.19492v2)**

> **作者:** Ryuto Koike; Liam Dugan; Masahiro Kaneko; Chris Callison-Burch; Naoaki Okazaki
>
> **摘要:** Although membership inference attacks (MIAs) and machine-generated text detection target different goals, their methods often exploit similar signals based on a language model's probability distribution, and the two tasks have been studied independently. This can result in conclusions that overlook stronger methods and valuable insights from the other task. In this work, we theoretically and empirically demonstrate the transferability, i.e., how well a method originally developed for one task performs on the other, between MIAs and machine text detection. We prove that the metric achieving asymptotically optimal performance is identical for both tasks. We unify existing methods under this optimal metric and hypothesize that the accuracy with which a method approximates this metric is directly correlated with its transferability. Our large-scale empirical experiments demonstrate very strong rank correlation ($ρ\approx 0.7$) in cross-task performance. Notably, we also find that a machine text detector achieves the strongest performance among evaluated methods on both tasks, demonstrating the practical impact of transferability. To facilitate cross-task development and fair evaluation, we introduce MINT, a unified evaluation suite for MIAs and machine-generated text detection, implementing 15 recent methods from both tasks.
>
---
#### [replaced 008] ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs
- **分类: cs.LG; cs.CL; cs.DB**

- **简介: 该论文属于长文本生成任务，解决KV缓存检索的分布偏移和高延迟问题。提出ParisKV框架，通过碰撞选择和量化重排序提升效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.07721v2](https://arxiv.org/pdf/2602.07721v2)**

> **作者:** Yanlin Qi; Xinhang Chen; Huiqiang Jiang; Qitong Wang; Botao Peng; Themis Palpanas
>
> **备注:** 25 pages, 16 figures. Under review
>
> **摘要:** KV-cache retrieval is essential for long-context LLM inference, yet existing methods struggle with distribution drift and high latency at scale. We introduce ParisKV, a drift-robust, GPU-native KV-cache retrieval framework based on collision-based candidate selection, followed by a quantized inner-product reranking estimator. For million-token contexts, ParisKV supports CPU-offloaded KV caches via Unified Virtual Addressing (UVA), enabling on-demand top-$k$ fetching with minimal overhead. ParisKV matches or outperforms full attention quality on long-input and long-generation benchmarks. It achieves state-of-the-art long-context decoding efficiency: it matches or exceeds full attention speed even at batch size 1 for long contexts, delivers up to 2.8$\times$ higher throughput within full attention's runnable range, and scales to million-token contexts where full attention runs out of memory. At million-token scale, ParisKV reduces decode latency by 17$\times$ and 44$\times$ compared to MagicPIG and PQCache, respectively, two state-of-the-art KV-cache Top-$k$ retrieval baselines.
>
---
#### [replaced 009] Analyzing the Effects of Supervised Fine-Tuning on Model Knowledge from Token and Parameter Levels
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型优化任务，旨在研究监督微调对模型知识的影响。通过实验发现微调样本数量和知识水平影响模型性能，分析参数和token层面的更新效果，为优化微调策略提供依据。**

- **链接: [https://arxiv.org/pdf/2509.16596v2](https://arxiv.org/pdf/2509.16596v2)**

> **作者:** Junjie Ye; Yuming Yang; Yang Nan; Shuo Li; Qi Zhang; Tao Gui; Xuanjing Huang; Peng Wang; Zhongchao Shi; Jianping Fan
>
> **备注:** Accepted by EMNLP 2025 Main Conference. Codes for parameter restoration are available at https://github.com/UmeanNever/ParamRestore
>
> **摘要:** Large language models (LLMs) acquire substantial world knowledge during pre-training, which is further shaped by post-training techniques such as supervised fine-tuning (SFT). However, the impact of SFT on a model's knowledge remains underexplored, limiting our ability to control knowledge change behavior in fine-tuned models. To address this gap, we evaluate closed-book question answering (CBQA) performance across five LLMs from the LLaMA-2 and LLaMA-3 families. Surprisingly, models fine-tuned on 1,920 samples perform up to 14% worse than those fine-tuned on only 240 samples. Furthermore, varying the level of knowledge mastery in the fine-tuning data leads to performance fluctuations of over 12%. To investigate these effects, we analyze model behavior at both the token and parameter levels. Our analysis reveals that up to 90% of parameter updates during SFT do not contribute to knowledge enhancement. Restoring these updates can improve performance on the CBQA task, depending on the characteristics of the fine-tuning data. These insights offer practical guidance for developing fine-tuning strategies that more effectively strengthen model knowledge.
>
---
#### [replaced 010] Improving Data and Reward Design for Scientific Reasoning in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于科学推理任务，旨在提升大语言模型在开放性科学问题上的表现。通过构建数据集和优化训练流程，解决监督不可靠和评估困难的问题。**

- **链接: [https://arxiv.org/pdf/2602.08321v2](https://arxiv.org/pdf/2602.08321v2)**

> **作者:** Zijie Chen; Zhenghao Lin; Xiao Liu; Zhenzhong Lan; Yeyun Gong; Peng Cheng
>
> **摘要:** Solving open-ended science questions remains challenging for large language models, particularly due to inherently unreliable supervision and evaluation. The bottleneck lies in the data construction and reward design for scientific post-training. We develop a large-scale, systematic data processing pipeline that transforms heterogeneous open-source science data into Dr. SCI dataset, which comprises of 1M questions across eight STEM subjects, with explicit verifiable/open-ended splits, scalable difficulty annotation, and fine-grained rubrics that operationalize evaluation for open-ended answers. Building on this dataset, we propose the Dr. SCI post-training pipeline, which redesigns the standard SFT -> RL workflow through three components: (i) Exploration-Expanding SFT, which broadens the model's reasoning pattern coverage prior to RL; (ii) Dynamic Difficulty Curriculum, which adapts training data to the model's evolving scientific capability; and (iii) SciRubric-Guided RL, which enables stable reinforcement learning on open-ended scientific questions via rubric-based evaluation with explicit answer correctness. Qwen3-4B-Base trained using Dr. SCI pipeline achieves 63.2 on GPQA-diamond and 32.4 on GPQA-general, consistently improves over strong post-trained baselines such as o1-mini and GPT-4o, demonstrating substantial gains in scientific reasoning, especially in open-ended settings.
>
---
#### [replaced 011] MEGConformer: Conformer-Based MEG Decoder for Robust Speech and Phoneme Classification
- **分类: cs.CL; cs.LG; cs.NE; cs.SD**

- **简介: 该论文提出MEGConformer，用于从MEG信号中解码语音和音素信息，解决脑机接口中的信号解析问题。通过改进的Conformer结构和数据增强方法，提升了任务性能。**

- **链接: [https://arxiv.org/pdf/2512.01443v2](https://arxiv.org/pdf/2512.01443v2)**

> **作者:** Xabier de Zuazo; Ibon Saratxaga; Eva Navas
>
> **备注:** 8 pages, 7 figures, 4 tables, v1 presentend in LibriBrain Workshop, NeurIPS 2025; v2 submitted to Odyssey 2026
>
> **摘要:** Decoding speech-related information from non-invasive MEG is a key step toward scalable brain-computer interfaces. We present compact Conformer-based decoders on the LibriBrain 2025 PNPL benchmark for two core tasks: Speech Detection and Phoneme Classification. Our approach adapts a compact Conformer to raw 306-channel MEG signals, with a lightweight convolutional projection layer and task-specific heads. For Speech Detection, a MEG-oriented SpecAugment provided a first exploration of MEG-specific augmentation. For Phoneme Classification, we used inverse-square-root class weighting and a dynamic grouping loader to handle 100-sample averaged examples. In addition, a simple instance-level normalization proved critical to mitigate distribution shifts on the holdout split. Using the official Standard track splits and F1-macro for model selection, our best systems achieved 88.9% (Speech) and 65.8% (Phoneme) on the leaderboard, winning the Phoneme Classification Standard track. For further implementation details, the technical documentation, source code, and checkpoints are available at https://github.com/neural2speech/libribrain-experiments.
>
---
#### [replaced 012] A large-scale pipeline for automatic corpus annotation using LLMs: variation and change in the English consider construction
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决大规模语料自动标注问题。通过构建四阶段流程，利用大语言模型对语料进行高效准确标注，揭示语言变化趋势。**

- **链接: [https://arxiv.org/pdf/2510.12306v2](https://arxiv.org/pdf/2510.12306v2)**

> **作者:** Cameron Morin; Matti Marttinen Larsson
>
> **摘要:** As natural language corpora expand at an unprecedented rate, manual annotation remains a significant methodological bottleneck in corpus linguistic work. We address this challenge by presenting a scalable pipeline for automating grammatical annotation in voluminous corpora using large language models (LLMs). Unlike previous supervised and iterative approaches, our method employs a four-phase workflow: prompt engineering, pre-hoc evaluation, automated batch processing, and post-hoc validation. We demonstrate the pipeline's accessibility and effectiveness through a diachronic case study of variation in the English evaluative consider construction (consider X as/to be/zero Y). We annotate 143,933 'consider' concordance lines from the Corpus of Historical American English (COHA) via the OpenAI API in under 60 hours, achieving 98 percent+ accuracy on two sophisticated annotation procedures. A Bayesian multinomial GAM fitted to 44,527 true positives of the evaluative construction reveals previously undocumented genre-specific trajectories of change, enabling us to advance new hypotheses about the relationship between register formality and competing pressures of morphosyntactic reduction and enhancement. Our results suggest that LLMs can perform a range of data preparation tasks at scale with minimal human intervention, unlocking substantive research questions previously beyond practical reach, though implementation requires attention to costs, licensing, and other ethical considerations.
>
---
#### [replaced 013] Rethinking Memory Mechanisms of Foundation Agents in the Second Half: A Survey
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于人工智能领域，旨在解决基础智能体在长期动态环境中的记忆机制问题。通过分析记忆类型、结构及评估方法，提出统一视角以提升智能体的实用性能。**

- **链接: [https://arxiv.org/pdf/2602.06052v3](https://arxiv.org/pdf/2602.06052v3)**

> **作者:** Wei-Chieh Huang; Weizhi Zhang; Yueqing Liang; Yuanchen Bei; Yankai Chen; Tao Feng; Xinyu Pan; Zhen Tan; Yu Wang; Tianxin Wei; Shanglin Wu; Ruiyao Xu; Liangwei Yang; Rui Yang; Wooseong Yang; Chin-Yuan Yeh; Hanrong Zhang; Haozhen Zhang; Siqi Zhu; Henry Peng Zou; Wanjia Zhao; Song Wang; Wujiang Xu; Zixuan Ke; Zheng Hui; Dawei Li; Yaozu Wu; Langzhou He; Chen Wang; Xiongxiao Xu; Baixiang Huang; Juntao Tan; Shelby Heinecke; Huan Wang; Caiming Xiong; Ahmed A. Metwally; Jun Yan; Chen-Yu Lee; Hanqing Zeng; Yinglong Xia; Xiaokai Wei; Ali Payani; Yu Wang; Haitong Ma; Wenya Wang; Chenguang Wang; Yu Zhang; Xin Wang; Yongfeng Zhang; Jiaxuan You; Hanghang Tong; Xiao Luo; Xue Liu; Yizhou Sun; Wei Wang; Julian McAuley; James Zou; Jiawei Han; Philip S. Yu; Kai Shu
>
> **摘要:** The research of artificial intelligence is undergoing a paradigm shift from prioritizing model innovations over benchmark scores towards emphasizing problem definition and rigorous real-world evaluation. As the field enters the "second half," the central challenge becomes real utility in long-horizon, dynamic, and user-dependent environments, where agents face context explosion and must continuously accumulate, manage, and selectively reuse large volumes of information across extended interactions. Memory, with hundreds of papers released this year, therefore emerges as the critical solution to fill the utility gap. In this survey, we provide a unified view of foundation agent memory along three dimensions: memory substrate (internal and external), cognitive mechanism (episodic, semantic, sensory, working, and procedural), and memory subject (agent- and user-centric). We then analyze how memory is instantiated and operated under different agent topologies and highlight learning policies over memory operations. Finally, we review evaluation benchmarks and metrics for assessing memory utility, and outline various open challenges and future directions.
>
---
#### [replaced 014] Is the Reversal Curse a Binding Problem? Uncovering Limitations of Transformers from a Basic Generalization Failure
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理领域，旨在解决LLMs的Reversal Curse问题。通过分析Transformer的表征不一致和纠缠，提出基于JEPA的模型设计，提升概念绑定能力。**

- **链接: [https://arxiv.org/pdf/2504.01928v2](https://arxiv.org/pdf/2504.01928v2)**

> **作者:** Boshi Wang; Huan Sun
>
> **备注:** ICLR 2026
>
> **摘要:** Despite their impressive capabilities, LLMs exhibit a basic generalization failure known as the Reversal Curse, where they struggle to learn reversible factual associations. Understanding why this occurs could help identify weaknesses in current models and advance their generalization and robustness. In this paper, we conjecture that the Reversal Curse in LLMs is a manifestation of the long-standing binding problem in cognitive science, neuroscience and AI. Specifically, we hypothesize two primary causes of the Reversal Curse stemming from transformers' limitations in conceptual binding: the inconsistency and entanglements of concept representations. We perform a series of experiments that support these conjectures. Our exploration leads to a model design based on JEPA (Joint-Embedding Predictive Architecture) that for the first time breaks the Reversal Curse without side-stepping it with specialized data augmentation or non-causal masking, and moreover, generalization could be further improved by incorporating special memory layers that support disentangled concept representations. Our research opens up the broader fundamental challenge of designing models capable of learning systematic conceptual binding with less human scaffolding.
>
---
#### [replaced 015] A Large-Scale Dataset for Molecular Structure-Language Description via a Rule-Regularized Method
- **分类: cs.CL; cs.AI; q-bio.BM**

- **简介: 该论文属于分子结构与语言描述对齐任务，旨在解决人工标注成本高、数据量不足的问题。通过规则化方法自动生成大规模精准的分子描述数据集。**

- **链接: [https://arxiv.org/pdf/2602.02320v2](https://arxiv.org/pdf/2602.02320v2)**

> **作者:** Feiyang Cai; Guijuan He; Yi Hu; Jingjing Wang; Joshua Luo; Tianyu Zhu; Srikanth Pilla; Gang Li; Ling Liu; Feng Luo
>
> **摘要:** Molecular function is largely determined by structure. Accurately aligning molecular structure with natural language is therefore essential for enabling large language models (LLMs) to reason about downstream chemical tasks. However, the substantial cost of human annotation makes it infeasible to construct large-scale, high-quality datasets of structure-grounded descriptions. In this work, we propose a fully automated annotation framework for generating precise molecular structure descriptions at scale. Our approach builds upon and extends a rule-based chemical nomenclature parser to interpret IUPAC names and construct enriched, structured XML metadata that explicitly encodes molecular structure. This metadata is then used to guide LLMs in producing accurate natural-language descriptions. Using this framework, we curate a large-scale dataset of approximately $163$k molecule-description pairs. A rigorous validation protocol combining LLM-based and expert human evaluation on a subset of $2,000$ molecules demonstrates a high description precision of $98.6\%$. The resulting dataset provides a reliable foundation for future molecule-language alignment, and the proposed annotation method is readily extensible to larger datasets and broader chemical tasks that rely on structural descriptions.
>
---
#### [replaced 016] MAPS: A Multilingual Benchmark for Agent Performance and Security
- **分类: cs.DB; cs.CL; cs.CR**

- **简介: 该论文提出MAPS，一个用于评估多语言智能体性能与安全性的基准。解决多语言环境下智能体性能下降和安全性问题，通过翻译现有数据集构建多语言任务集。**

- **链接: [https://arxiv.org/pdf/2505.15935v3](https://arxiv.org/pdf/2505.15935v3)**

> **作者:** Omer Hofman; Jonathan Brokman; Oren Rachmil; Shamik Bose; Vikas Pahuja; Toshiya Shimizu; Trisha Starostina; Kelly Marchisio; Seraphina Goldfarb-Tarrant; Roman Vainshtein
>
> **备注:** Accepted to EACL 2026 findings
>
> **摘要:** Agentic AI systems, which build on Large Language Models (LLMs) and interact with tools and memory, have rapidly advanced in capability and scope. Yet, since LLMs have been shown to struggle in multilingual settings, typically resulting in lower performance and reduced safety, agentic systems risk inheriting these limitations. This raises concerns about the accessibility of such systems, as users interacting in languages other than English may encounter unreliable or security-critical agent behavior. Despite growing interest in evaluating agentic AI and recent initial efforts toward multilingual interaction, existing benchmarks do not yet provide a comprehensive, multi-domain, security-aware evaluation of multilingual agentic systems. To address this gap, we propose MAPS, a multilingual benchmark suite designed to evaluate agentic AI systems across diverse languages and tasks. MAPS builds on four widely used agentic benchmarks - GAIA (real-world tasks), SWE-Bench (code generation), MATH (mathematical reasoning), and the Agent Security Benchmark (security). We translate each dataset into eleven diverse languages, resulting in 805 unique tasks and 9,660 total language-specific instances - enabling a systematic analysis of the Multilingual Effect on AI agents' performance and robustness. Empirically, we observe a degradation in both performance and security when transitioning from English to other languages, with severity varying by task and correlating with the amount of translated input. This work establishes the first standardized evaluation framework for multilingual agentic AI, encouraging future research towards equitable, reliable, and accessible agentic AI. MAPS benchmark suite is publicly available at https://huggingface.co/datasets/Fujitsu-FRE/MAPS
>
---
#### [replaced 017] LIBMoE: A Library for comprehensive benchmarking Mixture of Experts in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出LibMoE，一个用于Mixture of Experts（MoE）研究的统一框架，解决MoE研究中计算成本高、可复现性差的问题，通过分析路由动态、初始化影响和训练模式，推动MoE技术发展。**

- **链接: [https://arxiv.org/pdf/2411.00918v4](https://arxiv.org/pdf/2411.00918v4)**

> **作者:** Nam V. Nguyen; Thong T. Doan; Luong Tran; Van Nguyen; Quang Pham
>
> **备注:** 40 pages
>
> **摘要:** Mixture of experts (MoE) architectures have become a cornerstone for scaling up and are a key component in most large language models such as GPT-OSS, DeepSeek-V3, Llama-4, and Gemini-2.5. However, systematic research on MoE remains severely constrained by the prohibitive computational costs of training and evaluation, restricting large-scale studies accessible to most researchers. We introduce LibMoE, a unified framework for reproducible, efficient, and extensible MoE research that supports both pretraining and sparse-upcycling regimes. Beyond unified implementations, the framework provides transparent analytical tools for probing routing and expert dynamics. Leveraging this foundation, we conduct a comprehensive analysis along three dimensions: (i) routing dynamics, covering expert selection patterns, routing stability and optimality, and how routing entropy reveals task specialization and expert diversity; (ii) the effect of lightweight initialization on load balancing, demonstrating how subtle changes in router initialization shape early expert utilization; and (iii) training regime differences, revealing how sparse upcycling and full pretraining exhibit distinct routing patterns and stability profiles. By lowering the barrier to entry and standardizing evaluation, along with our comprehensive analysis, LibMoE broadens access to MoE research and establishes a reliable benchmark to guide future innovations. GitHub: \href{https://github.com/Fsoft-AIC/LibMoE}{https://github.com/Fsoft-AIC/LibMoE}.
>
---
#### [replaced 018] Does Memory Need Graphs? A Unified Framework and Empirical Analysis for Long-Term Dialog Memory
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话记忆任务，旨在分析图结构在长期对话记忆中的有效性。通过统一框架对比不同设计，发现性能差异多由基础设置决定，而非架构创新。**

- **链接: [https://arxiv.org/pdf/2601.01280v3](https://arxiv.org/pdf/2601.01280v3)**

> **作者:** Sen Hu; Yuxiang Wei; Jiaxin Ran; Zhiyuan Yao; Xueran Han; Huacan Wang; Ronghao Chen; Lei Zou
>
> **摘要:** Graph structures are increasingly used in dialog memory systems, but empirical findings on their effectiveness remain inconsistent, making it unclear which design choices truly matter. We present an experimental, system-oriented analysis of long-term dialog memory architectures. We introduce a unified framework that decomposes dialog memory systems into core components and supports both graph-based and non-graph approaches. Under this framework, we conduct controlled, stage-wise experiments on LongMemEval and HaluMem, comparing common design choices in memory representation, organization, maintenance, and retrieval. Our results show that many performance differences are driven by foundational system settings rather than specific architectural innovations. Based on these findings, we identify stable and reliable strong baselines for future dialog memory research. Code are available at https://github.com/AvatarMemory/UnifiedMem
>
---
#### [replaced 019] What Should Feature Distillation Transfer in LLMs? A Task-Tangent Geometry View
- **分类: cs.CL**

- **简介: 该论文研究知识蒸馏任务，解决如何有效传递教师模型特征的问题。提出从功能几何角度出发，强调保留对输出影响大的特征方向，提出Flex-KD方法提升蒸馏效果。**

- **链接: [https://arxiv.org/pdf/2507.10155v3](https://arxiv.org/pdf/2507.10155v3)**

> **作者:** Khouloud Saadi; Di Wang
>
> **摘要:** Feature-based knowledge distillation aims to transfer intermediate representations from a teacher LLM model to a student. Existing approaches typically rely on direct feature matching or learned projections, implicitly treating representations as objects with intrinsic meaning. However, the relevance of a representation dimension is determined solely by how it affects the model's output. In this work, we propose a functional perspective on feature-based distillation. We characterize knowledge transfer in terms of the teacher's functional geometry, i.e., how its output depends on internal representations, rather than direct representation alignment. This viewpoint reveals that effective distillation need not preserve full high-dimensional features, but instead should retain dominant directions of functional contribution, naturally inducing an effective functional dimension for each task. Building on this framework, we introduce Flex-KD, an architecture-agnostic and parameter-free distillation method that transfers the teacher's functional geometry while matching the student's representational capacity. Extensive experiments across language understanding and generation benchmarks demonstrate that Flex-KD consistently outperforms existing distillation approaches, particularly under severe teacher-student dimension mismatch.
>
---
#### [replaced 020] THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出THOR，用于解决数学推理中高精度任务的挑战。通过工具集成和强化学习优化，提升模型的推理与代码生成能力。**

- **链接: [https://arxiv.org/pdf/2509.13761v3](https://arxiv.org/pdf/2509.13761v3)**

> **作者:** Qikai Chang; Zhenrong Zhang; Pengfei Hu; Jun Du; Jiefeng Ma; Yicheng Pan; Jianshu Zhang; Quan Liu; Jianqing Gao
>
> **备注:** 22 pages, 13 figures, ICLR 2026
>
> **摘要:** Large Language Models (LLMs) have made remarkable progress in mathematical reasoning, but still continue to struggle with high-precision tasks like numerical computation and formal symbolic manipulation. Integrating external tools has emerged as a promising approach to bridge this gap. Despite recent advances, existing methods struggle with three key challenges: constructing tool-integrated reasoning data, performing fine-grained optimization, and enhancing inference. To overcome these limitations, we propose THOR (Tool-Integrated Hierarchical Optimization via RL). First, we introduce TIRGen, a multi-agent based pipeline for constructing high-quality datasets of tool-integrated reasoning paths, aligning with the policy and generalizing well across diverse models. Second, to perform fine-grained hierarchical optimization, we introduce an RL strategy that jointly optimizes for both episode-level problem solving and step-level code generation. This is motivated by our key insight that the success of an intermediate tool call is a strong predictor of the final answer's correctness. Finally, THOR incorporates a self-correction mechanism that leverages immediate tool feedback to dynamically revise erroneous reasoning paths during inference. Our approach demonstrates strong generalization across diverse models, performing effectively in both reasoning and non-reasoning models. It further achieves state-of-the-art performance for models of a similar scale on multiple mathematical benchmarks, while also delivering consistent improvements on code benchmarks. Our code will be publicly available at https://github.com/JingMog/THOR.
>
---
#### [replaced 021] Online Density-Based Clustering for Real-Time Narrative Evolution Monitorin
- **分类: cs.CL**

- **简介: 该论文属于实时叙事监控任务，解决批量聚类方法在处理连续数据流时的可扩展性问题。通过引入在线密度聚类算法，提升系统实时性和适应性。**

- **链接: [https://arxiv.org/pdf/2601.20680v2](https://arxiv.org/pdf/2601.20680v2)**

> **作者:** Ostap Vykhopen; Viktoria Skorik; Maksym Tereshchenko; Veronika Solopova
>
> **摘要:** Automated narrative intelligence systems for social media monitoring face significant scalability challenges when relying on batch clustering methods to process continuous data streams. We investigate replacing offline HDBSCAN with online density-based clustering algorithms in a production narrative report generation pipeline that processes large volumes of multilingual social media data. While HDBSCAN effectively discovers hierarchical clusters and handles noise, its batch-only nature requires full retraining for each time window, limiting scalability and real-time adaptability. We evaluate online clustering methods with respect to cluster quality, computational efficiency, memory footprint, and integration with downstream narrative extraction. Our evaluation combines standard clustering metrics, narrative-specific measures, and human validation of cluster correctness to assess both structural quality and semantic interpretability. Experiments using sliding-window simulations on historical data from the Ukrainian information space reveal trade-offs between temporal stability and narrative coherence, with DenStream achieving the strongest overall performance. These findings bridge the gap between batch-oriented clustering approaches and the streaming requirements of large-scale narrative monitoring systems.
>
---
#### [replaced 022] SPARC: Separating Perception And Reasoning Circuits for Test-time Scaling of VLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出SPARC框架，解决视觉语言模型测试时扩展的稳定性问题。通过分离感知与推理模块，提升模型在复杂任务中的准确性和效率。**

- **链接: [https://arxiv.org/pdf/2602.06566v2](https://arxiv.org/pdf/2602.06566v2)**

> **作者:** Niccolo Avogaro; Nayanika Debnath; Li Mi; Thomas Frick; Junling Wang; Zexue He; Hang Hua; Konrad Schindler; Mattia Rigotti
>
> **摘要:** Despite recent successes, test-time scaling - i.e., dynamically expanding the token budget during inference as needed - remains brittle for vision-language models (VLMs): unstructured chains-of-thought about images entangle perception and reasoning, leading to long, disorganized contexts where small perceptual mistakes may cascade into completely wrong answers. Moreover, expensive reinforcement learning with hand-crafted rewards is required to achieve good performance. Here, we introduce SPARC (Separating Perception And Reasoning Circuits), a modular framework that explicitly decouples visual perception from reasoning. Inspired by sequential sensory-to-cognitive processing in the brain, SPARC implements a two-stage pipeline where the model first performs explicit visual search to localize question-relevant regions, then conditions its reasoning on those regions to produce the final answer. This separation enables independent test-time scaling with asymmetric compute allocation (e.g., prioritizing perceptual processing under distribution shift), supports selective optimization (e.g., improving the perceptual stage alone when it is the bottleneck for end-to-end performance), and accommodates compressed contexts by running global search at lower image resolutions and allocating high-resolution processing only to selected regions, thereby reducing total visual tokens count and compute. Across challenging visual reasoning benchmarks, SPARC outperforms monolithic baselines and strong visual-grounding approaches. For instance, SPARC improves the accuracy of Qwen3VL-4B on the $V^*$ VQA benchmark by 6.7 percentage points, and it surpasses "thinking with images" by 4.6 points on a challenging OOD task despite requiring a 200$\times$ lower token budget.
>
---
#### [replaced 023] Text2SQL-Flow: A Robust SQL-Aware Data Augmentation Framework for Text-to-SQL
- **分类: cs.CL; cs.DB**

- **简介: 该论文属于Text-to-SQL任务，旨在解决数据稀缺与多样性不足的问题。提出Text2SQL-Flow框架，生成高质量SQL对数据集SQLFlow，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.10192v4](https://arxiv.org/pdf/2511.10192v4)**

> **作者:** Qifeng Cai; Hao Liang; Chang Xu; Tao Xie; Wentao Zhang; Bin Cui
>
> **摘要:** The data-centric paradigm has emerged as a pivotal direction in artificial intelligence (AI), emphasizing the role of high-quality training data. This shift is especially critical in the Text-to-SQL task, where the scarcity, limited diversity, and structural simplicity of existing datasets constrain model performance. To address these challenges, we propose Text2SQL-Flow, a SQL-aware data augmentation framework that systematically generates large-scale, semantically valid, and structurally diverse Text-to-SQL pairs from limited seed data. Our framework spans six augmentation dimensions and integrates an end-to-end pipeline with auxiliary database selection, SQL executability verification, natural language (NL) question generation, NL-SQL correspondence verification, and chain-of-thought (CoT) reasoning trace generation. Leveraging this framework, we construct SQLFlow, a high-quality dataset comprising 75,386 annotated examples. We demonstrate the utility of SQLFlow in both fine-tuning and prompt-based settings. (1) For open-source large language models (LLMs), fine-tuning with SQLFlow improves problem-solving ability, delivering competitive gains across multiple benchmarks under the same data budget. (2) For closed-source LLMs, we propose a masked alignment retrieval method that uses SQLFlow as both a knowledge base and training data for the retrieval model, enabling structure-aware example matching via fine-grained NL-SQL alignments. Experiments show that our retrieval strategy outperforms existing example retrieval methods, highlighting the combined value of SQLFlow's data quality and our retrieval technique. Overall, our work provides a scalable, data-centric foundation for advancing Text-to-SQL systems and underscores the importance of structured, high-fidelity data in modern AI development. Our code is available at https://github.com/TechNomad-ds/Text2SQL-Flow.
>
---
#### [replaced 024] EAMET: Robust Massive Model Editing via Embedding Alignment Optimization
- **分类: cs.CL**

- **简介: 该论文属于模型编辑任务，旨在解决大规模知识更新中嵌入对齐不足导致的编辑效果下降问题。通过优化关键与残差嵌入空间，提升编辑可靠性。**

- **链接: [https://arxiv.org/pdf/2505.11876v3](https://arxiv.org/pdf/2505.11876v3)**

> **作者:** Yanbo Dai; Zhenlan Ji; Zongjie Li; Shuai Wang
>
> **备注:** This paper was accepted to ICLR 2026
>
> **摘要:** Model editing techniques are essential for efficiently updating knowledge in large language models (LLMs). However, the effectiveness of existing approaches degrades in massive editing scenarios, particularly when evaluated with practical metrics. Their robustness is also limited in context-rich settings or when editing multiple facts of the same subject simultaneously. We attribute these failures to the embedding misalignment among knowledge items, which undermines editing reliability at scale. To address this, we propose EAMET (Embedding Alignment Model Editing in Transformers), which addresses this issue by aligning the space of key and residual embeddings. Extensive experiments across six LLMs and three datasets demonstrate that EAMET consistently outperforms existing methods, achieving about 90\% editing efficacy when editing 10k facts. Codes and datasets are publicly available at https://ybdai7.github.io/eamet-page/.
>
---
#### [replaced 025] Emergent Structured Representations Support Flexible In-Context Inference in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文研究大语言模型中的结构化表示是否支持上下文推理。任务是理解模型如何进行灵活推理。工作包括分析内部处理过程，发现中间层出现的语义子空间，并验证其对预测的因果作用。**

- **链接: [https://arxiv.org/pdf/2602.07794v2](https://arxiv.org/pdf/2602.07794v2)**

> **作者:** Ningyu Xu; Qi Zhang; Xipeng Qiu; Xuanjing Huang
>
> **备注:** 27 pages, 16 figures
>
> **摘要:** Large language models (LLMs) exhibit emergent behaviors suggestive of human-like reasoning. While recent work has identified structured, human-like conceptual representations within these models, it remains unclear whether they functionally rely on such representations for reasoning. Here we investigate the internal processing of LLMs during in-context concept inference. Our results reveal a conceptual subspace emerging in middle to late layers, whose representational structure persists across contexts. Using causal mediation analyses, we demonstrate that this subspace is not merely an epiphenomenon but is functionally central to model predictions, establishing its causal role in inference. We further identify a layer-wise progression where attention heads in early-to-middle layers integrate contextual cues to construct and refine the subspace, which is subsequently leveraged by later layers to generate predictions. Together, these findings provide evidence that LLMs dynamically construct and use structured, latent representations in context for inference, offering insights into the computational processes underlying flexible adaptation.
>
---
#### [replaced 026] TOPol: Capturing and Explaining Multidimensional Semantic Polarity Fields and Vectors
- **分类: cs.CL**

- **简介: 该论文提出TOPol框架，用于捕捉和解释多维语义极性场，解决传统单维度情感分析的不足。通过结合语言模型与聚类方法，实现对话语极性变化的精细分析。**

- **链接: [https://arxiv.org/pdf/2510.25069v2](https://arxiv.org/pdf/2510.25069v2)**

> **作者:** Gabin Taibi; Lucia Gomez
>
> **备注:** 7 pages, 3 figures and 2 tables
>
> **摘要:** Traditional approaches to semantic polarity in computational linguistics treat sentiment as a unidimensional scale, overlooking the multidimensional structure of language. This work introduces TOPol (Topic-Orientation POLarity), a semi-unsupervised framework for reconstructing and interpreting multidimensional narrative polarity fields under human-on-the-loop (HoTL) defined contextual boundaries (CBs). The framework embeds documents using a transformer-based large language model (tLLM), applies neighbor-tuned UMAP projection, and segments topics via Leiden partitioning. Given a CB between discourse regimes A and B, TOPol computes directional vectors between corresponding topic-boundary centroids, yielding a polarity field that quantifies fine-grained semantic displacement during regime shifts. This vectorial representation enables assessing CB quality and detecting polarity changes, guiding HoTL CB refinement. To interpret identified polarity vectors, the tLLM compares their extreme points and produces contrastive labels with estimated coverage. Robustness analyses show that only CB definitions (the main HoTL-tunable parameter) significantly affect results, confirming methodological stability. We evaluate TOPol on two corpora: (i) U.S. Central Bank speeches around a macroeconomic breakpoint, capturing non-affective semantic shifts, and (ii) Amazon product reviews across rating strata, where affective polarity aligns with NRC valence. Results demonstrate that TOPol consistently captures both affective and non-affective polarity transitions, providing a scalable, generalizable, and interpretable framework for context-sensitive multidimensional discourse analysis.
>
---
#### [replaced 027] The Condensate Theorem: Transformers are O(n), Not $O(n^2)$
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Condensate定理，解决Transformer的O(n²)计算瓶颈问题。通过发现注意力集中在特定拓扑结构上，实现高效注意力计算，显著提升推理速度。**

- **链接: [https://arxiv.org/pdf/2602.06317v2](https://arxiv.org/pdf/2602.06317v2)**

> **作者:** Jorge L. Ruiz Williams
>
> **备注:** 13 pages, 4 figures, 8 tables, 1 pseudocode algorithm
>
> **摘要:** We present the Condensate Theorem: attention sparsity is a learned topological property, not an architectural constraint. Through empirical analysis of trained language models, we find that attention mass concentrates on a distinct topological manifold -- and this manifold can be identified dynamically without checking every position. We prove a general result: for any query, projecting attention onto the Condensate Manifold (Anchor + Window + Dynamic Top-k) achieves 100% output equivalence with full $O(n^2)$ attention. This is not an approximation -- it is lossless parity. We validate this across GPT-2, Pythia, Qwen2, TinyLlama, and Mistral, demonstrating bit-exact token matching on 1,500+ generated tokens. By mapping this topology to hardware, our Topological Attention kernel achieves a 159x measured speedup at 131K tokens (3.94ms vs 628ms) and a projected >1,200x speedup at 1M tokens, reducing inference costs by >99.9% compared to Flash Attention. We conclude that the quadratic bottleneck is an artifact of naive implementation, not intelligence.
>
---
#### [replaced 028] Common Objects Out of Context (COOCo): Investigating Multimodal Context and Semantic Scene Violations in Referential Communication
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉语言模型（VLM）在参考生成中如何利用场景上下文，解决对象参照依赖场景的问题。通过构建COOCo数据集并分析模型表现，揭示了模型对上下文的动态适应机制。**

- **链接: [https://arxiv.org/pdf/2506.22274v2](https://arxiv.org/pdf/2506.22274v2)**

> **作者:** Filippo Merlo; Ece Takmaz; Wenkai Chen; Albert Gatt
>
> **备注:** Accepted to TACL (pre-MIT Press publication version)
>
> **摘要:** To what degree and under what conditions do VLMs rely on scene context when generating references to objects? To address this question, we introduce the $\textit{Common Objects Out-of-Context (COOCo)}$ dataset and conduct experiments on several VLMs under different degrees of scene-object congruency and noise. We find that models leverage scene context adaptively, depending on scene-object semantic relatedness and noise level. Based on these consistent trends across models, we turn to the question of how VLM attention patterns change as a function of target-scene semantic fit, and to what degree these patterns are predictive of categorisation accuracy. We find that successful object categorisation is associated with increased mid-layer attention to the target. We also find a non-monotonic dependency on semantic fit, with attention dropping at moderate fit and increasing for both low and high fit. These results suggest that VLMs dynamically balance local and contextual information for reference generation. Dataset and code are available here: $\href{https://github.com/cs-nlp-uu/scenereg}{https://github.com/cs-nlp-uu/scenereg}$.
>
---
#### [replaced 029] SAGE: An Agentic Explainer Framework for Interpreting SAE Features in Language Models
- **分类: cs.CL**

- **简介: 该论文提出SAGE框架，用于解释语言模型中SAE提取的特征。针对特征解释不准确的问题，通过主动迭代方法提升解释的准确性。属于模型解释任务。**

- **链接: [https://arxiv.org/pdf/2511.20820v2](https://arxiv.org/pdf/2511.20820v2)**

> **作者:** Jiaojiao Han; Wujiang Xu; Mingyu Jin; Mengnan Du
>
> **备注:** EACL 2026 Industry Track
>
> **摘要:** Large language models (LLMs) have achieved remarkable progress, yet their internal mechanisms remain largely opaque, posing a significant challenge to their safe and reliable deployment. Sparse autoencoders (SAEs) have emerged as a promising tool for decomposing LLM representations into more interpretable features, but explaining the features captured by SAEs remains a challenging task. In this work, we propose SAGE (SAE AGentic Explainer), an agent-based framework that recasts feature interpretation from a passive, single-pass generation task into an active, explanation-driven process. SAGE implements a rigorous methodology by systematically formulating multiple explanations for each feature, designing targeted experiments to test them, and iteratively refining explanations based on empirical activation feedback. Experiments on features from SAEs of diverse language models demonstrate that SAGE produces explanations with significantly higher generative and predictive accuracy compared to state-of-the-art baselines.an agent-based framework that recasts feature interpretation from a passive, single-pass generation task into an active, explanationdriven process. SAGE implements a rigorous methodology by systematically formulating multiple explanations for each feature, designing targeted experiments to test them, and iteratively refining explanations based on empirical activation feedback. Experiments on features from SAEs of diverse language models demonstrate that SAGE produces explanations with significantly higher generative and predictive accuracy compared to state-of-the-art baselines.
>
---
#### [replaced 030] Learning Tractable Distributions Of Language Model Continuations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型生成任务，解决自回归模型在条件生成中的计算不可行问题。提出LTLA方法，结合HMM与神经网络，提升生成效率与控制能力。**

- **链接: [https://arxiv.org/pdf/2511.16054v2](https://arxiv.org/pdf/2511.16054v2)**

> **作者:** Gwen Yidou-Weng; Ian Li; Anji Liu; Oliver Broadrick; Yuchen Cui; Guy Van den Broeck; Benjie Wang
>
> **摘要:** Controlled generation imposes sequence-level constraints (syntax, style, safety) that depend on future tokens, making exact conditioning of an autoregressive LM intractable. Tractable surrogates such as HMMs can approximate continuation distributions and steer decoding, but standard surrogates are often weakly context-aware. We propose Learning to Look Ahead (LTLA), a hybrid method that uses base-LM embeddings to condition a globally learned tractable surrogate: a neural head predicts only a prefix-dependent latent prior, while a shared HMM answers continuation queries exactly. LTLA is designed to avoid two common efficiency traps when adding neural context. First, it avoids vocabulary-sized prefix rescoring (V extra LM evaluations) by scoring all next-token candidates via a single batched HMM forward update. Second, it avoids predicting a new HMM per prefix by learning one shared HMM and conditioning only the latent prior, which enables reuse of cached future-likelihood (backward) messages across decoding steps. Empirically, LTLA improves continuation likelihood over standard HMM surrogates, enables lookahead control for vision--language models by incorporating continuous context, achieves 100% syntactic constraint satisfaction, and improves detoxification while adding only a 14% decoding-time overhead.
>
---
#### [replaced 031] Sri Lanka Document Datasets: A Large-Scale, Multilingual Resource for Law, News, and Policy
- **分类: cs.CL**

- **简介: 该论文介绍了一个多语言、大规模的斯里兰卡文档数据集，涵盖法律、新闻和政策等领域，用于自然语言处理和相关研究。**

- **链接: [https://arxiv.org/pdf/2510.04124v5](https://arxiv.org/pdf/2510.04124v5)**

> **作者:** Nuwan I. Senaratna
>
> **备注:** 4 pages. 253,817 documents (72.2 GB) across 26 datasets in Sinhala, Tamil, and English. Last updated on 2026-02-10 (10:51am)
>
> **摘要:** We present a collection of open, machine-readable document datasets covering parliamentary proceedings, legal judgments, government publications, news, and tourism statistics from Sri Lanka. The collection currently comprises of 253,817 documents (72.2 GB) across 26 datasets in Sinhala, Tamil, and English. The datasets are updated daily and mirrored on GitHub and Hugging Face. These resources aim to support research in computational linguistics, legal analytics, socio-political studies, and multilingual natural language processing. We describe the data sources, collection pipeline, formats, and potential use cases, while discussing licensing and ethical considerations. This manuscript is at version v2026-02-10-1051.
>
---
#### [replaced 032] Free(): Learning to Forget in Malloc-Only Reasoning Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于大模型推理任务，解决因冗余信息导致性能下降的问题。提出Free()LM，通过自遗忘机制动态清理无效上下文，提升模型表现。**

- **链接: [https://arxiv.org/pdf/2602.08030v2](https://arxiv.org/pdf/2602.08030v2)**

> **作者:** Yilun Zheng; Dongyang Ma; Tian Liang; Jiahao Xu; Xinting Huang; Lihui Chen; Haitao Mi; Yan Wang
>
> **摘要:** Reasoning models enhance problem-solving by scaling test-time compute, yet they face a critical paradox: excessive thinking tokens often degrade performance rather than improve it. We attribute this to a fundamental architectural flaw: standard LLMs operate as "malloc-only" engines, continuously accumulating valid and redundant steps alike without a mechanism to prune obsolete information. To break this cycle, we propose Free()LM, a model that introduces an intrinsic self-forgetting capability via the Free-Module, a plug-and-play LoRA adapter. By iteratively switching between reasoning and cleaning modes, Free()LM dynamically identifies and prunes useless context chunks, maintaining a compact and noise-free state. Extensive experiments show that Free()LM provides consistent improvements across all model scales (8B to 685B). It achieves a 3.3% average improvement over top-tier reasoning baselines, even establishing a new SOTA on IMOanswerBench using DeepSeek V3.2-Speciale. Most notably, in long-horizon tasks where the standard Qwen3-235B-A22B model suffers a total collapse (0% accuracy), Free()LM restores performance to 50%. Our findings suggest that sustainable intelligence requires the freedom to forget as much as the power to think.
>
---
#### [replaced 033] Survey of Video Diffusion Models: Foundations, Implementations, and Applications
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视频生成任务，旨在综述扩散模型在视频生成中的应用，解决运动一致性、效率及伦理等问题，系统分析方法、架构与应用。**

- **链接: [https://arxiv.org/pdf/2504.16081v3](https://arxiv.org/pdf/2504.16081v3)**

> **作者:** Yimu Wang; Xuye Liu; Wei Pang; Li Ma; Shuai Yuan; Paul Debevec; Ning Yu
>
> **备注:** Accepted by TMLR
>
> **摘要:** Recent advances in diffusion models have revolutionized video generation, offering superior temporal consistency and visual quality compared to traditional generative adversarial networks-based approaches. While this emerging field shows tremendous promise in applications, it faces significant challenges in motion consistency, computational efficiency, and ethical considerations. This survey provides a comprehensive review of diffusion-based video generation, examining its evolution, technical foundations, and practical applications. We present a systematic taxonomy of current methodologies, analyze architectural innovations and optimization strategies, and investigate applications across low-level vision tasks such as denoising and super-resolution. Additionally, we explore the synergies between diffusionbased video generation and related domains, including video representation learning, question answering, and retrieval. Compared to the existing surveys (Lei et al., 2024a;b; Melnik et al., 2024; Cao et al., 2023; Xing et al., 2024c) which focus on specific aspects of video generation, such as human video synthesis (Lei et al., 2024a) or long-form content generation (Lei et al., 2024b), our work provides a broader, more updated, and more fine-grained perspective on diffusion-based approaches with a special section for evaluation metrics, industry solutions, and training engineering techniques in video generation. This survey serves as a foundational resource for researchers and practitioners working at the intersection of diffusion models and video generation, providing insights into both the theoretical frameworks and practical implementations that drive this rapidly evolving field. A structured list of related works involved in this survey is also available on https://github.com/Eyeline-Research/Survey-Video-Diffusion.
>
---
#### [replaced 034] Reward-free Alignment for Conflicting Objectives
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于多目标对齐任务，解决冲突目标下模型训练不稳定问题。提出RACO框架，无需奖励模型，通过梯度裁剪实现有效对齐。**

- **链接: [https://arxiv.org/pdf/2602.02495v2](https://arxiv.org/pdf/2602.02495v2)**

> **作者:** Peter L. Chen; Xiaopeng Li; Xi Chen; Tianyi Lin
>
> **备注:** 27 pages
>
> **摘要:** Direct alignment methods are increasingly used to align large language models (LLMs) with human preferences. However, many real-world alignment problems involve multiple conflicting objectives, where naive aggregation of preferences can lead to unstable training and poor trade-offs. In particular, weighted loss methods may fail to identify update directions that simultaneously improve all objectives, and existing multi-objective approaches often rely on explicit reward models, introducing additional complexity and distorting user-specified preferences. The contributions of this paper are two-fold. First, we propose a Reward-free Alignment framework for Conflicted Objectives (RACO) that directly leverages pairwise preference data and resolves gradient conflicts via a novel clipped variant of conflict-averse gradient descent. We provide convergence guarantees to Pareto-critical points that respect user-specified objective weights, and further show that clipping can strictly improve convergence rate in the two-objective setting. Second, we improve our method using some heuristics and conduct experiments to demonstrate the compatibility of the proposed framework for LLM alignment. Both qualitative and quantitative evaluations on multi-objective summarization and safety alignment tasks across multiple LLM families (Qwen 3, Llama 3, Gemma 3) show that our method consistently achieves better Pareto trade-offs compared to existing multi-objective alignment baselines.
>
---
#### [replaced 035] Agentic Jigsaw Interaction Learning for Enhancing Visual Perception and Reasoning in Vision-Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出AGILE方法，解决VLM在视觉感知和推理上的不足。通过交互式拼图学习，提升模型能力，实验显示显著性能提升。**

- **链接: [https://arxiv.org/pdf/2510.01304v2](https://arxiv.org/pdf/2510.01304v2)**

> **作者:** Yu Zeng; Wenxuan Huang; Shiting Huang; Xikun Bao; Yukun Qi; Yiming Zhao; Qiuchen Wang; Lin Chen; Zehui Chen; Huaian Chen; Wanli Ouyang; Feng Zhao
>
> **摘要:** Although current large Vision-Language Models (VLMs) have advanced in multimodal understanding and reasoning, their fundamental perceptual and reasoning abilities remain limited. Specifically, even on simple jigsaw tasks, existing VLMs perform near randomly, revealing deficiencies in core perception and reasoning capabilities. While high-quality vision-language data can enhance these capabilities, its scarcity and limited scalability impose significant constraints. To address this, we propose AGILE, an Agentic jiGsaw Interaction Learning for Enhancing visual perception and reasoning in VLMs. AGILE formulates jigsaw solving as an interactive process, enabling the model to progressively engage with the environment. At each step, the model generates executable code to perform an action based on the current state, while the environment provides fine-grained visual feedback to guide task completion. Through this iterative cycle of observation and interaction, the model incrementally improves its perceptual and reasoning capabilities via exploration and feedback. Experimental results show that AGILE not only substantially boosts performance on jigsaw tasks of varying complexity (e.g., increasing accuracy from 9.5% to 82.8% under the 2 $\times$ 2 setting) but also demonstrates strong generalization across 9 general vision tasks, achieving an average improvement of 3.1%. These results indicate notable enhancements in both perceptual and reasoning abilities. This work opens a new avenue for advancing reasoning and generalization in multimodal models and provides an efficient, scalable solution to the scarcity of multimodal reinforcement learning data. The code and datasets is available at https://github.com/yuzeng0-0/AGILE .
>
---
#### [replaced 036] MolLangBench: A Comprehensive Benchmark for Language-Prompted Molecular Structure Recognition, Editing, and Generation
- **分类: cs.CL; cs.AI; cs.LG; q-bio.BM**

- **简介: 该论文提出MolLangBench，用于评估分子结构的识别、编辑和生成任务。旨在解决AI在化学领域处理分子语言接口的问题，通过高质量数据集推动更有效的AI系统发展。**

- **链接: [https://arxiv.org/pdf/2505.15054v3](https://arxiv.org/pdf/2505.15054v3)**

> **作者:** Feiyang Cai; Jiahui Bai; Tao Tang; Guijuan He; Joshua Luo; Tianyu Zhu; Srikanth Pilla; Gang Li; Ling Liu; Feng Luo
>
> **备注:** ICLR-2026 Camera-Ready version
>
> **摘要:** Precise recognition, editing, and generation of molecules are essential prerequisites for both chemists and AI systems tackling various chemical tasks. We present MolLangBench, a comprehensive benchmark designed to evaluate fundamental molecule-language interface tasks: language-prompted molecular structure recognition, editing, and generation. To ensure high-quality, unambiguous, and deterministic outputs, we construct the recognition tasks using automated cheminformatics tools, and curate editing and generation tasks through rigorous expert annotation and validation. MolLangBench supports the evaluation of models that interface language with different molecular representations, including linear strings, molecular images, and molecular graphs. Evaluations of state-of-the-art models reveal significant limitations: the strongest model (GPT-5) achieves $86.2\%$ and $85.5\%$ accuracy on recognition and editing tasks, which are intuitively simple for humans, and performs even worse on the generation task, reaching only $43.0\%$ accuracy. These results highlight the shortcomings of current AI systems in handling even preliminary molecular recognition and manipulation tasks. We hope MolLangBench will catalyze further research toward more effective and reliable AI systems for chemical applications.The dataset and code can be accessed at https://huggingface.co/datasets/ChemFM/MolLangBench and https://github.com/TheLuoFengLab/MolLangBench, respectively.
>
---
#### [replaced 037] IMAGINE: Integrating Multi-Agent System into One Model for Complex Reasoning and Planning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出IMAGINE框架，将多智能体系统整合到单一模型中，解决复杂推理与规划任务中的效率和性能问题。**

- **链接: [https://arxiv.org/pdf/2510.14406v2](https://arxiv.org/pdf/2510.14406v2)**

> **作者:** Xikai Zhang; Bo Wang; Likang Xiao; Yongzhi Li; Quan Chen; Wenjun Wu; Liu Liu
>
> **摘要:** Although large language models (LLMs) have made significant strides across various tasks, they still face significant challenges in complex reasoning and planning. For example, even with carefully designed prompts and prior information explicitly provided, GPT-4o achieves only a 7% Final Pass Rate on the TravelPlanner dataset in the sole-planning mode. Similarly, even in the thinking mode, Qwen3-8B-Instruct and DeepSeek-R1-671B, only achieve Final Pass Rates of 5.9% and 40%, respectively. Although well-organized Multi-Agent Systems (MAS) can offer improved collective reasoning, they often suffer from high reasoning costs due to multi-round internal interactions, long per-response latency, and difficulties in end-to-end training. To address these challenges, we propose a general and scalable framework called IMAGINE, short for Integrating Multi-Agent System into One Model. This framework not only integrates the reasoning and planning capabilities of MAS into a single, compact model, but also significantly surpass the capabilities of the MAS through a simple end-to-end training. Through this pipeline, a single small-scale model is not only able to acquire the structured reasoning and planning capabilities of a well-organized MAS but can also significantly outperform it. Experimental results demonstrate that, when using Qwen3-8B-Instruct as the base model and training it with our method, the model achieves an 82.7% Final Pass Rate on the TravelPlanner benchmark, far exceeding the 40% of DeepSeek-R1-671B, while maintaining a much smaller model size.
>
---
#### [replaced 038] Distribution-Aligned Decoding for Efficient LLM Task Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型任务适应领域，旨在解决高效微调问题。通过输出分布对齐，提出SVDecode方法，在不增加参数的情况下提升模型性能。**

- **链接: [https://arxiv.org/pdf/2509.15888v4](https://arxiv.org/pdf/2509.15888v4)**

> **作者:** Senkang Hu; Xudong Han; Jinqi Jiang; Yihang Tao; Zihan Fang; Yong Dai; Sam Tak Wu Kwong; Yuguang Fang
>
> **备注:** Accepted by NeurIPS'25
>
> **摘要:** Adapting billion-parameter language models to a downstream task is still costly, even with parameter-efficient fine-tuning (PEFT). We re-cast task adaptation as output-distribution alignment: the objective is to steer the output distribution toward the task distribution directly during decoding rather than indirectly through weight updates. Building on this view, we introduce Steering Vector Decoding (SVDecode), a lightweight, PEFT-compatible, and theoretically grounded method. We start with a short warm-start fine-tune and extract a task-aware steering vector from the Kullback-Leibler (KL) divergence gradient between the output distribution of the warm-started and pre-trained models. This steering vector is then used to guide the decoding process to steer the model's output distribution towards the task distribution. We theoretically prove that SVDecode is first-order equivalent to the gradient step of full fine-tuning and derive a globally optimal solution for the strength of the steering vector. Across three tasks and nine benchmarks, SVDecode paired with four standard PEFT methods improves multiple-choice accuracy by up to 5 percentage points and open-ended truthfulness by 2 percentage points, with similar gains (1-2 percentage points) on commonsense datasets without adding trainable parameters beyond the PEFT adapter. SVDecode thus offers a lightweight, theoretically grounded path to stronger task adaptation for large language models. Code is available at https://github.com/dl-m9/SVDecode.
>
---
#### [replaced 039] Subject islands do not reduce to construction-specific discourse function
- **分类: cs.CL**

- **简介: 该论文属于语法研究任务，旨在解决“主语是否为岛屿结构”的问题。通过实验验证不同句式中主语是否产生岛屿效应，结果支持语法结构独立于语用功能的解释。**

- **链接: [https://arxiv.org/pdf/2504.15688v2](https://arxiv.org/pdf/2504.15688v2)**

> **作者:** Mandy Cartner; Matthew Kogan; Nikolas Webster; Matthew Wagers; Ivy Sichel
>
> **摘要:** The term islands in linguistics refers to phrases from which extracting an element results in ungrammaticality (Ross, 1967). Grammatical subjects are considered islands because extracting a sub-part of a subject results in an ill-formed sentence, despite having a clear intended meaning (e.g., "Which topic did the article about inspire you?"). The generative tradition, which views syntax as autonomous of meaning and function, attributes this ungrammaticality to the abstract movement dependency between the wh-phrase and the subject-internal position with which it is associated for interpretation. However, research on language that emphasizes its communicative function suggests instead that syntactic constraints, including islands, can be explained based on the way different constructions package information. Accordingly, Abeillé et al. (2020) suggest that the islandhood of subjects is specific to the information structure of wh-questions, and propose that subjects are not islands for movement, but for focusing, due to their discourse-backgroundedness. This predicts that other constructions that differ in their information structure from wh-questions, but still involve movement, should not create a subject island effect. We test this prediction in three large-scale acceptability studies, using a super-additive design that singles out subject island violations, in three different constructions: wh-questions, relative clauses, and topicalization. We report evidence for a subject island effect in each construction type, despite only wh-questions introducing what Abeillé et al. (2020) call "a clash in information structure." We argue that this motivates an account of islands in terms of abstract, syntactic representations, independent of the communicative function associated with the constructions.
>
---
#### [replaced 040] Nudging the Boundaries of LLM Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM无法从难问题中学习的问题。通过生成提示提升模型推理上限，提升训练效果。**

- **链接: [https://arxiv.org/pdf/2509.25666v2](https://arxiv.org/pdf/2509.25666v2)**

> **作者:** Justin Chih-Yao Chen; Becky Xiangyu Peng; Prafulla Kumar Choubey; Kung-Hsiang Huang; Jiaxin Zhang; Mohit Bansal; Chien-Sheng Wu
>
> **备注:** ICLR 2026 (Camera-Ready)
>
> **摘要:** Current online reinforcement learning (RL) algorithms like GRPO share a key limitation in LLM reasoning: they cannot learn from problems that are "unsolvable" to the model. In other words, they can only improve performance on problems where the model is capable of exploring the correct answer. Consequently, the model's "upper limit" remains unchanged after RL training, even though the likelihood of solving easier, solvable problems may increase. These hard samples cannot contribute to training, as no rollouts yield rewards and thus no gradients are produced. To unlock learning from these hard samples, we propose NuRL, a "nudging" method that aims to push the upper bound of LLM reasoning using self-generated hints, i.e., abstract cues that help reduce the problem difficulty for the model. Given a question and its gold answer, the model generates a CoT and then produces a hint containing the core knowledge needed to solve the problem. During training, we generate G rollouts from the base policy and use the pass rate to decide whether the hint should be injected. For hard samples with a 0% pass rate, we inject the hint and regenerate a new batch of trajectories. This yields two benefits: (1) the hint boosts pass rates (from 0% to non-zero), thereby introducing training signals for previously unsolvable samples, and (2) the hints are self-generated, avoiding distributional shift and do not rely on external models. NuRL achieves consistent improvements across 6 benchmarks and 3 models, while remaining complementary to test-time scaling. Notably, NuRL can raise the model's upper limit, whereas GRPO leaves pass@1024 unchanged from the base model. Furthermore, we present a systematic study of what makes an effective hint and when hints are most useful. Interestingly, the best hints are abstract and high-level, and are most beneficial when applied necessarily and after GRPO has converged.
>
---
#### [replaced 041] Truth with a Twist: The Rhetoric of Persuasion in Professional vs. Community-Authored Fact-Checks
- **分类: cs.CL**

- **简介: 该论文属于事实核查研究，比较专业与社区撰写的内容在说服技巧上的差异，分析其效果与评价。**

- **链接: [https://arxiv.org/pdf/2601.14105v3](https://arxiv.org/pdf/2601.14105v3)**

> **作者:** Olesya Razuvayevskaya; Kalina Bontcheva
>
> **备注:** In Proceedings of the ACM Web Conference 2026 (WWW 2026)
>
> **摘要:** This study presents the first large-scale comparison of persuasion techniques present in crowd- versus professionally-written debunks. Using extensive datasets from Community Notes (CNs), EUvsDisinfo, and the Database of Known Fakes (DBKF), we quantify the prevalence and types of persuasion techniques across these fact-checking ecosystems. Contrary to prior hypothesis that community-produced debunks rely more heavily on subjective or persuasive wording, we find no evidence that CNs contain a higher average number of persuasion techniques than professional fact-checks. We additionally identify systematic rhetorical differences between CNs and professional debunking efforts, reflecting differences in institutional norms and topical coverage. Finally, we examine how the crowd evaluates persuasive language in CNs and show that, although notes with more persuasive elements receive slightly higher overall helpfulness ratings, crowd raters are effective at penalising the use of particular problematic rhetorical means
>
---
#### [replaced 042] Advancing General-Purpose Reasoning Models with Modular Gradient Surgery
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于机器学习任务，旨在解决多领域强化学习中的跨域干扰问题。通过引入模块化梯度手术（MGS），提升通用推理模型的性能。**

- **链接: [https://arxiv.org/pdf/2602.02301v2](https://arxiv.org/pdf/2602.02301v2)**

> **作者:** Min Cai; Yu Liang; Longzheng Wang; Yan Wang; Yueyang Zhang; Long Xia; Zhiyuan Sun; Xi Ye; Daiting Shi
>
> **备注:** Preprint; Code: https://github.com/StringNLPLAB/MGS Website: https://modular-gradient-surgery.github.io
>
> **摘要:** Reinforcement learning (RL) has played a central role in recent advances in large reasoning models (LRMs), yielding strong gains in verifiable and open-ended reasoning. However, training a single general-purpose LRM across diverse domains remains challenging due to pronounced domain heterogeneity. Through a systematic study of two widely used strategies, Sequential RL and Mixed RL, we find that both incur substantial cross-domain interference at the behavioral and gradient levels, resulting in limited overall gains. To address these challenges, we introduce **M**odular **G**radient **S**urgery (**MGS**), which resolves gradient conflicts at the module level within the transformer. When applied to Llama and Qwen models, MGS achieves average improvements of 4.3 (16.6\%) and 4.5 (11.1\%) points, respectively, over standard multi-task RL across three representative domains (math, general chat, and instruction following). Further analysis demonstrates that MGS remains effective under prolonged training. Overall, our study clarifies the sources of interference in multi-domain RL and presents an effective solution for training general-purpose LRMs.
>
---
#### [replaced 043] A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion Language Models
- **分类: cs.CL; cs.AI; cs.DC**

- **简介: 该论文属于文本生成任务，旨在解决传统自回归生成速度慢的问题。通过系统调研并分类分析并行文本生成方法，提出改进推理效率的思路。**

- **链接: [https://arxiv.org/pdf/2508.08712v4](https://arxiv.org/pdf/2508.08712v4)**

> **作者:** Lingzhe Zhang; Liancheng Fang; Chiming Duan; Minghua He; Leyi Pan; Pei Xiao; Shiyu Huang; Yunpeng Zhai; Xuming Hu; Philip S. Yu; Aiwei Liu
>
> **摘要:** As text generation has become a core capability of modern Large Language Models (LLMs), it underpins a wide range of downstream applications. However, most existing LLMs rely on autoregressive (AR) generation, producing one token at a time based on previously generated context-resulting in limited generation speed due to the inherently sequential nature of the process. To address this challenge, an increasing number of researchers have begun exploring parallel text generation-a broad class of techniques aimed at breaking the token-by-token generation bottleneck and improving inference efficiency. Despite growing interest, there remains a lack of comprehensive analysis on what specific techniques constitute parallel text generation and how they improve inference performance. To bridge this gap, we present a systematic survey of parallel text generation methods. We categorize existing approaches into AR-based and Non-AR-based paradigms, and provide a detailed examination of the core techniques within each category. Following this taxonomy, we assess their theoretical trade-offs in terms of speed, quality, and efficiency, and examine their potential for combination and comparison with alternative acceleration strategies. Finally, based on our findings, we highlight recent advancements, identify open challenges, and outline promising directions for future research in parallel text generation. We have also created a GitHub repository for indexing relevant papers and open resources available at https://github.com/zhanglingzhe0820/Awesome-Parallel-Text-Generation.
>
---
#### [replaced 044] Universal computation is intrinsic to language model decoding
- **分类: cs.CL**

- **简介: 该论文属于语言模型研究任务，探讨其计算能力。论文证明语言模型通过自回归输出可实现通用计算，即使未训练的模型也具备此能力，强调训练提升的是编程接口而非计算表达能力。**

- **链接: [https://arxiv.org/pdf/2601.08061v2](https://arxiv.org/pdf/2601.08061v2)**

> **作者:** Alex Lewandowski; Marlos C. Machado; Dale Schuurmans
>
> **备注:** Minor formatting corrections
>
> **摘要:** Language models now provide an interface to express and often solve general problems in natural language, yet their ultimate computational capabilities remain a major topic of scientific debate. Unlike a formal computer, a language model is trained to autoregressively predict successive elements in human-generated text. We prove that chaining a language model's autoregressive output is sufficient to perform universal computation. That is, a language model can simulate the execution of any algorithm on any input. The challenge of eliciting desired computational behaviour can thus be reframed in terms of programmability: the ease of finding a suitable prompt. Strikingly, we demonstrate that even randomly initialized language models are capable of universal computation before training. This implies that training does not give rise to computational expressiveness -- rather, it improves programmability, enabling a natural language interface for accessing these intrinsic capabilities.
>
---
#### [replaced 045] Fundamental Reasoning Paradigms Induce Out-of-Domain Generalization in Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究如何通过推理范式提升语言模型的域外泛化能力。工作包括构建数据集、尝试不同方法诱导推理技能，并验证其在真实任务中的效果。**

- **链接: [https://arxiv.org/pdf/2602.08658v2](https://arxiv.org/pdf/2602.08658v2)**

> **作者:** Mingzi Cao; Xingwei Tan; Mahmud Elahi Akhter; Marco Valentino; Maria Liakata; Xi Wang; Nikolaos Aletras
>
> **摘要:** Deduction, induction, and abduction are fundamental reasoning paradigms, core for human logical thinking. Although improving Large Language Model (LLM) reasoning has attracted significant research efforts, the extent to which the fundamental paradigms induce generalization has yet to be systematically explored. In this study, we shed light on how the interplay between these core paradigms influences LLMs' reasoning behavior. To this end, we first collect a new dataset of reasoning trajectories from symbolic tasks, each targeting one of the three fundamental paradigms, to abstract from concrete world knowledge. Then, we investigate effective ways for inducing these skills into LLMs. We experiment with a battery of methods including simple fine-tuning, and more complex approaches to increase model depth, or transform a dense model to a mixture-of-experts. We comprehensively evaluate induced models on realistic out-of-domain tasks, that are entirely formulated in natural language and contain real-world knowledge. Our results reveal that our approach yields strong generalizability with substantial performance gains (up to $14.60$) across realistic tasks.
>
---
#### [replaced 046] Evolving Interactive Diagnostic Agents in a Virtual Clinical Environment
- **分类: cs.CL**

- **简介: 该论文属于医疗诊断任务，旨在提升交互式诊断的准确性与效果。通过强化学习训练语言模型，使其在虚拟临床环境中自主决策，优化诊断流程。**

- **链接: [https://arxiv.org/pdf/2510.24654v2](https://arxiv.org/pdf/2510.24654v2)**

> **作者:** Pengcheng Qiu; Chaoyi Wu; Junwei Liu; Qiaoyu Zheng; Yusheng Liao; Haowen Wang; Yun Yue; Qianrui Fan; Shuai Zhen; Jian Wang; Jinjie Gu; Yanfeng Wang; Ya Zhang; Weidi Xie
>
> **摘要:** We present a framework for training large language models (LLMs) as diagnostic agents with reinforcement learning, enabling them to manage multi-turn interactive diagnostic processes, adaptively select examinations, and commit to final diagnoses. Unlike instruction-tuned models trained on static data, our method acquires diagnostic strategies through dynamic exploration and outcome-based feedback, mapping evolving patient states to the next optimal examination and subsequent diagnosis. Our contributions include: (i) DiagGym, a diagnostics world model trained with electronic health records, serving as a virtual clinical environment to support closed-loop in-silico training and evaluation for interactive diagnosis; (ii) DiagAgent, trained via end-to-end multi-turn RL to learn dynamic diagnostic policies that optimize both interactive effectiveness and final accuracy; (iii) DiagBench, a multi-center diagnostic benchmark designed to evaluate multi-turn diagnostic interaction trajectories. The benchmark comprises 2.2K physician-validated cases sourced from 4 distinct distributions, alongside 3.3K physician-written rubrics for granular process-oriented evaluation. (iv) Extensive evaluations demonstrate DiagAgent's superior performance across both in-domain and out-of-domain (OOD) settings. DiagAgent significantly outperforms 11 SOTA LLMs and 2 prompt-engineered agents. In the end-to-end setting, it delivers a 11.20% increase in diagnostic accuracy and a 17.58% boost in examination recommendation F1 score, while consistently maintaining SOTA performance across all three external centers. Furthermore, in rubric-based evaluations, it surpasses the next-best model by 7.1% in weighted rubric score. These findings indicate that learning policies in interactive clinical environments confers long-term diagnostic management abilities unattainable through passive training.
>
---
#### [replaced 047] DLLM Agent: See Farther, Run Faster
- **分类: cs.CL**

- **简介: 该论文属于智能代理任务，研究扩散大语言模型（DLLM）在多步骤决策中的表现。工作包括对比DLLM与自回归模型的效率，发现DLLM更高效且减少交互次数。**

- **链接: [https://arxiv.org/pdf/2602.07451v2](https://arxiv.org/pdf/2602.07451v2)**

> **作者:** Huiling Zhen; Weizhe Lin; Renxi Liu; Kai Han; Yiming Li; Yuchuan Tian; Hanting Chen; Xiaoguang Li; Xiaosong Li; Chen Chen; Xianzhi Yu; Mingxuan Yuan; Youliang Yan; Peifeng Qin; Jun Wang; Yu Wang; Dacheng Tao; Yunhe Wang
>
> **摘要:** Diffusion large language models (DLLMs) have emerged as an alternative to autoregressive (AR) decoding with appealing efficiency and modeling properties, yet their implications for agentic multi-step decision making remain underexplored. We ask a concrete question: when the generation paradigm is changed but the agent framework and supervision are held fixed, do diffusion backbones induce systematically different planning and tool-use behaviors, and do these differences translate into end-to-end efficiency gains? We study this in a controlled setting by instantiating DLLM and AR backbones within the same agent workflow (DeepDiver) and performing matched agent-oriented fine-tuning on the same trajectory data, yielding diffusion-backed DLLM Agents and directly comparable AR agents. Across benchmarks and case studies, we find that, at comparable accuracy, DLLM Agents are on average over 30% faster end to end than AR agents, with some cases exceeding 8x speedup. Conditioned on correct task completion, DLLM Agents also require fewer interaction rounds and tool invocations, consistent with higher planner hit rates that converge earlier to a correct action path with less backtracking. We further identify two practical considerations for deploying diffusion backbones in tool-using agents. First, naive DLLM policies are more prone to structured tool-call failures, necessitating stronger tool-call-specific training to emit valid schemas and arguments. Second, for multi-turn inputs interleaving context and action spans, diffusion-style span corruption requires aligned attention masking to avoid spurious context-action information flow; without such alignment, performance degrades. Finally, we analyze attention dynamics across workflow stages and observe paradigm-specific coordination patterns, suggesting stronger global planning signals in diffusion-backed agents.
>
---
#### [replaced 048] ReForm: Reflective Autoformalization with Prospective Bounded Sequence Optimization
- **分类: cs.CL**

- **简介: 该论文提出ReForm方法，解决自然语言数学到形式化陈述的自动翻译问题，通过引入语义一致性评估和迭代优化，提升翻译的准确性与语义保真度。**

- **链接: [https://arxiv.org/pdf/2510.24592v3](https://arxiv.org/pdf/2510.24592v3)**

> **作者:** Guoxin Chen; Jing Wu; Xinjie Chen; Wayne Xin Zhao; Ruihua Song; Chengxi Li; Kai Fan; Dayiheng Liu; Minpeng Liao
>
> **备注:** Camera Ready version for ICLR 2026. Code: https://github.com/Chen-GX/ReForm
>
> **摘要:** Autoformalization, which translates natural language mathematics into machine-verifiable formal statements, is critical for using formal mathematical reasoning to solve math problems stated in natural language. While Large Language Models can generate syntactically correct formal statements, they often fail to preserve the original problem's semantic intent. This limitation arises from the LLM approaches' treating autoformalization as a simplistic translation task which lacks mechanisms for self-reflection and iterative refinement that human experts naturally employ. To address these issues, we propose ReForm, a Reflective Autoformalization method that tightly integrates semantic consistency evaluation into the autoformalization process. This enables the model to iteratively generate formal statements, assess its semantic fidelity, and self-correct identified errors through progressive refinement. To effectively train this reflective model, we introduce Prospective Bounded Sequence Optimization (PBSO), which employs different rewards at different sequence positions to ensure that the model develops both accurate autoformalization and correct semantic validations, preventing superficial critiques that would undermine the purpose of reflection. Extensive experiments across four autoformalization benchmarks demonstrate that ReForm achieves an average improvement of 22.6 percentage points over the strongest baselines. To further ensure evaluation reliability, we introduce ConsistencyCheck, a benchmark of 859 expert-annotated items that not only validates LLMs as judges but also reveals that autoformalization is inherently difficult: even human experts produce semantic errors in up to 38.5% of cases.
>
---
#### [replaced 049] REPAIR: Robust Editing via Progressive Adaptive Intervention and Reintegration
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型的后训练优化任务，旨在解决知识更新成本高、错误修正易引发副作用的问题。提出REPAIR框架，实现高效、精准且稳定的模型编辑。**

- **链接: [https://arxiv.org/pdf/2510.01879v2](https://arxiv.org/pdf/2510.01879v2)**

> **作者:** Yisu Wang; Ming Wang; Haoyuan Song; Wenjie Huang; Chaozheng Wang; Yi Xie; Xuming Ran
>
> **摘要:** Post-training for large language models (LLMs) is constrained by the high cost of acquiring new knowledge or correcting errors and by the unintended side effects that frequently arise from retraining. To address these issues, we introduce REPAIR (Robust Editing via Progressive Adaptive Intervention and Reintegration), a lifelong editing framework designed to support precise and low-cost model updates while preserving non-target knowledge. REPAIR mitigates the instability and conflicts of large-scale sequential edits through a closed-loop feedback mechanism coupled with dynamic memory management. Furthermore, by incorporating frequent knowledge fusion and enforcing strong locality guards, REPAIR effectively addresses the shortcomings of traditional distribution-agnostic approaches that often overlook unintended ripple effects. Our experiments demonstrate that REPAIR boosts editing accuracy by 10%-30% across multiple model families and significantly reduces knowledge forgetting. This work introduces a robust framework for developing reliable, scalable, and continually evolving LLMs.
>
---
#### [replaced 050] An Iterative Question-Guided Framework for Knowledge Base Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识库问答任务，旨在解决多跳推理中的路径连贯性和关键连接丢失问题。提出iQUEST框架，通过迭代分解问题并结合图神经网络提升推理效果。**

- **链接: [https://arxiv.org/pdf/2506.01784v5](https://arxiv.org/pdf/2506.01784v5)**

> **作者:** Shuai Wang; Yinan Yu
>
> **备注:** Accepted to the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025), Main Track
>
> **摘要:** Large Language Models (LLMs) excel in many natural language processing tasks but often exhibit factual inconsistencies in knowledge-intensive settings. Integrating external knowledge resources, particularly knowledge graphs (KGs), provides a transparent and updatable foundation for more reliable reasoning. Knowledge Base Question Answering (KBQA), which queries and reasons over KGs, is central to this effort, especially for complex, multi-hop queries. However, multi-hop reasoning poses two key challenges: (1)~maintaining coherent reasoning paths, and (2)~avoiding prematurely discarding critical multi-hop connections. To tackle these challenges, we introduce iQUEST, a question-guided KBQA framework that iteratively decomposes complex queries into simpler sub-questions, ensuring a structured and focused reasoning trajectory. Additionally, we integrate a Graph Neural Network (GNN) to look ahead and incorporate 2-hop neighbor information at each reasoning step. This dual approach strengthens the reasoning process, enabling the model to explore viable paths more effectively. Detailed experiments demonstrate the consistent improvement delivered by iQUEST across four benchmark datasets and four LLMs.
>
---
#### [replaced 051] Modelling and Classifying the Components of a Literature Review
- **分类: cs.CL; cs.AI; cs.HC; cs.IR**

- **简介: 该论文属于自然语言处理中的文本分类任务，旨在解决科学文献中句子 rhetorical 角色的自动标注问题。研究提出一种新标注框架，并评估多种大模型在该任务上的表现。**

- **链接: [https://arxiv.org/pdf/2508.04337v2](https://arxiv.org/pdf/2508.04337v2)**

> **作者:** Francisco Bolaños; Angelo Salatino; Francesco Osborne; Enrico Motta
>
> **摘要:** Previous work has demonstrated that AI methods for analysing scientific literature benefit significantly from annotating sentences in papers according to their rhetorical roles, such as research gaps, results, limitations, extensions of existing methodologies, and others. Such representations also have the potential to support the development of a new generation of systems capable of producing high-quality literature reviews. However, achieving this goal requires the definition of a relevant annotation schema and effective strategies for large-scale annotation of the literature. This paper addresses these challenges in two ways: 1) it introduces a novel, unambiguous annotation schema that is explicitly designed for reliable automatic processing, and 2) it presents a comprehensive evaluation of a wide range of large language models (LLMs) on the task of classifying rhetorical roles according to this schema. To this end, we also present Sci-Sentence, a novel multidisciplinary benchmark comprising 700 sentences manually annotated by domain experts and 2,240 sentences automatically labelled using LLMs. We evaluate 37 LLMs on this benchmark, spanning diverse model families and sizes, using both zero-shot learning and fine-tuning approaches. The experiments reveal that modern LLMs achieve strong results on this task when fine-tuned on high-quality data, surpassing 96% F1, with both large proprietary models such as GPT-4o and lightweight open-source alternatives performing well. Moreover, augmenting the training set with semi-synthetic LLM-generated examples further boosts performance, enabling small encoders to achieve robust results and substantially improving several open decoder models.
>
---
#### [replaced 052] Can LLMs Automate Fact-Checking Article Writing?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于事实核查任务，旨在解决自动化生成适合公众传播的核查文章的问题。研究提出QRAFT框架，模拟人工写作流程，并通过评估验证其效果。**

- **链接: [https://arxiv.org/pdf/2503.17684v2](https://arxiv.org/pdf/2503.17684v2)**

> **作者:** Dhruv Sahnan; David Corney; Irene Larraz; Giovanni Zagni; Ruben Miguez; Zhuohan Xie; Iryna Gurevych; Elizabeth Churchill; Tanmoy Chakraborty; Preslav Nakov
>
> **备注:** Accepted to TACL 2026, pre-MIT Press publication version
>
> **摘要:** Automatic fact-checking aims to support professional fact-checkers by offering tools that can help speed up manual fact-checking. Yet, existing frameworks fail to address the key step of producing output suitable for broader dissemination to the general public: while human fact-checkers communicate their findings through fact-checking articles, automated systems typically produce little or no justification for their assessments. Here, we aim to bridge this gap. In particular, we argue for the need to extend the typical automatic fact-checking pipeline with automatic generation of full fact-checking articles. We first identify key desiderata for such articles through a series of interviews with experts from leading fact-checking organizations. We then develop QRAFT, an LLM-based agentic framework that mimics the writing workflow of human fact-checkers. Finally, we assess the practical usefulness of QRAFT through human evaluations with professional fact-checkers. Our evaluation shows that while QRAFT outperforms several previously proposed text-generation approaches, it lags considerably behind expert-written articles. We hope that our work will enable further research in this new and important direction. The code for our implementation is available at https://github.com/mbzuai-nlp/qraft.git.
>
---
#### [replaced 053] EPO: Entropy-regularized Policy Optimization for LLM Agents Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，解决多轮稀疏奖励环境下LLM代理的探索与利用问题。提出EPO框架，通过熵正则化等机制提升训练稳定性与性能。**

- **链接: [https://arxiv.org/pdf/2509.22576v2](https://arxiv.org/pdf/2509.22576v2)**

> **作者:** Wujiang Xu; Wentian Zhao; Zhenting Wang; Yu-Jhe Li; Can Jin; Mingyu Jin; Kai Mei; Kun Wan; Dimitris N. Metaxas
>
> **摘要:** Training LLM agents in multi-turn environments with sparse rewards, where completing a single task requires 30+ turns of interaction within an episode, presents a fundamental challenge for reinforcement learning. We identify a critical failure mode unique to this setting: the exploration-exploitation cascade failure. This cascade begins with early-stage policy premature convergence, where sparse feedback causes agents to commit to flawed, low-entropy strategies. Subsequently, agents enter late-stage policy collapse, where conventional entropy regularization becomes counterproductive, promoting chaotic exploration that destabilizes training. We propose Entropy-regularized Policy Optimization (EPO), a general framework that breaks this failure cycle through three synergistic mechanisms: (1) adopting entropy regularization in multi-turn settings to enhance exploration, (2) an entropy smoothing regularizer that bounds policy entropy within historical averages to prevent abrupt fluctuations, and (3) adaptive phase-based weighting that balances exploration and exploitation across training. Our analysis justifies that EPO guarantees monotonically decreasing entropy variance while maintaining convergence. EPO achieves up to 152% performance improvement on ScienceWorld and up to 19.8% on ALFWorld. Our work demonstrates that multi-turn sparse-reward settings require fundamentally different entropy control than traditional RL, with broad implications for LLM agent training.
>
---
#### [replaced 054] Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型对齐任务，解决prompt优化与推理策略脱节的问题。提出IAPO框架，联合优化提示和推理规模，提升模型对齐效果。**

- **链接: [https://arxiv.org/pdf/2508.10030v3](https://arxiv.org/pdf/2508.10030v3)**

> **作者:** Saaduddin Mahmud; Mason Nakamura; Kyle Hollins Wray; Shlomo Zilberstein
>
> **备注:** Accepted to AAAI 2026. Extended 17-page version
>
> **摘要:** Prompt optimization methods have demonstrated significant effectiveness in aligning black-box large language models (LLMs). In parallel, inference scaling strategies such as Best-of-N Sampling and Majority Voting have likewise been shown to improve alignment and performance by trading additional computation for better output. However, existing prompt optimization approaches are inference strategy agnostic; that is, they optimize prompts without accounting for the inference strategy. This constitutes a significant methodological gap, as our empirical and theoretical analysis reveals a strong interdependence between these two paradigms. Moreover, we find that user preferences regarding trade-offs among multiple objectives and inference budgets substantially influence the choice of prompt and inference configuration. To address this gap, we introduce a novel unified framework named IAPO (Inference-Aware Prompt Optimization) that jointly optimizes the prompt and inference scale, while being aware of the inference budget and different task objectives. We then develop a fixed-budget training algorithm for IAPO, called PSST (Prompt Scaling via Sequential Trimming), and establish finite-budget guarantees on the error probability. Finally, we evaluate the effectiveness of PSST on six tasks, including multi-objective text generation and reasoning, and demonstrate the critical role of incorporating inference-awareness in aligning black-box LLMs using prompt optimization.
>
---
#### [replaced 055] Structured Episodic Event Memory
- **分类: cs.CL**

- **简介: 该论文属于记忆建模任务，旨在解决LLM在复杂推理中缺乏结构化记忆的问题。提出SEEM框架，结合图记忆与动态情节记忆，提升叙事连贯性与逻辑一致性。**

- **链接: [https://arxiv.org/pdf/2601.06411v2](https://arxiv.org/pdf/2601.06411v2)**

> **作者:** Zhengxuan Lu; Dongfang Li; Yukun Shi; Beilun Wang; Longyue Wang; Baotian Hu
>
> **摘要:** Current approaches to memory in Large Language Models (LLMs) predominantly rely on static Retrieval-Augmented Generation (RAG), which often results in scattered retrieval and fails to capture the structural dependencies required for complex reasoning. For autonomous agents, these passive and flat architectures lack the cognitive organization necessary to model the dynamic and associative nature of long-term interaction. To address this, we propose Structured Episodic Event Memory (SEEM), a hierarchical framework that synergizes a graph memory layer for relational facts with a dynamic episodic memory layer for narrative progression. Grounded in cognitive frame theory, SEEM transforms interaction streams into structured Episodic Event Frames (EEFs) anchored by precise provenance pointers. Furthermore, we introduce an agentic associative fusion and Reverse Provenance Expansion (RPE) mechanism to reconstruct coherent narrative contexts from fragmented evidence. Experimental results on the LoCoMo and LongMemEval benchmarks demonstrate that SEEM significantly outperforms baselines, enabling agents to maintain superior narrative coherence and logical consistency.
>
---
#### [replaced 056] The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs
- **分类: cs.CL**

- **简介: 该论文属于安全漏洞研究任务，针对扩散型大语言模型（dLLMs）的对齐机制失效问题，提出DIJA攻击框架，揭示其在对抗性提示下的安全风险。**

- **链接: [https://arxiv.org/pdf/2507.11097v2](https://arxiv.org/pdf/2507.11097v2)**

> **作者:** Zichen Wen; Jiashu Qu; Zhaorun Chen; Xiaoya Lu; Dongrui Liu; Zhiyuan Liu; Ruixi Wu; Yicun Yang; Xiangqi Jin; Haoyun Xu; Xuyang Liu; Weijia Li; Chaochao Lu; Jing Shao; Conghui He; Linfeng Zhang
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Diffusion-based large language models (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs, offering faster inference and greater interactivity via parallel decoding and bidirectional modeling. However, despite strong performance in code generation and text infilling, we identify a fundamental safety concern: existing alignment mechanisms fail to safeguard dLLMs against context-aware, masked-input adversarial prompts, exposing novel vulnerabilities. To this end, we present DIJA, the first systematic study and jailbreak attack framework that exploits unique safety weaknesses of dLLMs. Specifically, our proposed DIJA constructs adversarial interleaved mask-text prompts that exploit the text generation mechanisms of dLLMs, i.e., bidirectional modeling and parallel decoding. Bidirectional modeling drives the model to produce contextually consistent outputs for masked spans, even when harmful, while parallel decoding limits model dynamic filtering and rejection sampling of unsafe content. This causes standard alignment mechanisms to fail, enabling harmful completions in alignment-tuned dLLMs, even when harmful behaviors or unsafe instructions are directly exposed in the prompt. Through comprehensive experiments, we demonstrate that DIJA significantly outperforms existing jailbreak methods, exposing a previously overlooked threat surface in dLLM architectures. Notably, our method achieves up to 100% keyword-based ASR on Dream-Instruct, surpassing the strongest prior baseline, ReNeLLM, by up to 78.5% in evaluator-based ASR on JailbreakBench and by 37.7 points in StrongREJECT score, while requiring no rewriting or hiding of harmful content in the jailbreak prompt. Our findings underscore the urgent need for rethinking safety alignment in this emerging class of language models. Code is available at https://github.com/ZichenWen1/DIJA.
>
---

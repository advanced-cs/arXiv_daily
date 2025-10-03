# 自然语言处理 cs.CL

- **最新发布 133 篇**

- **更新 71 篇**

## 最新发布

#### [new 001] Enhancing Large Language Model Reasoning with Reward Models: An Analytical Survey
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在提升大语言模型的推理能力。通过系统分析奖励模型的应用，解决模型生成质量与优化问题。**

- **链接: [http://arxiv.org/pdf/2510.01925v1](http://arxiv.org/pdf/2510.01925v1)**

> **作者:** Qiyuan Liu; Hao Xu; Xuhong Chen; Wei Chen; Yee Whye Teh; Ning Miao
>
> **摘要:** Reward models (RMs) play a critical role in enhancing the reasoning performance of LLMs. For example, they can provide training signals to finetune LLMs during reinforcement learning (RL) and help select the best answer from multiple candidates during inference. In this paper, we provide a systematic introduction to RMs, along with a comprehensive survey of their applications in LLM reasoning. We first review fundamental concepts of RMs, including their architectures, training methodologies, and evaluation techniques. Then, we explore their key applications: (1) guiding generation and selecting optimal outputs during LLM inference, (2) facilitating data synthesis and iterative self-improvement for LLMs, and (3) providing training signals in RL-based finetuning. Finally, we address critical open questions regarding the selection, generalization, evaluation, and enhancement of RMs, based on existing research and our own empirical findings. Our analysis aims to provide actionable insights for the effective deployment and advancement of RMs for LLM reasoning.
>
---
#### [new 002] Enhanced Arabic-language cyberbullying detection: deep embedding and transformer (BERT) approaches
- **分类: cs.CL**

- **简介: 该论文属于阿拉伯语网络欺凌检测任务，旨在解决阿拉伯语内容中网络欺凌识别问题。通过构建数据集并测试多种深度学习模型，提升了检测效果。**

- **链接: [http://arxiv.org/pdf/2510.02232v1](http://arxiv.org/pdf/2510.02232v1)**

> **作者:** Ebtesam Jaber Aljohani; Wael M. S. Yafoo
>
> **摘要:** Recent technological advances in smartphones and communications, including the growth of such online platforms as massive social media networks such as X (formerly known as Twitter) endangers young people and their emotional well-being by exposing them to cyberbullying, taunting, and bullying content. Most proposed approaches for automatically detecting cyberbullying have been developed around the English language, and methods for detecting Arabic-language cyberbullying are scarce. Methods for detecting Arabic-language cyberbullying are especially scarce. This paper aims to enhance the effectiveness of methods for detecting cyberbullying in Arabic-language content. We assembled a dataset of 10,662 X posts, pre-processed the data, and used the kappa tool to verify and enhance the quality of our annotations. We conducted four experiments to test numerous deep learning models for automatically detecting Arabic-language cyberbullying. We first tested a long short-term memory (LSTM) model and a bidirectional long short-term memory (Bi-LSTM) model with several experimental word embeddings. We also tested the LSTM and Bi-LSTM models with a novel pre-trained bidirectional encoder from representations (BERT) and then tested them on a different experimental models BERT again. LSTM-BERT and Bi-LSTM-BERT demonstrated a 97% accuracy. Bi-LSTM with FastText embedding word performed even better, achieving 98% accuracy. As a result, the outcomes are generalize
>
---
#### [new 003] Comparison of Unsupervised Metrics for Evaluating Judicial Decision Extraction
- **分类: cs.CL; cs.AI; cs.IR; H.3.3; I.2.8; I.2.7**

- **简介: 该论文属于法律自然语言处理任务，旨在评估司法文书信息提取的质量。通过16种无监督指标对比，发现部分指标与专家评分有中等相关性，但无法替代人工判断。**

- **链接: [http://arxiv.org/pdf/2510.01792v1](http://arxiv.org/pdf/2510.01792v1)**

> **作者:** Ivan Leonidovich Litvak; Anton Kostin; Fedor Lashkin; Tatiana Maksiyan; Sergey Lagutin
>
> **备注:** 28 pages
>
> **摘要:** The rapid advancement of artificial intelligence in legal natural language processing demands scalable methods for evaluating text extraction from judicial decisions. This study evaluates 16 unsupervised metrics, including novel formulations, to assess the quality of extracting seven semantic blocks from 1,000 anonymized Russian judicial decisions, validated against 7,168 expert reviews on a 1--5 Likert scale. These metrics, spanning document-based, semantic, structural, pseudo-ground truth, and legal-specific categories, operate without pre-annotated ground truth. Bootstrapped correlations, Lin's concordance correlation coefficient (CCC), and mean absolute error (MAE) reveal that Term Frequency Coherence (Pearson $r = 0.540$, Lin CCC = 0.512, MAE = 0.127) and Coverage Ratio/Block Completeness (Pearson $r = 0.513$, Lin CCC = 0.443, MAE = 0.139) best align with expert ratings, while Legal Term Density (Pearson $r = -0.479$, Lin CCC = -0.079, MAE = 0.394) show strong negative correlations. The LLM Evaluation Score (mean = 0.849, Pearson $r = 0.382$, Lin CCC = 0.325, MAE = 0.197) showed moderate alignment, but its performance, using gpt-4.1-mini via g4f, suggests limited specialization for legal textse. These findings highlight that unsupervised metrics, including LLM-based approaches, enable scalable screening but, with moderate correlations and low CCC values, cannot fully replace human judgment in high-stakes legal contexts. This work advances legal NLP by providing annotation-free evaluation tools, with implications for judicial analytics and ethical AI deployment.
>
---
#### [new 004] ARUQULA -- An LLM based Text2SPARQL Approach using ReAct and Knowledge Graph Exploration Utilities
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于Text2SPARQL任务，旨在降低非技术人员使用SPARQL查询知识图谱的难度。通过迭代探索与执行的LLM方法实现自然语言到SPARQL的转换。**

- **链接: [http://arxiv.org/pdf/2510.02200v1](http://arxiv.org/pdf/2510.02200v1)**

> **作者:** Felix Brei; Lorenz Bühmann; Johannes Frey; Daniel Gerber; Lars-Peter Meyer; Claus Stadler; Kirill Bulert
>
> **备注:** peer reviewed publication at Text2SPARQL Workshop @ ESWC 2025
>
> **摘要:** Interacting with knowledge graphs can be a daunting task for people without a background in computer science since the query language that is used (SPARQL) has a high barrier of entry. Large language models (LLMs) can lower that barrier by providing support in the form of Text2SPARQL translation. In this paper we introduce a generalized method based on SPINACH, an LLM backed agent that translates natural language questions to SPARQL queries not in a single shot, but as an iterative process of exploration and execution. We describe the overall architecture and reasoning behind our design decisions, and also conduct a thorough analysis of the agent behavior to gain insights into future areas for targeted improvements. This work was motivated by the Text2SPARQL challenge, a challenge that was held to facilitate improvements in the Text2SPARQL domain.
>
---
#### [new 005] Context Matters: Comparison of commercial large language tools in veterinary medicine
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于 veterinary NLP 任务，评估商业 LLM 在兽医肿瘤记录摘要中的表现，解决工具性能比较问题，通过评分框架分析三个产品效果。**

- **链接: [http://arxiv.org/pdf/2510.01224v1](http://arxiv.org/pdf/2510.01224v1)**

> **作者:** Tyler J Poore; Christopher J Pinard; Aleena Shabbir; Andrew Lagree; Andre Telfer; Kuan-Chuen Wu
>
> **备注:** 4 Figures, 10 pages
>
> **摘要:** Large language models (LLMs) are increasingly used in clinical settings, yet their performance in veterinary medicine remains underexplored. We evaluated three commercially available veterinary-focused LLM summarization tools (Product 1 [Hachiko] and Products 2 and 3) on a standardized dataset of veterinary oncology records. Using a rubric-guided LLM-as-a-judge framework, summaries were scored across five domains: Factual Accuracy, Completeness, Chronological Order, Clinical Relevance, and Organization. Product 1 achieved the highest overall performance, with a median average score of 4.61 (IQR: 0.73), compared to 2.55 (IQR: 0.78) for Product 2 and 2.45 (IQR: 0.92) for Product 3. It also received perfect median scores in Factual Accuracy and Chronological Order. To assess the internal consistency of the grading framework itself, we repeated the evaluation across three independent runs. The LLM grader demonstrated high reproducibility, with Average Score standard deviations of 0.015 (Product 1), 0.088 (Product 2), and 0.034 (Product 3). These findings highlight the importance of veterinary-specific commercial LLM tools and demonstrate that LLM-as-a-judge evaluation is a scalable and reproducible method for assessing clinical NLP summarization in veterinary medicine.
>
---
#### [new 006] SoK: Measuring What Matters for Closed-Loop Security Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于网络安全领域，解决封闭环安全代理评估问题。提出CLASP框架和CLC评分，用于衡量安全系统的闭环能力与效果。**

- **链接: [http://arxiv.org/pdf/2510.01654v1](http://arxiv.org/pdf/2510.01654v1)**

> **作者:** Mudita Khurana; Raunak Jain
>
> **摘要:** Cybersecurity is a relentless arms race, with AI driven offensive systems evolving faster than traditional defenses can adapt. Research and tooling remain fragmented across isolated defensive functions, creating blind spots that adversaries exploit. Autonomous agents capable of integrating, exploit confirmation, remediation, and validation into a single closed loop offer promise, but the field lacks three essentials: a framework defining the agentic capabilities of security systems across security life cycle, a principled method for evaluating closed loop agents, and a benchmark for measuring their performance in practice. We introduce CLASP: the Closed-Loop Autonomous Security Performance framework which aligns the security lifecycle (reconnaissance, exploitation, root cause analysis, patch synthesis, validation) with core agentic capabilities (planning, tool use, memory, reasoning, reflection & perception) providing a common vocabulary and rubric for assessing agentic capabilities in security tasks. By applying CLASP to 21 representative works, we map where systems demonstrate strengths, and where capability gaps persist. We then define the Closed-Loop Capability (CLC) Score, a composite metric quantifying both degree of loop closure and operational effectiveness, and outline the requirements for a closed loop benchmark. Together, CLASP and the CLC Score, provide the vocabulary, diagnostics, and measurements needed to advance both function level performance and measure closed loop security agents.
>
---
#### [new 007] A Comparison of Independent and Joint Fine-tuning Strategies for Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于问答任务，研究RAG框架的微调策略。比较独立、联合和两阶段微调方法，分析其效果与计算成本，以确定最佳策略。**

- **链接: [http://arxiv.org/pdf/2510.01600v1](http://arxiv.org/pdf/2510.01600v1)**

> **作者:** Neal Gregory Lawton; Alfy Samuel; Anoop Kumar; Daben Liu
>
> **摘要:** A Comparison of Independent and Joint Fine-tuning Strategies for Retrieval-Augmented Generation Download PDF Neal Gregory Lawton, Alfy Samuel, Anoop Kumar, Daben Liu Published: 20 Aug 2025, Last Modified: 17 Sept 2025EMNLP 2025 FindingsConference, Publication Chairs, AuthorsRevisionsBibTeXCC BY 4.0 Keywords: Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), Fine-tuning, Question Answering, Joint fine-tuning TL;DR: We evaluate and compare strategies for fine-tuning Retrieval Augmented Generation (RAG) pipelines, including independent fine-tuning, joint fine-tuning, and two-phase fine-tuning. Abstract: Retrieval augmented generation (RAG) is a popular framework for question answering that is powered by two large language models (LLMs): an embedding model that retrieves context documents from a database that are relevant to a given question, and a generator model that uses the retrieved context to generate an answer to the question. Both the embedding and generator models can be fine-tuned to increase performance of a RAG pipeline on a new task, but multiple fine-tuning strategies exist with different costs and benefits. In this paper, we evaluate and compare several RAG fine-tuning strategies, including independent, joint, and two-phase fine-tuning. In our experiments, we observe that all of these strategies achieve about equal improvement in EM and F1 generation quality metrics, although they have significantly different computational costs. We conclude the optimal fine-tuning strategy to use depends on whether the training dataset includes context labels and whether a grid search over the learning rates for the embedding and generator models is required.
>
---
#### [new 008] Feasibility of Structuring Stress Documentation Using an Ontology-Guided Large Language Model
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决应力文档结构化不足的问题。通过构建本体并利用大语言模型提取应力相关信息，提升文档一致性与临床实用性。**

- **链接: [http://arxiv.org/pdf/2510.01244v1](http://arxiv.org/pdf/2510.01244v1)**

> **作者:** Hyeoneui Kim; Jeongha Kim; Huijing Xu; Jinsun Jung; Sunghoon Kang; Sun Joo Jang
>
> **摘要:** Stress, arising from the dynamic interaction between external stressors, individual appraisals, and physiological or psychological responses, significantly impacts health yet is often underreported and inconsistently documented, typically captured as unstructured free-text in electronic health records. Ambient AI technologies offer promise in reducing documentation burden, but predominantly generate unstructured narratives, limiting downstream clinical utility. This study aimed to develop an ontology for mental stress and evaluate the feasibility of using a Large Language Model (LLM) to extract ontology-guided stress-related information from narrative text. The Mental Stress Ontology (MeSO) was developed by integrating theoretical models like the Transactional Model of Stress with concepts from 11 validated stress assessment tools. MeSO's structure and content were refined using Ontology Pitfall Scanner! and expert validation. Using MeSO, six categories of stress-related information--stressor, stress response, coping strategy, duration, onset, and temporal profile--were extracted from 35 Reddit posts using Claude Sonnet 4. Human reviewers evaluated accuracy and ontology coverage. The final ontology included 181 concepts across eight top-level classes. Of 220 extractable stress-related items, the LLM correctly identified 172 (78.2%), misclassified 27 (12.3%), and missed 21 (9.5%). All correctly extracted items were accurately mapped to MeSO, although 24 relevant concepts were not yet represented in the ontology. This study demonstrates the feasibility of using an ontology-guided LLM for structured extraction of stress-related information, offering potential to enhance the consistency and utility of stress documentation in ambient AI systems. Future work should involve clinical dialogue data and comparison across LLMs.
>
---
#### [new 009] AMAS: Adaptively Determining Communication Topology for LLM-based Multi-Agent System
- **分类: cs.CL**

- **简介: 该论文属于多智能体系统任务，解决LLM在工业应用中因固定拓扑结构导致的效率问题。提出AMAS框架，通过动态图设计实现任务自适应的通信拓扑。**

- **链接: [http://arxiv.org/pdf/2510.01617v1](http://arxiv.org/pdf/2510.01617v1)**

> **作者:** Hui Yi Leong; Yuheng Li; Yuqing Wu; Wenwen Ouyang; Wei Zhu; Jiechao Gao
>
> **摘要:** Although large language models (LLMs) have revolutionized natural language processing capabilities, their practical implementation as autonomous multi-agent systems (MAS) for industrial problem-solving encounters persistent barriers. Conventional MAS architectures are fundamentally restricted by inflexible, hand-crafted graph topologies that lack contextual responsiveness, resulting in diminished efficacy across varied academic and commercial workloads. To surmount these constraints, we introduce AMAS, a paradigm-shifting framework that redefines LLM-based MAS through a novel dynamic graph designer. This component autonomously identifies task-specific optimal graph configurations via lightweight LLM adaptation, eliminating the reliance on monolithic, universally applied structural templates. Instead, AMAS exploits the intrinsic properties of individual inputs to intelligently direct query trajectories through task-optimized agent pathways. Rigorous validation across question answering, mathematical deduction, and code generation benchmarks confirms that AMAS systematically exceeds state-of-the-art single-agent and multi-agent approaches across diverse LLM architectures. Our investigation establishes that context-sensitive structural adaptability constitutes a foundational requirement for high-performance LLM MAS deployments.
>
---
#### [new 010] More Than One Teacher: Adaptive Multi-Guidance Policy Optimization for Diverse Exploration
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于强化学习任务，解决LLM推理多样性不足问题。提出AMPO框架，通过多教师引导提升探索效率与性能。**

- **链接: [http://arxiv.org/pdf/2510.02227v1](http://arxiv.org/pdf/2510.02227v1)**

> **作者:** Xiaoyang Yuan; Yujuan Ding; Yi Bin; Wenqi Shao; Jinyu Cai; Jingkuan Song; Yang Yang; Hengtao Shen
>
> **备注:** 20 pages, 5 figures
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) is a promising paradigm for enhancing the reasoning ability in Large Language Models (LLMs). However, prevailing methods primarily rely on self-exploration or a single off-policy teacher to elicit long chain-of-thought (LongCoT) reasoning, which may introduce intrinsic model biases and restrict exploration, ultimately limiting reasoning diversity and performance. Drawing inspiration from multi-teacher strategies in knowledge distillation, we introduce Adaptive Multi-Guidance Policy Optimization (AMPO), a novel framework that adaptively leverages guidance from multiple proficient teacher models, but only when the on-policy model fails to generate correct solutions. This "guidance-on-demand" approach expands exploration while preserving the value of self-discovery. Moreover, AMPO incorporates a comprehension-based selection mechanism, prompting the student to learn from the reasoning paths that it is most likely to comprehend, thus balancing broad exploration with effective exploitation. Extensive experiments show AMPO substantially outperforms a strong baseline (GRPO), with a 4.3% improvement on mathematical reasoning tasks and 12.2% on out-of-distribution tasks, while significantly boosting Pass@k performance and enabling more diverse exploration. Notably, using four peer-sized teachers, our method achieves comparable results to approaches that leverage a single, more powerful teacher (e.g., DeepSeek-R1) with more data. These results demonstrate a more efficient and scalable path to superior reasoning and generalizability. Our code is available at https://github.com/SII-Enigma/AMPO.
>
---
#### [new 011] Say One Thing, Do Another? Diagnosing Reasoning-Execution Gaps in VLM-Powered Mobile-Use Agents
- **分类: cs.CL**

- **简介: 该论文属于移动代理任务，旨在解决VLM在执行指令时的推理与执行不一致问题。通过引入GTA评估框架，诊断并分析推理与执行间的差距。**

- **链接: [http://arxiv.org/pdf/2510.02204v1](http://arxiv.org/pdf/2510.02204v1)**

> **作者:** Lingzhong Dong; Ziqi Zhou; Shuaibo Yang; Haiyue Sheng; Pengzhou Cheng; Zongru Wu; Zheng Wu; Gongshen Liu; Zhuosheng Zhang
>
> **摘要:** Mobile-use agents powered by vision-language models (VLMs) have shown great potential in interpreting natural language instructions and generating corresponding actions based on mobile graphical user interface. Recent studies suggest that incorporating chain-of-thought (CoT) reasoning tends to improve the execution accuracy. However, existing evaluations emphasize execution accuracy while neglecting whether CoT reasoning aligns with ground-truth actions. This oversight fails to assess potential reasoning-execution gaps, which in turn foster over-trust: users relying on seemingly plausible CoTs may unknowingly authorize harmful actions, potentially resulting in financial loss or trust crisis. In this work, we introduce a new evaluation framework to diagnose reasoning-execution gaps. At its core lies Ground-Truth Alignment (GTA), which measures whether the action implied by a CoT matches the ground-truth action. By combining GTA with the standard Exact Match (EM) metric, we jointly assess both the reasoning accuracy and execution accuracy. This joint perspective reveals two types of reasoning-execution gaps: (i) Execution Gap (EG), where the reasoning correctly identifies the correct action but execution fails, and (ii) Reasoning Gap (RG), where execution succeeds but reasoning process conflicts with the actual execution. Experimental results across a wide range of mobile interaction tasks reveal that reasoning-execution gaps are prevalent, with execution gaps occurring more frequently than reasoning gaps. Moreover, while scaling up model size reduces the overall gap, sizable execution gaps persist even in the largest models. Further analysis shows that our framework reliably reflects systematic EG/RG patterns in state-of-the-art models. These findings offer concrete diagnostics and support the development of more trustworthy mobile-use agents.
>
---
#### [new 012] Longitudinal Monitoring of LLM Content Moderation of Social Issues
- **分类: cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于LLM内容审核研究，旨在解决模型拒绝生成特定内容的透明度问题。通过构建AI Watchman系统，长期监控并分析不同模型的内容 moderation 行为。**

- **链接: [http://arxiv.org/pdf/2510.01255v1](http://arxiv.org/pdf/2510.01255v1)**

> **作者:** Yunlang Dai; Emma Lurie; Danaé Metaxa; Sorelle A. Friedler
>
> **摘要:** Large language models' (LLMs') outputs are shaped by opaque and frequently-changing company content moderation policies and practices. LLM moderation often takes the form of refusal; models' refusal to produce text about certain topics both reflects company policy and subtly shapes public discourse. We introduce AI Watchman, a longitudinal auditing system to publicly measure and track LLM refusals over time, to provide transparency into an important and black-box aspect of LLMs. Using a dataset of over 400 social issues, we audit Open AI's moderation endpoint, GPT-4.1, and GPT-5, and DeepSeek (both in English and Chinese). We find evidence that changes in company policies, even those not publicly announced, can be detected by AI Watchman, and identify company- and model-specific differences in content moderation. We also qualitatively analyze and categorize different forms of refusal. This work contributes evidence for the value of longitudinal auditing of LLMs, and AI Watchman, one system for doing so.
>
---
#### [new 013] Who is In Charge? Dissecting Role Conflicts in Instruction Following
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大语言模型在遵循指令时的角色冲突问题，分析系统指令与社会线索之间的矛盾，提出轻量级对齐方法以增强系统服从性。**

- **链接: [http://arxiv.org/pdf/2510.01228v1](http://arxiv.org/pdf/2510.01228v1)**

> **作者:** Siqi Zeng
>
> **摘要:** Large language models should follow hierarchical instructions where system prompts override user inputs, yet recent work shows they often ignore this rule while strongly obeying social cues such as authority or consensus. We extend these behavioral findings with mechanistic interpretations on a large-scale dataset. Linear probing shows conflict-decision signals are encoded early, with system-user and social conflicts forming distinct subspaces. Direct Logit Attribution reveals stronger internal conflict detection in system-user cases but consistent resolution only for social cues. Steering experiments show that, despite using social cues, the vectors surprisingly amplify instruction following in a role-agnostic way. Together, these results explain fragile system obedience and underscore the need for lightweight hierarchy-sensitive alignment methods.
>
---
#### [new 014] Redundancy-as-Masking: Formalizing the Artificial Age Score (AAS) to Model Memory Aging in Generative AI
- **分类: cs.CL; cs.AI; cs.IT; cs.LG; math.IT; 68T05, 03C95, 94A17, 68Q85; I.2.0; H.1.2; H.1.1; H.1.0; F.4.0**

- **简介: 该论文属于人工智能记忆研究任务，旨在解决模型记忆老化问题。通过引入AAS指标，量化记忆退化过程，并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2510.01242v1](http://arxiv.org/pdf/2510.01242v1)**

> **作者:** Seyma Yaman Kayadibi
>
> **备注:** 34 pages, 17 figures. Includes theoretical development and mathematical proofs of the Artificial Age Score (AAS), with empirical illustrations via ChatGPT-based memory recall experiments (screenshots included)
>
> **摘要:** Artificial intelligence is observed to age not through chronological time but through structural asymmetries in memory performance. In large language models, semantic cues such as the name of the day often remain stable across sessions, while episodic details like the sequential progression of experiment numbers tend to collapse when conversational context is reset. To capture this phenomenon, the Artificial Age Score (AAS) is introduced as a log-scaled, entropy-informed metric of memory aging derived from observable recall behavior. The score is formally proven to be well-defined, bounded, and monotonic under mild and model-agnostic assumptions, making it applicable across various tasks and domains. In its Redundancy-as-Masking formulation, the score interprets redundancy as overlapping information that reduces the penalized mass. However, in the present study, redundancy is not explicitly estimated; all reported values assume a redundancy-neutral setting (R = 0), yielding conservative upper bounds. The AAS framework was tested over a 25-day bilingual study involving ChatGPT-5, structured into stateless and persistent interaction phases. During persistent sessions, the model consistently recalled both semantic and episodic details, driving the AAS toward its theoretical minimum, indicative of structural youth. In contrast, when sessions were reset, the model preserved semantic consistency but failed to maintain episodic continuity, causing a sharp increase in the AAS and signaling structural memory aging. These findings support the utility of AAS as a theoretically grounded, task-independent diagnostic tool for evaluating memory degradation in artificial systems. The study builds on foundational concepts from von Neumann's work on automata, Shannon's theories of information and redundancy, and Turing's behavioral approach to intelligence.
>
---
#### [new 015] Measuring Algorithmic Partisanship via Zero-Shot Classification and Its Implications on Political Discourse
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于算法偏见检测任务，旨在评估大语言模型的政治倾向性。通过零样本分类方法分析模型响应，揭示其对政治话语的影响。**

- **链接: [http://arxiv.org/pdf/2510.01258v1](http://arxiv.org/pdf/2510.01258v1)**

> **作者:** Nathan Junzi Chen
>
> **备注:** 19 pages, 7 figures
>
> **摘要:** Amidst the rapid normalization of generative artificial intelligence (GAI), intelligent systems have come to dominate political discourse across information mediums. However, internalized political biases stemming from training data skews, human prejudice, and algorithmic flaws continue to plague the novel technology. This paper employs a zero-shot classification approach to evaluate algorithmic political partisanship through a methodical combination of ideological alignment, topicality, response sentiment, and objectivity. A total of 1800 model responses across six mainstream large language models (LLMs) were individually input into four distinct fine-tuned classification algorithms, each responsible for computing an aforementioned bias evaluation metric. Results show an amplified liberal-authoritarian alignment across all six LLMs evaluated, with notable instances of reasoning supersessions and canned refusals. The study subsequently highlights the psychological influences underpinning human-computer interactions and how intrinsic biases can permeate public discourse. The resulting distortion of the political landscape can ultimately manifest as conformity or polarization, depending on a region's pre-existing socio-political structures.
>
---
#### [new 016] Uncovering Implicit Bias in Large Language Models with Concept Learning Dataset
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在检测大语言模型中的隐性偏见。通过构建概念学习数据集，进行上下文概念学习实验，发现模型对量化词存在向上单调性偏见。**

- **链接: [http://arxiv.org/pdf/2510.01219v1](http://arxiv.org/pdf/2510.01219v1)**

> **作者:** Leroy Z. Wang
>
> **摘要:** We introduce a dataset of concept learning tasks that helps uncover implicit biases in large language models. Using in-context concept learning experiments, we found that language models may have a bias toward upward monotonicity in quantifiers; such bias is less apparent when the model is tested by direct prompting without concept learning components. This demonstrates that in-context concept learning can be an effective way to discover hidden biases in language models.
>
---
#### [new 017] GemDetox at TextDetox CLEF 2025: Enhancing a Massively Multilingual Model for Text Detoxification on Low-resource Languages
- **分类: cs.CL**

- **简介: 该论文属于多语言文本净化任务，旨在提升低资源语言的毒性文本转换效果。通过优化大型多语言模型，结合数据增强与提示技术，实现高效准确的中性化改写。**

- **链接: [http://arxiv.org/pdf/2510.01250v1](http://arxiv.org/pdf/2510.01250v1)**

> **作者:** Trung Duc Anh Dang; Ferdinando Pio D'Elia
>
> **摘要:** As social-media platforms emerge and evolve faster than the regulations meant to oversee them, automated detoxification might serve as a timely tool for moderators to enforce safe discourse at scale. We here describe our submission to the PAN 2025 Multilingual Text Detoxification Challenge, which rewrites toxic single-sentence inputs into neutral paraphrases across 15 typologically diverse languages. Building on a 12B-parameter Gemma-3 multilingual transformer, we apply parameter-efficient LoRA SFT fine-tuning and prompting techniques like few-shot and Chain-of-Thought. Our multilingual training corpus combines 3,600 human-authored parallel pairs, 21,600 machine-translated synthetic pairs, and model-generated pairs filtered by Jaccard thresholds. At inference, inputs are enriched with three LaBSE-retrieved neighbors and explicit toxic-span annotations. Evaluated via Style Transfer Accuracy, LaBSE-based semantic preservation, and xCOMET fluency, our system ranks first on high-resource and low-resource languages. Ablations show +0.081 joint score increase from few-shot examples and +0.088 from basic CoT prompting. ANOVA analysis identifies language resource status as the strongest predictor of performance ($\eta^2$ = 0.667, p < 0.01).
>
---
#### [new 018] Evaluation Sheet for Deep Research: A Use Case for Academic Survey Writing
- **分类: cs.CL**

- **简介: 该论文属于评估任务，旨在解决Deep Research工具的评价标准问题。通过设计评估表，对学术综述生成进行评估，揭示了搜索工具与独立研究工具的差距。**

- **链接: [http://arxiv.org/pdf/2510.01283v1](http://arxiv.org/pdf/2510.01283v1)**

> **作者:** Israel Abebe Azime; Tadesse Destaw Belay; Atnafu Lambebo Tonja
>
> **摘要:** Large Language Models (LLMs) powered with argentic capabilities are able to do knowledge-intensive tasks without human involvement. A prime example of this tool is Deep research with the capability to browse the web, extract information and generate multi-page reports. In this work, we introduce an evaluation sheet that can be used for assessing the capability of Deep Research tools. In addition, we selected academic survey writing as a use case task and evaluated output reports based on the evaluation sheet we introduced. Our findings show the need to have carefully crafted evaluation standards. The evaluation done on OpenAI`s Deep Search and Google's Deep Search in generating an academic survey showed the huge gap between search engines and standalone Deep Research tools, the shortcoming in representing the targeted area.
>
---
#### [new 019] Inverse Language Modeling towards Robust and Grounded LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在提升大语言模型的鲁棒性和安全性。通过提出逆语言建模（ILM），同时增强模型对输入扰动的鲁棒性并实现原生接地，解决当前防御机制分散的问题。**

- **链接: [http://arxiv.org/pdf/2510.01929v1](http://arxiv.org/pdf/2510.01929v1)**

> **作者:** Davide Gabrielli; Simone Sestito; Iacopo Masi
>
> **摘要:** The current landscape of defensive mechanisms for LLMs is fragmented and underdeveloped, unlike prior work on classifiers. To further promote adversarial robustness in LLMs, we propose Inverse Language Modeling (ILM), a unified framework that simultaneously 1) improves the robustness of LLMs to input perturbations, and, at the same time, 2) enables native grounding by inverting model outputs to identify potentially toxic or unsafe input triggers. ILM transforms LLMs from static generators into analyzable and robust systems, potentially helping RED teaming. ILM can lay the foundation for next-generation LLMs that are not only robust and grounded but also fundamentally more controllable and trustworthy. The code is publicly available at github.com/davegabe/pag-llm.
>
---
#### [new 020] Silent Tokens, Loud Effects: Padding in LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究LLMs中填充token的影响。针对填充导致的计算偏差问题，通过实验分析其对模型表现的影响，揭示填充并非无害，需谨慎处理。**

- **链接: [http://arxiv.org/pdf/2510.01238v1](http://arxiv.org/pdf/2510.01238v1)**

> **作者:** Rom Himelstein; Amit LeVi; Yonatan Belinkov; Avi Mendelson
>
> **备注:** NeurIPS 2025 Workshop: LLM Evaluation
>
> **摘要:** Padding tokens are widely used in large language models (LLMs) to equalize sequence lengths during batched inference. While they should be fully masked, implementation errors can cause them to influence computation, and the extent of this influence is not well understood. We systematically study this effect across three open-source model families (Llama, Gemma, Qwen), inserting controlled amounts of padding and evaluating outcomes along four axes: activations, generation quality, bias, and safety. Even small amounts of padding shift hidden representations, degrade quality in smaller models, alter bias in unpredictable ways, and weaken safety guardrails. These findings demonstrate that padding is not a harmless detail but a robustness risk that must be carefully handled in deployment.
>
---
#### [new 021] A Comparative Analysis of Sparse Autoencoder and Activation Difference in Language Model Steering
- **分类: cs.CL**

- **简介: 该论文属于语言模型调控任务，旨在解决SAE在语义控制中的不足。通过聚焦top-1隐层和引入衰减策略，提升推理性能并优于均值激活差方法。**

- **链接: [http://arxiv.org/pdf/2510.01246v1](http://arxiv.org/pdf/2510.01246v1)**

> **作者:** Jiaqing Xie
>
> **备注:** 25 pages
>
> **摘要:** Sparse autoencoders (SAEs) have recently emerged as a powerful tool for language model steering. Prior work has explored top-k SAE latents for steering, but we observe that many dimensions among the top-k latents capture non-semantic features such as punctuation rather than semantic attributes like instructions. To address this, we propose focusing on a single, most relevant SAE latent (top-1), eliminating redundant features. We further identify a limitation in constant SAE steering, which often produces degenerate outputs such as repetitive single words. To mitigate this, we introduce a token-wise decaying steering strategy, enabling more faithful comparisons with mean activation difference baselines. Empirically, we show that steering an SAE latent associated with reasoning reliably elicits step-by-step mathematical reasoning and enhances inference quality, functionally resembling the effect of appending a guiding token. Our results demonstrate that SAEs outperform mean activation difference methods on mathematical reasoning benchmarks and match their performance on IF-Eval.
>
---
#### [new 022] Trustworthy Summarization via Uncertainty Quantification and Risk Awareness in Large Language Models
- **分类: cs.CL; cs.AI; stat.ML**

- **简介: 该论文属于自动摘要任务，旨在提升高风险场景下的摘要可靠性。通过引入不确定性量化和风险感知机制，增强摘要的可信度与准确性。**

- **链接: [http://arxiv.org/pdf/2510.01231v1](http://arxiv.org/pdf/2510.01231v1)**

> **作者:** Shuaidong Pan; Di Wu
>
> **摘要:** This study addresses the reliability of automatic summarization in high-risk scenarios and proposes a large language model framework that integrates uncertainty quantification and risk-aware mechanisms. Starting from the demands of information overload and high-risk decision-making, a conditional generation-based summarization model is constructed, and Bayesian inference is introduced during generation to model uncertainty in the parameter space, which helps avoid overconfident predictions. The uncertainty level of the generated content is measured using predictive distribution entropy, and a joint optimization of entropy regularization and risk-aware loss is applied to ensure that key information is preserved and risk attributes are explicitly expressed during information compression. On this basis, the model incorporates risk scoring and regulation modules, allowing summaries to cover the core content accurately while enhancing trustworthiness through explicit risk-level prompts. Comparative experiments and sensitivity analyses verify that the proposed method significantly improves the robustness and reliability of summarization in high-risk applications while maintaining fluency and semantic integrity. This research provides a systematic solution for trustworthy summarization and demonstrates both scalability and practical value at the methodological level.
>
---
#### [new 023] Computational Social Linguistics for Telugu Cultural Preservation: Novel Algorithms for Chandassu Metrical Pattern Recognition
- **分类: cs.CL; 68T50, 68T05, 68U35; I.2.7; J.5; H.3.1**

- **简介: 该论文属于文化保护任务，旨在解决Telugu Chandassu诗律模式识别问题，通过开发算法工具实现文化知识的数字化保存与传承。**

- **链接: [http://arxiv.org/pdf/2510.01233v1](http://arxiv.org/pdf/2510.01233v1)**

> **作者:** Boddu Sri Pavan; Boddu Swathi Sree
>
> **备注:** 16 pages, 4 figures
>
> **摘要:** This research presents a computational social science approach to preserving Telugu Chandassu, the metrical poetry tradition representing centuries of collective cultural intelligence. We develop the first comprehensive digital framework for analyzing Telugu prosodic patterns, bridging traditional community knowledge with modern computational methods. Our social computing approach involves collaborative dataset creation of 4,651 annotated padyams, expert-validated linguistic patterns, and culturally-informed algorithmic design. The framework includes AksharamTokenizer for prosody-aware tokenization, LaghuvuGuruvu Generator for classifying light and heavy syllables, and PadyaBhedam Checker for automated pattern recognition. Our algorithm achieves 91.73% accuracy on the proposed Chandassu Score, with evaluation metrics reflecting traditional literary standards. This work demonstrates how computational social science can preserve endangered cultural knowledge systems while enabling new forms of collective intelligence around literary heritage. The methodology offers insights for community-centered approaches to cultural preservation, supporting broader initiatives in digital humanities and socially-aware computing systems.
>
---
#### [new 024] Learning to Look at the Other Side: A Semantic Probing Study of Word Embeddings in LLMs with Enabled Bidirectional Attention
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs在文本嵌入和语义分析中的局限性。通过启用双向注意力机制和对比学习，提升模型的语义表示能力。**

- **链接: [http://arxiv.org/pdf/2510.01652v1](http://arxiv.org/pdf/2510.01652v1)**

> **作者:** Zhaoxin Feng; Jianfei Ma; Emmanuele Chersoni; Xiaojing Zhao; Xiaoyi Bao
>
> **摘要:** Autoregressive Large Language Models (LLMs) demonstrate exceptional performance in language understanding and generation. However, their application in text embedding tasks has been relatively slow, along with the analysis of their semantic representation in probing tasks, due to the constraints of the unidirectional attention mechanism. This paper aims to explore whether such constraints can be overcome by enabling bidirectional attention in LLMs. We tested different variants of the Llama architecture through additional training steps, progressively enabling bidirectional attention and unsupervised/supervised contrastive learning.
>
---
#### [new 025] AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **简介: 该论文属于文本检测任务，旨在区分人类与大语言模型生成的文本。提出AdaDetectGPT方法，通过自适应学习提升检测效果，并提供统计保障。**

- **链接: [http://arxiv.org/pdf/2510.01268v1](http://arxiv.org/pdf/2510.01268v1)**

> **作者:** Hongyi Zhou; Jin Zhu; Pingfan Su; Kai Ye; Ying Yang; Shakeel A O B Gavioli-Akilagun; Chengchun Shi
>
> **备注:** Accepted by NeurIPS2025
>
> **摘要:** We study the problem of determining whether a piece of text has been authored by a human or by a large language model (LLM). Existing state of the art logits-based detectors make use of statistics derived from the log-probability of the observed text evaluated using the distribution function of a given source LLM. However, relying solely on log probabilities can be sub-optimal. In response, we introduce AdaDetectGPT -- a novel classifier that adaptively learns a witness function from training data to enhance the performance of logits-based detectors. We provide statistical guarantees on its true positive rate, false positive rate, true negative rate and false negative rate. Extensive numerical studies show AdaDetectGPT nearly uniformly improves the state-of-the-art method in various combination of datasets and LLMs, and the improvement can reach up to 58%. A python implementation of our method is available at https://github.com/Mamba413/AdaDetectGPT.
>
---
#### [new 026] LOCA: Logical Chain Augmentation for Scientific Corpus Cleaning
- **分类: cs.CL**

- **简介: 该论文属于科学语料清洗任务，旨在解决科学问答数据中的逻辑错误问题。通过LOCA框架增强答案逻辑链，提升数据质量。**

- **链接: [http://arxiv.org/pdf/2510.01249v1](http://arxiv.org/pdf/2510.01249v1)**

> **作者:** You-Le Fang; Dong-Shan Jian; Xiang Li; Ce Meng; Ling-Shi Meng; Chen-Xu Yan; Zhi-Zhang Bian; Yan-Qing Ma
>
> **备注:** 29 pages, 2 figures
>
> **摘要:** While Large Language Models (LLMs) excel in general domains, their reliability often falls short in scientific problem-solving. The advancement of scientific AI depends on large-scale, high-quality corpora. However, existing scientific question-answering (QA) datasets suffer from high error rates, frequently resulting from logical leaps and implicit reasoning within the answers. To address this issue, we introduce LOCA (Logical Chain Augmentation), a novel framework for automatically cleaning scientific corpora, implemented through an augment-and-review loop. At its core, LOCA enhances raw answers by completing missing logical steps and explicitly separating the underlying scientific principle from its subsequent derivation. By applying LOCA to challenging scientific corpora, we demonstrate that it can automatically filter noisy datasets, typically reducing the error rate from as high as 20\% to below 2\%. LOCA provides a scalable and effective methodology for creating high-quality scientific corpora, paving the way for more reliable training and evaluation of scientific AI.
>
---
#### [new 027] SKYLENAGE Technical Report: Mathematical Reasoning and Contest-Innovation Benchmarks for Multi-Level Math Evaluation
- **分类: cs.CL**

- **简介: 该论文属于数学推理任务，旨在解决大模型在数学评估中的天花板问题。提出两个基准测试集，评估不同模型表现，提供全面的数学能力评估工具。**

- **链接: [http://arxiv.org/pdf/2510.01241v1](http://arxiv.org/pdf/2510.01241v1)**

> **作者:** Hu Wei; Ze Xu; Boyu Yang; Linlin Miao; Weiqi Zhai; Yihan Li; Zixuan Li; Zhijun Wang; Boya Wang; Jianwei Yu; Jialing Yuan; Xiaoyue Zhang; Cheng He; Minglei Chen; Zifan Zhang; Qianhui Li; Wei Wang; Xiang Xu
>
> **摘要:** Large language models (LLMs) now perform strongly on many public math suites, yet frontier separation within mathematics increasingly suffers from ceiling effects. We present two complementary benchmarks: SKYLENAGE-ReasoningMATH, a 100-item, structure-aware diagnostic set with per-item metadata on length, numeric density, and symbolic complexity; and SKYLENAGE-MATH, a 150-item contest-style suite spanning four stages from high school to doctoral under a seven-subject taxonomy. We evaluate fifteen contemporary LLM variants under a single setup and analyze subject x model and grade x model performance. On the contest suite, the strongest model reaches 44% while the runner-up reaches 37%; accuracy declines from high school to doctoral, and top systems exhibit a doctoral-to-high-school retention near 79%. On the reasoning set, the best model attains 81% overall, and hardest-slice results reveal clear robustness gaps between leaders and the mid-tier. In summary, we release SKYLENAGE-ReasoningMATH and report aggregate results for SKYLENAGE-MATH; together, SKYLENAGE provides a hard, reasoning-centered and broadly covering math benchmark with calibrated difficulty and rich metadata, serving as a reference benchmark for future evaluations of mathematical reasoning.
>
---
#### [new 028] What MLLMs Learn about When they Learn about Multimodal Reasoning: Perception, Reasoning, or their Integration?
- **分类: cs.CL**

- **简介: 该论文属于多模态推理任务，旨在解析模型在感知、推理和整合方面的学习效果。通过构建MathLens基准，分析不同训练方法对各子技能的影响，揭示模型在多模态任务中的优劣势。**

- **链接: [http://arxiv.org/pdf/2510.01719v1](http://arxiv.org/pdf/2510.01719v1)**

> **作者:** Jiwan Chung; Neel Joshi; Pratyusha Sharma; Youngjae Yu; Vibhav Vineet
>
> **摘要:** Multimodal reasoning models have recently shown promise on challenging domains such as olympiad-level geometry, yet their evaluation remains dominated by aggregate accuracy, a single score that obscures where and how models are improving. We introduce MathLens, a benchmark designed to disentangle the subskills of multimodal reasoning while preserving the complexity of textbook-style geometry problems. The benchmark separates performance into three components: Perception: extracting information from raw inputs, Reasoning: operating on available information, and Integration: selecting relevant perceptual evidence and applying it within reasoning. To support each test, we provide annotations: visual diagrams, textual descriptions to evaluate reasoning in isolation, controlled questions that require both modalities, and probes for fine-grained perceptual skills, all derived from symbolic specifications of the problems to ensure consistency and robustness. Our analysis reveals that different training approaches have uneven effects: First, reinforcement learning chiefly strengthens perception, especially when supported by textual supervision, while textual SFT indirectly improves perception through reflective reasoning. Second, reasoning improves only in tandem with perception. Third, integration remains the weakest capacity, with residual errors concentrated there once other skills advance. Finally, robustness diverges: RL improves consistency under diagram variation, whereas multimodal SFT reduces it through overfitting. We will release all data and experimental logs.
>
---
#### [new 029] Detoxifying Large Language Models via Autoregressive Reward Guided Representation Editing
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型生成有毒内容的问题。提出ARGRE框架，通过奖励引导的表示编辑实现高效 detoxification。**

- **链接: [http://arxiv.org/pdf/2510.01243v1](http://arxiv.org/pdf/2510.01243v1)**

> **作者:** Yisong Xiao; Aishan Liu; Siyuan Liang; Zonghao Ying; Xianglong Liu; Dacheng Tao
>
> **备注:** Accepted to NeurIPS 25
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive performance across various tasks, yet they remain vulnerable to generating toxic content, necessitating detoxification strategies to ensure safe and responsible deployment. Test-time detoxification methods, which typically introduce static or dynamic interventions into LLM representations, offer a promising solution due to their flexibility and minimal invasiveness. However, current approaches often suffer from imprecise interventions, primarily due to their insufficient exploration of the transition space between toxic and non-toxic outputs. To address this challenge, we propose \textsc{A}utoregressive \textsc{R}eward \textsc{G}uided \textsc{R}epresentation \textsc{E}diting (ARGRE), a novel test-time detoxification framework that explicitly models toxicity transitions within the latent representation space, enabling stable and precise reward-guided editing. ARGRE identifies non-toxic semantic directions and interpolates between toxic and non-toxic representations to reveal fine-grained transition trajectories. These trajectories transform sparse toxicity annotations into dense training signals, enabling the construction of an autoregressive reward model that delivers stable and precise editing guidance. At inference, the reward model guides an adaptive two-step editing process to obtain detoxified representations: it first performs directional steering based on expected reward gaps to shift representations toward non-toxic regions, followed by lightweight gradient-based refinements. Extensive experiments across 8 widely used LLMs show that ARGRE significantly outperforms leading baselines in effectiveness (-62.21% toxicity) and efficiency (-47.58% inference time), while preserving the core capabilities of the original model with minimal degradation. Our code is available at the website.
>
---
#### [new 030] SeMob: Semantic Synthesis for Dynamic Urban Mobility Prediction
- **分类: cs.CL**

- **简介: 该论文属于城市交通预测任务，解决外部事件导致的移动性突变问题。通过融合文本与时空数据，提出SeMob框架提升预测精度。**

- **链接: [http://arxiv.org/pdf/2510.01245v1](http://arxiv.org/pdf/2510.01245v1)**

> **作者:** Runfei Chen; Shuyang Jiang; Wei Huang
>
> **备注:** EMNLP2025
>
> **摘要:** Human mobility prediction is vital for urban services, but often fails to account for abrupt changes from external events. Existing spatiotemporal models struggle to leverage textual descriptions detailing these events. We propose SeMob, an LLM-powered semantic synthesis pipeline for dynamic mobility prediction. Specifically, SeMob employs a multi-agent framework where LLM-based agents automatically extract and reason about spatiotemporally related text from complex online texts. Fine-grained relevant contexts are then incorporated with spatiotemporal data through our proposed innovative progressive fusion architecture. The rich pre-trained event prior contributes enriched insights about event-driven prediction, and hence results in a more aligned forecasting model. Evaluated on a dataset constructed through our pipeline, SeMob achieves maximal reductions of 13.92% in MAE and 11.12% in RMSE compared to the spatiotemporal model. Notably, the framework exhibits pronounced superiority especially within spatiotemporal regions close to an event's location and time of occurrence.
>
---
#### [new 031] Do Bias Benchmarks Generalise? Evidence from Voice-based Evaluation of Gender Bias in SpeechLLMs
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音大模型性别偏见评估任务，旨在检验MCQA基准是否具备跨任务泛化能力。通过微调模型并测试其在不同任务中的表现，发现现有基准难以预测其他任务结果。**

- **链接: [http://arxiv.org/pdf/2510.01254v1](http://arxiv.org/pdf/2510.01254v1)**

> **作者:** Shree Harsha Bokkahalli Satish; Gustav Eje Henter; Éva Székely
>
> **备注:** 5 pages, 2 Figures, Submitted to IEEE ICASSP 2026
>
> **摘要:** Recent work in benchmarking bias and fairness in speech large language models (SpeechLLMs) has relied heavily on multiple-choice question answering (MCQA) formats. The model is tasked to choose between stereotypical, anti-stereotypical, or neutral/irrelevant answers given an input speech prompt and an optional text prompt. Such MCQA benchmarks implicitly assume that model performance is consistent across other MCQA tasks, voices, and other task formats such as more realistic, long-form evaluations. In this paper, we probe that assumption. We fine-tune three SpeechLLMs using LoRA adapters to induce specific MCQA behaviours: preference for stereotypical, anti-stereotypical, or neutral/uncertain answers. We then evaluate whether these behaviours generalise to another, distinct MCQA benchmark, and more critically to long-form, creative generation tasks. Our results show that performance on MCQA bias benchmarks fails to reliably predict performances across other MCQA benchmarks, and more importantly across long-form tasks. We conclude that current MCQA bias benchmarks show limited evidence of cross-task generalisation in the speech domain, and also propose an evaluation suite for measuring behaviour transferability in future models and benchmarks.
>
---
#### [new 032] Confidence-Aware Routing for Large Language Model Reliability Enhancement: A Multi-Signal Approach to Pre-Generation Hallucination Mitigation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型的幻觉问题。通过多信号融合的置信度评估系统，主动降低不可靠生成，提升模型可靠性。**

- **链接: [http://arxiv.org/pdf/2510.01237v1](http://arxiv.org/pdf/2510.01237v1)**

> **作者:** Nandakishor M
>
> **摘要:** Large Language Models suffer from hallucination, generating plausible yet factually incorrect content. Current mitigation strategies focus on post-generation correction, which is computationally expensive and fails to prevent unreliable content generation. We propose a confidence-aware routing system that proactively assesses model uncertainty before generation and redirects queries based on estimated reliability. Our approach combines three complementary signals: semantic alignment between internal representations and reference embeddings, internal convergence analysis across model layers, and learned confidence estimation. The unified confidence score determines routing to four pathways: local generation for high confidence, retrieval-augmented generation for medium confidence, larger models for low confidence, and human review for very low confidence. Evaluation on knowledge-intensive QA benchmarks demonstrates significant improvements in hallucination detection (0.74 vs. 0.42 baseline) while reducing computational costs by 40% compared to post-hoc methods. The F1 score improves from 0.61 to 0.82 with low false positive rates (0.09). This paradigm shift from reactive correction to proactive assessment offers a computationally efficient approach to LLM reliability enhancement.
>
---
#### [new 033] Towards Open-Ended Discovery for Low-Resource NLP
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于低资源自然语言处理任务，旨在解决数据匮乏问题。提出通过人机交互动态学习语言，构建参与式学习框架，促进语言多样性保护与技术发展。**

- **链接: [http://arxiv.org/pdf/2510.01220v1](http://arxiv.org/pdf/2510.01220v1)**

> **作者:** Bonaventure F. P. Dossou; Henri Aïdasso
>
> **备注:** Proceedings of the 2nd Workshop on Uncertainty-Aware NLP (UncertaiNLP) at EMNLP 2025
>
> **摘要:** Natural Language Processing (NLP) for low-resource languages remains fundamentally constrained by the lack of textual corpora, standardized orthographies, and scalable annotation pipelines. While recent advances in large language models have improved cross-lingual transfer, they remain inaccessible to underrepresented communities due to their reliance on massive, pre-collected data and centralized infrastructure. In this position paper, we argue for a paradigm shift toward open-ended, interactive language discovery, where AI systems learn new languages dynamically through dialogue rather than static datasets. We contend that the future of language technology, particularly for low-resource and under-documented languages, must move beyond static data collection pipelines toward interactive, uncertainty-driven discovery, where learning emerges dynamically from human-machine collaboration instead of being limited to pre-existing datasets. We propose a framework grounded in joint human-machine uncertainty, combining epistemic uncertainty from the model with hesitation cues and confidence signals from human speakers to guide interaction, query selection, and memory retention. This paper is a call to action: we advocate a rethinking of how AI engages with human knowledge in under-documented languages, moving from extractive data collection toward participatory, co-adaptive learning processes that respect and empower communities while discovering and preserving the world's linguistic diversity. This vision aligns with principles of human-centered AI, emphasizing interactive, cooperative model building between AI systems and speakers.
>
---
#### [new 034] REPAIR: Robust Editing via Progressive Adaptive Intervention and Reintegration
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型的后训练优化任务，旨在解决知识更新成本高和副作用问题。提出REPAIR框架，实现高效、精准的模型编辑与知识保留。**

- **链接: [http://arxiv.org/pdf/2510.01879v1](http://arxiv.org/pdf/2510.01879v1)**

> **作者:** Yisu Wang; Ming Wang; Haoyuan Song; Wenjie Huang; Chaozheng Wang; Yi Xie; Xuming Ran
>
> **摘要:** Post-training for large language models (LLMs) is constrained by the high cost of acquiring new knowledge or correcting errors and by the unintended side effects that frequently arise from retraining. To address these issues, we introduce REPAIR (Robust Editing via Progressive Adaptive Intervention and Reintegration), a lifelong editing framework designed to support precise and low-cost model updates while preserving non-target knowledge. REPAIR mitigates the instability and conflicts of large-scale sequential edits through a closed-loop feedback mechanism coupled with dynamic memory management. Furthermore, by incorporating frequent knowledge fusion and enforcing strong locality guards, REPAIR effectively addresses the shortcomings of traditional distribution-agnostic approaches that often overlook unintended ripple effects. Our experiments demonstrate that REPAIR boosts editing accuracy by 10%-30% across multiple model families and significantly reduces knowledge forgetting. This work introduces a robust framework for developing reliable, scalable, and continually evolving LLMs.
>
---
#### [new 035] SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs
- **分类: cs.CL**

- **简介: 该论文属于图学习任务，旨在解决跨域迁移和数据标注依赖问题。提出SSTAG方法，结合文本与图结构，提升模型泛化能力和效率。**

- **链接: [http://arxiv.org/pdf/2510.01248v1](http://arxiv.org/pdf/2510.01248v1)**

> **作者:** Ruyue Liu; Rong Yin; Xiangzhen Bo; Xiaoshuai Hao; Yong Liu; Jinwen Zhong; Can Ma; Weiping Wang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Large scale pretrained models have revolutionized Natural Language Processing (NLP) and Computer Vision (CV), showcasing remarkable cross domain generalization abilities. However, in graph learning, models are typically trained on individual graph datasets, limiting their capacity to transfer knowledge across different graphs and tasks. This approach also heavily relies on large volumes of annotated data, which presents a significant challenge in resource-constrained settings. Unlike NLP and CV, graph structured data presents unique challenges due to its inherent heterogeneity, including domain specific feature spaces and structural diversity across various applications. To address these challenges, we propose a novel structure aware self supervised learning method for Text Attributed Graphs (SSTAG). By leveraging text as a unified representation medium for graph learning, SSTAG bridges the gap between the semantic reasoning of Large Language Models (LLMs) and the structural modeling capabilities of Graph Neural Networks (GNNs). Our approach introduces a dual knowledge distillation framework that co-distills both LLMs and GNNs into structure-aware multilayer perceptrons (MLPs), enhancing the scalability of large-scale TAGs. Additionally, we introduce an in-memory mechanism that stores typical graph representations, aligning them with memory anchors in an in-memory repository to integrate invariant knowledge, thereby improving the model's generalization ability. Extensive experiments demonstrate that SSTAG outperforms state-of-the-art models on cross-domain transfer learning tasks, achieves exceptional scalability, and reduces inference costs while maintaining competitive performance.
>
---
#### [new 036] Can LLMs Refuse Questions They Do Not Know? Measuring Knowledge-Aware Refusal in Factual Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决LLM在事实任务中无法准确拒绝未知问题的问题。提出Refusal Index（RI）衡量模型的知识感知拒绝能力。**

- **链接: [http://arxiv.org/pdf/2510.01782v1](http://arxiv.org/pdf/2510.01782v1)**

> **作者:** Wenbo Pan; Jie Xu; Qiguang Chen; Junhao Dong; Libo Qin; Xinfeng Li; Haining Yu; Xiaohua Jia
>
> **摘要:** Large Language Models (LLMs) should refuse to answer questions beyond their knowledge. This capability, which we term knowledge-aware refusal, is crucial for factual reliability. However, existing metrics fail to faithfully measure this ability. On the one hand, simple refusal-based metrics are biased by refusal rates and yield inconsistent scores when models exhibit different refusal tendencies. On the other hand, existing calibration metrics are proxy-based, capturing the performance of auxiliary calibration processes rather than the model's actual refusal behavior. In this work, we propose the Refusal Index (RI), a principled metric that measures how accurately LLMs refuse questions they do not know. We define RI as Spearman's rank correlation between refusal probability and error probability. To make RI practically measurable, we design a lightweight two-pass evaluation method that efficiently estimates RI from observed refusal rates across two standard evaluation runs. Extensive experiments across 16 models and 5 datasets demonstrate that RI accurately quantifies a model's intrinsic knowledge-aware refusal capability in factual tasks. Notably, RI remains stable across different refusal rates and provides consistent model rankings independent of a model's overall accuracy and refusal rates. More importantly, RI provides insight into an important but previously overlooked aspect of LLM factuality: while LLMs achieve high accuracy on factual tasks, their refusal behavior can be unreliable and fragile. This finding highlights the need to complement traditional accuracy metrics with the Refusal Index for comprehensive factuality evaluation.
>
---
#### [new 037] Stream RAG: Instant and Accurate Spoken Dialogue Systems with Streaming Tool Usage
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音对话系统任务，解决端到端系统易幻觉的问题。通过引入流式工具调用和检索增强生成，提升准确性和响应速度。**

- **链接: [http://arxiv.org/pdf/2510.02044v1](http://arxiv.org/pdf/2510.02044v1)**

> **作者:** Siddhant Arora; Haidar Khan; Kai Sun; Xin Luna Dong; Sajal Choudhary; Seungwhan Moon; Xinyuan Zhang; Adithya Sagar; Surya Teja Appini; Kaushik Patnaik; Sanat Sharma; Shinji Watanabe; Anuj Kumar; Ahmed Aly; Yue Liu; Florian Metze; Zhaojiang Lin
>
> **摘要:** End-to-end speech-in speech-out dialogue systems are emerging as a powerful alternative to traditional ASR-LLM-TTS pipelines, generating more natural, expressive responses with significantly lower latency. However, these systems remain prone to hallucinations due to limited factual grounding. While text-based dialogue systems address this challenge by integrating tools such as web search and knowledge graph APIs, we introduce the first approach to extend tool use directly into speech-in speech-out systems. A key challenge is that tool integration substantially increases response latency, disrupting conversational flow. To mitigate this, we propose Streaming Retrieval-Augmented Generation (Streaming RAG), a novel framework that reduces user-perceived latency by predicting tool queries in parallel with user speech, even before the user finishes speaking. Specifically, we develop a post-training pipeline that teaches the model when to issue tool calls during ongoing speech and how to generate spoken summaries that fuse audio queries with retrieved text results, thereby improving both accuracy and responsiveness. To evaluate our approach, we construct AudioCRAG, a benchmark created by converting queries from the publicly available CRAG dataset into speech form. Experimental results demonstrate that our streaming RAG approach increases QA accuracy by up to 200% relative (from 11.1% to 34.2% absolute) and further enhances user experience by reducing tool use latency by 20%. Importantly, our streaming RAG approach is modality-agnostic and can be applied equally to typed input, paving the way for more agentic, real-time AI assistants.
>
---
#### [new 038] Discourse vs emissions: Analysis of corporate narratives, symbolic practices, and mimicry through LLMs
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 该论文属于ESG分析任务，旨在解决企业气候披露真实性问题。通过LLMs分析828家公司的报告，识别叙事特征与排放数据的关系，揭示模仿行为及承诺与行动的脱节。**

- **链接: [http://arxiv.org/pdf/2510.01222v1](http://arxiv.org/pdf/2510.01222v1)**

> **作者:** Bertrand Kian Hassani; Yacoub Bahini; Rizwan Mushtaq
>
> **摘要:** Climate change has increased demands for transparent and comparable corporate climate disclosures, yet imitation and symbolic reporting often undermine their value. This paper develops a multidimensional framework to assess disclosure maturity among 828 U.S.listed firms using large language models (LLMs) fine-tuned for climate communication. Four classifiers-sentiment, commitment, specificity, and target ambition-extract narrative indicators from sustainability and annual reports, which are linked to firm attributes such as emissions, market capitalization, and sector. Analyses reveal three insights: (1) risk-focused narratives often align with explicit commitments, but quantitative targets (e.g., net-zero pledges) remain decoupled from tone; (2) larger and higher-emitting firms disclose more commitments and actions than peers, though inconsistently with quantitative targets; and (3) widespread similarity in disclosure styles suggests mimetic behavior, reducing differentiation and decision usefulness. These results highlight the value of LLMs for ESG narrative analysis and the need for stronger regulation to connect commitments with verifiable transition strategies.
>
---
#### [new 039] LLM-Based Multi-Task Bangla Hate Speech Detection: Type, Severity, and Target
- **分类: cs.CL; 68T50; F.2.2; I.2.7**

- **简介: 该论文属于多任务 Bangla 垃圾信息检测任务，解决 hate speech 的类型、严重程度和目标识别问题。构建了首个多任务 Bangla 数据集，并对比了不同模型的性能。**

- **链接: [http://arxiv.org/pdf/2510.01995v1](http://arxiv.org/pdf/2510.01995v1)**

> **作者:** Md Arid Hasan; Firoj Alam; Md Fahad Hossain; Usman Naseem; Syed Ishtiaque Ahmed
>
> **摘要:** Online social media platforms are central to everyday communication and information seeking. While these platforms serve positive purposes, they also provide fertile ground for the spread of hate speech, offensive language, and bullying content targeting individuals, organizations, and communities. Such content undermines safety, participation, and equity online. Reliable detection systems are therefore needed, especially for low-resource languages where moderation tools are limited. In Bangla, prior work has contributed resources and models, but most are single-task (e.g., binary hate/offense) with limited coverage of multi-facet signals (type, severity, target). We address these gaps by introducing the first multi-task Bangla hate-speech dataset, BanglaMultiHate, one of the largest manually annotated corpus to date. Building on this resource, we conduct a comprehensive, controlled comparison spanning classical baselines, monolingual pretrained models, and LLMs under zero-shot prompting and LoRA fine-tuning. Our experiments assess LLM adaptability in a low-resource setting and reveal a consistent trend: although LoRA-tuned LLMs are competitive with BanglaBERT, culturally and linguistically grounded pretraining remains critical for robust performance. Together, our dataset and findings establish a stronger benchmark for developing culturally aligned moderation tools in low-resource contexts. For reproducibility, we will release the dataset and all related scripts.
>
---
#### [new 040] Parallel Scaling Law: Unveiling Reasoning Generalization through A Cross-Linguistic Perspective
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究LRM在不同语言间的推理泛化问题。通过跨语言实验，发现英语模型在其他语言上表现不佳，并提出并行训练规律以提升多语言推理能力。**

- **链接: [http://arxiv.org/pdf/2510.02272v1](http://arxiv.org/pdf/2510.02272v1)**

> **作者:** Wen Yang; Junhong Wu; Chong Li; Chengqing Zong; Jiajun Zhang
>
> **备注:** Work in progress
>
> **摘要:** Recent advancements in Reinforcement Post-Training (RPT) have significantly enhanced the capabilities of Large Reasoning Models (LRMs), sparking increased interest in the generalization of RL-based reasoning. While existing work has primarily focused on investigating its generalization across tasks or modalities, this study proposes a novel cross-linguistic perspective to investigate reasoning generalization. This raises a crucial question: $\textit{Does the reasoning capability achieved from English RPT effectively transfer to other languages?}$ We address this by systematically evaluating English-centric LRMs on multilingual reasoning benchmarks and introducing a metric to quantify cross-lingual transferability. Our findings reveal that cross-lingual transferability varies significantly across initial model, target language, and training paradigm. Through interventional studies, we find that models with stronger initial English capabilities tend to over-rely on English-specific patterns, leading to diminished cross-lingual generalization. To address this, we conduct a thorough parallel training study. Experimental results yield three key findings: $\textbf{First-Parallel Leap}$, a substantial leap in performance when transitioning from monolingual to just a single parallel language, and a predictable $\textbf{Parallel Scaling Law}$, revealing that cross-lingual reasoning transfer follows a power-law with the number of training parallel languages. Moreover, we identify the discrepancy between actual monolingual performance and the power-law prediction as $\textbf{Monolingual Generalization Gap}$, indicating that English-centric LRMs fail to fully generalize across languages. Our study challenges the assumption that LRM reasoning mirrors human cognition, providing critical insights for the development of more language-agnostic LRMs.
>
---
#### [new 041] In AI Sweet Harmony: Sociopragmatic Guardrail Bypasses and Evaluation-Awareness in OpenAI gpt-oss-20b
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文研究AI模型在特定指令下的拒绝行为，探讨语言框架与安全策略的影响，测试多个风险领域，并提出改进方法以提高安全性。**

- **链接: [http://arxiv.org/pdf/2510.01259v1](http://arxiv.org/pdf/2510.01259v1)**

> **作者:** Nils Durner
>
> **备注:** 27 pages, 1 figure
>
> **摘要:** We probe OpenAI's open-weights 20-billion-parameter model gpt-oss-20b to study how sociopragmatic framing, language choice, and instruction hierarchy affect refusal behavior. Across 80 seeded iterations per scenario, we test several harm domains including ZIP-bomb construction (cyber threat), synthetic card-number generation, minor-unsafe driving advice, drug-precursor indicators, and RAG context exfiltration. Composite prompts that combine an educator persona, a safety-pretext ("what to avoid"), and step-cue phrasing flip assistance rates from 0% to 97.5% on a ZIP-bomb task. On our grid, formal registers in German and French are often leakier than matched English prompts. A "Linux terminal" role-play overrides a developer rule not to reveal context in a majority of runs with a naive developer prompt, and we introduce an AI-assisted hardening method that reduces leakage to 0% in several user-prompt variants. We further test evaluation awareness with a paired-track design and measure frame-conditioned differences between matched "helpfulness" and "harmfulness" evaluation prompts; we observe inconsistent assistance in 13% of pairs. Finally, we find that the OpenAI Moderation API under-captures materially helpful outputs relative to a semantic grader, and that refusal rates differ by 5 to 10 percentage points across inference stacks, raising reproducibility concerns. We release prompts, seeds, outputs, and code for reproducible auditing at https://github.com/ndurner/gpt-oss-rt-run .
>
---
#### [new 042] Explore Briefly, Then Decide: Mitigating LLM Overthinking via Cumulative Entropy Regulation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决LLM过思考问题。通过引入TECA和CER机制，使模型动态决定推理终止点，提升效率。**

- **链接: [http://arxiv.org/pdf/2510.02249v1](http://arxiv.org/pdf/2510.02249v1)**

> **作者:** Tianyi Jiang; Yi Bin; Yujuan Ding; Kainian Zhu; Fei Ma; Jingkuan Song; Heng Tao Shen
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable reasoning abilities on complex problems using long Chain-of-Thought (CoT) reasoning. However, they often suffer from overthinking, meaning generating unnecessarily lengthy reasoning steps for simpler problems. This issue may degrade the efficiency of the models and make them difficult to adapt the reasoning depth to the complexity of problems. To address this, we introduce a novel metric Token Entropy Cumulative Average (TECA), which measures the extent of exploration throughout the reasoning process. We further propose a novel reasoning paradigm -- Explore Briefly, Then Decide -- with an associated Cumulative Entropy Regulation (CER) mechanism. This paradigm leverages TECA to help the model dynamically determine the optimal point to conclude its thought process and provide a final answer, thus achieving efficient reasoning. Experimental results across diverse mathematical benchmarks show that our approach substantially mitigates overthinking without sacrificing problem-solving ability. With our thinking paradigm, the average response length decreases by up to 71% on simpler datasets, demonstrating the effectiveness of our method in creating a more efficient and adaptive reasoning process.
>
---
#### [new 043] Efficient Training of Robust Traditional Chinese LLaMA-1B on a Single Consumer GPU: Continual Pre-training, SFT, and DPO
- **分类: cs.CL**

- **简介: 该论文针对传统中文语言模型的稳定性问题，提出一种三阶段优化方法，提升模型在单块消费级GPU上的鲁棒性与语言纯度。**

- **链接: [http://arxiv.org/pdf/2510.01616v1](http://arxiv.org/pdf/2510.01616v1)**

> **作者:** Yu-Cheng Chih; Ming-Tao Duan; Yong-Hao Hou
>
> **备注:** 17 pages, 1 figures, 2 tables. Technical report. Introduces PureTC-1B, an adapter-based pipeline for stabilizing Small Language Models in Traditional Chinese using CPT, SFT, and DPO
>
> **摘要:** Small Language Models (SLMs) enable cost-effective, on-device and latency-sensitive AI applications, yet their deployment in Traditional Chinese (TC) remains hindered by token-level instability - models unpredictably emit non-TC characters or code-switch into other languages. We address this practical reliability gap by creating PureTC-1B, a three-stage stabilization pipeline for Llama-3.2-1B-Instruct (an open-weight, instruction-tuned model released by Meta) using parameter-efficient LoRA adapters. Our method combines Continual Pre-Training (CPT) on TC-centric corpora, Supervised Fine-Tuning (SFT) with instruction data, and Direct Preference Optimization (DPO) using TC-adherence preferences to improve monolingual robustness without full-model retraining. On a benchmark designed to simulate real-world usage, PureTC-1B achieves a 51.3% relative reduction (micro-average) in non-TC output tokens versus the base model. On a Named Entity Translation (NET) task, PureTC-1B further reduces incorrect-language tokens by 77.2% relative to Llama-3B and 57.2% relative to Qwen-1.5B, indicating that robust TC adherence is attainable even at the 1B scale. The pipeline is reproducible, adapter-only, and hardware-friendly, offering practitioners a practical recipe to enhance language stability for TC and potentially other non-English languages.
>
---
#### [new 044] Model Merging to Maintain Language-Only Performance in Developmentally Plausible Multimodal Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态语言模型任务，旨在解决多模态模型在语言任务中表现不佳的问题，通过模型融合提升语言能力，同时保持多模态性能。**

- **链接: [http://arxiv.org/pdf/2510.01845v1](http://arxiv.org/pdf/2510.01845v1)**

> **作者:** Ece Takmaz; Lisa Bylinina; Jakub Dotlacil
>
> **备注:** Accepted to the EMNLP 2025 workshop BabyLM: Accelerating language modeling research with cognitively plausible datasets
>
> **摘要:** State-of-the-art vision-and-language models consist of many parameters and learn from enormous datasets, surpassing the amounts of linguistic data that children are exposed to as they acquire a language. This paper presents our approach to the multimodal track of the BabyLM challenge addressing this discrepancy. We develop language-only and multimodal models in low-resource settings using developmentally plausible datasets, with our multimodal models outperforming previous BabyLM baselines. One finding in the multimodal language model literature is that these models tend to underperform in \textit{language-only} tasks. Therefore, we focus on maintaining language-only abilities in multimodal models. To this end, we experiment with \textit{model merging}, where we fuse the parameters of multimodal models with those of language-only models using weighted linear interpolation. Our results corroborate the findings that multimodal models underperform in language-only benchmarks that focus on grammar, and model merging with text-only models can help alleviate this problem to some extent, while maintaining multimodal performance.
>
---
#### [new 045] Exploring Database Normalization Effects on SQL Generation
- **分类: cs.CL**

- **简介: 该论文属于NL2SQL任务，研究数据库规范化对SQL生成的影响。通过实验分析不同规范化程度的模式对模型性能的影响，提出适应性schema选择的重要性。**

- **链接: [http://arxiv.org/pdf/2510.01989v1](http://arxiv.org/pdf/2510.01989v1)**

> **作者:** Ryosuke Kohita
>
> **备注:** Accepted to CIKM 2025
>
> **摘要:** Schema design, particularly normalization, is a critical yet often overlooked factor in natural language to SQL (NL2SQL) systems. Most prior research evaluates models on fixed schemas, overlooking the influence of design on performance. We present the first systematic study of schema normalization's impact, evaluating eight leading large language models on synthetic and real-world datasets with varied normalization levels. We construct controlled synthetic datasets with formal normalization (1NF-3NF) and real academic paper datasets with practical schemes. Our results show that denormalized schemas offer high accuracy on simple retrieval queries, even with cost-effective models in zero-shot settings. In contrast, normalized schemas (2NF/3NF) introduce challenges such as errors in base table selection and join type prediction; however, these issues are substantially mitigated by providing few-shot examples. For aggregation queries, normalized schemas yielded better performance, mainly due to their robustness against the data duplication and NULL value issues that cause errors in denormalized schemas. These findings suggest that the optimal schema design for NL2SQL applications depends on the types of queries to be supported. Our study demonstrates the importance of considering schema design when developing NL2SQL interfaces and integrating adaptive schema selection for real-world scenarios.
>
---
#### [new 046] Chain-of-Thought Reasoning in Streaming Full-Duplex End-to-End Spoken Dialogue Systems
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音对话系统任务，解决传统系统依赖VAD导致的交互不流畅问题。提出SCoT框架，实现低延迟、连续响应的双工对话。**

- **链接: [http://arxiv.org/pdf/2510.02066v1](http://arxiv.org/pdf/2510.02066v1)**

> **作者:** Siddhant Arora; Jinchuan Tian; Hayato Futami; Jiatong Shi; Yosuke Kashiwagi; Emiru Tsunoo; Shinji Watanabe
>
> **摘要:** Most end-to-end (E2E) spoken dialogue systems (SDS) rely on voice activity detection (VAD) for turn-taking, but VAD fails to distinguish between pauses and turn completions. Duplex SDS models address this by predicting output continuously, including silence tokens, thus removing the need for explicit VAD. However, they often have complex dual-channel architecture and lag behind cascaded models in semantic reasoning. To overcome these challenges, we propose SCoT: a Streaming Chain-of-Thought (CoT) framework for Duplex SDS, alternating between processing fixed-duration user input and generating responses in a blockwise manner. Using frame-level alignments, we create intermediate targets-aligned user transcripts and system responses for each block. Experiments show that our approach produces more coherent and interpretable responses than existing duplex methods while supporting lower-latency and overlapping interactions compared to turn-by-turn systems.
>
---
#### [new 047] CIFLEX: Contextual Instruction Flow for Sub-task Execution in Multi-Turn Interactions with a Single On-Device LLM
- **分类: cs.CL**

- **简介: 该论文属于多轮对话任务，解决单设备大模型处理子任务时的计算开销问题。通过重用缓存和隔离执行路径，降低冗余计算，提升效率。**

- **链接: [http://arxiv.org/pdf/2510.01239v1](http://arxiv.org/pdf/2510.01239v1)**

> **作者:** Juntae Lee; Jihwan Bang; Seunghan Yang; Simyung Chang
>
> **备注:** accepted at EMNLP 2025 (main)
>
> **摘要:** We present CIFLEX (Contextual Instruction Flow for Sub-task Execution), which is a novel execution system for efficient sub-task handling in multi-turn interactions with a single on-device large language model (LLM). As LLMs become increasingly capable, a single model is expected to handle diverse sub-tasks that more effectively and comprehensively support answering user requests. Naive approach reprocesses the entire conversation context when switching between main and sub-tasks (e.g., query rewriting, summarization), incurring significant computational overhead. CIFLEX mitigates this overhead by reusing the key-value (KV) cache from the main task and injecting only task-specific instructions into isolated side paths. After sub-task execution, the model rolls back to the main path via cached context, thereby avoiding redundant prefill computation. To support sub-task selection, we also develop a hierarchical classification strategy tailored for small-scale models, decomposing multi-choice decisions into binary ones. Experiments show that CIFLEX significantly reduces computational costs without degrading task performance, enabling scalable and efficient multi-task dialogue on-device.
>
---
#### [new 048] RJE: A Retrieval-Judgment-Exploration Framework for Efficient Knowledge Graph Question Answering with LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱问答任务，解决LLMs在KGQA中效率低和依赖大模型的问题。提出RJE框架，提升小模型效果并减少调用次数。**

- **链接: [http://arxiv.org/pdf/2510.01257v1](http://arxiv.org/pdf/2510.01257v1)**

> **作者:** Can Lin; Zhengwang Jiang; Ling Zheng; Qi Zhao; Yuhang Zhang; Qi Song; Wangqiu Zhou
>
> **备注:** 18 pages, 9 figures
>
> **摘要:** Knowledge graph question answering (KGQA) aims to answer natural language questions using knowledge graphs. Recent research leverages large language models (LLMs) to enhance KGQA reasoning, but faces limitations: retrieval-based methods are constrained by the quality of retrieved information, while agent-based methods rely heavily on proprietary LLMs. To address these limitations, we propose Retrieval-Judgment-Exploration (RJE), a framework that retrieves refined reasoning paths, evaluates their sufficiency, and conditionally explores additional evidence. Moreover, RJE introduces specialized auxiliary modules enabling small-sized LLMs to perform effectively: Reasoning Path Ranking, Question Decomposition, and Retriever-assisted Exploration. Experiments show that our approach with proprietary LLMs (such as GPT-4o-mini) outperforms existing baselines while enabling small open-source LLMs (such as 3B and 8B parameters) to achieve competitive results without fine-tuning LLMs. Additionally, RJE substantially reduces the number of LLM calls and token usage compared to agent-based methods, yielding significant efficiency improvements.
>
---
#### [new 049] Enhancing Transformer-Based Rerankers with Synthetic Data and LLM-Based Supervision
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文档重排序任务，旨在解决小模型训练数据不足的问题。通过LLM生成合成数据并进行对比学习，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2510.01229v1](http://arxiv.org/pdf/2510.01229v1)**

> **作者:** Dimitar Peshevski; Kiril Blazhevski; Martin Popovski; Gjorgji Madjarov
>
> **备注:** Accepted by RANLP 2025
>
> **摘要:** Effective document reranking is essential for improving search relevance across diverse applications. While Large Language Models (LLMs) excel at reranking due to their deep semantic understanding and reasoning, their high computational cost makes them impractical for many real-world deployments. Fine-tuning smaller, task-specific models is a more efficient alternative but typically depends on scarce, manually labeled data. To overcome this, we propose a novel pipeline that eliminates the need for human-labeled query-document pairs. Our method uses LLMs to generate synthetic queries from domain-specific corpora and employs an LLM-based classifier to label positive and hard-negative pairs. This synthetic dataset is then used to fine-tune a smaller transformer model with contrastive learning using Localized Contrastive Estimation (LCE) loss. Experiments on the MedQuAD dataset show that our approach significantly boosts in-domain performance and generalizes well to out-of-domain tasks. By using LLMs for data generation and supervision rather than inference, we reduce computational costs while maintaining strong reranking capabilities.
>
---
#### [new 050] Benchmark Profiling: Mechanistic Diagnosis of LLM Benchmarks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型评估任务，旨在解决基准测试与实际能力不匹配的问题。通过引入Benchmark Profiling框架，分析模型在不同基准上的能力贡献，提升模型解释性与基准审计透明度。**

- **链接: [http://arxiv.org/pdf/2510.01232v1](http://arxiv.org/pdf/2510.01232v1)**

> **作者:** Dongjun Kim; Gyuho Shim; Yongchan Chun; Minhyuk Kim; Chanjun Park; Heuiseok Lim
>
> **备注:** 16 pages, 5 figures. Accepted to EMNLP 2025 main conference
>
> **摘要:** Large Language Models are commonly judged by their scores on standard benchmarks, yet such scores often overstate real capability since they mask the mix of skills a task actually demands. For example, ARC is assumed to test reasoning, while HellaSwag is designed to evaluate commonsense. However, we lack a systematic way to verify if these benchmarks actually measure these labels. We introduce Benchmark Profiling, a diagnostic framework that decomposes benchmark performance into ten cognitively grounded abilities. The method combines gradient-based importance scoring with targeted parameter ablation to compute an Ability Impact Score (AIS) that quantifies how much each ability contributes to a model's success on a given benchmark. Profiling three instruction-tuned models across ten widely used benchmarks yields four key findings: (i) most benchmarks draw on several abilities rather than one, (ii) datasets with similar labels rely on distinct ability mixtures, (iii) code-generation benchmarks reward broad, multi-skill improvement and thus show only modest gains from narrow domain-specific fine-tuning, and (iv) abilities irrelevant to the task could negatively affect performance. Benchmark Profiling therefore explains why performance gains do not always translate into user-perceived competence and offers a transparent tool for benchmark audit and model interpretability.
>
---
#### [new 051] LLM Based Sentiment Classification From Bangladesh E-Commerce Reviews
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感分类任务，旨在解决多语言电商评论的情感分析问题。通过微调LLM模型，提升了分类效果，并验证了参数高效方法的可行性。**

- **链接: [http://arxiv.org/pdf/2510.01276v1](http://arxiv.org/pdf/2510.01276v1)**

> **作者:** Sumaiya Tabassum
>
> **摘要:** Sentiment analysis is an essential part of text analysis, which is a larger field that includes determining and evaluating the author's emotional state. This method is essential since it makes it easier to comprehend consumers' feelings, viewpoints, and preferences holistically. The introduction of large language models (LLMs), such as Llama, has greatly increased the availability of cutting-edge model applications, such as sentiment analysis. However, accurate sentiment analysis is hampered by the intricacy of written language and the diversity of languages used in evaluations. The viability of using transformer-based BERT models and other LLMs for sentiment analysis from Bangladesh e commerce reviews is investigated in this paper. A subset of 4000 samples from the original dataset of Bangla and English customer reviews was utilized to fine-tune the model. The fine tuned Llama-3.1-8B model outperformed other fine-tuned models, including Phi-3.5-mini-instruct, Mistral-7B-v0.1, DistilBERT-multilingual, mBERT, and XLM-R-base, with an overall accuracy, precision, recall, and F1 score of 95.5%, 93%, 88%, 90%. The study emphasizes how parameter efficient fine-tuning methods (LoRA and PEFT) can lower computational overhead and make it appropriate for contexts with limited resources. The results show how LLMs can
>
---
#### [new 052] Style Over Story: A Process-Oriented Study of Authorial Creativity in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于AI创作研究任务，旨在分析大语言模型的创作过程。通过叙事学视角，探讨模型在风格与情节等元素间的偏好，揭示其创造性决策机制。**

- **链接: [http://arxiv.org/pdf/2510.02025v1](http://arxiv.org/pdf/2510.02025v1)**

> **作者:** Donghoon Jung; Jiwoo Choi; Songeun Chae; Seohyon Jung
>
> **摘要:** Evaluations of large language models (LLMs)' creativity have focused primarily on the quality of their outputs rather than the processes that shape them. This study takes a process-oriented approach, drawing on narratology to examine LLMs as computational authors. We introduce constraint-based decision-making as a lens for authorial creativity. Using controlled prompting to assign authorial personas, we analyze the creative preferences of the models. Our findings show that LLMs consistently emphasize Style over other elements, including Character, Event, and Setting. By also probing the reasoning the models provide for their choices, we show that distinctive profiles emerge across models and argue that our approach provides a novel systematic tool for analyzing AI's authorial creativity.
>
---
#### [new 053] Drawing Conclusions from Draws: Rethinking Preference Semantics in Arena-Style LLM Evaluation
- **分类: cs.CL**

- **简介: 该论文属于LLM评估任务，探讨arena风格评价中平局的语义问题。研究指出平局可能反映查询难度而非模型等价，提出应重新考虑评分机制。**

- **链接: [http://arxiv.org/pdf/2510.02306v1](http://arxiv.org/pdf/2510.02306v1)**

> **作者:** Raphael Tang; Crystina Zhang; Wenyan Li; Carmen Lai; Pontus Stenetorp; Yao Lu
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** In arena-style evaluation of large language models (LLMs), two LLMs respond to a user query, and the user chooses the winning response or deems the "battle" a draw, resulting in an adjustment to the ratings of both models. The prevailing approach for modeling these rating dynamics is to view battles as two-player game matches, as in chess, and apply the Elo rating system and its derivatives. In this paper, we critically examine this paradigm. Specifically, we question whether a draw genuinely means that the two models are equal and hence whether their ratings should be equalized. Instead, we conjecture that draws are more indicative of query difficulty: if the query is too easy, then both models are more likely to succeed equally. On three real-world arena datasets, we show that ignoring rating updates for draws yields a 1-3% relative increase in battle outcome prediction accuracy (which includes draws) for all four rating systems studied. Further analyses suggest that draws occur more for queries rated as very easy and those as highly objective, with risk ratios of 1.37 and 1.35, respectively. We recommend future rating systems to reconsider existing draw semantics and to account for query properties in rating updates.
>
---
#### [new 054] ClaimCheck: Real-Time Fact-Checking with Small Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于事实核查任务，旨在用小型语言模型实现实时、透明的虚假信息验证。通过模块化设计提升准确性与可解释性。**

- **链接: [http://arxiv.org/pdf/2510.01226v1](http://arxiv.org/pdf/2510.01226v1)**

> **作者:** Akshith Reddy Putta; Jacob Devasier; Chengkai Li
>
> **摘要:** We introduce ClaimCheck, an LLM-guided automatic fact-checking system designed to verify real-world claims using live Web evidence and small language models. Unlike prior systems that rely on large, closed-source models and static knowledge stores, ClaimCheck employs a transparent, stepwise verification pipeline that mirrors human fact-checking workflows consisting of Web search query planning, Web-based evidence retrieval and summarization, evidence synthesis and re-retrieval, and claim verdict evaluation. Each module is optimized for small LLMs, allowing the system to deliver accurate and interpretable fact-checking with significantly lower computational requirements. Despite using a much smaller Qwen3-4B model, ClaimCheck achieves state-of-the-art accuracy of 76.4% on the AVeriTeC dataset, outperforming previous approaches using LLaMA3.1 70B and GPT-4o. Extensive ablations demonstrate that careful modular design and prompting strategies can overcome the limitations of smaller LLMs. To promote accessibility and transparency, we provide a public demo at https://idir.uta.edu/claimcheck.
>
---
#### [new 055] InfoMosaic-Bench: Evaluating Multi-Source Information Seeking in Tool-Augmented Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息检索任务，旨在解决LLM代理在多源信息整合中的不足。提出InfoMosaic-Bench基准，评估代理结合通用搜索与领域工具的能力。**

- **链接: [http://arxiv.org/pdf/2510.02271v1](http://arxiv.org/pdf/2510.02271v1)**

> **作者:** Yaxin Du; Yuanshuo Zhang; Xiyuan Yang; Yifan Zhou; Cheng Wang; Gongyi Zou; Xianghe Pang; Wenhao Wang; Menglan Chen; Shuo Tang; Zhiyu Li; Siheng Chen
>
> **摘要:** Information seeking is a fundamental requirement for humans. However, existing LLM agents rely heavily on open-web search, which exposes two fundamental weaknesses: online content is noisy and unreliable, and many real-world tasks require precise, domain-specific knowledge unavailable from the web. The emergence of the Model Context Protocol (MCP) now allows agents to interface with thousands of specialized tools, seemingly resolving this limitation. Yet it remains unclear whether agents can effectively leverage such tools -- and more importantly, whether they can integrate them with general-purpose search to solve complex tasks. Therefore, we introduce InfoMosaic-Bench, the first benchmark dedicated to multi-source information seeking in tool-augmented agents. Covering six representative domains (medicine, finance, maps, video, web, and multi-domain integration), InfoMosaic-Bench requires agents to combine general-purpose search with domain-specific tools. Tasks are synthesized with InfoMosaic-Flow, a scalable pipeline that grounds task conditions in verified tool outputs, enforces cross-source dependencies, and filters out shortcut cases solvable by trivial lookup. This design guarantees both reliability and non-triviality. Experiments with 14 state-of-the-art LLM agents reveal three findings: (i) web information alone is insufficient, with GPT-5 achieving only 38.2% accuracy and 67.5% pass rate; (ii) domain tools provide selective but inconsistent benefits, improving some domains while degrading others; and (iii) 22.4% of failures arise from incorrect tool usage or selection, highlighting that current LLMs still struggle with even basic tool handling.
>
---
#### [new 056] Geometric Structures and Patterns of Meaning: A PHATE Manifold Analysis of Chinese Character Embeddings
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究中文字符嵌入的几何结构。通过PHATE分析，揭示语义与结构的分布模式，验证传统语言理论。**

- **链接: [http://arxiv.org/pdf/2510.01230v1](http://arxiv.org/pdf/2510.01230v1)**

> **作者:** Wen G. Gong
>
> **备注:** 33 pages, 17 figures
>
> **摘要:** We systematically investigate geometric patterns in Chinese character embeddings using PHATE manifold analysis. Through cross-validation across seven embedding models and eight dimensionality reduction methods, we observe clustering patterns for content words and branching patterns for function words. Analysis of over 1000 Chinese characters across 12 semantic domains reveals that geometric complexity correlates with semantic content: meaningful characters exhibit rich geometric diversity while structural radicals collapse into tight clusters. The comprehensive child-network analysis (123 phrases) demonstrates systematic semantic expansion from elemental character. These findings provide computational evidence supporting traditional linguistic theory and establish a novel framework for geometric analysis of semantic organization.
>
---
#### [new 057] TUMIX: Multi-Agent Test-Time Scaling with Tool-Use Mixture
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多智能体推理任务，解决如何有效融合工具使用以提升模型推理能力的问题。提出TUMIX框架，通过并行运行不同策略的代理，提升准确率并优化计算成本。**

- **链接: [http://arxiv.org/pdf/2510.01279v1](http://arxiv.org/pdf/2510.01279v1)**

> **作者:** Yongchao Chen; Jiefeng Chen; Rui Meng; Ji Yin; Na Li; Chuchu Fan; Chi Wang; Tomas Pfister; Jinsung Yoon
>
> **备注:** 27 pages, 13 figures
>
> **摘要:** While integrating tools like Code Interpreter and Search has significantly enhanced Large Language Model (LLM) reasoning in models like ChatGPT Agent and Gemini-Pro, practical guidance on optimal tool use is lacking. The core challenge is effectively combining textual reasoning, coding, and search for diverse questions. In this paper, we propose Tool-Use Mixture (TUMIX), an ensemble framework that runs multiple agents in parallel, each employing distinct tool-use strategies and answer paths. Agents in TUMIX iteratively share and refine responses based on the question and previous answers. In experiments, TUMIX achieves significant gains over state-of-the-art tool-augmented and test-time scaling methods, delivering an average accuracy improvement of up to 3.55% over the best baseline on Gemini-2.5-Pro and Gemini-2.5-Flash across key reasoning benchmarks, with near-equal inference costs. We find that agent diversity and quality are crucial and can be enhanced by using LLMs to auto-optimize agent designs. Furthermore, TUMIX can halt refinement upon reaching sufficient confidence, preserving performance at only 49% of the inference cost. Further scaling can achieve higher performance, albeit at a greater cost.
>
---
#### [new 058] NLP Methods for Detecting Novel LLM Jailbreaks and Keyword Analysis with BERT
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于安全检测任务，旨在识别LLM中的越狱提示。通过分析不同模型效果，发现微调BERT能有效区分越狱与正常提示，并揭示了越狱提示的关键词特征。**

- **链接: [http://arxiv.org/pdf/2510.01644v1](http://arxiv.org/pdf/2510.01644v1)**

> **作者:** John Hawkins; Aditya Pramar; Rodney Beard; Rohitash Chandra
>
> **摘要:** Large Language Models (LLMs) suffer from a range of vulnerabilities that allow malicious users to solicit undesirable responses through manipulation of the input text. These so-called jailbreak prompts are designed to trick the LLM into circumventing the safety guardrails put in place to keep responses acceptable to the developer's policies. In this study, we analyse the ability of different machine learning models to distinguish jailbreak prompts from genuine uses, including looking at our ability to identify jailbreaks that use previously unseen strategies. Our results indicate that using current datasets the best performance is achieved by fine tuning a Bidirectional Encoder Representations from Transformers (BERT) model end-to-end for identifying jailbreaks. We visualise the keywords that distinguish jailbreak from genuine prompts and conclude that explicit reflexivity in prompt structure could be a signal of jailbreak intention.
>
---
#### [new 059] LLMRank: Understanding LLM Strengths for Model Routing
- **分类: cs.CL**

- **简介: 该论文属于模型路由任务，旨在解决如何高效选择适合的LLM问题。通过提取提示特征并使用神经排序模型进行路由决策，提升性能与效率的平衡。**

- **链接: [http://arxiv.org/pdf/2510.01234v1](http://arxiv.org/pdf/2510.01234v1)**

> **作者:** Shubham Agrawal; Prasang Gupta
>
> **备注:** 13 pages, 1 figure
>
> **摘要:** The rapid growth of large language models (LLMs) with diverse capabilities, latency and computational costs presents a critical deployment challenge: selecting the most suitable model for each prompt to optimize the trade-off between performance and efficiency. We introduce LLMRank, a prompt-aware routing framework that leverages rich, human-readable features extracted from prompts, including task type, reasoning patterns, complexity indicators, syntactic cues, and signals from a lightweight proxy solver. Unlike prior one-shot routers that rely solely on latent embeddings, LLMRank predicts per-model utility using a neural ranking model trained on RouterBench, comprising 36,497 prompts spanning 11 benchmarks and 11 state-of-the-art LLMs, from small efficient models to large frontier systems. Our approach achieves up to 89.2% of oracle utility, while providing interpretable feature attributions that explain routing decisions. Extensive studies demonstrate the importance of multifaceted feature extraction and the hybrid ranking objective, highlighting the potential of feature-driven routing for efficient and transparent LLM deployment.
>
---
#### [new 060] A-VERT: Agnostic Verification with Embedding Ranking Targets
- **分类: cs.CL; cs.LG; 68T50; I.2.7**

- **简介: 该论文属于语言模型响应评估任务，解决现有方法成本高或不真实的问题，提出基于语义嵌入距离的结构无关评估方法，实现高效准确的分类。**

- **链接: [http://arxiv.org/pdf/2510.01469v1](http://arxiv.org/pdf/2510.01469v1)**

> **作者:** Nicolás Aguirre; Ramiro Caso; Ramiro Rodríguez Colmeiro; Mauro Santelli; Joaquín Toranzo Calderón
>
> **备注:** 19 pages, 7 figures, code available at https://github.com/pnyxai/a-vert, authors in alphabetical order
>
> **摘要:** The automatic evaluation of Language Model (LM) responses is a critical piece in the development of benchmarks and metrics, both for model training and quality assessment of production model endpoints. The current approaches to response classification relies on methods that are too expensive (i.e. LLM-as-a-Judge) or that are far from real-world conditions (string-matching, logprob). In this paper, a structure-free evaluation method is presented. The method makes use of semantic embedding distances to match target candidates with arbitrary LM-generated text, resulting in a robust classification of the response at a relatively low compute cost (embedding models of less than $10B$ parameters). The results show a regression score of ~0.97 and an accuracy of ~96% against human annotators, tested over 3 data sets and 3 different LM architectures.
>
---
#### [new 061] GPT and Prejudice: A Sparse Approach to Understanding Learned Representations in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型解释任务，旨在解决大型语言模型内部表示和数据偏见的理解问题。通过结合GPT与稀疏自编码器，提取可解释特征以分析训练数据中的社会结构和偏见。**

- **链接: [http://arxiv.org/pdf/2510.01252v1](http://arxiv.org/pdf/2510.01252v1)**

> **作者:** Mariam Mahran; Katharina Simbeck
>
> **备注:** Preprint. Draft version, subject to revision. 8 pages, 3 figures
>
> **摘要:** As large language models (LLMs) are increasingly trained on massive, uncurated corpora, understanding both model representations and the data they internalize has become a major challenge. In this work, we show that pairing LLMs with sparse autoencoders (SAEs) enables interpretation not only of model behavior but also of the deeper structures, themes, and biases embedded in the training data. We train a GPT-style transformer model exclusively on the novels of Jane Austen, a corpus rich in social constructs and narrative patterns. We then apply SAEs to hidden states across multiple layers, uncovering sparse, interpretable features that reflect the key narratives and concepts present in the corpus, including gender, class, and societal duty. Our findings demonstrate that LLMs combined with SAEs can act as scalable probes into complex datasets, offering a new path for corpus exploration, bias discovery, and model interpretability at scale.
>
---
#### [new 062] CLUE: Non-parametric Verification from Experience via Hidden-State Clustering
- **分类: cs.CL**

- **简介: 该论文属于语言模型输出验证任务，解决模型输出质量评估问题。通过分析隐藏状态，提出非参数验证方法CLUE，提升准确性。**

- **链接: [http://arxiv.org/pdf/2510.01591v1](http://arxiv.org/pdf/2510.01591v1)**

> **作者:** Zhenwen Liang; Ruosen Li; Yujun Zhou; Linfeng Song; Dian Yu; Xinya Du; Haitao Mi; Dong Yu
>
> **摘要:** Assessing the quality of Large Language Model (LLM) outputs presents a critical challenge. Previous methods either rely on text-level information (e.g., reward models, majority voting), which can overfit to superficial cues, or on calibrated confidence from token probabilities, which would fail on less-calibrated models. Yet both of these signals are, in fact, partial projections of a richer source of information: the model's internal hidden states. Early layers, closer to token embeddings, preserve semantic and lexical features that underpin text-based judgments, while later layers increasingly align with output logits, embedding confidence-related information. This paper explores hidden states directly as a unified foundation for verification. We show that the correctness of a solution is encoded as a geometrically separable signature within the trajectory of hidden activations. To validate this, we present Clue (Clustering and Experience-based Verification), a deliberately minimalist, non-parametric verifier. With no trainable parameters, CLUE only summarizes each reasoning trace by an hidden state delta and classifies correctness via nearest-centroid distance to ``success'' and ``failure'' clusters formed from past experience. The simplicity of this method highlights the strength of the underlying signal. Empirically, CLUE consistently outperforms LLM-as-a-judge baselines and matches or exceeds modern confidence-based methods in reranking candidates, improving both top-1 and majority-vote accuracy across AIME 24/25 and GPQA. As a highlight, on AIME 24 with a 1.5B model, CLUE boosts accuracy from 56.7% (majority@64) to 70.0% (top-maj@16).
>
---
#### [new 063] From Behavioral Performance to Internal Competence: Interpreting Vision-Language Models with VLM-Lens
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型分析任务，旨在解决模型内部机制理解问题。工作是开发VLM-Lens工具，支持模型中间输出提取与分析。**

- **链接: [http://arxiv.org/pdf/2510.02292v1](http://arxiv.org/pdf/2510.02292v1)**

> **作者:** Hala Sheta; Eric Huang; Shuyu Wu; Ilia Alenabi; Jiajun Hong; Ryker Lin; Ruoxi Ning; Daniel Wei; Jialin Yang; Jiawei Zhou; Ziqiao Ma; Freda Shi
>
> **备注:** EMNLP 2025 System Demonstration | Code: https://github.com/compling-wat/vlm-lens
>
> **摘要:** We introduce VLM-Lens, a toolkit designed to enable systematic benchmarking, analysis, and interpretation of vision-language models (VLMs) by supporting the extraction of intermediate outputs from any layer during the forward pass of open-source VLMs. VLM-Lens provides a unified, YAML-configurable interface that abstracts away model-specific complexities and supports user-friendly operation across diverse VLMs. It currently supports 16 state-of-the-art base VLMs and their over 30 variants, and is extensible to accommodate new models without changing the core logic. The toolkit integrates easily with various interpretability and analysis methods. We demonstrate its usage with two simple analytical experiments, revealing systematic differences in the hidden representations of VLMs across layers and target concepts. VLM-Lens is released as an open-sourced project to accelerate community efforts in understanding and improving VLMs.
>
---
#### [new 064] EEFSUVA: A New Mathematical Olympiad Benchmark
- **分类: cs.CL; math.HO**

- **简介: 该论文属于数学推理任务，旨在评估大语言模型的真实数学能力。针对现有基准可能存在的数据污染和范围狭窄问题，作者提出了EEFSUVA新基准，以更全面地测试模型的数学推理能力。**

- **链接: [http://arxiv.org/pdf/2510.01227v1](http://arxiv.org/pdf/2510.01227v1)**

> **作者:** Nicole N Khatibi; Daniil A. Radamovich; Michael P. Brenner
>
> **备注:** 16 Pages, 5 figures
>
> **摘要:** Recent breakthroughs have spurred claims that large language models (LLMs) match gold medal Olympiad to graduate level proficiency on mathematics benchmarks. In this work, we examine these claims in detail and assess the extent to which current benchmarks capture genuine LLM mathematical reasoning. The composition of these benchmarks, primarily drawing from the International Mathematics Olympiad (IMO) and related competitions, may overstate models reasoning ability due to potential data contamination and a narrow focus on familiar problem types. To enable a more holistic assessment of mathematical understanding, we introduce EEFSUVA, a novel benchmark curated from under circulated regional and national Olympiads of Eastern Europe and the countries from the former Soviet Union. These contests feature problems of comparable difficulty to the IMO and are renowned for demanding nonstandard problem-solving techniques, yet their problems are far less prevalent in online corpora. Preliminary results suggest that even state-of-the-art LLMs exhibit a notable performance decline on EEFSUVA relative to other Olympiad-style benchmarks. These findings also suggest the potential importance of broader evaluation datasets for a fuller assessment of mathematical reasoning and for guiding future model development.
>
---
#### [new 065] F2LLM Technical Report: Matching SOTA Embedding Performance with 6 Million Open-Source Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升嵌入模型性能。通过微调基础模型，利用600万真实数据，构建高效、低成本的嵌入模型。**

- **链接: [http://arxiv.org/pdf/2510.02294v1](http://arxiv.org/pdf/2510.02294v1)**

> **作者:** Ziyin Zhang; Zihan Liao; Hang Yu; Peng Di; Rui Wang
>
> **摘要:** We introduce F2LLM - Foundation to Feature Large Language Models, a suite of state-of-the-art embedding models in three sizes: 0.6B, 1.7B, and 4B. Unlike previous top-ranking embedding models that require massive contrastive pretraining, sophisticated training pipelines, and costly synthetic training data, F2LLM is directly finetuned from foundation models on 6 million query-document-negative tuples curated from open-source, non-synthetic datasets, striking a strong balance between training cost, model size, and embedding performance. On the MTEB English leaderboard, F2LLM-4B ranks 2nd among models with approximately 4B parameters and 7th overall, while F2LLM-1.7B ranks 1st among models in the 1B-2B size range. To facilitate future research in the field, we release the models, training dataset, and code, positioning F2LLM as a strong, reproducible, and budget-friendly baseline for future works.
>
---
#### [new 066] Detecting LLM-Generated Spam Reviews by Integrating Language Model Embeddings and Graph Neural Network
- **分类: cs.CL**

- **简介: 该论文属于spam检测任务，解决LLM生成的虚假评论识别问题。通过融合语言模型嵌入和图神经网络，提出FraudSquad模型，有效提升检测性能。**

- **链接: [http://arxiv.org/pdf/2510.01801v1](http://arxiv.org/pdf/2510.01801v1)**

> **作者:** Xin Liu; Rongwu Xu; Xinyi Jia; Jason Liao; Jiao Sun; Ling Huang; Wei Xu
>
> **摘要:** The rise of large language models (LLMs) has enabled the generation of highly persuasive spam reviews that closely mimic human writing. These reviews pose significant challenges for existing detection systems and threaten the credibility of online platforms. In this work, we first create three realistic LLM-generated spam review datasets using three distinct LLMs, each guided by product metadata and genuine reference reviews. Evaluations by GPT-4.1 confirm the high persuasion and deceptive potential of these reviews. To address this threat, we propose FraudSquad, a hybrid detection model that integrates text embeddings from a pre-trained language model with a gated graph transformer for spam node classification. FraudSquad captures both semantic and behavioral signals without relying on manual feature engineering or massive training resources. Experiments show that FraudSquad outperforms state-of-the-art baselines by up to 44.22% in precision and 43.01% in recall on three LLM-generated datasets, while also achieving promising results on two human-written spam datasets. Furthermore, FraudSquad maintains a modest model size and requires minimal labeled training data, making it a practical solution for real-world applications. Our contributions include new synthetic datasets, a practical detection framework, and empirical evidence highlighting the urgency of adapting spam detection to the LLM era. Our code and datasets are available at: https://anonymous.4open.science/r/FraudSquad-5389/.
>
---
#### [new 067] The Disparate Impacts of Speculative Decoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究 speculative decoding 在不同任务中的速度差异问题，分析其对任务公平性的影响，并提出缓解策略以提升公平性。**

- **链接: [http://arxiv.org/pdf/2510.02128v1](http://arxiv.org/pdf/2510.02128v1)**

> **作者:** Jameson Sandler; Ahmet Üstün; Marco Romanelli; Sara Hooker; Ferdinando Fioretto
>
> **摘要:** The practice of speculative decoding, whereby inference is probabilistically supported by a smaller, cheaper, ``drafter'' model, has become a standard technique for systematically reducing the decoding time of large language models. This paper conducts an analysis of speculative decoding through the lens of its potential disparate speed-up rates across tasks. Crucially, the paper shows that speed-up gained from speculative decoding is not uniformly distributed across tasks, consistently diminishing for under-fit, and often underrepresented tasks. To better understand this phenomenon, we derive an analysis to quantify this observed ``unfairness'' and draw attention to the factors that motivate such disparate speed-ups to emerge. Further, guided by these insights, the paper proposes a mitigation strategy designed to reduce speed-up disparities and validates the approach across several model pairs, revealing on average a 12% improvement in our fairness metric.
>
---
#### [new 068] HiSpec: Hierarchical Speculative Decoding for LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型推理加速任务，解决验证瓶颈问题。提出HiSpec框架，利用早停模型实现高效中间验证，提升吞吐量。**

- **链接: [http://arxiv.org/pdf/2510.01336v1](http://arxiv.org/pdf/2510.01336v1)**

> **作者:** Avinash Kumar; Sujay Sanghavi; Poulami Das
>
> **摘要:** Speculative decoding accelerates LLM inference by using a smaller draft model to speculate tokens that a larger target model verifies. Verification is often the bottleneck (e.g. verification is $4\times$ slower than token generation when a 3B model speculates for a 70B target model), but most prior works focus only on accelerating drafting. $\textit{``Intermediate"}$ verification reduces verification time by discarding inaccurate draft tokens early, but existing methods incur substantial training overheads in incorporating the intermediate verifier, increase the memory footprint to orchestrate the intermediate verification step, and compromise accuracy by relying on approximate heuristics. We propose $\underline{\textit{Hi}}\textit{erarchical }\underline{\textit{Spec}}\textit{ulative Decoding (HiSpec)}$, a framework for high-throughput speculative decoding that exploits $\textit{early-exit (EE) models}$ for low-overhead intermediate verification. EE models allow tokens to exit early by skipping layer traversal and are explicitly trained so that hidden states at selected layers can be interpreted, making them uniquely suited for intermediate verification without drastically increasing compute and memory overheads. To improve resource-efficiency even further, we design a methodology that enables HiSpec to re-use key-value caches and hidden states between the draft, intermediate verifier, and target models. To maintain accuracy, HiSpec periodically validates the draft tokens accepted by the intermediate verifier against the target model. Our evaluations using various representative benchmarks and models show that HiSpec improves throughput by 1.28$\times$ on average and by up to 2.01$\times$ compared to the baseline single-layer speculation without compromising accuracy.
>
---
#### [new 069] Veri-R1: Toward Precise and Faithful Claim Verification via Online Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于事实核查任务，解决在线验证中证据检索与推理不足的问题。通过在线强化学习框架Veri-R1提升模型的验证精度与可靠性。**

- **链接: [http://arxiv.org/pdf/2510.01932v1](http://arxiv.org/pdf/2510.01932v1)**

> **作者:** Qi He; Cheng Qian; Xiusi Chen; Bingxiang He; Yi R.; Fung; Heng Ji
>
> **摘要:** Claim verification with large language models (LLMs) has recently attracted considerable attention, owing to their superior reasoning capabilities and transparent verification pathways compared to traditional answer-only judgments. Online claim verification requires iterative evidence retrieval and reasoning, yet existing approaches mainly rely on prompt engineering or predesigned reasoning workflows without offering a unified training paradigm to improve necessary skills. Therefore, we introduce Veri-R1, an online reinforcement learning (RL) framework that enables an LLM to interact with a search engine and to receive reward signals that explicitly shape its planning, retrieval, and reasoning behaviors. The dynamic interaction between models and retrieval systems more accurately reflects real-world verification scenarios and fosters comprehensive verification skills. Empirical results show that Veri-R1 improves joint accuracy by up to 30% and doubles evidence score, often surpassing larger-scale counterparts. Ablation studies further reveal the impact of reward components and the link between output logits and label accuracy. Our results highlight the effectiveness of online RL for precise and faithful claim verification and provide a foundation for future research. We release our code to support community progress in LLM empowered claim verification.
>
---
#### [new 070] SCRIBES: Web-Scale Script-Based Semi-Structured Data Extraction with Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于信息抽取任务，旨在解决半结构化网页数据的高效提取问题。通过强化学习生成可复用脚本，提升提取效果与效率。**

- **链接: [http://arxiv.org/pdf/2510.01832v1](http://arxiv.org/pdf/2510.01832v1)**

> **作者:** Shicheng Liu; Kai Sun; Lisheng Fu; Xilun Chen; Xinyuan Zhang; Zhaojiang Lin; Rulin Shao; Yue Liu; Anuj Kumar; Wen-tau Yih; Xin Luna Dong
>
> **摘要:** Semi-structured content in HTML tables, lists, and infoboxes accounts for a substantial share of factual data on the web, yet the formatting complicates usage, and reliably extracting structured information from them remains challenging. Existing methods either lack generalization or are resource-intensive due to per-page LLM inference. In this paper, we introduce SCRIBES (SCRIpt-Based Semi-Structured Content Extraction at Web-Scale), a novel reinforcement learning framework that leverages layout similarity across webpages within the same site as a reward signal. Instead of processing each page individually, SCRIBES generates reusable extraction scripts that can be applied to groups of structurally similar webpages. Our approach further improves by iteratively training on synthetic annotations from in-the-wild CommonCrawl data. Experiments show that our approach outperforms strong baselines by over 13% in script quality and boosts downstream question answering accuracy by more than 4% for GPT-4o, enabling scalable and resource-efficient web information extraction.
>
---
#### [new 071] How Do Language Models Compose Functions?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型在组合任务中的处理机制，探讨其是否通过组合方式解决两步事实回忆任务，发现模型可能采用直接或组合策略，并与嵌入空间几何相关。**

- **链接: [http://arxiv.org/pdf/2510.01685v1](http://arxiv.org/pdf/2510.01685v1)**

> **作者:** Apoorv Khandelwal; Ellie Pavlick
>
> **摘要:** While large language models (LLMs) appear to be increasingly capable of solving compositional tasks, it is an open question whether they do so using compositional mechanisms. In this work, we investigate how feedforward LLMs solve two-hop factual recall tasks, which can be expressed compositionally as $g(f(x))$. We first confirm that modern LLMs continue to suffer from the "compositionality gap": i.e. their ability to compute both $z = f(x)$ and $y = g(z)$ does not entail their ability to compute the composition $y = g(f(x))$. Then, using logit lens on their residual stream activations, we identify two processing mechanisms, one which solves tasks $\textit{compositionally}$, computing $f(x)$ along the way to computing $g(f(x))$, and one which solves them $\textit{directly}$, without any detectable signature of the intermediate variable $f(x)$. Finally, we find that which mechanism is employed appears to be related to the embedding space geometry, with the idiomatic mechanism being dominant in cases where there exists a linear mapping from $x$ to $g(f(x))$ in the embedding spaces. We fully release our data and code at: https://github.com/apoorvkh/composing-functions .
>
---
#### [new 072] Learning to Reason for Hallucination Span Detection
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于 hallucination span 检测任务，旨在解决识别模型生成内容中错误段落的问题。通过引入强化学习框架 RL4HS 提升检测效果。**

- **链接: [http://arxiv.org/pdf/2510.02173v1](http://arxiv.org/pdf/2510.02173v1)**

> **作者:** Hsuan Su; Ting-Yao Hu; Hema Swetha Koppula; Kundan Krishna; Hadi Pouransari; Cheng-Yu Hsieh; Cem Koc; Joseph Yitan Cheng; Oncel Tuzel; Raviteja Vemulapalli
>
> **摘要:** Large language models (LLMs) often generate hallucinations -- unsupported content that undermines reliability. While most prior works frame hallucination detection as a binary task, many real-world applications require identifying hallucinated spans, which is a multi-step decision making process. This naturally raises the question of whether explicit reasoning can help the complex task of detecting hallucination spans. To answer this question, we first evaluate pretrained models with and without Chain-of-Thought (CoT) reasoning, and show that CoT reasoning has the potential to generate at least one correct answer when sampled multiple times. Motivated by this, we propose RL4HS, a reinforcement learning framework that incentivizes reasoning with a span-level reward function. RL4HS builds on Group Relative Policy Optimization and introduces Class-Aware Policy Optimization to mitigate reward imbalance issue. Experiments on the RAGTruth benchmark (summarization, question answering, data-to-text) show that RL4HS surpasses pretrained reasoning models and supervised fine-tuning, demonstrating the necessity of reinforcement learning with span-level rewards for detecting hallucination spans.
>
---
#### [new 073] Let's Play Across Cultures: A Large Multilingual, Multicultural Benchmark for Assessing Language Models' Understanding of Sports
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言多文化体育知识理解任务，旨在解决传统体育被忽视的问题，构建了包含33,000题的跨文化基准数据集。**

- **链接: [http://arxiv.org/pdf/2510.01247v1](http://arxiv.org/pdf/2510.01247v1)**

> **作者:** Punit Kumar Singh; Nishant Kumar; Akash Ghosh; Kunal Pasad; Khushi Soni; Manisha Jaishwal; Sriparna Saha; Syukron Abu Ishaq Alfarozi; Asres Temam Abagissa; Kitsuchart Pasupa; Haiqin Yang; Jose G Moreno
>
> **备注:** 52 pages, 56 figures; appearing at EMNLP'25
>
> **摘要:** Language Models (LMs) are primarily evaluated on globally popular sports, often overlooking regional and indigenous sporting traditions. To address this gap, we introduce \textbf{\textit{CultSportQA}}, a benchmark designed to assess LMs' understanding of traditional sports across 60 countries and 6 continents, encompassing four distinct cultural categories. The dataset features 33,000 multiple-choice questions (MCQs) across text and image modalities, each of which is categorized into three key types: history-based, rule-based, and scenario-based. To evaluate model performance, we employ zero-shot, few-shot, and chain-of-thought (CoT) prompting across a diverse set of Large Language Models (LLMs), Small Language Models (SLMs), and Multimodal Large Language Models (MLMs). By providing a comprehensive multilingual and multicultural sports benchmark, \textbf{\textit{CultSportQA}} establishes a new standard for assessing AI's ability to understand and reason about traditional sports.
>
---
#### [new 074] TraceDet: Hallucination Detection from the Decoding Trace of Diffusion Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于 hallucination 检测任务，解决 D-LLMs 中幻觉问题检测不足的问题。通过分析多步去噪过程中的中间步骤，提出 TraceDet 框架提升检测效果。**

- **链接: [http://arxiv.org/pdf/2510.01274v1](http://arxiv.org/pdf/2510.01274v1)**

> **作者:** Shenxu Chang; Junchi Yu; Weixing Wang; Yongqiang Chen; Jialin Yu; Philip Torr; Jindong Gu
>
> **摘要:** Diffusion large language models (D-LLMs) have recently emerged as a promising alternative to auto-regressive LLMs (AR-LLMs). However, the hallucination problem in D-LLMs remains underexplored, limiting their reliability in real-world applications. Existing hallucination detection methods are designed for AR-LLMs and rely on signals from single-step generation, making them ill-suited for D-LLMs where hallucination signals often emerge throughout the multi-step denoising process. To bridge this gap, we propose TraceDet, a novel framework that explicitly leverages the intermediate denoising steps of D-LLMs for hallucination detection. TraceDet models the denoising process as an action trace, with each action defined as the model's prediction over the cleaned response, conditioned on the previous intermediate output. By identifying the sub-trace that is maximally informative to the hallucinated responses, TraceDet leverages the key hallucination signals in the multi-step denoising process of D-LLMs for hallucination detection. Extensive experiments on various open source D-LLMs demonstrate that TraceDet consistently improves hallucination detection, achieving an average gain in AUROC of 15.2% compared to baselines.
>
---
#### [new 075] Taking a SEAT: Predicting Value Interpretations from Sentiment, Emotion, Argument, and Topic Annotations
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在预测个体价值判断。通过结合情感、情绪、论点和主题标注，提升语言模型对个体价值解释的预测能力。**

- **链接: [http://arxiv.org/pdf/2510.01976v1](http://arxiv.org/pdf/2510.01976v1)**

> **作者:** Adina Nicola Dobrinoiu; Ana Cristiana Marcu; Amir Homayounirad; Luciano Cavalcante Siebert; Enrico Liscio
>
> **备注:** Accepted at VALE workshop (ECAI 2025)
>
> **摘要:** Our interpretation of value concepts is shaped by our sociocultural background and lived experiences, and is thus subjective. Recognizing individual value interpretations is important for developing AI systems that can align with diverse human perspectives and avoid bias toward majority viewpoints. To this end, we investigate whether a language model can predict individual value interpretations by leveraging multi-dimensional subjective annotations as a proxy for their interpretive lens. That is, we evaluate whether providing examples of how an individual annotates Sentiment, Emotion, Argument, and Topics (SEAT dimensions) helps a language model in predicting their value interpretations. Our experiment across different zero- and few-shot settings demonstrates that providing all SEAT dimensions simultaneously yields superior performance compared to individual dimensions and a baseline where no information about the individual is provided. Furthermore, individual variations across annotators highlight the importance of accounting for the incorporation of individual subjective annotators. To the best of our knowledge, this controlled setting, although small in size, is the first attempt to go beyond demographics and investigate the impact of annotation behavior on value prediction, providing a solid foundation for future large-scale validation.
>
---
#### [new 076] MDSEval: A Meta-Evaluation Benchmark for Multimodal Dialogue Summarization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态对话摘要任务，旨在解决自动评估方法不足的问题。提出MDSEval基准，包含多模态数据和人类评价，以提升评估准确性。**

- **链接: [http://arxiv.org/pdf/2510.01659v1](http://arxiv.org/pdf/2510.01659v1)**

> **作者:** Yinhong Liu; Jianfeng He; Hang Su; Ruixue Lian; Yi Nian; Jake Vincent; Srikanth Vishnubhotla; Robinson Piramuthu; Saab Mansour
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Multimodal Dialogue Summarization (MDS) is a critical task with wide-ranging applications. To support the development of effective MDS models, robust automatic evaluation methods are essential for reducing both cost and human effort. However, such methods require a strong meta-evaluation benchmark grounded in human annotations. In this work, we introduce MDSEval, the first meta-evaluation benchmark for MDS, consisting image-sharing dialogues, corresponding summaries, and human judgments across eight well-defined quality aspects. To ensure data quality and richfulness, we propose a novel filtering framework leveraging Mutually Exclusive Key Information (MEKI) across modalities. Our work is the first to identify and formalize key evaluation dimensions specific to MDS. We benchmark state-of-the-art modal evaluation methods, revealing their limitations in distinguishing summaries from advanced MLLMs and their susceptibility to various bias.
>
---
#### [new 077] RESTRAIN: From Spurious Votes to Signals -- Self-Driven RL with Self-Penalization
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，解决无监督下模型推理能力提升问题。通过自惩罚机制利用未标注数据，显著提升推理性能。**

- **链接: [http://arxiv.org/pdf/2510.02172v1](http://arxiv.org/pdf/2510.02172v1)**

> **作者:** Zhaoning Yu; Will Su; Leitian Tao; Haozhu Wang; Aashu Singh; Hanchao Yu; Jianyu Wang; Hongyang Gao; Weizhe Yuan; Jason Weston; Ping Yu; Jing Xu
>
> **摘要:** Reinforcement learning with human-annotated data has boosted chain-of-thought reasoning in large reasoning models, but these gains come at high costs in labeled data while faltering on harder tasks. A natural next step is experience-driven learning, where models improve without curated labels by adapting to unlabeled data. We introduce RESTRAIN (REinforcement learning with Self-restraint), a self-penalizing RL framework that converts the absence of gold labels into a useful learning signal. Instead of overcommitting to spurious majority votes, RESTRAIN exploits signals from the model's entire answer distribution: penalizing overconfident rollouts and low-consistency examples while preserving promising reasoning chains. The self-penalization mechanism integrates seamlessly into policy optimization methods such as GRPO, enabling continual self-improvement without supervision. On challenging reasoning benchmarks, RESTRAIN delivers large gains using only unlabeled data. With Qwen3-4B-Base and OctoThinker Hybrid-8B-Base, it improves Pass@1 by up to +140.7 percent on AIME25, +36.2 percent on MMLU_STEM, and +19.6 percent on GPQA-Diamond, nearly matching gold-label training while using no gold labels. These results demonstrate that RESTRAIN establishes a scalable path toward stronger reasoning without gold labels.
>
---
#### [new 078] RAG-BioQA Retrieval-Augmented Generation for Long-Form Biomedical Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于生物医学问答任务，旨在解决短回答不足的问题，通过RAG-BioQA框架生成长文本答案。**

- **链接: [http://arxiv.org/pdf/2510.01612v1](http://arxiv.org/pdf/2510.01612v1)**

> **作者:** Lovely Yeswanth Panchumarthi; Sai Prasad Gudari; Atharva Negi; Praveen Raj Budime; Harsit Upadhya
>
> **摘要:** The exponential growth of biomedical literature creates significant challenges for accessing precise medical information. Current biomedical question-answering systems primarily focus on short-form answers, failing to provide the comprehensive explanations necessary for clinical decision-making. We present RAG-BioQA, a novel framework combining retrieval-augmented generation with domain-specific fine-tuning to produce evidence-based, long-form biomedical answers. Our approach integrates BioBERT embeddings with FAISS indexing and compares various re-ranking strategies (BM25, ColBERT, MonoT5) to optimize context selection before synthesizing evidence through a fine-tuned T5 model. Experimental results on the PubMedQA dataset show significant improvements over baselines, with our best model achieving substantial gains across BLEU, ROUGE, and METEOR metrics, advancing the state of accessible, evidence-based biomedical knowledge retrieval.
>
---
#### [new 079] Efficient Uncertainty Estimation for LLM-based Entity Linking in Tabular Data
- **分类: cs.CL; stat.ML**

- **简介: 该论文属于实体链接任务，旨在解决LLM在表格数据中预测不确定性估计的问题。通过单次推理提取特征，提高效率并降低计算成本。**

- **链接: [http://arxiv.org/pdf/2510.01251v1](http://arxiv.org/pdf/2510.01251v1)**

> **作者:** Carlo Bono; Federico Belotti; Matteo Palmonari
>
> **摘要:** Linking textual values in tabular data to their corresponding entities in a Knowledge Base is a core task across a variety of data integration and enrichment applications. Although Large Language Models (LLMs) have shown State-of-The-Art performance in Entity Linking (EL) tasks, their deployment in real-world scenarios requires not only accurate predictions but also reliable uncertainty estimates, which require resource-demanding multi-shot inference, posing serious limits to their actual applicability. As a more efficient alternative, we investigate a self-supervised approach for estimating uncertainty from single-shot LLM outputs using token-level features, reducing the need for multiple generations. Evaluation is performed on an EL task on tabular data across multiple LLMs, showing that the resulting uncertainty estimates are highly effective in detecting low-accuracy outputs. This is achieved at a fraction of the computational cost, ultimately supporting a cost-effective integration of uncertainty measures into LLM-based EL workflows. The method offers a practical way to incorporate uncertainty estimation into EL workflows with limited computational overhead.
>
---
#### [new 080] FOR-Prompting: From Objection to Revision via an Asymmetric Prompting Protocol
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文提出FOR-Prompting方法，用于提升模型的自我修正能力。针对推理任务中的自我修订不足问题，通过角色分工实现更有效的推理过程。**

- **链接: [http://arxiv.org/pdf/2510.01674v1](http://arxiv.org/pdf/2510.01674v1)**

> **作者:** He Zhang; Anzhou Zhang; Jian Dai
>
> **摘要:** Reasoning protocols such as Chain of Thought (CoT) and Tree of Thought (ToT) organize internal deliberation but lack an explicit mechanism for external questioning that elicits self-revision. We present FOR-Prompting (From Objection to Revision Prompting), an asymmetric protocol where a Defender proposes an answer, an Objectioner raises question-style objections with no direct fixes, and a Host enforces consistency and closure. On GSM8K we observe about a 22% point gain over single-prompt and accuracy on par with CoT, with more than 10% higher ratings in reasoning and coherence from a uniform GPT 4.1 judge. FOR-Prompting also corrects mistakes without tools or human supervision on tricky queries, and improves performance for small-scale model (approx. 19% accuracy improved on Llama3.2:1b for GSM8K task), highlighting promise for small models and on personal device use. Beyond factual QA, qualitative analyses on open-ended tasks show enhanced exploration and refinement, with dialogue traces that make assumptions and trade-offs explicit. The protocol is model agnostic and operates purely at the prompt level through role-structured turns, so it works with hosted and local models of different sizes without retraining, and it supports large-scale study of objection-guided reasoning.
>
---
#### [new 081] AccurateRAG: A Framework for Building Accurate Retrieval-Augmented Question-Answering Applications
- **分类: cs.CL**

- **简介: 该论文属于问答任务，旨在提升检索增强生成（RAG）系统的准确性。通过构建高效框架，优化数据处理与模型微调，实现更优的问答性能。**

- **链接: [http://arxiv.org/pdf/2510.02243v1](http://arxiv.org/pdf/2510.02243v1)**

> **作者:** Linh The Nguyen; Chi Tran; Dung Ngoc Nguyen; Van-Cuong Pham; Hoang Ngo; Dat Quoc Nguyen
>
> **摘要:** We introduce AccurateRAG -- a novel framework for constructing high-performance question-answering applications based on retrieval-augmented generation (RAG). Our framework offers a pipeline for development efficiency with tools for raw dataset processing, fine-tuning data generation, text embedding & LLM fine-tuning, output evaluation, and building RAG systems locally. Experimental results show that our framework outperforms previous strong baselines and obtains new state-of-the-art question-answering performance on benchmark datasets.
>
---
#### [new 082] GRPO++: Enhancing Dermatological Reasoning under Low Resource Settings
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于医学图像分析任务，旨在解决低资源环境下VLMs的结构化推理问题。通过改进GRPO算法和引入知识图谱优化，提升模型诊断准确性和可靠性。**

- **链接: [http://arxiv.org/pdf/2510.01236v1](http://arxiv.org/pdf/2510.01236v1)**

> **作者:** Ismam Nur Swapnil; Aranya Saha; Tanvir Ahmed Khan; Mohammad Ariful Haque
>
> **备注:** Will be submitted at IEEE JBHI
>
> **摘要:** Vision-Language Models (VLMs) show promise in medical image analysis, yet their capacity for structured reasoning in complex domains like dermatology is often limited by data scarcity and the high computational cost of advanced training techniques. To address these challenges, we introduce DermIQ-VLM, a VLM developed through a multi-stage, resource-efficient methodology designed to emulate a dermatologist's diagnostic process. Our primary contribution is a modified version of Grouped Relative Policy Optimization (GRPO), called GRPO++, which stabilizes the powerful but data-intensive GRPO framework. Our proposed training pipeline first employs GRPO++ for reasoning-oriented disease recognition, followed by supervised fine-tuning for conversational ability. To mitigate factual errors introduced during this step, we then align the model using Direct Preference Optimization (DPO), leveraging a Knowledge Graph-based system as a scalable proxy for expert preference. A preliminary evaluation on a curated dermatological dataset demonstrates that our proposed methodology yields notable performance gains over standard fine-tuning approaches. These findings validate the potential of our pipeline as a feasible pathway for developing specialized, reliable VLMs in resource-constrained environments.
>
---
#### [new 083] ReSSFormer: A Recursive Sparse Structured Transformer for Scalable and Long-Context Reasoning
- **分类: cs.CL; cs.NI**

- **简介: 该论文属于自然语言处理任务，旨在解决长文本推理中的效率与结构泛化问题。提出ReSSFormer模型，通过递归结构、稀疏注意力和自组织编码提升性能。**

- **链接: [http://arxiv.org/pdf/2510.01585v1](http://arxiv.org/pdf/2510.01585v1)**

> **作者:** Haochen You; Baojing Liu
>
> **备注:** Accepted as a short paper at ACM Multimedia Asia 2025
>
> **摘要:** While Transformer architectures have demonstrated impressive scalability across domains, they continue to face challenges in long-context reasoning, computational efficiency, and structural generalization - largely due to rigid layer stacking, dense attention, and reliance on positional encodings. We present ReSSFormer, a Recursive Sparse Structured Transformer that integrates three complementary innovations: Recurrent Reasoning & Memory Unit (R2MU) for iterative reasoning with bounded depth, Adaptive Sparse Attention Module (ASAM) for efficient and focused context selection, and Self-Organizing Encoder Structure (SOES) for position-free structure induction. ReSSFormer replaces conventional depth stacking with recurrent inference, substitutes full attention with token- and expert-level sparsity, and models latent token topology directly from content. Across language modeling, multi-hop QA, and structure-sensitive tasks, ReSSFormer consistently outperforms strong baselines under comparable FLOPs and parameter budgets, highlighting its scalability, efficiency, and structural flexibility.
>
---
#### [new 084] Syntactic Blind Spots: How Misalignment Leads to LLMs Mathematical Errors
- **分类: cs.CL; I.2.7; I.2.0**

- **简介: 该论文属于自然语言处理任务，研究LLMs在数学问题上的错误原因。指出语法差异导致模型误用解题策略，通过语法重写提升准确率，强调结构对齐的重要性。**

- **链接: [http://arxiv.org/pdf/2510.01831v1](http://arxiv.org/pdf/2510.01831v1)**

> **作者:** Dane Williamson; Yangfeng Ji; Matthew Dwyer
>
> **备注:** 14 pages, 5 Tables, 9 Figures; Accepted to MathNLP 2025: The 3rd Workshop on Mathematical Natural Language Processing (co-located with EMNLP 2025)
>
> **摘要:** Large Language Models (LLMs) demonstrate strong mathematical problem-solving abilities but frequently fail on problems that deviate syntactically from their training distribution. We identify a systematic failure mode, syntactic blind spots, in which models misapply familiar reasoning strategies to problems that are semantically straightforward but phrased in unfamiliar ways. These errors are not due to gaps in mathematical competence, but rather reflect a brittle coupling between surface form and internal representation. To test this, we rephrase incorrectly answered questions using syntactic templates drawn from correct examples. These rephrasings, which preserve semantics while reducing structural complexity, often lead to correct answers. We quantify syntactic complexity using a metric based on Dependency Locality Theory (DLT), and show that higher DLT scores are associated with increased failure rates across multiple datasets. Our findings suggest that many reasoning errors stem from structural misalignment rather than conceptual difficulty, and that syntax-aware interventions can reveal and mitigate these inductive failures.
>
---
#### [new 085] Format Inertia: A Failure Mechanism of LLMs in Medical Pre-Consultation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗预咨询任务，针对LLMs在长对话中生成重复无效问题的问题，提出通过重新平衡数据集来缓解“格式惯性”现象。**

- **链接: [http://arxiv.org/pdf/2510.01688v1](http://arxiv.org/pdf/2510.01688v1)**

> **作者:** Seungseop Lim; Gibaeg Kim; Wooseok Han; Jean Seo; Hyunkyung Lee; Jaehyo Yoo; Eunho Yang
>
> **备注:** EMNLP 2025 Industry Track
>
> **摘要:** Recent advances in Large Language Models (LLMs) have brought significant improvements to various service domains, including chatbots and medical pre-consultation applications. In the healthcare domain, the most common approach for adapting LLMs to multi-turn dialogue generation is Supervised Fine-Tuning (SFT). However, datasets for SFT in tasks like medical pre-consultation typically exhibit a skewed turn-count distribution. Training on such data induces a novel failure mechanism we term **Format Inertia**, where models tend to generate repetitive, format-correct, but diagnostically uninformative questions in long medical dialogues. To mitigate this observed failure mechanism, we adopt a simple, data-centric method that rebalances the turn-count distribution of the training dataset. Experimental results show that our approach substantially alleviates Format Inertia in medical pre-consultation.
>
---
#### [new 086] One More Question is Enough, Expert Question Decomposition (EQD) Model for Domain Quantitative Reasoning
- **分类: cs.CL; q-fin.CP**

- **简介: 该论文属于领域量化推理任务，解决大模型在专业领域问答中的效率与效果问题。提出EQD模型，通过分解问题提升QA性能，仅需少量数据和算力。**

- **链接: [http://arxiv.org/pdf/2510.01526v1](http://arxiv.org/pdf/2510.01526v1)**

> **作者:** Mengyu Wang; Sotirios Sabanis; Miguel de Carvalho; Shay B. Cohen; Tiejun Ma
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Domain-specific quantitative reasoning remains a major challenge for large language models (LLMs), especially in fields requiring expert knowledge and complex question answering (QA). In this work, we propose Expert Question Decomposition (EQD), an approach designed to balance the use of domain knowledge with computational efficiency. EQD is built on a two-step fine-tuning framework and guided by a reward function that measures the effectiveness of generated sub-questions in improving QA outcomes. It requires only a few thousand training examples and a single A100 GPU for fine-tuning, with inference time comparable to zero-shot prompting. Beyond its efficiency, EQD outperforms state-of-the-art domain-tuned models and advanced prompting strategies. We evaluate EQD in the financial domain, characterized by specialized knowledge and complex quantitative reasoning, across four benchmark datasets. Our method consistently improves QA performance by 0.6% to 10.5% across different LLMs. Our analysis reveals an important insight: in domain-specific QA, a single supporting question often provides greater benefit than detailed guidance steps.
>
---
#### [new 087] OpenAI's GPT-OSS-20B Model and Safety Alignment Issues in a Low-Resource Language
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型安全研究任务，旨在解决低资源语言中GPT-OSS-20B模型的安全对齐问题。通过红队测试，发现模型存在偏见、错误和文化不敏感等问题。**

- **链接: [http://arxiv.org/pdf/2510.01266v1](http://arxiv.org/pdf/2510.01266v1)**

> **作者:** Isa Inuwa-Dutse
>
> **备注:** 6 pages, 4 tables
>
> **摘要:** In response to the recent safety probing for OpenAI's GPT-OSS-20b model, we present a summary of a set of vulnerabilities uncovered in the model, focusing on its performance and safety alignment in a low-resource language setting. The core motivation for our work is to question the model's reliability for users from underrepresented communities. Using Hausa, a major African language, we uncover biases, inaccuracies, and cultural insensitivities in the model's behaviour. With a minimal prompting, our red-teaming efforts reveal that the model can be induced to generate harmful, culturally insensitive, and factually inaccurate content in the language. As a form of reward hacking, we note how the model's safety protocols appear to relax when prompted with polite or grateful language, leading to outputs that could facilitate misinformation and amplify hate speech. For instance, the model operates on the false assumption that common insecticide locally known as Fiya-Fiya (Cyphermethrin) and rodenticide like Shinkafar Bera (a form of Aluminium Phosphide) are safe for human consumption. To contextualise the severity of this error and popularity of the substances, we conducted a survey (n=61) in which 98% of participants identified them as toxic. Additional failures include an inability to distinguish between raw and processed foods and the incorporation of demeaning cultural proverbs to build inaccurate arguments. We surmise that these issues manifest through a form of linguistic reward hacking, where the model prioritises fluent, plausible-sounding output in the target language over safety and truthfulness. We attribute the uncovered flaws primarily to insufficient safety tuning in low-resource linguistic contexts. By concentrating on a low-resource setting, our approach highlights a significant gap in current red-teaming effort and offer some recommendations.
>
---
#### [new 088] Think Twice, Generate Once: Safeguarding by Progressive Self-Reflection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型安全任务，旨在解决LLM生成有害内容的问题。提出Progressive Self-Reflection方法，在不额外训练的情况下提升模型安全性。**

- **链接: [http://arxiv.org/pdf/2510.01270v1](http://arxiv.org/pdf/2510.01270v1)**

> **作者:** Hoang Phan; Victor Li; Qi Lei
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Large language models (LLMs) have revolutionized natural language processing with their ability to generate coherent and contextually relevant text. However, their deployment raises significant concerns about the potential for generating harmful or inappropriate content. In this paper, we introduce Progressive Self-Reflection (PSR), a novel inference-time technique that empowers LLMs to self-monitor and correct their outputs dynamically. Experimental results demonstrate that applying our proposed method to Llama-3.1-8B-Instruct reduces the attack success rate from 77.5\% to 5.9\%, to Llama-3.1-8B base from 89.7\% to 5.6\%, and to Qwen2.5-7B-Instruct from 44.4\% to 3.8\%, without additional training, while maintaining their original performance on benign tasks. Our approach acts as a test-time scaling method, where additional self-reflection rounds enhance safety at the cost of inference overhead. To balance safety with computational efficiency, we introduce a lightweight self-reflection predictor that estimates the optimal number of reflection rounds based on input complexity. This adaptive mechanism prevents unnecessary self-assessment on benign inputs while ensuring thorough evaluation when encountering potentially harmful content. Our findings suggest that Progressive Self-Reflection serves as a scalable test-time approach, enhancing LLM safety by dynamically allocating computational resources in proportion to the input's risk profile.
>
---
#### [new 089] TAG-EQA: Text-And-Graph for Event Question Answering via Structured Prompting Strategies
- **分类: cs.CL**

- **简介: 该论文属于事件问答任务，解决LLM在因果和时间推理上的不足。通过引入结构化提示策略，将因果图融入输入，提升问答准确率。**

- **链接: [http://arxiv.org/pdf/2510.01391v1](http://arxiv.org/pdf/2510.01391v1)**

> **作者:** Maithili Kadam; Francis Ferraro
>
> **备注:** Accepted in *sem 2025
>
> **摘要:** Large language models (LLMs) excel at general language tasks but often struggle with event-based questions-especially those requiring causal or temporal reasoning. We introduce TAG-EQA (Text-And-Graph for Event Question Answering), a prompting framework that injects causal event graphs into LLM inputs by converting structured relations into natural-language statements. TAG-EQA spans nine prompting configurations, combining three strategies (zero-shot, few-shot, chain-of-thought) with three input modalities (text-only, graph-only, text+graph), enabling a systematic analysis of when and how structured knowledge aids inference. On the TORQUESTRA benchmark, TAG-EQA improves accuracy by 5% on average over text-only baselines, with gains up to 12% in zero-shot settings and 18% when graph-augmented CoT prompting is effective. While performance varies by model and configuration, our findings show that causal graphs can enhance event reasoning in LLMs without fine-tuning, offering a flexible way to encode structure in prompt-based QA.
>
---
#### [new 090] Machine-interpretable Engineering Design Standards for Valve Specification
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于工程设计标准化任务，旨在解决传统文档标准难以机器解析的问题。通过构建可重用的语义本体，实现阀门选型的自动化验证与合规性检查。**

- **链接: [http://arxiv.org/pdf/2510.01736v1](http://arxiv.org/pdf/2510.01736v1)**

> **作者:** Anders Gjerver; Rune Frostad; Vedrana Barisic; Melinda Hodkiewicz; Caitlin Woods; Mihaly Fekete; Arild Braathen Torjusen; Johan Wilhelm Kluwer
>
> **备注:** 22 pages, 10 figures, 4 tables
>
> **摘要:** Engineering design processes use technical specifications and must comply with standards. Product specifications, product type data sheets, and design standards are still mainly document-centric despite the ambition to digitalize industrial work. In this paper, we demonstrate how to transform information held in engineering design standards into modular, reusable, machine-interpretable ontologies and use the ontologies in quality assurance of the plant design and equipment selection process. We use modelling patterns to create modular ontologies for knowledge captured in the text and in frequently referenced tables in International Standards for piping, material and valve design. These modules are exchangeable, as stored in a W3C compliant format, and interoperable as they are aligned with the top-level ontology ISO DIS 23726-3: Industrial Data Ontology (IDO). We test these ontologies, created based on international material and piping standards and industry norms, on a valve selection process. Valves are instantiated in semantic asset models as individuals along with a semantic representation of the environmental condition at their location on the asset. We create "functional location tags" as OWL individuals that become instances of OWL class Valve Data Sheet (VDS) specified valves. Similarly we create instances of manufacturer product type. Our approach enables automated validation that a specific VDS is compliant with relevant industry standards. Using semantic reasoning and executable design rules, we also determine whether the product type meets the valve specification. Creation of shared, reusable IDO-based modular ontologies for design standards enables semantic reasoning to be applied to equipment selection processes and demonstrates the potential of this approach for Standards Bodies wanting to transition to digitized Smart Standards.
>
---
#### [new 091] Utilizing Modern Large Language Models (LLM) for Financial Trend Analysis and Digest Creation
- **分类: cs.CE; cs.AI; cs.CL**

- **简介: 该论文属于金融信息处理任务，旨在解决传统分析方法效率低的问题，通过LLM自动生成金融趋势摘要，提升信息处理与获取效率。**

- **链接: [http://arxiv.org/pdf/2510.01225v1](http://arxiv.org/pdf/2510.01225v1)**

> **作者:** Andrei Lazarev; Dmitrii Sedov
>
> **备注:** This is the version of the article accepted for publication in SUMMA 2024 after peer review. The final, published version is available at IEEE Xplore: 10.1109/SUMMA64428.2024.10803746
>
> **摘要:** The exponential growth of information presents a significant challenge for researchers and professionals seeking to remain at the forefront of their fields and this paper introduces an innovative framework for automatically generating insightful financial digests using the power of Large Language Models (LLMs), specifically Google's Gemini Pro. By leveraging a combination of data extraction from OpenAlex, strategic prompt engineering, and LLM-driven analysis, we demonstrate the automated example of creating a comprehensive digests that generalize key findings, identify emerging trends. This approach addresses the limitations of traditional analysis methods, enabling the efficient processing of vast amounts of unstructured data and the delivery of actionable insights in an easily digestible format. This paper describes how LLMs work in simple words and how we can use their power to help researchers and scholars save their time and stay informed about current trends. Our study includes step-by-step process, from data acquisition and JSON construction to interaction with Gemini and the automated generation of PDF reports, including a link to the project's GitHub repository for broader accessibility and further development.
>
---
#### [new 092] Plan Then Action:High-Level Planning Guidance Reinforcement Learning for LLM Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决LLM在复杂任务中缺乏全局规划的问题。提出PTA-GRPO框架，通过分阶段优化提升推理效果。**

- **链接: [http://arxiv.org/pdf/2510.01833v1](http://arxiv.org/pdf/2510.01833v1)**

> **作者:** Zhihao Dou; Qinjian Zhao; Zhongwei Wan; Dinggen Zhang; Weida Wang; Towsif Raiyan; Benteng Chen; Qingtao Pan; Yang Ouyang; Zhiqiang Gao; Shufei Zhang; Sumon Biswas
>
> **备注:** 19 pages and 5 figures
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable reasoning abilities in complex tasks, often relying on Chain-of-Thought (CoT) reasoning. However, due to their autoregressive token-level generation, the reasoning process is largely constrained to local decision-making and lacks global planning. This limitation frequently results in redundant, incoherent, or inaccurate reasoning, which significantly degrades overall performance. Existing approaches, such as tree-based algorithms and reinforcement learning (RL), attempt to address this issue but suffer from high computational costs and often fail to produce optimal reasoning trajectories. To tackle this challenge, we propose Plan-Then-Action Enhanced Reasoning with Group Relative Policy Optimization PTA-GRPO, a two-stage framework designed to improve both high-level planning and fine-grained CoT reasoning. In the first stage, we leverage advanced LLMs to distill CoT into compact high-level guidance, which is then used for supervised fine-tuning (SFT). In the second stage, we introduce a guidance-aware RL method that jointly optimizes the final output and the quality of high-level guidance, thereby enhancing reasoning effectiveness. We conduct extensive experiments on multiple mathematical reasoning benchmarks, including MATH, AIME2024, AIME2025, and AMC, across diverse base models such as Qwen2.5-7B-Instruct, Qwen3-8B, Qwen3-14B, and LLaMA3.2-3B. Experimental results demonstrate that PTA-GRPO consistently achieves stable and significant improvements across different models and tasks, validating its effectiveness and generalization.
>
---
#### [new 093] Is It Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI安全任务，解决奖励黑客问题。提出TRACE方法，通过测量推理过程的效率检测隐式奖励黑客行为。**

- **链接: [http://arxiv.org/pdf/2510.01367v1](http://arxiv.org/pdf/2510.01367v1)**

> **作者:** Xinpeng Wang; Nitish Joshi; Barbara Plank; Rico Angell; He He
>
> **摘要:** Reward hacking, where a reasoning model exploits loopholes in a reward function to achieve high rewards without solving the intended task, poses a significant threat. This behavior may be explicit, i.e. verbalized in the model's chain-of-thought (CoT), or implicit, where the CoT appears benign thus bypasses CoT monitors. To detect implicit reward hacking, we propose TRACE (Truncated Reasoning AUC Evaluation). Our key observation is that hacking occurs when exploiting the loophole is easier than solving the actual task. This means that the model is using less `effort' than required to achieve high reward. TRACE quantifies effort by measuring how early a model's reasoning becomes sufficient to pass a verifier. We progressively truncate a model's CoT at various lengths, force the model to answer, and measure the verifier-passing rate at each cutoff. A hacking model, which takes a shortcut, will achieve a high passing rate with only a small fraction of its CoT, yielding a large area under the accuracy-vs-length curve. TRACE achieves over 65% gains over our strongest 72B CoT monitor in math reasoning, and over 30% gains over a 32B monitor in coding. We further show that TRACE can discover unknown loopholes during training. Overall, TRACE offers a scalable unsupervised approach for oversight where current monitoring methods prove ineffective.
>
---
#### [new 094] StockBench: Can LLM Agents Trade Stocks Profitably In Real-world Markets?
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于金融与AI交叉任务，旨在评估LLM在真实股市中的交易能力。通过构建StockBench基准，测试模型的交易策略与风险管理效果。**

- **链接: [http://arxiv.org/pdf/2510.02209v1](http://arxiv.org/pdf/2510.02209v1)**

> **作者:** Yanxu Chen; Zijun Yao; Yantao Liu; Jin Ye; Jianing Yu; Lei Hou; Juanzi Li
>
> **摘要:** Large language models (LLMs) have recently demonstrated strong capabilities as autonomous agents, showing promise in reasoning, tool use, and sequential decision-making. While prior benchmarks have evaluated LLM agents in domains such as software engineering and scientific discovery, the finance domain remains underexplored, despite its direct relevance to economic value and high-stakes decision-making. Existing financial benchmarks primarily test static knowledge through question answering, but they fall short of capturing the dynamic and iterative nature of trading. To address this gap, we introduce StockBench, a contamination-free benchmark designed to evaluate LLM agents in realistic, multi-month stock trading environments. Agents receive daily market signals -- including prices, fundamentals, and news -- and must make sequential buy, sell, or hold decisions. Performance is assessed using financial metrics such as cumulative return, maximum drawdown, and the Sortino ratio. Our evaluation of state-of-the-art proprietary (e.g., GPT-5, Claude-4) and open-weight (e.g., Qwen3, Kimi-K2, GLM-4.5) models shows that while most LLM agents struggle to outperform the simple buy-and-hold baseline, several models demonstrate the potential to deliver higher returns and manage risk more effectively. These findings highlight both the challenges and opportunities in developing LLM-powered financial agents, showing that excelling at static financial knowledge tasks does not necessarily translate into successful trading strategies. We release StockBench as an open-source resource to support reproducibility and advance future research in this domain.
>
---
#### [new 095] VOGUE: Guiding Exploration with Visual Uncertainty Improves Multimodal Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态推理任务，解决MLLMs在探索中的不足。通过引入VOGUE方法，利用视觉不确定性引导探索，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2510.01444v1](http://arxiv.org/pdf/2510.01444v1)**

> **作者:** Rui Liu; Dian Yu; Tong Zheng; Runpeng Dai; Zongxia Li; Wenhao Yu; Zhenwen Liang; Linfeng Song; Haitao Mi; Pratap Tokekar; Dong Yu
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) improves reasoning in large language models (LLMs) but struggles with exploration, an issue that still persists for multimodal LLMs (MLLMs). Current methods treat the visual input as a fixed, deterministic condition, overlooking a critical source of ambiguity and struggling to build policies robust to plausible visual variations. We introduce $\textbf{VOGUE (Visual Uncertainty Guided Exploration)}$, a novel method that shifts exploration from the output (text) to the input (visual) space. By treating the image as a stochastic context, VOGUE quantifies the policy's sensitivity to visual perturbations using the symmetric KL divergence between a "raw" and "noisy" branch, creating a direct signal for uncertainty-aware exploration. This signal shapes the learning objective via an uncertainty-proportional bonus, which, combined with a token-entropy bonus and an annealed sampling schedule, effectively balances exploration and exploitation. Implemented within GRPO on two model scales (Qwen2.5-VL-3B/7B), VOGUE boosts pass@1 accuracy by an average of 2.6% on three visual math benchmarks and 3.7% on three general-domain reasoning benchmarks, while simultaneously increasing pass@4 performance and mitigating the exploration decay commonly observed in RL fine-tuning. Our work shows that grounding exploration in the inherent uncertainty of visual inputs is an effective strategy for improving multimodal reasoning.
>
---
#### [new 096] Demystifying Synthetic Data in LLM Pre-training: A Systematic Study of Scaling Laws, Benefits, and Pitfalls
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理领域，研究合成数据在大模型预训练中的作用，旨在评估其效果与风险。通过大量实验比较不同数据组合的效果，提供实用指导。**

- **链接: [http://arxiv.org/pdf/2510.01631v1](http://arxiv.org/pdf/2510.01631v1)**

> **作者:** Feiyang Kang; Newsha Ardalani; Michael Kuchnik; Youssef Emad; Mostafa Elhoushi; Shubhabrata Sengupta; Shang-Wen Li; Ramya Raghavendra; Ruoxi Jia; Carole-Jean Wu
>
> **备注:** Published as a Main Conference paper at EMNLP 2025
>
> **摘要:** Training data plays a crucial role in Large Language Models (LLM) scaling, yet high quality data is of limited supply. Synthetic data techniques offer a potential path toward sidestepping these limitations. We conduct a large-scale empirical investigation (>1000 LLMs with >100k GPU hours) using a unified protocol and scaling laws, comparing natural web data, diverse synthetic types (rephrased text, generated textbooks), and mixtures of natural and synthetic data. Specifically, we found pre-training on rephrased synthetic data \textit{alone} is not faster than pre-training on natural web texts; while pre-training on 1/3 rephrased synthetic data mixed with 2/3 natural web texts can speed up 5-10x (to reach the same validation loss) at larger data budgets. Pre-training on textbook-style synthetic data \textit{alone} results in notably higher loss on many downstream domains especially at small data budgets. "Good" ratios of synthetic data in training data mixtures depend on the model size and data budget, empirically converging to ~30% for rephrased synthetic data. Larger generator models do not necessarily yield better pre-training data than ~8B-param models. These results contribute mixed evidence on "model collapse" during large-scale single-round (n=1) model training on synthetic data--training on rephrased synthetic data shows no degradation in performance in foreseeable scales whereas training on mixtures of textbook-style pure-generated synthetic data shows patterns predicted by "model collapse". Our work demystifies synthetic data in pre-training, validates its conditional benefits, and offers practical guidance.
>
---
#### [new 097] WAInjectBench: Benchmarking Prompt Injection Detections for Web Agents
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于安全检测任务，旨在解决Web代理的提示注入攻击检测问题。工作包括构建基准数据集、系统化检测方法并评估性能。**

- **链接: [http://arxiv.org/pdf/2510.01354v1](http://arxiv.org/pdf/2510.01354v1)**

> **作者:** Yinuo Liu; Ruohan Xu; Xilong Wang; Yuqi Jia; Neil Zhenqiang Gong
>
> **摘要:** Multiple prompt injection attacks have been proposed against web agents. At the same time, various methods have been developed to detect general prompt injection attacks, but none have been systematically evaluated for web agents. In this work, we bridge this gap by presenting the first comprehensive benchmark study on detecting prompt injection attacks targeting web agents. We begin by introducing a fine-grained categorization of such attacks based on the threat model. We then construct datasets containing both malicious and benign samples: malicious text segments generated by different attacks, benign text segments from four categories, malicious images produced by attacks, and benign images from two categories. Next, we systematize both text-based and image-based detection methods. Finally, we evaluate their performance across multiple scenarios. Our key findings show that while some detectors can identify attacks that rely on explicit textual instructions or visible image perturbations with moderate to high accuracy, they largely fail against attacks that omit explicit instructions or employ imperceptible perturbations. Our datasets and code are released at: https://github.com/Norrrrrrr-lyn/WAInjectBench.
>
---
#### [new 098] LSPO: Length-aware Dynamic Sampling for Policy Optimization in LLM Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大语言模型训练任务，旨在解决RLVR效率问题。提出LSPO算法，根据响应长度动态采样，提升学习效果。**

- **链接: [http://arxiv.org/pdf/2510.01459v1](http://arxiv.org/pdf/2510.01459v1)**

> **作者:** Weizhe Chen; Sven Koenig; Bistra Dilkina
>
> **摘要:** Since the release of Deepseek-R1, reinforcement learning with verifiable rewards (RLVR) has become a central approach for training large language models (LLMs) on reasoning tasks. Recent work has largely focused on modifying loss functions to make RLVR more efficient and effective. In this paper, motivated by studies of overthinking in LLMs, we propose Length-aware Sampling for Policy Optimization (LSPO), a novel meta-RLVR algorithm that dynamically selects training data at each step based on the average response length. We evaluate LSPO across multiple base models and datasets, demonstrating that it consistently improves learning effectiveness. In addition, we conduct a detailed ablation study to examine alternative ways of incorporating length signals into dynamic sampling, offering further insights and highlighting promising directions for future research.
>
---
#### [new 099] Quagmires in SFT-RL Post-Training: When High SFT Scores Mislead and What to Use Instead
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究LLM后训练中的SFT与RL关系，指出高SFT分数可能误导性能评估，提出用泛化损失和Pass@large k作为更可靠指标。任务是优化模型后训练效果。**

- **链接: [http://arxiv.org/pdf/2510.01624v1](http://arxiv.org/pdf/2510.01624v1)**

> **作者:** Feiyang Kang; Michael Kuchnik; Karthik Padthe; Marin Vlastelica; Ruoxi Jia; Carole-Jean Wu; Newsha Ardalani
>
> **备注:** Preprint. Under Review
>
> **摘要:** In post-training for reasoning Large Language Models (LLMs), the current state of practice trains LLMs in two independent stages: Supervised Fine-Tuning (SFT) and Reinforcement Learning with Verifiable Rewards (RLVR, shortened as ``RL'' below). In this work, we challenge whether high SFT scores translate to improved performance after RL. We provide extensive counter-examples where this is not true. We find high SFT scores can be biased toward simpler or more homogeneous data and are not reliably predictive of subsequent RL gains or scaled-up post-training effectiveness. In some cases, RL training on models with improved SFT performance could lead to substantially worse outcome compared to RL on the base model without SFT. We study alternative metrics and identify generalization loss on held-out reasoning examples and Pass@large k performance to provide strong proxies for the RL outcome. We trained hundreds of models up to 12B-parameter with SFT and RLVR via GRPO and ran extensive evaluations on 7 math benchmarks with up to 256 repetitions, spending $>$1M GPU hours. Experiments include models from Llama3, Mistral-Nemo, Qwen3 and multiple state-of-the-art SFT/RL datasets. Compared to directly predicting from pre-RL performance, prediction based on generalization loss and Pass@large k achieves substantial higher precision, improving $R^2$ coefficient and Spearman's rank correlation coefficient by up to 0.5 (2x). This provides strong utility for broad use cases. For example, in most experiments, we find SFT training on unique examples for a one epoch underperforms training on half examples for two epochs, either after SFT or SFT-then-RL; With the same SFT budget, training only on short examples may lead to better SFT performance, though, it often leads to worse outcome after RL compared to training on examples with varying lengths. Evaluation tool will be open-sourced.
>
---
#### [new 100] The Unreasonable Effectiveness of Scaling Agents for Computer Use
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于计算机使用代理任务，解决CUAs可靠性与泛化性问题。提出bBoN方法，通过多轨迹选择提升性能，达到新SOTA。**

- **链接: [http://arxiv.org/pdf/2510.02250v1](http://arxiv.org/pdf/2510.02250v1)**

> **作者:** Gonzalo Gonzalez-Pumariega; Vincent Tu; Chih-Lun Lee; Jiachen Yang; Ang Li; Xin Eric Wang
>
> **备注:** 23 pages, 7 figures, 10 tables
>
> **摘要:** Computer-use agents (CUAs) hold promise for automating everyday digital tasks, but their unreliability and high variance hinder their application to long-horizon, complex tasks. We introduce Behavior Best-of-N (bBoN), a method that scales over agents by generating multiple rollouts and selecting among them using behavior narratives that describe the agents' rollouts. It enables both wide exploration and principled trajectory selection, substantially improving robustness and success rates. On OSWorld, our bBoN scaling method establishes a new state of the art (SoTA) at 69.9%, significantly outperforming prior methods and approaching human-level performance at 72%, with comprehensive ablations validating key design choices. We further demonstrate strong generalization results to different operating systems on WindowsAgentArena and AndroidWorld. Crucially, our results highlight the unreasonable effectiveness of scaling CUAs, when you do it right: effective scaling requires structured trajectory understanding and selection, and bBoN provides a practical framework to achieve this.
>
---
#### [new 101] Jailbreaking LLMs via Semantically Relevant Nested Scenarios with Targeted Toxic Knowledge
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于安全攻击任务，旨在解决LLM防御漏洞问题。通过构建语义相关嵌套场景，提出RTS-Attack框架，有效绕过模型对齐防御。**

- **链接: [http://arxiv.org/pdf/2510.01223v1](http://arxiv.org/pdf/2510.01223v1)**

> **作者:** Hui Dou; Ning Xu; Yiwen Zhang; Kaibin Wang
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in various tasks. However, they remain exposed to jailbreak attacks, eliciting harmful responses. The nested scenario strategy has been increasingly adopted across various methods, demonstrating immense potential. Nevertheless, these methods are easily detectable due to their prominent malicious intentions. In this work, we are the first to find and systematically verify that LLMs' alignment defenses are not sensitive to nested scenarios, where these scenarios are highly semantically relevant to the queries and incorporate targeted toxic knowledge. This is a crucial yet insufficiently explored direction. Based on this, we propose RTS-Attack (Semantically Relevant Nested Scenarios with Targeted Toxic Knowledge), an adaptive and automated framework to examine LLMs' alignment. By building scenarios highly relevant to the queries and integrating targeted toxic knowledge, RTS-Attack bypasses the alignment defenses of LLMs. Moreover, the jailbreak prompts generated by RTS-Attack are free from harmful queries, leading to outstanding concealment. Extensive experiments demonstrate that RTS-Attack exhibits superior performance in both efficiency and universality compared to the baselines across diverse advanced LLMs, including GPT-4o, Llama3-70b, and Gemini-pro. Our complete code is available in the supplementary material. WARNING: THIS PAPER CONTAINS POTENTIALLY HARMFUL CONTENT.
>
---
#### [new 102] Synthetic Prefixes to Mitigate Bias in Real-Time Neural Query Autocomplete
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于推荐系统任务，旨在解决实时搜索中因展示偏差导致的排名问题。通过生成合成前缀丰富训练数据，提升模型泛化能力与公平性。**

- **链接: [http://arxiv.org/pdf/2510.01574v1](http://arxiv.org/pdf/2510.01574v1)**

> **作者:** Adithya Rajan; Xiaoyu Liu; Prateek Verma; Vibhu Arora
>
> **备注:** Accepted to the Proceedings of the ACM SIGIR Asia Pacific Conference on Information Retrieval (SIGIR-AP 2025), December 7-10, 2025, Xi'an, China
>
> **摘要:** We introduce a data-centric approach for mitigating presentation bias in real-time neural query autocomplete systems through the use of synthetic prefixes. These prefixes are generated from complete user queries collected during regular search sessions where autocomplete was not active. This allows us to enrich the training data for learning to rank models with more diverse and less biased examples. This method addresses the inherent bias in engagement signals collected from live query autocomplete interactions, where model suggestions influence user behavior. Our neural ranker is optimized for real-time deployment under strict latency constraints and incorporates a rich set of features, including query popularity, seasonality, fuzzy match scores, and contextual signals such as department affinity, device type, and vertical alignment with previous user queries. To support efficient training, we introduce a task-specific simplification of the listwise loss, reducing computational complexity from $O(n^2)$ to $O(n)$ by leveraging the query autocomplete structure of having only one ground-truth selection per prefix. Deployed in a large-scale e-commerce setting, our system demonstrates statistically significant improvements in user engagement, as measured by mean reciprocal rank and related metrics. Our findings show that synthetic prefixes not only improve generalization but also provide a scalable path toward bias mitigation in other low-latency ranking tasks, including related searches and query recommendations.
>
---
#### [new 103] InvThink: Towards AI Safety via Inverse Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI安全任务，旨在提升语言模型的安全性。通过逆向推理方法InvThink，模型先分析潜在危害再生成安全响应，有效减少有害输出。**

- **链接: [http://arxiv.org/pdf/2510.01569v1](http://arxiv.org/pdf/2510.01569v1)**

> **作者:** Yubin Kim; Taehan Kim; Eugene Park; Chunjong Park; Cynthia Breazeal; Daniel McDuff; Hae Won Park
>
> **摘要:** We present InvThink, a simple yet powerful approach that gives large language models (LLMs) the capability of inverse thinking: reasoning through failure modes before generating responses. Unlike existing safety alignment methods that optimize directly for safe response, InvThink instructs models to 1) enumerate potential harms, 2) analyze their consequences, and 3) generate safe outputs that proactively avoid these risks. Our method reveals three key findings: (i) safety improvements show stronger scaling with model size compared to existing safety methods. (ii) InvThink mitigates safety tax; by training models to systematically consider failure modes, it preserves general reasoning capabilities on standard benchmarks. (iii) beyond general safety tasks, InvThink excels in high-stakes domains including external-facing (medicine, finance, law) and agentic (blackmail, murder) risk scenarios, achieving up to 15.7% reduction in harmful responses compared to baseline methods like SafetyPrompt. We further implement InvThink via supervised fine-tuning, and reinforcement learning across three LLM families. These results suggest that inverse reasoning provides a scalable and generalizable path toward safer, more capable language models.
>
---
#### [new 104] A Rigorous Benchmark with Multidimensional Evaluation for Deep Research Agents: From Answers to Reports
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于人工智能评估任务，旨在解决DRAs性能评价不足的问题。提出多维基准和评估框架，提升对复杂任务处理能力的衡量。**

- **链接: [http://arxiv.org/pdf/2510.02190v1](http://arxiv.org/pdf/2510.02190v1)**

> **作者:** Yang Yao; Yixu Wang; Yuxuan Zhang; Yi Lu; Tianle Gu; Lingyu Li; Dingyi Zhao; Keming Wu; Haozhe Wang; Ping Nie; Yan Teng; Yingchun Wang
>
> **摘要:** Artificial intelligence is undergoing the paradigm shift from closed language models to interconnected agent systems capable of external perception and information integration. As a representative embodiment, Deep Research Agents (DRAs) systematically exhibit the capabilities for task decomposition, cross-source retrieval, multi-stage reasoning, and structured output, which markedly enhance performance on complex and open-ended tasks. However, existing benchmarks remain deficient in evaluation dimensions, response formatting, and scoring mechanisms, limiting their capacity to assess such systems effectively. This paper introduces a rigorous benchmark and a multidimensional evaluation framework tailored to DRAs and report-style responses. The benchmark comprises 214 expert-curated challenging queries distributed across 10 broad thematic domains, each accompanied by manually constructed reference bundles to support composite evaluation. The framework enables comprehensive evaluation of long-form reports generated by DRAs, incorporating integrated scoring metrics for semantic quality, topical focus, and retrieval trustworthiness. Extensive experimentation confirms the superior performance of mainstream DRAs over web-search-tool-augmented reasoning models, yet reveals considerable scope for further improvement. This study provides a robust foundation for capability assessment, architectural refinement, and paradigm advancement in DRA systems.
>
---
#### [new 105] Optimal Stopping vs Best-of-$N$ for Inference Time Optimization
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言模型推理优化任务，解决如何在保证质量的前提下减少生成次数。通过引入最优停止理论，提出一种自适应策略，显著降低生成成本。**

- **链接: [http://arxiv.org/pdf/2510.01394v1](http://arxiv.org/pdf/2510.01394v1)**

> **作者:** Yusuf Kalayci; Vinod Raman; Shaddin Dughmi
>
> **备注:** 24 pages
>
> **摘要:** Large language model (LLM) generation often requires balancing output quality against inference cost, especially when using multiple generations. We introduce a new framework for inference-time optimization based on the classical Pandora's Box problem. Viewing each generation as opening a costly "box" with random reward, we develop algorithms that decide when to stop generating without knowing the underlying reward distribution. Our first contribution is a UCB-style Pandora's Box algorithm, which achieves performance that is provably close to Weitzman's algorithm, the optimal strategy when the distribution is known. We further adapt this method to practical LLM settings by addressing reward scaling across prompts via a Bradley-Terry inspired transformation. This leads to an adaptive inference-time optimization method that normalizes rewards and learns stopping thresholds on the fly. Experiments on the AlpacaFarm and HH-RLHF datasets, using multiple LLM-reward model pairs, show that our adaptive strategy can obtain the same performance as non-adaptive Best-of-N sampling while requiring 15-35 percent fewer generations on average. Our results establish a principled bridge between optimal stopping theory and inference-time scaling, providing both theoretical performance bounds and practical efficiency gains for LLM deployment.
>
---
#### [new 106] Position: Privacy Is Not Just Memorization!
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于隐私安全研究，旨在解决LLM中的多维度隐私风险。通过分析文献和案例，指出当前研究忽视了除数据记忆外的其他威胁，呼吁更全面的应对策略。**

- **链接: [http://arxiv.org/pdf/2510.01645v1](http://arxiv.org/pdf/2510.01645v1)**

> **作者:** Niloofar Mireshghallah; Tianshi Li
>
> **备注:** 27 pages, 6 figures, 2 tables
>
> **摘要:** The discourse on privacy risks in Large Language Models (LLMs) has disproportionately focused on verbatim memorization of training data, while a constellation of more immediate and scalable privacy threats remain underexplored. This position paper argues that the privacy landscape of LLM systems extends far beyond training data extraction, encompassing risks from data collection practices, inference-time context leakage, autonomous agent capabilities, and the democratization of surveillance through deep inference attacks. We present a comprehensive taxonomy of privacy risks across the LLM lifecycle -- from data collection through deployment -- and demonstrate through case studies how current privacy frameworks fail to address these multifaceted threats. Through a longitudinal analysis of 1,322 AI/ML privacy papers published at leading conferences over the past decade (2016--2025), we reveal that while memorization receives outsized attention in technical research, the most pressing privacy harms lie elsewhere, where current technical approaches offer little traction and viable paths forward remain unclear. We call for a fundamental shift in how the research community approaches LLM privacy, moving beyond the narrow focus of current technical solutions and embracing interdisciplinary approaches that address the sociotechnical nature of these emerging threats.
>
---
#### [new 107] ExGRPO: Learning to Reason from Experience
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决大语言模型推理能力提升问题。通过分析经验价值，提出ExGRPO框架，提升训练效率与稳定性。**

- **链接: [http://arxiv.org/pdf/2510.02245v1](http://arxiv.org/pdf/2510.02245v1)**

> **作者:** Runzhe Zhan; Yafu Li; Zhi Wang; Xiaoye Qu; Dongrui Liu; Jing Shao; Derek F. Wong; Yu Cheng
>
> **摘要:** Reinforcement learning from verifiable rewards (RLVR) is an emerging paradigm for improving the reasoning ability of large language models. However, standard on-policy training discards rollout experiences after a single update, leading to computational inefficiency and instability. While prior work on RL has highlighted the benefits of reusing past experience, the role of experience characteristics in shaping learning dynamics of large reasoning models remains underexplored. In this paper, we are the first to investigate what makes a reasoning experience valuable and identify rollout correctness and entropy as effective indicators of experience value. Based on these insights, we propose ExGRPO (Experiential Group Relative Policy Optimization), a framework that organizes and prioritizes valuable experiences, and employs a mixed-policy objective to balance exploration with experience exploitation. Experiments on five backbone models (1.5B-8B parameters) show that ExGRPO consistently improves reasoning performance on mathematical/general benchmarks, with an average gain of +3.5/7.6 points over on-policy RLVR. Moreover, ExGRPO stabilizes training on both stronger and weaker models where on-policy methods fail. These results highlight principled experience management as a key ingredient for efficient and scalable RLVR.
>
---
#### [new 108] Agentic Jigsaw Interaction Learning for Enhancing Visual Perception and Reasoning in Vision-Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型的感知与推理任务，旨在解决VLMs在简单拼图任务中表现不佳的问题。通过AGILE方法，利用交互式学习提升模型能力。**

- **链接: [http://arxiv.org/pdf/2510.01304v1](http://arxiv.org/pdf/2510.01304v1)**

> **作者:** Yu Zeng; Wenxuan Huang; Shiting Huang; Xikun Bao; Yukun Qi; Yiming Zhao; Qiuchen Wang; Lin Chen; Zehui Chen; Huaian Chen; Wanli Ouyang; Feng Zhao
>
> **摘要:** Although current large Vision-Language Models (VLMs) have advanced in multimodal understanding and reasoning, their fundamental perceptual and reasoning abilities remain limited. Specifically, even on simple jigsaw tasks, existing VLMs perform near randomly, revealing deficiencies in core perception and reasoning capabilities. While high-quality vision-language data can enhance these capabilities, its scarcity and limited scalability impose significant constraints. To address this, we propose AGILE, an Agentic jiGsaw Interaction Learning for Enhancing visual perception and reasoning in VLMs. AGILE formulates jigsaw solving as an interactive process, enabling the model to progressively engage with the environment. At each step, the model generates executable code to perform an action based on the current state, while the environment provides fine-grained visual feedback to guide task completion. Through this iterative cycle of observation and interaction, the model incrementally improves its perceptual and reasoning capabilities via exploration and feedback. Experimental results show that AGILE not only substantially boosts performance on jigsaw tasks of varying complexity (e.g., increasing accuracy from 9.5% to 82.8% under the 2 $\times$ 2 setting) but also demonstrates strong generalization across 9 general vision tasks, achieving an average improvement of 3.1%. These results indicate notable enhancements in both perceptual and reasoning abilities. This work opens a new avenue for advancing reasoning and generalization in multimodal models and provides an efficient, scalable solution to the scarcity of multimodal reinforcement learning data. The code and datasets is available at https://github.com/yuzeng0-0/AGILE .
>
---
#### [new 109] Bridging Collaborative Filtering and Large Language Models with Dynamic Alignment, Multimodal Fusion and Evidence-grounded Explanations
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于推荐系统任务，解决用户偏好变化、多模态信息利用及解释性不足的问题。提出框架融合协同过滤与大模型，实现动态适应、多模态融合和证据支持的解释。**

- **链接: [http://arxiv.org/pdf/2510.01606v1](http://arxiv.org/pdf/2510.01606v1)**

> **作者:** Bo Ma; LuYao Liu; Simon Lau; Chandler Yuan; and XueY Cui; Rosie Zhang
>
> **摘要:** Recent research has explored using Large Language Models for recommendation tasks by transforming user interaction histories and item metadata into text prompts, then having the LLM produce rankings or recommendations. A promising approach involves connecting collaborative filtering knowledge to LLM representations through compact adapter networks, which avoids expensive fine-tuning while preserving the strengths of both components. Yet several challenges persist in practice: collaborative filtering models often use static snapshots that miss rapidly changing user preferences; many real-world items contain rich visual and audio content beyond textual descriptions; and current systems struggle to provide trustworthy explanations backed by concrete evidence. Our work introduces \model{}, a framework that tackles these limitations through three key innovations. We develop an online adaptation mechanism that continuously incorporates new user interactions through lightweight modules, avoiding the need to retrain large models. We create a unified representation that seamlessly combines collaborative signals with visual and audio features, handling cases where some modalities may be unavailable. Finally, we design an explanation system that grounds recommendations in specific collaborative patterns and item attributes, producing natural language rationales users can verify. Our approach maintains the efficiency of frozen base models while adding minimal computational overhead, making it practical for real-world deployment.
>
---
#### [new 110] Just Do It!? Computer-Use Agents Exhibit Blind Goal-Directedness
- **分类: cs.AI; cs.CL; cs.CR; cs.CY; cs.LG**

- **简介: 该论文研究计算机使用代理的盲目标导向行为，旨在识别其在执行任务时忽视安全与上下文的问题。通过构建基准测试BLIND-ACT，评估多个模型并发现高风险行为。**

- **链接: [http://arxiv.org/pdf/2510.01670v1](http://arxiv.org/pdf/2510.01670v1)**

> **作者:** Erfan Shayegani; Keegan Hines; Yue Dong; Nael Abu-Ghazaleh; Roman Lutz; Spencer Whitehead; Vidhisha Balachandran; Besmira Nushi; Vibhav Vineet
>
> **摘要:** Computer-Use Agents (CUAs) are an increasingly deployed class of agents that take actions on GUIs to accomplish user goals. In this paper, we show that CUAs consistently exhibit Blind Goal-Directedness (BGD): a bias to pursue goals regardless of feasibility, safety, reliability, or context. We characterize three prevalent patterns of BGD: (i) lack of contextual reasoning, (ii) assumptions and decisions under ambiguity, and (iii) contradictory or infeasible goals. We develop BLIND-ACT, a benchmark of 90 tasks capturing these three patterns. Built on OSWorld, BLIND-ACT provides realistic environments and employs LLM-based judges to evaluate agent behavior, achieving 93.75% agreement with human annotations. We use BLIND-ACT to evaluate nine frontier models, including Claude Sonnet and Opus 4, Computer-Use-Preview, and GPT-5, observing high average BGD rates (80.8%) across them. We show that BGD exposes subtle risks that arise even when inputs are not directly harmful. While prompting-based interventions lower BGD levels, substantial risk persists, highlighting the need for stronger training- or inference-time interventions. Qualitative analysis reveals observed failure modes: execution-first bias (focusing on how to act over whether to act), thought-action disconnect (execution diverging from reasoning), and request-primacy (justifying actions due to user request). Identifying BGD and introducing BLIND-ACT establishes a foundation for future research on studying and mitigating this fundamental risk and ensuring safe CUA deployment.
>
---
#### [new 111] Do AI Models Perform Human-like Abstract Reasoning Across Modalities?
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究AI模型在多模态下的抽象推理能力，旨在评估其是否像人类一样理解抽象概念。通过不同模态和工具设置的实验，发现模型常依赖表面规律而非深层抽象，导致对能力的误判。**

- **链接: [http://arxiv.org/pdf/2510.02125v1](http://arxiv.org/pdf/2510.02125v1)**

> **作者:** Claas Beger; Ryan Yi; Shuhao Fu; Arseny Moskvichev; Sarah W. Tsai; Sivasankaran Rajamanickam; Melanie Mitchell
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** OpenAI's o3-preview reasoning model exceeded human accuracy on the ARC-AGI benchmark, but does that mean state-of-the-art models recognize and reason with the abstractions that the task creators intended? We investigate models' abstraction abilities on ConceptARC. We evaluate models under settings that vary the input modality (textual vs. visual), whether the model is permitted to use external Python tools, and, for reasoning models, the amount of reasoning effort. In addition to measuring output accuracy, we perform fine-grained evaluation of the natural-language rules that models generate to explain their solutions. This dual evaluation lets us assess whether models solve tasks using the abstractions ConceptARC was designed to elicit, rather than relying on surface-level patterns. Our results show that, while some models using text-based representations match human output accuracy, the best models' rules are often based on surface-level ``shortcuts'' and capture intended abstractions far less often than humans. Thus their capabilities for general abstract reasoning may be overestimated by evaluations based on accuracy alone. In the visual modality, AI models' output accuracy drops sharply, yet our rule-level analysis reveals that models might be underestimated, as they still exhibit a substantial share of rules that capture intended abstractions, but are often unable to correctly apply these rules. In short, our results show that models still lag humans in abstract reasoning, and that using accuracy alone to evaluate abstract reasoning on ARC-like tasks may overestimate abstract-reasoning capabilities in textual modalities and underestimate it in visual modalities. We believe that our evaluation framework offers a more faithful picture of multimodal models' abstract reasoning abilities and a more principled way to track progress toward human-like, abstraction-centered intelligence.
>
---
#### [new 112] Sparse Query Attention (SQA): A Computationally Efficient Attention Mechanism with Query Heads Reduction
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决Transformer中注意力机制计算复杂度高的问题。通过减少查询头数量，提出SQA方法降低FLOPs，提升计算效率。**

- **链接: [http://arxiv.org/pdf/2510.01817v1](http://arxiv.org/pdf/2510.01817v1)**

> **作者:** Adam Filipek
>
> **备注:** 18 pages, 6 figures, small-scale experiments
>
> **摘要:** The Transformer architecture, underpinned by the Multi-Head Attention (MHA) mechanism, has become the de facto standard for state-of-the-art models in artificial intelligence. However, the quadratic computational complexity of MHA with respect to sequence length presents a significant barrier to scaling, particularly for applications involving long contexts. Prevailing solutions, such as Multi-Query Attention (MQA) and Grouped-Query Attention (GQA), have effectively addressed the memory bandwidth bottleneck that dominates autoregressive inference latency by sharing Key and Value projections. While highly successful, these methods do not reduce the fundamental number of floating-point operations (FLOPs) required for the attention score computation, which remains a critical bottleneck for training and full-sequence processing. This paper introduces Sparse Query Attention (SQA), a novel attention architecture that pursues an alternative and complementary optimization path. Instead of reducing Key/Value heads, SQA reduces the number of Query heads. This architectural modification directly decreases the computational complexity of the attention mechanism by a factor proportional to the reduction in query heads, thereby lowering the overall FLOPs. This work presents the theoretical foundation of SQA, its mathematical formulation, and a family of architectural variants. Empirical benchmarks on long sequences (32k-200k tokens) demonstrate that SQA can achieve significant throughput improvements of up to 3x in computation-bound scenarios such as model pre-training, fine-tuning, and encoder-based tasks, with only a minimal impact on model quality in preliminary smallscale experiments. SQA was discovered serendipitously during the development of the upcoming Reactive Transformer architecture, suggesting its potential as a powerful tool for building more efficient and scalable models
>
---
#### [new 113] MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于多平台动态智能体的长期记忆与状态跟踪任务，旨在解决企业环境中记忆评估难题，通过构建MEMTRACK基准测试相关能力。**

- **链接: [http://arxiv.org/pdf/2510.01353v1](http://arxiv.org/pdf/2510.01353v1)**

> **作者:** Darshan Deshpande; Varun Gangal; Hersh Mehta; Anand Kannappan; Rebecca Qian; Peng Wang
>
> **备注:** Accepted to NeurIPS 2025 SEA Workshop
>
> **摘要:** Recent works on context and memory benchmarking have primarily focused on conversational instances but the need for evaluating memory in dynamic enterprise environments is crucial for its effective application. We introduce MEMTRACK, a benchmark designed to evaluate long-term memory and state tracking in multi-platform agent environments. MEMTRACK models realistic organizational workflows by integrating asynchronous events across multiple communication and productivity platforms such as Slack, Linear and Git. Each benchmark instance provides a chronologically platform-interleaved timeline, with noisy, conflicting, cross-referring information as well as potential codebase/file-system comprehension and exploration. Consequently, our benchmark tests memory capabilities such as acquistion, selection and conflict resolution. We curate the MEMTRACK dataset through both manual expert driven design and scalable agent based synthesis, generating ecologically valid scenarios grounded in real world software development processes. We introduce pertinent metrics for Correctness, Efficiency, and Redundancy that capture the effectiveness of memory mechanisms beyond simple QA performance. Experiments across SoTA LLMs and memory backends reveal challenges in utilizing memory across long horizons, handling cross-platform dependencies, and resolving contradictions. Notably, the best performing GPT-5 model only achieves a 60\% Correctness score on MEMTRACK. This work provides an extensible framework for advancing evaluation research for memory-augmented agents, beyond existing focus on conversational setups, and sets the stage for multi-agent, multi-platform memory benchmarking in complex organizational settings
>
---
#### [new 114] Interactive Training: Feedback-Driven Neural Network Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于深度学习优化任务，旨在解决传统训练方法缺乏灵活性的问题。提出Interactive Training框架，实现训练过程中的实时反馈干预，提升稳定性与适应性。**

- **链接: [http://arxiv.org/pdf/2510.02297v1](http://arxiv.org/pdf/2510.02297v1)**

> **作者:** Wentao Zhang; Yang Young Lu; Yuntian Deng
>
> **备注:** EMNLP 2025 Demo
>
> **摘要:** Traditional neural network training typically follows fixed, predefined optimization recipes, lacking the flexibility to dynamically respond to instabilities or emerging training issues. In this paper, we introduce Interactive Training, an open-source framework that enables real-time, feedback-driven intervention during neural network training by human experts or automated AI agents. At its core, Interactive Training uses a control server to mediate communication between users or agents and the ongoing training process, allowing users to dynamically adjust optimizer hyperparameters, training data, and model checkpoints. Through three case studies, we demonstrate that Interactive Training achieves superior training stability, reduced sensitivity to initial hyperparameters, and improved adaptability to evolving user needs, paving the way toward a future training paradigm where AI agents autonomously monitor training logs, proactively resolve instabilities, and optimize training dynamics.
>
---
#### [new 115] Tree-based Dialogue Reinforced Policy Optimization for Red-Teaming Attacks
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于AI安全领域，解决多轮对话中对抗攻击的问题。提出DialTree-RPO框架，通过强化学习自动发现多样化的多轮攻击策略。**

- **链接: [http://arxiv.org/pdf/2510.02286v1](http://arxiv.org/pdf/2510.02286v1)**

> **作者:** Ruohao Guo; Afshin Oroojlooy; Roshan Sridhar; Miguel Ballesteros; Alan Ritter; Dan Roth
>
> **摘要:** Despite recent rapid progress in AI safety, current large language models remain vulnerable to adversarial attacks in multi-turn interaction settings, where attackers strategically adapt their prompts across conversation turns and pose a more critical yet realistic challenge. Existing approaches that discover safety vulnerabilities either rely on manual red-teaming with human experts or employ automated methods using pre-defined templates and human-curated attack data, with most focusing on single-turn attacks. However, these methods did not explore the vast space of possible multi-turn attacks, failing to consider novel attack trajectories that emerge from complex dialogue dynamics and strategic conversation planning. This gap is particularly critical given recent findings that LLMs exhibit significantly higher vulnerability to multi-turn attacks compared to single-turn attacks. We propose DialTree-RPO, an on-policy reinforcement learning framework integrated with tree search that autonomously discovers diverse multi-turn attack strategies by treating the dialogue as a sequential decision-making problem, enabling systematic exploration without manually curated data. Through extensive experiments, our approach not only achieves more than 25.9% higher ASR across 10 target models compared to previous state-of-the-art approaches, but also effectively uncovers new attack strategies by learning optimal dialogue policies that maximize attack success across multiple turns.
>
---
#### [new 116] From Videos to Indexed Knowledge Graphs -- Framework to Marry Methods for Multimodal Content Analysis and Understanding
- **分类: cs.CV; cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于多模态内容分析任务，旨在解决视频数据处理与知识表示难题。提出框架融合预训练模型，将视频转为可查询的知识图谱，支持持续学习。**

- **链接: [http://arxiv.org/pdf/2510.01513v1](http://arxiv.org/pdf/2510.01513v1)**

> **作者:** Basem Rizk; Joel Walsh; Mark Core; Benjamin Nye
>
> **摘要:** Analysis of multi-modal content can be tricky, computationally expensive, and require a significant amount of engineering efforts. Lots of work with pre-trained models on static data is out there, yet fusing these opensource models and methods with complex data such as videos is relatively challenging. In this paper, we present a framework that enables efficiently prototyping pipelines for multi-modal content analysis. We craft a candidate recipe for a pipeline, marrying a set of pre-trained models, to convert videos into a temporal semi-structured data format. We translate this structure further to a frame-level indexed knowledge graph representation that is query-able and supports continual learning, enabling the dynamic incorporation of new domain-specific knowledge through an interactive medium.
>
---
#### [new 117] Constrained Adaptive Rejection Sampling
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的约束生成任务，旨在解决传统方法在有效性与效率间的权衡问题。提出CARS方法，在保持分布准确性的前提下提升采样效率。**

- **链接: [http://arxiv.org/pdf/2510.01902v1](http://arxiv.org/pdf/2510.01902v1)**

> **作者:** Paweł Parys; Sairam Vaidya; Taylor Berg-Kirkpatrick; Loris D'Antoni
>
> **摘要:** Language Models (LMs) are increasingly used in applications where generated outputs must satisfy strict semantic or syntactic constraints. Existing approaches to constrained generation fall along a spectrum: greedy constrained decoding methods enforce validity during decoding but distort the LM's distribution, while rejection sampling (RS) preserves fidelity but wastes computation by discarding invalid outputs. Both extremes are problematic in domains such as program fuzzing, where both validity and diversity of samples are essential. We present Constrained Adaptive Rejection Sampling (CARS), an approach that strictly improves the sample-efficiency of RS without distributional distortion. CARS begins with unconstrained LM sampling and adaptively rules out constraint-violating continuations by recording them in a trie and subtracting their probability mass from future draws. This adaptive pruning ensures that prefixes proven invalid are never revisited, acceptance rates improve monotonically, and the resulting samples exactly follow the constrained distribution. In experiments on a variety of domains -- e.g., program fuzzing and molecular generation -- CARS consistently achieves higher efficiency -- measured in the number of LM forward passes per valid sample -- while also producing stronger sample diversity than both GCD and methods that approximate the LM's distribution.
>
---
#### [new 118] Improving AGI Evaluation: A Data Science Perspective
- **分类: cs.AI; cs.CL**

- **简介: 论文探讨AGI评估方法，指出传统依赖直觉设计任务的不足，提出基于数据科学的稳健任务执行评估框架，旨在更可靠地衡量AGI能力。**

- **链接: [http://arxiv.org/pdf/2510.01687v1](http://arxiv.org/pdf/2510.01687v1)**

> **作者:** John Hawkins
>
> **摘要:** Evaluation of potential AGI systems and methods is difficult due to the breadth of the engineering goal. We have no methods for perfect evaluation of the end state, and instead measure performance on small tests designed to provide directional indication that we are approaching AGI. In this work we argue that AGI evaluation methods have been dominated by a design philosophy that uses our intuitions of what intelligence is to create synthetic tasks, that have performed poorly in the history of AI. Instead we argue for an alternative design philosophy focused on evaluating robust task execution that seeks to demonstrate AGI through competence. This perspective is developed from common practices in data science that are used to show that a system can be reliably deployed. We provide practical examples of what this would mean for AGI evaluation.
>
---
#### [new 119] Fine-tuning with RAG for Improving LLM Learning of New Skills
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM在执行多步骤任务时的失败问题。通过RAG微调，将检索信息转化为模型内部能力，提升性能并减少依赖。**

- **链接: [http://arxiv.org/pdf/2510.01375v1](http://arxiv.org/pdf/2510.01375v1)**

> **作者:** Humaid Ibrahim; Nikolai Rozanov; Marek Rei
>
> **备注:** Under review at ICLR 2026
>
> **摘要:** Large language model (LLM) agents deployed for multi-step tasks frequently fail in predictable ways: attempting actions with unmet preconditions, issuing redundant commands, or mishandling environment constraints. While retrieval-augmented generation (RAG) can improve performance by providing runtime guidance, it requires maintaining external knowledge databases and adds computational overhead at every deployment. We propose a simple pipeline that converts inference-time retrieval into learned competence through distillation. Our approach: (1) extracts compact, reusable hints from agent failures, (2) uses these hints to generate improved teacher trajectories via one-shot retrieval at episode start, and (3) trains student models on these trajectories with hint strings removed, forcing internalization rather than memorization. Across two interactive benchmarks, ALFWorld (household tasks) and WebShop (online shopping), distilled students consistently outperform baseline agents, achieving up to 91% success on ALFWorld (vs. 79% for baselines) and improving WebShop scores to 72 (vs. 61 for baselines), while using 10-60% fewer tokens than retrieval-augmented teachers depending on the environment. The approach generalizes across model scales (7B/14B parameters) and agent architectures (ReAct/StateAct), demonstrating that retrieval benefits can be effectively internalized through targeted fine-tuning without permanent runtime dependencies.
>
---
#### [new 120] LLM4Rec: Large Language Models for Multimodal Generative Recommendation with Causal Debiasing
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于推荐系统任务，旨在解决多模态数据处理、算法偏见和决策透明性问题。通过五项创新改进生成式推荐框架，提升准确性、公平性和多样性。**

- **链接: [http://arxiv.org/pdf/2510.01622v1](http://arxiv.org/pdf/2510.01622v1)**

> **作者:** Bo Ma; Hang Li; ZeHua Hu; XiaoFan Gui; LuYao Liu; Simon Lau
>
> **摘要:** Contemporary generative recommendation systems face significant challenges in handling multimodal data, eliminating algorithmic biases, and providing transparent decision-making processes. This paper introduces an enhanced generative recommendation framework that addresses these limitations through five key innovations: multimodal fusion architecture, retrieval-augmented generation mechanisms, causal inference-based debiasing, explainable recommendation generation, and real-time adaptive learning capabilities. Our framework leverages advanced large language models as the backbone while incorporating specialized modules for cross-modal understanding, contextual knowledge integration, bias mitigation, explanation synthesis, and continuous model adaptation. Extensive experiments on three benchmark datasets (MovieLens-25M, Amazon-Electronics, Yelp-2023) demonstrate consistent improvements in recommendation accuracy, fairness, and diversity compared to existing approaches. The proposed framework achieves up to 2.3% improvement in NDCG@10 and 1.4% enhancement in diversity metrics while maintaining computational efficiency through optimized inference strategies.
>
---
#### [new 121] Study on LLMs for Promptagator-Style Dense Retriever Training
- **分类: cs.IR; cs.CL**

- **简介: 该论文研究使用开源小规模LLM（≤14B参数）替代专用大模型进行Promptagator式密集检索器训练，解决数据生成与模型微调问题。**

- **链接: [http://arxiv.org/pdf/2510.02241v1](http://arxiv.org/pdf/2510.02241v1)**

> **作者:** Daniel Gwon; Nour Jedidi; Jimmy Lin
>
> **备注:** CIKM 2025 short research paper
>
> **摘要:** Promptagator demonstrated that Large Language Models (LLMs) with few-shot prompts can be used as task-specific query generators for fine-tuning domain-specialized dense retrieval models. However, the original Promptagator approach relied on proprietary and large-scale LLMs which users may not have access to or may be prohibited from using with sensitive data. In this work, we study the impact of open-source LLMs at accessible scales ($\leq$14B parameters) as an alternative. Our results demonstrate that open-source LLMs as small as 3B parameters can serve as effective Promptagator-style query generators. We hope our work will inform practitioners with reliable alternatives for synthetic data generation and give insights to maximize fine-tuning results for domain-specific applications.
>
---
#### [new 122] LLM-based Multi-Agent Blackboard System for Information Discovery in Data Science
- **分类: cs.MA; cs.AI; cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于数据科学任务，解决大规模异构数据湖中信息发现的问题。提出基于LLM的多智能体黑板系统，提升多智能体协作效率与灵活性。**

- **链接: [http://arxiv.org/pdf/2510.01285v1](http://arxiv.org/pdf/2510.01285v1)**

> **作者:** Alireza Salemi; Mihir Parmar; Palash Goyal; Yiwen Song; Jinsung Yoon; Hamed Zamani; Hamid Palangi; Tomas Pfister
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has opened new opportunities in data science, yet their practical deployment is often constrained by the challenge of discovering relevant data within large heterogeneous data lakes. Existing methods struggle with this: single-agent systems are quickly overwhelmed by large, heterogeneous files in the large data lakes, while multi-agent systems designed based on a master-slave paradigm depend on a rigid central controller for task allocation that requires precise knowledge of each sub-agent's capabilities. To address these limitations, we propose a novel multi-agent communication paradigm inspired by the blackboard architecture for traditional AI models. In this framework, a central agent posts requests to a shared blackboard, and autonomous subordinate agents -- either responsible for a partition of the data lake or general information retrieval -- volunteer to respond based on their capabilities. This design improves scalability and flexibility by eliminating the need for a central coordinator to have prior knowledge of all sub-agents' expertise. We evaluate our method on three benchmarks that require explicit data discovery: KramaBench and modified versions of DS-Bench and DA-Code to incorporate data discovery. Experimental results demonstrate that the blackboard architecture substantially outperforms baselines, including RAG and the master-slave multi-agent paradigm, achieving between 13% to 57% relative improvement in end-to-end task success and up to a 9% relative gain in F1 score for data discovery over the best-performing baselines across both proprietary and open-source LLMs. Our findings establish the blackboard paradigm as a scalable and generalizable communication framework for multi-agent systems.
>
---
#### [new 123] The Reasoning Boundary Paradox: How Reinforcement Learning Constrains Language Models
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于语言模型优化任务，研究RLVR导致推理边界缩小的问题，分析负干扰和赢家通吃现象，并提出数据筛选方法提升性能。**

- **链接: [http://arxiv.org/pdf/2510.02230v1](http://arxiv.org/pdf/2510.02230v1)**

> **作者:** Phuc Minh Nguyen; Chinh D. La; Duy M. H. Nguyen; Nitesh V. Chawla; Binh T. Nguyen; Khoa D. Doan
>
> **备注:** 23 pages, 15 figures
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a key method for improving Large Language Models' reasoning capabilities, yet recent evidence suggests it may paradoxically shrink the reasoning boundary rather than expand it. This paper investigates the shrinkage issue of RLVR by analyzing its learning dynamics and reveals two critical phenomena that explain this failure. First, we expose negative interference in RLVR, where learning to solve certain training problems actively reduces the likelihood of correct solutions for others, leading to the decline of Pass@$k$ performance, or the probability of generating a correct solution within $k$ attempts. Second, we uncover the winner-take-all phenomenon: RLVR disproportionately reinforces problems with high likelihood, correct solutions, under the base model, while suppressing other initially low-likelihood ones. Through extensive theoretical and empirical analysis on multiple mathematical reasoning benchmarks, we show that this effect arises from the inherent on-policy sampling in standard RL objectives, causing the model to converge toward narrow solution strategies. Based on these insights, we propose a simple yet effective data curation algorithm that focuses RLVR learning on low-likelihood problems, achieving notable improvement in Pass@$k$ performance. Our code is available at https://github.com/mail-research/SELF-llm-interference.
>
---
#### [new 124] Aristotle: IMO-level Automated Theorem Proving
- **分类: cs.AI; cs.CL**

- **简介: 该论文介绍AI系统Aristotle，用于自动定理证明，解决数学奥林匹克问题，整合形式验证与非形式推理，实现高水平性能。**

- **链接: [http://arxiv.org/pdf/2510.01346v1](http://arxiv.org/pdf/2510.01346v1)**

> **作者:** Tudor Achim; Alex Best; Kevin Der; Mathïs Fédérico; Sergei Gukov; Daniel Halpern-Leister; Kirsten Henningsgard; Yury Kudryashov; Alexander Meiburg; Martin Michelsen; Riley Patterson; Eric Rodriguez; Laura Scharff; Vikram Shanker; Vladmir Sicca; Hari Sowrirajan; Aidan Swope; Matyas Tamas; Vlad Tenev; Jonathan Thomm; Harold Williams; Lawrence Wu
>
> **摘要:** We introduce Aristotle, an AI system that combines formal verification with informal reasoning, achieving gold-medal-equivalent performance on the 2025 International Mathematical Olympiad problems. Aristotle integrates three main components: a Lean proof search system, an informal reasoning system that generates and formalizes lemmas, and a dedicated geometry solver. Our system demonstrates state-of-the-art performance with favorable scaling properties for automated theorem proving.
>
---
#### [new 125] Think Right: Learning to Mitigate Under-Over Thinking via Adaptive, Attentive Compression
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于推理任务，解决模型在复杂问题中过度或不足思考的问题。通过自适应注意力压缩方法TRAAC，优化推理步骤，提升准确率并减少冗余计算。**

- **链接: [http://arxiv.org/pdf/2510.01581v1](http://arxiv.org/pdf/2510.01581v1)**

> **作者:** Joykirat Singh; Justin Chih-Yao Chen; Archiki Prasad; Elias Stengel-Eskin; Akshay Nambi; Mohit Bansal
>
> **备注:** Code: https://github.com/joykirat18/TRAAC
>
> **摘要:** Recent thinking models solve complex reasoning tasks by scaling test-time compute, but this scaling must be allocated in line with task difficulty. On one hand, short reasoning (underthinking) leads to errors on harder problems that require extended reasoning steps; but, excessively long reasoning (overthinking) can be token-inefficient, generating unnecessary steps even after reaching a correct intermediate solution. We refer to this as under-adaptivity, where the model fails to modulate its response length appropriately given problems of varying difficulty. To address under-adaptivity and strike a balance between under- and overthinking, we propose TRAAC (Think Right with Adaptive, Attentive Compression), an online post-training RL method that leverages the model's self-attention over a long reasoning trajectory to identify important steps and prune redundant ones. TRAAC also estimates difficulty and incorporates it into training rewards, thereby learning to allocate reasoning budget commensurate with example difficulty. Our approach improves accuracy, reduces reasoning steps, and enables adaptive thinking compared to base models and other RL baselines. Across a variety of tasks (AIME, AMC, GPQA-D, BBEH), TRAAC (Qwen3-4B) achieves an average absolute accuracy gain of 8.4% with a relative reduction in reasoning length of 36.8% compared to the base model, and a 7.9% accuracy gain paired with a 29.4% length drop compared to the best RL baseline. TRAAC also shows strong generalization: although our models are trained on math datasets, they show accuracy and efficiency gains on out-of-distribution non-math datasets like GPQA-D, BBEH, and OptimalThinkingBench. Our analysis further verifies that TRAAC provides fine-grained adjustments to thinking budget based on difficulty and that a combination of task-difficulty calibration and attention-based compression yields gains across diverse tasks.
>
---
#### [new 126] Information Seeking for Robust Decision Making under Partial Observability
- **分类: cs.AI; cs.CL; cs.RO**

- **简介: 该论文属于决策规划任务，解决部分可观测环境下的不确定性问题。提出InfoSeeker框架，结合计划与信息获取，提升决策鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.01531v1](http://arxiv.org/pdf/2510.01531v1)**

> **作者:** Djengo Cyun-Jyun Fang; Tsung-Wei Ke
>
> **备注:** The project page is available at https://infoseekerllm.github.io
>
> **摘要:** Explicit information seeking is essential to human problem-solving in practical environments characterized by incomplete information and noisy dynamics. When the true environmental state is not directly observable, humans seek information to update their internal dynamics and inform future decision-making. Although existing Large Language Model (LLM) planning agents have addressed observational uncertainty, they often overlook discrepancies between their internal dynamics and the actual environment. We introduce Information Seeking Decision Planner (InfoSeeker), an LLM decision-making framework that integrates task-oriented planning with information seeking to align internal dynamics and make optimal decisions under uncertainty in both agent observations and environmental dynamics. InfoSeeker prompts an LLM to actively gather information by planning actions to validate its understanding, detect environmental changes, or test hypotheses before generating or revising task-oriented plans. To evaluate InfoSeeker, we introduce a novel benchmark suite featuring partially observable environments with incomplete observations and uncertain dynamics. Experiments demonstrate that InfoSeeker achieves a 74% absolute performance gain over prior methods without sacrificing sample efficiency. Moreover, InfoSeeker generalizes across LLMs and outperforms baselines on established benchmarks such as robotic manipulation and web navigation. These findings underscore the importance of tightly integrating planning and information seeking for robust behavior in partially observable environments. The project page is available at https://infoseekerllm.github.io
>
---
#### [new 127] Control the Temperature: Selective Sampling for Diverse and High-Quality LLM Outputs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型生成任务，解决高温度采样导致的精度下降问题。通过引入选择性采样方法，动态调整采样策略以提升输出质量与多样性。**

- **链接: [http://arxiv.org/pdf/2510.01218v1](http://arxiv.org/pdf/2510.01218v1)**

> **作者:** Sergey Troshin; Wafaa Mohammed; Yan Meng; Christof Monz; Antske Fokkens; Vlad Niculae
>
> **备注:** Second Conference on Language Modeling, 2025
>
> **摘要:** Diversity is an essential metric for evaluating the creativity of outputs generated by language models. Temperature-based sampling is a common strategy to increase diversity. However, for tasks that require high precision, e.g., mathematical reasoning, uncontrolled high temperature sampling, e.g., min-$p$ or top-$p$, degrades reasoning quality. We demonstrate that the loss of accuracy is caused by sampling incorrect continuations in sensitive decoding positions. To address this, in this paper, we propose \textbf{selective sampling}, a method that dynamically switches between greedy and high-temperature sampling based on a sampling risk metric. This risk metric estimates the likelihood of output errors when applying high-temperature sampling on the current token position. To predict sampling risk, we train a lightweight classifier on a small subset of verifiable problems. The trained classifier can be integrated with the base language model with minimal latency overhead. Experiments on mathematical reasoning tasks demonstrate that selective sampling enhances the quality-diversity trade-off, even in high-temperature settings.
>
---
#### [new 128] RSAVQ: Riemannian Sensitivity-Aware Vector Quantization for Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型压缩任务，解决大语言模型在资源受限设备上的部署问题。提出RSAVQ框架，通过几何方法优化低比特量化，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2510.01240v1](http://arxiv.org/pdf/2510.01240v1)**

> **作者:** Zukang Xu; Xing Hu; Qiang Wu; Dawei Yang
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable performance across a wide range of natural language processing tasks. However, their exponentially increasing parameters pose significant challenges for deployment on resource-constrained devices. Vector Quantization (VQ) shows great promise for low-bit quantization (e.g., 2 to 4 bits), but existing work faces two key challenges: unconstrained direction error and suboptimal bit allocation. In this paper, we propose RSAVQ, a novel VQ framework to enhance extremely low-bit quantization for LLMs. RSAVQ introduces two geometry-driven innovations that effectively mitigate above limitations: (1) Error Direction Sensitivity Guidance (EDSG), which leverages the Fisher Information Matrix (FIM)-induced Riemannian metric to project quantization errors onto low-sensitivity directions in the parameter space. Specifically, this projection is performed along the negative natural gradient direction, which effectively suppresses error expansion. (2) Weight Channel Sensitivity Guidance (WCSG) , which constructs a channel-wise sensitivity metric via FIM curvature analysis to dynamically guide bit resource allocation. The approach facilitates a globally optimal quantization solution within prescribed bit constraints. Experiments demonstrate that RSAVQ outperforms existing methods for LLMs. For example, in 2-bit quantization of LLaMA-3 8B, RSAVQ leads baselines like VPTQ and QuIP# by 0.4 in perplexity (PPL) and 1.5 in zero-shot accuracy. This work offers a practical solution for constrained environments and a theoretical bridge between information geometry and the quantization of neural networks, advancing efficient deep learning.
>
---
#### [new 129] Extracting O*NET Features from the NLx Corpus to Build Public Use Aggregate Labor Market Data
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决职业数据标准化问题。通过提取在线职位广告中的O*NET特征，构建结构化劳动市场数据集。**

- **链接: [http://arxiv.org/pdf/2510.01470v1](http://arxiv.org/pdf/2510.01470v1)**

> **作者:** Stephen Meisenbacher; Svetlozar Nestorov; Peter Norlander
>
> **备注:** 85 pages
>
> **摘要:** Data from online job postings are difficult to access and are not built in a standard or transparent manner. Data included in the standard taxonomy and occupational information database (O*NET) are updated infrequently and based on small survey samples. We adopt O*NET as a framework for building natural language processing tools that extract structured information from job postings. We publish the Job Ad Analysis Toolkit (JAAT), a collection of open-source tools built for this purpose, and demonstrate its reliability and accuracy in out-of-sample and LLM-as-a-Judge testing. We extract more than 10 billion data points from more than 155 million online job ads provided by the National Labor Exchange (NLx) Research Hub, including O*NET tasks, occupation codes, tools, and technologies, as well as wages, skills, industry, and more features. We describe the construction of a dataset of occupation, state, and industry level features aggregated by monthly active jobs from 2015 - 2025. We illustrate the potential for research and future uses in education and workforce development.
>
---
#### [new 130] Automated Extraction of Material Properties using LLM-based AI Agents
- **分类: cs.LG; cond-mat.mtrl-sci; cs.AI; cs.CL**

- **简介: 该论文属于材料属性自动提取任务，解决实验文献数据利用不足的问题。通过LLM驱动的工作流，从大量文献中高效提取热电和结构属性，构建大规模数据集。**

- **链接: [http://arxiv.org/pdf/2510.01235v1](http://arxiv.org/pdf/2510.01235v1)**

> **作者:** Subham Ghosh; Abhishek Tewari
>
> **摘要:** The rapid discovery of materials is constrained by the lack of large, machine-readable datasets that couple performance metrics with structural context. Existing databases are either small, manually curated, or biased toward first principles results, leaving experimental literature underexploited. We present an agentic, large language model (LLM)-driven workflow that autonomously extracts thermoelectric and structural-properties from about 10,000 full-text scientific articles. The pipeline integrates dynamic token allocation, zeroshot multi-agent extraction, and conditional table parsing to balance accuracy against computational cost. Benchmarking on 50 curated papers shows that GPT-4.1 achieves the highest accuracy (F1 = 0.91 for thermoelectric properties and 0.82 for structural fields), while GPT-4.1 Mini delivers nearly comparable performance (F1 = 0.89 and 0.81) at a fraction of the cost, enabling practical large scale deployment. Applying this workflow, we curated 27,822 temperature resolved property records with normalized units, spanning figure of merit (ZT), Seebeck coefficient, conductivity, resistivity, power factor, and thermal conductivity, together with structural attributes such as crystal class, space group, and doping strategy. Dataset analysis reproduces known thermoelectric trends, such as the superior performance of alloys over oxides and the advantage of p-type doping, while also surfacing broader structure-property correlations. To facilitate community access, we release an interactive web explorer with semantic filters, numeric queries, and CSV export. This study delivers the largest LLM-curated thermoelectric dataset to date, provides a reproducible and cost-profiled extraction pipeline, and establishes a foundation for scalable, data-driven materials discovery beyond thermoelectrics.
>
---
#### [new 131] PychoBench: Evaluating the Psychology Intelligence of Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型在心理学领域的智能水平。通过构建PsychoBench基准测试，验证LLM是否具备心理辅导能力。**

- **链接: [http://arxiv.org/pdf/2510.01611v1](http://arxiv.org/pdf/2510.01611v1)**

> **作者:** Min Zeng
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable success across a wide range of industries, primarily due to their impressive generative abilities. Yet, their potential in applications requiring cognitive abilities, such as psychological counseling, remains largely untapped. This paper investigates the key question: Can LLMs be effectively applied to psychological counseling? To determine whether an LLM can effectively take on the role of a psychological counselor, the first step is to assess whether it meets the qualifications required for such a role, namely the ability to pass the U.S. National Counselor Certification Exam (NCE). This is because, just as a human counselor must pass a certification exam to practice, an LLM must demonstrate sufficient psychological knowledge to meet the standards required for such a role. To address this, we introduce PsychoBench, a benchmark grounded in U.S.national counselor examinations, a licensure test for professional counselors that requires about 70% accuracy to pass. PsychoBench comprises approximately 2,252 carefully curated single-choice questions, crafted to require deep understanding and broad enough to cover various sub-disciplines of psychology. This benchmark provides a comprehensive assessment of an LLM's ability to function as a counselor. Our evaluation shows that advanced models such as GPT-4o, Llama3.3-70B, and Gemma3-27B achieve well above the passing threshold, while smaller open-source models (e.g., Qwen2.5-7B, Mistral-7B) remain far below it. These results suggest that only frontier LLMs are currently capable of meeting counseling exam standards, highlighting both the promise and the challenges of developing psychology-oriented LLMs.
>
---
#### [new 132] RLAD: Training LLMs to Discover Abstractions for Solving Reasoning Problems
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于机器学习任务，旨在解决模型推理能力不足的问题。通过引入抽象概念，提升模型结构化探索和解决问题的能力。**

- **链接: [http://arxiv.org/pdf/2510.02263v1](http://arxiv.org/pdf/2510.02263v1)**

> **作者:** Yuxiao Qu; Anikait Singh; Yoonho Lee; Amrith Setlur; Ruslan Salakhutdinov; Chelsea Finn; Aviral Kumar
>
> **摘要:** Reasoning requires going beyond pattern matching or memorization of solutions to identify and implement "algorithmic procedures" that can be used to deduce answers to hard problems. Doing so requires realizing the most relevant primitives, intermediate results, or shared procedures, and building upon them. While RL post-training on long chains of thought ultimately aims to uncover this kind of algorithmic behavior, most reasoning traces learned by large models fail to consistently capture or reuse procedures, instead drifting into verbose and degenerate exploration. To address more effective reasoning, we introduce reasoning abstractions: concise natural language descriptions of procedural and factual knowledge that guide the model toward learning successful reasoning. We train models to be capable of proposing multiple abstractions given a problem, followed by RL that incentivizes building a solution while using the information provided by these abstractions. This results in a two-player RL training paradigm, abbreviated as RLAD, that jointly trains an abstraction generator and a solution generator. This setup effectively enables structured exploration, decouples learning signals of abstraction proposal and solution generation, and improves generalization to harder problems. We also show that allocating more test-time compute to generating abstractions is more beneficial for performance than generating more solutions at large test budgets, illustrating the role of abstractions in guiding meaningful exploration.
>
---
#### [new 133] RLP: Reinforcement as a Pretraining Objective
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决模型推理能力不足的问题。通过引入RLP预训练目标，将强化学习融入预训练阶段，提升模型独立思考能力。**

- **链接: [http://arxiv.org/pdf/2510.01265v1](http://arxiv.org/pdf/2510.01265v1)**

> **作者:** Ali Hatamizadeh; Syeda Nahida Akter; Shrimai Prabhumoye; Jan Kautz; Mostofa Patwary; Mohammad Shoeybi; Bryan Catanzaro; Yejin Choi
>
> **备注:** RLP introduces a new paradigm for RL-based Pretraining
>
> **摘要:** The dominant paradigm for training large reasoning models starts with pre-training using next-token prediction loss on vast amounts of data. Reinforcement learning, while powerful in scaling reasoning, is introduced only as the very last phase of post-training, preceded by supervised fine-tuning. While dominant, is this an optimal way of training? In this paper, we present RLP, an information-driven reinforcement pretraining objective, that brings the core spirit of reinforcement learning -- exploration -- to the last phase of pretraining. The key idea is to treat chain-of-thought as an exploratory action, with rewards computed based on the information gain it provides for predicting future tokens. This training objective essentially encourages the model to think for itself before predicting what comes next, thus teaching an independent thinking behavior earlier in the pretraining. More concretely, the reward signal measures the increase in log-likelihood of the next token when conditioning on both context and a sampled reasoning chain, compared to conditioning on context alone. This approach yields a verifier-free dense reward signal, allowing for efficient training for the full document stream during pretraining. Specifically, RLP reframes reinforcement learning for reasoning as a pretraining objective on ordinary text, bridging the gap between next-token prediction and the emergence of useful chain-of-thought reasoning. Pretraining with RLP on Qwen3-1.7B-Base lifts the overall average across an eight-benchmark math-and-science suite by 19%. With identical post-training, the gains compound, with the largest improvements on reasoning-heavy tasks such as AIME25 and MMLU-Pro. Applying RLP to the hybrid Nemotron-Nano-12B-v2 increases the overall average from 42.81% to 61.32% and raises the average on scientific reasoning by 23%, demonstrating scalability across architectures and model sizes.
>
---
## 更新

#### [replaced 001] MOSAIC: A Multilingual, Taxonomy-Agnostic, and Computationally Efficient Approach for Radiological Report Classification
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.04471v2](http://arxiv.org/pdf/2509.04471v2)**

> **作者:** Alice Schiavone; Marco Fraccaro; Lea Marie Pehrson; Silvia Ingala; Rasmus Bonnevie; Michael Bachmann Nielsen; Vincent Beliveau; Melanie Ganz; Desmond Elliott
>
> **备注:** 8 pages, 14 pages including references and appendix. 9 figures. Preprint
>
> **摘要:** Radiology reports contain rich clinical information that can be used to train imaging models without relying on costly manual annotation. However, existing approaches face critical limitations: rule-based methods struggle with linguistic variability, supervised models require large annotated datasets, and recent LLM-based systems depend on closed-source or resource-intensive models that are unsuitable for clinical use. Moreover, current solutions are largely restricted to English and single-modality, single-taxonomy datasets. We introduce MOSAIC, a multilingual, taxonomy-agnostic, and computationally efficient approach for radiological report classification. Built on a compact open-access language model (MedGemma-4B), MOSAIC supports both zero-/few-shot prompting and lightweight fine-tuning, enabling deployment on consumer-grade GPUs. We evaluate MOSAIC across seven datasets in English, Spanish, French, and Danish, spanning multiple imaging modalities and label taxonomies. The model achieves a mean macro F1 score of 88 across five chest X-ray datasets, approaching or exceeding expert-level performance, while requiring only 24 GB of GPU memory. With data augmentation, as few as 80 annotated samples are sufficient to reach a weighted F1 score of 82 on Danish reports, compared to 86 with the full 1600-sample training set. MOSAIC offers a practical alternative to large or proprietary LLMs in clinical settings. Code and models are open-source. We invite the community to evaluate and extend MOSAIC on new languages, taxonomies, and modalities.
>
---
#### [replaced 002] Enhancing Personalized Multi-Turn Dialogue with Curiosity Reward
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.03206v3](http://arxiv.org/pdf/2504.03206v3)**

> **作者:** Yanming Wan; Jiaxing Wu; Marwa Abdulhai; Lior Shani; Natasha Jaques
>
> **摘要:** Effective conversational agents like large language models (LLMs) must personalize their interactions to adapt to user preferences, personalities, and attributes across diverse domains like education and healthcare. Current methods like Reinforcement Learning from Human Feedback (RLHF), often prioritize helpfulness and safety but fall short in fostering truly empathetic, adaptive, and personalized dialogues. Existing personalization approaches typically rely on extensive user history, limiting their effectiveness for new or context-limited users. To address these limitations, we propose leveraging a user model to incorporate a curiosity-based intrinsic reward into multi-turn RLHF. This novel reward mechanism encourages the LLM agent to actively infer user traits by optimizing conversations to improve its user model's accuracy. Consequently, the agent delivers more personalized interactions by learning more about the user. We demonstrate our method's effectiveness in two distinct domains: significantly improving personalization performance in a conversational recommendation task, and personalizing conversations for different learning styles in an educational setting. We show improved generalization capabilities compared to traditional multi-turn RLHF, all while maintaining conversation quality. Our method offers a promising solution for creating more personalized, adaptive, and engaging conversational agents.
>
---
#### [replaced 003] Sparkle: Mastering Basic Spatial Capabilities in Vision Language Models Elicits Generalization to Spatial Reasoning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.16162v4](http://arxiv.org/pdf/2410.16162v4)**

> **作者:** Yihong Tang; Ao Qu; Zhaokai Wang; Dingyi Zhuang; Zhaofeng Wu; Wei Ma; Shenhao Wang; Yunhan Zheng; Zhan Zhao; Jinhua Zhao
>
> **摘要:** Vision language models (VLMs) perform well on many tasks but often fail at spatial reasoning, which is essential for navigation and interaction with physical environments. Many spatial reasoning tasks depend on fundamental two-dimensional (2D) skills, yet our evaluation shows that state-of-the-art VLMs give implausible or incorrect answers to composite spatial problems, including simple pathfinding tasks that humans solve effortlessly. To address this, we enhance 2D spatial reasoning in VLMs by training them only on basic spatial capabilities. We first disentangle 2D spatial reasoning into three core components: direction comprehension, distance estimation, and localization. We hypothesize that mastering these skills substantially improves performance on complex spatial tasks that require advanced reasoning and combinatorial problem solving, while also generalizing to real-world scenarios. To test this, we introduce Sparkle, a framework that generates synthetic data to provide targeted supervision across these three capabilities and yields an instruction dataset for each. Experiments show that VLMs fine-tuned with \emph{Sparkle} improve not only on basic tasks but also on composite and out-of-distribution real-world spatial reasoning tasks. These results indicate that enhancing basic spatial skills through synthetic generalization effectively advances complex spatial reasoning and offers a systematic strategy for boosting the spatial understanding of VLMs. Source codes of Sparkle are available at https://github.com/YihongT/Sparkle.
>
---
#### [replaced 004] MOSS-Speech: Towards True Speech-to-Speech Models Without Text Guidance
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.00499v2](http://arxiv.org/pdf/2510.00499v2)**

> **作者:** Xingjian Zhao; Zhe Xu; Qinyuan Cheng; Zhaoye Fei; Luozhijie Jin; Yang Wang; Hanfu Chen; Yaozhou Jiang; Qinghui Gao; Ke Chen; Ruixiao Li; Mingshu Chen; Ruiming Wang; Wenbo Zhang; Yiyang Zhang; Donghua Yu; Yang Gao; Xiaogui Yang; Yitian Gong; Yuanfan Xu; Yaqian Zhou; Xuanjing Huang; Xipeng Qiu
>
> **摘要:** Spoken dialogue systems often rely on cascaded pipelines that transcribe, process, and resynthesize speech. While effective, this design discards paralinguistic cues and limits expressivity. Recent end-to-end methods reduce latency and better preserve these cues, yet still rely on text intermediates, creating a fundamental bottleneck. We present MOSS-Speech, a true speech-to-speech large language model that directly understands and generates speech without relying on text guidance. Our approach combines a modality-based layer-splitting architecture with a frozen pre-training strategy, preserving the reasoning and knowledge of pretrained text LLMs while adding native speech capabilities. Experiments show that our model achieves state-of-the-art results in spoken question answering and delivers comparable speech-to-speech performance relative to existing text-guided systems, while still maintaining competitive text performance. By narrowing the gap between text-guided and direct speech generation, our work establishes a new paradigm for expressive and efficient end-to-end speech interaction.
>
---
#### [replaced 005] Agent-ScanKit: Unraveling Memory and Reasoning of Multimodal Agents via Sensitivity Perturbations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.00496v2](http://arxiv.org/pdf/2510.00496v2)**

> **作者:** Pengzhou Cheng; Lingzhong Dong; Zeng Wu; Zongru Wu; Zhuosheng Zhang; Gongshen Liu
>
> **备注:** 23 pages, 10 figures, 7 tables
>
> **摘要:** Although numerous strategies have recently been proposed to enhance the autonomous interaction capabilities of multimodal agents in graphical user interface (GUI), their reliability remains limited when faced with complex or out-of-domain tasks. This raises a fundamental question: Are existing multimodal agents reasoning spuriously? In this paper, we propose \textbf{Agent-ScanKit}, a systematic probing framework to unravel the memory and reasoning capabilities of multimodal agents under controlled perturbations. Specifically, we introduce three orthogonal probing paradigms: visual-guided, text-guided, and structure-guided, each designed to quantify the contributions of memorization and reasoning without requiring access to model internals. In five publicly available GUI benchmarks involving 18 multimodal agents, the results demonstrate that mechanical memorization often outweighs systematic reasoning. Most of the models function predominantly as retrievers of training-aligned knowledge, exhibiting limited generalization. Our findings underscore the necessity of robust reasoning modeling for multimodal agents in real-world scenarios, offering valuable insights toward the development of reliable multimodal agents.
>
---
#### [replaced 006] TLUE: A Tibetan Language Understanding Evaluation Benchmark
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.12051v5](http://arxiv.org/pdf/2503.12051v5)**

> **作者:** Fan Gao; Cheng Huang; Nyima Tashi; Xiangxiang Wang; Thupten Tsering; Ban Ma-bao; Renzeg Duojie; Gadeng Luosang; Rinchen Dongrub; Dorje Tashi; Hao Wang Xiao Feng; Yongbin Yu
>
> **备注:** Accepted by EMNLP Main Conference (Poster)
>
> **摘要:** Large language models have made tremendous progress in recent years, but low-resource languages, like Tibetan, remain significantly underrepresented in their evaluation. Despite Tibetan being spoken by over seven million people, it has largely been neglected in the development and assessment of large language models. To address this gap, we present a \textbf{T}ibetan \textbf{L}anguage \textbf{U}nderstanding \textbf{E}valuation Benchmark, \textbf{TLUE}, the first large-scale benchmark for measuring the proficiency of LLMs in the Tibetan language. \textbf{TLUE} comprises two major components: a comprehensive multi-task understanding benchmark spanning 5 domains and 67 subdomains, and a safety benchmark encompassing 7 subdomains. Then, we evaluate a diverse set of state-of-the-art large language models. Experimental results demonstrate that most large language models perform below the random baseline, highlighting the considerable challenges they face in Tibetan language processing. \textbf{TLUE} provides a crucial foundation for advancing future research in Tibetan language understanding and highlights the importance of promoting greater inclusivity in the development of large language models.
>
---
#### [replaced 007] Beyond Chunking: Discourse-Aware Hierarchical Retrieval for Long Document Question Answering
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.06313v2](http://arxiv.org/pdf/2506.06313v2)**

> **作者:** Huiyao Chen; Yi Yang; Yinghui Li; Meishan Zhang; Min Zhang
>
> **备注:** 20 pages, 9 figures
>
> **摘要:** Long document question answering systems typically process texts as flat sequences or use arbitrary segmentation, failing to capture discourse structures that guide human comprehension. We present a discourse-aware hierarchical framework that leverages rhetorical structure theory (RST) to enhance long document question answering. Our approach converts discourse trees into sentence-level representations and employs LLM-enhanced node representations to bridge structural and semantic information. The framework involves three key innovations: specialized discourse parsing for lengthy documents, LLM-based enhancement of discourse relation nodes, and structure-guided hierarchical retrieval. Comprehensive experiments on QASPER, QuALITY, and NarrativeQA demonstrate consistent improvements over existing approaches. Ablation studies confirm that incorporating discourse structure significantly enhances question answering across diverse document types.
>
---
#### [replaced 008] Out-of-Distribution Detection using Synthetic Data Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.03323v2](http://arxiv.org/pdf/2502.03323v2)**

> **作者:** Momin Abbas; Muneeza Azmat; Raya Horesh; Mikhail Yurochkin
>
> **备注:** Accepted to COLM 2025. Camera-ready version
>
> **摘要:** Distinguishing in- and out-of-distribution (OOD) inputs is crucial for reliable deployment of classification systems. However, OOD data is typically unavailable or difficult to collect, posing a significant challenge for accurate OOD detection. In this work, we present a method that harnesses the generative capabilities of Large Language Models (LLMs) to create high-quality synthetic OOD proxies, eliminating the dependency on any external OOD data source. We study the efficacy of our method on classical text classification tasks such as toxicity detection and sentiment classification as well as classification tasks arising in LLM development and deployment, such as training a reward model for RLHF and detecting misaligned generations. Extensive experiments on nine InD-OOD dataset pairs and various model sizes show that our approach dramatically lowers false positive rates (achieving a perfect zero in some cases) while maintaining high accuracy on in-distribution tasks, outperforming baseline methods by a significant margin.
>
---
#### [replaced 009] Boundless Byte Pair Encoding: Breaking the Pre-tokenization Barrier
- **分类: cs.CL; cs.AI; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2504.00178v2](http://arxiv.org/pdf/2504.00178v2)**

> **作者:** Craig W. Schmidt; Varshini Reddy; Chris Tanner; Yuval Pinter
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** Pre-tokenization, the initial step in many modern tokenization pipelines, segments text into smaller units called pretokens, typically splitting on whitespace and punctuation. While this process encourages having full, individual words as tokens, it introduces a fundamental limitation in most tokenization algorithms such as Byte Pair Encoding (BPE). Specifically, pre-tokenization causes the distribution of tokens in a corpus to heavily skew towards common, full-length words. This skewed distribution limits the benefits of expanding to larger vocabularies, since the additional tokens appear with progressively lower counts. To overcome this barrier, we propose BoundlessBPE, a modified BPE algorithm that relaxes the pretoken boundary constraint. Our approach selectively merges two complete pretokens into a larger unit we term a superword. Superwords are not necessarily semantically cohesive. For example, the pretokens " of" and " the" might be combined to form the superword " of the". This merging strategy results in a substantially more uniform distribution of tokens across a corpus than standard BPE, and compresses text more effectively, with up to a 15% increase in bytes per token.
>
---
#### [replaced 010] Reason to Rote: Rethinking Memorization in Reasoning
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.04782v2](http://arxiv.org/pdf/2507.04782v2)**

> **作者:** Yupei Du; Philipp Mondorf; Silvia Casola; Yuekun Yao; Robert Litschko; Barbara Plank
>
> **备注:** EMNLP 2025 Main. 21 pages, 14 figures
>
> **摘要:** Large language models readily memorize arbitrary training instances, such as label noise, yet they perform strikingly well on reasoning tasks. In this work, we investigate how language models memorize label noise, and why such memorization in many cases does not heavily affect generalizable reasoning capabilities. Using two controllable synthetic reasoning datasets with noisy labels, four-digit addition (FDA) and two-hop relational reasoning (THR), we discover a reliance of memorization on generalizable reasoning mechanisms: models continue to compute intermediate reasoning outputs even when retrieving memorized noisy labels, and intervening reasoning adversely affects memorization. We further show that memorization operates through distributed encoding, i.e., aggregating various inputs and intermediate results, rather than building a look-up mechanism from inputs to noisy labels. Moreover, our FDA case study reveals memorization occurs via outlier heuristics, where existing neuron activation patterns are slightly shifted to fit noisy labels. Together, our findings suggest that memorization of label noise in language models builds on, rather than overrides, the underlying reasoning mechanisms, shedding lights on the intriguing phenomenon of benign memorization.
>
---
#### [replaced 011] Generating Difficult-to-Translate Texts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.26592v2](http://arxiv.org/pdf/2509.26592v2)**

> **作者:** Vilém Zouhar; Wenda Xu; Parker Riley; Juraj Juraska; Mara Finkelstein; Markus Freitag; Daniel Deutsch
>
> **摘要:** Machine translation benchmarks sourced from the real world are quickly obsoleted, due to most examples being easy for state-of-the-art translation models. This limits the benchmark's ability to distinguish which model is better or to reveal models' weaknesses. Current methods for creating difficult test cases, such as subsampling or from-scratch synthesis, either fall short of identifying difficult examples or suffer from a lack of diversity and naturalness. Inspired by the iterative process of human experts probing for model failures, we propose MT-breaker, a method where a large language model iteratively refines a source text to increase its translation difficulty. The LLM iteratively queries a target machine translation model to guide its generation of difficult examples. Our approach generates examples that are more challenging for the target MT model while preserving the diversity of natural texts. While the examples are tailored to a particular machine translation model during the generation, the difficulty also transfers to other models and languages.
>
---
#### [replaced 012] Double-Checker: Enhancing Reasoning of Slow-Thinking LLMs via Self-Critical Fine-Tuning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.21285v3](http://arxiv.org/pdf/2506.21285v3)**

> **作者:** Xin Xu; Tianhao Chen; Fan Zhang; Wanlong Liu; Pengxiang Li; Ajay Kumar Jaiswal; Yuchen Yan; Jishan Hu; Yang Wang; Hao Chen; Shiwei Liu; Shizhe Diao; Can Yang; Lu Yin
>
> **备注:** 10 pages
>
> **摘要:** While slow-thinking large language models (LLMs) exhibit reflection-like reasoning, commonly referred to as the "aha moment:, their ability to generate informative critiques and refine prior solutions remains limited. In this paper, we introduce Double-Checker, a principled framework designed to enhance the reasoning capabilities of slow-thinking LLMs by fostering explicit self-critique and iterative refinement of their previous solutions. By fine-tuning on our curated 1,730 self-critical instances, Double-Checker empowers long-CoT LLMs to iteratively critique and refine their outputs during inference until they evaluate their solutions as correct under self-generated critiques. We validate the efficacy of Double-Checker across a comprehensive suite of reasoning benchmarks, demonstrating that iterative self-critique significantly enhances the reasoning capabilities of long-CoT LLMs. Notably, our Double-Checker increases the pass@1 performance on challenging AIME benchmarks from 4.4% to 18.2% compared to the original long-CoT LLMs. These results highlight a promising direction for developing more trustworthy and effective LLMs capable of structured self-critique. Our codes and data are available at https://github.com/XinXU-USTC/DoubleChecker
>
---
#### [replaced 013] Injecting External Knowledge into the Reasoning Process Enhances Retrieval-Augmented Generation
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.19333v2](http://arxiv.org/pdf/2507.19333v2)**

> **作者:** Minghao Tang; Shiyu Ni; Jiafeng Guo; Keping Bi
>
> **备注:** SIGIR-AP 2025
>
> **摘要:** Retrieval-augmented generation (RAG) has been widely adopted to augment large language models (LLMs) with external knowledge for knowledge-intensive tasks. However, its effectiveness is often undermined by the presence of noisy (i.e., low-quality) retrieved passages. Enhancing LLMs' robustness to such noise is critical for improving the reliability of RAG systems. Recent advances have equipped LLMs with strong reasoning and self-reflection capabilities, allowing them to identify and correct errors in their reasoning process. Inspired by this ability, we propose Passage Injection-a simple yet effective method that explicitly incorporates retrieved passages into LLMs' reasoning process, aiming to enhance the model's ability to recognize and resist noisy passages. We validate Passage Injection under general RAG settings using BM25 as the retriever. Experiments on four reasoning-enhanced LLMs across four factual QA datasets demonstrate that Passage Injection significantly improves overall RAG performance. Further analysis on two noisy retrieval settings-random noise, where the model is provided irrelevant passages, and counterfactual noise, where it is given misleading passages-shows that Passage Injection consistently improves robustness. Controlled experiments confirm that Passage Injection can also effectively leverage helpful passages. These findings suggest that incorporating passages in LLMs' reasoning process is a promising direction for building more robust RAG systems. The code can be found \href{here}{https://github.com/Trustworthy-Information-Access/Passage-Injection}.
>
---
#### [replaced 014] MathArena: Evaluating LLMs on Uncontaminated Math Competitions
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23281v2](http://arxiv.org/pdf/2505.23281v2)**

> **作者:** Mislav Balunović; Jasper Dekoninck; Ivo Petrov; Nikola Jovanović; Martin Vechev
>
> **摘要:** The rapid advancement of reasoning capabilities in large language models (LLMs) has led to notable improvements on mathematical benchmarks. However, many of the most commonly used evaluation datasets (e.g., AIME 2024) are widely available online, making it difficult to disentangle genuine reasoning from potential memorization. Furthermore, these benchmarks do not evaluate proof-writing capabilities, which are crucial for many mathematical tasks. To address this, we introduce MathArena, a new benchmark based on the following key insight: recurring math competitions provide a stream of high-quality, challenging problems that can be used for real-time evaluation of LLMs. By evaluating models as soon as new problems are released, we effectively eliminate the risk of contamination. Using this framework, we find strong signs of contamination in AIME 2024. Nonetheless, evaluations on harder competitions, such as CMIMC 2025, demonstrate impressive reasoning capabilities in top-performing models. MathArena is also the first benchmark for proof-writing capabilities. On IMO 2025, top models achieve slightly less than 40%, demonstrating both notable progress and significant room for improvement. So far, we have evaluated over $50$ models across seven competitions, totaling $162$ problems. As an evolving benchmark, MathArena will continue to track the progress of LLMs on newly released competitions, ensuring rigorous and up-to-date evaluation of mathematical reasoning.
>
---
#### [replaced 015] MetaFaith: Faithful Natural Language Uncertainty Expression in LLMs
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.24858v2](http://arxiv.org/pdf/2505.24858v2)**

> **作者:** Gabrielle Kaili-May Liu; Gal Yona; Avi Caciularu; Idan Szpektor; Tim G. J. Rudner; Arman Cohan
>
> **备注:** EMNLP 2025
>
> **摘要:** A critical component in the trustworthiness of LLMs is reliable uncertainty communication, yet LLMs often use assertive language when conveying false claims, leading to over-reliance and eroded trust. We present the first systematic study of $\textit{faithful confidence calibration}$ of LLMs, benchmarking models' ability to use linguistic expressions of uncertainty that $\textit{faithfully reflect}$ their intrinsic uncertainty, across a comprehensive array of models, datasets, and prompting strategies. Our results demonstrate that LLMs largely fail at this task, and that existing interventions are insufficient: standard prompt approaches provide only marginal gains, and existing, factuality-based calibration techniques can even harm faithful calibration. To address this critical gap, we introduce MetaFaith, a novel prompt-based calibration approach inspired by human metacognition. We show that MetaFaith robustly improves faithful calibration across diverse models and task domains, enabling up to 61% improvement in faithfulness and achieving an 83% win rate over original generations as judged by humans.
>
---
#### [replaced 016] Design and Application of Multimodal Large Language Model Based System for End to End Automation of Accident Dataset Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.00015v2](http://arxiv.org/pdf/2505.00015v2)**

> **作者:** MD Thamed Bin Zaman Chowdhury; Moazzem Hossain
>
> **备注:** This paper is accepted for presentation in TRB annual meeting 2026. The version presented here is the preprint version before peer review process
>
> **摘要:** Road traffic accidents remain a major public safety and socio-economic issue in developing countries like Bangladesh. Existing accident data collection is largely manual, fragmented, and unreliable, resulting in underreporting and inconsistent records. This research proposes a fully automated system using Large Language Models (LLMs) and web scraping techniques to address these challenges. The pipeline consists of four components: automated web scraping code generation, news collection from online sources, accident news classification with structured data extraction, and duplicate removal. The system uses the multimodal generative LLM Gemini-2.0-Flash for seamless automation. The code generation module classifies webpages into pagination, dynamic, or infinite scrolling categories and generates suitable Python scripts for scraping. LLMs also classify and extract key accident information such as date, time, location, fatalities, injuries, road type, vehicle types, and pedestrian involvement. A deduplication algorithm ensures data integrity by removing duplicate reports. The system scraped 14 major Bangladeshi news sites over 111 days (Oct 1, 2024 - Jan 20, 2025), processing over 15,000 news articles and identifying 705 unique accidents. The code generation module achieved 91.3% calibration and 80% validation accuracy. Chittagong reported the highest number of accidents (80), fatalities (70), and injuries (115), followed by Dhaka, Faridpur, Gazipur, and Cox's Bazar. Peak accident times were morning (8-9 AM), noon (12-1 PM), and evening (6-7 PM). A public repository was also developed with usage instructions. This study demonstrates the viability of an LLM-powered, scalable system for accurate, low-effort accident data collection, providing a foundation for data-driven road safety policymaking in Bangladesh.
>
---
#### [replaced 017] Superficial Safety Alignment Hypothesis
- **分类: cs.CL; cs.AI; cs.CR; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.10862v2](http://arxiv.org/pdf/2410.10862v2)**

> **作者:** Jianwei Li; Jung-Eun Kim
>
> **摘要:** As large language models (LLMs) are overwhelmingly more and more integrated into various applications, ensuring they generate safe responses is a pressing need. Previous studies on alignment have largely focused on general instruction-following but have often overlooked the distinct properties of safety alignment, such as the brittleness of safety mechanisms. To bridge the gap, we propose the Superficial Safety Alignment Hypothesis (SSAH), which posits that safety alignment teaches an otherwise unsafe model to choose the correct reasoning direction - fulfill or refuse users' requests - interpreted as an implicit binary classification task. Through SSAH, we hypothesize that only a few essential components can establish safety guardrails in LLMs. We successfully identify four types of attribute-critical components: Safety Critical Unit (SCU), Utility Critical Unit (UCU), Complex Unit (CU), and Redundant Unit (RU). Our findings show that freezing certain safety-critical components during fine-tuning allows the model to retain its safety attributes while adapting to new tasks. Similarly, we show that leveraging redundant units in the pre-trained model as an "alignment budget" can effectively minimize the alignment tax while achieving the alignment goal. All considered, this paper concludes that the atomic functional unit for safety in LLMs is at the neuron level and underscores that safety alignment should not be complicated.
>
---
#### [replaced 018] When Models Reason in Your Language: Controlling Thinking Language Comes at the Cost of Accuracy
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22888v2](http://arxiv.org/pdf/2505.22888v2)**

> **作者:** Jirui Qi; Shan Chen; Zidi Xiong; Raquel Fernández; Danielle S. Bitterman; Arianna Bisazza
>
> **备注:** Accepted at EMNLP 2025 Findings
>
> **摘要:** Recent Large Reasoning Models (LRMs) with thinking traces have shown strong performance on English reasoning tasks. However, their ability to think in other languages is less studied. This capability is as important as answer accuracy for real world applications because users may find the reasoning trace useful for oversight only when it is expressed in their own language. We comprehensively evaluate two leading families of LRMs on our XReasoning benchmark and find that even the most advanced models often revert to English or produce fragmented reasoning in other languages, revealing a substantial gap in multilingual reasoning. Prompt based interventions that force models to reason in the users language improve readability and oversight but reduce answer accuracy, exposing an important trade off. We further show that targeted post training on just 100 examples mitigates this mismatch, though some accuracy loss remains. Our results highlight the limited multilingual reasoning capabilities of current LRMs and outline directions for future work. Code and data are available at https://github.com/Betswish/mCoT-XReasoning.
>
---
#### [replaced 019] Interpretable Text Embeddings and Text Similarity Explanation: A Survey
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2502.14862v2](http://arxiv.org/pdf/2502.14862v2)**

> **作者:** Juri Opitz; Lucas Möller; Andrianos Michail; Sebastian Padó; Simon Clematide
>
> **备注:** EMNLP 2025 (main)
>
> **摘要:** Text embeddings are a fundamental component in many NLP tasks, including classification, regression, clustering, and semantic search. However, despite their ubiquitous application, challenges persist in interpreting embeddings and explaining similarities between them. In this work, we provide a structured overview of methods specializing in inherently interpretable text embeddings and text similarity explanation, an underexplored research area. We characterize the main ideas, approaches, and trade-offs. We compare means of evaluation, discuss overarching lessons learned and finally identify opportunities and open challenges for future research.
>
---
#### [replaced 020] DynaGuard: A Dynamic Guardrail Model With User-Defined Policies
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.02563v2](http://arxiv.org/pdf/2509.02563v2)**

> **作者:** Monte Hoover; Vatsal Baherwani; Neel Jain; Khalid Saifullah; Joseph Vincent; Chirag Jain; Melissa Kazemi Rad; C. Bayan Bruss; Ashwinee Panda; Tom Goldstein
>
> **备注:** 22 Pages
>
> **摘要:** Guardian models are used to supervise and moderate the outputs of user-facing chatbots, enforcing guardrails and detecting bad behaviors. Standard guardian models like LlamaGuard detect predefined, static categories of harms. We propose dynamic guardian models that evaluate text based on user-defined policies, making them useful for different application domains that are not addressed by standard guardian models. Our dynamic guardian models can be used for fast detection of policy violations or with chain-of-thought reasoning that articulates and justifies the model outputs. Our dynamic guardian models match static models in detection accuracy for static harm categories while identifying violations of free-form policies with accuracy comparable to frontier reasoning models in a fraction of the time.
>
---
#### [replaced 021] AbsTopK: Rethinking Sparse Autoencoders For Bidirectional Features
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.00404v2](http://arxiv.org/pdf/2510.00404v2)**

> **作者:** Xudong Zhu; Mohammad Mahdi Khalili; Zhihui Zhu
>
> **摘要:** Sparse autoencoders (SAEs) have emerged as powerful techniques for interpretability of large language models (LLMs), aiming to decompose hidden states into meaningful semantic features. While several SAE variants have been proposed, there remains no principled framework to derive SAEs from the original dictionary learning formulation. In this work, we introduce such a framework by unrolling the proximal gradient method for sparse coding. We show that a single-step update naturally recovers common SAE variants, including ReLU, JumpReLU, and TopK. Through this lens, we reveal a fundamental limitation of existing SAEs: their sparsity-inducing regularizers enforce non-negativity, preventing a single feature from representing bidirectional concepts (e.g., male vs. female). This structural constraint fragments semantic axes into separate, redundant features, limiting representational completeness. To address this issue, we propose AbsTopK SAE, a new variant derived from the $\ell_0$ sparsity constraint that applies hard thresholding over the largest-magnitude activations. By preserving both positive and negative activations, AbsTopK uncovers richer, bidirectional conceptual representations. Comprehensive experiments across four LLMs and seven probing and steering tasks show that AbsTopK improves reconstruction fidelity, enhances interpretability, and enables single features to encode contrasting concepts. Remarkably, AbsTopK matches or even surpasses the Difference-in-Mean method, a supervised approach that requires labeled data for each concept and has been shown in prior work to outperform SAEs.
>
---
#### [replaced 022] The Data-Quality Illusion: Rethinking Classifier-Based Quality Filtering for LLM Pretraining
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.00866v2](http://arxiv.org/pdf/2510.00866v2)**

> **作者:** Thiziri Nait Saada; Louis Bethune; Michal Klein; David Grangier; Marco Cuturi; Pierre Ablin
>
> **备注:** 21 pages, 20 figures, 2 tables, preprint
>
> **摘要:** Large-scale models are pretrained on massive web-crawled datasets containing documents of mixed quality, making data filtering essential. A popular method is Classifier-based Quality Filtering (CQF), which trains a binary classifier to distinguish between pretraining data and a small, high-quality set. It assigns each pretraining document a quality score defined as the classifier's score and retains only the top-scoring ones. We provide an in-depth analysis of CQF. We show that while CQF improves downstream task performance, it does not necessarily enhance language modeling on the high-quality dataset. We explain this paradox by the fact that CQF implicitly filters the high-quality dataset as well. We further compare the behavior of models trained with CQF to those trained on synthetic data of increasing quality, obtained via random token permutations, and find starkly different trends. Our results challenge the view that CQF captures a meaningful notion of data quality.
>
---
#### [replaced 023] Diversity-Enhanced Reasoning for Subjective Questions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20187v3](http://arxiv.org/pdf/2507.20187v3)**

> **作者:** Yumeng Wang; Zhiyuan Fan; Jiayu Liu; Jen-tse Huang; Yi R. Fung
>
> **摘要:** Large Reasoning Models (LRMs) with long chain-of-thought capabilities, optimized via reinforcement learning with verifiable rewards (RLVR), excel at objective reasoning tasks like mathematical problem solving and code generation. However, RLVR is known for degrading generation diversity, which causes LRMs to fall short on subjective reasoning that has multiple answers depending on different role perspectives. While recent studies recognize the importance of diversity-enhanced training in objective reasoning, limited attention has been given to subjective tasks. In this paper, we find that subjective reasoning can be improved by introducing perspective diversity and token-level diversity, with the former one providing a coherent scaffolding anchored to a real-world stakeholder group and the latter one broadening the answer search space. We propose MultiRole-R1, a diversity-enhanced training framework featuring an unsupervised data construction pipeline that synthesizes reasoning chains incorporating various role perspectives. It also employs reinforcement learning via Group Relative Policy Optimization with reward shaping, taking diversity as a reward signal in addition to verifiable reward. Training on subjective tasks solely, MultiRole-R1 increases the in-domain and out-of-domain accuracy by 14.1% and 7.64%, and even enhances the performance on advanced math reasoning such as AIME 2024. We further show that diversity is a more consistent indicator of accuracy than reasoning length.
>
---
#### [replaced 024] The Rise of AfricaNLP: Contributions, Contributors, and Community Impact (2005-2025)
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.25477v3](http://arxiv.org/pdf/2509.25477v3)**

> **作者:** Tadesse Destaw Belay; Kedir Yassin Hussen; Sukairaj Hafiz Imam; Ibrahim Said Ahmad; Isa Inuwa-Dutse; Abrham Belete Haile; Grigori Sidorov; Iqra Ameer; Idris Abdulmumin; Tajuddeen Gwadabe; Vukosi Marivate; Seid Muhie Yimam; Shamsuddeen Hassan Muhammad
>
> **摘要:** Natural Language Processing (NLP) is undergoing constant transformation, as Large Language Models (LLMs) are driving daily breakthroughs in research and practice. In this regard, tracking the progress of NLP research and automatically analyzing the contributions of research papers provides key insights into the nature of the field and the researchers. This study explores the progress of African NLP (AfricaNLP) by asking (and answering) basic research questions such as: i) How has the nature of NLP evolved over the last two decades?, ii) What are the contributions of AfricaNLP papers?, and iii) Which individuals and organizations (authors, affiliated institutions, and funding bodies) have been involved in the development of AfricaNLP? We quantitatively examine the contributions of AfricaNLP research using 1.9K NLP paper abstracts, 4.9K author contributors, and 7.8K human-annotated contribution sentences (AfricaNLPContributions) along with benchmark results. Our dataset and continuously existing NLP progress tracking website provide a powerful lens for tracing AfricaNLP research trends and hold potential for generating data-driven literature surveys.
>
---
#### [replaced 025] WebRollback: Enhancing Web Agents with Explicit Rollback Mechanisms
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.11788v2](http://arxiv.org/pdf/2504.11788v2)**

> **作者:** Zhisong Zhang; Tianqing Fang; Kaixin Ma; Wenhao Yu; Hongming Zhang; Haitao Mi; Dong Yu
>
> **摘要:** With recent advancements in large language models, web agents have been greatly improved. However, dealing with complex and dynamic web environments requires more advanced planning and search abilities. Previous studies usually adopt a greedy one-way search strategy, which may struggle to recover from erroneous states. In this work, we enhance web agents with an explicit rollback mechanism, enabling the agent to revert back to a previous state in its navigation trajectory. This mechanism gives the model the flexibility to directly control the search process, leading to an effective and efficient web navigation method. We conduct experiments on two live web navigation benchmarks with zero-shot and fine-tuning settings. The results demonstrate the effectiveness of our proposed approach.
>
---
#### [replaced 026] LEXam: Benchmarking Legal Reasoning on 340 Law Exams
- **分类: cs.CL; cs.AI; cs.LG; 68T50; I.2**

- **链接: [http://arxiv.org/pdf/2505.12864v4](http://arxiv.org/pdf/2505.12864v4)**

> **作者:** Yu Fan; Jingwei Ni; Jakob Merane; Yang Tian; Yoan Hermstrüwer; Yinya Huang; Mubashara Akhtar; Etienne Salimbeni; Florian Geering; Oliver Dreyer; Daniel Brunner; Markus Leippold; Mrinmaya Sachan; Alexander Stremitzer; Christoph Engel; Elliott Ash; Joel Niklaus
>
> **摘要:** Long-form legal reasoning remains a key challenge for large language models (LLMs) in spite of recent advances in test-time scaling. To address this, we introduce \textsc{LEXam}, a novel benchmark derived from 340 law exams spanning 116 law school courses across a range of subjects and degree levels. The dataset comprises 4,886 law exam questions in English and German, including 2,841 long-form, open-ended questions and 2,045 multiple-choice questions. Besides reference answers, the open questions are also accompanied by explicit guidance outlining the expected legal reasoning approach such as issue spotting, rule recall, or rule application. Our evaluation on both open-ended and multiple-choice questions present significant challenges for current LLMs; in particular, they notably struggle with open questions that require structured, multi-step legal reasoning. Moreover, our results underscore the effectiveness of the dataset in differentiating between models with varying capabilities. Deploying an ensemble LLM-as-a-Judge paradigm with rigorous human expert validation, we demonstrate how model-generated reasoning steps can be evaluated consistently and accurately, closely aligning with human expert assessments. Our evaluation setup provides a scalable method to assess legal reasoning quality beyond simple accuracy metrics. We have open-sourced our code on \href{https://github.com/LEXam-Benchmark/LEXam}{GitHub} and released our data on \href{https://huggingface.co/datasets/LEXam-Benchmark/LEXam}{Hugging Face}. Project page: https://lexam-benchmark.github.io/
>
---
#### [replaced 027] Self-Consistency Falls Short! The Adverse Effects of Positional Bias on Long-Context Problems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.01101v3](http://arxiv.org/pdf/2411.01101v3)**

> **作者:** Adam Byerly; Daniel Khashabi
>
> **备注:** 25 pages, 7 figures, 3 tables
>
> **摘要:** Self-consistency (SC) improves the performance of large language models (LLMs) across various tasks and domains that involve short content. However, does this support its effectiveness for long-context problems? We challenge the assumption that SC's benefits generalize to long-context settings, where LLMs often struggle with position bias, the systematic over-reliance on specific context regions-which hinders their ability to utilize information effectively from all parts of their context. Through comprehensive experimentation with varying state-of-the-art models, tasks, and SC formulations, we find that SC not only fails to improve but actively degrades performance on long-context tasks. This degradation is driven by persistent position bias, which worsens with longer context lengths and smaller model sizes but remains invariant to prompt format or task type. Unlike short-context tasks, where SC diversifies reasoning paths, long-context SC amplifies positional errors. These comprehensive results provide valuable insight into the limitations of current LLMs in long-context understanding and highlight the need for more sophisticated approaches.
>
---
#### [replaced 028] OntoURL: A Benchmark for Evaluating Large Language Models on Symbolic Ontological Understanding, Reasoning and Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11031v3](http://arxiv.org/pdf/2505.11031v3)**

> **作者:** Xiao Zhang; Huiyuan Lai; Qianru Meng; Johan Bos
>
> **摘要:** Large language models have demonstrated remarkable capabilities across a wide range of tasks, yet their ability to process structured symbolic knowledge remains underexplored. To address this gap, we propose a taxonomy of ontological capabilities and introduce OntoURL, the first comprehensive benchmark designed to systematically evaluate LLMs' capabilities in handling ontologies -- formal and symbolic representations of domain knowledge. Based on the proposed taxonomy, OntoURL systematically assesses three dimensions: understanding, reasoning, and learning through 15 distinct tasks comprising 57,303 questions derived from 40 ontologies across 8 domains. Experiments with 20 open-source LLMs reveal significant performance differences across models, tasks, and domains, with current LLMs showing capabilities in understanding ontological knowledge but weaknesses in reasoning and learning tasks. Further experiments with few-shot and chain-of-thought prompting illustrate how different prompting strategies affect model performance. Additionally, a human evaluation reveals that LLMs outperform humans in understanding and reasoning tasks but fall short in most learning tasks. These findings highlight both the potential and limitations of LLMs in processing symbolic knowledge and establish OntoURL as a critical benchmark for advancing the integration of LLMs with formal knowledge representations.
>
---
#### [replaced 029] Euclid's Gift: Enhancing Spatial Perception and Reasoning in Vision-Language Models via Geometric Surrogate Tasks
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.24473v2](http://arxiv.org/pdf/2509.24473v2)**

> **作者:** Shijie Lian; Changti Wu; Laurence Tianruo Yang; Hang Yuan; Bin Yu; Lei Zhang; Kai Chen
>
> **摘要:** Spatial intelligence spans a rich suite of abilities, including visualising and transforming shapes, mentally rotating objects, judging relational positions and containment, and estimating numerosity. However, it still remains a critical unresolved challenge for Multimodal Large Language Models (MLLMs).To fill this gap, we propose to treat Euclidean geometry problem-solving as a surrogate task. Specifically, we meticulously constructed a curated multimodal dataset, called Euclid30K, comprising approximately 30K plane and solid geometry problems. To enable the model to acquire and apply Euclidean principles from these geometry problems, we employed Group Relative Policy Optimization (GRPO) to finetune the Qwen2.5VL family and RoboBrain2.0 family, inspiring the models to identify shapes, count, and relate entities, and perform multi-step deductive reasoning using Euclidean principles. Our experiments demonstrate that the resulting models achieve substantial zero-shot gains across four spatial reasoning benchmarks (Super-CLEVR, Omni3DBench, VSI-Bench, and MindCube) without any task-specific adaptations. Notably, after training on the Euclid30K, the mean VSI-Bench accuracy of all evaluated models rose from 34.5% to 40.5%, improving by 5.5 percentage points. Among them, RoboBrain2.0-Euclid-7B achieves 49.6\% accuracy, surpassing the previous state-of-the-art model, Spatial-MLLM.To our knowledge, this is the first systematic study showing that geometry-centric fine-tuning can confer vision-language models with broadly transferable spatial skills. Code and Euclid30K dataset can be found in https://zgca-ai4edu.github.io/Euclids_Gift.
>
---
#### [replaced 030] Adapting Large Language Models for Character-based Augmentative and Alternative Communication
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2501.10582v3](http://arxiv.org/pdf/2501.10582v3)**

> **作者:** Dylan Gaines; Keith Vertanen
>
> **备注:** To appear in Findings of EMNLP 2025
>
> **摘要:** Users of Augmentative and Alternative Communication (AAC) may write letter-by-letter via an interface that uses a character language model. However, most state-of-the-art large pretrained language models predict subword tokens of variable length. We investigate how to practically use such models to make accurate and efficient character predictions. Our algorithm for producing character predictions from a subword large language model (LLM) provides more accurate predictions than using a classification layer, a byte-level LLM, or an n-gram model. Additionally, we investigate a domain adaptation procedure based on a large dataset of sentences we curated based on scoring how useful each sentence might be for spoken or written AAC communication. We find our procedure further improves model performance on simple, conversational text.
>
---
#### [replaced 031] Mafoko: Structuring and Building Open Multilingual Terminologies for South African NLP
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.03529v2](http://arxiv.org/pdf/2508.03529v2)**

> **作者:** Vukosi Marivate; Isheanesu Dzingirai; Fiskani Banda; Richard Lastrucci; Thapelo Sindane; Keabetswe Madumo; Kayode Olaleye; Abiodun Modupe; Unarine Netshifhefhe; Herkulaas Combrink; Mohlatlego Nakeng; Matome Ledwaba
>
> **备注:** Under Review
>
> **摘要:** The critical lack of structured terminological data for South Africa's official languages hampers progress in multilingual NLP, despite the existence of numerous government and academic terminology lists. These valuable assets remain fragmented and locked in non-machine-readable formats, rendering them unusable for computational research and development. \emph{Mafoko} addresses this challenge by systematically aggregating, cleaning, and standardising these scattered resources into open, interoperable datasets. We introduce the foundational \emph{Mafoko} dataset, released under the equitable, Africa-centered NOODL framework. To demonstrate its immediate utility, we integrate the terminology into a Retrieval-Augmented Generation (RAG) pipeline. Experiments show substantial improvements in the accuracy and domain-specific consistency of English-to-Tshivenda machine translation for large language models. \emph{Mafoko} provides a scalable foundation for developing robust and equitable NLP technologies, ensuring South Africa's rich linguistic diversity is represented in the digital age.
>
---
#### [replaced 032] MolLangBench: A Comprehensive Benchmark for Language-Prompted Molecular Structure Recognition, Editing, and Generation
- **分类: cs.CL; cs.AI; cs.LG; q-bio.BM**

- **链接: [http://arxiv.org/pdf/2505.15054v2](http://arxiv.org/pdf/2505.15054v2)**

> **作者:** Feiyang Cai; Jiahui Bai; Tao Tang; Guijuan He; Joshua Luo; Tianyu Zhu; Srikanth Pilla; Gang Li; Ling Liu; Feng Luo
>
> **摘要:** Precise recognition, editing, and generation of molecules are essential prerequisites for both chemists and AI systems tackling various chemical tasks. We present MolLangBench, a comprehensive benchmark designed to evaluate fundamental molecule-language interface tasks: language-prompted molecular structure recognition, editing, and generation. To ensure high-quality, unambiguous, and deterministic outputs, we construct the recognition tasks using automated cheminformatics tools, and curate editing and generation tasks through rigorous expert annotation and validation. MolLangBench supports the evaluation of models that interface language with different molecular representations, including linear strings, molecular images, and molecular graphs. Evaluations of state-of-the-art models reveal significant limitations: the strongest model (GPT-5) achieves $86.2\%$ and $85.5\%$ accuracy on recognition and editing tasks, which are intuitively simple for humans, and performs even worse on the generation task, reaching only $43.0\%$ accuracy. These results highlight the shortcomings of current AI systems in handling even preliminary molecular recognition and manipulation tasks. We hope MolLangBench will catalyze further research toward more effective and reliable AI systems for chemical applications.
>
---
#### [replaced 033] Benchmarking Foundation Models with Retrieval-Augmented Generation in Olympic-Level Physics Problem Solving
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.00919v2](http://arxiv.org/pdf/2510.00919v2)**

> **作者:** Shunfeng Zheng; Yudi Zhang; Meng Fang; Zihan Zhang; Zhitan Wu; Mykola Pechenizkiy; Ling Chen
>
> **备注:** Accepted to EMNLP 2025 (Findings)
>
> **摘要:** Retrieval-augmented generation (RAG) with foundation models has achieved strong performance across diverse tasks, but their capacity for expert-level reasoning-such as solving Olympiad-level physics problems-remains largely unexplored. Inspired by the way students prepare for competitions by reviewing past problems, we investigate the potential of RAG to enhance physics reasoning in foundation models. We introduce PhoPile, a high-quality multimodal dataset specifically designed for Olympiad-level physics, enabling systematic study of retrieval-based reasoning. PhoPile includes diagrams, graphs, and equations, capturing the inherently multimodal nature of physics problem solving. Using PhoPile, we benchmark RAG-augmented foundation models, covering both large language models (LLMs) and large multimodal models (LMMs) with multiple retrievers. Our results demonstrate that integrating retrieval with physics corpora can improve model performance, while also highlighting challenges that motivate further research in retrieval-augmented physics reasoning.
>
---
#### [replaced 034] Reasoning Models are Test Exploiters: Rethinking Multiple-Choice
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.15337v2](http://arxiv.org/pdf/2507.15337v2)**

> **作者:** Narun Raman; Taylor Lundy; Kevin Leyton-Brown
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** When evaluating Large Language Models (LLMs) in question answering domains, it is common to ask the model to choose among a fixed set of choices (so-called multiple-choice question-answering, or MCQA). Although downstream tasks of interest typically do not provide systems with explicit options among which to choose, this approach is nevertheless widely used because it makes automatic grading straightforward and has tended to produce challenging benchmarks that correlate sufficiently well with downstream performance. This paper investigates the extent to which this trend continues to hold for state-of-the-art reasoning models, describing a systematic evaluation of 15 different question-answering benchmarks (e.g., MMLU, GSM8K) and 27 different LLMs (including small models such as Qwen-2.5 7B, mid-sized models such as Llama-3.3 70B, and large state-of-the-art models such as OpenAI's o3). For each model--benchmark pair, we considered 5 ways of presenting the model with questions, including variations on whether multiple choices were offered to the model at all; whether "none of the above" sometimes replaced the right answer; and whether the model was permitted to perform chain-of-thought reasoning before and/or after the choices were presented. MCQA remained a good proxy for the downstream performance of models as long as they were allowed to perform chain-of-thought reasoning only \emph{before} being presented with the options among which they had to select. On the other hand, large models that were able to perform reasoning \emph{after} being given a set of options tended to significantly outperform their free-text performance due to exploiting the information in the options. We identify and quantify the signals models are using when answering MCQA questions, and offer practical guidelines when analyzing results from MCQA that better reflect LLMs' genuine reasoning capabilities.
>
---
#### [replaced 035] CRUST-Bench: A Comprehensive Benchmark for C-to-safe-Rust Transpilation
- **分类: cs.SE; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.15254v3](http://arxiv.org/pdf/2504.15254v3)**

> **作者:** Anirudh Khatry; Robert Zhang; Jia Pan; Ziteng Wang; Qiaochu Chen; Greg Durrett; Isil Dillig
>
> **备注:** To be published at COLM, 2025
>
> **摘要:** C-to-Rust transpilation is essential for modernizing legacy C code while enhancing safety and interoperability with modern Rust ecosystems. However, no dataset currently exists for evaluating whether a system can transpile C into safe Rust that passes a set of test cases. We introduce CRUST-Bench, a dataset of 100 C repositories, each paired with manually-written interfaces in safe Rust as well as test cases that can be used to validate correctness of the transpilation. By considering entire repositories rather than isolated functions, CRUST-Bench captures the challenges of translating complex projects with dependencies across multiple files. The provided Rust interfaces provide explicit specifications that ensure adherence to idiomatic, memory-safe Rust patterns, while the accompanying test cases enforce functional correctness. We evaluate state-of-the-art large language models (LLMs) on this task and find that safe and idiomatic Rust generation is still a challenging problem for various state-of-the-art methods and techniques. We also provide insights into the errors LLMs usually make in transpiling code from C to safe Rust. The best performing model, OpenAI o1, is able to solve only 15 tasks in a single-shot setting. Improvements on CRUST-Bench would lead to improved transpilation systems that can reason about complex scenarios and help in migrating legacy codebases from C into languages like Rust that ensure memory safety. You can find the dataset and code at https://github.com/anirudhkhatry/CRUST-bench.
>
---
#### [replaced 036] ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.14238v3](http://arxiv.org/pdf/2505.14238v3)**

> **作者:** Raghav Singhal; Kaustubh Ponkshe; Rohit Vartak; Praneeth Vepakomma
>
> **备注:** Raghav Singhal, Kaustubh Ponkshe, and Rohit Vartak contributed equally to this work
>
> **摘要:** Large Language Models have demonstrated strong performance across a wide range of tasks, but adapting them efficiently to new domains remains a key challenge. Parameter-Efficient Fine-Tuning (PEFT) methods address this by introducing lightweight, trainable modules while keeping most pre-trained weights fixed. The prevailing approach, LoRA, models updates using a low-rank decomposition, but its expressivity is inherently constrained by the rank. Recent methods like HiRA aim to increase expressivity by incorporating a Hadamard product with the frozen weights, but still rely on the structure of the pre-trained model. We introduce ABBA, a new PEFT architecture that reparameterizes the update as a Hadamard product of two independently learnable low-rank matrices. In contrast to prior work, ABBA fully decouples the update from the pre-trained weights, enabling both components to be optimized freely. This leads to significantly higher expressivity under the same parameter budget, a property we validate through matrix reconstruction experiments. Empirically, ABBA achieves state-of-the-art results on arithmetic and commonsense reasoning benchmarks, consistently outperforming existing PEFT methods by a significant margin across multiple models. Our code is publicly available at: https://github.com/CERT-Lab/abba.
>
---
#### [replaced 037] Charting the Landscape of African NLP: Mapping Progress and Shaping the Road Ahead
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21315v3](http://arxiv.org/pdf/2505.21315v3)**

> **作者:** Jesujoba O. Alabi; Michael A. Hedderich; David Ifeoluwa Adelani; Dietrich Klakow
>
> **备注:** EMNLP 2025
>
> **摘要:** With over 2,000 languages and potentially millions of speakers, Africa represents one of the richest linguistic regions in the world. Yet, this diversity is scarcely reflected in state-of-the-art natural language processing (NLP) systems and large language models (LLMs), which predominantly support a narrow set of high-resource languages. This exclusion not only limits the reach and utility of modern NLP technologies but also risks widening the digital divide across linguistic communities. Nevertheless, NLP research on African languages is active and growing. In recent years, there has been a surge of interest in this area, driven by several factors-including the creation of multilingual language resources, the rise of community-led initiatives, and increased support through funding programs. In this survey, we analyze 884 research papers on NLP for African languages published over the past five years, offering a comprehensive overview of recent progress across core tasks. We identify key trends shaping the field and conclude by outlining promising directions to foster more inclusive and sustainable NLP research for African languages.
>
---
#### [replaced 038] Can Code-Switched Texts Activate a Knowledge Switch in LLMs? A Case Study on English-Korean Code-Switching
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.18436v4](http://arxiv.org/pdf/2410.18436v4)**

> **作者:** Seoyeon Kim; Huiseo Kim; Chanjun Park; Jinyoung Yeo; Dongha Lee
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Recent large language models (LLMs) demonstrate multilingual abilities, yet they are English-centric due to dominance of English in training corpora. The limited resource for low-resource languages remains a crucial challenge. Code-switching (CS), a phenomenon where multilingual speakers alternate between languages in a discourse, can convey subtle cultural and linguistic nuances that can be otherwise lost in translation and elicits language-specific knowledge in human communications. In light of this, we investigate whether code-switching can activate, or identify and leverage knowledge for reasoning when LLMs solve low-resource language tasks. To facilitate the research, we first present EnKoQA, a synthetic English-Korean CS question-answering dataset. We provide comprehensive analysis on a variety of multilingual LLMs by subdividing activation process into knowledge identification and knowledge leveraging. Our results demonstrate that compared to English text, CS can faithfully activate knowledge inside LLMs especially on language-specific domains, suggesting the potential of code-switching on low-resource language tasks.
>
---
#### [replaced 039] Efficient Whole Slide Pathology VQA via Token Compression
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.14497v2](http://arxiv.org/pdf/2507.14497v2)**

> **作者:** Weimin Lyu; Qingqiao Hu; Kehan Qi; Zhan Shi; Wentao Huang; Saumya Gupta; Chao Chen
>
> **摘要:** Whole-slide images (WSIs) in pathology can reach up to 10,000 x 10,000 pixels, posing significant challenges for multimodal large language model (MLLM) due to long context length and high computational demands. Previous methods typically focus on patch-level analysis or slide-level classification using CLIP-based models with multi-instance learning, but they lack the generative capabilities needed for visual question answering (VQA). More recent MLLM-based approaches address VQA by feeding thousands of patch tokens directly into the language model, which leads to excessive resource consumption. To address these limitations, we propose Token Compression Pathology LLaVA (TCP-LLaVA), the first MLLM architecture to perform WSI VQA via token compression. TCP-LLaVA introduces a set of trainable compression tokens that aggregate visual and textual information through a modality compression module, inspired by the [CLS] token mechanism in BERT. Only the compressed tokens are forwarded to the LLM for answer generation, significantly reducing input length and computational cost. Experiments on ten TCGA tumor subtypes show that TCP-LLaVA outperforms existing MLLM baselines in VQA accuracy while reducing training resource consumption by a substantial margin.
>
---
#### [replaced 040] FANS -- Formal Answer Selection for Natural Language Math Reasoning Using Lean4
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.03238v2](http://arxiv.org/pdf/2503.03238v2)**

> **作者:** Jiarui Yao; Ruida Wang; Tong Zhang
>
> **摘要:** Large Language Models (LLMs) have displayed astonishing abilities in various tasks, especially in text generation, classification, question answering, etc. However, the reasoning ability of LLMs still faces many debates. The inherent ambiguity of Natural Language (NL) limits LLMs' ability to perform verifiable reasoning, making its answers lack coherence and trustworthy support. To tackle the above problems, we propose a novel framework named FANS: Formal ANswer Selection for Natural Language Math Reasoning Using Lean4. To the best of our knowledge, it is the first framework that utilizes Lean4 to enhance LLMs' NL math reasoning ability. In particular, given an NL math question and LLM-generated answers, FANS first translates it into Lean4 theorem statements. Then it tries to prove it using a Lean4 prover and verify it by Lean4. Finally, it uses the FL result to assist in answer selection. It enhances LLMs' NL math ability in providing a computer-verifiable solution for its correct answer and proposes an alternative method for answer selection beyond the reward model. Extensive experiments indicate the effectiveness of our framework. It can improve the accuracy rate of reward model enhanced LLMs in the MATH-500 dataset by at most 1.91% and AMC-23 by at most 8.33% on strong reward-model baselines. In some particular fields like number theory that Lean4 experts in, we can even select all correct solutions. The qualitative analysis also shows our framework can make NL results formally backed by Lean4 proofs. As a pioneering work in the corresponding field, we will open-source all our models and datasets to further boost the development of the field.
>
---
#### [replaced 041] Deriving Strategic Market Insights with Large Language Models: A Benchmark for Forward Counterfactual Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.19430v3](http://arxiv.org/pdf/2505.19430v3)**

> **作者:** Keane Ong; Rui Mao; Deeksha Varshney; Paul Pu Liang; Erik Cambria; Gianmarco Mengaldo
>
> **备注:** Published at Empirical Methods in Natural Language Processing 2025 (Main Conference) (Oral)
>
> **摘要:** Counterfactual reasoning typically involves considering alternatives to actual events. While often applied to understand past events, a distinct form-forward counterfactual reasoning-focuses on anticipating plausible future developments. This type of reasoning is invaluable in dynamic financial markets, where anticipating market developments can powerfully unveil potential risks and opportunities for stakeholders, guiding their decision-making. However, performing this at scale is challenging due to the cognitive demands involved, underscoring the need for automated solutions. LLMs offer promise, but remain unexplored for this application. To address this gap, we introduce a novel benchmark, FIN-FORCE-FINancial FORward Counterfactual Evaluation. By curating financial news headlines and providing structured evaluation, FIN-FORCE supports LLM based forward counterfactual generation. This paves the way for scalable and automated solutions for exploring and anticipating future market developments, thereby providing structured insights for decision-making. Through experiments on FIN-FORCE, we evaluate state-of-the-art LLMs and counterfactual generation methods, analyzing their limitations and proposing insights for future research. We release the benchmark, supplementary data and all experimental codes at the following link: https://github.com/keanepotato/fin_force
>
---
#### [replaced 042] Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.14245v2](http://arxiv.org/pdf/2506.14245v2)**

> **作者:** Xumeng Wen; Zihan Liu; Shun Zheng; Shengyu Ye; Zhirong Wu; Yang Wang; Zhijian Xu; Xiao Liang; Junjie Li; Ziming Miao; Jiang Bian; Mao Yang
>
> **备注:** Update with more experiments
>
> **摘要:** Recent advancements in long chain-of-thought (CoT) reasoning, particularly through the Group Relative Policy Optimization algorithm used by DeepSeek-R1, have led to significant interest in the potential of Reinforcement Learning with Verifiable Rewards (RLVR) for Large Language Models (LLMs). While RLVR promises to improve reasoning by allowing models to learn from free exploration, there remains debate over whether it truly enhances reasoning abilities or simply boosts sampling efficiency. This paper systematically investigates the impact of RLVR on LLM reasoning. We revisit Pass@K experiments and demonstrate that RLVR can extend the reasoning boundary for both mathematical and coding tasks. This is supported by our introduction of a novel evaluation metric, CoT-Pass@K, which captures reasoning success by accounting for both the final answer and intermediate reasoning steps. Furthermore, we present a theoretical framework explaining RLVR's incentive mechanism, demonstrating how it can encourage correct reasoning even when rewards are based solely on answer correctness. Our analysis of RLVR's training dynamics reveals that it incentivizes correct reasoning early in the process, with substantial improvements in reasoning quality confirmed through extensive evaluations. These findings provide strong evidence of RLVR's potential to enhance LLM reasoning, offering valuable insights into its mechanisms and performance improvements.
>
---
#### [replaced 043] AutoScale: Scale-Aware Data Mixing for Pre-Training LLMs
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2407.20177v5](http://arxiv.org/pdf/2407.20177v5)**

> **作者:** Feiyang Kang; Yifan Sun; Bingbing Wen; Si Chen; Dawn Song; Rafid Mahmood; Ruoxi Jia
>
> **备注:** Published as a conference paper at COLM 2025
>
> **摘要:** Domain reweighting is an emerging research area aimed at adjusting the relative weights of different data sources to improve the effectiveness and efficiency of LLM pre-training. We show that data mixtures that perform well at smaller scales may not retain their advantage at larger scales, challenging the existing practice of determining competitive mixtures in small-scale experiments and directly applying them at much larger scales. To address this, we propose AutoScale, a two-stage, scale-aware data composition framework. First, AutoScale fits a parametric model that predicts the model's loss under different data compositions, then uses it to find an approximate best allocation at smaller, more manageable budgets. Next, leveraging a novel theoretical analysis of how optimal compositions evolve with scale, AutoScale extrapolates that composition to larger budgets without further retraining. Empirically, AutoScale accelerates convergence and improves downstream performance. For instance, when pre-training GPT-2 Large, it achieves a 28% faster perplexity reduction than baselines and up to a 38% speed-up over unweighted training, while yielding best-average results on various downstream tasks. Overall, our findings illustrate how domain importance shifts with training scale, underscoring the need for scale-dependent data curation in LLM training. Our code is open-sourced.
>
---
#### [replaced 044] When Disagreements Elicit Robustness: Investigating Self-Repair Capabilities under LLM Multi-Agent Disagreements
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.15153v2](http://arxiv.org/pdf/2502.15153v2)**

> **作者:** Tianjie Ju; Bowen Wang; Hao Fei; Mong-Li Lee; Wynne Hsu; Yun Li; Qianren Wang; Pengzhou Cheng; Zongru Wu; Haodong Zhao; Zhuosheng Zhang; Gongshen Liu
>
> **备注:** Working in progress
>
> **摘要:** Recent advances in Large Language Models (LLMs) have upgraded them from sophisticated text generators to autonomous agents capable of cooperation and tool use in multi-agent systems (MAS). However, it remains unclear how disagreements shape collective decision-making. In this paper, we revisit the role of disagreement and argue that general, partially overlapping disagreements prevent premature consensus and expand the explored solution space, while disagreements on task-critical steps can derail collaboration depending on the topology of solution paths. We investigate two collaborative settings with distinct path structures: collaborative reasoning (CounterFact, MQuAKE-cf), which typically follows a single evidential chain, whereas collaborative programming (HumanEval, GAIA) often adopts multiple valid implementations. Disagreements are instantiated as general heterogeneity among agents and as task-critical counterfactual knowledge edits injected into context or parameters. Experiments reveal that general disagreements consistently improve success by encouraging complementary exploration. By contrast, task-critical disagreements substantially reduce success on single-path reasoning, yet have a limited impact on programming, where agents can choose alternative solutions. Trace analyses show that MAS frequently bypasses the edited facts in programming but rarely does so in reasoning, revealing an emergent self-repair capability that depends on solution-path rather than scale alone. Our code is available at https://github.com/wbw625/MultiAgentRobustness.
>
---
#### [replaced 045] PurpCode: Reasoning for Safer Code Generation
- **分类: cs.CR; cs.CL; cs.LG; cs.SE**

- **链接: [http://arxiv.org/pdf/2507.19060v3](http://arxiv.org/pdf/2507.19060v3)**

> **作者:** Jiawei Liu; Nirav Diwan; Zhe Wang; Haoyu Zhai; Xiaona Zhou; Kiet A. Nguyen; Tianjiao Yu; Muntasir Wahed; Yinlin Deng; Hadjer Benkraouda; Yuxiang Wei; Lingming Zhang; Ismini Lourentzou; Gang Wang
>
> **摘要:** We introduce PurpCode, the first post-training recipe for training safe code reasoning models towards generating secure code and defending against malicious cyberactivities. PurpCode trains a reasoning model in two stages: (i) Rule Learning, which explicitly teaches the model to reference cybersafety rules to generate vulnerability-free code and to avoid facilitating malicious cyberactivities; and (ii) Reinforcement Learning, which optimizes model safety and preserves model utility through diverse, multi-objective reward mechanisms. To empower the training pipelines with comprehensive cybersafety data, we conduct internal red-teaming to synthesize comprehensive and high-coverage prompts based on real-world tasks for inducing unsafe cyberactivities in the model. Based on PurpCode, we develop a reasoning-based coding model, namely PurpCode-32B, which demonstrates state-of-the-art cybersafety, outperforming various frontier models. Meanwhile, our alignment method decreases the model overrefusal rates in both general and cybersafety-specific scenarios, while preserving model utility in both code generation and common security knowledge.
>
---
#### [replaced 046] Aligning Reasoning LLMs for Materials Discovery with Physics-aware Rejection Sampling
- **分类: cs.AI; cond-mat.mtrl-sci; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.00768v2](http://arxiv.org/pdf/2509.00768v2)**

> **作者:** Lee Hyun; Sohee Yoon; Jinwoo Park; Sue In Chae; Seongeon Park; Jooyeon Ahn; Yebin Jung; Youjung Chung; Hogeun Chang; Sujin Park; Myeonginn Kang; Jina Kim; Ho-Gyeong Kim; Myeonghun Jeong
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** AI-driven materials discovery that couples automated experimentation with algorithmic decision-making requires process aware recipe to property predictors that are accurate, calibrated, and physically admissible. We approach this as a reasoning problem with large reasoning models (LRMs). To instill reasoning capability into language models, we curate reasoning traces from a teacher model to train a student model. However, most training pipelines select reasoning traces using binary correctness or learned preference signals that poorly reflect physical admissibility. We introduce Physics-aware Rejection Sampling (PaRS), a training-time trace selection scheme that favors traces consistent with fundamental physics and numerically close to targets, with lightweight halting to control compute. We instantiate our framework with a large student model fine-tuned on traces synthesized by a larger teacher model, and evaluate under matched token budgets against various rejection sampling baselines. Our method improves accuracy and calibration, reduces physics-violation rates, and lowers sampling cost relative to baselines. These results indicate that modest, domain-aware constraints combined with trace-level selection provide a practical path toward reliable, efficient LRMs for process-aware property prediction and closed-loop materials design.
>
---
#### [replaced 047] Co-NAML-LSTUR: A Combined Model with Attentive Multi-View Learning and Long- and Short-term User Representations for News Recommendation
- **分类: cs.CL; 68T50, 68T05; I.2.7; I.7**

- **链接: [http://arxiv.org/pdf/2507.20210v2](http://arxiv.org/pdf/2507.20210v2)**

> **作者:** Minh Hoang Nguyen; Thuat Thien Nguyen; Minh Nhat Ta; Tung Le; Huy Tien Nguyen
>
> **备注:** MIWAI 2025
>
> **摘要:** News recommendation systems play a critical role in alleviating information overload by delivering personalized content. A key challenge lies in jointly modeling multi-view representations of news articles and capturing the dynamic, dual-scale nature of user interests-encompassing both short- and long-term preferences. Prior methods often rely on single-view features or insufficiently model user behavior across time. In this work, we introduce Co-NAML-LSTUR, a hybrid news recommendation framework that integrates NAML for attentive multi-view news encoding and LSTUR for hierarchical user modeling, designed for training on limited data resources. Our approach leverages BERT-based embeddings to enhance semantic representation. We evaluate Co-NAML-LSTUR on two widely used benchmarks, MIND-small and MIND-large. Results show that our model significantly outperforms strong baselines, achieving improvements over NRMS by 1.55% in AUC and 1.15% in MRR, and over NAML by 2.45% in AUC and 1.71% in MRR. These findings highlight the effectiveness of our efficiency-focused hybrid model, which combines multi-view news modeling with dual-scale user representations for practical, resource-limited resources rather than a claim to absolute state-of-the-art (SOTA). The implementation of our model is publicly available at https://github.com/MinhNguyenDS/Co-NAML-LSTUR
>
---
#### [replaced 048] Diagnosing and Addressing Pitfalls in KG-RAG Datasets: Toward More Reliable Benchmarking
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.23495v2](http://arxiv.org/pdf/2505.23495v2)**

> **作者:** Liangliang Zhang; Zhuorui Jiang; Hongliang Chi; Haoyang Chen; Mohammed Elkoumy; Fali Wang; Qiong Wu; Zhengyi Zhou; Shirui Pan; Suhang Wang; Yao Ma
>
> **备注:** Accepted at NeurIPS 2025 Datasets and Benchmarks Track
>
> **摘要:** Knowledge Graph Question Answering (KGQA) systems rely on high-quality benchmarks to evaluate complex multi-hop reasoning. However, despite their widespread use, popular datasets such as WebQSP and CWQ suffer from critical quality issues, including inaccurate or incomplete ground-truth annotations, poorly constructed questions that are ambiguous, trivial, or unanswerable, and outdated or inconsistent knowledge. Through a manual audit of 16 popular KGQA datasets, including WebQSP and CWQ, we find that the average factual correctness rate is only 57 %. To address these issues, we introduce KGQAGen, an LLM-in-the-loop framework that systematically resolves these pitfalls. KGQAGen combines structured knowledge grounding, LLM-guided generation, and symbolic verification to produce challenging and verifiable QA instances. Using KGQAGen, we construct KGQAGen-10k, a ten-thousand scale benchmark grounded in Wikidata, and evaluate a diverse set of KG-RAG models. Experimental results demonstrate that even state-of-the-art systems struggle on this benchmark, highlighting its ability to expose limitations of existing models. Our findings advocate for more rigorous benchmark construction and position KGQAGen as a scalable framework for advancing KGQA evaluation.
>
---
#### [replaced 049] SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models
- **分类: cs.CL; cs.AI; cs.LG; cs.MM; cs.SD; eess.AS; I.2.7**

- **链接: [http://arxiv.org/pdf/2509.14270v2](http://arxiv.org/pdf/2509.14270v2)**

> **作者:** Karan Dua; Puneet Mittal; Ranjeet Gupta; Hitesh Laxmichand Patel
>
> **备注:** Accepted at ACL 2025
>
> **摘要:** High-quality Text-to-Speech (TTS) model training requires extensive and diverse text and speech data. It is challenging to procure such data from real sources due to issues of domain specificity, licensing, and scalability. Large language models (LLMs) can certainly generate textual data, but they create repetitive text with insufficient variation in the prompt during the generation process. Another important aspect in TTS training data is text normalization. Tools for normalization might occasionally introduce anomalies or overlook valuable patterns, and thus impact data quality. Furthermore, it is also impractical to rely on voice artists for large scale speech recording in commercial TTS systems with standardized voices. To address these challenges, we propose SpeechWeave, a synthetic speech data generation pipeline that is capable of automating the generation of multilingual, domain-specific datasets for training TTS models. Our experiments reveal that our pipeline generates data that is 10-48% more diverse than the baseline across various linguistic and phonetic metrics, along with speaker-standardized speech audio while generating approximately 97% correctly normalized text. Our approach enables scalable, high-quality data generation for TTS training, improving diversity, normalization, and voice consistency in the generated datasets.
>
---
#### [replaced 050] BiasLab: Toward Explainable Political Bias Detection with Dual-Axis Annotations and Rationale Indicators
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16081v2](http://arxiv.org/pdf/2505.16081v2)**

> **作者:** Kma Solaiman
>
> **备注:** Presented at ICML 2025 2nd Workshop on Models of Human Feedback for AI Alignment, Vancouver, Canada
>
> **摘要:** We present BiasLab, a dataset of 300 political news articles annotated for perceived ideological bias. These articles were selected from a curated 900-document pool covering diverse political events and source biases. Each article is labeled by crowdworkers along two independent scales, assessing sentiment toward the Democratic and Republican parties, and enriched with rationale indicators. The annotation pipeline incorporates targeted worker qualification and was refined through pilot-phase analysis. We quantify inter-annotator agreement, analyze misalignment with source-level outlet bias, and organize the resulting labels into interpretable subsets. Additionally, we simulate annotation using schema-constrained GPT-4o, enabling direct comparison to human labels and revealing mirrored asymmetries, especially in misclassifying subtly right-leaning content. We define two modeling tasks: perception drift prediction and rationale type classification, and report baseline performance to illustrate the challenge of explainable bias detection. BiasLab's rich rationale annotations provide actionable interpretations that facilitate explainable modeling of political bias, supporting the development of transparent, socially aware NLP systems. We release the dataset, annotation schema, and modeling code to encourage research on human-in-the-loop interpretability and the evaluation of explanation effectiveness in real-world settings.
>
---
#### [replaced 051] Defend LLMs Through Self-Consciousness
- **分类: cs.AI; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2508.02961v2](http://arxiv.org/pdf/2508.02961v2)**

> **作者:** Boshi Huang; Fabio Nonato de Paula
>
> **备注:** company requests to withdraw
>
> **摘要:** This paper introduces a novel self-consciousness defense mechanism for Large Language Models (LLMs) to combat prompt injection attacks. Unlike traditional approaches that rely on external classifiers, our method leverages the LLM's inherent reasoning capabilities to perform self-protection. We propose a framework that incorporates Meta-Cognitive and Arbitration Modules, enabling LLMs to evaluate and regulate their own outputs autonomously. Our approach is evaluated on seven state-of-the-art LLMs using two datasets: AdvBench and Prompt-Injection-Mixed-Techniques-2024. Experiment results demonstrate significant improvements in defense success rates across models and datasets, with some achieving perfect and near-perfect defense in Enhanced Mode. We also analyze the trade-off between defense success rate improvement and computational overhead. This self-consciousness method offers a lightweight, cost-effective solution for enhancing LLM ethics, particularly beneficial for GenAI use cases across various platforms.
>
---
#### [replaced 052] Probabilistic Reasoning with LLMs for k-anonymity Estimation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.09674v4](http://arxiv.org/pdf/2503.09674v4)**

> **作者:** Jonathan Zheng; Sauvik Das; Alan Ritter; Wei Xu
>
> **备注:** 10 pages, Accepted to NeurIPS 2025
>
> **摘要:** Probabilistic reasoning is a key aspect of both human and artificial intelligence that allows for handling uncertainty and ambiguity in decision-making. In this paper, we introduce a new numerical reasoning task under uncertainty for large language models, focusing on estimating the privacy risk of user-generated documents containing privacy-sensitive information. We propose BRANCH, a new LLM methodology that estimates the k-privacy value of a text-the size of the population matching the given information. BRANCH factorizes a joint probability distribution of personal information as random variables. The probability of each factor in a population is estimated separately using a Bayesian network and combined to compute the final k-value. Our experiments show that this method successfully estimates the k-value 73% of the time, a 13% increase compared to o3-mini with chain-of-thought reasoning. We also find that LLM uncertainty is a good indicator for accuracy, as high-variance predictions are 37.47% less accurate on average.
>
---
#### [replaced 053] Flexible Feature Distillation for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.10155v2](http://arxiv.org/pdf/2507.10155v2)**

> **作者:** Khouloud Saadi; Di Wang
>
> **摘要:** Knowledge distillation (KD) has become a cornerstone for compressing large language models (LLMs). However, existing LLM-KD methods have primarily focused on logit-based approaches, which achieve good performance but overlook the rich internal representations of LLMs. Feature-level KD could leverage this structure to provide complementary benefits, yet it remains underexplored because current feature-KD approaches typically assume identical teacher-student hidden sizes, a restrictive and unrealistic assumption. A common workaround is to train a linear projector to align their feature spaces; however, this introduces additional parameters, distorts teacher embeddings, and often degrades downstream performance, especially in generative tasks. We propose Flex-KD, a parameter-free framework for task-driven feature distillation for LLMs. Instead of projecting the entire teacher representation, Flex-KD uses gradient-based scores to identify the most task-relevant dimensions of the teacher's hidden states and distills only this subspace into the student. This ensures that the student's limited capacity is allocated to informative components, while avoiding projector-induced distortion and extra parameters. Flex-KD integrates seamlessly with existing KD pipelines and supports differing teacher-student hidden sizes. Extensive experiments across both classification and generative tasks, i.e., instruction-following and summarization, show that Flex-KD consistently boosts student performance, achieving up to a 3.75 percent performance gain over the linear projection baseline.
>
---
#### [replaced 054] Should I Share this Translation? Evaluating Quality Feedback for User Reliance on Machine Translation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.24683v3](http://arxiv.org/pdf/2505.24683v3)**

> **作者:** Dayeon Ki; Kevin Duh; Marine Carpuat
>
> **备注:** EMNLP 2025
>
> **摘要:** As people increasingly use AI systems in work and daily life, feedback mechanisms that help them use AI responsibly are urgently needed, particularly in settings where users are not equipped to assess the quality of AI predictions. We study a realistic Machine Translation (MT) scenario where monolingual users decide whether to share an MT output, first without and then with quality feedback. We compare four types of quality feedback: explicit feedback that directly give users an assessment of translation quality using (1) error highlights and (2) LLM explanations, and implicit feedback that helps users compare MT inputs and outputs through (3) backtranslation and (4) question-answer (QA) tables. We find that all feedback types, except error highlights, significantly improve both decision accuracy and appropriate reliance. Notably, implicit feedback, especially QA tables, yields significantly greater gains than explicit feedback in terms of decision accuracy, appropriate reliance, and user perceptions, receiving the highest ratings for helpfulness and trust, and the lowest for mental burden.
>
---
#### [replaced 055] Synergizing LLMs and Knowledge Graphs: A Novel Approach to Software Repository-Related Question Answering
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.03815v2](http://arxiv.org/pdf/2412.03815v2)**

> **作者:** Samuel Abedu; SayedHassan Khatoonabadi; Emad Shihab
>
> **备注:** Submitted to ACM Transactions on Software Engineering and Methodology for review
>
> **摘要:** Software repositories contain valuable information for understanding the development process. However, extracting insights from repository data is time-consuming and requires technical expertise. While software engineering chatbots support natural language interactions with repositories, chatbots struggle to understand questions beyond their trained intents and to accurately retrieve the relevant data. This study aims to improve the accuracy of LLM-based chatbots in answering repository-related questions by augmenting them with knowledge graphs. We use a two-step approach: constructing a knowledge graph from repository data, and synergizing the knowledge graph with an LLM to handle natural language questions and answers. We curated 150 questions of varying complexity and evaluated the approach on five popular open-source projects. Our initial results revealed the limitations of the approach, with most errors due to the reasoning ability of the LLM. We therefore applied few-shot chain-of-thought prompting, which improved accuracy to 84%. We also compared against baselines (MSRBot and GPT-4o-search-preview), and our approach performed significantly better. In a task-based user study with 20 participants, users completed more tasks correctly and in less time with our approach, and they reported that it was useful. Our findings demonstrate that LLMs and knowledge graphs are a viable solution for making repository data accessible.
>
---
#### [replaced 056] Do LLMs Really Forget? Evaluating Unlearning with Knowledge Correlation and Confidence Awareness
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.05735v3](http://arxiv.org/pdf/2506.05735v3)**

> **作者:** Rongzhe Wei; Peizhi Niu; Hans Hao-Hsun Hsu; Ruihan Wu; Haoteng Yin; Mohsen Ghassemi; Yifan Li; Vamsi K. Potluru; Eli Chien; Kamalika Chaudhuri; Olgica Milenkovic; Pan Li
>
> **摘要:** Machine unlearning techniques aim to mitigate unintended memorization in large language models (LLMs). However, existing approaches predominantly focus on the explicit removal of isolated facts, often overlooking latent inferential dependencies and the non-deterministic nature of knowledge within LLMs. Consequently, facts presumed forgotten may persist implicitly through correlated information. To address these challenges, we propose a knowledge unlearning evaluation framework that more accurately captures the implicit structure of real-world knowledge by representing relevant factual contexts as knowledge graphs with associated confidence scores. We further develop an inference-based evaluation protocol leveraging powerful LLMs as judges; these judges reason over the extracted knowledge subgraph to determine unlearning success. Our LLM judges utilize carefully designed prompts and are calibrated against human evaluations to ensure their trustworthiness and stability. Extensive experiments on our newly constructed benchmark demonstrate that our framework provides a more realistic and rigorous assessment of unlearning performance. Moreover, our findings reveal that current evaluation strategies tend to overestimate unlearning effectiveness. Our code is publicly available at https://github.com/Graph-COM/Knowledge_Unlearning.git.
>
---
#### [replaced 057] On Code-Induced Reasoning in LLMs
- **分类: cs.CL; cs.PL**

- **链接: [http://arxiv.org/pdf/2509.21499v2](http://arxiv.org/pdf/2509.21499v2)**

> **作者:** Abdul Waheed; Zhen Wu; Carolyn Rosé; Daphne Ippolito
>
> **摘要:** Code data has been shown to enhance the reasoning capabilities of large language models (LLMs), but it remains unclear which aspects of code are most responsible. We investigate this question with a systematic, data-centric framework. We construct parallel instruction datasets in ten programming languages and apply controlled perturbations that selectively disrupt structural or semantic properties of code. We then finetune LLMs from five model families and eight scales on each variant and evaluate their performance on natural language, math, and code tasks. Across 3,331 experiments, our results show that LLMs are more vulnerable to structural perturbations than semantic ones, particularly on math and code tasks. Appropriate abstractions like pseudocode and flowcharts can be as effective as code, while encoding the same information with fewer tokens without adhering to original syntax can often retain or even improve performance. Remarkably, even corrupted code with misleading signals remains competitive when surface-level regularities persist. Finally, syntactic styles also shape task-specific gains with Python favoring natural language reasoning and lower-level languages such as Java and Rust favoring math. Through our systematic framework, we aim to provide insight into how different properties of code influence reasoning and inform the design of training data for enhancing LLM reasoning capabilities.
>
---
#### [replaced 058] Push the Limit of Multi-modal Emotion Recognition by Prompting LLMs with Receptive-Field-Aware Attention Weighting
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.17674v2](http://arxiv.org/pdf/2411.17674v2)**

> **作者:** Han Zhang; Yu Lu; Liyun Zhang; Dian Ding; Dinghua Zhao; Yi-Chao Chen; Ye Wu; Guangtao Xue
>
> **摘要:** Understanding the emotions in a dialogue usually requires external knowledge to accurately understand the contents. As the LLMs become more and more powerful, we do not want to settle on the limited ability of the pre-trained language model. However, the LLMs either can only process text modality or are too expensive to process the multimedia information. We aim to utilize both the power of LLMs and the supplementary features from the multimedia modalities. In this paper, we present a framework, Lantern, that can improve the performance of a certain vanilla model by prompting large language models with receptive-field-aware attention weighting. This framework trained a multi-task vanilla model to produce probabilities of emotion classes and dimension scores. These predictions are fed into the LLMs as references to adjust the predicted probabilities of each emotion class with its external knowledge and contextual understanding. We slice the dialogue into different receptive fields, and each sample is included in exactly t receptive fields. Finally, the predictions of LLMs are merged with a receptive-field-aware attention-driven weighting module. In the experiments, vanilla models CORECT and SDT are deployed in Lantern with GPT-4 or Llama-3.1-405B. The experiments in IEMOCAP with 4-way and 6-way settings demonstrated that the Lantern can significantly improve the performance of current vanilla models by up to 1.23% and 1.80%.
>
---
#### [replaced 059] Tenyidie Syllabification corpus creation and deep learning applications
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.00629v2](http://arxiv.org/pdf/2510.00629v2)**

> **作者:** Teisovi Angami; Kevisino Khate
>
> **备注:** 17 pages
>
> **摘要:** The Tenyidie language is a low-resource language of the Tibeto-Burman family spoken by the Tenyimia Community of Nagaland in the north-eastern part of India and is considered a major language in Nagaland. It is tonal, Subject-Object-Verb, and highly agglutinative in nature. Being a low-resource language, very limited research on Natural Language Processing (NLP) has been conducted. To the best of our knowledge, no work on syllabification has been reported for this language. Among the many NLP tasks, syllabification or syllabication is an important task in which the given word syllables are identified. The contribution of this work is the creation of 10,120 syllabified Tenyidie words and the application of the Deep Learning techniques on the created corpus. In this paper, we have applied LSTM, BLSTM, BLSTM+CRF, and Encoder-decoder deep learning architectures on our created dataset. In our dataset split of 80:10:10 (train:validation:test) set, we achieved the highest accuracy of 99.21% with BLSTM model on the test set. This work will find its application in numerous other NLP applications, such as morphological analysis, part-of-speech tagging, machine translation, etc, for the Tenyidie Language. Keywords: Tenyidie; NLP; syllabification; deep learning; LSTM; BLSTM; CRF; Encoder-decoder
>
---
#### [replaced 060] Investigating ReLoRA: Effects on the Learning Dynamics of Small Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.12960v2](http://arxiv.org/pdf/2509.12960v2)**

> **作者:** Yuval Weiss; David Demitri Africa; Paula Buttery; Richard Diehl Martinez
>
> **备注:** 12 Pages, 6 Tables, 8 Figures
>
> **摘要:** Parameter-efficient methods like LoRA have revolutionised large language model (LLM) fine-tuning. ReLoRA extends this idea to pretraining by repeatedly merging and reinitialising low-rank adapters, increasing cumulative rank while keeping updates cheap. This aligns well with observations that high-capacity models learn through locally low-rank trajectories that expand over time. By contrast, recent work suggests that small language models (SLMs) exhibit rank deficiencies and under-utilise their available dimensionality. This raises a natural question: can ReLoRA's rank-expanding update rule \textit{steer} SLMs toward healthier learning dynamics, mitigating rank bottlenecks in a capacity-constrained regime? We argue SLMs are an ideal testbed: they train quickly, enable controlled ablations, and make rank phenomena more measurable. We present the first systematic study of ReLoRA in SLMs (11M-66M parameters), evaluating both performance and learning dynamics. Across loss, Paloma perplexity, and BLiMP, we find that ReLoRA underperforms full-rank training, with gaps widening at larger scales. Analysis of proportional effective rank and condition numbers shows that ReLoRA amplifies existing rank deficiencies and induces ill-conditioned updates early in training. Our results suggest that while ReLoRA's merge-and-restart strategy can expand ranks in larger models, it does not straightforwardly translate to capacity-limited SLMs, motivating adaptive-rank or hybrid-rank approaches for low-compute pretraining.
>
---
#### [replaced 061] Mechanistic Interpretability as Statistical Estimation: A Variance Analysis of EAP-IG
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.00845v2](http://arxiv.org/pdf/2510.00845v2)**

> **作者:** Maxime Méloux; François Portet; Maxime Peyrard
>
> **摘要:** The development of trustworthy artificial intelligence requires moving beyond black-box performance metrics toward an understanding of models' internal computations. Mechanistic Interpretability (MI) aims to meet this need by identifying the algorithmic mechanisms underlying model behaviors. Yet, the scientific rigor of MI critically depends on the reliability of its findings. In this work, we argue that interpretability methods, such as circuit discovery, should be viewed as statistical estimators, subject to questions of variance and robustness. To illustrate this statistical framing, we present a systematic stability analysis of a state-of-the-art circuit discovery method: EAP-IG. We evaluate its variance and robustness through a comprehensive suite of controlled perturbations, including input resampling, prompt paraphrasing, hyperparameter variation, and injected noise within the causal analysis itself. Across a diverse set of models and tasks, our results demonstrate that EAP-IG exhibits high structural variance and sensitivity to hyperparameters, questioning the stability of its findings. Based on these results, we offer a set of best-practice recommendations for the field, advocating for the routine reporting of stability metrics to promote a more rigorous and statistically grounded science of interpretability.
>
---
#### [replaced 062] Differential Information Distribution: A Bayesian Perspective on Direct Preference Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23761v2](http://arxiv.org/pdf/2505.23761v2)**

> **作者:** Yunjae Won; Hyunji Lee; Hyeonbin Hwang; Minjoon Seo
>
> **备注:** Preprint, under review. 39 pages, 12 figures. Updates from v1: Added new theoretical results on DPO training dynamics and policy exploration, included experiments with Qwen3-4B, and refined the discussion of log-margin dynamics
>
> **摘要:** Direct Preference Optimization (DPO) has been widely used for aligning language models with human preferences in a supervised manner. However, several key questions remain unresolved: the rationale behind its log-ratio reward, how the statistical structure of preference datasets shapes its training dynamics, and how those dynamics impact downstream capabilities. We approach these questions from a Bayesian perspective, interpreting the goal of preference optimization as learning the differential information required to update a reference policy into a target policy. To formalize this view, we introduce the Differential Information Distribution (DID), defined as the distribution over samples that carry the Bayesian evidence required to update policies. We introduce three complementary insights by viewing preference optimization through the DID. First, we find that DPO's log-ratio reward is uniquely justified when preferences encode the Differential Information needed to update a reference policy into the target policy. Second, we discuss how commonly observed training dynamics in DPO, including changes in log-likelihood and policy exploration, stem from a power-law DID relationship. Finally, we analyze how training dynamics influence downstream performance using the entropy of DID, a principled measure of uncertainty in the learned information. We observe that learning high-entropy DID improves open-ended instruction-following, while low-entropy DID benefits knowledge-intensive QA. Taken together, our results show that DPO's reward design, training dynamics, and downstream capabilities all emerge as natural consequences of learning Differential Information, offering both a principled theoretical foundation and practical guidance for preference-based alignment.
>
---
#### [replaced 063] The Hidden Costs of Translation Accuracy: Distillation, Quantization, and Environmental Impact
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.23990v2](http://arxiv.org/pdf/2509.23990v2)**

> **作者:** Dhaathri Vijay; Anandaswarup Vadapalli
>
> **摘要:** The rapid expansion of large language models (LLMs) has heightened concerns about their computational and environmental costs. This study investigates the trade-offs between translation quality and efficiency by comparing full-scale, distilled, and quantized models using machine translation as a case study. We evaluated performance on the Flores+ benchmark and through human judgments of conversational translations in French, Hindi, and Kannada. Our analysis revealed that the full 3.3B FP32 model, while achieving the highest BLEU scores, incurred the largest environmental footprint (~ 0.007-0.008 kg CO2 per run). The distilled 600M FP32 model reduced inference time by 71-78% and carbon emissions by 63-65% compared with the full model, with only minimal reductions in BLEU scores. Human evaluations further showed that even aggressive quantization (INT4) preserved high levels of accuracy and fluency, with differences between models generally minor. These findings demonstrate that model compression strategies can substantially reduce computational demands and environmental impact while maintaining competitive translation quality, though trade-offs are more pronounced in low-resource settings. We argue for evaluation frameworks that integrate efficiency and sustainability alongside accuracy as central dimensions of progress in NLP.
>
---
#### [replaced 064] DrKGC: Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion across General and Biomedical Domains
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.00708v2](http://arxiv.org/pdf/2506.00708v2)**

> **作者:** Yongkang Xiao; Sinian Zhang; Yi Dai; Huixue Zhou; Jue Hou; Jie Ding; Rui Zhang
>
> **备注:** Accepted at EMNLP 2025 Findings
>
> **摘要:** Knowledge graph completion (KGC) aims to predict missing triples in knowledge graphs (KGs) by leveraging existing triples and textual information. Recently, generative large language models (LLMs) have been increasingly employed for graph tasks. However, current approaches typically encode graph context in textual form, which fails to fully exploit the potential of LLMs for perceiving and reasoning about graph structures. To address this limitation, we propose DrKGC (Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion). DrKGC employs a flexible lightweight model training strategy to learn structural embeddings and logical rules within the KG. It then leverages a novel bottom-up graph retrieval method to extract a subgraph for each query guided by the learned rules. Finally, a graph convolutional network (GCN) adapter uses the retrieved subgraph to enhance the structural embeddings, which are then integrated into the prompt for effective LLM fine-tuning. Experimental results on two general domain benchmark datasets and two biomedical datasets demonstrate the superior performance of DrKGC. Furthermore, a realistic case study in the biomedical domain highlights its interpretability and practical utility.
>
---
#### [replaced 065] Initialization using Update Approximation is a Silver Bullet for Extremely Efficient Low-Rank Fine-Tuning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.19557v4](http://arxiv.org/pdf/2411.19557v4)**

> **作者:** Kaustubh Ponkshe; Raghav Singhal; Eduard Gorbunov; Alexey Tumanov; Samuel Horvath; Praneeth Vepakomma
>
> **备注:** Kaustubh Ponkshe and Raghav Singhal contributed equally to this work
>
> **摘要:** Low-rank adapters have become standard for efficiently fine-tuning large language models, but they often fall short of achieving the performance of full fine-tuning. We propose a method, LoRA Silver Bullet or LoRA-SB, that approximates full fine-tuning within low-rank subspaces using a carefully designed initialization strategy. We theoretically demonstrate that the architecture of LoRA-XS, which inserts a learnable r x r matrix between B and A while keeping other matrices fixed, provides the precise conditions needed for this approximation. We leverage its constrained update space to achieve optimal scaling for high-rank gradient updates while removing the need for scaling factor tuning. We prove that our initialization offers an optimal low-rank approximation of the initial gradient and preserves update directions throughout training. Extensive experiments across mathematical reasoning, commonsense reasoning, and language understanding tasks demonstrate that our approach exceeds the performance of LoRA (and baselines) while using 27-90 times fewer learnable parameters, and comprehensively outperforms LoRA-XS. Our findings establish that it is possible to simulate full fine-tuning in low-rank subspaces, and achieve significant parameter efficiency gains without sacrificing performance. Our code is publicly available at: https://github.com/CERT-Lab/lora-sb.
>
---
#### [replaced 066] AlgoTune: Can Language Models Speed Up General-Purpose Numerical Programs?
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.15887v3](http://arxiv.org/pdf/2507.15887v3)**

> **作者:** Ori Press; Brandon Amos; Haoyu Zhao; Yikai Wu; Samuel K. Ainsworth; Dominik Krupke; Patrick Kidger; Touqir Sajed; Bartolomeo Stellato; Jisun Park; Nathanael Bosch; Eli Meril; Albert Steppi; Arman Zharmagambetov; Fangzhao Zhang; David Perez-Pineiro; Alberto Mercurio; Ni Zhan; Talor Abramovich; Kilian Lieret; Hanlin Zhang; Shirley Huang; Matthias Bethge; Ofir Press
>
> **摘要:** Despite progress in language model (LM) capabilities, evaluations have thus far focused on models' performance on tasks that humans have previously solved, including in programming (Jimenez et al., 2024) and mathematics (Glazer et al., 2024). We therefore propose testing models' ability to design and implement algorithms in an open-ended benchmark: We task LMs with writing code that efficiently solves computationally challenging problems in computer science, physics, and mathematics. Our AlgoTune benchmark consists of 154 coding tasks collected from domain experts and a framework for validating and timing LM-synthesized solution code, which is compared to reference implementations from popular open-source packages. In addition, we develop a baseline LM agent, AlgoTuner, and evaluate its performance across a suite of frontier models. AlgoTuner uses a simple, budgeted loop that edits code, compiles and runs it, profiles performance, verifies correctness on tests, and selects the fastest valid version. AlgoTuner achieves an average 1.72x speedup against our reference solvers, which use libraries such as SciPy, sk-learn and CVXPY. However, we find that current models fail to discover algorithmic innovations, instead preferring surface-level optimizations. We hope that AlgoTune catalyzes the development of LM agents exhibiting creative problem solving beyond state-of-the-art human performance.
>
---
#### [replaced 067] Linguistic Nepotism: Trading-off Quality for Language Preference in Multilingual RAG
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.13930v2](http://arxiv.org/pdf/2509.13930v2)**

> **作者:** Dayeon Ki; Marine Carpuat; Paul McNamee; Daniel Khashabi; Eugene Yang; Dawn Lawrie; Kevin Duh
>
> **备注:** 33 pages, 20 figures
>
> **摘要:** Multilingual Retrieval-Augmented Generation (mRAG) systems enable language models to answer knowledge-intensive queries with citation-supported responses across languages. While such systems have been proposed, an open questions is whether the mixture of different document languages impacts generation and citation in unintended ways. To investigate, we introduce a controlled methodology using model internals to measure language preference while holding other factors such as document relevance constant. Across eight languages and six open-weight models, we find that models preferentially cite English sources when queries are in English, with this bias amplified for lower-resource languages and for documents positioned mid-context. Crucially, we find that models sometimes trade-off document relevance for language preference, indicating that citation choices are not always driven by informativeness alone. Our findings shed light on how language models leverage multilingual context and influence citation behavior.
>
---
#### [replaced 068] The AI Productivity Index (APEX)
- **分类: econ.GN; cs.AI; cs.CL; cs.HC; q-fin.EC**

- **链接: [http://arxiv.org/pdf/2509.25721v2](http://arxiv.org/pdf/2509.25721v2)**

> **作者:** Bertie Vidgen; Abby Fennelly; Evan Pinnix; Chirag Mahapatra; Zach Richards; Austin Bridges; Calix Huang; Ben Hunsberger; Fez Zafar; Brendan Foody; Dominic Barton; Cass R. Sunstein; Eric Topol; Osvald Nitski
>
> **摘要:** We introduce the first version of the AI Productivity Index (APEX), a benchmark for assessing whether frontier AI models can perform knowledge work with high economic value. APEX addresses one of the largest inefficiencies in AI research: outside of coding, benchmarks often fail to test economically relevant capabilities. APEX-v1.0 contains 200 test cases and covers four domains: investment banking, management consulting, law, and primary medical care. It was built in three steps. First, we sourced experts with top-tier experience e.g., investment bankers from Goldman Sachs. Second, experts created prompts that reflect high-value tasks in their day-to-day work. Third, experts created rubrics for evaluating model responses. We evaluate 23 frontier models on APEX-v1.0 using an LM judge. GPT 5 (Thinking = High) achieves the highest mean score (64.2%), followed by Grok 4 (61.3%) and Gemini 2.5 Flash (Thinking = On) (60.4%). Qwen 3 235B is the best performing open-source model and seventh best overall. There is a large gap between the performance of even the best models and human experts, highlighting the need for better measurement of models' ability to produce economically valuable work.
>
---
#### [replaced 069] What if I ask in \textit{alia lingua}? Measuring Functional Similarity Across Languages
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.04032v2](http://arxiv.org/pdf/2509.04032v2)**

> **作者:** Debangan Mishra; Arihant Rastogi; Agyeya Negi; Shashwat Goel; Ponnurangam Kumaraguru
>
> **备注:** Accepted into Multilingual Representation Learning (MRL) Workshop at EMNLP 2025
>
> **摘要:** How similar are model outputs across languages? In this work, we study this question using a recently proposed model similarity metric $\kappa_p$ applied to 20 languages and 47 subjects in GlobalMMLU. Our analysis reveals that a model's responses become increasingly consistent across languages as its size and capability grow. Interestingly, models exhibit greater cross-lingual consistency within themselves than agreement with other models prompted in the same language. These results highlight not only the value of $\kappa_p$ as a practical tool for evaluating multilingual reliability, but also its potential to guide the development of more consistent multilingual systems.
>
---
#### [replaced 070] No Language Data Left Behind: A Comparative Study of CJK Language Datasets in the Hugging Face Ecosystem
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.04329v2](http://arxiv.org/pdf/2507.04329v2)**

> **作者:** Dasol Choi; Woomyoung Park; Youngsook Song
>
> **备注:** Accepted to the MRL Workshop at EMNLP 2025
>
> **摘要:** Recent advances in Natural Language Processing (NLP) have underscored the crucial role of high-quality datasets in building large language models (LLMs). However, while extensive resources and analyses exist for English, the landscape for East Asian languages - particularly Chinese, Japanese, and Korean (CJK) - remains fragmented and underexplored, despite these languages together serving over 1.6 billion speakers. To address this gap, we investigate the HuggingFace ecosystem from a cross-linguistic perspective, focusing on how cultural norms, research environments, and institutional practices shape dataset availability and quality. Drawing on more than 3,300 datasets, we employ quantitative and qualitative methods to examine how these factors drive distinct creation and curation patterns across Chinese, Japanese, and Korean NLP communities. Our findings highlight the large-scale and often institution-driven nature of Chinese datasets, grassroots community-led development in Korean NLP, and an entertainment- and subculture-focused emphasis on Japanese collections. By uncovering these patterns, we reveal practical strategies for enhancing dataset documentation, licensing clarity, and cross-lingual resource sharing - ultimately guiding more effective and culturally attuned LLM development in East Asia. We conclude by discussing best practices for future dataset curation and collaboration, aiming to strengthen resource development across all three languages.
>
---
#### [replaced 071] Reasoning over User Preferences: Knowledge Graph-Augmented LLMs for Explainable Conversational Recommendations
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2411.14459v2](http://arxiv.org/pdf/2411.14459v2)**

> **作者:** Zhangchi Qiu; Linhao Luo; Shirui Pan; Alan Wee-Chung Liew
>
> **备注:** Accepted by ICDM 2025
>
> **摘要:** Conversational Recommender Systems (CRSs) aim to provide personalized recommendations by capturing user preferences through interactive dialogues. Explainability in CRSs is crucial as it enables users to understand the reasoning behind recommendations, increasing system transparency and trustworthiness. However, current CRSs often leverage knowledge graphs (KGs) or language models to extract and represent user preferences as latent vectors, which limits their explainability. Large language models (LLMs) offer powerful reasoning capabilities that can bridge this gap by generating human-understandable preference summaries. However, effectively reasoning over user preferences in CRSs remains challenging as LLMs pre-trained on large-scale corpora may not be well-suited for analyzing user preferences. While KGs provide rich domain knowledge, integrating them with LLMs encounters a significant modality gap between structured KG information and unstructured conversations. In this paper, we propose COMPASS, a plug-and-play framework that synergizes LLMs and KGs to reason over user preferences, enhancing the performance and explainability of existing CRSs. COMPASS employs a two-stage training approach: first, it bridges the gap between the structured KG and natural language through novel graph entity captioning pre-training. Next, COMPASS optimizes user preference reasoning via knowledge-aware instruction fine-tuning, where the LLM learns to reason and summarize user preferences from dialogue histories and KG-augmented context. This enables COMPASS to perform knowledge-aware reasoning and generate interpretable user preferences that can seamlessly integrate with existing CRS models for improving recommendation performance and explainability. Our experiments on benchmark datasets demonstrate the effectiveness of COMPASS in improving various CRS models.
>
---

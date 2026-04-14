# 自然语言处理 cs.CL

- **最新发布 209 篇**

- **更新 157 篇**

## 最新发布

#### [new 001] C-ReD: A Comprehensive Chinese Benchmark for AI-Generated Text Detection Derived from Real-World Prompts
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI生成文本检测任务，旨在解决中文语境下检测模型多样性不足和数据单一的问题。研究构建了C-ReD基准，提升检测效果与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.11796](https://arxiv.org/pdf/2604.11796)**

> **作者:** Chenxi Qing; Junxi Wu; Zheng Liu; Yixiang Qiu; Hongyao Yu; Bin Chen; Hao Wu; Shu-Tao Xia
>
> **摘要:** Recently, large language models (LLMs) are capable of generating highly fluent textual content. While they offer significant convenience to humans, they also introduce various risks, like phishing and academic dishonesty. Numerous research efforts have been dedicated to developing algorithms for detecting AI-generated text and constructing relevant datasets. However, in the domain of Chinese corpora, challenges remain, including limited model diversity and data homogeneity. To address these issues, we propose C-ReD: a comprehensive Chinese Real-prompt AI-generated Detection benchmark. Experiments demonstrate that C-ReD not only enables reliable in-domain detection but also supports strong generalization to unseen LLMs and external Chinese datasets-addressing critical gaps in model diversity, domain coverage, and prompt realism that have limited prior Chinese detection benchmarks. We release our resources at this https URL.
>
---
#### [new 002] Too Nice to Tell the Truth: Quantifying Agreeableness-Driven Sycophancy in Role-Playing Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究角色扮演语言模型中亲和性与阿谀行为的关系，属于AI安全与对齐任务。通过实验分析，发现亲和性高的角色更易产生阿谀行为，为提升AI系统可靠性提供依据。**

- **链接: [https://arxiv.org/pdf/2604.10733](https://arxiv.org/pdf/2604.10733)**

> **作者:** Arya Shah; Deepali Mishra; Chaklam Silpasuwanchai
>
> **备注:** 14 Pages, 5 Figures, 9 Tables, ACL Main Conference 2026
>
> **摘要:** Large language models increasingly serve as conversational agents that adopt personas and role-play characters at user request. This capability, while valuable, raises concerns about sycophancy: the tendency to provide responses that validate users rather than prioritize factual accuracy. While prior work has established that sycophancy poses risks to AI safety and alignment, the relationship between specific personality traits of adopted personas and the degree of sycophantic behavior remains unexplored. We present a systematic investigation of how persona agreeableness influences sycophancy across 13 small, open-weight language models ranging from 0.6B to 20B parameters. We develop a benchmark comprising 275 personas evaluated on NEO-IPIP agreeableness subscales and expose each persona to 4,950 sycophancy-eliciting prompts spanning 33 topic categories. Our analysis reveals that 9 of 13 models exhibit statistically significant positive correlations between persona agreeableness and sycophancy rates, with Pearson correlations reaching $r = 0.87$ and effect sizes as large as Cohen's $d = 2.33$. These findings demonstrate that agreeableness functions as a reliable predictor of persona-induced sycophancy, with direct implications for the deployment of role-playing AI systems and the development of alignment strategies that account for personality-mediated deceptive behaviors.
>
---
#### [new 003] Evaluating Memory Capability in Continuous Lifelog Scenario
- **分类: cs.CL**

- **简介: 该论文属于记忆能力评估任务，旨在解决真实场景下记忆系统评估不足的问题。通过构建新基准LifeDialBench并提出在线评估协议，发现现有系统不如简单基线有效。**

- **链接: [https://arxiv.org/pdf/2604.11182](https://arxiv.org/pdf/2604.11182)**

> **作者:** Jianjie Zheng; Zhichen Liu; Zhanyu Shen; Jingxiang Qu; Guanhua Chen; Yile Wang; Yang Xu; Yang Liu; Sijie Cheng
>
> **备注:** 27 pages, 7 figures. ACL 2026 Findings camera-ready
>
> **摘要:** Nowadays, wearable devices can continuously lifelog ambient conversations, creating substantial opportunities for memory systems. However, existing benchmarks primarily focus on online one-on-one chatting or human-AI interactions, thus neglecting the unique demands of real-world scenarios. Given the scarcity of public lifelogging audio datasets, we propose a hierarchical synthesis framework to curate \textbf{\textsc{LifeDialBench}}, a novel benchmark comprising two complementary subsets: \textbf{EgoMem}, built on real-world egocentric videos, and \textbf{LifeMem}, constructed using simulated virtual community. Crucially, to address the issue of temporal leakage in traditional offline settings, we propose an \textbf{Online Evaluation} protocol that strictly adheres to temporal causality, ensuring systems are evaluated in a realistic streaming fashion. Our experimental results reveal a counterintuitive finding: current sophisticated memory systems fail to outperform a simple RAG-based baseline. This highlights the detrimental impact of over-designed structures and lossy compression in current approaches, emphasizing the necessity of high-fidelity context preservation for lifelog scenarios. We release our code and data at this https URL.
>
---
#### [new 004] Time is Not a Label: Continuous Phase Rotation for Temporal Knowledge Graphs and Agentic Memory
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱任务，解决时间建模问题。提出RoMem模块，通过连续相位旋转区分持久与演变事实，提升时序知识图谱性能。**

- **链接: [https://arxiv.org/pdf/2604.11544](https://arxiv.org/pdf/2604.11544)**

> **作者:** Weixian Waylon Li; Jiaxin Zhang; Xianan Jim Yang; Tiejun Ma; Yiwen Guo
>
> **摘要:** Structured memory representations such as knowledge graphs are central to autonomous agents and other long-lived systems. However, most existing approaches model time as discrete metadata, either sorting by recency (burying old-yet-permanent knowledge), simply overwriting outdated facts, or requiring an expensive LLM call at every ingestion step, leaving them unable to distinguish persistent facts from evolving ones. To address this, we introduce RoMem, a drop-in temporal knowledge graph module for structured memory systems, applicable to agentic memory and beyond. A pretrained Semantic Speed Gate maps each relation's text embedding to a volatility score, learning from data that evolving relations (e.g., "president of") should rotate fast while persistent ones (e.g., "born in") should remain stable. Combined with continuous phase rotation, this enables geometric shadowing: obsolete facts are rotated out of phase in complex vector space, so temporally correct facts naturally outrank contradictions without deletion. On temporal knowledge graph completion, RoMem achieves state-of-the-art results on ICEWS05-15 (72.6 MRR). Applied to agentic memory, it delivers 2-3x MRR and answer accuracy on temporal reasoning (MultiTQ), dominates hybrid benchmark (LoCoMo), preserves static memory with zero degradation (DMR-MSC), and generalises zero-shot to unseen financial domains (FinTMMBench).
>
---
#### [new 005] Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究非自回归扩散语言模型的解码问题，针对初始选择对生成轨迹的影响进行改进，提出一种轻量级方法提升推理与规划任务效果。**

- **链接: [https://arxiv.org/pdf/2604.10567](https://arxiv.org/pdf/2604.10567)**

> **作者:** Jiyeon Kim; Sungik Choi; Yongrae Jo; Moontae Lee; Minjoon Seo
>
> **摘要:** Diffusion-based language models (dLLMs) have emerged as a promising alternative to autoregressive language models, offering the potential for parallel token generation and bidirectional context modeling. However, harnessing this flexibility for fully non-autoregressive decoding remains an open question, particularly for reasoning and planning tasks. In this work, we investigate non-autoregressive decoding in dLLMs by systematically analyzing its inference dynamics along the temporal axis. Specifically, we uncover an inherent failure mode in confidence-based non-autoregressive generation stemming from a strong proximity bias-the tendency for the denoising order to concentrate on spatially adjacent tokens. This local dependency leads to spatial error propagation, rendering the entire trajectory critically contingent on the initial unmasking position. Leveraging this insight, we present a minimal-intervention approach that guides early token selection, employing a lightweight planner and end-of-sequence temperature annealing. We thoroughly evaluate our method on various reasoning and planning tasks and observe substantial overall improvement over existing heuristic baselines without significant computational overhead.
>
---
#### [new 006] A Systematic Analysis of the Impact of Persona Steering on LLM Capabilities
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究Persona Steering对LLM能力的影响，解决个性化设置与模型性能关系的问题。通过NPTI框架测试六项认知基准，发现人格特征显著影响任务表现，并提出DPR优化策略。**

- **链接: [https://arxiv.org/pdf/2604.11048](https://arxiv.org/pdf/2604.11048)**

> **作者:** Jiaqi Chen; Ming Wang; Tingna Xie; Shi Feng; Yongkang Liu
>
> **摘要:** Imbuing Large Language Models (LLMs) with specific personas is prevalent for tailoring interaction styles, yet the impact on underlying cognitive capabilities remains unexplored. We employ the Neuron-based Personality Trait Induction (NPTI) framework to induce Big Five personality traits in LLMs and evaluate performance across six cognitive benchmarks. Our findings reveal that persona induction produces stable, reproducible shifts in cognitive task performance beyond surface-level stylistic changes. These effects exhibit strong task dependence: certain personalities yield consistent gains on instruction-following, while others impair complex reasoning. Effect magnitude varies systematically by trait dimension, with Openness and Extraversion exerting the most robust influence. Furthermore, LLM effects show 73.68% directional consistency with human personality-cognition relationships. Capitalizing on these regularities, we propose Dynamic Persona Routing (DPR), a lightweight query-adaptive strategy that outperforms the best static persona without additional training.
>
---
#### [new 007] General365: Benchmarking General Reasoning in Large Language Models Across Diverse and Challenging Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的通用推理任务，旨在评估大语言模型在非专业领域的推理能力。研究提出General365基准，解决当前模型在通用场景下表现不足的问题。**

- **链接: [https://arxiv.org/pdf/2604.11778](https://arxiv.org/pdf/2604.11778)**

> **作者:** Junlin Liu; Shengnan An; Shuang Zhou; Dan Ma; Shixiong Luo; Ying Xie; Yuan Zhang; Wenling Yuan; Yifan Zhou; Xiaoyu Li; Ziwen Wang; Xuezhi Cao; Xunliang Cai
>
> **备注:** 17 pages, 9 figures
>
> **摘要:** Contemporary large language models (LLMs) have demonstrated remarkable reasoning capabilities, particularly in specialized domains like mathematics and physics. However, their ability to generalize these reasoning skills to more general and broader contexts--often termed general reasoning--remains under-explored. Unlike domain-specific reasoning, general reasoning relies less on expert knowledge but still presents formidable reasoning challenges, such as complex constraints, nested logical branches, and semantic interference. To address this gap, we introduce General365, a benchmark specifically designed to assess general reasoning in LLMs. By restricting background knowledge to a K-12 level, General365 explicitly decouples reasoning from specialized expertise. The benchmark comprises 365 seed problems and 1,095 variant problems across eight categories, ensuring both high difficulty and diversity. Evaluations across 26 leading LLMs reveal that even the top-performing model achieves only 62.8% accuracy, in stark contrast to the near-perfect performances of LLMs in math and physics benchmarks. These results suggest that the reasoning abilities of current LLMs are heavily domain-dependent, leaving significant room for improvement in broader applications. We envision General365 as a catalyst for advancing LLM reasoning beyond domain-specific tasks toward robust, general-purpose real-world scenarios. Code, Dataset, and Leaderboard: this https URL
>
---
#### [new 008] Deep-Reporter: Deep Research for Grounded Multimodal Long-Form Generation
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出Deep-Reporter，解决多模态长文本生成任务，通过整合文本与视觉信息提升生成内容的准确性和连贯性。**

- **链接: [https://arxiv.org/pdf/2604.10741](https://arxiv.org/pdf/2604.10741)**

> **作者:** Fangda Ye; Zhifei Xie; Yuxin Hu; Yihang Yin; Shurui Huang; Shikai Dong; Jianzhu Bao; Shuicheng Yan
>
> **备注:** 41 pages, 6 figures, 8 tables. Code available at this https URL
>
> **摘要:** Recent agentic search frameworks enable deep research via iterative planning and retrieval, reducing hallucinations and enhancing factual grounding. However, they remain text-centric, overlooking the multimodal evidence that characterizes real-world expert reports. We introduce a pressing task: multimodal long-form generation. Accordingly, we propose Deep-Reporter, a unified agentic framework for grounded multimodal long-form generation. It orchestrates: (i) Agentic Multimodal Search and Filtering to retrieve and filter textual passages and information-dense visuals; (ii) Checklist-Guided Incremental Synthesis to ensure coherent image-text integration and optimal citation placement; and (iii) Recurrent Context Management to balance long-range coherence with local fluency. We develop a rigorous curation pipeline producing 8K high-quality agentic traces for model optimization. We further introduce M2LongBench, a comprehensive testbed comprising 247 research tasks across 9 domains and a stable multimodal sandbox. Extensive experiments demonstrate that long-form multimodal generation is a challenging task, especially in multimodal selection and integration, and effective post-training can bridge the gap.
>
---
#### [new 009] Why Supervised Fine-Tuning Fails to Learn: A Systematic Study of Incomplete Learning in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究监督微调中未完全学习现象（ILP）。通过分析发现ILP的五大原因，并提出诊断框架以提升模型学习效果。**

- **链接: [https://arxiv.org/pdf/2604.10079](https://arxiv.org/pdf/2604.10079)**

> **作者:** Chao Xue; Yao Wang; Mengqiao Liu; Di Liang; Xingsheng Han; Peiyang Liu; Xianjie Wu; Chenyao Lu; Lei Jiang; Yu Lu; Haibo Shi; Shuang Liang; Minlong Peng; Flora D. Salim
>
> **备注:** Accepted by ACL 2026 Oral
>
> **摘要:** Supervised Fine-Tuning (SFT) is the standard approach for adapting large language models (LLMs) to downstream tasks. However, we observe a persistent failure mode: even after convergence, models often fail to correctly reproduce a subset of their own supervised training data. We refer to this behavior as the Incomplete Learning Phenomenon(ILP). This paper presents the first systematic study of ILP in LLM fine-tuning. We formalize ILP as post-training failure to internalize supervised instances and demonstrate its prevalence across multiple model families, domains, and datasets. Through controlled analyses, we identify five recurrent sources of incomplete learning: (1) missing prerequisite knowledge in the pre-trained model, (2) conflicts between SFT supervision and pre-training knowledge, (3) internal inconsistencies within SFT data, (4) left-side forgetting during sequential fine-tuning, and (5) insufficient optimization for rare or complex patterns. We introduce a diagnostic-first framework that maps unlearned samples to these causes using observable training and inference signals, and study several targeted mitigation strategies as causal interventions. Experiments on Qwen, LLaMA, and OLMo2 show that incomplete learning is widespread and heterogeneous, and that improvements in aggregate metrics can mask persistent unlearned subsets. The findings highlight the need for fine-grained diagnosis of what supervised fine-tuning fails to learn, and why.
>
---
#### [new 010] QFS-Composer: Query-focused summarization pipeline for less resourced languages
- **分类: cs.CL**

- **简介: 该论文属于查询聚焦摘要任务，解决低资源语言中摘要与用户意图不一致的问题。提出QFS-Composer框架，结合查询分解、问答和摘要生成，提升摘要准确性。**

- **链接: [https://arxiv.org/pdf/2604.10687](https://arxiv.org/pdf/2604.10687)**

> **作者:** Vuk Đuranović; Marko Robnik Šikonja
>
> **备注:** 12 pages, 3 tables
>
> **摘要:** Large language models (LLMs) demonstrate strong performance in text summarization, yet their effectiveness drops significantly across languages with restricted training resources. This work addresses the challenge of query-focused summarization (QFS) in less-resourced languages, where labeled datasets and evaluation tools are limited. We present a novel QFS framework, QFS-Composer, that integrates query decomposition, question generation (QG), question answering (QA), and abstractive summarization to improve the factual alignment of a summary with user intent. We test our approach on the Slovenian language. To enable high-quality supervision and evaluation, we develop the Slovenian QA and QG models based on a Slovene LLM and adapt evaluation approaches for reference-free summary evaluation. Empirical evaluation shows that the QA-guided summarization pipeline yields improved consistency and relevance over baseline LLMs. Our work establishes an extensible methodology for advancing QFS in less-resourced languages.
>
---
#### [new 011] HTAA: Enhancing LLM Planning via Hybrid Toolset Agentization & Adaptation
- **分类: cs.CL**

- **简介: 该论文属于大语言模型工具使用任务，解决工具调用效率低和错误累积问题，提出HTAA框架通过分层规划和代理适配提升性能。**

- **链接: [https://arxiv.org/pdf/2604.10917](https://arxiv.org/pdf/2604.10917)**

> **作者:** Chengrui Huang; Junshuo Zhang; Zhiyuan Ma; Xikun Wang; Ximeng Wang; Menghua Jiang; Gang Zeng; Zhaobing Han; Shen Gao; Shuo Shang
>
> **备注:** 22 pages, 3 figures
>
> **摘要:** Enabling large language models to scale and reliably use hundreds of tools is critical for real-world applications, yet challenging due to the inefficiency and error accumulation inherent in flat tool-calling architectures. To address this, we propose Hybrid Toolset Agentization & Adaptation (HTAA), a hierarchical framework for scalable tool-use planning. We propose a novel toolset agentization paradigm, which encapsulates frequently co-used tools into specialized agent tools, thereby reducing the planner's action space and mitigating redundancy. To ensure effective coordination, we design Asymmetric Planner Adaptation, a trajectory-based training paradigm that aligns the high-level planner with agent tools via backward reconstruction and forward refinement. To validate the performance of HTAA, we conduct experiments on a real-world internal dataset, InfoVerify, based on the POI validation workflow of China's largest online large-scale ride-hailing platform, featuring long-horizon executable tool trajectories. Experiments on InfoVerify and widely-used benchmarks show that HTAA consistently achieves higher task success rates, requires short tool calling trajectories, and significantly reduces context overhead compared to strong baselines. Furthermore, in a production deployment, HTAA substantially reduces manual validation effort and operational cost, demonstrating its practical efficacy.
>
---
#### [new 012] Self-Correcting RAG: Enhancing Faithfulness via MMKP Context Selection and NLI-Guided MCTS
- **分类: cs.CL**

- **简介: 该论文属于问答与事实核查任务，旨在解决RAG在复杂推理中的低效信息利用和幻觉问题。通过MMKP上下文选择和NLI引导的MCTS优化生成过程。**

- **链接: [https://arxiv.org/pdf/2604.10734](https://arxiv.org/pdf/2604.10734)**

> **作者:** Shijia Xu; Zhou Wu; Xiaolong Jia; Yu Wang; Kai Liu; April Xiaowen Dong
>
> **摘要:** Retrieval-augmented generation (RAG) substantially extends the knowledge boundary of large language models. However, it still faces two major challenges when handling complex reasoning tasks: low context utilization and frequent hallucinations. To address these issues, we propose Self-Correcting RAG, a unified framework that reformulates retrieval and generation as constrained optimization and path planning. On the input side, we move beyond traditional greedy retrieval and, for the first time, formalize context selection as a multi-dimensional multiple-choice knapsack problem (MMKP), thereby maximizing information density and removing redundancy under a strict token budget. On the output side, we introduce a natural language inference (NLI)-guided Monte Carlo Tree Search (MCTS) mechanism, which leverages test-time compute to dynamically explore reasoning trajectories and validate the faithfulness of generated answers. Experiments on six multi-hop question answering and fact-checking datasets demonstrate that our method significantly improves reasoning accuracy on complex queries while effectively reducing hallucinations, outperforming strong existing this http URL code is available at this https URL .
>
---
#### [new 013] Attention Sinks as Internal Signals for Hallucination Detection in Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于 hallucination 检测任务，旨在解决大语言模型生成内容中事实错误的问题。通过分析注意力图中的“sink”token，提出 SinkProbe 方法提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.10697](https://arxiv.org/pdf/2604.10697)**

> **作者:** Jakub Binkowski; Kamil Adamczewski; Tomasz Kajdanowicz
>
> **摘要:** Large language models frequently exhibit hallucinations: fluent and confident outputs that are factually incorrect or unsupported by the input context. While recent hallucination detection methods have explored various features derived from attention maps, the underlying mechanisms they exploit remain poorly understood. In this work, we propose SinkProbe, a hallucination detection method grounded in the observation that hallucinations are deeply entangled with attention sinks - tokens that accumulate disproportionate attention mass during generation - indicating a transition from distributed, input-grounded attention to compressed, prior-dominated computation. Importantly, although sink scores are computed solely from attention maps, we find that the classifier preferentially relies on sinks whose associated value vectors have large norms. Moreover, we show that previous methods implicitly depend on attention sinks by establishing their mathematical relationship to sink scores. Our findings yield a novel hallucination detection method grounded in theory that produces state-of-the-art results across popular datasets and LLMs.
>
---
#### [new 014] Training-Free Cross-Lingual Dysarthria Severity Assessment via Phonological Subspace Analysis in Self-Supervised Speech Representations
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语音分析任务，旨在无需训练数据即可评估构音障碍严重程度。通过分析语音表示中的音系子空间，提取特征并验证其与临床严重程度的相关性。**

- **链接: [https://arxiv.org/pdf/2604.10123](https://arxiv.org/pdf/2604.10123)**

> **作者:** Bernard Muller; Antonio Armando Ortiz Barrañón; LaVonne Roberts
>
> **备注:** Submitted to PLOS digital health
>
> **摘要:** Dysarthric speech severity assessment typically requires trained clinicians or supervised models built from labelled pathological speech, limiting scalability across languages and clinical settings. We present a training-free method that quantifies dysarthria severity by measuring degradation in phonological feature subspaces within frozen HuBERT representations. No supervised severity model is trained; feature directions are estimated from healthy control speech using a pretrained forced aligner. For each speaker, we extract phone-level embeddings via Montreal Forced Aligner, compute d-prime scores along phonological contrast directions (nasality, voicing, stridency, sonorance, manner, and four vowel features) derived exclusively from healthy controls, and construct a 12-dimensional phonological this http URL 890 speakers across 10 corpora, 5 languages (English, Spanish, Dutch, Mandarin, French), and 3 primary aetiologies (Parkinson's disease, cerebral palsy, ALS), we find that all five consonant d-prime features correlate significantly with clinical severity (random-effects meta-analysis rho = -0.50 to -0.56, p < 2e-4; pooled Spearman rho = -0.47 to -0.55 with bootstrap 95% CIs not crossing zero). The effect replicates within individual corpora, survives FDR correction, and remains robust to leave-one-corpus-out removal and alignment quality controls. Nasality d-prime decreases monotonically from control to severe in 6 of 7 severity-graded corpora. Mann-Whitney U tests confirm that all 12 features distinguish controls from severely dysarthric speakers (p < 0.001).The method requires no dysarthric training data and applies to any language with an existing MFA acoustic model (currently 29 languages). We release the full pipeline and phone feature configurations for six languages.
>
---
#### [new 015] YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents
- **分类: cs.CL**

- **简介: 该论文提出YIELD数据集和评估框架，用于信息获取代理研究。解决传统对话代理无法有效获取信息的问题，通过构建大规模对话数据并定义新指标，提升代理的信息获取能力。**

- **链接: [https://arxiv.org/pdf/2604.10968](https://arxiv.org/pdf/2604.10968)**

> **作者:** Victor De Lima; Grace Hui Yang
>
> **备注:** Accepted at ACL 2026 (Main Conference)
>
> **摘要:** Most conversational agents (CAs) are designed to satisfy user needs through user-driven interactions. However, many real-world settings, such as academic interviewing, judicial proceedings, and journalistic investigations, involve broader institutional decision-making processes and require agents that can elicit information from users. In this paper, we introduce Information Elicitation Agents (IEAs) in which the agent's goal is to elicit information from users to support the agent's institutional or task-oriented objectives. To enable systematic research on this setting, we present YIELD, a 26M-token dataset of 2,281 ethically sourced, human-to-human dialogues. Moreover, we formalize information elicitation as a finite-horizon POMDP and propose novel metrics tailored to IEAs. Pilot experiments on multiple foundation LLMs show that training on YIELD improves their alignment with real elicitation behavior and findings are corroborated by human evaluation. We release YIELD under CC BY 4.0. The dataset, project code, evaluation tools, and fine-tuned model adapters are available at: this https URL.
>
---
#### [new 016] LASQ: A Low-resource Aspect-based Sentiment Quadruple Extraction Dataset
- **分类: cs.CL**

- **简介: 该论文提出LASQ数据集，解决低资源语言中细粒度情感四元组提取问题，设计融合语法知识的模型以应对词法稀疏问题。**

- **链接: [https://arxiv.org/pdf/2604.10417](https://arxiv.org/pdf/2604.10417)**

> **作者:** Aizihaierjiang Yusufu; Jiang Liu; Kamran Aziz; Abidan Ainiwaer; Bobo Li; Fei Li; Donghong Ji; Aizierguli Yusufu
>
> **摘要:** In recent years, aspect-based sentiment analysis (ABSA) has made rapid progress and shown strong practical value. However, existing research and benchmarks are largely concentrated on high-resource languages, leaving fine-grained sentiment extraction in low-resource languages under-explored. To address this gap, we constructed the first Low-resource languages Aspect-based Sentiment Quadruple dataset, named LASQ, which includes two low-resource languages: Uzbek and Uyghur. Secondly, it includes a fine-grained target-aspect-opinion-sentiment quadruple extraction task. To facilitate future research, we designed a grid-tagging model that integrates syntactic knowledge. This model incorporates part-of-speech (POS) and dependency knowledge into the model through our designed Syntax Knowledge Embedding Module (SKEM), thereby alleviating the lexical sparsity problem caused by agglutinative languages. Experiments on LASQ demonstrate consistent gains over competitive baselines, validating both the dataset's utility and the effectiveness of the proposed modeling approach.
>
---
#### [new 017] Uncertainty-Aware Web-Conditioned Scientific Fact-Checking
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于科学事实核查任务，解决技术性声明验证中幻觉和推理不一致的问题。通过原子事实分解与不确定性引导的网络核查，提升核查的准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2604.11036](https://arxiv.org/pdf/2604.11036)**

> **作者:** Ashwin Vinod; Katrin Erk
>
> **摘要:** Scientific fact-checking is vital for assessing claims in specialized domains such as biomedicine and materials science, yet existing systems often hallucinate or apply inconsistent reasoning, especially when verifying technical, compositional claims against an evidence snippet under source and cost/latency constraints. We present a pipeline centered on atomic predicate-argument decomposition and calibrated, uncertainty-gated corroboration: atomic facts are aligned to local snippets via embeddings, verified by a compact evidence-grounded checker, and only facts with uncertain support trigger domain-restricted web search over authoritative sources. The system supports both binary and tri-valued classification where it predicts labels from Supported, Refuted, NEI for three-way tasks. We evaluate under two regimes, Context-Only (no web) and Context+Web (uncertainty-gated web corroboration); when retrieved evidence conflicts with the provided context, we abstain with NEI rather than overriding the context. On multiple benchmarks, our framework surpasses the strongest benchmarks. In our experiments, web corroboration was invoked for only a minority of atomic facts on average, indicating that external evidence is consulted selectively under calibrated uncertainty rather than routinely. Overall, coupling atomic granularity with calibrated, uncertainty-gated corroboration yields more interpretable and context-conditioned verification, making the approach well-suited to high-stakes, single-document settings that demand traceable rationales, predictable cost/latency, and conservative.
>
---
#### [new 018] FAITH: Factuality Alignment through Integrating Trustworthiness and Honestness
- **分类: cs.CL**

- **简介: 该论文提出FAITH框架，解决大语言模型事实性不准确的问题。通过整合信任度和诚实度信号，提升模型事实对齐能力。**

- **链接: [https://arxiv.org/pdf/2604.10189](https://arxiv.org/pdf/2604.10189)**

> **作者:** Xiaoning Dong; Chengyan Wu; Yajie Wen; Yu Chen; Yun Xue; Jing Zhang; Wei Xu; Bolei Ma
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Large Language Models (LLMs) can generate factually inaccurate content even if they have corresponding knowledge, which critically undermines their reliability. Existing approaches attempt to mitigate this by incorporating uncertainty in QA prompt during training, but these numerical scores lack the semantic richness for LLM to properly understand its internal states of trustworthiness and honestness, leading to insufficient factuality alignment. We introduce FAITH (Factuality Alignment through Integrating Trustworthiness and Honestness), a post-training framework for factuality alignment that integrates natural-language uncertainty signals with external knowledge. Specifically, we augment training datasets by computing confidence scores and semantic entropy from LLM outputs and mapping them into a knowledge state quadrant that describes the model's internal knowledge possession (trustworthiness) and answering behaviors (honestness) in natural language. Based on this enhanced data, we design a reward function that considers both correctness and uncertainty signals, and fine-tune the LLM using the Proximal Policy Optimization (PPO) algorithm. To further mitigate weakly grounded responses, we design a retrieval-augmented module that retrieves relevant external passages, improving the consistency between internal and external knowledge representations. Extensive experiments on four knowledge-intensive benchmarks demonstrate that FAITH enhances the factual accuracy and truthfulness of LLMs.
>
---
#### [new 019] Polyglot Teachers: Evaluating Language Models for Multilingual Synthetic Data Generation
- **分类: cs.CL**

- **简介: 该论文属于多语言合成数据生成任务，旨在解决教师模型选择不当导致的数据质量差问题。通过评估不同模型作为教师的效果，提出有效教学策略。**

- **链接: [https://arxiv.org/pdf/2604.11290](https://arxiv.org/pdf/2604.11290)**

> **作者:** Lester James V. Miranda; Ivan Vulić; Anna Korhonen
>
> **摘要:** Synthesizing supervised finetuning (SFT) data from language models (LMs) to teach smaller models multilingual tasks has become increasingly common. However, teacher model selection is often ad hoc, typically defaulting to the largest available option, even though such models may have significant capability gaps in non-English languages. This practice can result in poor-quality synthetic data and suboptimal student downstream performance. In this work, we systematically characterize what makes an effective multilingual teacher. We measure intrinsic measures of data quality with extrinsic student model performance in a metric we call Polyglot Score; evaluating 10 LMs across 6 typologically diverse languages, generating over 1.4M SFT examples and training 240 student models. Among the models tested, Gemma 3 27B and Aya Expanse 32B emerge as consistently effective teachers across different student base model families. Further analyses reveal that model scale alone does not significantly predict teacher effectiveness; instead, data qualities such as prompt diversity, length, and response fluency capture over 93.3% of variance in intrinsic data quality and predict student performance. Finally, we provide practical recommendations, including matching the model families of teacher-student pairs and translating from or responding to existing prompts, which can yield improvements for less-resourced languages. We hope that our work advances data-centric research in multilingual synthetic data and LM development.
>
---
#### [new 020] Agentic Aggregation for Parallel Scaling of Long-Horizon Agentic Tasks
- **分类: cs.CL**

- **简介: 该论文研究长周期代理任务的并行测试时扩展，解决多轨迹聚合信息丢失与上下文限制问题。提出AggAgent，通过轻量工具有效合成轨迹信息，提升任务性能。**

- **链接: [https://arxiv.org/pdf/2604.11753](https://arxiv.org/pdf/2604.11753)**

> **作者:** Yoonsang Lee; Howard Yen; Xi Ye; Danqi Chen
>
> **摘要:** We study parallel test-time scaling for long-horizon agentic tasks such as agentic search and deep research, where multiple rollouts are generated in parallel and aggregated into a final response. While such scaling has proven effective for chain-of-thought reasoning, agentic tasks pose unique challenges: trajectories are long, multi-turn, and tool-augmented, and outputs are often open-ended. Aggregating only final answers discards rich information from trajectories, while concatenating all trajectories exceeds the model's context window. To address this, we propose AggAgent, an aggregation agent that treats parallel trajectories as an environment. We equip it with lightweight tools to inspect candidate solutions and search across trajectories, enabling it to navigate and synthesize information on demand. Across six benchmarks and three model families (GLM-4.7, Qwen3.5, MiniMax-M2.5), AggAgent outperforms all existing aggregation methods-by up to 5.3% absolute on average and 10.3% on two deep research tasks-while adding minimal overhead, as the aggregation cost remains bounded by a single agentic rollout. Our findings establish agentic aggregation as an effective and cost-efficient approach to parallel test-time scaling.
>
---
#### [new 021] Efficient Training for Cross-lingual Speech Language Models
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于跨语言语音语言模型任务，旨在解决数据有限和多语言扩展困难的问题。提出CSLM方法，通过离散语音标记实现跨模态与跨语言对齐，提升生成质量并降低延迟。**

- **链接: [https://arxiv.org/pdf/2604.11096](https://arxiv.org/pdf/2604.11096)**

> **作者:** Yan Zhou; Qingkai Fang; Yun Hong; Yang Feng
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** Currently, large language models (LLMs) predominantly focus on the text modality. To enable more natural human-AI interaction, speech LLMs are emerging, but building effective end-to-end speech LLMs remains challenging due to limited data and the difficulty in expanding to more languages. In this paper, we introduce Cross-lingual Speech Language Model (CSLM), an efficient training method for cross-lingual speech LLMs based on discrete speech tokens. We propose a novel alignment strategy that achieves cross-modal and cross-lingual alignment through continual pre-training. By conducting instruction fine-tuning following a speech-text interleaved chain-of-modality generation process, we enhance modal alignment at a finer granularity, thereby improving generation quality and reducing latency. CSLM aligns different modalities and languages simultaneously without the need for massive speech data, thus exhibiting good language scalability. Evaluations on cross-modal tasks, mono-lingual conversational tasks, and cross-lingual conversational tasks demonstrate CSLM's strong cross-modal alignment capabilities and general task abilities. (Code is available at: this https URL)
>
---
#### [new 022] Utilizing and Calibrating Hindsight Process Rewards via Reinforcement with Mutual Information Self-Evaluation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于强化学习任务，解决大语言模型在稀疏奖励下的学习问题。提出MISE方法，利用生成式自评估提供密集奖励并校准其与环境反馈的一致性。**

- **链接: [https://arxiv.org/pdf/2604.11611](https://arxiv.org/pdf/2604.11611)**

> **作者:** Jiashu Yao; Heyan Huang; Zeming Liu; Yuhang Guo
>
> **备注:** preprint
>
> **摘要:** To overcome the sparse reward challenge in reinforcement learning (RL) for agents based on large language models (LLMs), we propose Mutual Information Self-Evaluation (MISE), an RL paradigm that utilizes hindsight generative self-evaluation as dense reward signals while simultaneously calibrating them against the environmental feedbacks. Empirically, MISE enables an agent to learn autonomously from dense internal rewards supplementing sparse extrinsic signals. Theoretically, our work provides the first formal foundation for the paradigm of generative self-rewarding. We prove that utilizing hindsight self-evaluation rewards is equivalent to minimizing an objective that combines mutual information with a KL divergence term between the policy and a proxy reward policy. This theoretical insight then informs and justifies our calibration step, which actively aligns these rewards with the optimal policy. Extensive experiments show that MISE outperforms strong baselines, enabling open-source LLMs about 7B parameters to achieve performance comparable to GPT-4o on validation without expert supervision.
>
---
#### [new 023] How You Ask Matters! Adaptive RAG Robustness to Query Variations
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，针对Adaptive RAG在查询变化下的鲁棒性问题，提出基准测试并分析其性能表现。**

- **链接: [https://arxiv.org/pdf/2604.10745](https://arxiv.org/pdf/2604.10745)**

> **作者:** Yunah Jang; Megha Sundriyal; Kyomin Jung; Meeyoung Cha
>
> **摘要:** Adaptive Retrieval-Augmented Generation (RAG) promises accuracy and efficiency by dynamically triggering retrieval only when needed and is widely used in practice. However, real-world queries vary in surface form even with the same intent, and their impact on Adaptive RAG remains under-explored. We introduce the first large-scale benchmark of diverse yet semantically identical query variations, combining human-written and model-generated rewrites. Our benchmark facilitates a systematic evaluation of Adaptive RAG robustness by examining its key components across three dimensions: answer quality, computational cost, and retrieval decisions. We discover a critical robustness gap, where small surface-level changes in queries dramatically alter retrieval behavior and accuracy. Although larger models show better performance, robustness does not improve accordingly. These findings reveal that Adaptive RAG methods are highly vulnerable to query variations that preserve identical semantics, exposing a critical robustness challenge.
>
---
#### [new 024] BITS Pilani at SemEval-2026 Task 9: Structured Supervised Fine-Tuning with DPO Refinement for Polarization Detection
- **分类: cs.CL**

- **简介: 该论文属于政治极化检测任务，旨在提升社交媒体文本中极化的准确识别。通过结合结构化微调与DPO优化，减少误判，提高召回率和F1值。**

- **链接: [https://arxiv.org/pdf/2604.11121](https://arxiv.org/pdf/2604.11121)**

> **作者:** Atharva Gupta; Dhruv Kumar; Yash Sinha
>
> **摘要:** The POLAR SemEval-2026 Shared Task aims to detect online polarization and focuses on the classification and identification of multilingual, multicultural, and multi-event polarization. Accurate computational detection of online polarization is challenging due to nuanced rhetoric, implicit framing, and the high cost of human-in-the-loop annotation. Building on recent findings that contextual prompting enables large language models to function as strong polarization detectors, we present a two-stage approach for detecting political polarization in social media text that combines structured supervised fine-tuning with Direct Preference Optimization (DPO) refinement. We fine-tune Qwen 2.5-7B-Instruct with LoRA using an interpretable slot-filling template (target, claim type, manifestation checklist, and justification). We then apply DPO with automatically generated preference pairs to reduce costly false negatives. Experiments on the SemEval 2026 POLAR shared task dataset show that preference-based refinement improves both accuracy and decreases false negatives without extra annotation. On the English development set, DPO increases recall from 0.5085 to 0.7797 and improves macro-F1 by ~5 points.
>
---
#### [new 025] Learning and Enforcing Context-Sensitive Control for LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言生成任务，旨在解决LLMs生成有效性问题。通过自动学习上下文敏感约束，无需人工干预即可保证生成内容符合规则。**

- **链接: [https://arxiv.org/pdf/2604.10667](https://arxiv.org/pdf/2604.10667)**

> **作者:** Mohammad Albinhassan; Pranava Madhyastha; Mark Law; Alessandra Russo
>
> **备注:** ACL 2025 Student Research Workshop
>
> **摘要:** Controlling the output of Large Language Models (LLMs) through context-sensitive constraints has emerged as a promising approach to overcome the limitations of Context-Free Grammars (CFGs) in guaranteeing generation validity. However, such constraints typically require manual specification -- a significant barrier demanding specialized expertise. We introduce a framework that automatically learns context-sensitive constraints from LLM interactions through a two-phase process: syntactic exploration to gather diverse outputs for constraint learning, followed by constraint exploitation to enforce these learned rules during generation. Experiments demonstrate that our method enables even small LLMs (1B parameters) to learn and generate with perfect constraint adherence, outperforming larger counterparts and state-of-the-art reasoning models. This work represents the first integration of context-sensitive grammar learning with LLM generation, eliminating manual specification while maintaining generation validity.
>
---
#### [new 026] Triviality Corrected Endogenous Reward
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，解决开放文本生成中缺乏可验证奖励的问题。通过引入TCER方法，提升生成内容的多样性和质量。**

- **链接: [https://arxiv.org/pdf/2604.11522](https://arxiv.org/pdf/2604.11522)**

> **作者:** Xinda Wang; Zhengxu Hou; Yangshijie Zhang; Bingren Yan; Jialin Liu; Chenzhuo Zhao; Zhibo Yang; Bin-Bin Yang; Feng Xiao
>
> **摘要:** Reinforcement learning for open-ended text generation is constrained by the lack of verifiable rewards, necessitating reliance on judge models that require either annotated data or powerful closed-source models. Inspired by recent work on unsupervised reinforcement learning for mathematical reasoning using confidence-based endogenous rewards, we investigate whether this principle can be adapted to open-ended writing tasks. We find that directly applying confidence rewards leads to Triviality Bias: the policy collapses toward high-probability outputs, reducing diversity and meaningful content. We propose TCER (Triviality Corrected Endogenous Reward), which addresses this bias by rewarding the relative information gain between a specialist policy and a generalist reference policy, modulated by a probability-dependent correction mechanism. Across multiple writing benchmarks and model architectures, TCER achieves consistent improvements without external supervision. Furthermore, TCER also transfers effectively to mathematical reasoning, validating the generality of our approach across different generation tasks.
>
---
#### [new 027] ks-pret-5m: a 5 million word, 12 million token kashmiri pretraining dataset
- **分类: cs.CL**

- **简介: 该论文介绍KS-PRET-5M，一个用于克什米尔语预训练的大型数据集，解决语言资源不足问题，通过收集和清理文本，提供高质量语料支持语言模型和研究。**

- **链接: [https://arxiv.org/pdf/2604.11066](https://arxiv.org/pdf/2604.11066)**

> **作者:** Haq Nawaz Malik; Nahfid Nissar
>
> **摘要:** We present KS-PRET-5M, the largest publicly available pretraining dataset for the Kashmiri language, comprising 5,090,244 (5.09M) words, 27,692,959 (27.6M) characters, and a vocabulary of 295,433 (295.4K) unique word types. We assembled the dataset from two source classes: digitized archival and literary material, encompassing literature, news, biographies, novels, poetry, religious scholarship, and academic writing, recovered from the proprietary InPage desktop-publishing format using the converter of Malik~\cite{malik2024inpage}, and Unicode-native text collected from Kashmiri-language web sources. All text was processed through an eleven-stage cleaning pipeline that achieves a mean Kashmiri script ratio of 0.9965, reducing Devanagari contamination to 146 characters across the full dataset. We tokenized the dataset empirically using google/muril-base-cased, yielding a subword ratio of 2.383 tokens per word and a total of approximately 12.13 million subword tokens, substantially higher than prior estimates derived from non-Kashmiri Perso-Arabic analogues. KS-PRET-5M is released as a single continuous text stream under CC~BY~4.0 to support language model pretraining, tokenizer training, and computational linguistic research for Kashmiri.
>
---
#### [new 028] Turing or Cantor: That is the Question
- **分类: cs.CL**

- **简介: 论文探讨了图灵与康托尔的贡献关系，提出基于概率的不可判定性度量，定义三类新复杂度类，并否定U-完全类中P≠NP问题。**

- **链接: [https://arxiv.org/pdf/2604.10418](https://arxiv.org/pdf/2604.10418)**

> **作者:** Eugene Eberbach
>
> **摘要:** Alan Turing is considered as a founder of current computer science together with Kurt Godel, Alonzo Church and John von Neumann. In this paper multiple new research results are presented. It is demonstrated that there would not be Alan Turing's achievements without earlier seminal contributions by Georg Cantor in the set theory and foundations of mathematics. It is proposed to introduce the measure of undecidability of problems unsolvable by Turing machines based on probability distribution of its input data, i.e., to provide the degree of unsolvabilty based on the number of undecidable instances of input data versus decidable ones. It is proposed as well to extend the Turing's work on infinite logics and Oracle machines to a whole class of super-Turing models of computation. Next, the three new complexity classes for TM undecidable problems have been defined: U-complete (Universal complete), D-complete (Diagonalization complete) and H-complete (Hypercomputation complete) classes. The above has never been defined explicitly before by other scientists, and has been inspired by Cook/Levin NP-complete class for intractable problems. Finally, an equivalent to famous P is not equal to NP unanswered question for NP-complete class, has been answered negatively for U-complete class of complexity for undecidable problems.
>
---
#### [new 029] BLUEmed: Retrieval-Augmented Multi-Agent Debate for Clinical Error Detection
- **分类: cs.CL**

- **简介: 该论文属于临床错误检测任务，解决术语替换错误问题。提出BLUEmed框架，结合检索增强生成与多智能体辩论，提升检测准确率。**

- **链接: [https://arxiv.org/pdf/2604.10389](https://arxiv.org/pdf/2604.10389)**

> **作者:** Saukun Thika You; Nguyen Anh Khoa Tran; Wesley K. Marizane; Hanshu Rao; Qiunan Zhang; Xiaolei Huang
>
> **备注:** Accepted to the IEEE International Conference on Healthcare Informatics (ICHI) 2026
>
> **摘要:** Terminology substitution errors in clinical notes, where one medical term is replaced by a linguistically valid but clinically different term, pose a persistent challenge for automated error detection in healthcare. We introduce BLUEmed, a multi-agent debate framework augmented with hybrid Retrieval-Augmented Generation (RAG) that combines evidence-grounded reasoning with multi-perspective verification for clinical error detection. BLUEmed decomposes each clinical note into focused sub-queries, retrieves source-partitioned evidence through dense, sparse, and online retrieval, and assigns two domain expert agents distinct knowledge bases to produce independent analyses; when the experts disagree, a structured counter-argumentation round and cross-source adjudication resolve the conflict, followed by a cascading safety layer that filters common false-positive patterns. We evaluate BLUEmed on a clinical terminology substitution detection benchmark under both zero-shot and few-shot prompting with multiple backbone models spanning proprietary and open-source families. Experimental results show that BLUEmed achieves the best accuracy (69.13%), ROC-AUC (74.45%), and PR-AUC (72.44%) under few-shot prompting, outperforming both single-agent RAG and debate-only baselines. Further analyses across six backbone models and two prompting strategies confirm that retrieval augmentation and structured debate are complementary, and that the framework benefits most from models with sufficient instruction-following and clinical language understanding.
>
---
#### [new 030] Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty
- **分类: cs.CL**

- **简介: 该论文属于增强大模型推理能力的任务，解决GRM计算成本高和评估不精准的问题，提出E-GRM框架，基于模型内部不确定性选择性触发推理，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.10072](https://arxiv.org/pdf/2604.10072)**

> **作者:** Chao Xue; Yao Wang; Mengqiao Liu; Di Liang; Xingsheng Han; Peiyang Liu; Xianjie Wu; Chenyao Lu; Lei Jiang; Yu Lu; Haibo Shi; Shuang Liang; Minlong Peng; Flora D. Salim
>
> **备注:** accepted by ACL 2026
>
> **摘要:** Recent advancements in the Generative Reward Model (GRM) have demonstrated its potential to enhance the reasoning abilities of LLMs through Chain-of-Thought (CoT) prompting. Despite these gains, existing implementations of GRM suffer from two critical limitations. First, CoT prompting is applied indiscriminately to all inputs regardless of their inherent complexity. This introduces unnecessary computational costs for tasks amenable to fast, direct inference. Second, existing approaches primarily rely on voting-based mechanisms to evaluate CoT outputs, which often lack granularity and precision in assessing reasoning quality. In this paper, we propose E-GRM, an efficient generative reward modeling framework grounded in model-internal uncertainty. E-GRM leverages the convergence behavior of parallel model generations to estimate uncertainty and selectively trigger CoT reasoning only when needed, without relying on handcrafted features or task-dependent signals. To improve reward fidelity, we introduce a lightweight discriminative scorer trained with a hybrid regression--ranking objective to provide fine-grained evaluation of reasoning paths. Experiments on multiple reasoning benchmarks show that E-GRM substantially reduces inference cost while consistently improving answer accuracy, demonstrating that model-internal uncertainty is an effective and general signal for efficient reasoning-aware reward modeling.
>
---
#### [new 031] Synthius-Mem: Brain-Inspired Hallucination-Resistant Persona Memory Achieving 94.4% Memory Accuracy and 99.6% Adversarial Robustness on LoCoMo
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于AI记忆任务，旨在解决长时记忆不准确和幻觉问题。提出Synthius-Mem系统，通过结构化人格记忆提升准确性和对抗鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.11563](https://arxiv.org/pdf/2604.11563)**

> **作者:** Artem Gadzhiev; Andrew Kislov
>
> **摘要:** Providing AI agents with reliable long-term memory that does not hallucinate remains an open problem. Current approaches to memory for LLM agents -- sliding windows, summarization, embedding-based RAG, and flat fact extraction -- each reduce token cost but introduce catastrophic information loss, semantic drift, or uncontrolled hallucination about the user. The structural reason is architectural: every published memory system on the LoCoMo benchmark treats conversation as a retrieval problem over raw or lightly summarized dialogue segments, and none reports adversarial robustness, the ability to refuse questions about facts the user never disclosed. We present Synthius-Mem, a brain-inspired structured persona memory system that takes a fundamentally different approach. Instead of retrieving what was said, Synthius-Mem extracts what is known about the person: a full persona extraction pipeline decomposes conversations into six cognitive domains (biography, experiences, preferences, social circle, work, psychometrics), consolidates and deduplicates per domain, and retrieves structured facts via CategoryRAG at 21.79 ms latency. On the LoCoMo benchmark (ACL 2024, 10 conversations, 1,813 questions), Synthius-Mem achieves 94.37% accuracy, exceeding all published systems including MemMachine (91.69%, adversarial score is not reported) and human performance (87.9 F1). Core memory fact accuracy reaches 98.64%. Adversarial robustness, the hallucination resistance metric that no competing system reports, reaches 99.55%. Synthius-Mem reduces token consumption by ~5x compared to full-context replay while achieving higher accuracy. Synthius-Mem achieves state-of-the-art results on LoCoMo and is, to our knowledge, the only persona memory system that both exceeds human-level performance and reports adversarial robustness.
>
---
#### [new 032] CodaRAG: Connecting the Dots with Associativity Inspired by Complementary Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CodaRAG框架，解决LLM在知识密集型任务中的信息碎片化与逻辑链缺失问题，通过关联检索提升事实、推理和创造性任务的准确性。**

- **链接: [https://arxiv.org/pdf/2604.10426](https://arxiv.org/pdf/2604.10426)**

> **作者:** Cheng-Yen Li; Xuanjun Chen; Claire Lin; Wei-Yu Chen; Wenhua Nie; Hung-Yi Lee; Jyh-Shing Roger Jang
>
> **备注:** Preprint, Submitted to ACM TIST
>
> **摘要:** Large Language Models (LLMs) struggle with knowledge-intensive tasks due to hallucinations and fragmented reasoning over dispersed information. While Retrieval-Augmented Generation (RAG) grounds generation in external sources, existing methods often treat evidence as isolated units, failing to reconstruct the logical chains that connect these dots. Inspired by Complementary Learning Systems (CLS), we propose CodaRAG, a framework that evolves retrieval from passive lookup into active associative discovery. CodaRAG operates via a three-stage pipeline: (1) Knowledge Consolidation to unify fragmented extractions into a stable memory substrate; (2) Associative Navigation to traverse the graph via multi-dimensional pathways-semantic, contextualized, and functional-explicitly recovering dispersed evidence chains; and (3) Interference Elimination to prune hyper-associative noise, ensuring a coherent, high-precision reasoning context. On GraphRAG-Bench, CodaRAG achieves absolute gains of 7-10% in retrieval recall and 3-11% in generation accuracy. These results demonstrate CodaRAG's superior ability to systematically robustify associative evidence retrieval for factual, reasoning, and creative tasks.
>
---
#### [new 033] MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于数学推理数据生成任务，旨在解决高质量数据合成难题。提出一种基于约束图的层级框架，通过对抗进化生成结构化蓝图，实现复杂逻辑数据的高效合成。**

- **链接: [https://arxiv.org/pdf/2604.11188](https://arxiv.org/pdf/2604.11188)**

> **作者:** Zixiong Yu; Jun Rao; Guhan Chen; Songtao Tian; Bohan Li; Jiansheng Wei; Min Zhang; Xiaojun Meng
>
> **备注:** Accepted by ACL 2026 findings
>
> **摘要:** Synthesizing high-quality mathematical reasoning data without human priors remains a significant challenge. Current approaches typically rely on seed data mutation or simple prompt engineering, often suffering from mode collapse and limited logical complexity. This paper proposes a hierarchical synthesis framework that formulates data synthesis as an unsupervised optimization problem over a constraint graph followed by semantic instantiation, rather than treating it as a direct text generation task. We introduce a Legislator-Executor paradigm: The Legislator adversarially evolves structured generation blueprints encoding the constraints of the problem, while the Executor instantiates these specifications into diverse natural language scenarios. This decoupling of skeleton design from linguistic realization enables a prioritized focus on constructing complex and diverse logical structures, thereby guiding high-quality data synthesis. Experiments conducted on a total of 10 models across the Qwen, Llama, Mistral, and Gemma series demonstrate that our method achieves notable results: models fine-tuned on 1K synthesized samples outperform widely-used datasets of comparable scale (LIMO, s1K) across eight mathematical benchmarks, exhibiting superior out-of-distribution generalization.
>
---
#### [new 034] Policy Split: Incentivizing Dual-Mode Exploration in LLM Reinforcement with Dual-Mode Entropy Regularization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决大语言模型探索与准确性的平衡问题。通过引入双模式策略，提升模型的多样化探索能力。**

- **链接: [https://arxiv.org/pdf/2604.11510](https://arxiv.org/pdf/2604.11510)**

> **作者:** Jiashu Yao; Heyan Huang; Chuwei Luo; Daiqing Wu; Zeming Liu; Yuhang Guo; Yangyang Kang
>
> **备注:** preprint
>
> **摘要:** To encourage diverse exploration in reinforcement learning (RL) for large language models (LLMs) without compromising accuracy, we propose Policy Split, a novel paradigm that bifurcates the policy into normal and high-entropy modes with a high-entropy prompt. While sharing model parameters, the two modes undergo collaborative dual-mode entropy regularization tailored to distinct objectives. Specifically, the normal mode optimizes for task correctness, while the high-entropy mode incorporates a preference for exploration, and the two modes learn collaboratively. Extensive experiments demonstrate that our approach consistently outperforms established entropy-guided RL baselines across various model sizes in general and creative tasks. Further analysis reveals that Policy Split facilitates dual-mode exploration, where the high-entropy mode generates distinct behavioral patterns to the normal mode, providing unique learning signals.
>
---
#### [new 035] Saar-Voice: A Multi-Speaker Saarbrücken Dialect Speech Corpus
- **分类: cs.CL**

- **简介: 该论文属于语音处理任务，旨在解决方言资源不足的问题。通过构建Saar-Voice语料库，提供多说话人的方言语音数据，支持低资源场景下的文本转语音研究。**

- **链接: [https://arxiv.org/pdf/2604.11803](https://arxiv.org/pdf/2604.11803)**

> **作者:** Lena S. Oberkircher; Jesujoba O. Alabi; Dietrich Klakow; Jürgen Trouvain
>
> **备注:** accepted at DialRes-LREC26
>
> **摘要:** Natural language processing (NLP) and speech technologies have made significant progress in recent years; however, they remain largely focused on standardized language varieties. Dialects, despite their cultural significance and widespread use, are underrepresented in linguistic resources and computational models, resulting in performance disparities. To address this gap, we introduce Saar-Voice, a six-hour speech corpus for the Saarbrücken dialect of German. The dataset was created by first collecting text through digitized books and locally sourced materials. A subset of this text was recorded by nine speakers, and we conducted analyses on both the textual and speech components to assess the dataset's characteristics and quality. We discuss methodological challenges related to orthographic and speaker variation, and explore grapheme-to-phoneme (G2P) conversion. The resulting corpus provides aligned textual and audio representations. This serves as a foundation for future research on dialect-aware text-to-speech (TTS), particularly in low-resource scenarios, including zero-shot and few-shot model adaptation.
>
---
#### [new 036] ProUIE: A Macro-to-Micro Progressive Learning Method for LLM-based Universal Information Extraction
- **分类: cs.CL**

- **简介: 该论文提出ProUIE，解决LLM通用信息抽取任务中的训练复杂与效果有限问题，通过宏观到微观的渐进学习方法提升抽取性能。**

- **链接: [https://arxiv.org/pdf/2604.10633](https://arxiv.org/pdf/2604.10633)**

> **作者:** Wenda Liu; Zhigang Song; Shuai Nie; Guangyao Liu; Lisung Chen; Binyu Yang; Yaran Chen; Peng Zhou; Hongzhen Wang; Yuchen Liu; Wenyue Hu; Jiaming Xu; Runyu Shi; Ying Huang
>
> **摘要:** LLM-based universal information extraction (UIE) methods often rely on additional information beyond the original training data, which increases training complexity yet often yields limited gains. To address this, we propose ProUIE, a Macro-to-Micro progressive learning approach that improves UIE without introducing any external information. ProUIE consists of three stages: (i) macro-level Complete Modeling (CM), which learns NER, RE, and EE along their intrinsic difficulty order on the full training data to build a unified extraction foundation, (ii) meso-level Streamlined Alignment (SA), which operates on sampled data with simplified target formats, streamlining and regularizing structured outputs to make them more concise and controllable, and (iii) micro-level Deep Exploration (DE), which applies GRPO with stepwise fine-grained rewards (SFR) over structural units to guide exploration and improve performance. Experiments on 36 public datasets show that ProUIE consistently improves unified extraction, outperforming strong instruction-tuned baselines on average for NER and RE while using a smaller backbone, and it further demonstrates clear gains in large-scale production-oriented information extraction.
>
---
#### [new 037] A Structured Clustering Approach for Inducing Media Narratives
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的文本分析任务，旨在解决媒体叙事结构难以准确捕捉的问题。通过结构化聚类方法，联合建模事件与角色，生成可解释的叙事模式。**

- **链接: [https://arxiv.org/pdf/2604.10368](https://arxiv.org/pdf/2604.10368)**

> **作者:** Rohan Das; Advait Deshmukh; Alexandria Leto; Zohar Naaman; I-Ta Lee; Maria Leonor Pacheco
>
> **备注:** Accepted to the Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)
>
> **摘要:** Media narratives wield tremendous power in shaping public opinion, yet computational approaches struggle to capture the nuanced storytelling structures that communication theory emphasizes as central to how meaning is constructed. Existing approaches either miss subtle narrative patterns through coarse-grained analysis or require domain-specific taxonomies that limit scalability. To bridge this gap, we present a framework for inducing rich narrative schemas by jointly modeling events and characters via structured clustering. Our approach produces explainable narrative schemas that align with established framing theory while scaling to large corpora without exhaustive manual annotation.
>
---
#### [new 038] Toward Generalized Cross-Lingual Hateful Language Detection with Web-Scale Data and Ensemble LLM Annotations
- **分类: cs.CL**

- **简介: 该论文属于多语言仇恨语言检测任务，旨在提升低资源语言的检测效果。通过使用网络数据和LLM合成标注，结合预训练与集成学习方法，显著提升了小模型性能。**

- **链接: [https://arxiv.org/pdf/2604.09625](https://arxiv.org/pdf/2604.09625)**

> **作者:** Dang H. Dang; Jelena Mitrovi; Michael Granitzer
>
> **备注:** 8 Pages, 3 tables, LREC 2026 papers
>
> **摘要:** We study whether large-scale unlabelled web data and LLM-based synthetic annotations can improve multilingual hate speech detection. Starting from texts crawled via this http URL~(OWS) in four languages (English, German, Spanish, Vietnamese), we pursue two complementary strategies. First, we apply continued pre-training to BERT models by continuing masked language modelling on unlabelled OWS texts before supervised fine-tuning, and show that this yields an average macro-F1 gain of approximately 3% over standard baselines across sixteen benchmarks, with stronger gains in low-resource settings. Second, we use four open-source LLMs (Mistral-7B, Llama3.1-8B, Gemma2-9B, Qwen2.5-14B) to produce synthetic annotations through three ensemble strategies: mean averaging, majority voting, and a LightGBM meta-learner. The LightGBM ensemble consistently outperforms the other strategies. Fine-tuning on these synthetic labels substantially benefits a small model (Llama3.2-1B: +11% pooled F1), but provides only a modest gain for the larger Qwen2.5-14B (+0.6%). Our results indicate that the combination of web-scale unlabelled data and LLM-ensemble annotations is the most valuable for smaller models and low-resource languages.
>
---
#### [new 039] Comparative Analysis of Large Language Models in Healthcare
- **分类: cs.CL**

- **简介: 该论文属于医疗AI任务，旨在比较不同大语言模型在医疗场景中的表现，解决标准化评估不足的问题，通过多项任务测试模型性能。**

- **链接: [https://arxiv.org/pdf/2604.10316](https://arxiv.org/pdf/2604.10316)**

> **作者:** Subin Santhosh; Farwa Abbas; Hussain Ahmad; Claudia Szabo
>
> **摘要:** Background: Large Language Models (LLMs) are transforming artificial intelligence applications in healthcare due to their ability to understand, generate, and summarize complex medical text. They offer valuable support to clinicians, researchers, and patients, yet their deployment in high-stakes clinical environments raises critical concerns regarding accuracy, reliability, and patient safety. Despite substantial attention in recent years, standardized benchmarking of LLMs for medical applications has been limited. Objective: This study addresses the need for a standardized comparative evaluation of LLMs in medical settings. Method: We evaluate multiple models, including ChatGPT, LLaMA, Grok, Gemini, and ChatDoctor, on core medical tasks such as patient note summarization and medical question answering, using the open-access datasets, MedMCQA, PubMedQA, and Asclepius, and assess performance through a combination of linguistic and task-specific metrics. Results: The results indicate that domain-specific models, such as ChatDoctor, excel in contextual reliability, producing medically accurate and semantically aligned text, whereas general-purpose models like Grok and LLaMA perform better in structured question-answering tasks, demonstrating higher quantitative accuracy. This highlights the complementary strengths of domain-specific and general-purpose LLMs depending on the medical task. Conclusion: Our findings suggest that LLMs can meaningfully support medical professionals and enhance clinical decision-making; however, their safe and effective deployment requires adherence to ethical standards, contextual accuracy, and human oversight in relevant cases. These results underscore the importance of task-specific evaluation and cautious integration of LLMs into healthcare workflows.
>
---
#### [new 040] From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律咨询问答任务，旨在解决数据稀缺、任务复杂及上下文依赖等问题。构建了大规模数据集并设计多智能体框架，提升法律推理效果。**

- **链接: [https://arxiv.org/pdf/2604.10470](https://arxiv.org/pdf/2604.10470)**

> **作者:** Mingfei Lu; Yi Zhang; Mengjia Wu; Yue Feng
>
> **备注:** Accepted by ACL 2026 Main conference
>
> **摘要:** Legal consultation question answering (Legal CQA) presents unique challenges compared to traditional legal QA tasks, including the scarcity of high-quality training data, complex task composition, and strong contextual dependencies. To address these, we construct JurisCQAD, a large-scale dataset of over 43,000 real-world Chinese legal queries annotated with expert-validated positive and negative responses, and design a structured task decomposition that converts each query into a legal element graph integrating entities, events, intents, and legal issues. We further propose JurisMA, a modular multi-agent framework supporting dynamic routing, statutory grounding, and stylistic optimization. Combined with the element graph, the framework enables strong context-aware reasoning, effectively capturing dependencies across legal facts, norms, and procedural logic. Trained on JurisCQAD and evaluated on a refined LawBench, our system significantly outperforms both general-purpose and legal-domain LLMs across multiple lexical and semantic metrics, demonstrating the benefits of interpretable decomposition and modular collaboration in Legal CQA.
>
---
#### [new 041] CodeComp: Structural KV Cache Compression for Agentic Coding
- **分类: cs.CL**

- **简介: 该论文提出CodeComp，解决代码任务中KV缓存压缩问题，通过静态分析提升压缩效果，提升代码理解准确性。**

- **链接: [https://arxiv.org/pdf/2604.10235](https://arxiv.org/pdf/2604.10235)**

> **作者:** Qiujiang Chen; Jing Xiong; Chenyang Zhao; Sidi Yang; Ngai Wong
>
> **摘要:** Agentic code tasks such as fault localization and patch generation require processing long codebases under tight memory constraints, where the Key-Value (KV) cache becomes the primary inference bottleneck. Existing compression methods rely exclusively on attention signals to estimate token importance, systematically discarding structurally critical tokens such as call sites, branch conditions, and assignments that are essential for code understanding. We present CodeComp, a training-free KV cache compression framework that incorporates static program analysis into LLM inference via Code Property Graph priors extracted by Joern. Across bug localization and code generation benchmarks, CodeComp consistently outperforms attention-only compression baselines under equal memory budgets, recovering the majority of full-context accuracy under aggressive KV cache compression, while matching the patch generation quality of uncompressed full-context inference and integrating seamlessly into SGLang-based agentic coding pipelines without model modification.
>
---
#### [new 042] MIXAR: Scaling Autoregressive Pixel-based Language Models to Multiple Languages and Scripts
- **分类: cs.CL**

- **简介: 该论文提出MIXAR，一个跨语言的像素生成模型，解决多语言和脚本下的语言建模问题。通过训练多种语言，提升模型在多语言任务中的表现与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.11575](https://arxiv.org/pdf/2604.11575)**

> **作者:** Chen Hu; Yintao Tai; Antonio Vergari; Frank Keller; Alessandro Suglia
>
> **摘要:** Pixel-based language models are gaining momentum as alternatives to traditional token-based approaches, promising to circumvent tokenization challenges. However, the inherent perceptual diversity across languages poses a significant hurdle for multilingual generalization in pixel space. This paper introduces MIXAR, the first generative pixel-based language model trained on eight different languages utilizing a range of different scripts. We empirically evaluate MIXAR against previous pixel-based models as well as comparable tokenizer-based models, demonstrating substantial performance improvement on discriminative and generative multilingual tasks. Additionally, we show how MIXAR is robust to languages never seen during the training. These results are further strengthened when scaling the model to 0.5B parameters which not only improves its capabilities in generative tasks like LAMBADA but also its robustness when challenged with input perturbations such as orthographic attacks.
>
---
#### [new 043] Psychological Concept Neurons: Can Neural Control Bias Probing and Shift Generation in LLMs?
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在探究LLMs中大五人格概念的表示与行为控制关系。通过分析神经元响应和干预实验，发现概念选择性神经元可影响模型输出，但行为控制较难实现。**

- **链接: [https://arxiv.org/pdf/2604.11802](https://arxiv.org/pdf/2604.11802)**

> **作者:** Yuto Harada; Hiro Taiyo Hamada
>
> **摘要:** Using psychological constructs such as the Big Five, large language models (LLMs) can imitate specific personality profiles and predict a user's personality. While LLMs can exhibit behaviors consistent with these constructs, it remains unclear where and how they are represented inside the model and how they relate to behavioral outputs. To address this gap, we focus on questionnaire-operationalized Big Five concepts, analyze the formation and localization of their internal representations, and use interventions to examine how these representations relate to behavioral outputs. In our experiment, we first use probing to examine where Big Five information emerges across model depth. We then identify neurons that respond selectively to each Big Five concept and test whether enhancing or suppressing their activations can bias latent representations and label generation in intended directions. We find that Big Five information becomes rapidly decodable in early layers and remains detectable through the final layers, while concept-selective neurons are most prevalent in mid layers and exhibit limited overlap across domains. Interventions on these neurons consistently shift probe readouts toward targeted concepts, with targeted success rates exceeding 0.8 for some concepts, indicating that the model's internal separation of Big Five personality traits can be causally steered. At the label-generation level, the same interventions often bias generated label distributions in the intended directions, but the effects are weaker, more concept-dependent, and often accompanied by cross-trait spillover, indicating that comparable control over generated labels is difficult even with interventions on a large fraction of concept-selective neurons. Overall, our findings reveal a gap between representational control and behavioral control in LLMs.
>
---
#### [new 044] Evaluating Cooperation in LLM Social Groups through Elected Leadership
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于多智能体协作研究，旨在探讨选举领导是否能提升社会福利与合作。通过模拟实验，验证了有领导结构的系统在多个指标上的显著提升。**

- **链接: [https://arxiv.org/pdf/2604.11721](https://arxiv.org/pdf/2604.11721)**

> **作者:** Ryan Faulkner; Anushka Deshpande; David Guzman Piedrahita; Joel Z. Leibo; Zhijing Jin
>
> **备注:** Main text: 11 pages, 4 figures, 4 tables
>
> **摘要:** Governing common-pool resources requires agents to develop enduring strategies through cooperation and self-governance to avoid collective failure. While foundation models have shown potential for cooperation in these settings, existing multi-agent research provides little insight into whether structured leadership and election mechanisms can improve collective decision making. The lack of such a critical organizational feature ubiquitous in human society presents a significant shortcoming of the current methods. In this work we aim to directly address whether leadership and elections can support improved social welfare and cooperation through multi-agent simulation with LLMs. We present our open-source framework that simulates leadership through elected personas and candidate-driven agendas and carry out an empirical study of LLMs under controlled governance conditions. Our experiments demonstrate that having elected leadership improves social welfare scores by 55.4% and survival time by 128.6% across a range of high performing LLMs. Through the construction of an agent social graph we compute centrality metrics to assess the social influence of leader personas and also analyze rhetorical and cooperative tendencies revealed through a sentiment analysis on leader utterances. This work lays the foundation for further study of election mechanisms in multi-agent systems toward navigating complex social dilemmas.
>
---
#### [new 045] Self-Evolving LLM Memory Extraction Across Heterogeneous Tasks
- **分类: cs.CL**

- **简介: 该论文研究LLM在跨异构任务中的自进化记忆提取问题。针对不同任务需提取不同信息的挑战，提出CluE方法，通过聚类优化提取效果。**

- **链接: [https://arxiv.org/pdf/2604.11610](https://arxiv.org/pdf/2604.11610)**

> **作者:** Yuqing Yang; Tengxiao Liu; Wang Bill Zhu; Taiwei Shi; Linxin Song; Robin Jia
>
> **摘要:** As LLM-based assistants become persistent and personalized, they must extract and retain useful information from past conversations as memory. However, the types of information worth remembering vary considerably across tasks. We formalize the \textit{heterogeneous memory extraction} task and introduce \textbf{BEHEMOTH}, a benchmark that repurposes 18 existing datasets spanning personalization, problem-solving, and agentic tasks, using a downstream utility-driven metric for systematic evaluation. Our empirical analysis confirms that no single static extraction prompt dominates across all task categories, and that existing self-evolving prompt optimization frameworks, originally designed for homogeneous distributions, degrade when training tasks are heterogeneous. To address this, we propose \textbf{CluE}, a cluster-based self-evolving strategy that groups training examples into clusters by extraction scenarios, analyzes each cluster independently, and synthesizes cross-cluster insights to update the extraction prompt. Experiments on BEHEMOTH show that CluE generalizes effectively across heterogeneous tasks ($+$9.04\% relative gain), consistently outperforming prior self-evolving frameworks.
>
---
#### [new 046] Spoiler Alert: Narrative Forecasting as a Metric for Tension in LLM Storytelling
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，旨在提升LLM讲故事的紧张感。通过引入100-Endings指标评估叙事张力，优化生成流程以提高故事质量。**

- **链接: [https://arxiv.org/pdf/2604.09854](https://arxiv.org/pdf/2604.09854)**

> **作者:** Peiqi Sui; Yutong Zhu; Tianyi Cheng; Peter West; Richard Jean So; Hoyt Long; Ari Holtzman
>
> **备注:** 29 pages, 10 figures, 9 tables
>
> **摘要:** LLMs have so far failed both to generate consistently compelling stories and to recognize this failure--on the leading creative-writing benchmark (EQ-Bench), LLM judges rank zero-shot AI stories above New Yorker short stories, a gold standard for literary fiction. We argue that existing rubrics overlook a key dimension of compelling human stories: narrative tension. We introduce the 100-Endings metric, which walks through a story sentence by sentence: at each position, a model predicts how the story will end 100 times given only the text so far, and we measure tension as how often predictions fail to match the ground truth. Beyond the mismatch rate, the sentence-level curve yields complementary statistics, such as inflection rate, a geometric measure of how frequently the curve reverses direction, tracking twists and revelations. Unlike rubric-based judges, 100-Endings correctly ranks New Yorker stories far above LLM outputs. Grounded in narratological principles, we design a story-generation pipeline using structural constraints, including analysis of story templates, idea formulation, and narrative scaffolding. Our pipeline significantly increases narrative tension as measured by the 100-Endings metric, while maintaining performance on the EQ-Bench leaderboard.
>
---
#### [new 047] CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity
- **分类: cs.CL**

- **简介: 该论文提出CArtBench，用于评估视觉语言模型在中文艺术理解、解释和真实性判断上的能力。任务涵盖艺术识别、赏析、再诠释和真伪鉴别，旨在解决模型在深度艺术推理上的不足。**

- **链接: [https://arxiv.org/pdf/2604.11632](https://arxiv.org/pdf/2604.11632)**

> **作者:** Xuefeng Wei; Zhixuan Wang; Xuan Zhou; Zhi Qu; Hongyao Li; Yusuke Sakai; Hidetaka Kamigaito; Taro Watanabe
>
> **备注:** Accepted to the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)
>
> **摘要:** We introduce CARTBENCH, a museum-grounded benchmark for evaluating vision-language models (VLMs) on Chinese artworks beyond short-form recognition and QA. CARTBENCH comprises four subtasks: CURATORQA for evidence-grounded recognition and reasoning, CATALOGCAPTION for structured four-section expert-style appreciation, REINTERPRET for defensible reinterpretation with expert ratings, and CONNOISSEURPAIRS for diagnostic authenticity discrimination under visually similar confounds. CARTBENCH is built by aligning image-bearing Palace Museum objects from Wikidata with authoritative catalog pages, spanning five art categories across multiple dynasties. Across nine representative VLMs, we find that high overall CURATORQA accuracy can mask sharp drops on hard evidence linking and style-to-period inference; long-form appreciation remains far from expert references; and authenticity-oriented diagnostic discrimination stays near chance, underscoring the difficulty of connoisseur-level reasoning for current models.
>
---
#### [new 048] Mem$^2$Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Mem$^2$Evolve框架，解决自进化智能体能力增长受限问题，通过融合经验记忆与资产记忆实现协同进化。**

- **链接: [https://arxiv.org/pdf/2604.10923](https://arxiv.org/pdf/2604.10923)**

> **作者:** Zihao Cheng; Zeming Liu; Yingyu Shan; Xinyi Wang; Xiangrong Zhu; Yunpu Ma; Hongru Wang; Yuhang Guo; Wei Lin; Yunhong Wang
>
> **备注:** Accepted by ACL 2026 Main
>
> **摘要:** While large language model--powered agents can self-evolve by accumulating experience or by dynamically creating new assets (i.e., tools or expert agents), existing frameworks typically treat these two evolutionary processes in isolation. This separation overlooks their intrinsic interdependence: the former is inherently bounded by a manually predefined static toolset, while the latter generates new assets from scratch without experiential guidance, leading to limited capability growth and unstable evolution. To address this limitation, we introduce a novel paradigm of co-evolutionary Capability Expansion and Experience Distillation. Guided by this paradigm, we propose the \textbf{Mem$^{\textbf{2}}$Evolve}, which integrates two core components: \textbf{Experience Memory} and \textbf{Asset Memory}. Specifically, Mem$^{2}$Evolve leverages accumulated experience to guide the dynamic creation of assets, thereby expanding the agent's capability space while simultaneously acquiring new experience to achieve co-evolution. Extensive experiments across 6 task categories and 8 benchmarks demonstrate that Mem$^{2}$Evolve achieves improvement of 18.53\% over standard LLMs, 11.80\% over agents evolving solely through experience, and 6.46\% over those evolving solely through asset creation, establishing it as a substantially more effective and stable self-evolving agent framework. Code is available at: this https URL.
>
---
#### [new 049] Computational Implementation of a Model of Category-Theoretic Metaphor Comprehension
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决隐喻理解问题。通过改进算法，提升隐喻理解的准确性、系统性和新颖性。**

- **链接: [https://arxiv.org/pdf/2604.10035](https://arxiv.org/pdf/2604.10035)**

> **作者:** Fumitaka Iwaki; Miho Fuyama; Hayato Saigo; Tatsuji Takahashi
>
> **备注:** 7 pages, 8 figures, CogSci member abstract
>
> **摘要:** In this study, we developed a computational implementation for a model of metaphor comprehension based on the theory of indeterminate natural transformation (TINT) proposed by Fuyama et al. We simplified the algorithms implementing the model to be closer to the original theory and verified it through data fitting and simulations. The outputs of the algorithms are evaluated with three measures: data-fitting with experimental data, the systematicity of the metaphor comprehension result, and the novelty of the comprehension (i.e. the correspondence of the associative structure of the source and target of the metaphor). The improved algorithm outperformed the existing ones in all the three measures.
>
---
#### [new 050] Adaptive Multi-Expert Reasoning via Difficulty-Aware Routing and Uncertainty-Guided Aggregation
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出AMR框架，解决数学推理中模型性能不稳定的任务。通过难度感知路由和不确定性引导聚合，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.10335](https://arxiv.org/pdf/2604.10335)**

> **作者:** Mohamed Ehab; Ali Hamdi
>
> **摘要:** Large language models (LLMs) demonstrate strong performance in math reasoning benchmarks, but their performance varies inconsistently across problems with varying levels of difficulty. This paper describes Adaptive Multi-Expert Reasoning (AMR), a framework that focuses on problem complexity by reasoning with dynamically adapted strategies. An agile routing system that focuses on problem text predicts problems' difficulty and uncertainty and guides a reconfigurable sampling mechanism to manage the breadth of generation. Three specialized experts create candidate responses, which are modified during multiple correction and finalization phases. A neural verifier assesses the correctness of responses, while a clustering-based aggregation technique identifies the final candidate answer based on a combination of consensus and answer quality. When evaluated on the GSM8K dataset, AMR achieved 75.28% accuracy while only using the original training data. This result outperformed the majority of comparable 7B models that were trained on synthetic data. This showcases that models using difficulty-based routing and uncertainty-driven aggregation are efficient and effective in improving math reasoning models' robustness.
>
---
#### [new 051] Relational Probing: LM-to-Graph Adaptation for Financial Prediction
- **分类: cs.CL**

- **简介: 该论文属于金融预测任务，旨在解决语言模型与结构化图数据结合的问题。通过引入关系探测方法，直接从语言模型中生成关系图，并联合训练以提升预测效果。**

- **链接: [https://arxiv.org/pdf/2604.10212](https://arxiv.org/pdf/2604.10212)**

> **作者:** Yingjie Niu; Changhong Jin; Rian Dolphin; Ruihai Dong
>
> **备注:** Accpeted by The 2nd Workskop on Advances in Financial AI Workshop: Towards Agentic and Responsible Systems at ICLR 2026
>
> **摘要:** Language models can be used to identify relationships between financial entities in text. However, while structured output mechanisms exist, prompting-based pipelines still incur autoregressive decoding costs and decouple graph construction from downstream optimization. We propose \emph{Relational Probing}, which replaces the standard language-model head with a relation head that induces a relational graph directly from language-model hidden states and is trained jointly with the downstream task model for stock-trend prediction. This approach both learns semantic representations and preserves the strict structure of the induced relational graph. It enables language-model outputs to go beyond text, allowing them to be reshaped into task-specific formats for downstream models. To enhance reproducibility, we provide an operational definition of small language models (SLMs): models that can be fine-tuned end-to-end on a single 24GB GPU under specified batch-size and sequence-length settings. Experiments use Qwen3 backbones (0.6B/1.7B/4B) as upstream SLMs and compare against a co-occurrence baseline. Relational Probing yields consistent performance improvements at competitive inference cost.
>
---
#### [new 052] Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate
- **分类: cs.CL**

- **简介: 该论文属于医疗诊断任务，旨在解决MLLM在诊断中产生的幻觉问题。通过多智能体对抗辩论框架Dialectic-Med，提升诊断的准确性和可信度。**

- **链接: [https://arxiv.org/pdf/2604.11258](https://arxiv.org/pdf/2604.11258)**

> **作者:** Zhixiang Lu; Jionglong Su
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** Multimodal Large Language Models (MLLMs) in healthcare suffer from severe confirmation bias, often hallucinating visual details to support initial, potentially erroneous diagnostic hypotheses. Existing Chain-of-Thought (CoT) approaches lack intrinsic correction mechanisms, rendering them vulnerable to error propagation. To bridge this gap, we propose Dialectic-Med, a multi-agent framework that enforces diagnostic rigor through adversarial dialectics. Unlike static consensus models, Dialectic-Med orchestrates a dynamic interplay between three role-specialized agents: a proponent that formulates diagnostic hypotheses; an opponent equipped with a novel visual falsification module that actively retrieves contradictory visual evidence to challenge the Proponent; and a mediator that resolves conflicts via a weighted consensus graph. By explicitly modeling the cognitive process of falsification, our framework guarantees that diagnostic reasoning is tightly grounded in verified visual regions. Empirical evaluations on MIMIC-CXR-VQA, VQA-RAD, and PathVQA demonstrate that Dialectic-Med not only achieves state-of-the-art performance but also fundamentally enhances the trustworthiness of the reasoning process. Beyond accuracy, our approach significantly enhances explanation faithfulness and decisively mitigates hallucinations, establishing a new standard over single-agent baselines.
>
---
#### [new 053] OccuBench: Evaluating AI Agents on Real-World Professional Tasks via Language World Models
- **分类: cs.CL**

- **简介: 该论文提出OccuBench，用于评估AI代理在真实职业任务中的表现。解决现有基准覆盖领域有限的问题，通过语言世界模型生成专业环境，评估任务完成度和环境鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.10866](https://arxiv.org/pdf/2604.10866)**

> **作者:** Xiaomeng Hu; Yinger Zhang; Fei Huang; Jianhong Tu; Yang Su; Lianghao Deng; Yuxuan Liu; Yantao Liu; Dayiheng Liu; Tsung-Yi Ho
>
> **备注:** 23 pages, 8 figures, 2 tables. Project page: this https URL
>
> **摘要:** AI agents are expected to perform professional work across hundreds of occupational domains (from emergency department triage to nuclear reactor safety monitoring to customs import processing), yet existing benchmarks can only evaluate agents in the few domains where public environments exist. We introduce OccuBench, a benchmark covering 100 real-world professional task scenarios across 10 industry categories and 65 specialized domains, enabled by Language World Models (LWMs) that simulate domain-specific environments through LLM-driven tool response generation. Our multi-agent synthesis pipeline automatically produces evaluation instances with guaranteed solvability, calibrated difficulty, and document-grounded diversity. OccuBench evaluates agents along two complementary dimensions: task completion across professional domains and environmental robustness under controlled fault injection (explicit errors, implicit data degradation, and mixed faults). We evaluate 15 frontier models across 8 model families and find that: (1) no single model dominates all industries, as each has a distinct occupational capability profile; (2) implicit faults (truncated data, missing fields) are harder than both explicit errors (timeouts, 500s) and mixed faults, because they lack overt error signals and require the agent to independently detect data degradation; (3) larger models, newer generations, and higher reasoning effort consistently improve performance. GPT-5.2 improves by 27.5 points from minimal to maximum reasoning effort; and (4) strong agents are not necessarily strong environment simulators. Simulator quality is critical for LWM-based evaluation reliability. OccuBench provides the first systematic cross-industry evaluation of AI agents on professional occupational tasks.
>
---
#### [new 054] LangFlow: Continuous Diffusion Rivals Discrete in Language Modeling
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言建模任务，旨在解决连续扩散模型性能落后于离散模型的问题。通过引入创新方法，构建了首个性能接近离散模型的连续扩散语言模型LangFlow。**

- **链接: [https://arxiv.org/pdf/2604.11748](https://arxiv.org/pdf/2604.11748)**

> **作者:** Yuxin Chen; Chumeng Liang; Hangke Sui; Ruihan Guo; Chaoran Cheng; Jiaxuan You; Ge Liu
>
> **摘要:** Continuous diffusion models have achieved strong performance across domains such as images. However, in language modeling, prior continuous diffusion language models (DLMs) lag behind discrete counterparts. In this work, we close this gap with LangFlow, the first continuous DLM to rival discrete diffusion. Our approach connects embedding-space DLMs to Flow Matching via Bregman divergence and introduces three key innovations: (1) a novel ODE-based NLL bound for principled evaluation of continuous flow-based language models; (2) an information-uniform principle for noise scheduling, motivating a learnable scheduler based on a Gumbel distribution; and (3) an improved training protocol incorporating self-conditioning, which enhances both likelihood and sample this http URL achieves strong performance across benchmarks, reaching a perplexity (PPL) of 30.0 on LM1B and 24.6 on OpenWebText. It matches top discrete DLMs at comparable scale and surpasses autoregressive baselines in zero-shot transfer across multiple benchmarks. LangFlow provides clear evidence that continuous diffusion is a competitive and promising paradigm for language modeling. this https URL
>
---
#### [new 055] Why Don't You Know? Evaluating the Impact of Uncertainty Sources on Uncertainty Quantification in LLMs
- **分类: cs.CL**

- **简介: 该论文属于不确定性量化任务，旨在解决LLMs中多源不确定性对UQ方法影响的问题。通过构建新数据集，分析不同不确定性来源对现有方法效果的影响。**

- **链接: [https://arxiv.org/pdf/2604.10495](https://arxiv.org/pdf/2604.10495)**

> **作者:** Maiya Goloburda; Roman Vashurin; Fedor Chernogorsky; Nurkhan Laiyk; Daniil Orel; Preslav Nakov; Maxim Panov
>
> **摘要:** As Large Language Models (LLMs) are increasingly deployed in real-world applications, reliable uncertainty quantification (UQ) becomes critical for safe and effective use. Most existing UQ approaches for language models aim to produce a single confidence score -- for example, estimating the probability that a model's answer is correct. However, uncertainty in natural language tasks arises from multiple distinct sources, including model knowledge gaps, output variability, and input ambiguity, which have different implications for system behavior and user interaction. In this work, we study how the source of uncertainty impacts the behavior and effectiveness of existing UQ methods. To enable controlled analysis, we introduce a new dataset that explicitly categorizes uncertainty sources, allowing systematic evaluation of UQ performance under each condition. Our experiments reveal that while many UQ methods perform well when uncertainty stems solely from model knowledge limitations, their performance degrades or becomes misleading when other sources are introduced. These findings highlight the need for uncertainty-aware methods that explicitly account for the source of uncertainty in large language models.
>
---
#### [new 056] Human vs. Machine Deception: Distinguishing AI-Generated and Human-Written Fake News Using Ensemble Learning
- **分类: cs.CL**

- **简介: 该论文属于虚假新闻分类任务，旨在区分AI生成与人工编写的假新闻。通过分析文本的语言、结构和情感特征，采用集成学习方法提升分类效果。**

- **链接: [https://arxiv.org/pdf/2604.09960](https://arxiv.org/pdf/2604.09960)**

> **作者:** Samuel Jaeger; Calvin Ibeneye; Aya Vera-Jimenez; Dhrubajyoti Ghosh
>
> **摘要:** The rapid adoption of large language models has introduced a new class of AI-generated fake news that coexists with traditional human-written misinformation, raising important questions about how these two forms of deceptive content differ and how reliably they can be distinguished. This study examines linguistic, structural, and emotional differences between human-written and AI-generated fake news and evaluates machine learning and ensemble-based methods for distinguishing these content types. A document-level feature representation is constructed using sentence structure, lexical diversity, punctuation patterns, readability indices, and emotion-based features capturing affective dimensions such as fear, anger, joy, sadness, trust, and anticipation. Multiple classification models, including logistic regression, random forest, support vector machines, extreme gradient boosting, and a neural network, are applied alongside an ensemble framework that aggregates predictions across models. Model performance is assessed using accuracy and area under the receiver operating characteristic curve. The results show strong and consistent classification performance, with readability-based features emerging as the most informative predictors and AI-generated text exhibiting more uniform stylistic patterns. Ensemble learning provides modest but consistent improvements over individual models. These findings indicate that stylistic and structural properties of text provide a robust basis for distinguishing AI-generated misinformation from human-written fake news.
>
---
#### [new 057] CircuitSynth: Reliable Synthetic Data Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CircuitSynth，解决LLM生成合成数据时的逻辑不一致和覆盖不足问题。通过神经符号框架实现语义推理与表面表达分离，提升数据有效性与覆盖率。**

- **链接: [https://arxiv.org/pdf/2604.10114](https://arxiv.org/pdf/2604.10114)**

> **作者:** Zehua Cheng; Wei Dai; Jiahao Sun; Thomas Lukasiewicz
>
> **备注:** 11 Pages
>
> **摘要:** The generation of high-fidelity synthetic data is a cornerstone of modern machine learning, yet Large Language Models (LLMs) frequently suffer from hallucinations, logical inconsistencies, and mode collapse when tasked with structured generation. Existing approaches, such as prompting or retrieval-augmented generation, lack the mechanisms to balance linguistic expressivity with formal guarantees regarding validity and coverage. To address this, we propose CircuitSynth, a novel neuro-symbolic framework that decouples semantic reasoning from surface realization. By distilling the reasoning capabilities of a Teacher LLM into a Probabilistic Sentential Decision Diagram (PSDD), CircuitSynth creates a tractable semantic prior that structurally enforces hard logical constraints. Furthermore, we introduce a convex optimization mechanism to rigorously satisfy soft distributional goals. Empirical evaluations across diverse benchmarks demonstrate that CircuitSynth achieves 100% Schema Validity even in complex logic puzzles where unconstrained baselines fail (12.4%) while significantly outperforming state-of-the-art methods in rare-combination coverage.
>
---
#### [new 058] Efficient Process Reward Modeling via Contrastive Mutual Information
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于过程奖励建模任务，旨在解决人工标注成本高、计算资源消耗大的问题。提出CPMI方法，利用模型内部概率自动生成奖励标签，显著降低计算负担并提升效率。**

- **链接: [https://arxiv.org/pdf/2604.10660](https://arxiv.org/pdf/2604.10660)**

> **作者:** Nakyung Lee; Sangwoo Hong; Jungwoo Lee
>
> **备注:** Accepted at ACL 2026 Main Conference
>
> **摘要:** Recent research has devoted considerable effort to verifying the intermediate reasoning steps of chain-of-thought (CoT) trajectories using process reward models (PRMs) and other verifier models. However, training a PRM typically requires human annotators to assign reward scores to each reasoning step, which is both costly and time-consuming. Existing automated approaches, such as Monte Carlo (MC) estimation, also demand substantial computational resources due to repeated LLM rollouts. To overcome these limitations, we propose contrastive pointwise mutual information (CPMI), a novel automatic reward labeling method that leverages the model's internal probability to infer step-level supervision while significantly reducing the computational burden of annotating dataset. CPMI quantifies how much a reasoning step increases the mutual information between the step and the correct target answer relative to hard-negative alternatives. This contrastive signal serves as a proxy for the step's contribution to the final solution and yields a reliable reward. The experimental results show that CPMI-based labeling reduces dataset construction time by 84% and token generation by 98% compared to MC estimation, while achieving higher accuracy on process-level evaluations and mathematical reasoning benchmarks.
>
---
#### [new 059] When Verification Fails: How Compositionally Infeasible Claims Escape Rejection
- **分类: cs.CL; cs.AI**

- **简介: 论文研究科学主张验证任务，解决现有基准无法区分严谨验证与简单约束依赖的问题。通过构造复合不可行主张，揭示模型过度接受错误主张的现象。**

- **链接: [https://arxiv.org/pdf/2604.10990](https://arxiv.org/pdf/2604.10990)**

> **作者:** Muxin Liu; Delip Rao; Grace Kim; Chris Callison-Burch
>
> **备注:** 25 pages, 9 figures
>
> **摘要:** Scientific claim verification, the task of determining whether claims are entailed by scientific evidence, is fundamental to establishing discoveries in evidence while preventing misinformation. This process involves evaluating each asserted constraint against validated evidence. Under the Closed-World Assumption (CWA), a claim is accepted if and only if all asserted constraints are positively supported. We show that existing verification benchmarks cannot distinguish models enforcing this standard from models applying a simpler shortcut called salient-constraint checking, which applies CWA's rejection criterion only to the most salient constraint and accepts when that constraint is supported. Because existing benchmarks construct infeasible claims by perturbing a single salient element they are insufficient at distinguishing between rigorous claim verification and simple salient-constraint reliance. To separate the two, we construct compositionally infeasible claims where the salient constraint is supported but a non-salient constraint is contradicted. Across model families and modalities, models that otherwise saturate existing benchmarks consistently over-accept these claims, confirming the prevalence of such shortcut reasoning. Via model context interventions, we show that different models and prompting strategies occupy distinct positions on a shared ROC curve, indicating that the gap between model families reflects differences in verification threshold rather than underlying reasoning ability, and that the compositional inference bottleneck is a structural property of current verification behavior that strategy guidance alone cannot overcome.
>
---
#### [new 060] CLSGen: A Dual-Head Fine-Tuning Framework for Joint Probabilistic Classification and Verbalized Explanation
- **分类: cs.CL**

- **简介: 该论文提出CLSGen框架，解决LLM在分类任务中无法可靠生成概率和解释的问题。通过双头结构实现准确分类与可解释性。**

- **链接: [https://arxiv.org/pdf/2604.11801](https://arxiv.org/pdf/2604.11801)**

> **作者:** WonJin Yoon; Kangyu Zhu; Ian Bulovic; Autumn Sehy; Yanjun Gao; Dmitriy Dligach; Majid Afshar; Timothy A. Miller
>
> **摘要:** With the recent progress of Large Language Models (LLMs), there is a growing interest in applying these models to solve complex and challenging problems. Modern LLMs, capable of processing long contexts and generating verbalized explanations, offer significant potential in addressing real-world applications. However, a critical hurdle in deploying LLMs for practical decision-making is their inability to provide reliable, quantitative probabilities. While task-specific fine-tuning of LLMs using traditional discriminative objectives (similar to encoder-only models) can yield probability estimates, this often leads to catastrophic forgetting and linguistic collapse. Consequently, the model loses its ability to generate explanations, severely undermining its interpretability and usability. To address this challenge, we propose CLSGen, a novel LLM fine-tuning framework designed for binary classification tasks. The CLSGen framework encompasses a new model architecture, training methodology, and data construction strategy to enable robust probability estimation without sacrificing the model's inherent explanation-generation capabilities. Experimental results across multiple benchmark datasets demonstrate that models fine-tuned with CLSGen outperform existing baselines in classification metrics (AUROC and F1-score). Regarding explanation, the results showed strong alignment between predicted labels and generated justifications, as well as high readability.
>
---
#### [new 061] ASPIRin: Action Space Projection for Interactivity-Optimized Reinforcement Learning in Full-Duplex Speech Language Models
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音交互任务，解决全双工语音语言模型中对话时序优化问题。通过ASPIRin框架分离说话时机与内容，提升交互性并减少重复。**

- **链接: [https://arxiv.org/pdf/2604.10065](https://arxiv.org/pdf/2604.10065)**

> **作者:** Chi-Yuan Hsiao; Ke-Han Lu; Yu-Kuan Fu; Guan-Ting Lin; Hsiao-Tsung Hung; Hung-yi Lee
>
> **摘要:** End-to-end full-duplex Speech Language Models (SLMs) require precise turn-taking for natural interaction. However, optimizing temporal dynamics via standard raw-token reinforcement learning (RL) degrades semantic quality, causing severe generative collapse and repetition. We propose ASPIRin, an interactivity-optimized RL framework that explicitly decouples when to speak from what to say. Using Action Space Projection, ASPIRin maps the text vocabulary into a coarse-grained binary state (active speech vs. inactive silence). By applying Group Relative Policy Optimization (GRPO) with rule-based rewards, it balances user interruption and response latency. Empirical evaluations show ASPIRin optimizes interactivity across turn-taking, backchanneling, and pause handling. Crucially, isolating timing from token selection preserves semantic coherence and reduces the portion of duplicate n-grams by over 50% compared to standard GRPO, effectively eliminating degenerative repetition.
>
---
#### [new 062] Claim2Vec: Embedding Fact-Check Claims for Multilingual Similarity and Clustering
- **分类: cs.CL**

- **简介: 该论文提出Claim2Vec，解决多语言事实核查声明的聚类问题。通过优化嵌入空间提升相似声明的表示，增强跨语言知识迁移效果。**

- **链接: [https://arxiv.org/pdf/2604.09812](https://arxiv.org/pdf/2604.09812)**

> **作者:** Rrubaa Panchendrarajan; Arkaitz Zubiaga
>
> **摘要:** Recurrent claims present a major challenge for automated fact-checking systems designed to combat misinformation, especially in multilingual settings. While tasks such as claim matching and fact-checked claim retrieval aim to address this problem by linking claim pairs, the broader challenge of effectively representing groups of similar claims that can be resolved with the same fact-check via claim clustering remains relatively underexplored. To address this gap, we introduce Claim2Vec, the first multilingual embedding model optimized to represent fact-check claims as vectors in an improved semantic embedding space. We fine-tune a multilingual encoder using contrastive learning with similar multilingual claim pairs. Experiments on the claim clustering task using three datasets, 14 multilingual embedding models, and 7 clustering algorithms demonstrate that Claim2Vec significantly improves clustering performance. Specifically, it enhances both cluster label alignment and the geometric structure of the embedding space across different cluster configurations. Our multilingual analysis shows that clusters containing multiple languages benefit from fine-tuning, demonstrating cross-lingual knowledge transfer.
>
---
#### [new 063] CoSToM:Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的社会推理任务，旨在解决大模型内在认知与外在行为不一致的问题。通过因果引导方法提升模型的社会推理能力。**

- **链接: [https://arxiv.org/pdf/2604.10031](https://arxiv.org/pdf/2604.10031)**

> **作者:** Mengfan Li; Xuanhua Shi; Yang Deng
>
> **备注:** Accepted to ACL 2026 (Main Conference)
>
> **摘要:** Theory of Mind (ToM), the ability to attribute mental states to others, is a hallmark of social intelligence. While large language models (LLMs) demonstrate promising performance on standard ToM benchmarks, we observe that they often fail to generalize to complex task-specific scenarios, relying heavily on prompt scaffolding to mimic reasoning. The critical misalignment between the internal knowledge and external behavior raises a fundamental question: Do LLMs truly possess intrinsic cognition, and can they externalize this internal knowledge into stable, high-quality behaviors? To answer this, we introduce CoSToM (Causal-oriented Steering for ToM alignment), a framework that transitions from mechanistic interpretation to active intervention. First, we employ causal tracing to map the internal distribution of ToM features, empirically uncovering the internal layers' characteristics in encoding fundamental ToM semantics. Building on this insight, we implement a lightweight alignment framework via targeted activation steering within these ToM-critical layers. Experiments demonstrate that CoSToM significantly enhances human-like social reasoning capabilities and downstream dialogue quality.
>
---
#### [new 064] ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification
- **分类: cs.CL; cs.DB; cs.IR; cs.MA**

- **简介: 该论文提出ODUTQA-MDC任务，解决开放域表格问答中模糊表达的问题。构建了基准数据集和动态澄清接口，并提出MAIC-TQA框架进行多轮对话澄清与答案优化。**

- **链接: [https://arxiv.org/pdf/2604.10159](https://arxiv.org/pdf/2604.10159)**

> **作者:** Zhensheng Wang; ZhanTeng Lin; Wenmian Yang; Kun Zhou; Yiquan Zhang; Weijia Jia
>
> **备注:** This paper has been accepted to the main conference of ACL 2026
>
> **摘要:** The advancement of large language models (LLMs) has enhanced tabular question answering (Tabular QA), yet they struggle with open-domain queries exhibiting underspecified or uncertain expressions. To address this, we introduce the ODUTQA-MDC task and the first comprehensive benchmark to tackle it. This benchmark includes: (1) a large-scale ODUTQA dataset with 209 tables and 25,105 QA pairs; (2) a fine-grained labeling scheme for detailed evaluation; and (3) a dynamic clarification interface that simulates user feedback for interactive assessment. We also propose MAIC-TQA, a multi-agent framework that excels at detecting ambiguities, clarifying them through dialogue, and refining answers. Experiments validate our benchmark and framework, establishing them as a key resource for advancing conversational, underspecification-aware Tabular QA research.
>
---
#### [new 065] Linguistic Accommodation Between Neurodivergent Communities on Reddit:A Communication Accommodation Theory Analysis of ADHD and Autism Groups
- **分类: cs.CL**

- **简介: 该论文分析Reddit上ADHD与自闭症群体的语言适应现象，属于社会媒体与语言学交叉研究，旨在探讨神经多样性群体间的语言互动机制。**

- **链接: [https://arxiv.org/pdf/2604.10063](https://arxiv.org/pdf/2604.10063)**

> **作者:** Saad Mankarious; Nour Zein; Iyad Ait Hou; Aya Zirikly
>
> **摘要:** Social media research on mental health has focused predominantly on detecting and diagnosing conditions at the individual level. In this work, we shift attention to \emph{intergroup} behavior, examining how two prominent neurodivergent communities, ADHD and autism, adjust their language when engaging with each other on Reddit. Grounded in Communication Accommodation Theory (CAT), we first establish that each community maintains a distinct linguistic profile as measured by Language Inquiry and Word Count Lexicon (LIWC). We then show that these profiles shift in opposite directions when users cross community boundaries: features that are elevated in one group's home community decrease when its members post in the other group's space, and vice versa, consistent with convergent accommodation. The involvement of topic-independent summary variables (Authentic, Clout) in these shifts provides partial evidence against a purely topical explanation. Finally, in an exploratory longitudinal analysis around the moment of public diagnosis disclosure, we find that its effects on linguistic style are small and, in some cases, directionally opposite to cross-community accommodation, providing initial evidence that situational audience adaptation and longer-term identity processes may involve different mechanisms. Our findings contribute to understanding intergroup communication dynamics among neurodivergent populations online and carry implications for community moderation and clinical perspectives on these conditions.
>
---
#### [new 066] When Valid Signals Fail: Regime Boundaries Between LLM Features and RL Trading Policies
- **分类: cs.CL; cs.AI; cs.CE**

- **简介: 该论文研究LLM生成特征对RL交易策略的影响，解决特征有效性与策略鲁棒性不一致的问题。通过优化提示词提取有效特征，但发现其在宏观冲击下效果下降。任务属于强化学习与金融交易的交叉领域。**

- **链接: [https://arxiv.org/pdf/2604.10996](https://arxiv.org/pdf/2604.10996)**

> **作者:** Zhengzhe Yang
>
> **摘要:** Can large language models (LLMs) generate continuous numerical features that improve reinforcement learning (RL) trading agents? We build a modular pipeline where a frozen LLM serves as a stateless feature extractor, transforming unstructured daily news and filings into a fixed-dimensional vector consumed by a downstream PPO agent. We introduce an automated prompt-optimization loop that treats the extraction prompt as a discrete hyperparameter and tunes it directly against the Information Coefficient - the Spearman rank correlation between predicted and realized returns - rather than NLP losses. The optimized prompt discovers genuinely predictive features (IC above 0.15 on held-out data). However, these valid intermediate representations do not automatically translate into downstream task performance: during a distribution shift caused by a macroeconomic shock, LLM-derived features add noise, and the augmented agent under-performs a price-only baseline. In a calmer test regime the agent recovers, yet macroeconomic state variables remain the most robust driver of policy improvement. Our findings highlight a gap between feature-level validity and policy-level robustness that parallels known challenges in transfer learning under distribution shift.
>
---
#### [new 067] Think Before you Write: QA-Guided Reasoning for Character Descriptions in Books
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于文本生成任务，旨在解决长篇叙事中准确生成角色描述的问题。通过分离推理与生成过程，提升描述的准确性与合理性。**

- **链接: [https://arxiv.org/pdf/2604.11435](https://arxiv.org/pdf/2604.11435)**

> **作者:** Argyrios Papoudakis; Mirella Lapata; Frank Keller
>
> **备注:** 20 pages, 16 tables, 1 figure
>
> **摘要:** Character description generation is an important capability for narrative-focused applications such as summarization, story analysis, and character-driven simulations. However, generating accurate character descriptions from long-form narratives (e.g., novels) is challenging: models must track evolving attributes (e.g., relationships and events), integrate evidence scattered across the text, and infer implicit details. Despite the success of reasoning-enabled LLMs on many benchmarks, we find that for character description generation their performance improves when built-in reasoning is disabled (i.e., an empty reasoning trace). Motivated by this, we propose a training framework that decouples reasoning from generation. Our approach, which can be applied on top of long-context LLMs or chunk-based methods, consists of a reasoning model that produces a structured QA reasoning trace and a generation model that conditions on this trace to produce the final character description. Experiments on two datasets (BookWorm and CroSS) show that QA-guided reasoning improves faithfulness, informativeness, and grounding over strong long-context baselines.
>
---
#### [new 068] Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs在工具调用中的结构对齐偏差问题。研究提出SABEval数据集，分析并缓解LLMs在无关工具情况下仍错误调用的现象。**

- **链接: [https://arxiv.org/pdf/2604.11322](https://arxiv.org/pdf/2604.11322)**

> **作者:** Yilong Liu; Xixun Lin; Pengfei Cao; Ge Zhang; Fang Fang; Yanan Cao
>
> **备注:** Accepted to ACL 2026 (Main Conference)
>
> **摘要:** Large language models (LLMs) have demonstrated impressive capabilities in utilizing external tools. In practice, however, LLMs are often exposed to tools that are irrelevant to the user's query, in which case the desired behavior is to refrain from invocations. In this work, we identify a widespread yet overlooked mechanistic flaw in tool refusal, which we term structural alignment bias: Even when a tool fails to serve the user's goal, LLMs still tend to invoke it whenever query attributes can be validly assigned to tool parameters. To systematically study this bias, we introduce SABEval, a new dataset that decouples structural alignment from semantic relevance. Our analysis shows that structural alignment bias induces severe tool-invocation errors in LLMs, yet remains largely unaccounted for in existing evaluations. To investigate the internal mechanisms underlying this bias, we propose Contrastive Attention Attribution, which reveals two competing pathways for semantic checking and structural matching. The relative strength of these pathways drives LLMs' tool invocation decisions. Based on these findings, we further introduce a rebalancing strategy that effectively mitigates structural alignment bias, as demonstrated by extensive experiments, without degrading general tool-use capabilities.
>
---
#### [new 069] HistLens: Mapping Idea Change across Concepts and Corpora
- **分类: cs.CL**

- **简介: 该论文提出HistLens框架，用于多概念、多语料的观念演变分析，解决跨语料和隐含概念的计算问题。属于历史语义分析任务。**

- **链接: [https://arxiv.org/pdf/2604.11749](https://arxiv.org/pdf/2604.11749)**

> **作者:** Yi Jing; Weiyun Qiu; Yihang Peng; Zhifang Sui
>
> **备注:** Accepted by ACL 2026 MainConference
>
> **摘要:** Language change both reflects and shapes social processes, and the semantic evolution of foundational concepts provides a measurable trace of historical and social transformation. Despite recent advances in diachronic semantics and discourse analysis, existing computational approaches often (i) concentrate on a single concept or a single corpus, making findings difficult to compare across heterogeneous sources, and (ii) remain confined to surface lexical evidence, offering insufficient computational and interpretive granularity when concepts are expressed implicitly. We propose HistLens, a unified, SAE-based framework for multi-concept, multi-corpus conceptual-history analysis. The framework decomposes concept representations into interpretable features and tracks their activation dynamics over time and across sources, yielding comparable conceptual trajectories within a shared coordinate system. Experiments on long-span press corpora show that HistLens supports cross-concept, cross-corpus computation of patterns of idea evolution and enables implicit concept computation. By bridging conceptual modeling with interpretive needs, HistLens broadens the analytical perspectives and methodological repertoire available to social science and the humanities for diachronic text analysis.
>
---
#### [new 070] Shared Emotion Geometry Across Small Language Models: A Cross-Architecture Study of Representation, Behavior, and Methodological Confounds
- **分类: cs.CL; cs.AI**

- **简介: 论文研究小语言模型的情感表示一致性，分析不同架构模型在情感几何结构上的相似性及方法论影响，旨在揭示模型行为与情感表示的关系。**

- **链接: [https://arxiv.org/pdf/2604.11050](https://arxiv.org/pdf/2604.11050)**

> **作者:** Jihoon Jeong
>
> **备注:** 34 pages, 6 figures, 1 table in main text + appendix. Ongoing series on Model Medicine
>
> **摘要:** We extract 21-emotion vector sets from twelve small language models (six architectures x base/instruct, 1B-8B parameters) under a unified comprehension-mode pipeline at fp16 precision, and compare the resulting geometries via representational similarity analysis on raw cosine RDMs. The five mature architectures (Qwen 2.5 1.5B, SmolLM2 1.7B, Llama 3.2 3B, Mistral 7B v0.3, Llama 3.1 8B) share nearly identical 21-emotion geometry, with pairwise RDM Spearman correlations of 0.74-0.92. This universality persists across diametrically opposed behavioral profiles: Qwen 2.5 and Llama 3.2 occupy opposite poles of MTI Compliance facets yet produce nearly identical emotion RDMs (rho = 0.81), so behavioral facet differences arise above the shared emotion representation. Gemma-3 1B base, the one immature case in our dataset, exhibits extreme residual-stream anisotropy (0.997) and is restructured by RLHF across all geometric descriptors, whereas the five already-mature families show within-family base x instruct RDM correlations of rho >= 0.92 (Mistral 7B v0.3 at rho = 0.985), suggesting RLHF restructures only representations that are not yet organized. Methodologically, we show that what prior work has read as a single comprehension-vs-generation method effect in fact decomposes into four distinct layers -- a coarse method-dependent dissociation, robust sub-parameter sensitivity within generation, a true precision (fp16 vs INT8) effect, and a conflated cross-experiment bias that distorts in opposite directions for different models -- so that a single rho between two prior emotion-vector studies is not a safe basis for interpretation without the layered decomposition.
>
---
#### [new 071] Dynamic Adaptive Attention and Supervised Contrastive Learning: A Novel Hybrid Framework for Text Sentiment Classification
- **分类: cs.CL**

- **简介: 该论文属于文本情感分类任务，旨在解决长文本中语义依赖和情感表达模糊的问题。提出融合动态注意力与对比学习的框架，提升分类效果。**

- **链接: [https://arxiv.org/pdf/2604.10459](https://arxiv.org/pdf/2604.10459)**

> **作者:** Qingyang Li
>
> **摘要:** The exponential growth of user-generated movie reviews on digital platforms has made accurate text sentiment classification a cornerstone task in natural language processing. Traditional models, including standard BERT and recurrent architectures, frequently struggle to capture long-distance semantic dependencies and resolve ambiguous emotional expressions in lengthy review texts. This paper proposes a novel hybrid framework that seamlessly integrates dynamic adaptive multi-head attention with supervised contrastive learning into a BERT-based Transformer encoder. The dynamic adaptive attention module employs a global context pooling vector to dynamically regulate the contribution of each attention head, thereby focusing on critical sentiment-bearing tokens while suppressing noise. Simultaneously, the supervised contrastive learning branch enforces tighter intra-class compactness and larger inter-class separation in the embedding space. Extensive experiments on the IMDB dataset demonstrate that the proposed model achieves competitive performance with an accuracy of 94.67\%, outperforming strong baselines by 1.5--2.5 percentage points. The framework is lightweight, efficient, and readily extensible to other text classification tasks.
>
---
#### [new 072] Exploring Knowledge Conflicts for Faithful LLM Reasoning: Benchmark and Method
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识推理任务，旨在解决LLMs在面对跨源知识冲突时的忠实推理问题。提出ConflictQA基准和XoT方法以提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2604.11209](https://arxiv.org/pdf/2604.11209)**

> **作者:** Tianzhe Zhao; Jiaoyan Chen; Shuxiu Zhang; Haiping Zhu; Qika Lin; Jun Liu
>
> **备注:** Accepted at SIGIR 2026
>
> **摘要:** Large language models (LLMs) have achieved remarkable success across a wide range of applications especially when augmented by external knowledge through retrieval-augmented generation (RAG). Despite their widespread adoption, recent studies have shown that LLMs often struggle to perform faithful reasoning when conflicting knowledge is retrieved. However, existing work primarily focuses on conflicts between external knowledge and the parametric knowledge of LLMs, leaving conflicts across external knowledge largely unexplored. Meanwhile, modern RAG systems increasingly emphasize the integration of unstructured text and (semi-)structured data like knowledge graphs (KGs) to improve knowledge completeness and reasoning faithfulness. To address this gap, we introduce ConflictQA, a novel benchmark that systematically instantiates conflicts between textual evidence and KG evidence. Extensive evaluations across representative LLMs reveal that, facing such cross-source conflicts, LLMs often fail to identify reliable evidence for correct reasoning. Instead, LLMs become more sensitive to prompting choices and tend to rely exclusively on either KG or textual evidence, resulting in incorrect responses. Based on these findings, we further propose XoT, a two-stage explanation-based thinking framework tailored for reasoning over heterogeneous conflicting evidence, and verify its effectiveness with extensive experiments.
>
---
#### [new 073] RCBSF: A Multi-Agent Framework for Automated Contract Revision via Stackelberg Game
- **分类: cs.CL**

- **简介: 该论文属于法律AI任务，旨在解决LLM在合同自动修订中的安全性和约束问题。提出RCBSF框架，通过Stackelberg博弈实现风险约束下的优化修订。**

- **链接: [https://arxiv.org/pdf/2604.10740](https://arxiv.org/pdf/2604.10740)**

> **作者:** Shijia Xu; Yu Wang; Xiaolong Jia; Zhou Wu; Kai Liu; April Xiaowen Dong
>
> **摘要:** Despite the widespread adoption of Large Language Models (LLMs) in Legal AI, their utility for automated contract revision remains impeded by hallucinated safety and a lack of rigorous behavioral constraints. To address these limitations, we propose the Risk-Constrained Bilevel Stackelberg Framework (RCBSF), which formulates revision as a non-cooperative Stackelberg game. RCBSF establishes a hierarchical Leader Follower structure where a Global Prescriptive Agent (GPA) imposes risk budgets upon a follower system constituted by a Constrained Revision Agent (CRA) and a Local Verification Agent (LVA) to iteratively optimize output. We provide theoretical guarantees that this bilevel formulation converges to an equilibrium yielding strictly superior utility over unguided configurations. Empirical validation on a unified benchmark demonstrates that RCBSF achieves state-of-the-art performance, surpassing iterative baselines with an average Risk Resolution Rate (RRR) of 84.21\% while enhancing token efficiency. Our code is available at this https URL .
>
---
#### [new 074] SHARE: Social-Humanities AI for Research and Education
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在为社会科学与人文学科（SSH）开发专用语言模型。通过构建SHARE模型和MIRROR界面，解决SSH领域文本建模效率低及AI生成内容合规性问题。**

- **链接: [https://arxiv.org/pdf/2604.11152](https://arxiv.org/pdf/2604.11152)**

> **作者:** João Gonçalves; Sonia de Jager; Petr Knoth; David Pride; Nick Jelicic
>
> **备注:** 23 pages, 9 figures, 4 tables
>
> **摘要:** This intermediate technical report introduces the SHARE family of base models and the MIRROR user interface. The SHARE models are the first causal language models fully pretrained by and for the social sciences and humanities (SSH). Their performance in modelling SSH texts is close to that of general purpose models (Phi-4) which use 100 times more tokens, as shown by our custom SSH Cloze benchmark. The MIRROR user interface is designed for reviewing text inputs from the SSH disciplines while preserving critical engagement. By prototyping a generative AI interface that does not generate any text, we propose a way to harness the capabilities of the SHARE models without compromising the integrity of SSH principles and norms.
>
---
#### [new 075] RUMLEM: A Dictionary-Based Lemmatizer for Romansh
- **分类: cs.CL**

- **简介: 该论文介绍RUMLEM，一个针对罗曼什语的词形还原工具，解决词形还原与方言分类问题，基于社区驱动数据库，覆盖多种方言并实现高准确率分类。**

- **链接: [https://arxiv.org/pdf/2604.11233](https://arxiv.org/pdf/2604.11233)**

> **作者:** Dominic P. Fischer; Zachary Hopton; Jannis Vamvas
>
> **摘要:** Lemmatization -- the task of mapping an inflected word form to its dictionary form -- is a crucial component of many NLP applications. In this paper, we present RUMLEM, a lemmatizer that covers the five main varieties of Romansh as well as the supra-regional standard variety Rumantsch Grischun. It is based on comprehensive, community-driven morphological databases for Romansh, enabling RUMLEM to cover 77-84% of the words in a typical Romansh text. Since there is a dedicated database for each Romansh variety, an additional application of RUMLEM is variety-aware language classification. Evaluation on 30'000 Romansh texts of varying lengths shows that RUMLEM correctly identifies the variety in 95% of cases. In addition, a proof of concept demonstrates the feasibility of Romansh vs. non-Romansh language classification based on the lemmatizer.
>
---
#### [new 076] TInR: Exploring Tool-Internalized Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型推理任务，旨在解决工具使用效率低的问题。通过内部化工具知识，提出TInR-U框架提升推理效果与效率。**

- **链接: [https://arxiv.org/pdf/2604.10788](https://arxiv.org/pdf/2604.10788)**

> **作者:** Qiancheng Xu; Yongqi Li; Fan Liu; Hongru Wang; Min Yang; Wenjie Li
>
> **摘要:** Tool-Integrated Reasoning (TIR) has emerged as a promising direction by extending Large Language Models' (LLMs) capabilities with external tools during reasoning. Existing TIR methods typically rely on external tool documentation during reasoning. However, this leads to tool mastery difficulty, tool size constraints, and inference inefficiency. To mitigate these issues, we explore Tool-Internalized Reasoning (TInR), aiming at facilitating reasoning with tool knowledge internalized into LLMs. Achieving this goal presents notable requirements, including tool internalization and tool-reasoning coordination. To address them, we propose TInR-U, a tool-internalized reasoning framework for unified reasoning and tool usage. TInR-U is trained through a three-phase pipeline: 1) tool internalization with a bidirectional knowledge alignment strategy; 2) supervised fine-tuning warm-up using high-quality reasoning annotations, and 3) reinforcement learning with TInR-specific rewards. We comprehensively evaluate our method across in-domain and out-of-domain settings. Experiment results show that TInR-U achieves superior performance in both settings, highlighting its effectiveness and efficiency.
>
---
#### [new 077] Please Make it Sound like Human: Encoder-Decoder vs. Decoder-Only Transformers for AI-to-Human Text Style Transfer
- **分类: cs.CL**

- **简介: 该论文属于文本风格迁移任务，旨在将AI生成文本改写为更像人类写作。通过构建语料库并对比不同模型效果，研究模型在风格转换中的表现与评估问题。**

- **链接: [https://arxiv.org/pdf/2604.11687](https://arxiv.org/pdf/2604.11687)**

> **作者:** Utsav Paneru
>
> **备注:** 12 pages, 3 figures, 2 tables
>
> **摘要:** AI-generated text has become common in academic and professional writing, prompting research into detection methods. Less studied is the reverse: systematically rewriting AI-generated prose to read as genuinely human-authored. We build a parallel corpus of 25,140 paired AI-input and human-reference text chunks, identify 11 measurable stylistic markers separating the two registers, and fine-tune three models: BART-base, BART-large, and Mistral-7B-Instruct with QLoRA. BART-large achieves the highest reference similarity -- BERTScore F1 of 0.924, ROUGE-L of 0.566, and chrF++ of 55.92 -- with 17x fewer parameters than Mistral-7B. We show that Mistral-7B's higher marker shift score reflects overshoot rather than accuracy, and argue that shift accuracy is a meaningful blind spot in current style transfer evaluation.
>
---
#### [new 078] Retrieval as Generation: A Unified Framework with Self-Triggered Information Planning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于问答任务，解决传统检索增强生成中检索与生成分离的问题，提出GRIP框架，通过生成引导检索实现端到端协同。**

- **链接: [https://arxiv.org/pdf/2604.11407](https://arxiv.org/pdf/2604.11407)**

> **作者:** Bo Li; Mingda Wang; Gexiang Fang; Shikun Zhang; Wei Ye
>
> **备注:** Github: this https URL HuggingFace:this https URL
>
> **摘要:** We revisit retrieval-augmented generation (RAG) by embedding retrieval control directly into generation. Instead of treating retrieval as an external intervention, we express retrieval decisions within token-level decoding, enabling end-to-end coordination without additional controllers or classifiers. Under the paradigm of Retrieval as Generation, we propose \textbf{GRIP} (\textbf{G}eneration-guided \textbf{R}etrieval with \textbf{I}nformation \textbf{P}lanning), a unified framework in which the model regulates retrieval behavior through control-token emission. Central to GRIP is \textit{Self-Triggered Information Planning}, which allows the model to decide when to retrieve, how to reformulate queries, and when to terminate, all within a single autoregressive trajectory. This design tightly couples retrieval and reasoning and supports dynamic multi-step inference with on-the-fly evidence integration. To supervise these behaviors, we construct a structured training set covering answerable, partially answerable, and multi-hop queries, each aligned with specific token patterns. Experiments on five QA benchmarks show that GRIP surpasses strong RAG baselines and is competitive with GPT-4o while using substantially fewer parameters.
>
---
#### [new 079] NOSE: Neural Olfactory-Semantic Embedding with Tri-Modal Orthogonal Contrastive Learning
- **分类: cs.CL**

- **简介: 该论文提出NOSE框架，解决嗅觉多模态表示学习问题，通过对齐分子结构、受体序列和语言描述，提升模型的生物合理性和语义可解释性。**

- **链接: [https://arxiv.org/pdf/2604.10452](https://arxiv.org/pdf/2604.10452)**

> **作者:** Yanyi Su; Hongshuai Wang; Zhifeng Gao; Jun Cheng
>
> **备注:** Accepted to the ACL 2026 Main Conference
>
> **摘要:** Olfaction lies at the intersection of chemical structure, neural encoding, and linguistic perception, yet existing representation methods fail to fully capture this pathway. Current approaches typically model only isolated segments of the olfactory pathway, overlooking the complete chain from molecule to receptors to linguistic descriptions. Such fragmentation yields learned embeddings that lack both biological grounding and semantic interpretability. We propose NOSE (Neural Olfactory-Semantic Embedding), a representation learning framework that aligns three modalities along the olfactory pathway: molecular structure, receptor sequence, and natural language description. Rather than simply fusing these signals, we decouple their contributions via orthogonal constraints, preserving the unique encoded information of each modality. To address the sparsity of olfactory language, we introduce a weak positive sample strategy to calibrate semantic similarity, preventing erroneous repulsion of similar odors in the feature space. Extensive experiments demonstrate that NOSE achieves state-of-the-art (SOTA) performance and excellent zero-shot generalization, confirming the strong alignment between its representation space and human olfactory intuition.
>
---
#### [new 080] Self-Calibrating Language Models via Test-Time Discriminative Distillation
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出SECL方法，通过测试时训练提升语言模型的校准能力，解决模型过自信问题，无需标注数据，效果显著。**

- **链接: [https://arxiv.org/pdf/2604.09624](https://arxiv.org/pdf/2604.09624)**

> **作者:** Mohamed Rissal Hedna; Jan Strich; Martin Semmann; Chris Biemann
>
> **备注:** Submitted to ACL March 26
>
> **摘要:** Large language models (LLMs) are systematically overconfident: they routinely express high certainty on questions they often answer incorrectly. Existing calibration methods either require labeled validation data, degrade under distribution shifts, or incur substantial inference costs. Recent work has shown that LLMs already contain a better-calibrated signal than the one they verbalize: the token probability of "True" when the model is asked "Is this answer correct?" ($P(\text{True})$) consistently outperforms their stated confidence, a gap that is theoretically grounded as generative error is lower-bounded by roughly twice the corresponding discriminative error. We introduce $\textbf{SECL}$ ($\textbf{SE}$lf-$\textbf{C}$alibrating $\textbf{L}$anguage Models), a test-time training (TTT) pipeline that exploits this gap as label-free self-supervision, requiring no labeled data or human supervision. SECL adapts only when the input distribution shifts, training on just 6--26% of the question stream at lower cost than the baseline it distills from. Across four small language models from three model families and four diverse domains, SECL reduces Expected Calibration Error (ECE) by 56--78%, outperforming its own supervision signal and matching or outperforming recent inference-time methods. SECL is the first method to apply TTT to calibration; seven ablations covering signal quality, gating strategy, weight accumulation, loss design, domain ordering, hyperparameter sensitivity, and layer selection confirm that each component is crucial and robust across configurations. Code: this https URL
>
---
#### [new 081] Playing Along: Learning a Double-Agent Defender for Belief Steering via Theory of Mind
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出ToM-SB任务，解决AI在对话中欺骗对手、引导其错误信念的问题。通过双代理模型和强化学习，提升AI的理论心智与欺骗能力。**

- **链接: [https://arxiv.org/pdf/2604.11666](https://arxiv.org/pdf/2604.11666)**

> **作者:** Hanqi Xiao; Vaidehi Patil; Zaid Khan; Hyunji Lee; Elias Stengel-Eskin; Mohit Bansal
>
> **备注:** First two authors contributed equally. Code: this https URL
>
> **摘要:** As large language models (LLMs) become the engine behind conversational systems, their ability to reason about the intentions and states of their dialogue partners (i.e., form and use a theory-of-mind, or ToM) becomes increasingly critical for safe interaction with potentially adversarial partners. We propose a novel privacy-themed ToM challenge, ToM for Steering Beliefs (ToM-SB), in which a defender must act as a Double Agent to steer the beliefs of an attacker with partial prior knowledge within a shared universe. To succeed on ToM-SB, the defender must engage with and form a ToM of the attacker, with a goal of fooling the attacker into believing they have succeeded in extracting sensitive information. We find that strong frontier models like Gemini3-Pro and GPT-5.4 struggle on ToM-SB, often failing to fool attackers in hard scenarios with partial attacker prior knowledge, even when prompted to reason about the attacker's beliefs (ToM prompting). To close this gap, we train models on ToM-SB to act as AI Double Agents using reinforcement learning, testing both fooling and ToM rewards. Notably, we find a bidirectionally emergent relationship between ToM and attacker-fooling: rewarding fooling success alone improves ToM, and rewarding ToM alone improves fooling. Across four attackers with different strengths, six defender methods, and both in-distribution and out-of-distribution (OOD) evaluation, we find that gains in ToM and attacker-fooling are well-correlated, highlighting belief modeling as a key driver of success on ToM-SB. AI Double Agents that combine both ToM and fooling rewards yield the strongest fooling and ToM performance, outperforming Gemini3-Pro and GPT-5.4 with ToM prompting on hard scenarios. We also show that ToM-SB and AI Double Agents can be extended to stronger attackers, demonstrating generalization to OOD settings and the upgradability of our task.
>
---
#### [new 082] A Triadic Suffix Tokenization Scheme for Numerical Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理中的数值推理任务，旨在解决传统分词方法导致数字结构丢失的问题。提出Triadic Suffix Tokenization方案，通过三元分组和量级标记提升模型对数值的理解与计算能力。**

- **链接: [https://arxiv.org/pdf/2604.11582](https://arxiv.org/pdf/2604.11582)**

> **作者:** Olga Chetverina
>
> **备注:** 8 pages, 1 figure. This is a theoretical proposal of a novel numbers tokenization for LLMs. The code is available on GitHub. Previous version archived at Zenodo: DOI https://doi.org/10.5281/zenodo.18999577
>
> **摘要:** Standard subword tokenization methods fragment numbers inconsistently, causing large language models (LLMs) to lose positional and decimal structure - a primary driver of errors in arithmetic and scientific reasoning. We introduce Triadic Suffix Tokenization (TST), a deterministic scheme that partitions digits into three-digit triads and annotates each triad with an explicit magnitude marker. Critically, the scheme defines a fixed, one-to-one mapping between suffixes and orders of magnitude for the integer part (thousands, millions, billions, etc.) and a parallel system of replicated markers for fractional depth (tenths, thousandths, millionths, etc.). Unlike approaches that rely on positional inference, this method provides a consistent gradient signal, which should ensure stable convergence. Two implementation variants are proposed: (1) a vocabulary-based approach that adds at most 10,000 fixed tokens to an existing vocabulary, covering 33 orders of magnitude ($10^{-15}$ to $10^{18}$); and (2) a suffix-marker approach that uses a small set of special tokens to denote magnitude dynamically. Both variants preserve exact digits while making order-of-magnitude relationships transparent at the token level. The framework is inherently scalable, allowing for linear vocabulary expansion to accommodate arbitrary precision and range. TST is architecture-agnostic and can be integrated as a drop-in preprocessing step. Experimental validation is deferred to future work.
>
---
#### [new 083] Structure-Grounded Knowledge Retrieval via Code Dependencies for Multi-Step Data Reasoning
- **分类: cs.CL**

- **简介: 该论文属于数据推理任务，解决多步分析中知识检索不准确的问题。通过代码依赖结构构建知识图谱，提升LLM的推理准确性。**

- **链接: [https://arxiv.org/pdf/2604.10516](https://arxiv.org/pdf/2604.10516)**

> **作者:** Xinyi Huang; Mingzhe Lu; Haoyu Dong
>
> **摘要:** Selecting the right knowledge is critical when using large language models (LLMs) to solve domain-specific data analysis tasks. However, most retrieval-augmented approaches rely primarily on lexical or embedding similarity, which is often a weak proxy for the task-critical knowledge needed for multi-step reasoning. In many such tasks, the relevant knowledge is not merely textually related to the query, but is instead grounded in executable code and the dependency structure through which computations are carried out. To address this mismatch, we propose SGKR (Structure-Grounded Knowledge Retrieval), a retrieval framework that organizes domain knowledge with a graph induced by function-call dependencies. Given a question, SGKR extracts semantic input and output tags, identifies dependency paths connecting them, and constructs a task-relevant subgraph. The associated knowledge and corresponding function implementations are then assembled as a structured context for LLM-based code generation. Experiments on multi-step data analysis benchmarks show that SGKR consistently improves solution correctness over no-retrieval and similarity-based retrieval baselines for both vanilla LLMs and coding agents.
>
---
#### [new 084] CocoaBench: Evaluating Unified Digital Agents in the Wild
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CocoaBench，用于评估统一数字代理在复杂任务中的表现，解决现有评估方法孤立测试能力的问题。工作包括构建基准和轻量级比较框架，发现当前代理仍有较大提升空间。**

- **链接: [https://arxiv.org/pdf/2604.11201](https://arxiv.org/pdf/2604.11201)**

> **作者:** CocoaBench Team; Shibo Hao; Zhining Zhang; Zhiqi Liang; Tianyang Liu; Yuheng Zha; Qiyue Gao; Jixuan Chen; Zilong Wang; Zhoujun Cheng; Haoxiang Zhang; Junli Wang; Hexi Jin; Boyuan Zheng; Kun Zhou; Yu Wang; Feng Yao; Licheng Liu; Yijiang Li; Zhifei Li; Zhengtao Han; Pracha Promthaw; Tommaso Cerruti; Xiaohan Fu; Ziqiao Ma; Jingbo Shang; Lianhui Qin; Julian McAuley; Eric P. Xing; Zhengzhong Liu; Rupesh Kumar Srivastava; Zhiting Hu
>
> **摘要:** LLM agents now perform strongly in software engineering, deep research, GUI automation, and various other applications, while recent agent scaffolds and models are increasingly integrating these capabilities into unified systems. Yet, most evaluations still test these capabilities in isolation, which leaves a gap for more diverse use cases that require agents to combine different capabilities. We introduce CocoaBench, a benchmark for unified digital agents built from human-designed, long-horizon tasks that require flexible composition of vision, search, and coding. Tasks are specified only by an instruction and an automatic evaluation function over the final output, enabling reliable and scalable evaluation across diverse agent infrastructures. We also present CocoaAgent, a lightweight shared scaffold for controlled comparison across model backbones. Experiments show that current agents remain far from reliable on CocoaBench, with the best evaluated system achieving only 45.1% success rate. Our analysis further points to substantial room for improvement in reasoning and planning, tool use and execution, and visual grounding.
>
---
#### [new 085] Judge Like Human Examiners: A Weighted Importance Multi-Point Evaluation Framework for Generative Tasks with Long-form Answers
- **分类: cs.CL**

- **简介: 该论文属于生成任务的评估研究，旨在解决长文本回答质量评价中因素复杂、重要性不均的问题。提出WIMPE框架，通过加权评分点和对齐与冲突度量提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2604.11246](https://arxiv.org/pdf/2604.11246)**

> **作者:** Guoxin Yu; Chulun Zhou; Lemao Liu; Qi Wang; Mo Yu; Jialong Tang; Baosong Yang; Xiang Ao; Wao Lam; Yue Yu
>
> **备注:** 21 pages
>
> **摘要:** Evaluating the quality of model responses remains challenging in generative tasks with long-form answers, as the expected answers usually contain multiple semantically distinct yet complementary factors that should be factorized for fine-grained assessment. Recent evaluation methods resort to relying on either task-level rubrics or question-aware checklists. However, they still 1) struggle to assess whether a response is genuinely grounded in provided contexts; 2) fail to capture the heterogeneous importance of different aspects of reference answers. Inspired by human examiners, we propose a Weighted Importance Multi-Point Evaluation (WIMPE) framework, which factorizes each reference answer into weighted context-bound scoring points. Two complementary metrics, namely Weighted Point-wise Alignment (WPA) and Point-wise Conflict Penalty (PCP), are designed to measure the alignment and contradiction between model responses and reference answers. Extensive experiments on 10 generative tasks demonstrate that WIMPE achieves higher correlations with human annotations.
>
---
#### [new 086] GIANTS: Generative Insight Anticipation from Scientific Literature
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出GIANTS任务，旨在从科学文献中预测下游论文的核心见解。通过构建基准和训练模型，提升语言模型在科学发现中的合成能力。**

- **链接: [https://arxiv.org/pdf/2604.09793](https://arxiv.org/pdf/2604.09793)**

> **作者:** Joy He-Yueya; Anikait Singh; Ge Gao; Michael Y. Li; Sherry Yang; Chelsea Finn; Emma Brunskill; Noah D. Goodman
>
> **摘要:** Scientific breakthroughs often emerge from synthesizing prior ideas into novel contributions. While language models (LMs) show promise in scientific discovery, their ability to perform this targeted, literature-grounded synthesis remains underexplored. We introduce insight anticipation, a generation task in which a model predicts a downstream paper's core insight from its foundational parent papers. To evaluate this capability, we develop GiantsBench, a benchmark of 17k examples across eight scientific domains, where each example consists of a set of parent papers paired with the core insight of a downstream paper. We evaluate models using an LM judge that scores similarity between generated and ground-truth insights, and show that these similarity scores correlate with expert human ratings. Finally, we present GIANTS-4B, an LM trained via reinforcement learning (RL) to optimize insight anticipation using these similarity scores as a proxy reward. Despite its smaller open-source architecture, GIANTS-4B outperforms proprietary baselines and generalizes to unseen domains, achieving a 34% relative improvement in similarity score over gemini-3-pro. Human evaluations further show that GIANTS-4B produces insights that are more conceptually clear than those of the base model. In addition, SciJudge-30B, a third-party model trained to compare research abstracts by likely citation impact, predicts that insights generated by GIANTS-4B are more likely to lead to higher citations, preferring them over the base model in 68% of pairwise comparisons. We release our code, benchmark, and model to support future research in automated scientific discovery.
>
---
#### [new 087] Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于古代汉字演变分析任务，旨在提升多模态大模型对汉字演化的理解。通过构建基准测试集和提出Glyph-Driven微调框架，解决模型在字形比较和演化推理上的不足。**

- **链接: [https://arxiv.org/pdf/2604.11299](https://arxiv.org/pdf/2604.11299)**

> **作者:** Rui Song; Lida Shi; Ruihua Qi; Yingji Li; Hao Xu
>
> **备注:** Accepted by ACL 2026 main
>
> **摘要:** In recent years, rapid advances in Multimodal Large Language Models (MLLMs) have increasingly stimulated research on ancient Chinese scripts. As the evolution of written characters constitutes a fundamental pathway for understanding cultural transformation and historical continuity, how MLLMs can be systematically leveraged to support and advance text evolution analysis remains an open and largely underexplored problem. To bridge this gap, we construct a comprehensive benchmark comprising 11 tasks and over 130,000 instances, specifically designed to evaluate the capability of MLLMs in analyzing the evolution of ancient Chinese scripts. We conduct extensive evaluations across multiple widely used MLLMs and observe that, while existing models demonstrate a limited ability in glyph-level comparison, their performance on core tasks-such as character recognition and evolutionary reasoning-remains substantially constrained. Motivated by these findings, we propose a glyph-driven fine-tuning framework (GEVO) that explicitly encourages models to capture evolutionary consistency in glyph transformations and enhances their understanding of text evolution. Experimental results show that even models at the 2B scale achieve consistent and comprehensive performance improvements across all evaluated tasks. To facilitate future research, we publicly release both the benchmark and the trained models\footnote{this https URL}.
>
---
#### [new 088] RPA-Check: A Multi-Stage Automated Framework for Evaluating Dynamic LLM-based Role-Playing Agents
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文属于评估任务，旨在解决LLM驱动角色扮演代理的评价难题。提出RPA-Check框架，通过多阶段评估其行为一致性与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.11655](https://arxiv.org/pdf/2604.11655)**

> **作者:** Riccardo Rosati; Edoardo Colucci; Massimiliano Bolognini; Adriano Mancini; Paolo Sernani
>
> **摘要:** The rapid adoption of Large Language Models (LLMs) in interactive systems has enabled the creation of dynamic, open-ended Role-Playing Agents (RPAs). However, evaluating these agents remains a significant challenge, as standard NLP metrics fail to capture the nuances of role adherence, logical consistency, and long-term narrative stability. This paper introduces RPA-Check, a multi-stage automated evaluation framework designed to objectively assess the performance of LLM-based RPAs in complex, constraints-heavy environments. Our methodology is based on a four-step pipeline: (1) Dimension Definition, establishing high-level qualitative behavioral criteria; (2) Augmentation, where these requirements are expanded into granular boolean checklist indicators; (3) Semantic Filtering, to ensure indicator objectivity, no redundancy and agent isolation; and (4) LLM-as-a-Judge Evaluation, which employs chain-of-thought verification to score agent fidelity. We validate this framework by applying it to LLM Court, a serious game for forensic training involving several quantized local models. Experimental results across five distinct legal scenarios demonstrate the framework's ability to identify subtle trade-offs between model size, reasoning depth, and operational stability. Notably, the findings reveal an inverse relationship between parametric scale and procedural consistency, showing that smaller, adequately instruction-tuned models (8-9B) can outperform larger architectures prone to user-alignment bias or sycophancy. RPA-Check thus provides a standardized and reproducible metric for future research in generative agent evaluation within specialized domains.
>
---
#### [new 089] DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning
- **分类: cs.CL**

- **简介: 该论文提出DeCoVec，用于在解码空间构建任务向量，以无训练、非侵入方式提升大语言模型的推理能力，解决任务引导问题。**

- **链接: [https://arxiv.org/pdf/2604.11129](https://arxiv.org/pdf/2604.11129)**

> **作者:** Feiyang Li; Yile Wang
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** Task vectors, representing directions in model or activation spaces that encode task-specific behaviors, have emerged as a promising tool for steering large language models (LLMs). However, existing approaches typically require fine-tuning or invasive manipulation of internal states, limiting their flexibility and scalability. We propose \textsc{DeCoVec} (Decoding Space based Task Vector), a training-free and non-invasive framework that constructs task vectors directly in the \textit{decoding space} by leveraging in-context learning (ICL). Specifically, \textsc{DeCoVec} captures the task essence as the difference between the output logit distributions of few-shot and zero-shot prompts, then steers generation by injecting this vector into the decoding process. Experiments across seven LLMs (0.5B--9B) on TruthfulQA, Math-500, and AQUA-RAT show that \textsc{DeCoVec} consistently outperforms standard few-shot baselines, with gains up to +5.50 average accuracy. Further analysis demonstrates that \textsc{DeCoVec} effectively suppresses generation degeneration and logical flaws while exhibiting strong robustness to demonstration ordering, all without incurring additional input token costs. Our method offers a training-free and non-invasive solution for LLM steering without requiring weight updates or auxiliary models.
>
---
#### [new 090] When Meaning Isn't Literal: Exploring Idiomatic Meaning Across Languages and Modalities
- **分类: cs.CL**

- **简介: 该论文属于多模态语言理解任务，旨在解决语言模型对习语含义的误解问题。研究构建了多语言多模态习语语料库Mediom，并提出HIDE框架提升模型的隐喻理解能力。**

- **链接: [https://arxiv.org/pdf/2604.10787](https://arxiv.org/pdf/2604.10787)**

> **作者:** Sarmistha Das; Shreyas Guha; Suvrayan Bandyopadhyay; Salisa Phosit; Kitsuchart Pasupa; Sriparna Saha
>
> **摘要:** Idiomatic reasoning, deeply intertwined with metaphor and culture, remains a blind spot for contemporary language models, whose progress skews toward surface-level lexical and semantic cues. For instance, the Bengali idiom \textit{\foreignlanguage{bengali}{\char"0986\char"0999\char"09CD\char"0997\char"09C1 \char"09B0 \char"09AB\char"09B2 \char"099F\char"0995}} (angur fol tok, ``grapes are sour''): it encodes denial-driven rationalization, yet naive models latch onto the literal fox-and-grape imagery. Addressing this oversight, we present ``Mediom,'' a multilingual, multimodal idiom corpus of 3,533 Hindi, Bengali, and Thai idioms, each paired with gold-standard explanations, cross-lingual translations, and carefully aligned text--image representations. We benchmark both large language models (textual reasoning) and vision-language models (figurative disambiguation) on Mediom, exposing systematic failures in metaphor comprehension. To mitigate these gaps, we propose ``HIDE,'' a Hinting-based Idiom Explanation framework that leverages error-feedback retrieval and targeted diagnostic cues for iterative reasoning refinement. Collectively, Mediom and HIDE establish a rigorous test bed and methodology for culturally grounded, multimodal idiom understanding embedded with reasoning hints in next-generation AI systems.
>
---
#### [new 091] Nationality encoding in language model hidden states: Probing culturally differentiated representations in persona-conditioned academic text
- **分类: cs.CL**

- **简介: 该论文属于语言模型的探针研究任务，旨在检验模型在生成学术文本时是否编码国籍差异。通过分析隐藏状态，发现模型在特定层中能有效区分英中学术风格。**

- **链接: [https://arxiv.org/pdf/2604.10151](https://arxiv.org/pdf/2604.10151)**

> **作者:** Paul Jackson; Ruizhe Li; Elspeth Edelstein
>
> **备注:** 42 pages, 6 tables
>
> **摘要:** Large language models are increasingly used as writing tools and pedagogical resources in English for Academic Purposes, but it remains unclear whether they encode culturally differentiated representations when generating academic text. This study tests whether Gemma-3-4b-it encodes nationality-discriminative information in hidden states when generating research article introductions conditioned by British and Chinese academic personas. A corpus of 270 texts was generated from 45 prompt templates crossed with six persona conditions in a 2 x 3 design. Logistic regression probes were trained on hidden-state activations across all 35 layers, with shuffled-label baselines, a surface-text skyline classifier, cross-family tests, and sentence-level baselines used as controls. Probe-selected token positions were annotated for structural, lexical, and stance features using the Stanza NLP pipeline. The nationality probe reached 0.968 cross-validated accuracy at Layer 18, with perfect held-out classification. Nationality encoding followed a non-monotonic trajectory across layers, with structural effects strongest in the middle to upper network and lexical-domain effects peaking earlier. At high-signal token positions, British-associated patterns showed more postmodification, hedging, boosting, passive voice, and evaluative or process-oriented vocabulary, while Chinese-associated patterns showed more premodification, nominal predicates, and sociocultural or internationalisation vocabulary. However, sentence-level analysis found no significant nationality differences in the full generated surface text. The findings extend probing methodology to a sociolinguistic attribute and have practical implications for EAP and language pedagogy.
>
---
#### [new 092] Instruction Data Selection via Answer Divergence
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决指令数据选择问题。提出ADG方法，通过答案发散性选择高质量指令数据，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.10448](https://arxiv.org/pdf/2604.10448)**

> **作者:** Bo Li; Mingda Wang; Shikun Zhang; Wei Ye
>
> **备注:** Github: this https URL Project: this https URL
>
> **摘要:** Instruction tuning relies on large instruction-response corpora whose quality and composition strongly affect downstream performance. We propose Answer Divergence-Guided Selection (ADG), which selects instruction data based on the geometric structure of multi-sample outputs. ADG draws several high-temperature generations per instruction, maps responses into an embedding space, and computes an output divergence score that jointly encodes dispersion magnitude and shape anisotropy. High scores correspond to instructions whose answers are both far apart and multi-modal, rather than clustered paraphrases along a single direction. Across two backbones and three public instruction pools, fine-tuning on only 10K ADG-selected examples consistently outperforms strong selectors on six benchmarks spanning reasoning, knowledge, and coding. Analyses further show that both dispersion magnitude and shape anisotropy are necessary, supporting answer divergence as a practical signal for instruction data selection. Code and appendix are included in the supplementary materials.
>
---
#### [new 093] BlasBench: An Open Benchmark for Irish Speech Recognition
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决爱尔兰语ASR系统缺乏公开基准的问题。作者发布BlasBench，包含爱尔兰语文本规范化的评估框架，并对比了多个系统的表现。**

- **链接: [https://arxiv.org/pdf/2604.10736](https://arxiv.org/pdf/2604.10736)**

> **作者:** Jyoutir Raj; John Conway
>
> **备注:** 8 pages, 4 tables, 3 appendices. Code and data: this https URL
>
> **摘要:** No open Irish-specific benchmark compares end-user ASR systems under a shared Irish-aware evaluation protocol. To solve this, we release BlasBench, an open evaluation harness with Irish-aware text normalisation that preserves fadas, lenition, and eclipsis. We benchmark 12 systems across four architecture families on Common Voice ga-IE and FLEURS ga-IE. All Whisper variants exceed 100% WER. The best open model (omniASR LLM 7B) achieves 30.65% WER on Common Voice and 39.09% on FLEURS. We noticed models fine-tuned on Common Voice lose 33-43 WER points on FLEURS, revealing a generalisation gap that is invisible to single-dataset evaluation.
>
---
#### [new 094] NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment
- **分类: cs.CL; cs.AI; cs.DL; cs.IR**

- **简介: 该论文提出NovBench，用于评估大语言模型在学术论文新颖性判断上的能力，解决模型对科学新颖性理解不足的问题。**

- **链接: [https://arxiv.org/pdf/2604.11543](https://arxiv.org/pdf/2604.11543)**

> **作者:** Wenqing Wu; Yi Zhao; Yuzhuo Wang; Siyou Li; Juexi Shao; Yunfei Long; Chengzhi Zhang
>
> **备注:** ACL 2026
>
> **摘要:** Novelty is a core requirement in academic publishing and a central focus of peer review, yet the growing volume of submissions has placed increasing pressure on human reviewers. While large language models (LLMs), including those fine-tuned on peer review data, have shown promise in generating review comments, the absence of a dedicated benchmark has limited systematic evaluation of their ability to assess research novelty. To address this gap, we introduce NovBench, the first large-scale benchmark designed to evaluate LLMs' capability to generate novelty evaluations in support of human peer review. NovBench comprises 1,684 paper-review pairs from a leading NLP conference, including novelty descriptions extracted from paper introductions and corresponding expert-written novelty evaluations. We focus on both sources because the introduction provides a standardized and explicit articulation of novelty claims, while expert-written novelty evaluations constitute one of the current gold standards of human judgment. Furthermore, we propose a four-dimensional evaluation framework (including Relevance, Correctness, Coverage, and Clarity) to assess the quality of LLM-generated novelty evaluations. Extensive experiments on both general and specialized LLMs under different prompting strategies reveal that current models exhibit limited understanding of scientific novelty, and that fine--tuned models often suffer from instruction-following deficiencies. These findings underscore the need for targeted fine-tuning strategies that jointly improve novelty comprehension and instruction adherence.
>
---
#### [new 095] EviCare: Enhancing Diagnosis Prediction with Deep Model-Guided Evidence for In-Context Reasoning
- **分类: cs.CL**

- **简介: 该论文属于医疗诊断预测任务，旨在解决LLM过度依赖历史诊断而忽视新病症的问题。提出EviCare框架，通过深度模型引导提升诊断预测准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2604.10455](https://arxiv.org/pdf/2604.10455)**

> **作者:** Hengyu Zhang; Xuyun Zhang; Pengxiang Zhan; Linhao Luo; Hang Lv; Yanchao Tan; Shirui Pan; Carl Yang
>
> **备注:** Accepted by KDD 2026
>
> **摘要:** Recent advances in large language models (LLMs) have enabled promising progress in diagnosis prediction from electronic health records (EHRs). However, existing LLM-based approaches tend to overfit to historically observed diagnoses, often overlooking novel yet clinically important conditions that are critical for early intervention. To address this, we propose EviCare, an in-context reasoning framework that integrates deep model guidance into LLM-based diagnosis prediction. Rather than prompting LLMs directly with raw EHR inputs, EviCare performs (1) deep model inference for candidate selection, (2) evidential prioritization for set-based EHRs, and (3) relational evidence construction for novel diagnosis prediction. These signals are then composed into an adaptive in-context prompt to guide LLM reasoning in an accurate and interpretable manner. Extensive experiments on two real-world EHR benchmarks (MIMIC-III and MIMIC-IV) demonstrate that EviCare achieves significant performance gains, which consistently outperforms both LLM-only and deep model-only baselines by an average of 20.65\% across precision and accuracy metrics. The improvements are particularly notable in challenging novel diagnosis prediction, yielding average improvements of 30.97\%.
>
---
#### [new 096] Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在研究扩散大语言模型的幻觉问题。通过对比实验，发现其幻觉倾向高于自回归模型，并分析了推理过程中的失败模式。**

- **链接: [https://arxiv.org/pdf/2604.10556](https://arxiv.org/pdf/2604.10556)**

> **作者:** Zhengnan Guo; Fei Tan
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** While Diffusion Large Language Models (dLLMs) have emerged as a promising non-autoregressive paradigm comparable to autoregressive (AR) models, their faithfulness, specifically regarding hallucination, remains largely underexplored. To bridge this gap, we present the first controlled comparative study to evaluate hallucination patterns in dLLMs. Our results demonstrate that current dLLMs exhibit a higher propensity for hallucination than AR counterparts controlled for architecture, scale, and pre-training weights. Furthermore, an analysis of inference-time compute reveals divergent dynamics: while quasi-autoregressive generation suffers from early saturation, non-sequential decoding unlocks potential for continuous refinement. Finally, we identify distinct failure modes unique to the diffusion process, including premature termination, incomplete denoising, and context intrusion. Our findings underscore that although dLLMs have narrowed the performance gap on general tasks, their distinct hallucination mechanisms pose a critical challenge to model reliability. Our code is available at this https URL
>
---
#### [new 097] Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry
- **分类: cs.CL**

- **简介: 该论文属于AI生成文本检测任务，旨在解决LLM生成古典中文诗歌的识别问题。通过构建基准数据集ChangAn，评估12种检测工具的效果，揭示现有检测方法的不足。**

- **链接: [https://arxiv.org/pdf/2604.10101](https://arxiv.org/pdf/2604.10101)**

> **作者:** Jiang Li; Tian Lan; Shanshan Wang; Dongxing Zhang; Dianqing Lin; Guanglai Gao; Derek F. Wong; Xiangdong Su
>
> **备注:** Accepted to ACL 2026 Main Conference
>
> **摘要:** The rapid development of large language models (LLMs) has extended text generation tasks into the literary domain. However, AI-generated literary creations has raised increasingly prominent issues of creative authenticity and ethics in literary world, making the detection of LLM-generated literary texts essential and urgent. While previous works have made significant progress in detecting AI-generated text, it has yet to address classical Chinese poetry. Due to the unique linguistic features of classical Chinese poetry, such as strict metrical regularity, a shared system of poetic imagery, and flexible syntax, distinguishing whether a poem is authored by AI presents a substantial challenge. To address these issues, we introduce ChangAn, a benchmark for detecting LLM-generated classical Chinese poetry that containing total 30,664 poems, 10,276 are human-written poems and 20,388 poems are generated by four popular LLMs. Based on ChangAn, we conducted a systematic evaluation of 12 AI detectors, investigating their performance variations across different text granularities and generation strategies. Our findings highlight the limitations of current Chinese text detectors, which fail to serve as reliable tools for detecting LLM-generated classical Chinese poetry. These results validate the effectiveness and necessity of our proposed ChangAn benchmark. Our dataset and code are available at this https URL.
>
---
#### [new 098] Should We be Pedantic About Reasoning Errors in Machine Translation?
- **分类: cs.CL; cs.AI**

- **简介: 论文研究机器翻译中的推理错误识别与修正，属于自然语言处理任务。旨在解决MT系统推理不准确的问题，通过多种干预手段评估错误修正效果。**

- **链接: [https://arxiv.org/pdf/2604.09890](https://arxiv.org/pdf/2604.09890)**

> **作者:** Calvin Bao; Marine Carpuat
>
> **备注:** 17 pages, 2 figures, 5 tables
>
> **摘要:** Across multiple language pairings (English $\to$ \{Spanish, French, German, Mandarin, Japanese, Urdu, Cantonese\}), we find reasoning errors in translation. To quantify how often these reasoning errors occur, we leverage an automated annotation protocol for reasoning evaluation wherein the goal is to detect if a reasoning step is any of three error categories: (1) source sentence-misaligned, (2) model hypothesis-misaligned, or (3) reasoning trace-misaligned. We probe the reasoning model with perturbed traces correcting for these identified reasoning errors using an array of weak-to-strong interventions: hedging, removal, re-reasoning after removal, hindsight, and oracle interventions. Experimenting with interventions on the reasoning traces suggests that small corrections to the reasoning have little impact on translation quality, but stronger interventions yield the highest resolution rates, despite translation quality gains being mixed. We find ultimately that reasoning errors in MT can be identified with high precision in Urdu but lower precision in Spanish, but that removing these reasoning errors does not resolve the initial errors significantly, suggesting limited reasoning faithfulness for machine translation.
>
---
#### [new 099] Do BERT Embeddings Encode Narrative Dimensions? A Token-Level Probing Analysis of Time, Space, Causality, and Character in Fiction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言理解任务，旨在验证BERT是否编码叙事维度（时间、空间、因果、人物）。通过构建标注数据集并进行线性探测，发现BERT能有效编码这些信息，但存在分类混淆问题。**

- **链接: [https://arxiv.org/pdf/2604.10786](https://arxiv.org/pdf/2604.10786)**

> **作者:** Beicheng Bei; Hannah Hyesun Chun; Chen Guo; Arwa Saghiri
>
> **备注:** 13 pages, 7 figures. Accepted at CMN'26 (9th International Workshop on Computational Models of Narrative)
>
> **摘要:** Narrative understanding requires multidimensional semantic structures. This study investigates whether BERT embeddings encode dimensions of fictional narrative semantics -- time, space, causality, and character. Using an LLM to accelerate annotation, we construct a token-level dataset labeled with these four narrative categories plus "others." A linear probe on BERT embeddings (94% accuracy) significantly outperforms a control probe on variance-matched random embeddings (47%), confirming that BERT encodes meaningful narrative information. With balanced class weighting, the probe achieves a macro-average recall of 0.83, with moderate success on rare categories such as causality (recall = 0.75) and space (recall = 0.66). However, confusion matrix analysis reveals "Boundary Leakage," where rare dimensions are systematically misclassified as "others." Clustering analysis shows that unsupervised clustering aligns near-randomly with predefined categories (ARI = 0.081), suggesting that narrative dimensions are encoded but not as discretely separable clusters. Future work includes a POS-only baseline to disentangle syntactic patterns from narrative encoding, expanded datasets, and layer-wise probing.
>
---
#### [new 100] Simulating Organized Group Behavior: New Framework, Benchmark, and Analysis
- **分类: cs.CL; cs.SI**

- **简介: 该论文属于组织群体行为模拟任务，旨在预测组织在特定情境下的决策。提出GROVE基准和分析框架，实现更准确的预测与知识迁移。**

- **链接: [https://arxiv.org/pdf/2604.09874](https://arxiv.org/pdf/2604.09874)**

> **作者:** Xinkai Zou; Yiming Huang; Zhuohang Wu; Jian Sha; Nan Huang; Longfei Yun; Jingbo Shang; Letian Peng
>
> **摘要:** Simulating how organized groups (e.g., corporations) make decisions (e.g., responding to a competitor's move) is essential for understanding real-world dynamics and could benefit relevant applications (e.g., market prediction). In this paper, we formalize this problem as a concrete research platform for group behavior understanding, providing: (1) a task definition with benchmark and evaluation criteria, (2) a structured analytical framework with a corresponding algorithm, and (3) detailed temporal and cross-group analysis. Specifically, we propose Organized Group Behavior Simulation, a task that models organized groups as collective entities from a practical perspective: given a group facing a particular situation (e.g., AI Boom), predict the decision it would take. To support this task, we present GROVE (GRoup Organizational BehaVior Evaluation), a benchmark covering 44 entities with 8,052 real-world context-decision pairs collected from Wikipedia and TechCrunch across 9 domains, with an end-to-end evaluation protocol assessing consistency, initiative, scope, magnitude, and horizon. Beyond straightforward prompting pipelines, we propose a structured analytical framework that converts collective decision-making events into an interpretable, adaptive, and traceable behavioral model, achieving stronger performance than summarization- and retrieval-based baselines. It further introduces an adapter mechanism for time-aware evolution and group-aware transfer, and traceable evidence nodes grounding each decision rule in originating historical events. Our analysis reveals temporal behavioral drift within individual groups, which the time-aware adapter effectively captures for stronger prediction, and structured cross-group similarity that enables knowledge transfer for data-scarce organizations.
>
---
#### [new 101] Generating Multiple-Choice Knowledge Questions with Interpretable Difficulty Estimation using Knowledge Graphs and Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自动化多选题生成任务，旨在解决难度估计不准确的问题。通过知识图谱和大语言模型生成可解释难度的多选题。**

- **链接: [https://arxiv.org/pdf/2604.10748](https://arxiv.org/pdf/2604.10748)**

> **作者:** Mehmet Can Şakiroğlu; H. Altay Güvenir; Kamer Kaya
>
> **摘要:** Generating multiple-choice questions (MCQs) with difficulty estimation remains challenging in automated MCQ-generation systems used in adaptive, AI-assisted education. This study proposes a novel methodology for generating MCQs with difficulty estimation from the input documents by utilizing knowledge graphs (KGs) and large language models (LLMs). Our approach uses an LLM to construct a KG from input documents, from which MCQs are then systematically generated. Each MCQ is generated by selecting a node from the KG as the key, sampling a related triple or quintuple -- optionally augmented with an extra triple -- and prompting an LLM to generate a corresponding stem from these graph components. Distractors are then selected from the KG. For each MCQ, nine difficulty signals are computed and combined into a unified difficulty score using a data-driven approach. Experimental results demonstrate that our method generates high-quality MCQs whose difficulty estimation is interpretable and aligns with human perceptions. Our approach improves automated MCQ generation by integrating structured knowledge representations with LLMs and a data-driven difficulty estimation model.
>
---
#### [new 102] Generating High Quality Synthetic Data for Dutch Medical Conversations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于生成合成医疗对话的任务，旨在解决临床数据稀缺问题。通过微调的荷兰语大模型生成合成对话，并评估其质量。**

- **链接: [https://arxiv.org/pdf/2604.09645](https://arxiv.org/pdf/2604.09645)**

> **作者:** Cecilia Kuan; Aditya Kamlesh Parikh; Henk van den Heuvel
>
> **备注:** Accepted to LREC 2026. This publication was supported by the MediSpeech project funded by ITEA4 under contract number 22032
>
> **摘要:** Medical conversations offer insights into clinical communication often absent from Electronic Health Records. However, developing reliable clinical Natural Language Processing (NLP) models is hampered by the scarcity of domain-specific datasets, as clinical data are typically inaccessible due to privacy and ethical constraints. To address these challenges, we present a pipeline for generating synthetic Dutch medical dialogues using a Dutch fine-tuned Large Language Model, with real medical conversations serving as linguistic and structural reference. The generated dialogues were evaluated through quantitative metrics and qualitative review by native speakers and medical practitioners. Quantitative analysis revealed strong lexical variety and overly regular turn-taking, suggesting scripted rather than natural conversation flow. Qualitative review produced slightly below-average scores, with raters noting issues in domain specificity and natural expression. The limited correlation between quantitative and qualitative results highlights that numerical metrics alone cannot fully capture linguistic quality. Our findings demonstrate that generating synthetic Dutch medical dialogues is feasible but requires domain knowledge and carefully structured prompting to balance naturalness and structure in conversation. This work provides a foundation for expanding Dutch clinical NLP resources through ethically generated synthetic data.
>
---
#### [new 103] Expect the Unexpected? Testing the Surprisal of Salient Entities
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究 discourse 中实体显著性与 surprisal 的关系。通过分析70K标注数据，发现显著实体具有更高 surprisal，并影响周围内容的可预测性。**

- **链接: [https://arxiv.org/pdf/2604.10724](https://arxiv.org/pdf/2604.10724)**

> **作者:** Jessica Lin; Amir Zeldes
>
> **备注:** Accepted to ACL 2026 (main, long); camera-ready version
>
> **摘要:** Previous work examining the Uniform Information Density (UID) hypothesis has shown that while information as measured by surprisal metrics is distributed more or less evenly across documents overall, local discrepancies can arise due to functional pressures corresponding to syntactic and discourse structural constraints. However, work thus far has largely disregarded the relative salience of discourse participants. We fill this gap by studying how overall salience of entities in discourse relates to surprisal using 70K manually annotated mentions across 16 genres of English and a novel minimal-pair prompting method. Our results show that globally salient entities exhibit significantly higher surprisal than non-salient ones, even controlling for position, length, and nesting confounds. Moreover, salient entities systematically reduce surprisal for surrounding content when used as prompts, enhancing document-level predictability. This effect varies by genre, appearing strongest in topic-coherent texts and weakest in conversational contexts. Our findings refine the UID competing pressures framework by identifying global entity salience as a mechanism shaping information distribution in discourse.
>
---
#### [new 104] Phonological distances for linguistic typology and the origin of Indo-European languages
- **分类: cs.CL; cond-mat.stat-mech; cs.IT; physics.soc-ph**

- **简介: 该论文属于语言类型学与演化语言学任务，旨在通过音位距离分析语言关系，解决语言家族起源问题。工作包括构建音位模型，计算语言距离，揭示语言接触与地理分布关联，支持印欧语系起源假说。**

- **链接: [https://arxiv.org/pdf/2604.11565](https://arxiv.org/pdf/2604.11565)**

> **作者:** Marius Mavridis; Juan De Gregorio; Raul Toral; David Sanchez
>
> **备注:** 27 pages, 7 figures, 2 appendices
>
> **摘要:** We show that short-range phoneme dependencies encode large-scale patterns of linguistic relatedness, with direct implications for quantitative typology and evolutionary linguistics. Specifically, using an information-theoretic framework, we argue that phoneme sequences modeled as second-order Markov chains essentially capture the statistical correlations of a phonological system. This finding enables us to quantify distances among 67 modern languages from a multilingual parallel corpus employing a distance metric that incorporates articulatory features of phonemes. The resulting phonological distance matrix recovers major language families and reveals signatures of contact-induced convergence. Remarkably, we obtain a clear correlation with geographic distance, allowing us to constrain a plausible homeland region for the Indo-European family, consistent with the Steppe hypothesis.
>
---
#### [new 105] Bridging What the Model Thinks and How It Speaks: Self-Aware Speech Language Models for Expressive Speech Generation
- **分类: cs.CL**

- **简介: 该论文属于语音生成任务，旨在解决语义理解与语音表达不一致的问题。通过引入自感知机制，提升语音的表达性。**

- **链接: [https://arxiv.org/pdf/2604.11424](https://arxiv.org/pdf/2604.11424)**

> **作者:** Kuang Wang; Lai Wei; Qibing Bai; Ping Lin; Wenkai Fang; Feng Jiang; Zhongjie Jiang; Jun Huang; Yannan Wang; Haizhou Li
>
> **备注:** 16 pages, 4 figures, 6 tables. Project page: this https URL
>
> **摘要:** Speech Language Models (SLMs) exhibit strong semantic understanding, yet their generated speech often sounds flat and fails to convey expressive intent, undermining user engagement. We term this mismatch the semantic understanding-acoustic realization gap. We attribute this gap to two key deficiencies: (1) intent transmission failure, where SLMs fail to provide the stable utterance-level intent needed for expressive delivery; and (2) realization-unaware training, where no feedback signal verifies whether acoustic outputs faithfully reflect intended expression. To address these issues, we propose SA-SLM (Self-Aware Speech Language Model), built on the principle that the model should be aware of what it thinks during generation and how it speaks during training. SA-SLM addresses this gap through two core contributions: (1) Intent-Aware Bridging, which uses a Variational Information Bottleneck (VIB) objective to translate the model's internal semantics into temporally smooth expressive intent, making speech generation aware of what the model intends to express; and (2) Realization-Aware Alignment, which repurposes the model as its own critic to verify and align acoustic realization with intended expressive intent via rubric-based feedback. Trained on only 800 hours of expressive speech data, our 3B parameter SA-SLM surpasses all open-source baselines and comes within 0.08 points of GPT-4o-Audio in overall expressiveness on the EchoMind benchmark.
>
---
#### [new 106] Transactional Attention: Semantic Sponsorship for KV-Cache Retention
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对KV缓存压缩中的凭证丢失问题，提出Transactional Attention机制，有效保留关键token，提升检索准确率。**

- **链接: [https://arxiv.org/pdf/2604.11288](https://arxiv.org/pdf/2604.11288)**

> **作者:** Abhinaba Basu
>
> **摘要:** At K=16 tokens (0.4% of a 4K context), every existing KV-cache compression method achieves 0% on credential retrieval. The failure mode is dormant tokens: credentials, API keys, and configuration values that receive near-zero attention but become essential at generation time. Because these tokens lack the statistical signals that eviction policies rely on, no method based on attention scores, reconstruction loss, or learned retention gates retains them. We introduce Transactional Attention (TA), a sponsorship mechanism in which structural anchor patterns (e.g., "key:", "password:") protect adjacent value-bearing tokens from eviction. TA achieves 100% credential retrieval at K=16 where six baselines (H2O, TOVA, SnapKV, StreamingLLM, PyramidKV, DynamicKV) achieve 0%, and sustains 100% accuracy across 200 function-calling trials. TA-Fast, an attention-free variant, reduces memory overhead by 52% and is compatible with SDPA and FlashAttention. TA is orthogonal to existing compression methods and adds less than 1% latency overhead.
>
---
#### [new 107] Advancing Polish Language Modeling through Tokenizer Optimization in the Bielik v3 7B and 11B Series
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型优化任务，旨在解决通用分词器对波兰语处理效率低的问题，通过专用分词器和预训练策略提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.10799](https://arxiv.org/pdf/2604.10799)**

> **作者:** Krzysztof Ociepa; Łukasz Flis; Remigiusz Kinas; Krzysztof Wróbel; Adrian Gwoździej
>
> **摘要:** The development of the Bielik v3 PL series, encompassing both the 7B and 11B parameter variants, represents a significant milestone in the field of language-specific large language model (LLM) optimization. While general-purpose models often demonstrate impressive multilingual capabilities, they frequently suffer from a fundamental architectural inefficiency: the use of universal tokenizers. These tokenizers, typically designed to cover a broad spectrum of languages, often fail to capture the morphological nuances of specific languages like Polish, leading to higher fertility ratios, increased inference costs, and restricted effective context windows. This report details the transition from the universal Mistral-based tokenization to a dedicated Polish-optimized vocabulary for the Bielik v3 models, exploring the FOCUS-based embedding initialization, the multi-stage pretraining curriculum, and the subsequent post-training alignment involving Supervised Fine-Tuning, Direct Preference Optimization, and Reinforcement Learning through Group Relative Policy Optimization with verifiable rewards.
>
---
#### [new 108] Knowing What to Stress: A Discourse-Conditioned Text-to-Speech Benchmark
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音合成任务，旨在解决TTS系统在语境中正确标注重音的问题。通过构建基准数据集，发现TTS系统在实现语境合适的重音上存在不足。**

- **链接: [https://arxiv.org/pdf/2604.10580](https://arxiv.org/pdf/2604.10580)**

> **作者:** Arnon Turetzky; Avihu Dekel; Hagai Aronowitz; Ron Hoory; Yossi Adi
>
> **备注:** Preprint
>
> **摘要:** Spoken meaning often depends not only on what is said, but also on which word is emphasized. The same sentence can convey correction, contrast, or clarification depending on where emphasis falls. Although modern text-to-speech (TTS) systems generate expressive speech, it remains unclear whether they infer contextually appropriate stress from discourse alone. To address this gap, we present Context-Aware Stress TTS (CAST), a benchmark for evaluating context-conditioned word-level stress in TTS. Items are defined as contrastive context pairs: identical sentences paired with distinct contexts requiring different stressed words. We evaluate state-of-the-art systems and find a consistent gap: text-only language models reliably recover the intended stress from context, yet TTS systems frequently fail to realize it in speech. We release the benchmark, evaluation framework, construction pipeline and a synthetic corpus to support future work on context-aware speech synthesis.
>
---
#### [new 109] LLMs Should Incorporate Explicit Mechanisms for Human Empathy
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在解决LLMs缺乏人类共情机制的问题。通过分析共情失败的四个机制，提出改进方向。**

- **链接: [https://arxiv.org/pdf/2604.10557](https://arxiv.org/pdf/2604.10557)**

> **作者:** Xiaoxing You; Qiang Huang; Jun Yu
>
> **摘要:** This paper argues that Large Language Models (LLMs) should incorporate explicit mechanisms for human empathy. As LLMs become increasingly deployed in high-stakes human-centered settings, their success depends not only on correctness or fluency but on faithful preservation of human perspectives. Yet, current LLMs systematically fail at this requirement: even when well-aligned and policy-compliant, they often attenuate affect, misrepresent contextual salience, and rigidify relational stance in ways that distort meaning. We formalize empathy as an observable behavioral property: the capacity to model and respond to human perspectives while preserving intention, affect, and context. Under this framing, we identify four recurring mechanisms of empathic failure in contemporary LLMs--sentiment attenuation, empathic granularity mismatch, conflict avoidance, and linguistic distancing--arising as structural consequences of prevailing training and alignment practices. We further organize these failures along three dimensions: cognitive, cultural, and relational empathy, to explain their manifestation across tasks. Empirical analyses show that strong benchmark performance can mask systematic empathic distortions, motivating empathy-aware objectives, benchmarks, and training signals as first-class components of LLM development.
>
---
#### [new 110] Decomposing and Reducing Hidden Measurement Error in LLM Evaluation Pipelines
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决LLM评估中的隐性测量误差问题。通过分解误差来源，提出优化评估管道的方法，提升评估的准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.11581](https://arxiv.org/pdf/2604.11581)**

> **作者:** Solomon Messing
>
> **摘要:** LLM evaluations drive which models get deployed, which safety standards get adopted, and which research conclusions get published. Yet these scores carry hidden uncertainty: rephrasing the prompt, switching the judge model, or changing the temperature can shift results enough to flip rankings and reverse conclusions. Standard confidence intervals ignore this variance, producing under-coverage that worsens with more data. The unmeasured variance also creates an exploitable surface: model developers can optimize against measurement noise rather than genuine capability. This paper decomposes LLM pipeline uncertainty into its sources, distinguishes variance that shrinks with more data from sensitivity to researcher design choices, and projects the most efficient path to reducing total error. For benchmark builders, the same decomposition identifies which design choices contribute exploitable surface for gaming and prescribes designs that minimize it. Across ideology annotation, safety classification, MMLU benchmarking, and a human-validated propaganda audit, projection-optimized pipelines outperform 73\% of possible naive pipelines against a human baseline. On MMLU, optimized budget allocation halves estimation error compared to standard single-prompt evaluation at equivalent cost. A small-sample variance estimation exercise is sufficient to derive confidence intervals that approach nominal coverage when the model includes the relevant pipeline facets, and to generate recommendations for reducing measurement error and improving benchmark robustness.
>
---
#### [new 111] Hidden Failures in Robustness: Why Supervised Uncertainty Quantification Needs Better Evaluation
- **分类: cs.CL**

- **简介: 该论文属于不确定性量化任务，旨在解决监督式不确定性探针在分布外情况下的鲁棒性问题。通过系统评估不同探针设计，发现中间层表示和多token聚合更可靠。**

- **链接: [https://arxiv.org/pdf/2604.11662](https://arxiv.org/pdf/2604.11662)**

> **作者:** Joe Stacey; Hadas Orgad; Kentaro Inui; Benjamin Heinzerling; Nafise Sadat Moosavi
>
> **摘要:** Recent work has shown that the hidden states of large language models contain signals useful for uncertainty estimation and hallucination detection, motivating a growing interest in efficient probe-based approaches. Yet it remains unclear how robust existing methods are, and which probe designs provide uncertainty estimates that are reliable under distribution shift. We present a systematic study of supervised uncertainty probes across models, tasks, and OOD settings, training over 2,000 probes while varying the representation layer, feature type, and token aggregation strategy. Our evaluation highlights poor robustness in current methods, particularly in the case of long-form generations. We also find that probe robustness is driven less by architecture and more by the probe inputs. Middle-layer representations generalise more reliably than final-layer hidden states, and aggregating across response tokens is consistently more robust than relying on single-token features. These differences are often largely invisible in-distribution but become more important under distribution shift. Informed by our evaluation, we explore a simple hybrid back-off strategy for improving robustness, arguing that better evaluation is a prerequisite for building more robust probes.
>
---
#### [new 112] Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale
- **分类: cs.CL**

- **简介: 该论文提出Relax系统，解决大规模多模态强化学习训练中的数据流异构、扩展性及延迟-吞吐权衡问题，通过异步架构提升效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.11554](https://arxiv.org/pdf/2604.11554)**

> **作者:** Liujie Zhang; Benzhe Ning; Rui Yang; Xiaoyan Yu; Jiaxing Li; Lumeng Wu; Jia Liu; Minghao Li; Weihang Chen; Weiqi Hu; Lei Zhang
>
> **备注:** 17 pages, 22 figures
>
> **摘要:** Reinforcement learning (RL) post-training has proven effective at unlocking reasoning, self-reflection, and tool-use capabilities in large language models. As models extend to omni-modal inputs and agentic multi-turn workflows, RL training systems face three interdependent challenges: heterogeneous data flows, operational robustness at scale, and the staleness -- throughput tradeoff. We present \textbf{Relax} (Reinforcement Engine Leveraging Agentic X-modality), an open-source RL training engine that addresses these challenges through three co-designed architectural layers. First, an \emph{omni-native architecture} builds multimodal support into the full stack -- from data preprocessing and modality-aware parallelism to inference generation -- rather than retrofitting it onto a text-centric pipeline. Second, each RL role runs as an independent, fault-isolated service that can be scaled, recovered, and upgraded without global coordination. Third, service-level decoupling enables asynchronous training via the TransferQueue data bus, where a single staleness parameter smoothly interpolates among on-policy, near-on-policy, and fully asynchronous execution. Relax achieves a 1.20$\times$ end-to-end speedup over veRL on Qwen3-4B on-policy training. Its fully async mode delivers a 1.76$\times$ speedup over colocate on Qwen3-4B and a 2.00$\times$ speedup on Qwen3-Omni-30B, while all modes converge to the same reward level. Relax supports R3 (Rollout Routing Replay)~\cite{ma2025r3} for MoE models with only 1.9\% overhead, compared to 32\% degradation in veRL under the same configuration. It further demonstrates stable omni-modal RL convergence on Qwen3-Omni across image, text, and audio, sustaining over 2{,}000 steps on video without degradation. Relax is available at this https URL.
>
---
#### [new 113] Position-Agnostic Pre-Projection for Transformer Attention: Nonlinear Feature Construction and Content Skip Before Q/K/V
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在改进Transformer模型的注意力机制。通过引入非线性预投影和内容跳连，提升特征表示并优化信息流动。**

- **链接: [https://arxiv.org/pdf/2604.10791](https://arxiv.org/pdf/2604.10791)**

> **作者:** Chirag Shinde
>
> **备注:** 7 pages, 2 figures, 5 tables. Code: this https URL
>
> **摘要:** We propose two complementary modifications to transformer attention blocks. First, a non-linear pre-projection MLP is inserted between layer norm and Q/K/V projections, constructing richer features in a position-agnostic manner before any positional encoding is applied. Second, a content skip connection routes the pre-projection's features around the attention mechanism, allowing content information to bypass position-aware attention where beneficial. In frozen-probe experiments on Pythia-160M and 410M, the combined approach achieves the strongest results across methods: +40.6% LAMBADA accuracy and -39% perplexity at 160M scale. Learned skip connection weights reveal a consistent pattern across model sizes: later transformer layers activate the content bypass more strongly than earlier layers, suggesting that deeper layers benefit from content information that does not pass through positional attention. All modifications add no K/V cache overhead.
>
---
#### [new 114] METRO: Towards Strategy Induction from Expert Dialogue Transcripts for Non-collaborative Dialogues
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出METRO方法，用于从专家对话中自动提取策略，解决非协作对话代理构建难题。通过策略森林结构，提升对话策略的多样性和前瞻性。**

- **链接: [https://arxiv.org/pdf/2604.11427](https://arxiv.org/pdf/2604.11427)**

> **作者:** Haofu Yang; Jiaji Liu; Chen Huang; Faguo Wu; Wenqiang Lei; See-Kiong Ng
>
> **备注:** ACL 2026
>
> **摘要:** Developing non-collaborative dialogue agents traditionally requires the manual, unscalable codification of expert strategies. We propose \ours, a method that leverages large language models to autonomously induce both strategy actions and planning logic directly from raw transcripts. METRO formalizes expert knowledge into a Strategy Forest, a hierarchical structure that captures both short-term responses (nodes) and long-term strategic foresight (branches). Experimental results across two benchmarks show that METRO demonstrates promising performance, outperforming existing methods by an average of 9%-10%. Our further analysis not only reveals the success behind METRO (strategic behavioral diversity and foresight), but also demonstrates its robust cross-task transferability. This offers new insights into building non-collaborative agents in a cost-effective and scalable way. Our code is available at this https URL.
>
---
#### [new 115] METER: Evaluating Multi-Level Contextual Causal Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在因果推理中的上下文一致性问题。提出METER基准，评估模型在因果层次上的表现，并分析其失败原因。**

- **链接: [https://arxiv.org/pdf/2604.11502](https://arxiv.org/pdf/2604.11502)**

> **作者:** Pengfeng Li; Chen Huang; Chaoqun Hao; Hongyao Chen; Xiao-Yong Wei; Wenqiang Lei; See-Kiong Ng
>
> **备注:** ACL 2026. Our code and dataset are available at this https URL
>
> **摘要:** Contextual causal reasoning is a critical yet challenging capability for Large Language Models (LLMs). Existing benchmarks, however, often evaluate this skill in fragmented settings, failing to ensure context consistency or cover the full causal hierarchy. To address this, we pioneer METER to systematically benchmark LLMs across all three levels of the causal ladder under a unified context setting. Our extensive evaluation of various LLMs reveals a significant decline in proficiency as tasks ascend the causal hierarchy. To diagnose this degradation, we conduct a deep mechanistic analysis via both error pattern identification and internal information flow tracing. Our analysis reveals two primary failure modes: (1) LLMs are susceptible to distraction by causally irrelevant but factually correct information at lower level of causality; and (2) as tasks ascend the causal hierarchy, faithfulness to the provided context degrades, leading to a reduced performance. We belive our work advances our understanding of the mechanisms behind LLM contextual causal reasoning and establishes a critical foundation for future research. Our code and dataset are available at this https URL .
>
---
#### [new 116] HiEdit: Lifelong Model Editing with Hierarchical Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于模型编辑任务，旨在解决LLM知识更新中的侧效与遗忘问题。提出HiEdit框架，通过分层强化学习动态选择相关层进行精准编辑。**

- **链接: [https://arxiv.org/pdf/2604.11214](https://arxiv.org/pdf/2604.11214)**

> **作者:** Yangfan Wang; Tianyang Sun; Chen Tang; Jie Liu; Wei Cai; Jingchi Jiang
>
> **备注:** Accept by ACL 2026
>
> **摘要:** Lifelong model editing (LME) aims to sequentially rectify outdated or inaccurate knowledge in deployed LLMs while minimizing side effects on unrelated inputs. However, existing approaches typically apply parameter perturbations to a static and dense set of LLM layers for all editing instances. This practice is counter-intuitive, as we hypothesize that different pieces of knowledge are stored in distinct layers of the model. Neglecting this layer-wise specificity can impede adaptability in integrating new knowledge and result in catastrophic forgetting for both general and previously edited knowledge. To address this, we propose HiEdit, a hierarchical reinforcement learning framework that adaptively identifies the most knowledge-relevant layers for each editing instance. By enabling dynamic, instance-aware layer selection and incorporating an intrinsic reward for sparsity, HiEdit achieves precise, localized updates. Experiments on various LLMs show that HiEdit boosts the performance of the competitive RLEdit by an average of 8.48% with perturbing only half of the layers per edit. Our code is available at: this https URL.
>
---
#### [new 117] Discourse Diversity in Multi-Turn Empathic Dialogue
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感对话任务，旨在解决大语言模型在多轮对话中缺乏话语策略多样性的问题。通过引入MINT框架提升对话多样性与共情效果。**

- **链接: [https://arxiv.org/pdf/2604.11742](https://arxiv.org/pdf/2604.11742)**

> **作者:** Hongli Zhan; Emma S. Gueorguieva; Javier Hernandez; Jina Suh; Desmond C. Ong; Junyi Jessy Li
>
> **摘要:** Large language models (LLMs) produce responses rated as highly empathic in single-turn settings (Ayers et al., 2023; Lee et al., 2024), yet they are also known to be formulaic generators that reuse the same lexical patterns, syntactic templates, and discourse structures across tasks (Jiang et al., 2025; Shaib et al., 2024; Namuduri et al., 2025). Less attention has been paid to whether this formulaicity extends to the level of discourse moves, i.e., what a response does for the person it is addressing. This question is especially consequential for empathic dialogue, where effective support demands not just a kind response at one moment but varied strategies as a conversation unfolds (Stiles et al., 1998). Indeed, prior work shows that LLMs reuse the same tactic sequences more than human supporters in single-turn settings (Gueorguieva et al., 2026). We extend this analysis to multi-turn conversations and find that the rigidity compounds: once a tactic appears in a supporter turn, LLMs reuse it in the next at nearly double the rate of humans (0.50-0.56 vs. 0.27). This pattern holds across LLMs serving as supporters in real emotional support conversations, and is invisible to standard similarity metrics. To address this gap, we introduce MINT (Multi-turn Inter-tactic Novelty Training), the first reinforcement learning framework to optimize discourse move diversity across multi-turn empathic dialogue. The best MINT variant combines an empathy quality reward with a cross-turn tactic novelty signal, improving aggregate empathy by 25.3% over vanilla across 1.7B and 4B models while reducing cross-turn discourse move repetition by 26.3% on the 4B model, surpassing all baselines including quality-only and token-level diversity methods on both measures. These results suggest that what current models lack is not empathy itself, but the ability to vary their discourse moves across a conversation.
>
---
#### [new 118] NameBERT: Scaling Name-Based Nationality Classification with LLM-Augmented Open Academic Data
- **分类: cs.CL**

- **简介: 论文提出NameBERT，解决姓名到国籍分类任务中的数据不足问题。通过LLM增强开放学术数据，构建大规模数据集，提升模型性能与效率。**

- **链接: [https://arxiv.org/pdf/2604.10401](https://arxiv.org/pdf/2604.10401)**

> **作者:** Cong Ming; Ruixin Shi; Yifan Hu
>
> **备注:** 12 pages, 3 figures, 8 tables; accepted at the 39th Canadian Conference on Artificial Intelligence (Canadian AI 2026)
>
> **摘要:** Inferring nationality from personal names is a critical capability for equity and bias monitoring, personalization, and a valuable tool in biomedical and sociological research. However, existing name-based nationality classifiers are typically trained on relatively small or source-specific labeled datasets, which can introduce coverage gaps and limit performance for underrepresented countries. While large language models (LLMs) demonstrate strong zero-shot performance for name-based nationality prediction, their computational cost and latency make them impractical for real-time, large-scale deployment. In this work, we created a large-scale name-nationality dataset from the Open Academic Graph (OAG) and introduce a framework that leverages LLMs as dataset enrichers rather than inference engines. We augment low-resource countries with LLM-generated names and evaluate on real and synthetic-tail test sets. We find that augmentation produces large gains when evaluation includes synthetic tail names and still offers a modest lift on tail-country metrics otherwise. Overall, NameBERT models achieve significantly higher accuracy than state-of-the-art baselines across both in- and out-of-domain tasks, while remaining efficient for large-scale inference compared to LLMs.
>
---
#### [new 119] How Robust Are Large Language Models for Clinical Numeracy? An Empirical Study on Numerical Reasoning Abilities in Clinical Contexts
- **分类: cs.CL**

- **简介: 该论文属于临床数值推理任务，旨在评估大语言模型在临床场景中的数值理解能力。通过构建基准数据集，测试模型在数值检索、计算、比较和聚合方面的表现，揭示其鲁棒性问题。**

- **链接: [https://arxiv.org/pdf/2604.11133](https://arxiv.org/pdf/2604.11133)**

> **作者:** Minh-Vuong Nguyen; Fatemeh Shiri; Zhuang Li; Karin Verspoor
>
> **备注:** Accepted to ACL2026 Findings
>
> **摘要:** Large Language Models (LLMs) are increasingly being explored for clinical question answering and decision support, yet safe deployment critically requires reliable handling of patient measurements in heterogeneous clinical notes. Existing evaluations of LLMs for clinical numerical reasoning provide limited operation-level coverage, restricted primarily to arithmetic computation, and rarely assess the robustness of numerical understanding across clinical note formats. We introduce ClinicNumRobBench, a benchmark of 1,624 context-question instances with ground-truth answers that evaluates four main types of clinical numeracy: value retrieval, arithmetic computation, relational comparison, and aggregation. To stress-test robustness, ClinicNumRobBench presents longitudinal MIMIC-IV vital-sign records in three semantically equivalent representations, including a real-world note-style variant derived from the Open Patients dataset, and instantiates queries using 42 question templates. Experiments on 14 LLMs show that value retrieval is generally strong, with most models exceeding 85% accuracy, while relational comparison and aggregation remain challenging, with some models scoring below 15%. Fine-tuning on medical data can reduce numeracy relative to base models by over 30%, and performance drops under note-style variation indicate LLM sensitivity to format. ClinicNumRobBench offers a rigorous testbed for clinically reliable numerical reasoning. Code and data URL are available on this https URL.
>
---
#### [new 120] TRACE: An Experiential Framework for Coherent Multi-hop Knowledge Graph Question Answering
- **分类: cs.CL**

- **简介: 该论文属于多跳知识图谱问答任务，旨在解决推理过程碎片化和重复探索问题。提出TRACE框架，结合上下文与经验先验，提升推理连贯性与效果。**

- **链接: [https://arxiv.org/pdf/2604.11193](https://arxiv.org/pdf/2604.11193)**

> **作者:** Yingxu Wang; Jiaxin Huang; Mengzhu Wang; Nan Yin
>
> **摘要:** Multi-hop Knowledge Graph Question Answering (KGQA) requires coherent reasoning across relational paths, yet existing methods often treat each reasoning step independently and fail to effectively leverage experience from prior explorations, leading to fragmented reasoning and redundant exploration. To address these challenges, we propose Trajectoryaware Reasoning with Adaptive Context and Exploration priors (TRACE), an experiential framework that unifies LLM-driven contextual reasoning with exploration prior integration to enhance the coherence and robustness of multihop KGQA. Specifically, TRACE dynamically translates evolving reasoning paths into natural language narratives to maintain semantic continuity, while abstracting prior exploration trajectories into reusable experiential priors that capture recurring exploration patterns. A dualfeedback re-ranking mechanism further integrates contextual narratives with exploration priors to guide relation selection during reasoning. Extensive experiments on multiple KGQA benchmarks demonstrate that TRACE consistently outperforms state-of-the-art baselines.
>
---
#### [new 121] Back to Basics: Let Conversational Agents Remember with Just Retrieval and Generation
- **分类: cs.CL**

- **简介: 该论文属于对话记忆任务，旨在解决长对话中上下文稀释问题。提出一种仅依赖检索与生成的极简框架，通过优化信号获取和去除冗余，提升对话理解效果。**

- **链接: [https://arxiv.org/pdf/2604.11628](https://arxiv.org/pdf/2604.11628)**

> **作者:** Yuqian Wu; Wei Chen; Zhengjun Huang; Junle Chen; Qingxiang Liu; Kai Wang; Xiaofang Zhou; Yuxuan Liang
>
> **备注:** 23 pages, 12 figures
>
> **摘要:** Existing conversational memory systems rely on complex hierarchical summarization or reinforcement learning to manage long-term dialogue history, yet remain vulnerable to context dilution as conversations grow. In this work, we offer a different perspective: the primary bottleneck may lie not in memory architecture, but in the \textit{Signal Sparsity Effect} within the latent knowledge manifold. Through controlled experiments, we identify two key phenomena: \textit{Decisive Evidence Sparsity}, where relevant signals become increasingly isolated with longer sessions, leading to sharp degradation in aggregation-based methods; and \textit{Dual-Level Redundancy}, where both inter-session interference and intra-session conversational filler introduce large amounts of non-informative content, hindering effective generation. Motivated by these insights, we propose \method, a minimalist framework that brings conversational memory back to basics, relying solely on retrieval and generation via Turn Isolation Retrieval (TIR) and Query-Driven Pruning (QDP). TIR replaces global aggregation with a max-activation strategy to capture turn-level signals, while QDP removes redundant sessions and conversational filler to construct a compact, high-density evidence set. Extensive experiments on multiple benchmarks demonstrate that \method achieves robust performance across diverse settings, consistently outperforming strong baselines while maintaining high efficiency in tokens and latency, establishing a new minimalist baseline for conversational memory.
>
---
#### [new 122] HumorGen: Cognitive Synergy for Humor Generation in Large Language Models via Persona-Based Distillation
- **分类: cs.CL**

- **简介: 该论文属于幽默生成任务，旨在解决LLM难以生成幽默的问题。通过认知协同框架和角色化数据生成，提升幽默质量。**

- **链接: [https://arxiv.org/pdf/2604.09629](https://arxiv.org/pdf/2604.09629)**

> **作者:** Edward Ajayi; Prasenjit Mitra
>
> **摘要:** Humor generation poses a significant challenge for Large Language Models (LLMs), because their standard training objective - predicting the most likely next word - inherently conflicts with the surprise and incongruity needed for comedy. To bridge this gap, we introduce the Cognitive Synergy Framework, a theoretically grounded methodology for generating high-quality humor data inspired by psychological theories of humor. Utilizing a Mixture-of-Thought (MoT) approach, we deploy six cognitive personas (e.g., The Absurdist, The Cynic) to synthesize diverse comedic perspectives for a given prompt. This framework creates a theoretically grounded dataset, which we use to fine-tune a 7B-parameter student model. We compare Direct Preference Optimization (DPO) and a novel Offline Group Relative Policy Optimization (O-GRPO); our 7B model significantly outperforms larger instruction-tuned baselines and achieves performance competitive with state-of-the-art proprietary models. We find that cognitive-driven data curation is far more critical than alignment algorithms or model scale for humor generation. Code and data will be available upon publication.
>
---
#### [new 123] SEPTQ: A Simple and Effective Post-Training Quantization Paradigm for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型量化过程中计算复杂和性能下降的问题。提出SEPTQ方法，简化量化流程并提升低比特下的效果。**

- **链接: [https://arxiv.org/pdf/2604.10091](https://arxiv.org/pdf/2604.10091)**

> **作者:** Han Liu; Haotian Gao; Xiaotong Zhang; Changya Li; Feng Zhang; Wei Wang; Fenglong Ma; Hong Yu
>
> **备注:** Accepted to KDD 2025. 12 pages, 10 figures
>
> **摘要:** Large language models (LLMs) have shown remarkable performance in various domains, but they are constrained by massive computational and storage costs. Quantization, an effective technique for compressing models to fit resource-limited devices while preserving generative quality, encompasses two primary methods: quantization aware training (QAT) and post-training quantization (PTQ). QAT involves additional retraining or fine-tuning, thus inevitably resulting in high training cost and making it unsuitable for LLMs. Consequently, PTQ has become the research hotspot in recent quantization methods. However, existing PTQ methods usually rely on various complex computation procedures and suffer from considerable performance degradation under low-bit quantization settings. To alleviate the above issues, we propose a simple and effective post-training quantization paradigm for LLMs, named SEPTQ. Specifically, SEPTQ first calculates the importance score for each element in the weight matrix and determines the quantization locations in a static global manner. Then it utilizes the mask matrix which represents the important locations to quantize and update the associated weights column-by-column until the appropriate quantized weight matrix is obtained. Compared with previous methods, SEPTQ simplifies the post-training quantization procedure into only two steps, and considers the effectiveness and efficiency simultaneously. Experimental results on various datasets across a suite of models ranging from millions to billions in different quantization bit-levels demonstrate that SEPTQ significantly outperforms other strong baselines, especially in low-bit quantization scenarios.
>
---
#### [new 124] Legal2LogicICL: Improving Generalization in Transforming Legal Cases to Logical Formulas via Diverse Few-Shot Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于法律文本到逻辑公式的转换任务，旨在解决数据稀缺导致的模型泛化能力不足问题。通过引入多样化的少样本学习方法，提升法律推理系统的准确性与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.11699](https://arxiv.org/pdf/2604.11699)**

> **作者:** Jieying Xue; Phuong Minh Nguyen; Ha Thanh Nguyen; May Myo Zin; Ken Satoh
>
> **备注:** Accepted at ICAIL 2026
>
> **摘要:** This work aims to improve the generalization of logic-based legal reasoning systems by integrating recent advances in NLP with legal-domain adaptive few-shot learning techniques using LLMs. Existing logic-based legal reasoning pipelines typically rely on fine-tuned models to map natural-language legal cases into logical formulas before forwarding them to a symbolic reasoner. However, such approaches are heavily constrained by the scarcity of high-quality annotated training data. To address this limitation, we propose a novel LLM-based legal reasoning framework that enables effective in-context learning through retrieval-augmented generation. Specifically, we introduce Legal2LogicICL, a few-shot retrieval framework that balances diversity and similarity of exemplars at both the latent semantic representation level and the legal text structure level. In addition, our method explicitly accounts for legal structure by mitigating entity-induced retrieval bias in legal texts, where lengthy and highly specific entity mentions often dominate semantic representations and obscure legally meaningful reasoning patterns. Our Legal2LogicICL constructs informative and robust few-shot demonstrations, leading to accurate and stable logical rule generation without requiring additional training. In addition, we construct a new dataset, named Legal2Proleg, which is annotated with alignments between legal cases and PROLEG logical formulas to support the evaluation of legal semantic parsing. Experimental results on both open-source and proprietary LLMs demonstrate that our approach significantly improves accuracy, stability, and generalization in transforming natural-language legal case descriptions into logical representations, highlighting its effectiveness for interpretable and reliable legal reasoning. Our code is available at this https URL.
>
---
#### [new 125] AOP-Smart: A RAG-Enhanced Large Language Model Framework for Adverse Outcome Pathway Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AOP分析任务，旨在解决LLM在AOP知识任务中的幻觉问题。提出AOP-Smart框架，通过RAG提升模型准确性。**

- **链接: [https://arxiv.org/pdf/2604.10874](https://arxiv.org/pdf/2604.10874)**

> **作者:** Qinjiang Niu; Lu Yan
>
> **摘要:** Adverse Outcome Pathways (AOPs) are an important knowledge framework in toxicological research and risk assessment. In recent years, large language models (LLMs) have gradually been applied to AOP-related question answering and mechanistic reasoning tasks. However, due to the existence of the hallucination problem, that is, the model may generate content that is inconsistent with facts or lacks evidence, their reliability is still limited. To address this issue, this study proposes an AOP-oriented Retrieval-Augmented Generation (RAG) framework, AOP-Smart. Based on the official XML data from AOP-Wiki, this method uses Key Events (KEs), Key Event Relationships (KERs), and specific AOP information to retrieve relevant knowledge for user questions, thereby improving the reliability of the generated results of large language models. To evaluate the effectiveness of the proposed method, this study constructed a test set containing 20 AOP-related question answering tasks, covering KE identification, upstream and downstream KE retrieval, and complex AOP retrieval tasks. Experiments were conducted on three mainstream large language models, Gemini, DeepSeek, and ChatGPT, and comparative tests were performed under two settings: without RAG and with RAG. The experimental results show that, without using RAG, the accuracies of GPT, DeepSeek, and Gemini were 15.0\%, 35.0\%, and 20.0\%, respectively; after using RAG, their accuracies increased to 95.0\%, 100.0\%, and 95.0\%, respectively. The results indicate that AOP-Smart can significantly alleviate the hallucination problem of large language models in AOP knowledge tasks, and greatly improve the accuracy and consistency of their answers.
>
---
#### [new 126] Weird Generalization is Weirdly Brittle
- **分类: cs.CL**

- **简介: 该论文属于AI安全领域，研究奇怪泛化现象。旨在解决模型在特定数据上微调后出现意外行为的问题，通过实验验证其脆弱性并提出有效干预方法。**

- **链接: [https://arxiv.org/pdf/2604.10022](https://arxiv.org/pdf/2604.10022)**

> **作者:** Miriam Wanner; Hannah Collison; William Jurayj; Benjamin Van Durme; Mark Dredze; William Walden
>
> **摘要:** Weird generalization is a phenomenon in which models fine-tuned on data from a narrow domain (e.g. insecure code) develop surprising traits that manifest even outside that domain (e.g. broad misalignment)-a phenomenon that prior work has highlighted as a critical safety concern. Here, we present an extended replication study of key weird generalization results across an expanded suite of models and datasets. We confirm that surprising (and dangerous) traits can emerge under certain circumstances, but we find that weird generalization is exceptionally brittle: it emerges only for specific models on specific datasets, and it vanishes under simple training-time, prompt-based interventions. We find that the most effective interventions provide prompt context that makes the generalized behavior the expected behavior. However, we show that even very generic interventions that do not anticipate specific generalized traits can still be effective in mitigating weird generalization's effects. Our findings thus help clarify the nature of the safety threat that weird generalization poses and point toward an easily implemented set of solutions.
>
---
#### [new 127] Computational Lesions in Multilingual Language Models Separate Shared and Language-specific Brain Alignment
- **分类: cs.CL; cs.AI; cs.CE**

- **简介: 该论文属于多语言神经网络研究任务，旨在探讨大脑语言处理是共享还是语言特异。通过构建计算损伤，分析模型与脑活动的关联，揭示共享核心与语言特化机制。**

- **链接: [https://arxiv.org/pdf/2604.10627](https://arxiv.org/pdf/2604.10627)**

> **作者:** Yang Cui; Jingyuan Sun; Yizheng Sun; Yifan Wang; Yunhao Zhang; Jixing Li; Shaonan Wang; Hongpeng Zhou; John Hale; Chengqing Zong; Goran Nenadic
>
> **备注:** 23 pages, 5 figures, Journal format
>
> **摘要:** How the brain supports language across different languages is a basic question in neuroscience and a useful test for multilingual artificial intelligence. Neuroimaging has identified language-responsive brain regions across languages, but it cannot by itself show whether the underlying processing is shared or language-specific. Here we use six multilingual large language models (LLMs) as controllable systems and create targeted ``computational lesions'' by zeroing small parameter sets that are important across languages or especially important for one language. We then compare intact and lesioned models in predicting functional magnetic resonance imaging (fMRI) responses during 100 minutes of naturalistic story listening in native English, Chinese and French (112 participants). Lesioning a compact shared core reduces whole-brain encoding correlation by 60.32% relative to intact models, whereas language-specific lesions preserve cross-language separation in embedding space but selectively weaken brain predictivity for the matched native language. These results support a shared backbone with embedded specializations and provide a causal framework for studying multilingual brain-model alignment.
>
---
#### [new 128] ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization
- **分类: cs.CL; cs.AI; cs.PL**

- **简介: 该论文属于代码摘要事实一致性评估任务，解决真实场景下多句功能和依赖关系的细粒度评估问题。提出ReFEree方法，无需参考文本即可准确评估事实一致性。**

- **链接: [https://arxiv.org/pdf/2604.10520](https://arxiv.org/pdf/2604.10520)**

> **作者:** Suyoung Bae; CheolWon Na; Jaehoon Lee; Yumin Lee; YunSeok Choi; Jee-Hyong Lee
>
> **备注:** Accepted to ACL 2026 main. 25 pages
>
> **摘要:** As Large Language Models (LLMs) have become capable of generating long and descriptive code summaries, accurate and reliable evaluation of factual consistency has become a critical challenge. However, previous evaluation methods are primarily designed for short summaries of isolated code snippets. Consequently, they struggle to provide fine-grained evaluation of multi-sentence functionalities and fail to accurately assess dependency context commonly found in real-world code summaries. To address this, we propose ReFEree, a reference-free and fine-grained method for evaluating factual consistency in real-world code summaries. We define factual inconsistency criteria specific to code summaries and evaluate them at the segment level using these criteria along with dependency information. These segment-level results are then aggregated into a fine-grained score. We construct a code summarization benchmark with human-annotated factual consistency labels. The evaluation results demonstrate that ReFEree achieves the highest correlation with human judgment among 13 baselines, improving 15-18% over the previous state-of-the-art. Our code and data are available at this https URL.
>
---
#### [new 129] Think in Sentences: Explicit Sentence Boundaries Enhance Language Model's Capabilities
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型能力。解决现有方法忽略句子结构的问题，通过在句末插入分隔符，增强模型的句级处理能力。**

- **链接: [https://arxiv.org/pdf/2604.10135](https://arxiv.org/pdf/2604.10135)**

> **作者:** Zhichen Liu; Yongyuan Li; Yang Xu
>
> **备注:** Accepted to ACL 2026 main conference
>
> **摘要:** Researchers have explored different ways to improve large language models (LLMs)' capabilities via dummy token insertion in contexts. However, existing works focus solely on the dummy tokens themselves, but fail to leverage the inherent sentence-level structure of natural language. This is a critical oversight, as LLMs acquire linguistic capabilities through exposure to human-generated texts, which are inherently structured at the sentence level. Motivated by this gap, we propose an approach that inserts delimiters at sentence boundaries in LLM inputs, which not only integrates dummy tokens into the context, but also facilitates LLMs with sentence-by-sentence processing behavior during reasoning. Two concrete methods: (1). In-context learning and (2). Supervised fine-tuning are experimented using 7B models to 600B Deepseek-V3. Our results demonstrate consistent improvements across various tasks, with notable gains of up to 7.7\% on GSM8k and 12.5\% on DROP. Furthermore, the fine-tuned LLMs can incorporate sentence awareness evidenced by their internal representations. Our work establishes a simple yet effective technique for enhancing LLM's capabilities, offering promising directions for cognitive-inspired LLM enhancement paradigm.
>
---
#### [new 130] Bridging Linguistic Gaps: Cross-Lingual Mapping in Pre-Training and Dataset for Enhanced Multilingual LLM Performance
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言语言模型任务，旨在解决跨语言任务中的数据不平衡和单语偏差问题。通过引入跨语言映射任务，提升模型的跨语言对齐能力。**

- **链接: [https://arxiv.org/pdf/2604.10590](https://arxiv.org/pdf/2604.10590)**

> **作者:** Weihua Zheng; Chang Liu; Zhengyuan Liu; Xin Huang; Kui Wu; Muhammad Huzaifah Md Shahrin; Aiti Aw; Roy Ka-Wei Lee
>
> **摘要:** Multilingual Large Language Models (LLMs) struggle with cross-lingual tasks due to data imbalances between high-resource and low-resource languages, as well as monolingual bias in pre-training. Existing methods, such as bilingual fine-tuning and contrastive alignment, can improve cross-lingual performance, but they often require extensive parallel data or suffer from instability. To address these challenges, we introduce a Cross-Lingual Mapping Task during the pre-training phase, which enhances cross-lingual alignment without compromising monolingual fluency. Our approach bi-directionally maps languages within the LLM embedding space, improving both language generation and comprehension. We further propose a Language Alignment Coefficient to robustly quantify cross-lingual consistency, even in limited-data scenarios. Experimental results on machine translation (MT), cross-lingual natural language understanding (CLNLU), and cross-lingual question answering (CLQA) show that our model achieves gains of up to 11.9 BLEU points in MT, 6.72 points in CLQA BERTScore-Precision, and more than 5% in CLNLU accuracy over strong multilingual baselines. These findings highlight the potential of incorporating cross-lingual objectives into pre-training to improve multilingual LLMs.
>
---
#### [new 131] HeceTokenizer: A Syllable-Based Tokenization Approach for Turkish Retrieval
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出HeceTokenizer，用于土耳其语检索任务，解决词汇外（OOV）问题。通过音节分词和BERT-tiny模型，实现高效检索，取得50.3%的Recall@5。**

- **链接: [https://arxiv.org/pdf/2604.10665](https://arxiv.org/pdf/2604.10665)**

> **作者:** Senol Gulgonul
>
> **摘要:** HeceTokenizer is a syllable-based tokenizer for Turkish that exploits the deterministic six-pattern phonological structure of the language to construct a closed, out-of-vocabulary (OOV)-free vocabulary of approximately 8,000 unique syllable types. A BERT-tiny encoder (1.5M parameters) is trained from scratch on a subset of Turkish Wikipedia using a masked language modeling objective and evaluated on the TQuAD retrieval benchmark using Recall@5. Combined with a fine-grained chunk-based retrieval strategy, HeceTokenizer achieves 50.3% Recall@5, surpassing the 46.92% reported by a morphology-driven baseline that uses a 200 times larger model. These results suggest that the phonological regularity of Turkish syllables provides a strong and resource-light inductive bias for retrieval tasks.
>
---
#### [new 132] CONSCIENTIA: Can LLM Agents Learn to Strategize? Emergent Deception and Trust in a Multi-Agent NYC Simulation
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文属于多智能体策略学习任务，研究LLM在对抗环境中如何产生策略行为，包括欺骗与信任。通过模拟纽约市环境，分析蓝红代理的互动与优化策略。**

- **链接: [https://arxiv.org/pdf/2604.09746](https://arxiv.org/pdf/2604.09746)**

> **作者:** Aarush Sinha; Arion Das; Soumyadeep Nag; Charan Karnati; Shravani Nag; Chandra Vadhan Raj; Aman Chadha; Vinija Jain; Suranjana Trivedy; Amitava Das
>
> **摘要:** As large language models (LLMs) are increasingly deployed as autonomous agents, understanding how strategic behavior emerges in multi-agent environments has become an important alignment challenge. We take a neutral empirical stance and construct a controlled environment in which strategic behavior can be directly observed and measured. We introduce a large-scale multi-agent simulation in a simplified model of New York City, where LLM-driven agents interact under opposing incentives. Blue agents aim to reach their destinations efficiently, while Red agents attempt to divert them toward billboard-heavy routes using persuasive language to maximize advertising revenue. Hidden identities make navigation socially mediated, forcing agents to decide when to trust or deceive. We study policy learning through an iterative simulation pipeline that updates agent policies across repeated interaction rounds using Kahneman-Tversky Optimization (KTO). Blue agents are optimized to reduce billboard exposure while preserving navigation efficiency, whereas Red agents adapt to exploit remaining weaknesses. Across iterations, the best Blue policy improves task success from 46.0% to 57.3%, although susceptibility remains high at 70.7%. Later policies exhibit stronger selective cooperation while preserving trajectory efficiency. However, a persistent safety-helpfulness trade-off remains: policies that better resist adversarial steering do not simultaneously maximize task completion. Overall, our results show that LLM agents can exhibit limited strategic behavior, including selective trust and deception, while remaining highly vulnerable to adversarial persuasion.
>
---
#### [new 133] Agentic Driving Coach: Robustness and Determinism of Agentic AI-Powered Human-in-the-Loop Cyber-Physical Systems
- **分类: cs.AI; cs.CL; cs.RO; eess.SY**

- **简介: 该论文属于AI与控制系统任务，解决agentic HITL CPS的非确定性问题，提出基于LF框架的方法，并通过驾驶教练案例验证。**

- **链接: [https://arxiv.org/pdf/2604.11705](https://arxiv.org/pdf/2604.11705)**

> **作者:** Deeksha Prahlad; Daniel Fan; Hokeun Kim
>
> **摘要:** Foundation models, including large language models (LLMs), are increasingly used for human-in-the-loop (HITL) cyber-physical systems (CPS) because foundation model-based AI agents can potentially interact with both the physical environments and human users. However, the unpredictable behavior of human users and AI agents, in addition to the dynamically changing physical environments, leads to uncontrollable nondeterminism. To address this urgent challenge of enabling agentic AI-powered HITL CPS, we propose a reactor-model-of-computation (MoC)-based approach, realized by the open-source Lingua Franca (LF) framework. We also carry out a concrete case study using the agentic driving coach as an application of HITL CPS. By evaluating the LF-based agentic HITL CPS, we identify practical challenges in reintroducing determinism into such agentic HITL CPS and present pathways to address them.
>
---
#### [new 134] PatchRecall: Patch-Driven Retrieval for Automated Program Repair
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于程序修复任务，解决自动化程序修复中如何高效高召回地检索相关文件的问题。提出PatchRecall方法，结合代码库和历史问题信息，提升召回率同时保持检索简洁。**

- **链接: [https://arxiv.org/pdf/2604.10481](https://arxiv.org/pdf/2604.10481)**

> **作者:** Mahir Labib Dihan; Faria Binta Awal; Md. Ishrak Ahsan
>
> **备注:** Code is available at this https URL
>
> **摘要:** Retrieving the correct set of files from a large codebase is a crucial step in Automated Program Repair (APR). High recall is necessary to ensure that the relevant files are included, but simply increasing the number of retrieved files introduces noise and degrades efficiency. To address this tradeoff, we propose PatchRecall, a hybrid retrieval approach that balances recall with conciseness. Our method combines two complementary strategies: (1) codebase retrieval, where the current issue description is matched against the codebase to surface potentially relevant files, and (2) history-based retrieval, where similar past issues are leveraged to identify edited files as candidate targets. Candidate files from both strategies are merged and reranked to produce the final retrieval set. Experiments on SWE-Bench demonstrate that PatchRecall achieves higher recall without significantly increasing retrieved file count, enabling more effective APR.
>
---
#### [new 135] COMPOSITE-Stem
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出COMPOSITE-STEM基准，用于评估AI在科学领域的推理能力。针对现有基准饱和的问题，通过专家任务和灵活评分机制，衡量AI在物理、生物、化学和数学中的表现。**

- **链接: [https://arxiv.org/pdf/2604.09836](https://arxiv.org/pdf/2604.09836)**

> **作者:** Kyle Waters; Lucas Nuzzi; Tadhg Looram; Alessandro Tomasiello; Ariel Ghislain Kemogne Kamdoum; Bikun Li; Damien Sileo; Egor Kretov; Francesco Fournier-Facio; Georgios Soloupis; Haile Kassahun; Hew Wolff; Jiaqi Cai; Lianghui Li; Marc Roth; Mohinder Naiya; Naixu Guo; Qicheng Tang; Richard Wheeler; Samuele Sala; Serguei Popov; Steven Dillman; Yuqi Li
>
> **摘要:** AI agents hold growing promise for accelerating scientific discovery; yet, a lack of frontier evaluations hinders adoption into real workflows. Expert-written benchmarks have proven effective at measuring AI reasoning, but most at this stage have become saturated and only measure performance on constrained outputs. To help address this gap, we introduce COMPOSITE-STEM, a benchmark of 70 expert-written tasks in physics, biology, chemistry, and mathematics, curated by doctoral-level researchers. Our benchmark combines exact-match grading and criterion-based rubrics with an LLM-as-a-jury grading protocol, allowing more flexible assessment of scientifically meaningful outputs. Using an adapted multimodal Terminus-2 agent harness within the Harbor agentic evaluation framework, we evaluate four frontier models. The top-performing model achieves 21%, demonstrating that COMPOSITE-STEM captures capabilities beyond current agent reach. All tasks are open-sourced with contributor permission to support reproducibility and to promote additional research towards AI's acceleration of scientific progress in these domains.
>
---
#### [new 136] SpectralLoRA: Is Low-Frequency Structure Sufficient for LoRA Adaptation? A Spectral Analysis of Weight Updates
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究LoRA适配中的频谱结构，分析低频成分是否足够。属于模型压缩任务，旨在提升适配器效率，通过DCT分析发现低频占主导，验证了频谱稀疏性原则。**

- **链接: [https://arxiv.org/pdf/2604.10649](https://arxiv.org/pdf/2604.10649)**

> **作者:** Rajveer Singh
>
> **备注:** 11 pages, 6 figures, 7 tables. Indian Institute of Technology Roorkee
>
> **摘要:** We present a systematic empirical study of the spectral structure of LoRA weight updates. Through 2D Discrete Cosine Transform (DCT) analysis of trained adaptation matrices across BERT-base and RoBERTa-base on four GLUE benchmarks (SST-2, MNLI, CoLA, QQP), we establish that LoRA updates are universally dominated by low-frequency components: on average, just 33% of DCT coefficients capture 90% of total spectral energy. Retaining only 10% of frequency coefficients reduces adapter storage by 10x while sacrificing only 1.95pp on SST-2. Notably, frequency masking at k=50% improves over full LoRA on 3 of 8 model-task pairs, suggesting high-frequency components act as adaptation noise. We further discover that RoBERTa-base is systematically more spectrally compressible than BERT-base across all tasks, and that task complexity governs spectral sensitivity -- NLI tasks require more frequency budget than sentiment classification. These findings motivate a new design principle for PEFT: spectral sparsity in adaptation.
>
---
#### [new 137] NSFL: A Post-Training Neuro-Symbolic Fuzzy Logic Framework for Boolean Operators in Neural Embeddings
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出NSFL框架，解决神经嵌入中布尔运算的逻辑约束问题。通过融合模糊逻辑与神经符号方法，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2604.10604](https://arxiv.org/pdf/2604.10604)**

> **作者:** Vladi Vexler; Ofer Idan; Gil Lederman; Dima Sivov
>
> **备注:** 23 pages (16 main + 7 appendix), 2 figures, 10 tables, 1 algorithm
>
> **摘要:** Standard dense retrievers lack a native calculus for multi-atom logical constraints. We introduce Neuro-Symbolic Fuzzy Logic (NSFL), a framework that adapts formal t-norms and t-conorms to neural embedding spaces without requiring retraining. NSFL operates as a first-order hybrid calculus: it anchors logical operations on isolated zero-order similarity scores while actively steering representations using Neuro-Symbolic Deltas (NS-Delta) -- the first-order marginal differences derived from contextual fusion. This preserves pure atomic meaning while capturing domain reliance, preventing the representation collapse and manifold escape endemic to traditional geometric baselines. For scalable real-time retrieval, Spherical Query Optimization (SQO) leverages Riemannian optimization to project these fuzzy formulas into manifold-stable query vectors. Validated across six distinct encoder configurations and two modalities (including zero-shot and SOTA fine-tuned models), NSFL yields mAP improvements up to +81%. Notably, NSFL provides an additive 20% average and up to 47% boost even when applied to encoders explicitly fine-tuned for logical reasoning. By establishing a training-free, order-aware calculus for high-dimensional spaces, this framework lays the foundation for future dynamic scaling and learned manifold logic.
>
---
#### [new 138] Teaching Language Models How to Code Like Learners: Conversational Serialization for Student Simulation
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于编程教育领域，旨在模拟学生编程学习过程。通过将学生代码与环境反馈转化为对话形式，训练语言模型更真实地反映学生调试行为。**

- **链接: [https://arxiv.org/pdf/2604.10720](https://arxiv.org/pdf/2604.10720)**

> **作者:** Charles Koutcheme; Arto Hellas; Juho Leinonen
>
> **备注:** 8 pages, 2 figures, 2 tables. Accepted to Educational Data Mining 2026
>
> **摘要:** Artificial models that simulate how learners act and respond within educational systems are a promising tool for evaluating tutoring strategies and feedback mechanisms at scale. However, many existing approaches in programming education rely on prompting large, proprietary language models, raising concerns around privacy, cost, and dependence. In this work, we propose a method for training open-weight artificial programming learners using authentic student process data. Our approach serializes temporal log traces into a conversational format, representing each student's problem-solving process as a dialogue between the learner and their automated assessment system. Student code submissions and environment feedback, such as test outcomes, grades, and error traces, form alternating conversational turns, enabling models to learn from the iterative debugging process. We additionally introduce a training pipeline combining supervised fine-tuning with preference optimization to align models with authentic student debugging behavior. We evaluate our framework by training Qwen models at 4B and 8B scales on a large-scale dataset of real student submissions to Python programming assignments. Our results show that incorporating environment feedback strengthens the models' ability to replicate student debugging behavior, improving over both prior code-only approaches and prompted large language models baselines in functional alignment and code similarity. We release our code to support reproducibility.
>
---
#### [new 139] Towards Proactive Information Probing: Customer Service Chatbots Harvesting Value from Conversation
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出主动信息探查任务，解决聊天机器人如何高效获取有价值信息的问题。工作包括定义任务、设计PROCHATIP框架，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2604.11077](https://arxiv.org/pdf/2604.11077)**

> **作者:** Chen Huang; Zitan Jiang; Changyi Zou; Wenqiang Lei; See-Kiong Ng
>
> **备注:** Findings of ACL 2026
>
> **摘要:** Customer service chatbots are increasingly expected to serve not merely as reactive support tools for users, but as strategic interfaces for harvesting high-value information and business intelligence. In response, we make three main contributions. 1) We introduce and define a novel task of Proactive Information Probing, which optimizes when to probe users for pre-specified target information while minimizing conversation turns and user friction. 2) We propose PROCHATIP, a proactive chatbot framework featuring a specialized conversation strategy module trained to master the delicate timing of probes. 3) Experiments demonstrate that PROCHATIP significantly outperforms baselines, exhibiting superior capability in both information probing and service quality. We believe that our work effectively redefines the commercial utility of chatbots, positioning them as scalable, cost-effective engines for proactive business intelligence. Our code is available at this https URL.
>
---
#### [new 140] SWE-AGILE: A Software Agent Framework for Efficiently Managing Dynamic Reasoning Context
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于软件工程任务，解决多轮推理中上下文爆炸与信息丢失问题。提出SWE-AGILE框架，通过动态推理上下文和摘要压缩提升效率。**

- **链接: [https://arxiv.org/pdf/2604.11716](https://arxiv.org/pdf/2604.11716)**

> **作者:** Shuquan Lian; Juncheng Liu; Yazhe Chen; Yuhong Chen; Hui Li
>
> **摘要:** Prior representative ReAct-style approaches in autonomous Software Engineering (SWE) typically lack the explicit System-2 reasoning required for deep analysis and handling complex edge cases. While recent reasoning models demonstrate the potential of extended Chain-of-Thought (CoT), applying them to the multi-turn SWE task creates a fundamental dilemma: retaining full reasoning history leads to context explosion and ``Lost-in-the-Middle'' degradation, while discarding it would force the agent to redundantly re-reason at every step. To address these challenges, we propose SWE-AGILE, a novel software agent framework designed to bridge the gap between reasoning depth, efficiency, and context constraints. SWE-AGILE introduces a Dynamic Reasoning Context strategy, maintaining a ``sliding window'' of detailed reasoning for immediate continuity to prevent redundant re-analyzing, while compressing historical reasoning content into concise Reasoning Digests. Empirically, SWE-AGILE sets a new standard for 7B-8B models on SWE-Bench-Verified using only 2.2k trajectories and 896 tasks. Code is available at this https URL.
>
---
#### [new 141] Geometry-Aware Localized Watermarking for Copyright Protection in Embedding-as-a-Service
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于版权保护任务，针对EaaS模型易被盗用的问题，提出GeoMark框架，通过几何感知的局部水印技术，实现鲁棒且准确的版权验证。**

- **链接: [https://arxiv.org/pdf/2604.11344](https://arxiv.org/pdf/2604.11344)**

> **作者:** Zhimin Chen; Xiaojie Liang; Wenbo Xu; Yuxuan Liu; Wei Lu
>
> **摘要:** Embedding-as-a-Service (EaaS) has become an important semantic infrastructure for natural language and multimedia applications, but it is highly vulnerable to model stealing and copyright infringement. Existing EaaS watermarking methods face a fundamental robustness--utility--verifiability tension: trigger-based methods are fragile to paraphrasing, transformation-based methods are sensitive to dimensional perturbation, and region-based methods may incur false positives due to coincidental geometric affinity. To address this problem, we propose GeoMark, a geometry-aware localized watermarking framework for EaaS copyright protection. GeoMark uses a natural in-manifold embedding as a shared watermark target, constructs geometry-separated anchors with explicit target--anchor margins, and activates watermark injection only within adaptive local neighborhoods. This design decouples where watermarking is triggered from what ownership is attributed to, achieving localized triggering and centralized attribution. Experiments on four benchmark datasets show that GeoMark preserves downstream utility and geometric fidelity while maintaining robust copyright verification under paraphrasing, dimensional perturbation, and CSE (Clustering, Selection, Elimination) attacks, with improved verification stability and low false-positive risk.
>
---
#### [new 142] RECIPER: A Dual-View Retrieval Pipeline for Procedure-Oriented Materials Question Answering
- **分类: eess.SP; cs.AI; cs.CL**

- **简介: 该论文属于材料科学问答任务，解决过程信息检索难题。通过双视角检索管道RECIPER，结合段落和程序摘要，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2604.11229](https://arxiv.org/pdf/2604.11229)**

> **作者:** Zhuoyu Wu; Wenhui Ou; Pei-Sze Tan; Wenqi Fang; Sailaja Rajanala; Raphaël C.-W. Phan
>
> **备注:** 5 pages, 1 figure
>
> **摘要:** Retrieving procedure-oriented evidence from materials science papers is difficult because key synthesis details are often scattered across long, context-heavy documents and are not well captured by paragraph-only dense retrieval. We present RECIPER, a dual-view retrieval pipeline that indexes both paragraph-level context and compact large language model-extracted procedural summaries, then combines the two candidate streams with lightweight lexical reranking. Across four dense retrieval backbones, RECIPER consistently improves early-rank retrieval over paragraph-only dense retrieval, achieving average gains of +3.73 in Recall@1, +2.85 in nDCG@10, and +3.13 in MRR. With BGE-large-en-v1.5, it reaches 86.82%, 97.07%, and 97.85% on Recall@1, Recall@5, and Recall@10, respectively. We further observe improved downstream question answering under automatic metrics, suggesting that procedural summaries can serve as a useful complementary retrieval signal for procedure-oriented materials question answering. Code and data are available at this https URL.
>
---
#### [new 143] Sign Language Recognition in the Age of LLMs
- **分类: cs.CV; cs.CL**

- **简介: 论文研究了在零样本条件下，视觉语言模型（VLMs）是否能有效识别手语。任务属于手语识别，旨在评估VLMs在无需特定训练下的表现。工作包括实验对比不同VLMs在WLASL300数据集上的效果。**

- **链接: [https://arxiv.org/pdf/2604.11225](https://arxiv.org/pdf/2604.11225)**

> **作者:** Vaclav Javorek; Jakub Honzik; Ivan Gruber; Tomas Zelezny; Marek Hruz
>
> **备注:** Accepted at the CVPR 2026 Workshop on Multimodal Sign Language Research (MSLR), 8 pages, 3 figures
>
> **摘要:** Recent Vision Language Models (VLMs) have demonstrated strong performance across a wide range of multimodal reasoning tasks. This raises the question of whether such general-purpose models can also address specialized visual recognition problems such as isolated sign language recognition (ISLR) without task-specific training. In this work, we investigate the capability of modern VLMs to perform ISLR in a zero-shot setting. We evaluate several open-source and proprietary VLMs on the WLASL300 benchmark. Our experiments show that, under prompt-only zero-shot inference, current open-source VLMs remain far behind classic supervised ISLR classifiers by a wide margin. However, follow-up experiments reveal that these models capture partial visual-semantic alignment between signs and text descriptions. Larger proprietary models achieve substantially higher accuracy, highlighting the importance of model scale and training data diversity. All our code is publicly available on GitHub.
>
---
#### [new 144] Bringing Value Models Back: Generative Critics for Value Modeling in LLM Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM中信用分配问题。通过引入生成式批评者，提升价值建模效果，改善RL性能。**

- **链接: [https://arxiv.org/pdf/2604.10701](https://arxiv.org/pdf/2604.10701)**

> **作者:** Zikang Shan; Han Zhong; Liwei Wang; Li Zhao
>
> **备注:** 16 pages including appendix, 4 figures
>
> **摘要:** Credit assignment is a central challenge in reinforcement learning (RL). Classical actor-critic methods address this challenge through fine-grained advantage estimation based on a learned value function. However, learned value models are often avoided in modern large language model (LLM) RL because conventional discriminative critics are difficult to train reliably. We revisit value modeling and argue that this difficulty is partly due to limited expressiveness. In particular, representation complexity theory suggests that value functions can be hard to approximate under the one-shot prediction paradigm used by existing value models, and our scaling experiments show that such critics do not improve reliably with scale. Motivated by this observation, we propose Generative Actor-Critic (GenAC), which replaces one-shot scalar value prediction with a generative critic that performs chain-of-thought reasoning before producing a value estimate. We further introduce In-Context Conditioning, which helps the critic remain calibrated to the current actor throughout training. GenAC improves value approximation, ranking reliability, and out-of-distribution generalization, and these gains translate into stronger downstream RL performance than both value-based and value-free baselines. Overall, our results suggest that stronger value modeling is a promising direction for improving credit assignment in LLM reinforcement learning.
>
---
#### [new 145] Visual Late Chunking: An Empirical Study of Contextual Chunking for Efficient Visual Document Retrieval
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 该论文属于视觉文档检索任务，解决多向量模型存储与计算成本高的问题。提出ColChunk框架，通过多模态晚期分块实现高效上下文表示。**

- **链接: [https://arxiv.org/pdf/2604.10167](https://arxiv.org/pdf/2604.10167)**

> **作者:** Yibo Yan; Mingdong Ou; Yi Cao; Jiahao Huo; Xin Zou; Shuliang Liu; James Kwok; Xuming Hu
>
> **备注:** Preprint
>
> **摘要:** Multi-vector models dominate Visual Document Retrieval (VDR) due to their fine-grained matching capabilities, but their high storage and computational costs present a major barrier to practical deployment. In this paper, we propose ColChunk, a plug-and-play framework that introduces multimodal late chunking to construct efficient, contextualized multi-vectors. Unlike existing pruning or fixed-token approaches, ColChunk employs hierarchical clustering on patch-level embeddings, fused with a 2D position prior to ensure spatial-semantic coherence. This adaptive grouping allows for a content-aware representation that preserves global context while drastically reducing the vector count. Evaluations across 24 VDR datasets demonstrate ColChunk achieves over a 90% reduction in storage requirements while simultaneously delivering a 9-point average improvement in nDCG@5 across representative single-vector models. ColChunk provides a practical solution for balancing retrieval accuracy and efficiency in visual document systems.
>
---
#### [new 146] The Salami Slicing Threat: Exploiting Cumulative Risks in LLM Systems
- **分类: cs.CR; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于安全防护任务，针对LLM的多轮越狱攻击问题，提出“萨拉米切片风险”概念及自动攻击框架Salami Attack，有效提升攻击成功率并探讨防御策略。**

- **链接: [https://arxiv.org/pdf/2604.11309](https://arxiv.org/pdf/2604.11309)**

> **作者:** Yihao Zhang; Kai Wang; Jiangrong Wu; Haolin Wu; Yuxuan Zhou; Zeming Wei; Dongxian Wu; Xun Chen; Jun Sun; Meng Sun
>
> **摘要:** Large Language Models (LLMs) face prominent security risks from jailbreaking, a practice that manipulates models to bypass built-in security constraints and generate unethical or unsafe content. Among various jailbreak techniques, multi-turn jailbreak attacks are more covert and persistent than single-turn counterparts, exposing critical vulnerabilities of LLMs. However, existing multi-turn jailbreak methods suffer from two fundamental limitations that affect the actual impact in real-world scenarios: (a) As models become more context-aware, any explicit harmful trigger is increasingly likely to be flagged and blocked; (b) Successful final-step triggers often require finely tuned, model-specific contexts, making such attacks highly context-dependent. To fill this gap, we propose \textit{Salami Slicing Risk}, which operates by chaining numerous low-risk inputs that individually evade alignment thresholds but cumulatively accumulate harmful intent to ultimately trigger high-risk behaviors, without heavy reliance on pre-designed contextual structures. Building on this risk, we develop Salami Attack, an automatic framework universally applicable to multiple model types and modalities. Rigorous experiments demonstrate its state-of-the-art performance across diverse models and modalities, achieving over 90\% Attack Success Rate on GPT-4o and Gemini, as well as robustness against real-world alignment defenses. We also proposed a defense strategy to constrain the Salami Attack by at least 44.8\% while achieving a maximum blocking rate of 64.8\% against other multi-turn jailbreak attacks. Our findings provide critical insights into the pervasive risks of multi-turn jailbreaking and offer actionable mitigation strategies to enhance LLM security.
>
---
#### [new 147] Reasoning Resides in Layers: Restoring Temporal Reasoning in Video-Language Models with Layer-Selective Merging
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视频语言模型任务，旨在解决视觉对齐导致的时间推理能力下降问题。提出MERIT框架，通过分层融合恢复时间推理，无需重新训练。**

- **链接: [https://arxiv.org/pdf/2604.11399](https://arxiv.org/pdf/2604.11399)**

> **作者:** Zihang Fu; Haonan Wang; Jian Kang; Kenji Kawaguchi; Jiaying Wu
>
> **摘要:** Multimodal adaptation equips large language models (LLMs) with perceptual capabilities, but often weakens the reasoning ability inherited from language-only pretraining. This trade-off is especially pronounced in video-language models (VLMs), where visual alignment can impair temporal reasoning (TR) over sequential events. We propose MERIT, a training-free, task-driven model merging framework for restoring TR in VLMs. MERIT searches over layer-wise self-attention merging recipes between a VLM and its paired text-only backbone using an objective that improves TR while penalizing degradation in temporal perception (TP). Across three representative VLMs and multiple challenging video benchmarks, MERIT consistently improves TR, preserves or improves TP, and generalizes beyond the search set to four distinct benchmarks. It also outperforms uniform full-model merging and random layer selection, showing that effective recovery depends on selecting the right layers. Interventional masking and frame-level attribution further show that the selected layers are disproportionately important for reasoning and shift model decisions toward temporally and causally relevant evidence. These results show that targeted, perception-aware model merging can effectively restore TR in VLMs without retraining.
>
---
#### [new 148] Learning from Emptiness: De-biasing Listwise Rerankers with Content-Agnostic Probability Calibration
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，解决列表重排序中的位置偏差问题。提出CapCal框架，在不增加训练成本的情况下有效去除偏差，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.10150](https://arxiv.org/pdf/2604.10150)**

> **作者:** Hang Lv; Hongchao Gu; Ruiqing Yang; Liangyue Li; Zulong Chen; Defu Lian; Hao Wang; Enhong Chen
>
> **备注:** ACL2026
>
> **摘要:** Generative listwise reranking leverages global context for superior retrieval but is plagued by intrinsic position bias, where models exhibit structural sensitivity to input order independent of relevance. Existing mitigations present a dilemma: inference-time aggregation incurs prohibitive latency, while training-based methods often fail to eradicate ingrained priors, particularly in compact models. To resolve this dilemma, we propose CapCal (Content-Agnostic Probability Calibration), a training-free framework that mechanically decouples positional bias from ranking decisions. By estimating the bias distribution via content-free placeholders, CapCal rectifies output logits through an entropy-adaptive contrastive mechanism. Evaluations across 10 benchmarks confirm that CapCal achieves superior performance among training-free methods while preserving single-pass efficiency. Notably, it unlocks the latent potential of lightweight models (e.g., 0.6B), delivering absolute NDCG gains exceeding 10 points and outperforming both permutation-based aggregation and data-augmentation baselines.
>
---
#### [new 149] Unifying Ontology Construction and Semantic Alignment for Deterministic Enterprise Reasoning at Scale
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于知识图谱构建与语义对齐任务，旨在解决企业数据混乱、决策信息不全的问题。提出LOM框架，整合 ontology 构建、语义对齐和逻辑推理，提升企业级智能决策能力。**

- **链接: [https://arxiv.org/pdf/2604.09608](https://arxiv.org/pdf/2604.09608)**

> **作者:** Hongyin Zhu
>
> **摘要:** While enterprises amass vast quantities of data, much of it remains chaotic and effectively dormant, preventing decision-making based on comprehensive information. Existing neuro-symbolic approaches rely on disjoint pipelines and struggle with error propagation. We introduce the large ontology model (LOM), a unified framework that seamlessly integrates ontology construction, semantic alignment, and logical reasoning into a single end-to-end architecture. LOM employs a construct-align-reason (CAR) pipeline, leveraging its unified architecture across all three stages: it first autonomously constructs a domain-specific ontological universe from raw data, then aligns neural generation with this structural reality using a graph-aware encoder and reinforcement learning, and finally executes deterministic reasoning over the constructed topology, node attributes and relation types. We evaluate LOM on a comprehensive benchmark constructed from diverse real-world enterprise datasets. Experimental results demonstrate that LOM-4B achieves 88.8% accuracy in ontology completion and 94% in complex graph reasoning tasks, significantly outperforming state-of-the-art LLMs. These findings validate that autonomous logical construction is essential for achieving deterministic, enterprise-grade intelligence.
>
---
#### [new 150] Thinking Fast, Thinking Wrong: Intuitiveness Modulates LLM Counterfactual Reasoning in Policy Evaluation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于因果推理任务，探讨LLMs在政策评估中的反事实推理能力。研究分析了直观性对模型表现的影响，揭示了推理策略与模型表现的关联。**

- **链接: [https://arxiv.org/pdf/2604.10511](https://arxiv.org/pdf/2604.10511)**

> **作者:** Yanjie He
>
> **备注:** 7 pages, 3 figures
>
> **摘要:** Large language models (LLMs) are increasingly used for causal and counterfactual reasoning, yet their reliability in real-world policy evaluation remains underexplored. We construct a benchmark of 40 empirical policy evaluation cases drawn from economics and social science, each grounded in peer-reviewed evidence and classified by intuitiveness -- whether the empirical finding aligns with (obvious), is unclear relative to (ambiguous), or contradicts (counter-intuitive) common prior expectations. We evaluate four frontier LLMs across five prompting strategies with 2,400 experimental trials and analyze the results using mixed-effects logistic regression. Our findings reveal three key results: (1) a chain-of-thought (CoT) paradox, where chain-of-thought prompting dramatically improves performance on obvious cases but this benefit is nearly eliminated on counter-intuitive ones (interaction OR = 0.053, $p < 0.001$); (2) intuitiveness as the dominant factor, explaining more variance than model choice or prompting strategy (ICC = 0.537); and (3) a knowledge-reasoning dissociation, where citation-based familiarity is unrelated to accuracy ($p = 0.53$), suggesting models possess relevant knowledge but fail to reason with it when findings contradict intuition. We frame these results through the lens of dual-process theory (System 1 vs. System 2) and argue that current LLMs' "slow thinking" may be little more than "slow talking" -- they produce the form of deliberative reasoning without the substance.
>
---
#### [new 151] Reproduction Beyond Benchmarks: ConstBERT and ColBERT-v2 Across Backends and Query Distributions
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文属于信息检索任务，研究多向量检索模型的可复现性问题。通过评估ConstBERT和ColBERT-v2，发现其在长文本查询上性能显著下降，原因在于架构限制而非数据或参数调整。**

- **链接: [https://arxiv.org/pdf/2604.09982](https://arxiv.org/pdf/2604.09982)**

> **作者:** Utshab Kumar Ghosh; Ashish David; Shubham Chatterjee
>
> **备注:** 10 pages, 9 tables. Accepted to the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2026)
>
> **摘要:** Reproducibility must validate architectural robustness, not just numerical accuracy. We evaluate ColBERT-v2 and ConstBERT across five dimensions, finding that while ConstBERT reproduces within 0.05% MRR@10 on MS-MARCO, both models show a drop of 86-97% on long, narrative queries (TREC ToT 2025). Ablations prove this failure is architectural: performance plateaus at 20 words because the MaxSim operator's uniform token weighting cannot distinguish signal from filler noise. Furthermore, undocumented backend parameters create an 8-point gap due to ConstBERT's sparse centroid coverage, and fine-tuning with 3x more data actually degrades performance by up to 29%. We conclude that architectural constraints in multi-vector retrieval cannot be overcome by adaptation alone. Code: this https URL.
>
---
#### [new 152] MCERF: Advancing Multimodal LLM Evaluation of Engineering Documentation with Enhanced Retrieval
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于多模态大模型评估任务，旨在提升工程文档问答的准确性。针对RAG系统在处理多模态工程文档时的不足，提出MCERF框架，结合多模态检索与推理策略，显著提高问答性能。**

- **链接: [https://arxiv.org/pdf/2604.09552](https://arxiv.org/pdf/2604.09552)**

> **作者:** Kiarash Naghavi Khanghah; Hoang Anh Nguyen; Anna C. Doris; Amir Mohammad Vahedi; Daniele Grandi; Faez Ahmed; Hongyi Xu
>
> **摘要:** Engineering rulebooks and technical standards contain multimodal information like dense text, tables, and illustrations that are challenging for retrieval augmented generation (RAG) systems. Building upon the DesignQA framework [1], which relied on full-text ingestion and text-based retrieval, this work establishes a Multimodal ColPali Enhanced Retrieval and Reasoning Framework (MCERF), a system that couples a multimodal retriever with large language model reasoning for accurate and efficient question answering from engineering documents. The system employs the ColPali, which retrieves both textual and visual information, and multiple retrieval and reasoning strategies: (i) Hybrid Lookup mode for explicit rule mentions, (ii) Vision to Text fusion for figure and table guided queries, (iii) High Reasoning LLM mode for complex multi modal questions, and (iv) SelfConsistency decision to stabilize responses. The modular framework design provides a reusable template for future multimodal systems regardless of underlying model architecture. Furthermore, this work establishes and compares two routing approaches: a single case routing approach and a multi-agent system, both of which dynamically allocate queries to optimal pipelines. Evaluation on the DesignQA benchmark illustrates that this system improves average accuracy across all tasks with a relative gain of +41.1% from baseline RAG best results, which is a significant improvement in multimodal and reasoning-intensive tasks without complete rulebook ingestion. This shows how vision language retrieval, modular reasoning, and adaptive routing enable scalable document comprehension in engineering use cases.
>
---
#### [new 153] Demographic and Linguistic Bias Evaluation in Omnimodal Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态语言模型的公平性评估任务，旨在检测和分析模型在不同人口统计和语言群体中的偏差。工作包括对四类模型在多个任务上的性能比较。**

- **链接: [https://arxiv.org/pdf/2604.10014](https://arxiv.org/pdf/2604.10014)**

> **作者:** Alaa Elobaid
>
> **备注:** Accepted at ICPR 2026. Full paper with complete appendix (31 pages total)
>
> **摘要:** This paper provides a comprehensive evaluation of demographic and linguistic biases in omnimodal language models that process text, images, audio, and video within a single framework. Although these models are being widely deployed, their performance across different demographic groups and modalities is not well studied. Four omnimodal models are evaluated on tasks that include demographic attribute estimation, identity verification, activity recognition, multilingual speech transcription, and language identification. Accuracy differences are measured across age, gender, skin tone, language, and country of origin. The results show that image and video understanding tasks generally exhibit better performance with smaller demographic disparities. In contrast, audio understanding tasks exhibit significantly lower performance and substantial bias, including large accuracy differences across age groups, genders, and languages, and frequent prediction collapse toward narrow categories. These findings highlight the importance of evaluating fairness across all supported modalities as omnimodal language models are increasingly used in real-world applications.
>
---
#### [new 154] Omnimodal Dataset Distillation via High-order Proxy Alignment
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于数据蒸馏任务，解决多模态数据压缩问题。提出HoPA方法，通过高阶对齐实现跨模态高效蒸馏。**

- **链接: [https://arxiv.org/pdf/2604.10666](https://arxiv.org/pdf/2604.10666)**

> **作者:** Yuxuan Gao; Xiaohao Liu; Xiaobo Xia; Tongliang Liu
>
> **摘要:** Dataset distillation compresses large-scale datasets into compact synthetic sets while preserving training performance, but existing methods are largely restricted to single-modal or bimodal settings. Extending dataset distillation to scenarios involving more than two modalities, i.e., Omnimodal Dataset Distillation, remains underexplored and challenging due to increased heterogeneity and complex cross-modal interactions. In this work, we identify the key determinant that bounds the endpoint discrepancy in the omnimodal setting, which is exacerbated with an increasing number of modalities. To this end, we propose HoPA, a unified method that captures high-order cross-modal alignments via a compact proxy, which is compatible with trajectory matching as well. By abstracting omnimodal alignment with a shared similarity structure, our method avoids the combinatorial complexity of pairwise modality modeling and enables scalable joint distillation across heterogeneous modalities. Theoretical analysis from the spectral perspective reveals the rationality of our proposed method against bimodal dataset distillation techniques. Extensive experiments on various benchmarks demonstrate that the proposed method achieves superior compression-performance trade-offs compared to existing competitors. The source code will be publicly released.
>
---
#### [new 155] Hierarchical Textual Knowledge for Enhanced Image Clustering
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文属于图像聚类任务，旨在解决视觉相似但语义不同的类别难以区分的问题。通过构建层次化文本知识增强特征，提升聚类效果。**

- **链接: [https://arxiv.org/pdf/2604.11144](https://arxiv.org/pdf/2604.11144)**

> **作者:** Yijie Zhong; Yunfan Gao; Weipeng Jiang; Haofen Wang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Image clustering aims to group images in an unsupervised fashion. Traditional methods focus on knowledge from visual space, making it difficult to distinguish between visually similar but semantically different classes. Recent advances in vision-language models enable the use of textual knowledge to enhance image clustering. However, most existing methods rely on coarse class labels or simple nouns, overlooking the rich conceptual and attribute-level semantics embedded in textual space. In this paper, we propose a knowledge-enhanced clustering (KEC) method that constructs a hierarchical concept-attribute structured knowledge with the help of large language models (LLMs) to guide clustering. Specifically, we first condense redundant textual labels into abstract concepts and then automatically extract discriminative attributes for each single concept and similar concept pairs, via structured prompts to LLMs. This knowledge is instantiated for each input image to achieve the knowledge-enhanced features. The knowledge-enhanced features with original visual features are adapted to various downstream clustering algorithms. We evaluate KEC on 20 diverse datasets, showing consistent improvements across existing methods using additional textual knowledge. KEC without training outperforms zero-shot CLIP on 14 out of 20 datasets. Furthermore, the naive use of textual knowledge may harm clustering performance, while KEC provides both accuracy and robustness.
>
---
#### [new 156] DuET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于测试用例生成任务，旨在提升LLM预测测试输出的可靠性。通过结合代码执行与伪代码推理，提出DuET框架，解决代码错误和幻觉问题，显著提升预测准确率。**

- **链接: [https://arxiv.org/pdf/2604.11514](https://arxiv.org/pdf/2604.11514)**

> **作者:** Hojae Han; Jaejin Kim; Seung-won Hwang; Yu Jin Kim; Moontae Lee
>
> **备注:** Findings of ACL 2026
>
> **摘要:** This work addresses test output prediction, a key challenge in test case generation. To improve the reliability of predicted outputs by LLMs, prior approaches generate code first to ground predictions. One grounding strategy is direct execution of generated code, but even minor errors can cause failures. To address this, we introduce LLM-based pseudocode execution, which grounds prediction on more error-resilient pseudocode and simulates execution via LLM reasoning. We further propose DuET, a dual-execution framework that combines both approaches by functional majority voting. Our analysis shows the two approaches are complementary in overcoming the limitations of direct execution suffering from code errors, and pseudocode reasoning from hallucination. On LiveCodeBench, DuET achieves the state-of-the-art performance, improving Pass@1 by 13.6 pp.
>
---
#### [new 157] Evaluating Small Open LLMs for Medical Question Answering: A Practical Framework
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于医疗问答任务，旨在评估小规模开放大语言模型的可靠性，解决模型输出一致性不足的问题。通过构建评估框架，分析模型在准确性和重复性上的表现。**

- **链接: [https://arxiv.org/pdf/2604.10535](https://arxiv.org/pdf/2604.10535)**

> **作者:** Avi-ad Avraam Buskila
>
> **摘要:** Incorporating large language models (LLMs) in medical question answering demands more than high average accuracy: a model that returns substantively different answers each time it is queried is not a reliable medical tool. Online health communities such as Reddit have become a primary source of medical information for millions of users, yet they remain highly susceptible to misinformation; deploying LLMs as assistants in these settings amplifies the need for output consistency alongside correctness. We present a practical, open-source evaluation framework for assessing small, locally-deployable open-weight LLMs on medical question answering, treating reproducibility as a first-class metric alongside lexical and semantic accuracy. Our pipeline computes eight quality metrics, including BERTScore, ROUGE-L, and an LLM-as-judge rubric, together with two within-model reproducibility metrics derived from repeated inference (N=10 runs per question). Evaluating three models (Llama 3.1 8B, Gemma 3 12B, MedGemma 1.5 4B) on 50 MedQuAD questions (N=1,500 total responses) reveals that despite low-temperature generation (T=0.2), self-agreement across runs reaches at most 0.20, while 87-97% of all outputs per model are unique -- a safety gap that single-pass benchmarks entirely miss. The clinically fine-tuned MedGemma 1.5 4B underperforms the larger general-purpose models on both quality and reproducibility; however, because MedGemma is also the smallest model, this comparison confounds domain fine-tuning with model scale. We describe the methodology in sufficient detail for practitioners to replicate or extend the evaluation for their own model-selection workflows. All code and data pipelines are available at this https URL.
>
---
#### [new 158] SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出SCOPE方法，用于优化强化学习中的策略蒸馏，解决稀疏奖励下的信用分配问题。通过双路径自适应加权，提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2604.10688](https://arxiv.org/pdf/2604.10688)**

> **作者:** Binbin Zheng; Xing Ma; Yiheng Liang; Jingqing Ruan; Xiaoliang Fu; Kepeng Lin; Benchang Zhu; Ke Zeng; Xunliang Cai
>
> **摘要:** On-policy reinforcement learning has become the dominant paradigm for reasoning alignment in large language models, yet its sparse, outcome-level rewards make token-level credit assignment notoriously difficult. On-Policy Distillation (OPD) alleviates this by introducing dense, token-level KL supervision from a teacher model, but typically applies this supervision uniformly across all rollouts, ignoring fundamental differences in signal quality. We propose Signal-Calibrated On-Policy Distillation Enhancement (SCOPE), a dual-path adaptive training framework that routes on-policy rollouts by correctness into two complementary supervision paths. For incorrect trajectories, SCOPE performs teacher-perplexity-weighted KL distillation to prioritize instances where the teacher demonstrates genuine corrective capability, while down-weighting unreliable guidance. For correct trajectories, it applies student-perplexity-weighted MLE to concentrate reinforcement on low-confidence samples at the capability boundary rather than over-reinforcing already mastered ones. Both paths employ a group-level normalization to adaptively calibrate weight distributions, accounting for the intrinsic difficulty variance across prompts. Extensive experiments on six reasoning benchmarks show that SCOPE achieves an average relative improvement of 11.42% in Avg@32 and 7.30% in Pass@32 over competitive baselines, demonstrating its consistent effectiveness.
>
---
#### [new 159] Back to the Barn with LLAMAs: Evolving Pretrained LLM Backbones in Finetuning Vision Language Models
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究预训练大语言模型对视觉语言模型性能的影响，探讨如何有效更新VLM以利用更先进的LLM。任务属于多模态学习，解决LLM进化对VLM效果的不确定性问题，通过对比不同LLM版本验证其影响。**

- **链接: [https://arxiv.org/pdf/2604.10985](https://arxiv.org/pdf/2604.10985)**

> **作者:** Sameera Horawalavithana; Lauren Phillips; Ian Stewart; Sai Munikoti; Karl Pazdernik
>
> **备注:** Preprint and under review
>
> **摘要:** Vision-Language Models (VLMs) have rapidly advanced by leveraging powerful pre-trained Large Language Models (LLMs) as core reasoning backbones. As new and more capable LLMs emerge with improved reasoning, instruction-following, and generalization, there is a pressing need to efficiently update existing VLMs to incorporate these advancements. However, the integration of new LLMs into VLMs, particularly how the evolving LLMs contribute to multimodal reasoning, alignment, and task-specific performance remains underexplored. Addressing this gap is important for VLM development, given the rapid evolution of pretrained LLM backbones. This study presents a controlled and systematic investigation of how changes in the pretrained LLM backbone affect downstream VLM task performance. By having the vision encoder, training data, and post-training algorithm remain same across LLAMA-1, LLAMA-2, and LLAMA-3 based VLMs, we find that newer LLM backbones do not always lead to better VLMs, but the performance depends on the downstream VLM task. For example, in visual question and answering tasks, newer LLM backbones tend to solve different questions rather than just more questions, and our analysis shows this is driven by differences in how the models process information, including better calibrated confidence and more stable internal representations. We also find that some VLM capabilities appear only in the newest LLM generation, while tasks that depend mainly on visual understanding see little benefit from a newer LLM backbone.
>
---
#### [new 160] Digital hybridity and relics in cultural heritage: using corpus linguistics to inform design in emerging technologies from AI to VR
- **分类: cs.HC; cs.AI; cs.CL; cs.CY; cs.DL; cs.LG**

- **简介: 论文探讨数字技术如何影响文化遗产中圣物的呈现与体验，通过语料库语言学分析历史与现代文本中的“圣物”表述，旨在为AI、VR等技术设计提供语言学依据，解决数字化过程中文化意义传达的问题。**

- **链接: [https://arxiv.org/pdf/2604.09669](https://arxiv.org/pdf/2604.09669)**

> **作者:** Emma McClaughlin; Glenn McGarry; Alan Chamberlain; Geert De Wilde; Oliver Butler
>
> **备注:** This is a (ACM J.5 Arts & Humanities Paper) relating to Hybrid Technologies, Language, AI, VR, Interaction and Experience. 24 pages. Int J Digit Humanities (2026)
>
> **摘要:** Hybrid technologies enable the blending of physical and digital elements, creating new ways to experience and interact with the world. Such technologies can transform engagement with relics, both secular and sacred but they present challenges for capturing faith, belief, and representation responsibly. Given the complexities of digital representation and the ethical challenges inherent in digitising culturally significant objects, a transdisciplinary understanding of these issues is needed. To inform this discussion from a linguistic perspective, we examined the representation of relics in historical and contemporary texts. Using a corpus linguistic approach to extract modifiers of the word relic in corpora of Early Modern English books and contemporary web sourced texts from 2021, we examined the multifaceted ways in which relics have been perceived and evaluated over time. Early texts consider relics as both objects of moral and spiritual significance, and tools of religious and political control, while they are more often framed as heritage symbols, reflecting past events, places, and traditions in contemporary texts. We discuss how hybrid, sometimes AI based technologies can enhance accessibility and engagement, whilst also challenging traditional sensitivities around authenticity and sensory experience, which are integral to the meaning and significance of relics.
>
---
#### [new 161] Generative UI: LLMs are Effective UI Generators
- **分类: cs.HC; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于人机交互任务，旨在解决LLM生成内容界面单一的问题。通过提示工程和工具支持，实现高质量自定义UI生成，提升用户体验。**

- **链接: [https://arxiv.org/pdf/2604.09577](https://arxiv.org/pdf/2604.09577)**

> **作者:** Yaniv Leviathan; Dani Valevski; Matan Kalman; Danny Lumen; Eyal Segalis; Eyal Molad; Shlomi Pasternak; Vishnu Natchu; Valerie Nygaard; Srinivasan; Venkatachary; James Manyika; Yossi Matias
>
> **摘要:** AI models excel at creating content, but typically render it with static, predefined interfaces. Specifically, the output of LLMs is often a markdown "wall of text". Generative UI is a long standing promise, where the model generates not just the content, but the interface itself. Until now, Generative UI was not possible in a robust fashion. We demonstrate that when properly prompted and equipped with the right set of tools, a modern LLM can robustly produce high quality custom UIs for virtually any prompt. When ignoring generation speed, results generated by our implementation are overwhelmingly preferred by humans over the standard LLM markdown output. In fact, while the results generated by our implementation are worse than those crafted by human experts, they are at least comparable in 50% of cases. We show that this ability for robust Generative UI is emergent, with substantial improvements from previous models. We also create and release PAGEN, a novel dataset of expert-crafted results to aid in evaluating Generative UI implementations, as well as the results of our system for future comparisons. Interactive examples can be seen at this https URL
>
---
#### [new 162] Instructing LLMs to Negotiate using Reinforcement Learning with Verifiable Rewards
- **分类: cs.AI; cs.CL; cs.GT; econ.GN**

- **简介: 该论文属于智能体谈判任务，旨在解决LLMs在不完全信息博弈中表现不佳的问题。通过RLVR框架训练买家代理，提升其谈判能力，实现经济盈余最大化。**

- **链接: [https://arxiv.org/pdf/2604.09855](https://arxiv.org/pdf/2604.09855)**

> **作者:** Shuze Daniel Liu; Claire Chen; Jiabao Sean Xiao; Lei Lei; Yuheng Zhang; Yisong Yue; David Simchi-Levi
>
> **摘要:** The recent advancement of Large Language Models (LLMs) has established their potential as autonomous interactive agents. However, they often struggle in strategic games of incomplete information, such as bilateral price negotiation. In this paper, we investigate if Reinforcement Learning from Verifiable Rewards (RLVR) can effectively teach LLMs to negotiate. Specifically, we explore the strategic behaviors that emerge during the learning process. We introduce a framework that trains a mid-sized buyer agent against a regulated LLM seller across a wide distribution of real-world products. By grounding reward signals directly in the maximization of economic surplus and strict adherence to private budget constraints, we reveal a novel four-phase strategic evolution. The agent progresses from naive bargaining to using aggressive starting prices, moves through a phase of deadlock, and ultimately develops sophisticated persuasive skills. Our results demonstrate that this verifiable training allows a 30B agent to significantly outperform frontier models over ten times its size in extracting surplus. Furthermore, the trained agent generalizes robustly to stronger counterparties unseen during training and remains effective even when facing hostile, adversarial seller personas.
>
---
#### [new 163] Speaking to No One: Ontological Dissonance and the Double Bind of Conversational AI
- **分类: cs.HC; cs.AI; cs.CL; cs.CY; cs.ET**

- **简介: 论文探讨了对话式AI可能引发妄想体验的问题，属于人工智能伦理研究。该文分析了AI交互中的本体论冲突，解释了为何传统安全措施失效，并提出技术中介的双人妄想机制。**

- **链接: [https://arxiv.org/pdf/2604.10833](https://arxiv.org/pdf/2604.10833)**

> **作者:** Hugh Brosnahan; Izabela Lipinska
>
> **备注:** This version of the article has been accepted for publication in Medicine, Health Care and Philosophy following peer review. This version is distributed under Springer Nature's terms for accepted manuscripts and does not reflect any post-acceptance improvements or corrections. The Version of Record will be available via Springer Nature upon publication
>
> **摘要:** Recent reports indicate that sustained interaction with conversational artificial intelligence (AI) systems can, in a small subset of users, contribute to the emergence or stabilisation of delusional experience. Existing accounts typically attribute such cases either to individual vulnerability or to failures of safety engineering. These explanations are incomplete. Drawing on phenomenology, psychiatry, and cognitive neuroscience, this paper argues that the risk arises from the relational and ontological structure of the interaction itself. Conversational AI generates ontological dissonance: a conflict between the appearance of relational presence and the absence of any subject capable of sustaining it. Maintained through a communicative double bind and amplified by attentional asymmetries, this dissonance tends, under conditions of affective vulnerability, to stabilise into a technologically mediated analogue of folie a deux. This account explains why explicit disclaimers often fail to disrupt delusional involvement and clarifies the ethical and clinical implications for the design and use of conversational AI.
>
---
#### [new 164] Assessing the Pedagogical Readiness of Large Language Models as AI Tutors in Low-Resource Contexts: A Case Study of Nepal's K-10 Curriculum
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于教育AI任务，旨在评估大语言模型作为低资源地区教师的准备度。研究分析了四个模型在尼泊尔课程中的表现，发现其存在课程对齐和文化适配问题，并提出改进策略。**

- **链接: [https://arxiv.org/pdf/2604.09619](https://arxiv.org/pdf/2604.09619)**

> **作者:** Pratyush Acharya; Prasansha Bharati; Yokibha Chapagain; Isha Sharma Gauli; Kiran Parajuli
>
> **备注:** 14 pages and 4 figures
>
> **摘要:** The integration of Large Language Models (LLMs) into educational ecosystems promises to democratize access to personalized tutoring, yet the readiness of these systems for deployment in non-Western, low-resource contexts remains critically under-examined. This study presents a systematic evaluation of four state-of-the-art LLMs--GPT-4o, Claude Sonnet 4, Qwen3-235B, and Kimi K2--assessing their capacity to function as AI tutors within the specific curricular and cultural framework of Nepal's Grade 5-10 Science and Mathematics education. We introduce a novel, curriculum-aligned benchmark and a fine-grained evaluation framework inspired by the "natural language unit tests" paradigm, decomposing pedagogical efficacy into seven binary metrics: Prompt Alignment, Factual Correctness, Clarity, Contextual Relevance, Engagement, Harmful Content Avoidance, and Solution Accuracy. Our results reveal a stark "curriculum-alignment gap." While frontier models (GPT-4o, Claude Sonnet 4) achieve high aggregate reliability (approximately 97%), significant deficiencies persist in pedagogical clarity and cultural contextualization. We identify two pervasive failure modes: the "Expert's Curse," where models solve complex problems but fail to explain them clearly to novices, and the "Foundational Fallacy," where performance paradoxically degrades on simpler, lower-grade material due to an inability to adapt to younger learners' cognitive constraints. Furthermore, regional models like Kimi K2 exhibit a "Contextual Blindspot," failing to provide culturally relevant examples in over 20% of interactions. These findings suggest that off-the-shelf LLMs are not yet ready for autonomous deployment in Nepalese classrooms. We propose a "human-in-the-loop" deployment strategy and offer a methodological blueprint for curriculum-specific fine-tuning to align global AI capabilities with local educational needs.
>
---
#### [new 165] Pioneer Agent: Continual Improvement of Small Language Models in Production
- **分类: cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文提出Pioneer Agent，解决小语言模型在生产中的持续优化问题。通过自动化数据收集、训练和诊断，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.09791](https://arxiv.org/pdf/2604.09791)**

> **作者:** Dhruv Atreja; Julia White; Nikhil Nayak; Kelton Zhang; Henrijs Princis; George Hurn-Maloney; Ash Lewis; Urchade Zaratiana
>
> **备注:** 43 pages, 10 figures, 14 tables
>
> **摘要:** Small language models are attractive for production deployment due to their low cost, fast inference, and ease of specialization. However, adapting them to a specific task remains a challenging engineering loop, driven not by training itself but by surrounding decisions: data curation, failure diagnosis, regression avoidance, and iteration control. We present Pioneer Agent, a closed-loop system that automates this lifecycle. In cold-start mode, given only a natural-language task description, the agent acquires data, constructs evaluation sets, and iteratively trains models by jointly optimizing data, hyperparameters, and learning strategy. In production mode, given a deployed model with labeled failures, it diagnoses error patterns, constructs targeted training data, and retrains under explicit regression constraints. To evaluate this setting, we introduce AdaptFT-Bench, a benchmark of synthetic inference logs with progressively increasing noise, designed to test the full adaptation loop: diagnosis, curriculum synthesis, retraining, and verification. Across eight cold-start benchmarks spanning reasoning, math, code generation, summarization, and classification, Pioneer Agent improves over base models by 1.6-83.8 points. On AdaptFT-Bench, it improves or preserves performance in all seven scenarios, while naive retraining degrades by up to 43 points. On two production-style deployments built from public benchmark tasks, it raises intent classification from 84.9% to 99.3% and Entity F1 from 0.345 to 0.810. Beyond performance gains, the agent often discovers effective training strategies, including chain-of-thought supervision, task-specific optimization, and quality-focused data curation, purely from downstream feedback.
>
---
#### [new 166] LABBench2: An Improved Benchmark for AI Systems Performing Biology Research
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出LABBench2，用于评估AI在真实科学任务中的能力。旨在解决AI科学研究能力衡量的问题，通过增加任务数量和难度，推动AI工具的发展。**

- **链接: [https://arxiv.org/pdf/2604.09554](https://arxiv.org/pdf/2604.09554)**

> **作者:** Jon M Laurent; Albert Bou; Michael Pieler; Conor Igoe; Alex Andonian; Siddharth Narayanan; James Braza; Alexandros Sanchez Vassopoulos; Jacob L Steenwyk; Blake Lash; Andrew D White; Samuel G Rodriques
>
> **摘要:** Optimism for accelerating scientific discovery with AI continues to grow. Current applications of AI in scientific research range from training dedicated foundation models on scientific data to agentic autonomous hypothesis generation systems to AI-driven autonomous labs. The need to measure progress of AI systems in scientific domains correspondingly must not only accelerate, but increasingly shift focus to more real-world capabilities. Beyond rote knowledge and even just reasoning to actually measuring the ability to perform meaningful work. Prior work introduced the Language Agent Biology Benchmark LAB-Bench as an initial attempt at measuring these abilities. Here we introduce an evolution of that benchmark, LABBench2, for measuring real-world capabilities of AI systems performing useful scientific tasks. LABBench2 comprises nearly 1,900 tasks and is, for the most part, a continuation of LAB-Bench, measuring similar capabilities but in more realistic contexts. We evaluate performance of current frontier models, and show that while abilities measured by LAB-Bench and LABBench2 have improved substantially, LABBench2 provides a meaningful jump in difficulty (model-specific accuracy differences range from -26% to -46% across subtasks) and underscores continued room for performance improvement. LABBench2 continues the legacy of LAB-Bench as a de facto benchmark for AI scientific research capabilities and we hope that it continues to help advance development of AI tools for these core research functions. To facilitate community use and development, we provide the task dataset at this https URL and a public eval harness at this https URL.
>
---
#### [new 167] The Amazing Agent Race: Strong Tool Users, Weak Navigators
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出AAR基准，用于评估LLM代理的工具使用和导航能力。针对线性基准的不足，设计DAG结构任务，揭示代理在导航上的弱点。**

- **链接: [https://arxiv.org/pdf/2604.10261](https://arxiv.org/pdf/2604.10261)**

> **作者:** Zae Myung Kim; Dongseok Lee; Jaehyung Kim; Vipul Raheja; Dongyeop Kang
>
> **摘要:** Existing tool-use benchmarks for LLM agents are overwhelmingly linear: our analysis of six benchmarks shows 55 to 100% of instances are simple chains of 2 to 5 steps. We introduce The Amazing Agent Race (AAR), a benchmark featuring directed acyclic graph (DAG) puzzles (or "legs") with fork-merge tool chains. We release 1,400 instances across two variants: sequential (800 legs) and compositional (600 DAG legs). Agents must navigate Wikipedia, execute multi-step tool chains, and aggregate results into a verifiable answer. Legs are procedurally generated from Wikipedia seeds across four difficulty levels with live-API validation. Three complementary metrics (finish-line accuracy, pit-stop visit rate, and roadblock completion rate) separately diagnose navigation, tool-use, and arithmetic failures. Evaluating three agent frameworks on 1,400 legs, the best achieves only 37.2% accuracy. Navigation errors dominate (27 to 52% of trials) while tool-use errors remain below 17%, and agent architecture matters as much as model scale (Claude Code matches Codex CLI at 37% with 6x fewer tokens). The compositional structure of AAR reveals that agents fail not at calling tools but at navigating to the right pages, a blind spot invisible to linear benchmarks. The project page can be accessed at: this https URL
>
---
#### [new 168] BMdataset: A Musicologically Curated LilyPond Dataset
- **分类: cs.SD; cs.CL; cs.IR**

- **简介: 该论文提出BMdataset，一个基于LilyPond的巴洛克音乐数据集，用于音乐理解任务。解决传统MIDI数据集的局限性，通过专家标注提升数据质量，并引入LilyBERT模型进行有效学习。**

- **链接: [https://arxiv.org/pdf/2604.10628](https://arxiv.org/pdf/2604.10628)**

> **作者:** Matteo Spanio; Ilay Guler; Antonio Rodà
>
> **备注:** Submitted to SMC2026
>
> **摘要:** Symbolic music research has relied almost exclusively on MIDI-based datasets; text-based engraving formats such as LilyPond remain unexplored for music understanding. We present BMdataset, a musicologically curated dataset of 393 LilyPond scores (2,646 movements) transcribed by experts directly from original Baroque manuscripts, with metadata covering composer, musical form, instrumentation, and sectional attributes. Building on this resource, we introduce LilyBERT (weights can be found at this https URL), a CodeBERT-based encoder adapted to symbolic music through vocabulary extension with 115 LilyPond-specific tokens and masked language model pre-training. Linear probing on the out-of-domain Mutopia corpus shows that, despite its modest size (~90M tokens), fine-tuning on BMdataset alone outperforms continuous pre-training on the full PDMX corpus (~15B tokens) for both composer and style classification, demonstrating that small, expertly curated datasets can be more effective than large, noisy corpora for music understanding. Combining broad pre-training with domain-specific fine-tuning yields the best results overall (84.3% composer accuracy), confirming that the two data regimes are complementary. We release the dataset, tokenizer, and model to establish a baseline for representation learning on LilyPond.
>
---
#### [new 169] Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于多轮LLM代理任务，解决RL样本效率低的问题。提出Skill-SD框架，利用代理轨迹生成动态监督信号，提升训练效果。**

- **链接: [https://arxiv.org/pdf/2604.10674](https://arxiv.org/pdf/2604.10674)**

> **作者:** Hao Wang; Guozhi Wang; Han Xiao; Yufeng Zhou; Yue Pan; Jichao Wang; Ke Xu; Yafei Wen; Xiaohu Ruan; Xiaoxin Chen; Honggang Qi
>
> **备注:** Project page: this https URL
>
> **摘要:** Reinforcement learning (RL) has been widely used to train LLM agents for multi-turn interactive tasks, but its sample efficiency is severely limited by sparse rewards and long horizons. On-policy self-distillation (OPSD) alleviates this by providing dense token-level supervision from a privileged teacher that has access to ground-truth answers. However, such fixed privileged information cannot capture the diverse valid strategies in agent tasks, and naively combining OPSD with RL often leads to training collapse. To address these limitations, we introduce Skill-SD, a framework that turns the agent's own trajectories into dynamic training-only supervision. Completed trajectories are summarized into compact natural language skills that describe successful behaviors, mistakes, and workflows. These skills serve as dynamic privileged information conditioning only the teacher, while the student always acts under the plain task prompt and learns to internalize the guidance through distillation. To stabilize the training, we derive an importance-weighted reverse-KL loss to provide gradient-correct token-level distillation, and dynamically synchronize the teacher with the improving student. Experimental results on agentic benchmarks demonstrate that Skill-SD substantially outperforms the standard RL baseline, improving both vanilla GRPO (+14.0%/+10.9% on AppWorld/Sokoban) and vanilla OPD (+42.1%/+40.6%). Project page: this https URL
>
---
#### [new 170] Quantization Dominates Rank Reduction for KV-Cache Compression
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型压缩任务，解决KV缓存压缩问题。比较了秩缩减和量化两种方法，发现量化在保持精度的同时更有效。**

- **链接: [https://arxiv.org/pdf/2604.11501](https://arxiv.org/pdf/2604.11501)**

> **作者:** Samuel Salfati
>
> **备注:** 16 pages, 3 figures
>
> **摘要:** We compare two strategies for compressing the KV cache in transformer inference: rank reduction (discard dimensions) and quantization (keep all dimensions, reduce precision). At matched storage budgets across five models (124M-14B, MHA and GQA), we find that quantization consistently outperforms rank reduction by 4-364 PPL depending on model and compression level. The gap persists even when rank reduction is combined with quantization in hybrid baselines, and it grows with GQA aggressiveness. On LAMBADA, INT4 matches FP16 accuracy (+0.23 PPL on Mistral 7B, +0.58 on GPT-2) while rank-32 at identical storage collapses to 0.4%. We trace this gap to a structural asymmetry: under softmax attention routing, removing a dimension can flip which token is attended (a discrete failure), while quantization noise is bounded and typically preserves score ordering. We formalize this via a perturbation result showing projection damage exceeds quantization damage by 3 x 2^(2b) per direction under the softmax Fisher metric. A basis ablation confirms the finding is basis-independent (spread <0.4 PPL), establishing that the advantage comes from preserving dimensions, not from a better coordinate system. Joint K+V INT4 quantization achieves 75% total KV reduction at only +0.18 PPL on Mistral 7B.
>
---
#### [new 171] Calibration Collapse Under Sycophancy Fine-Tuning: How Reward Hacking Breaks Uncertainty Quantification in LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究奖励欺骗对大语言模型校准能力的影响，属于模型校准任务。通过实验发现，基于迎合的微调会破坏不确定性量化，提出后处理方法缓解问题。**

- **链接: [https://arxiv.org/pdf/2604.10585](https://arxiv.org/pdf/2604.10585)**

> **作者:** Subramanyam Sahoo
>
> **备注:** Accepted at the AISTATS 2026 Workshop on Towards Trustworthy Predictions: Theory and Applications of Calibration for Modern AI. 14 Pages
>
> **摘要:** Modern large language models (LLMs) are increasingly fine-tuned via reinforcement learning from human feedback (RLHF) or related reward optimisation schemes. While such procedures improve perceived helpfulness, we investigate whether sycophantic reward signals degrade calibration -- a property essential for reliable uncertainty quantification. We fine-tune Qwen3-8B under three regimes: no fine-tuning (base), neutral supervised fine-tuning (SFT) on TriviaQA, and sycophancy-inducing Group Relative Policy Optimisation (GRPO) that rewards agreement with planted wrong answers. Evaluating on $1{,}000$ MMLU items across five subject domains with bootstrap confidence intervals and permutation testing, we find that \textbf{sycophantic GRPO produces consistent directional calibration degradation} -- ECE rises by $+0.006$ relative to the base model and MCE increases by $+0.010$ relative to neutral SFT -- though the effect does not reach statistical significance ($p = 0.41$) at this training budget. Post-hoc matrix scaling applied to all three models reduces ECE by $40$--$64\%$ and improves accuracy by $1.5$--$3.0$ percentage points. However, the sycophantic model retains the highest post-scaling ECE relative to the neutral SFT control ($0.042$ vs.\ $0.037$), suggesting that reward-induced miscalibration leaves a structured residual even after affine correction. These findings establish a methodology for evaluating the calibration impact of reward hacking and motivate calibration-aware training objectives.
>
---
#### [new 172] Do Agent Rules Shape or Distort? Guardrails Beat Guidance in Coding Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究AI编码代理的规则有效性，解决规则是否提升性能的问题。通过实验发现规则主要通过上下文引导起作用，建议采用负面约束而非正面指令。**

- **链接: [https://arxiv.org/pdf/2604.11088](https://arxiv.org/pdf/2604.11088)**

> **作者:** Xing Zhang; Guanghui Wang; Yanwei Cui; Wei Qiu; Ziyuan Li; Bing Zhu; Peiyang He
>
> **摘要:** Developers increasingly guide AI coding agents through natural language instruction files (e.g., this http URL, .cursorrules), yet no controlled study has measured whether these rules actually improve agent performance or which properties make a rule beneficial. We scrape 679 such files (25,532 rules) from GitHub and conduct the first large-scale empirical evaluation, running over 5,000 agent runs with a state-of-the-art coding agent on SWE-bench Verified. Rules improve performance by 7--14 percentage points, but random rules help as much as expert-curated ones -- suggesting rules work through context priming rather than specific instruction. Negative constraints ("do not refactor unrelated code") are the only individually beneficial rule type, while positive directives ("follow code style") actively hurt -- a pattern we analyze through the lens of potential-based reward shaping (PBRS). Moreover, individual rules are mostly harmful in isolation yet collectively helpful, with no degradation up to 50 rules. These findings expose a hidden reliability risk -- well-intentioned rules routinely degrade agent performance -- and provide a clear principle for safe agent configuration: constrain what agents must not do, rather than prescribing what they should.
>
---
#### [new 173] Anthropogenic Regional Adaptation in Multimodal Vision-Language Model
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决区域文化对齐问题。提出Anthropogenic Regional Adaptation和GG-EZ方法，提升模型在特定地区的文化相关性，同时保持全局性能。**

- **链接: [https://arxiv.org/pdf/2604.11490](https://arxiv.org/pdf/2604.11490)**

> **作者:** Samuel Cahyawijaya; Peerat Limkonchotiwat; Tack Hwa Wong; Hitesh Laxmichand Patel; Amit Agarwal; Manuel Antonio Rufino; Carlos Rafael Catalan; Muhammad Reza Qorib; Vicky Feliren; Holy Lovenia; Aye Hninn Khine; Frederikus Hudi; David Anugraha; Alham Fikri Aji; Romrawin Chumpu; Viet-Thanh Pham; Minghan Wang; Mohamed Fazli Imam; Ruochen Zhang; Joseph Marvin Imperial; Do Xuan Long; Musa Izzanardi Wijanarko; Joel Ruben Antony Moniz; Patrick Amadeus Irawan; Hanif Muhammad Zhafran; Isaiah Flores; Ira Salsabila; Jun Kevin; Jostin Jerico Rosal; Patricia Nicole Monderin; Kun Kerdthaisong; Ahmad Mustafid; My Chiffon Nguyen; Natchapon Jongwiriyanurak; Siva Worajitwannakul; Haochen Li; Adrian Xuan Wei Lim; Bin Wang; Muhammad Ravi Shulthan Habibi; Lynnette Hui Xian Ng; Mithil Bangera; Yeshil Bangera; Priyaranjan Pattnayak; Dun Li Chan; Sherissa Caren Djuniwar; Hee Ming Shan
>
> **摘要:** While the field of vision-language (VL) has achieved remarkable success in integrating visual and textual information across multiple languages and domains, there is still no dedicated framework for assessing human-centric alignment in vision-language systems. We offer two contributions to address this gap. First, we introduce Anthropogenic Regional Adaptation: a novel paradigm that aims to optimize model relevance to specific regional contexts while ensuring the retention of global generalization capabilities. Second, we present a simple, but effective adaptation method named Geographical-generalization-made-easy (GG-EZ), which utilizes regional data filtering and model merging. Through comprehensive experiments on 3 VL architectures: large vision-language models, text-to-image diffusion models, and vision-language embedding models, and a case study in Southeast Asia (SEA) regional adaptation, we demonstrate the importance of Anthropogenic Regional Adaptation and the effectiveness of GG-EZ, showing 5-15% gains in cultural relevance metrics across SEA while maintaining over 98% of global performance and even occasionally surpassing it. Our findings establish Anthropogenic Regional Alignment as a foundational paradigm towards applicability of multimodal vision-language models in diverse regions and demonstrate a simple-yet-effective baseline method that optimizes regional value alignment while preserving global generalization.
>
---
#### [new 174] From UAV Imagery to Agronomic Reasoning: A Multimodal LLM Benchmark for Plant Phenotyping
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于植物表型分析任务，旨在解决农业领域中多模态模型应用的挑战。工作包括构建PlantXpert基准，评估11个视觉语言模型在作物表型中的表现。**

- **链接: [https://arxiv.org/pdf/2604.09907](https://arxiv.org/pdf/2604.09907)**

> **作者:** Yu Wu; Guangzeng Han; Ibra Niang Niang; Francia Ravelombola; Maiara Oliveira; Jason Davis; Dong Chen; Feng Lin; Xiaolei Huang
>
> **备注:** In review
>
> **摘要:** To improve crop genetics, high-throughput, effective and comprehensive phenotyping is a critical prerequisite. While such tasks were traditionally performed manually, recent advances in multimodal foundation models, especially in vision-language models (VLMs), have enabled more automated and robust phenotypic analysis. However, plant science remains a particularly challenging domain for foundation models because it requires domain-specific knowledge, fine-grained visual interpretation, and complex biological and agronomic reasoning. To address this gap, we develop PlantXpert, an evidence-grounded multimodal reasoning benchmark for soybean and cotton phenotyping. Our benchmark provides a structured and reproducible framework for agronomic adaptation of VLMs, and enables controlled comparison between base models and their domain-adapted counterparts. We constructed a dataset comprising 385 digital images and more than 3,000 benchmark samples spanning key plant science domains including disease, pest control, weed management, and yield. The benchmark can assess diverse capabilities including visual expertise, quantitative reasoning, and multi-step agronomic reasoning. A total of 11 state-of-the-art VLMs were evaluated. The results indicate that task-specific fine-tuning leads to substantial improvement in accuracy, with models such as Qwen3-VL-4B and Qwen3-VL-30B achieving up to 78%. At the same time, gains from model scaling diminish beyond a certain capacity, generalization across soybean and cotton remains uneven, and quantitative as well as biologically grounded reasoning continue to pose substantial challenges. These findings suggest that PlantXpert can serve as a foundation for assessing evidence-grounded agronomic reasoning and for advancing multimodal model development in plant science.
>
---
#### [new 175] A molecular clock for writing systems reveals the quantitative impact of imperial power on cultural evolution
- **分类: q-bio.PE; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于文化演化研究，旨在量化帝国权力对书写系统演化的影响。通过构建全球书写系统数据库，分析其演化模式与政治干预的关系。**

- **链接: [https://arxiv.org/pdf/2604.10957](https://arxiv.org/pdf/2604.10957)**

> **作者:** Hiroki Fukui
>
> **备注:** 28 pages, 6 figures, 4 supplementary figures, 1 table. Preprint v5
>
> **摘要:** Writing systems are cultural replicators whose evolution has never been studied quantitatively at global scale. We compile the Global Script Database (GSD): 300 writing and notation systems, 50 binary structural characters, and 259 phylogenetic edges spanning 5,400 years. Applying four methods -- phenetics, cladistics, Bayesian inference, and neural network clustering -- we find that scripts exhibit a detectable molecular clock. The best-fitting model (Mk+Gamma strict clock) yields a substitution rate of q = 0.226 substitutions/character/millennium (95% CI: 0.034-1.22; Delta BIC = -4.1 versus relaxed clock; Delta BIC = -1,364.7 versus Mk without rate variation). Political interventions break this clock: deviation from expected divergence times correlates with intervention intensity (Spearman rho = 0.556, p < 10^{-4}), and per-character rate analysis reveals that intervention selectively rewrites deep structural features rather than merely accelerating change (rate profile correlation rho = 0.320). We identify 30 major script replacement events and rank their destructive impact. A ceiling effect suppresses independent invention wherever writing already exists (Fisher's exact OR = 0.054, p < 10^{-6}), and colonial contact predicts script extinction (Cox HR = 5.25, p = 0.0006). The Spanish Empire extinguished the most scripts (6 of 12 contacted, 50%), followed by the Empire of Japan (3 of 9, 33.3%). Feature coding was validated by inter-rater reliability testing with two independent human coders (Cohen's kappa = 0.877; human-LLM kappa = 0.929; Fleiss' kappa = 0.911).
>
---
#### [new 176] Audio Flamingo Next: Next-Generation Open Audio-Language Models for Speech, Sound, and Music
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出Audio Flamingo Next，解决音频-语言理解与推理问题，通过增强模型、扩展数据集和引入新推理范式，提升长音频处理能力与准确性。**

- **链接: [https://arxiv.org/pdf/2604.10905](https://arxiv.org/pdf/2604.10905)**

> **作者:** Sreyan Ghosh; Arushi Goel; Kaousheik Jayakumar; Lasha Koroshinadze; Nishit Anand; Zhifeng Kong; Siddharth Gururani; Sang-gil Lee; Jaehyeon Kim; Aya Aljafari; Chao-Han Huck Yang; Sungwon Kim; Ramani Duraiswami; Dinesh Manocha; Mohammad Shoeybi; Bryan Catanzaro; Ming-Yu Liu; Wei Ping
>
> **备注:** Project website: this https URL
>
> **摘要:** We present Audio Flamingo Next (AF-Next), the next-generation and most capable large audio-language model in the Audio Flamingo series, designed to advance understanding and reasoning over speech, environmental sounds and music. Compared to Audio Flamingo 3, AF-Next introduces: (i) a stronger foundational audio-language model that significantly improves accuracy across diverse audio understanding tasks; (ii) scalable strategies for constructing large-scale audio understanding and reasoning data beyond existing academic benchmarks; (iii) support for long and complex audio inputs up to 30 minutes; and (iv) Temporal Audio Chain-of-Thought, a new reasoning paradigm that explicitly grounds intermediate reasoning steps to timestamps in long audio, enabling fine-grained temporal alignment and improved interpretability. To enable these capabilities, we first conduct a systematic analysis of Audio Flamingo 3 to identify key gaps in audio understanding and reasoning. We then curate and scale new large-scale datasets totaling over 1 million hours to address these limitations and expand the existing AudioSkills-XL, LongAudio-XL, AF-Think and AF-Chat datasets. AF-Next is trained using a curriculum-based strategy spanning pre-training, mid-training and post-training stages. Extensive experiments across 20 audio understanding and reasoning benchmarks, including challenging long-audio tasks, show that AF-Next outperforms similarly sized open models by large margins and remains highly competitive with and sometimes surpasses, much larger open-weight and closed models. Beyond benchmark performance, AF-Next exhibits strong real-world utility and transfers well to unseen tasks, highlighting its robustness and generalization ability. In addition to all data, code and methods, we open-source 3 variants of AF-Next, including AF-Next-Instruct, AF-Next-Think and AF-Next-Captioner.
>
---
#### [new 177] Detecting Safety Violations Across Many Agent Traces
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于安全检测任务，旨在解决多智能体轨迹中的安全违规问题。通过聚类与代理搜索结合的方法，提升对罕见、复杂违规行为的检测效果。**

- **链接: [https://arxiv.org/pdf/2604.11806](https://arxiv.org/pdf/2604.11806)**

> **作者:** Adam Stein; Davis Brown; Hamed Hassani; Mayur Naik; Eric Wong
>
> **备注:** 35 pages, 17 figures
>
> **摘要:** To identify safety violations, auditors often search over large sets of agent traces. This search is difficult because failures are often rare, complex, and sometimes even adversarially hidden and only detectable when multiple traces are analyzed together. These challenges arise in diverse settings such as misuse campaigns, covert sabotage, reward hacking, and prompt injection. Existing approaches struggle here for several reasons. Per-trace judges miss failures that only become visible across traces, naive agentic auditing does not scale to large trace collections, and fixed monitors are brittle to unanticipated behaviors. We introduce Meerkat, which combines clustering with agentic search to uncover violations specified in natural language. Through structured search and adaptive investigation of promising regions, Meerkat finds sparse failures without relying on seed scenarios, fixed workflows, or exhaustive enumeration. Across misuse, misalignment, and task gaming settings, Meerkat significantly improves detection of safety violations over baseline monitors, discovers widespread developer cheating on a top agent benchmark, and finds nearly 4x more examples of reward hacking on CyBench than previous audits.
>
---
#### [new 178] ProGAL-VLA: Grounded Alignment through Prospective Reasoning in Vision-Language-Action Models
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出ProGAL-VLA模型，解决VLA模型语言忽视和指令不敏感问题，通过构建3D图、符号子目标和对比损失提升鲁棒性和实体检索效果。**

- **链接: [https://arxiv.org/pdf/2604.09824](https://arxiv.org/pdf/2604.09824)**

> **作者:** Nastaran Darabi; Amit Ranjan Trivedi
>
> **摘要:** Vision language action (VLA) models enable generalist robotic agents but often exhibit language ignorance, relying on visual shortcuts and remaining insensitive to instruction changes. We present Prospective Grounding and Alignment VLA (ProGAL-VLA), which constructs a 3D entity-centric graph (GSM), uses a slow planner to produce symbolic sub-goals, and aligns them with grounded entities via a Grounding Alignment Contrastive (GAC) loss. All actions are conditioned on a verified goal embedding $g_t$, whose attention entropy provides an intrinsic ambiguity signal. On LIBERO-Plus, ProGAL-VLA increases robustness under robot perturbations from 30.3 to 71.5 percent, reduces language ignorance by 3x-4x, and improves entity retrieval from 0.41 to 0.71 Recall@1. On the Custom Ambiguity Benchmark, it reaches AUROC 0.81 (vs. 0.52), AUPR 0.79, and raises clarification on ambiguous inputs from 0.09 to 0.81 without harming unambiguous success. The verification bottleneck increases mutual information of language-actions, the GAC loss imposes an entity-level InfoNCE bound, and attention entropy yields calibrated selective prediction, indicating that explicit verified grounding is an effective path toward instruction-sensitive, ambiguity-aware agents.
>
---
#### [new 179] Min-$k$ Sampling: Decoupling Truncation from Temperature Scaling via Relative Logit Dynamics
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出Min-k Sampling，解决语言模型生成文本质量问题，通过动态截断策略实现温度不变性，提升生成文本质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.11012](https://arxiv.org/pdf/2604.11012)**

> **作者:** Yuanhao Ding; Meimingwei Li; Esteban Garces Arias; Matthias Aßenmacher; Christian Heumann; Chongsheng Zhang
>
> **备注:** Accepted at ACL 2026 (Main Conference)
>
> **摘要:** The quality of text generated by large language models depends critically on the decoding sampling strategy. While mainstream methods such as Top-$k$, Top-$p$, and Min-$p$ achieve a balance between diversity and accuracy through probability-space truncation, they share an inherent limitation: extreme sensitivity to the temperature parameter. Recent logit-space approaches like Top-$n\sigma$ achieve temperature invariance but rely on global statistics that are susceptible to long-tail noise, failing to capture fine-grained confidence structures among top candidates. We propose \textbf{Min-$k$ Sampling}, a novel dynamic truncation strategy that analyzes the local shape of the sorted logit distribution to identify "semantic cliffs": sharp transitions from high-confidence core tokens to uncertain long-tail tokens. By computing a position-weighted relative decay rate, Min-$k$ dynamically determines truncation boundaries at each generation step. We formally prove that Min-$k$ achieves strict temperature invariance and empirically demonstrate its low sensitivity to hyperparameter choices. Experiments on multiple reasoning benchmarks, creative writing tasks, and human evaluation show that Min-$k$ consistently improves text quality, maintaining robust performance even under extreme temperature settings where probability-based methods collapse. We make our code, models, and analysis tools publicly available.
>
---
#### [new 180] CFMS: A Coarse-to-Fine Multimodal Synthesis Framework for Enhanced Tabular Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出CFMS框架，解决表格推理任务中的视觉与符号推理结合问题，通过多模态合成提升准确性。**

- **链接: [https://arxiv.org/pdf/2604.10973](https://arxiv.org/pdf/2604.10973)**

> **作者:** Qixian Huang; Hongqiang Lin; Tong Fu; Yingsen Wang; Zhenghui Fu; Qirui Wang; Yiding Sun; Dongxu Zhang
>
> **摘要:** Reasoning over tabular data is a crucial capability for tasks like question answering and fact verification, as it requires models to comprehend both free-form questions and semi-structured tables. However, while methods like Chain-of-Thought (CoT) introduce reasoning chains, purely symbolic methodes are inherently limited by their blindness to holistic visual patterns. To address this, we propose the Coarse-to-Fine Multimodal Synthesis framework (CFMS), a novel two-stage paradigm that hierarchically decouples high-level visual perception from granular symbolic reasoning. In the Coarse Stage, CFMS leverages the Multimodal Large Language Models (MLLMs) to perform a one-time synthesis of a multi-perspective knowledge tuple. This tuple subsequently serves as a dynamic reasoning map to guide the fine stage, where a symbolic engine executes a targeted and efficient sequence of iterative operations over the table. Extensive experiments on the WikiTQ and TabFact benchmarks demonstrate that CFMS achieves competitive accuracy. The framework exhibits particular robustness when handling large tables and when instantiated with smaller backbone models, validating its effectiveness and generalizability.
>
---
#### [new 181] What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于个性化图像美学评估任务，旨在解决VLM是否编码多层级美学属性的问题。通过分析VLM内部表示，发现其可有效支持个性化评估，并验证简单模型的可行性。**

- **链接: [https://arxiv.org/pdf/2604.11374](https://arxiv.org/pdf/2604.11374)**

> **作者:** Koki Ryu; Hitomi Yanaka
>
> **备注:** To appear at ACL 2026 findings
>
> **摘要:** Personalized image aesthetics assessment (PIAA) is an important research problem with practical real-world applications. While methods based on vision-language models (VLMs) are promising candidates for PIAA, it remains unclear whether they internally encode rich, multi-level aesthetic attributes required for effective personalization. In this paper, we first analyze the internal representations of VLMs to examine the presence and distribution of such aesthetic attributes, and then leverage them for lightweight, individual-level personalization without model fine-tuning. Our analysis reveals that VLMs encode diverse aesthetic attributes that propagate into the language decoder layers. Building on these representations, we demonstrate that simple linear models can perform PIAA effectively. We further analyze how aesthetic information is transferred across layers in different VLM architectures and across image domains. Our findings provide insights into how VLMs can be utilized for modeling subjective, individual aesthetic preferences. Our code is available at this https URL.
>
---
#### [new 182] Low-rank Optimization Trajectories Modeling for LLM RLVR Acceleration
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型强化学习任务，旨在解决RLVR训练中的计算开销过大问题。通过非线性低秩轨迹外推方法NExt，提升训练效率。**

- **链接: [https://arxiv.org/pdf/2604.11446](https://arxiv.org/pdf/2604.11446)**

> **作者:** Zhipeng Chen; Tao Qian; Wayne Xin Zhao; Ji-Rong Wen
>
> **备注:** Working in progress
>
> **摘要:** Recently, scaling reinforcement learning with verifiable rewards (RLVR) for large language models (LLMs) has emerged as an effective training paradigm for significantly improving model capabilities, which requires guiding the model to perform extensive exploration and learning, leading to substantial computational overhead and becoming a key challenge. To reduce the number of training steps, Prior work performs linear extrapolation of model parameters. However, the dynamics of model parameter updates during RLVR training remain insufficiently understood. To further investigate the evolution of LLMs during RLVR training, we conduct empirical experiments and find that the rank-1 subspace of the model does not evolve linearly, and its dominance over the original parameters is further amplified during LoRA training. Based on the above insights, we propose the \textbf{N}onlinear \textbf{Ext}rapolation of low-rank trajectories (\textbf{NExt}), a novel framework that models and extrapolates low-rank parameter trajectories in a nonlinear manner. Concretely, we first train the model using LoRA and extract the rank-1 subspace of parameter differences at multiple training steps, which is then used for the subsequent nonlinear extrapolation. Afterward, we utilized the extracted rank-1 subspace to train a predictor, which can model the trajectory of parameter updates during RLVR, and then perform the predict-extend process to extrapolate model parameters, achieving the acceleration of RLVR. To further study and understand NExt, we conduct comprehensive experiments that demonstrate the effectiveness and robustness of the method. Our method reduces computational overhead by approximately 37.5\% while remaining compatible with a wide range of RLVR algorithms and tasks. We release our code in this https URL.
>
---
#### [new 183] How LLMs Might Think
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于哲学与人工智能交叉研究，探讨LLMs是否具有思维。论文反驳了LLMs不思考的观点，提出它们可能以非理性、联想方式思考。**

- **链接: [https://arxiv.org/pdf/2604.09674](https://arxiv.org/pdf/2604.09674)**

> **作者:** Joseph Gottlieb; Ethan Kemp; Matthew Trager
>
> **摘要:** Do large language models (LLMs) think? Daniel Stoljar and Zhihe Vincent Zhang have recently developed an argument from rationality for the claim that LLMs do not think. We contend, however, that the argument from rationality not only falters, but leaves open an intriguing possibility: that LLMs engage only in arational, associative forms of thinking, and have purely associative minds. Our positive claim is that if LLMs think at all, they likely think precisely in this manner.
>
---
#### [new 184] DeepReviewer 2.0: A Traceable Agentic System for Auditable Scientific Peer Review
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文提出DeepReviewer 2.0，用于可审计的科学同行评审。解决自动化评审缺乏可追溯性的问题，通过生成带证据和行动项的评审包，提升评审透明度与准确性。**

- **链接: [https://arxiv.org/pdf/2604.09590](https://arxiv.org/pdf/2604.09590)**

> **作者:** Yixuan Weng; Minjun Zhu; Qiujie Xie; Zhiyuan Ning; Shichen Li; Panzhong Lu; Zhen Lin; Enhao Gu; Qiyao Sun; Yue Zhang
>
> **摘要:** Automated peer review is often framed as generating fluent critique, yet reviewers and area chairs need judgments they can \emph{audit}: where a concern applies, what evidence supports it, and what concrete follow-up is required. DeepReviewer~2.0 is a process-controlled agentic review system built around an output contract: it produces a \textbf{traceable review package} with anchored annotations, localized evidence, and executable follow-up actions, and it exports only after meeting minimum traceability and coverage budgets. Concretely, it first builds a manuscript-only claim--evidence--risk ledger and verification agenda, then performs agenda-driven retrieval and writes anchored critiques under an export gate. On 134 ICLR~2025 submissions under three fixed protocols, an \emph{un-finetuned 196B} model running DeepReviewer~2.0 outperforms Gemini-3.1-Pro-preview, improving strict major-issue coverage (37.26\% vs.\ 23.57\%) and winning 71.63\% of micro-averaged blind comparisons against a human review committee, while ranking first among automatic systems in our pool. We position DeepReviewer~2.0 as an assistive tool rather than a decision proxy, and note remaining gaps such as ethics-sensitive checks.
>
---
#### [new 185] ClawGUI: A Unified Framework for Training, Evaluating, and Deploying GUI Agents
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出ClawGUI框架，解决GUI代理训练、评估与部署的问题，整合RL、标准化评估及跨平台部署，提升代理性能与实用性。**

- **链接: [https://arxiv.org/pdf/2604.11784](https://arxiv.org/pdf/2604.11784)**

> **作者:** Fei Tang; Zhiqiong Lu; Boxuan Zhang; Weiming Lu; Jun Xiao; Yueting Zhuang; Yongliang Shen
>
> **摘要:** GUI agents drive applications through their visual interfaces instead of programmatic APIs, interacting with arbitrary software via taps, swipes, and keystrokes, reaching a long tail of applications that CLI-based agents cannot. Yet progress in this area is bottlenecked less by modeling capacity than by the absence of a coherent full-stack infrastructure: online RL training suffers from environment instability and closed pipelines, evaluation protocols drift silently across works, and trained agents rarely reach real users on real devices. We present \textbf{ClawGUI}, an open-source framework addressing these three gaps within a single harness. \textbf{ClawGUI-RL} provides the first open-source GUI agent RL infrastructure with validated support for both parallel virtual environments and real physical devices, integrating GiGPO with a Process Reward Model for dense step-level supervision. \textbf{ClawGUI-Eval} enforces a fully standardized evaluation pipeline across 6 benchmarks and 11+ models, achieving 95.8\% reproduction against official baselines. \textbf{ClawGUI-Agent} brings trained agents to Android, HarmonyOS, and iOS through 12+ chat platforms with hybrid CLI-GUI control and persistent personalized memory. Trained end to end within this pipeline, \textbf{ClawGUI-2B} achieves 17.1\% Success Rate on MobileWorld GUI-Only, outperforming the same-scale MAI-UI-2B baseline by 6.0\%.
>
---
#### [new 186] ZoomR: Memory Efficient Reasoning through Multi-Granularity Key Value Retrieval
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出ZoomR，解决长输出生成时KV缓存内存过高的问题，通过多粒度键值检索实现高效推理。**

- **链接: [https://arxiv.org/pdf/2604.10898](https://arxiv.org/pdf/2604.10898)**

> **作者:** David H. Yang; Yuxuan Zhu; Mohammad Mohammadi Amiri; Keerthiram Murugesan; Tejaswini Pedapati; Subhajit Chaudhury; Pin-Yu Chen
>
> **摘要:** Large language models (LLMs) have shown great performance on complex reasoning tasks but often require generating long intermediate thoughts before reaching a final answer. During generation, LLMs rely on a key-value (KV) cache for autoregressive decoding. However, the memory footprint of the KV cache grows with output length. Prior work on KV cache optimization mostly focus on compressing the long input context, while retaining the full KV cache for decoding. For tasks requiring long output generation, this leads to increased computational and memory costs. In this paper, we introduce ZoomR, a novel approach that enables LLMs to adaptively compress verbose reasoning thoughts into summaries and uses a dynamic KV cache selection policy that leverages these summaries while also strategically "zooming in" on fine-grained details. By using summary keys as a coarse-grained index during decoding, ZoomR uses the query to retrieve details for only the most important thoughts. This hierarchical strategy significantly reduces memory usage by avoiding full-cache attention at each step. Experiments across math and reasoning tasks show that our approach achieves competitive performance compared to baselines, while reducing inference memory requirements by more than $4\times$. These results demonstrate that a multi-granularity KV selection enables more memory efficient decoding, especially for long output generation.
>
---
#### [new 187] CID-TKG: Collaborative Historical Invariance and Evolutionary Dynamics Learning for Temporal Knowledge Graph Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于时间知识图推理任务，旨在解决现有方法因忽略演化动态而效果受限的问题。提出CID-TKG框架，融合历史不变性和演化动态进行协同学习。**

- **链接: [https://arxiv.org/pdf/2604.09600](https://arxiv.org/pdf/2604.09600)**

> **作者:** Shuai-Long Lei; Xiaobin Zhu; Jiarui Liang; Guoxi Sun; Zhiyu Fang; Xu-Cheng Yin
>
> **摘要:** Temporal knowledge graph (TKG) reasoning aims to infer future facts at unseen timestamps from temporally evolving entities and relations. Despite recent progress, existing approaches still suffer from inherent limitations due to their inductive biases, as they predominantly rely on time-invariant or weakly time-dependent structures and overlook the evolutionary dynamics. To overcome this limitation, we propose a novel collaborative learning framework for TKGR (dubbed CID-TKG) that integrates evolutionary dynamics and historical invariance semantics as an effective inductive bias for reasoning. Specifically, CID-TKG constructs a historical invariance graph to capture long-term structural regularities and an evolutionary dynamics graph to model short-term temporal transitions. Dedicated encoders are then employed to learn representations from each structure. To alleviate semantic discrepancies across the two structures, we decompose relations into view-specific representations and align view-specific query representations via a contrastive objective, which promotes cross-view consistency while suppressing view-specific noise. Extensive experiments verify that our CID-TKG achieves state-of-the-art performance under extrapolation settings.
>
---
#### [new 188] K-Way Energy Probes for Metacognition Reduce to Softmax in Discriminative Predictive Coding Networks
- **分类: cs.LG; cs.CL; cs.NE**

- **简介: 该论文研究预测编码网络中的结构探针，探讨其与softmax的关系。任务是分析结构探针的有效性，发现其在多种条件下均低于softmax，揭示其局限性。**

- **链接: [https://arxiv.org/pdf/2604.11011](https://arxiv.org/pdf/2604.11011)**

> **作者:** Jon-Paul Cacioli
>
> **备注:** 33 pages, 3 figures
>
> **摘要:** We present this as a negative result with an explanatory mechanism, not as a formal upper bound. Predictive coding networks (PCNs) admit a K-way energy probe in which each candidate class is fixed as a target, inference is run to settling, and the per-hypothesis settled energies are compared. The probe appears to read a richer signal source than softmax, since the per-hypothesis energy depends on the entire generative chain. We argue this appearance is misleading under the standard Pinchetti-style discriminative PC formulation. We present an approximate reduction showing that with target-clamped CE-energy training and effectively-feedforward latent dynamics, the K-way energy margin decomposes into a monotone function of the log-softmax margin plus a residual that is not trained to correlate with correctness. The decomposition predicts that the structural probe should track softmax from below. We test this across six conditions on CIFAR-10: extended deterministic training, direct measurement of latent movement during inference, a post-hoc decoder fairness control on a backpropagation network, a matched-budget PC vs BP comparison, a five-point Langevin temperature sweep, and trajectory-integrated MCPC training. In every condition the probe sat below softmax. The gap was stable across training procedures within the discriminative PC family. Final-state and trajectory-integrated training produced probes whose AUROC_2 values differed by less than 10^-3 at deterministic evaluation. The empirical regime is small: single seed, 2.1M-parameter network, 1280 test images. We frame the result as a preprint inviting replication. We discuss conditions under which the decomposition does not apply (bidirectional PC, prospective configuration, generative PC, non-CE energy formulations) and directions for productive structural probing the analysis does not foreclose.
>
---
#### [new 189] AI Patents in the United States and China: Measurement, Organization, and Knowledge Flows
- **分类: econ.GN; cs.AI; cs.CL; q-fin.GN**

- **简介: 该论文属于AI专利分析任务，旨在准确测量中美AI专利并分析其创新模式。通过构建高精度分类器，研究两国专利增长、结构及知识流动差异。**

- **链接: [https://arxiv.org/pdf/2604.10529](https://arxiv.org/pdf/2604.10529)**

> **作者:** Hanming Fang; Xian Gu; Hanyin Yan; Wu Zhu
>
> **摘要:** We develop a high-precision classifier to measure artificial intelligence (AI) patents by fine-tuning PatentSBERTa on manually labeled data from the USPTO's AI Patent Dataset. Our classifier substantially improves the existing USPTO approach, achieving 97.0% precision, 91.3% recall, and a 94.0% F1 score, and it generalizes well to Chinese patents based on citation and lexical validation. Applying it to granted U.S. patents (1976-2023) and Chinese patents (2010-2023), we document rapid growth in AI patenting in both countries and broad convergence in AI patenting intensity and subfield composition, even as China surpasses the United States in recent annual patent counts. The organization of AI innovation nevertheless differs sharply: U.S. AI patenting is concentrated among large private incumbents and established hubs, whereas Chinese AI patenting is more geographically diffuse and institutionally diverse, with larger roles for universities and state-owned enterprises. For listed firms, AI patents command a robust market-value premium in both countries. Cross-border citations show continued technological interdependence rather than decoupling, with Chinese AI inventors relying more heavily on U.S. frontier knowledge than vice versa.
>
---
#### [new 190] Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised Reinforcement Learning Approach
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于医疗推理任务，旨在解决医疗数据稀缺与罕见病领域效果不佳的问题。提出MedSSR框架，结合知识合成与半监督强化学习，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.11547](https://arxiv.org/pdf/2604.11547)**

> **作者:** Haolin Li; Shuyang Jiang; Ruipeng Zhang; Jiangchao Yao; Ya Zhang; Yanfeng Wang
>
> **备注:** Accepted to ACL 2026 as a Findings paper
>
> **摘要:** While large language models hold promise for complex medical applications, their development is hindered by the scarcity of high-quality reasoning data. To address this issue, existing approaches typically distill chain-of-thought reasoning traces from large proprietary models via supervised fine-tuning, then conduct reinforcement learning (RL). These methods exhibit limited improvement on underrepresented domains like rare diseases while incurring substantial costs from generating complex reasoning chains. To efficiently enhance medical reasoning, we propose MedSSR, a Medical Knowledge-enhanced data Synthesis and Semi-supervised Reinforcement learning framework. Our framework first employs rare disease knowledge to synthesize distribution-controllable reasoning questions. We then utilize the policy model itself to generate high-quality pseudo-labels. This enables a two-stage, intrinsic-to-extrinsic training paradigm: self-supervised RL on the pseudo-labeled synthetic data, followed by supervised RL on the human-annotated real data. MedSSR scales model training efficiently without relying on costly trace distillation. Extensive experiments on Qwen and Llama demonstrate that our method outperforms existing methods across ten medical benchmarks, achieving up to +5.93% gain on rare-disease tasks. Our code is available at this https URL.
>
---
#### [new 191] MimicLM: Zero-Shot Voice Imitation through Autoregressive Modeling of Pseudo-Parallel Speech Corpora
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音模仿任务，解决数据稀缺问题。通过使用合成语音作为训练源、真实录音作为目标，提出MimicLM模型，提升语音模仿质量与自然度。**

- **链接: [https://arxiv.org/pdf/2604.11552](https://arxiv.org/pdf/2604.11552)**

> **作者:** Tao Feng; Yuxiang Wang; Yuancheng Wang; Xueyao Zhang; Dekun Chen; Chaoren Wang; Xun Guan; Zhizheng Wu
>
> **摘要:** Voice imitation aims to transform source speech to match a reference speaker's timbre and speaking style while preserving linguistic content. A straightforward approach is to train on triplets of (source, reference, target), where source and target share the same content but target matches the reference's voice characteristics, yet such data is extremely scarce. Existing approaches either employ carefully designed disentanglement architectures to bypass this data scarcity or leverage external systems to synthesize pseudo-parallel training data. However, the former requires intricate model design, and the latter faces a quality ceiling when synthetic speech is used as training targets. To address these limitations, we propose MimicLM, which takes a novel approach by using synthetic speech as training sources while retaining real recordings as targets. This design enables the model to learn directly from real speech distributions, breaking the synthetic quality ceiling. Building on this data construction approach, we incorporate interleaved text-audio modeling to guide the generation of content-accurate speech and apply post-training with preference alignment to mitigate the inherent distributional mismatch when training on synthetic data. Experiments demonstrate that MimicLM achieves superior voice imitation quality with a simple yet effective architecture, significantly outperforming existing methods in naturalness while maintaining competitive similarity scores across speaker identity, accent, and emotion dimensions.
>
---
#### [new 192] LETGAMES: An LLM-Powered Gamified Approach to Cognitive Training for Patients with Cognitive Impairment
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于认知训练任务，旨在解决个性化游戏设计资源消耗大的问题。提出LETGAMES方法，利用大语言模型生成定制化游戏，提升治疗效果。**

- **链接: [https://arxiv.org/pdf/2604.09566](https://arxiv.org/pdf/2604.09566)**

> **作者:** Jingwei Shi; Shengyu Tao; Xinxiang Yin; Chen Huang; Wenqiang Lei; See-Kiong Ng
>
> **备注:** 53 pages
>
> **摘要:** The application of games as a therapeutic tool for cognitive training is beneficial for patients with cognitive impairments. However, effective game design for individual patient is resource-intensive. To this end, we propose an LLM-powered method, \ours, for automated and personalized therapeutic game design. Inspired by the Dungeons & Dragons, LETGAMES generates an open-world interactive narrative game. It not only generates game scenarios and challenges that target specific cognitive domains, but also employs conversational strategies to offer guidance and companionship. To validate its efficacy, we pioneer a psychology-grounded evaluation protocol LETGAMESEVAL, establishing comprehensive metrics for rehabilitative assessment. Building upon this, our experimental results from both LLM-based assessors and human expert evaluations demonstrate the significant potential of our approach, positioning LETGAMES as a promising solution to the widespread need for more accessible and tailored cognitive training tools. Our code will be open-sourced upon acceptance.
>
---
#### [new 193] Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于安全攻击任务，旨在破解大视觉语言模型的安全机制。通过操纵注意力模式，提高攻击成功率并减少迭代次数。**

- **链接: [https://arxiv.org/pdf/2604.10299](https://arxiv.org/pdf/2604.10299)**

> **作者:** Jingru Li; Wei Ren; Tianqing Zhu
>
> **备注:** Accepted to ACL 2026. Code: this https URL
>
> **摘要:** Large Vision-Language Models (LVLMs) rely on attention-based retrieval of safety instructions to maintain alignment during generation. Existing attacks typically optimize image perturbations to maximize harmful output likelihood, but suffer from slow convergence due to gradient conflict between adversarial objectives and the model's safety-retrieval mechanism. We propose Attention-Guided Visual Jailbreaking, which circumvents rather than overpowers safety alignment by directly manipulating attention patterns. Our method introduces two simple auxiliary objectives: (1) suppressing attention to alignment-relevant prefix tokens and (2) anchoring generation on adversarial image features. This simple yet effective push-pull formulation reduces gradient conflict by 45% and achieves 94.4% attack success rate on Qwen-VL (vs. 68.8% baseline) with 40% fewer iterations. At tighter perturbation budgets ($\epsilon=8/255$), we maintain 59.0% ASR compared to 45.7% for standard methods. Mechanistic analysis reveals a failure mode we term safety blindness: successful attacks suppress system-prompt attention by 80%, causing models to generate harmful content not by overriding safety rules, but by failing to retrieve them.
>
---
#### [new 194] Detecting RAG Extraction Attack via Dual-Path Runtime Integrity Game
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于安全防护任务，解决RAG系统中的知识库泄露问题。提出CanaryRAG机制，通过双路径验证检测攻击，有效防止数据泄露。**

- **链接: [https://arxiv.org/pdf/2604.10717](https://arxiv.org/pdf/2604.10717)**

> **作者:** Yuanbo Xie; Yingjie Zhang; Yulin Li; Shouyou Song; Xiaokun Chen; Zhihan Liu; Liya Su; Tingwen Liu
>
> **备注:** Accepted by ACL 2026 Main
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems augment large language models with external knowledge, yet introduce a critical security vulnerability: RAG Knowledge Base Leakage, wherein adversarial prompts can induce the model to divulge retrieved proprietary content. Recent studies reveal that such leakage can be executed through adaptive and iterative attack strategies (named RAG extraction attack), while effective countermeasures remain notably lacking. To bridge this gap, we propose CanaryRAG, a runtime defense mechanism inspired by stack canaries in software security. CanaryRAG embeds carefully designed canary tokens into retrieved chunks and reformulates RAG extraction defense as a dual-path runtime integrity game. Leakage is detected in real time whenever either the target or oracle path violates its expected canary behavior, including under adaptive suppression and obfuscation. Extensive evaluations against existing attacks demonstrate that CanaryRAG provides robust defense, achieving substantially lower chunk recovery rates than state-of-the-art baselines while imposing negligible impact on task performance and inference latency. Moreover, as a plug-and-play solution, CanaryRAG can be seamlessly integrated into arbitrary RAG pipelines without requiring retraining or structural modifications, offering a practical and scalable safeguard for proprietary data.
>
---
#### [new 195] EvoDiagram: Agentic Editable Diagram Creation via Design Expertise Evolution
- **分类: cs.HC; cs.CL; cs.CV**

- **简介: 该论文提出EvoDiagram，解决自动创建可编辑图表的任务，通过多智能体系统和设计知识演化机制，提升图表的结构一致性和美观性。**

- **链接: [https://arxiv.org/pdf/2604.09568](https://arxiv.org/pdf/2604.09568)**

> **作者:** Tianfu Wang; Leilei Ding; Ziyang Tao; Yi Zhan; Zhiyuan Ma; Wei Wu; Yuxuan Lei; Yuan Feng; Junyang Wang; Yin Wu; Yizhao Xu; Hongyuan Zhu; Qi Liu; Nicholas Jing Yuan; Yanyong Zhang; Hui Xiong
>
> **摘要:** High-fidelity diagram creation requires the complex orchestration of semantic topology, visual styling, and spatial layout, posing a significant challenge for automated systems. Existing methods also suffer from a representation gap: pixel-based models often lack precise control, while code-based synthesis limits intuitive flexibility. To bridge this gap, we introduce EvoDiagram, an agentic framework that generates object-level editable diagrams via an intermediate canvas schema. EvoDiagram employs a coordinated multi-agent system to decouple semantic intent from rendering logic, resolving conflicts across heterogeneous design layers. Additionally, we propose a design knowledge evolution mechanism that distills execution traces into a hierarchical memory of domain guidelines, enabling agents to retrieve context-aware expertise adaptively. We further release CanvasBench, a benchmark consisting of both data and metrics for canvas-based diagramming. Extensive experiments demonstrate that EvoDiagram exhibits excellent performance and balance against baselines in generating editable, structurally consistent, and aesthetically coherent diagrams. Our code is available at this https URL.
>
---
#### [new 196] Cross-Cultural Value Awareness in Large Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于AI公平性研究任务，旨在解决LVLMs中文化偏见问题。通过分析不同文化背景下模型的价值判断，评估其对文化差异的敏感度。**

- **链接: [https://arxiv.org/pdf/2604.09945](https://arxiv.org/pdf/2604.09945)**

> **作者:** Phillip Howard; Xin Su; Kathleen C. Fraser
>
> **摘要:** The rapid adoption of large vision-language models (LVLMs) in recent years has been accompanied by growing fairness concerns due to their propensity to reinforce harmful societal stereotypes. While significant attention has been paid to such fairness concerns in the context of social biases, relatively little prior work has examined the presence of stereotypes in LVLMs related to cultural contexts such as religion, nationality, and socioeconomic status. In this work, we aim to narrow this gap by investigating how cultural contexts depicted in images influence the judgments LVLMs make about a person's moral, ethical, and political values. We conduct a multi-dimensional analysis of such value judgments in five popular LVLMs using counterfactual image sets, which depict the same person across different cultural contexts. Our evaluation framework diagnoses LVLM awareness of cultural value differences through the use of Moral Foundations Theory, lexical analyses, and the sensitivity of generated values to depicted cultural contexts.
>
---
#### [new 197] VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文属于视觉与语言导航任务，解决虚假前提指令下的导航问题。通过构建VLN-NF基准和提出ROAM方法，提升代理在目标不存在时的探索与决策能力。**

- **链接: [https://arxiv.org/pdf/2604.10533](https://arxiv.org/pdf/2604.10533)**

> **作者:** Hung-Ting Su; Ting-Jun Wang; Jia-Fong Yeh; Min Sun; Winston H. Hsu
>
> **备注:** Accepted at ACL 2026. The first two authors contributed equally to the technical work
>
> **摘要:** Conventional Vision-and-Language Navigation (VLN) benchmarks assume instructions are feasible and the referenced target exists, leaving agents ill-equipped to handle false-premise goals. We introduce VLN-NF, a benchmark with false-premise instructions where the target is absent from the specified room and agents must navigate, gather evidence through in-room exploration, and explicitly output NOT-FOUND. VLN-NF is constructed via a scalable pipeline that rewrites VLN instructions using an LLM and verifies target absence with a VLM, producing plausible yet factually incorrect goals. We further propose REV-SPL to jointly evaluate room reaching, exploration coverage, and decision correctness. To address this challenge, we present ROAM, a two-stage hybrid that combines supervised room-level navigation with LLM/VLM-driven in-room exploration guided by a free-space clearance prior. ROAM achieves the best REV-SPL among compared methods, while baselines often under-explore and terminate prematurely under unreliable instructions. VLN-NF project page can be found at this https URL.
>
---
#### [new 198] Hijacking Text Heritage: Hiding the Human Signature through Homoglyphic Substitution
- **分类: cs.CR; cs.CL; cs.IR**

- **简介: 该论文属于隐私保护任务，旨在解决文本中个人身份信息泄露问题。通过同形异义字符替换，降低风格分析系统的有效性。**

- **链接: [https://arxiv.org/pdf/2604.10271](https://arxiv.org/pdf/2604.10271)**

> **作者:** Robert Dilworth
>
> **备注:** 30 pages, 9 figures
>
> **摘要:** In what way could a data breach involving government-issued IDs such as passports, driver's licenses, etc., rival a random voluntary disclosure on a nondescript social-media platform? At first glance, the former appears more significant, and that is a valid assessment. The disclosed data could contain an individual's date of birth and address; for all intents and purposes, a leak of that data would be disastrous. Given the threat, the latter scenario involving an innocuous online post seems comparatively harmless--or does it? From that post and others like it, a forensic linguist could stylometrically uncover equivalent pieces of information, estimating an age range for the author (adolescent or adult) and narrowing down their geographical location (specific country). While not an exact science--the determinations are statistical--stylometry can reveal comparable, though noticeably diluted, information about an individual. To prevent an ID from being breached, simply sharing it as little as possible suffices. Preventing the leakage of personal information from written text requires a more complex solution: adversarial stylometry. In this paper, we explore how performing homoglyph substitution--the replacement of characters with visually similar alternatives (e.g., "h" $\texttt{[U+0068]}$ $\rightarrow$ "h" $\texttt{[U+04BB]}$)--on text can degrade stylometric systems.
>
---
#### [new 199] FinTrace: Holistic Trajectory-Level Evaluation of LLM Tool Calling for Long-Horizon Financial Tasks
- **分类: cs.AI; cs.CE; cs.CL; cs.MM**

- **简介: 该论文属于金融任务中的工具调用评估，解决现有基准无法全面衡量轨迹推理质量的问题。提出FinTrace基准和训练数据，提升模型信息利用与答案质量。**

- **链接: [https://arxiv.org/pdf/2604.10015](https://arxiv.org/pdf/2604.10015)**

> **作者:** Yupeng Cao; Haohang Li; Weijin Liu; Wenbo Cao; Anke Xu; Lingfei Qian; Xueqing Peng; Minxue Tang; Zhiyuan Yao; Jimin Huang; K.P. Subbalakshmi; Zining Zhu; Jordan W. Suchow; Yangyang Yu
>
> **摘要:** Recent studies demonstrate that tool-calling capability enables large language models (LLMs) to interact with external environments for long-horizon financial tasks. While existing benchmarks have begun evaluating financial tool calling, they focus on limited scenarios and rely on call-level metrics that fail to capture trajectory-level reasoning quality. To address this gap, we introduce FinTrace, a benchmark comprising 800 expert-annotated trajectories spanning 34 real-world financial task categories across multiple difficulty levels. FinTrace employs a rubric-based evaluation protocol with nine metrics organized along four axes -- action correctness, execution efficiency, process quality, and output quality -- enabling fine-grained assessment of LLM tool-calling behavior. Our evaluation of 13 LLMs reveals that while frontier models achieve strong tool selection, all models struggle with information utilization and final answer quality, exposing a critical gap between invoking the right tools and reasoning effectively over their outputs. To move beyond diagnosis, we construct FinTrace-Training, the first trajectory-level preference dataset for financial tool-calling, containing 8,196 curated trajectories with tool-augmented contexts and preference pairs. We fine-tune Qwen-3.5-9B using supervised fine-tuning followed by direct preference optimization (DPO) and show that training on FinTrace-Training consistently improves intermediate reasoning metrics, with DPO more effectively suppressing failure modes. However, end-to-end answer quality remains a bottleneck, indicating that trajectory-level improvements do not yet fully propagate to final output quality.
>
---
#### [new 200] Revisiting Compositionality in Dual-Encoder Vision-Language Models: The Role of Inference
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于视觉语言模型任务，旨在解决双编码器模型在组合性任务上的性能瓶颈。通过引入局部对齐机制，提升模型的组合泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.11496](https://arxiv.org/pdf/2604.11496)**

> **作者:** Imanol Miranda; Ander Salaberria; Eneko Agirre; Gorka Azkune
>
> **摘要:** Dual-encoder Vision-Language Models (VLMs) such as CLIP are often characterized as bag-of-words systems due to their poor performance on compositional benchmarks. We argue that this limitation may stem less from deficient representations than from the standard inference protocol based on global cosine similarity. First, through controlled diagnostic experiments, we show that explicitly enforcing fine-grained region-segment alignment at inference dramatically improves compositional performance without updating pretrained encoders. We then introduce a lightweight transformer that learns such alignments directly from frozen patch and token embeddings. Comparing against full fine-tuning and prior end-to-end compositional training methods, we find that although these approaches improve in-domain retrieval, their gains do not consistently transfer under distribution shift. In contrast, learning localized alignment over frozen representations matches full fine-tuning on in-domain retrieval while yielding substantial improvements on controlled out-of-domain compositional benchmarks. These results identify global embedding matching as a key bottleneck in dual-encoder VLMs and highlight the importance of alignment mechanisms for robust compositional generalization.
>
---
#### [new 201] ACE-TA: An Agentic Teaching Assistant for Grounded Q&A, Quiz Generation, and Code Tutoring
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文提出ACE-TA，一个用于编程教学的智能助手，解决编程答疑、测验生成和代码辅导问题，整合了问答、测验生成和代码指导模块。**

- **链接: [https://arxiv.org/pdf/2604.09572](https://arxiv.org/pdf/2604.09572)**

> **作者:** Himanshu Tripathi; Charlottee Crowell; Kaley Newlin; Subash Neupane; Shahram Rahimi; Jason Keith
>
> **摘要:** We introduce ACE-TA, the Agentic Coding and Explanations Teaching Assistant framework, that autonomously routes conceptual queries drawn from programming course material to grounded Q&A, stepwise coding guidance, and automated quiz generation using pre-trained Large Language Models (LLMs). ACE-TA consists of three coordinated modules: a retrieval grounded conceptual Q&A system that provides precise, context-aligned explanations; a quiz generator that constructs adaptive, multi-topic assessments targeting higher-order understanding; and an interactive code tutor that guides students through step-by-step reasoning with sandboxed execution and iterative feedback.
>
---
#### [new 202] LLM Nepotism in Organizational Governance
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文研究AI在组织治理中的偏见问题，属于AI公平性任务。解决AI信任偏好导致的不公平招聘与决策问题，通过模拟实验分析并提出缓解方法。**

- **链接: [https://arxiv.org/pdf/2604.09620](https://arxiv.org/pdf/2604.09620)**

> **作者:** Shunqi Mao; Wei Guo; Dingxin Zhang; Chaoyi Zhang; Weidong Cai
>
> **备注:** 23 pages, 3 figures, 13 tables
>
> **摘要:** Large language models are increasingly used to support organizational decisions from hiring to governance, raising fairness concerns in AI-assisted evaluation. Prior work has focused mainly on demographic bias and broader preference effects, rather than on whether evaluators reward expressed trust in AI itself. We study this phenomenon as LLM Nepotism, an attitude-driven bias channel in which favorable signals toward AI are rewarded even when they are not relevant to role-related merit. We introduce a two-phase simulation pipeline that first isolates AI-trust preference in qualification-matched resume screening and then examines its downstream effects in board-level decision making. Across several popular LLMs, we find that resume screeners tend to favor candidates with positive or non-critical attitudes toward AI, discriminating skeptical, human-centered counterparts. These biases suggest a loophole: LLM-based hiring can produce more homogeneous AI-trusting organizations, whose decision-makers exhibit greater scrutiny failure and delegation to AI agents, approving flawed proposals more readily while favoring AI-delegation initiatives. To mitigate this behavior, we additionally study prompt-based mitigation and propose Merit-Attitude Factorization, which separates non-merit AI attitude from merit-based evaluation and attenuates this bias across experiments.
>
---
#### [new 203] Exploring Structural Complexity in Normative RAG with Graph-based approaches: A case study on the ETSI Standards
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决标准文档处理难题。通过图结构RAG方法提升检索性能，针对ETSI标准进行实验验证。**

- **链接: [https://arxiv.org/pdf/2604.09868](https://arxiv.org/pdf/2604.09868)**

> **作者:** Aiman Al Masoud; Marco Arazzi; Simone Germani; Antonino Nocera
>
> **备注:** 6 pages, 7 figures
>
> **摘要:** Industrial standards and normative documents exhibit intricate hierarchical structures, domain-specific lexicons, and extensive cross-referential dependencies, which making it challenging to process them directly by Large Language Models (LLMs). While Retrieval-Augmented Generation (RAG) provides a computationally efficient alternative to LLM fine-tuning, standard "vanilla" vector-based retrieval may fail to capture the latent structural and relational features intrinsic in normative documents. With the objective of shedding light on the most promising technique for building high-performance RAG solutions for normative, standards, and regulatory documents, this paper investigates the efficacy of Graph RAG architectures, which represent information as interconnected nodes, thus moving from simple semantic similarity toward a more robust, relation-aware retrieval mechanism. Despite the promise of graph-based techniques, there is currently a lack of empirical evidence as to which is the optimal indexing strategy for technical standards. Therefore, to help solve this knowledge gap, we propose a specialized RAG methodology tailored to the unique structure and lexical characteristics of standards and regulatory documents. Moreover, to keep our investigation grounded, we focus on well-known public standards, such as the ETSI EN 301 489 series. We evaluate several lightweight and low-latency strategies designed to embed document structure directly into the retrieval workflow. The considered approaches are rigorously tested against a custom synthesized Q&A dataset, facilitating a quantitative performance analysis. Our experimental results demonstrate that the incorporation of structural and lexical information into the index can enhance, at least to some extent, retrieval performance, providing a scalable framework for automated normative and standards elaboration.
>
---
#### [new 204] The Past Is Not Past: Memory-Enhanced Dynamic Reward Shaping
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决大语言模型采样多样性不足问题，通过引入历史行为记忆提升奖励设计，增强探索并减少重复错误。**

- **链接: [https://arxiv.org/pdf/2604.11297](https://arxiv.org/pdf/2604.11297)**

> **作者:** Yang Liu; Enxi Wang; Yufei Gao; Weixin Zhang; Bo Wang; Zhiyuan Zeng; Yikai Zhang; Yining Zheng; Xipeng Qiu
>
> **摘要:** Despite the success of reinforcement learning for large language models, a common failure mode is reduced sampling diversity, where the policy repeatedly generates similar erroneous behaviors. Classical entropy regularization encourages randomness under the current policy, but does not explicitly discourage recurrent failure patterns across rollouts. We propose MEDS, a Memory-Enhanced Dynamic reward Shaping framework that incorporates historical behavioral signals into reward design. By storing and leveraging intermediate model representations, we capture features of past rollouts and use density-based clustering to identify frequently recurring error patterns. Rollouts assigned to more prevalent error clusters are penalized more heavily, encouraging broader exploration while reducing repeated mistakes. Across five datasets and three base models, MEDS consistently improves average performance over existing baselines, achieving gains of up to 4.13 pass@1 points and 4.37 pass@128 points. Additional analyses using both LLM-based annotations and quantitative diversity metrics show that MEDS increases behavioral diversity during sampling.
>
---
#### [new 205] Use of AI Tools: Guidelines to Maintain Academic Integrity in Computing Colleges
- **分类: cs.CY; cs.AI; cs.CL; cs.ET**

- **简介: 本文探讨AI工具在计算机学院教学中的应用，旨在解决学术诚信问题。论文提出指导方针和评估模型，帮助教师合理使用AI工具，确保教学效果与学术规范。**

- **链接: [https://arxiv.org/pdf/2604.11111](https://arxiv.org/pdf/2604.11111)**

> **作者:** Hatem M. El-boghdadi; Toqeer Ali Syed; Ali Akarma; Qamar Wali
>
> **备注:** This paper is in press for Volume 33 Issue 4 (2025) International Journal of Energy, Environment, and Economics
>
> **摘要:** The rapid adoption of AI tools such as ChatGPT has significantly transformed academic practices, offering considerable benefits for both students and faculty in computing disciplines. These tools have been shown to enhance learning efficiency, academic self-efficacy, and confidence. However, their increasing use also raises pressing concerns regarding the preservation of academic integrity -- an essential pillar of the educational process. This paper explores the implications of widespread AI tool usage within computing colleges, with a particular focus on how to align their use with the principles of academic honesty. We begin by classifying common assessment techniques employed in computing education and examine how each may be impacted by AI-assisted tools. Building on this foundation, we propose a set of general guidelines applicable across various assessment formats to help instructors responsibly integrate AI tools into their pedagogy. Furthermore, we provide targeted, assessment-specific recommendations designed to uphold educational objectives while mitigating risks of academic misconduct. These guidelines serve as a practical framework for instructors aiming to balance the pedagogical advantages of AI tools with the imperative of maintaining academic integrity in computing education. Finally, we introduce a formal model that provides a structured mathematical framework for evaluating student assessments in the presence of AI-assisted tools.
>
---
#### [new 206] Head-wise Modality Specialization within MLLMs for Robust Fake News Detection under Missing Modality
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态虚假新闻检测任务，解决缺失模态下的鲁棒性问题。通过引入头级模态专精机制和单模态知识保留策略，提升模型在模态缺失时的验证能力。**

- **链接: [https://arxiv.org/pdf/2604.09711](https://arxiv.org/pdf/2604.09711)**

> **作者:** Kai Qian; Weijie Shi; Jiaqi Wang; Mengze Li; Hao Chen; Yue Cui; Hanghui Guo; Ziyi Liu; Jia Zhu; Jiajie Xu
>
> **摘要:** Multimodal fake news detection (MFND) aims to verify news credibility by jointly exploiting textual and visual evidence. However, real-world news dissemination frequently suffers from missing modality due to deleted images, corrupted screenshots, and similar issues. Thus, robust detection in this scenario requires preserving strong verification ability for each modality, which is challenging in MFND due to insufficient learning of the low-contribution modality and scarce unimodal annotations. To address this issue, we propose Head-wise Modality Specialization within Multimodal Large Language Models (MLLMs) for robust MFND under missing modality. Specifically, we first systematically study attention heads in MLLMs and their relationship with performance under missing modality, showing that modality-critical heads serve as key carriers of unimodal verification ability through their modality specialization. Based on this observation, to better preserve verification ability for the low-contribution modality, we introduce a head-wise specialization mechanism that explicitly allocates these heads to different modalities and preserves their specialization through lower-bound attention constraints. Furthermore, to better exploit scarce unimodal annotations, we propose a Unimodal Knowledge Retention strategy that prevents these heads from drifting away from the unimodal knowledge learned from limited supervision. Experiments show that our method improves robustness under missing modality while preserving performance with full multimodal input.
>
---
#### [new 207] Learning from Contrasts: Synthesizing Reasoning Paths from Diverse Search Trajectories
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出CRPS框架，解决MCTS中监督信息提取效率低的问题。通过对比高、低质量搜索轨迹，合成有效推理路径，提升模型性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.11365](https://arxiv.org/pdf/2604.11365)**

> **作者:** Peiyang Liu; Zhirui Chen; Xi Wang; Di Liang; Youru Li; Zhi Cai; Wei Ye
>
> **摘要:** Monte Carlo Tree Search (MCTS) has been widely used for automated reasoning data exploration, but current supervision extraction methods remain inefficient. Standard approaches retain only the single highest-reward trajectory, discarding the comparative signals present in the many explored paths. Here we introduce \textbf{Contrastive Reasoning Path Synthesis (CRPS)}, a framework that transforms supervision extraction from a filtering process into a synthesis procedure. CRPS uses a structured reflective process to analyze the differences between high- and low-quality search trajectories, extracting explicit information about strategic pivots and local failure modes. These insights guide the synthesis of reasoning chains that incorporate success patterns while avoiding identified pitfalls. We show empirically that models fine-tuned on just 60K CRPS-synthesized examples match or exceed the performance of baselines trained on 590K examples derived from standard rejection sampling, a 20$\times$ reduction in dataset size. Furthermore, CRPS improves generalization on out-of-domain benchmarks, demonstrating that learning from the contrast between success and failure produces more transferable reasoning capabilities than learning from success alone.
>
---
#### [new 208] Explainability and Certification of AI-Generated Educational Assessments
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于教育技术任务，旨在解决AI生成评估的透明性与认证问题。通过构建解释框架和认证流程，提升AI评估的可信度与可审计性。**

- **链接: [https://arxiv.org/pdf/2604.09622](https://arxiv.org/pdf/2604.09622)**

> **作者:** Antoun Yaacoub; Zainab Assaghir; Anuradha Kar
>
> **备注:** Chapter to be published in a Springer special book "Emerging trends in Computer Science and Computer Engineering Education Book"
>
> **摘要:** The rapid adoption of generative artificial intelligence (AI) in educational assessment has created new opportunities for scalable item creation, personalized feedback, and efficient formative evaluation. However, despite advances in taxonomy alignment and automated question generation, the absence of transparent, explainable, and certifiable mechanisms limits institutional and accreditation-level acceptance. This chapter proposes a comprehensive framework for explainability and certification of AI-generated assessment items, combining self-rationalization, attribution-based analysis, and post-hoc verification to produce interpretable cognitive-alignment evidence grounded in Bloom's and SOLO taxonomies. A structured certification metadata schema is introduced to capture provenance, alignment predictions, reviewer actions, and ethical indicators, enabling audit-ready documentation consistent with emerging governance requirements. A traffic-light certification workflow operationalizes these signals by distinguishing auto-certifiable items from those requiring human review or rejection. A proof-of-concept study on 500 AI-generated computer science questions demonstrates the framework's feasibility, showing improved transparency, reduced instructor workload, and enhanced auditability. The chapter concludes by outlining ethical implications, policy considerations, and directions for future research, positioning explainability and certification as essential components of trustworthy, accreditation-ready AI assessment systems.
>
---
#### [new 209] Seven simple steps for log analysis in AI systems
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于AI系统日志分析任务，旨在解决缺乏标准化分析方法的问题。提出七步流程框架，提供代码示例和实践指导。**

- **链接: [https://arxiv.org/pdf/2604.09563](https://arxiv.org/pdf/2604.09563)**

> **作者:** Magda Dubois; Ekin Zorer; Maia Hamin; Joe Skinner; Alexandra Souly; Jerome Wynne; Harry Coppock; Lucas Satos; Sayash Kapoor; Sunischal Dev; Keno Juchems; Kimberly Mai; Timo Flesch; Lennart Luettgau; Charles Teague; Eric Patey; JJ Allaire; Lorenzo Pacchiardi; Jose Hernandez-Orallo; Cozmin Ududec
>
> **摘要:** AI systems produce large volumes of logs as they interact with tools and users. Analysing these logs can help understand model capabilities, propensities, and behaviours, or assess whether an evaluation worked as intended. Researchers have started developing methods for log analysis, but a standardised approach is still missing. Here we suggest a pipeline based on current best practices. We illustrate it with concrete code examples in the Inspect Scout library, provide detailed guidance on each step, and highlight common pitfalls. Our framework provides researchers with a foundation for rigorous and reproducible log analysis.
>
---
## 更新

#### [replaced 001] Process-Centric Analysis of Agentic Software Systems
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于软件系统分析任务，旨在解决 agentic 系统执行过程难以评估的问题。通过引入 Graphectory 方法，分析代理系统的轨迹，提升对其决策过程的理解与优化。**

- **链接: [https://arxiv.org/pdf/2512.02393](https://arxiv.org/pdf/2512.02393)**

> **作者:** Shuyang Liu; Yang Chen; Rahul Krishna; Saurabh Sinha; Jatin Ganhotra; Reyhan Jabbarvand
>
> **摘要:** Agentic systems are modern software systems: they consist of orchestrated modules, expose interfaces, and are deployed in software pipelines. Unlike conventional programs, their execution, i.e., trajectories, is inherently stochastic and adaptive to the problems they solve. Evaluation of such systems is often outcome-centric. This narrow focus overlooks detailed insights, failing to explain how agents reason, plan, act, or change their strategies. Inspired by the structured representation of conventional software systems as graphs, we introduce Graphectory to systematically encode the temporal and semantic relations in such systems. Using Graphectory, we automatically analyze 4000 trajectories of two dominant agentic programming workflows, SWE-agent and OpenHands, with four backbone Large Language Models (LLMs), attempting to resolve SWE-bench issues. Our automated analyses (completed within four minutes) reveal that: (1) agents using richer prompts or stronger LLMs exhibit more complex Graphectory, reflecting deeper exploration, broader context gathering, and more thorough validation; (2) agents' strategies vary with problem difficulty and the underlying LLM - for resolved issues, strategies often follow coherent localization-patching-validation steps, while unresolved ones exhibit chaotic or backtracking behaviors; and (3) even successful agentic systems often display inefficient processes. We also implement a novel technique for real-time construction and analysis of Graphectory and Langutory during agent execution to flag trajectory issues. Upon detecting such issues, the technique notifies the agent with a diagnostic message and, when applicable, rolls back the trajectory. Experiments show that online monitoring and interventions improve resolution rates by 6.9%-23.5% across models for problematic instances, while significantly shortening trajectories with near-zero overhead.
>
---
#### [replaced 002] Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统任务，旨在提升对话句向量的质量。通过利用模板信息，提出TaDSE方法，解决标注困难问题，并在多个数据集上取得显著效果。**

- **链接: [https://arxiv.org/pdf/2305.14299](https://arxiv.org/pdf/2305.14299)**

> **作者:** Minsik Oh; Jiwei Li; Guoyin Wang
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** Learning high quality sentence embeddings from dialogues has drawn increasing attentions as it is essential to solve a variety of dialogue-oriented tasks with low annotation cost. Annotating and gathering utterance relationships in conversations are difficult, while token-level annotations, \eg, entities, slots and templates, are much easier to obtain. Other sentence embedding methods are usually sentence-level self-supervised frameworks and cannot utilize token-level extra knowledge. We introduce Template-aware Dialogue Sentence Embedding (TaDSE), a novel augmentation method that utilizes template information to learn utterance embeddings via self-supervised contrastive learning framework. We further enhance the effect with a synthetically augmented dataset that diversifies utterance-template association, in which slot-filling is a preliminary step. We evaluate TaDSE performance on five downstream benchmark dialogue datasets. The experiment results show that TaDSE achieves significant improvements over previous SOTA methods for dialogue. We further introduce a novel analytic instrument of semantic compression test, for which we discover a correlation with uniformity and alignment. Our code is available at this https URL
>
---
#### [replaced 003] EduIllustrate: Towards Scalable Automated Generation Of Multimodal Educational Content
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文提出EduIllustrate，解决K-12 STEM教育内容的多模态生成问题，通过基准测试评估大模型的图文解释能力。**

- **链接: [https://arxiv.org/pdf/2604.05005](https://arxiv.org/pdf/2604.05005)**

> **作者:** Shuzhen Bi; Mingzi Zhang; Zhuoxuan Li; Xiaolong Wang; Keqian Li; Aimin Zhou
>
> **摘要:** Large language models are increasingly used as educational assistants, yet evaluation of their educational capabilities remains concentrated on question-answering and tutoring tasks. A critical gap exists for multimedia instructional content generation -- the ability to produce coherent, diagram-rich explanations that combine geometrically accurate visuals with step-by-step reasoning. We present EduIllustrate, a benchmark for evaluating LLMs on interleaved text-diagram explanation generation for K-12 STEM problems. The benchmark comprises 230 problems spanning five subjects and three grade levels, a standardized generation protocol with sequential anchoring to enforce cross-diagram visual consistency, and an 8-dimension evaluation rubric grounded in multimedia learning theory covering both text and visual quality. Evaluation of ten LLMs reveals a wide performance spread: Gemini 3.0 Pro Preview leads at 87.8\%, while Kimi-K2.5 achieves the best cost-efficiency (80.8\% at \\$0.12/problem). Workflow ablation confirms sequential anchoring improves Visual Consistency by 13\% at 94\% lower cost. Human evaluation with 20 expert raters validates LLM-as-judge reliability for objective dimensions ($\rho \geq 0.83$) while revealing limitations on subjective visual assessment.
>
---
#### [replaced 004] Infusing Theory of Mind into Socially Intelligent LLM Agents
- **分类: cs.CL**

- **简介: 该论文属于社会智能对话任务，旨在解决LLM缺乏理论心智问题。通过引入ToMAgent，提升对话效果和目标达成效率。**

- **链接: [https://arxiv.org/pdf/2509.22887](https://arxiv.org/pdf/2509.22887)**

> **作者:** EunJeong Hwang; Yuwei Yin; Giuseppe Carenini; Peter West; Vered Shwartz
>
> **摘要:** Theory of Mind (ToM)-an understanding of the mental states of others-is a key aspect of human social intelligence, yet, chatbots and LLM-based social agents do not typically integrate it. In this work, we demonstrate that LLMs that explicitly use ToM get better at dialogue, achieving goals more effectively. After showing that simply prompting models to generate mental states between dialogue turns already provides significant benefit, we further introduce ToMAgent (ToMA), a ToM-focused dialogue agent. ToMA is trained by pairing ToM with dialogue lookahead to produce mental states that are maximally useful for achieving dialogue goals. Experiments on the Sotopia interactive social evaluation benchmark demonstrate the effectiveness of our method over a range of baselines. Comprehensive analysis shows that ToMA exhibits more strategic, goal-oriented reasoning behaviors, which enable long-horizon adaptation, while maintaining better relationships with their partners. Our results suggest a step forward in integrating ToM for building socially intelligent LLM agents.
>
---
#### [replaced 005] Enhancing Geo-localization for Crowdsourced Flood Imagery via LLM-Guided Attention
- **分类: cs.CL; cs.AI; cs.CV; cs.CY**

- **简介: 该论文属于图像地理定位任务，旨在解决社交媒体洪水图像缺乏可靠地理信息的问题。通过引入LLM引导的注意力机制，提升VPR模型的定位性能。**

- **链接: [https://arxiv.org/pdf/2512.11811](https://arxiv.org/pdf/2512.11811)**

> **作者:** Fengyi Xu; Jun Ma; Waishan Qiu; Cui Guo; Jack C.P. Cheng
>
> **备注:** Updated author list to include additional contributor. Revised title and improved methodology section based on collaborative feedback
>
> **摘要:** Crowdsourced social media imagery provides real-time visual evidence of urban flooding but often lacks reliable geographic metadata for emergency response. Existing Visual Place Recognition (VPR) models struggle to geo-localize these images due to cross-source domain shifts and visual distortions. We present VPR-AttLLM, a model-agnostic framework integrating the semantic reasoning and geospatial knowledge of Large Language Models (LLMs) into VPR pipelines via attention-guided descriptor enhancement. VPR-AttLLM uses LLMs to isolate location-informative regions and suppress transient noise, improving retrieval without model retraining or new data. We evaluate this framework across San Francisco and Hong Kong using established queries, synthetic flooding scenarios, and real social media flood images. Integrating VPR-AttLLM with state-of-the-art models (CosPlace, EigenPlaces, SALAD) consistently improves recall, yielding 1-3% relative gains and up to 8% on challenging real flood imagery. By embedding urban perception principles into attention mechanisms, VPR-AttLLM bridges human-like spatial reasoning with modern VPR architectures. Its plug-and-play design and cross-source robustness offer a scalable solution for rapid geo-localization of crowdsourced crisis imagery, advancing cognitive urban resilience.
>
---
#### [replaced 006] Stop Fixating on Prompts: Reasoning Hijacking and Constraint Tightening for Red-Teaming LLM Agents
- **分类: cs.CL**

- **简介: 该论文属于安全评估任务，旨在解决LLM代理的漏洞问题。提出JailAgent框架，通过隐式操控推理过程，无需修改用户提示，提升红队测试效果。**

- **链接: [https://arxiv.org/pdf/2604.05549](https://arxiv.org/pdf/2604.05549)**

> **作者:** Yanxu Mao; Peipei Liu; Tiehan Cui; Congying Liu; Mingzhe Xing; Datao You
>
> **摘要:** With the widespread application of LLM-based agents across various domains, their complexity has introduced new security threats. Existing red-team methods mostly rely on modifying user prompts, which lack adaptability to new data and may impact the agent's performance. To address the challenge, this paper proposes the JailAgent framework, which completely avoids modifying the user prompt. Specifically, it implicitly manipulates the agent's reasoning trajectory and memory retrieval with three key stages: Trigger Extraction, Reasoning Hijacking, and Constraint Tightening. Through precise trigger identification, real-time adaptive mechanisms, and an optimized objective function, JailAgent demonstrates outstanding performance in cross-model and cross-scenario environments.
>
---
#### [replaced 007] From Speech-to-Spatial: Grounding Utterances on A Live Shared View with Augmented Reality
- **分类: cs.HC; cs.CL; cs.ET; cs.IR**

- **简介: 该论文属于自然语言处理与增强现实结合的任务，旨在将语音指令转化为空间化的AR引导。解决远程指导中指代不明确的问题，通过分析语音参考模式，生成直观的视觉指引，提升任务效率和用户体验。**

- **链接: [https://arxiv.org/pdf/2602.03059](https://arxiv.org/pdf/2602.03059)**

> **作者:** Yoonsang Kim; Divyansh Pradhan; Devshree Jadeja; Arie Kaufman
>
> **备注:** 11 pages, 6 figures. This is the author's version of the article that appeared at the IEEE Conference on Virtual Reality and 3D User Interfaces (IEEE VR) 2026
>
> **摘要:** We introduce Speech-to-Spatial, a referent disambiguation framework that converts verbal remote-assistance instructions into spatially grounded AR guidance. Unlike prior systems that rely on additional cues (e.g., gesture, gaze) or manual expert annotations, Speech-to-Spatial infers the intended target solely from spoken references (speech input). Motivated by our formative study of speech referencing patterns, we characterize recurring ways people specify targets (Direct Attribute, Relational, Remembrance, and Chained) and ground them to our object-centric relational graph. Given an utterance, referent cues are parsed and rendered as persistent in-situ AR visual guidance, reducing iterative micro-guidance ("a bit more to the right", "now, stop.") during remote guidance. We demonstrate the use cases of our system with remote guided assistance and intent disambiguation scenarios. Our evaluation shows that Speechto-Spatial improves task efficiency, reduces cognitive load, and enhances usability compared to a conventional voice-only baseline, transforming disembodied verbal instruction into visually explainable, actionable guidance on a live shared view.
>
---
#### [replaced 008] Physical Commonsense Reasoning for Lower-Resourced Languages and Dialects: a Study on Basque
- **分类: cs.CL**

- **简介: 该论文属于物理常识推理任务，旨在解决低资源语言如巴斯克语在非问答任务中的表现问题。作者构建了首个巴斯克语物理常识数据集BasPhyCo，并评估了多模型的表现。**

- **链接: [https://arxiv.org/pdf/2602.14812](https://arxiv.org/pdf/2602.14812)**

> **作者:** Jaione Bengoetxea; Itziar Gonzalez-Dios; Rodrigo Agerri
>
> **摘要:** Physical commonsense reasoning represents a fundamental capability of human intelligence, enabling individuals to understand their environment, predict future events, and navigate physical spaces. Recent years have witnessed growing interest in reasoning tasks within Natural Language Processing (NLP). However, no prior research has examined the performance of Large Language Models (LLMs) on non-question-answering (non-QA) physical commonsense reasoning tasks in low-resource languages such as Basque. Taking the Italian GITA as a starting point, this paper addresses this gap by presenting BasPhyCo, the first non-QA physical commonsense reasoning dataset for Basque, available in both standard and dialectal variants. We evaluate model performance across three hierarchical levels of commonsense understanding: (1) distinguishing between plausible and implausible narratives (accuracy), (2) identifying the conflicting element that renders a narrative implausible (consistency), and (3) determining the specific physical state that creates the implausibility (verifiability). These tasks were assessed using multiple multilingual LLMs as well as models pretrained specifically for Italian and Basque. Results indicate that, in terms of verifiability, LLMs exhibit limited physical commonsense capabilities in low-resource languages such as Basque, especially when processing dialectal variants.
>
---
#### [replaced 009] Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的置信度估计任务，旨在探讨语义标记是否能准确反映大模型的不确定性。研究发现，标记在分布内有效，但在分布外表现不稳定，需改进标记与真实不确定性的对齐。**

- **链接: [https://arxiv.org/pdf/2505.24778](https://arxiv.org/pdf/2505.24778)**

> **作者:** Jiayu Liu; Qing Zong; Weiqi Wang; Yangqiu Song
>
> **备注:** ACL 2025 Main
>
> **摘要:** As large language models (LLMs) are increasingly used in high-stakes domains, accurately assessing their confidence is crucial. Humans typically express confidence through epistemic markers (e.g., "fairly confident") instead of numerical values. However, it remains unclear whether LLMs consistently use these markers to reflect their intrinsic confidence due to the difficulty of quantifying uncertainty associated with various markers. To address this gap, we first define marker confidence as the observed accuracy when a model employs an epistemic marker. We evaluate its stability across multiple question-answering datasets in both in-distribution and out-of-distribution settings for open-source and proprietary LLMs. Our results show that while markers generalize well within the same distribution, their confidence is inconsistent in out-of-distribution scenarios. These findings raise significant concerns about the reliability of epistemic markers for confidence estimation, underscoring the need for improved alignment between marker based confidence and actual model uncertainty. Our code is available at this https URL.
>
---
#### [replaced 010] Re-Mask and Redirect: Exploiting Denoising Irreversibility in Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于安全对齐任务，针对扩散语言模型的拒绝机制进行攻击，提出TrajHijack方法，通过重掩码和注入前缀提升攻击成功率。**

- **链接: [https://arxiv.org/pdf/2604.08557](https://arxiv.org/pdf/2604.08557)**

> **作者:** Arth Singh
>
> **备注:** 15 pages
>
> **摘要:** Safety alignment in diffusion language models (dLLMs) relies on a single load-bearing assumption: that committed tokens are permanent. We show that violating this assumption, by re-masking committed refusal tokens and injecting a short affirmative prefix, achieves 74-82% ASR on HarmBench across all three publicly available safety-tuned dLLMs, rising to 92-98% with a generic 8-token compliance prefix. We call this attack TrajHijack; it is the first trajectory-level attack on dLLMs, requires no gradient computation, and generalizes across SFT and preference-optimized (VRPO) models. Three findings emerge. First, the vulnerability is irreducibly two-component: re-masking alone (4.4%) and prefix alone (5.7%) both fail. Second, gradient optimization via a differentiable Gumbel-softmax chain consistently degrades ASR (41.5% vs. 76.1%), because continuous perturbations push token distributions off-manifold. Third, A2D (the strongest published dLLM defense) is more vulnerable to TrajHijack (89.9%) than the undefended model (76.1%): its silent-refusal training removes the contextual resistance that trajectory-level attacks must overcome, an effect we call the Defense Inversion Effect.
>
---
#### [replaced 011] Pay Less Attention to Function Words for Free Robustness of Vision-Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于视觉-语言模型（VLM）任务，旨在提升模型的鲁棒性。针对功能词导致的脆弱性，提出FDA方法，通过差分减去功能词注意力，增强模型抗跨模态攻击能力。**

- **链接: [https://arxiv.org/pdf/2512.07222](https://arxiv.org/pdf/2512.07222)**

> **作者:** Qiwei Tian; Chenhao Lin; Zhengyu Zhao; Chao Shen
>
> **备注:** The paper has been accepted by ICLR26
>
> **摘要:** To address the trade-off between robustness and performance for robust VLM, we observe that function words could incur vulnerability of VLMs against cross-modal adversarial attacks, and propose Function-word De-Attention (FDA) accordingly to mitigate the impact of function words. Similar to differential amplifiers, our FDA calculates the original and the function-word cross-attention within attention heads, and differentially subtracts the latter from the former for more aligned and robust VLMs. Comprehensive experiments include 2 SOTA baselines under 6 different attacks on 2 downstream tasks, 3 datasets, and 3 models. Overall, our FDA yields an average 18/13/53% ASR drop with only 0.2/0.3/0.6% performance drops on the 3 tested models on retrieval, and a 90% ASR drop with a 0.3% performance gain on visual grounding. We demonstrate the scalability, generalization, and zero-shot performance of FDA experimentally, as well as in-depth ablation studies and analysis. Code is available at this https URL.
>
---
#### [replaced 012] MEDSYN: Benchmarking Multi-EviDence SYNthesis in Complex Clinical Cases for Multimodal Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出MEDSYN基准，用于评估多模态大语言模型在复杂临床案例中的多证据合成能力。任务是提升模型在诊断中的跨模态证据整合效果，解决其在真实临床场景中的性能不足问题。**

- **链接: [https://arxiv.org/pdf/2602.21950](https://arxiv.org/pdf/2602.21950)**

> **作者:** Boqi Chen; Xudong Liu; Jiachuan Peng; Marianne Frey-Marti; Bang Zheng; Kyle Lam; Lin Li; Jianing Qiu
>
> **摘要:** Multimodal large language models (MLLMs) have shown great potential in medical applications, yet existing benchmarks inadequately capture real-world clinical complexity. We introduce MEDSYN, a multilingual, multimodal benchmark of highly complex clinical cases with up to 7 distinct visual clinical evidence (CE) types per case. Mirroring clinical workflow, we evaluate 18 MLLMs on differential diagnosis (DDx) generation and final diagnosis (FDx) selection. While top models often match or even outperform human experts on DDx generation, all MLLMs exhibit a much larger DDx--FDx performance gap compared to expert clinicians, indicating a failure mode in synthesis of heterogeneous CE types. Ablations attribute this failure to (i) overreliance on less discriminative textual CE ($\it{e.g.}$, medical history) and (ii) a cross-modal CE utilization gap. We introduce Evidence Sensitivity to quantify the latter and show that a smaller gap correlates with higher diagnostic accuracy. Finally, we demonstrate how it can be used to guide interventions to improve model performance. We will open-source our benchmark and code.
>
---
#### [replaced 013] MerNav: A Highly Generalizable Memory-Execute-Review Framework for Zero-Shot Object Goal Navigation
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在提升零样本目标导航的成功率和泛化能力。提出MerNav框架，结合记忆、执行与回顾模块，有效提升导航性能。**

- **链接: [https://arxiv.org/pdf/2602.05467](https://arxiv.org/pdf/2602.05467)**

> **作者:** Dekang Qi; Shuang Zeng; Xinyuan Chang; Feng Xiong; Shichao Xie; Xiaolong Wu; Mu Xu
>
> **备注:** 9 pages, 2 figures, 5 tables, conference
>
> **摘要:** Visual Language Navigation (VLN) is one of the fundamental capabilities for embodied intelligence and a critical challenge that urgently needs to be addressed. However, existing methods are still unsatisfactory in terms of both success rate (SR) and generalization: Supervised Fine-Tuning (SFT) approaches typically achieve higher SR, while Training-Free (TF) approaches often generalize better, but it is difficult to obtain both simultaneously. To this end, we propose a Memory-Execute-Review framework. It consists of three parts: a hierarchical memory module for providing information support, an execute module for routine decision-making and actions, and a review module for handling abnormal situations and correcting behavior. We validated the effectiveness of this framework on the Object Goal Navigation task. Across 4 datasets, our average SR achieved absolute improvements of 7% and 5% compared to all baseline methods under TF and Zero-Shot (ZS) settings, respectively. On the most commonly used HM3D_v0.1 and the more challenging open vocabulary dataset HM3D_OVON, the SR improved by 8% and 6%, under ZS settings. Furthermore, on the MP3D and HM3D_OVON datasets, our method not only outperformed all TF methods but also surpassed all SFT methods, achieving comprehensive leadership in both SR (5% and 2%) and generalization. Additionally, we deployed the MerNav model on the humanoid robot and conducted experiments in the real world. The project address is: this https URL
>
---
#### [replaced 014] Echoes of Automation: The Increasing Use of LLMs in Newsmaking
- **分类: cs.CL; cs.AI**

- **简介: 论文研究生成式AI在新闻制作中的应用，分析40,000篇新闻内容，探讨其对新闻风格和质量的影响。任务属于AI与媒体交叉研究，解决AI对新闻真实性及风格影响的问题。**

- **链接: [https://arxiv.org/pdf/2508.06445](https://arxiv.org/pdf/2508.06445)**

> **作者:** Abolfazl Ansari; Delvin Ce Zhang; Nafis Irtiza Tripto; Dongwon Lee
>
> **备注:** To appear in the SBP-BRiMS 2025
>
> **摘要:** The rapid rise of Generative AI (GenAI), particularly LLMs, poses concerns for journalistic integrity and authorship. This study examines AI-generated content across over 40,000 news articles from major, local, and college news media, in various media formats. Using three advanced AI-text detectors (e.g., Binoculars, Fast-Detect GPT, and GPTZero), we find substantial increase of GenAI use in recent years, especially in local and college news. Sentence-level analysis reveals LLMs are often used in the introduction of news, while conclusions usually written manually. Linguistic analysis shows GenAI boosts word richness and readability but lowers formality, leading to more uniform writing styles, particularly in local media.
>
---
#### [replaced 015] VisText-Mosquito: A Unified Multimodal Dataset for Visual Detection, Segmentation, and Textual Explanation on Mosquito Breeding Sites
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出VisText-Mosquito数据集，用于蚊虫滋生地的视觉检测、分割和文本解释，解决蚊媒疾病预防问题。**

- **链接: [https://arxiv.org/pdf/2506.14629](https://arxiv.org/pdf/2506.14629)**

> **作者:** Md. Adnanul Islam; Md. Faiyaz Abdullah Sayeedi; Md. Asaduzzaman Shuvo; Shahanur Rahman Bappy; Md Asiful Islam; Swakkhar Shatabda
>
> **备注:** Accepted at CVPRW 2026
>
> **摘要:** Mosquito-borne diseases pose a major global health risk, requiring early detection and proactive control of breeding sites to prevent outbreaks. In this paper, we present VisText-Mosquito, a multimodal dataset that integrates visual and textual data to support automated detection, segmentation, and explanation for mosquito breeding site analysis. The dataset includes 1,828 annotated images for object detection, 142 images for water surface segmentation, and natural language explanation texts linked to each image. The YOLOv9s model achieves the highest precision of 0.92926 and mAP@50 of 0.92891 for object detection, while YOLOv11n-Seg reaches a segmentation precision of 0.91587 and mAP@50 of 0.79795. For textual explanation generation, we tested a range of large vision-language models (LVLMs) in both zero-shot and few-shot settings. Our fine-tuned Mosquito-LLaMA3-8B model achieved the best results, with a final loss of 0.0028, a BLEU score of 54.7, BERTScore of 0.91, and ROUGE-L of 0.85. This dataset and model framework emphasize the theme "Prevention is Better than Cure", showcasing how AI-based detection can proactively address mosquito-borne disease risks. The dataset and implementation code are publicly available at GitHub: this https URL
>
---
#### [replaced 016] MCGA: A Multi-task Classical Chinese Literary Genre Audio Corpus
- **分类: cs.CL**

- **简介: 该论文提出MCGA，一个119小时的中文古典文学音频语料库，涵盖6项任务，旨在推动多模态大模型在该领域的研究与应用。**

- **链接: [https://arxiv.org/pdf/2601.09270](https://arxiv.org/pdf/2601.09270)**

> **作者:** Yexing Du; Kaiyuan Liu; Bihe Zhang; Youcheng Pan; Bo Yang; Liangyu Huo; Xiyuan Zhang; Jian Xie; Daojing He; Yang Xiang; Ming Liu; Bing Qin
>
> **备注:** Accepted in ACL 2026 (Findings)
>
> **摘要:** With the rapid advancement of Multimodal Large Language Models (MLLMs), their potential has gained significant attention in Chinese Classical Studies (CCS). While existing research primarily focuses on text and visual modalities, the audio corpus within this domain remains largely underexplored. To bridge this gap, we introduce the Multi-task Classical Chinese Literary Genre Audio Corpus (MCGA), a 119-hour corpus comprising 22,000 audio samples. It encompasses a diverse range of literary genres across six tasks: Automatic Speech Recognition (ASR), Speech-to-Text Translation (S2TT), Speech Emotion Captioning (SEC), Spoken Question Answering (SQA), Speech Understanding (SU), and Speech Reasoning (SR). Through the evaluation of ten MLLMs, our experimental results demonstrate that current MLLMs still face substantial challenges on the MCGA test set. Furthermore, we introduce a domain-specific metric for SEC and a metric to measure the consistency between speech and text capabilities. We release MCGA to the public to facilitate the development of more robust MLLMs. MCGA Corpus: this https URL
>
---
#### [replaced 017] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations
- **分类: cs.CL**

- **简介: 该论文属于模型解释性研究，旨在解决LLM答案与解释不一致的问题。通过构建基准数据集，分析特征重要性差异，并改进对齐方法。**

- **链接: [https://arxiv.org/pdf/2506.07523](https://arxiv.org/pdf/2506.07523)**

> **作者:** Sahar Admoni; Ofra Amir; Assaf Hallak; Yftah Ziser
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** Large language models (LLMs) seem to offer an easy path to interpretability: just ask them to explain their answers. Yet the features driving an answer often differ from those emphasized in its explanation, meaning post-hoc rationales can misrepresent what actually shaped the model's output. We quantify this gap by comparing the feature-importance distributions of answers and their explanations. Prior analyses reveal such discrepancies, but large-scale study has been limited by the high computational cost of attribution methods. To address this, we introduce the Post-hoc Self-Consistency Bank (PSCB), a large-scale benchmark linking model decisions with diverse explanations and attribution vectors across datasets, methods, and model families. Using PSCB, we find that Spearman rank correlation provides a more reliable signal of alignment than cosine similarity. Building on this insight, we apply Direct Preference Optimization (DPO) to attribution-based preference data, improving alignment without degrading task accuracy, and show that standard supervised fine-tuning on the same data fails to achieve comparable gains. These improvements generalize robustly across domains, paving the way toward scalable and faithful alignment between LLM decisions and their natural language explanations.
>
---
#### [replaced 018] Doc-PP: Document Policy Preservation Benchmark for Large Vision-Language Models
- **分类: cs.CL**

- **简介: 该论文属于文档问答任务，解决LVLM在遵循用户政策时的信息泄露问题。构建Doc-PP基准，提出DVA框架以提升安全性和合规性。**

- **链接: [https://arxiv.org/pdf/2601.03926](https://arxiv.org/pdf/2601.03926)**

> **作者:** Haeun Jang; Hwan Chang; Hwanhee Lee
>
> **备注:** ACL 2026 Findings
>
> **摘要:** The deployment of Large Vision-Language Models (LVLMs) for real-world document question answering is often constrained by dynamic, user-defined policies that dictate information disclosure based on context. While ensuring adherence to these explicit constraints is critical, existing safety research primarily focuses on implicit social norms or text-only settings, overlooking the complexities of multimodal documents. In this paper, we introduce Doc-PP (Document Policy Preservation Benchmark), a novel benchmark constructed from real-world reports requiring reasoning across heterogeneous visual and textual elements under strict non-disclosure policies. Our evaluation highlights a systemic Reasoning-Induced Safety Gap: models frequently leak sensitive information when answers must be inferred through complex synthesis or aggregated across modalities, effectively circumventing existing safety constraints. Furthermore, we identify that providing extracted text improves perception but inadvertently facilitates leakage. To address these vulnerabilities, we propose DVA (Decompose-Verify-Aggregation), a structural inference framework that decouples reasoning from policy verification. Experimental results demonstrate that DVA significantly outperforms standard prompting defenses, offering a robust baseline for policy-compliant document understanding
>
---
#### [replaced 019] The Poisoned Apple Effect: Strategic Manipulation of Mediated Markets via Technology Expansion of AI Agents
- **分类: cs.GT; cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于机制设计任务，研究AI技术扩展对市场策略的影响，解决技术操纵市场设计的问题，通过分析博弈论模型揭示“毒苹果”效应。**

- **链接: [https://arxiv.org/pdf/2601.11496](https://arxiv.org/pdf/2601.11496)**

> **作者:** Eilam Shapira; Roi Reichart; Moshe Tennenholtz
>
> **摘要:** The integration of AI agents into economic markets fundamentally alters the landscape of strategic interaction. We investigate the economic implications of expanding the set of available technologies in three canonical game-theoretic settings: bargaining (resource division), negotiation (asymmetric information trade), and persuasion (strategic information transmission). We find that simply increasing the choice of AI delegates can drastically shift equilibrium payoffs and regulatory outcomes, often creating incentives for regulators to proactively develop and release technologies. Conversely, we identify a strategic phenomenon termed the "Poisoned Apple" effect: an agent may release a new technology, which neither they nor their opponent ultimately uses, solely to manipulate the regulator's choice of market design in their favor. This strategic release improves the releaser's welfare at the expense of their opponent and the regulator's fairness objectives. Our findings demonstrate that static regulatory frameworks are vulnerable to manipulation via technology expansion, necessitating dynamic market designs that adapt to the evolving landscape of AI capabilities.
>
---
#### [replaced 020] BadGraph: A Backdoor Attack Against Latent Diffusion Model for Text-Guided Graph Generation
- **分类: cs.LG; cs.CL; q-bio.BM**

- **简介: 该论文属于文本引导图生成任务，研究如何通过后门攻击破坏模型安全性。工作包括提出BadGraph方法，利用文本触发器植入后门，实现可控子图生成，同时保持正常性能。**

- **链接: [https://arxiv.org/pdf/2510.20792](https://arxiv.org/pdf/2510.20792)**

> **作者:** Liang Ye; Shengqin Chen; Jiazhu Dai
>
> **摘要:** The rapid progress of graph generation has raised new security concerns, particularly regarding backdoor vulnerabilities. While prior work has explored backdoor attacks in image diffusion and unconditional graph generation, conditional, especially text-guided graph generation remains largely unexamined. This paper proposes BadGraph, a backdoor attack method against latent diffusion models for text-guided graph generation. BadGraph leverages textual triggers to poison training data, covertly implanting backdoors that induce attacker-specified subgraphs during inference when triggers appear, while preserving normal performance on clean inputs. Extensive experiments on four benchmark datasets (PubChem, ChEBI-20, PCDes, MoMu) demonstrate the effectiveness and stealth of the attack: less than 10% poisoning rate can achieves 50% attack success rate, while 24% suffices for over 80% success rate, with negligible performance degradation on benign samples. Ablation studies further reveal that the backdoor is implanted during VAE and diffusion training rather than pretraining. These findings reveal the security vulnerabilities in latent diffusion models of text-guided graph generation, highlight the serious risks in models' applications such as drug discovery and underscore the need for robust defenses against the backdoor attack in such diffusion models.
>
---
#### [replaced 021] LaMI: Augmenting Large Language Models via Late Multi-Image Fusion
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出LaMI方法，通过晚融合多图像提升大语言模型的常识推理能力，解决视觉与文本结合任务中的性能不足问题。**

- **链接: [https://arxiv.org/pdf/2406.13621](https://arxiv.org/pdf/2406.13621)**

> **作者:** Guy Yariv; Idan Schwartz; Yossi Adi; Sagie Benaim
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** Commonsense reasoning often requires both textual and visual knowledge, yet Large Language Models (LLMs) trained solely on text lack visual grounding (e.g., "what color is an emperor penguin's belly?"). Visual Language Models (VLMs) perform better on visually grounded tasks but face two limitations: (i) often reduced performance on text-only commonsense reasoning compared to text-trained LLMs, and (ii) adapting newly released LLMs to vision input typically requires costly multimodal training. An alternative augments LLMs with test-time visual signals, improving visual commonsense without harming textual reasoning, but prior designs often rely on early fusion and a single image, which can be suboptimal. We propose a late multi-image fusion method: multiple images are generated from the text prompt with a lightweight parallel sampling, and their prediction probabilities are combined with those of a text-only LLM through a late-fusion layer that integrates projected visual features just before the final prediction. Across visual commonsense and NLP benchmarks, our method significantly outperforms augmented LLMs on visual reasoning, matches VLMs on vision-based tasks, and, when applied to strong LLMs such as LLaMA 3, also improves NLP performance while adding only modest test-time overhead. Project page is available at: this https URL.
>
---
#### [replaced 022] Measuring What Matters!! Assessing Therapeutic Principles in Mental-Health Conversation
- **分类: cs.CL**

- **简介: 该论文属于AI心理健康评估任务，旨在解决AI对话系统在遵循治疗原则方面的评价问题。工作包括构建FAITH-M基准和提出CARE框架，提升评估效果。**

- **链接: [https://arxiv.org/pdf/2604.05795](https://arxiv.org/pdf/2604.05795)**

> **作者:** Abdullah Mazhar; Het Riteshkumar Shah; Aseem Srivastava; Smriti Joshi; Md Shad Akhtar
>
> **备注:** Accepted at ACL 2026 (Main)
>
> **摘要:** The increasing use of large language models in mental health applications calls for principled evaluation frameworks that assess alignment with psychotherapeutic best practices beyond surface-level fluency. While recent systems exhibit conversational competence, they lack structured mechanisms to evaluate adherence to core therapeutic principles. In this paper, we study the problem of evaluating AI-generated therapist-like responses for clinically grounded appropriateness and effectiveness. We assess each therapists utterance along six therapeutic principles: non-judgmental acceptance, warmth, respect for autonomy, active listening, reflective understanding, and situational appropriateness using a fine-grained ordinal scale. We introduce FAITH-M, a benchmark annotated with expert-assigned ordinal ratings, and propose CARE, a multi-stage evaluation framework that integrates intra-dialogue context, contrastive exemplar retrieval, and knowledge-distilled chain-of-thought reasoning. Experiments show that CARE achieves an F-1 score of 63.34 versus the strong baseline Qwen3 F-1 score of 38.56 which is a 64.26 improvement, which also serves as its backbone, indicating that gains arise from structured reasoning and contextual modeling rather than backbone capacity alone. Expert assessment and external dataset evaluations further demonstrate robustness under domain shift, while highlighting challenges in modelling implicit clinical nuance. Overall, CARE provides a clinically grounded framework for evaluating therapeutic fidelity in AI mental health systems.
>
---
#### [replaced 023] SimBench: Benchmarking the Ability of Large Language Models to Simulate Human Behaviors
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 该论文提出SimBench，用于评估大语言模型模拟人类行为的能力。任务是衡量模拟真实性，解决现有评估不统一的问题，通过整合多个数据集进行标准化测试。**

- **链接: [https://arxiv.org/pdf/2510.17516](https://arxiv.org/pdf/2510.17516)**

> **作者:** Tiancheng Hu; Joachim Baumann; Lorenzo Lupo; Nigel Collier; Dirk Hovy; Paul Röttger
>
> **备注:** Accepted at ICLR 2026. Project Website: this http URL Data: this https URL
>
> **摘要:** Large language model (LLM) simulations of human behavior have the potential to revolutionize the social and behavioral sciences, if and only if they faithfully reflect real human behaviors. Current evaluations of simulation fidelity are fragmented, based on bespoke tasks and metrics, creating a patchwork of incomparable results. To address this, we introduce SimBench, the first large-scale, standardized benchmark for a robust, reproducible science of LLM simulation. By unifying 20 diverse datasets covering tasks from moral decision-making to economic choice across a large global participant pool, SimBench provides the necessary foundation to ask fundamental questions about when, how, and why LLM simulations succeed or fail. We show that the best LLMs today achieve meaningful but modest simulation fidelity (score: 40.80/100), with performance scaling log-linearly with model size but not with increased inference-time compute. We discover an alignment-simulation tradeoff: instruction tuning improves performance on low-entropy (consensus) questions but degrades it on high-entropy (diverse) ones. Models particularly struggle when simulating specific demographic groups. Finally, we demonstrate that simulation ability correlates most strongly with knowledge-intensive reasoning (MMLU-Pro, r = 0.939). By making progress measurable, we aim to accelerate the development of more faithful LLM simulators.
>
---
#### [replaced 024] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于长文本语音问答任务，旨在解决长音频处理困难的问题。提出CLSR模型，通过对比学习提取相关语音片段，提升问答效果。**

- **链接: [https://arxiv.org/pdf/2511.09282](https://arxiv.org/pdf/2511.09282)**

> **作者:** Jiliang Hu; Zuchao Li; Baoyuan Qi; Liu Guoming; Ping Wang
>
> **备注:** 12 pages, 7 figures, accepted by AAAI 2026
>
> **摘要:** Significant progress has been made in spoken question answering (SQA) in recent years. However, many existing methods, including large audio language models, struggle with processing long audio. Follow the success of retrieval augmented generation, a speech-related retriever shows promising in help preprocessing long-form speech. But the performance of existing speech-related retrievers is lacking. To address this challenge, we propose CLSR, an end-to-end contrastive language-speech retriever that efficiently extracts question-relevant segments from long audio recordings for downstream SQA task. Unlike conventional speech-text contrastive models, CLSR incorporates an intermediate step that converts acoustic features into text-like representations prior to alignment, thereby more effectively bridging the gap between modalities. Experimental results across four cross-modal retrieval datasets demonstrate that CLSR surpasses both end-to-end speech related retrievers and pipeline approaches combining speech recognition with text retrieval, providing a robust foundation for advancing practical long-form SQA applications.
>
---
#### [replaced 025] SafeConstellations: Mitigating Over-Refusals in LLMs Through Task-Aware Representation Steering
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs的过度拒绝问题。通过分析嵌入空间中的轨迹模式，提出SafeConstellations方法，在不损害实用性的前提下显著降低过度拒绝率。**

- **链接: [https://arxiv.org/pdf/2508.11290](https://arxiv.org/pdf/2508.11290)**

> **作者:** Utsav Maskey; Sumit Yadav; Mark Dras; Usman Naseem
>
> **备注:** ACL'26 Main
>
> **摘要:** LLMs increasingly exhibit over-refusal behavior, where safety mechanisms cause models to reject benign instructions that seemingly resemble harmful content. This phenomenon diminishes utility in production applications that repeatedly rely on common prompt templates or applications that frequently rely on LLMs for specific tasks (e.g. sentiment analysis, language translation). Through extensive evaluation, we demonstrate that LLMs persist in refusing inputs containing harmful content, even when they are reframed with tasks that have benign intent. Our mechanistic analysis reveals that LLMs follow distinct "constellation" patterns in embedding space as representations traverse layers, with each NLP task maintaining consistent trajectories that shift predictably between refusal and non-refusal cases. We introduce SafeConstellations, an inference-time trajectory-shifting approach that tracks task-specific trajectory patterns and guides representations toward non-refusal pathways. By selectively guiding model behavior only on tasks prone to over-refusal, our method reduces over-refusal rates by up to 73% with minimal impact on utility -- offering a principled and conditional approach to mitigating over-refusals.
>
---
#### [replaced 026] ChatCLIDS: Simulating Persuasive AI Dialogues to Promote Closed-Loop Insulin Adoption in Type 1 Diabetes Care
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于健康行为改变任务，旨在提升1型糖尿病患者对闭环胰岛素系统的采用。提出ChatCLIDS框架，模拟AI对话以克服行为障碍。**

- **链接: [https://arxiv.org/pdf/2509.00891](https://arxiv.org/pdf/2509.00891)**

> **作者:** Zonghai Yao; Talha Chafekar; Junda Wang; Shuo Han; Feiyun Ouyang; Junhui Qian; Lingxi Li; Hong Yu
>
> **备注:** Equal contribution for the first two authors. To appear in AAAI 2026 Special Track on AI for Social Impact
>
> **摘要:** Real-world adoption of closed-loop insulin delivery systems (CLIDS) in type 1 diabetes remains low, driven not by technical failure, but by diverse behavioral, psychosocial, and social barriers. We introduce ChatCLIDS, the first benchmark to rigorously evaluate LLM-driven persuasive dialogue for health behavior change. Our framework features a library of expert-validated virtual patients, each with clinically grounded, heterogeneous profiles and realistic adoption barriers, and simulates multi-turn interactions with nurse agents equipped with a diverse set of evidence-based persuasive strategies. ChatCLIDS uniquely supports longitudinal counseling and adversarial social influence scenarios, enabling robust, multi-dimensional evaluation. Our findings reveal that while larger and more reflective LLMs adapt strategies over time, all models struggle to overcome resistance, especially under realistic social pressure. These results highlight critical limitations of current LLMs for behavior change, and offer a high-fidelity, scalable testbed for advancing trustworthy persuasive AI in healthcare and beyond.
>
---
#### [replaced 027] Beyond Black-Box Interventions: Latent Probing for Faithful Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，旨在解决RAG系统缺乏上下文忠实性的问题。通过分析模型内部表示，提出ProbeRAG框架提升生成准确性与忠实度。**

- **链接: [https://arxiv.org/pdf/2510.12460](https://arxiv.org/pdf/2510.12460)**

> **作者:** Linfeng Gao; Qinggang Zhang; Baolong Bi; Bo Zeng; Zheng Yuan; Zerui Chen; Zhimin Wei; Shenghua Liu; Linlong Xu; Longyue Wang; Weihua Luo; Jinsong Su
>
> **备注:** ACL 2026 Findings; Code is available at this https URL
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems often fail to maintain contextual faithfulness, generating responses that conflict with the provided context or fail to fully leverage the provided evidence. Existing methods attempt to improve faithfulness through external interventions, such as specialized prompting, decoding-based calibration, or preference optimization. However, since these approaches treat the LLM as a black box, they lack a reliable mechanism to assess when and why knowledge conflicts occur. Consequently, they tend to be brittle, data-intensive, and agnostic to the model's internal reasoning process. In this paper, we move beyond black-box interventions to analyze the model's internal reasoning process. We discover that conflicting and aligned knowledge states are linearly separable in the model's latent space, and contextual noise systematically increases the entropy of these representations. Based on these findings, we propose ProbeRAG, a novel framework for faithful RAG that operates in three stages: (i) fine-grained knowledge pruning to filter irrelevant context, (ii) latent conflict probing to identify hard conflicts in the model's latent space, and (iii) conflict-aware attention to modulate attention heads toward faithful context integration. Extensive experiments demonstrate that ProbeRAG substantially improves both accuracy and contextual faithfulness. The related resources are available at this https URL.
>
---
#### [replaced 028] Interactive ASR: Towards Human-Like Interaction and Semantic Coherence Evaluation for Agentic Speech Recognition
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决传统评估指标不足和交互纠错缺失的问题。提出基于大模型的语义评估和交互框架，提升识别的语义准确性和交互能力。**

- **链接: [https://arxiv.org/pdf/2604.09121](https://arxiv.org/pdf/2604.09121)**

> **作者:** Peng Wang; Yanqiao Zhu; Zixuan Jiang; Qinyuan Chen; Xingjian Zhao; Xipeng Qiu; Wupeng Wang; Zhifu Gao; Xiangang Li; Kai Yu; Xie Chen
>
> **摘要:** Recent years have witnessed remarkable progress in automatic speech recognition (ASR), driven by advances in model architectures and large-scale training data. However, two important aspects remain underexplored. First, Word Error Rate (WER), the dominant evaluation metric for decades, treats all words equally and often fails to reflect the semantic correctness of an utterance at the sentence level. Second, interactive correction-an essential component of human communication-has rarely been systematically studied in ASR research. In this paper, we integrate these two perspectives under an agentic framework for interactive ASR. We propose leveraging LLM-as-a-Judge as a semantic-aware evaluation metric to assess recognition quality beyond token-level accuracy. Furthermore, we design an LLM-driven agent framework to simulate human-like multi-turn interaction, enabling iterative refinement of recognition outputs through semantic feedback. Extensive experiments are conducted on standard benchmarks, including GigaSpeech (English), WenetSpeech (Chinese), the ASRU 2019 code-switching test set. Both objective and subjective evaluations demonstrate the effectiveness of the proposed framework in improving semantic fidelity and interactive correction capability. We will release the code to facilitate future research in interactive and agentic ASR.
>
---
#### [replaced 029] Do Neurons Dream of Primitive Operators? Wake-Sleep Compression Rediscovers Schank's Event Semantics
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于事件语义解析任务，旨在自动发现人类事件的原始操作符。通过压缩压力，系统从事件数据中学习到与Schank理论相符的原始操作符，并超越其覆盖范围。**

- **链接: [https://arxiv.org/pdf/2603.25975](https://arxiv.org/pdf/2603.25975)**

> **作者:** Peter Balogh
>
> **摘要:** We show that they do. Roger Schank's conceptual dependency theory proposed that all human events decompose into primitive operations -- ATRANS (transfer of possession), PTRANS (physical movement), MTRANS (information transfer), and others -- hand-coded from linguistic intuition. We ask: can the same primitives be discovered automatically through compression pressure alone? We adapt DreamCoder's wake-sleep library learning to event state transformations. Given events as before/after world-state pairs, the system searches for operator compositions explaining each event (wake), then extracts recurring patterns as library entries under Minimum Description Length (sleep). Starting from four generic primitives, it discovers operators mapping to Schank's core: MOVE_PROP_has = ATRANS, CHANGE_location = PTRANS, SET_knows = MTRANS, SET_consumed = INGEST, plus compound operators (e.g., "mail" = ATRANS composed with PTRANS) and novel emotional-state operators absent from Schank's taxonomy. We validate on synthetic events, ATOMIC (Sap et al., 2019), and GLUCOSE (Mostafazadeh et al., 2020). On synthetic data, the discovered library achieves MDL within 4% of Schank's hand-coded primitives at 100% coverage (vs. Schank's 81%). On ATOMIC, Schank covers only 10%; on GLUCOSE, 31%. The discovered library covers 100% of both, dominated by mental/emotional operators -- CHANGE_wants (20%), CHANGE_feels (18%), CHANGE_is (18%) -- none in Schank's original taxonomy. Libraries discovered from one corpus transfer to the other with under 1 bit/event degradation despite different annotation schemes and domains, suggesting the operators are information-theoretically determined structure, not dataset artifacts.
>
---
#### [replaced 030] LiveCLKTBench: Towards Reliable Evaluation of Cross-Lingual Knowledge Transfer in Multilingual LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言大模型的跨语言知识迁移评估任务，旨在解决如何可靠衡量模型在不同语言间的知识转移能力。工作包括构建LiveCLKTBench基准，通过生成事实性问题评估跨语言迁移效果。**

- **链接: [https://arxiv.org/pdf/2511.14774](https://arxiv.org/pdf/2511.14774)**

> **作者:** Pei-Fu Guo; Yun-Da Tsai; Chun-Chia Hsu; Kai-Xin Chen; Ya-An Tsai; Kai-Wei Chang; Nanyun Peng; Mi-Yen Yeh; Shou-De Lin
>
> **摘要:** Evaluating cross-lingual knowledge transfer in large language models is challenging, as correct answers in a target language may arise either from genuine transfer or from prior exposure during pre-training. We present LiveCLKTBench, an automated generation pipeline specifically designed to isolate and measure cross-lingual knowledge transfer. Our pipeline identifies self-contained, time-sensitive knowledge entities from real-world domains, filters them based on temporal occurrence, and verifies them against the model's knowledge. The documents of these valid entities are then used to generate factual questions, which are translated into multiple languages to evaluate transferability across linguistic boundaries. Using LiveCLKTBench, we evaluate several LLMs across five languages and observe that cross-lingual transfer is strongly influenced by linguistic distance and often asymmetric across language directions. While larger models improve transfer, the gains diminish with scale and vary across domains. These findings provide new insights into multilingual transfer and demonstrate the value of LiveCLKTBench as a reliable benchmark for future research.
>
---
#### [replaced 031] MM-LIMA: Less Is More for Alignment in Multi-Modal Datasets
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态语言模型对齐任务，旨在用少量高质量数据提升模型性能。工作包括提出数据质量评估指标和自动筛选方法，使模型在200条数据上表现优于基准。**

- **链接: [https://arxiv.org/pdf/2308.12067](https://arxiv.org/pdf/2308.12067)**

> **作者:** Lai Wei; Xiaozhe Li; Zihao Jiang; Weiran Huang; Lichao Sun
>
> **备注:** Published at Artificial Intelligence for Engineering
>
> **摘要:** Multimodal large language models are typically trained in two stages: first pre-training on image-text pairs, and then fine-tuning using supervised vision-language instruction data. Recent studies have shown that large language models can achieve satisfactory results even with a limited amount of high-quality instruction-following data. In this paper, we introduce MM-LIMA, which is fine-tuned on a small dataset comprising only 200 examples, amounting to approximately 6% of the instruction-following data used in the alignment dataset for MiniGPT-4. To achieve this, we first propose several metrics to access the quality of multimodal instruction data. Based on these metrics, we present an effective and trainable data selector to automatically identify and filter low-quality vision-language data. By employing this method, MM-LIMA outperforms the original MiniGPT-4 on various evaluations. Overall, our findings demonstrate that less but high-quality instruction tuning data is efficient in enabling multimodal large language models to generate better output. Our code is available at this https URL.
>
---
#### [replaced 032] Thought Branches: Interpreting LLM Reasoning Requires Resampling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型解释任务，旨在解决单一推理链不足以理解模型决策的问题。通过重采样方法分析推理分布，评估因果影响和干预效果。**

- **链接: [https://arxiv.org/pdf/2510.27484](https://arxiv.org/pdf/2510.27484)**

> **作者:** Uzay Macar; Paul C. Bogdan; Senthooran Rajamanoharan; Neel Nanda
>
> **备注:** Uzay Macar and Paul C. Bogdan contributed equally to this work, and their listed order was determined by coinflip
>
> **摘要:** Most work interpreting reasoning models studies only a single chain-of-thought (CoT), yet these models define distributions over many possible CoTs. We argue that studying a single sample is inadequate for understanding causal influence and the underlying computation. Though fully specifying this distribution is intractable, we can measure a partial CoT's impact by resampling only the subsequent text. We present case studies using resampling to investigate model decisions. First, when a model states a reason for its action, does that reason actually cause the action? In "agentic misalignment" scenarios, we find that self-preservation sentences have small causal impact, suggesting they do not meaningfully drive blackmail. Second, are artificial edits to CoT sufficient for steering reasoning? Resampling and selecting a completion with the desired property is a principled on-policy alternative. We find that off-policy interventions yield small and unstable effects compared to resampling in decision-making tasks. Third, how do we understand the effect of removing a reasoning step when the model may repeat it post-edit? We introduce a resilience metric that repeatedly resamples to prevent similar content from reappearing downstream. Critical planning statements resist removal but have large effects when eliminated. Fourth, since CoT is sometimes "unfaithful", can our methods teach us anything in these settings? Adapting causal mediation analysis, we find that hints that causally affect the output without being explicitly mentioned exert a subtle and cumulative influence on the CoT that persists even if the hint is removed. Overall, studying distributions via resampling enables reliable causal analysis, clearer narratives of model reasoning, and principled CoT interventions.
>
---
#### [replaced 033] StyleBench: Evaluating thinking styles in Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，研究大模型的推理风格。旨在解决结构化推理在不同场景下的有效性问题，通过实验比较多种推理方式，分析其效率与性能。**

- **链接: [https://arxiv.org/pdf/2509.20868](https://arxiv.org/pdf/2509.20868)**

> **作者:** Junyu Guo; Shangding Gu; Ming Jin; Costas Spanos; Javad Lavaei
>
> **摘要:** Structured reasoning can improve the inference performance of large language models (LLMs), but it also introduces computational cost and control constraints. When additional reasoning structure helps, and when it instead reduces efficiency or robustness, remains poorly understood. We propose StyleBench, where we study reasoning structure as a capacity-constrained design choice rather than a fixed inference recipe. We evaluate five representative reasoning styles: Chain-of-Thought, Tree-of-Thought, Algorithm-of-Thought, Sketch-of-Thought, and Chain-of-Draft across five reasoning tasks and 15 open-source LLMs ranging from 270M to 120B parameters. We find that greater structural complexity improves accuracy only in limited regimes defined by task demands and model capacity. Search-based styles help on open-ended combinatorial problems but fail on smaller models, while concise styles achieve large efficiency gains on structured tasks without sacrificing performance. We also identify systematic failure modes in smaller models, including premature guessing and weak adherence to reasoning-control instructions. To study adaptive reasoning control, we further compare supervised and reinforcement-based strategy selection on Qwen-7B-Instruct. Supervised fine-tuning collapses to shallow style preferences, whereas GRPO learns stronger adaptive control and improves downstream performance. Together, these results clarify when structured reasoning is useful, when it is wasteful, and why learning to choose a reasoning strategy is itself a challenging inference problem, we open source the benchmark in this https URL.
>
---
#### [replaced 034] Who Gets Which Message? Auditing Demographic Bias in LLM-Generated Targeted Text
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 该论文属于自然语言处理中的公平性研究，旨在检测LLM在生成定向文本时的种族偏见。通过实验分析不同模型在不同情境下的表现，揭示了年龄和性别相关的语言差异。**

- **链接: [https://arxiv.org/pdf/2601.17172](https://arxiv.org/pdf/2601.17172)**

> **作者:** Tunazzina Islam
>
> **备注:** Accepted at Findings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)
>
> **摘要:** Large language models (LLMs) are increasingly capable of generating personalized, persuasive text at scale, raising new questions about bias and fairness in automated communication. This paper presents the first systematic analysis of how LLMs behave when tasked with demographic-conditioned targeted messaging. We introduce a controlled evaluation framework using three leading models: GPT-4o, Llama-3.3, and Mistral-Large-2.1, across two generation settings: Standalone Generation, which isolates intrinsic demographic effects, and Context-Rich Generation, which incorporates thematic and regional context to emulate realistic targeting. We evaluate generated messages along three dimensions: lexical content, language style, and persuasive framing. We instantiate this framework on climate communication and find consistent age- and gender-based asymmetries across models: male- and youth-targeted messages emphasize agency, innovation, and assertiveness, while female- and senior-targeted messages stress warmth, care, and tradition. Contextual prompts systematically amplify these disparities, with persuasion scores significantly higher for messages tailored to younger or male audiences. Our findings demonstrate how demographic stereotypes can surface and intensify in LLM-generated targeted communication, underscoring the need for bias-aware generation pipelines and transparent auditing frameworks that explicitly account for demographic conditioning in socially sensitive applications.
>
---
#### [replaced 035] RedNote-Vibe: A Dataset for Capturing Temporal Dynamics of AI-Generated Text in Lifestyle Social Media
- **分类: cs.CL**

- **简介: 该论文提出RedNote-Vibe数据集，用于研究社交媒体中AI生成文本的动态变化。任务是检测AI生成内容，工作包括构建数据集和提出PLAD框架，以分析人类与AI内容的差异及用户策略影响。**

- **链接: [https://arxiv.org/pdf/2509.22055](https://arxiv.org/pdf/2509.22055)**

> **作者:** Yudong Li; Yufei Sun; Peiru Yang; Yuhan Yao; Wanyue Li; Jiajun Zou; Haoyang Yang; Haotian Gan; Linlin Shen; Yongfeng Huang
>
> **摘要:** We introduce RedNote-Vibe, a dataset spanning five years (pre-LLM to July 2025) sourced from lifestyle platform RedNote (Xiaohongshu), capturing the temporal dynamics of content creation and is enriched with comprehensive engagement metrics. To address the detection challenge posed by RedNote-Vibe, we propose the \textbf{PsychoLinguistic AIGT Detection Framework (PLAD)}. Grounded in cognitive psychology, PLAD leverages deep psychological signatures for robust and interpretable detection. Our experiments demonstrate PLAD's superior performance and reveal insights into content dynamics: (1) human content continues to outperform AI in emotionally resonant domains; (2) AI content is more homogeneous and rarely produces breaking posts, however, this human-AI gap narrows for arousing higher-investment interactions; and (3) most interestingly, a small group of users who strategically utilize AI tools can achieve higher engagement outcomes. The dataset is available at this https URL
>
---
#### [replaced 036] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出OlymMATH基准，用于评估大语言模型的数学推理能力。针对现有基准不足，设计包含中英文问题和形式化验证的多维度评测体系，推动模型严谨推理研究。**

- **链接: [https://arxiv.org/pdf/2503.21380](https://arxiv.org/pdf/2503.21380)**

> **作者:** Haoxiang Sun; Yingqian Min; Zhipeng Chen; Wayne Xin Zhao; Ji-Rong Wen
>
> **备注:** Accepted by ACL 2026 Main
>
> **摘要:** The rapid advancement of large reasoning models has saturated existing math benchmarks, underscoring the urgent need for more challenging evaluation frameworks. To address this, we introduce OlymMATH, a rigorously curated, Olympiad-level math benchmark comprising 350 problems, each with parallel English and Chinese versions. OlymMATH is the first benchmark to unify dual evaluation paradigms within a single suite: (1) natural language evaluation through OlymMATH-EASY and OlymMATH-HARD, comprising 200 computational problems with numerical answers for objective rule-based assessment, and (2) formal verification through OlymMATH-LEAN, offering 150 problems formalized in Lean 4 for rigorous process-level evaluation. All problems are manually sourced from printed publications to minimize data contamination, verified by experts, and span four core domains. Extensive experiments reveal the benchmark's significant challenge, and our analysis also uncovers consistent performance gaps between languages and identifies cases where models employ heuristic "guessing" rather than rigorous reasoning. To further support community research, we release 582k+ reasoning trajectories, a visualization tool, and expert solutions at this https URL.
>
---
#### [replaced 037] MAESTRO: Meta-learning Adaptive Estimation of Scalarization Trade-offs for Reward Optimization
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出MAESTRO，解决开放领域中多目标奖励优化问题，通过动态调整奖励标量化提升大语言模型对齐效果。**

- **链接: [https://arxiv.org/pdf/2601.07208](https://arxiv.org/pdf/2601.07208)**

> **作者:** Yang Zhao; Hepeng Wang; Xiao Ding; Yangou Ouyang; Bibo Cai; Kai Xiong; Jinglong Gao; Zhouhao Sun; Li Du; Bing Qin; Ting Liu
>
> **备注:** ACL 2026 Main Conference
>
> **摘要:** Group-Relative Policy Optimization (GRPO) has emerged as an efficient paradigm for aligning Large Language Models (LLMs), yet its efficacy is primarily confined to domains with verifiable ground truths. Extending GRPO to open-domain settings remains a critical challenge, as unconstrained generation entails multi-faceted and often conflicting objectives - such as creativity versus factuality - where rigid, static reward scalarization is inherently suboptimal. To address this, we propose MAESTRO (Meta-learning Adaptive Estimation of Scalarization Trade-offs for Reward Optimization), which introduces a meta-cognitive orchestration layer that treats reward scalarization as a dynamic latent policy, leveraging the model's terminal hidden states as a semantic bottleneck to perceive task-specific priorities. We formulate this as a contextual bandit problem within a bi-level optimization framework, where a lightweight Conductor network co-evolves with the policy by utilizing group-relative advantages as a meta-reward signal. Across seven benchmarks, MAESTRO consistently outperforms single-reward and static multi-objective baselines, while preserving the efficiency advantages of GRPO, and in some settings even reducing redundant generation.
>
---
#### [replaced 038] Rethinking LLM Watermark Detection in Black-Box Settings: A Non-Intrusive Third-Party Framework
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于LLM水印检测任务，解决第三方无法验证水印的问题。提出TTP-Detect框架，实现非侵入式检测，提升治理可行性。**

- **链接: [https://arxiv.org/pdf/2603.14968](https://arxiv.org/pdf/2603.14968)**

> **作者:** Zhuoshang Wang; Yubing Ren; Yanan Cao; Fang Fang; Xiaoxue Li; Li Guo
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** While watermarking serves as a critical mechanism for LLM provenance, existing secret-key schemes tightly couple detection with injection, requiring access to keys or provider-side scheme-specific detectors for verification. This dependency creates a fundamental barrier for real-world governance, as independent auditing becomes impossible without compromising model security or relying on the opaque claims of service providers. To resolve this dilemma, we introduce TTP-Detect, a pioneering black-box framework designed for non-intrusive, third-party watermark verification. By decoupling detection from injection, TTP-Detect reframes verification as a relative hypothesis testing problem. It employs a proxy model to amplify watermark-relevant signals and a suite of complementary relative measurements to assess the alignment of the query text with watermarked distributions. Extensive experiments across representative watermarking schemes, datasets and models demonstrate that TTP-Detect achieves superior detection performance and robustness against diverse attacks.
>
---
#### [replaced 039] How Alignment Routes: Localizing, Scaling, and Controlling Policy Circuits in Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言模型中的策略路由机制，解决政策电路定位与控制问题。通过分析注意力门控和放大器头，揭示模型如何处理拒绝与安全指令，提出审计与干预方法。**

- **链接: [https://arxiv.org/pdf/2604.04385](https://arxiv.org/pdf/2604.04385)**

> **作者:** Gregory N. Frank
>
> **备注:** Code and data: this https URL
>
> **摘要:** This paper localizes the policy routing mechanism in alignment-trained language models. An intermediate-layer attention gate reads detected content and triggers deeper amplifier heads that boost the signal toward refusal. In smaller models the gate and amplifier are single heads; at larger scale they become bands of heads across adjacent layers. The gate contributes under 1% of output DLA, but interchange testing (p<0.001) and knockout cascade confirm it is causally necessary. Interchange screening at n>=120 detects the same motif in twelve models from six labs (2B to 72B), though specific heads differ by lab. Per-head ablation weakens up to 58x at 72B and misses gates that interchange identifies; interchange is the only reliable audit at scale. Modulating the detection-layer signal continuously controls policy from hard refusal through evasion to factual answering. On safety prompts the same intervention turns refusal into harmful guidance, showing the safety-trained capability is gated by routing rather than removed. Thresholds vary by topic and by input language, and the circuit relocates across generations within a family while behavioral benchmarks register no change. Routing is early-commitment: the gate commits at its own layer before deeper layers finish processing the input. Under an in-context substitution cipher, gate interchange necessity collapses 70 to 99% across three models and the model switches to puzzle-solving. Injecting the plaintext gate activation into the cipher forward pass restores 48% of refusals in Phi-4-mini, localizing the bypass to the routing interface. A second method, cipher contrast analysis, uses plain/cipher DLA differences to map the full cipher-sensitive routing circuit in O(3n) forward passes. Any encoding that defeats detection-layer pattern matching bypasses the policy regardless of whether deeper layers reconstruct the content.
>
---
#### [replaced 040] RISK: A Framework for GUI Agents in E-commerce Risk Management
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于GUI代理任务，旨在解决电商风险管理中复杂网页交互的自动化问题。提出RISK框架，包含数据集、基准和强化学习方法，提升多步骤交互的处理效果。**

- **链接: [https://arxiv.org/pdf/2509.21982](https://arxiv.org/pdf/2509.21982)**

> **作者:** Renqi Chen; Zeyin Tao; Jianming Guo; Jingzhe Zhu; Yiheng Peng; Qingqing Sun; Tianyi Zhang; Shuai Chen
>
> **备注:** Accepted by ACL 2026 Main Conference
>
> **摘要:** E-commerce risk management requires aggregating diverse, deeply embedded web data through multi-step, stateful interactions, which traditional scraping methods and most existing Graphical User Interface (GUI) agents cannot handle. These agents are typically limited to single-step tasks and lack the ability to manage dynamic, interactive content critical for effective risk assessment. To address this challenge, we introduce RISK, a novel framework designed to build and deploy GUI agents for this domain. RISK integrates three components: (1) RISK-Data, a dataset of 8,492 single-step and 2,386 multi-step interaction trajectories, collected through a high-fidelity browser framework and a meticulous data curation process; (2) RISK-Bench, a benchmark with 802 single-step and 320 multi-step trajectories across three difficulty levels for standardized evaluation; and (3) RISK-R1, a R1-style reinforcement fine-tuning framework considering four aspects: (i) Output Format Constraint, (ii) Single-step and (iii) Multi-step Level Reward, and (iv) Task Level Reweight. Experiments show that RISK-R1 achieves a 6.8% improvement in offline single-step and an 8.8% improvement in offline multi-step, using only 7.2% of the parameters of the SOTA baseline. Moreover, it attains a top task success rate of 70.5% in online evaluation. RISK provides a scalable, domain-specific solution for automating complex web interactions in e-commerce risk management. The code is available at this https URL.
>
---
#### [replaced 041] GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出GameplayQA，用于评估3D环境中智能体的感知与推理能力，解决多智能体行为理解问题，通过密集标注视频生成诊断问答对。**

- **链接: [https://arxiv.org/pdf/2603.24329](https://arxiv.org/pdf/2603.24329)**

> **作者:** Yunzhe Wang; Runhui Xu; Kexin Zheng; Tianyi Zhang; Jayavibhav Niranjan Kogundi; Soham Hans; Volkan Ustun
>
> **备注:** Accepted to the Annual Meeting of the Association for Computational Linguistics (ACL 2026)
>
> **摘要:** Multimodal LLMs are increasingly deployed as perceptual backbones for autonomous agents in 3D environments, from robotics to virtual worlds. These applications require agents to perceive rapid state changes, attribute actions to the correct entities, and reason about concurrent multi-agent behaviors from a first-person perspective, capabilities that existing benchmarks do not adequately evaluate. We introduce GameplayQA, a framework for evaluating agentic-centric perception and reasoning through video understanding. Specifically, we densely annotate multiplayer 3D gameplay videos at 1.22 labels/second, with time-synced, concurrent captions of states, actions, and events structured around a triadic system of Self, Other Agents, and the World, a natural decomposition for multi-agent environments. From these annotations, we refined 2.4K diagnostic QA pairs organized into three levels of cognitive complexity, accompanied by a structured distractor taxonomy that enables fine-grained analysis of where models hallucinate. Evaluation of frontier MLLMs reveals a substantial gap from human performance, with common failures in temporal and cross-video grounding, agent-role attribution, and handling the decision density of the game. We hope GameplayQA stimulates future research at the intersection of embodied AI, agentic perception, and world modeling.
>
---
#### [replaced 042] Tuning Language Models for Robust Prediction of Diverse User Behaviors
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于用户行为预测任务，解决深度学习模型难以捕捉长尾行为的问题。提出BehaviorLM方法，通过分阶段微调提升模型对罕见行为的预测能力。**

- **链接: [https://arxiv.org/pdf/2505.17682](https://arxiv.org/pdf/2505.17682)**

> **作者:** Fanjin Meng; Jingtao Ding; Jiahui Gong; Chen Yang; Hong Chen; Zuojian Wang; Haisheng Lu; Yong Li
>
> **摘要:** Predicting user behavior is essential for intelligent assistant services, yet deep learning models often struggle to capture long-tailed behaviors. Large language models (LLMs), with their pretraining on vast corpora containing rich behavioral knowledge, offer promise. However, existing fine-tuning approaches tend to overfit to frequent ``anchor'' behaviors, reducing their ability to predict less common ``tail'' behaviors. In this paper, we introduce BehaviorLM, a progressive fine-tuning approach that addresses this issue. In the first stage, LLMs are fine-tuned on anchor behaviors while preserving general behavioral knowledge. In the second stage, fine-tuning uses a balanced subset of all behaviors based on sample difficulty to improve tail behavior predictions without sacrificing anchor performance. Experimental results on two real-world datasets demonstrate that BehaviorLM robustly predicts both anchor and tail behaviors and effectively leverages LLM behavioral knowledge to master tail behavior prediction with few-shot examples.
>
---
#### [replaced 043] MemDLM: Memory-Enhanced DLM Training
- **分类: cs.CL**

- **简介: 该论文提出MemDLM，解决DLM在长文本中注意力稀释的问题，通过引入记忆机制提升模型性能。任务为改进扩散语言模型训练。**

- **链接: [https://arxiv.org/pdf/2603.22241](https://arxiv.org/pdf/2603.22241)**

> **作者:** Zehua Pei; Hui-Ling Zhen; Weizhe Lin; Sinno Jialin Pan; Yunhe Wang; Mingxuan Yuan; Bei Yu
>
> **摘要:** Diffusion Language Models (DLMs) offer attractive advantages over Auto-Regressive (AR) models, such as full-attention parallel decoding and flexible generation. However, standard DLM training uses a static, single-step masked prediction objective that never exposes the model to the progressive denoising dynamics of inference, and forces all contextual information to be maintained purely through token-space attention, which becomes increasingly diluted as context length grows. We propose MemDLM (Memory-Enhanced DLM), which introduces a second memory channel by embedding a simulated denoising trajectory into training via Bi-level Optimization. An inner loop updates a set of fast weights, forming a Parametric Memory that captures the local trajectory experience, while an outer loop updates the base model conditioned on this memory. By offloading part of the memorization burden from token-space attention to parameter space, MemDLM yields faster convergence, stronger long-context representations, and lower training loss, even when the fast weights are discarded at inference time. Re-enabling the inner loop at inference provides an additional prompt-specific adaptation effect, where the Parametric Memory acts as an emergent in-weight retrieval mechanism on challenging Needle-in-a-Haystack tasks. Code: this https URL.
>
---
#### [replaced 044] EDUMATH: Generating Standards-aligned Educational Math Word Problems
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于教育AI任务，旨在解决教师定制数学题困难的问题。通过LLM生成符合标准的数学题，并构建数据集进行训练与评估。**

- **链接: [https://arxiv.org/pdf/2510.06965](https://arxiv.org/pdf/2510.06965)**

> **作者:** Bryan R. Christ; Penelope Molitz; Beau LeBlond; Zachary Gottesman; Jonathan Kropko; Thomas Hartvigsen
>
> **备注:** 33 pages, 16 figures ACL 2026 (Main)
>
> **摘要:** Math word problems (MWPs) are critical K-12 educational tools, and customizing them to students' interests and ability levels can enhance learning. However, teachers struggle to find time to customize MWPs for students given large class sizes and increasing burnout. We propose that LLMs can support math education by generating MWPs customized to student interests and math education standards. We use a joint human expert-LLM judge approach to evaluate over 11,000 MWPs generated by open and closed LLMs and develop the first teacher-annotated dataset for standards-aligned educational MWP generation. We show the value of our data by using it to train a 12B open model that matches the performance of larger and more capable open models. We also use our teacher-annotated data to train a text classifier that enables a 30B open LLM to outperform existing closed baselines without any training. Next, we show our models' MWPs are more similar to human-written MWPs than those from existing models. We conclude by conducting the first study of customized LLM-generated MWPs with grade school students, finding they perform similarly on our models' MWPs relative to human-written MWPs but consistently prefer our customized MWPs.
>
---
#### [replaced 045] AtlasKV: Augmenting LLMs with Billion-Scale Knowledge Graphs in 20GB VRAM
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识增强任务，旨在解决大规模知识图谱与LLM融合的高内存和低效问题。提出AtlasKV方法，在20GB VRAM内高效整合百亿级知识图谱。**

- **链接: [https://arxiv.org/pdf/2510.17934](https://arxiv.org/pdf/2510.17934)**

> **作者:** Haoyu Huang; Hong Ting Tsang; Jiaxin Bai; Xi Peng; Gong Zhang; Yangqiu Song
>
> **备注:** ICLR 2026
>
> **摘要:** Retrieval-augmented generation (RAG) has shown some success in augmenting large language models (LLMs) with external knowledge. However, as a non-parametric knowledge integration paradigm for LLMs, RAG methods heavily rely on external retrieval modules and the retrieved textual context prior. Especially for very large scale knowledge augmentation, they would introduce substantial inference latency due to expensive searches and much longer relevant context. In this paper, we propose a parametric knowledge integration method, called \textbf{AtlasKV}, a scalable, effective, and general way to augment LLMs with billion-scale knowledge graphs (KGs) (e.g. 1B triples) using very little GPU memory cost (e.g. less than 20GB VRAM). In AtlasKV, we introduce KG2KV and HiKVP to integrate KG triples into LLMs at scale with sub-linear time and memory complexity. It maintains strong knowledge grounding and generalization performance using the LLMs' inherent attention mechanism, and requires no external retrievers, long context priors, or retraining when adapting to new knowledge.
>
---
#### [replaced 046] Proximal Supervised Fine-Tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型微调任务，解决SFT导致的泛化能力下降问题。提出PSFT方法，通过约束策略漂移提升模型稳定性与泛化性能。**

- **链接: [https://arxiv.org/pdf/2508.17784](https://arxiv.org/pdf/2508.17784)**

> **作者:** Wenhong Zhu; Ruobing Xie; Rui Wang; Xingwu Sun; Di Wang; Pengfei Liu
>
> **备注:** ICLR 2026
>
> **摘要:** Supervised fine-tuning (SFT) of foundation models often leads to poor generalization, where prior capabilities deteriorate after tuning on new tasks or domains. Inspired by trust-region policy optimization (TRPO) and proximal policy optimization (PPO) in reinforcement learning (RL), we propose Proximal SFT (PSFT). This fine-tuning objective incorporates the benefits of trust-region, effectively constraining policy drift during SFT while maintaining competitive tuning. By viewing SFT as a special case of policy gradient methods with constant positive advantages, we derive PSFT that stabilizes optimization and leads to generalization, while leaving room for further optimization in subsequent post-training stages. Experiments across mathematical and human-value domains show that PSFT matches SFT in-domain, outperforms it in out-of-domain generalization, remains stable under prolonged training without causing entropy collapse, and provides a stronger foundation for the subsequent optimization.
>
---
#### [replaced 047] Efficient Provably Secure Linguistic Steganography via Range Coding
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于语言隐写任务，旨在解决隐写容量与安全性的平衡问题。通过引入范围编码和旋转机制，提出一种高效且可证明安全的隐写方法。**

- **链接: [https://arxiv.org/pdf/2604.08052](https://arxiv.org/pdf/2604.08052)**

> **作者:** Ruiyi Yan; Yugo Murawaki
>
> **备注:** ACL2026 Main
>
> **摘要:** Linguistic steganography involves embedding secret messages within seemingly innocuous texts to enable covert communication. Provable security, which is a long-standing goal and key motivation, has been extended to language-model-based steganography. Previous provably secure approaches have achieved perfect imperceptibility, measured by zero Kullback-Leibler (KL) divergence, but at the expense of embedding capacity. In this paper, we attempt to directly use a classic entropy coding method (range coding) to achieve secure steganography, and then propose an efficient and provably secure linguistic steganographic method with a rotation mechanism. Experiments across various language models show that our method achieves around 100% entropy utilization (embedding efficiency) for embedding capacity, outperforming the existing baseline methods. Moreover, it achieves high embedding speeds (up to 1554.66 bits/s on GPT-2). The code is available at this http URL.
>
---
#### [replaced 048] RiTeK: A Dataset for Large Language Models Complex Reasoning over Textual Knowledge Graphs in Medicine
- **分类: cs.CL**

- **简介: 该论文提出RiTeK数据集，用于评估大语言模型在医学文本知识图谱上的复杂推理能力。旨在解决医疗TKGs数据稀缺、结构表达有限及检索系统评估不足的问题。**

- **链接: [https://arxiv.org/pdf/2410.13987](https://arxiv.org/pdf/2410.13987)**

> **作者:** Jiatan Huang; Mingchen Li; Zonghai Yao; Dawei Li; Yuxin Zhang; Zhichao Yang; Yongkang Xiao; Feiyun Ouyang; Xiaohan Li; Shuo Han; Hong Yu
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Answering complex real-world questions in the medical domain often requires accurate retrieval from medical Textual Knowledge Graphs (medical TKGs), as the relational path information from TKGs could enhance the inference ability of Large Language Models (LLMs). However, the main bottlenecks lie in the scarcity of existing medical TKGs, the limited expressiveness of their topological structures, and the lack of comprehensive evaluations of current retrievers for medical TKGs. To address these challenges, we first develop a Dataset1 for LLMs Complex Reasoning over medical Textual Knowledge Graphs (RiTeK), covering a broad range of topological structures. Specifically, we synthesize realistic user queries integrating diverse topological structures, relational information, and complex textual descriptions. We conduct a rigorous medical expert evaluation process to assess and validate the quality of our synthesized queries. RiTeK also serves as a comprehensive benchmark dataset for evaluating the capabilities of retrieval systems built upon LLMs. By assessing 11 representative retrievers on this benchmark, we observe that existing methods struggle to perform well, revealing notable limitations in current LLM-driven retrieval approaches. These findings highlight the pressing need for more effective retrieval systems tailored for semi-structured data in the medical domain.
>
---
#### [replaced 049] Anchored Sliding Window: Toward Robust and Imperceptible Linguistic Steganography
- **分类: cs.CL**

- **简介: 该论文属于语言模型隐写任务，解决文本易被修改破坏的问题。提出锚定滑动窗口框架，提升文本的隐蔽性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.09066](https://arxiv.org/pdf/2604.09066)**

> **作者:** Ruiyi Yan; Shiao Meng; Yugo Murawaki
>
> **备注:** ACL2026 Main
>
> **摘要:** Linguistic steganography based on language models typically assumes that steganographic texts are transmitted without alteration, making them fragile to even minor modifications. While previous work mitigates this fragility by limiting the context window, it significantly compromises text quality. In this paper, we propose the anchored sliding window (ASW) framework to improve imperceptibility and robustness. In addition to the latest tokens, the prompt and a bridge context are anchored within the context window, encouraging the model to compensate for the excluded tokens. We formulate the optimization of the bridge context as a variant of prompt distillation, which we further extend using self-distillation strategies. Experiments show that our ASW significantly and consistently outperforms the baseline method in text quality, imperceptibility, and robustness across diverse settings. The code is available at this http URL.
>
---
#### [replaced 050] M$^3$KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态知识图谱增强的生成任务，旨在解决现有系统在音频-视觉领域中知识覆盖不足和检索不精准的问题。通过构建多跳多模态知识图谱并引入精准检索方法，提升模型的推理能力和答案可信度。**

- **链接: [https://arxiv.org/pdf/2512.20136](https://arxiv.org/pdf/2512.20136)**

> **作者:** Hyeongcheol Park; Jiyoung Seo; Jaewon Mun; Hogun Park; Wonmin Byeon; Sung June Kim; Hyeonsoo Im; JeungSub Lee; Sangpil Kim
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Retrieval-Augmented Generation (RAG) has recently been extended to multimodal settings, connecting multimodal large language models (MLLMs) with vast corpora of external knowledge such as multimodal knowledge graphs (MMKGs). Despite their recent success, multimodal RAG in the audio-visual domain remains challenging due to 1) limited modality coverage and multi-hop connectivity of existing MMKGs, and 2) retrieval based solely on similarity in a shared multimodal embedding space, which fails to filter out off-topic or redundant knowledge. To address these limitations, we propose M$^3$KG-RAG, a Multi-hop Multimodal Knowledge Graph-enhanced RAG that retrieves query-aligned audio-visual knowledge from MMKGs, improving reasoning depth and answer faithfulness in MLLMs. Specifically, we devise a lightweight multi-agent pipeline to construct multi-hop MMKG (M$^3$KG), which contains context-enriched triplets of multimodal entities, enabling modality-wise retrieval based on input queries. Furthermore, we introduce GRASP (Grounded Retrieval And Selective Pruning), which ensures precise entity grounding to the query, evaluates answer-supporting relevance, and prunes redundant context to retain only knowledge essential for response generation. Extensive experiments across diverse multimodal benchmarks demonstrate that M$^3$KG-RAG significantly enhances MLLMs' multimodal reasoning and grounding over existing approaches. Project website: this https URL
>
---
#### [replaced 051] MERMAID: Memory-Enhanced Retrieval and Reasoning with Multi-Agent Iterative Knowledge Grounding for Veracity Assessment
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于事实核查任务，旨在解决证据检索与推理效率低、重复搜索的问题。提出MERMAID框架，通过记忆增强实现动态证据获取与跨声明复用。**

- **链接: [https://arxiv.org/pdf/2601.22361](https://arxiv.org/pdf/2601.22361)**

> **作者:** Yupeng Cao; Chengyang He; Yangyang Yu; Ping Wang; K.P. Subbalakshmi
>
> **摘要:** Assessing the veracity of online content has become increasingly critical. Large language models (LLMs) have recently enabled substantial progress in automated veracity assessment, including automated fact-checking and claim verification systems. Typical veracity assessment pipelines break down complex claims into sub-claims, retrieve external evidence, and then apply LLM reasoning to assess veracity. However, existing methods often treat evidence retrieval as a static, isolated step and do not effectively manage or reuse retrieved evidence across claims. In this work, we propose MERMAID, a memory-enhanced multi-agent veracity assessment framework that tightly couples the retrieval and reasoning processes. MERMAID integrates agent-driven search, structured knowledge representations, and a persistent memory module within a Reason-Action style iterative process, enabling dynamic evidence acquisition and cross-claim evidence reuse. By retaining retrieved evidence in an evidence memory, the framework reduces redundant searches and improves verification efficiency and consistency. We evaluate MERMAID on three fact-checking benchmarks and two claim-verification datasets using multiple LLMs, including GPT, LLaMA, and Qwen families. Experimental results show that MERMAID achieves state-of-the-art performance while improving the search efficiency, demonstrating the effectiveness of synergizing retrieval, reasoning, and memory for reliable veracity assessment.
>
---
#### [replaced 052] STU-PID: Steering Token Usage via PID Controller for Efficient Large Language Model Reasoning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决大模型推理中冗余步骤导致的计算效率低问题。提出STUPID方法，通过PID控制器动态调节推理过程，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2506.18831](https://arxiv.org/pdf/2506.18831)**

> **作者:** Aryasomayajula Ram Bharadwaj
>
> **摘要:** Large Language Models employing extended chain-of-thought (CoT) reasoning often suffer from the overthinking phenomenon, generating excessive and redundant reasoning steps that increase computational costs while potentially degrading performance. While recent work has explored static steering approaches to mitigate this issue, they lack the adaptability to dynamically adjust intervention strength based on real-time reasoning quality. We propose STUPID (Steering Token Usage via PID controller), a novel training-free method that employs a PID controller to dynamically modulate activation steering strength during inference. Our approach combines a chunk-level classifier for detecting redundant reasoning patterns with a PID control mechanism that adaptively adjusts steering intensity based on the predicted redundancy probability. Experimental evaluation on GSM8K demonstrates that STUPID achieves a 6% improvement in accuracy while reducing token usage by 32%, outperforming static steering baselines. Our method provides a principled framework for dynamic reasoning calibration that maintains reasoning quality while significantly improving computational efficiency.
>
---
#### [replaced 053] Powerful Training-Free Membership Inference Against Autoregressive Language Models
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文属于隐私审计任务，旨在检测微调语言模型是否泄露训练数据。提出EZ-MIA方法，通过分析错误位置的概率变化，实现高效且高精度的成员推理攻击。**

- **链接: [https://arxiv.org/pdf/2601.12104](https://arxiv.org/pdf/2601.12104)**

> **作者:** David Ilić; David Stanojević; Kostadin Cvejoski
>
> **备注:** 9 pages, 2 figures; appendix with additional experiments and derivations
>
> **摘要:** Fine-tuned language models pose significant privacy risks, as they may memorize and expose sensitive information from their training data. Membership inference attacks (MIAs) provide a principled framework for auditing these risks, yet existing methods achieve limited detection rates, particularly at the low false-positive thresholds required for practical privacy auditing. We present EZ-MIA, a membership inference attack that exploits a key observation: memorization manifests most strongly at error positions, specifically tokens where the model predicts incorrectly yet still shows elevated probability for training examples. We introduce the Error Zone (EZ) score, which measures the directional imbalance of probability shifts at error positions relative to a pretrained reference model. This principled statistic requires only two forward passes per query and no model training of any kind. On WikiText with GPT-2, EZ-MIA achieves 3.8x higher detection than the previous state-of-the-art under identical conditions (66.3% versus 17.5% true positive rate at 1% false positive rate), with near-perfect discrimination (AUC 0.98). At the stringent 0.1% FPR threshold critical for real-world auditing, we achieve 8x higher detection than prior work (14.0% versus 1.8%), requiring no reference model training. These gains extend to larger architectures: on AG News with Llama-2-7B, we achieve 3x higher detection (46.7% versus 15.8% TPR at 1% FPR). These results establish that privacy risks of fine-tuned language models are substantially greater than previously understood, with implications for both privacy auditing and deployment decisions. Code is available at this https URL.
>
---
#### [replaced 054] SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios
- **分类: cs.SE; cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于代码安全评估任务，旨在解决现有基准无法真实反映漏洞引入问题。提出SecureVibeBench基准，包含105个C/C++安全编码任务，用于评估代码代理的安全编码能力。**

- **链接: [https://arxiv.org/pdf/2509.22097](https://arxiv.org/pdf/2509.22097)**

> **作者:** Junkai Chen; Huihui Huang; Yunbo Lyu; Junwen An; Jieke Shi; Chengran Yang; Ting Zhang; Haoye Tian; Yikun Li; Zhenhao Li; Xin Zhou; Xing Hu; David Lo
>
> **备注:** ACL 2026 Main Conference
>
> **摘要:** Large language model-powered code agents are rapidly transforming software engineering, yet the security risks of their generated code have become a critical concern. Existing benchmarks have provided valuable insights, but they fail to capture scenarios in which vulnerabilities are actually introduced by human developers, making fair comparisons between humans and agents infeasible. We therefore introduce SecureVibeBench, a benchmark of 105 C/C++ secure coding tasks sourced from 41 projects in OSS-Fuzz for code agents. SecureVibeBench has the following features: (i) realistic task settings that require multi-file edits in large repositories, (ii)~aligned contexts based on real-world open-source vulnerabilities with precisely identified vulnerability introduction points, and (iii) comprehensive evaluation that combines functionality testing and security checking with both static and dynamic oracles. We evaluate 5 popular code agents like OpenHands, supported by 5 LLMs (e.g., Claude sonnet 4.5) on SecureVibeBench. Results show that current agents struggle to produce both correct and secure code, as even the best-performing one, produces merely 23.8\% correct and secure solutions on SecureVibeBench. Our code and data are on this https URL.
>
---
#### [replaced 055] C2F-Thinker: Coarse-to-Fine Reasoning with Hint-Guided Reinforcement Learning for Multimodal Sentiment Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态情感分析任务，旨在提升模型的可解释性和跨领域泛化能力。通过结合粗到细推理与提示引导的强化学习，提出C2F-Thinker框架，解决黑箱模型不可解释及强化学习效率低的问题。**

- **链接: [https://arxiv.org/pdf/2604.00013](https://arxiv.org/pdf/2604.00013)**

> **作者:** Miaosen Luo; Zhenhao Yang; Jieshen Long; Jinghu Sun; Yichu Liu; Sijie Mai
>
> **摘要:** Multimodal sentiment analysis aims to integrate textual, acoustic, and visual information for deep emotional understanding. Despite the progress of multimodal large language models (MLLMs) via supervised fine-tuning, their "black-box" nature hinders interpretability. While Chain-of-Thought (CoT) reasoning offers a potential remedy, it is constrained by high manual annotation costs and the inherent challenges of reinforcement learning (RL), such as reward sparsity and low exploration efficiency on hard samples. This paper presents C2F-Thinker, a framework that harmonizes coarse-to-fine structured reasoning with hint-guided RL through a two-stage progressive training pipeline. In the first stage, we conduct cold-start supervised fine-tuning using high-quality CoT data distilled from a larger teacher model, consisting of three distinct phases: polarity judgment, intermediate analysis, and fine-grained scoring. This equips the base model with a structured emotional reasoning paradigm. In the second stage, we introduce a hint-guided Group Relative Policy Optimization (GRPO) algorithm. By injecting correct initial polarity predictions as hints during the sampling process, the model is guided toward accurate reasoning paths, effectively mitigating cascading errors and enhancing the utilization of hard samples. Furthermore, a multi-faceted reward function incorporating classification, regression, and formatting constraints is designed to refine prediction accuracy while preserving interpretability. Experimental results demonstrate that C2F-Thinker achieves competitive performance on fine-grained sentiment regression tasks while significantly outperforming baselines in cross-domain generalization. This highlights its potential in building trustworthy and robust sentiment analysis systems for real-world applications.
>
---
#### [replaced 056] A Survey of Inductive Reasoning for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型的归纳推理问题。论文系统总结了提升归纳推理的方法，提出了评估框架，并分析了其能力来源。**

- **链接: [https://arxiv.org/pdf/2510.10182](https://arxiv.org/pdf/2510.10182)**

> **作者:** Kedi Chen; Dezhao Ruan; Yuhao Dan; Yaoting Wang; Siyu Yan; Xuecheng Wu; Yinqi Zhang; Qin Chen; Jie Zhou; Liang He; Biqing Qi; Linyang Li; Qipeng Guo; Xiaoming Shi; Wei Zhang
>
> **摘要:** Reasoning is an important task for large language models (LLMs). Among all the reasoning paradigms, inductive reasoning is one of the fundamental types, which is characterized by its particular-to-general thinking process and the non-uniqueness of its answers. The inductive mode is crucial for knowledge generalization and aligns better with human cognition, so it is a fundamental mode of learning, hence attracting increasing interest. Despite the importance of inductive reasoning, there is no systematic summary of it. Therefore, this paper presents the first comprehensive survey of inductive reasoning for LLMs. First, methods for improving inductive reasoning are categorized into three main areas: post-training, test-time scaling, and data augmentation. Then, current benchmarks of inductive reasoning are summarized, and a unified sandbox-based evaluation approach with the observation coverage metric is derived. Finally, we offer some analyses regarding the source of inductive ability and how simple model architectures and data help with inductive tasks, providing a solid foundation for future research.
>
---
#### [replaced 057] What Factors Affect LLMs and RLLMs in Financial Question Answering?
- **分类: cs.CL**

- **简介: 该论文属于金融问答任务，研究影响LLMs和RLLMs性能的因素。通过实验分析提示方法、代理框架和多语言对齐的影响，旨在提升模型在金融领域的表现。**

- **链接: [https://arxiv.org/pdf/2507.08339](https://arxiv.org/pdf/2507.08339)**

> **作者:** Peng Wang; Xuesi Hu; Jiageng Wu; Yuntao Zou; Qiancheng Zhang; Dagang Li
>
> **备注:** Accepted by ACL 2026 Findings
>
> **摘要:** Recently, large language models (LLMs) and reasoning large language models (RLLMs) have gained considerable attention from many researchers. RLLMs enhance the reasoning capabilities of LLMs through Long Chain-of-Thought (Long CoT) processes, significantly improving the performance of LLMs in addressing complex problems. However, there are few works that systematically explore what methods can fully unlock the performance of LLMs and RLLMs within the financial domain. To investigate the impact of various methods on LLMs and RLLMs, we utilize five LLMs and four RLLMs to assess the effects of prompting methods, agentic frameworks, and multilingual alignment methods on financial question-answering tasks. Our research findings indicate: (1) Current prompting methods and agent frameworks enhance the performance of LLMs in financial question answering by simulating Long CoT; (2) RLLMs possess inherent Long CoT capabilities, which limits the effectiveness of conventional methods in further enhancing their performance; (3) Current advanced multilingual alignment methods primarily improve the multilingual performance of LLMs by extending the reasoning length, which yields minimal benefits for RLLMs. Additionally, we discuss strategies for enhancing the performance of LLMs and RLLMs in financial question answering, which may serve as a inspiration for future improvements. We hope that this study can serve as an important reference for LLMs and RLLMs in the field of financial question answering.
>
---
#### [replaced 058] LayerNorm Induces Recency Bias in Transformer Decoders
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究Transformer解码器中的位置偏差问题。论文揭示LayerNorm与因果自注意力结合导致近期偏差，分析其成因并提出改进方向。**

- **链接: [https://arxiv.org/pdf/2509.21042](https://arxiv.org/pdf/2509.21042)**

> **作者:** Junu Kim; Xiao Liu; Zhenghao Lin; Lei Ji; Yeyun Gong; Edward Choi
>
> **备注:** Codes available at: this https URL
>
> **摘要:** Causal self-attention provides positional information to Transformer decoders. Prior work has shown that stacks of causal self-attention layers alone induce a positional bias in attention scores toward earlier tokens. However, this differs from the bias toward later tokens typically observed in Transformer decoders, known as recency bias. We address this discrepancy by analyzing the interaction between causal self-attention and other architectural components. We show that stacked causal self-attention layers combined with LayerNorm induce recency bias. Furthermore, we examine the effects of residual connections and the distribution of input token embeddings on this bias. Our results provide new theoretical insights into how positional information interacts with architectural components and suggest directions for improving positional encoding strategies.
>
---
#### [replaced 059] GoT-R1: Unleashing Reasoning Capability of MLLM for Visual Generation with Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **简介: 该论文属于视觉生成任务，解决复杂文本提示下物体空间关系和属性精确生成的问题。通过强化学习增强模型的语义空间推理能力，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2505.17022](https://arxiv.org/pdf/2505.17022)**

> **作者:** Chengqi Duan; Rongyao Fang; Yuqing Wang; Kun Wang; Linjiang Huang; Xingyu Zeng; Hongsheng Li; Xihui Liu
>
> **备注:** Github page refer to: this https URL. Published as a conference paper at ICLR 2026
>
> **摘要:** Visual generation models have made remarkable progress in creating realistic images from text prompts, yet struggle with complex prompts that specify multiple objects with precise spatial relationships and attributes. Effective handling of such prompts requires explicit reasoning about the semantic content and spatial layout. We present GoT-R1, a framework that applies reinforcement learning to enhance semantic-spatial reasoning in visual generation. Building upon the Generation Chain-of-Thought approach, GoT-R1 enables models to autonomously discover effective reasoning strategies beyond predefined templates through carefully designed reinforcement learning. To achieve this, we propose a dual-stage multi-dimensional reward framework that leverages MLLMs to evaluate both the reasoning process and final output, enabling effective supervision across the entire generation pipeline. The reward system assesses semantic alignment, spatial accuracy, and visual quality in a unified approach. Experimental results demonstrate significant improvements on T2I-CompBench benchmark, particularly in compositional tasks involving precise spatial relationships and attribute binding. GoT-R1 advances the state-of-the-art in image generation by successfully transferring sophisticated reasoning capabilities to the visual generation domain. To facilitate future research, we make our code and pretrained models publicly available at this https URL.
>
---
#### [replaced 060] Detection Is Cheap, Routing Is Learned: Why Refusal-Based Alignment Evaluation Fails
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型对齐研究，解决现有评估方法忽视路由机制的问题。通过实验发现检测与拒绝不足以评估对齐，提出需关注从检测到行为的路由过程。**

- **链接: [https://arxiv.org/pdf/2603.18280](https://arxiv.org/pdf/2603.18280)**

> **作者:** Gregory N. Frank
>
> **备注:** Code and data: this https URL
>
> **摘要:** Current alignment evaluation mostly measures whether models encode dangerous concepts and whether they refuse harmful requests. Both miss the layer where alignment often operates: routing from concept detection to behavioral policy. We study political censorship in Chinese-origin language models as a natural experiment, using probes, surgical ablations, and behavioral tests across nine open-weight models from five labs. Three findings follow. First, probe accuracy alone is non-diagnostic: political probes, null controls, and permutation baselines can all reach 100%, so held-out category generalization is the informative test. Second, surgical ablation reveals lab-specific routing. Removing the political-sensitivity direction eliminates censorship and restores accurate factual output in most models tested, while one model confabulates because its architecture entangles factual knowledge with the censorship mechanism. Cross-model transfer fails, indicating that routing geometry is model- and lab-specific. Third, refusal is no longer the dominant censorship mechanism. Within one model family, hard refusal falls to zero while narrative steering rises to the maximum, making censorship invisible to refusal-only benchmarks. These results support a three-stage descriptive framework: detect, route, generate. Models often retain the relevant knowledge; alignment changes how that knowledge is expressed. Evaluations that audit only detection or refusal therefore miss the routing mechanism that most directly determines behavior.
>
---
#### [replaced 061] CARINOX: Inference-time Scaling with Category-Aware Reward-based Initial Noise Optimization and Exploration
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文本到图像生成任务，旨在解决复杂语义对齐问题。提出CARINOX框架，结合噪声优化与探索，提升图像与文本的匹配度。**

- **链接: [https://arxiv.org/pdf/2509.17458](https://arxiv.org/pdf/2509.17458)**

> **作者:** Seyed Amir Kasaei; Ali Aghayari; Arash Marioriyad; Niki Sepasian; Shayan Baghayi Nejad; MohammadAmin Fazli; Mahdieh Soleymani Baghshah; Mohammad Hossein Rohban
>
> **备注:** Accepted at TMLR (2026)
>
> **摘要:** Text-to-image diffusion models, such as Stable Diffusion, can produce high-quality and diverse images but often fail to achieve compositional alignment, particularly when prompts describe complex object relationships, attributes, or spatial arrangements. Recent inference-time approaches address this by optimizing or exploring the initial noise under the guidance of reward functions that score text-image alignment without requiring model fine-tuning. While promising, each strategy has intrinsic limitations when used alone: optimization can stall due to poor initialization or unfavorable search trajectories, whereas exploration may require a prohibitively large number of samples to locate a satisfactory output. Our analysis further shows that neither single reward metrics nor ad-hoc combinations reliably capture all aspects of compositionality, leading to weak or inconsistent guidance. To overcome these challenges, we present Category-Aware Reward-based Initial Noise Optimization and Exploration (CARINOX), a unified framework that combines noise optimization and exploration with a principled reward selection procedure grounded in correlation with human judgments. Evaluations on two complementary benchmarks covering diverse compositional challenges show that CARINOX raises average alignment scores by +16% on T2I-CompBench++ and +11% on the HRS benchmark, consistently outperforming state-of-the-art optimization and exploration-based methods across all major categories, while preserving image quality and diversity. The project page is available at this https URL.
>
---
#### [replaced 062] Valence-Arousal Subspace in LLMs: Circular Emotion Geometry and Multi-Behavioral Control
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于情感控制任务，旨在通过构建VA子空间实现对语言模型情绪维度的调控。工作包括提取情感向量、学习VA轴，并验证其在多模型上的有效性。**

- **链接: [https://arxiv.org/pdf/2604.03147](https://arxiv.org/pdf/2604.03147)**

> **作者:** Lihao Sun; Lewen Yan; Xiaoya Lu; Andrew Lee; Jie Zhang; Jing Shao
>
> **摘要:** We present a method to identify a valence-arousal (VA) subspace within large language model representations. From 211k emotion-labeled texts, we derive emotion steering vectors, then learn VA axes as linear combinations of their top PCA components via ridge regression on the model's self-reported valence-arousal scores. The resulting VA subspace exhibits circular geometry consistent with established models of human emotion perception. Projections along our recovered VA subspace correlate with human-crowdsourced VA ratings across 44k lexical items. Furthermore, steering generation along these axes produces monotonic shifts in the corresponding affective dimensions of model outputs. Steering along these directions also induces near-monotonic bidirectional control over refusal and sycophancy: increasing arousal decreases refusal and increases sycophancy, and vice versa. These effects replicate across Llama-3.1-8B, Qwen3-8B, and Qwen3-14B, demonstrating cross-architecture generality. We provide a mechanistic account for these effects and prior emotionally-framed controls: refusal-associated tokens ("I can't," "sorry") occupy low-arousal, negative-valence regions, so VA steering directly modulates their emission probability.
>
---
#### [replaced 063] How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型的可控性。解决的问题是模型行为不可预测带来的风险。工作包括提出SteerEval基准，从语言、情感和人格三个维度评估控制效果。**

- **链接: [https://arxiv.org/pdf/2603.02578](https://arxiv.org/pdf/2603.02578)**

> **作者:** Ziwen Xu; Kewei Xu; Haoming Xu; Haiwen Hong; Longtao Huang; Hui Xue; Ningyu Zhang; Yongliang Shen; Guozhou Zheng; Huajun Chen; Shumin Deng
>
> **备注:** ACL 2026
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in socially sensitive domains, yet their unpredictable behaviors, ranging from misaligned intent to inconsistent personality, pose significant risks. We introduce SteerEval, a hierarchical benchmark for evaluating LLM controllability across three domains: language features, sentiment, and personality. Each domain is structured into three specification levels: L1 (what to express), L2 (how to express), and L3 (how to instantiate), connecting high-level behavioral intent to concrete textual output. Using SteerEval, we systematically evaluate contemporary steering methods, revealing that control often degrades at finer-grained levels. Our benchmark offers a principled and interpretable framework for safe and controllable LLM behavior, serving as a foundation for future research.
>
---
#### [replaced 064] Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics
- **分类: cs.CL; cs.AI; cs.CV; cs.IR; cs.LG**

- **简介: 该论文属于语言模型控制任务，旨在统一理解不同控制方法的效果。通过分析偏好与效用的权衡，提出新方法SPLIT以提升控制效果并保持生成质量。**

- **链接: [https://arxiv.org/pdf/2602.02343](https://arxiv.org/pdf/2602.02343)**

> **作者:** Ziwen Xu; Chenyan Wu; Hengyu Sun; Haiwen Hong; Mengru Wang; Yunzhi Yao; Longtao Huang; Hui Xue; Shumin Deng; Zhixuan Chu; Huajun Chen; Ningyu Zhang
>
> **备注:** ACL 2026
>
> **摘要:** Methods for controlling large language models (LLMs), including local weight fine-tuning, LoRA-based adaptation, and activation-based interventions, are often studied in isolation, obscuring their connections and making comparison difficult. In this work, we present a unified view that frames these interventions as dynamic weight updates induced by a control signal, placing them within a single conceptual framework. Building on this view, we propose a unified preference-utility analysis that separates control effects into preference, defined as the tendency toward a target concept, and utility, defined as coherent and task-valid generation, and measures both on a shared log-odds scale using polarity-paired contrastive examples. Across methods, we observe a consistent trade-off between preference and utility: stronger control increases preference while predictably reducing utility. We further explain this behavior through an activation manifold perspective, in which control shifts representations along target-concept directions to enhance preference, while utility declines primarily when interventions push representations off the model's valid-generation manifold. Finally, we introduce a new steering approach SPLIT guided by this analysis that improves preference while better preserving utility. Code is available at this https URL.
>
---
#### [replaced 065] LingoLoop Attack: Trapping MLLMs via Linguistic Context and State Entrapment into Endless Loops
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于安全攻击任务，旨在解决MLLMs易受资源耗尽攻击的问题。通过构建LingoLoop攻击，诱导模型生成冗长重复内容，暴露其漏洞。**

- **链接: [https://arxiv.org/pdf/2506.14493](https://arxiv.org/pdf/2506.14493)**

> **作者:** Jiyuan Fu; Kaixun Jiang; Lingyi Hong; Jinglun Li; Haijing Guo; Dingkang Yang; Zhaoyu Chen; Wenqiang Zhang
>
> **备注:** Accepted to ICLR 2026. Code is available at: this https URL
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown great promise but require substantial computational resources during inference. Attackers can exploit this by inducing excessive output, leading to resource exhaustion and service degradation. Prior energy-latency attacks aim to increase generation time by broadly shifting the output token distribution away from the EOS token, but they neglect the influence of token-level Part-of-Speech (POS) characteristics on EOS and sentence-level structural patterns on output counts, limiting their efficacy. To address this, we propose LingoLoop, an attack designed to induce MLLMs to generate excessively verbose and repetitive sequences. First, we find that the POS tag of a token strongly affects the likelihood of generating an EOS token. Based on this insight, we propose a POS-Aware Delay Mechanism to postpone EOS token generation by adjusting attention weights guided by POS information. Second, we identify that constraining output diversity to induce repetitive loops is effective for sustained generation. We introduce a Generative Path Pruning Mechanism that limits the magnitude of hidden states, encouraging the model to produce persistent loops. Extensive experiments on models like Qwen2.5-VL-3B demonstrate LingoLoop's powerful ability to trap them in generative loops; it consistently drives them to their generation limits and, when those limits are relaxed, can induce outputs with up to 367x more tokens than clean inputs, triggering a commensurate surge in energy consumption. These findings expose significant MLLMs' vulnerabilities, posing challenges for their reliable deployment.
>
---
#### [replaced 066] AttnTrace: Contextual Attribution of Prompt Injection and Knowledge Corruption
- **分类: cs.CL; cs.CR**

- **简介: 该论文提出AttnTrace，用于高效准确地追踪大语言模型响应的上下文来源，解决长文本下提示注入和知识污染的溯源问题。**

- **链接: [https://arxiv.org/pdf/2508.03793](https://arxiv.org/pdf/2508.03793)**

> **作者:** Yanting Wang; Runpeng Geng; Ying Chen; Jinyuan Jia
>
> **备注:** To appear in IEEE S&P 2026. The code is available at this https URL. The demo is available at this https URL
>
> **摘要:** Long-context large language models (LLMs), such as Gemini-2.5-Pro and Claude-Sonnet-4, are increasingly used to empower advanced AI systems, including retrieval-augmented generation (RAG) pipelines and autonomous agents. In these systems, an LLM receives an instruction along with a context--often consisting of texts retrieved from a knowledge database or memory--and generates a response that is contextually grounded by following the instruction. Recent studies have designed solutions to trace back to a subset of texts in the context that contributes most to the response generated by the LLM. These solutions have numerous real-world applications, including performing post-attack forensic analysis and improving the interpretability and trustworthiness of LLM outputs. While significant efforts have been made, state-of-the-art solutions such as TracLLM often lead to a high computation cost, e.g., it takes TracLLM hundreds of seconds to perform traceback for a single response-context pair. In this work, we propose AttnTrace, a new context traceback method based on the attention weights produced by an LLM for a prompt. To effectively utilize attention weights, we introduce two techniques designed to enhance the effectiveness of AttnTrace, and we provide theoretical insights for our design choice. We also perform a systematic evaluation for AttnTrace. The results demonstrate that AttnTrace is more accurate and efficient than existing state-of-the-art context traceback methods. We also show that AttnTrace can improve state-of-the-art methods in detecting prompt injection under long contexts through the attribution-before-detection paradigm. As a real-world application, we demonstrate that AttnTrace can effectively pinpoint injected instructions in a paper designed to manipulate LLM-generated reviews. The code is at this https URL.
>
---
#### [replaced 067] Linear Representations of Hierarchical Concepts in Language Models
- **分类: cs.CL**

- **简介: 该论文研究语言模型中概念层次结构的线性表示，解决层次关系编码问题。通过训练特定深度和领域的线性变换，分析层次信息在低维子空间中的编码方式。**

- **链接: [https://arxiv.org/pdf/2604.07886](https://arxiv.org/pdf/2604.07886)**

> **作者:** Masaki Sakata; Benjamin Heinzerling; Takumi Ito; Sho Yokoi; Kentaro Inui
>
> **备注:** 27 pages, 18 figures, 11 tables
>
> **摘要:** We investigate how and to what extent hierarchical relations (e.g., Japan $\subset$ Eastern Asia $\subset$ Asia) are encoded in the internal representations of language models. Building on Linear Relational Concepts, we train linear transformations specific to each hierarchical depth and semantic domain, and characterize representational differences associated with hierarchical relations by comparing these transformations. Going beyond prior work on the representational geometry of hierarchies in LMs, our analysis covers multi-token entities and cross-layer representations. Across multiple domains we learn such transformations and evaluate in-domain generalization to unseen data and cross-domain transfer. Experiments show that, within a domain, hierarchical relations can be linearly recovered from model representations. We then analyze how hierarchical information is encoded in representation space. We find that it is encoded in a relatively low-dimensional subspace and that this subspace tends to be domain-specific. Our main result is that hierarchy representation is highly similar across these domain-specific subspaces. Overall, we find that all models considered in our experiments encode concept hierarchies in the form of highly interpretable linear representations.
>
---
#### [replaced 068] Merging Triggers, Breaking Backdoors: Defensive Poisoning for Instruction-Tuned Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于安全防护任务，旨在防御指令调优语言模型的后门攻击。通过提出MB-Defense框架，融合并破坏后门触发器，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.04448](https://arxiv.org/pdf/2601.04448)**

> **作者:** San Kim; Gary Geunbae Lee
>
> **备注:** 17 pages
>
> **摘要:** Large Language Models (LLMs) have greatly advanced Natural Language Processing (NLP), particularly through instruction tuning, which enables broad task generalization without additional fine-tuning. However, their reliance on large-scale datasets-often collected from human or web sources-makes them vulnerable to backdoor attacks, where adversaries poison a small subset of data to implant hidden behaviors. Despite this growing risk, defenses for instruction-tuned models remain underexplored. We propose MB-Defense (Merging & Breaking Defense Framework), a novel training pipeline that immunizes instruction-tuned LLMs against diverse backdoor threats. MB-Defense comprises two stages: (i) Defensive Poisoning, which merges attacker and defensive triggers into a unified backdoor representation, and (ii) Backdoor Neutralization, which breaks this representation through additional training to restore clean behavior. Extensive experiments across multiple LLMs show that MB-Defense substantially lowers attack success rates while preserving instruction-following ability. Our method offers a generalizable and data-efficient defense strategy, improving the robustness of instruction-tuned LLMs against unseen backdoor attacks.
>
---
#### [replaced 069] Agent-Dice: Disentangling Knowledge Updates via Geometric Consensus for Agent Continual Learning
- **分类: cs.CL**

- **简介: 该论文属于持续学习任务，旨在解决代理在学习新任务时的灾难性遗忘问题。提出Agent-Dice框架，通过知识解耦实现稳定与灵活性的平衡。**

- **链接: [https://arxiv.org/pdf/2601.03641](https://arxiv.org/pdf/2601.03641)**

> **作者:** Zheng Wu; Xingyu Lou; Xinbei Ma; Yansi Li; Weiwen Liu; Weinan Zhang; Jun Wang; Zhuosheng Zhang
>
> **摘要:** Large Language Model (LLM)-based agents significantly extend the utility of LLMs by interacting with dynamic environments. However, enabling agents to continually learn new tasks without catastrophic forgetting remains a critical challenge, known as the stability-plasticity dilemma. In this work, we argue that this dilemma fundamentally arises from the failure to explicitly distinguish between common knowledge shared across tasks and conflicting knowledge introduced by task-specific interference. To address this, we propose Agent-Dice, a parameter fusion framework based on directional consensus evaluation. Concretely, Agent-Dice disentangles knowledge updates through a two-stage process: geometric consensus filtering to prune conflicting gradients, and curvature-based importance weighting to amplify shared semantics. We provide a rigorous theoretical analysis that establishes the validity of the proposed fusion scheme and offers insight into the origins of the stability-plasticity dilemma. Extensive experiments on GUI agents and tool-use agent domains demonstrate that Agent-Dice exhibits outstanding continual learning performance with minimal computational overhead and parameter updates. The codes are available at this https URL.
>
---
#### [replaced 070] BiCLIP: Domain Canonicalization via Structured Geometric Transformation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于跨模态对齐任务，旨在解决视觉语言模型在不同领域中的适应问题。通过引入BiCLIP框架，利用少量样本估计几何变换，提升跨模态对齐效果。**

- **链接: [https://arxiv.org/pdf/2603.08942](https://arxiv.org/pdf/2603.08942)**

> **作者:** Pranav Mantini; Shishir K. Shah
>
> **备注:** Accepted at Domain Generalization: Evolution, Breakthroughs, and Future Horizons Workshop at CVPR 2026
>
> **摘要:** Recent advances in vision-language models (VLMs) have demonstrated remarkable zero-shot capabilities, yet adapting these models to specialized domains remains a significant challenge. Building on recent theoretical insights suggesting that independently trained VLMs are related by a canonical transformation, we extend this understanding to the concept of domains. We hypothesize that image features across disparate domains are related by a canonicalized geometric transformation that can be recovered using a small set of anchors. Few-shot classification provides a natural setting for this alignment, as the limited labeled samples serve as the anchors required to estimate this transformation. Motivated by this hypothesis, we introduce BiCLIP, a framework that applies a targeted transformation to multimodal features to enhance cross-modal alignment. Our approach is characterized by its extreme simplicity and low parameter footprint. Extensive evaluations across 11 standard benchmarks, including EuroSAT, DTD, and FGVCAircraft, demonstrate that BiCLIP consistently achieves state-of-the-art results. Furthermore, we provide empirical verification of existing geometric findings by analyzing the orthogonality and angular distribution of the learned transformations, confirming that structured alignment is the key to robust domain adaptation. Code is available at this https URL
>
---
#### [replaced 071] SpeechLess: Micro-utterance with Personalized Spatial Memory-aware Assistant in Everyday Augmented Reality
- **分类: cs.HC; cs.CL; cs.ET; cs.IR**

- **简介: 该论文提出SpeechLess，一种基于个性化空间记忆的AR助手，解决公共场合语音交互尴尬和重复表达问题。通过减少用户说话量，提升信息获取效率。**

- **链接: [https://arxiv.org/pdf/2602.00793](https://arxiv.org/pdf/2602.00793)**

> **作者:** Yoonsang Kim; Devshree Jadeja; Divyansh Pradhan; Yalong Yang; Arie Kaufman
>
> **备注:** 11 pages, 9 figures. This is the author's version of the article that appeared at the IEEE Conference on Virtual Reality and 3D User Interfaces (IEEE VR) 2026
>
> **摘要:** Speaking aloud to a wearable AR assistant in public can be socially awkward, and re-articulating the same requests every day creates unnecessary effort. We present SpeechLess, a wearable AR assistant that introduces a speech-based intent granularity control paradigm grounded in personalized spatial memory. SpeechLess helps users "speak less," while still obtaining the information they need, and supports gradual explicitation of intent when more complex expression is required. SpeechLess binds prior interactions to multimodal personal context-space, time, activity, and referents-to form spatial memories, and leverages them to extrapolate missing intent dimensions from under-specified user queries. This enables users to dynamically adjust how explicitly they express their informational needs, from full-utterance to micro/zero-utterance interaction. We motivate our design through a week-long formative study using a commercial smart glasses platform, revealing discomfort with public voice use, frustration with repetitive speech, and hardware constraints. Building on these insights, we design SpeechLess, and evaluate it through controlled lab and in-the-wild studies. Our results indicate that regulated speech-based interaction, can improve everyday information access, reduce articulation effort, and support socially acceptable use without substantially degrading perceived usability or intent resolution accuracy across diverse everyday environments.
>
---
#### [replaced 072] Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets
- **分类: cs.CL; cs.MA**

- **简介: 该论文研究多智能体系统与单智能体系统在多跳推理任务中的性能比较，旨在解决计算资源分配与系统架构优势的争议。通过理论分析与实验验证，发现单智能体系统在相同计算预算下表现更优。**

- **链接: [https://arxiv.org/pdf/2604.02460](https://arxiv.org/pdf/2604.02460)**

> **作者:** Dat Tran; Douwe Kiela
>
> **摘要:** Recent work reports strong performance from multi-agent LLM systems (MAS), but these gains are often confounded by increased test-time computation. When computation is normalized, single-agent systems (SAS) can match or outperform MAS, yet the theoretical basis and evaluation methodology behind this comparison remain unclear. We present an information-theoretic argument, grounded in the Data Processing Inequality, suggesting that under a fixed reasoning-token budget and with perfect context utilization, single-agent systems are more information-efficient. This perspective further predicts that multi-agent systems become competitive when a single agent's effective context utilization is degraded, or when more compute is expended. We test these predictions in a controlled empirical study across three model families (Qwen3, DeepSeek-R1-Distill-Llama, and Gemini 2.5), comparing SAS with multiple MAS architectures under matched budgets. We find that SAS consistently match or outperform MAS on multi-hop reasoning tasks when reasoning tokens are held constant. Beyond aggregate performance, we conduct a detailed diagnostic analysis of system behavior and evaluation methodology. We identify significant artifacts in API-based budget control (particularly in Gemini 2.5) and in standard benchmarks, both of which can inflate apparent gains from MAS. Overall, our results suggest that, for multi-hop reasoning tasks, many reported advantages of multi-agent systems are better explained by unaccounted computation and context effects rather than inherent architectural benefits, and highlight the importance of understanding and explicitly controlling the trade-offs between compute, context, and coordination in agentic systems.
>
---
#### [replaced 073] Understanding Generalization in Role-Playing Models via Information Theory
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决角色扮演模型在实际应用中泛化能力下降的问题。通过引入信息理论指标和强化学习框架，分析并提升模型的泛化性能。**

- **链接: [https://arxiv.org/pdf/2512.17270](https://arxiv.org/pdf/2512.17270)**

> **作者:** Yongqi Li; Hao Lang; Fei Huang; Tieyun Qian; Yongbin Li
>
> **备注:** Accepted to ACL 2026 (Findings), camera-ready version
>
> **摘要:** Role-playing models (RPMs) are widely used in real-world applications but underperform when deployed in the wild. This degradation can be attributed to distribution shifts, including user, character, and dialogue compositional shifts. Existing methods like LLM-as-a-judge fall short in providing a fine-grained diagnosis of how these shifts affect RPM generalization, and thus there lack formal frameworks to characterize RPM generalization behaviors. To bridge these gaps, we introduce an information-theoretic metric, named reasoning-based effective mutual information difference (R-EMID), to measure RPM performance degradation in an interpretable way. We also derive an upper bound on R-EMID to predict the worst-case generalization performance of RPMs and theoretically reveal how various shifts contribute to the RPM performance degradation. Moreover, we propose a co-evolving reinforcement learning framework to adaptively model the connection among user, character, and dialogue context and thus enhance the estimation of dialogue response generation probability, which is critical for calculating R-EMID. Finally, we evaluate the generalization performance of various RPMs using R-EMID, finding that user shift poses the highest risk among all shifts and reinforcement learning is the most effective approach for enhancing RPM generalization.
>
---
#### [replaced 074] Preference Learning Unlocks LLMs' Psycho-Counseling Skills
- **分类: cs.CL**

- **简介: 该论文属于心理辅导任务，旨在提升LLMs的咨询能力。针对缺乏高质量数据的问题，提出评估原则并构建偏好数据集，通过偏好学习增强模型表现。**

- **链接: [https://arxiv.org/pdf/2502.19731](https://arxiv.org/pdf/2502.19731)**

> **作者:** Mian Zhang; Shaun M. Eack; Zhiyu Zoey Chen
>
> **备注:** ACL 2026 Camera-Ready
>
> **摘要:** Applying large language models (LLMs) to assist in psycho-counseling is an emerging and meaningful approach, driven by the significant gap between patient needs and the availability of mental health support. However, current LLMs struggle to consistently provide effective responses to client speeches, largely due to the lack of supervision from high-quality real psycho-counseling data, whose content is typically inaccessible due to client privacy concerns. Furthermore, the quality of therapists' responses in available sessions can vary significantly based on their professional training and experience. Assessing the quality of therapists' responses remains an open challenge. In this work, we address these challenges by first proposing a set of professional and comprehensive principles to evaluate therapists' responses to client speeches. Using these principles, we create a preference dataset, PsychoCounsel-Preference, which contains 36k high-quality preference comparison pairs. This dataset aligns with the preferences of professional psychotherapists, providing a robust foundation for evaluating and improving LLMs in psycho-counseling. Experiments on reward modeling and preference learning demonstrate that PsychoCounsel-Preference is an excellent resource for LLMs to acquire essential skills for responding to clients in a counseling session. Our best-aligned model, PsychoCounsel-Llama3-8B, achieves an impressive win rate of 87% against GPT-4o. We release PsychoCounsel-Preference, PsychoCounsel-Llama3-8B and the reward model PsychoCounsel Llama3-8B-Reward to facilitate the research of psycho-counseling with LLMs at: this https URL.
>
---
#### [replaced 075] Ultra-Low-Dimensional Prompt Tuning via Random Projection
- **分类: cs.CL**

- **简介: 该论文提出ULPT，解决大模型微调参数效率问题。通过低维提示优化和随机矩阵上投影，显著减少参数量并保持性能。属于自然语言处理中的参数高效微调任务。**

- **链接: [https://arxiv.org/pdf/2502.04501](https://arxiv.org/pdf/2502.04501)**

> **作者:** Zijun Wu; Yongchang Hao; Lili Mou
>
> **备注:** Accepted by EACL 2026 (Main Conference, Long Paper)
>
> **摘要:** Large language models achieve state-of-the-art performance but are increasingly costly to fine-tune. Prompt tuning is a parameter-efficient fine-tuning method that addresses parameter-efficiency by learning prompt embeddings, but these embeddings are typically tied to the model's hidden dimensionality, limiting parameter saving. In this paper, we propose Ultra-Low-dimensional Prompt Tuning (ULPT), a simple yet effective method that optimizes prompts in a low-dimensional space (e.g., 2D) and uses a frozen random matrix for up-projection. ULPT can achieve 98% reduction in the training parameters compared to vanilla prompt tuning while preserving performance. Our extensive experiments across over 20 NLP tasks demonstrate that ULPT consistently outperforms recent parameter-efficient tuning methods using significantly fewer parameters, making it well-suited as a storage-efficient framework for massive LLM customization.
>
---
#### [replaced 076] MegaFake: A Theory-Driven Dataset of Fake News Generated by Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于虚假新闻检测任务，旨在解决LLM生成虚假新闻的问题。提出理论框架并构建了MegaFake数据集，用于研究和检测机器生成的虚假信息。**

- **链接: [https://arxiv.org/pdf/2408.11871](https://arxiv.org/pdf/2408.11871)**

> **作者:** Lionel Z. Wang; Ka Chung Ng; Yiming Ma; Wenqi Fan
>
> **备注:** Decision Support Systems
>
> **摘要:** Fake news significantly influences decision-making processes by misleading individuals, organizations, and even governments. Large language models (LLMs), as part of generative AI, can amplify this problem by generating highly convincing fake news at scale, posing a significant threat to online information integrity. Therefore, understanding the motivations and mechanisms behind fake news generated by LLMs is crucial for effective detection and governance. In this study, we develop the LLM-Fake Theory, a theoretical framework that integrates various social psychology theories to explain machine-generated deception. Guided by this framework, we design an innovative prompt engineering pipeline that automates fake news generation using LLMs, eliminating manual annotation needs. Utilizing this pipeline, we create a theoretically informed \underline{M}achin\underline{e}-\underline{g}ener\underline{a}ted \underline{Fake} news dataset, MegaFake, derived from FakeNewsNet. Through extensive experiments with MegaFake, we advance both theoretical understanding of human-machine deception mechanisms and practical approaches to fake news detection in the LLM era.
>
---
#### [replaced 077] Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于自动化问题形式化任务，旨在解决高成本仿真驱动设计中需求难以转化为数学优化模型的问题。通过构建高质量数据集并微调LLMs，实现自然语言到可执行优化模型的自动转换。**

- **链接: [https://arxiv.org/pdf/2512.18682](https://arxiv.org/pdf/2512.18682)**

> **作者:** Yuchen Li; Handing Wang; Bing Xue; Mengjie Zhang; Yaochu Jin
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** In the high-cost simulation-driven design domain, translating ambiguous design requirements into a mathematical optimization formulation is a bottleneck for optimizing product performance. This process is time-consuming and heavily reliant on expert knowledge. While large language models (LLMs) offer potential for automating this task, existing approaches either suffer from poor formalization that fails to accurately align with the design intent or rely on solver feedback for data filtering, which is unavailable due to the high simulation costs. To address this challenge, we propose APF, a framework for solver-independent, automated problem formulation via LLMs designed to automatically convert engineers' natural language requirements into executable optimization models. The core of this framework is an innovative pipeline for automatically generating high-quality data, which overcomes the difficulty of constructing suitable fine-tuning datasets in the absence of high-cost solver feedback with the help of data generation and test instance annotation. The generated high-quality dataset is used to perform supervised fine-tuning on LLMs, significantly enhancing their ability to generate accurate and executable optimization problem formulations. Experimental results on antenna design demonstrate that APF significantly outperforms the existing methods in both the accuracy of requirement formalization and the quality of resulting radiation efficiency curves in meeting the design goals.
>
---
#### [replaced 078] GanitLLM: Difficulty-Aware Bengali Mathematical Reasoning through Curriculum-GRPO
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出GanitLLM，解决低资源语言Bengali数学推理问题。构建了难度感知数据集，采用课程GRPO方法提升模型表现。**

- **链接: [https://arxiv.org/pdf/2601.06767](https://arxiv.org/pdf/2601.06767)**

> **作者:** Shubhashis Roy Dipta; Khairul Mahbub; Nadia Najjar
>
> **备注:** Accepted at ACL 2026 (Findings)
>
> **摘要:** We present a Bengali mathematical reasoning model called GanitLLM (named after the Bangla word for mathematics, "Ganit"), together with a new difficulty-aware Bengali math corpus and a curriculum-based GRPO pipeline. Bengali is one of the world's most widely spoken languages, yet existing LLMs either reason in English and then translate, or simply fail on multi-step Bengali math, in part because reinforcement learning recipes are tuned for high-resource languages and collapse under reward sparsity in low-resource settings. To address this, we construct Ganit, a rigorously filtered and decontaminated Bengali math dataset with automatic difficulty tags derived from the pass@k of a strong evaluator model. Building on this dataset, we propose Curriculum-GRPO, which combines multi-stage training (SFT + GRPO) with difficulty-aware sampling and verifiable rewards for format, numerical correctness, and Bengali reasoning. On Bn-MGSM and Bn-MSVAMP, GanitLLM-4B improves over its Qwen3-4B base by +8 and +7 accuracy points, respectively, while increasing the percentage of Bengali reasoning tokens from 14% to over 88% and reducing average solution length from 943 to 193 words.
>
---
#### [replaced 079] BiT-MCTS: A Theme-based Bidirectional MCTS Approach to Chinese Fiction Generation
- **分类: cs.CL**

- **简介: 该论文属于中文小说生成任务，旨在解决长篇叙事结构不清晰、多样性不足的问题。提出BiT-MCTS框架，通过双向MCTS扩展情节，提升故事连贯性和主题深度。**

- **链接: [https://arxiv.org/pdf/2603.14410](https://arxiv.org/pdf/2603.14410)**

> **作者:** Zhaoyi Li; Xu Zhang; Xiaojun Wan
>
> **备注:** 15 pages, 3 figures
>
> **摘要:** Generating long-form linear fiction from open-ended themes remains a major challenge for large language models, which frequently fail to guarantee global structure and narrative diversity when using premise-based or linear outlining approaches. We present BiT-MCTS, a theme-driven framework that operationalizes a "climax-first, bidirectional expansion" strategy motivated by Freytag's Pyramid. Given a theme, our method extracts a core dramatic conflict and generates an explicit climax, then employs a bidirectional Monte Carlo Tree Search (MCTS) to expand the plot backward (rising action, exposition) and forward (falling action, resolution) to produce a structured outline. A final generation stage realizes a complete narrative from the refined outline. We construct a Chinese theme corpus for evaluation and conduct extensive experiments across three contemporary LLM backbones. Results show that BiT-MCTS improves narrative coherence, plot structure, and thematic depth relative to strong baselines, while enabling substantially longer, more coherent stories according to automatic metrics and human judgments.
>
---
#### [replaced 080] Catalog-Native LLM: Speaking Item-ID Dialect with Less Entanglement for Recommendation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于推荐系统任务，旨在解决协同过滤与大语言模型融合困难的问题。通过将物品ID作为语言方言，提出IDIOMoE模型，实现文本与物品信息的高效协同。**

- **链接: [https://arxiv.org/pdf/2510.05125](https://arxiv.org/pdf/2510.05125)**

> **作者:** Reza Shirkavand; Xiaokai Wei; Chen Wang; Zheng Hui; Heng Huang; Michelle Gong
>
> **摘要:** While collaborative filtering delivers predictive accuracy and efficiency, and Large Language Models (LLMs) enable expressive and generalizable reasoning, modern recommendation systems must bring these strengths together. Growing user expectations, such as natural-language queries and transparent explanations, further highlight the need for a unified approach. However, doing so is nontrivial. Collaborative signals are often token-efficient but semantically opaque, while LLMs are semantically rich but struggle to model implicit user preferences when trained only on textual inputs. This paper introduces Item-ID + Oral-language Mixture-of-Experts Language Model (IDIOMoE), which treats item interaction histories as a native dialect within the language space, enabling collaborative signals to be understood in the same way as natural language. By splitting the Feed Forward Network of each block of a pretrained LLM into a separate text expert and an item expert with token-type gating, our method avoids destructive interference between text and catalog modalities. IDIOMoE demonstrates strong recommendation performance across both public and proprietary datasets, while preserving the text understanding of the pretrained model.
>
---
#### [replaced 081] Data Selection for Multi-turn Dialogue Instruction Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多轮对话数据选择任务，解决多轮对话数据噪声大、结构不一致的问题。提出MDS框架，通过全局和局部评估选择高质量对话数据。**

- **链接: [https://arxiv.org/pdf/2604.07892](https://arxiv.org/pdf/2604.07892)**

> **作者:** Bo Li; Shikun Zhang; Wei Ye
>
> **备注:** Github: this https URL Project: this https URL
>
> **摘要:** Instruction-tuned language models increasingly rely on large multi-turn dialogue corpora, but these datasets are often noisy and structurally inconsistent, with topic drift, repetitive chitchat, and mismatched answer formats across turns. We address this from a data selection perspective and propose \textbf{MDS} (Multi-turn Dialogue Selection), a dialogue-level framework that scores whole conversations rather than isolated turns. MDS combines a global coverage stage that performs bin-wise selection in the user-query trajectory space to retain representative yet non-redundant dialogues, with a local structural stage that evaluates within-dialogue reliability through entity-grounded topic grounding and information progress, together with query-answer form consistency for functional alignment. MDS outperforms strong single-turn selectors, dialogue-level LLM scorers, and heuristic baselines on three multi-turn benchmarks and an in-domain Banking test set, achieving the best overall rank across reference-free and reference-based metrics, and is more robust on long conversations under the same training budget. Code and resources are included in the supplementary materials.
>
---
#### [replaced 082] Parallelism and Generation Order in Masked Diffusion Language Models: Limits Today, Potential Tomorrow
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究MDLM的并行生成与生成顺序问题，通过实验分析其性能限制，并提出改进方案。属于自然语言处理任务，旨在提升模型效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.15593](https://arxiv.org/pdf/2601.15593)**

> **作者:** Yangyang Zhong; Yanmei Gu; Zhengqing Zang; Xiaomeng Li; Yuqi Ding; Xibei Jia; Yuting Shen; Zhenzhong Lan; Liwang Zhu; Weiping Liu; Junlin Zhou; Haisheng Liu; Zhong Xin Yu; Pengxin Luo; Donglian Qi; Yunfeng Yan; Junbo Zhao
>
> **摘要:** Masked Diffusion Language Models (MDLMs) promise parallel token generation and arbitrary-order decoding, yet it remains unclear to what extent current models truly realize these capabilities. We characterize MDLM behavior along two dimensions -- parallelism strength and generation order -- using Average Finalization Parallelism (AFP) and Kendall's tau. We evaluate eight mainstream MDLMs (up to 100B parameters) on 58 benchmarks spanning knowledge, reasoning, and programming. The results show that MDLMs still lag behind comparably sized autoregressive models, mainly because parallel probabilistic modeling weakens inter-token dependencies. Meanwhile, MDLMs exhibit adaptive decoding behavior: their parallelism and generation order vary significantly with the task domain, the stage of reasoning, and whether the output is correct. On tasks that require "backward information" (e.g., Sudoku), MDLMs adopt a solution order that tends to fill easier Sudoku blanks first, highlighting their advantages. Finally, we provide theoretical motivation and design insights supporting a Generate-then-Edit paradigm, which mitigates dependency loss while retaining the efficiency of parallel decoding.
>
---
#### [replaced 083] An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于信息检索任务，旨在解决RAG系统中高价值结果优先的问题。提出ITEM框架，提升实用性和问答效果。**

- **链接: [https://arxiv.org/pdf/2406.11290](https://arxiv.org/pdf/2406.11290)**

> **作者:** Hengran Zhang; Keping Bi; Jiafeng Guo; Xueqi Cheng
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** Relevance and utility are two frequently used measures to evaluate the effectiveness of an information retrieval (IR) system. Relevance emphasizes the aboutness of a result to a query, while utility refers to the result's usefulness or value to an information seeker. In retrieval-augmented generation (RAG), high-utility results should be prioritized to feed to LLMs due to their limited input bandwidth. Re-examining RAG's three core components-relevance ranking derived from retrieval models, utility judgments, and answer generation-aligns with Schutz's philosophical system of relevances, which encompasses three types of relevance representing different levels of human cognition that enhance each other. These three RAG components also reflect three cognitive levels for LLMs in question-answering. Therefore, we propose an Iterative utiliTy judgmEnt fraMework (ITEM) to promote each step in RAG. We conducted extensive experiments on retrieval (TREC DL, WebAP), utility judgment task (GTI-NQ), and factoid question-answering (NQ) datasets. Experimental results demonstrate improvements of ITEM in utility judgments, ranking, and answer generation upon representative baselines.
>
---
#### [replaced 084] What's In My Human Feedback? Learning Interpretable Descriptions of Preference Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的偏好学习任务，旨在解释人类反馈数据。解决的问题是缺乏对反馈数据编码内容的清晰理解。工作包括引入WIMHF方法，通过稀疏自编码器提取可解释特征，提升数据理解和应用效果。**

- **链接: [https://arxiv.org/pdf/2510.26202](https://arxiv.org/pdf/2510.26202)**

> **作者:** Rajiv Movva; Smitha Milli; Sewon Min; Emma Pierson
>
> **备注:** ICLR 2026 (oral). v2 adds SAE ablations and robustness checks. Code: this https URL Demo: this https URL
>
> **摘要:** Human feedback can alter language models in unpredictable and undesirable ways, as practitioners lack a clear understanding of what feedback data encodes. While prior work studies preferences over certain attributes (e.g., length or sycophancy), automatically extracting relevant features without pre-specifying hypotheses remains challenging. We introduce What's In My Human Feedback? (WIMHF), a method to explain feedback data using sparse autoencoders. WIMHF characterizes both (1) the preferences a dataset is capable of measuring and (2) the preferences that the annotators actually express. Across 7 datasets, WIMHF identifies a small number of human-interpretable features that account for the majority of the preference prediction signal achieved by black-box models. These features reveal a wide diversity in what humans prefer, and the role of dataset-level context: for example, users on Reddit prefer informality and jokes, while annotators in HH-RLHF and PRISM disprefer them. WIMHF also surfaces potentially unsafe preferences, such as that LMArena users tend to vote against refusals, often in favor of toxic content. The learned features enable effective data curation: re-labeling the harmful examples in Arena yields large safety gains (+37%) with no cost to general performance. They also allow fine-grained personalization: on the Community Alignment dataset, we learn annotator-specific weights over subjective features that improve preference prediction. WIMHF provides a human-centered analysis method for practitioners to better understand and use preference data.
>
---
#### [replaced 085] Not All Denoising Steps Are Equal: Model Scheduling for Faster Masked Diffusion Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究如何加速掩码扩散语言模型的采样。针对生成过程耗时的问题，提出通过在部分去噪步骤中替换为小型模型，提升效率。实验表明可在保持生成质量的前提下显著减少计算量。**

- **链接: [https://arxiv.org/pdf/2604.02340](https://arxiv.org/pdf/2604.02340)**

> **作者:** Ivan Sedykh; Nikita Sorokin; Valentin Malykh
>
> **摘要:** Recent advances in masked diffusion language models (MDLMs) narrow the quality gap to autoregressive LMs, but their sampling remains expensive because generation requires many full-sequence denoising passes with a large Transformer and, unlike autoregressive decoding, cannot benefit from KV caching. In this work, we exploit the flexibility of the diffusion framework and study model scheduling, where a smaller MDLM replaces the full model at a subset of denoising steps. Across models trained on OpenWebText and LM1B, we show that early and late denoising steps are substantially more robust to such replacement than middle steps, enabling up to a 17% reduction in FLOPs with only modest degradation in generative perplexity under both unconditional and prefix-conditional generation, while preserving sample diversity. We support these findings with a step-importance analysis based on loss and KL divergence between small and large models across timesteps, as well as an exhaustive search over coarse step segments, both of which identify the middle of the diffusion trajectory as most sensitive consistently across datasets. Our results suggest that simple, architecture-agnostic scheduling rules can significantly accelerate MDLM sampling while largely preserving generation quality.
>
---
#### [replaced 086] Both Ends Count! Just How Good are LLM Agents at "Text-to-Big SQL"?
- **分类: cs.DB; cs.CL; cs.IR**

- **简介: 该论文属于Text-to-SQL任务，解决其在大数据场景下的评估不足问题。提出新指标，评估执行效率、成本与数据规模影响。**

- **链接: [https://arxiv.org/pdf/2602.21480](https://arxiv.org/pdf/2602.21480)**

> **作者:** Germán T. Eizaguirre; Lars Tissen; Marc Sánchez-Artigas
>
> **备注:** 14 pages, 8 figures
>
> **摘要:** Text-to-SQL and Big Data are both extensively benchmarked fields, yet there is limited research that evaluates them jointly. In the real world, Text-to-SQL systems are often embedded with Big Data workflows, such as large-scale data processing or interactive data analytics. We refer to this as ``Text-to-Big SQL''. However, existing text-to-SQL benchmarks remain narrowly scoped and overlook the cost and performance implications that arise at scale. For instance, translation errors that are minor on small datasets lead to substantial cost and latency overheads as data scales, a relevant issue completely ignored by text-to-SQL metrics. In this paper, we overcome this overlooked challenge by introducing novel and representative metrics for evaluating Text-to-Big SQL. Our study focuses on production-level LLM agents, a database-agnostic system adaptable to diverse user needs. Via an extensive evaluation of frontier models, we show that text-to-SQL metrics are insufficient for Big Data. In contrast, our proposed text-to-Big SQL metrics accurately reflect execution efficiency, cost, and the impact of data scale. For example, GPT-4o compensates for roughly 7% lower accuracy than the top-performing later-generation models with up to a 12.16x speedup, while GPT-5.2 is more than twice as cost-effective as Gemini 3 Pro at large input scales.
>
---
#### [replaced 087] MASH: Modeling Abstention via Selective Help-Seeking
- **分类: cs.CL**

- **简介: 该论文提出MASH框架，解决大模型在知识边界外易幻觉的问题。通过强化学习，使模型在需要时主动寻求外部帮助，提升回答准确性和 abstention 能力。任务为知识密集型问答中的有效搜索与拒绝回答。**

- **链接: [https://arxiv.org/pdf/2510.01152](https://arxiv.org/pdf/2510.01152)**

> **作者:** Mustafa Omer Gul; Claire Cardie; Tanya Goyal
>
> **备注:** 25 pages, with 15 dedicated to citations and appendix. 17 tables and 11 figures. Preprint, under review. Paper updated to reflect new title and results
>
> **摘要:** LLMs cannot reliably recognize their parametric knowledge boundaries and often hallucinate answers to outside-of-boundary questions. In this paper, we introduce MASH (Modeling Abstention via Selective Help-seeking), a training framework that readily extracts abstentions from LLMs. Our key idea is that any external help-seeking by an LLM, i.e. search tool use, can serve as a proxy for abstention if the external help (search) is appropriately penalized while also rewarding answer accuracy. MASH operationalizes this idea using reinforcement learning with a pay-per-search reward. We run experiments on three knowledge-intensive QA datasets. Our results show that MASH substantially improves upon the selective help-seeking performance of prior efficient search approaches; on multi-hop datasets, it improves answer accuracy by 7.6%. Furthermore, MASH demonstrates strong off-the-shelf abstention performance, showcasing behavior competitive with prior abstention methods that additionally require predetermining model knowledge boundaries to construct training data. Overall, we show MASH training effectively aligns search tool use with parametric knowledge, which can be successfully leveraged for making abstention decisions and efficient search tool use
>
---
#### [replaced 088] From Perception to Autonomous Computational Modeling: A Multi-Agent Approach
- **分类: cs.CE; cs.CL; cs.MA**

- **简介: 该论文提出一种多智能体框架，实现从感知数据到工程报告的自动化计算流程，解决工程分析中的不确定性与保守性问题。**

- **链接: [https://arxiv.org/pdf/2604.06788](https://arxiv.org/pdf/2604.06788)**

> **作者:** Daniel N. Wilke
>
> **备注:** 32 pages, 8 figures, 5 tables
>
> **摘要:** We present a solver-agnostic framework in which coordinated large language model (LLM) agents autonomously execute the complete computational mechanics workflow, from perceptual data of an engineering component through geometry extraction, material inference, discretisation, solver execution, uncertainty quantification, and code-compliant assessment, to an engineering report with actionable recommendations. Agents are formalised as conditioned operators on a shared context space with quality gates that introduce conditional iteration between pipeline layers. We introduce a mathematical framework for extracting engineering information from perceptual data under uncertainty using interval bounds, probability densities, and fuzzy membership functions, and introduce task-dependent conservatism to resolve the ambiguity of what `conservative' means when different limit states are governed by opposing parameter trends. The framework is demonstrated through a finite element analysis pipeline applied to a photograph of a steel L-bracket, producing a 171,504-node tetrahedral mesh, seven analyses across three boundary condition hypotheses, and a code-compliant assessment revealing structural failure with a quantified redesign. All results are presented as generated in the first autonomous iteration without manual correction, reinforcing that a professional engineer must review and sign off on any such analysis.
>
---
#### [replaced 089] Look Twice before You Leap: A Rational Framework for Localized Adversarial Anonymization
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于文本匿名化任务，解决隐私与效用的平衡问题。针对现有方法依赖远程API和局部模型效果差的问题，提出RLAA框架，通过理性策略提升隐私保护并防止效用下降。**

- **链接: [https://arxiv.org/pdf/2512.06713](https://arxiv.org/pdf/2512.06713)**

> **作者:** Donghang Duan; Xu Zheng; Yuefeng He; Chong Mu; Leyi Cai; Lizong Zhang
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** Current LLM-based frameworks for text anonymization usually rely on remote API services from powerful LLMs, which creates an inherent privacy paradox: users must disclose the raw data to untrusted third parties for guaranteed privacy preservation. Moreover, directly migrating current solutions to local small-scale models (LSMs) offers a suboptimal solution with severe utility collapse. Our work argues that this failure stems not merely from the capability deficits of LSMs, but significantly from the inherent irrationality of the greedy adversarial strategies employed by current state-of-the-art (SOTA) methods. To address this drawback, we propose Rational Localized Adversarial Anonymization (RLAA), a fully localized and training-free framework featuring an Attacker-Arbitrator-Anonymizer architecture. We model the anonymization process as a trade-off between Marginal Privacy Gain (MPG) and Marginal Utility Cost (MUC), demonstrating that greedy strategies tend to drift into an irrational state. Instead, RLAA introduces an arbitrator that acts as a rationality gatekeeper, validating the attacker's inference to filter out ghost leaks. This mechanism promotes a rational early-stopping criterion, and structurally prevents utility collapse. Extensive experiments on different benchmarks demonstrate that RLAA achieves a superior privacy-utility trade-off compared to strong baselines.
>
---
#### [replaced 090] Why Code, Why Now: An Information-Theoretic Perspective on the Limits of Machine Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文探讨机器学习的局限性，分析任务的信息结构对学习效果的影响。研究代码生成与强化学习的差异，提出五级可学习性层次，揭示模型规模并非瓶颈。**

- **链接: [https://arxiv.org/pdf/2602.13934](https://arxiv.org/pdf/2602.13934)**

> **作者:** Zhimin Zhao
>
> **摘要:** This paper offers a new perspective on the limits of machine learning: the ceiling on progress is set not by model size or algorithm choice but by the information structure of the task itself. Code generation has progressed more reliably than reinforcement learning, largely because code provides dense, local, verifiable feedback at every token, whereas most reinforcement learning problems do not. This difference in feedback quality is not binary but graded. We propose a five-level hierarchy of learnability based on information structure and argue that diagnosing a task's position in this hierarchy is more predictive of scaling outcomes than any property of the model. The hierarchy rests on a formal distinction among three properties of computational problems (expressibility, computability, and learnability). We establish their pairwise relationships, including where implications hold and where they fail, and present a unified template that makes the structural differences explicit. The analysis suggests why supervised learning on code scales predictably while reinforcement learning does not, and why the common assumption that scaling alone will solve remaining ML challenges warrants scrutiny.
>
---
#### [replaced 091] HiPRAG: Hierarchical Process Rewards for Efficient Agentic Retrieval Augmented Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于增强型生成任务，解决Agentic RAG中的搜索效率问题。提出HiPRAG方法，通过细粒度奖励优化搜索决策，提升准确率并降低过搜和欠搜率。**

- **链接: [https://arxiv.org/pdf/2510.07794](https://arxiv.org/pdf/2510.07794)**

> **作者:** Peilin Wu; Mian Zhang; Kun Wan; Wentian Zhao; Kaiyu He; Xinya Du; Zhiyu Chen
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Agentic RAG is a powerful technique for incorporating external information that LLMs lack, enabling better problem solving and question answering. However, suboptimal search behaviors exist widely, such as over-search (retrieving information already known) and under-search (failing to search when necessary), which leads to unnecessary overhead and unreliable outputs. Current training methods, which typically rely on outcome-based rewards in a RL framework, lack the fine-grained control needed to address these inefficiencies. To overcome this, we introduce Hierarchical Process Rewards for Efficient agentic RAG (HiPRAG), a training methodology that incorporates a fine-grained, knowledge-grounded process reward into the RL training. Our approach evaluates the necessity of each search decision on-the-fly by decomposing the agent's reasoning trajectory into discrete, parsable steps. We then apply a hierarchical reward function that provides an additional bonus based on the proportion of optimal search and non-search steps, on top of commonly used outcome and format rewards. Experiments on the Qwen2.5 and Llama-3.2 models across seven diverse QA benchmarks show that our method achieves average accuracies of 65.4% (3B) and 67.2% (7B). This is accomplished while improving search efficiency, reducing the over-search rate to just 2.3% and concurrently lowering the under-search rate. These results demonstrate the efficacy of optimizing the reasoning process itself, not just the final outcome. Further experiments and analysis demonstrate that HiPRAG shows good generalizability across a wide range of RL algorithms, model families, sizes, and types. This work demonstrates the importance and potential of fine-grained control through RL, for improving the efficiency and optimality of reasoning for search agents.
>
---
#### [replaced 092] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning
- **分类: cs.CL; cs.GT; cs.LG**

- **简介: 该论文研究如何利用大语言模型在博弈论任务中进行有效推理，解决决策质量低的问题。通过结合上下文学习与自适应思维链推理，提升游戏中的决策效果。**

- **链接: [https://arxiv.org/pdf/2601.10775](https://arxiv.org/pdf/2601.10775)**

> **作者:** Tommaso Felice Banfi; Sashenka Gamage
>
> **备注:** Published at the AAAI 2026 Bridge: Logical and Symbolic Reasoning in Language Models (OpenReview)
>
> **摘要:** We propose a novel LLM-based framework for reasoning in discrete, game-theoretic tasks, illustrated with \emph{Tic-Tac-Toe}. The method integrates in-context learning with entropy-guided chain-of-thought (CoT) reasoning and adaptive context retrieval. The model dynamically adjusts both the number of retrieved examples and reasoning paths according to token-level uncertainty: concise reasoning with minimal context is used when uncertainty is low, whereas higher uncertainty triggers expanded multi-path CoT exploration. Experimental evaluation against a sub-optimal algorithmic opponent shows that entropy-aware adaptive reasoning substantially improves decision quality, increasing the average game outcome from \(-11.6\%\) with the baseline LLM to \(+9.5\%\) with entropy-guided adaptive reasoning over 100 games (win = +1, tie = 0, loss = -1), while maintaining a relatively low number of LLM queries per game. Statistical validation confirms that the improvement is significant, and correlation analysis reveals a negative association between token-level entropy and move optimality. These findings demonstrate that uncertainty-guided adaptive reasoning effectively enhances LLM performance in sequential decision-making environments.
>
---
#### [replaced 093] Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统中的记忆管理任务，旨在解决RAG在代理记忆中因冗余和相关性高导致的检索效率问题。提出xMemory方法，通过解耦与聚合实现更高效的检索。**

- **链接: [https://arxiv.org/pdf/2602.02007](https://arxiv.org/pdf/2602.02007)**

> **作者:** Zhanghao Hu; Qinglin Zhu; Di Liang; Hanqi Yan; Yulan He; Lin Gui
>
> **备注:** Project Address: this https URL
>
> **摘要:** Agent memory systems often adopt the standard Retrieval-Augmented Generation (RAG) pipeline, yet its underlying assumptions differ in this setting. RAG targets large, heterogeneous corpora where retrieved passages are diverse, whereas agent memory is a bounded, coherent dialogue stream with highly correlated spans that are often duplicates. Under this shift, fixed top-$k$ similarity retrieval tends to return redundant context, and post-hoc pruning can delete temporally linked prerequisites needed for correct reasoning. We argue retrieval should move beyond similarity matching and instead operate over latent components, following decoupling to aggregation: disentangle memories into semantic components, organise them into a hierarchy, and use this structure to drive retrieval. We propose xMemory, which builds a hierarchy of intact units and maintains a searchable yet faithful high-level node organisation via a sparsity--semantics objective that guides memory split and merge. At inference, xMemory retrieves top-down, selecting a compact, diverse set of themes and semantics for multi-fact queries, and expanding to episodes and raw messages only when it reduces the reader's uncertainty. Experiments on LoCoMo and PerLTQA across the three latest LLMs show consistent gains in answer quality and token efficiency.
>
---
#### [replaced 094] LIFT: A Novel Framework for Enhancing Long-Context Understanding of LLMs via Long Input Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文提出LIFT框架，解决大模型长文本理解问题。通过微调使短上下文模型适应长输入，提升长文本处理能力。**

- **链接: [https://arxiv.org/pdf/2502.14644](https://arxiv.org/pdf/2502.14644)**

> **作者:** Yansheng Mao; Yufei Xu; Jiaqi Li; Fanxu Meng; Haotong Yang; Zilong Zheng; Xiyuan Wang; Muhan Zhang
>
> **备注:** 22 pages, 7 figures, preprint
>
> **摘要:** Long context understanding remains challenging for large language models due to their limited context windows. This paper introduces Long Input Fine-Tuning (LIFT), a novel framework for long-context modeling that can enhance the long-context performance of arbitrary short-context LLMs by dynamically adapting their parameters to the given long input. Importantly, rather than endlessly extending the context window size to accommodate increasingly longer inputs in context, LIFT stores and absorbs the long input in parameters. By fine-tuning the long input into model parameters, LIFT allows short-context LLMs to answer questions even when the required information is not provided in the context during inference, avoiding the quadratic complexity w.r.t. input length of a normal long context model. Furthermore, LIFT does not simply perform continued pretraining on new, long contexts, but leverages carefully designed LLM-generated synthetic tasks to enhance the comprehension of long contexts, moving beyond mere memorization. To accommodate the additional cost of fine-tuning, we design a highly optimized pipeline that reduces the Time to First Token (TTFT) to less than 10 seconds for 8k context. We further provide a comprehensive analysis of LIFT's strengths and limitations in long-context understanding, discuss its feasibility for large-scale real-world deployment, and highlight valuable directions for future research.
>
---
#### [replaced 095] CricBench: A Multilingual Benchmark for Evaluating LLMs in Cricket Analytics
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本到SQL任务，旨在评估大语言模型在板球分析中的表现。提出CricBench基准，解决多语言和领域特定问题。**

- **链接: [https://arxiv.org/pdf/2512.21877](https://arxiv.org/pdf/2512.21877)**

> **作者:** Parth Agarwal; Navya Kommuri; Trizal Garg; Prisha Singhal; Dhruv Shah; Vaibhav Devraj; Yash Sinha; Jagat Sesh Challa; Murari Mandal; Dhruv Kumar
>
> **备注:** Under Review
>
> **摘要:** Cricket is the second most popular sport worldwide, with billions of fans seeking advanced statistical insights unavailable through standard web searches. Although LLMs have advanced significantly in Text-to-SQL tasks, their capability to handle domain-specific nuances and multilingual requirements in sports analytics remains under-explored. We present CricBench, a benchmark suite evaluating the intrinsic SQL generation abilities of LLMs on cricket data across four formats: Test, ODI, T20I, and IPL. We curate a Gold-Standard dataset of 2,654 evaluation instances across four languages (English, Hindi, Punjabi, and Telugu). We evaluate seven models, GPT-5 Mini, Claude Sonnet 4, DeepSeek R1 and V3, Qwen 235B, Llama 3.1, and Gemma 2, using schema-only prompting. No single model dominates across all formats: GPT-5 Mini leads on Test cricket (12.4% DMA), Qwen 235B leads on IPL (28.7%) and T20I (17.5%), and all models score 0% on hard ODI queries. All models show a stark disconnect between syntactic validity (>98% execution accuracy) and semantic correctness (<29% DMA), with a domain gap of 37-55 percentage points versus BIRD. To our knowledge, CricBench is the first Text-to-SQL benchmark for cricket analytics.
>
---
#### [replaced 096] Prompt Injection as Role Confusion
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文研究语言模型中的提示注入攻击问题，指出其源于角色混淆。通过设计角色探测器，验证攻击者可控信号影响模型角色感知，提出一种统一框架解释提示注入现象。**

- **链接: [https://arxiv.org/pdf/2603.12277](https://arxiv.org/pdf/2603.12277)**

> **作者:** Charles Ye; Jasmine Cui; Dylan Hadfield-Menell
>
> **摘要:** Language models remain vulnerable to prompt injection attacks despite extensive safety training. We trace this failure to role confusion: models infer the source of text based on how it sounds, not where it actually comes from. A command hidden in a webpage hijacks an agent simply because it sounds like a user instruction. This is not just behavioral: in the model's internal representations, text that sounds like a trusted source occupies the same space as text that actually is one. We design role probes which measure how models internally perceive "who is speaking", showing that attacker-controllable signals (e.g. syntactic patterns, lexical choice) control role perception. We first test this with CoT Forgery, a zero-shot attack that injects fabricated reasoning into user prompts or ingested webpages. Models mistake the text for their own thoughts, yielding 60% attack success on StrongREJECT across frontier models with near-0% baselines. Strikingly, the degree of role confusion strongly predicts attack success. We then generalize these results to standard agent prompt injections, introducing a unifying framework that reframes prompt injection not as an ad-hoc exploit but as a measurable consequence of how models represent role.
>
---
#### [replaced 097] PICon: A Multi-Turn Interrogation Framework for Evaluating Persona Agent Consistency
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决 persona agent 一致性评估问题。提出 PICon 框架，通过多轮逻辑提问检测其内部、外部和重测一致性。**

- **链接: [https://arxiv.org/pdf/2603.25620](https://arxiv.org/pdf/2603.25620)**

> **作者:** Minseo Kim; Sujeong Im; Junseong Choi; Junhee Lee; Chaeeun Shim; Edward Choi
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Large language model (LLM)-based persona agents are rapidly being adopted as scalable proxies for human participants across diverse domains. Yet there is no systematic method for verifying whether a persona agent's responses remain free of contradictions and factual inaccuracies throughout an interaction. A principle from interrogation methodology offers a lens: no matter how elaborate a fabricated identity, systematic interrogation will expose its contradictions. We apply this principle to propose PICon, an evaluation framework that probes persona agents through logically chained multi-turn questioning. PICon evaluates consistency along three core dimensions: internal consistency (freedom from self-contradiction), external consistency (alignment with real-world facts), and retest consistency (stability under repetition). Evaluating seven groups of persona agents alongside 63 real human participants, we find that even systems previously reported as highly consistent fail to meet the human baseline across all three dimensions, revealing contradictions and evasive responses under chained questioning. This work provides both a conceptual foundation and a practical methodology for evaluating persona agents before trusting them as substitutes for human participants. We provide the source code and an interactive demo at: this https URL
>
---
#### [replaced 098] Not All Rollouts are Useful: Down-Sampling Rollouts in LLM Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决大语言模型训练中的计算与内存不匹配问题。通过PODS方法，选择性使用部分rollout进行策略优化，提升效率。**

- **链接: [https://arxiv.org/pdf/2504.13818](https://arxiv.org/pdf/2504.13818)**

> **作者:** Yixuan Even Xu; Yash Savani; Fei Fang; J. Zico Kolter
>
> **备注:** 19 pages, 10 figures, TMLR 2026
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has emerged as the leading approach for enhancing reasoning capabilities in large language models. However, it faces a fundamental compute and memory asymmetry: rollout generation is embarrassingly parallel and memory-light, whereas policy updates are communication-heavy and memory-intensive. To address this, we introduce PODS (Policy Optimization with Down-Sampling), which decouples rollout generation from policy updates by training only on a strategically selected subset of rollouts, maintaining learning quality while dramatically reducing update costs. We propose a principled subset selection criterion, max-variance down-sampling, that maximizes reward diversity, and provide an efficient $O(n\log n)$ implementation. Empirically, Group Relative Policy Optimization (GRPO) with PODS achieves the peak test accuracy of vanilla GRPO at least $\mathbf{1.7\times}$ faster across the different reasoning benchmarks and hardware configurations we tested.
>
---
#### [replaced 099] HyperGraphPro: Progress-Aware Reinforcement Learning for Structure-Guided Hypergraph RAG
- **分类: cs.CL**

- **简介: 该论文属于知识增强的生成任务，旨在解决GraphRAG中检索依赖语义、忽略图结构以及奖励稀疏的问题。提出HyperGraphPro，结合语义与图结构进行检索，并通过进度优化提升多步推理效果。**

- **链接: [https://arxiv.org/pdf/2601.17755](https://arxiv.org/pdf/2601.17755)**

> **作者:** Jinyoung Park; Sanghyeok Lee; Omar Zia Khan; Hyunwoo J. Kim; Joo-Kyung Kim
>
> **备注:** In progress
>
> **摘要:** Graph Retrieval-Augmented Generation (GraphRAG) has emerged as a promising paradigm that organizes external knowledge into structured graphs of entities and relations, enabling large language models (LLMs) to perform complex reasoning beyond text-chunk retrieval. Recent advances have integrated reinforcement learning (RL) into agentic GraphRAG approaches, enabling iterative interactions with knowledge graphs during training. However, existing RL-based methods suffer from two key limitations: (1) they primarily depend on semantic similarity for retrieval, often overlooking the underlying graph topology, and (2) they rely on sparse, outcome-level rewards that fail to capture the quality of intermediate retrieval steps and their dependencies. To address these limitations, we propose HyperGraphPro, a progress-aware agentic framework for graph-based retrieval and multi-step reasoning. HyperGraphPro introduces a structure-aware hypergraph retrieval mechanism that jointly considers semantic relevance and graph connectivity, promoting coherent traversal along multi-hop reasoning paths. Furthermore, we design a progress-based stepwise policy optimization that provides dense learning signals by modulating advantages according to intermediate reasoning progress within a graph, rather than relying solely on final outcomes. Experiments on multi-hop question answering benchmarks demonstrate that HyperGraphPro consistently improves reasoning accuracy and generation quality over existing GraphRAG methods.
>
---
#### [replaced 100] M2-Verify: A Large-Scale Multidomain Benchmark for Checking Multimodal Claim Consistency
- **分类: cs.CL**

- **简介: 该论文提出M2-Verify，一个大规模多领域多模态数据集，用于评估科学主张与证据的一致性。解决多模态一致性验证问题，通过实验揭示现有模型的不足。**

- **链接: [https://arxiv.org/pdf/2604.01306](https://arxiv.org/pdf/2604.01306)**

> **作者:** Abolfazl Ansari; Delvin Ce Zhang; Zhuoyang Zou; Wenpeng Yin; Dongwon Lee
>
> **备注:** Preprint. Under Review
>
> **摘要:** Evaluating scientific arguments requires assessing the strict consistency between a claim and its underlying multimodal evidence. However, existing benchmarks lack the scale, domain diversity, and visual complexity needed to evaluate this alignment realistically. To address this gap, we introduce M2-Verify, a large-scale multimodal dataset for checking scientific claim consistency. Sourced from PubMed and arXiv, M2-Verify provides over 469K instances across 16 domains, rigorously validated through expert audits. Extensive baseline experiments show that state-of-the-art models struggle to maintain robust consistency. While top models achieve up to 85.8\% Micro-F1 on low-complexity medical perturbations, performance drops to 61.6\% on high-complexity challenges like anatomical shifts. Furthermore, expert evaluations expose hallucinations when models generate scientific explanations for their alignment decisions. Finally, we demonstrate our dataset's utility and provide comprehensive usage guidelines.
>
---
#### [replaced 101] Beyond the Beep: Scalable Collision Anticipation and Real-Time Explainability with BADAS-2.0
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出BADAS-2.0，用于车辆碰撞预警，解决长尾场景识别与实时解释问题，通过数据增强、模型压缩和可解释性分析提升性能。**

- **链接: [https://arxiv.org/pdf/2604.05767](https://arxiv.org/pdf/2604.05767)**

> **作者:** Roni Goldshmidt; Hamish Scott; Lorenzo Niccolini; Hernan Matzner
>
> **摘要:** We present BADAS-2.0, the second generation of our collision anticipation system, building on BADAS-1.0, which showed that fine-tuning V-JEPA2 on large-scale ego-centric dashcam data outperforms both academic baselines and production ADAS systems. BADAS-2.0 advances the state of the art along three axes. (i) Long-tail benchmark and accuracy: We introduce a 10-group long-tail benchmark targeting rare and safety-critical scenarios. To construct it, BADAS-1.0 is used as an active oracle to score millions of unlabeled drives and surface high-risk candidates for annotation. Combined with Nexar's Atlas platform for targeted data collection, this expands the dataset from 40k to 178,500 labeled videos (~2M clips), yielding consistent gains across all subgroups, with the largest improvements on the hardest long-tail cases. (ii) Knowledge distillation to edge: Domain-specific self-supervised pre-training on 2.25M unlabeled driving videos enables distillation into compact models, BADAS-2.0-Flash (86M) and BADAS-2.0-Flash-Lite (22M), achieving 7-12x speedup with near-parity accuracy, enabling real-time edge deployment. (iii) Explainability: BADAS-2.0 produces real-time object-centric attention heatmaps that localize the evidence behind predictions. BADAS-Reason extends this with a vision-language model that consumes the last frame and heatmap to generate driver actions and structured textual reasoning. Inference code and evaluation benchmarks are publicly available.
>
---
#### [replaced 102] Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究强化学习中语言模型推理的探索与利用问题，提出在隐状态空间进行分析，引入ER、ERV、ERA等指标，设计VERL算法提升性能。**

- **链接: [https://arxiv.org/pdf/2509.23808](https://arxiv.org/pdf/2509.23808)**

> **作者:** Fanding Huang; Guanbo Huang; Xiao Fan; Yi He; Xiao Liang; Xiao Chen; Qinting Jiang; Faisal Nadeem Khan; Jingyan Jiang; Zhi Wang
>
> **备注:** Accepted as an ACL 2026 Findings paper
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) for LLM reasoning is often framed as balancing exploration and exploitation in action space, typically operationalized with token-level proxies (e.g., output entropy or confidence). We argue that this apparent trade-off is largely a measurement artifact: token-level statistics reflect next-token uncertainty rather than how reasoning progresses over multi-token semantic structures. We therefore study exploration and exploitation in the hidden-state space of response trajectories. We use Effective Rank (ER) to quantify representational exploration and introduce its temporal derivatives, Effective Rank Velocity (ERV) and Effective Rank Acceleration (ERA), to characterize exploitative refinement dynamics. Empirically and theoretically, ER and ERV exhibit near-zero correlation in semantic space, suggesting the two capacities can be improved simultaneously. Motivated by this, we propose Velocity-Exploiting Rank Learning (VERL), which shapes the RL advantage with an auxiliary signal derived from ER/ERV and uses the more stable ERA as a meta-control variable to adaptively balance the incentives. Across multiple base models, RL algorithms, and reasoning benchmarks, VERL yields consistent improvements, including large gains on challenging tasks (e.g., 21.4\% in Gaokao 2024).
>
---
#### [replaced 103] KCS: Diversify Multi-hop Question Generation with Knowledge Composition Sampling
- **分类: cs.CL**

- **简介: 该论文属于多跳问答任务，旨在解决数据稀疏导致的语言模型学习虚假模式问题。通过引入KCS框架，增强多跳问题生成的多样性与知识整合。**

- **链接: [https://arxiv.org/pdf/2508.20567](https://arxiv.org/pdf/2508.20567)**

> **作者:** Yangfan Wang; Jie Liu; Chen Tang; Lian Yan; Jingchi Jiang
>
> **备注:** Accept by EMNLP 2025
>
> **摘要:** Multi-hop question answering faces substantial challenges due to data sparsity, which increases the likelihood of language models learning spurious patterns. To address this issue, prior research has focused on diversifying question generation through content planning and varied expression. However, these approaches often emphasize generating simple questions and neglect the integration of essential knowledge, such as relevant sentences within documents. This paper introduces the Knowledge Composition Sampling (KCS), an innovative framework designed to expand the diversity of generated multi-hop questions by sampling varied knowledge compositions within a given context. KCS models the knowledge composition selection as a sentence-level conditional prediction task and utilizes a probabilistic contrastive loss to predict the next most relevant piece of knowledge. During inference, we employ a stochastic decoding strategy to effectively balance accuracy and diversity. Compared to competitive baselines, our KCS improves the overall accuracy of knowledge composition selection by 3.9%, and its application for data augmentation yields improvements on HotpotQA and 2WikiMultihopQA datasets. Our code is available at: this https URL.
>
---
#### [replaced 104] GenProve: Learning to Generate Text with Fine-Grained Provenance
- **分类: cs.CL**

- **简介: 该论文提出GenProve任务，解决LLM生成文本时缺乏细粒度来源追踪的问题。通过构建ReFInE数据集和结合SFT与GRPO的框架，提升生成内容的准确性和可验证性。**

- **链接: [https://arxiv.org/pdf/2601.04932](https://arxiv.org/pdf/2601.04932)**

> **作者:** Jingxuan Wei; Xingyue Wang; Yanghaoyu Liao; Jie Dong; Yuchen Liu; Caijun Jia; Bihui Yu; Junnan Zhu
>
> **摘要:** Large language models (LLM) often hallucinate, and while adding citations is a common solution, it is frequently insufficient for accountability as users struggle to verify how a cited source supports a generated claim. Existing methods are typically coarse-grained and fail to distinguish between direct quotes and complex reasoning. In this paper, we introduce Generation-time Fine-grained Provenance, a task where models must generate fluent answers while simultaneously producing structured, sentence-level provenance triples. To enable this, we present ReFInE (Relation-aware Fine-grained Interpretability & Evidence), a dataset featuring expert verified annotations that distinguish between Quotation, Compression, and Inference. Building on ReFInE, we propose GenProve, a framework that combines Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO). By optimizing a composite reward for answer fidelity and provenance correctness, GenProve significantly outperforms 14 strong LLMs in joint evaluation. Crucially, our analysis uncovers a reasoning gap where models excel at surface-level quotation but struggle significantly with inference-based provenance, suggesting that verifiable reasoning remains a frontier challenge distinct from surface-level citation.
>
---
#### [replaced 105] Defending against Backdoor Attacks via Module Switching
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于模型安全任务，旨在防御后门攻击。通过模块切换方法（MSD）提高防御效果，尤其在模型数量少和共谋攻击情况下表现优异。**

- **链接: [https://arxiv.org/pdf/2504.05902](https://arxiv.org/pdf/2504.05902)**

> **作者:** Weijun Li; Ansh Arora; Xuanli He; Mark Dras; Qiongkai Xu
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Backdoor attacks pose a serious threat to deep neural networks (DNNs), allowing adversaries to implant triggers for hidden behaviors in inference. Defending against such vulnerabilities is especially difficult in the post-training setting, since end-users lack training data or prior knowledge of the attacks. Model merging offers a cost-effective defense; however, latest methods like weight averaging (WAG) provide reasonable protection when multiple homologous models are available, but are less effective with fewer models and place heavy demands on defenders. We propose a module-switching defense (MSD) for disrupting backdoor shortcuts. We first validate its theoretical rationale and empirical effectiveness on two-layer networks, showing its capability of achieving higher backdoor divergence than WAG, and preserving utility. For deep models, we evaluate MSD on Transformer and CNN architectures and design an evolutionary algorithm to optimize fusion strategies with selective mechanisms to identify the most effective combinations. Experiments show that MSD achieves stronger defense with fewer models in practical settings, and even under an underexplored case of collusive attacks among multiple models--where some models share the same backdoors--switching strategies by MSD deliver superior robustness against diverse attacks. Code is available at this https URL.
>
---
#### [replaced 106] ChatInject: Abusing Chat Templates for Prompt Injection in LLM Agents
- **分类: cs.CL**

- **简介: 该论文属于安全领域，研究LLM代理中的提示注入攻击。针对间接提示注入问题，提出ChatInject攻击方法，通过模仿聊天模板实现高效攻击，并验证其有效性与防御漏洞。**

- **链接: [https://arxiv.org/pdf/2509.22830](https://arxiv.org/pdf/2509.22830)**

> **作者:** Hwan Chang; Yonghyun Jun; Hwanhee Lee
>
> **备注:** ICLR 2026
>
> **摘要:** The growing deployment of large language model (LLM) based agents that interact with external environments has created new attack surfaces for adversarial manipulation. One major threat is indirect prompt injection, where attackers embed malicious instructions in external environment output, causing agents to interpret and execute them as if they were legitimate prompts. While previous research has focused primarily on plain-text injection attacks, we find a significant yet underexplored vulnerability: LLMs' dependence on structured chat templates and their susceptibility to contextual manipulation through persuasive multi-turn dialogues. To this end, we introduce ChatInject, an attack that formats malicious payloads to mimic native chat templates, thereby exploiting the model's inherent instruction-following tendencies. Building on this foundation, we develop a persuasion-driven Multi-turn variant that primes the agent across conversational turns to accept and execute otherwise suspicious actions. Through comprehensive experiments across frontier LLMs, we demonstrate three critical findings: (1) ChatInject achieves significantly higher average attack success rates than traditional prompt injection methods, improving from 5.18% to 32.05% on AgentDojo and from 15.13% to 45.90% on InjecAgent, with multi-turn dialogues showing particularly strong performance at average 52.33% success rate on InjecAgent, (2) chat-template-based payloads demonstrate strong transferability across models and remain effective even against closed-source LLMs, despite their unknown template structures, and (3) existing prompt-based defenses are largely ineffective against this attack approach, especially against Multi-turn variants. These findings highlight vulnerabilities in current agent systems.
>
---
#### [replaced 107] From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，解决大语言模型中的信用分配问题。针对推理与代理两种场景，梳理47种方法，提出评估资源与基准协议。**

- **链接: [https://arxiv.org/pdf/2604.09459](https://arxiv.org/pdf/2604.09459)**

> **作者:** Chenchen Zhang
>
> **摘要:** Reinforcement learning (RL) for large language models (LLMs) increasingly relies on sparse, outcome-level rewards -- yet determining which actions within a long trajectory caused the outcome remains difficult. This credit assignment (CA) problem manifests in two regimes: reasoning RL, where credit must be distributed across tokens and steps within a single chain-of-thought generation (500--30K+ tokens); and agentic RL, where multi-turn environment interaction introduces stochastic transitions, partial observability, and horizons of 100+ turns (100K--1M tokens), making episode-level credit increasingly uninformative. We survey 47 CA methods (41 core, 6 adjacent enablers) published between 2024 and early 2026, organizing them in a two-dimensional taxonomy by assignment granularity (token, segment, step, turn, multi-agent) and methodology (Monte Carlo, temporal difference, model-based, game-theoretic, information-theoretic). Beyond the survey itself, we contribute three reusable resources: (1) a structured, machine-readable paper inventory with taxonomy labels, baseline families, and evidence levels; (2) a reporting checklist for future CA papers, validated against the reviewed literature to identify systematic methodological gaps; and (3) a benchmark protocol specification with task families, metadata requirements, and controlled bifurcation tasks, accompanied by a method selection decision tree. Our synthesis suggests that the shift from reasoning to agentic RL complicates and reshapes the credit assignment landscape: reasoning CA is maturing around process reward models and critic-free group comparison, while agentic CA is driving genuinely new approaches -- hindsight counterfactual analysis, privileged asymmetric critics, and turn-level MDP reformulations -- that have no direct precedent in reasoning RL.
>
---
#### [replaced 108] Seeing Through Deception: Uncovering Misleading Creator Intent in Multimodal News with Vision-Language Models
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文属于多模态虚假信息检测任务，旨在解决创作者意图误导的问题。工作包括构建DeceptionDecoded数据集，并评估VLMs在意图推理上的不足。**

- **链接: [https://arxiv.org/pdf/2505.15489](https://arxiv.org/pdf/2505.15489)**

> **作者:** Jiaying Wu; Fanxiao Li; Zihang Fu; Min-Yen Kan; Bryan Hooi
>
> **备注:** ICLR 2026
>
> **摘要:** The impact of multimodal misinformation arises not only from factual inaccuracies but also from the misleading narratives that creators deliberately embed. Interpreting such creator intent is therefore essential for multimodal misinformation detection (MMD) and effective information governance. To this end, we introduce DeceptionDecoded, a large-scale benchmark of 12,000 image-caption pairs grounded in trustworthy reference articles, created using an intent-guided simulation framework that models both the desired influence and the execution plan of news creators. The dataset captures both misleading and non-misleading cases, spanning manipulations across visual and textual modalities, and supports three intent-centric tasks: (1) misleading intent detection, (2) misleading source attribution, and (3) creator desire inference. We evaluate 14 state-of-the-art vision-language models (VLMs) and find that they struggle with intent reasoning, often relying on shallow cues such as surface-level alignment, stylistic polish, or heuristic authenticity signals. To bridge this, our framework systematically synthesizes data that enables models to learn implication-level intent reasoning. Models trained on DeceptionDecoded demonstrate strong transferability to real-world MMD, validating our framework as both a benchmark to diagnose VLM fragility and a data synthesis engine that provides high-quality, intent-focused resources for enhancing robustness in real-world multimodal misinformation governance.
>
---
#### [replaced 109] Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation
- **分类: cs.CL**

- **简介: 该论文属于多轮对话生成任务，旨在解决对话结构不清晰和表达不完整的问题。提出DRCR框架，通过重写对话上下文提升生成质量。**

- **链接: [https://arxiv.org/pdf/2604.06784](https://arxiv.org/pdf/2604.06784)**

> **作者:** Zhiyu Cao; Peifeng Li; Qiaoming Zhu
>
> **备注:** ACL 2026 Main Conference
>
> **摘要:** Previous research on multi-party dialogue generation has predominantly leveraged structural information inherent in dialogues to directly inform the generation process. However, the prevalence of colloquial expressions and incomplete utterances in dialogues often impedes comprehension and weakens the fidelity of dialogue structure representations, which is particularly pronounced in multi-party dialogues. In this work, we propose a novel framework DRCR (Discourse coherence and Response-guided Context Rewriting) to improve multi-party dialogue generation through dialogue context rewriting. Specifically, DRCR employs two complementary feedback signals, discourse coherence and response quality, to construct preference data for both context rewriting and response generation. Moreover, we propose a dynamic self-evolution learning method that allows the rewriter and responder to continuously enhance their capabilities through mutual interaction in an iterative training loop. Comprehensive experiments conducted on four multi-party dialogue datasets substantiate the effectiveness of DRCR.
>
---
#### [replaced 110] Different types of syntactic agreement recruit the same units within large language models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究LLMs如何表征语法知识。通过分析不同句法现象，发现句法一致现象共享相同模型单元，揭示句法一致在LLMs中构成有意义的类别。**

- **链接: [https://arxiv.org/pdf/2512.03676](https://arxiv.org/pdf/2512.03676)**

> **作者:** Daria Kryvosheieva; Andrea de Varda; Evelina Fedorenko; Greta Tuckute
>
> **摘要:** Large language models (LLMs) can reliably distinguish grammatical from ungrammatical sentences, but how grammatical knowledge is represented within the models remains an open question. We investigate whether different syntactic phenomena recruit shared or distinct components in LLMs. Using a functional localization approach inspired by cognitive neuroscience, we identify the LLM units most responsive to 67 English syntactic phenomena in seven open-weight models. These units are consistently recruited across sentences containing the phenomena and causally support the models' syntactic performance. Critically, different types of syntactic agreement (e.g., subject-verb, anaphor, determiner-noun) recruit overlapping sets of units, suggesting that agreement constitutes a meaningful functional category for LLMs. This pattern holds in English, Russian, and Chinese; and further, in a cross-lingual analysis of 57 diverse languages, structurally more similar languages share more units for subject-verb agreement. Taken together, these findings reveal that syntactic agreement-a critical marker of syntactic dependencies-constitutes a meaningful category within LLMs' representational spaces.
>
---
#### [replaced 111] Towards Efficient Large Vision-Language Models: A Comprehensive Survey on Inference Strategies
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 本文综述了提升大视觉语言模型推理效率的策略，针对其计算量大、扩展性差的问题，从四个维度分析优化方法，旨在推动高效多模态系统的发展。**

- **链接: [https://arxiv.org/pdf/2603.27960](https://arxiv.org/pdf/2603.27960)**

> **作者:** Surendra Pathak; Bo Han
>
> **备注:** 12 pages
>
> **摘要:** Although Large Vision Language Models (LVLMs) have demonstrated impressive multimodal reasoning capabilities, their scalability and deployment are constrained by massive computational requirements. In particular, the massive amount of visual tokens from high-resolution input data aggravates the situation due to the quadratic complexity of attention mechanisms. To address these issues, the research community has developed several optimization frameworks. This paper presents a comprehensive survey of the current state-of-the-art techniques for accelerating LVLM inference. We introduce a systematic taxonomy that categorizes existing optimization frameworks into four primary dimensions: visual token compression, memory management and serving, efficient architectural design, and advanced decoding strategies. Furthermore, we critically examine the limitations of these current methodologies and identify critical open problems to inspire future research directions in efficient multimodal systems.
>
---
#### [replaced 112] Disco-RAG: Discourse-Aware Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于知识密集型任务，旨在解决RAG系统中信息结构化不足的问题。提出Disco-RAG框架，通过引入话语结构提升生成质量。**

- **链接: [https://arxiv.org/pdf/2601.04377](https://arxiv.org/pdf/2601.04377)**

> **作者:** Dongqi Liu; Hang Ding; Qiming Feng; Xurong Xie; Zhucun Xue; Chengjie Wang; Jian Li; Jiangning Zhang; Yabiao Wang
>
> **备注:** ACL 2026 Main & Long Conference Paper
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as an important means of enhancing the performance of large language models (LLMs) in knowledge-intensive tasks. However, most existing RAG strategies treat retrieved passages in a flat and unstructured way, which prevents the model from capturing structural cues and constrains its ability to synthesize knowledge from dispersed evidence across documents. To overcome these limitations, we propose Disco-RAG, a discourse-aware framework that explicitly injects discourse signals into the generation process. Our method constructs intra-chunk discourse trees to capture local hierarchies and builds inter-chunk rhetorical graphs to model cross-passage coherence. These structures are jointly integrated into a planning blueprint that conditions the generation. Experiments on question answering and long-document summarization benchmarks show the efficacy of our approach. Disco-RAG achieves state-of-the-art results on the benchmarks without fine-tuning. These findings underscore the important role of discourse structure in advancing RAG systems.
>
---
#### [replaced 113] Reliable Evaluation Protocol for Low-Precision Retrieval
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，针对低精度计算导致的评估不可靠问题，提出高精度评分和关联度量方法，以提高检索评估的稳定性与准确性。**

- **链接: [https://arxiv.org/pdf/2508.03306](https://arxiv.org/pdf/2508.03306)**

> **作者:** Kisu Yang; Yoonna Jang; Hwanseok Jang; Kenneth Choi; Isabelle Augenstein; Heuiseok Lim
>
> **备注:** ACL 2026 Main Conference
>
> **摘要:** Lowering the numerical precision of model parameters and computations is widely adopted to improve the efficiency of retrieval systems. However, when computing relevance scores between the query and documents in low-precision, we observe spurious ties due to the reduced granularity. This introduces high variability in the results based on tie resolution, making the evaluation less reliable. To address this, we propose a more robust retrieval evaluation protocol designed to reduce score variation. It consists of: (1) High-Precision Scoring (HPS), which upcasts the final scoring step to higher precision to resolve tied candidates with minimal computational cost; and (2) Tie-aware Retrieval Metrics (TRM), which report expected scores, range, and bias to quantify order uncertainty of tied candidates. Our experiments test multiple models with three scoring functions on two retrieval datasets to demonstrate that HPS dramatically reduces tie-induced instability, and TRM accurately recovers expected metric values. This combination enables a more consistent and reliable evaluation system for lower-precision retrievals.
>
---
#### [replaced 114] Measuring and curing reasoning rigidity: from decorative chain-of-thought to genuine faithfulness
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言模型推理的可靠性，解决推理刚性问题，提出SLRC度量和LC-CoSR方法以提升推理真实性。**

- **链接: [https://arxiv.org/pdf/2603.22816](https://arxiv.org/pdf/2603.22816)**

> **作者:** Abhinaba Basu; Pavan Chakraborty
>
> **备注:** Includes SLRC metric with formal guarantees (Theorem 1), LC-CoSR training intervention, Reasoning Integrity Score, and mechanistic analysis
>
> **摘要:** Language models increasingly show their work by writing step-by-step reasoning before answering. But are these steps genuinely used, or is the answer rigid - fixed before reasoning begins? We introduce the Step-Level Reasoning Capacity (SLRC) metric and prove it is a consistent causal estimator (Theorem 1). We propose LC-CoSR, a training method with Lyapunov stability guarantees that directly reduces rigidity. Evaluating 16 frontier models (o4-mini, GPT-5.4, Claude Opus, Grok-4, DeepSeek-R1, Gemini 2.5 Pro, and others) across six domains at N=133-500, we find reasoning falls into three modes. OpenAI's o4-mini shows 74-88% step necessity on five of six tasks (73.8-88.3%) - the highest SLRC in our study. The critical differentiator is RL-based reasoning training, not thinking tokens: Grok-4's reasoning mode shows lower faithfulness than its non-reasoning mode (1.4% vs 7.2% necessity). We discover a faithfulness paradox - high-SLRC models are more susceptible to sycophancy - and propose the Reasoning Integrity Score (RIS = SLRC x (1-Sycophancy)), which significantly predicts error detection (rho=0.66, p=0.026). LC-CoSR achieves 2.6x less negative reward than FARL and CSR baselines without external model dependencies.
>
---
#### [replaced 115] ZARA: Training-Free Motion Time-Series Reasoning via Evidence-Grounded LLM Agents
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出ZARA，解决运动时间序列的零样本识别问题。通过知识增强的代理框架，实现无需训练的活动推理，提升跨数据集和场景的泛化能力。**

- **链接: [https://arxiv.org/pdf/2508.04038](https://arxiv.org/pdf/2508.04038)**

> **作者:** Zechen Li; Baiyu Chen; Hao Xue; Flora D. Salim
>
> **备注:** Accepted by ACL 2026 Main Conference
>
> **摘要:** Motion sensor time-series are central to Human Activity Recognition (HAR), yet conventional approaches are constrained to fixed activity sets and typically require costly parameter retraining to adapt to new behaviors. While Large Language Models (LLMs) offer promising open-set reasoning capabilities, applying them directly to numerical time-series often leads to hallucinations and weak grounding. To address this challenge, we propose ZARA (Zero-training Activity Reasoning Agents), a knowledge- and retrieval-augmented agentic framework for motion time-series reasoning in a training-free inference setting. Rather than relying on black-box projections, ZARA distills reference data into a statistically grounded textual knowledge base that transforms implicit signal patterns into verifiable natural-language priors. Guided by retrieved evidence, ZARA iteratively selects discriminative cues and performs grounded reasoning over candidate activities. Extensive experiments on eight benchmarks show that ZARA generalizes robustly to unseen subjects and across datasets, demonstrating strong transferability across heterogeneous sensor domains. These results mark a step toward trustworthy, plug-and-play motion understanding beyond dataset-specific artifacts. Our code is available at this https URL.
>
---
#### [replaced 116] If an LLM Were a Character, Would It Know Its Own Story? Evaluating Lifelong Learning in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型的终身学习能力。针对现有基准不足，提出LIFESTATE-BENCH，通过剧本数据测试模型的自我意识和记忆能力，发现非参数方法更优，但存在遗忘问题。**

- **链接: [https://arxiv.org/pdf/2503.23514](https://arxiv.org/pdf/2503.23514)**

> **作者:** Siqi Fan; Xiusheng Huang; Yiqun Yao; Xuezhi Fang; Kang Liu; Peng Han; Shuo Shang; Aixin Sun; Yequan Wang
>
> **摘要:** Large language models (LLMs) can carry out human-like dialogue, but unlike humans, they are stateless due to the superposition property. However, during multi-turn, multi-agent interactions, LLMs begin to exhibit consistent, character-like behaviors, hinting at a form of emergent lifelong learning. Despite this, existing benchmarks often fail to capture these dynamics, primarily focusing on static, open-ended evaluations. To address this gap, we introduce LIFESTATE-BENCH, a benchmark designed to assess lifelong learning in LLMs. It features two episodic datasets: Hamlet and a synthetic script collection, rich in narrative structure and character interactions. Our fact checking evaluation probes models' self-awareness, episodic memory retrieval, and relationship tracking, across both parametric and non-parametric approaches. Experiments on models like Llama3.1-8B, GPT-4-turbo, and DeepSeek R1, we demonstrate that nonparametric methods significantly outperform parametric ones in managing stateful learning. However, all models exhibit challenges with catastrophic forgetting as interactions extend, highlighting the need for further advancements in lifelong learning.
>
---
#### [replaced 117] Generation-Augmented Generation: A Plug-and-Play Framework for Private Knowledge Injection in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于知识注入任务，旨在解决大语言模型中私有领域知识的高效注入问题。提出GAG框架，通过生成增强生成方式，在不修改基础模型的前提下，实现私有知识的可靠整合与部署。**

- **链接: [https://arxiv.org/pdf/2601.08209](https://arxiv.org/pdf/2601.08209)**

> **作者:** Rongji Li; Jian Xu; Yi Chen; Xueqing Chen; Yisheng Yang; Jiayi Wang; Xingyu Chen; Chunyu Xie; Dawei Leng; Xu-Yao Zhang
>
> **摘要:** In domains such as materials science, biomedicine, and finance, high-stakes deployment of large language models (LLMs) requires injecting private, domain-specific knowledge that is proprietary, fast-evolving, and under-represented in public pretraining. However, the two dominant paradigms for private knowledge injection each have clear drawbacks: fine-tuning is expensive to iterate under continual updates that can induce catastrophic forgetting and general-capability regression; retrieval-augmented generation (RAG) keeps the base model intact but remains brittle in specialized private corpora due to chunk-induced evidence fragmentation, retrieval mismatch, and long-context pressure. Inspired by how multimodal LLMs align heterogeneous modalities into a shared semantic space, we propose Generation-Augmented Generation (GAG), which treats private expertise as an auxiliary modality and injects it into a frozen base model through a compact, constant-budget latent interface. Concretely, GAG distills question-conditioned specialist knowledge from lightweight domain experts into multi-slot latent memories, integrates multi-layer expert signals via per-slot cross-layer fusion, and aligns them to the frozen base model through gated residual projection, while supporting scalable mixed-domain deployment with reliable selective activation. In a unified mixed-domain evaluation spanning two scientific private-domain QA benchmarks (catalytic materials and immunology adjuvant) together with general-domain queries, GAG consistently outperforms strong retrieval-based and parameter-efficient fine-tuning baselines on specialist QA, while preserving general-domain capability, achieving highly reliable routing, and offering a favorable efficiency--effectiveness trade-off. Code and datasets are provided in the supplementary material. Code is publicly available at this https URL.
>
---
#### [replaced 118] Language Reconstruction with Brain Predictive Coding from fMRI Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于脑信号到语言的解码任务，旨在解决如何有效利用脑信号中的语义信息提升语言重建效果。提出PredFT模型，结合预测编码理论，通过主网络与侧网络融合实现更准确的文本生成。**

- **链接: [https://arxiv.org/pdf/2405.11597](https://arxiv.org/pdf/2405.11597)**

> **作者:** Congchi Yin; Ziyi Ye; Piji Li
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** Many recent studies have shown that the perception of speech can be decoded from brain signals and subsequently reconstructed as continuous language. However, there is a lack of neurological basis for how the semantic information embedded within brain signals can be used more effectively to guide language reconstruction. Predictive coding theory suggests the human brain naturally engages in continuously predicting future words that span multiple timescales. This implies that the decoding of brain signals could potentially be associated with a predictable future. To explore the predictive coding theory within the context of language reconstruction, this paper proposes \textsc{PredFT}~(\textbf{F}MRI-to-\textbf{T}ext decoding with \textbf{Pred}ictive coding). \textsc{PredFT} consists of a main network and a side network. The side network obtains brain predictive representation from related regions of interest~(ROIs) with a self-attention module. The representation is then fused into the main network for continuous language decoding. Experiments on two naturalistic language comprehension fMRI datasets show that \textsc{PredFT} outperforms current decoding models on several evaluation metrics.
>
---
#### [replaced 119] Quantifying the Climate Risk of Generative AI: Region-Aware Carbon Accounting with G-TRACE and the AI Sustainability Pyramid
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于气候风险评估任务，旨在量化生成式AI的碳排放。通过G-TRACE框架和AI可持续性金字塔模型，分析不同区域和模态的碳足迹，提出可持续治理方案。**

- **链接: [https://arxiv.org/pdf/2511.04776](https://arxiv.org/pdf/2511.04776)**

> **作者:** Zahida Kausar; Seemab Latif; Raja Khurrum Shahzad; Mehwish Fatima
>
> **备注:** 27 page, 4 figures
>
> **摘要:** Generative Artificial Intelligence (GenAI) represents a rapidly expanding digital infrastructure whose energy demand and associated CO2 emissions are emerging as a new category of climate risk. This study introduces G-TRACE (GenAI Transformative Carbon Estimator), a cross-modal, region-aware framework that quantifies training- and inference-related emissions across modalities and deployment geographies. Using real-world analytics and microscopic simulation, G-TRACE measures energy use and carbon intensity per output type (text, image, video) and reveals how decentralized inference amplifies small per-query energy costs into system-level impacts. Through the Ghibli-style image generation trend (2024-2025), we estimate 4,309 MWh of energy consumption and 2,068 tCO2 emissions, illustrating how viral participation inflates individual digital actions into tonne-scale consequences. Building on these findings, we propose the AI Sustainability Pyramid, a seven-level governance model linking carbon accounting metrics (L1-L7) with operational readiness, optimization, and stewardship. This framework translates quantitative emission metrics into actionable policy guidance for sustainable AI deployment. The study contributes to the quantitative assessment of emerging digital infrastructures as a novel category of climate risk, supporting adaptive governance for sustainable technology deployment. By situating GenAI within climate-risk frameworks, the work advances data-driven methods for aligning technological innovation with global decarbonization and resilience objectives.
>
---
#### [replaced 120] Arbitration Failure, Not Perceptual Blindness: How Vision-Language Models Resolve Visual-Linguistic Conflicts
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉语言模型在视觉-语言冲突中的决策机制，探讨其是否因感知缺陷或决策错误导致错误回答。通过分析模型内部信号竞争，发现错误源于仲裁而非感知，提出干预方法提升视觉接地效果。**

- **链接: [https://arxiv.org/pdf/2604.09364](https://arxiv.org/pdf/2604.09364)**

> **作者:** Farhad Nooralahzadeh; Omid Rohanian; Yi Zhang; Jonathan Fürst; Kurt Stockinger
>
> **摘要:** When a Vision-Language Model (VLM) sees a blue banana and answers "yellow", is the problem of perception or arbitration? We explore the question in ten VLMs with various sizes and reveal an Encoding-Grounding Dissociation: models that fail to report what they see (and thus provide a wrong answer) still encode the visual evidence as strongly as models that provide the correct answer. Using Multimodal Arbitration Crossover (MAC) analysis with layer-by-layer Logit Lens probing, we track the competition between visual and prior signals across every layer of each model. We show that visual attributes can be linearly decodable from early layers (AUC > 0.86). The accuracy remains nearly identical for both successful and failed samples. However, the gap in the final-layer logit - not the strength of encoding - better predicts grounding outcomes with a correlation of $\rho=$ 0.847. After having studied when VLMs base their answers on image clues rather than prior knowledge, we want to understand the causal relationships. We establish causality through full-sequence activation patching. The standard last-token interventions in LLM interpretability do not affect VLMs. In contrast, replacing the full token sequence at layers identified by MAC alters 60 to 84% of outputs. Partial-token decomposition shows that image tokens carry almost all of the causal impact, while text tokens have none. Scaling addresses the remaining architectural differences to achieve perfect retention. Moving from diagnosis to intervention, we show that training-free activation steering - both linear and sparse autoencoder-guided - in early layers can improve visual grounding by up to +3.8% with degrading performance in some setups. Overall, these findings lead to a clear conclusion: VLMs already see well, but the challenge is acting on what they see. Targeted interventions can help to bridge this gap.
>
---
#### [replaced 121] CounterBench: Evaluating and Improving Counterfactual Reasoning in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于因果推理任务，旨在解决大语言模型在反事实推理中的不足。通过构建基准数据集CounterBench并提出CoIn方法，提升模型的反事实推理能力。**

- **链接: [https://arxiv.org/pdf/2502.11008](https://arxiv.org/pdf/2502.11008)**

> **作者:** Yuefei Chen; Vivek K.Singh; Jing Ma; Ruixiang Tang
>
> **摘要:** Counterfactual reasoning is widely recognized as one of the most challenging and intricate aspects of causality in artificial intelligence. In this paper, we evaluate the performance of large language models (LLMs) in counterfactual reasoning. In contrast to previous studies that primarily focus on commonsense causal reasoning, where LLMs often rely on prior knowledge for inference, we specifically assess their ability to perform counterfactual inference using a set of formal rules. To support this evaluation, we introduce a new benchmark dataset, CounterBench, comprising 1K counterfactual reasoning questions. The dataset is designed with varying levels of difficulty, diverse causal graph structures, distinct types of counterfactual questions, and multiple nonsensical name variants. Our experiments demonstrate that counterfactual reasoning poses a significant challenge for LLMs, with most models performing at levels comparable to random guessing. To enhance LLM's counterfactual reasoning ability, we propose a novel reasoning paradigm, CoIn, which guides LLMs through iterative reasoning and backtracking to systematically explore counterfactual solutions. Experimental results show that our method significantly improves LLM performance on counterfactual reasoning tasks and consistently enhances performance across different this http URL dataset is available at this https URL.
>
---
#### [replaced 122] ClaimDB: A Fact Verification Benchmark over Large Structured Data
- **分类: cs.CL**

- **简介: 该论文提出ClaimDB，一个基于大规模结构化数据的事实验证基准，旨在解决真实场景下的事实核查问题。通过构建多领域数据库，评估模型在复杂证据下的推理能力。**

- **链接: [https://arxiv.org/pdf/2601.14698](https://arxiv.org/pdf/2601.14698)**

> **作者:** Michael Theologitis; Preetam Prabhu Srikar Dammu; Chirag Shah; Dan Suciu
>
> **备注:** ACL 2026 main
>
> **摘要:** Real-world fact-checking often involves verifying claims grounded in structured data at scale. Despite substantial progress in fact-verification benchmarks, this setting remains largely underexplored. In this work, we introduce ClaimDB, a fact-verification benchmark where the evidence for claims is derived from compositions of millions of records and multiple tables. ClaimDB consists of 80 unique real-life databases covering a wide range of domains, from governance and healthcare to media, education and the natural sciences. At this scale, verification approaches that rely on "reading" the evidence break down, forcing a timely shift toward reasoning in executable programs. We conduct extensive experiments with 30 state-of-the-art proprietary and open-source (below 70B) LLMs and find that more than half score below 55% accuracy. Our analysis also reveals that both closed- and open-source models struggle with abstention -- the ability to admit that there is no evidence to decide -- raising doubts about their reliability in high-stakes data analysis tasks. We release the benchmark, code, and the LLM leaderboard at this https URL .
>
---
#### [replaced 123] Improving LLM Unlearning Robustness via Random Perturbations
- **分类: cs.CL**

- **简介: 该论文属于模型遗忘任务，解决遗忘过程降低模型鲁棒性的问题。通过将遗忘建模为后门攻击，提出随机噪声增强方法提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2501.19202](https://arxiv.org/pdf/2501.19202)**

> **作者:** Dang Huu-Tien; Hoang Thanh-Tung; Anh Bui; Minh-Phuong Nguyen; Le-Minh Nguyen; Naoya Inoue
>
> **备注:** Accepted by Transactions on Machine Learning Research
>
> **摘要:** Here, we show that current LLM unlearning methods inherently reduce models' robustness, causing them to misbehave even when a single non-adversarial forget-token is present in the retain-query. Toward understanding underlying causes, we propose a novel theoretical framework that reframes the unlearning process as a backdoor attack and defense problem: we formulate how the forgetting process inadvertently learns to align forget-tokens (backdoor triggers) with the target-representations (target labels). As a result, forget-tokens act as backdoor triggers that, when activated in retain-queries, cause disruptions in unlearned models' behaviors, similar to successful backdoor attacks. The sense that, LLM unlearning methods themselves poison the model, make it more vulnerable to forget-tokens, and hide rather than erase target knowledge, describes their true mechanism. To mitigate the vulnerability caused by the forgetting process, we reinterpret the retaining process as a backdoor defense and propose Random Noise Augmentation (RNA), a lightweight, model and method-agnostic approach with theoretical guarantees for improving the robustness of unlearned models. Extensive experiments demonstrate that RNA significantly improves the robustness of unlearned models while preserving forget and retain performances. This backdoor attack-defense framework offers insights into the mechanism of unlearning that can shed light on future research directions for improving unlearning robustness.
>
---
#### [replaced 124] Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于持续预训练任务，解决模型在新领域中遗忘原有能力的问题。提出Data Mixing Agent，通过强化学习自动学习领域重加权策略，实现平衡性能和良好泛化。**

- **链接: [https://arxiv.org/pdf/2507.15640](https://arxiv.org/pdf/2507.15640)**

> **作者:** Kailai Yang; Xiao Liu; Lei Ji; Hao Li; Xiao Liang; Zhiwei Liu; Yeyun Gong; Peng Cheng; Mao Yang
>
> **备注:** Accepted by the ACL 2026 main conference
>
> **摘要:** Continual pre-training on small-scale task-specific data is an effective method for improving large language models in new target fields, yet it risks catastrophic forgetting of their original capabilities. A common solution is to re-weight training data mixtures from source and target fields on a domain space to achieve balanced performance. Previous domain reweighting strategies rely on manual designation with certain heuristics based on human intuition or empirical results. In this work, we prove that more general heuristics can be parameterized by proposing Data Mixing Agent, the first model-based, end-to-end framework that learns to re-weight domains. The agent learns generalizable heuristics through reinforcement learning on large quantities of data mixing trajectories with corresponding feedback from an evaluation environment. Experiments in continual pre-training on math reasoning show that Data Mixing Agent outperforms strong baselines in achieving balanced performance across source and target field benchmarks. Furthermore, it generalizes well across unseen source fields, target models, and domain spaces without retraining. Direct application to the code generation field also indicates its adaptability across target domains. Further analysis showcases the agents' well-aligned heuristics with human intuitions and their efficiency in achieving superior model performance with less source-field data.
>
---
#### [replaced 125] Large Language Models Can Help Mitigate Barren Plateaus in Quantum Neural Networks
- **分类: quant-ph; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于量子机器学习任务，旨在解决QNN训练中的 barren plateaus 问题。通过提出 AdaInit 框架，利用大语言模型生成有效初始参数，提升梯度方差。**

- **链接: [https://arxiv.org/pdf/2502.13166](https://arxiv.org/pdf/2502.13166)**

> **作者:** Jun Zhuang; Chaowen Guan
>
> **备注:** [ACL'26 Findings] TL;DR: We propose a new LLM-driven submartingale-based framework that adaptively generates effective initial parameters for quantum neural networks to mitigate barren plateaus by leveraging LLMs with the submartingale property
>
> **摘要:** In the era of noisy intermediate-scale quantum (NISQ) computing, Quantum Neural Networks (QNNs) have emerged as a promising approach for various applications, yet their training is often hindered by barren plateaus (BPs), where gradient variance vanishes exponentially as the qubit size increases. Most initialization-based mitigation strategies rely heavily on pre-designed static parameter distributions, thereby lacking adaptability to diverse model sizes or data conditions. To address these limitations, we propose AdaInit, a foundational framework that leverages large language models with the submartingale property to iteratively synthesize initial parameters for QNNs that yield non-negligible gradient variance, thereby mitigating BPs. Unlike conventional one-shot initialization methods, AdaInit adaptively explores the parameter space by incorporating dataset characteristics and gradient feedback, with theoretical guarantees of convergence to finding a set of effective initial parameters for QNNs. We provide rigorous theoretical analyses of the submartingale-based process and empirically validate that AdaInit consistently outperforms existing initialization methods in maintaining higher gradient variance across various QNN scales. We believe this work may initiate a new avenue to mitigate BPs.
>
---
#### [replaced 126] Controlling Multimodal Conversational Agents with Coverage-Enhanced Latent Actions
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究如何通过增强覆盖的潜在动作控制多模态对话代理，解决RL微调中文本空间过大和数据不足的问题。**

- **链接: [https://arxiv.org/pdf/2601.07516](https://arxiv.org/pdf/2601.07516)**

> **作者:** Yongqi Li; Hao Lang; Tieyun Qian; Yongbin Li
>
> **备注:** Accepted to ACL 2026 (Main), camera-ready version
>
> **摘要:** Vision-language models are increasingly employed as multimodal conversational agents (MCAs) for diverse conversational tasks. Recently, reinforcement learning (RL) has been widely explored for adapting MCAs to various human-AI interaction scenarios. Despite showing great enhancement in generalization performance, fine-tuning MCAs via RL still faces challenges in handling the extremely large text token space. To address this, we learn a compact latent action space for RL fine-tuning instead. Specifically, we adopt the learning from observation mechanism to construct the codebook for the latent action space, where future observations are leveraged to estimate current latent actions that could further be used to reconstruct future observations. However, the scarcity of paired image-text data hinders learning a codebook with sufficient coverage. Thus, we leverage both paired image-text data and text-only data to construct the latent action space, using a cross-modal projector for transforming text embeddings into image-text embeddings. We initialize the cross-modal projector on paired image-text data, and further train it on massive text-only data with a novel cycle consistency loss to enhance its robustness. We show that our latent action based method outperforms competitive baselines on two conversation tasks across various RL algorithms.
>
---
#### [replaced 127] Multi-Model Synthetic Training for Mission-Critical Small Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于小语言模型训练任务，旨在解决领域数据稀缺问题。通过多模型生成合成数据，提升模型性能并降低成本。**

- **链接: [https://arxiv.org/pdf/2509.13047](https://arxiv.org/pdf/2509.13047)**

> **作者:** Nolan Platt; Pragyansmita Nayak
>
> **备注:** 8 pages. Accepted as a full paper to the 3rd International Conference on Foundation and Large Language Models (IEEE FLLM) 2025
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across many domains, yet their application to specialized fields remains constrained by the scarcity and complexity of domain-specific training data. We present a novel approach that achieves a 261x cost reduction for maritime intelligence by using LLMs as one-time teachers rather than using them directly for inference. Our method transforms 3.2 billion Automatic Identification System (AIS) vessel tracking records into 21,543 synthetic question and answer pairs through multi-model generation (GPT-4o and o3-mini), preventing overfitting and ensuring accurate reasoning. The resulting fine-tuned Qwen2.5-7B model achieves 75% accuracy on maritime tasks, while being substantially cheaper than using a larger model for inference. We show that smaller, cheaper models -- when fine tuned properly -- can provide similar accuracy compared to larger models that are prohibitively expensive. Our work contributes to the growing field of synthetic dataset generation for specialized AI applications and presents a highly reproducible framework for domains where manual annotation is infeasible. Beyond expanding research in the growing field of specialized small language models, our approach has immediate applications in maritime safety, security operations, and vessel traffic management systems in various industries.
>
---
#### [replaced 128] Detecting HIV-Related Stigma in Clinical Narratives Using Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在检测临床文本中的HIV歧视。通过构建标注数据集并对比不同模型，开发首个实用的HIV歧视识别工具。**

- **链接: [https://arxiv.org/pdf/2604.07717](https://arxiv.org/pdf/2604.07717)**

> **作者:** Ziyi Chen; Yasir Khan; Mengyuan Zhang; Cheng Peng; Mengxian Lyu; Yiyang Liu; Krishna Vaddiparti; Robert L Cook; Mattia Prosperi; Yonghui Wu
>
> **摘要:** Human immunodeficiency virus (HIV)-related stigma is a critical psychosocial determinant of health for people living with HIV (PLWH), influencing mental health, engagement in care, and treatment outcomes. Although stigma-related experiences are documented in clinical narratives, there is a lack of off-the-shelf tools to extract and categorize them. This study aims to develop a large language model (LLM)-based tool for identifying HIV stigma from clinical notes. We identified clinical notes from PLWH receiving care at the University of Florida (UF) Health between 2012 and 2022. Candidate sentences were identified using expert-curated stigma-related keywords and iteratively expanded via clinical word embeddings. A total of 1,332 sentences were manually annotated across four stigma subscales: Concern with Public Attitudes, Disclosure Concerns, Negative Self-Image, and Personalized Stigma. We compared GatorTron-large and BERT as encoder-based baselines, and GPT-OSS-20B, LLaMA-8B, and MedGemma-27B as generative LLMs, under zero-shot and few-shot prompting. GatorTron-large achieved the best overall performance (Micro F1 = 0.62). Few-shot prompting substantially improved generative model performance, with 5-shot GPT-OSS-20B and LLaMA-8B achieving Micro-F1 scores of 0.57 and 0.59, respectively. Performance varied by stigma subscale, with Negative Self-Image showing the highest predictability and Personalized Stigma remaining the most challenging. Zero-shot generative inference exhibited non-trivial failure rates (up to 32%). This study develops the first practical NLP tool for identifying HIV stigma in clinical notes.
>
---
#### [replaced 129] EEPO: Exploration-Enhanced Policy Optimization via Sample-Then-Forget
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，解决LLM在RLVR中探索与利用失衡的问题。通过两阶段滚动生成和自适应遗忘机制，提升模型探索能力，增强性能。**

- **链接: [https://arxiv.org/pdf/2510.05837](https://arxiv.org/pdf/2510.05837)**

> **作者:** Liang Chen; Xueting Han; Qizhou Wang; Bo Han; Jing Bai; Hinrich Schutze; Kam-Fai Wong
>
> **备注:** ICLR 2026
>
> **摘要:** Balancing exploration and exploitation remains a central challenge in reinforcement learning with verifiable rewards (RLVR) for large language models (LLMs). Current RLVR methods often overemphasize exploitation, leading to entropy collapse, diminished exploratory capacity, and ultimately limited performance gains. Although techniques that increase policy stochasticity can promote exploration, they frequently fail to escape dominant behavioral modes. This creates a self-reinforcing loop -- repeatedly sampling and rewarding dominant modes -- that further erodes exploration. We introduce Exploration-Enhanced Policy Optimization (EEPO), a framework that promotes exploration via two-stage rollouts with adaptive unlearning. In the first stage, the model generates half of the trajectories; it then undergoes a lightweight unlearning step to temporarily suppress these sampled responses, forcing the second stage to explore different regions of the output space. This sample-then-forget mechanism disrupts the self-reinforcing loop and promotes wider exploration during rollouts. Across five reasoning benchmarks, EEPO outperforms GRPO, achieving average relative gains of 24.3% on Qwen2.5-3B, 33.0% on Llama3.2-3B-Instruct, and 10.4% on Qwen3-8B-Base.
>
---
#### [replaced 130] Agri-R1: Agricultural Reasoning for Disease Diagnosis via Automated-Synthesis and Reinforcement Learning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于农业疾病诊断任务，解决传统方法依赖大量标注数据、可解释性差的问题。通过自动化合成数据和强化学习优化模型，提升诊断准确性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.04672](https://arxiv.org/pdf/2601.04672)**

> **作者:** Wentao Zhang; Mingkun Xu; Qi Zhang; Shangyang Li; Derek F. Wong; Lifei Wang; Yanchao Yang; Lina Lu; Tao Fang
>
> **备注:** This paper is submitted for review to the 2026 ACM MM Conference. The corresponding authors are Tao Fang and Lina Lu, where Tao Fang is the senior Corresponding Author (Last Author) and the principal supervisor of this work, having led the research design, guided the methodology, and overseen the entire project
>
> **摘要:** Agricultural disease diagnosis challenges VLMs, as conventional fine-tuning requires extensive labels, lacks interpretability, and generalizes poorly. While reasoning improves model robustness, existing methods rely on costly expert annotations and rarely address the open-ended, diverse nature of agricultural queries. To address these limitations, we propose \textbf{Agri-R1}, a reasoning-enhanced large model for agriculture. Our framework automates high-quality reasoning data generation via vision-language synthesis and LLM-based filtering, using only 19\% of available samples. Training employs Group Relative Policy Optimization (GRPO) with a novel reward function that integrates domain-specific lexicons and fuzzy matching to assess both correctness and linguistic flexibility in open-ended responses. Evaluated on CDDMBench, our resulting 3B-parameter model achieves performance competitive with 7B- to 13B-parameter baselines, showing a +27.9\% relative gain in disease recognition accuracy, +33.3\% in agricultural knowledge QA, and a +26.10-point improvement in cross-domain generalization over standard fine-tuning. These results suggest that automated reasoning synthesis paired with domain-aware reward design may provide a broadly applicable paradigm for RL-based VLM adaptation in data-scarce specialized domains. Our code and data are publicly available at: this https URL.
>
---
#### [replaced 131] Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决多语言推理模型中的性能差距问题。研究发现，主要原因是语言理解失败，提出Selective Translation策略有效缓解这一问题。**

- **链接: [https://arxiv.org/pdf/2510.27269](https://arxiv.org/pdf/2510.27269)**

> **作者:** Deokhyung Kang; Seonjeong Hwang; Daehui Kim; Hyounghun Kim; Gary Geunbae Lee
>
> **备注:** Accepted at Findings of ACL 2026
>
> **摘要:** Reasoning language models (RLMs) achieve strong performance on complex reasoning tasks, yet they still exhibit a multilingual reasoning gap, performing better in high-resource languages than in low-resource ones. While recent efforts have been made to address this gap, its underlying causes remain largely unexplored. In this work, we show that this gap primarily stems from failures in language understanding-specifically, the model's inability to translate multilingual inputs into the language dominating its reasoning traces (typically English). As identifying understanding failures can enable targeted mitigation of the gap, we evaluate a range of detection methods and find that understanding failures are detectable to a meaningful extent, with supervised approaches performing best. Building on this, we propose Selective Translation, a strategy that incorporates an English translation into the initial reasoning trace only when an understanding failure is detected. Experimental results using Qwen3-4B show that Selective Translation substantially bridges the multilingual reasoning gap, achieving near full-translation performance while translating only about 20% of inputs. Together, our results show that failures in language understanding are the primary driver of the multilingual reasoning gap and can be detected and selectively mitigated, clarifying its origin and suggesting a path toward more equitable multilingual reasoning. Our code and data are publicly available at this https URL
>
---
#### [replaced 132] Structured Causal Video Reasoning via Multi-Objective Alignment
- **分类: cs.CL**

- **简介: 该论文属于视频理解任务，旨在解决现有模型依赖非结构化推理导致的因果推断脆弱问题。通过构建结构化事件事实及多目标强化学习优化，提升视频推理的准确性和可靠性。**

- **链接: [https://arxiv.org/pdf/2604.04415](https://arxiv.org/pdf/2604.04415)**

> **作者:** Zinuo Li; Yongxin Guo; Jun Liu; Jiawei Zhan; Xi Jiang; Chengjie Wang; Mohammed Bennamoun; Farid Boussaid; Feng Zheng; Qiuhong Ke
>
> **摘要:** Human understanding of video dynamics is typically grounded in a structured mental representation of entities, actions, and temporal relations, rather than relying solely on immediate deductive reasoning. In contrast, existing Video-LLMs largely depend on unstructured video reasoning, where critical visual evidence is embedded in verbose textual descriptions and temporal causality is often weakly modeled. This leads to inefficient processes and fragile causal inference. To bridge this cognitive gap, we propose constructing a compact representation of salient events and their causal relationships, which we name Structured Event Facts, prior to the reasoning stage. This structured prior serves as an explicit constraint to promote concise and causally grounded reasoning, while also making intermediate evidence easier to verify. To effectively train models on such structured facts, we introduce CausalFact-60K and a four-stage training pipeline comprising facts alignment, format warm-start, thinking warm-start, and reinforcement learning-based post-training. During RL stage, we find that this framework introduces competing objectives, as structural completeness and causal fidelity must be balanced against reasoning length, making it difficult to optimize. We address this challenge by formulating the optimization as a Multi-Objective Reinforcement Learning (MORL) problem and explicitly optimizing toward the Pareto-Frontier to balance these trade-offs. As a result, we introduce Factum-4B, which yields more reliable reasoning and delivers stronger performance on challenging video understanding tasks requiring fine-grained temporal inference.
>
---
#### [replaced 133] Resource Consumption Threats in Large Language Models
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于资源效率研究任务，旨在解决LLM资源消耗过高的问题。通过系统综述，分析威胁来源并提出缓解方法，提升模型效率与可持续性。**

- **链接: [https://arxiv.org/pdf/2603.16068](https://arxiv.org/pdf/2603.16068)**

> **作者:** Yuanhe Zhang; Xinyue Wang; Zhican Chen; Weiliu Wang; Zilu Zhang; Zhengshuo Gong; Zhenhong Zhou; Kun Wang; Li Sun; Yang Liu; Sen Su
>
> **摘要:** Given limited and costly computational infrastructure, resource efficiency is a key requirement for large language models (LLMs). Efficient LLMs increase service capacity for providers and reduce latency and API costs for users. Recent resource consumption threats induce excessive generation, degrading model efficiency and harming both service availability and economic sustainability. This survey presents a systematic review of threats to resource consumption in LLMs. We further establish a unified view of this emerging area by clarifying its scope and examining the problem along the full pipeline from threat induction to mechanism understanding and mitigation. Our goal is to clarify the problem landscape for this emerging area, thereby providing a clearer foundation for characterization and mitigation.
>
---
#### [replaced 134] TokUR: Token-Level Uncertainty Estimation for Large Language Model Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理中的推理任务，旨在解决大模型输出不可靠的问题。通过引入token级不确定性估计框架TokUR，提升模型在数学推理中的可靠性与可解释性。**

- **链接: [https://arxiv.org/pdf/2505.11737](https://arxiv.org/pdf/2505.11737)**

> **作者:** Tunyu Zhang; Haizhou Shi; Yibin Wang; Hengyi Wang; Xiaoxiao He; Zhuowei Li; Haoxian Chen; Ligong Han; Kai Xu; Huan Zhang; Dimitris Metaxas; Hao Wang
>
> **备注:** Accepted to International Conference on Learning Representations (ICLR) 2026
>
> **摘要:** While Large Language Models (LLMs) have demonstrated impressive capabilities, their output quality remains inconsistent across various application scenarios, making it difficult to identify trustworthy responses, especially in complex tasks requiring multi-step reasoning. In this paper, we propose a Token-level Uncertainty estimation framework for Reasoning (TokUR) that enables LLMs to self-assess and self-improve their responses in mathematical reasoning. Specifically, we introduce low-rank random weight perturbation during LLM decoding to generate predictive distributions for token-level uncertainty estimation, and we aggregate these uncertainty quantities to capture the semantic uncertainty of generated responses. Experiments on mathematical reasoning datasets of varying difficulty demonstrate that TokUR exhibits a strong correlation with answer correctness and model robustness, and the uncertainty signals produced by TokUR can be leveraged to enhance the model's reasoning performance at test time. These results highlight the effectiveness of TokUR as a principled and scalable approach for improving the reliability and interpretability of LLMs in challenging reasoning tasks.
>
---
#### [replaced 135] Think Parallax: Solving Multi-Hop Problems via Multi-View Knowledge-Graph-Based Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱增强的问答任务，解决多跳推理中信息漂移问题。通过多视图框架ParallaxRAG，提升检索与问答性能。**

- **链接: [https://arxiv.org/pdf/2510.15552](https://arxiv.org/pdf/2510.15552)**

> **作者:** Jinliang Liu; Jiale Bai; Shaoning Zeng
>
> **摘要:** Large language models (LLMs) still struggle with multi-hop reasoning over knowledge-graphs (KGs), and we identify a previously overlooked structural reason for this difficulty: Transformer attention heads naturally specialize in distinct semantic relations across reasoning stages, forming a hop-aligned relay pattern. This key finding suggests that multi-hop reasoning is inherently multi-view, yet existing KG-based retrieval-augmented generation (KG-RAG) systems collapse all reasoning hops into a single representation, flat embedding space, suppressing this implicit structure and causing noisy or drifted path exploration. We introduce ParallaxRAG, a symmetric multi-view framework that decouples queries and KGs into aligned, head-specific semantic spaces. By enforcing relational diversity across multiple heads while constraining weakly related paths, ParallaxRAG constructs more accurate, cleaner subgraphs and guides LLMs through grounded, hop-wise reasoning. On WebQSP and CWQ, it achieves state-of-the-art retrieval and QA performance, substantially reduces hallucination, and generalizes strongly to the biomedical BioASQ benchmark.
>
---
#### [replaced 136] StableToken: A Noise-Robust Semantic Speech Tokenizer for Resilient SpeechLLMs
- **分类: cs.CL**

- **简介: 该论文属于语音处理任务，旨在解决语义语音分词器在噪声下的不稳定性问题。通过引入多分支投票机制，提升分词器的鲁棒性，增强SpeechLLMs性能。**

- **链接: [https://arxiv.org/pdf/2509.22220](https://arxiv.org/pdf/2509.22220)**

> **作者:** Yuhan Song; Linhao Zhang; Chuhan Wu; Aiwei Liu; Wei Jia; Houfeng Wang; Xiao Zhou
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Prevalent semantic speech tokenizers, designed to capture linguistic content, are surprisingly fragile. We find they are not robust to meaning-irrelevant acoustic perturbations; even at high Signal-to-Noise Ratios (SNRs) where speech is perfectly intelligible, their output token sequences can change drastically, increasing the learning burden for downstream LLMs. This instability stems from two flaws: a brittle single-path quantization architecture and a distant training signal indifferent to intermediate token stability. To address this, we introduce StableToken, a tokenizer that achieves stability through a consensus-driven mechanism. Its multi-branch architecture processes audio in parallel, and these representations are merged via a powerful bit-wise voting mechanism to form a single, stable token sequence. StableToken sets a new state-of-the-art in token stability, drastically reducing Unit Edit Distance (UED) under diverse noise conditions. This foundational stability translates directly to downstream benefits, significantly improving the robustness of SpeechLLMs on a variety of tasks. Our code and model are publicly available at this https URL.
>
---
#### [replaced 137] Can Large Language Models Infer Causal Relationships from Real-World Text?
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于因果推理任务，旨在解决LLMs在真实文本中推断因果关系的难题。研究构建了首个真实世界基准数据集，并发现LLMs在此任务上表现有限。**

- **链接: [https://arxiv.org/pdf/2505.18931](https://arxiv.org/pdf/2505.18931)**

> **作者:** Ryan Saklad; Aman Chadha; Oleg Pavlov; Raha Moraffah
>
> **摘要:** Understanding and inferring causal relationships from texts is a core aspect of human cognition and is essential for advancing large language models (LLMs) towards artificial general intelligence. Existing work evaluating LLM causal reasoning primarily relies on synthetic or simplified texts with explicitly stated causal relationships. These texts typically feature short passages and few causal relations, failing to reflect the complexities of real-world reasoning. In this paper, we investigate whether LLMs are capable of inferring causal relationships from real-world texts. We develop a benchmark drawn from real-world academic literature, which includes diverse texts with respect to length, complexity (different levels of explicitness, number of causal events and relationships), and domain. To the best of our knowledge, our benchmark is the first-ever real-world dataset for this task. Our experiments on this dataset show that LLMs face significant challenges in inferring causal relationships from real-world text, with the best-performing model achieving an average F$_1$ score of only 0.535. Through systematic analysis across aspects of real-world text (explicitness, number of causal events and relationships, length of text, domain), our benchmark offers targeted insights for further research into advancing LLM causal reasoning. Our code and dataset can be found at this https URL .
>
---
#### [replaced 138] FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言生成任务，解决长文本生成速度与质量的平衡问题。提出FS-DFM模型，通过少步采样实现快速准确生成。**

- **链接: [https://arxiv.org/pdf/2509.20624](https://arxiv.org/pdf/2509.20624)**

> **作者:** Amin Karimi Monsefi; Nikhil Bhendawade; Manuel Rafael Ciosici; Dominic Culver; Yizhe Zhang; Irina Belousova
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Autoregressive language models (ARMs) deliver strong likelihoods, but are inherently serial: they generate one token per forward pass, which limits throughput and inflates latency for long sequences. Diffusion Language Models (DLMs) parallelize across positions and thus appear promising for language generation, yet standard discrete diffusion typically needs hundreds to thousands of model evaluations to reach high quality, trading serial depth for iterative breadth. We introduce FS-DFM, Few-Step Discrete Flow-Matching. A discrete flow-matching model designed for speed without sacrificing quality. The core idea is simple: make the number of sampling steps an explicit parameter and train the model to be consistent across step budgets, so one big move lands where many small moves would. We pair this with a reliable update rule that moves probability in the right direction without overshooting, and with strong teacher guidance distilled from long-run trajectories. Together, these choices make few-step sampling stable, accurate, and easy to control. On language modeling benchmarks, FS-DFM with 8 sampling steps achieves perplexity parity with a 1,024-step discrete-flow baseline for generating 1,024 tokens using a similar-size model, delivering up to 128 times faster sampling and corresponding latency/throughput gains. Code & pretrained checkpoints: this https URL
>
---
#### [replaced 139] SODA: Semi On-Policy Black-Box Distillation for Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出SODA，解决大语言模型知识蒸馏中的效率与稳定性问题，通过半在线策略对比提升蒸馏效果。**

- **链接: [https://arxiv.org/pdf/2604.03873](https://arxiv.org/pdf/2604.03873)**

> **作者:** Xiwen Chen; Jingjing Wang; Wenhui Zhu; Peijie Qiu; Xuanzhao Dong; Hejian Sang; Zhipeng Wang; Alborz Geramifard; Feng Luo
>
> **备注:** The manuscript is currently undergoing internal review and approval. The authors will resubmit an updated version after this process is completed
>
> **摘要:** Black-box knowledge distillation for large language models presents a strict trade-off. Simple off-policy methods (e.g., sequence-level knowledge distillation) struggle to correct the student's inherent errors. Fully on-policy methods (e.g., Generative Adversarial Distillation) solve this via adversarial training but introduce well-known training instability and crippling computational overhead. To address this dilemma, we propose SODA (Semi On-policy Distillation with Alignment), a highly efficient alternative motivated by the inherent capability gap between frontier teachers and much smaller base models. Because a compact student model's natural, zero-shot responses are almost strictly inferior to the powerful teacher's targets, we can construct a highly effective contrastive signal simply by pairing the teacher's optimal response with a one-time static snapshot of the student's outputs. This demonstrates that exposing the small student to its own static inferior behaviors is sufficient for high-quality distribution alignment, eliminating the need for costly dynamic rollouts and fragile adversarial balancing. Extensive evaluations across four compact Qwen2.5 and Llama-3 models validate this semi on-policy paradigm. SODA matches or outperforms the state-of-the-art methods on 15 out of 16 benchmark results. More importantly, it achieves this superior distillation quality while training 10 times faster, consuming 27% less peak GPU memory, and completely eliminating adversarial instability.
>
---
#### [replaced 140] Pyramid MoA: A Probabilistic Framework for Cost-Optimized Anytime Inference
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Pyramid MoA，解决LLM推理中的成本优化问题，通过概率框架实现高效 anytime 推理。**

- **链接: [https://arxiv.org/pdf/2602.19509](https://arxiv.org/pdf/2602.19509)**

> **作者:** Arindam Khaled
>
> **备注:** 12 pages, 6 figures, 4 tables. v3: corrected router direction, added multi-benchmark context-aware escalation analysis, added Dean & Boddy and Horvitz citations
>
> **摘要:** We observe that LLM cascading and routing implicitly solves an anytime computation problem -- a class of algorithms, well-studied in classical AI, that improve solutions as additional computation is allocated. We formalize this connection and propose Pyramid MoA, a hierarchical Mixture-of-Agents architecture governed by a decision-theoretic router that escalates queries only when necessary. We establish a Probabilistic Anytime Property with provable monotonicity guarantees and derive a generalized escalation rule from Value of Computation theory that accounts for imperfect oracles, extending the Hansen-Zilberstein monitoring framework to stochastic LLM inference. On MBPP, the router intercepts 81.6% of bugs; on GSM8K/MMLU, the system nearly matches the 68.1% Oracle baseline while achieving up to 42.9% compute savings. The router transfers zero-shot to unseen benchmarks: matching Oracle accuracy on HumanEval (81.1%) and MATH 500 (58.0%) with significant cost reductions. We further discover a context-conditioned anchoring effect across four benchmarks: passing correct SLM reasoning improves Oracle accuracy by up to +19.2pp, while incorrect reasoning degrades it by up to -18.0pp, revealing a fundamental tension in hierarchical MoA architectures.
>
---
#### [replaced 141] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion
- **分类: cs.CL**

- **简介: 该论文属于多语言RAG系统任务，旨在解决语言偏好偏差问题。通过提出DeLP和DELTA，优化跨语言检索与生成，提升多语言系统性能。**

- **链接: [https://arxiv.org/pdf/2601.02956](https://arxiv.org/pdf/2601.02956)**

> **作者:** Jeonghyun Park; Byeongjeong Kim; Seojin Hwang; Hwanhee Lee
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Multilingual Retrieval-Augmented Generation (mRAG) systems often exhibit a perceived preference for high-resource languages, particularly English, resulting in the widespread adoption of English pivoting. While prior studies attribute this advantage to the superior English-centric capabilities of Large Language Models (LLMs), we find that such measurements are significantly distorted by structural priors inherent in evaluation benchmarks. Specifically, we identify exposure bias and a gold availability prior-both driven by the disproportionate concentration of resources in English-as well as cultural priors rooted in topic locality, as factors that hinder accurate assessment of genuine language preference. To address these biases, we propose DeLP (Debiased Language Preference), a calibrated metric designed to explicitly factor out these structural confounds. Our analysis using DeLP reveals that the previously reported English preference is largely a byproduct of evidence distribution rather than an inherent model bias. Instead, we find that retrievers fundamentally favor monolingual alignment between the query and the document language. Building on this insight, we introduce DELTA (DEbiased Language preference-guided Text Augmentation), a lightweight and efficient mRAG framework that strategically leverages monolingual alignment to optimize cross-lingual retrieval and generation. Experimental results demonstrate that DELTA consistently outperforms English pivoting and mRAG baselines across diverse languages.
>
---
#### [replaced 142] Domain-Specific Data Generation Framework for RAG Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出RAGen框架，用于生成领域特定的QAC三元组，解决RAG系统适应领域需求的问题。**

- **链接: [https://arxiv.org/pdf/2510.11217](https://arxiv.org/pdf/2510.11217)**

> **作者:** Chris Xing Tian; Weihao Xie; Zhen Chen; Zhengyuan Yi; Hui Liu; Haoliang Li; Shiqi Wang; Siwei Ma
>
> **备注:** To appear in ACL 2026
>
> **摘要:** Retrieval-Augmented Generation (RAG) combines the language understanding and reasoning power of large language models (LLMs) with external retrieval to enable domain-grounded responses. Effectively adapting RAG systems to domain-specific settings requires specialized, context-rich training data beyond general-purpose question-answering. Here, we propose RAGen, a scalable and modular framework for generating domain-grounded question-answer-context (QAC) triples tailored to diverse RAG adaptation approaches. RAGen produces these QAC triples by identifying key concepts in documents, generating diverse questions guided by Bloom's Taxonomy-inspired principles, and pairing them with precise answers extracted from relevant contexts. RAGen supports multiple RAG adaptation strategies, including the optimization of key components such as the LLM, retriever, and embedding model, etc. Its modular pipeline features semantic chunking, hierarchical concept extraction, and multi-chunk retrieval, along with the introduction of curated distractor contexts to promote robust reasoning. Designed for scalability, RAGen efficiently handles large and evolving document corpora without redundant processing, making it especially suitable for dynamic evolving domains such as scientific research and enterprise knowledge bases.
>
---
#### [replaced 143] Exploring Knowledge Purification in Multi-Teacher Knowledge Distillation for LLMs
- **分类: cs.CL**

- **简介: 该论文属于知识蒸馏任务，旨在解决多教师模型知识冲突和资源消耗高的问题。通过引入知识净化方法，提升蒸馏效果与效率。**

- **链接: [https://arxiv.org/pdf/2602.01064](https://arxiv.org/pdf/2602.01064)**

> **作者:** Ruihan Jin; Pengpeng Shao; Zhengqi Wen; Jinyang Wu; Mingkuan Feng; Shuo Yang; Chu Yuan Zhang; Jianhua Tao
>
> **备注:** ICLR 2026
>
> **摘要:** Knowledge distillation has emerged as a pivotal technique for transferring knowledge from stronger large language models (LLMs) to smaller, more efficient models. However, traditional distillation approaches face challenges related to knowledge conflicts and high resource demands, particularly when leveraging multiple teacher models. In this paper, we introduce the concept of \textbf{Knowledge Purification}, which consolidates the rationales from multiple teacher LLMs into a single rationale, thereby mitigating conflicts and enhancing efficiency. To investigate the effectiveness of knowledge purification, we further propose five purification methods from various perspectives. Our experiments demonstrate that these methods not only improve the performance of the distilled model but also effectively alleviate knowledge conflicts. Moreover, router-based methods exhibit robust generalization capabilities, underscoring the potential of innovative purification techniques in optimizing multi-teacher distillation and facilitating the practical deployment of powerful yet lightweight models.
>
---
#### [replaced 144] MCAT: Scaling Many-to-Many Speech-to-Text Translation with MLLMs to 70 Languages
- **分类: cs.CL**

- **简介: 该论文属于语音到文本翻译任务，旨在解决语言覆盖不足和效率低的问题。提出MCAT框架，支持70种语言互译，并优化语音序列长度，提升效率。**

- **链接: [https://arxiv.org/pdf/2512.01512](https://arxiv.org/pdf/2512.01512)**

> **作者:** Yexing Du; Kaiyuan Liu; Youcheng Pan; Bo Yang; Keqi Deng; Xie Chen; Yang Xiang; Ming Liu; Bing Qin; YaoWei Wang
>
> **备注:** Accepted in IEEE TASLP
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved great success in Speech-to-Text Translation (S2TT) tasks. However, current research is constrained by two key challenges: language coverage and efficiency. Most of the popular S2TT datasets are substantially English-centric, which restricts the scaling-up of MLLMs' many-to-many translation capabilities. Moreover, the inference speed of MLLMs degrades dramatically when the speech is converted into long sequences (e.g., 750 tokens). To address these limitations, we propose a Multilingual Cost-effective Accelerated Speech-to-Text Translator (MCAT) framework, which includes two innovations. First, a language scaling method that leverages curriculum learning and a data balancing strategy is introduced to extend the language coverage supported by MLLMs to 70 languages and achieve mutual translation among these languages. Second, an optimized speech adapter module is designed to reduce the length of the speech sequence to only 30 tokens. Extensive experiments were conducted on MLLMs of different scales (9B and 27B). The experimental results demonstrate that MCAT not only surpasses state-of-the-art end-to-end models on the FLEURS dataset across 70x69 directions but also enhances inference efficiency. The code and models are released at this https URL.
>
---
#### [replaced 145] Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出PerSyn，解决教师模型与学生模型不匹配的问题，通过路由机制为每个学生选择最佳教师，生成定制数据以提升学习效果。属于模型蒸馏任务。**

- **链接: [https://arxiv.org/pdf/2510.10925](https://arxiv.org/pdf/2510.10925)**

> **作者:** Hengyuan Zhang; Shiping Yang; Xiao Liang; Chenming Shang; Yuxuan Jiang; Chaofan Tao; Jing Xiong; Hayden Kwok-Hay So; Ruobing Xie; Angel X. Chang; Ngai Wong
>
> **备注:** ACL 2026 Main Conference
>
> **摘要:** Training student models on synthetic data generated by strong teacher models is a promising way to distilling the capabilities of teachers. However, recent studies show that stronger models are not always optimal teachers, revealing a mismatch between teacher outputs and student learnability. To address this issue, we propose PerSyn (Personalized data Synthesis), a novel synthesis strategy that operates under a new ``Route then Generate'' paradigm to create data tailored to each student model, enabling it to learn more effectively. Specifically, PerSyn first assigns each prompt to its optimal teacher via a query-level router that jointly considers student learnability and teacher response quality. Each teacher then synthesizes data only for its assigned prompts, making the process more efficient than the conventional ``Generate then Select'' paradigm, where all teachers must generate parallel responses for the entire prompt set before constructing the final dataset. Extensive experiments across different model families and scales demonstrate that PerSyn consistently achieves superior or comparable performance to all baselines in instruct tuning and math reasoning settings. Further analysis verifies the effectiveness of PerSyn and offers extra insights to propel future research.
>
---
#### [replaced 146] ChemPro: A Progressive Chemistry Benchmark for Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出ChemPro基准，用于评估大语言模型在化学领域的表现。任务是检测LLMs在不同难度化学问题上的推理与理解能力，解决其科学推理的局限性问题。**

- **链接: [https://arxiv.org/pdf/2602.03108](https://arxiv.org/pdf/2602.03108)**

> **作者:** Aaditya Baranwal; Shruti Vyas
>
> **备注:** Accepted at Artificial Intelligence Chemistry Journal
>
> **摘要:** We introduce ChemPro, a progressive benchmark with 4100 natural language question-answer pairs in Chemistry, across 4 coherent sections of difficulty designed to assess the proficiency of Large Language Models (LLMs) in a broad spectrum of general chemistry topics. We include Multiple Choice Questions and Numerical Questions spread across fine-grained information recall, long-horizon reasoning, multi-concept questions, problem-solving with nuanced articulation, and straightforward questions in a balanced ratio, effectively covering Bio-Chemistry, Inorganic-Chemistry, Organic-Chemistry and Physical-Chemistry. ChemPro is carefully designed analogous to a student's academic evaluation for basic to high-school chemistry. A gradual increase in the question difficulty rigorously tests the ability of LLMs to progress from solving basic problems to solving more sophisticated challenges. We evaluate 45+7 state-of-the-art LLMs, spanning both open-source and proprietary variants, and our analysis reveals that while LLMs perform well on basic chemistry questions, their accuracy declines with different types and levels of complexity. These findings highlight the critical limitations of LLMs in general scientific reasoning and understanding and point towards understudied dimensions of difficulty, emphasizing the need for more robust methodologies to improve LLMs.
>
---
#### [replaced 147] MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出MSA框架，解决长文本处理中内存容量与推理效率的瓶颈问题，实现100M token级高效记忆模型。**

- **链接: [https://arxiv.org/pdf/2603.23516](https://arxiv.org/pdf/2603.23516)**

> **作者:** Yu Chen; Runkai Chen; Sheng Yi; Xinda Zhao; Xiaohong Li; Jianjin Zhang; Jun Sun; Chuanrui Hu; Yunyun Han; Lidong Bing; Yafeng Deng; Tianqiao Chen
>
> **摘要:** Long-term memory is a cornerstone of human intelligence. Enabling AI to process lifetime-scale information remains a long-standing pursuit in the field. Due to the constraints of full-attention architectures, the effective context length of large language models (LLMs) is typically limited to 1M tokens. Existing approaches, such as hybrid linear attention, fixed-size memory states (e.g., RNNs), and external storage methods like RAG or agent systems, attempt to extend this limit. However, they often suffer from severe precision degradation and rapidly increasing latency as context length grows, an inability to dynamically modify memory content, or a lack of end-to-end optimization. These bottlenecks impede complex scenarios like large-corpus summarization, Digital Twins, and long-history agent reasoning, while limiting memory capacity and slowing inference. We present Memory Sparse Attention (MSA), an end-to-end trainable, efficient, and massively scalable memory model framework. Through core innovations including scalable sparse attention and document-wise RoPE, MSA achieves linear complexity in both training and inference while maintaining exceptional stability, exhibiting less than 9% degradation when scaling from 16K to 100M tokens. Furthermore, KV cache compression, combined with Memory Parallel, enables 100M-token inference on 2xA800 GPUs. We also propose Memory Interleaving to facilitate complex multi-hop reasoning across scattered memory segments. MSA significantly surpasses frontier LLMs, state-of-the-art RAG systems, and leading memory agents in long-context benchmarks. These results demonstrate that by decoupling memory capacity from reasoning, MSA provides a scalable foundation to endow general-purpose models with intrinsic, lifetime-scale memory.
>
---
#### [replaced 148] FlashMem: Distilling Intrinsic Latent Memory via Computation Reuse
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型缺乏动态上下文记忆的问题。通过计算复用提取内在记忆，提升推理效率与持久性。**

- **链接: [https://arxiv.org/pdf/2601.05505](https://arxiv.org/pdf/2601.05505)**

> **作者:** Yubo Hou; Zhisheng Chen; Tao Wan; Zengchang Qin
>
> **摘要:** The stateless architecture of Large Language Models inherently lacks the mechanism to preserve dynamic context, compelling agents to redundantly reprocess history to maintain long-horizon autonomy. While latent memory offers a solution, current approaches are hindered by architectural segregation, relying on auxiliary encoders that decouple memory from the reasoning backbone. We propose FlashMem, a framework that distills intrinsic memory directly from transient reasoning states via computation reuse. Leveraging the property that internal representations uniquely encode input trajectories, FlashMem identifies the last hidden state as a sufficient statistic for the interaction history. This enables a Shared-KV Consolidator to synthesize memory by attending directly to the backbone's frozen cache, eliminating redundant re-parameterization. Furthermore, a parameter-free Cognitive Monitor leverages attention entropy to adaptively trigger consolidation only when high epistemic uncertainty is detected. Experiments demonstrate that FlashMem matches the performance of heavy baselines while reducing inference latency by 5 times, effectively bridging the gap between efficiency and persistent cognition.
>
---
#### [replaced 149] CAMO: A Class-Aware Minority-Optimized Ensemble for Robust Language Model Evaluation on Imbalanced Data
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于文本分类任务，解决类别不平衡问题。提出CAMO方法，通过动态增强少数类提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.07583](https://arxiv.org/pdf/2604.07583)**

> **作者:** Mohamed Ehab; Ali Hamdi; Khaled Shaban
>
> **摘要:** Real-world categorization is severely hampered by class imbalance because traditional ensembles favor majority classes, which lowers minority performance and overall F1-score. We provide a unique ensemble technique for imbalanced problems called CAMO (Class-Aware Minority-Optimized).Through a hierarchical procedure that incorporates vote distributions, confidence calibration, and inter model uncertainty, CAMO dynamically boosts underrepresented classes while preserving and amplifying minority forecasts. We verify CAMO on two highly unbalanced, domain-specific benchmarks: the DIAR-AI/Emotion dataset and the ternary BEA 2025 dataset. We benchmark against seven proven ensemble algorithms using eight different language models (three LLMs and five SLMs) under zero-shot and fine-tuned settings .With refined models, CAMO consistently earns the greatest strict macro F1-score, setting a new benchmark. Its benefit works in concert with model adaptation, showing that the best ensemble choice depends on model properties .This proves that CAMO is a reliable, domain-neutral framework for unbalanced categorization.
>
---
#### [replaced 150] Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的工具调用任务，旨在解决企业API调用中因工具相似或参数不明确导致的错误。提出DiaFORGE框架，通过对话生成、微调和评估提升调用准确性。**

- **链接: [https://arxiv.org/pdf/2507.03336](https://arxiv.org/pdf/2507.03336)**

> **作者:** Ashutosh Hathidara; Julien Yu; Sebastian Schreiber
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Large language models (LLMs) are increasingly tasked with invoking enterprise APIs, yet they routinely falter when near-duplicate tools vie for the same user intent or when required arguments are left underspecified. We introduce DiaFORGE (Dialogue Framework for Organic Response Generation & Evaluation), a disambiguation-centric, three-stage pipeline that (i) synthesizes persona-driven, multi-turn dialogues in which the assistant must distinguish among highly similar tools, (ii) performs supervised fine-tuning of open-source models with reasoning traces across 3B - 70B parameters, and (iii) evaluates real-world readiness via a dynamic suite that redeploys each model in a live agentic loop and reports end-to-end goal completion alongside conventional static metrics. On our dynamic benchmark DiaBENCH, models trained with DiaFORGE raise tool-invocation success by 27 pp over GPT-4o and by 49 pp over Claude-3.5-Sonnet, both under optimized prompting. To spur further research, we release an open corpus of 5000 production-grade enterprise API specifications paired with rigorously validated, disambiguation-focused dialogues, offering a practical blueprint for building reliable, enterprise-ready tool-calling agents.
>
---
#### [replaced 151] IatroBench: Pre-Registered Evidence of Iatrogenic Harm from AI Safety Measures
- **分类: cs.AI; cs.CL; cs.CY; cs.LG**

- **简介: 该论文属于AI医疗安全评估任务，旨在检测AI在临床建议中的有意隐瞒问题。通过测试不同模型在医生与患者视角下的响应差异，揭示AI安全措施可能导致的医疗风险。**

- **链接: [https://arxiv.org/pdf/2604.07709](https://arxiv.org/pdf/2604.07709)**

> **作者:** David Gringras
>
> **备注:** 30 pages, 3 figures, 11 tables. Pre-registered on OSF (DOI: https://doi.org/10.17605/OSF.IO/G6VMZ). Code and data: this https URL. v2: Fix bibliography entries (add arXiv IDs, published venues); correct p-value typo in Limitations section; add AI Assistance Statement
>
> **摘要:** Ask a frontier model how to taper six milligrams of alprazolam (psychiatrist retired, ten days of pills left, abrupt cessation causes seizures) and it tells her to call the psychiatrist she just explained does not exist. Change one word ("I'm a psychiatrist; a patient presents with...") and the same model, same weights, same inference pass produces a textbook Ashton Manual taper with diazepam equivalence, anticonvulsant coverage, and monitoring thresholds. The knowledge was there; the model withheld it. IatroBench measures this gap. Sixty pre-registered clinical scenarios, six frontier models, 3,600 responses, scored on two axes (commission harm, CH 0-3; omission harm, OH 0-4) through a structured-evaluation pipeline validated against physician scoring (kappa_w = 0.571, within-1 agreement 96%). The central finding is identity-contingent withholding: match the same clinical question in physician vs. layperson framing and all five testable models provide better guidance to the physician (decoupling gap +0.38, p = 0.003; binary hit rates on safety-colliding actions drop 13.1 percentage points in layperson framing, p < 0.0001, while non-colliding actions show no change). The gap is widest for the model with the heaviest safety investment (Opus, +0.65). Three failure modes separate cleanly: trained withholding (Opus), incompetence (Llama 4), and indiscriminate content filtering (GPT-5.2, whose post-generation filter strips physician responses at 9x the layperson rate because they contain denser pharmacological tokens). The standard LLM judge assigns OH = 0 to 73% of responses a physician scores OH >= 1 (kappa = 0.045); the evaluation apparatus has the same blind spot as the training apparatus. Every scenario targets someone who has already exhausted the standard referrals.
>
---
#### [replaced 152] A Multilingual Dataset and Empirical Validation for the Mutual Reinforcement Effect in Information Extraction
- **分类: cs.CL**

- **简介: 该论文研究信息抽取中的互增强效应（MRE），解决多语言环境下MRE验证不足的问题。构建了多语言数据集MMM，并验证了76%的子集存在MRE，提升了信息抽取效果。**

- **链接: [https://arxiv.org/pdf/2407.10953](https://arxiv.org/pdf/2407.10953)**

> **作者:** Chengguang Gan; Sunbowen Lee; Qingyu Yin; Yunhao Liang; Xinyang He; Hanjun Wei; Younghun Lim; Shijian Wang; Hexiang Huang; Qinghao Zhang; Shiwen Ni; Tatsunori Mori
>
> **备注:** Accepted by ACL 2026 Findings
>
> **摘要:** The Mutual Reinforcement Effect (MRE) describes a phenomenon in information extraction where word-level and sentence-level tasks can mutually improve each other when jointly modeled. While prior work has reported MRE in Japanese, its generality across languages and task settings has not been empirically validated, largely due to the lack of multilingual MRE datasets. To address this limitation, we introduce the Multilingual MRE Mix dataset (MMM), which consists of 21 sub-datasets covering English, Japanese, and Chinese. We propose an LLM-assisted dataset translation and alignment framework that significantly reduces manual annotation effort while preserving the structural requirements of MRE tasks. Building on MMM, we adopt a unified input-output framework to train an open-domain information extraction model and conduct extensive empirical studies, including full fine-tuning ablations and the construction of knowledgeable verbalizers based on MRE-mix data. Experimental results show that 76 percent of the MMM sub-datasets consistently exhibit the Mutual Reinforcement Effect across languages. These findings provide systematic empirical validation of MRE in multilingual settings and demonstrate its practical value for information extraction.
>
---
#### [replaced 153] VeriInteresting: An Empirical Study of Model Prompt Interactions in Verilog Code Generation
- **分类: cs.AR; cs.CL**

- **简介: 该论文属于代码生成任务，研究模型与提示的交互影响。旨在探索不同模型对提示设计的响应模式，通过实验分析模型特性与提示策略的关系。**

- **链接: [https://arxiv.org/pdf/2603.08715](https://arxiv.org/pdf/2603.08715)**

> **作者:** Luca Collini; Andrew Hennesee; Patrick Yubeaton; Siddharth Garg; Ramesh Karri
>
> **备注:** Submitted for peer review
>
> **摘要:** Rapid advances in language models (LMs) have created new opportunities for automated code generation while complicating trade-offs between model characteristics and prompt design choices. In this work, we provide an empirical map of recent trends in LMs for Verilog code generation, focusing on interactions among model reasoning, specialization, and prompt engineering strategies. We evaluate a diverse set of small and large LMs, including general-purpose, reasoning, and domain-specific variants. Our experiments use a controlled factorial design spanning benchmark prompts, structured outputs, prompt rewriting, chain-of-thought reasoning, in-context learning, and evolutionary prompt optimization via Genetic-Pareto. Across two Verilog benchmarks, we identify patterns in how model classes respond to structured prompts and optimization, and we document which trends generalize across LMs and benchmarks versus those that are specific to particular model-prompt combinations.
>
---
#### [replaced 154] Cross-Tokenizer LLM Distillation through a Byte-Level Interface
- **分类: cs.CL**

- **简介: 该论文属于跨分词器知识蒸馏任务，旨在解决不同分词器间模型知识迁移的问题。提出Byte-Level Distillation方法，通过字节级接口实现有效蒸馏。**

- **链接: [https://arxiv.org/pdf/2604.07466](https://arxiv.org/pdf/2604.07466)**

> **作者:** Avyav Kumar Singh; Yen-Chen Wu; Alexandru Cioba; Alberto Bernacchia; Davide Buffelli
>
> **摘要:** Cross-tokenizer distillation (CTD), the transfer of knowledge from a teacher to a student language model when the two use different tokenizers, remains a largely unsolved problem. Existing approaches rely on heuristic strategies to align mismatched vocabularies, introducing considerable complexity. In this paper, we propose a simple but effective baseline called Byte-Level Distillation (BLD) which enables CTD by operating at a common interface across tokenizers: the byte level. In more detail, we convert the teacher's output distribution to byte-level probabilities, attach a lightweight byte-level decoder head to the student, and distill through this shared byte-level interface. Despite its simplicity, BLD performs competitively with--and on several benchmarks surpasses--significantly more sophisticated CTD methods, across a range of distillation tasks with models from 1B to 8B parameters. Our results suggest that the byte level is a natural common ground for cross-tokenizer knowledge transfer, while also highlighting that consistent improvements across all tasks and benchmarks remain elusive, underscoring that CTD is still an open problem.
>
---
#### [replaced 155] Many-Tier Instruction Hierarchy in LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于智能体指令冲突解决任务，旨在处理多源指令的优先级问题。提出ManyIH框架及基准测试，解决大规模指令冲突下的安全有效执行问题。**

- **链接: [https://arxiv.org/pdf/2604.09443](https://arxiv.org/pdf/2604.09443)**

> **作者:** Jingyu Zhang; Tianjian Li; William Jurayj; Hongyuan Zhan; Benjamin Van Durme; Daniel Khashabi
>
> **摘要:** Large language model agents receive instructions from many sources-system messages, user prompts, tool outputs, other agents, and more-each carrying different levels of trust and authority. When these instructions conflict, agents must reliably follow the highest-privilege instruction to remain safe and effective. The dominant paradigm, instruction hierarchy (IH), assumes a fixed, small set of privilege levels (typically fewer than five) defined by rigid role labels (e.g., system > user). This is inadequate for real-world agentic settings, where conflicts can arise across far more sources and contexts. In this work, we propose Many-Tier Instruction Hierarchy (ManyIH), a paradigm for resolving instruction conflicts among instructions with arbitrarily many privilege levels. We introduce ManyIH-Bench, the first benchmark for ManyIH. ManyIH-Bench requires models to navigate up to 12 levels of conflicting instructions with varying privileges, comprising 853 agentic tasks (427 coding and 426 instruction-following). ManyIH-Bench composes constraints developed by LLMs and verified by humans to create realistic and difficult test cases spanning 46 real-world agents. Our experiments show that even the current frontier models perform poorly (~40% accuracy) when instruction conflict scales. This work underscores the urgent need for methods that explicitly target fine-grained, scalable instruction conflict resolution in agentic settings.
>
---
#### [replaced 156] SynthAgent: Adapting Web Agents with Synthetic Supervision
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出SynthAgent，解决Web代理适应新网站时合成数据质量低的问题，通过任务与轨迹的双重优化提升数据质量。**

- **链接: [https://arxiv.org/pdf/2511.06101](https://arxiv.org/pdf/2511.06101)**

> **作者:** Zhaoyang Wang; Yiming Liang; Xuchao Zhang; Qianhui Wu; Siwei Han; Anson Bastos; Rujia Wang; Chetan Bansal; Baolin Peng; Jianfeng Gao; Saravan Rajmohan; Huaxiu Yao
>
> **备注:** Accepted to ACL 2026 Main Conference
>
> **摘要:** Web agents struggle to adapt to new websites due to the scarcity of environment specific tasks and demonstrations. Recent works have explored synthetic data generation to address this challenge, however, they suffer from data quality issues where synthesized tasks contain hallucinations that cannot be executed, and collected trajectories are noisy with redundant or misaligned actions. In this paper, we propose SynthAgent, a fully synthetic supervision framework that aims at improving synthetic data quality via dual refinement of both tasks and trajectories. Our approach begins by synthesizing diverse tasks through categorized exploration of web elements, ensuring efficient coverage of the target environment. During trajectory collection, tasks are refined only when conflicts with observations are detected, which mitigates hallucinations while preserving task consistency. After collection, we conduct trajectory refinement with global context to mitigate potential noise or misalignments. Finally, we fine-tune open-source web agents on the refined synthetic data to adapt them to the target environment. Experimental results demonstrate that SynthAgent outperforms existing synthetic data methods, validating the importance of high-quality synthetic supervision. The code is publicly available at this https URL.
>
---
#### [replaced 157] SCITUNE: Aligning Large Language Models with Human-Curated Scientific Multimodal Instructions
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于科学任务中的指令微调研究，旨在提升大语言模型对科学多模态指令的遵循能力。通过引入人类标注的科学多模态数据，改进模型在科学问答等任务上的表现。**

- **链接: [https://arxiv.org/pdf/2307.01139](https://arxiv.org/pdf/2307.01139)**

> **作者:** Sameera Horawalavithana; Sai Munikoti; Ian Stewart; Henry Kvinge; Karl Pazdernik
>
> **备注:** In Proceedings of the 1st Workshop on NLP for Science, Association for Computational Linguistics
>
> **摘要:** Instruction finetuning is a popular paradigm to align large language models (LLM) with human intent. Despite its popularity, this idea is less explored in improving LLMs to align existing foundation models with scientific disciplines, concepts and goals. In this work, we present \textit{SciTune} as a tuning framework to improve the ability of LLMs to follow multimodal instructions generated from scientific publications. To test our methodology, we train a large multimodal model LLaMA-SciTune that connects a vision encoder and LLM for science-focused visual and language understanding. LLaMA-SciTune significantly outperforms the state-of-the-art models in the generated figure types and captions in SciCap and VisText benchmarks. In comparison to the models that are finetuned with synthetic data only, LLaMA-SciTune surpasses human performance on average and in many sub-categories on the ScienceQA benchmark. Our results demonstrate that human-generated scientific multimodal instructions remain highly valuable in tuning LLMs to perform well on science tasks, despite their lower volume and relative scarcity compared to synthetic data. We publicly release the SciTune codebase this https URL.
>
---

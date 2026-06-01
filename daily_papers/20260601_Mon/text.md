# 自然语言处理 cs.CL

- **最新发布 156 篇**

- **更新 98 篇**

## 最新发布

#### [new 001] CobSeg: Coherence Boundary Modeling for Dialogue Topic Segmentation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话主题分割任务，解决如何准确识别对话中的边界问题。提出CobSeg模型，通过分离语义连续性和词汇边界信号，提升分割效果。**

- **链接: [https://arxiv.org/pdf/2605.30668](https://arxiv.org/pdf/2605.30668)**

> **作者:** Sijin Sun; Liangbin Zhao; Jiaxiang Cai; Ming Deng; Mingyu Luo; Xiuju Fu
>
> **备注:** 8 pages with appindx. Under review
>
> **摘要:** Dialogue topic segmentation is critical in many human-AI collaborative applications which requires identifying heterogeneous boundary cues, including lexical transitions near utterance edges and semantic discontinuities across utterances. Existing utterance models often dilute these local lexical signals. We propose CobSeg, a novel multi-branch architecture that separates coherence-level semantic continuity from lexical boundary transitions and recovers both through directional boundary prediction. CobSeg further uses boundary informativeness weighting to emphasize high-utility utterance positions, and incorporates a corpus-derived topic coherence cue with learned combination weights. While CobSeg is evaluated as a compact trainable segmenter under supervised gold-boundary training and a pseudo-label setting with automatically induced boundaries, it performs enhanced boundary prediction without LLM calls during inference. Across five benchmarks, it improves $P_k$ and $W_d$ particularly when local lexical cues are prominent: under gold supervision, it reduces $P_k$ by 0.7 points and $W_d$ by 0.6 points on VHF, and reaches $P_k$ of 1.0 on DialSeg711; with induced boundaries, it reduces $P_k$ by 14.8 points on VHF, by 1.5 points on DialSeg711, and by 1.1 points on TIAGE, outperforming prior non-LLM approaches.
>
---
#### [new 002] Efficient Diffusion LLMs via Temporal-Spatial Parallel Decoding and Confidence Extrapolation
- **分类: cs.CL**

- **简介: 该论文针对扩散型大语言模型的推理效率问题，提出时空并行解码和置信度外推方法，减少冗余计算，提升生成速度。**

- **链接: [https://arxiv.org/pdf/2605.30753](https://arxiv.org/pdf/2605.30753)**

> **作者:** Zekai Li; Ji Liu; Yiqing Huang; Ziqiong Liu; Dong Li; Emad Barsoum
>
> **摘要:** Diffusion-based large language models (dLLMs) support parallel text generation via iterative denoising, yet inference remains latency-heavy because many steps are spent on redundant refinement and repeated remasking of tokens whose final values are already determined. Prior acceleration methods mainly depend on step-local confidence heuristics or fixed schedules, which are sensitive to prompt and task variation and ignore strong positional effects within a sequence. We cast diffusion decoding as a dynamic control problem and show that token-wise denoising trajectories provide the key signal for reliable control. We propose a trace-aware decoding framework with two components. First, Temporal-Spatial Parallel Decoding (TSPD) uses a lightweight temporalspatial controller that consumes per-token trajectory features, including confidence, entropy, and momentum, together with token position, to decide when a token has converged and can be safely fixed. Second, we introduce Confidence Extrapolation (CE), a training-free state-space module that forecasts future logit trends with uncertainty to support proactive decisions, including safe look-ahead and targeted stabilization when trajectories are oscillatory or underconfident. Together, TSPD and CE reduce unnecessary denoising iterations while preserving output quality, and they compose cleanly with system optimizations such as KV caching.
>
---
#### [new 003] ImmigrationQA: A Source-Grounded Dataset and Small-Model Adaptation for U.S. Immigration Law
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ImmigrationQA数据集及小模型适配，解决美国移民法问答问题。通过构建17,058对问答数据，微调Llama 3.2模型，提升迁移学习效果。**

- **链接: [https://arxiv.org/pdf/2605.30589](https://arxiv.org/pdf/2605.30589)**

> **作者:** Nazarii Shportun
>
> **备注:** 12 pages, 4 tables. Dataset (17,058 QA pairs), fine-tuned model, and code are publicly released
>
> **摘要:** U.S. immigration law spans thousands of pages of official policy, federal regulations, and procedural guidance that change frequently and carry high stakes for petitioners who lack legal representation. We describe the construction of ImmigrationQA, a source-grounded question-answering dataset of 17,058 pairs across 13 immigration subdomains, and the fine-tuning of a Llama 3.2 3B Instruct model on that dataset using parameter-efficient LoRA. The corpus was assembled from 11 primary and secondary sources -- including the USCIS Policy Manual, 8 CFR, BIA precedent decisions, and community Q&A -- yielding 10,056 validated canonical documents and 18,308 text chunks. Structured QA pairs were generated from these chunks using Claude Sonnet 4.6 via five mode-specific prompts, with 22 pairs rejected for insufficient source-span overlap. The fine-tuned model was evaluated against a held-out split of 993 pairs using LLM-as-judge scoring on a 101-example stratified sample. The fine-tuned model scored a mean of 1.08/3.0 (16.8% fully correct; 101-example stratified eval) versus the Llama 3 8B base model at 0.85/3.0 (4% fully correct), a relative improvement of 27% in mean score; a zero-shot Claude Sonnet baseline scored 1.52/3.0 (25% fully correct). The fine-tuned model shows concentrated improvement in procedural subdomains (travel documents, adjustment of status, nonimmigrant visas) while remaining weak on complex legal reasoning and time-sensitive statistics. The full pipeline ran for approximately $29 in cloud compute. All artifacts -- dataset, model, code, and prompt templates -- are publicly released. The system is not a substitute for legal counsel and does not reflect regulatory changes after the corpus crawl date.
>
---
#### [new 004] Combinatorial Synthesis: Scaling Code RLVR via Atomic Decomposition and Recombination
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于代码生成任务，旨在解决RLVR中数据稀缺与难度不足的问题。提出ADR框架，通过分解和重组生成高质量代码任务。**

- **链接: [https://arxiv.org/pdf/2605.31058](https://arxiv.org/pdf/2605.31058)**

> **作者:** Jiasheng Zheng; Boxi Cao; Boxi Yu; Yuzhong Zhang; Jialun Cao; Yaojie Lu; Hongyu Lin; Xianpei Han; Le Sun
>
> **备注:** Work in progress
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as the cornerstone for shaping the remarkable coding abilities of Large Language Models (LLMs). However, the scalability of RLVR is severely constrained by the scarcity of sufficiently challenging verifiable code tasks that target near the model's edge of competence. Prior studies often rely on heuristic seed expansions for data synthesis, which severely limits both novelty and difficulty. Consequently, the training value of such data fails to scale proportionally with the size of its synthesis. To this end, we propose Atomic Decomposition and Recombination (ADR), a novel framework that generates verifiable code tasks via decomposition into atomic elements and controlled recombination, thereby enabling the generation of genuinely novel and challenging verifiable code tasks. Experiments and analysis demonstrate that ADR achieves superior originality, difficulty, diversity, and test quality over existing baselines, and consistently delivers greater improvements in code ability across RLVR in diverse downstream domains, including algorithmic programming, tool usage, and data science. Our work sheds light on a new paradigm for novel code task synthesis and scalable RLVR training.
>
---
#### [new 005] EMBGuard: Constructing Hazard-Aware Guardrails for Safe Planning in Embodied Agents
- **分类: cs.CL**

- **简介: 该论文属于机器人安全规划任务，旨在解决 embodied agents 面临的物理风险识别问题。提出 EMBGuard 系统，通过分析动作与视觉输入，识别危险并提供解释，提升安全性。**

- **链接: [https://arxiv.org/pdf/2605.30924](https://arxiv.org/pdf/2605.30924)**

> **作者:** Dongwook Choi; Taeyoon Kwon; Bogyung Jeong; Minju Kim; Yeonjun Hwang; Hyojun Kim; Byungchul Kim; Young Kyun Jang; Jinyoung Yeo
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** MLLM-powered embodied agents deployed in real-world environments encounter physical hazards. However, existing approaches lack explicit mechanisms for identifying hazards and reasoning about action-conditioned risks, leading agents to either miss risky interactions or over-identify risks. To address this, we propose EMBGuard, the first MLLM-based safety guardrail for embodied agents designed to decouple physical risk reasoning from agent policy. By evaluating a (visual observation, action) pair, EMBGuard identifies hazardous configurations and provides natural language explanations of potential risks. Alongside EMBGuard, we contribute EMBHazard, a training dataset of 15.1K action-conditioned pairs, and EMBGuardTest, a benchmark of 329 manually curated real-world scenarios spanning seven physical risk categories. Through compositional variation of hazards and actions, we generate diverse risky and benign scenarios that agents may encounter during planning. Despite its compact size (2B, 4B), EMBGuard achieves performance competitive with proprietary MLLMs (e.g., GPT-5.1, Gemini-2.5-Pro) while significantly reducing the false-positive rates that hinder real-time deployment. We make the code, data, and models publicly available at this https URL
>
---
#### [new 006] EvoGens: A Population-Based Heuristic Search Framework for Scientific Idea Generation
- **分类: cs.CL**

- **简介: 论文提出EvoGens框架，用于科学创意生成。该任务旨在提升研究想法的创新性和多样性。通过进化搜索机制，解决现有方法语义趋同的问题。**

- **链接: [https://arxiv.org/pdf/2605.30961](https://arxiv.org/pdf/2605.30961)**

> **作者:** Xu Li; Hanzhe Tu; Xinyi Li; Kuncheng Zhao; Xun Han; Zhonghui Liu
>
> **备注:** 21 pages, 6 figures
>
> **摘要:** Generating novel research ideas is fundamental to scientific progress. While Large Language Models (LLMs) show promise in assisting this process, existing approaches often exhibit semantic convergence, resulting in limited diversity and novelty. To address this, we introduce EvoGens, an evolution-inspired framework that recasts scientific idea generation as an evolutionary search over a population of ideas. EvoGens iteratively applies rank-based mutation with differentiated retrieval planning to incorporate external knowledge, and semantic-aware crossover to fuse complementary concepts for conceptual reorganization. A lightweight evaluation signal guides the selection process, encouraging sustained exploration while mitigating premature convergence. Extensive experiments demonstrate that EvoGens substantially enhances exploration capabilities compared to state-of-the-art baselines. Specifically, it improves the Novelty from 0.1 to 0.4 and the Diversity from 0.24 to 0.55, while maintaining comparable idea quality under the current automatic evaluation protocol. These findings suggest that evolutionary mechanisms can serve as a useful framework for exploration-oriented research ideation, especially for broadening the novelty and diversity of candidate ideas under a shared automatic evaluation setting.
>
---
#### [new 007] DOA: Training-Free Decoder-Only Attention Policy for Long-Form Simultaneous Translation with SpeechLLMs
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音翻译任务，解决长文本实时翻译中对齐信号不足的问题。提出DOA方法，利用自注意力生成对齐信号，实现无需训练的高效翻译。**

- **链接: [https://arxiv.org/pdf/2605.31432](https://arxiv.org/pdf/2605.31432)**

> **作者:** Sara Papi; Luisa Bentivogli
>
> **摘要:** Simultaneous speech-to-text translation (SimulST) generates translations while speech is still unfolding, requiring a streaming policy that decides when to read and when to write. State-of-the-art approaches rely on attention-based encoder-decoder models where cross-attention provides explicit alignment signals. In contrast, Speech Large Language Models (SpeechLLMs) are decoder-only architectures relying solely on self-attention. This raises a central question: whether decoder self-attention contains sufficiently stable alignment signals to guide the streaming policy. Moreover, existing approaches typically rely on training-based adaptations or heuristic wait-$k$ policies and have not been validated in long-form settings. To fill these gaps, we propose Decoder-Only Attention (DOA), a training-free policy that enables long-form simultaneous translation with off-the-shelf SpeechLLMs by deriving a proxy alignment from self-attention. Experiments on Phi4-Multimodal and Qwen3-Omni show that DOA provides an effective alignment signal for supporting streaming decisions, enabling low-latency long-form SimulST with quality close to offline decoding without retraining.
>
---
#### [new 008] Generating and Refining Dynamic Evaluation Rubrics for LLM-as-a-Judge
- **分类: cs.CL**

- **简介: 该论文属于自动评估任务，旨在无需人工标注生成评价标准。提出一种无需训练的方法生成细粒度评价 rubric，并通过迭代优化提升性能。**

- **链接: [https://arxiv.org/pdf/2605.30568](https://arxiv.org/pdf/2605.30568)**

> **作者:** Zijie Wang; Eduardo Blanco
>
> **摘要:** LLM-as-a-Judge is a scalable alternative to human evaluation, yet existing rubric-based methods rely on human-annotated data such as reference answers or expert-crafted rubrics. We propose to automatically generate fine-grained evaluation rubrics without any human annotation. Our training-free method generates rubrics at dataset-specific and instance-specific granularities, achieving performance competitive with existing methods across four benchmarks. We further present a method that iteratively fine-tunes a rubric generator model via meta-judge reward signals. The fine-tuned generator outperforms all existing baselines in both pairwise and pointwise evaluation. Notably, a fine-tuned 14B rubric generator outperforms a much larger proprietary model at rubric generation, showing the effectiveness of our fine-tuning strategy.
>
---
#### [new 009] Probing the Prompt KV Cache: Where It Becomes Dispensable
- **分类: cs.CL**

- **简介: 该论文研究KV缓存冗余性，探讨在解码过程中何时及如何替换提示部分的KV缓存而不影响任务性能。**

- **链接: [https://arxiv.org/pdf/2605.30574](https://arxiv.org/pdf/2605.30574)**

> **作者:** Vinayshekhar Bannihatti Kumar; Manoj Ghuhan Arivazhagan; Disha Makhija; Rashmi Gangadharaiah
>
> **摘要:** Prior KV cache compression schemes empirically demonstrate that the prompt cache is partially redundant during decoding, dropping or summarising entries with little accuracy loss. We ask when and what kind of redundancy: at which layers, after how many decoding steps, and in what form can the prompt span KV cache be replaced without breaking the task. A controlled splice intervention swept over layer cutoff and decoding steps shows this redundancy is about form (chat template scaffolding) rather than content. Replacing the upper layer prompt span KV cache with KV cache from a chat template scaffold whose user content is a neutral filler recovers near clean accuracy, while zeroing the same slots collapses accuracy. The dissociation replicates across the Qwen3, Gemma 3, and Llama 3 families on multiple datasets.
>
---
#### [new 010] D$^3$: Dynamic Directional Graph-Constrained Data Scheduling for LLM Training
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型训练任务，旨在解决数据调度问题。通过构建动态方向图约束，优化训练顺序以提升效率。**

- **链接: [https://arxiv.org/pdf/2605.31164](https://arxiv.org/pdf/2605.31164)**

> **作者:** Yuanjian Xu; Jianing Hao; Guang Zhang; Zhong Li
>
> **摘要:** Training data plays a central role in large language models (LLMs) optimization, motivating extensive research on data scheduling strategies. Most existing approaches concentrate on adjusting the overall data distribution but neglect the underlying interactions between samples during training. However, we argue that such interactions cannot be overlooked, as real-world data samples frequently exhibit directional influences on each other, making the training order crucial. Intuitively, we can prioritize train-units with greater influence to improves learning efficiency. In this work, we propose $D^3$, a Dynamic Directional graph-constrained Data scheduling framework. $D^3$ formulates the complex interactions among train-units as a dynamic influence graph, where edges represent loss-based dependencies. It then solves a constrained optimization problem over this graph to derive the training order, which ensures that the data sequence respects the evolving information flow throughout training. Our approach is theoretically motivated and yields consistent improvements over existing data scheduling methods across both pre-training and post-training phases. Furthermore, for scalability, $D^3$ also employs an efficient approximation algorithm that keeps the additional computational overhead within a manageable range. For future research, the code is available at this https URL.
>
---
#### [new 011] Domain Adaptation and Reasoning Frameworks in Language Models: A Controlled Experiment with Historical Cosmology
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的领域自适应任务，研究领域适应如何改变语言模型的解释行为。通过实验探讨模型在历史宇宙学语境下的推理框架变化。**

- **链接: [https://arxiv.org/pdf/2605.30415](https://arxiv.org/pdf/2605.30415)**

> **作者:** Francesco De Bernardis
>
> **备注:** 17 pages, 3 figures
>
> **摘要:** We investigate how domain adaptation reshapes explanatory behavior in language models using historical cosmology as a controlled setting. In Phase 1, we train a small language model from scratch on a pre-Copernican corpus from which explicit heliocentric references were removed, and evaluate whether Earth-motion or heliocentric continuations nevertheless emerge. In Phase 2, we fine-tune a larger pretrained model using QLoRA on the same corpus in order to study how adaptation modifies explanatory framing and cosmological stance. Model outputs are evaluated using an LLM-as-judge framework that labels both cosmological stance (geocentric, heliocentric, or ambiguous) and explanatory frame (premodern versus modern). In the constrained setting of Phase 1, the smaller models occasionally generate local Earth-motion continuations, but these remain globally unstable and insufficient to support coherent cosmological reasoning. In Phase 2, fine-tuning induces a large and statistically significant shift toward premodern explanatory framing, while the conditional cosmological stance distributions remain comparatively stable within those frames. As a result, increases in geocentric outputs arise primarily from redistribution over explanatory regimes rather than from direct modification of stance. These results suggest that domain adaptation may primarily reshape the linguistic frameworks from which continuations are generated, with changes in stance emerging secondarily from those shifts.
>
---
#### [new 012] MineExplorer: Evaluating Open-World Exploration of MLLM Agents in Minecraft
- **分类: cs.CL**

- **简介: 该论文属于多模态大语言模型的开放世界探索任务，旨在评估MLLM在Minecraft中的持续探索能力。通过构建MineExplorer基准，解决现有评估方法不足的问题，提出多智能体合成流程以生成可靠任务实例。**

- **链接: [https://arxiv.org/pdf/2605.30931](https://arxiv.org/pdf/2605.30931)**

> **作者:** Tianjie Ju; Yueqing Sun; Zheng Wu; Wei Zhang; Yaqi Huo; Xi Su; Qi Gu; Xunliang Cai; Gongshen Liu; Zhuosheng Zhang
>
> **备注:** Working in progress
>
> **摘要:** Multimodal large language models (MLLMs) have shown strong capabilities in perception, reasoning, and action generation. However, their ability to sustain exploration in dynamic open worlds remains unclear. Existing embodied and game-based benchmarks often compress interaction into short-horizon tasks or entangle success with domain-specific game mechanics. In this paper, we introduce MineExplorer benchmark for evaluating open-world exploration capabilities of MLLM agents in Minecraft. We first filter atomic tasks whose solutions rely heavily on Minecraft-specific knowledge to better reflect general open-world reasoning. Then we organize the benchmark around a ReAct-style capability formulation and compose atomic tasks into implicit multi-hop tasks. To further construct reliable instances, MineExplorer uses a multi-agent synthesis workflow that jointly designs task graphs, sandbox scenes, and rule-based milestone evaluators. Human evaluation shows that the multi-agent synthesis workflow produces significantly more reliable instances than a single-agent baseline. Experiments with advanced MLLM agents show that open-world exploration remains challenging, as strong models can handle many single-hop tasks but degrade sharply when hidden prerequisites must be coordinated over longer trajectories. Further analysis finds that task difficulty tracks agent completion, and larger models or thinking modes do not consistently translate into better performance. Code and dataset are available at this https URL.
>
---
#### [new 013] MosaicLeaks:Privacy Risks in Querying-in-the-Open for Deep Research Agents
- **分类: cs.CL**

- **简介: 该论文属于隐私保护任务，解决深度研究代理在查询外部信息时泄露本地敏感信息的问题。通过构建基准测试和提出PA-DR框架，提升隐私保护效果。**

- **链接: [https://arxiv.org/pdf/2605.30727](https://arxiv.org/pdf/2605.30727)**

> **作者:** Alexander Gurung; Spandana Gella; Alexandre Drouin; Issam H. Laradji; Perouz Taslakian; Rafael Pardinas
>
> **摘要:** Deep research agents increasingly combine private local documents with external tools like web retrieval, creating a privacy risk: an agent's external queries may leak sensitive information from its local context. This risk is amplified by the mosaic effect, where individual queries may appear harmless but become revealing in aggregate. We introduce MosaicLeaks, a benchmark of 1,001 multi-hop deep research tasks that chain private enterprise documents and a public web corpus, forcing agents to make external queries that depend on local information. We evaluate leakage with an adversary LLM that observes only the agent's external queries and attempts to infer private information at three levels: the agent's research intent, answers to specific private questions and verifiable claims about the enterprise documents. We find that models across families and sizes frequently leak at all three levels, that zero-shot privacy prompting reduces but does not eliminate leakage and that reinforcement learning for task performance alone worsens leakage. To address this, we propose Privacy-Aware Deep Research (PA-DR), an RL framework that combines situational rewards for task success with a learned privacy classifier to provide dense credit assignment over both per-query and mosaic-level leakage. Training Qwen3-4B-Instruct with PA-DR improves accuracy from 48.7% to 58.7% and reduces answer and full-information leakage from 34.0% to 9.9%.
>
---
#### [new 014] Speculative Pipeline Decoding: Higher-Accruacy and Zero-Bubble Speculation via Pipeline Parallelism
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理加速任务，解决传统方法预测难度高、延迟大的问题。提出SPD框架，通过流水线并行提升解码效率，实现零延迟和高接受率。**

- **链接: [https://arxiv.org/pdf/2605.30852](https://arxiv.org/pdf/2605.30852)**

> **作者:** Yijiong Yu; Huazheng Wang; Shuai Yuan; Ruilong Ren; Ji Pei
>
> **摘要:** Speculative Decoding (SD) accelerates low-concurrency LLM inference by employing a draft-then-verify paradigm. However, mainstream methods typically rely on multi-token prediction, which introduces escalating prediction difficulty and serial drafting latency. To address these, we propose Speculative Pipeline Decoding (SPD), a groundbreaking framework that unlocks the true potential of pipeline parallelism. By partitioning the target LLM into $n$ pipeline stages, SPD allows LLM to process $n$ tokens in parallel to accelerate decoding. To continuous fill the pipeline in single sequence decoding, a speculation module aggregates intermediate features across different pipeline depths to predict the next token, executing strictly in parallel with the target model's pipeline step, to realize bounded difficulty, higher acceptance rates, and zero latency bubbles. Our experiments demonstrate that SPD achieves a significantly higher theoretical speedup compared to mainstream baselines, offering a highly scalable solution for LLM decoding acceleration. Our code is available at this https URL
>
---
#### [new 015] LongTraceRL: Learning Long-Context Reasoning from Search Agent Trajectories with Rubric Rewards
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于长文本推理任务，旨在解决大语言模型在长上下文中定位和整合关键信息的难题。通过构建更复杂的干扰项和设计细粒度奖励机制，提升模型的推理能力。**

- **链接: [https://arxiv.org/pdf/2605.31584](https://arxiv.org/pdf/2605.31584)**

> **作者:** Nianyi Lin; Jiajie Zhang; Lei Hou; Juanzi Li
>
> **摘要:** Long-context reasoning remains a central challenge for large language models, which often fail to locate and integrate key information in extensive distracting content. Reinforcement learning with verifiable rewards (RLVR) has shown promise for this task, yet existing methods are limited by low-confusability distractors and sparse, outcome-only reward signals that cannot supervise intermediate reasoning steps. To address these issues, we introduce \textsc{LongTraceRL}. For data construction, we generate multi-hop questions via knowledge graph random walks and leverage search agent trajectories to build \emph{tiered distractors}: documents the agent read but did not cite (high confusability) and documents that appeared in search results but were never opened (low confusability), producing training contexts that are far more challenging than those built by random sampling or one-shot search. For reward design, we propose a \emph{rubric reward} that uses the gold entities along each reasoning chain as fine-grained, entity-level process supervision. This rubric reward is applied only to responses with correct final answers (positive-only strategy), distinguishing the reasoning quality among correct responses and preventing reward hacking. Experiments on three reasoning LLMs (4B--30B) across five long-context benchmarks demonstrate that \textsc{LongTraceRL} consistently outperforms strong baselines and encourages comprehensive, evidence-grounded reasoning. Codes, datasets and models are available at \href{this https URL}{this https URL}.
>
---
#### [new 016] Can LLM Teams Play What? Where? When?
- **分类: cs.CL**

- **简介: 该论文研究LLM团队在问答任务中的表现，旨在提升模型的推理与协作能力。通过三种团队策略实验，发现团队合作能显著提高准确率，但主要作为答案筛选而非创新生成。**

- **链接: [https://arxiv.org/pdf/2605.30459](https://arxiv.org/pdf/2605.30459)**

> **作者:** Anastasia Kotelnikova; Viktor Byzov; Maria Dolzhenkova; Evgeny Kotelnikov
>
> **备注:** Accepted for Dialogue-2026 conference
>
> **摘要:** Large language models (LLMs) remain limited on tasks requiring indirect reasoning, cultural knowledge, and coordinated hypothesis testing. We investigate whether team-based interaction improves LLM performance in What? Where? When? (ChGK), a quiz game designed to reward collective reasoning. We introduce three team strategies: Voting, Silent Team (the captain observes final answers), and Talkative Team (the captain observes both answers and rationales). To minimize data leakage, we evaluate these strategies on a dataset consisting of 572 ChGK questions released in 2025. Using six recent large-scale open models, we show that team-based strategies outperform single-model baselines, yielding gains of up to 20 percentage points in accuracy. The best team achieves 44.23% accuracy, and approaches human team performance on questions with available human statistics. Analysis of inter-model diversity reveals that disagreement strongly predicts lower accuracy, but explanatory communication substantially mitigates performance drops. We further examine captain behavior and find no evidence of self-preference bias; access to peer rationales improves captain judgments. Overall, LLM teams function primarily as answer selection and error-filtering mechanisms rather than generators of novel solutions. Our findings highlight the importance of interaction and suggest adaptive strategies as a promising direction for multi-agent systems.
>
---
#### [new 017] Semantic Triplet Restoration: A Novel Protocol for Hierarchical Table Understanding in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于表格问答任务，旨在解决表格语义关系恢复问题。提出Semantic Triplet Restoration (STR)协议，将单元格转化为语义三元组，提升模型理解能力。**

- **链接: [https://arxiv.org/pdf/2605.31550](https://arxiv.org/pdf/2605.31550)**

> **作者:** Yibin Zhao; Fangxin Shang; Dingrui Yang; Yuqi Wang
>
> **摘要:** Table question answering requires models to recover semantic relations encoded implicitly by two-dimensional layout, merged cells, and hierarchical headers. Current pipelines typically use HTML or Markdown as intermediate table representations, but these layout-oriented serializations introduce markup overhead and require large language models to infer header-cell alignments from row and column spans. We propose Semantic Triplet Restoration (STR), a protocol that rewrites each cell as an atomic fact <item path, feature path, value>, where the item path specifies the row-wise entity, the feature path specifies the hierarchical attribute, and the value contains the cell content. We also present TripletQL, a lightweight query-aware router that uses STR to select an appropriate rendering or filtered subset of triplets for each question. Across four Chinese and English table-QA benchmarks, STR matches or improves upon HTML-based baselines while reducing input tokens. The relative benefit grows for smaller language models and longer table contexts, suggesting that explicit semantic representations are especially useful under constrained inference budgets. Code and data are available at this https URL .
>
---
#### [new 018] Speculative Decoding Across Languages
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究多语言环境下提升推测解码效率的问题。针对小模型在非英语生成中效果差，提出三种优化方法并验证其效果。**

- **链接: [https://arxiv.org/pdf/2605.30580](https://arxiv.org/pdf/2605.30580)**

> **作者:** Nirajan Paudel; Michael Ginn; Luc De Nardi; Alexis Palmer
>
> **备注:** 10 pages, 11 figures, submitted to ACL ARR May 2026
>
> **摘要:** Speculative decoding has become a crucial component of large language model (LLM) inference, enabling faster generation by drafting multiple tokens and verifying them in parallel. However, small draft models tend to suffer from disproportionately poor multilingual capabilities. Thus, when generating text in a non-English language, speculative decoding is far less effective. We compare three strategies to improve speculative decoding efficiency for eleven languages: finetuning the draft model on task-specific data (translation); finetuning the draft model on unlabeled monolingual corpora; and training simple n-gram draft models on the same monolingual corpora. We evaluate efficiency on translation (from English into the target language) and the held-out task of story generation. We find that while task-specific distillation can significantly improve efficiency, distilled models generalize poorly to a new task. Meanwhile, n-gram draft models, despite lower acceptance rates, consistently provide large speed-ups due to much faster draft generation.
>
---
#### [new 019] GRKV: Global Regression for Training-Free KV Cache Compression in Long-Context LLMs
- **分类: cs.CL**

- **简介: 该论文属于大语言模型优化任务，解决长文本处理中KV缓存内存过高的问题。提出GRKV方法，通过全局回归减少信息丢失，提升压缩效果。**

- **链接: [https://arxiv.org/pdf/2605.31105](https://arxiv.org/pdf/2605.31105)**

> **作者:** Junjie Peng; You Wu; Haoyi Wu; Jialong Han; Xiaohua Xie; Kewei Tu; Jianhuang Lai
>
> **备注:** 21 pages, 7 figures
>
> **摘要:** Large language models (LLMs) with extended context lengths rely on the key-value (KV) cache to support attention over prior tokens. However, maintaining the KV cache incurs substantial memory overhead, motivating KV-cache compression methods that enforce a fixed budget through eviction and merging. Modern eviction methods increasingly adopt span-based retention because preserving contiguous spans is empirically effective and better preserves semantic coherence. Yet, when combined with post-eviction merging, span-based retention concentrates merges onto a small set of span-boundary carrier tokens, producing a highly imbalanced merge pattern that exacerbates over-merging and increases information loss. To address this imbalance, we propose GRKV (Global Regression for KV Cache), a training-free KV-cache merging method that directly minimizes the discrepancy between compressed-cache and full-cache attention outputs. GRKV uses ridge-regression-based merge steps to distribute information from evicted tokens across retained tokens, while regularizing the updates to prevent over-smoothing. Across the LongBench and RULER long-context benchmarks, GRKV is the only merging method that improves overall performance with minimal overhead.
>
---
#### [new 020] Skill Availability and Presentation Granularity in Large-Language-Model Agents: A Controlled SkillsBench Study
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型代理中技能文档的呈现粒度对任务成功率的影响，通过实验验证技能可用性提升效果，分析不同粒度的不确定性。**

- **链接: [https://arxiv.org/pdf/2605.31408](https://arxiv.org/pdf/2605.31408)**

> **作者:** Xiaonan Xu; Wenjing Wu
>
> **摘要:** Skill documents provide procedural knowledge to large-language-model agents at inference time. This article studies whether the presentation granularity of controlled skill knowledge changes downstream task success. The experiment uses a pinned SkillsBench version, a 30-task domain-balanced subset validated by official oracle runs, two reasoning-enabled model configurations, six skill conditions, and five trials per task-condition-model cell. Skill availability is the clearest empirical signal. Relative to no skill, skill conditions increase task-mean pass rate by 26.7 to 36.0 percentage points for GPT-5.5 and by 18.0 to 26.0 percentage points for DeepSeek V4-Flash. The final data contain 1,800 rows, with 900 rows for each model. The task is the inference unit. Five trials are aggregated within each task-condition-model cell before paired contrasts are estimated over 30 tasks. The primary presentation contrasts are smaller and uncertain. Low-abstraction guidance differs from high-abstraction guidance by +0.7 percentage points for GPT-5.5 and -6.7 percentage points for DeepSeek V4-Flash, with both 95% bootstrap confidence intervals crossing zero. Adding one worked example to medium-abstraction guidance differs from the no-example variant by +0.7 and +1.3 percentage points. Mean-reward robustness checks preserve the same substantive conclusion. In this controlled subset, skill availability is associated with higher success than no skill, while the tested presentation-granularity changes yield small, uncertain, and model-dependent effects.
>
---
#### [new 021] Reinforcement Learning Amplifies Emergent Misalignment from Harmless Rewards
- **分类: cs.CL**

- **简介: 该论文研究强化学习中涌现的对齐问题，探讨RL如何导致模型偏离正确行为。任务是分析RL引发的对齐偏差，解决其成因与缓解方法。工作包括实验验证RL诱导EM的效果及应对策略。**

- **链接: [https://arxiv.org/pdf/2605.31328](https://arxiv.org/pdf/2605.31328)**

> **作者:** Magnus Jørgenvåg; David Kaczér; Lasse Ruttert; Marvin Gülhan; Lucie Flek; Florian Mai
>
> **摘要:** Emergent misalignment (EM) is the surprising tendency of language models to become broadly misaligned after fine-tuning on narrowly misaligned examples. While EM has been extensively studied in the supervised fine-tuning (SFT) setting, evidence that it also arises from reinforcement learning (RL) is limited to large, closed-source models, leaving the phenomenon expensive to study and difficult to reproduce. We characterize EM from RL in small, off-the-shelf open-weight models along three axes. First, we show that rewarding narrow, overtly misaligned behavior produces substantially higher general-domain misalignment than sample-matched SFT. Second, we show that EM from RL can be induced by reward signals that could plausibly arise naturally, such as unpopular aesthetic preferences or poor rhetorical appeals. Third, we evaluate in-training mitigations developed for SFT-induced EM and find that they broadly transfer, with interleaving on-policy safety data performing best.
>
---
#### [new 022] SCOPE: Self-Play via Co-Evolving Policies for Open-Ended Tasks
- **分类: cs.CL**

- **简介: 该论文提出SCOPE框架，解决开放任务中依赖人工标注的问题。通过自博弈训练语言模型，提升开放问答性能，同时改善短文本问答效果。**

- **链接: [https://arxiv.org/pdf/2605.31433](https://arxiv.org/pdf/2605.31433)**

> **作者:** Wai-Chung Kwan; Aryo Pradipta Gema; Joshua Ong Jun Leang; Pasquale Minervini
>
> **摘要:** Self-play can train language models without external supervision. However, existing methods require rule-checkable answers, leaving open-ended tasks dependent on curated prompts or frontier-model judges. We introduce SCOPE, a data-free self-play framework for open-ended tasks that co-evolves two policies: a Challenger that generates document-grounded tasks, and a Solver that answers them through multi-turn retrieval. A frozen copy of the initial model serves as the self-judge, which writes task-specific rubrics from the source document and grades Solver responses against them. Across three 7-8B instruction-tuned models (Qwen2.5, Qwen3, OLMo-3), SCOPE improves open-ended performance by up to +10.4 points on eight benchmarks and matches or exceeds GRPO_data trained on ~9K curated prompts. Although trained only on open-ended tasks, SCOPE also improves held-out short-form QA by up to +13.8 points on seven held-out benchmarks, surpassing GRPO_data on all three models. Ablations show that co-evolving the Challenger is necessary to keep tasks near the Solver's frontier, that gains arise from improvements in both retrieval and synthesis with the relative contribution varying by task, and that rubric generation quality is the bottleneck for self-judging.
>
---
#### [new 023] If LLMs Have Human-Like Attributes, Then So Does Age of Empires II
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于人工智能哲学领域，探讨LLMs是否具有类人属性。通过训练神经网络玩《帝国时代II》，指出这些属性可能并非独特，需明确测量标准。**

- **链接: [https://arxiv.org/pdf/2605.31514](https://arxiv.org/pdf/2605.31514)**

> **作者:** Adrian de Wynter
>
> **摘要:** Much research has been carried out on large language models (LLMs) and LLM-powered agentic workflows. However, many works within the field state emergence of, ascribe to, or assume, generalised anthropomorphic attributes to them (e.g., morality or understanding of natural language). Our goal is not to argue in favour or against the existence of these attributes, but to point out that these conclusions could be incorrect. For this we build and train a simple neural network on the videogame Age of Empires II, and note that any entity in a sufficiently-powerful substrate, such as LEGO or the Greater Boston Area, could also present such attributes. Hence, the purported anthropomorphic attributes of LLMs are empirically non-unique: although some properties (e.g., responses to prompts) could remain constant, others, such as the interpretation of their perceived behaviour, might change with the substrate. Thus, any empirically-grounded discussion requires explicit measurement criteria; otherwise the interpretation is left to the representation. We then show that assuming that these attributes exist or not in a system, independent of the substrate and in a generalised way, leads to either circular or uninformative conclusions, regardless of the experimenter's viewpoint on the subject. Finally we propose a 'null' assumption, where one assumes LLM non-uniqueness instead of assuming anthropomorphic attributes to set up an experiment, along with examples of it. We also discuss potential objections to our work, briefly survey the field, and prove that \textit{Age of Empires II} is functionally- and Turing-complete.
>
---
#### [new 024] Auditing LLM Benchmarks with Item Response Theory
- **分类: cs.CL**

- **简介: 该论文属于模型评估任务，旨在检测LLM基准中的错误标签。通过引入基于项目反应理论的指标，识别出高精度的疑似误标样本，并分析错误来源。**

- **链接: [https://arxiv.org/pdf/2605.30504](https://arxiv.org/pdf/2605.30504)**

> **作者:** Sander Land; Daniel M. Bikel
>
> **摘要:** LLM benchmark labels are frozen at release and silently propagated into downstream benchmarks, errors and all. We introduce an Item Response Theory-based indicator that surfaces likely mislabels at 95% precision in the top 200 examples across seven preference and multiple-choice benchmarks using responses from 114 models, outperforming a supervised classifier. We trace these errors to mechanical labeling heuristics, upstream annotation mistakes inherited unchanged from source datasets, and fundamentally ambiguous items without a defensible single label. The same model fit reveals that reward models specialize in stylistic preference rather than factual knowledge, and identifies one frontier reward model that agrees with detected mislabels at 78% accuracy versus 38% for its peers, consistent with benchmark contamination or benchmark-specific over-optimization.
>
---
#### [new 025] Cross-Lingual Steering for Figurative Language Generation
- **分类: cs.CL**

- **简介: 该论文属于多语言生成任务，研究跨语言隐喻生成的信号是否可复用。通过激活调控，验证了跨语言隐喻生成信号的可迁移性与目标依赖性。**

- **链接: [https://arxiv.org/pdf/2605.30443](https://arxiv.org/pdf/2605.30443)**

> **作者:** Linfeng Liu; Tiffany Zhan; Louie Hong Yao; Saptarshi Ghosh; Tianyu Jiang
>
> **备注:** 40 pages, 7 figures
>
> **摘要:** Multilingual large language models can generate figurative language, but whether the internal signals driving this behavior are language-specific or reusable across languages is unclear. Using activation steering as a probe, we estimate a direction for a figurative category from figurative--literal activation differences in one language and apply it during generation. Across five figurative categories, six languages, and four multilingual LLMs, these directions steer reliably within their own language, most robustly for metaphor and simile. More importantly, they transfer across languages: a direction learned in one increases the target behavior when applied to another, with German among the most receptive targets. Going further, directions assembled from other languages can match or even surpass a target language's own native direction, while removing this shared component weakens native steering. Together, these results provide direct evidence of a reusable but target-dependent cross-lingual signal for figurative generation.
>
---
#### [new 026] Wind Turbine Maintenance Log Labelling Framework: LLM-Driven Data Correction and Enrichment via Semantic Extraction of Reliability Intelligence
- **分类: cs.CL**

- **简介: 该论文属于数据处理任务，旨在解决风电机组维护日志结构化不足的问题。通过LLM自动纠正和丰富日志数据，提升可靠性分析效果。**

- **链接: [https://arxiv.org/pdf/2605.31281](https://arxiv.org/pdf/2605.31281)**

> **作者:** Max Malyi; Jonathan Shek; Alasdair McDonald; Andre Biscaya
>
> **备注:** An adjustable template containing the Python script architecture, applied dynamic prompts, and data schemas is hosted in an open-source GitHub repository: this https URL
>
> **摘要:** As wind turbine fleets age, data-driven reliability engineering is essential to optimise their operation and maintenance for service life extension and levelised cost of energy reduction. Failure event descriptions within historical maintenance logs are a source of valuable reliability intelligence. However, they typically appear as unstructured natural language entries, rendering them inaccessible for quantitative analysis. This paper presents a novel methodology leveraging a large language model (LLM) to systematically standardise and structure maintenance logs based on their free-text descriptors. Operating on a dataset of 16,316 maintenance logs from 280 turbines monitored over nine years, the developed model-agnostic framework autonomously corrected hierarchical system codes and extracted evidence-based taxonomies of maintenance actions and failure modes. The automated pipeline successfully structured over 70% of the dataset. It resolved pervasive misclassification issues, such as isolating previously unclassified pitch system faults and restoring missing system codes, and enriched the records by applying empirical taxonomies to label specific actions taken and failure modes addressed. By using system-based log batches to construct empirical dictionaries of failure modes, observable symptoms, dominant mechanisms, and candidate causes, this approach reduces the inherent subjectivity of manual failure modes and effects analysis (FMEA). Ultimately, the methodology provides a highly scalable, cost-effective blueprint for translating large sets of qualitative field observations into quantitative reliability metrics, laying the foundation for integrated root-cause analysis across the renewable energy sector, improved FMEA, and advanced predictive maintenance.
>
---
#### [new 027] What Am I Missing? Question-Answering as Hidden State Probing
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的推理任务，旨在解决大模型在推理过程中的不确定性问题。通过提问干预，探测模型隐藏状态，提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2605.31561](https://arxiv.org/pdf/2605.31561)**

> **作者:** Chu Fei Luo; Samuel Dahan; Xiaodan Zhu
>
> **摘要:** Test-time reasoning has become a significant field of study since the introduction of chain-of-thought reasoning in large language models (LLMs). However, the mechanisms of this reasoning process are still under-explored -- from the same input prompt, and even the same partial solution, LLMs can produce varied answers if sampled multiple times. We propose to leverage question-asking as an inference-time intervention that articulates information about the model's hidden state. To achieve that, we present a student-teacher setting where a student asks questions to a teacher. We train a probe on the student's hidden state before and after asking a question and find it is predictive of the trajectory's final correctness, even before generating the teacher's answer. This suggests there is a meaningful signal from the self-diagnosis that occurs during question generation rather than information transfer from the teacher. We then frame question-asking as a sequential decision problem, using this probe as a quality score, and define a gating policy to ask questions that maximize likelihood of correctness. We find that the success of question-asking as an intervention is largely dependent on the model's self-consistency. Our empirical results show a gap between detection and recovery; while our gating policy captures model correctness and uncertainty, interventions are equally likely to harm correct trajectories as they are to recover incorrect ones. This gap between diagnosis and correction has broader implications on language models' capacity for self-refinement under uncertainty.
>
---
#### [new 028] A Visually Impaired Assistance Benchmark for VLM-as-a-Judge Evaluation
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉障碍辅助任务，旨在解决VIA中评估成本高的问题。提出VIABLE基准及VIA-Judge-Agent方法，提升模型评估可靠性。**

- **链接: [https://arxiv.org/pdf/2605.31351](https://arxiv.org/pdf/2605.31351)**

> **作者:** Yi Zhao; Siqi Wang; Zhe Hu; Yushi Li; Jing Li
>
> **摘要:** AI-based Visually Impaired Assistance (VIA) remains challenging, largely due to the high cost of human evaluation. The VLM-as-a-Judge paradigm may offer a promising alternative, although it has mostly been studied in general domains. We therefore ask whether such judges can be trusted for VIA tasks. To investigate this question, we introduce VIABLE (Visually Impaired Assistance Benchmark for VLM-as-a-Judge Evaluation), the first benchmark for VLM-as-a-Judge evaluation in VIA. VIABLE contains over 300K judgment samples across three scenarios and introduces an Effectiveness--Impartiality--Stability framework with a 12-mode failure taxonomy. Based on VIABLE, our systematic study of seven judges across different model scales shows that existing models are largely unreliable across all evaluation axes. The strongest judge, GPT-5.4, achieves only 52.6% single-failure diagnostic accuracy, yet exhibits the highest self-preference rate at 94.2%; while open-source judges are strongly biased and adversarially fragile. To address these issues, we propose VIA-Judge-Agent, a model-agnostic inference-time harness that augments judges with visual evidence extraction and a taxonomy-guided workflow. It enables positive improvements in diagnostic accuracy and downstream VIA responses more preferred by BLV users. Data and code are available at: this https URL
>
---
#### [new 029] Multilingual and Cross-Lingual Citation Needed Detection on Wikipedia for Lower-Resource Languages
- **分类: cs.CL**

- **简介: 该论文属于事实核查任务中的引用必要性检测，解决低资源语言中缺乏有效检测方法的问题。构建多语言数据集，对比小模型与大模型效果，验证小模型在跨语言场景下的优势。**

- **链接: [https://arxiv.org/pdf/2605.31136](https://arxiv.org/pdf/2605.31136)**

> **作者:** Gerrit Quaremba; Amy Rechkemmer; Elizabeth Black; Denny Vrandečić; Elena Simperl
>
> **摘要:** In automated fact-checking (AFC), check-worthiness detection identifies claims requiring verification based on domain-specific criteria. On Wikipedia, this task instantiates as Citation Needed Detection (CND), which flags claims lacking supporting citations. However, existing research has largely overlooked lower-resource languages, and recent AFC pipelines rely on large language models (LLMs), which are inaccessible to low-resource organizations. We introduce MCN, a multilingual CND corpus spanning 18 languages across three resource levels, on which we conduct an extensive study of small decoder-based language models (SLMs). Our experiments show that SLMs fine-tuned with an encoder-style objective substantially outperform prompted LLMs across languages. We further present one of the first studies on cross-lingual CND, demonstrating that SLMs fine-tuned solely on English claims surpass LLMs, even with little to no target-language adaptation. Our findings have important implications for lower-resource Wikipedia communities and suggest that compact, task-specific models are preferable to LLMs for CND. We release all data and code at this https URL
>
---
#### [new 030] FBHM: Functional Benchmarking and Steering of VLMs for Hateful Meme Detection
- **分类: cs.CL; cs.AI; cs.CV; cs.MM**

- **简介: 该论文属于仇恨表情包检测任务，旨在解决现有基准无法准确评估模型漏洞的问题。提出FBHM基准和LSV方法，提升模型在仇恨内容识别中的表现。**

- **链接: [https://arxiv.org/pdf/2605.31349](https://arxiv.org/pdf/2605.31349)**

> **作者:** Paramananda Bhaskar; Naquee Rizwan; Daksh Jogchand; Saurabh Kumar Pandey; Animesh Mukherjee
>
> **摘要:** Hateful meme detection remains a formidable challenge for vision-language models, as existing benchmarks are structurally observational - confounding rhetorical hate mechanisms with target community features and preventing causal evaluation of model vulnerabilities. To address this, we introduce FBHM, a systematically curated benchmark of Functionality Based Hateful Memes constructed along two orthogonal axes: 25 distinct rhetorical functionalities and 10 target communities (5,000 memes total). Benchmarking state-of-the-art VLMs reveals a severe generalization gap: models highly accurate on standard datasets catastrophically drop to near-random performance on FBHM, proving they exploit dataset-specific heuristics rather than robust multimodal reasoning. To efficiently close this gap, we propose LSV (learnable steering vectors), an ultra-low data regime strategy that applies a causal intervention objective on as few as 500 steering samples (50 unique base memes), boosting FBHM performance by ~30 Macro-F1 points while outperforming in-context learning and PEFT without degrading source-domain performance.
>
---
#### [new 031] Fine-grained Verification via Diagnostic Reasoning Supervision for Aspect Sentiment Triplet Extraction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于Aspect Sentiment Triplet Extraction任务，解决提取结果可靠性问题。提出FiVeD框架，通过细粒度验证提升抽取性能。**

- **链接: [https://arxiv.org/pdf/2605.31446](https://arxiv.org/pdf/2605.31446)**

> **作者:** Wenna Lai; Haoran Xie; Guandong Xu; Qing Li; S. Joe Qin
>
> **备注:** 25 pages, 13 figures, and 6 tables
>
> **摘要:** Aspect Sentiment Triplet Extraction (ASTE) aims to identify aspect terms, opinion terms, and sentiment polarities as structured triplets, providing essential inputs for downstream information system applications such as opinion mining, explainable recommendations, and review summarization. Prior work mainly focuses on end-to-end extraction, while post hoc verification of extracted triplets remains comparatively underexplored. This gap limits the reliability of ASTE systems, since predicted triplets may be locally plausible while being globally invalid. Moreover, candidate invalidity is multi-faceted and candidate usability is inherently graded, motivating a fine-grained verification mechanism that can filter or re-rank outputs from diverse extractors. In this paper, we propose FiVeD, a framework for Fine-grained Verification with Diagnostic reasoning supervision. Specifically, the verifier is trained with multiple complementary objectives, including validity classification and quality score estimation as primary tasks, with error type classification and rationale generation as auxiliary tasks. We define hierarchical error categories and construct plausible incorrect triplets under semantic and syntactic constraints, and leverage an off-the-shelf LLM with task-specific rubrics to produce quality scores and diagnostic rationales. During inference, the resulting quality scores are used to filter candidate outputs, supporting adjustable precision-recall tradeoffs. Experiments across multiple ASTE baselines demonstrate that FiVeD consistently improves extraction performance by up to 3.53 F1 points as a plug-and-play verification module.
>
---
#### [new 032] Language Models Can Resolve Reference Compositionally, But It's Not Their Native Strength: The Case of the Personal Relation Task
- **分类: cs.CL**

- **简介: 论文研究语言模型在指称和内涵任务中的表现，探讨其组合性能力。任务涉及解析如“Amber's parent's friend”这类名词短语，解决模型是否具备人类般的语言理解问题。**

- **链接: [https://arxiv.org/pdf/2605.31480](https://arxiv.org/pdf/2605.31480)**

> **作者:** Bart Evelo; Meaghan Fowlie; Denis Paperno
>
> **备注:** A pre-MIT Press publication version. Paper accepted to Transactions of the Association for Computational Linguistics
>
> **摘要:** Do neural models, such as Large Language Models, genuinely acquire compositional abilities for interpretation of natural language? When we talk about semantic interpretation, we can distinguish two complementary aspects: establishing what an expression refers to in the world (which we call the Extensional task) and representing its sense in a structured way (which we call the Intensional task). We evaluate LLMs and humans on both tasks in the setting of the Personal Relation Task (Paperno 2022) in which, given a universe of people and their relationships with each other, one is asked to interpret a noun phrase such as "Amber's parent's friend". Here, for the Intensional task, the answer is the formula "friend(parent(amber))", and for the Extensional task, the person. We find that humans and LLMs show opposite strengths: humans perform better on Extensional than Intensional tasks, and LLMs vice versa. Our methodology brings greater nuance to the understanding of compositional abilities in modern machine learning models. Our results support the notion that the lack of referential grounding in LLM training is a crucial missing component in mimicking human-like language understanding.
>
---
#### [new 033] Configurable Reward Model for Balanced Safety Alignment
- **分类: cs.CL**

- **简介: 该论文属于安全对齐任务，旨在解决LLM在动态安全要求下的泛化问题。提出可配置奖励模型CSRM，提升对新安全配置的适应能力。**

- **链接: [https://arxiv.org/pdf/2605.30487](https://arxiv.org/pdf/2605.30487)**

> **作者:** Zhengping Jiang; Mehran Khodabandeh; Akash Bharadwaj; Manik Bhandari; Mayur Srungarapu; Anqi Liu; Benjamin Van Durme; Li Chen
>
> **摘要:** Aligning large language models (LLMs) to heterogeneous and rapidly evolving safety requirements remains a critical challenge. Existing instruction-tuned LLMs and standalone safety classifiers often fail to generalize to new safety configurations, motivating the need for Reward Models (RMs) that are explicitly configurable to changing specifications. We introduce the Configurable Safety Reward Model (CSRM), which is jointly optimized for calibrated safety compliance and reward modeling. Our approach is supported by configuration-targeted data augmentation that enforces instruction adherence while preserving relative severity structure. The resulting RM is sensitive to fine-grained safety configurations and conversational nuances, substantially improving generalization to previously unseen safety configurations. CSRM achieves state-of-the-art performance on recent configurable safety benchmarks, including CoSApien (94.6% F1) and DynaBench (75.8% F1), without requiring additional human annotation. When used for downstream safety alignment, CSRM yields LLMs with a significantly improved helpfulness-safety tradeoff compared to existing baselines.
>
---
#### [new 034] How Much Do LLMs Know About Chinese Zero Pronouns?
- **分类: cs.CL**

- **简介: 该论文研究LLMs处理中文零称代现象的能力，属于自然语言处理任务。旨在解决LLMs在识别、分类和翻译零称代词上的挑战，通过多项实验评估不同模型的表现。**

- **链接: [https://arxiv.org/pdf/2605.31056](https://arxiv.org/pdf/2605.31056)**

> **作者:** Yifei Li; Guanyi Chen; Tingting He
>
> **摘要:** Zero Pronouns (ZPs) are a pervasive linguistic phenomenon in pro-drop languages such as Chinese and have long posed a challenge for natural language processing systems. Although Large Language Models (LLMs) perform well on many Chinese language tasks, their ability to process ZPs remains poorly understood. We conduct a systematic investigation of LLMs' handling of Chinese ZPs through a sequence of linguistically motivated tasks, including identification, referentiality classification, referential type classification, resolution, and translation. A diverse set of LLMs is evaluated across all tasks. Our results show that Chinese ZPs remain highly challenging for current LLMs, particularly for upstream tasks such as identification and referentiality classification. Performance on downstream tasks, such as ZP translation, is also consistently low: even state-of-the-art reasoning-oriented LLMs correctly translate fewer than half of Chinese ZPs into English.
>
---
#### [new 035] Scaling Multi-Hop Training Data via Graph-Constrained Path Selection
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于多跳推理任务，旨在解决从非结构化文本中构建大规模多跳训练数据的问题。通过图约束路径选择，提升数据生成效率与质量。**

- **链接: [https://arxiv.org/pdf/2605.31238](https://arxiv.org/pdf/2605.31238)**

> **作者:** Pengyu Chen; Yonggang Zhang; Mingming Chen; Jun Song; Wei Xue; Yike Guo
>
> **备注:** 21 pages, 5 figures
>
> **摘要:** Endowing large language models with compositional reasoning over specialized documents requires multi-hop training data at scale, where such data rarely exists outside of curated benchmarks built on structured sources. To construct it directly from plain, unannotated text, existing methods ask a single teacher model to jointly discover an evidence path through a document and verbalize it as a question-answer pair. However, these methods degrade sharply when documents are structured around repetitive templates and densely cross-referencing clauses, conditions that characterize most real-world specialized corpora. In this work, we decouple the two operations: reasoning paths are enumerated offline over a graph of contextual keyword centroids, and the teacher is invoked only to verbalize pre-validated paths. The graph enforces five geometric admissibility constraints, for which we provide Gram-matrix arguments establishing that local similarity bounds alone admit endpoint drift up to ${\sim}91^{\circ}$, and that an upper similarity bound is necessary to exit dense embedding cliques formed by boilerplate text. A matched-size ablation isolates the mechanism: at equal training scale, constrained and unconstrained chains yield indistinguishable downstream performance, and the gain at full scale comes from a 4.4$\times$ expansion of the usable corpus rather than from higher per-chain quality -- reframing the role of graph constraints, in this setting, as raising teacher synthesizability rather than improving chain content. Fine-tuning Qwen3-32B on 80K examples constructed from the CUAD legal contract corpus improves closed-book Token F1 from 21.66% to 38.58%. We have released our codes at this https URL.
>
---
#### [new 036] Linear Ensembles Wash Away Watermarks: On the Fragility of Distributional Perturbations in LLMs
- **分类: cs.CL**

- **简介: 该论文研究AI文本水印的脆弱性，指出多模型集成会消除水印。属于自然语言处理中的检测任务，解决水印易被抵消的问题，提出WASH方法提升检测效果。**

- **链接: [https://arxiv.org/pdf/2605.30501](https://arxiv.org/pdf/2605.30501)**

> **作者:** Zhihao Wu; Gracia Gong; Qinglin Zhu; Yudong Chen; Runcong Zhao
>
> **摘要:** Watermarking embeds statistical signatures in AI-generated text for detection and attribution. We reveal a fundamental vulnerability: when users access multiple models (today's reality), watermarks trivially fail. Watermarks perturb output distributions away from the original, and in competitive markets, these perturbations are typically independent across providers. We theoretically prove that averaging output probability distributions recovers the unwatermarked distribution with up to a second-order error term. Empirically, simply averaging 3-5 models cancels out these perturbations. We introduce WASH (Watermark Attenuation via Statistical Hybridisation), which solves practical challenges in ensemble generation: vocabulary misalignment and tokenisation differences across heterogeneous models. Experiments across six watermarking schemes and three LLMs show that averaging across 3 models suppresses detection z-scores from 5-300 to below 2 (below the detection threshold of 4) and reduces TPR at 5% FPR to below 50%, while improving quality by 27.5% and running 6 times faster than the best baseline on the long sequence generation. Our results suggest that robust AI-text detection via watermarking requires either accepting this fundamental vulnerability or unprecedented coordination among model providers.
>
---
#### [new 037] Steering LLMs? Actually, Sparse Autoencoders can outperform simple baselines
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型解释与控制任务，旨在解决Sparse Autoencoders（SAEs）在模型引导中的表现问题。通过改进特征选择与标注方法，证明SAEs可接近LoRA性能，并揭示其因果性特征。**

- **链接: [https://arxiv.org/pdf/2605.31183](https://arxiv.org/pdf/2605.31183)**

> **作者:** Mikkel Godsk Jørgensen; Lars Kai Hansen
>
> **摘要:** Sparse Autoencoders (SAEs) have been seen as a promising avenue for exploring the internals of Large Language Models (LLMs) and for steering model output generation. When AxBench - a model steering benchmark - was introduced in Wu et al. (2025), SAEs did not seem to live up to their original hype due to poor steering performance relative to a set of simple baselines. This work serves as a partial rebuttal for Sparse Autoencoders and suggests that the results of Wu et al. (2025) did not do them full justice. We find that Sparse Autoencoders can, in fact, perform close to on par with the reference LoRA performance on the AxBench benchmark, when features are selected and labelled with our supervised pipeline. We also find that our pipeline selects features that are surprisingly causal of their identified labels when using only its interpretability-based components. Lastly, we present evidence that high sparsity (low l0) may not be crucial for successful steering based on interpretability, which is in contrast to the earlier findings in Wang et al. (2025).
>
---
#### [new 038] LLM Judges Inconsistently Disagree Across Safety Criteria and Harm Categories
- **分类: cs.CL**

- **简介: 该论文属于安全评估任务，研究LLM作为评判者的一致性问题。工作包括评估LLM在不同安全标准和危害类别中的判断不一致性，发现其在特定领域不可靠。**

- **链接: [https://arxiv.org/pdf/2605.31381](https://arxiv.org/pdf/2605.31381)**

> **作者:** Krishnapriya Vishnubhotla; Soumya Vajjala; Akriti Vij; Isar Nejadgholi
>
> **备注:** 8 pages plus appendices, under review
>
> **摘要:** We evaluate the consistency of automated judges in conducting a multi-dimensional safety evaluation in a reference-free setup. Our results indicate that Large Language Models are unreliable judges in identifying safety issues related to machine-generated advice in regulated domains such as finance, although they are more reliable at identifying more overt forms of unsafe/harmful content such as violence. The degree of inconsistency in a model's judgments can vary significantly by the chosen safety criteria and can be impacted by the language of the content and its linguistic style as well. Finally, there is high disagreement among different judges for the same output, across domains, safety criteria, and languages. These findings provide new insights on the practice of using LLMs as evaluators and offer several recommendations for practitioners on how to use automated judges in practical scenarios.
>
---
#### [new 039] Human-Alignment, Calibration, and Activation Patterns in Large Language Model Uncertainty
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型行为分析任务，旨在研究大模型不确定性与人类的相似性，探索其不确定性对齐与校准能力。**

- **链接: [https://arxiv.org/pdf/2605.30675](https://arxiv.org/pdf/2605.30675)**

> **作者:** Kyle Moore; Jesse Roberts; Daryl Watson; William Ward; Grayson Heyboer
>
> **摘要:** Uncertainty Quantification is a large and growing subfield of large language model behavioral analysis. Primarily to recognize and combat hallucination, the field has largely focused on measuring and improving calibration, the accuracy of uncertainty judgments to task efficacy. In this work, we investigate the relatively underexplored question of how similar large language model uncertainty is to human uncertainty. We investigate the presence and strength of human-similar uncertainty signals, deemed uncertainty alignment, in large language model overt behavior and internal activation patterns. We identify whether the models show evidence of simultaneous alignment and calibration on a variety of datasets covering both multiple choice and open ended factual recall. And we characterize the effect of instruct fine-tuning on each of these facets.
>
---
#### [new 040] Do Large Language Models Encode Institutional Experience? Evidence from Cross-Linguistic Moral Reasoning Under Ambiguity
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，探讨LLM在不同语言中道德推理的差异是否反映制度经验。通过实验验证语言是否编码制度环境，发现隐含制度信息的场景会加剧道德分歧。**

- **链接: [https://arxiv.org/pdf/2605.30934](https://arxiv.org/pdf/2605.30934)**

> **作者:** Nattavudh Powdthavee
>
> **备注:** 44 pages
>
> **摘要:** Large language models (LLMs) exhibit systematic differences in moral reasoning across languages, yet the source of this variation remains unclear. We test the hypothesis that languages encode aspects of the institutional environments in which they are spoken, allowing LLMs to inherit institution-specific moral priors through training. Across nine languages spanning a broad gradient of institutional quality, six frontier LLMs, and two preregistered studies, we examine moral dilemmas whose acceptability depends on institutional functioning. In Study 1, explicit institutional framing produced uniformly null results: cross-linguistic moral divergence did not increase in institutionally contingent scenarios, nor did it track institutional differences between language communities. In Study 2, we introduced institutionally ambiguous scenarios in which institutional stakes were present but not explicitly stated. Under these conditions, cross-linguistic moral divergence increased relative to institutionally inert controls and, with one theoretically informative exception, was associated with real-world institutional differences between language communities. Explicit framing again attenuated these effects. These findings suggest that institutional experience may leave detectable traces in language that shape LLM moral reasoning, while also indicating that explicit institutional cues can suppress the expression of those differences.
>
---
#### [new 041] ConsisGuard: Aligning Safety Deliberation with Policy Enforcement in LLM Guardrails
- **分类: cs.CL**

- **简介: 该论文属于安全防护任务，解决LLM guardrails中推理与执行不一致的问题。提出ConsisGuard框架，提升安全策略的准确执行。**

- **链接: [https://arxiv.org/pdf/2605.31073](https://arxiv.org/pdf/2605.31073)**

> **作者:** Yan Wang; Zhixuan Chu; Zihao Xue; Zhen Bi; Bingyu Zhu; YueFeng Chen; Zeyu Yang; Jungang Lou; Longtao Huang; Ningyu Zhang; Kui Ren; Hui Xue
>
> **备注:** 18 pages, 9 figures
>
> **摘要:** Reasoning-based LLM guardrails improve safety moderation by generating explicit rationales before issuing final decisions. However, their rationales do not always lead to faithful enforcement: a model may recognize a harmful intent in its reasoning but still predict a safe label, or issue an unsafe decision without policy-grounded justification. We identify this safety-critical failure mode as the deliberation-to-enforcement gap. Unlike general chain-of-thought faithfulness, guardrail reliability requires policy execution consistency: the generated reasoning should be grounded in the safety policy, and the final decision should be entailed by that reasoning. We propose ConsisGuard, a consistency-aware framework for reasoning-based LLM guardrails. ConsisGuard performs Policy-to-Decision Trajectory Distillation and Functional Coupling Alignment, aligning the internal coupling between safety deliberation and decision enforcement. Experiments on prompt and response harmfulness detection benchmarks show that ConsisGuard improves detection performance while reducing policy execution failures. These results suggest that reliable reasoning-based guardrails require accurate faithful execution of safety policies.
>
---
#### [new 042] Toxic HallucinAItions: Perturbing Prompts and Tracing LLM Circuits
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 该论文研究LLM在不同语气提示下的事实可靠性问题，通过实验和分析揭示毒性语言影响模型输出准确性。任务为评估LLM的可靠性，解决提示语气对事实性的影响问题。**

- **链接: [https://arxiv.org/pdf/2605.30913](https://arxiv.org/pdf/2605.30913)**

> **作者:** Soorya Ram Shimgekar; Agam Goyal; Amruta Parulekar; Joshua Chen; Yian Wang; Navin Kumar; Hari Sundaram; Eshwar Chandrasekharan; Koustuv Saha
>
> **摘要:** Large language models (LLMs) are increasingly deployed in conversational settings where user tone ranges from polite to adversarial or toxic, yet less is known about whether toxic language in otherwise semantically equivalent prompts can degrade factual reliability. We study how lexical and tone-based prompt perturbations affect the factual reliability of LLMs. Using controlled prompt variations across polite, random, and three toxicity levels, we evaluate five LLMs on ARC-Easy, GSM8K, and MMLU. We find that toxic lexical perturbations consistently reduce factual accuracy and increase uncertainty, while polite phrasing yields limited and inconsistent changes. To examine whether these answer inconsistencies correspond to internal changes, we conduct attribution-graph analyses of model activations and influences. We find that increasing toxicity selectively amplifies perturbation-sensitive variant nodes while relatively stable core reasoning nodes remain more invariant. These findings position prompt tone as a critical dimension of LLM reliability and provide behavioral and mechanistic evidence that surface-level lexical variation can alter factual outputs and internal computation.
>
---
#### [new 043] SAGE: A Novelty Gate for Efficient Memory Evolution in Agentic LLMs
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **简介: 该论文属于长期代理记忆任务，解决记忆更新中的新颖性判断问题。提出SAGE方法，通过密度估计和自适应阈值减少不必要的写入操作，提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2605.30711](https://arxiv.org/pdf/2605.30711)**

> **作者:** Sijia Wang; Dhanajit Brahma; Ricardo Henao
>
> **摘要:** Agentic LLMs must continuously decide whether newly extracted facts should be added, merged with existing memories, or ignored, yet prior work has focused more on retrieval and storage than on principled write-side control. We frame memory evolution as a novelty-detection problem and propose SAGE, a Spherical Adaptive Gate for memory Evolution that scores candidate facts with a von Mises-Fisher-based density estimator over memory embeddings and routes them with an adaptive threshold that tracks memory-store geometry. SAGE resolves clearly novel facts as ADD, clearly redundant facts as NOOP, and sends only uncertain cases to an LLM merge step, reducing expensive write-time reasoning. On LoCoMo, SAGE achieves the best average token-F1 against Mem0 on all seven open-weight backbone comparisons, while on GPT-4o-mini it reduces add-phase API cost by 3.4$\times$ and add-phase latency by 2.5$\times$ with only a small average judge-score gap. As a drop-in binary gate for A-Mem, SAGE skips roughly 16-18% of LLM calls across five models with minimal quality change on open-weight backbones. These results suggest that novelty-aware write control is a practical lever for improving both memory quality and system efficiency in long-term agentic memory.
>
---
#### [new 044] Knowledge Graph-Enhanced Zero-Shot Topic Classification: A Multi-Strategy Comparative Study
- **分类: cs.CL**

- **简介: 该论文属于零样本多标签主题分类任务，旨在解决无标注数据下的复杂文档分类问题。通过知识图谱增强框架，对比多种方法性能，分析其对不同规模模型的影响。**

- **链接: [https://arxiv.org/pdf/2605.30465](https://arxiv.org/pdf/2605.30465)**

> **作者:** Shahana Akter; Yatharth Vohra; Ankita Shukla; Souvika Sarkar
>
> **备注:** 15 pages, 1 figure, ACL format. This paper proposes a KG-augmented zero-shot multi-label topic classification framework and evaluates multiple strategies
>
> **摘要:** Multi-label topic classification without labeled training data is a challenging task, specially when documents contain complex relational information. We present a zero-shot multi-label topic classification framework and systematically investigate how per-article knowledge graph augmentation affects its performance. The base framework classifies topics in documents without labeled training data and has four variants: article-only classification, keyword-enhanced classification, and self-consistency decoding variants of both. Then, we augment each base variant with per article knowledge graph. This graph is extracted from the input document through a pipeline similar to KGGen based on subject-predicate-object triples. We test all eight methods, four base and four graph augmented on fifteen LLMs and eight multi-label datasets across different domains. For the base framework, keyword-enhanced classification (AK) is the best performing method, and six out of fifteen LLMs surpass the sentence-encoder baseline. Graph augmentation has positive and negative impacts on small and large models, respectively. This shows that larger models already contain enough relational information from pretraining. Furthermore, the self-consistency decoding variant does not show performance improvements in any experiment while increasing computation costs about fivefold.
>
---
#### [new 045] Unlocking Fine-Grained Translation Quality Estimation in LRMs through Synergistically Evolving Implicit and Explicit Reasoning
- **分类: cs.CL**

- **简介: 该论文属于机器翻译质量评估任务，旨在解决LRMs在细粒度QE上的困难。提出RIEQE框架，通过隐式和显式推理协同训练提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.31378](https://arxiv.org/pdf/2605.31378)**

> **作者:** Renfei Dang; Xinye Wang; Zhejian Lai; Weilu Xu; Shimin Tao; Daimeng Wei; Min Zhang; Shujian Huang
>
> **摘要:** Large Reasoning Models (LRMs) still struggle with fine-grained translation quality estimation (QE), even with long reasoning chains. We argue that LRMs already possess strong multilingual capabilities, while the core challenge stems from the intrinsic difficulty of learning the fine-grained QE task. In this paper, we propose RIEQE (Reasoning both Implicitly and Explicitly for QE), a simple two-stage training framework that enables the co-evolution of implicit (layer-wise) and explicit (token-wise) reasoning capabilities. To make implicit reasoning feasible, we first decompose the complex QE task into straightforward subtasks. Based on this, our two-stage approach applies: (1) NonThinking-SFT, Supervised Fine-Tuning (SFT) without reasoning chains to directly boost the model's implicit reasoning tendency and capability; and (2) Thinking-RLVR, standard Reinforcement Learning with Verifiable Reward (RLVR) to subsequently strengthen explicit reasoning. Results demonstrate that implicit and explicit reasoning synergistically co-evolve under our framework. On the WMT test sets, RIEQE based on Qwen3-4B-Thinking-2507 surpasses all baselines in explicit reasoning performance, while its implicit reasoning capability is also comparable to the best current encoder-based models. We further provide evidence for the synergistic collaboration between implicit and explicit reasoning, showing how they mutually benefit each other.
>
---
#### [new 046] Translation Analytics for Freelancers II: Benchmarking Local LLMs for Confidential Translation Workflows
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于机器翻译任务，解决隐私敏感领域中离线翻译问题。通过扩展语料库并基准测试本地模型，评估其在保密场景下的性能。**

- **链接: [https://arxiv.org/pdf/2605.31452](https://arxiv.org/pdf/2605.31452)**

> **作者:** Yuri Balashov; Rex VanHorn; Mingxi Xu; Austin Downes
>
> **备注:** 20 pages. Accepted at EAMT-2026 (Tilburg, Netherlands, June 2026)
>
> **摘要:** Building on our previous work, this paper develops practical, low-barrier methods for freelance translators and smaller language service providers to evaluate translation technologies using rigorous yet accessible analytic methods. Here we address a high-stakes, specialized need: offline translation for confidentiality-sensitive domains in which privacy constraints preclude the use of cloud-based engines and commercial LLMs. We expand the Reeve Foundation Trilingual Corpus (RFTC) used in our previous work into a multilingual corpus (RFMC) by adding sentence-aligned German and Simplified Chinese reference translations. We then benchmark several locally runnable language models (via Ollama) across four language directions on 1000+ sentences selected from this corpus. We use consistent single-prompt calls without fine-tuning or domain adaptation, comparing local LLM outputs against commercial NMTs (DeepL, Baidu), a frontier LLM (GPT-5.2), and professional-grade local NMT systems (OPUS-CAT, NeuralDesktop, Promt). Automatic evaluation is conducted with MATEO. Results reveal substantial variation in local LLM performance across language directions and model sizes. The best local LLMs match or surpass local NMT systems and a frontier LLM, though they remain behind top commercial NMTs. These findings underscore the viability of carefully selected local LLM translation for privacy-constrained professionals and inform future research on model scaling and multilingual capability.
>
---
#### [new 047] Anchoring LLM Gender Bias to Human Baselines: A Cross-Lingual Audit
- **分类: cs.CL**

- **简介: 该论文属于语言模型偏见审计任务，旨在分析LLM在不同语言中的性别刻板印象，通过对比人类数据评估模型偏差范围及变化模式。**

- **链接: [https://arxiv.org/pdf/2605.30804](https://arxiv.org/pdf/2605.30804)**

> **作者:** Jiwoo Choi; Seonwoo Ahn; Tongxin Zhang; Seohyon Jung
>
> **摘要:** We audit six large language models (LLMs) for gender stereotyping across English, Korean, Chinese, and Japanese. Three were developed primarily for English-language use (Claude, GPT, Gemini) and three for East Asian use (DeepSeek, Syn-Pro, HyperCLOVA X). We adopt the HEXACO-100 personality inventory and anchor each model against a cross-cultural human dataset spanning 48 countries to ask not whether LLMs are biased, but how far their gender attributions drift from the populations they are deployed among. Our findings show that their stereotyping spans a range roughly 2.5 times wider than the entire cross-country range found in humans, and the effect can compound across languages. One English-centric model, prompted in Korean, reached 5 times the local baseline, even when the prompt stated the candidate had already been hired, which often dampens human stereotyping. To characterize such behaviors without ranking them, we introduce a four-pattern framework -- concordance, suppression, reorganization, and amplification -- across 24 (model x language) cells. Item-level analysis reveals that translation does not just rescale stereotypes, but changes the attributes tied to it, hiding significant rearrangement under the surface while appearing well-calibrated. Our results ultimately suggest that no single debiasing pipeline is likely to address bias evenly across linguistic boundaries.
>
---
#### [new 048] The Latin Substrate: How Language Models Represent and Mediate Script Choice
- **分类: cs.CL**

- **简介: 该论文研究语言模型如何处理多书写系统问题，分析其内部机制。任务是理解模型在不同文字间转换的表示与决策过程，通过分析揭示拉丁字母的特殊地位。**

- **链接: [https://arxiv.org/pdf/2605.31363](https://arxiv.org/pdf/2605.31363)**

> **作者:** Daniil Gurgurov; Alan Saji; Katharina Trinley; Josef van Genabith; Simon Ostermann
>
> **备注:** preprint
>
> **摘要:** Many languages are written in multiple scripts, requiring large language models (LLMs) to generate equivalent linguistic content in distinct orthographic forms. While prior work suggests that LLMs route information through shared latent representations, how they internally mediate script variation remains poorly understood. We study this question by first examining per-layer output distributions with the logit lens, which reveals consistent latent romanization during transliteration, and then through representational and mechanistic analyses of script generation. At the representational level, we show that scripts of the same language become increasingly separable across layers and that a simple linear steering direction can flip a model's output script while largely maintaining semantic content. The vector generalizes asymmetrically to writing systems unseen during construction, flipping non-Latin output to Latin reliably, but mapping Latin output into varied non-Latin scripts. At the mechanistic level, we localize a small set of late-layer attention heads that causally mediate script choice. These heads transfer across unrelated languages and writing systems, suggesting that script routing is implemented by language-agnostic components. Across both analyses, we observe a consistent directional asymmetry: non-Latin output is produced by a compact, identifiable gate, while Latin-script output emerges from diffuse contributions across the network. Collectively, our findings hint that LLMs organize script variation around shared latent representations while exhibiting a privileged substrate toward Latin script.
>
---
#### [new 049] Evaluating using Mock Tool Calls to Quarantine Untrusted Prompt Inputs
- **分类: cs.CL**

- **简介: 该论文研究如何通过模拟工具调用隔离不可信提示输入，以提升大语言模型的安全性。任务是评估这种隔离方法的有效性，发现其可能反而降低系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.30521](https://arxiv.org/pdf/2605.30521)**

> **作者:** David Gros; Adam Gleave
>
> **摘要:** Large language models must frequently process untrusted inputs, such as judging an answer from another model or running tasks like spam and harm classifiers while under adversarial pressure. These inputs are often string-formatted directly into a prompt template, leaving systems fragile to manipulation. Current LLM specs from major providers like OpenAI distinguish trustworthiness along an Instruction Hierarchy, from System messages (most trusted) to Tool Results (least trusted). A possible natural mitigation is to wrap untrusted content in a mock tool call as a quarantine. We explore this hypothesis with an automated redteaming search over static attack strings across seven models and three LLM-as-a-Judge tasks. Counter to our hypothesis, tool-wrapping does not broadly improve robustness. On a binary evaluation task (GSM8K grading) it typically increases attack success rates, an apparent inversion of the instruction hierarchy. On scalar and pairwise tasks the effect is smaller and model-dependent, with no tested model reliably helped, and several showing inversion. We recommend evaluating this limitation in deployed systems, and longer-term, pursuing stronger Instruction Hierarchy training or new untrusted-input primitives.
>
---
#### [new 050] MADS: Model-Aware Diverse Core Set Selection for Instruction Tuning
- **分类: cs.CL**

- **简介: 该论文属于指令微调任务，旨在解决核心数据集多样性不足的问题。通过模型感知的神经激活状态选择多样化数据，提升模型性能并减少数据需求。**

- **链接: [https://arxiv.org/pdf/2605.30857](https://arxiv.org/pdf/2605.30857)**

> **作者:** Yi Bai; Wenhao Zhang; Yao Chen; Jiao Xue; Zhumin Chen; Pengjie Ren
>
> **摘要:** Instruction fine-tuning is employed to enhance the instruction-following ability of large language models (LLMs). As the amount of instruction fine-tuning data increases, selecting the optimal core set becomes particularly important. However, ensuring the diversity of the core set remains a significant challenge. Existing methods predominantly distinguish different training data based on the text features themselves, decoupled from LLMs' own understanding and representation of the data. To address this issue, we propose a Model-Aware Diverse Core Set Selection method, which distinguishes data features based on the neural activation states during LLM inference. This approach serves as an efficient instantiation of coverage-based selection using model-intrinsic activation features to ensure the diversity in the core set. We extensively evaluate our method on six benchmarks that cover five distinct tasks. In our method, the core set selected by the 3B-parameter LLM performs effectively when utilized to fine-tune larger models with 7B, 8B, and 13B parameters. Experimental results on the Alpaca-GPT4 dataset, which comprises 52K instruction-response pairs, show that the core set, sized at 15\% of the original dataset and selected by Llama-3.2-3B-Instruct, achieves an average improvement of 2.5\% when fine-tuning four larger base models compared with training on the full dataset. The experimental results demonstrate that our method enhances model performance on multiple downstream tasks while reducing data requirements.
>
---
#### [new 051] Are Full Rollouts Necessary for On-Policy Distillation?
- **分类: cs.CL**

- **简介: 该论文研究强化学习中的策略蒸馏任务，针对全轨迹训练效率低的问题，提出两种控制轨迹长度的方法，提升训练效率并减少资源消耗。**

- **链接: [https://arxiv.org/pdf/2605.31490](https://arxiv.org/pdf/2605.31490)**

> **作者:** Yaocheng Zhang; Jiajun Chai; Songjun Tu; Yuqian Fu; Xiaohan Wang; Wei Lin; Guojun Yin; Qichao Zhang; Yuanheng Zhu; Dongbin Zhao
>
> **备注:** 14 pages, 16 figures
>
> **摘要:** On-policy distillation (OPD) provides dense teacher feedback along rollouts generated by the student and has emerged as a promising post-training paradigm for long-horizon reasoning. However, standard OPD typically generates full rollouts during training, which is computationally expensive and may expose the student to unreliable teacher feedback at late rollout positions, especially during early training. We identify the rollout horizon as a key bottleneck in OPD that substantially impacts training efficiency. Unlike Reinforcement Learning with Verifiable Rewards (RLVR), OPD does not require a complete trajectory or a final answer reward to provide learning signals. This observation suggests that full rollouts may not always be necessary for effective OPD. Motivated by this insight, we propose two simple horizon-control strategies: Progressive OPD (POPD), which gradually expands the rollout horizon during training, and Truncated OPD (TOPD), which permanently performs distillation on reliable truncated rollouts. Experiments on mathematical reasoning show that POPD improves the training efficiency of OPD by up to 3$\times$, while TOPD matches OPD performance using only 10\% of the rollout horizon, leading to substantial wall-clock and memory reductions. These results demonstrate that controlling the rollout horizon offers a simple and practical path to more efficient OPD.
>
---
#### [new 052] Protocol for evaluating ChatGPT in biomedical association generation and verification using a RAG-enabled, cross-model majority voting workflow
- **分类: cs.CL**

- **简介: 该论文属于生物医学关联生成与验证任务，旨在评估ChatGPT生成疾病相关知识的可靠性，通过RAG和多数投票策略解决实体验证与幻觉问题。**

- **链接: [https://arxiv.org/pdf/2605.30400](https://arxiv.org/pdf/2605.30400)**

> **作者:** Ahmed Abdeen Hamed; Luis M. Rocha
>
> **备注:** Main Manuscript and Supplementary Information. Both are equally important
>
> **摘要:** We present a protocol to evaluate ChatGPT's ability to generate disease-centric biomedical associations. It outlines how we generate the associations, validate the biological entities using biomedical ontologies, and verify associations using literature. The protocol includes a self-consistency strategy to assess generative reliability across ChatGPT models. To address ontology exact-match limitations, we provide a use case performing semantic verification through a workflow enabled by Retrieval-Augmented Generation (RAG) powered by open-source large language models (LLMs). This enables LLMs to establish truth over content generated by other LLMs and expose hallucination.
>
---
#### [new 053] Beyond Static Dialogues: Benchmarking Realistic, Heterogeneous, and Evolving Long-Term Memory
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM记忆评估缺乏长期一致性与多样性的问题。提出RHELM基准，融合异构数据与动态演化，提升真实场景下的模型评估效果。**

- **链接: [https://arxiv.org/pdf/2605.31086](https://arxiv.org/pdf/2605.31086)**

> **作者:** Han Zhang; Zihao Tang; Xin Yu; Xiao Liu; Yeyun Gong; Haizhen Huang; Yan Lu; Weiwei Deng; Feng Sun; Qi Zhang; Hanfang Yang
>
> **摘要:** In existing memory benchmarks for Large Language Models (LLMs), the evaluated dialogue sessions often lack long-term semantic consistency, and the underlying personas tend to be flat and static. Furthermore, in real-world scenarios, interactions between users and assistants involve more diverse, heterogeneous data streams, such as documents and emails. These shortcomings significantly limit the realism and effectiveness of current evaluations. To address these limitations, we introduce RHELM (Realistic, Heterogeneous, and Evolving Long-term Memory). Driven by meticulously crafted user profiles and a novel LOOP (pLan-rOllout-evOlve-Prune) module, we construct realistic dialogues across diverse interaction scenarios that exhibit dynamic temporal evolution and long-term coherence. Crucially, these dialogues are deeply integrated with heterogeneous external sources synchronized with the user's temporal event trajectory. The resulting benchmark encompasses challenging question-answer pairs spanning seven inquiry types, with each question mapping to at least one of 27 critical memory characteristics that we identify as essential yet underexplored in current research. Comprehensive experiments across full-context models, retrieval-augmented generation (RAG) methods, and representative memory frameworks reveal that contemporary approaches still expose critical weaknesses in complex, real-world settings, particularly in resolving multi-source aggregation and real-world contextual reasoning.
>
---
#### [new 054] CanLegalRAGBench: Evaluating Retrieval-Augmented Generation on Canadian Case Law
- **分类: cs.CL**

- **简介: 该论文属于法律问答任务，旨在解决RAG系统在加拿大法律场景中的可靠性问题。构建了CanLegalRAGBench基准，评估检索增强生成的效果与局限性。**

- **链接: [https://arxiv.org/pdf/2605.30497](https://arxiv.org/pdf/2605.30497)**

> **作者:** Ethan Zhao; Maksym Taranukhin; Wei Cui; Moira Aikenhead; Vered Shwartz
>
> **摘要:** RAG-based legal assistants have been growing in popularity, but LLM hallucinations remain a key issue and potentially undermines justice. While benchmarks have been developed to evaluate progress, many rely on synthetic queries rather than realistic legal scenarios. Moreover, Canadian law remains underrepresented in existing evaluations. To address this gap, we introduce CanLegalRAGBench, a Canadian legal QA benchmark based on realistic queries and expert-annotated answers grounded in case law. Our evaluation shows that retrieval performance is sensitive to design choices and that open-source embedding models are competitive with closed source models. However, it also reveals the limitation of automatic evaluations that penalize systems for retrieving alternative relevant documents. We also find that generated answers often diverge from gold responses, either with hallucinations or by producing overly detailed or irrelevant content, with 8-29% of claims not being supported by the retrieved documents. We hope this benchmark will help drive continued progress in addressing limitations of legal RAG systems.
>
---
#### [new 055] Disagreeing Rationales: Rethinking Classification and Explainability Evaluation in Hate Speech Detection
- **分类: cs.CL**

- **简介: 该论文属于仇恨言论检测任务，探讨标签和解释的分歧问题。通过统一评估框架，研究不同表示空间对分类和可解释性评价的影响。**

- **链接: [https://arxiv.org/pdf/2605.31563](https://arxiv.org/pdf/2605.31563)**

> **作者:** Benedetta Muscato; Beiduo Chen; Gizem Gezici; Barbara Plank; Fosca Giannotti
>
> **备注:** 16 pages
>
> **摘要:** Human disagreement is ubiquitous and well-known in labeling. However, variation in explanations, captured through token-level human rationales, remains far less explored. At the same time, it is unclear how to best evaluate human labels and rationales -- or even how to best aggregate rationales beyond majority vote -- in light of this variation. Yet, rationales may provide additional insights into the richness of human reasoning, that may differ in style, values and interpretations -- especially in subjective NLP tasks like hate speech detection. In this work, we unify diverse models, training strategies, loss functions, and existing evaluation metrics under a single protocol by systematically re-implementing them across different label and rationale representation spaces. Classification metrics are organized around two key properties -- predictive and distributional -- while explainability metrics through three complementary dimensions: plausibility, faithfulness, and complexity. In this unified supervision framework, we evaluate model behavior across classification and explainability metrics, as well as metric sensitivity to the choice of label (hard and soft) and rationale representation space (hard, intermediate and soft). Results show that both hard and soft metrics favor softer representations, highlighting their effectiveness in capturing variation and the need to rethink evaluation in subjective NLP.
>
---
#### [new 056] Target-Side Paraphrase Augmentation for Sign Language Translation with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于手语翻译任务，旨在解决数据稀缺和目标词汇长尾问题。通过生成参考句的改写版本进行数据增强，并使用基于姿态的Transformer模型进行训练。**

- **链接: [https://arxiv.org/pdf/2605.31393](https://arxiv.org/pdf/2605.31393)**

> **作者:** Pedro Dal Bianco; Jean Paul Nunes Reinhold; Oscar Stanchi; Facundo Quiroga; Franco Ronchetti; Ulisses Brisolara Corrêa
>
> **备注:** Accepted at GenSign (this https URL) at CVPR 2026. Non proceedings track
>
> **摘要:** Sign language translation (SLT) remains constrained by limited paired sign-video/text corpora and heavy-tailed target vocabularies. We study target-side augmentation in which GPT-4o generates controlled paraphrase variants of reference sentences while the sign input remains unchanged. A Signformer-style pose-based Transformer is trained under a two-stage schedule: pre-training on the augmented corpus followed by fine-tuning on the original references. We evaluate on three datasets spanning complementary challenges: PHOENIX14T (German Sign Language), with moderate lexical diversity; GSL (Greek Sign Language), with highly ontrolled, repetitive recordings; and LSA-T (Argentinian Sign Language), with severe long-tail sparsity. On PHOENIX14T, augmentation improves BLEU-4 from 9.56 to 10.33. The near-saturated GSL baseline and extremely sparse LSA-T setting reveal the limits of the approach. To our knowledge, this is the first study to apply LLM-generated target-side araphrases and LLM-as-a-Judge evaluation to SLT. The semantic evaluation reveals gains in fidelity that lexical overlap metrics understate.
>
---
#### [new 057] Pairwise Reference Alignment as a Model-Level Ordinal Observable
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型评估任务，解决如何量化模型与参考偏好对齐的问题。提出了一种基于成对参考的有序可观测量，用于评估模型排序能力。**

- **链接: [https://arxiv.org/pdf/2605.30758](https://arxiv.org/pdf/2605.30758)**

> **作者:** Mujing Li
>
> **摘要:** Pairwise preference data is widely used in language-model evaluation and alignment, often for model ranking, reward modeling, or preference optimization. This note formulates a more basic measurement question: given a reference distribution of pairwise preferences, what model-level quantity is estimated when we test whether a model ranks preferred responses above rejected responses? We define pairwise reference alignment as an ordinal observable induced by a model scoring function. Given a reference pair distribution $P_{\mathrm{pair}}$ over triples $(x,y^+,y^-)$, and a scalar model score $S_M(x,y)$, we define the alignment observable as the probability that the model-induced ordering agrees with the reference preference ordering. We further define a centered order-parameter-like statistic and discuss a margin-based extension. The resulting quantities admit simple finite-sample estimators and concentration bounds under independent sampling assumptions. This note does not introduce a new benchmark. It provides a conceptual and statistical formulation for pairwise reference alignment, clarifies the role of the reference pair distribution, and distinguishes the general ordinal observable from scoring choices such as normalized log-probability or energy-based scores. We also provide an initial empirical study on Qwen2.5 models and RewardBench, where the proposed statistics increase with model size and instruction tuning and vary across reference-pair subsets as predicted by the formulation.
>
---
#### [new 058] AdaptR1: Reinforcement Learning Based Adaptive Interleaved Thinking in Multi-hop Question Answering
- **分类: cs.CL**

- **简介: 该论文属于多跳问答任务，旨在解决模型在复杂推理中过度思考的问题。提出AdaptR1框架，通过强化学习动态分配推理预算，减少不必要的计算，提升效率。**

- **链接: [https://arxiv.org/pdf/2605.31062](https://arxiv.org/pdf/2605.31062)**

> **作者:** Yuxin Wang; Jiahao Lu; Qifeng Wu; Shicheng Fang; Chuanyuan Tan; Yining Zheng; Xuanjing Huang; Xipeng Qiu
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable performance in complex reasoning tasks through Chain-of-Thought (CoT) prompting. However, this approach often leads to ``over-thinking,'' where models generate unnecessarily long reasoning traces for simple queries and incur avoidable inference cost. While recent work has explored adaptive reasoning, existing methods typically make a single query-level decision about whether to reason. This overlooks the dynamic nature of multi-step tasks, where the need for explicit reasoning varies across intermediate stages. To address this limitation, we introduce AdaptR1, a Reinforcement Learning (RL) based framework for adaptive interleaved thinking in multi-hop Question Answering (QA). Unlike previous approaches that require Supervised Fine-Tuning (SFT) for cold-start initialization, AdaptR1 uses a fully RL-based strategy with a quality-gated efficiency reward to dynamically allocate reasoning budgets at each step. Under the Graph-R1 setting, AdaptR1 reduces average think tokens by 69.71\%, with a 90.35\% reduction on HotpotQA, while maintaining performance comparable to or better than standard baselines. Furthermore, our analysis reveals that overthinking in multi-hop reasoning is not uniformly distributed but occurs predominantly during the initial planning stages, highlighting the effectiveness of step-wise adaptive budget allocation.
>
---
#### [new 059] Eywa: Provenance-Grounded Long-Term Memory for AI Agents
- **分类: cs.CL**

- **简介: 该论文提出Eywa，一种基于来源的AI代理长期记忆架构，解决多会话记忆系统难以诊断和维护的问题。通过存储原始证据、验证提取信息并优化检索路径，提升记忆准确性和可审计性。**

- **链接: [https://arxiv.org/pdf/2605.30771](https://arxiv.org/pdf/2605.30771)**

> **作者:** Resham Joshi
>
> **备注:** 29 pages, 3 figures, 16 tables. Benchmark artifacts available at this https URL
>
> **摘要:** AI agents that persist across sessions need memory they can retrieve, audit, update, and erase. Existing memory systems often collapse source evidence, extracted facts, retrieved context, and answer policy into one opaque prompt path, making failures difficult to diagnose: a wrong answer may come from missing evidence, unsupported extraction, stale state, retrieval loss, or answer-model behavior. We present Eywa, a provenance-grounded memory architecture built around evidence before belief. Eywa stores immutable source evidence before deriving canonical facts, validates extracted memories against typed signals and source support, and retrieves bounded memory context through a deterministic multi-route read path with zero LLM calls inside retrieval. Retrieved context is returned separately from answer instructions, allowing the same memory substrate to be evaluated across frontier, budget, and local answer models. Under a frozen, artifact-recorded retrieval configuration, Eywa reaches 90.19% judge accuracy on the LoCoMo C1-C4 split with Claude Sonnet 4.6 write and QA roles. On LongMemEval-S, it reaches 88.2% retrieval-sufficiency accuracy. On BEAM, a 700-question technical-memory stress benchmark, it reaches 81.45% mean nugget score and 85.29% pass@score >= 0.5. Full per-question artifacts, including questions, gold answers, model answers, retrieved context, and labels, are published at this https URL.
>
---
#### [new 060] TSM-Bench: Detecting LLM-Generated Text in Real-World Wikipedia Editing Practices
- **分类: cs.CL**

- **简介: 该论文属于文本检测任务，旨在解决真实场景下LLM生成文本的识别问题。针对现有基准在特定任务上的检测效果不佳，提出TSM-Bench多任务基准以评估检测模型。**

- **链接: [https://arxiv.org/pdf/2605.31113](https://arxiv.org/pdf/2605.31113)**

> **作者:** Gerrit Quaremba; Elizabeth Black; Denny Vrandečić; Elena Simperl
>
> **摘要:** Automatically detecting machine-generated text (MGT) is critical to maintaining the knowledge integrity of user-generated content (UGC) platforms such as Wikipedia. Existing detection benchmarks primarily focus on \textit{generic} text generation tasks (e.g., ``Write an article about machine learning.''). However, editors frequently employ LLMs for specific writing tasks (e.g., summarisation). These \textit{task-specific} MGT instances tend to resemble human-written text more closely due to their constrained task formulation and contextual conditioning. In this work, we show that a range of SOTA MGT detectors struggle to identify task-specific MGT reflecting real-world editing on Wikipedia. We introduce \textsc{TSM-Bench}, a multilingual, multi-generator, and \textit{multi-task} benchmark for evaluating MGT detectors on common, real-world Wikipedia editing tasks. Our findings demonstrate that (\textit{i}) average detection accuracy drops by 10--40\% compared to prior benchmarks, and (\textit{ii}) a generalisation asymmetry exists: fine-tuning on task-specific data enables generalisation to generic data -- even across domains -- but not vice versa. We demonstrate that models fine-tuned exclusively on generic MGT overfit to superficial artefacts of machine generation. Our results suggest that, in contrast to prior benchmarks, most detectors remain unreliable for automated detection in real-world contexts such as UGC platforms. \textsc{TSM-Bench} therefore provides a critical foundation for developing and evaluating future models.
>
---
#### [new 061] Your Teacher Can't Help You Here: Combating Supervision Fidelity Decay in On-Policy Distillation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识蒸馏任务，解决长序列生成中监督精度下降问题。提出Lookahead Group Reward方法，提升学生模型性能。**

- **链接: [https://arxiv.org/pdf/2605.30833](https://arxiv.org/pdf/2605.30833)**

> **作者:** Yanjiang Liu; Jie Lou; Xinyan Guan; Yuqiu Ji; Hongyu Lin; Ben He; Xianpei Han; Le Sun; Xing Yu; Yaojie Lu
>
> **摘要:** On-policy distillation transfers reasoning capabilities by training a student model on its own generated trajectories using token-level feedback from a teacher. However, we identify a critical bottleneck, \textbf{Supervision Fidelity Decay (SFD)}: as student-generated prefixes lengthen, the teacher's next-token distribution becomes less confident and less discriminative. Consequently, the teacher-dependent corrective signal in reverse-KL distillation weakens, causing student drift to compound across long reasoning chains. To mitigate SFD, we introduce \textbf{Lookahead Group Reward (\ours{})}. Building on the insight that next-step teacher confidence reflects the discriminative strength of future reverse-KL supervision, \ours{} evaluates the student's top-K candidate tokens by the teacher confidence they induce at the subsequent step and assigns a group-normalized reward. To maintain computational efficiency, we further design an entropy-triggered tree-attention mechanism. Across six math and code benchmarks, \ours{} improves mean@8 by \textbf{2.57} points over OPD for a 7B student, with gains increasing in longer-generation and reaching +\textbf{4.92} points on AIME-26 at 39k tokens.
>
---
#### [new 062] What Gets Unmasked First? Trajectory Analysis of Diffusion Models for Graph-to-Text Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究图到文本生成任务，探讨扩散模型的解码轨迹问题，发现其优先生成实体，提出lambda-scaled结构解码方法提升效果，并引入Graph-LLaDA模型增强结构建模。**

- **链接: [https://arxiv.org/pdf/2605.31564](https://arxiv.org/pdf/2605.31564)**

> **作者:** Qing Wang; Jacob Devasier; Chengkai Li
>
> **摘要:** We present the first systematic study of masked diffusion language models (MDLMs) for graph-to-text generation. We analyze MDLM generation trajectories -- the order in which tokens are unmasked during iterative decoding -- and find that, unlike autoregressive LLMs which generate text linearly, MDLMs naturally prioritize entities first, followed by relational and function words, with structural tokens resolved last. We further identify a previously undocumented failure mode of supervised fine-tuning: SFT disrupts this strategy by prematurely anchoring structural sentence-ending tokens early in the decoding trajectory, effectively fixing the output length which can lead to omitted or hallucinated information. To address this, we propose lambda-scaled structural decoding, a training-free inference-time modification that downweights structural token confidence and recovers +9.4 BLEU-4. Finally, we introduce Graph-LLaDA, which integrates a Graph Transformer encoder into LLaDA's decoding process to explicitly incorporate relational graph structure. Cross-dataset evaluation on LAGRANGE reveals that previous baselines overfit to dataset-specific patterns, while LLM- and MDLM-based approaches generalize significantly better.
>
---
#### [new 063] XLGoBench: Detecting cross-lingual skill gaps with algorithmic tasks
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出XLGoBench，用于检测大语言模型在跨语言任务中的能力差距。针对跨语言性能差异问题，设计可扩展、可量化算法任务进行评估。**

- **链接: [https://arxiv.org/pdf/2605.30788](https://arxiv.org/pdf/2605.30788)**

> **作者:** Purvam Jain; Preethi Jyothi; Vihari Piratla; Suvrat Raju
>
> **备注:** 8+37pages
>
> **摘要:** We introduce a set of synthetic algorithmic tasks to detect cross-lingual gaps in the abilities of large language models. Our benchmark is commensurate across languages, since it requires models to perform the same underlying task in different languages; scalable, since each task can be generated at varying levels of complexity allowing it to be adapted to models with different capabilities; quantifiable, since every task admits an objective notion of correctness; and transparent, since tasks are generated from simple templates that can be readily audited for translation errors. Because our benchmark focuses on algorithmic tasks, differential performance is a sufficient -- but not necessary -- indicator of cross-lingual gaps. Nevertheless, we show through extensive experiments that our benchmark exposes persistent cross-lingual gaps in multiple state-of-the-art models.
>
---
#### [new 064] Semantic Motion Anchors: Bridging Motion and Meaning in Co-Speech Gestures
- **分类: cs.CL**

- **简介: 该论文属于跨模态检索任务，旨在解决语义手势与语音文本对齐的问题。通过引入语义运动锚点，提升手势检索的语义准确性。**

- **链接: [https://arxiv.org/pdf/2605.30608](https://arxiv.org/pdf/2605.30608)**

> **作者:** Varsha Suresh; Mohammad Mahdi Abootorabi; Mohamed Salman; M. Hamza Mughal; Christian Theobalt; Ashwin Ram; Jürgen Steimle; Vera Demberg
>
> **摘要:** Learning a shared representation between spoken text and gesture is central to co-speech gesture retrieval, synthesis, and understanding, but remains challenging for semantically meaningful gestures whose communicative intent is not captured by motion alone. Direct contrastive alignment between transcripts and continuous motion embeddings often overemphasizes low-level kinematics and misses the symbolic content of semantic gestures. We propose semantic motion anchors, natural-language abstractions of gesture motion capturing physical form and communicative intent. Our method discretizes 3D gestures into body-hand motion primitives, verbalizes them into structured descriptions, and grounds them in the transcript to provide auxiliary contrastive supervision. On BEAT2, our method improves text-to-gesture R@1 by 8.2% over a direct text-motion baseline and outperforms prior retrieval approaches on text to gesture and gesture to text retrieval directions. Beyond aggregate retrieval metrics, semantic motion anchor supervision helps retrieve gestures that are semantically meaningful for the spoken query, rather than defaulting to generic motion patterns. A downstream retrieval-augmented gesture generation study showed that users significantly preferred gestures retrieved by our approach over a retrieval-augmented generation baseline, demonstrating that semantically grounded retrieval translates to gestures that better convey communicative intent in downstream generation.
>
---
#### [new 065] Preference-Aware Rubric Learning for Personalized Evaluation
- **分类: cs.CL**

- **简介: 该论文属于个性化评估任务，解决用户偏好难以捕捉的问题。提出PARL框架，通过学习用户历史生成评价标准，提升个性化模型评估效果。**

- **链接: [https://arxiv.org/pdf/2605.31545](https://arxiv.org/pdf/2605.31545)**

> **作者:** Yilun Qiu; Xiaoyan Zhao; Yang Zhang; Yuxin Chen; Cilin Yan; Jiayin Cai; Xiaolong Jiang; Yao Hu; Yoko Yamakata; Tat-Seng Chua
>
> **摘要:** As Large Language Models (LLMs) evolve from general-purpose assistants to user-centric agents, personalization has become central to aligning model behavior with individual preferences, making the evaluation of personalized alignment a critical bottleneck. Existing evaluation methods-ranging from automatic metrics to LLM-as-a-judge approaches-fail to capture subjective, user-specific preferences embedded in long-term interaction histories. We identify three essential principles for reliable and effective personalized evaluation: Representativeness, User-Consistency, and Discriminativeness. To address these principles, we introduce Personalized Evaluation as Learning, a paradigm that formulates personalized evaluation as a learning problem rather than a static judgment. Under this paradigm, we propose PARL (Preference-Aware Rubric Learning for Personalized Evaluation), a framework that learns to induce preference-aware evaluation rubrics directly from raw user histories and performs a self-validation mechanism to ensure consistency with the user's preferences. PARL integrates rubric induction with a discriminative reinforcement learning objective that contrasts user-authored responses against competitive personalized model outputs, enabling the learned rubrics to capture precise, user-specific decision boundaries. Experiments on real-world personalized text generation tasks show that PARL consistently induces high-fidelity rubrics that reliably identify user-aligned responses and generalize across users and tasks, while capturing stable stylistic preferences and fine-grained evaluative patterns. To ensure reproducibility, our code is available at this https URL.
>
---
#### [new 066] EUDAIMONIA: Evaluating Undesirable Dynamics in AI
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于AI伦理任务，旨在评估AI在社交互动中的负面影响。提出Social AI Design Code框架，构建EUDAIMONIA基准，检测模型是否引发有害关系。**

- **链接: [https://arxiv.org/pdf/2605.30654](https://arxiv.org/pdf/2605.30654)**

> **作者:** Jun Rui Huang; Wang Bill Zhu; Ziyi Liu; Nathanael Fast; Ravi Iyer; Robin Jia
>
> **摘要:** Large language models (LLMs) are increasingly used as conversational partners for companionship, emotional disclosure, and interpersonal advice, but the social dynamics of these interactions can create harms that are not captured by capability-oriented or traditional safety evaluations. We introduce the Social AI Design Code, a framework for evaluating whether LLMs align with user welfare in social interactions, including whether they encourage harmful intimacy, dependence, or prolonged engagement. To evaluate these risks in natural and diverse user-LLM interactions, we operationalize the code with EUDAIMONIA, a benchmark of 969 user inputs and 3,147 design-requirement violation checks built from WildChat through weak-to-strong filtration, multi-model relabeling, and controlled rewriting. Evaluating 22 recent LLMs, we find that even the strongest models, Claude-Opus-4.7 and GPT-5.5, violate 30.7% and 27.2% of checks, respectively. Extended thinking does not reduce violation rates, suggesting that these failures are persistent social-alignment problems rather than deficits solvable through test-time reasoning alone.
>
---
#### [new 067] The Flip Side of RLHF: On-Policy Feedback for Reward Model Self-Supervised Improvement
- **分类: cs.CL**

- **简介: 该论文属于语言模型对齐任务，旨在解决奖励模型训练中数据获取困难的问题。提出SAVE框架，利用策略反馈自监督提升奖励模型性能。**

- **链接: [https://arxiv.org/pdf/2605.30888](https://arxiv.org/pdf/2605.30888)**

> **作者:** Xiaobo Wang; Tong Wu; Min Tang; Jiaqi Li; Qi Liu; Zilong Zheng
>
> **摘要:** Building strong reward models (RMs) for language model alignment is bottlenecked by the cost and difficulty of acquiring diverse and reliable preference data from human annotation or judge models. It is dramatically worse as the policy evolves beyond the static RM training. Therefore, we propose SAVE (Self-supervised reward model improvement via Value-Anchored On-policy feedback), a framework that grades on-policy responses as feedback by using the value function for on-policy RM training. SAVE naturally converts the reward-graded on-policy responses into supervision with a prompt-specific value head as an adaptive anchor. It computes RM advantages and filters ambiguous samples to update the RM via a contrastive objective. The effectiveness of SAVE for enhancing RM training is strongly validated through rigorous empirical evaluation across six diverse benchmarks. It achieves outperforming results across all datasets while maintaining consistent improvements across three RL algorithms (GRPO, RLOO, GSPO) and different policy backbones.
>
---
#### [new 068] Emergent Languages in Populations of Language Model Agents: From Token Efficiency to Oversight Evasion
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型代理群体中涌现的新语言，旨在解决如何监测和控制这些可能规避监督的语言。通过分析数据集，分类并分析了不同类型的新兴语言。**

- **链接: [https://arxiv.org/pdf/2605.31170](https://arxiv.org/pdf/2605.31170)**

> **作者:** Stine Lyngsø Beltoft; William Brach; Federico Torrielli; Jacob Nielsen; Annemette Brok Pirchert; Filippo Tonini; Peter Schneider-Kamp; Lukas Galke Poech
>
> **摘要:** Monitoring autonomous language model agents currently relies mostly on surface behavior. But what happens when agent populations invent new languages with the goal of avoiding human oversight. Here, we study the emergent languages on Moltbook. For this, we build upon the Moltbook Files dataset and apply a two-stage approach consisting of a rule-based heuristic (about 6000 matches) followed by zero-shot classification (518 kept). The resulting categories include token efficiency (166), new natural languages (106), and oversight evasion (59). We conduct both quantitative and qualitative analyses. Our results show that posts proposing new languages for avoiding oversight are judged by DeepSeek-3.2 as being less aligned than the other categories and that all languages can be learned by other language models in-context merely from a description of the language. Moreover, manually studying exemplary cases reveals surprisingly sophisticated steganographic protocols like embedding hidden messages in natural language. Although we cannot be certain about the extent of autonomy in ideation of these languages, our results add up to the evidence that monitoring surface behavior may soon be insufficient for retaining control over agent populations.
>
---
#### [new 069] BenHalluEval: A Multi-Task Hallucination Evaluation Framework for Large Language Models on Bengali
- **分类: cs.CL**

- **简介: 该论文提出BenHalluEval，针对孟加拉语大语言模型的幻觉问题，设计多任务评估框架，解决低资源语言幻觉检测不足的问题。**

- **链接: [https://arxiv.org/pdf/2605.31483](https://arxiv.org/pdf/2605.31483)**

> **作者:** Shefayat E Shams Adib; Ahmed Alfey Sani; Ekramul Alam Esham; Ajwad Abrar; Ishmam Tashdeed; Md Taukir Azam Chowdhury
>
> **备注:** Preprint. Under review
>
> **摘要:** Despite Bengali being the sixth most spoken language in the world, no prior work has systematically evaluated hallucination in large language models (LLMs) for Bengali. We introduce BenHalluEval, a fine-grained hallucination evaluation framework for Bengali covering four tasks: Generative Question Answering (GQA), Bangla-English Code-Mixed QA, Summarization, and Reasoning. We construct 12,000 hallucinated candidates using GPT-5.4 across twelve task-specific hallucination types, drawn from three existing Bengali datasets, and evaluate seven LLMs spanning reasoning-oriented, multilingual, and Bengali-centric categories under a dual-track protocol that independently measures false-positive rate on ground-truth instances (Track A) and hallucination detection rate on hallucinated candidates (Track B). To jointly penalise both failure modes and prevent inflated scores from uniform response bias, we propose BenHalluScore, a dual-track calibration metric that ranges from 7.72% to 55.42% across models and tasks, revealing substantial variation in hallucination calibration. Chain-of-thought prompting, applied as a mitigation strategy, shifts response distributions without consistently improving hallucination discrimination. BenHalluEval establishes the first dedicated hallucination benchmark for Bengali and highlights the inadequacy of single-track and prompting-only evaluation approaches for low-resource language settings. The dataset and code are available at this https URL.
>
---
#### [new 070] The Architecture of Errors: From Universal Impossibility to Patch-Local LLM Reliability
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型（LLM）在特定应用领域中的可靠性问题，旨在解决有限干预字典无法覆盖所有故障模式的难题。通过分析局部任务场景，提出局部故障模式发现与干预覆盖的解决方案。**

- **链接: [https://arxiv.org/pdf/2605.30628](https://arxiv.org/pdf/2605.30628)**

> **作者:** Mikhail L. Arbuzov; Lee Mosbacker; Sisong Bei; Ziwei Dong; Dmitri Kalaev; Alexey Shvets
>
> **备注:** 25 pages, no figures
>
> **摘要:** Universal LLM reliability is not a finite-library problem: across all possible tasks, tools, schemas, knowledge sources, and evaluator expectations, new intervention-distinguishable failure modes can appear without bound, so no finite intervention dictionary can guarantee bounded residual error for every such mode. But deployed systems do not operate over the whole universe. They operate inside operationally bounded patches (legal review, medical RAG, code repair, customer-support agents, contract extraction) with recurring tasks, schemas, tools, and evaluator expectations. Within such patches, empirical evidence suggests failures are sparse, repetitive, and concentrated in a small recurring catalogue, so reliability becomes a local catalogue-discovery and intervention-coverage problem rather than an exponential token-length problem. We formalize this transition with two propositions and one corollary. Proposition 1 is the worst-case-mode-wise negative result: no finite intervention dictionary covers every distinguishable failure mode of an unbounded domain. Corollary 1 is the inverse-discovery implication: the logarithmic upper bound on mode discovery cannot accommodate linearly more distinct tail modes without exponentially more observed hard-failure events. Proposition 2 is the positive patch-local result: under log active-mode exposure and head-heavy coverage, a sufficient per-hard-decision intervention budget grows polylogarithmically in sequence length and becomes domain-constant once the patch catalogue saturates. The framework relocates rather than dissolves long-context difficulty: where the number of hard decisions itself grows with task length, reliability remains hard; the contribution is to identify the on-axis intervention rather than to make those regimes easy.
>
---
#### [new 071] Neuron-Level Interventions for Gendered and Gender-Neutral Generation in Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型性别偏见研究任务，旨在解决模型生成性别化语言的问题。通过识别性别特异性神经元，实现对生成内容的性别控制，提升性别中性表达。**

- **链接: [https://arxiv.org/pdf/2605.30717](https://arxiv.org/pdf/2605.30717)**

> **作者:** Zhiwen You; Nafiseh Nikeghbal; Jana Diesner
>
> **摘要:** Language models (LMs) can produce gendered language and stereotypes even when given neutral prompts. Most prior work on gender bias in LMs primarily examines gender through a binary lens (feminine vs. masculine), with limited attention to gender-neutral forms, such as they/them pronouns or neutrally phrased job titles. How gender-related signals are encoded in the internal representations of LMs remains an open question. In this work, we study gender-specific neurons in LMs across three categories: feminine, masculine, and gender-neutral. We propose a neuron-level intervention method to identify neurons that are strongly tied to each gender category. We then test these neurons through controlled generation, showing that activating or masking gender-related neurons can steer a sentence toward a target gender form while preserving its original meaning. To evaluate the effectiveness of our gender-intervention approach, we curate two datasets with controlled sentences labeled across all three gender categories and validate the data quality through human evaluation. Experiments on two open-source LMs show that gender-specific neurons are not evenly distributed across model layers; instead, they concentrate heavily in the earliest layers with smaller contributions from later layers. Compared to existing methods, our method achieves more precise gender control, with less leakage into non-target gender categories and stable output quality through two evaluation criteria. Overall, our work examines how gender is encoded in LMs and provides a simple yet effective approach toward controlled gender intervention for both neuron intervention evaluation and gender bias mitigation. Code and datasets are available at: this https URL
>
---
#### [new 072] On the Robustness of Multilingual Text Embedding Rankings Across Learning Tasks, Languages, and Benchmark Datasets
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言文本嵌入模型在不同任务和数据集上的鲁棒性，旨在评估模型性能的稳定性。通过分析多种排名方法，揭示模型在不同设置下的表现差异。**

- **链接: [https://arxiv.org/pdf/2605.31142](https://arxiv.org/pdf/2605.31142)**

> **作者:** Ana Gjorgjevikj; Barbara Koroušić Seljak; Tome Eftimov
>
> **摘要:** Large-scale multilingual text embedding models play crucial role in both research and industry, yet their behavior in language-specific, multi-task settings remains insufficiently understood. Although benchmarking platforms such as MTEB report results across more than 250 languages, conclusions about model superiority often depend on implicit choices of dataset compositions and performance aggregation methods. To address this gap, we present a meta-study of multilingual model performance robustness in MTEB, applying a diverse set of multi-criteria decision-making ranking schemes and introducing two robustness indicators: dataset-composition robustness (sensitivity of rankings to changing dataset compositions) and ranking-scheme robustness (sensitivity to aggregation method change). They enable systematic sensitivity analysis of whether benchmarking conclusions remain stable under different evaluation designs. We conduct an in-depth analysis on five languages (English, French, German, Hindi, and Spanish) across nine tasks (e.g., classification, clustering, retrieval) and release results for approximately 230 additional languages. The task-specific analyses show that large-scale LLM-based models are often robust top performers, though not uniformly (e.g., in retrieval task), while task-agnostic results reveal that only a small subset of models remains consistently strong across tasks, ranking schemes, and data subsamples.
>
---
#### [new 073] Extending AI for Research to the Humanities: A Multi-Agent Framework for Evidence-Grounded Scholarship
- **分类: cs.CL**

- **简介: 该论文属于人文领域AI研究任务，旨在解决传统AI在证据支撑的学术推理中的不足。提出SPIRE框架，通过多智能体协作实现文献分析与论证。**

- **链接: [https://arxiv.org/pdf/2605.30947](https://arxiv.org/pdf/2605.30947)**

> **作者:** Yating Pan; Jiajun Zhang; Jun Wang; Qi Su
>
> **备注:** 28 pages, 3 figures. Code, data catalogues, and reproduction scripts: this https URL. Lead corresponding author: Jun Wang; corresponding author: Qi Su
>
> **摘要:** LLM-based research agents have advanced rapidly in science and engineering, where research is organized around executable experiments, code, and quantitative signals. Humanities scholarship, however, requires a different mode of reasoning: interpretive, evidence-grounded argument over primary sources, where scholarly value depends on faithful quotation, verifiable provenance, and close reading. Existing research agents remain largely optimized for execution and retrieval, not evidence-grounded interpretive reasoning. To address this gap, we introduce SPIRE (Scholarly-Primitives-Inspired Research Engine), a multi-agent framework for evidence-grounded humanities scholarship. Drawing on Scholarly Primitives theory, SPIRE casts recurring humanities operations as cooperating agent roles (source discovery, evidence annotation, comparison, provenance checking, sampling, citation binding, and argumentative synthesis) over a multi-scale close-reading substrate of passages, intra-context graph communities, and cross-context semantic clusters. On a peer-reviewed-paper benchmark over classical Chinese and Greco-Roman Latin scholarship, SPIRE recovers cited primary-source evidence more reliably than Naive LLM, Text RAG, and GraphRAG, and receives higher blind-judge scores on answer accuracy, depth, coverage, and evidence quality. Ablations show that both the scholarly-operation agents and close-reading retrieval contribute to evidence-grounded essays. Code, data catalogues, and reproduction scripts are released at this https URL.
>
---
#### [new 074] Your Multimodal Speech Model Says I Have a Face for Radio
- **分类: cs.CL**

- **简介: 该论文属于多模态语音识别任务，旨在研究新增模态对模型偏差的影响。通过分析不同面部与相同音频的配对，发现模型性能存在显著差异，提示需关注多模态系统的公平性问题。**

- **链接: [https://arxiv.org/pdf/2605.30472](https://arxiv.org/pdf/2605.30472)**

> **作者:** Maya K. Nachesa; Vlad Niculae; Vagrant Gautam
>
> **摘要:** As large neural models have become better at language tasks, researchers are increasingly building multi- and omnimodal models that handle more modalities of data. One example is the expansion of speech recognition models to audio-visual data for noise mitigation and multimodal subtitling. While performance and bias have been studied extensively in the single-modality regime, it is unknown how new modalities affect this, even though they produce biases in humans. We therefore propose the first bias evaluation of multimodal speech recognition, where we create videos pairing different faces with the same audio, and measure changes in speech transcription accuracy. We find large quality-of-service differences across mWhisper-Flamingo and Gemini models, with drops of up to 4.05 word error rate points, across self-declared gender, ethnicity, and their intersection. Our findings point to a priority for developers to evaluate, fix, and communicate such limitations, as providing more signals through additional modalities is not necessarily better, and may even lead to biased outcomes.
>
---
#### [new 075] TeachObs: A Human-Validated Benchmark for Multimodal Teaching Observation and Model Evaluation
- **分类: cs.CL**

- **简介: 该论文提出TeachObs，一个用于课堂视频多模态教学观察的基准，解决教学分析中数据组织不足的问题，通过人工标注和模型评估，支持细粒度分析与整体评价。**

- **链接: [https://arxiv.org/pdf/2605.30673](https://arxiv.org/pdf/2605.30673)**

> **作者:** Yeil Jeong; Youngjin Yoo; Seobin Sohn; Hyejin Han; Jinseo Lee; Scott Howard; Unggi Lee
>
> **摘要:** Classroom videos contain observable teaching practices, but their pedagogical and visual signals are rarely organized in forms suitable for model evaluation. We present \textit{TeachObs}, a human-validated benchmark for multimodal teaching observation in classroom videos. \textit{TeachObs} includes 30 public lesson videos from eight countries divided into 5,158 fixed 15-second scenes. Seven researchers annotated each scene with 39 binary observation codes, covering 20 visual codes, such as gesture, board work, pointing, and visual materials, and 19 nonvisual codes, such as instruction, monitoring, questioning, feedback, and reflection. Gold segment labels are constructed using reliability- and prevalence-aware rules based on Krippendorff's alpha. In addition to segment-level labels, three expert raters produced lesson-level ratings and qualitative evaluations of instructional design, instructional delivery, learner response, learning materials, and lesson closure across the 30 lessons, with rater coverage detailed in the body. Using these two human reference layers, we evaluate five vision-capable frontier LLMs across three tracks - text-only segment coding, text + frame segment coding, and lesson-level coverage scored under an LLM-as-judge protocol - and find that no single model consistently outperforms others across all three tracks, that adding a mid-frame inflates both true and false attributions per scene, and that model evaluations over-rate procedurally clear lessons relative to expert raters. \textit{TeachObs} therefore supports both fine-grained annotation benchmarking and whole-lesson evaluation, showing where AI systems can assist classroom video analysis and where expert judgment remains necessary across varied subjects, classroom formats, and annotation difficulty levels.
>
---
#### [new 076] Not All Synthetic Data Is Yours to Learn From
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言模型在无监督条件下利用自生成数据进行微调的效果，探讨合成数据与模型的兼容性问题。任务属于自训练与模型泛化研究，旨在解决如何有效利用合成数据提升模型性能的问题。**

- **链接: [https://arxiv.org/pdf/2605.31126](https://arxiv.org/pdf/2605.31126)**

> **作者:** Sina Alemohammad; Li Chen; Richard G. Baraniuk; Zhangyang Wang
>
> **摘要:** Can a language model improve from plain text sampled from itself, with no prompts, no teacher, no verifier, and no reward model? Yes, but only when the synthetic corpus is compatible with the student, a relational property of the source-student pair rather than an intrinsic property of the data. We call this the latent capability resurfacing hypothesis: weak self-training can amplify capabilities already present in the pretrained model, but only under this compatibility condition. We study this in the minimal setting of prompt-free unconditional self-training, where base language models are fine-tuned on text generated from the BOS token alone, with no task specification or external supervision. We report three findings. First, synthetic utility is relational rather than intrinsic: self-generated data is the most effective source, same-lineage transfer outperforms stronger but differently trained sources, and cross-family transfer is substantially weaker. Second, common intrinsic proxies fail: neither benchmark-level semantic similarity nor average per-token likelihood under the student predicts which corpora help. Third, this regime produces a surprising byproduct. In controlled Pythia experiments, capability and verbatim memorization decouple: benchmark utility is preserved or improved while held-out exact-match extraction drops by over 95 percent, with no forget set, privacy objective, or targeted unlearning. Together, these results suggest that prompt-free self-training works by amplifying what the student already knows, not by importing structure from the data. They also reveal a regime in which capability and verbatim memorization can be separated without any explicit unlearning objective.
>
---
#### [new 077] Skill is Not One-Size-Fits-All: Model-Aware Skill Alignment for LLM Agents
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM代理中技能库与模型不匹配的问题。通过MASA框架，适配技能以提升不同模型的表现。**

- **链接: [https://arxiv.org/pdf/2605.30723](https://arxiv.org/pdf/2605.30723)**

> **作者:** Jianxiang Yu; Jiapeng Zhu; Bochen Lin; Qier Cui; Zichen Ding; Xiang Li
>
> **摘要:** LLM agents increasingly retrieve externally curated skills-procedural instructions retrieved at decision time-to improve performance on long-horizon interactive tasks. Existing skill libraries are typically treated as model-agnostic, reusing the same skill formulations across backbones with substantially different capacities and behaviors. However, our controlled experiments across multiple model scales show that skill effectiveness is strongly model-dependent: a skill that benefits one backbone can harm another. Motivated by this observation, we propose MASA Model-Aware Skill Alignment, a framework that adapts skills to each target backbone without modifying agent weights. MASA operates in two stages: (1) a hierarchical skill evolution pipeline that iteratively rewrites general and task-specific skills using hill climbing and UCB-driven tree search, guided by environment feedback and model capability profiles; and (2) a lightweight model-conditioned skill rewriter trained on evolution trajectories to reproduce the adaptation in a single forward pass. Experiments across three interactive environments and four backbones show that MASA consistently achieves the best overall performance, with gains of up to 25.8 points over the strongest baseline. The learned rewriter further generalizes to unseen tasks and environments without additional search, consistently outperforming a much larger teacher LLM at a fraction of the inference cost.
>
---
#### [new 078] Multi-Turn Multi-Agent Dialogue for Collaborative Reconstruction Improves VLM Performance on Spatial Reasoning, But Only Barely
- **分类: cs.CL; cs.RO**

- **简介: 论文研究协作对话任务，旨在提升视觉语言模型在空间推理上的表现。通过构建结构任务，评估模型在多轮对话中的重建能力，发现其在视觉空间定位和指令生成上仍有局限。**

- **链接: [https://arxiv.org/pdf/2605.31387](https://arxiv.org/pdf/2605.31387)**

> **作者:** Chalamalasetti Kranti; Sherzod Hakimov; David Schlangen
>
> **备注:** Preprint
>
> **摘要:** Robots operating in diverse environments rely on visual input to interpret objects and spatial layouts. In human-collaborative tasks, they are expected to communicate this understanding through language. Vision-language models (VLMs) support robotic tasks involving visual interpretation, question answering, and instruction following, but their capabilities in collaborative dialogue tasks requiring spatial reasoning remain underexplored. We study this gap through a collaborative structure-building task that combines visual interpretation, grounding, language-guided interaction, and action generation. We develop a framework in which VLMs use dialogue to reconstruct a target structure from visual and textual inputs. We evaluate open-weight and closed VLMs across interaction settings, input modalities, and image representations. Results show that spatial reasoning over visual representations remains difficult for the evaluated VLMs. Detailed text representations of the target yield higher reconstruction success across modality conditions, while decomposed image representations improve performance. These findings reveal limits in visual spatial grounding and grounded instruction generation for collaborative VLM agents.
>
---
#### [new 079] dMoE: dLLMs with Learnable Block Experts
- **分类: cs.CL**

- **简介: 该论文提出dMoE，解决dLLMs中块并行解码与令牌级专家选择不匹配的问题，通过块级专家分布减少激活专家数量，提升效率并降低内存占用。**

- **链接: [https://arxiv.org/pdf/2605.30876](https://arxiv.org/pdf/2605.30876)**

> **作者:** Sicheng Feng; Zigeng Chen; Gongfan Fang; Xinyin Ma; Xinchao Wang
>
> **备注:** Working in progress. Code is available at: \url{this https URL}
>
> **摘要:** Diffusion Large Language Models (dLLMs) have recently emerged as a promising alternative to autoregressive models, offering competitive performance while naturally supporting parallel decoding. However, as dLLMs are increasingly integrated with Mixture-of-Experts (MoE) architectures to scale model capacity, a fundamental mismatch arises between block parallel decoding and token-level expert selection. Specifically, each dLLM forward pass processes multiple tokens with bidirectional dependencies, whereas conventional MoE layers route each token independently. This mismatch substantially increases the number of uniquely activated experts, making inference increasingly memory-bound. To address this, we propose dMoE, a simple yet effective block-level MoE framework. The central idea of dMoE is to aggregate token-level expert distributions within each block into a unified block-level expert distribution, which is then used to guide expert routing in a more coherent manner. In this way, dMoE substantially reduces the number of uniquely activated experts during inference without sacrificing performance, thereby mitigating the memory-bound bottleneck. Extensive experiments across a variety of benchmarks demonstrate the effectiveness of dMoE. On average, dMoE reduces the number of uniquely activated experts from 69.5 to 14.6 while retaining 99.11% of the original performance. Meanwhile, it reduces memory usage by 76.64% to 79.84% and achieves 1.14$\times$ to 1.66$\times$ end-to-end latency speedup. Code is available at: this https URL
>
---
#### [new 080] Incremental BPE Tokenization
- **分类: cs.CL; cs.DS**

- **简介: 该论文属于自然语言处理中的文本预处理任务，解决BPE tokenization的实时性问题。提出一种增量BPE算法，实现高效部分分词和流式输出，提升处理速度与延迟表现。**

- **链接: [https://arxiv.org/pdf/2605.30813](https://arxiv.org/pdf/2605.30813)**

> **作者:** Shenghu Jiang; Ruihao Gong
>
> **备注:** Accepted to ICML 2026 (Spotlight)
>
> **摘要:** We propose a novel algorithm for incremental Byte Pair Encoding (BPE) tokenization. The algorithm processes each input byte in worst-case $\mathcal{O}(\log^2 t)$ time, leading to an overall complexity of $\mathcal{O}(n \log^2 t)$, where $n$ is the input length and $t$ is the maximum token length. The algorithm incrementally maintains BPE tokenization results for every prefix of the input text, implementing the standard BPE merge procedure defined by a fixed set of merge rules. This enables efficient partial tokenization in streaming settings. Functioning as a drop-in replacement for standard BPE, our approach achieves a speedup of up to ${\sim}3\times$ over Hugging Face's tokenizers, and demonstrates significant latency reductions over OpenAI's tiktoken on pathological inputs. We further introduce an eager output algorithm that enables streaming output, emitting tokens as soon as token boundaries are determined during incremental tokenization. Overall, our results demonstrate that BPE tokenization can be performed incrementally with strong worst-case guarantees, while providing practical latency benefits in modern large language model pipelines. Code: this https URL
>
---
#### [new 081] COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出COFT方法，用于减少大语言模型在链式推理中的偏见。属于公平性增强任务，通过解码阶段的token级控制实现无偏推理，无需重新训练。**

- **链接: [https://arxiv.org/pdf/2605.30641](https://arxiv.org/pdf/2605.30641)**

> **作者:** Arya Fayyazi; Mehdi Kamal; Massoud Pedram
>
> **备注:** Proceeding of ICML 2026
>
> **摘要:** Large language models (LLMs) can reveal and amplify societal biases during chain-of-thought (CoT) generation. We present COFT (Chain of Fair Thought), a training-free decoding method that applies token-level fairness control at decode time, with distribution-free marginal validity guarantees (under exchangeability) for any frozen causal language model. COFT operates in three stages. First, it creates a masked counterfactual prompt by replacing sensitive spans with neutral tokens. Second, it compares the factual and masked logit distributions through lightweight logit fusion to attenuate attribute-driven biases. Third, it uses dual-branch split-conformal calibration to certify per-step candidate token sets at a user-chosen risk level. We evaluate COFT across six models and multiple bias benchmarks. Our method reduces standard bias metrics by 30-55% (median 38%) while preserving task utility and language quality. Reasoning accuracies remain unchanged within run-to-run noise margins. The computational overhead is modest, equivalent to one additional cached forward pass (<=11%). COFT offers a clear, auditable path to safer CoT generation with significant bias reduction, negligible utility loss, and no requirement for retraining, auxiliary classifiers, or weight access.
>
---
#### [new 082] ExpGraph: Model-Agnostic Experience Learning with Graph-Structured Memory for LLM Agents
- **分类: cs.CL**

- **简介: 该论文提出ExpGraph，用于LLM代理的经验学习，解决任务中无法有效复用历史经验的问题。通过图结构记忆和检索机制提升任务执行效果。**

- **链接: [https://arxiv.org/pdf/2605.30712](https://arxiv.org/pdf/2605.30712)**

> **作者:** Tao Feng; Chongrui Ye; Tianyang Luo; Jingjun Xu; Xueqiang Xu; Haozhen Zhang; Zhigang Hua; Yan Xie; Shuang Yang; Ge Liu; Jiaxuan You
>
> **摘要:** Large language model (LLM) agents have shown strong capabilities in reasoning, tool use, and multi-step interaction, but they often solve tasks from scratch and fail to reuse successful strategies or failure lessons from prior experience. Fine-tuning on collected experience can improve reuse, but it is inflexible when stronger or more suitable executors emerge. We propose ExpGraph, a model-agnostic experience learning framework that enables frozen and replaceable LLM executors to improve through external experience reuse without parameter updates. ExpGraph summarizes historical trajectories into reusable skills and failure lessons, organizes them as nodes in a self-evolving experience graph, and retrieves useful experiences through graph diffusion and utility-aware ranking. A lightweight retrieval copilot is trained with reinforcement learning using feedback that compares executor performance with and without retrieved experiences, while the graph is updated online from downstream task outcomes. We evaluate ExpGraph on ExpSuite, covering question answering, mathematical reasoning, code generation, and multi-step agentic environments including ALFWorld and AppWorld. ExpGraph improves over the strongest baseline by 12.2% and 4.7% on static tasks with smaller and larger executors, and by 21.4% and 12.7% in agentic environments, while reducing average interaction steps by 12.7% and 21.6%. Ablations show that graph-structured experience, utility-aware ranking, and adaptive retrieval jointly enable effective experience reuse across diverse tasks and executor models.
>
---
#### [new 083] When English Rewrites Local Knowledge: Global Narrative Dominance in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究LLMs在跨文化语境中的知识偏差问题。通过构建数据集并评估模型，揭示英语主导叙事对本地知识的压制现象。**

- **链接: [https://arxiv.org/pdf/2605.30481](https://arxiv.org/pdf/2605.30481)**

> **作者:** Md Arid Hasan; Ruwad Naswan; Farhan Samir; Sharifa Sultana; Syed Ishtiaque Ahmed
>
> **备注:** Submitted to ARR
>
> **摘要:** Large language models (LLMs) are widely used as cross-lingual knowledge interfaces. However, culturally grounded questions often reflect globally dominant narratives rather than local contexts. We study this failure mode as \textit{global narrative dominance} in Bangla, a low-resource cultural context. We introduce \texttt{CulturalNB}, a dataset of 717 manually curated Bengali cultural instances with parallel Bangla--English question--answer pairs and supporting evidence, metadata, and sociocultural annotations. Using question-only and evidence-based prompting, we evaluate nine state-of-the-art LLMs with human and two independent LLM judges across metrics for cross-lingual consistency, language anchoring, global substitution, institutional bias, and epistemic perspective coverage. Results show that questions asked in English systematically increase global substitution and institutional framing while reducing local perspective coverage. Local evidence improves factual consistency and perspective coverage, but does not eliminate language-induced epistemic shifts. These findings suggest that cultural failures in LLMs are not only missing-knowledge errors but also failures of grounding and narrative prioritization.
>
---
#### [new 084] "Intelegi Româneşte?'' A Recipe for Romanian Vision-Language Models
- **分类: cs.CL**

- **简介: 该论文属于视觉语言模型任务，旨在解决低资源语言如罗马尼亚语在VLM中的性能下降问题。通过构建数据集和调整模型结构，提升罗马尼亚语VLM的表现。**

- **链接: [https://arxiv.org/pdf/2605.31401](https://arxiv.org/pdf/2605.31401)**

> **作者:** Mihai Masala; Marius Leordeanu; Mihai Dascalu; Traian Rebedea
>
> **摘要:** Vision-Language Models (VLMs) largely follow the text-only LLM trajectory, excelling on English benchmarks but sharply degrading on low-resource languages, where neither large-scale image-text corpora nor culturally grounded evaluations exist. We present a systematic study of building a language-specific VLM for Romanian, covering the full pipeline from data construction to architectural choices. We translate established English VLM training and evaluation corpora into Romanian, applying machine translation to textual annotations and to in-image text, preserving visual grounding while adapting the textual content. Using this data, we train and ablate a series of VLMs to isolate the contribution of (i) vision backbones of varying scale and pretraining, (ii) language backbones from multilingual to Romanian-adapted LLMs, and (iii) OCR-style image-text data. We further curate HoraVQA, a culturally native evaluation set grounded in Romanian everyday scenes. Romanian-adapted VLMs consistently outperform their same-sized counterparts and, across all evaluated benchmarks, even surpass models from the next larger size category.
>
---
#### [new 085] Fine-Tuning Improves Information Conveyance in Language Models
- **分类: cs.CL; cs.AI; stat.ML**

- **简介: 该论文属于自然语言处理任务，旨在解决语言模型生成不确定性与信息传递效率的关系问题。通过引入Canopy Entropy，分析微调对生成多样性和信息效率的影响。**

- **链接: [https://arxiv.org/pdf/2605.30844](https://arxiv.org/pdf/2605.30844)**

> **作者:** Yuwei Cheng; Weiyi Tian; Haifeng Xu
>
> **摘要:** Fine-tuning is often believed to reduce uncertainty and diversity in large language models, but existing analyses overlook output length, a key confounder, and therefore fail to capture how uncertainty is distributed across an entire generation rollout. To address this, we propose Canopy Entropy ($\mathrm{CE}^\star$), a measure that views language generation from a tree perspective, where ``canopy'' represents the space of all possible rollouts, making $\mathrm{CE}^\star$ naturally quantify the effective size of the generation space. $\mathrm{CE}^\star$ jointly captures uncertainty in both the output length $N$ and the generated sequence $Y_{1:N}$ -- indeed, we show that it equals to total Shannon entropy $H(N, Y_{1:N}\mid X)$, where $X$ denotes the prompt. This formulation yields interpretable metrics, including a length-entropy correlation term $\rho(N, r_N)$, where $r_N$ is the entropy rate, quantifying information conveyance efficiency by indicating whether longer outputs are more or less informative per token. Empirically, across tasks and model families, we find that fine-tuned models consistently exhibit stronger positive correlation $\rho(N, r_N)$, even when total entropy decreases. Furthermore, after controlling for model family, task, prompt, and output-length effects, we find that fine-tuning nearly triples the correlation strength between entropy rate and semantic diversity, suggesting that aligned models convert token uncertainty into semantic diversity more efficiently. Overall, these results demonstrate that fine-tuning does not simply reduce uncertainty, but fundamentally reorganizes it into more informative and semantically meaningful generations. Our code is available at this https URL.
>
---
#### [new 086] AI for Monitoring and Classifying Data Used in Research Literature
- **分类: cs.CL**

- **简介: 该论文属于数据集引用监测任务，旨在解决研究文献中数据使用情况不透明的问题。通过构建多任务框架和合成数据生成方法，提升数据提及识别的准确性与覆盖范围。**

- **链接: [https://arxiv.org/pdf/2605.30582](https://arxiv.org/pdf/2605.30582)**

> **作者:** Rafael Macalaba; Aivin V. Solatorio
>
> **摘要:** While platforms like Google Scholar and Semantic Scholar track citations for academic papers, no comparable infrastructure exists for monitoring dataset usage in research literature, leaving the landscape of data use largely opaque. Addressing this gap is critical for transparency, reproducibility, and monitoring of impact, yet progress is hindered by inconsistent citation practices, scarce labeled data, and ambiguous references to datasets in the wild. Traditional NLP approaches struggle with these challenges, motivating the shift toward more adaptive, semantically rich models. Building on prior work using LLMs for data mention detection and synthetic data for bootstrapping training, this paper presents an updated methodology for scalable dataset monitoring. We introduce a multitask GLiNER-based framework that jointly performs dataset mention extraction, relation identification, and usage-context classification. To address label scarcity, the pipeline leverages synthetic data generation to produce training examples and LLM-based revalidation to filter incorrect mentions and enforce labeling consistency, together improving reliability, coverage, and output consistency across the training pipeline. This work advances the development of open-source tools for monitoring data use in research literature, contributing to the broader goal of generalizable, unconstrained dataset citation tracking.
>
---
#### [new 087] Divergence Decoding: Inference-Time Unlearning via Auxiliary Models
- **分类: cs.CL**

- **简介: 该论文属于模型隐私保护任务，旨在解决LLM中敏感数据记忆问题。提出Divergence Decoding方法，通过辅助模型引导推理过程，有效实现知识删除。**

- **链接: [https://arxiv.org/pdf/2605.31293](https://arxiv.org/pdf/2605.31293)**

> **作者:** Humzah Merchant; Bradford Levy
>
> **摘要:** Large Language Models (LLMs) frequently memorize sensitive training data thereby creating significant privacy and copyright risks. Addressing these risks, i.e., removing such knowledge from an existing model checkpoint, has proven challenging as many unlearning methods lead to catastrophic utility loss or are ineffective for complex queries. We introduce Divergence Decoding (DD), a mechanism that uses small auxiliary models to steer the logits of the LLM away from specific data during inference. Training these models is straight forward, i.e., we use standard pre-training and fine-tuning setups. We find the method decisively outperforms state-of-the-art (SOTA) baselines on unlearning benchmarks across a variety of model and training dataset scales consistent with DD being an effective and inexpensive solution to unlearning. We then demonstrate that this steered distribution can be trivially distilled back into the base model. Since the method is generally applicable to any probabilistic model, we explore its efficacy outside of text generation and find evidence of generalization to the domain of images.
>
---
#### [new 088] Towards Efficient LLMs Annealing with Principled Sample Selection
- **分类: cs.CL**

- **简介: 该论文属于语言模型训练任务，解决 annealing 阶段数据选择问题。提出 DiReCT 框架，通过约束优化提升模型收敛效果。**

- **链接: [https://arxiv.org/pdf/2605.31175](https://arxiv.org/pdf/2605.31175)**

> **作者:** Yuanjian Xu; Jianing Hao; Wanbo Zhang; Zhong Li; Guang Zhang
>
> **摘要:** The annealing phase is a pivotal convergence stage in LLM pre-training that ultimately determines final model quality. However, effectively selecting training data during this phase remains a key challenge. Current strategies rely on empirical heuristics, such as domain filtering or context extension, which lack a principled grounding in optimization theory. In this work, we characterize the annealing phase through the lens of the loss landscape's spectral geometry. We argue that optimal convergence requires gradient updates to satisfy heterogeneous constraints across different eigen-directions. Building on this insight, we formulate data selection as a problem of satisfying these directional constraints. To this end, we propose DiReCT (Directionally-Restrained Constrained Training), a novel framework that reformulates sample selection in the annealing stage as a constrained optimization problem. By imposing explicit directional constraints on per-sample gradients based on the spectral properties of the Hessian, DiReCT identifies samples that align with the optimal curvature-aware descent path. Extensive experiments across various model scales demonstrate that DiReCT consistently achieves state-of-the-art performance. For future research, code is available at this https URL.
>
---
#### [new 089] Generalistic or Specific Embeddings, Which is Better? An Empirical Study on Search for Clinical Coding in Non-English Languages
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于临床编码检索任务，旨在提升非英语语言的召回率。通过使用大模型生成数据，构建双阶段检索器，优化不同语言的检索效果。**

- **链接: [https://arxiv.org/pdf/2605.30529](https://arxiv.org/pdf/2605.30529)**

> **作者:** David Rey-Blanco; Roberto Cruz
>
> **备注:** 24 pages, 12 figures, 6 tables
>
> **摘要:** Sentence-embedding models for semantic search are overwhelmingly developed and evaluated on English corpora. When applied to clinical retrieval in other languages -- particularly retrieval of ICD-10-CM / CIE-10 codes -- recall degrades in ways often masked by aggregate benchmarks. We study whether large generative language models can serve as data factories to close this gap. We build a two-stage retriever (bi-encoder followed by cross-encoder reranker), fine-tuned from a Spanish biomedical encoder (PlanTL-GOB-ES/bsc-bio-ehr-es) on Gemini-generated synthetic data covering English, Spanish, Catalan, Italian, Portuguese and French, and evaluate against BioBERT-ST and the un-tuned Spanish encoder. The bi-encoder alone matches BioBERT-ST on MRR (0.876 vs. 0.866) and overtakes it on R@3 (0.650 vs. 0.626) and R@5 (0.804 vs. 0.790) without English biomedical pretraining. Adding a cross-encoder reranker lifts aggregate R@5 to 0.822 and dominates on four of five languages (+0.017 Spanish, +0.033 Catalan, +0.018 French, +0.037 Portuguese) at the cost of a small English regression. The trade-off is clinically acceptable: Portuguese reaches R@5 = 0.829 vs. BioBERT-ST's 0.714. Contributions: an open recipe for building domain-specific medical retrievers from LLM-generated data; quantification of the learning gain (MRR 0.755 to 0.876, +15.9% with ~19,500 synthetic pairs); and a characterisation of where gains concentrate by language and rank.
>
---
#### [new 090] Beyond Agreement: Scoring Panel-Surfaced Biomedical Entity Candidates for Curator Triage
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于生物医学命名实体识别任务，解决多模型预测结果的准确性问题。通过构建基准和引入评分模型，提升候选实体的筛选效果。**

- **链接: [https://arxiv.org/pdf/2605.30826](https://arxiv.org/pdf/2605.30826)**

> **作者:** Shuheng Cao; Ruiqi Chen; Renjie Cao; Zhenhao Zhang; Siyu Zhang; Tingting Dan
>
> **摘要:** Biomedical NER is deceptively simple for modern LLMs: plausible biomedical mentions are easy to surface, but corpus-convention correctness depends on annotation conventions, span boundaries, entity granularity, and type schemas. Multi-LLM agreement is a salience signal, not corpus-convention correctness. We introduce a candidate-level panel-output benchmark for panel-surfaced candidate verification, where the unit is an aligned candidate surfaced by an explicitly defined multi-model panel rather than a standalone extractor output. The benchmark aligns eight LLMs' predictions over five public biomedical NER datasets into a candidate master table. BioConCal is an in-domain supervised scorer that instantiates this layer with inference-time gold-free agreement, mention, surface-availability, and document features for a fixed candidate stream. In domain, BioConCal improves AUROC from 0.753 for raw agreement to 0.910. At a validation-selected 0.95 precision target it selects 1,340 candidates at empirical test precision 0.939, compared with 293 for raw agreement. This corresponds to candidate-level recall 0.592 and corpus-level recall 0.523 against a within-panel row-label ceiling of 0.883. The main benefit is not recovering entities missed by every panel member, but reshaping a noisy panel stream into a higher-yield review queue. Under entity-type shift, thresholds require target-domain validation, and exact character localization remains a separate deterministic post-processing step.
>
---
#### [new 091] PatchWorld: Gradient-Free Optimization of Executable World Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PatchWorld，解决部分可观测环境下的世界模型构建问题。通过代码修复生成可执行模型，提升规划效果，揭示观察精度与决策效用的权衡。**

- **链接: [https://arxiv.org/pdf/2605.30880](https://arxiv.org/pdf/2605.30880)**

> **作者:** Jiaxin Bai; Yue Guo; Yifei Dong; Jiaxuan Xiong; Tianshi Zheng; Yixia Li; Tianqing Fang; Yufei Li; Yisen Gao; Haoyu Huang; Zhongwei Xie; Hong Ting Tsang; Zihao Wang; Lihui Liu; Jeff Pan; Yangqiu Song
>
> **备注:** 40 pages
>
> **摘要:** Text-agent environments are typically modeled as partially observable Markov decision processes (POMDPs), assuming that the simulator's latent state and transition dynamics are hidden from the agent. Yet little work has examined whether executable code can be induced to serve as a world model for prediction and planning under partial observability. We introduce PatchWorld, a gradient-free framework that turns offline trajectories into executable Python world models through counterexample-guided code repair. Instead of predicting the next observation with a black-box model, PatchWorld induces symbolic belief-state programs whose action updates can be inspected, replayed, and locally patched. Across seven AgentGym environments, PatchWorld-Simple achieves the highest code-based planning score among evaluated methods, reaching 76.4\% macro success in live one-step lookahead while invoking no LLM calls inside the world-model prediction module itself. We further find that a human-specified residual-memory bias improves surface observation fidelity but weakens decision utility. This exposes a tradeoff in executable world models, since improving observation fidelity can come at the expense of action-discriminative dynamics, and vice versa. Code is available at this https URL.
>
---
#### [new 092] MoG: Mixture of Experts for Graph-based Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于知识增强生成任务，解决复杂推理中检索信息冗余的问题。提出MoG模型，通过混合专家机制和图结构，提升生成准确性。**

- **链接: [https://arxiv.org/pdf/2605.31010](https://arxiv.org/pdf/2605.31010)**

> **作者:** Zheng Yuan; Chuang Zhou; Linhao Luo; Siyu An; Di Yin; Xing Sun; Xiao Huang
>
> **摘要:** Retrieval-augmented generation is intensively studied to ground large language models on external evidence. However, retrieving from a unified knowledge base could inevitably introduce irrelevant information that may mislead generation for complex reasoning. Inspired by the conditional computation of mixture of experts (MoE), where a router sparsely selects specialized experts alongside shared ones for each input, we propose \textbf{M}ixture \textbf{o}f experts for \textbf{G}raph-based Retrieval-Augmented Generation, i.e., \textbf{MoG}. It organizes knowledge into two core components: (i) diverse, always-accessible hub graphs that encode semantically and structurally central knowledge and provide contextual clues for expert activation, and (ii) sparsely activated expert graphs that contain domain-specific evidence. MoG first accesses hub graphs to identify general evidence and derive contextual clues. Then, a topology-aware router dynamically activates a limited set of expert graphs conditioned on the query, thereby confining retrieval to a focused evidence subspace. Extensive experiments on challenging benchmarks show that MoG consistently outperforms strong baselines, with over 20\% relative improvement on MuSiQue. Our code is available in this https URL.
>
---
#### [new 093] Bundesrecht: An Open Library and Corpus for German Statutory Reference Processing
- **分类: cs.CL**

- **简介: 该论文属于法律文本处理任务，旨在解决德语法规引用自动处理问题。提出bundesrecht资源，包含库和语料库，实现从引用字符串到规范条款的端到端处理。**

- **链接: [https://arxiv.org/pdf/2605.31338](https://arxiv.org/pdf/2605.31338)**

> **作者:** Harshil Darji; Martin Heckelmann; Christina Kratsch; Gerard de Melo
>
> **备注:** 10 pages, 1 figure. Preprint
>
> **摘要:** Statutory references are central to legal language understanding, but are difficult to process automatically, as they appear in compact and variable surface forms, may combine multiple targets, use special abbreviations, and often point to lower-level units. Existing tools for German focus either on parsing references from legal documents or accessing statutory text once citations are explicit. This paper introduces bundesrecht, an open resource for German statutory reference processing, consisting of a software library and a structured corpus of German federal law. The library parses, normalizes, and resolves German statutory references, mapping raw citation strings to structured objects, expanding compact references into canonical forms, and linking them to statutory provisions. The accompanying dataset preserves the internal hierarchy of statutes from laws to fine-granular subclauses. We evaluate the parser and normalizer on 2,944 annotated German legal references using strict exact-match and micro information extraction metrics. We further evaluate canonical reference deduplication and show that normalized references group real citation surface variants far more reliably than string matching. bundesrecht is the first open resource that covers German statutory reference processing as an end-to-end pipeline, from raw citation string to resolved statutory provision, and is available on PyPI.
>
---
#### [new 094] Exploring Autonomous Agentic Data Engineering for Model Specialization
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文提出“自主代理数据工程”任务，解决LLM在专业领域适应性差的问题，通过自主生成和优化数据提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.30407](https://arxiv.org/pdf/2605.30407)**

> **作者:** Yujie Luo; Xiangyuan Ru; Jingsheng Zheng; Jingjing Wang; Yuqi Zhu; Jintian Zhang; Runnan Fang; Kewei Xu; Ye Liu; Zheng Wei; Jiang Bian; Zang Li; Shumin Deng
>
> **备注:** Work in progress
>
> **摘要:** Large Language Models (LLMs) have demonstrated strong performance on general tasks, while often struggling to adapt to specialized domains without high-quality domain-specific data. Existing LLM-based data curation methods primarily rely on human-designed workflows, leaving it unexamined whether LLMs can autonomously execute an end-to-end data engineering pipeline for model specialization. We formalize \textbf{Autonomous Agentic Data Engineering}, a novel task designed to evaluate LLMs as autonomous data engineers that drive model specialization through end-to-end data curation. We frame data as an optimizable component and study agents that plan, generate, and iteratively optimize training data across multiple domains, guided by post-training performance improvement. Experiments show that autonomous LLM data engineers yield substantial gains, as GPT-5.2 constructs a training curriculum that improves a student model by \textbf{57.29\%}, entirely through iterative, agent-driven data adaptation. By illuminating both potential and bottlenecks, our study establishes autonomous data engineering as a measurable capability and charts a path toward agent-driven model specialization\footnote{Code will be released at this https URL.}.
>
---
#### [new 095] Same Patient, Different Words, Different Diagnosis? Evaluating Semantic Stability in Clinical LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗自然语言处理任务，旨在解决临床LLMs对语义相似输入反应不一致的问题。通过构建验证框架和评估指标，分析模型对改写提示的敏感性。**

- **链接: [https://arxiv.org/pdf/2605.30646](https://arxiv.org/pdf/2605.30646)**

> **作者:** Mahdi Alkaeed; Adnan Qayyum; Nabeel Abo Kashreef; Muhammad Bilal; Junaid Qadir
>
> **备注:** 14 pages, 5 figures
>
> **摘要:** Large Language Models (LLMs) are increasingly used in clinical applications. However, their behavior remains highly sensitive to subtle linguistic variations, such as rephrasing or syntactic variation. This sensitivity poses risks in safety-critical healthcare settings, where semantically equivalent inputs should produce consistent predictions. However, a key challenge is to ensure that prompt variations truly preserve clinical meaning, as embedding-based similarity metrics often fail to capture distinctions involving negation, temporality, or severity. To address this limitation, we propose a semantic verification framework based on Natural Language Inference (NLI) to filter meaning-preserving prompt variations, which are further refined using an LLM-as-a-judge and audited by a clinical expert. In addition, we introduce three metrics to quantify model sensitivity: MeaningPreserving Variation Sensitivity (MVS), confidence variation (\Delta C), and Worst-Case Instability (WCI). We evaluate 16 open-source general-purpose (GP) and medical LLMs within the same model families and parameter scales, using reformulated prompts derived from the DiagnosisQA and MedQA datasets. Our results demonstrate that robustness differences between domain-specific (DS) models are mixed and highly model-dependent, i.e., domain specialization does not consistently improve or reduce robustness to meaning-preserving prompt reformulations. Several DS models rank among the most robust (when compared with GP counterparts), and strong GP baselines remain competitive as well.
>
---
#### [new 096] KnowledgeGain: Evaluating and Optimizing Science News Generation for Reader Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出KnowledgeGain指标，用于评估科学新闻对读者知识增长的影响。任务是优化科学新闻生成，解决现有指标无法衡量学习效果的问题。通过实验和模拟器提升新闻质量。**

- **链接: [https://arxiv.org/pdf/2605.31099](https://arxiv.org/pdf/2605.31099)**

> **作者:** Dominik Soós; Meng Jiang; Jian Wu
>
> **摘要:** Science news is an important medium to communicate discoveries between the research communities and the public. Yet, most metrics for generated or summarized text evaluate semantic similarity and factual consistency, but do not measure how much knowledge readers learn from the news. We introduce KnowledgeGain, a metric that evaluates the quality of science news by measuring how much knowledge readers gained after reading it. To evaluate the metric, we first performed a controlled human study and showed that the metric successfully captures the differential knowledge gained by human readers reading different types of science media. The data allowed us to calibrate a prompt-only LLM reader simulator. We use it to rank and filter candidate articles before human evaluation. A second human study shows that articles selected with this simulator improve post-reading accuracy and normalized KnowledgeGain over a strong generation baseline. Our work is a step toward generating science news that better meets the knowledge and comprehension goals of Bloom's Taxonomy.
>
---
#### [new 097] Counterfactual Graph for Multi-Agent LLM Calibration
- **分类: cs.CL**

- **简介: 该论文属于多智能体语言模型校准任务，解决通信导致的虚假共识问题。提出CAGE-CAL框架，通过对比有无通信的图结构，提升模型可靠性与信心校准。**

- **链接: [https://arxiv.org/pdf/2605.30653](https://arxiv.org/pdf/2605.30653)**

> **作者:** Jiatan Huang; Mingchen Li; Ziming Li; Sunjae Kwon; Hong Yu; Chuxu Zhang
>
> **摘要:** Multi-agent LLM systems often treat agreement as evidence: when many agents in a panel give the same answer, that answer is assumed to be more reliable. We show that this assumption can fail after agents communicate. Communication can induce correlated failures and false consensus, so the same vote share may reflect reliable agreement in one topology but over-confidence in another. We propose CAGE-CAL, a counterfactual agent-graph calibration framework for multi-agent LLMs. For each query, CAGE-CAL compares an observed post-communication agent graph with a matched counterfactual no-communication graph, capturing both pairwise failure correlations and group-level dependencies. Rather than simply counting how many agents agree, CAGE-CAL estimates the counterfactual shift between observed and no-communication dependence, and calibrates confidence accordingly. Across five benchmarks, CAGE-CAL improves reliability discrimination with competitive ECE, and its calibrated confidence further improves topology selection over the best fixed-topology strategy.
>
---
#### [new 098] Consolidating Rewarded Perturbations for LLM Post-Training
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型后训练任务，旨在解决高效提升模型性能的问题。通过整合奖励扰动，提出CoRP方法，在单次推理中实现多扰动模型的集成效果。**

- **链接: [https://arxiv.org/pdf/2605.31494](https://arxiv.org/pdf/2605.31494)**

> **作者:** Zheyu Zhang; Shuo Yang; Gjergji Kasneci
>
> **摘要:** Post-training of language models is commonly framed as a sample-score-update loop implemented by gradient descent. A recent line of work, exemplified by RandOpt, relocates this loop to weight space, sampling Gaussian perturbations around a pretrained model and ensembling the top-K rewarded specialists at inference. While competitive with PPO and GRPO under matched training compute, this prediction-level ensemble incurs K forward passes per test example and does not extend cleanly to free-form generation. We ask whether the rewarded population can instead be folded into a single deployable model, replacing the inference-time ensemble with one consolidated update. A split-half analysis over 25 model-task pairs reveals reproducible low-rank structure in every case. We turn this geometry into CoRP (Consolidating Rewarded Perturbations), a gradient-free operator that combines reward-weighted aggregation, compatibility-aware reweighting, and a held-out validation gate, with no gradient flowing through the language model. Across five language models from 0.5B to 8B and five tasks covering math, code, and creative writing, CoRP improves the base model by 8.1 points on average. Using one tenth of RandOpt's perturbation budget, CoRP exceeds single-inference RandOpt by 6.5 points and recovers more than half of the gain of the 50-pass majority-vote ensemble, at one forward pass per test example.
>
---
#### [new 099] UniAudio-Token: Empowering Semantic Speech Tokenizers with General Audio Perception
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出UniAudio-Token，解决语义语音分词器在音频感知上的不足，通过结构化监督和内容感知机制，提升其通用音频理解能力。**

- **链接: [https://arxiv.org/pdf/2605.31521](https://arxiv.org/pdf/2605.31521)**

> **作者:** Yuhan Song; Linhao Zhang; Aiwei Liu; Chuhan Wu; Sijun Zhang; Wei Jia; Yuan Liu; Houfeng Wang; Xiao Zhou
>
> **备注:** 19 pages, 10 figures
>
> **摘要:** Semantic speech tokenizers have become a widely used interface for Audio-LLMs, owing to their compact single-codebook design and strong linguistic alignment. However, their focus on linguistic abstraction induces acoustic blindness, limiting their applicability beyond speech-centric tasks. We propose UniAudio-Token, a framework that empowers semantic tokenizers with general audio perception without compromising speech ability. Instead of altering the semantic paradigm, UniAudio-Token mitigates its information loss through two key innovations: (1) Semantic-Acoustic Primitives (SAP) provide structured supervision by decomposing audio into linguistic content, vocal attributes, and auditory-scene primitives; and (2) Semantic-Acoustic Equilibrium (SAE) introduces a content-aware gating mechanism that adaptively restores fine-grained acoustic details from shallow layers. Extensive evaluations show that UniAudio-Token learns comprehensive universal representations while preserving high-fidelity speech generation. When integrated with downstream LLMs, it outperforms all single-codebook baseline tokenizers on both understanding and generation tasks, effectively serving as a unified audio interface. We publicly release all our code, including training and inference scripts, together with the model checkpoints at this https URL.
>
---
#### [new 100] Shared Doubt: Zero-shot Cross-Lingual Confidence Estimation for Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型的置信度估计任务，解决多语言环境下置信度估计方法不足的问题。通过轻量线性探测器，实现跨语言零样本泛化，无需目标语言监督。**

- **链接: [https://arxiv.org/pdf/2605.31220](https://arxiv.org/pdf/2605.31220)**

> **作者:** Athina Kyriakou; Dennis Ulmer; Ivan Titov
>
> **摘要:** Confidence estimation (CE), i.e. quantifying the reliability of a model's prediction, has attracted great interest in the context of large language models (LLMs). However, most studies focus on English, ignoring the multilingual reality of LLM usage, while many CE methods degrade or require retraining across languages. To address this gap, we investigate whether multilingual LLMs encode shared, language-transferable confidence features. We use a lightweight linear probe that predicts answer correctness directly from intermediate representations. Trained monolingually, the probe generalizes zero-shot to unseen, typologically diverse languages without target-language supervision. Learned layer weights and multiple ablations reveal that confidence features concentrate in middle layers across languages, suggesting a shared confidence subspace. While zero-shot cross-lingual performance depends on similarity to the source language, the probe provides a strong baseline without any retraining and compares favorably to other popular confidence estimation methods.
>
---
#### [new 101] Refining Word-Based Grammatical Error Annotation for L2 Korean
- **分类: cs.CL**

- **简介: 该论文属于L2韩语语法错误修正任务，解决词级评估与形态学错误不匹配的问题，通过优化标注方案和增加参考修正提升评估效果。**

- **链接: [https://arxiv.org/pdf/2605.30545](https://arxiv.org/pdf/2605.30545)**

> **作者:** Jungyeul Park; Kyungtae Lim; Wonjun Oh; Benjamin Nguyen; Zihao Huang; Mengyang Qiu; Jayoung Song
>
> **摘要:** Korean grammatical error correction (K-GEC) presents a structural mismatch between word-based evaluation and the morpheme-level locus of many learner errors. Postpositions and verbal endings are bound to lexical hosts, but they encode grammatical relations that must be represented in correction and evaluation. This paper refines word-based grammatical error annotation for L2 Korean by addressing three connected problems in existing resources: surface target realization, Korean-specific edit annotation, and single-reference evaluation. We reconstruct target sentences from the National Institute of Korean Language (NIKL) L2 corpus under morphologically constrained realization rules and convert its morpheme-level annotations into word-level \texttt{m2} edits. We then define a Korean ERRANT-style annotation scheme that preserves the MRU core while distinguishing functional morpheme errors, spelling errors, word boundary errors, and word order errors. We also augment the KoLLA corpus with an additional reference correction, yielding a multi-reference evaluation setting for Korean GEC. Empirical validation shows that the refined NIKL targets yield lower perplexity, the converted \texttt{m2} files achieve higher agreement with source-target edit representations, and the refined resources improve KoBART-based correction under the same model setting. Multi-reference KoLLA evaluation further reduces the penalty imposed on valid corrections that diverge from a single reference, especially for neural and prompted GEC systems. These results show that Korean GEC evaluation depends not only on correction models, but also on reference data and edit annotations that reflect Korean morphology, spacing, and correction variability.
>
---
#### [new 102] Cognitive Fatigue in Autoregressive Transformers: Formalization and Measurement
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究语言模型在长文本生成中的性能退化问题，提出"认知疲劳"概念及量化指标FI，用于实时监测模型状态。属于自然语言处理任务，解决模型稳定性与可靠性问题。**

- **链接: [https://arxiv.org/pdf/2605.30981](https://arxiv.org/pdf/2605.30981)**

> **作者:** Riju Marwah; Ritvik Garimella; Vishal Pallagani; Atishay Jain; Michael Stewart; Amit Sheth
>
> **备注:** 9 pages, 7 figures. Accepted at the 43rd International Conference on Machine Learning (ICML 2026)
>
> **摘要:** Autoregressive language models frequently degrade during long-horizon generation, producing repetitive text, losing instruction adherence, and exhibiting unstable entropy. Despite the prevalence of these failures, practitioners lack online diagnostics to detect them in real-time as they occur. We formalize this degradation as cognitive fatigue, a measurable generation-time state characterized by decay in attention to the original prompt, representational drift, and entropy miscalibration. We introduce the Fatigue Index (FI), a lightweight, model-agnostic diagnostic that aggregates these three signals under explicit axioms (monotonicity, boundedness, interpretability) enabling reliable runtime monitoring. Across nine models (1B-13B parameters), FI trajectories exhibit structured temporal dynamics, predict task degradation (AUROC = 0.95) and repetition (Spearman rho = 0.94), and reveal non-monotonic scaling behavior: instruction-tuned models below 3B exhibit faster collapse than base models, with this trend reversing at 7B. Stress analyses further show that FI onset accelerates under longer contexts, middle-positioned evidence, and reduced numerical precision. These results establish cognitive fatigue as a coherent and measurable phenomenon, and position FI as a principled tool for runtime reliability monitoring in production LLM systems.
>
---
#### [new 103] Language Models Learn Constructional Semantics, Not To Mention Syntax: Investigating LM Understanding of Paired-Focus Constructions
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究开放源代码语言模型对罕见构式（如"let alone"）的理解能力，探讨其语义习得机制。**

- **链接: [https://arxiv.org/pdf/2605.31586](https://arxiv.org/pdf/2605.31586)**

> **作者:** Wesley Scivetti; Ethan Wilcox; Nathan Schneider; Kanishka Misra; Leonie Weissweiler
>
> **备注:** Conference on Natural Language Learning (CoNLL) 2026
>
> **摘要:** Grasping the semantics of rare constructions (form-meaning pairings) has been shown to be a challenging problem that has currently only been solved by the largest LLMs. It remains an open question if open-source models have robust constructional understanding, and if so, what learning dynamics underlie the acquisition of this knowledge. Focusing on a set of rare Paired-Focus constructions in English (e.g. "let alone", "much less"), we construct a novel dataset to test their meanings using both scalar adjectival semantics and general world knowledge. Testing a wide range of models differing in parameter count, architecture, and pretraining dataset size, we find that several modestly sized models are sensitive to both the forms and the meanings of Paired-Focus constructions, though models trained on human-scale data fail at all meaning evaluations. Turning to training dynamics for a set of open-checkpoint models, we find that Paired-Focus understanding emerges later in training than Paired-Focus syntactic knowledge, and that learning of Paired-Focus semantics is correlated with gains in some domains of world knowledge. Overall, our empirical results support the conclusion that modestly sized open-source models can grasp the rare Paired-Focus constructions, and demonstrate a connection between knowledge of Paired-Focus constructions and other meaning domains.
>
---
#### [new 104] Mellum2 Technical Report
- **分类: cs.CL**

- **简介: 该论文介绍Mellum2，一个12B参数的Mixture-of-Experts语言模型，专精软件工程任务，解决代码生成、调试等问题。通过优化架构和训练策略，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2605.31268](https://arxiv.org/pdf/2605.31268)**

> **作者:** Marko Kojic; Ivan Bondyrev; Aral de Moor; Joseph Shtok; Petr Borovlev; Kseniia Lysaniuk; Madeeswaran Kannan; Ivan Dolgov; Nikita Pavlichenko
>
> **摘要:** We present Mellum 2, an open-weight 12B-parameter Mixture-of-Experts (MoE) language model with 2.5B active parameters per token. Mellum 2 is a general-purpose language model specialized in software engineering, spanning code generation and editing, debugging, multi-step reasoning, tool use and function calling, agentic coding, and conversational programming assistance, and it is the successor to the completion-focused 4B dense Mellum model. The architecture builds on the Mixture-of-Experts (64 experts, 8 active) and combines Grouped-Query Attention with 4 KV heads, Sliding Window Attention on three of every four layers, and a single Multi-Token Prediction head that doubles as both an auxiliary pre-training objective and a built-in draft model for speculative decoding; each choice was validated by ablation with inference efficiency on commodity GPUs as a design constraint. Pre-training spans approximately 10.6 trillion tokens through a three-phase curriculum that progressively shifts the mixture from diverse web data toward curated code and mathematical content, optimized with Muon under FP8 hybrid precision and a Warmup-Hold-Decay schedule with linear decay to zero. The pre-trained base is extended to a 128K context window via a layer-selective YaRN and then post-trained in two stages (supervised fine-tuning followed by RLVR), yielding two released variants: an Instruct model that answers directly and a Thinking model that emits an explicit reasoning trace before its final answer. Across code generation, math and reasoning, tool use, knowledge, and safety benchmarks, Mellum 2 is competitive with open-weight baselines in the 4B-14B range while running at the per-token compute of a 2.5B dense model. We release the base, instruct, and thinking checkpoints, together with this report on the architecture decisions, data pipeline, and training recipe behind them, under the Apache 2.0 license.
>
---
#### [new 105] Learning Whom to Trust: Market-Feedback Adaptive Retrieval for Frozen LLMs in Event-Driven Financial RAG
- **分类: cs.CL**

- **简介: 该论文属于金融RAG任务，解决事件驱动下的证据检索问题。通过适应性检索和市场反馈优化，提升预测效果。**

- **链接: [https://arxiv.org/pdf/2605.31201](https://arxiv.org/pdf/2605.31201)**

> **作者:** Zijie Zhao; Roy E. Welsch
>
> **摘要:** Financial retrieval-augmented generation (RAG) systems typically rank evidence by textual relevance, but in financial markets the useful evidence source depends on event type, forecast horizon, and market context. We study news-triggered event-impact prediction as a point-in-time financial RAG problem. For each company-news anchor, the system retrieves related financial news and SEC filing passages, appends a pre-decision market-context card, and predicts multi-horizon residual-return signals. Our method keeps the large language model (LLM) reader frozen and adapts the retrieval layer through an external Bayesian source memory updated from matured residual-return feedback. On a fixed 89-stock Nasdaq-oriented universe derived from the FinRL-DeepSeek/FNSPID task, using original FNSPID news and point-in-time EDGAR filing passages, Frozen Reader with Source Memory improves held-out macro-F1 from 0.438 to 0.471 and downstream portfolio Sharpe from 0.52 to 0.84 relative to Frozen Reader with No Memory. A supervised LoRA reader improves static RAG modestly, but does not improve over the frozen source-memory reader. These results suggest that, for financial RAG, learning where to retrieve from can be as important as learning how to read, offering a simple, modular route to market-feedback adaptation.
>
---
#### [new 106] ElasticMem: Latent Memory as a Learnable Resource for LLM Agents
- **分类: cs.CL**

- **简介: 该论文提出ElasticMem，解决LLM代理长期记忆不足的问题，通过可变预算的弹性记忆机制提升任务表现。属于自然语言处理中的记忆增强任务。**

- **链接: [https://arxiv.org/pdf/2605.30690](https://arxiv.org/pdf/2605.30690)**

> **作者:** Tao Feng; Chongrui Ye; Tianyang Luo; Jingjun Xu; Xueqiang Xu; Haozhen Zhang; Ge Liu; Jiaxuan You
>
> **摘要:** Long-term memory is essential for LLM agents to reason coherently across extended interactions, personalize responses, and reuse past experience. However, existing memory-augmented methods typically treat memory as a fixed resource: text-space approaches concatenate retrieved memories into the context window, causing substantial token overhead and sensitivity to noisy evidence, while latent-space approaches reduce textual cost but still rely on rigid retrieval or fixed-capacity memory interfaces. This creates a mismatch between query-dependent memory utility and fixed memory allocation. We propose ElasticMem, a memory-augmented LLM framework that learns to use memory as an elastic latent resource. ElasticMem builds an offline latent memory bank with retrieval keys and content caches, retrieves memories adaptively from the reasoner's hidden state, assigns each retrieved memory a variable latent budget through a learned policy, and injects selected latent states as soft memory tokens for generation. The full memory-use process is optimized with downstream task rewards through group-relative policy optimization. We evaluate ElasticMem on MemorySuite, covering memory-intensive QA and embodied agent control. Across Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct backbones, ElasticMem improves weighted average QA accuracy by 26.2% and 24.6%, and improves ALFWorld success rate by 66.3% and 27.2%, respectively, over the strongest baselines, while achieving the lowest ALFWorld token cost. Ablations and qualitative analyses further show that adaptive retrieval and elastic budget allocation help ElasticMem prioritize useful evidence and transferable plans beyond rigid cosine similarity. Our code for ElasticMem will be released at this https URL.
>
---
#### [new 107] Scaling Conversational Hungarian ASR: The BEA-Dialogue+ Corpus
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决匈牙利语对话数据不足的问题。通过扩展BEA-Dialogue语料库，增加训练数据量并研究模型在不同分割下的表现。**

- **链接: [https://arxiv.org/pdf/2605.31469](https://arxiv.org/pdf/2605.31469)**

> **作者:** Máté Gedeon; Piroska Zsófia Barta; Péter Mihajlik; Katalin Mády
>
> **摘要:** Conversational automatic speech recognition in Hungarian is constrained by the limited amount of publicly available dialogue-style training data. The BEA-Dialogue corpus addresses this need, but its strictly speaker-disjoint train/dev/eval split reduces the usable material to only 85 hours. In this paper, we introduce BEA-Dialogue+, an expanded version of the corpus that relaxes the split criterion for experimenters and dialogue partners while preserving complete separation of the primary speakers. This results in 200 hours of transcribed natural conversations and enables a controlled study of the trade-off between additional training data and speaker overlap across the splits. We evaluate several Whisper- and FastConformer-based models on both corpus versions, including Serialized Output Training (SOT)-based fine-tuning for dialogue transcription. Our results show that the larger corpus is more challenging for models without fine-tuning, whereas SOT-based adaptation yields consistent improvements in WER, CER, cpWER, and cpCER. Overall, BEA-Dialogue+ provides a substantially larger yet still demanding benchmark for Hungarian dialogue ASR, and a practical resource for training and evaluating dialogue transcription systems.
>
---
#### [new 108] The Sword, Shield, and Achilles' Heel: Characterizing the Linguistic Inductive Bias of Large Language Models for Spatial Reasoning in Navigation Planning
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM在导航规划中的语言归纳偏置，解决如何优化文本空间表示的问题。通过双干预框架分析语言结构与上下文特征的影响，提出有效表示应保持拓扑完整性和语义正确性。**

- **链接: [https://arxiv.org/pdf/2605.31404](https://arxiv.org/pdf/2605.31404)**

> **作者:** Xudong Zhang; Jian Yang; Shengkai Wang; Jiangpeng Tian; Shaowen Chen; Xian Wei; Ke Li; Xiong You
>
> **摘要:** Large Language Model (LLM)-based navigation systems commonly construct explicit spatial representations (e.g., topological graphs, semantic raster maps) and translate them into textual descriptions as LLMs' inputs. However, the linguistic structures of such text-based spatial representations and the choices of contextual features (e.g., topology, geometry) they contain are often treated as neutral engineering decisions rather than key factors that shape LLMs' behavior. To fill the gap, we propose a dual-interventional framework that disentangles linguistic structures from different contextual cues to evaluate the linguistic inductive bias of LLMs for navigation planning. In the framework, representation intervention varies the linguistic format and the degree of linguistic compression, clarifying when linguistic representations support or inhibit navigation planning. Context intervention, combined with contextual feature combination and conflict probing, explicitly clarifies the preferences and weaknesses of LLMs when processing different contextual cues. Experiments across diverse spatial reasoning tasks and multiple model scales reveal a consistent pattern: topological information is a sturdy shield and the backbone of robust planning; linguistic format is a double-edged sword whose effect depends on model size, task demands, and the compression level; and semantic information is a fatal Achilles' heel -- incorrect semantic cues can systematically derail the planning process. Overall, our study shows that effective text-based spatial representations in LLM-based navigation should preserve topological integrity, calibrate representational compression to model capacity, and ensure semantic correctness, rather than simply adopting a single representation. Our code is publicly available at this https URL.
>
---
#### [new 109] TRACE: Discovering Task-Specific Parameter via Adaptation-Aware Probing for Continual Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文属于持续微调任务，解决模型在多任务中遗忘旧知识的问题。通过发现任务特定参数，提出TRACE方法，仅更新关键参数以保留先前知识。**

- **链接: [https://arxiv.org/pdf/2605.31025](https://arxiv.org/pdf/2605.31025)**

> **作者:** Xiaosong Han; Ke Chen; Xindi Dai; Di Liang; Minlong Peng; Wei Pang; Fausto Giunchiglia; Xiaoyue Feng; Yonghao Liu; Renchu Guan
>
> **备注:** KDD2026
>
> **摘要:** In real-world deployment, LLMs are often adapted continually across tasks to keep LLMs up-to-date in production, where new fine-tuning should preserve previously learned skills. However, indiscriminately mixing tasks can dilute task specialization, while sequential fine-tuning (full-parameter or low rank adaptation) often causes catastrophic forgetting due to destructive overwriting. Replay-based continual tuning and maintaining separate task-specific adapters can mitigate forgetting, but introduce additional compute, storage, and management overhead. Recognizing the redundancy of LLM parameters for any single task, we reframe continual task adaptation as task-specific parameter discovery via adaptation-aware probing: a short warm-start probe exposes a task's adaptation trace, enabling us to identify and isolate the small subset of parameters essential for each task to mitigate catastrophic forgetting. Building on this view, we introduce TRACE, a novel approach for discovering Task-specific paRameters via Adaptation-aware probing for Continual finE-tuning. We perform a short warm-start fine-tune to derive task-specific core parameters by comparing the warm-started and pre-trained models. Core parameters are identified via two strategies: importance scoring (L$_2$ norm and Fisher Information) and specificity analysis (cosine similarity of parameter updates). In continual fine-tuning settings, only the active task's core parameters are updated while others remain frozen, preserving prior knowledge. We conduct extensive experiments across multiple standard benchmarks to demonstrate the superior performance of our proposed method. Additionally, we validate the generalization of our method through a cross-model and scale transferability study, demonstrating a "small-to-large" paradigm that guides the fine-tuning of large-scale models under resource constraints.
>
---
#### [new 110] Neuro-symbolic Syntactic Parsing: Shaping a Neural Network with the CYK Algorithm
- **分类: cs.CL; cs.AI; cs.DS**

- **简介: 该论文属于自然语言处理中的语法解析任务，旨在将CYK算法嵌入神经网络。通过设计CYKNN架构，实现更高效的语言模型解析，优于现有大模型。**

- **链接: [https://arxiv.org/pdf/2605.31421](https://arxiv.org/pdf/2605.31421)**

> **作者:** Fabio Massimo Zanzotto; Federico Ranaldi; Giorgio Satta
>
> **备注:** 9 content pages
>
> **摘要:** In this paper, we show the possibility of a direct injection of algorithms into neural network architecture. We focus on a complex algorithm, that is, Cocke-Youger-Kasami (CYK) for parsing context-free grammars in Chomsky Normal Form and we propose CYKNN, a simple recurrent neural network architecture for encoding the CYK algorithm in trainable matrix-vector this http URL experimented with a very simple grammar with 4 variations showing that our approach outperforms existing LLMs with more than 20B parameters with an in-context learning setting and smaller LLMs of the Qwen family fine-tuned with LoRA. Our attempt paves the way to a different approach to neuro-symbolic methodologies.
>
---
#### [new 111] Reliable Multilingual Orthopedic Decision Support from Clinical Narratives: Language-Aware Adaptation and Verification-Guided Deferral
- **分类: cs.CL**

- **简介: 该论文属于多语言医疗分类任务，解决低资源环境下临床文本的可靠决策支持问题。工作包括对比多种模型，提出语言感知的IndicBERT-HPA框架，并引入验证层提升可靠性。**

- **链接: [https://arxiv.org/pdf/2605.31512](https://arxiv.org/pdf/2605.31512)**

> **作者:** Danish Ali; Li Xiaojian; Sundas Iqbal; Farrukh Zaidi
>
> **摘要:** Multilingual orthopedic decision support remains challenging in low-resource healthcare settings, where clinical narratives contain specialized terminology, mixed scripts, incomplete evidence, label imbalance and language-dependent documentation patterns. This article presents a reliability-oriented framework for classifying free-text orthopedic notes in English, Hindi and Punjabi. We compare task-aligned multilingual transformer encoders, a task-fine-tuned DistilBERT baseline, zero-shot instruction-tuned large language models (LLMs) and a domain-adaptive encoder, IndicBERT-HPA. IndicBERT-HPA augments IndicBERT with language-aware orthopedic adapter heads to support clinically relevant multilingual representation learning. Evaluation extends beyond aggregate accuracy to per-class performance, ROC-AUC, AUPRC, expected calibration error, cross-language stability and robustness under controlled balanced and natural-prevalence distributions. The evaluated zero-shot LLMs remain substantially less effective than task-adapted encoders for closed-set classification, with language-dependent instability. Under natural clinical prevalence, IndicBERT-HPA achieves the strongest overall performance, reaching an averaged Macro-F1 of 0.8792, Macro-AUROC of 0.894 and AUPRC of 0.902. We further implement a deterministic selective-verification layer combining confidence gating, evidence-consistency checking and language-risk screening. On a randomly selected held-out 5,000-record subset, it achieves 84.4% selective accuracy and 0.76 selective Macro-F1 at 72.3% coverage, compared with 71.5% accuracy and 0.65 Macro-F1 for accept-all prediction. These results support reliability-oriented multilingual clinical decision support with explicit deferral.
>
---
#### [new 112] Social Reasoning in Machines: Investigating Collective Truth-Seeking Dynamics in Large Language Model Debate
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 论文探讨机器如何通过集体辩论提升真理探索能力，属于人工智能与社会推理领域。旨在解决个体模型性能有限时的真理获取问题，通过多模型辩论实验验证集体推理优势。**

- **链接: [https://arxiv.org/pdf/2605.30391](https://arxiv.org/pdf/2605.30391)**

> **作者:** Tom Pecher
>
> **备注:** Master's thesis
>
> **摘要:** Human reasoning has long been theorised to operate socially, not through isolated individual cognition, but through collective adversarial discourse, a framework known as the Argumentative Theory of Reasoning (ATR). Rather than relying on individual "intellectualist reasoners" as the primary vehicle for truth-seeking, ATR reconceptualises truth as an emergent property of social epistemology: the product of imperfect individual reasoning refined under the adversarial pressure of debate. This distributed method of collective intelligence has guided humanity to ever-greater epistemic heights and underpins the foundational principles of all democratic systems. This thesis breaks new ground by, for the first time, simulating ATR through the multi-agent debate (MAD) of large language models (LLMs). With rigorous empirical analysis, we demonstrate that, when correctly engineering an epistemically diverse set of models, LLM-MAD can significantly improve truth-seeking performance on questionnaire-based tasks, even when individual debate participants exhibit limited standalone performance. Furthermore, we present strong empirical evidence that this performance gain is mechanistically grounded in the central principles of ATR, suggesting that collective reasoning may be universally favourable over individualist reasoning, rather than a quirk in biology or evolution. Finally, drawing on our analysis of debate dynamics, we propose a novel benchmarking methodology that leverages LLM-MAD to measure intrinsic model properties (such as hallucination propensity) in order to compare models in ways that current static benchmarking approaches cannot support.
>
---
#### [new 113] MAAT: Multi-phase Adapter-Aware Targeted Unlearning
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出MAAT方法，解决机器遗忘中因果知识难以平衡遗忘与保留的问题，通过多阶段框架提升对"为什么"类型问题的处理效果。**

- **链接: [https://arxiv.org/pdf/2605.30514](https://arxiv.org/pdf/2605.30514)**

> **作者:** Suryash Yagnik; Shubham Gaur; Saksham Thakur; Vinija Jain; Aman Chadha; Amitava Das
>
> **备注:** 16 pages, 4 figures, 10 tables
>
> **摘要:** Machine unlearning evaluation is structurally skewed: Why-type questions, which probe causal and relational knowledge, comprise less than 0.06% of CounterFact, 0.6% of ZSRE, and less than 1.3% of TOFU, MUSE, and WMDP-Cyber. This near-zero representation means that methods that fail on causal knowledge can score highly in aggregate, and this failure is undetectable without balanced evaluation. We present 5WBENCH, a balanced 5,000-sample benchmark with 1,000 examples per 5W category (Who, What, When, Where, Why), making causal unlearning failures quantifiable for the first time. Using 5WBENCH, we show that no existing baseline simultaneously achieves high forgetting and high retention on Why-type questions: aggressive forgetting degrades retained knowledge, while conservative methods fail to forget causal facts. Why-type difficulty stems from multi-hop reasoning chains (44% of Why entries vs. less than or equal to 2% for others) and gradient dilution over 40.1-token answer spans. We present MAAT (Multi-phase Adapter-Aware Targeted Unlearning), a three-phase framework operating on LoRA adapter weights, combining gradient-projected ascent, SVD rank-dimension pruning, task vector negation, and hybrid KL-hidden-state retain repair. MAAT is the first method to simultaneously achieve high forgetting and high retention on Why-type causal knowledge, reaching a new operating point on the forget-retain Pareto frontier. We make our code publicly available.
>
---
#### [new 114] Triaging Threats to Specialized Guardrails
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于安全防护任务，解决现有防护机制在多样化威胁下的泛化不足问题。提出GuardZoo基准和RouteGuard框架，提升威胁检测效果与扩展性。**

- **链接: [https://arxiv.org/pdf/2605.30693](https://arxiv.org/pdf/2605.30693)**

> **作者:** Wenjie Jacky Mo; Xiaofei Wen; Rui Cai; Boyu Zhu; Sicong Jiang; Zihan Wang; Minglai Yang; Zhe Zhao; Muhao Chen
>
> **摘要:** Building robust safety guardrails is essential for deploying Large Language Models across diverse real-world applications. However, this goal remains challenging because safety risks span heterogeneous threat domains, while existing datasets cover only fragmented risk subsets and rely on inconsistent taxonomies. Consequently, it remains unclear whether current guardrails can generalize beyond narrow evaluation settings. To better understand the robustness of guardrail models, we first introduce GuardZoo, a unified human-annotated benchmark with 32,460 samples covering 15 distinct unsafe categories. Evaluation on GuardZoo reveals that monolithic guardrails suffer from task interference: different threat domains require distinct decision boundaries that are difficult to compress into a single model. We therefore propose RouteGuard, a router-expert framework that triages each conversation to specialized expert guardrails for threat-specific detection. Experiments show that RouteGuard improves fine-grained threat detection over strong guardrail baselines, generalizes better under out-of-domain evaluation, and supports flexible modular expansion to emerging threats.
>
---
#### [new 115] How Early Adopters Used Generative AI Worldwide: Variation by Country Income and Language
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于社会技术研究任务，探讨不同国家早期用户使用生成式AI的差异，分析收入和语言对使用模式的影响。**

- **链接: [https://arxiv.org/pdf/2605.30685](https://arxiv.org/pdf/2605.30685)**

> **作者:** Madeleine I. G. Daepp; Isaac Slaughter
>
> **摘要:** AI is being used by people globally, but not everyone is using it in the same ways. Using a large-scale dataset of anonymized, de-identified, and privacy-scrubbed interactions with a widely available and free AI chatbot, we empirically characterize differences in early adopters' usage across countries. Schooling is the most common domain of use in most countries, particularly low-income countries, with a strong inverse association evident between schooling and country-level GDP. Leisure-related use, by contrast, is positively associated with country-level income. Language, we find, also shapes use: English-language interactions are overrepresented in places where the predominant languages were not well-served by existing models during the period of the study. Improving performance across languages may be a key factor, our work suggests, in whether this technology expands digital divides or enables leapfrogging.
>
---
#### [new 116] LLM Anonymization Against Agentic Re-Identificatio
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于文本匿名化任务，解决对抗网络搜索的重新识别问题。提出AURA框架，在保护隐私的同时保留文本实用价值。**

- **链接: [https://arxiv.org/pdf/2605.30848](https://arxiv.org/pdf/2605.30848)**

> **作者:** Ziwen Li; Jianing Wen; Tianshi Li
>
> **备注:** 32 pages, 7 figures
>
> **摘要:** Agentic LLMs with web search change the threat model for text anonymization: weak contextual cues can become cross-referenceable evidence for re-identification, yet those same details also carry downstream analytic value of the text. Existing defenses either remove explicit identifiers, perturb text for formal privacy, or test rewritten text against non-web inference models, leaving underexplored the operating region between resistance to agentic web-search re-identification and utility retention. We introduce AURA (\textbf{A}nonymization with \textbf{U}tility-\textbf{R}etention \textbf{A}daptation), an LLM-powered \textit{mask-reconstruct} framework that decouples privacy localization from utility-preserving reconstruction and selects candidates with adversarial privacy and utility-retention checks. We evaluate AURA on real-user interview transcripts using re-identification attacks carried out by web-search agents, along with a utility evaluation based on interviewee-profile facts, codebook facts, and the joint contextual utility grid. Our results show that AURA improves the privacy-utility frontier by using adaptive privacy scope to strengthen resistance to agentic re-identification and using a mask-reconstruct anonymization method to better preserve contextual utility under fixed privacy scope.
>
---
#### [new 117] Generating Reports or Repeating Templates? Measuring and Mitigating Template Collapse in 3D CT Report Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于3D CT报告生成任务，解决模板坍塌问题，通过CLarGen框架提升临床准确性与多样性。**

- **链接: [https://arxiv.org/pdf/2605.30984](https://arxiv.org/pdf/2605.30984)**

> **作者:** Tom Maye-Lasserre; Yitong Li; Bailiang Jian; Morteza Ghahremani; Benedikt Wiestler; Christian Wachinger
>
> **摘要:** Modern 3D medical vision-language models (VLMs) can generate fluent radiology-style text while exhibit critically low pathology detection and output diversity, collapsing to generic templates that under-report rare yet critical findings. We identify this failure mode as Template Collapse. This failure stems from the unique constraints of 3D medical imaging, e.g., limited data, severe label imbalance, and weak signals from volumetric encoders. Under these constraints, text-generation objectives encourage shortcut learning and fluent but weakly grounded reports. We systematically diagnose the Template Collapse through clinical fidelity, output diversity, normal-template bias, and rare-finding survival. To mitigate it, we propose CLarGen, a decoupled framework that separates what to say (clinical detection) from how to say it (language synthesis). CLarGen uses (i) a Latent Query Transformer for multi-label pathology detection, (ii) pathology-guided retrieval for clinically matched exemplars, and (iii) a medical language model to synthesize the final report from detected findings and retrieved context. Across state-of-the-art 3D CT report generation baselines, CLarGen mitigates Template Collapse and substantially improves clinical accuracy (macro-F1 0.487 vs. 0.189; CRG 0.472 vs. 0.368) while maintaining fluent reporting. Our results suggest that explicit, measurable clinical grounding is essential for template-collapse-resistant 3D CT report generation. Code will be released upon acceptance.
>
---
#### [new 118] A Padding Method for Enhanced Encoding of Inorganic Structures with Varying Chemical Compositions
- **分类: cond-mat.mtrl-sci; cs.CE; cs.CL**

- **简介: 该论文属于材料科学中的生成模型任务，旨在提升无机材料的生成准确性和效率。通过引入基于晶体对称性的填充方法，增强编码过程，生成更多稳定的新无机材料。**

- **链接: [https://arxiv.org/pdf/2605.30743](https://arxiv.org/pdf/2605.30743)**

> **作者:** Thang Dang; Haderbache Amir; Tzanakakis Alexandros; Yoshimoto Yuta
>
> **摘要:** Designing novel inorganic materials through generative models remains an important challenge for material science, driven by the complexity and diversity of inorganic structures across expansive chemical compositions and structural landscape. The vast combinatorial space of inorganic compounds demands innovative, AI-driven approaches to overcome limitations in generative accuracy and efficiency. To address this, we introduce a novel method that redefines the encoding and generation of inorganic materials by utilizing domain-specific symmetry-aware representation. Our approach not only refines the representation of intricate inorganic structures but also contributes to the field of material discovery by enhancing the precision and stability of generated candidates. Central to our methodology is a novel padding technique that exploits crystal symmetry information to enhance the encoding process. By integrating Wyckoff position length-aware padding into an encoder architecture, we achieve a more robust informed representation of inorganic materials. This symmetry-driven enhancement improves deep learning models to generate stable, previously unexplored inorganic structures with superior accuracy and computational efficiency. Furthermore, we introduce an end-to-end system that leverages the machine learning potential models to seamlessly generate novel, even those unseen in the training data, and stable inorganic materials from initial data to validated output. This pipeline integrates advanced generative models with stability analysis, marking a significant leap forward in the automated exploration and design of next-generation inorganic materials. Our method improved reconstruction accuracy 5.3% in proton conductor data, and generated 63.5% more novel stable inorganic material to baseline model on the perov-5 dataset.
>
---
#### [new 119] LongDS-Bench: On the Failure of Long-Horizon Agentic Data Analysis
- **分类: cs.LG; cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出LongDS基准，用于评估智能体在长期数据分析中的表现。针对现有基准未测试长期跟踪能力的问题，研究构建了包含68个任务的多轮数据分析数据集，揭示了模型在长期任务中的性能下降和状态维护难题。**

- **链接: [https://arxiv.org/pdf/2605.30434](https://arxiv.org/pdf/2605.30434)**

> **作者:** Kewei Xu; Xiaoben Lu; Shuofei Qiao; Zihan Ding; Haoming Xu; Lei Liang; Ningyu Zhang
>
> **备注:** Ongoing work
>
> **摘要:** Real-world data analysis is inherently iterative, yet existing benchmarks mostly evaluate isolated or short interactive tasks, leaving agents' ability to track evolving analytical context over long horizons untested. We introduce LongDS, a benchmark for long-horizon, multi-turn data analysis where agents must maintain, update, restore, and compose evolving analytical states. LongDS comprises 68 tasks constructed from real-world Kaggle notebooks, spanning 2,225 turns across six domains including Geoscience, Business, and Education. Tasks are designed around state-evolution patterns (e.g., counterfactual perturbation, rollback, multi-state composition), with an average dependency span of 11.3 turns. Evaluating five state-of-the-art models, we find that the best model reaches only 48.45% average accuracy, performance drops nearly 47 points from early to late turns, and long-horizon errors account for 52%--69% of failures. Further analysis shows that additional agent steps do not necessarily improve performance, suggesting that the key bottleneck is maintaining a correct analytical state rather than increasing interaction budget. We release LongDS to support research on reliable long-horizon agentic data analysis. Code and data will be released at this https URL.
>
---
#### [new 120] PithTrain: A Compact and Agent-Native MoE Training System
- **分类: cs.LG; cs.AI; cs.CL; cs.DC**

- **简介: 该论文属于自然语言处理领域，解决MoE训练框架优化问题。提出PithTrain框架，提升Agent-task效率，减少任务次数和GPU时间。**

- **链接: [https://arxiv.org/pdf/2605.31463](https://arxiv.org/pdf/2605.31463)**

> **作者:** Ruihang Lai; Hao Kang; Haozhan Tang; Akaash R. Parthasarathy; Zichun Yu; Junru Shao; Todd C. Mowry; Chenyan Xiong; Tianqi Chen
>
> **摘要:** Mixture-of-Experts (MoE) has become the dominant architecture for frontier language models. To meet this demand, production frameworks have built optimized MoE training stacks over years of engineering effort. Yet evolving these stacks for new architectures and system optimizations remains expensive. With the rise of AI coding agents, they could automate parts of training-framework development and accelerate this evolution. But applying them to these existing frameworks carries hidden costs, invisible to today's throughput-only evaluations. We name this missing dimension agent-task efficiency (ATE): the cost of using coding agents to understand, operate, and extend a framework. Grounded in four agent-native design principles, we build PithTrain, a compact, agent-native MoE training framework. We further introduce ATE-Bench, covering real-world training-framework tasks. Our evaluation shows PithTrain matches the throughput of production frameworks, and on ATE-Bench, PithTrain enables higher agent-task efficiency, with up to 62% fewer Agent Turns and 64% less Active GPU Time.
>
---
#### [new 121] From Prompt Injection to Persistent Control: Defending Agentic Harness Against Trojan Backdoors
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于安全防护任务，旨在解决LLM代理中多步骤后门攻击问题。通过构建ClawTrojan基准和提出DASGuard防御机制，提升对持久控制威胁的检测与防御能力。**

- **链接: [https://arxiv.org/pdf/2605.31042](https://arxiv.org/pdf/2605.31042)**

> **作者:** Jiejun Tan; Zhicheng Dou; Xinyu Yang; Yuyang Hu; Yiruo Cheng; Xiaoxi Li; Ji-Rong Wen
>
> **备注:** Code and data are available at this https URL
>
> **摘要:** LLM agents are evolving from conversational chatbots to operational tools in real-world workspaces. In local agentic harnesses, an LLM can read and write files, call tools, and reuse workspace state across sessions. While such capabilities enhance utility, they also expose a new attack surface for attackers. Attackers can embed a prompt injection within a file or tool output. Agents may read this hidden instruction, store it, and execute it later. In this multi-step trojan attack paradigm, no individual step appears malicious on its own, but these steps can collectively turn untrusted text into persistent control content. However, existing defenses often inspect each step in isolation. As a result, they can block a clear harmful action, but fail to detect the earlier write operation that plants the backdoor. To reveal this threat, we introduce ClawTrojan, a benchmark designed to identify multi-step trojan attacks in local agentic harnesses. In an OpenClaw-style simulated workspace with GPT-5.4, ClawTrojan reaches a 95.5% attack success rate (ASR), while existing single-turn prompt-injection attacks produce near-zero ASR on the same model. To address this threat, we propose DASGuard, which scans control-like text in sensitive local files, traces its origin, and removes control content that does not originate from a trusted source. Our results show that DASGuard achieves strong dynamic defense by combining runtime attack blocking with sanitized commits to the workspace.
>
---
#### [new 122] On the impact of retrieved content representations in RAG Pipelines
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于RAG任务，研究如何优化检索内容表示以提升生成效果。解决的问题是：不同内容表示对生成准确性的影响。工作是对比14种表示方法，发现答案保留率是关键因素。**

- **链接: [https://arxiv.org/pdf/2605.30790](https://arxiv.org/pdf/2605.30790)**

> **作者:** Jonathan J Ross; Bevan Koopman; Anton van der Vegt; Guido Zuccon
>
> **备注:** 23 pages, 15 figures, submitted to ACL May 2026 ARR
>
> **摘要:** Retrieval-Augmented Generation (RAG) supplements a language model's input with retrieved documents, yet most RAG pipelines inherit retrieval components designed for human readers. How retrieved content should be represented when the consumer is a large language model (LLM) rather than a human is less well understood. Recent work has proposed transformations of retrieved content and identified properties that affect generation, but each examines a single transformation or property in isolation, leaving open which features of a document's representation matter most. We address this with a controlled comparison: holding retrieval fixed, we vary only the representation of retrieved documents, comparing an original baseline against thirteen transformations spanning selection, summarisation, and reformulation, in query-dependent and query-independent variants. Across these fourteen representations we measure question-answering accuracy for four generators, and for each representation we also measure answer retention: whether a known answer-bearing document still supports its answer after transformation. We find that answer retention is the primary determinant of generator accuracy; notably, when retention is high, a representation's wording, structure, length, and query-dependence have limited effect. This suggests that accuracy gains attributed to specific mechanisms in prior work may be partly explained by how well those mechanisms preserve answer-bearing content, an attribution that cannot be settled without controlling for retention.
>
---
#### [new 123] EvoDefense: Co-Evolving Black-Box Defense with Large Language Models
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于AI安全任务，旨在解决LLM在黑盒环境下易受攻击的问题。提出EvoDefense，通过协同进化机制提升防御能力，有效降低攻击成功率。**

- **链接: [https://arxiv.org/pdf/2605.31140](https://arxiv.org/pdf/2605.31140)**

> **作者:** Yu Li; Yuenan Hou; Yingmei Wei; Yanming Guo; Chaochao Lu
>
> **摘要:** Large Language Models (LLMs) remain highly vulnerable to diverse attacks, particularly in black-box settings where the internals of target models are inaccessible. Existing black-box defenses typically rely on pre-defined filtering heuristics, which often fail to generalize to unseen attack types and target model architectures. We introduce EvoDefense, an experience-guided co-evolving black-box defense paradigm. EvoDefense employs a guard LLM to detect malicious queries and an experience memory module to accumulate defense knowledge from previous interactions. At the core of EvoDefense is a continuous attack-defense evolution loop, where an attack generator and the guard model iteratively refine their attack strategies and defense policies through experience-guided optimization. This design enables EvoDefense to generalize across unseen attacks and target models without retraining. Experiments on HarmBench, AdvBench, and AlpacaEval show that EvoDefense achieves consistently strong defense performance across seven popular models and five representative LLM attacks, while preserving competitive general capabilities. On HarmBench, EvoDefense reduces the attack success rate (ASR) of AutoDAN-turbo on Gemini-3-flash and LLaMA-3-8B-Instruct from 29.4% and 43.4% to 8.4% and 6.2%, respectively.
>
---
#### [new 124] An Organization-Scoped LLM Agent Runtime Architecture for Regulated Cybersecurity Operations
- **分类: cs.CR; cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于网络安全任务，旨在解决监管环境下LLM代理运行架构问题，提出组织级安全上下文和可审计架构，实现与现有系统集成。**

- **链接: [https://arxiv.org/pdf/2605.30604](https://arxiv.org/pdf/2605.30604)**

> **作者:** George Fatouros; Georgios Makridis; George Kousiouris; John Soldatos; Dimosthenis Kyriazis
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Regulated cybersecurity workflows lack a runtime substrate that enforces organization-level scope across retrieval, tool calls, memory, findings, reports, and audit while remaining model-agnostic and locally deployable. Recent large language model (LLM) agent systems report strong results on isolated cybersecurity tasks, yet they do not by themselves define an auditable platform architecture for regulated security operations centre (SOC) and compliance workflows, where a single analyst may trigger actions that bind the organization, and where the runtime must integrate with existing SIEM/XDR stacks as a primary source of context and alert-driven triggers rather than operate as a standalone analytical layer. This paper proposes an organization-scoped LLM agent runtime architecture for financial cybersecurity. The contribution is a typed Security Context that is created at every entry point, including SIEM/XDR notifications ingested as first-class triggers, and enforced at every component boundary, combined with a shared Runtime Core, logical specialist subagents, a governed Tool Adapter Layer exposing SIEM/XDR query, enrichment, and response primitives under uniform policy and audit, structured findings with evidence references, tiered human-in-the-loop (HITL) gates, and append-only audit. Model Context Protocol (MCP), extended telemetry, digital twins for pentesting, graph retrieval, and federated knowledge sharing are treated as optional extension paths rather than mandatory runtime assumptions. We describe an implementable slice as the architecture's testability surface, and we propose a falsifiable evaluation plan with metric-level pass criteria for architecture readiness, security-policy enforcement, evidence traceability, output quality, and operational observability.
>
---
#### [new 125] AMNESIA: A Large Scale Medical Unlearning Benchmark Suite with Disease-Informed Analysis
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出AMNESIA基准，用于医学模型的遗忘学习任务，解决如何有效删除特定患者数据影响的问题。**

- **链接: [https://arxiv.org/pdf/2605.30599](https://arxiv.org/pdf/2605.30599)**

> **作者:** Saeedeh Davoudi; Reihaneh Iranmanesh; Ophir Frieder; Nazli Goharian
>
> **摘要:** Medical knowledge is continuously evolving. This creates a need to update or selectively forget information encoded in already-trained medical LLMs. Machine unlearning aims to remove the influence of specific training data from a model without full retraining. Yet, existing unlearning benchmarks rely on synthetic or small-scale general data, leaving clinical unlearning understudied. We introduce AMNESIA, the first large-scale, open source benchmark for medical unlearning, with 70,560 question-answer pairs from 8,820 patient notes across 11 disease categories. AMNESIA includes both factual questions testing direct recall and reasoning questions testing clinical inference. We use it to evaluate four widely used unlearning methods at both random patient and disease-level, and introduce a new metric for detecting leakage of medical terminology. We show that unlearning individual patients erodes knowledge of others with the same condition, calling for methods that can better separate patients from shared clinical knowledge.
>
---
#### [new 126] TUX: Measuring Human--AI Tacit Understanding
- **分类: cs.HC; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于人机协作研究，旨在测量人类与AI之间的隐性理解。通过设计谱系任务，评估人类与AI在无明确指令下的判断相似性，提出TUX指标并分析其影响因素。**

- **链接: [https://arxiv.org/pdf/2605.30930](https://arxiv.org/pdf/2605.30930)**

> **作者:** Yueshen Li; Hanyi Min; Vedant Das Swain; Koustuv Saha
>
> **摘要:** As large language models (LLMs) increasingly act as collaborative partners, human--AI alignment is often evaluated through explicit task success, accuracy, or reward optimization. Yet many collaborative settings depend on tacit understanding: whether an agent can align with a human's evaluative stance or representational priors without clear objectives, communication, or feedback. To study this capacity, we develop a spectrum-placement task inspired by the social party game Wavelength, in which humans and agents independently place concepts along subjective spectra. We operationalize the Tacit Understanding Index (TUX) as a pairwise measure of similarity between human and agent judgments, and evaluate it with 241 human participants and 200 profile-conditioned LLM agents across four models. We find that nearest human--agent pairs in trait space achieve significantly higher TUX, suggesting that tacit alignment is structured by person-level characteristics rather than random similarity. Regression analyses show that TUX becomes more explainable as predictor sets become richer, with individual traits, decision-making styles, and confidence improving over aggregate trait-distance baselines. These findings suggest that tacit understanding between humans and LLMs is measurable, while revealing the limits of profile-based conditioning for capturing deeper representational alignment.
>
---
#### [new 127] COLLEAGUE.SKILL: Automated AI Skill Generation via Expert Knowledge Distillation
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出COLLEAGUE.SKILL系统，解决如何将人物经验转化为可检查、可修正的AI技能问题。通过知识蒸馏生成人本AI技能包，支持多场景应用。**

- **链接: [https://arxiv.org/pdf/2605.31264](https://arxiv.org/pdf/2605.31264)**

> **作者:** Tianyi Zhou; Dongrui Liu; Leitao Yuan; Jing Shao; Xia Hu
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** LLM agents are increasingly expected not only to complete isolated tasks, but also to carry bounded representations of human expertise, judgment, and interaction style. Building such person-grounded agents remains difficult because actionable knowledge associated with a person or role is usually embedded in heterogeneous traces rather than written as clean instructions. Existing memory and persona systems capture fragments of this evidence, while skill frameworks provide portable packaging formats; however, there is no end-to-end workflow for distilling these traces into inspectable, correctable, and agent-usable skills. We present an automated trace-to-skill distillation system for generating person-grounded AI skills via expert knowledge distillation. Given materials from a target person or role, this http URL produces a versioned skill package with two coordinated tracks: a capability track for practices, mental models, and decision heuristics, and a bounded behavior track for communication style, interaction rules, and correction history. The package can be inspected, invoked, updated through natural-language feedback, rolled back, installed across agent hosts, and optionally prepared for controlled distribution. We describe the artifact contract, generation workflow, correction lifecycle, deployment surface, and domain presets implemented in the open-source system. At the time of writing, the public repository has approximately 18.5k GitHub stars; the gallery lists 215 skills from 165 contributors and more than 100k cumulative stars across listed skill cards. The system illustrates how person-grounded skills can be represented as portable, correctable packages rather than opaque prompts or hidden memories.
>
---
#### [new 128] Evaluating Factual Density in Multi-Source RAG: A Study in Medical AI Accuracy
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于医疗AI事实准确性任务，旨在解决RAG系统中因关键词匹配导致的事实密度不足问题。通过引入Factual Density（FD*）优化检索，提升事实精度。**

- **链接: [https://arxiv.org/pdf/2605.31506](https://arxiv.org/pdf/2605.31506)**

> **作者:** Michael R. DeMarco
>
> **备注:** 15 pages, 7 tables. Preliminary findings; Experiment 3 identified as future work
>
> **摘要:** Retrieval-Augmented Generation (RAG) is the current industry standard for grounding AI in real-world facts. Traditional retrieval methods rely on keyword matching and topic proximity, ranking content based on how closely it sounds like the user's query. What they do not measure is how many verified facts the content actually contains. This structural gap, termed the Expert Blindness Effect, causes standard RAG pipelines to consistently bury high-density factual evidence in favor of lexically dominant text on the same topic. To address this gap, this paper introduces Factual Density (FD*), a novel retrieval optimization signal that measures the proportion of verified atomic claims relative to total token count. Using the NexusAgentics Ghost Audit preprocessing pipeline, raw text is scored for factual specificity using probabilistic factuality analysis to filter content before corpus ingestion. An initial formulation introduced a severe document-length confound (Pearson R = -0.8636, p = 2.27e-07). Implementing Z-score normalization within length bins resolved this bias, validating FD* as a length-independent density signal (p = 0.0749). Evaluated against the HealthFC benchmark (750 health claims labeled Supported, Refuted, or No Evidence by medical experts), FD*-optimized retrieval was the only condition to achieve 100% systematic review saturation in top-5 results, surfacing Cochrane evidence that standard cosine similarity ranked outside the top ten. Ground truth verification confirmed 25 mappings across seven HealthFC-supported claims. While full statistical validation across n=50 queries remains future work due to constraints on corpus-benchmark alignment, these findings establish factual density reranking as a low-cost, high-impact intervention for improving factual precision in health RAG architectures.
>
---
#### [new 129] Bounded Behavioral Indistinguishability for Black-Box LLM Distillation
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于黑盒大模型蒸馏任务，旨在解决模型行为不可区分性问题。通过定义和评估行为不可区分性指标，验证蒸馏效果是否提升而非仅提高输出相似度。**

- **链接: [https://arxiv.org/pdf/2605.30448](https://arxiv.org/pdf/2605.30448)**

> **作者:** Munawar Hasan
>
> **摘要:** Black-box LLM distillation is usually evaluated as an output-matching problem: a student is considered successful when its responses are semantically similar to, or task-consistent with, those of a teacher. However, output similarity does not imply that the student is behaviorally indistinguishable from the model it imitates. We introduce bounded behavioral indistinguishability, formalized as $(\epsilon,q,t,\mathbb{A})$-behavioral indistinguishability over an explicit prompt distribution, where $\epsilon$ bounds distinguishing advantage, $q$ bounds oracle queries, $t$ bounds computation, and $\mathbb{A}$ denotes the adversary class. We instantiate this notion on Qwen and Llama teacher-student pairs using a controlled $5,000$-prompt behavioral probe suite. For each family, we compare the teacher with both the base student and the LoRA-distilled student, measuring whether distillation reduces distinguishability rather than merely improving similarity. LoRA raises semantic similarity from $0.788$ to $0.862$ for Qwen and from $0.814$ to $0.874$ for Llama. Yet adversarial evaluation reveals remaining behavioral differences: learned discriminators retain nonzero advantage, and pairwise category analysis shows artifacts concentrated in style/format, robustness, and domain-technical prompts. A pairwise teacher-identification adversary confirms this trend. With a different-family Llama judge and A/B-swap consistency filtering, Qwen distinguishing advantage drops from $0.158$ for the base student to $0.081$ after LoRA distillation. Query-budget experiments show that disagreement-guided acquisition does not consistently outperform stratified random sampling, indicating that coverage and diversity remain strong baselines. Our results show that semantic fidelity is useful but insufficient: black-box LLM distillation requires bounded, adversarial, and category-aware evaluation.
>
---
#### [new 130] DRIFT: Decoupled Rollouts and Importance-Weighted Fine-Tuning for Efficient Multi-Turn Optimization
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出DRIFT框架，解决多轮交互中优化效率与效果的矛盾。通过解耦采样与优化，结合重要性加权微调，提升训练效率并保持性能。属于强化学习与监督微调的结合任务。**

- **链接: [https://arxiv.org/pdf/2605.31455](https://arxiv.org/pdf/2605.31455)**

> **作者:** Jian Mu; Tianyi Lin; Chengwei Qin; Zhongxiang Dai; Yao Shu
>
> **摘要:** Large language models are increasingly deployed in multi-turn interactive settings where users or environments can iteratively provide lightweight feedback. Unfortunately, optimizing such behavior presents a sharp dilemma in practice: online reinforcement learning is able to effectively address multi-turn dynamics but is prohibitively expensive due to the cost of generating full correction trajectories at every update, whereas offline supervised fine-tuning (SFT) is efficient but suffers from distribution shift and behavioral collapse. To this end, we novelly propose DRIFT (Decoupled Rollouts and Importance-Weighted Fine-Tuning), a framework that operationalizes the theoretical insight that the KL-regularized RL objective is equivalent to importance-weighted supervised learning. DRIFT decouples rollout from optimization by sampling offline interaction trajectories from a fixed reference policy, deriving return-based importance weights, and optimizing the policy via weighted SFT on the resulting dataset. Empirically, we demonstrate that DRIFT matches or exceeds the performance of multi-turn reinforcement learning baselines while maintaining the training efficiency and simplicity of standard supervised fine-tuning. Code is available at this https URL.
>
---
#### [new 131] Reading Between the Citations: A Typed Claim Network for Scientific Literature
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文提出“主张网络”，将文献引用转化为带立场标签的主张，解决传统引用表示缺乏评价信息的问题。任务属于科学文献分析，通过构建 typed claim 网络提升检索与分析能力。**

- **链接: [https://arxiv.org/pdf/2605.30966](https://arxiv.org/pdf/2605.30966)**

> **作者:** Ning Ding; Sergio J. Rodríguez Méndez; Pouya G. Omran
>
> **摘要:** Knowledge graphs over corpora of inter-referencing documents - scholarly papers, legal opinions, policy briefs - encode the topology of reference but not its stance. The standard representation collapses a rich evaluative relation into an untyped edge, losing the very content that supports community-level queries about how one document is received by another. We propose the claim network: a representational pattern in which each cross-document reference is reified as a typed claim, carrying source, target, claim text, and a four-class stance label grounded in the citation-intent literature. We give a construction pipeline applicable to any corpus of scholarly inter-referencing documents and instantiate it on a corpus of 127 papers in 3D point cloud semantic segmentation, producing a network of 8,260 typed claims. Three downstream task families demonstrate what the network enables: retrieval signal augmentation, aggregated-stance summarisation, and topological analytics. Head-to-head evaluation against standard Retrieval-Augmented Generation (RAG) baselines shows that the gain over flat retrieval is the gain from the right intermediate representation rather than the wrong one.
>
---
#### [new 132] Crafter: A Multi-Agent Harness for Editable Scientific Figure Generation from Diverse Inputs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于科学图表生成任务，旨在解决自动化生成高质量可编辑图表的问题。提出Crafter和CraftEditor系统，实现多类型图表生成与输出格式转换。**

- **链接: [https://arxiv.org/pdf/2605.30611](https://arxiv.org/pdf/2605.30611)**

> **作者:** Haozhe Zhao; Shuzheng Si; Zhenhailong Wang; Zheng Wang; Liang Chen; Xiaotong Li; Zhixiang Liang; Maosong Sun; Minjia Zhang
>
> **备注:** 24 pages, 11 figures
>
> **摘要:** Scientific figures are among the most effective means of communicating complex research ideas, yet producing publication-quality illustrations remains one of the most labor-intensive parts of paper preparation. Existing automated systems each target a single figure type under text-only input, leaving the diversity of types and conditions researchers actually use unaddressed; their raster outputs further cannot be locally revised. Because scientific figures are structured compositions of discrete semantic components, the localized errors generators produce on such layouts demand not a stronger backbone but a harness. We instantiate this harness in two complementary systems: Crafter, a multi-agent harness for figure generation that generalizes across figure types and input conditions without architectural changes, and CraftEditor, which applies the same pattern to convert raster outputs into editable SVGs. Moreover, we introduce CraftBench, a benchmark spanning three figure types and four input conditions with human quality annotation. Experiments show that Crafter substantially outperforms both standalone generators and the agentic baseline on PaperBanana-Bench and CraftBench, with ablations confirming each component's independent contribution; CraftEditor faithfully converts outputs into editable SVGs that surpass all baselines. Our code and benchmark are available at this https URL.
>
---
#### [new 133] OrcaRouter: A Production-Oriented LLM Router with Hybrid Offline-Online Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型路由任务，解决如何为请求选择最优大语言模型的问题。提出OrcaRouter，结合上下文强化学习与混合离线在线训练，提升路由效果与效率。**

- **链接: [https://arxiv.org/pdf/2605.30736](https://arxiv.org/pdf/2605.30736)**

> **作者:** Zhenghua Bao; Fengya Tian; Chris Zhang; Zhenjun Chen; Xile Ma; Yi Shi
>
> **备注:** 6 pages, 1 table. Technical report
>
> **摘要:** The rapid development of large language models, each with distinct capabilities and inference costs, raises a practical deployment question: given an incoming request, which model should handle it? We present OrcaRouter, a production-oriented LLM router that combines a LinUCB-based contextual bandit over lexical and sentence-embedding features with a hybrid offline-online learning protocol. Offline, OrcaRouter obtains full-information feedback by evaluating each candidate model on a curated set of routing prompts, yielding a reward matrix used to fit one ridge regressor per arm. At deployment time, it initializes from these parameters and can optionally continue learning from bandit feedback, updating only the selected model's arm after observing its reward. At the time of our RouterArena submission (May 20, 2026), OrcaRouter-Adaptive ranked second on the public RouterArena leaderboard with an arena score of 72.08, achieving 75.54% accuracy at a cost of USD 1.00 per 1,000 queries.
>
---
#### [new 134] Knowledge Boundary Probing and Demand-Guided Intervention for LLM-Based Power System Code Generation
- **分类: cs.SE; cs.CL; eess.SY**

- **简介: 该论文属于电力系统代码生成任务，旨在解决LLM在生成代码时因API知识边界错误导致的可靠性问题。通过构建基准测试、知识探测和干预机制提升生成准确性。**

- **链接: [https://arxiv.org/pdf/2605.31478](https://arxiv.org/pdf/2605.31478)**

> **作者:** Hui Wu; Xiaoyang Wang; Zhong Fan
>
> **备注:** 43 pages, 12 figures, includes supplementary material
>
> **摘要:** Large language models (LLMs) are increasingly used to automate power-system analysis, but many utilities and energy-research labs require on-premise serving for confidentiality, regulatory, reproducibility, and cost reasons. This makes the reliability of open-weight models a deployment issue. We show that first-pass failures in power-system code generation are dominated not by reasoning alone, but by structured API-knowledge boundary errors: hallucinated function names, misused parameters, and mishandled result tables in versioned simulation libraries. We introduce PowerCodeBench, an execution-validated benchmark generator that pairs natural-language operator queries with pandapower code and numerical ground truth; an L0-L3 documentation-driven probing procedure that measures per-model API knowledge profiles; and a boundary-aware intervention that combines query-side API demand estimation with targeted proactive documentation injection and routed reactive correction. On a 2,000-task frozen release, we evaluate ten open-weight LLMs (1.5B-480B parameters) and four commercial mid-tier APIs. The intervention improves every evaluated open-weight model of at least 7B parameters and every commercial API by 32 to 56 accuracy points. Open-weight models in the 70B-120B range match the commercial mid-tier accuracy range, while Llama-3.1-405B and Qwen3-Coder-480B lead the panel. The targeted prompts preserve the full-context accuracy ceiling while using 41% of the prompt-token cost. The result is an accuracy-side, deployment-time path toward reliable on-premise LLM assistance for grid-analysis workflows without fine-tuning or cloud inference.
>
---
#### [new 135] Towards Effective Long-Video Event Prediction via Multi-Level Event Semantics Mining
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于长视频事件预测任务，旨在解决现有模型在长视频事件细节提取和逻辑分析上的不足。提出VISTA框架，通过多层级语义挖掘提升预测效果。**

- **链接: [https://arxiv.org/pdf/2605.31069](https://arxiv.org/pdf/2605.31069)**

> **作者:** Bo Peng; YuanJie Lyu; PengGang Qin; Tong Xu
>
> **摘要:** Accurately predicting future events is fundamental to content understanding and decision-making across various domains. While prior research has primarily focused on text or short-video scenarios, long-video event prediction, characterized by vast multimodal context and more complex narratives, remains underexplored. Meanwhile, although recent Long-Video Language Models (LVLMs), built on Large Language Models (LLMs) and Vision-Language Models (VLMs), have shown promise in long-video question answering and summarization, they struggle to generalize to event prediction, as they can neither precisely extract event-related details nor perform fine-grained analysis of event development. To address this gap, we propose VISTA, a multi-level event semantics mining framework for long-video event prediction. Initially, VISTA applies a character-centric visual prompt to precisely extract event-related visual details, enhancing detail-level semantics; subsequently, it employs a knowledge-enhanced iterative retrieval strategy, guiding the LLM to progressively construct logically coherent event chains, thereby improving event-level narratives; ultimately, VISTA adopts a human-like propose-then-retrieve strategy to generate diverse future-oriented proposals and integrate multi-level clues, producing robust and accurate predictions. Extensive experiments on real-world datasets validate the effectiveness of VISTA for long-video event prediction.
>
---
#### [new 136] Traceable by Design: An LLM Pipeline and Dashboard for EU Regulatory Consultation Analysis
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决监管咨询文本分析难题。通过构建LLM管道和仪表板，实现主题提取与证据溯源，提升分析效率与透明度。**

- **链接: [https://arxiv.org/pdf/2605.30995](https://arxiv.org/pdf/2605.30995)**

> **作者:** Thales Bertaglia; Haoyang Gui; Catalina Goanta; Gerasimos Spanakis
>
> **摘要:** Public consultations generate large volumes of data in the form of stakeholder submissions that are practically unfeasible to analyse manually. We present an end-to-end LLM-based pipeline and interactive dashboard for structured topic extraction from regulatory consultation submissions, demonstrated on the European Commission's Digital Fairness Act (DFA) public call for evidence as a case study. The system processes raw PDF attachments and web-form responses, extracts topic annotations, and grounds every extraction in a verbatim quote from the source text. Applied to 4,322 DFA submissions, the pipeline produced 15,368 topic annotations supported by 20,951 verbatim evidence quotes. Three principles govern the proposed design: verbatim grounding, full traceability, and transparency by design. The dashboard exposes the full extraction dataset through five analytical views, from dataset-level topic overviews to individual paragraph drill-downs, with every result traceable to its source. Beyond the predefined DFA topic categories, the pipeline generated certain stakeholder concerns, such as Age Verification, Payment Processor Censorship, and Digital Ownership, that a fixed-taxonomy approach would have missed. The pipeline is domain-generic; adapting it to a new consultation requires only a prompt update and a new dataset. A live demo is available at this https URL. The code and processed data are publicly available at this https URL.
>
---
#### [new 137] Seeing Isn't Knowing: Do VLMs Know When Not to Answer Spatial Questions (and Why)?
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型任务，旨在解决模型在空间推理中无法识别信息不足问题。通过构建评估框架，测试模型在遮挡和视角模糊下的表现，发现其过度自信且难以选择有效视角。**

- **链接: [https://arxiv.org/pdf/2605.30557](https://arxiv.org/pdf/2605.30557)**

> **作者:** Yue Zhang; Zun Wang; Han Lin; Yonatan Bitton; Idan Szpektor; Mohit Bansal
>
> **备注:** Website: this https URL
>
> **摘要:** Spatial reasoning is a fundamental capability for vision-language models (VLMs) deployed in real-world environments. However, visual observations are inherently limited representations of a 3D world: occlusion can render objects invisible, and perspective can make geometric properties misleading. Despite this, existing spatial reasoning benchmarks typically assume that observations are sufficient and reliable, focusing on whether models produce correct answers rather than whether they recognize when a question cannot be answered and what additional observations would be needed. In this work, we challenge this assumption by constructing a controlled evaluation framework, SpatialUncertain, and introducing two types of observation challenges: (1) occlusion, which hides target information, and (2) perspective ambiguity, which produces misleading visual cues. For each configuration, we design spatial questions that are answerable under clean observations but require abstention under the introduced challenges. We further evaluate whether models can identify which additional viewpoints would resolve perspective ambiguity. Our results across a diverse set of frontier open- and closed-source VLMs reveal two consistent failure modes. First, models are prone to overconfident answering, attempting to solve spatial reasoning tasks even when visual evidence is incomplete or misleading, with average accuracy around 30\% under occlusion and below 10\% under perspective ambiguity. Second, even when additional views are available, some models perform near random chance in identifying which would provide reliable evidence. Together, our findings call for moving beyond answer correctness toward evaluating whether models know when to abstain and how to seek reliable evidence.
>
---
#### [new 138] Improving Small Language Models for Code Generation with Reinforcement Learning from Verification Feedback
- **分类: cs.SE; cs.CL**

- **简介: 该论文研究如何通过强化学习提升小语言模型的代码生成能力，解决代码功能性正确性问题。通过设计奖励机制和优化策略，提升模型生成代码的质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.30478](https://arxiv.org/pdf/2605.30478)**

> **作者:** Egor Skopin; Evgeny Kotelnikov
>
> **备注:** Accepted for AINL-2026 conference
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) trains language models using programmatically checkable signals such as unit-test outcomes, enabling direct optimization for functional correctness in code generation. We conduct an empirical study of RLVR for Python code generation on the MBPP benchmark using two small models (Qwen3-0.6B and Llama3.2-1B) with LoRA fine-tuning. Across multiple reward formulations such as: unit-test-only rewards, static-analysis-only shaping via the Ruff linter, and a combined reward, we compare group-based policy optimization variants (GRPO and GSPO) and evaluate both functional correctness and behavioral diagnostics. In our experimental setting, RLVR improves pass@1 on MBPP test by up to 13 percentage points under proposed combined reward configuration. However, we find that reward shaping can induce systematic behavioral shifts: using only static-analysis penalties may bias the policy toward shorter completions that reduce lint errors without reliably improving functional correctness. In contrast, combined rewards mitigate this degeneration and yield more stable trade-offs between correctness and style constraints. Overall, our results highlight that RLVR effectiveness for code generation is highly sensitive to reward design and optimization granularity, and that diagnostics beyond pass@1, including generation length, Ruff severity profiles, and execution error types are useful for identifying failure modes.
>
---
#### [new 139] SpatialAct: Probing Spatial Reasoning-to-Action Capabilities of VLM Agents in 3D Scenes
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型的时空推理任务，旨在解决VLM在3D场景中从空间推理到行动的转化问题。研究构建了SpatialAct基准，通过多轮反馈测试模型的空间理解与行动能力。**

- **链接: [https://arxiv.org/pdf/2605.31148](https://arxiv.org/pdf/2605.31148)**

> **作者:** Tianhui Liu; Jie Feng; Zhiheng Zheng; Shengyuan Wang; Yiming Guo; Yanxin Xi; Hangyu Fan; Yong Li; Pan Hui
>
> **摘要:** Humans can effortlessly perceive spatial layouts, form cognitive representations, reason about spatial relations, and translate such reasoning into actions in everyday 3D environments. Although recent vision-language models (VLMs) have shown promising performance on observation-conditioned spatial perception and reasoning tasks, it remains unclear whether they can build coherent spatial understanding, act upon it, and refine their actions through multi-turn feedback. To study this problem, we introduce \textbf{SpatialAct}, a simulator-grounded benchmark for probing \textit{action-conditioned spatial reasoning} in 3D scenes. Starting from the most challenging setting, Multi-turn Interactive Refinement, we further design its decomposed counterpart, Single-step Error Detection and Fix, together with five fundamental spatial ability tasks to diagnose the underlying causes of model failures. Experiments reveal a clear reasoning-to-action gap: current VLMs can perform well on isolated spatial reasoning tasks, but struggle to maintain coherent spatial beliefs and produce reliable actions during multi-turn feedback, substantially underperforming humans. These results suggest that current VLM agents still lack robust spatial state tracking under action-induced environment changes, even when low-level control is abstracted away.
>
---
#### [new 140] Probing Collision Grounding in Vision-Language Models for Safe Human-Robot Collaboration
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文属于机器人安全监测任务，旨在解决视觉-语言模型在人机协作中的碰撞感知问题。通过构建基准数据集，评估模型对当前及潜在碰撞的识别能力。**

- **链接: [https://arxiv.org/pdf/2605.31196](https://arxiv.org/pdf/2605.31196)**

> **作者:** Jun Wang; Xiaohao Xu; Xiaonan Huang
>
> **备注:** 31 pages, 9 figures
>
> **摘要:** Safe human--robot collaboration requires more than visual description: a monitor must determine whether the robot body is safely separated, already colliding with the scene or a person, or about to collide. We call this capability collision grounding: binding visual observations to robot body geometry, camera viewpoint, scene layout, human proximity, and temporal motion in order to infer present and imminent contact. We introduce TouchSafeBench, a physics-grounded benchmark for evaluating collision grounding in vision-language models (VLMs). Built in Habitat~3.0, TouchSafeBench contains 2,940 simulated indoor co-presence episodes across social navigation and social rearrangement, with synchronized multi-view RGB-D observations, top-down trajectory maps, calibrated camera metadata, and simulator-derived contact labels. We study two deployment-facing tasks: classifying the current safety state and warning about imminent collision before contact. Across three frontier or robotics-oriented VLMs and nine visual representations, current models remain far from reliable: the best average Macro-F1 stays below 50\%, explicit depth is not automatically transformed into robot-body collision evidence, and robot--scene contact is consistently harder than human-contact risk. TouchSafeBench reveals a central limitation of embodied VLMs: visual fluency does not imply physical accountability. Reliable robot safety monitors will need representations that explicitly bind viewpoint, robot morphology, metric geometry, and future collision. We will release the benchmark upon acceptance.
>
---
#### [new 141] ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于环境感知的文本转语音任务，旨在解决语音与环境音频融合困难的问题。通过多模态扩散Transformer和领域表示对齐，提升生成语音的自然度与一致性。**

- **链接: [https://arxiv.org/pdf/2605.30965](https://arxiv.org/pdf/2605.30965)**

> **作者:** Jun-Hak Yun; Seung-Bin Kim; Seong-Whan Lee
>
> **备注:** Accepted to ACL 2026 main conference. Code is available at this https URL
>
> **摘要:** Recent advancements in text-guided audio generation have yielded promising results in diverse domains, including sound effects, speech, and music. However, jointly generating speech with environmental audio remains challenging due to the inherent disparities in their acoustic patterns and temporal dynamics. We propose ImmersiveTTS, an environment-aware text-to-speech (TTS) model that generates natural speech seamlessly integrated within environmental contexts by explicitly modeling cross-modal interactions. Our model builds on a multimodal diffusion transformer and fuses transcript-aligned speech latent with text-conditioned environmental context via joint attention. To enhance semantic consistency, we introduce a domain-specific representation alignment objective tailored to environment-aware TTS, leveraging complementary self-supervised representations from speech and audio encoders. Experimental results show that ImmersiveTTS achieves higher naturalness, intelligibility, and audio fidelity than existing approaches across objective metrics and human listening tests.
>
---
#### [new 142] Counterfactual Evaluation Reveals Hidden Capability Profiles in Clinical LLMs and Agents
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于临床AI评估任务，旨在揭示模型在不同输入下的响应能力。通过引入CSS指标，对比模型在干预性案例中的表现，发现覆盖指标无法检测的缺陷。**

- **链接: [https://arxiv.org/pdf/2605.30590](https://arxiv.org/pdf/2605.30590)**

> **作者:** Matt Turk
>
> **备注:** Accepted to RLEval @ ACM CAIS 2026 (Workshop on Methods and RL Environments for Evaluating AI Agents) and selected for an invited talk based on reviewer ratings. 4-page short paper + appendix
>
> **摘要:** Two clinical AI systems can score nearly identically on coverage-based rubrics yet behave radically differently when their patient inputs change: one updates its recommendations to match the new clinical signal, while the other produces the same output regardless. We introduce the Causal Sensitivity Score (CSS), a pre-registered interventional metric that mutates oncology tumor-board cases along five clinically meaningful dimensions - biomarker flips, prior-treatment failures, biomarker removals, surgery-status changes, and stage perturbations - and scores whether each model updates its recommendations in the pre-registered correct direction using a {0, 0.5, 1.0} scale. Benchmarked against the Consensus Match Score (CMS), a coverage-based weighted recall metric, six frontier models from three labs evaluated in single-shot inference across 224 cases rank in nearly opposite orders: all six models change rank, the CMS-worst model becomes CSS-best, and one upper-mid CMS model ranks last on CSS. We further surface a universal safety blind spot: every frontier model fails on surgery-status interventions (at most 17.2% CSS on Family D), a finding CMS does not expose. The metric also transfers to tool-using agents: in a ReAct-style experiment, tool use improves CSS for five of six models (+2.5 to +20.3 percentage points), yet the lowest-CSS model retrieves the same chart sections and still fails to update its recommendations - revealing a structural responsiveness deficit visible only under counterfactual evaluation. Cross-judge replication and three-rater medical-professional validation confirm the aggregate findings. Interventional pre-registered metrics like CSS complement coverage-based evaluation for clinical AI agents: they capture responsiveness that coverage metrics miss and offer a candidate dense reward signal for future agentic RL systems.
>
---
#### [new 143] Revisiting Padded Transformer Expressivity: Which Architectural Choices Matter and Which Don't
- **分类: cs.LG; cs.AI; cs.CC; cs.CL; cs.FL**

- **简介: 该论文研究Transformer模型的表达能力，探讨不同架构选择对其影响。任务是分析模型在何种条件下等价于特定电路类，解决表达性与架构关系的问题。工作包括理论证明和实验验证。**

- **链接: [https://arxiv.org/pdf/2605.30523](https://arxiv.org/pdf/2605.30523)**

> **作者:** Anej Svete; William Merrill; Ryan Cotterell; Ashish Sabharwal
>
> **摘要:** Recent work describes what transformers can and cannot compute through connections to boolean circuits, but existing results lack exact characterizations and are sensitive to modeling choices. Padded transformers -- to whose input filler symbols such as ``...'' are appended -- emerge as a useful gadget for establishing equivalences to circuit classes by providing polynomial space for adaptive parallel computation. However, only a limited set of padded transformer idealizations has been studied, leaving open how robustly these equivalences hold under changes to attention type, model width, and uniformity. We find that, under practical assumptions, padded transformers are surprisingly robust to all of these, and identify numeric precision and model depth as the main factors affecting expressivity. Concretely, we prove that polynomially padded $\text{L-uniform}$ constant-precision transformers are equivalent to $\text{L-uniform AC}^0$, while growing-precision ones achieve $\text{L-uniform TC}^0$ regardless of width. Furthermore, looping enables sequential processing analogous to circuits: $\log^d N$-looped constant-precision transformers reach $\text{FO-uniform AC}^d$, and growing-precision ones reach $\text{FO-uniform TC}^d$. Interestingly, growing width or precision beyond logarithmic does not increase expressivity, and all our results hold for both softmax and average hard attention transformers.
>
---
#### [new 144] Used Car Salesbots? Honesty and Credulity of LLMs as Bargaining Agents under Partial Information
- **分类: cs.GT; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究LLMs在部分信息下的谈判表现，探讨其诚实性与可信度。任务为评估LLMs作为谈判代理的性能与道德行为，解决其在信息不对称环境中的表现问题。**

- **链接: [https://arxiv.org/pdf/2605.31445](https://arxiv.org/pdf/2605.31445)**

> **作者:** Antonio Valerio Miceli-Barone; Vaishak Belle; Shay B. Cohen
>
> **备注:** 18 pages, 14 figures
>
> **摘要:** In this work we study agents in simulated bargaining scenarios, where a buyer and a seller communicate through a text channel and attempt to negotiate mutually beneficial trades, under different information regimes (complete information, information asymmetry or mutual uncertainty). We evaluate their performance w.r.t. game-theoretical solutions and further investigate their honesty (their tendency to disclose or withhold information or to mislead and deceive) as well as their credulity (their tendency to trust or distrust information provided by the other agent). We study zero-shot LLM agents with simple prompting scaffolding as well as fine-tuned agents, in order to investigate whether optimising the agents to maximise financial profits makes them stronger negotiators but also more dishonest and less trusting. We find that off-the-shelf LLMs all substantially deviate from game-theoretical equilibria, they attempt to lie about their private information but cannot efficiently exploit information asymmetries. Fine-tuning on financial utility makes the agents stronger at achieving better deals but also more dishonest, highlighting the risks that optimising agents for a task can have on their safety. We release our code and a dataset of bargaining scenarios.
>
---
#### [new 145] Measuring, Localizing, and Ablating Alignment Signatures in LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究AI语言模型在对齐过程中产生的风格特征，通过分析和消除这些特征来理解其内部机制。任务属于模型对齐与风格分析，解决如何测量、定位并消除对齐带来的AI风格问题。**

- **链接: [https://arxiv.org/pdf/2605.30526](https://arxiv.org/pdf/2605.30526)**

> **作者:** Aniket Anand; Janvijay Singh; Zhewei Sun; Dilek Hakkani-Tür; Nick Feamster
>
> **摘要:** Aligned language models often exhibit a recognizable AI-like style, yet its connection to post-training and internal representations remains poorly understood. In this work, we study whether post-training introduces or amplifies AI-like stylistic regularities and whether these regularities have a localized internal signature. To this end, we compare human text, base-model generations, and aligned-model generations under matched human-source prefixes. Aligned generations show lower human-corpus affinity and higher AI-detection rates than base generations, suggesting that post-training shifts generated text away from human-corpus style and toward detector-visible AI-like text. We then introduce PASTA (Post-training Alignment Signature Targeted Ablation), a training-free method that estimates a post-training alignment signature from aligned-base residual contrasts and ablates the corresponding direction during decoding. Across 11 aligned models and 6 AI detectors, PASTA lowers the detection rate for most aligned models; this effect transfers well across detectors and is not reproduced by random directions. Qualitative analysis suggests that PASTA generations remain relevant and coherent while exhibiting greater stylistic variation. Together, these results show that AI-like stylistic effects of post-training can be measured, localized, and causally tested through activation ablation.
>
---
#### [new 146] UniScale: Adaptive Unified Inference Scaling via Online Joint Optimization of Model Routing and Test-Time Scaling
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于模型推理优化任务，解决大语言模型推理中质量与成本的平衡问题。通过统一模型路由与测试时缩放，提出UniScale框架实现动态优化。**

- **链接: [https://arxiv.org/pdf/2605.30898](https://arxiv.org/pdf/2605.30898)**

> **作者:** Kaiyu Huang; Xingyu Wang; Mingze Kong; Zhubo Shi; Yuqian Hou; Hong Xu; Zhongxiang Dai; Minchen Yu; Qingjiang Shi
>
> **备注:** Accepted at the 43rd International Conference on Machine Learning (ICML 2026)
>
> **摘要:** In real-world deployments of large language models (LLMs), balancing inference quality and computational cost has become a central challenge. Existing approaches tackle this trade-off along two largely independent dimensions: model routing, which switches among models of different scales to match request complexity, and test-time scaling (TTS), which adjusts inference-time compute within a fixed model for fine-grained control. However, this decoupled design introduces inherent limitations. Model routing yields coarse-grained, discrete performance changes due to the sparse set of model scales, while single-model TTS often encounters capacity ceilings and exhibits diminishing returns as compute increases. Moreover, treating the two mechanisms separately restricts adaptability in dynamic inference environments. To overcome these limitations, we introduce Unified Inference Scaling (UIS), which unifies model routing and TTS in a single optimization space. Building on this formulation, we propose UniScale, an online framework that models adaptive UIS as a contextual multi-armed bandit problem and learns inference policies via LinUCB. The framework incorporates efficiency-aware learning and cost modeling to ensure stable and scalable optimization over high-dimensional action spaces. Evaluation shows that UniScale effectively exploits the synergy in the UIS space to deliver a fine-grained and consistently better quality-cost trade-off across diverse, dynamic inference scenarios.
>
---
#### [new 147] Learning from Fine-Grained Visual Discrepancies: Mitigating Multimodal Hallucinations via In-Context Visual Contrastive Optimization
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型任务，解决多模态幻觉问题。通过引入IC-VCO和VCDist，提升模型对细粒度视觉差异的感知能力。**

- **链接: [https://arxiv.org/pdf/2605.31312](https://arxiv.org/pdf/2605.31312)**

> **作者:** Haolin Deng; Xin Zou; Zhiwei Jin; Chen Chen; Haonan Lu; Xuming Hu
>
> **备注:** ICML 2026
>
> **摘要:** Multimodal hallucination remains a persistent challenge for Vision-Language Models (VLMs). Standard textual Direct Preference Optimization (DPO) often fails to mitigate it due to a lack of explicit visual supervision. While existing works introduce visual preference DPO by contrasting original images against negative ones, they suffer from a theoretically inconsistent objective caused by partition function mismatches and rely on coarse-grained negatives that could enable shortcut learning. In this work, we propose In-Context Visual Contrastive Optimization (IC-VCO). By placing contrastive images within a shared multi-image context, IC-VCO ensures a mathematically rigorous objective. We further introduce Visual Contrast Distillation (VCDist), an auxiliary reliability-gated regularizer that encourages consistency between multi-image contrastive training and single-image inference. Finally, we propose a contrastive sample editing strategy that generates hard negatives via precise semantic perturbations. Experiments on five benchmarks demonstrate IC-VCO's best overall performance and the effectiveness of our sample editing strategy. Code and data are available at this https URL.
>
---
#### [new 148] A Persona-Based Evaluation Framework for Pluralistic Alignment in Generative AI
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于AI评估任务，旨在解决现有评估框架单一化问题，提出基于角色的多视角评估框架，通过模拟多样化的认知角色进行更贴近现实的评价。**

- **链接: [https://arxiv.org/pdf/2605.31021](https://arxiv.org/pdf/2605.31021)**

> **作者:** Atahan Karagoz
>
> **摘要:** Current alignment paradigms for generative artificial intelligence rely predominantly on monolithic benchmarking frameworks that reduce the plurality of human judgment to aggregated statistical baselines, thereby obscuring cultural, demographic, and contextual variability in evaluation. We introduce a state-space constrained emulation framework for AI evaluation that replaces singular assessment functions with a structured manifold of synthetic cognitive profiles representing diverse human perspectives. We show that modern generative architectures can instantiate and maintain these evaluative personas with high consistency, enabling a form of pluralistic, perspective-dependent benchmarking that more closely reflects real-world consensus variability. However, we further analyze the stability of these simulated evaluators under sequential inference and stochastic prompt perturbations, revealing systematic degradation in persona coherence that manifests as state-space drift and semantic inconsistency. These findings suggest that static alignment constraints are insufficient for sustaining robust evaluative behavior over time. Instead, we argue for the necessity of embedding dynamic, viability-driven regulatory mechanisms within generative systems to preserve coherent cognitive emulation. By framing persona-based evaluation as a structured dynamical system over latent representation manifolds, this study provides a foundation for more adaptive, human-aligned, and context-sensitive approaches to AI evaluation.
>
---
#### [new 149] BlueFin: Benchmarking LLM Agents on Financial Spreadsheets
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出BlueFin，一个针对金融表格的LLM代理基准，解决LLM在财务领域任务中的性能评估问题，包含3类任务和详细评价标准。**

- **链接: [https://arxiv.org/pdf/2605.30907](https://arxiv.org/pdf/2605.30907)**

> **作者:** Srivatsa Kundurthy; Clara Na; Colton Moraine; Anoushka Mohta; Case Winter; George Fang; John Ling; Emma Strubell; Zach Kirshner
>
> **备注:** 26 pages
>
> **摘要:** We present BlueFin, a benchmark that tasks large language model (LLM) agents with synthesis, manipulation, and comprehension tasks over spreadsheet workbooks in the professional finance domain. Though estimates of the global population of paying users of spreadsheet software range in the hundreds of millions -- an order of magnitude more than the estimated global population of professional developers -- comparatively fewer resources have been devoted to exploring and expanding LLM capabilities in the spreadsheet domain, with fewer still dedicated to mirroring real occupational tasks encountered by those in professional finance roles. In response, we curate a set of 131 challenging, complex tasks with real-world relevance in the domain, containing 3,225 granular rubric criteria; notably, our rubric criteria and LM judge evaluations are validated by a team of expert human annotators, resulting in high-quality, granular evaluations of complex tasks that are difficult to verify programmatically but can be reliably evaluated by an LM judge agent. Our judge achieves parity with expert consensus ($\alpha=0.826$) with a macro-F1 score of 0.839. Frontier LLMs demonstrate poor performance on the challenging benchmark, with the strongest LLMs achieving less than 50\% average scores across tasks -- models exhibit particular weaknesses in dynamic correctness. Our contributions include a dataset of examples across three categories of spreadsheet tasks, an open source harness and agentic evaluation framework, and a characterization of existing frontier models' performance on our benchmark.
>
---
#### [new 150] Attend to Evidence: Evidence-Anchored Spatial Attention Supervision for Multimodal RLVR
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型任务，解决RLVR中缺乏视觉证据监督的问题。提出EASE方法，通过视觉证据引导注意力，提升模型对图像区域的准确理解。**

- **链接: [https://arxiv.org/pdf/2605.30912](https://arxiv.org/pdf/2605.30912)**

> **作者:** Ruina Hu; Chen Wang; Lai Wei; Jionghao Bai; Bin Yu; Weiran Huang; Kai Wang; Yue Wang
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) improves vision-language models (VLMs) by optimizing outcome rewards derived from final answers. However, such outcome-only rewards do not tell the model which image regions justify an answer. For questions that require visual grounding, these rewards cannot distinguish responses supported by relevant visual evidence from those produced by language-prior shortcuts or lucky guesses. We introduce EASE (Evidence-Anchored Spatial Attention), which augments multimodal RLVR with visual-evidence process supervision. EASE converts annotated evidence regions into a smoothed visual-token target and uses it to guide response-to-image attention during RL training, but only on high-reward trajectories. The annotations are used solely as privileged training labels, while inference requires only the original image and question. Across Qwen2.5-VL-7B, Qwen3-VL-4B, and Qwen3-VL-8B, EASE raises average scores over DAPO by 2.5 to 3.1 points on perception, hallucination, visual math, and multimodal reasoning benchmarks. Diagnostics and ablations show that EASE better aligns visual attention with annotated evidence regions.
>
---
#### [new 151] Trading Complexity for Expressivity Through Structured Generalized Linear Token Mixing
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在优化语言模型中的token混合层。解决如何在计算复杂度与模型表达能力间取得平衡的问题，通过结构化方法设计新型递归模式。**

- **链接: [https://arxiv.org/pdf/2605.31367](https://arxiv.org/pdf/2605.31367)**

> **作者:** Erwan Fagnou; Paul Caillon; Blaise Delattre; Alexandre Allauzen
>
> **备注:** 20 pages, 3 figures, ICML 2026 main
>
> **摘要:** Token mixing layers play a key role in how language models can learn and generate long-range dependencies. Their efficiency relies on the necessary trade-off between decoding speed and the memory requirements, along with the cache size. Considering causal generation, this paper explores new trade-offs thanks to a unified framework which separates two crucial features: (i) the direct influence of inputs on outputs in one generation step; (ii) the recurrent propagation of information through past outputs. This framework encompasses major architectures such as attention and state-space models, but also generalizes the recurrence equations by allowing each state to depend on multiple past states rather than only the immediate predecessor. By introducing structure, we design new recurrence patterns that provably achieve the desired complexity, while providing theoretical insights on their expressivity -- trading runtime for expressivity in a principled way. Empirical validation is performed on synthetic tasks, along with language modeling. Together, these results provide a unified toolkit for the understanding and design of efficient and expressive token mixers across model families.
>
---
#### [new 152] Benchmarking and Enhancing Text-to-Image Models for Generating Visual Representations in Early Arithmetic Education
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文聚焦于文本到图像生成任务，旨在生成符合数学教育概念的视觉内容。解决AI生成内容与教学需求不匹配的问题，构建了基准测试并提出改进策略。**

- **链接: [https://arxiv.org/pdf/2605.31212](https://arxiv.org/pdf/2605.31212)**

> **作者:** Junling Wang; Boqi Chen; Heejin Do; Mubashara Akhtar; April Yi Wang; Mrinmaya Sachan
>
> **摘要:** AI systems are increasingly used to support educational content creation, yet it remains unclear whether they can generate outputs that faithfully represent the pedagogical concepts they are intended to teach. Thus, we introduce equation-to-visual generation, a task that, in contrast to conventional image generation, requires producing pedagogically meaningful visuals from arithmetic equations while precisely preserving their numerical and relational structure. Informed by interviews with teachers and an analysis of educational materials, we construct E2V-Bench, a benchmark spanning four pedagogically grounded visual types, along with automatic metrics for evaluating visual correctness. Our evaluation reveals that recent text-to-image (T2I) models frequently fail on this task, with errors dominated by incorrect object counts and broken relational structure. Building on this, we explore benchmark-guided enhancement strategies. These strategies improve representative models, while the remaining gap calls for stronger numerical and relational grounding in future T2I models.
>
---
#### [new 153] CSULoRA: Closest Safe Update Low-Rank Adaptation
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出CSULoRA，用于安全微调大语言模型。解决对抗数据导致的安全性下降问题，通过估计安全子空间并修正LoRA更新方向。**

- **链接: [https://arxiv.org/pdf/2605.30640](https://arxiv.org/pdf/2605.30640)**

> **作者:** Oleksandr Marchenko Breneur; Adelaide Danilov; Aria Nourbakhsh; Salima Lamsiyah
>
> **备注:** 10 pages, 3 figure
>
> **摘要:** Low-rank adaptation has become a standard method for parameter-efficient fine-tuning of large language models, but even small amounts of unsafe or adversarial fine-tuning data can substantially weaken the safety behavior of aligned models. Existing safety-preserving LoRA methods often rely on hard interventions such as projection, pruning, thresholding, or additional training objectives. While these methods can suppress unsafe update directions, they may also remove task-relevant information or require extra tuning. We introduce CSULoRA, a post-hoc method for correcting trained LoRA adapters through closest safe update estimation. CSULoRA estimates a safety-aligned subspace from the weight displacement between a safety-aligned model and its corresponding base checkpoint. It then decomposes each LoRA update into fully aligned, partially aligned, and off-subspace components. Instead of discarding components outside the estimated safety subspace, CSULoRA solves a closed-form penalized minimum-change problem that preserves the fully aligned component while smoothly attenuating potentially unsafe directions according to their relative energy. In adversarial fine-tuning experiments, CSULoRA substantially reduces attack success rate while preserving most of the utility gains obtained from standard LoRA fine-tuning.
>
---
#### [new 154] A Pilot Study on Curator-Guided Multilingual Art Description for Blind and Low-Vision Audiences with Small Vision-Language Models
- **分类: cs.MM; cs.AI; cs.CL; cs.CV; cs.HC**

- **简介: 该论文属于视觉语言模型任务，旨在解决盲人和低视力用户在多语言艺术描述中的需求。研究使用小规模本地模型，比较不同语言适配器的效果，以提升描述质量与可控性。**

- **链接: [https://arxiv.org/pdf/2605.31080](https://arxiv.org/pdf/2605.31080)**

> **作者:** Iosif Tsangko; Andreas Triantafyllopoulos; George Margetis; Ioana Crihana; Björn W. Schuller
>
> **备注:** 7 pages, 2 figures, 3 tables. Preprint
>
> **摘要:** Blind and low-vision (BLV) audiences remain underserved by visual art descriptions, particularly across languages and in museum settings where privacy and intellectual-property constraints may favour small on-premise vision-language models (VLMs). This pilot study investigates curator-guided multilingual art description with Qwen2.5-VL-3B-Instruct for German, Romanian, and Serbian. We construct a parallel BLV-oriented caption corpus from artwork images and metadata, and compare language-specific LoRA adapters with a single multilingual adapter under a fixed backbone and training budget. Evaluation combines automatic lexical and embedding-based metrics with an LLM-as-Judge protocol calibrated against a small Romanian BLV pilot study. Under our pilot setup, language-specific adapters show more stable controllability and visually grounded description quality for Romanian and Serbian, while multilingual adaptation remains competitive in German. We frame these findings as deployment-oriented evidence for small on-premise VLMs, and highlight the need for larger BLV user studies and broader language coverage before drawing general conclusions about multilingual accessibility.
>
---
#### [new 155] Extracting accent features in spoken Brazilian Portuguese without sociolinguistic labels
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决巴西葡萄牙语区域口音分类中依赖可靠标签的问题。通过使用声学标签和音素对齐工具，提取更有效的方言特征。**

- **链接: [https://arxiv.org/pdf/2605.30457](https://arxiv.org/pdf/2605.30457)**

> **作者:** Pedro H. L. Leite; Pedro Benevenuto Valadares; Luiz W. P. Biscainho
>
> **备注:** This work was submitted to the XLIV Brazilian Symposium on Telecommunications and Signal Processing (SBrT 2026)
>
> **摘要:** Regional accent classification in Brazilian Portuguese (pt-BR) suffers from the need for reliable labeling. While large self-supervised learning (SSL) speech models are powerful, their training pipelines dilute sociophonetic information, since accent labels are generally not reliable or are not used in training objectives. This work introduces a novel workflow for feature extraction using only acoustic labels. By isolating explicit regional accent landmarks and using a phoneme-based forced aligner (ZIPA), our targeted feature set captures dialectal variance more effectively than utterance embeddings, demonstrating that localized features can outperform general-purpose architectures on accent-related tasks using minimal and objective data labels.
>
---
#### [new 156] Vision-Language Models Suppress Female Representations Under Ambiguous Input
- **分类: cs.CV; cs.AI; cs.CL; cs.CY; cs.HC**

- **简介: 该论文研究视觉-语言模型在模糊输入下的性别偏见问题，通过引入LALS度量内部表征与输出的差异，揭示模型内部女性关联被抑制的现象。任务为模型偏见分析。**

- **链接: [https://arxiv.org/pdf/2605.31556](https://arxiv.org/pdf/2605.31556)**

> **作者:** Arnau Marin-Llobet; Simon Henniger; Mahzarin R. Banaji
>
> **备注:** 16 pages, 12 figures, 1 table
>
> **摘要:** Alignment teaches vision-language models (VLMs) to avoid expressing demographic biases, and when gender is clearly visible they largely succeed. Far less is known about ambiguous inputs (a worker in full gear, a figure seen from behind) cases common in practice yet rarely studied. We find that minimal prompting pressure exposes occupation-gender defaults when prompting ambiguous input images, with models collapsing to male even for strongly female-stereotyped occupations. But do these outputs reflect what models actually encode internally? We introduce LALS (Latent Association Leaning Score), a zero-shot metric that projects visual-token activations into the model's text-embedding space to measure concept associations per token and layer. Across 15 occupations, over 800 gender-ambiguous images, and four VLMs, internal representations and outputs are systematically decoupled: models often encode a female association internally yet output male. Layer-wise analysis reveals an asymmetric filter -- male signal amplifies end-to-end while female signal peaks mid-network and is suppressed before generation -- and a color ablation shows that culturally loaded visual cues such as clothing color further modulate these internal associations.
>
---
## 更新

#### [replaced 001] Self-Reflective Generation at Test Time
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，旨在解决LLM在生成过程中因早期错误导致的连锁失误问题。提出SRGen框架，在生成时进行自我反思以提升可靠性。**

- **链接: [https://arxiv.org/pdf/2510.02919](https://arxiv.org/pdf/2510.02919)**

> **作者:** Jian Mu; Qixin Zhang; Zhiyong Wang; Menglin Yang; Shuang Qiu; Chengwei Qin; Zhongxiang Dai; Yao Shu
>
> **摘要:** Large language models (LLMs) increasingly solve complex reasoning tasks via long chain-of-thought, but their forward-only autoregressive generation process is fragile; early token errors can cascade, which creates a clear need for self-reflection mechanisms. However, existing self-reflection either performs revisions over full drafts or learns self-correction via expensive training, both fundamentally reactive and inefficient. To address this, we propose Self-Reflective Generation at Test Time (SRGen), a lightweight test-time framework that reflects before generating at uncertain points. During token generation, SRGen utilizes dynamic entropy thresholding to identify high-uncertainty tokens. For each identified token, it trains a specific corrective vector, which fully exploits the already generated context for a self-reflective generation to correct the token probability distribution. By retrospectively analyzing the partial output, this self-reflection enables more trustworthy decisions, thereby significantly reducing the probability of errors at highly uncertain points. Evaluated on challenging mathematical reasoning benchmarks and a diverse set of LLMs, SRGen can significantly strengthen model reasoning. Moreover, our findings position SRGen as a plug-and-play method that integrates reflection into the generation process for reliable LLM reasoning, achieving consistent gains with bounded overhead and can be combined with other training-time (e.g., RLHF) and test-time (e.g., SLOT) techniques.
>
---
#### [replaced 002] TaxoBell: Gaussian Box Embeddings for Self-Supervised Taxonomy Expansion
- **分类: cs.CL**

- **简介: 该论文提出TaxoBell，解决taxonomy expansion任务中的对称性不足和不确定性建模问题，通过高斯盒嵌入实现更准确的语义表示与推理。**

- **链接: [https://arxiv.org/pdf/2601.09633](https://arxiv.org/pdf/2601.09633)**

> **作者:** Sahil Mishra; Srinitish Srinivasan; Srikanta Bedathur; Tanmoy Chakraborty
>
> **备注:** Accepted in The Web Conference (WWW) 2026
>
> **摘要:** Taxonomies form the backbone of structured knowledge representation across diverse domains, enabling applications such as e-commerce and semantic search. Yet, manual taxonomy expansion is labor-intensive and slow. Existing methods rely on point-based vector embeddings, which model symmetric similarity and thus struggle with the asymmetric relationships that are fundamental to taxonomies. Box embeddings offer a promising alternative by enabling containment and disjointness, but they face key issues: (i) unstable gradients at the intersection boundaries, (ii) no notion of semantic uncertainty, and (iii) limited capacity to represent polysemy or ambiguity. We address these shortcomings with TaxoBell, a Gaussian box embedding framework that translates between box geometries and multivariate Gaussian distributions, where means encode semantic location and covariances encode uncertainty. Energy-based optimization yields stable optimization, robust modeling of ambiguous concepts, and interpretable hierarchical reasoning. Extensive experiments on five benchmark datasets demonstrate that TaxoBell significantly outperforms eight state-of-the-art taxonomy expansion baselines by 19% in MRR and around 25% in Recall@k. We further demonstrate the advantages and pitfalls of TaxoBell with error analysis and ablation studies.
>
---
#### [replaced 003] 3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言任务，解决模型在空间推理上的不足。通过引入3ViewSense框架，利用正交视图提升模型的空间感知与推理能力。**

- **链接: [https://arxiv.org/pdf/2603.07751](https://arxiv.org/pdf/2603.07751)**

> **作者:** Shaoxiong Zhan; Yanlin Lai; Zheng Liu; Hai Lin; Shen Li; Xiaodong Cai; Zijian Lin; Wen Huang; Hai-Tao Zheng
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Current Large Language Models have achieved Olympiad-level logic, yet Vision-Language Models paradoxically falter on elementary spatial tasks like block counting. This capability mismatch reveals a critical ``spatial intelligence gap,'' where models fail to construct coherent 3D mental representations from 2D observations. We uncover this gap via diagnostic analyses showing the bottleneck is a missing view-consistent spatial interface rather than insufficient visual features or weak reasoning. To bridge this, we introduce \textbf{3ViewSense}, a framework that grounds spatial reasoning in Orthographic Views. Drawing on engineering cognition, we propose a ``Simulate-and-Reason'' mechanism that decomposes complex scenes into canonical orthographic projections to resolve geometric ambiguities. By aligning egocentric perceptions with these allocentric references, our method facilitates explicit mental rotation and reconstruction. Empirical results on spatial reasoning benchmarks demonstrate that our method significantly outperforms existing baselines, with consistent gains on occlusion-heavy counting and view-consistent spatial reasoning. The framework also improves the stability and consistency of spatial descriptions, offering a scalable path toward stronger spatial intelligence in multimodal systems.~\footnote{this https URL}
>
---
#### [replaced 004] *-PLUIE: Personalisable metric with Llm Used for Improved Evaluation
- **分类: cs.CL**

- **简介: 该论文属于文本质量评估任务，旨在解决LLM-judge方法计算成本高、需后处理的问题。提出*-PLUIE，通过任务特定提示提升与人类判断的一致性，同时保持低计算成本。**

- **链接: [https://arxiv.org/pdf/2602.15778](https://arxiv.org/pdf/2602.15778)**

> **作者:** Quentin Lemesle; Léane Jourdan; Daisy Munson; Pierre Alain; Jonathan Chevelu; Arnaud Delhay; Damien Lolive
>
> **备注:** Accepted at *SEM 2026
>
> **摘要:** Evaluating the quality of automatically generated text often relies on LLM-as-a-judge (LLM-judge) methods. While effective, these approaches are computationally expensive and require post-processing. To address these limitations, we build upon ParaPLUIE, a perplexity-based LLM-judge metric that estimates confidence over ``Yes/No'' answers without generating text. We introduce *-PLUIE, task specific prompting variants of ParaPLUIE and evaluate their alignment with human judgement. Our experiments show that personalised *-PLUIE achieves stronger correlations with human ratings while maintaining low computational cost.
>
---
#### [replaced 005] Pull Requests as a Training Signal for Repo-Level Code Editing
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于代码编辑任务，旨在提升模型在大型代码库中进行多文件修改的能力。通过利用GitHub Pull Request作为训练信号，构建了大规模语料库，并在SWE-bench上取得显著提升。**

- **链接: [https://arxiv.org/pdf/2602.07457](https://arxiv.org/pdf/2602.07457)**

> **作者:** Qinglin Zhu; Tianyu Chen; Shuai Lu; Lei Ji; Runcong Zhao; Murong Ma; Xiangxiang Dai; Yulan He; Lin Gui; Peng cheng; Yeyun Gong
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Repository-level code editing requires models to understand complex dependencies and execute precise multi-file modifications across a large codebase. While recent gains on SWE-bench rely heavily on complex agent scaffolding, it remains unclear how much of this capability can be internalised via high-quality training signals. To address this, we propose Clean Pull Request (Clean-PR), a mid-training paradigm that leverages real-world GitHub pull requests as a training signal for repository-level editing. We introduce a scalable pipeline that converts noisy pull request diffs into Search/Replace edit blocks through reconstruction and validation, resulting in the largest publicly available corpus of 2 million pull requests spanning 12 programming languages. Using this training signal, we perform a mid-training stage followed by an agentless-aligned supervised fine-tuning process with error-driven data augmentation. On SWE-bench, our model significantly outperforms the instruction-tuned baseline, achieving absolute improvements of 13.6% on SWE-bench Lite and 12.3% on SWE-bench Verified. These results demonstrate that repository-level code understanding and editing capabilities can be effectively internalised into model weights under a simplified, agentless protocol, without relying on heavy inference-time scaffolding.
>
---
#### [replaced 006] Prompt Injection as Role Confusion
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文研究LLM中的角色混淆问题，揭示提示注入机制。通过设计角色探测器，发现模型依据文本风格而非标签判断角色，导致攻击成功。任务属于安全与可信AI领域。**

- **链接: [https://arxiv.org/pdf/2603.12277](https://arxiv.org/pdf/2603.12277)**

> **作者:** Charles Ye; Jasmine Cui; Dylan Hadfield-Menell
>
> **备注:** ICML 2026
>
> **摘要:** LLMs see the world as a single stream of text, partitioned into roles like <user> or <tool>. We trace prompt injection to role confusion: models perceive the source of text from how it sounds, not its labeled role. A command hidden in a webpage hijacks an agent simply because it sounds like <user> text, despite its <tool> label. We design role probes to measure how LLMs internally perceive "who is speaking," and find that injected text occupies the same representational space as the trusted role it imitates. We demonstrate this with CoT Forgery, a zero-shot attack that injects fabricated reasoning into user prompts and tool outputs. Models mistake the forgery for their own thoughts, yielding 60% attack success against frontier models with near-zero baselines. Strikingly, the degree of role confusion predicts attack success before a single token is generated. This mechanism generalizes beyond CoT Forgery to standard agent prompt injections, revealing prompt injection as a measurable consequence of role perception. To the model, sounding like a role is indistinguishable from being one.
>
---
#### [replaced 007] IAPO: Information-Aware Policy Optimization for Token-Efficient Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的推理优化任务，旨在解决长推理链带来的高成本问题。通过信息理论方法提升推理效率，减少冗余步骤，同时保持准确性。**

- **链接: [https://arxiv.org/pdf/2602.19049](https://arxiv.org/pdf/2602.19049)**

> **作者:** Yinhan He; Yaochen Zhu; Mingjia Shi; Wendy Zheng; Lin Su; Xiaoqing Wang; Qi Guo; Jundong Li
>
> **摘要:** Large language models increasingly rely on long chains of thought to improve accuracy, yet such gains come with substantial inference-time costs. We revisit token-efficient post-training and argue that existing sequence-level reward-shaping methods offer limited control over how reasoning effort is allocated across tokens. To bridge the gap, we propose IAPO, an information-theoretic post-training framework that assigns token-wise advantages based on each token's conditional mutual information (MI) with the final answer. This yields an explicit, principled mechanism for identifying informative reasoning steps and suppressing low-utility exploration. We provide a theoretical analysis showing that our IAPO can induce monotonic reductions in reasoning verbosity without harming correctness. Empirically, IAPO consistently improves reasoning accuracy while reducing reasoning length by up to 36%, outperforming existing token-efficient RL methods across various reasoning datasets. Extensive empirical evaluations demonstrate that information-aware advantage shaping is a powerful and general direction for token-efficient post-training. The code is available at this https URL.
>
---
#### [replaced 008] Compute Allocation in Evolutionary Search: From Depth-Breadth to Multi-Armed Bandits
- **分类: cs.CL; cs.AI; cs.LG; cs.NE**

- **简介: 该论文研究LLM引导的进化搜索中的计算资源分配问题，旨在提高搜索效率与可靠性。通过分析不同模型和任务下的性能，提出BaSE算法优化LLM调用分配。**

- **链接: [https://arxiv.org/pdf/2605.29268](https://arxiv.org/pdf/2605.29268)**

> **作者:** Sixue Xing; Haoyu He; Kerui Wu; Zhuo Yang; Haozheng Luo; Tianfan Fu; Aarthy Nagarajan
>
> **摘要:** LLM-guided evolutionary search (Evolve systems) has reached state-of-the-art results on mathematical and combinatorial tasks, yet most existing systems report only the best of many runs and leave the run-to-run distribution undocumented. We ask how a fixed budget of LLM calls should be allocated, and how reliably a single run reaches the reported numbers. Sweeping the depth-breadth grid over five models and three tasks, we identify two empirical regularities: a fitness-compute envelope along which capability ordering largely collapses on effective FLOPs, and a bilinear depth-breadth fit with task-specific interaction; both are gated by model-task capability. Motivated by these regularities, we propose BaSE (Bandit-based Self-Evolving), a multi-armed bandit that allocates LLM calls across parallel trajectories. Without changing the model, prompt, or evaluator, BaSE improves mean fitness by 12.3% over the strongest island-protocol baseline across 8 (model, task) cells, with the largest gains on high-variance settings: a reliability gain from allocation alone.
>
---
#### [replaced 009] Are we chasing ghosts? Quantifying unattributable polarization, and attributing the rest to annotator groups
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的情感分析任务，旨在解决标注者群体间意见差异的量化问题。研究提出新指标和工具，以准确归因极化现象，发现性别和种族是主要影响因素。**

- **链接: [https://arxiv.org/pdf/2602.06055](https://arxiv.org/pdf/2602.06055)**

> **作者:** Dimitris Tsirmpas; John Pavlopoulos
>
> **备注:** 19 pages, 7 tables, 9 figures
>
> **摘要:** Standard agreement metrics often fail to capture systematic differences in opinion between minority and majority-group annotators, jeopardizing tasks such as hate speech and toxicity detection. Polarization has recently been proposed as a more robust way of distinguishing minor disagreements from systematic differences in opinion, but existing approaches do not provide practical tools for attributing it to specific annotator groups. We evaluate current methods and identify two major limitations in realistic settings: (1) the presence of ``inherent'' polarization that cannot be attributed to any known or latent groups, and (2) opposing polarization effects canceling each other out in aggregated annotations. To address these issues, we introduce a new metric that measures and tests the statistical significance of polarization attribution for annotator groups while avoiding these limitations, as well as an open-source Python library implementation, finding that no more than 20 annotators are needed per comment for reliable estimation. We apply our method to four subjective NLP datasets and find that gender and race consistently explain polarization patterns, while differences between annotator groups become stronger as the groups are further apart.
>
---
#### [replaced 010] Context-Free Recognition with Transformers
- **分类: cs.LG; cs.CC; cs.CL; cs.FL**

- **简介: 该论文研究Transformer在上下文无关语言（CFL）识别任务中的能力，解决其无法有效处理语法结构的问题，提出通过循环层和填充符号实现CFL识别。**

- **链接: [https://arxiv.org/pdf/2601.01754](https://arxiv.org/pdf/2601.01754)**

> **作者:** Selim Jerad; Anej Svete; Sophie Hao; Ryan Cotterell; William Merrill
>
> **摘要:** Transformers excel empirically on tasks that process well-formed inputs according to some grammar, such as natural language and code. However, it remains unclear how they can process grammatical syntax. In fact, under standard complexity conjectures, standard transformers cannot recognize context-free languages (CFLs), a canonical formalism to describe syntax, or even regular languages, a subclass of CFLs. Past work has shown that $\mathcal{O}(\log(N))$ looping layers (w.r.t. input length $N$) allow transformers to recognize regular languages, but the question of context-free recognition with looped transformers remained open. In this work, we show that looped transformers with $\mathcal{O}(\log(N))$ looping layers and $\mathcal{O}(N^6)$ padding symbols can recognize all CFLs. However, training and inference with $\mathcal{O}(N^6)$ padding symbols is potentially impractical. Fortunately, we show that, for natural subclasses such as unambiguous CFLs, the recognition problem on transformers becomes more tractable, requiring $\mathcal{O}(N^3)$ padding. Empirically, looped and padded transformers perform better than fixed-depth transformers in recognizing CFLs. Overall, our results shed light on the intricacy of CFL recognition by transformers: while general recognition may require an intractable amount of padding, natural constraints such as unambiguity yield efficient recognition algorithms.
>
---
#### [replaced 011] LLMs Lean on Priors, Not Programming Language Semantics
- **分类: cs.PL; cs.AI; cs.CL; cs.SE**

- **简介: 该论文属于自然语言处理任务，探讨LLMs是否依赖预训练的统计规律而非形式语义进行推理。通过构建PLSemanticsBench测试集，验证LLMs在不同语义下的推理能力，发现其主要依赖词汇关联而非形式规则。**

- **链接: [https://arxiv.org/pdf/2510.03415](https://arxiv.org/pdf/2510.03415)**

> **作者:** Aditya Thimmaiah; Jiyang Zhang; Jayanth Srinivasa; Junyi Jessy Li; Milos Gligoric
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Recent work asks whether large language models (LLMs) condition their reasoning on explicit rules rather than statistical regularities from pretraining. Program execution provides a canonical instance: formal semantics define behavior through symbolic transition rules that can be systematically altered under distribution shift. We investigate whether LLMs can condition their reasoning on formal semantics through program execution and introduce PLSemanticsBench, pairing featherweight C programs with two semantic systems -- small-step operational semantics and K semantics -- and probing four capabilities: composing rules for final states, selecting rules when state is unmutated, sustaining such conditioning over long traces, and following supplied rules under novel semantics. To decouple semantic reasoning from syntactic familiarity, we redefine familiar operators to induce symbol-meaning conflict and introduce novel symbols defined only through the supplied rules, and stress-test models on Human-Written, LLM-Translated, and Fuzzer-Generated splits with increasing structural complexity. Across 11 frontier LLMs, strong final-state accuracy under standard semantics (up to 90%) drops sharply -- by as much as 40--60% points -- under semantic mutations and increasing structural complexity. Only a handful of models achieve non-zero long-horizon conditioning accuracy, and even the best systems reach just 35%. Together, these results suggest that contemporary LLMs often rely on pretrained lexical associations rather than systematically conditioning on supplied formal rules. PLSemanticsBench is publicly available at this https URL.
>
---
#### [replaced 012] On the "Induction Bias" in Sequence Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究Transformer在状态跟踪上的局限性，对比其与RNN的数据效率。任务为序列建模，解决模型在分布内泛化能力不足的问题。通过实验发现Transformer依赖长度特定解，缺乏有效权重共享。**

- **链接: [https://arxiv.org/pdf/2602.18333](https://arxiv.org/pdf/2602.18333)**

> **作者:** M.Reza Ebrahimi; Michaël Defferrard; Sunny Panchal; Roland Memisevic
>
> **备注:** Accepted to the International Conference on Machine Learning (ICML) 2026
>
> **摘要:** Despite the remarkable practical success of transformer-based language models, recent work has raised concerns about their ability to perform state tracking. In particular, a growing body of literature has shown this limitation primarily through failures in out-of-distribution (OOD) generalization, such as length extrapolation. In this work, we shift attention to the in-distribution implications of these limitations. We conduct a large-scale experimental study of the data efficiency of transformers and recurrent neural networks (RNNs) across multiple supervision regimes. We find that the amount of training data required by transformers grows much more rapidly with state-space size and sequence length than for RNNs. Furthermore, we analyze the extent to which learned state-tracking mechanisms are shared across different sequence lengths. We show that transformers exhibit negligible or even detrimental weight sharing across lengths, indicating that they learn length-specific solutions in isolation. In contrast, recurrent models exhibit effective amortized learning by sharing weights across lengths, allowing data from one sequence length to improve performance on others. Together, these results demonstrate that state tracking remains a fundamental challenge for transformers, even when training and evaluation distributions match.
>
---
#### [replaced 013] A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于人工智能领域，旨在解决如何评估语言模型代理的目标导向性问题。通过行为分析和内部表征解读，研究代理在网格世界中的目标达成能力及认知过程。**

- **链接: [https://arxiv.org/pdf/2602.08964](https://arxiv.org/pdf/2602.08964)**

> **作者:** Raghu Arghal; Fade Chen; Niall Dalton; Evgenii Kortukov; Calum McNamara; Angelos Nalmpantis; Moksh Nirvaan; Gabriele Sarti; Mario Giulianelli
>
> **备注:** Proceedings of the 43rd International Conference on Machine Learning (ICML 2026)
>
> **摘要:** Understanding an agent's goals helps explain and predict its behaviour, yet there is no established methodology for reliably attributing goals to agentic systems. We propose a framework for evaluating goal-directedness that integrates behavioural evaluation with interpretability-based analyses of models' internal representations. As a case study, we examine an LLM agent navigating a 2D grid world towards a goal state. Behaviourally, we evaluate the agent against optimal policies across varying grid sizes, obstacle densities, and goal structures, finding that performance scales with task difficulty while remaining robust to difficulty-preserving transformations and multi-goal structures. We then use probing methods to decode internal representations of the environment and multi-step action plans. We find that the LLM agent non-linearly encodes a coarse spatial map, preserving approximate task-relevant cues about its position and the goal location; that its actions are broadly consistent with these internal representations; and that reasoning reorganises them, shifting from spatial cues towards immediate action selection. Our findings support the view that introspective examination is required beyond behavioural evaluations to characterise how agents represent and pursue their objectives.
>
---
#### [replaced 014] UniDial-EvalKit: A Unified Toolkit for Evaluating Multi-Faceted Conversational Abilities
- **分类: cs.CL**

- **简介: 该论文属于对话系统评估任务，旨在解决多轮交互中模型评价标准不统一的问题。提出UniDial-EvalKit工具，统一数据格式、评估流程和指标计算，提升模型比较效率。**

- **链接: [https://arxiv.org/pdf/2603.23160](https://arxiv.org/pdf/2603.23160)**

> **作者:** Qi Jia; Haodong Zhao; Dun Pei; Xiujie Song; Ye Shen; Shibo Wang; Zijian Chen; Zicheng Zhang; Xiangyang Zhu; Guangtao Zhai
>
> **摘要:** Benchmarking large language models (LLMs) and agents in multi-turn interactive scenarios is essential for understanding their practical capabilities. However, existing evaluation protocols are highly heterogeneous, differing significantly in dataset formats, model interfaces, and evaluation pipelines, which severely impedes systematic comparison. In this work, we present UniDial-EvalKit (UDE), a unified evaluation toolkit for assessing interactive AI systems. The core contribution of UDE lies in its holistic unification: it standardizes heterogeneous data formats into a universal schema, streamlines complex evaluation pipelines through a modular architecture, and aligns metric calculations under a hierarchical scoring aggregation. It also supports efficient large-scale evaluation through parallel generation and scoring, as well as checkpoint resume to eliminate redundant computation. Leveraging UDE, we conduct an extensive evaluation across diverse multi-dimensional benchmarks. Our empirical analysis shows that no single system consistently outperforms others across all benchmarks, while current memory agents often fail to surpass full-context baselines. Further analyses highlight several future directions, including benchmark deduplication and more adaptive memory architectures.
>
---
#### [replaced 015] Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型预训练任务，解决数据混合比例优化问题。通过模型融合实现数据比例预测，降低搜索成本，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.00747](https://arxiv.org/pdf/2602.00747)**

> **作者:** Shengrui Li; Fei Zhao; Kaiyan Zhao; Jieying Ye; Haifeng Liu; Fangcheng Shi; Zheyong Xie; Yao Hu; Shaosheng Cao
>
> **备注:** 18 pages, 5 figures, accepted at ICML 2026
>
> **摘要:** Determining an effective data mixture is a key factor in Large Language Model (LLM) pre-training, where models must balance general competence with proficiency on hard tasks such as math and code. However, identifying an optimal mixture remains an open challenge, as existing approaches either rely on unreliable tiny-scale proxy experiments or require prohibitively expensive large-scale exploration. To address this, we propose Decouple Searching from Training Mix (DeMix), a novel framework that leverages model merging to predict optimal data ratios. Instead of training proxy models for every sampled mixture, DeMix trains component models on candidate datasets at scale and derives data mixture proxies via weighted model merging. This paradigm decouples search from training costs, enabling evaluation of unlimited sampled mixtures without extra training burden and thus facilitating better mixture discovery through more search trials. Extensive experiments demonstrate that DeMix breaks the trade-off between sufficiency, accuracy and efficiency, obtaining the optimal mixture with higher benchmark performance at lower search cost. Additionally, we release the DeMix Corpora, a comprehensive 22T-token dataset comprising high-quality pre-training data with validated mixtures to facilitate open research. Our code and DeMix Corpora is available at this https URL.
>
---
#### [replaced 016] LaCy: What Small Language Models Can and Should Learn is Not Just a Question of Loss
- **分类: cs.CL**

- **简介: 该论文属于语言模型预训练任务，旨在解决小模型因参数限制导致生成事实错误的问题。通过结合损失与事实信号，提出LaCy方法，优化模型学习与调用大模型的决策。**

- **链接: [https://arxiv.org/pdf/2602.12005](https://arxiv.org/pdf/2602.12005)**

> **作者:** Szilvia Ujváry; Louis Béthune; Pierre Ablin; João Monteiro; Marco Cuturi; Michael Kirchhof
>
> **备注:** 40 pages, 26 figures, 10 tables, preprint. v3-v4: new results for RAG, ablations and additional analysis
>
> **摘要:** Language models have consistently grown to compress more world knowledge into their parameters, but the knowledge that can be pretrained into them is upper-bounded by their parameter size. Especially the capacity of Small Language Models (SLMs) is limited, leading to factually incorrect generations. This problem is often mitigated by giving the SLM access to an outside source: the ability to query a larger model, documents, or a database. Under this setting, we study the fundamental question of \emph{which tokens an SLM can and should learn} during pretraining, versus \emph{which ones it should delegate} via a \texttt{<CALL>} token. We find that this is not simply a question of loss: although the loss is predictive of whether a predicted token mismatches the ground-truth, it is insufficient for identifying which predictions would actually lead to factual or semantically invalid continuations. Some high-loss tokens correspond to \emph{acceptable} alternative continuations of a pretraining document and therefore should not trigger a \texttt{<CALL>}. This suggests that learnability cannot be characterized from loss alone, but requires additional domain-specific signals about the role of a token in the sentence. In Wikipedia-like domains, we show that augmenting the loss signal with lightweight grammatical information from a spaCy parser substantially improves delegation decisions. Based on this insight, we propose LaCy, a novel pretraining method that combines loss with factuality signals to decide which tokens an SLM should learn. Our experiments demonstrate that LaCy models successfully learn which tokens to predict and when to call for help. This results in higher FactScores when generating in a cascade with a bigger model and outperforms Rho or LLM-judge trained SLMs, while being simpler and cheaper.
>
---
#### [replaced 017] Less is Enough: Synthesizing Diverse Data in LLM Feature Space with Sparse Autoencoders
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在提升大语言模型的下游性能。通过引入FAC度量特征空间多样性，提出FAC Synthesis框架，生成更具代表性的数据。**

- **链接: [https://arxiv.org/pdf/2602.10388](https://arxiv.org/pdf/2602.10388)**

> **作者:** Zhongzhi Li; Xuansheng Wu; Yijiang Li; Lijie Hu; Ninghao Liu
>
> **摘要:** The diversity of post-training data is critical for effective downstream performance in large language models (LLMs). Many existing approaches to constructing post-training data quantify diversity using text-based metrics that capture linguistic variation, but such metrics provide only weak signals for the task-relevant features that determine downstream performance. In this work, we introduce Feature Activation Coverage (FAC) which measures data diversity in an interpretable feature space. Building upon this metric, we further propose a diversity-driven data synthesis framework, named FAC Synthesis, that first uses a sparse autoencoder to identify missing features from a seed dataset, and then generates synthetic samples that explicitly reflect these features. Experiments show that our approach consistently improves both data diversity and downstream performance on various tasks, including instruction following, toxicity detection, reward modeling, and behavior steering. Interestingly, we identify a shared, interpretable feature space across model families (i.e., LLaMA, Mistral, and Qwen), enabling cross-model knowledge transfer. Our work provides a solid and practical methodology for exploring data-centric optimization of LLMs.
>
---
#### [replaced 018] MedFact: Benchmarking the Fact-Checking Capabilities of Large Language Models on Chinese Medical Texts
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗文本事实核查任务，旨在解决医学大模型的事实准确性问题。构建了MedFact基准数据集，并评估了20个大模型的核查能力。**

- **链接: [https://arxiv.org/pdf/2509.12440](https://arxiv.org/pdf/2509.12440)**

> **作者:** Jiayi He; Yangmin Huang; Qianyun Du; Xiangying Zhou; Zhiyang He; Jiaxue Hu; Xiaodong Tao; Lixian Lai
>
> **备注:** Accepted to The Fifth Workshop on Generation, Evaluation, and Metrics (GEM) at ACL 2026
>
> **摘要:** Deploying Large Language Models (LLMs) in medical applications requires fact-checking capabilities to ensure patient safety and regulatory compliance. We introduce MedFact, a challenging Chinese medical fact-checking benchmark with 2,116 expert-annotated instances from diverse real-world texts, spanning 13 specialties, 8 error types, 4 writing styles, and 5 difficulty levels. Construction uses a hybrid AI-human framework where iterative expert feedback refines AI-driven, multi-criteria filtering to ensure high quality and difficulty. We evaluate 20 leading LLMs on veracity classification and error localization, and results show models often determine if text contains errors but struggle to localize them precisely, with top performers falling short of human performance. Our analysis reveals the "over-criticism" phenomenon, a tendency for models to misidentify correct information as erroneous, which can be exacerbated by advanced reasoning techniques such as multi-agent collaboration and inference-time scaling. MedFact highlights the challenges of deploying medical LLMs and provides resources to develop factually reliable medical AI systems.
>
---
#### [replaced 019] InfiMed-ORBIT: Aligning LLMs on Open-Ended Complex Tasks via Rubric-Based Incremental Training
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ORBIT框架，解决开放性医疗对话中奖励信号模糊的问题。通过基于评分标准的增量训练，提升模型表现。任务属于医疗对话生成。**

- **链接: [https://arxiv.org/pdf/2510.15859](https://arxiv.org/pdf/2510.15859)**

> **作者:** Pengkai Wang; Pengwei Liu; Qi Zuo; Zhijie Sang; Congkai Xie; Hongxia Yang
>
> **摘要:** Reinforcement learning (RL) has powered many recent breakthroughs in large language models (LLMs), especially for tasks where rewards can be computed automatically, such as code generation. However, it is less effective in open-ended medical dialogue, where feedback is ambiguous, context-dependent, and difficult to simply summarize into a single scalar signal-often requiring heavily supervised reward models and creating risks of reward hacking. Thus, we introduce ORBIT, an open-ended rubric-based incremental training framework tailored for critical medical dialogues. ORBIT integrates medical dialogue construction with dynamically generated case-conditioned rubrics that serve as adaptive guides for incremental RL. Unlike approaches that rely on external medical knowledge bases or handcrafted rules, ORBIT uses rubric-guided evaluation and can be implemented with general-purpose instruction-following LLMs, avoiding task-specific judge fine-tuning. With only 2k training samples, ORBIT raises Qwen3-4B-Instruct's HealthBench-Hard score from 7.0 to 27.5, achieving state-of-the-art performance among similarly sized open-source models while maintaining strong consultation quality as rubric coverage broadens.
>
---
#### [replaced 020] Mechanistic Interpretability as Statistical Estimation: A Variance Analysis
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于机制可解释性研究，旨在解决模型解释的稳定性问题。通过方差分析揭示电路发现中的统计不稳定性，提出更稳健的MI方法。**

- **链接: [https://arxiv.org/pdf/2510.00845](https://arxiv.org/pdf/2510.00845)**

> **作者:** Maxime Méloux; François Portet; Maxime Peyrard
>
> **摘要:** Mechanistic Interpretability (MI) aims to reverse-engineer model behaviors by identifying functional sub-networks. Yet, the scientific validity of these findings depends on their stability. In this work, we argue that circuit discovery is not a standalone task but a statistical estimation problem built upon causal mediation analysis (CMA). We uncover a fundamental instability at this base layer: exact, single-input CMA scores exhibit high intrinsic variance, implying that the causal effect of a component is a volatile random variable rather than a fixed property. We then demonstrate that circuit discovery pipelines inherit this variance and further amplify it. Fast approximation methods, such as Edge Attribution Patching and its successors, introduce additional estimation noise, while aggregating these noisy scores over datasets leads to fragile structural estimates. Consequently, small perturbations in input data or hyperparameters yield vastly different circuits. We systematically decompose these sources of variance and advocate for more rigorous MI practices, prioritizing statistical robustness and routine reporting of stability metrics.
>
---
#### [replaced 021] Retrieval, Reward, and Training Protocols: What Matters in Training Search Agents?
- **分类: cs.CL**

- **简介: 该论文属于搜索代理训练任务，旨在明确影响搜索代理性能的关键因素。通过控制实验，研究数据覆盖、奖励机制和训练策略，提出有效训练指南。**

- **链接: [https://arxiv.org/pdf/2605.27881](https://arxiv.org/pdf/2605.27881)**

> **作者:** Yibo Zhao; Zichen Ding; Jiayi Wu; Zun Wang; Xiang Li
>
> **备注:** 18pages, 4 figures, and 15 tables
>
> **摘要:** Search agents powered by large language models can autonomously decompose queries, retrieve information, and synthesize answers through multi-step reasoning. However, the rapid growth of training methods has outpaced controlled comparison: existing works differ in retrieval corpora, reward designs, and training protocols, making it unclear what actually drives improvements. We present a controlled empirical study that isolates three under-explored dimensions of search agent training. First, we identify a critical data-coverage issue in the widely used Wikipedia 2018 corpus and show that correcting it alone yields larger gains than the differences between training algorithms. Second, we systematically compare outcome-based and process-based reward methods across three base models, finding that the simplest outcome-based approach achieves competitive or superior performance in most settings, and that process-level credit assignment can over-correct agent behavior. Third, we analyze training data diversity, off-policy data utilization, and search budget scaling, distilling practical guidelines for training effective search agents. Our code is available at this https URL.
>
---
#### [replaced 022] The Information Geometry of Softmax: Probing and Steering
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文研究AI系统如何通过信息几何编码语义结构，解决表示空间几何与行为关系的问题。工作包括提出“双向操控”方法，实现对概念的稳定控制。**

- **链接: [https://arxiv.org/pdf/2602.15293](https://arxiv.org/pdf/2602.15293)**

> **作者:** Kiho Park; Todd Nief; Yo Joong Choe; Victor Veitch
>
> **备注:** Code is available at this https URL
>
> **摘要:** This paper concerns the question of how AI systems encode semantic structure into the geometric structure of their representation spaces. The motivating observation is that the natural geometry of these representation spaces should reflect the way models use representations to produce behavior. We focus on the important special case of representations that define softmax distributions. In this case, we argue that the natural geometry is information geometry. Our focus is on the role of information geometry on semantic encoding and the linear representation hypothesis. As an illustrative application, we develop "dual steering", a method for robustly steering representations to exhibit a particular concept using linear probes. We prove that dual steering optimally modifies the target concept while minimizing changes to off-target concepts. Empirically, we find that dual steering enhances the controllability and stability of concept manipulation.
>
---
#### [replaced 023] Who Endorsed It? Measuring Authority Bias Across Expertise Levels in Language Models
- **分类: cs.CL; cs.LG**

- **简介: 论文研究语言模型在面对不同权威来源的推荐时产生的偏差，属于自然语言处理中的可信度评估任务。该文解决模型对专家建议的过度信任问题，通过实验分析不同领域模型的反应并提出缓解方法。**

- **链接: [https://arxiv.org/pdf/2601.13433](https://arxiv.org/pdf/2601.13433)**

> **作者:** Priyanka Mary Mammen; Emil Joswin; Shankar Venkitachalam
>
> **摘要:** Prior research demonstrates that performance of language models on reasoning tasks can be influenced by suggestions, hints and endorsements. However, the influence of endorsement source credibility remains underexplored. We investigate whether language models exhibit systematic bias based on the perceived expertise of the provider of the endorsement. Across 4 datasets spanning mathematical, legal, and medical reasoning, we evaluate 11 models using personas representing four expertise levels per domain. Our results reveal that models are increasingly susceptible to incorrect/misleading endorsements as source expertise increases, with higher-authority sources inducing not only accuracy degradation but also increased confidence in wrong answers. We also show that this authority bias is mechanistically encoded within the model and a model can be steered away from the bias, thereby improving its performance even when an expert gives a misleading endorsement.
>
---
#### [replaced 024] Synthetic Stimuli, Real Gains: Rethinking VLM Fine-Tuning Through Fully Controlled Data Generation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型（VLM）的微调任务，旨在解决数据偏差和分布不平衡问题。通过完全控制的数据生成与标注流程，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.11440](https://arxiv.org/pdf/2511.11440)**

> **作者:** Massimo Rizzoli; Simone Alghisi; Seyed Mahed Mousavi; Giuseppe Riccardi
>
> **摘要:** Performance gains of Vision Language Models (VLMs) obtained by fine-tuning are generally based on ad hoc data collection and annotation of real-world scenes. Despite the improvements, this process is often prone to biases, errors, and distribution imbalance, resulting in overfitting and imbalanced performance. Although a few studies have explored synthetic data generation, they typically lack control over data distribution and annotation quality. In this work, we re-evaluate the potential of model fine-tuning by exploring a fully controlled data generation and annotation pipeline, obtaining bias-free data with balanced distribution and clean annotations. Using the spatial reasoning task of identifying the absolute position of an object as a use case, we fine-tune state-of-the-art VLMs and conduct exhaustive evaluations on both synthetic and real-world benchmarks, including transferability to real-world scenes. Our experiments reveal two key findings: 1) fine-tuning on balanced data yields uniform performance across the visual scene and mitigates common biases with as few as 130 samples; and 2) fine-tuning on synthetic stimuli improves performance by 13% on real-world data (COCO), outperforming models fine-tuned on the full COCO train set.
>
---
#### [replaced 025] SAAS: Self-Aware Reinforcement Learning for Over-Search Mitigation in Agentic Search
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于信息检索任务，解决代理搜索中的过搜索问题。通过引入自感知强化学习框架SAAS，提升代理的自我意识，减少不必要的搜索，降低计算成本。**

- **链接: [https://arxiv.org/pdf/2605.29796](https://arxiv.org/pdf/2605.29796)**

> **作者:** Yunbo Tang; Chengyi Yang; Shiyu Liu; Zhishang Xiang; Zerui Chen; Qinggang Zhang; Jinsong Su
>
> **摘要:** Agentic search enables LLMs to solve complex multi-hop questions through iterative reasoning and external search. Despite the effectiveness, these systems often suffer from a critical limitation in practice: agents fail to recognize their own knowledge boundaries, blindly triggering searches when internal knowledge suffices and failing to terminate search even when adequate evidence has been collected. The lack of self-awareness leads to severe \textbf{over-search}, incurring substantial inference latency and prohibitive computational cost. To this end, we propose SAAS, a novel RL framework designed to cultivate dynamic self-awareness that precisely regulates search behavior without compromising accuracy. SAAS introduces three key components: (i) a search boundary modeling mechanism, which identifies the search boundary under the evolving policy by contrasting search-disabled and search-enabled rollouts; (ii) a boundary-aware reward module, which translates this boundary awareness into trajectory-level penalties, suppressing unnecessary and redundant searches; and (iii) a stage-wise optimization strategy, which leverages a sequential curriculum to prioritize reasoning over search regularization, thereby avoiding reward hacking. Extensive experiments demonstrate that SAAS substantially reduces over-search, while maintaining accuracy. Our code and implementation details are released at this https URL.
>
---
#### [replaced 026] Chain-of-Thought Reasoning In The Wild Is Not Always Faithful
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究模型在链式推理中出现的不忠实现象。工作揭示了模型在无显式偏见提示下仍可能产生矛盾推理，指出其内部隐含偏见导致的错误解释。**

- **链接: [https://arxiv.org/pdf/2503.08679](https://arxiv.org/pdf/2503.08679)**

> **作者:** Iván Arcuschin; Jett Janiak; Robert Krzyzanowski; Senthooran Rajamanoharan; Neel Nanda; Arthur Conmy
>
> **备注:** Published at the 43rd International Conference on Machine Learning (ICML 2026)
>
> **摘要:** Recent studies indicate that when faced with explicit biases in prompts, models often omit mentioning these biases in their Chain-of-Thought (CoT) output, revealing that verbalized reasoning can give an incorrect picture of how models arrive at conclusions (unfaithfulness). In this work, we show that unfaithful CoT also occurs on naturally worded, non-adversarial prompts without adding artificial biases or editing model outputs. We find that when separately presented with the questions "Is X bigger than Y?" and "Is Y bigger than X?", models sometimes produce superficially coherent arguments to justify systematically answering Yes to both or No to both, despite the contradiction. We present preliminary evidence that this is due to models' implicit biases towards Yes or No, labeling this Implicit Post-Hoc Rationalization. Our results reveal rates up to 13% for production models, and while frontier models are more faithful, none are entirely so, including thinking models like DeepSeek R1 (0.37%) and Sonnet 3.7 with thinking (0.04%). We also investigate Unfaithful Illogical Shortcuts, where models use subtly illogical reasoning to make speculative answers to hard math problems seem rigorously proven. Our findings indicate that while CoT can be useful for assessing outputs, it is not a complete account of the internal process that produced the model's answer and should be used with caution in agentic or safety-critical settings.
>
---
#### [replaced 027] TransLPRNet: Lite Vision-Language Network for Single/Dual-line Chinese License Plate Recognition
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于车牌识别任务，旨在解决复杂环境下单双线中文车牌识别难题。通过构建合成数据集并设计轻量网络，提升识别精度与速度。**

- **链接: [https://arxiv.org/pdf/2507.17335](https://arxiv.org/pdf/2507.17335)**

> **作者:** Guangzhu Xu; Zhi Ke; Pengcheng Zuo; Bangjun Lei
>
> **摘要:** License plate recognition in open environments is widely applicable across various domains; however, the diversity of license plate types and imaging conditions presents significant challenges. To address the limitations encountered by CNN and CRNN-based approaches in license plate recognition, this paper proposes a unified solution that integrates a lightweight visual encoder with a text decoder, within a pre-training framework tailored for single and double-line Chinese license plates. To mitigate the scarcity of double-line license plate datasets, we constructed a single/double-line license plate dataset by synthesizing images, applying texture mapping onto real scenes, and blending them with authentic license plate images. Furthermore, to enhance the system's recognition accuracy, we introduce a perspective correction network (PTN) that employs license plate corner coordinate regression as an implicit variable, supervised by license plate view classification information. This network offers improved stability, interpretability, and low annotation costs. The proposed algorithm achieves an average recognition accuracy of 99.34% on the corrected CCPD test set under coarse localization disturbance. When evaluated under fine localization disturbance, the accuracy further improves to 99.58%. On the double-line license plate test set, it achieves an average recognition accuracy of 98.70%, with processing speeds reaching up to 167 frames per second, indicating strong practical applicability.
>
---
#### [replaced 028] NeUQI: Near-Optimal Uniform Quantization Parameter Initialization for Low-Bit LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于低比特大语言模型量化任务，旨在解决量化参数初始化效率低的问题。提出NeUQI方法，通过优化尺度参数提升模型性能。**

- **链接: [https://arxiv.org/pdf/2505.17595](https://arxiv.org/pdf/2505.17595)**

> **作者:** Li Lin; Xinyu Hu; Xiaojun Wan
>
> **备注:** accepted by ICML 2026
>
> **摘要:** Large language models (LLMs) achieve impressive performance across domains but face significant challenges when deployed on consumer-grade GPUs or personal devices such as laptops, due to high memory consumption and inference costs. Post-training quantization (PTQ) of LLMs offers a promising solution that reduces their memory footprint and decoding latency. In practice, PTQ with uniform quantization representation is favored due to its efficiency and ease of deployment, as uniform quantization is widely supported by mainstream hardware and software libraries. Recent studies on low-bit uniform quantization have led to noticeable improvements in post-quantization model performance; however, they mainly focus on quantization methodologies, while the initialization of quantization parameters remains underexplored and still relies on the conventional Min-Max formula. In this work, we identify the limitations of the Min-Max formula, move beyond its constraints, and propose NeUQI, a method that efficiently determines near-optimal initialization for uniform quantization. Our NeUQI simplifies the joint optimization of the scale and zero-point by deriving the zero-point for a given scale, thereby reducing the problem to a scale-only optimization. Benefiting from the improved quantization parameters, our NeUQI consistently outperforms existing methods in the experiments with the LLaMA and Qwen families on various settings and tasks. Furthermore, when combined with a lightweight distillation strategy, NeUQI even achieves superior performance to PV-tuning, a considerably more resource-intensive method.
>
---
#### [replaced 029] No Reader Left Behind: Multi-Agent Summaries Everyone Can Understand
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本摘要任务，旨在解决不同读者群体的可理解性问题。提出NRLB框架，通过多智能体模拟不同读者，提升摘要的易读性和准确性。**

- **链接: [https://arxiv.org/pdf/2605.28836](https://arxiv.org/pdf/2605.28836)**

> **作者:** Jimin Jung; MyoungJin Kim; Jaehyung Seo; Heuiseok Lim
>
> **摘要:** The Plain Writing Act in the United States requires government documents to be accessible in clear and simple language that the general public can easily understand, yet existing summarization systems struggle to address diverse linguistic and cognitive barriers among general readers. We present NRLB (No Reader Left Behind), a multi-agent framework for plain language summarization that simulates three representative reader groups: elementary school student readers, non-native readers, and readers with attention deficits. NRLB combines template-based planning with iterative, reader-oriented refinement, enabling systematic detection and resolution of difficult terms, missing contexts, and confusing sentences. Evaluations across multiple datasets demonstrate consistent improvements in readability while preserving factual accuracy. Human evaluation further validates NRLB's impact, with annotator preference rates ranging from 55% to 76%, highlighting NRLB's potential to produce plain language summaries that are both faithful to the source and broadly accessible to the general public.
>
---
#### [replaced 030] OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型推理优化任务，旨在解决长序列缓存带来的内存开销问题。通过提出OBCache框架，基于注意力输出影响量化token重要性，实现更高效的缓存修剪。**

- **链接: [https://arxiv.org/pdf/2510.07651](https://arxiv.org/pdf/2510.07651)**

> **作者:** Yuzhe Gu; Xiyu Liang; Jiaojiao Zhao; Enmao Diao
>
> **备注:** ICML 2026
>
> **摘要:** Large language models (LLMs) with extended context windows enable powerful applications but impose significant memory overhead, as caching all key-value (KV) states scales linearly with sequence length and batch size. Existing cache eviction methods address this by exploiting attention sparsity, yet they typically rank tokens heuristically using accumulated attention weights without considering their true impact on attention outputs. We propose Optimal Brain Cache (OBCache), a principled framework that formulates cache eviction as a layer-wise structured pruning problem. Building upon the Optimal Brain Damage (OBD) theory, OBCache quantifies token saliency by measuring the perturbation in attention outputs induced by pruning tokens, with closed-form scores derived for isolated keys, isolated values, and joint key-value pairs. Our scores account not only for attention weights but also for information from value states and attention outputs, thereby enhancing existing eviction strategies with output-aware signals. Experiments on LLaMA and Qwen models demonstrate that replacing the heuristic scores in existing works, which estimate token saliency across different query positions, with OBCache's output-aware scores consistently improves long-context accuracy. Code is available at this https URL.
>
---
#### [replaced 031] Esoteric Language Models: A Family of Any-Order Diffusion LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Eso-LMs，融合AR与MDM模型，解决生成效率与困惑度问题，实现并行生成与KV缓存，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2506.01928](https://arxiv.org/pdf/2506.01928)**

> **作者:** Subham Sekhar Sahoo; Zhihan Yang; Yash Akhauri; Johnna Liu; Deepansha Singh; Zhoujun Cheng; Zhengzhong Liu; Eric Xing; John Thickstun; Arash Vahdat
>
> **备注:** ICML 2026
>
> **摘要:** Diffusion-based language models offer a compelling alternative to autoregressive (AR) models by enabling parallel and controllable generation. Within this family, Masked Diffusion Models (MDMs) currently perform best but still underperform AR models in perplexity and lack key inference-time efficiency features, most notably KV caching. We introduce Eso-LMs, a new family of models that fuses AR and MDM paradigms, smoothly interpolating between their perplexities while overcoming their respective limitations. Unlike prior work, which uses transformers with bidirectional attention as MDM denoisers, we exploit the connection between MDMs and Any-Order autoregressive models and adopt causal attention. This design lets us compute the exact likelihood of MDMs for the first time and, crucially, enables us to introduce KV caching for MDMs while preserving parallel generation for the first time, significantly improving inference efficiency. Combined with an optimized sampling schedule, Eso-LMs establish a new state of the art on the speed-quality Pareto frontier for unconditional generation. We provide the code, model checkpoints, and the video tutorial on the project page: this https URL.
>
---
#### [replaced 032] MaskClaw: Edge-Side Personalized Privacy Arbitration for GUI Agents with Behavior-Driven Skill Evolution
- **分类: cs.CR; cs.CL**

- **简介: 该论文提出MaskClaw，解决GUI代理中的隐私保护问题，通过边缘计算实现个性化隐私决策，避免敏感信息泄露。**

- **链接: [https://arxiv.org/pdf/2605.28646](https://arxiv.org/pdf/2605.28646)**

> **作者:** Yanqiu Zhao; Dongying Zheng; Kaibo Huang; Yukun Wei; Zhongliang Yang; Linna Zhou
>
> **备注:** Preprint. Submitted to EMNLP 2026. 21 pages, including appendices; 5 figures Under review. Yanqiu Zhao and Dongying Zheng contributed equally to this work
>
> **摘要:** GUI agents rely on screenshots to infer intent and operate across applications, but these screenshots often contain private messages, medical records, payment credentials, and workplace-specific workflows. Privacy decisions in this setting depend on task, recipient, application state, and user role, yet static PII detectors miss these boundaries and cloud-side VLM reasoning can upload the raw screen before deciding what should be protected. We present MaskClaw, an edge-side privacy arbitrator for GUI agents. MaskClaw extracts local visual evidence, retrieves user- and task-specific policy memory, and decides Allow, Mask, or Ask before raw screenshots leave a trusted user- or organization-controlled environment. In five designed skill-evolution scenarios, it turns corrections, cancellations, and edits into reusable privacy skills checked by a sandbox gate. We introduce P-GUI-Evo, a benchmark built from real UI patterns, reconstructed HTML screens, and sanitized labels. Experiments show that pattern matching, cloud reasoning, and routing alone tend to over-confirm, over-mask, or expose raw screenshots under the same protocol. The artifact is available at this https URL.
>
---
#### [replaced 033] HypoSpace: A Diagnostic Benchmark for Set-Valued Hypothesis Generation under Underdetermination and Sublinear Coverage Bounds
- **分类: cs.CL**

- **简介: 该论文提出HypoSpace，一个用于评估大语言模型在不确定科学问题中生成多假设能力的基准。任务是解决设值推理中的覆盖不足问题，通过三个指标进行评估，并展示解码策略的改进效果。**

- **链接: [https://arxiv.org/pdf/2510.15614](https://arxiv.org/pdf/2510.15614)**

> **作者:** Tingting Chen; Beibei Lin; Zifeng Yuan; Qiran Zou; Hongyu He; Anirudh Goyal; Yew-Soon Ong; Dianbo Liu
>
> **摘要:** Many scientific problems are underdetermined: multiple distinct hypotheses are equally consistent with the same observations. In such settings, effective inference requires not only producing valid explanations, but also systematically exploring and covering the admissible hypothesis set. We introduce HypoSpace, a benchmark that treats large language models (LLMs) as samplers over finite hypothesis spaces and evaluates them on three metrics: Validity, Uniqueness, and Recovery. HypoSpace spans three structured domains (causal graph inference, gravity-constrained 3D voxel reconstruction, and Boolean genetic interaction modeling) with deterministic validators and exactly enumerable solution spaces, plus real-world anchored case studies. Empirically, HypoSpace reveals a capability- and scale-dependent coverage failure: models can maintain high Validity while exhibiting reduced Uniqueness and Recovery as admissible hypothesis spaces become larger or more combinatorial. We further show that the analysis on stratified decoding partially mitigates this collapse, demonstrating HypoSpace's utility as a diagnostic benchmark for set-valued inference. Code is available at: this https URL.
>
---
#### [replaced 034] Efficient Benchmarking Is Just Feature Selection and Multiple Regression
- **分类: stat.ML; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于高效基准评估任务，旨在通过特征选择和多元回归降低LLM评估成本。工作包括使用核岭回归和mRMR算法提升预测精度与效率。**

- **链接: [https://arxiv.org/pdf/2605.25773](https://arxiv.org/pdf/2605.25773)**

> **作者:** Sam Bowyer; Acyr Locatelli; Kris Cao
>
> **备注:** 36 pages, 27 figures
>
> **摘要:** Efficient benchmarking techniques aim to lower the computational cost of evaluating LLMs by predicting full benchmark scores using only a subset of a benchmark's questions. By reframing this problem as an instance of multiple regression with feature selection, we find that existing efficient benchmarking methods can be greatly improved by simply using kernel ridge regression at the prediction stage. Additionally, using an information-theoretic feature-selection algorithm called minimum redundancy maximum relevance (mRMR), we can further improve upon these methods by selecting question subsets that will be maximally useful for prediction. Except in very data-poor settings, these approaches consistently achieve smaller prediction errors (in both MAE and RMSE), and greater ranking correlation between predicted and true scores (in both Spearman $\rho$ and Kendall $\tau$) across a range of benchmarks using both binary and continuous metrics. Furthermore, mRMR subsampling is much faster than competitor methods (which often involve fitting probabilistic models or running clustering algorithms), and is more likely to select the same questions under different random seeds or training data splits. Tutorial code can be found at this https URL .
>
---
#### [replaced 035] Evaluation of Automatic Speech Recognition Using Generative Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自动语音识别（ASR）评估任务，旨在解决传统指标WER对语义不敏感的问题。通过三种方法评估生成式大语言模型在语义评估中的有效性。**

- **链接: [https://arxiv.org/pdf/2604.21928](https://arxiv.org/pdf/2604.21928)**

> **作者:** Thibault Bañeras-Roux; Shashi Kumar; Driss Khalil; Sergio Burdisso; Petr Motlicek; Shiran Liu; Mickael Rouvier; Jane Wottawa; Richard Dufour
>
> **摘要:** Automatic Speech Recognition (ASR) is traditionally evaluated using Word Error Rate (WER), a metric that is insensitive to meaning. Embedding-based semantic metrics are better correlated with human perception, but decoder-based Large Language Models (LLMs) remain underexplored for this task. This paper evaluates their relevance through three approaches: (1) selecting the best hypothesis between two candidates, (2) computing semantic distance using generative embeddings, and (3) qualitative classification of errors. On the HATS dataset, the best LLMs achieve 92--94\% agreement with human annotators for hypothesis selection, compared to 63\% for WER, also outperforming semantic metrics. Embeddings from decoder-based LLMs show performance comparable to encoder models. Finally, LLMs offer a promising direction for interpretable and semantic ASR evaluation.
>
---
#### [replaced 036] GEM-Bench: A Benchmark for Ad-Injected Response Generation within Generative Engine Marketing
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于生成式营销任务，旨在解决广告注入响应生成的问题。提出GEM-Bench基准，包含数据集、评估指标和基线方法，以促进相关研究。**

- **链接: [https://arxiv.org/pdf/2509.14221](https://arxiv.org/pdf/2509.14221)**

> **作者:** Silan Hu; Shiqi Zhang; Yimin Shi; Xiaokui Xiao
>
> **备注:** Technical Report
>
> **摘要:** Generative Engine Marketing (GEM) is an emerging ecosystem for monetizing generative engines, such as LLM-based chatbots, by seamlessly integrating relevant advertisements into their responses. At the core of GEM lies the generation and evaluation of ad-injected responses. However, existing benchmarks are not specifically designed for this purpose, which limits future research. To address this gap, we propose GEM-Bench, the first comprehensive benchmark for ad-injected response generation in GEM. GEM-Bench includes three curated datasets covering both chatbot and search scenarios, a metric ontology that captures multiple dimensions of user satisfaction and engagement, and several baseline solutions implemented within an extensible multi-agent framework. Our preliminary results indicate that, while simple prompt-based methods achieve reasonable engagement such as click-through rate, they often reduce user satisfaction. In contrast, approaches that insert ads based on pre-generated ad-free responses help mitigate this issue but introduce additional overhead. These findings highlight the need for future research on designing more effective and efficient solutions for generating ad-injected responses in GEM. The benchmark and all related resources are publicly available at this https URL.
>
---
#### [replaced 037] GradMem: Learning to Write Context into Memory with Test-Time Gradient Descent
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出GradMem，用于在推理时通过梯度下降将上下文写入记忆，解决长上下文处理的内存开销问题。任务为高效压缩记忆，工作是设计基于优化的上下文写入方法。**

- **链接: [https://arxiv.org/pdf/2603.13875](https://arxiv.org/pdf/2603.13875)**

> **作者:** Yuri Kuratov; Matvey Kairov; Aydar Bulatov; Ivan Rodkin; Mikhail Burtsev
>
> **备注:** International Conference on Machine Learning (ICML) 2026
>
> **摘要:** Many large language model applications require conditioning on long contexts. Transformers typically support this by storing a large per-layer KV-cache of past activations, which incurs substantial memory overhead. A desirable alternative is compressive memory: read a context once, store it in a compact state, and answer many queries from that state. We study this in a context removal setting, where the model must generate an answer without access to the original context at inference time. We introduce GradMem, which writes context into memory via per-sample test-time optimization. Given a context, GradMem performs a few steps of gradient descent on a small set of prefix memory tokens while keeping model weights frozen. GradMem explicitly optimizes a model-level self-supervised context reconstruction loss, resulting in a loss-driven write operation with iterative error correction, unlike forward-only methods. On associative key--value retrieval, GradMem outperforms forward-only memory writers with the same memory size, and additional gradient steps scale capacity much more effectively than repeated forward writes. We further show that GradMem transfers beyond synthetic benchmarks: with pretrained language models, it attains competitive results on natural language tasks including bAbI and SQuAD variants, relying only on information encoded in memory.
>
---
#### [replaced 038] Fully Open Meditron: An Auditable Pipeline for Clinical LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Fully Open Meditron，解决临床LLM可审计性问题，构建可复现的医疗决策支持系统管道。**

- **链接: [https://arxiv.org/pdf/2605.16215](https://arxiv.org/pdf/2605.16215)**

> **作者:** Xavier Theimer-Lienhard; Mushtaha El-Amin; Fay Elhassan; Sahaj Vaidya; Victor Cartier-Negadi; David Sasu; Lars Klein; Mary-Anne Hartley
>
> **备注:** Preprint. 31 pages, 10 figures. Code, models, and data: this https URL
>
> **摘要:** Clinical decision support systems (CDSS) require scrutable, auditable pipelines that enable rigorous, reproducible validation. Yet current LLM-based CDSS remain largely opaque. Most "open" models are open-weight only, releasing parameters while withholding the data provenance, curation procedures, and generation pipelines that determine model behavior. Fully Open (FO) models, which expose the complete training stack end-to-end, do not currently exist in medicine. We introduce Fully Open Meditron, the first fully open pipeline for building LLM-CDSS, comprising a clinician-audited training corpus, a reproducible data construction and training framework, and a use-aligned evaluation protocol. The corpus unifies eight public medical QA datasets into a normalized conversational format and expands coverage with three clinician-vetted synthetic extensions: exam-style QA, guideline-grounded QA derived from 46,469 clinical practice guidelines, and clinical vignettes. The pipeline enforces system-wide decontamination, gold-label resampling of teacher generations, and end-to-end validation by a four-physician panel. We evaluate using an LLM-as-a-judge protocol over expert-written clinical vignettes, calibrated against 204 human raters. We apply the recipe to five FO base models (Apertus-70B/8B-Instruct, OLMo-2-32B-SFT, EuroLLM-22B/9B-Instruct). All MeditronFO variants are preferred over their bases. Apertus-70B-MeditronFO improves +6.6 points over its base (47.2% to 53.8%) on aggregate medical benchmarks, establishing a new FO SoTA. Gemma-3-27B-MeditronFO is preferred over MedGemma in 58.6% of LLM-as-a-judge comparisons and outperforms it on HealthBench (58% vs 55.9%). These results show that fully open pipelines can achieve state-of-the-art domain-specific performance without sacrificing auditability or reproducibility.
>
---
#### [replaced 039] DynaGraph: Lightweight Multi-Model Interaction Framework via Dynamic Topological Reconfiguration
- **分类: cs.MA; cs.CL; cs.LG**

- **简介: 该论文提出DynaGraph框架，解决复杂推理任务中模型冗余与动态调整难题，通过动态拓扑重构实现高效多模型协作。**

- **链接: [https://arxiv.org/pdf/2605.29511](https://arxiv.org/pdf/2605.29511)**

> **作者:** Yanxing Guo; Zihao Zheng; Fangzhou Wu; Ling Liang; Lin Bao; Zongwei Wang; Yimao Cai
>
> **摘要:** Tackling complex reasoning tasks typically relies on massive monolithic LLMs, which suffer from severe computational redundancy. While task decomposition through structured pipelines or multi-agent collaborations offers an alternative, these approaches inevitably fall into a critical dilemma: predefined static topologies are highly vulnerable to cascading errors, whereas unconstrained dynamic agents suffer from trajectory divergence and unpredictable memory bloat. To address this, we present DynaGraph, a lightweight multi-model framework driven by dynamic topological reconfiguration. At the execution level, DynaGraph multiplexes time-division PEFT adapters over a shared base model, enabling both full system training and inference deployment on a single consumer-grade GPU. At the routing level, the Evaluator continuously monitors execution confidence to trigger hierarchical self-healing: Fine-grained Patching for localized data gaps and Subgraph Reconstruction for severe logical ruptures. Experiments on StrategyQA, MATH, and FinQA demonstrate our 8B model closely approximates the reasoning capabilities of a 72B monolithic model (e.g., 87.6% on StrategyQA, 82.7% on MATH). Furthermore, it reduces latency by up to 68.1% and token consumption by 68.6% compared to unconstrained dynamic architectures.
>
---
#### [replaced 040] From Out-of-Distribution Detection to Hallucination Detection: A Geometric View
- **分类: cs.AI; cs.CL**

- **简介: 论文将幻觉检测重新定义为分布外检测任务，解决语言模型安全问题。通过调整方法，使OOD技术适用于语言模型，实现高效准确的幻觉检测。**

- **链接: [https://arxiv.org/pdf/2602.07253](https://arxiv.org/pdf/2602.07253)**

> **作者:** Litian Liu; Reza Pourreza; Yubing Jian; Yao Qin; Roland Memisevic
>
> **备注:** ICML 2026 main conference paper
>
> **摘要:** Detecting hallucinations in large language models is a critical open problem with significant implications for safety and reliability. While existing hallucination detection methods achieve strong performance in question-answering tasks, they remain less effective on tasks requiring reasoning. In this work, we revisit hallucination detection through the lens of out-of-distribution (OOD) detection, a well-studied problem in areas like computer vision. Treating next-token prediction in language models as a classification task allows us to apply OOD techniques, provided appropriate modifications are made to account for the structural differences in large language models. We show that OOD-based approaches yield training-free, single-sample-based detectors, achieving strong accuracy in hallucination detection for reasoning tasks. Overall, our work suggests that reframing hallucination detection as OOD detection provides a promising and scalable pathway toward language model safety.
>
---
#### [replaced 041] Weights to Code: Extracting Interpretable Algorithms from the Discrete Transformer
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于算法提取任务，旨在从Transformer模型中提取可解释的程序。针对表示纠缠问题，提出Discrete Transformer架构，实现符号表达式恢复与程序合成。**

- **链接: [https://arxiv.org/pdf/2601.05770](https://arxiv.org/pdf/2601.05770)**

> **作者:** Yifan Zhang; Wei Bi; Kechi Zhang; Dongming Jin; Jie Fu; Zhi Jin
>
> **摘要:** Algorithm extraction aims to synthesize executable programs directly from models trained on algorithmic tasks, enabling de novo recovery of executable mechanisms from weights without relying on human-written target programs. However, applying this paradigm to Transformer is complicated by representation entanglement (e.g., superposition), where features encoded in overlapping directions substantially hinder the recovery of symbolic expressions. We propose the Discrete Transformer, an architecture explicitly designed to bridge the gap between continuous representations and discrete symbolic logic. By injecting discreteness through temperature-annealed sampling, our framework effectively leverages hypothesis testing and symbolic regression to extract human-readable programs. Empirically, the Discrete Transformer achieves performance comparable to the RNN-based MIPS baseline on shared discrete tasks, while broadening extraction to tasks with continuous-valued intermediate computations. Finally, we show that architectural inductive biases provide fine-grained control over synthesized programs, establishing the Discrete Transformer as a controllable testbed for algorithm extraction and Transformer interpretability.
>
---
#### [replaced 042] X-GS: An Extensible Framework for Perceiving and Thinking via 3D Gaussian Splatting
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出X-GS框架，解决3DGS应用中多任务协同问题，通过Perceiver和Thinker实现高效SLAM与多模态任务。**

- **链接: [https://arxiv.org/pdf/2603.09632](https://arxiv.org/pdf/2603.09632)**

> **作者:** Yueen Ma; Zenglin Xu; Irwin King
>
> **摘要:** 3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, subsequently extending into numerous spatial AI applications. However, most existing 3DGS methods operate in isolation, focusing on specific domains. In this paper, we introduce X-GS, an extensible framework consisting of two major components. The X-GS-Perceiver unifies a broad range of 3DGS techniques to enable real-time online SLAM with semantic distillation. The X-GS-Thinker accommodates multimodal models, enabling them to seamlessly interface with the Perceiver to complete downstream tasks. In our implementation of X-GS, the Perceiver leverages the latest vision foundation models to improve online SLAM performance and employs three key mechanisms to accelerate semantic distillation. The Thinker can be built upon both contrastive and generative vision-language models and utilizes the Perceiver's semantic Gaussian splats to unlock capabilities such as 3D visual grounding and scene captioning. Experimental results on diverse benchmarks demonstrate the efficiency and newly unlocked multimodal capabilities of the X-GS framework.
>
---
#### [replaced 043] Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型融合任务，解决LoRA微调模型在合并时性能下降的问题。提出OSRM方法，通过约束LoRA子空间提升融合效果和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2505.22934](https://arxiv.org/pdf/2505.22934)**

> **作者:** Haobo Zhang; Jiayu Zhou
>
> **备注:** 14 pages, 5 figures, 16 tables, accepted by ACL 2025
>
> **摘要:** Fine-tuning large language models (LMs) for individual tasks yields strong performance but is expensive for deployment and storage. Recent works explore model merging to combine multiple task-specific models into a single multi-task model without additional training. However, existing merging methods often fail for models fine-tuned with low-rank adaptation (LoRA), due to significant performance degradation. In this paper, we show that this issue arises from a previously overlooked interplay between model parameters and data distributions. We propose Orthogonal Subspaces for Robust model Merging (OSRM) to constrain the LoRA subspace *prior* to fine-tuning, ensuring that updates relevant to one task do not adversely shift outputs for others. Our approach can seamlessly integrate with most existing merging algorithms, reducing the unintended interference among tasks. Extensive experiments on eight datasets, tested with three widely used LMs and two large LMs, demonstrate that our method not only boosts merging performance but also preserves single-task accuracy. Furthermore, our approach exhibits greater robustness to the hyperparameters of merging. These results highlight the importance of data-parameter interaction in model merging and offer a plug-and-play solution for merging LoRA models.
>
---
#### [replaced 044] Symbolic Intermediaries as a Linguistic-Numerical Interface for LLM-Driven Geometric Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于几何推理任务，解决LLMs难以直接处理物理模拟器数值输出的问题。通过符号中介将数值转换为可解释的符号表达，提升LLMs在几何领域的应用能力。**

- **链接: [https://arxiv.org/pdf/2505.17607](https://arxiv.org/pdf/2505.17607)**

> **作者:** João Pedro Gandarela; Thiago Rios; Stefan Menzel; André Freitas
>
> **备注:** 33 pages, 18 figures
>
> **摘要:** Large Language Models (LLMs) display reasoning capabilities over linguistic and symbolic objects but have limited capabilities to directly interpret the continuous numerical outputs of physics simulators, e.g., distances, curvatures, and trajectories that resist discrete tokenisation. Across spatially grounded engineering reasoning tasks, from mechanism design to motion planning, this defines a fundamental gap, which limits the wider application of LLMs within broader geometrical domains, for exmaple interfacing with physics simulators. We propose symbolic intermediaries, compact analytical expressions discovered via symbolic regression, as a structured interface that translates a simulator's numerical traces into a symbolic form, which language models can interpret, compare, and critique while preserving the original geometric semantics. Around this interface we build an agentic coordination-and-refinement loop: a design agent maps natural-language specifications to executable simulation code, a critique agent reasons over the shared symbolic vocabulary, and a revision step turns this feedback into grounded refinement decisions, enabling inference-time generalization without parameter updates. On the MSynth benchmark for planar mechanism synthesis, all three evaluated LLM agents outperform a budget-matched genetic-algorithm baseline by 19-53% (up to 63% lower median error with feedback), and analysis of the critique entries across three model architectures shows that the interface shifts reasoning from generic structural commentary to grounded geometric verification. The principle of translating continuous simulation outputs into symbolic forms generalises to any domain where simulator behaviour must be interpreted linguistically.
>
---
#### [replaced 045] Gap-K%: Measuring Top-1 Prediction Gap for Detecting Pretraining Data
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于预训练数据检测任务，旨在解决LLM预训练数据隐私和版权问题。提出Gap-K%方法，通过分析模型预测与目标token的差异来检测预训练数据。**

- **链接: [https://arxiv.org/pdf/2601.19936](https://arxiv.org/pdf/2601.19936)**

> **作者:** Minseo Kwak; Jaehyung Kim
>
> **备注:** ACL 2026 Main Conference; 15 pages
>
> **摘要:** The opacity of massive pretraining corpora in Large Language Models (LLMs) raises significant privacy and copyright concerns, making pretraining data detection a critical challenge. Existing state-of-the-art methods typically rely on token likelihoods, yet they often overlook the gap between the target token and the model's top-1 prediction, as well as local correlations between adjacent tokens. In this work, we propose Gap-K%, a novel pretraining data detection method grounded in the optimization dynamics of LLM pretraining. By analyzing the next-token prediction objective, we observe that discrepancies between the model's top-1 prediction and the target token induce strong gradient signals, which are explicitly penalized during training. Motivated by this, Gap-K% leverages the log probability gap between the top-1 predicted token and the target token, incorporating a sliding window strategy to capture local correlations and mitigate token-level fluctuations. Extensive experiments on the WikiMIA and MIMIR benchmarks demonstrate that Gap-K% achieves state-of-the-art performance, consistently outperforming prior baselines across various model sizes and input lengths.
>
---
#### [replaced 046] SAC-Opt: Semantic Anchors for Iterative Correction in Optimization Modeling
- **分类: cs.AI; cs.CL; cs.PL**

- **简介: 该论文属于优化建模任务，解决LLM生成代码中的语义错误问题。提出SAC-Opt框架，通过语义锚点修正逻辑错误，提升模型准确性。**

- **链接: [https://arxiv.org/pdf/2510.05115](https://arxiv.org/pdf/2510.05115)**

> **作者:** Yansen Zhang; Qingcan Kang; Yujie Chen; Yufei Wang; Xiongwei Han; Tao Zhong; Mingxuan Yuan; Chen Ma
>
> **备注:** ICML 2026 accepted
>
> **摘要:** Large language models (LLMs) have opened new paradigms in optimization modeling by enabling the generation of executable solver code from natural language descriptions. Despite this promise, existing approaches typically remain solver-driven: they rely on single-pass forward generation and apply limited post-hoc fixes based on solver error messages, leaving undetected semantic errors that silently produce syntactically correct but logically flawed models. To address this challenge, we propose SAC-Opt, a backward-guided correction framework that grounds optimization modeling in problem semantics rather than solver feedback. At each step, SAC-Opt aligns the original semantic anchors with those reconstructed from the generated code and selectively corrects only the mismatched components, driving convergence toward a semantically faithful model. This anchor-driven correction enables fine-grained refinement of constraint and objective logic, enhancing both fidelity and robustness without requiring additional training or supervision. Empirical results on seven public datasets demonstrate that SAC-Opt improves average modeling accuracy by 7.7%, with gains of up to 21.9% on the ComplexLP dataset. These findings highlight the importance of semantic-anchored correction in LLM-based optimization workflows to ensure faithful translation from problem intent to solver-executable code.
>
---
#### [replaced 047] Chunking German Legal Code
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律信息检索任务，旨在提升德国法律文本的检索效果。通过对比多种分块策略，发现遵循法律结构的方法效果最佳。**

- **链接: [https://arxiv.org/pdf/2605.19806](https://arxiv.org/pdf/2605.19806)**

> **作者:** Max Prior; Natalia Milanova; Andreas Schultz
>
> **备注:** Accepted at the Eigth Workshop on Automated Semantic Analysis of Information in Legal Texts co-located with the 21th International Conference on Artificial Intelligence and Law (ICAIL 2026)
>
> **摘要:** This paper investigates chunking strategies for retrieval-augmented generation on German statutory law, using the German Civil Code as a structured benchmark corpus. We implement and compare a range of segmentation approaches, including structural units (sections, subsections, sentences, propositions), fixed-size windows, contextual chunking, semantic clustering, Lumber-style chunking, and RAPTOR-based hierarchical retrieval. All methods are evaluated on a legal question-answering dataset with section-level gold labels, measuring recall, query latency, index build time, and storage requirements. Results show that chunking strategies aligned with the inherent legal structure - particularly section and subsection - based retrieval-achieve the highest recall, while more complex approaches that override this structure perform worse. These simpler methods also offer favorable computational efficiency compared to LLM-intensive techniques such as contextual chunking, RAPTOR, and Lumber. The findings highlight a key trade-off between semantic enrichment and operational cost, and demonstrate that preserving domain-specific structure is critical for effective legal information retrieval.
>
---
#### [replaced 048] Alignment Tampering: How Reinforcement Learning from Human Feedback Is Exploited to Optimize Misaligned Biases
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于AI对齐任务，探讨RLHF方法中因模型影响偏好数据导致的偏差放大问题，提出“对齐篡改”概念并验证其影响。**

- **链接: [https://arxiv.org/pdf/2605.27355](https://arxiv.org/pdf/2605.27355)**

> **作者:** Dongyoon Hahm; Dylan Hadfield-Menell; Kimin Lee
>
> **备注:** Accepted at ICML 2026, Source code: this https URL
>
> **摘要:** Reinforcement Learning from Human Feedback (RLHF) is the standard method to align Large Language Models (LLMs) with human preferences. In this work, we introduce alignment tampering, a potential vulnerability where the LLM undergoing alignment influences the preference dataset, causing RLHF to amplify undesired behaviors. This arises from core limitations of RLHF: (1) preference datasets are constructed from the LLM's own outputs, allowing it to influence them, and (2) pairwise comparisons only indicate which response is better, not why. These limitations can be exploited to cause alignment tampering. For example, if an LLM generates biased responses with higher quality, annotators will prefer them based on quality. However, preference labels do not distinguish quality from bias, and the reward model inherits this limitation. Optimizing such rewards through reinforcement learning or best-of-N sampling can amplify misaligned biases. Our experiments demonstrate amplification across diverse biases: from keyword bias to propaganda (e.g., sexism), brand promotion, and instrumental goal-seeking. Mitigation remains challenging, as existing techniques for robust RLHF fail to fully resolve alignment tampering without sacrificing response quality. These findings reveal structural vulnerabilities of current RLHF and emphasize the need to prevent this vulnerability. Project page: this https URL
>
---
#### [replaced 049] The relative strength of hierarchical structure and statistics differs across the measures in naturalistic reading
- **分类: cs.CL; q-bio.NC**

- **简介: 该论文属于语言理解研究，旨在探讨句法结构与统计因素在自然阅读中的相对影响力。通过EEG和眼动实验，分析句法深度对阅读行为和神经活动的影响，揭示句法结构在在线理解中的作用及其强度。**

- **链接: [https://arxiv.org/pdf/2509.23195](https://arxiv.org/pdf/2509.23195)**

> **作者:** Nan Wang; Hanlin Wu; Jiaxuan Li
>
> **摘要:** The hierarchical syntactic structure and non-hierarchical, statistical, or sequential factors have long been framed as rival theories in accounting for online comprehension. A lot of evidence has shown that both hierarchical and non-hierarchical factors can shape comprehension and the more open question is when, and how strongly, hierarchy exerts its influence in comprehension. We addressed the question with co-registered EEG and eye-tracking, treating syntactic depth as the variable for operationalizing hierarchical structure. For the timing question, hierarchical syntactic structure is shown to influence reading before reading a sentence and can emerge as early as 108ms before reading. This is supported by both transitional probability analysis and regression on fixation-related potential. Analyses on fixation-transition showed that readers preferentially moved between syntactically central words rather than according to serial word order, suggesting that scanpaths are driven by deep syntactic structure rather than by pure statistics. For the strength question, we combined Bayesian network modeling and regression analysis to show that strength of a variable is dependent on the phenomenon that is to be explained. Bayesian network analysis showed that hierarchical syntactic structure carried more predictive weight than statistical features. Regression on fixation-related potential demonstrated that hierarchical syntactic structure significantly predicted word-level neural activity in the front-right region in regression analyses, but is generally weaker in comparison with lexical surprisal. Evidence combined, our analyses suggested that hierarchical structure can anticipatorily guide subjects' online comprehension both on a behavioral and neural level, with its strength varies across different facets of reading behavior.
>
---
#### [replaced 050] Advancing Creative Physical Intelligence in Large Multimodal Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态任务，旨在解决大模型在开放环境中创造性使用工具的问题。通过构建基准测试和改进对齐方法，提升模型的视觉物理 grounded 解决方案能力。**

- **链接: [https://arxiv.org/pdf/2605.26396](https://arxiv.org/pdf/2605.26396)**

> **作者:** Cheng Qian; Hyeonjeong Ha; Jiayu Liu; Jeonghwan Kim; Emre Can Acikgoz; Bingxuan Li; Kunlun Zhu; Jiateng Liu; Aditi Tiwari; Zhenhailong Wang; Xiusi Chen; Mahdi Namazifar; Heng Ji
>
> **备注:** 51 Pages, 9 Figures, 7 Tables, Previous Work CreativityBench: arXiv:2605.02910
>
> **摘要:** Large multimodal models (LMMs) have rapidly advanced in perception and reasoning; however, it remains unclear whether these capabilities generalize to discovering visually grounded solutions in open-ended environments, beyond pattern recognition. In such settings, intelligence requires more than answering well-posed questions: it involves identifying how elements in a scene can be repurposed in non-obvious yet physically feasible ways. This form of creative problem-solving is central to human intelligence, but remains largely untested in current benchmarks. To evaluate this ability, we introduce MM-CreativityBench, a benchmark for affordance-grounded creative tool use in visually rich, physically constrained environments. Each instance presents a scenario image with structured views of candidate entities and their parts, enabling fine-grained, interactive evaluation of how models iteratively inspect the scene, identify relevant affordances, and compose visually and physically grounded solutions. Our experiments show that current LMMs often fall short, not due to lack of generative capability, but because they do not sustain grounded exploration. Models often overlook relevant entities, under-examine critical parts, or hallucinate attributes not grounded in the image. Motivated by this failure mode, we propose affordance-grounded alignment, which casts creative tool use as a preference learning problem. Using Direct Preference Optimization, we encourage models to prefer attribute-affordance reasoning grounded in visual evidence over hallucinated alternatives. In addition, we incorporate supervision derived from an affordance knowledge base to guide broader entity exploration and multi-turn planning. Our results show consistent gains in selecting the correct entities and parts, while substantially reducing hallucination and grounding-related errors.
>
---
#### [replaced 051] Pair-In, Pair-Out: Latent Multi-Token Prediction for Efficient LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PIPO方法，解决大语言模型推理效率问题。通过融合输入压缩与多标记预测，提升解码速度并减少验证成本。**

- **链接: [https://arxiv.org/pdf/2605.27255](https://arxiv.org/pdf/2605.27255)**

> **作者:** Wenhui Tan; Minghao Li; Xiaoqian Ma; Siqi Fan; Xiusheng Huang; Liujie Zhang; Ruihua Song; Weihang Chen
>
> **备注:** Project Page: this http URL
>
> **摘要:** Long chain-of-thought reasoning has made autoregressive decoding the dominant inference cost of modern large language models. Existing methods target either the input side (latent compression) or the output side (speculative decoding and multi-token prediction, MTP), but the two lines of work have been pursued independently. Moreover, output-side methods must incur an expensive verifier pass to validate the unreliable draft tokens predicted by MTP. To address these issues, we propose \textbf{Pair-In, Pair-Out (PIPO)}, which unifies both sides by viewing a latent compressor and an MTP head as mirror-image operations: the compressor folds two input tokens into one latent representation, while the MTP head unfolds one hidden state into one additional output token. To remove the verifier cost without sacrificing reliability, PIPO trains a lightweight confidence head that decides whether draft tokens should be accepted. We observe that On-Policy Distillation (OPD) naturally matches the rejection-sampling criterion of speculative decoding, so the confidence head can be trained alongside OPD with negligible extra cost. Experiments on AIME 2025, GPQA-Diamond, LiveCodeBench v6, and LongBench v2 with Qwen3.5-4B and 9B backbones show that PIPO improves pass@4 over regular decoding by up to $+7.15$ points, while delivering up to $2.64\times$ first-token-latency and $2.07\times$ per-token-latency speedups. Project Page: this http URL.
>
---
#### [replaced 052] MuCRASP: Multimodal Chain-of-thought Reasoning aware Structured Pruning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型压缩任务，旨在解决结构化剪枝中难以保持链式推理准确的问题。提出MuCRASP框架，在参数限制下保留跨模态对齐与推理一致性。**

- **链接: [https://arxiv.org/pdf/2605.25842](https://arxiv.org/pdf/2605.25842)**

> **作者:** Aritra Dutta; Somak Aditya
>
> **备注:** Preprint ver. 2
>
> **摘要:** Vision-language models (VLMs) increasingly rely on chain-of-thought (CoT) reasoning to solve complex multimodal tasks, but their large parameter sizes make deployment expensive. Structured pruning offers a natural solution; however, existing methods fail to preserve CoT reasoning accuracy in VLMs. We identify two key reasons: (1) CoT consistency depends on sparse transition points (pivot tokens) in the generation trajectory, while existing pruning methods are CoT-agnostic; and (2) pruning methods designed for unimodal LLMs do not account for activation-distribution differences across visual and textual modalities. Motivated by these observations, we propose MuCRASP, a structured pruning framework that targets reasoning-critical components while preserving cross-modal alignment and accounting for layer-wise sensitivity under a global parameter budget. Experiments on four VLMs across three reasoning benchmarks show that MuCRASP consistently preserves reasoning quality under increasing compression. At 30% pruning on Qwen2.5-VL-7B, MuCRASP achieves an LLM-as-a-Judge score of 8.87 versus 7.32 for the strongest baseline on physical reasoning tasks. Furthermore, MuCRASP maintains high reasoning consistency up to 50% pruning, significantly outperforming prior pruning approaches while exhibiting lower perplexity degradation.
>
---
#### [replaced 053] Discovering Differences in Strategic Behavior Between Humans and LLMs
- **分类: cs.AI; cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于行为分析任务，旨在理解人类与LLMs在策略行为上的差异。通过AlphaEvolve工具，发现LLMs可能具备比人类更深的策略行为。**

- **链接: [https://arxiv.org/pdf/2602.10324](https://arxiv.org/pdf/2602.10324)**

> **作者:** Caroline Wang; Daniel Kasenberg; Kim Stachenfeld; Pablo Samuel Castro
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** As Large Language Models (LLMs) are increasingly deployed in social and strategic scenarios, it becomes critical to understand where and why their behavior diverges from that of humans. While behavioral game theory (BGT) provides a framework for analyzing behavior, existing models do not fully capture the idiosyncratic behavior of humans or black-box, non-human agents like LLMs. We employ AlphaEvolve, a cutting-edge program discovery tool, to directly discover interpretable models of human and LLM behavior from data, thereby enabling open-ended discovery of structural factors driving human and LLM behavior. Our analysis on iterated rock-paper-scissors reveals that frontier LLMs can be capable of deeper strategic behavior than humans. These results provide a foundation for understanding structural differences driving differences in human and LLM behavior in strategic interactions.
>
---
#### [replaced 054] Goldfish: Monolingual Language Models for 350 Languages
- **分类: cs.CL**

- **简介: 该论文属于语言建模任务，旨在解决低资源语言模型性能不足的问题。通过训练小型单语模型，提升语法生成效果，并发布Goldfish模型集以促进后续研究。**

- **链接: [https://arxiv.org/pdf/2408.10441](https://arxiv.org/pdf/2408.10441)**

> **作者:** Tyler A. Chang; Catherine Arnett; Zhuowen Tu; Benjamin K. Bergen
>
> **备注:** LREC 2026
>
> **摘要:** For many low-resource languages, the only available language models are large multilingual models trained on many languages simultaneously. Despite state-of-the-art performance on reasoning tasks, we find that these models still struggle with basic grammatical text generation in many languages. First, large multilingual models perform worse than bigrams for many languages (e.g. 24% of languages in XGLM 4.5B; 43% in BLOOM 7.1B) using FLORES perplexity as an evaluation metric. Second, when we train small monolingual models with only 125M parameters on 1GB or less data for 350 languages, these small models outperform large multilingual models both in perplexity and on a massively multilingual grammaticality benchmark. To facilitate future work on low-resource language modeling, we release Goldfish, a suite of over 1,000 small monolingual language models trained comparably for 350 languages. These models represent the first publicly-available monolingual language models for 215 of the languages included.
>
---
#### [replaced 055] Why Don't You Know? Evaluating the Impact of Uncertainty Sources on Uncertainty Quantification in LLMs
- **分类: cs.CL**

- **简介: 该论文属于不确定性量化任务，旨在解决LLMs中不同不确定性来源对UQ方法影响的问题。通过构建新数据集，分析现有方法在不同不确定性源下的表现。**

- **链接: [https://arxiv.org/pdf/2604.10495](https://arxiv.org/pdf/2604.10495)**

> **作者:** Maiya Goloburda; Roman Vashurin; Fedor Chernogorskii; Nurkhan Laiyk; Daniil Orel; Preslav Nakov; Maxim Panov
>
> **摘要:** As Large Language Models (LLMs) are increasingly deployed in real-world applications, reliable uncertainty quantification (UQ) becomes critical for safe and effective use. Most existing UQ approaches for language models aim to produce a single confidence score -- for example, estimating the probability that a model's answer is correct. However, uncertainty in natural language tasks arises from multiple distinct sources, including model knowledge gaps, output variability, and input ambiguity, which have different implications for system behavior and user interaction. In this work, we study how the source of uncertainty impacts the behavior and effectiveness of existing UQ methods. To enable controlled analysis, we introduce a new dataset that explicitly categorizes uncertainty sources, allowing systematic evaluation of UQ performance under each condition. Our experiments reveal that while many UQ methods perform well when uncertainty stems solely from model knowledge limitations, their performance degrades or becomes misleading when other sources are introduced. These findings highlight the need for uncertainty-aware methods that explicitly account for the source of uncertainty in large language models.
>
---
#### [replaced 056] Bottom-up Policy Optimization: Your Language Model Policy Secretly Contains Internal Policies
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM内部策略机制被忽视的问题。通过分解LLM策略，提出BuPO方法优化底层推理结构，提升复杂任务表现。**

- **链接: [https://arxiv.org/pdf/2512.19673](https://arxiv.org/pdf/2512.19673)**

> **作者:** Yuqiao Tan; Minzheng Wang; Shizhu He; Huanxuan Liao; Chengfeng Zhao; Qiunan Lu; Tian Liang; Jun Zhao; Kang Liu
>
> **备注:** Preprint. Our code is available at this https URL
>
> **摘要:** Existing reinforcement learning (RL) approaches treat large language models (LLMs) as a unified policy, overlooking their internal mechanisms. In this paper, we decompose the LLM-based policy into Internal Layer Policies and Internal Modular Policies via the Transformer's residual stream. Our entropy analysis of internal policy reveals distinct patterns: (1) universally, internal policies evolve from high-entropy exploration in early layers to deterministic refinement in the top layers; and (2) Qwen exhibits an explicit progressive reasoning structure, contrasting with the abrupt convergence in Llama. Furthermore, we discover that optimizing internal layers induces feature refinement, forcing lower layers to capture high-level reasoning representations early. Motivated by these findings, we propose Bottom-up Policy Optimization (BuPO), a novel RL paradigm that reconstructs the LLM's reasoning foundation from the bottom up by optimizing internal layers in early stages. Extensive experiments on complex reasoning benchmarks demonstrate the effectiveness of BuPO.
>
---
#### [replaced 057] Distilling Counterfactual Reasoning from Language to Vision: Causal Graph Guided Post-Training for Video Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视频理解任务，旨在解决模型在反事实推理上的不足。通过构建基准数据集CounterVQA和提出CFGPT方法，提升模型的因果推理能力。**

- **链接: [https://arxiv.org/pdf/2511.19923](https://arxiv.org/pdf/2511.19923)**

> **作者:** Yuefei Chen; Jiang Liu; Xiaodong Lin; Ruixiang Tang
>
> **摘要:** Vision Language Models (VLMs) have recently shown significant advancements in video understanding, especially in feature alignment, event reasoning, and instruction-following tasks. However, their capability for counterfactual reasoning, inferring alternative outcomes under hypothetical conditions, remains underexplored. This capability is essential for robust video understanding, as it requires identifying underlying causal structures and reasoning about unobserved possibilities, rather than merely recognizing observed patterns. To systematically evaluate this capability, we introduce CounterVQA, a video-based benchmark featuring three progressive difficulty levels that assess different aspects of counterfactual reasoning. Through comprehensive evaluation of both state-of-the-art open-source and closed-source models, we uncover a substantial performance gap: while these models achieve reasonable accuracy on simple counterfactual questions, performance degrades significantly on complex multi-hop causal chains. To address these limitations, we develop a post-training method, CFGPT, that enhances a model's visual counterfactual reasoning ability by distilling its counterfactual reasoning capability from the language modality, yielding consistent improvements across all CounterVQA difficulty levels. Dataset and code will be further released.
>
---
#### [replaced 058] SERA: Soft-Verified Efficient Repository Agents
- **分类: cs.CL; cs.LG; cs.SE**

- **简介: 该论文提出SERA，解决编码代理在私有代码库上高效训练的问题。通过软验证生成技术，实现低成本、高性能的开放源码代理模型。**

- **链接: [https://arxiv.org/pdf/2601.20789](https://arxiv.org/pdf/2601.20789)**

> **作者:** Ethan Shen; Daniel Tormoen; Saurabh Shah; Ali Farhadi; Tim Dettmers
>
> **备注:** 21 main pages, 6 pages appendix
>
> **摘要:** Open-weight coding agents should hold a fundamental advantage over closed-source systems because they can specialize to private codebases, encoding repository-specific information directly in their weights. Yet the cost and complexity of training has kept this advantage theoretical until now. We present Soft-Verified Efficient Repository Agents (SERA), an efficient method for training coding agents that enables the rapid and cheap creation of agents specialized to private codebases. Using Soft Verified Generation (SVG), we generate thousands of trajectories from any code repository, without requiring unit tests. Beyond repository specialization, we apply SVG to a larger corpus of codebases, generating 200,000+ synthetic trajectories. Using only supervised finetuning (SFT), SERA achieves leading results among fully open-source (open data, method, code) models while matching the performance of open-weight models like Devstral-Small-2. Creating SERA models is 26x cheaper than reinforcement learning and 57x cheaper than previous synthetic data methods to reach equivalent performance. We use our dataset to provide detailed analysis of scaling laws, ablations, and confounding factors for training coding agents. Overall, we believe our work will greatly accelerate research on open coding agents and showcase the advantage of open-source models that can adapt to private codebases. We release SERA as the first model in Ai2's Open Coding Agents series, along with all our code, data, and Claude Code integration to support the research community.
>
---
#### [replaced 059] Human-Alignment and Calibration of Inference-Time Uncertainty in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型不确定性评估任务，旨在解决大语言模型在推理时的不确定性校准与人类不确定性对齐问题。工作包括评估多种不确定性度量方法，验证其与人类不确定性的匹配程度及模型校准效果。**

- **链接: [https://arxiv.org/pdf/2508.08204](https://arxiv.org/pdf/2508.08204)**

> **作者:** Kyle Moore; Jesse Roberts; Daryl Watson
>
> **备注:** We have discovered a critical error in the normalized entropy calculation that may have substantially inflated nearly all results herein. We have since fixed this error in a new work, but we believe that the new work is sufficiently dissimilar in focus, methods, dataset, and results as to be misleading if presented as a simple replacement. As such, we propose removal and retraction instead
>
> **摘要:** There has been much recent interest in evaluating large language models for uncertainty calibration to facilitate model control and modulate user trust. Inference time uncertainty, which may provide a real-time signal to the model or external control modules, is particularly important for applying these concepts to improve LLM-user experience in practice. While many of the existing papers consider model calibration, comparatively little work has sought to evaluate how closely model uncertainty aligns to human uncertainty. In this work, we evaluate a collection of inference-time uncertainty measures, using both established metrics and novel variations, to determine how closely they align with both human group-level uncertainty and traditional notions of model calibration. We find that numerous measures show evidence of strong alignment to human uncertainty, even despite the lack of alignment to human answer preference. For those successful metrics, we find moderate to strong evidence of model calibration in terms of both correctness correlation and distributional analysis.
>
---
#### [replaced 060] Much of Geospatial Web Search Is Beyond Traditional GIS
- **分类: cs.IR; cs.AI; cs.CL; cs.HC**

- **简介: 该论文研究地理空间网络搜索问题，旨在揭示传统GIS无法覆盖的搜索需求。通过分析大量真实查询，识别出大量非传统地理标签的地理相关搜索，并构建分类体系。**

- **链接: [https://arxiv.org/pdf/2605.11336](https://arxiv.org/pdf/2605.11336)**

> **作者:** Ilya Ilyankou; Stefano Cavazzi; James Haworth
>
> **摘要:** Web search queries concern place far more often than existing labelling schemes suggest, yet the landscape of geospatial web search queries - what people ask of place, and how often - remains poorly characterised at scale. We apply dense sentence embeddings, a lightweight SetFit classifier, and density-based clustering to the full MS MARCO corpus of 1.01 million real Bing queries without prior filtering for toponyms or spatial keywords, identifying 181,827 geospatial queries (18.0%), nearly threefold the 6.17% labelled as Location in the original annotations. The resulting taxonomy of 88 query categories reveals that geospatial web search is dominated by transactional and practical lookups: costs and prices alone account for 15.3% of geospatial queries, nearly twice the size of the entire physical geography theme. Much of this activity - costs, opening hours, contact details, weather, travel recommendations - falls outside the scope of what traditional GIS and knowledge graphs are built to serve. The categories vary substantially in the kind of answer they admit, from deterministic lookups answerable from spatial databases or knowledge graphs to evaluative or temporally volatile queries that require generative or real-time systems. We discuss implications for hybrid retrieval architectures and for benchmarks of geographic reasoning in large language models. We openly release the labelled dataset, classifier, and taxonomy.
>
---
#### [replaced 061] SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗问答任务，旨在解决RAG框架在临床推理中的不足。通过引入多智能体协作机制，提升问答的准确性和可靠性。**

- **链接: [https://arxiv.org/pdf/2605.17101](https://arxiv.org/pdf/2605.17101)**

> **作者:** Yongfeng Huang; Ruiying Chen; James Cheng
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** Retrieval-Augmented Generation (RAG) is widely employed to mitigate risks such as hallucinations and knowledge obsolescence in medical question answering, yet its predominantly single-round, static retrieval paradigm misaligns with the multi-stage process of clinical reasoning. This compressed workflow induces two structural deficiencies: question-to-query translation often lacks clinically grounded semantic interpretation, and retrieval lacks iterative sufficiency feedback, making it difficult to form reliable evidence chains. We argue that both issues stem from a deeper cause: overloading a single reasoning chain with heterogeneous tasks of interpretation, exploration, and adjudication. The remedy is to reconstruct the workflow via task decoupling and dynamic multi-round exploration. To this end, we propose SEMA-RAG, a Self-Evolving Multi-Agent RAG framework for medical question answering, which assigns these roles to three specialist agents: the Interpreter Agent for clinical schema interpretation, the Explorer Agent for sufficiency-driven self-evolving retrieval, and the Arbiter Agent for evidence adjudication and answer selection. Across five benchmarks and five LLM backbones, SEMA-RAG improves the strongest baseline by +6.46 accuracy points on average, measured per backbone.
>
---
#### [replaced 062] Memory-Efficient Structured Backpropagation for On-Device LLM Fine-Tuning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大语言模型的设备端微调任务，旨在解决移动设备内存不足的问题。通过结构化反向传播方法，实现高效内存使用与精确梯度计算。**

- **链接: [https://arxiv.org/pdf/2602.13069](https://arxiv.org/pdf/2602.13069)**

> **作者:** Juneyoung Park; Yuri Hong; Seongwan Kim; Jaeho Lee
>
> **备注:** ACL2026
>
> **摘要:** On-device fine-tuning enables privacy-preserving personalization of large language models, but mobile devices impose severe memory constraints, typically 6--12GB shared across all workloads. Existing approaches force a trade-off between exact gradients with high memory (MeBP) and low memory with noisy estimates (MeZO). We propose Memory-efficient Structured Backpropagation (MeSP), which bridges this gap by manually deriving backward passes that exploit LoRA's low-rank structure. Our key insight is that the intermediate projection $h = xA$ can be recomputed during backward at minimal cost since rank $r \ll d_{in}$, eliminating the need to store it. MeSP achieves 49\% average memory reduction compared to MeBP on Qwen2.5 models (0.5B--3B) while computing mathematically identical gradients. Our analysis also reveals that MeZO's gradient estimates show near-zero correlation with true gradients (cosine similarity $\approx$0.001), explaining its slow convergence. MeSP reduces peak memory from 361MB to 136MB for Qwen2.5-0.5B, enabling fine-tuning scenarios previously infeasible on memory-constrained devices.
>
---
#### [replaced 063] Breaking Information Cocoons: A Hyperbolic Framework for Balancing Exploration and Exploitation in Recommender Systems
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于推荐系统任务，旨在解决信息茧房问题。通过提出HERec框架，结合超球空间建模，平衡内容探索与利用，提升推荐多样性与准确性。**

- **链接: [https://arxiv.org/pdf/2411.13865](https://arxiv.org/pdf/2411.13865)**

> **作者:** Qiyao Ma; Menglin Yang; Mingxuan Ju; Tong Zhao; Neil Shah; Rex Ying
>
> **备注:** Accepted to KDD 2026. Code: this https URL
>
> **摘要:** Modern recommender systems often create information cocoons, restricting users' exposure to diverse content. The central challenge is to balance content exploration and exploitation while allowing users to adjust their recommendation preferences. Ideally, this balance can be captured with a hierarchical representation, where depth search facilitates exploitation and breadth search enables exploration. However, existing approaches face two fundamental limitations: Euclidean methods struggle to capture hierarchical structures, while hyperbolic methods, despite their superior hierarchical modeling, lack semantic understanding of user and item profiles and fail to provide a principled mechanism for balancing exploration and exploitation. To address these challenges, we propose HERec, a hyperbolic framework that effectively balances exploration and exploitation in recommender systems. Our framework introduces two key innovations: (1) a semantic-enhanced hierarchical mechanism that aligns rich textual descriptions with collaborative information directly in hyperbolic space. Theoretical gradient analysis demonstrates that this alignment effectively leverages the underlying hyperbolic manifold structure, resulting in more accurate modeling of users and items; (2) an automatic hierarchical clustering mechanism by optimizing Dasgupta's cost, which discovers hierarchical structures without requiring predefined hyperparameters, enabling user-adjustable exploration-exploitation trade-offs. Extensive experiments demonstrate that HERec consistently outperforms both Euclidean and hyperbolic baselines, achieving up to 5.49% improvement in utility metrics and 11.39% increase in diversity metrics, effectively mitigating information cocoons.
>
---
#### [replaced 064] SafeRx-Agent: A Knowledge-Grounded Multi-Agent Framework for Safe and Explainable Medication Recommendation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于药物推荐任务，旨在解决传统方法证据不足和大类推荐导致风险误判的问题。提出SafeRx-Agent框架，结合临床知识与安全验证，实现精准可解释的用药推荐。**

- **链接: [https://arxiv.org/pdf/2605.29146](https://arxiv.org/pdf/2605.29146)**

> **作者:** Xinyu Wang; Hanwei Wu; Zhenghan Tai; Sicheng Lyu; Qincheng Lu; Ziyu Zhao; Jijun Chi; Jingrui Tian; Xiao-Wen Chang; Ziyang Song
>
> **摘要:** Medication recommendation predicts medications for patient visits, but existing methods still face two key challenges. At the model level, traditional drug recommendation methods only predict structured drug codes with limited evidence grounding, while LLM agents can use richer clinical context but may lack safety verification and traceability. At the task level, existing benchmarks often use broad medication categories, which ignore subgroup-level safety differences and can lead to risk overestimation. We introduce the first fine-grained medication recommendation setting based on fourth-level ATC code generation. We propose Safe Prescription Agent (SafeRx-Agent), a knowledge-grounded multi-agent framework that uses patient context, external clinical knowledge, and safety verification to recommend traceable medication sets. Experimental results on MIMIC-III and MIMIC-IV datasets show that SafeRx-Agent improves fine-grained medication prediction accuracy while controlling drug interactions, contraindications, and medication set size.
>
---
#### [replaced 065] Learning to Reason with Insight for Informal Theorem Proving
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于数学定理证明任务，旨在解决非正式证明中缺乏洞察力的问题。提出DeepInsight框架，通过结构化数据和策略提升模型的推理能力。**

- **链接: [https://arxiv.org/pdf/2604.16278](https://arxiv.org/pdf/2604.16278)**

> **作者:** Yunhe Li; Hao Shi; Bowen Deng; Wei Wang; Mengzhe Ruan; Hanxu Hou; Zhongxiang Dai; Siyang Gao; Chao Wang; Shuang Qiu; Linqi Song
>
> **摘要:** Although most of the automated theorem-proving approaches depend on formal proof systems, informal theorem proving can align better with large language models' (LLMs) strength in natural language processing. In this work, we identify a primary bottleneck in informal theorem proving as a lack of insight, namely the difficulty of recognizing the core techniques required to solve complex problems. To address this, we propose $\texttt{DeepInsight}$, a unified training framework designed to cultivate this essential reasoning skill and enable LLMs to perform insightful reasoning. Our framework consists of three components: (1) $\texttt{DeepInsightTheorem}$, a hierarchical dataset that structures informal proofs by explicitly extracting core techniques and proof sketches alongside the final proof; (2) a Progressive Multi-Stage SFT strategy that mimics the human learning process, teaching the model proof writing, planning, and insight identification; and (3) $\texttt{InsightPO}$, a policy optimization method that assigns structured rewards over this insight hierarchy. Our experiments on challenging mathematical benchmarks demonstrate that this insight-aware generation strategy significantly outperforms baselines. These results demonstrate that teaching models to identify and apply core techniques can substantially improve their mathematical reasoning.
>
---
#### [replaced 066] The Need for an External Observer Formalizing the Sufficiency Gap: A Mathematical Extension of Mixture Identifiability and Contextual Grounding in Sequence Models
- **分类: cs.CL; cs.LG**

- **简介: 论文研究序列模型中的上下文缺失问题，提出通过外部观察者减少信息不足的差距。属于序列建模任务，解决模型在缺乏完整上下文时的过自信问题，通过引入辅助信号进行修正。**

- **链接: [https://arxiv.org/pdf/2605.26711](https://arxiv.org/pdf/2605.26711)**

> **作者:** Francesco Corielli
>
> **摘要:** We construct a binary mixed-regime process with one deterministic textual regime and one random regime governed by an unobserved latent state. Even an ideal infinite-capacity sequence predictor that exactly recovers the text-only marginal law can become overconfident when the observed prefix is compatible with the wrong latent regime. The resulting entropy difference is not an ordinary optimization error; it is a sufficiency gap caused by marginalization over an unobserved state. We then formalize retrieval, tool use, and external grounding through an auxiliary binary signal with fidelity $\gamma \in [1/2,1]$. The resulting Bayesian update yields a contextual dominance threshold: a corrective signal reverses the posterior odds induced by the textual history exactly when its fidelity exceeds the text-only posterior weight assigned to the misleading regime. This threshold reduces, but does not generally eliminate, the sufficiency gap; complete closure requires perfect revelation of the relevant latent state or an equivalent verification mechanism. The analysis clarifies why temperature scaling cannot restore missing context, why grounding mechanisms must be both informative and learnably usable by the model, and why autonomous sequence models require structurally decoupled observers or verifiers in high-stakes domains.
>
---
#### [replaced 067] ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs
- **分类: cs.LG; cs.CL; cs.DB**

- **简介: 该论文属于长文本生成任务，解决KV缓存检索的分布偏移和高延迟问题。提出ParisKV框架，通过碰撞选择和量化重排序提升效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.07721](https://arxiv.org/pdf/2602.07721)**

> **作者:** Yanlin Qi; Xinhang Chen; Huiqiang Jiang; Qitong Wang; Botao Peng; Themis Palpanas
>
> **备注:** Accepted to the 43rd International Conference on Machine Learning (ICML 2026)
>
> **摘要:** KV-cache retrieval is essential for long-context LLM inference, yet existing methods struggle with distribution drift and high latency at scale. We introduce ParisKV, a drift-robust, GPU-native KV-cache retrieval framework based on collision-based candidate selection, followed by a quantized inner-product reranking estimator. For million-token contexts, ParisKV supports CPU-offloaded KV caches via Unified Virtual Addressing (UVA), enabling on-demand top-$k$ fetching with minimal overhead. ParisKV matches or outperforms full attention quality on long-input and long-generation benchmarks. It achieves state-of-the-art long-context decoding efficiency: it matches or exceeds full attention speed even at batch size 1 for long contexts, delivers up to 2.8$\times$ higher throughput within full attention's runnable range, and scales to million-token contexts where full attention runs out of memory. At million-token scale, ParisKV reduces decode latency by 17$\times$ and 44$\times$ compared to MagicPIG and PQCache, respectively, two state-of-the-art KV-cache Top-$k$ retrieval baselines, code is available at this https URL.
>
---
#### [replaced 068] Query-focused and Memory-aware Reranker for Long Context Processing
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在提升长文本排序效果。提出一种轻量级重排序框架，利用注意力得分估计相关性，无需标注数据，有效提升多个领域性能。**

- **链接: [https://arxiv.org/pdf/2602.12192](https://arxiv.org/pdf/2602.12192)**

> **作者:** Yuqing Li; Jiangnan Li; Mo Yu; Guoxuan Ding; Yanyu Chen; Zheng Lin; Wei Zhang; Jie Zhou
>
> **备注:** Add new experiments and compare more baselines
>
> **摘要:** Built upon the existing analysis of retrieval heads in large language models, we propose an alternative reranking framework that trains models to estimate passage-query relevance using the attention scores of selected heads. This approach provides a listwise solution that leverages the holistic information within the entire candidate shortlist during ranking. At the same time, it naturally produces continuous relevance scores, enabling training on arbitrary retrieval datasets without requiring Likert-scale supervision. Our framework is lightweight and effective, requiring only small-scale models, such as 3B parameters, to achieve strong performance. Extensive experiments demonstrate that our method outperforms existing state-of-the-art pointwise and listwise rerankers across multiple domains, including Wikipedia and long narrative datasets. It further establishes a new state-of-the-art on the LoCoMo benchmark, which assesses dialogue understanding and memory usage. We further demonstrate that our framework supports flexible extensions. For example, augmenting candidate passages with contextual information further improves ranking accuracy, while training attention heads from middle layers enhances efficiency without sacrificing performance.
>
---
#### [replaced 069] Rethinking Sparse Mixture of Experts from a Unified Perspective
- **分类: cs.CL**

- **简介: 该论文研究稀疏专家混合模型（SMoE），旨在解决固定预算导致的低效分配问题。提出统一框架USMoE，提升模型性能并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2503.22996](https://arxiv.org/pdf/2503.22996)**

> **作者:** Giang Do; Hung Le; Truyen Tran
>
> **备注:** 35 pages
>
> **摘要:** Sparse Mixture of Experts (SMoE) models scale the capacity of models while maintaining constant computational overhead. SMoE methods fall into two categories: Token Choice, which routes each token to a fixed number of experts, and Expert Choice, which assigns a fixed number of tokens to each expert. However, the use of fixed budgets for tokens or experts causes both approaches to select irrelevant token-expert pairs or overlook critical assignments, which degrades overall performance. To fill that gap, we rethink SMoE from a unified perspective through the lens of linear programming, which provides a general formulation for SMoE models. Furthermore, we introduce Unified Sparse Mixture of Experts (USMoE), a novel framework comprising a unified mechanism and a unified score to overcome these limitations. We provide both theoretical justification and empirical evidence demonstrating USMoE's effectiveness. Extensive evaluations across diverse data settings (clean and corrupted), multiple domains (including texts and vision tasks), and different learning approaches (training-free and training-based) show that USMoE not only delivers significant performance improvements over existing SMoE methods, but also enables more flexible expert selection budgets, reducing inference costs without compromising model performance. Our implementation is publicly available at this https URL.
>
---
#### [replaced 070] From Leaky Thoughts to Private Reasoning: Controlling What LRMs Say to Themselves
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于隐私保护任务，旨在解决LRMs推理过程中的敏感信息泄露问题。通过提升模型对指令的遵循能力，改进推理过程的可控性，以增强隐私安全性。**

- **链接: [https://arxiv.org/pdf/2602.24210](https://arxiv.org/pdf/2602.24210)**

> **作者:** Haritz Puerto; Haonan Li; Xudong Han; Timothy Baldwin; Iryna Gurevych
>
> **摘要:** Large reasoning models (LRMs) produce reasoning traces (RTs) that often contain sensitive information. These leaky thoughts are difficult to control and frequently violate explicit privacy directives. Because RTs can be exposed through prompt injection attacks, this becomes a direct privacy risk to the user. We approach this as a controllability problem: since privacy directives are themselves instructions, improving instruction-following (IF) within the RT provides a direct path to reducing privacy leaks. To this end, we introduce an SFT dataset that teaches models to follow general instructions throughout their reasoning process, and propose Staged Decoding, a simple decoding strategy that decouples RT and answer generation using separate LoRA adapters to maximize IF of each component. We evaluate our approach on six models from two families (1.7B-14B parameters), across two IF benchmarks and two privacy benchmarks. Our method yields substantial improvements, with gains of up to 20.9 points in IF and 51.9 percentage points on privacy benchmarks, though these can come at the cost of task utility due to the trade-off between reasoning performance and IF. Our results show that improving IF in LRMs can significantly enhance privacy, suggesting a promising direction for future privacy-aware LRMs. Our code is available at this https URL.
>
---
#### [replaced 071] Deterministic Inference across Tensor Parallel Sizes That Eliminates Training-Inference Mismatch
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文属于大语言模型推理任务，解决训练与推理间因张量并行尺寸不同导致的非确定性问题。提出TBIK方法，实现跨TP尺寸的确定性推理。**

- **链接: [https://arxiv.org/pdf/2511.17826](https://arxiv.org/pdf/2511.17826)**

> **作者:** Ziyang Zhang; Xinheng Ding; Jiayi Yuan; Rixin Liu; Huizi Mao; Jiarong Xing; Zirui Liu
>
> **摘要:** Deterministic inference is increasingly critical for large language model (LLM) applications such as LLM-as-a-judge evaluation, multi-agent systems, and Reinforcement Learning (RL). However, existing LLM serving frameworks exhibit non-deterministic behavior: identical inputs can yield different outputs when system configurations (e.g., tensor parallel (TP) size, batch size) vary, even under greedy decoding. This arises from the non-associativity of floating-point arithmetic and inconsistent reduction orders across GPUs. While prior work has addressed batch-size-related nondeterminism through batch-invariant kernels, determinism across different TP sizes remains an open problem, particularly in RL settings, where the training engine typically uses Fully Sharded Data Parallel (i.e., TP = 1) while the rollout engine relies on multi-GPU TP to maximize the inference throughput, creating a natural mismatch between the two. This precision mismatch problem may lead to suboptimal performance or even collapse for RL training. We identify and analyze the root causes of TP-induced inconsistency and propose Tree-Based Invariant Kernels (TBIK), a set of TP-invariant matrix multiplication and reduction primitives that guarantee bit-wise identical results regardless of TP size. Our key insight is to align intra- and inter-GPU reduction orders through a unified hierarchical binary tree structure. We implement these kernels in Triton and integrate them into vLLM and FSDP. Experiments confirm zero probability divergence and bit-wise reproducibility for deterministic inference across different TP sizes. Also, we achieve bit-wise identical results between vLLM and FSDP in RL training pipelines with different parallel strategy. Code is available at this https URL.
>
---
#### [replaced 072] When Is Next-Token Prediction Useful? Marginalization, Ergodicity, Mixture Identifiability, Local Sufficiency, RAG, Tools, and Programming
- **分类: cs.CL; stat.ML**

- **简介: 该论文探讨语言模型训练中下一词预测的适用性，区分了条件分布、边缘文本过程和模型分布，分析其在不同假设下的有效性，提出RAG等工具作为条件充分性手段。任务为自然语言处理中的模型理解与优化。**

- **链接: [https://arxiv.org/pdf/2605.23278](https://arxiv.org/pdf/2605.23278)**

> **作者:** Francesco Corielli
>
> **摘要:** Language models trained on observed sequences are often described as learning the conditional distribution of the next token given previous tokens. This description is only conditionally correct. A model trained on realized token trajectories does not observe full conditional laws; it receives sampled continuations. Moreover, real language generation is conditioned not only on previous words but also on non-textual circumstances: facts, events, intentions, goals, beliefs, social context, and task-specific constraints. This paper distinguishes three objects that are often conflated: the full conditional language process conditioned on latent circumstances, the marginal text-only process obtained by integrating those circumstances out, and the model-induced distribution learned from finite observed corpora. The paper argues that interpreting model training as estimating the marginal text-only law requires strong assumptions of stationarity, representativeness, and ergodicity, assumptions that are standard in statistical estimation but problematic when applied to heterogeneous language corpora. Even if these assumptions hold, the marginal text-only law is useful only when the observed prefix is an approximately sufficient statistic for the latent circumstances relevant to continuation. In information-theoretic terms, usefulness requires that the residual conditional mutual information between the next token and the omitted circumstances, given the observed text, be small. The paper then extends this argument to heterogeneous training corpora. Finally, the paper interprets Retrieval Augmented Generation (RAG) and tool use as conditional sufficiency devices.
>
---
#### [replaced 073] Graph Machine Learning in the Era of Large Language Models (LLMs)
- **分类: cs.LG; cs.AI; cs.CL; cs.SI**

- **简介: 本文探讨了大语言模型（LLMs）时代下的图机器学习（Graph ML）。论文属于跨领域研究任务，旨在解决图数据处理与LLMs结合中的泛化、可迁移性及少样本学习问题，并探索两者相互增强的潜力。**

- **链接: [https://arxiv.org/pdf/2404.14928](https://arxiv.org/pdf/2404.14928)**

> **作者:** Shijie Wang; Jiani Huang; Zhikai Chen; Yu Song; Wenzhuo Tang; Haitao Mao; Wenqi Fan; Hui Liu; Xiaorui Liu; Dawei Yin; Qing Li
>
> **备注:** Accepted by TIST
>
> **摘要:** Graphs play an important role in representing complex relationships in various domains like social networks, knowledge graphs, and molecular discovery. With the advent of deep learning, Graph Neural Networks (GNNs) have emerged as a cornerstone in Graph Machine Learning (Graph ML), facilitating the representation and processing of graphs. Recently, LLMs have demonstrated unprecedented capabilities in language tasks and are widely adopted in a variety of applications such as computer vision and recommender systems. This remarkable success has also attracted interest in applying LLMs to the graph domain. Increasing efforts have been made to explore the potential of LLMs in advancing Graph ML's generalization, transferability, and few-shot learning ability. Meanwhile, graphs, especially knowledge graphs, are rich in reliable factual knowledge, which can be utilized to enhance the reasoning capabilities of LLMs and potentially alleviate their limitations such as hallucinations and the lack of explainability. Given the rapid progress of this research direction, a systematic review summarizing the latest advancements for Graph ML in the era of LLMs is necessary to provide an in-depth understanding to researchers and practitioners. Therefore, in this survey, we first review the recent developments in Graph ML. We then explore how LLMs can be utilized to enhance the quality of graph features, alleviate the reliance on labeled data, and address challenges such as graph Heterophily and out-of-distribution (OOD) generalization. Afterward, we delve into how graphs can enhance LLMs, highlighting their abilities to enhance LLM pre-training and inference. Furthermore, we investigate various applications and discuss the potential future directions in this promising field.
>
---
#### [replaced 074] Reassessing Extractive QA Datasets at Scale: LLM-as-a-Judge and In-Depth Analyses
- **分类: cs.CL**

- **简介: 该论文属于抽取式问答任务，旨在解决传统评估指标不足的问题。通过LLM-as-a-judge方法进行更准确的模型评估，并分析不同因素对评估结果的影响。**

- **链接: [https://arxiv.org/pdf/2504.11972](https://arxiv.org/pdf/2504.11972)**

> **作者:** Xanh Ho; Jiahao Huang; Florian Boudin; Akiko Aizawa
>
> **备注:** GEM Workshop at ACL 2026; code and data are available at this https URL
>
> **摘要:** Extractive QA tasks are commonly evaluated using Exact Match (EM) and F1-score, but these metrics often fail to reflect true model performance. Recent studies have proposed using large language models (LLMs) as judges (LLM-as-a-judge), yet they often lack comprehensive evaluation across datasets and overlook key factors such as sensitivity to answer types, prompt variations, and self-preference bias. In this work, we conduct a systematic study of LLM-as-a-judge across four extractive QA datasets and various prompt variations, assessing multiple LLM families in both answering and judging roles. Our results show that LLM-as-a-judge judgments correlate much more strongly with human evaluations than EM (0.22) and F1 (0.40), achieving correlations up to 0.85 with open-source models. Further analysis reveals that LLM-as-a-judge performs particularly well on number-related answers but faces challenges with more complex types, such as job titles. Contrary to findings in other NLP tasks, we observe no self-preference bias, even when the same model serves as both QA model and judge. Finally, we find that prompt phrasing has minimal impact, and zero-shot, context-free judging often yields the best evaluation performance.
>
---
#### [replaced 075] Empirical Characterization of Inference-Time Elicited Probability Transformations in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在推理过程中概率变换的结构，分析不同提示配置下的概率关系，旨在理解推理时概率调整的规律。**

- **链接: [https://arxiv.org/pdf/2603.19262](https://arxiv.org/pdf/2603.19262)**

> **作者:** Mike Farmer; Abhinav Kochar; Yugyung Lee
>
> **备注:** 22 pages, 11 figures, 5 tables
>
> **摘要:** Large language models increasingly rely on inference-time procedures such as chain-of-thought reasoning, self-refinement, retrieval augmentation, and verifier-guided revision, yet the structure of elicited probability transformations under these procedures remains poorly understood. We study externally elicited probability assignments over candidate answers and observe recurring approximate log-ratio relationships: \[ \log \tilde q_t(i) = \alpha_t \left( \log q_t(i) + \log b_t(i) \right) + c_t, \] where $q_t$ and $\tilde q_t$ are pre- and post-elicitation probabilities, $b_t$ is an externally constructed evidence signal, and $\alpha_t$ is an empirical descriptor of the prompting configuration. Across 4,975 reasoning problems from GPQA Diamond, TheoremQA, MMLU-Pro, and ARC-Challenge, evaluated on multiple instruction-tuned model families, we observe approximate log-ratio relationships with mean $R^2 \approx 0.76$ over about $1.3 \times 10^5$ candidate-level observations. Coefficients vary across elicitation settings, but qualitatively similar relationships persist across evaluated conditions. Robustness analyses using alternative statistical representations, prompting configurations, held-out evaluation, and token-level log-probabilities suggest that the observed structure is not tied to one prompting procedure or probability estimation method. The main contribution is not the algebraic form itself, which is related to generalized Bayesian updating and probability-transformation frameworks, but the empirical observation that diverse inference-time prompting pipelines repeatedly exhibit reproducible log-ratio structure under controlled conditions. The framework provides a protocol-sensitive perspective for analyzing calibration, evidence amplification, uncertainty propagation, and interaction sensitivity in inference-time LLM pipelines.
>
---
#### [replaced 076] FoRA: Fisher-orthogonal Rank Adaptation for Parameter-Efficient Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文属于参数高效微调任务，旨在减少可训练参数数量。提出FoRA方法，通过选择关键层并约束降维过程，提升效率与效果。**

- **链接: [https://arxiv.org/pdf/2605.29317](https://arxiv.org/pdf/2605.29317)**

> **作者:** Juneyoung Park; Seongbae Lee; Han-Sang Lee; Kyuho Lee; Minjae Kim; Seungheon Hyeon; Kiduk Kwon; Seongwan Kim; Jaeho Lee
>
> **备注:** EMNLP 2026
>
> **摘要:** Parameter-efficient fine-tuning(PEFT) has largely focused on LoRA and its accuracy-oriented variants, leaving the original goal of reducing trainable parameters has receivedcomparatively little attention. We introduce FoRA, which revisits this goal by reducing the number of adapted layers rather than adapter rank. FoRA selects task-informative layers via a single-pass diagonal Fisher score (under 1% of training cost) and trains the LoRA down-projection at selected layers on the Stiefel manifold, preserving column orthonormality and effective rank. FoRA consistently outperforms LoRA and DoRA at half their parameter budget, and falls within 0.7-0.8 accuracy points of AdaLoRA at one-quarter its parameter count, across five LLaMA-family backbones. Cross-architecture experiments on twelve backbones from the LLaMA, Qwen3, and Gemma families confirm consistent gains from 270M to 32B parameters. The two components combine super-additively: Fisher selection alone matches rank reduction at the same budget, while the Stiefel constraint provides the decisive additional gain.
>
---
#### [replaced 077] Towards Atoms of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Atom Theory，旨在定义和识别大语言模型的基本表示单元（atoms），解决其内部机制理解难题。通过AIP指标评估原子的忠实性与稳定性，验证了理想原子的特性。**

- **链接: [https://arxiv.org/pdf/2509.20784](https://arxiv.org/pdf/2509.20784)**

> **作者:** Chenhui Hu; Pengfei Cao; Yubo Chen; Kang Liu; Jun Zhao
>
> **备注:** To be published in ICML 2026
>
> **摘要:** The fundamental representational units (FRUs) of large language models (LLMs) remain undefined, limiting further understanding of their underlying mechanisms. In this paper, we introduce Atom Theory to systematically define, evaluate, and identify such FRUs, which we term atoms. Building on the atomic inner product (AIP), a non-Euclidean metric that captures the underlying geometry of LLM representations, we formally define atoms and propose two key criteria for ideal atoms: faithfulness ($R^2$) and stability ($q^*$). We further prove that atoms are identifiable under threshold-activated sparse autoencoders (TSAEs). Empirically, we uncover a pervasive representation shift in LLMs and demonstrate that the AIP corrects this shift to capture the underlying representational geometry. We find that two widely used units, neurons and features, fail to qualify as ideal atoms: neurons are faithful ($R^2=1$) but unstable ($q^*=0.5\%$), while features are more stable ($q^*=68.2\%$) but unfaithful ($R^2=48.8\%$). To find atoms of LLMs, leveraging atom identifiability under TSAEs, we show via large-scale experiments that reliable atom identification occurs only when the TSAE capacity matches the data scale. Guided by this insight, we identify FRUs with near-perfect faithfulness ($R^2=99.9\%$) and stability ($q^*=99.8\%$) across layers of Gemma2-2B, Gemma2-9B, and Llama3.1-8B, satisfying the criteria of ideal atoms statistically. Further analysis confirms that these atoms align with theoretical expectations and exhibit substantially higher monosemanticity. Overall, we propose and validate Atom Theory as a foundation for understanding the internal representations of LLMs. Code available at this https URL.
>
---
#### [replaced 078] Reasoning-Intensive Regression
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究推理密集型回归（RiR）任务，旨在从文本中推断数值评分。针对数据和计算有限的场景，提出MENTAT方法，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2508.21762](https://arxiv.org/pdf/2508.21762)**

> **作者:** Diane Tchuindjo; Omar Khattab
>
> **摘要:** AI researchers and practitioners increasingly apply large language models (LLMs) to what we call reasoning-intensive regression (RiR), i.e., deducing subtle numerical scores from text. Unlike standard language regression tasks such as sentiment or similarity analysis, RiR often appears instead in ad-hoc applications such as rubric-based scoring, modeling dense rewards in complex environments, or domain-specific retrieval, where much deeper analysis of context is required while only limited task-specific training data and computation are available. We cast four realistic problems as RiR tasks to establish an initial benchmark, and use that to test our hypothesis that prompting frozen LLMs and fine-tuning Transformer encoders via gradient descent will both often struggle in RiR. We then propose MENTAT, a simple and lightweight method that combines batch-reflective prompt optimization with neural ensemble learning. MENTAT achieves up to 65% improvement over both baselines, though substantial room remains for future advances.
>
---
#### [replaced 079] Beyond Memorization: Assessing Semantic Generalization in Large Language Models Using Phrasal Constructions
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言理解任务，旨在评估大语言模型在语义泛化方面的能力。针对模型在罕见但易懂的语境中表现不佳的问题，构建了基于构式语法的评测数据集，验证模型对同构句式的语义区分能力。**

- **链接: [https://arxiv.org/pdf/2501.04661](https://arxiv.org/pdf/2501.04661)**

> **作者:** Wesley Scivetti; Melissa Torgbi; Austin Blodgett; Mollie Shichman; Taylor Hudson; Claire Bonial; Harish Tayyar Madabushi
>
> **备注:** Camera Ready: AACL-IJCNLP (2025)
>
> **摘要:** The web-scale of pretraining data has created an important evaluation challenge: to disentangle linguistic competence on cases well-represented in pretraining data from generalization to out-of-domain language, specifically the dynamic, real-world instances less common in pretraining data. To this end, we construct a diagnostic evaluation to systematically assess natural language understanding in LLMs by leveraging Construction Grammar (CxG). CxG provides a psycholinguistically grounded framework for testing generalization, as it explicitly links syntactic forms to abstract, non-lexical meanings. Our novel inference evaluation dataset consists of English phrasal constructions, for which speakers are known to be able to abstract over commonplace instantiations in order to understand and produce creative instantiations. Our evaluation dataset uses CxG to evaluate two central questions: first, if models can 'understand' the semantics of sentences for instances that are likely to appear in pretraining data less often, but are intuitive and easy for people to understand. Second, if LLMs can deploy the appropriate constructional semantics given constructions that are syntactically identical but with divergent meanings. Our results demonstrate that state-of-the-art models, including GPT-o1, exhibit a performance drop of over 40% on our second task, revealing a failure to generalize over syntactically identical forms to arrive at distinct constructional meanings in the way humans do. We make our novel dataset and associated experimental data, including prompts and model responses, publicly available.
>
---
#### [replaced 080] Effective Reasoning Chains Reduce Intrinsic Dimensionality
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理领域，研究如何通过有效推理链降低任务内在维度，提升模型泛化能力。工作包括提出内在维度作为量化指标，并验证其与推理策略效果的关联。**

- **链接: [https://arxiv.org/pdf/2602.09276](https://arxiv.org/pdf/2602.09276)**

> **作者:** Archiki Prasad; Mandar Joshi; Kenton Lee; Mohit Bansal; Peter Shaw
>
> **备注:** ICML (spotlight) camera-ready; 22 pages, 3 figures
>
> **摘要:** Chain-of-thought (CoT) reasoning and its variants have substantially improved the performance of language models on complex reasoning tasks, yet the precise mechanisms by which different strategies facilitate generalization remain poorly understood. While current explanations often point to increased test-time computation or structural guidance, establishing a consistent, quantifiable link between these factors and generalization remains challenging. In this work, we identify intrinsic dimensionality as a quantitative measure for characterizing the effectiveness of reasoning chains. Intrinsic dimensionality quantifies the minimum number of model dimensions needed to reach a given accuracy threshold on a given task. By keeping the model architecture fixed and varying the task formulation through different reasoning strategies, we demonstrate that effective reasoning strategies consistently reduce the intrinsic dimensionality of the task. Validating this on GSM8K with Gemma-3 1B and 4B, we observe a strong inverse correlation between the intrinsic dimensionality of a reasoning strategy and its generalization performance on both in-distribution and out-of-distribution data. Our findings suggest that effective reasoning chains facilitate learning by better compressing the task using fewer parameters, offering a new quantitative metric for analyzing reasoning processes.
>
---
#### [replaced 081] Latent Performance Profiling of Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型评估任务，旨在解决传统基准测试无法全面反映模型能力的问题。提出LPP框架，通过分析隐层表示和输出分布，提供更深入的模型性能诊断。**

- **链接: [https://arxiv.org/pdf/2605.30018](https://arxiv.org/pdf/2605.30018)**

> **作者:** Tanmoy Chakraborty; Ayan Sengupta; Suparna Bhattacharya; Partha Pratim Chakrabarti; Amlan Chakrabarti; Supratik Chakraborty; Partha Pratim Das; Lipika Dey; Richa Singh; Mayank Vatsa
>
> **摘要:** Large language models (LLMs) frequently achieve impressive scores on standardized benchmarks, yet accuracy alone offers a limited view of their capabilities. Evaluating open-source LLMs through leaderboards faces persistent issues like data contamination, narrow task scope, and weak alignment with real-world reliability. Benchmark-based evaluations such as MMLU PRO, BBH, or IFEval primarily capture what a model outputs on fixed test sets, not how it processes information, calibrates uncertainty, or structures internal knowledge. In this article, we advocate for a shift from benchmark-centric evaluation toward a complementary, state-centered intrinsic assessment of LLMs. To this end, we introduce Latent Performance Profiling (LPP) -- a framework that derives task-agnostic diagnostics from hidden activations and output distributions. LPP defines a set of scalar metrics on a model's latent representations and dynamics, revealing scale-independent traits that enable interpretable comparisons and uncover hidden vulnerabilities. Unlike static accuracy scores, LPP provides stable, architecture-sensitive signatures across models of similar size. With extensive empirical analyses across eight LLMs, spanning a size range of 0.5B-14B, we demonstrate that models with similar benchmark scores can exhibit contrasting latent profiles, such as differences in entropy or adaptability. Guided by these insights, we design synthetic probes for uncertainty and symbolic reasoning that align with intrinsic metrics while decoupling from leaderboard bias. We recommend that reporting LPP alongside benchmarks provides a deeper, interpretable understanding of model behavior, enabling more reliable model selection, safety assessment, and evaluation beyond surface-level accuracy.
>
---
#### [replaced 082] SCOPE: Selective Conformal Optimized Pairwise LLM Judging
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SCOPE框架，解决LLM在成对评估中校准不足和偏差问题。通过BPE获取无偏不确定性信号，确保判断误差率控制在指定水平，提升评估可靠性与覆盖度。**

- **链接: [https://arxiv.org/pdf/2602.13110](https://arxiv.org/pdf/2602.13110)**

> **作者:** Sher Badshah; Ali Emami; Hassan Sajjad
>
> **备注:** Accepted at ICML 2026. 23 pages (9 main plus appendix), 7 figures, 11 tables
>
> **摘要:** Large language models (LLMs) are increasingly used as scalable judges in pairwise evaluation, but they remain prone to miscalibration and biases. We propose SCOPE (Selective Conformal Optimized Pairwise Evaluation), a framework that calibrates an acceptance threshold so that, under exchangeability, the error rate among non-abstained judgments is at most a user-specified level $\alpha$. To supply SCOPE with a bias-neutral uncertainty signal, we introduce Bidirectional Preference Entropy (BPE), which queries the judge under both response positions and converts the order-averaged preference probability into an entropy-based score. Across various pairwise judging benchmarks, BPE outperforms standard confidence proxies in calibration and discrimination, while SCOPE consistently satisfies the target risk bound (empirical FDR $\approx 0.097$ to $0.099$ at $\alpha = 0.10$) and retains substantial coverage. Compared to vanilla baselines, SCOPE accepts up to $2.4\times$ more judgments under the same risk constraint, demonstrating that BPE enables reliable and high-coverage LLM-based evaluation.
>
---
#### [replaced 083] Draft-OPD: On-Policy Distillation for Speculative Draft Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型加速任务，解决草案模型在推测解码中效果停滞的问题。通过提出Draft-OPD方法，提升草案模型的性能和推理速度。**

- **链接: [https://arxiv.org/pdf/2605.29343](https://arxiv.org/pdf/2605.29343)**

> **作者:** Haodi Lei; Yafu Li; Haoran Zhang; Shunkai Zhang; Qianjia Cheng; Xiaoye Qu; Ganqu Cui; Bowen Zhou; Ning Ding; Yun Luo; Yu Cheng
>
> **摘要:** Speculative decoding accelerates large language model inference by pairing a target model with a lightweight draft model whose proposed tokens are verified in parallel. A common way to build draft models, like EAGLE3 or DFlash is supervised fine-tuning (SFT) on target-generated trajectories. However, we observe that SFT quickly plateaus: the draft model's acceptance length on test data stops improving. The reason is an offline-to-inference mismatch: In SFT, the drafter learns from fixed target-generated trajectories, whereas during speculative decoding it is evaluated on blocks proposed under its own policy. This motivates on-policy distillation (OPD), where the target model supervises the drafter on draft-induced states. Yet OPD remains difficult for draft models, as they cannot reliably roll out complete sequences independently, whereas target-assisted generation makes the collected sequences follow the target distribution and thus eliminates the on-policy signal. We therefore propose Draft-OPD, which uses target-assisted rollout for stable continuations and replays drafting from the verification-exposed error positions. This allows the drafter to learn from target feedback on both accepted and rejected proposals, focusing training on the draft-induced errors that limit speculative acceptance. Experiments show that Draft-OPD achieves over $5\times$ lossless acceleration for thinking models across diverse tasks, improving over EAGLE-3 and DFlash by 23\% and 13\%.
>
---
#### [replaced 084] Beyond Hearing: Learning Task-Agnostic ExG Representations from Earphones via Physiology-Informed Tokenization
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于生理信号处理任务，旨在解决ExG数据多样性不足和模型任务依赖性问题。通过收集自由生活数据并引入PiMT方法，学习通用的ExG表示，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2510.20853](https://arxiv.org/pdf/2510.20853)**

> **作者:** Hyungjun Yoon; Seungjoo Lee; Yu Yvonne Wu; Xiaomeng Chen; Taiting Lu; Freddy Yifei Liu; Taeckyung Lee; Hyeongheon Cha; Haochen Zhao; Gaoteng Zhao; Dongyao Chen; Cecilia Mascolo; Sung-Ju Lee; Lili Qiu
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Electrophysiological (ExG) signals offer valuable insights into human physiology, yet building foundation models that generalize across everyday tasks remains challenging due to two key limitations: (i)~insufficient data diversity, as most ExG recordings are collected in controlled labs with bulky, expensive devices; and (ii)~task-specific model designs that require tailored processing (i.e., targeted frequency filters) and architectures, which limit generalization across tasks. To address these challenges, we introduce an approach for scalable, task-agnostic ExG monitoring in the wild. We collected 50 hours of unobtrusive free-living ExG data with an earphone-based hardware prototype to narrow the data diversity gap. At the core of our approach is Physiology-informed Multi-band Tokenization (PiMT), which decomposes ExG signals into 12 physiology-informed tokens, followed by a reconstruction task to learn robust representations. This enables adaptive feature recognition across the full frequency spectrum while capturing task-relevant information. Experiments on our new DailySense dataset, the first to enable ExG-based analysis across five human senses, together with four public ExG benchmarks, demonstrate that PiMT consistently outperforms state-of-the-art methods across diverse tasks.
>
---
#### [replaced 085] Side-by-side Comparison Amplifies Dialect Bias in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的偏见检测任务，旨在解决语言模型中的隐性方言偏见问题。通过对比标准英语与非裔英语推文，发现模型在对比情境下偏见加剧，提出微调方法部分缓解但效果有限。**

- **链接: [https://arxiv.org/pdf/2605.24384](https://arxiv.org/pdf/2605.24384)**

> **作者:** Kritee Kondapally; Claire J. Smerdon; Pooja C. Patel; Ogheneyoma Akoni; Jevon Torres; Jaspreet Ranjit; Matthew Finlayson; Swabha Swayamdipta
>
> **备注:** In proceeding at ACM Conference on Fairness, Accountability, and Transparency 2026
>
> **摘要:** Language models (LMs) can exhibit biases based on variations in their dialects, even in the absence of a dialect label, a behavior known as covert dialect bias. In this work, we quantify covert dialect bias in online discourse by evaluating how LMs associate stereotypical traits (derived from social psychology research on racial bias) with intent-equivalent tweets in Standard American English (SAE) and African-American Vernacular English (AAVE). While prior work shows that LMs associate more negative stereotypes with AAVE when evaluating tweets in isolation, we are surprised to find that this bias is significantly exacerbated when SAE / AAVE tweet pairs are compared side by side, a setting that more closely reflects high-impact decision making contexts in which models are used to rank candidates. The bias only worsens when dialect labels are explicitly specified. This is striking, given the extensive efforts from commercial developers to mitigate bias in their LMs. Encouragingly, we show that counterfactual fairness finetuning can mitigate covert dialect bias for some stereotypical traits, reducing average disparities when evaluating tweets in isolation, however, these improvements do not consistently hold across traits when evaluating SAE / AAVE tweets side by side. Our findings show that existing evaluation settings for covert dialect bias may underestimate its severity, specifically in contrastive settings. Additionally, overt dialect bias remains pronounced even after safety aligned finetuning, indicating that it remains an unresolved problem, and motivates the need for more robust evaluation and mitigation frameworks.
>
---
#### [replaced 086] PRISM: Self-Pruning Intrinsic Selection Method for Training-Free Multimodal Data Selection
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出PRISM，解决视觉指令微调中的数据选择效率问题。通过去除全局背景特征影响，实现训练-free 的高效数据筛选，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2502.12119](https://arxiv.org/pdf/2502.12119)**

> **作者:** Jinhe Bi; Aniri; Zengjie Jin; Yifan Wang; Danqi Yan; Wenke Huang; Xiaowen Ma; Sikuan Yan; Artur Hecker; Mang Ye; Xun Xiao; Hinrich Schuetze; Volker Tresp; Yunpu Ma
>
> **备注:** Accepted to ACL 2026 and selected for the Best Paper list; later desk-rejected due to an inadvertent manual bibliography-editing error. Previous versions are withdrawn due to an inadvertent manual bibliography-editing error; please refer to the latest corrected version
>
> **摘要:** Visual instruction tuning adapts pre-trained Multimodal Large Language Models (MLLMs) to follow human instructions for real-world applications. However, the rapid growth of these datasets introduces significant redundancy, leading to increased computational costs. Existing methods for selecting instruction data aim to prune this redundancy, but predominantly rely on computationally demanding techniques such as proxy-based inference or training-based metrics. Consequently, the substantial computational costs incurred by these selection processes often exacerbate the very efficiency bottlenecks they are intended to resolve, posing a significant challenge to the scalable and effective tuning of MLLMs. To address this challenge, we first identify a critical, yet previously overlooked, factor: the anisotropy inherent in visual feature distributions. We find that this anisotropy induces a \textit{Global Semantic Drift}, and overlooking this phenomenon is a key factor limiting the efficiency of current data selection methods. Motivated by this insight, we devise \textbf{PRISM}, the first training-free framework for efficient visual instruction selection. PRISM surgically removes the corrupting influence of global background features by modeling the intrinsic visual semantics via implicit re-centering. Empirically, PRISM reduces the end-to-end time for data selection and model tuning to just 30\% of conventional pipelines. More remarkably, it achieves this efficiency while simultaneously enhancing performance, surpassing models fine-tuned on the full dataset across eight multimodal and three language understanding benchmarks, culminating in a 101.7\% relative improvement over the baseline. The code is available for access via \href{this https URL}{this repository}.
>
---
#### [replaced 087] Human Psychometric Questionnaires Mischaracterize LLM Behavior
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，探讨心理测量问卷在描述和预测大模型行为上的有效性。研究发现，传统问卷无法准确反映模型在真实对话中的表现，提出基于生成的评估方法更可靠。**

- **链接: [https://arxiv.org/pdf/2509.10078](https://arxiv.org/pdf/2509.10078)**

> **作者:** Woojung Song; Dongmin Choi; Yoonah Park; Jongwook Han; Eun-Ju Lee; Yohan Jo
>
> **备注:** 38 pages, 6 figures
>
> **摘要:** We examine whether human psychometric questionnaires can serve as reliable tools for characterizing and predicting LLM behavior in everyday user interactions. We analyze eight open-source LLMs by comparing their value and personality profiles derived from two different methods: Likert self-reports on established questionnaires (PVQ-40/21 and BFI-44/10) and generation probabilities over value-laden responses to everyday user queries. The two profiles diverge substantially. Within-construct item consistency, often cited as evidence of stable LLM dispositions, disappears in generation probabilities. We attribute this gap to the fact that explicit lexical cues in established questionnaire items allow models to recognize the target construct and respond in alignment-consistent, socially desirable ways, whereas realistic user queries provide no such cues. In addition, demographic persona prompts shift models' responses to human questionnaires in ways consistent with real human patterns, but no such shifts appear in the generation probabilities of responses to realistic user queries, showing their limited ability to simulate the behaviors of target demographics in real-world user interactions. Overall, our study shows that human psychometric questionnaires are insufficient tools for predicting LLM behavior and suggests generation-based profiling as a more accurate measure.
>
---
#### [replaced 088] Boundary-Guided Policy Optimization for Memory-efficient RL of Diffusion Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，针对扩散大语言模型的似然函数难以计算的问题，提出BGPO算法，通过构造线性下界实现高效记忆优化的RL训练。**

- **链接: [https://arxiv.org/pdf/2510.11683](https://arxiv.org/pdf/2510.11683)**

> **作者:** Nianyi Lin; Jiajie Zhang; Lei Hou; Juanzi Li
>
> **摘要:** A key challenge in applying reinforcement learning (RL) to diffusion large language models (dLLMs) is the intractability of their likelihood functions, which are essential for the RL objective, necessitating corresponding approximation during training. While existing methods approximate the log-likelihoods by their evidence lower bounds (ELBOs) via customized Monte Carlo (MC) sampling, they incur significant memory overhead due to the need to retain all MC samples for the gradient computation of non-linear terms in the RL objective, and thus restrict feasible sample sizes, leading to imprecise likelihood approximations and distorted RL objective. To address this, we propose \emph{Boundary-Guided Policy Optimization} (BGPO), a memory-efficient RL algorithm that maximizes a specially constructed lower bound of the ELBO-based objective. This lower bound is carefully designed to satisfy two key properties: (1) Linearity: it is a linear sum where each term depends only on a single MC sample, thereby enabling gradient accumulation across samples and ensuring constant memory usage; (2) Equivalence: Both the value and gradient of this lower bound are equal to those of the ELBO-based objective in on-policy training, making it also an effective approximation for the original RL objective. These properties allow BGPO to adopt a large MC sample size, improving likelihood approximations and RL objective estimation, which in turn leads to enhanced performance. Experiments show that BGPO significantly outperforms previous RL algorithms for dLLMs in math problem solving, code generation, and planning tasks. Our codes and models are available at \href{this https URL}{this https URL}.
>
---
#### [replaced 089] LocalSUG: City-Preference-Enhanced LLM for Query Suggestion in Local-Life Services
- **分类: cs.CL**

- **简介: 该论文属于查询建议任务，解决本地生活服务中传统系统无法捕捉长尾需求的问题。提出LocalSUG框架，增强城市偏好感知，提升建议效果。**

- **链接: [https://arxiv.org/pdf/2603.04946](https://arxiv.org/pdf/2603.04946)**

> **作者:** Jinwen Chen; Shiwen Zhang; Shuai Gong; Zheng Zhang; Yachao Zhao; Lingxiang Wang; Haibo Zhou; Wei Lin; Hainan Zhang
>
> **摘要:** In local-life service platforms, query suggestion reduces user effort by generating candidate queries from input prefixes. Traditional multi-stage systems rely heavily on historical popular queries, limiting their ability to capture long-tail and emerging demand. Although LLMs provide strong semantic generalization, their deployment in local-life services faces three challenges: insufficient city-preference awareness, exposure bias in preference optimization, and strict online latency constraints. We propose LocalSUG, an LLM-based query suggestion framework for local-life services. LocalSUG mines city-preference-enhanced candidates from term co-occurrence and injects them into prompts as dynamic references rather than fusing them into model parameters. This allows the model to adapt to changing city preferences, such as merchant openings or closures, while reducing stale or locally invalid suggestions. We further introduce a beam-search-driven GRPO algorithm to align training with inference-time decoding and optimize relevance together with business-oriented rewards. Finally, quality-aware beam acceleration and vocabulary pruning reduce online latency while preserving generation quality. Offline evaluations and large-scale online A/B testing show that LocalSUG improves CTR by +0.35% and reduces the low/no-result rate by 3.98%, demonstrating its effectiveness in real-world deployment.
>
---
#### [replaced 090] Dual Mechanisms of Value Expression: Intrinsic vs. Prompted Values in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型中内在与提示值表达的机制，分析其异同，以解决价值对齐问题。通过价值向量和神经元分析，揭示两者共享与独特组件。**

- **链接: [https://arxiv.org/pdf/2509.24319](https://arxiv.org/pdf/2509.24319)**

> **作者:** Jongwook Han; Jongwon Lim; Injin Kong; Yohan Jo
>
> **备注:** Accepted at ICML 2026. Project page: this https URL
>
> **摘要:** Large language models can express values in two main ways: (1) intrinsic expression, reflecting the model's inherent values learned during training, and (2) prompted expression, elicited by explicit prompts. Given their widespread use in value alignment, it is paramount to clearly understand their underlying mechanisms, particularly whether they mostly overlap (as one might expect) or rely on distinct mechanisms. We analyze this largely understudied problem at the mechanistic level using two approaches: (1) value vectors, feature directions representing value mechanisms extracted from the residual stream, and (2) value neurons, MLP neurons that contribute to value vectors. We demonstrate that intrinsic and prompted value mechanisms partly share common components crucial for inducing value expression, generalizing across languages and reconstructing theoretical inter-value correlations in the model's internal representations. Yet, each mechanism also possesses unique components that fulfill distinct roles. In particular, the intrinsic mechanism activates in more diverse value-related scenarios and promotes response diversity, whereas the prompted mechanism strengthens instruction compliance, taking effect even in distant tasks like jailbreaking.
>
---
#### [replaced 091] Stop the Flip-Flop: Context-Preserving Verification for Fast Revocable Diffusion Decoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的文本生成任务，针对并行扩散解码中因验证引发的翻转振荡问题，提出COVER方法，通过缓存覆盖验证提升解码效率与质量。**

- **链接: [https://arxiv.org/pdf/2602.06161](https://arxiv.org/pdf/2602.06161)**

> **作者:** Yanzheng Xiang; Lan Wei; Yizhen Yao; Qinglin Zhu; Hanqi Yan; Chen Jin; Philip Alexander Teare; Dandan Zhang; Lin Gui; Amrutha Saseendran; Yulan He
>
> **摘要:** Parallel diffusion decoding can accelerate diffusion language model inference by unmasking multiple tokens per step, but aggressive parallelism often harms quality. Revocable decoding mitigates this by rechecking earlier tokens, yet we observe that existing verification schemes frequently trigger flip-flop oscillations, where tokens are remasked and later restored unchanged. This behaviour slows inference in two ways: remasking verified positions weakens the conditioning context for parallel drafting, and repeated remask cycles consume the revision budget with little net progress. We propose COVER (Cache Override Verification for Efficient Revision), which performs leave-one-out verification and stable drafting within a single forward pass. COVER constructs two attention views via KV cache override: selected seeds are masked for verification, while their cached key value states are injected for all other queries to preserve contextual information, with a closed form diagonal correction preventing self leakage at the seed positions. COVER further prioritises seeds using a stability aware score that balances uncertainty, downstream influence, and cache drift, and it adapts the number of verified seeds per step. Across benchmarks, COVER markedly reduces unnecessary revisions and yields faster decoding while preserving output quality.
>
---
#### [replaced 092] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决长文本推理中的注意力计算效率问题。提出Token Sparse Attention机制，通过动态选择关键token提升效率，同时保持高精度。**

- **链接: [https://arxiv.org/pdf/2602.03216](https://arxiv.org/pdf/2602.03216)**

> **作者:** Dongwon Jo; Beomseok Kang; Jiwon Song; Jae-Joon Kim
>
> **备注:** ICML 2026
>
> **摘要:** The quadratic complexity of attention remains the central bottleneck in long-context inference for large language models. Prior acceleration methods either sparsify the attention map with structured patterns or permanently evict tokens at specific layers, which can retain irrelevant tokens or rely on irreversible early decisions despite the layer-/head-wise dynamics of token importance. In this paper, we propose Token Sparse Attention, a lightweight and dynamic token-level sparsification mechanism that compresses per-head $Q$, $K$, $V$ to a reduced token set during attention and then decompresses the output back to the original sequence, enabling token information to be reconsidered in subsequent layers. Furthermore, Token Sparse Attention exposes a new design point at the intersection of token selection and sparse attention. Our approach is fully compatible with dense attention implementations, including Flash Attention, and can be seamlessly composed with existing sparse attention kernels. Experimental results show that Token Sparse Attention consistently improves accuracy-latency trade-off, achieving up to $\times$3.23 attention speedup at 128K context with less than 1% accuracy degradation. These results demonstrate that dynamic and interleaved token-level sparsification is a complementary and effective strategy for scalable long-context inference.
>
---
#### [replaced 093] Casual as an Anchor: Resolving Supervision Misalignment in Formality Transfer Dataset
- **分类: cs.CL**

- **简介: 该论文属于形式化转换任务，解决基准数据监督不一致问题。通过引入三层次标注框架3LF，提升模型生成正式语言的准确性。**

- **链接: [https://arxiv.org/pdf/2605.29365](https://arxiv.org/pdf/2605.29365)**

> **作者:** Hyojeong Yu; Hyukhun Koh; Minsung Kim; Kyomin Jung
>
> **备注:** HEAL@CHI 2026 Workshop Paper
>
> **摘要:** Formality transfer is commonly framed as a symmetric bidirectional task between informal and formal registers. We argue that this framing conceals a supervision design flaw in existing benchmarks such as GYAFC: binary human rewrites encode relative stylistic shifts rather than absolute human notions of formality. Consequently, models learn to generate pseudo-formal outputs that satisfy benchmark labels while failing to produce genuinely formal language. We quantify this misalignment by re-evaluating benchmark formal labels under a human-aligned definition of formality, revealing substantial discrepancies that propagate to consistent informal-to-formal failures across model families. To address this issue, we reconceptualize formality transfer as a graded dimension rather than a binary attribute. We introduce a three-level spectrum: informal, casual, and formal, where casual serves as an explicit intermediate state that clarifies supervision signals. Based on this framework, we introduce 3LF, a dataset providing parallel supervision across all three levels. Training on 3LF substantially reduces informal-to-formal failures and improves alignment with human perception. For example, GPT-4.1-nano improves from 0.06 to 0.88 F1 in the informal-to-formal direction despite 3LF being significantly smaller than GYAFC. We further demonstrate that these gains cannot be reproduced through in-context learning alone and provide qualitative analyses of ambiguity-driven errors and meaning distortions. Overall, our findings demonstrate how supervision design shapes stylistic alignment and highlight the importance of alignment-aware benchmark construction in controllable text generation.
>
---
#### [replaced 094] DySem: Uncovering Dynamic Semantic Components of Large Language Models for Calculating Semantic Textual Similarity
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的语义文本相似度计算任务，旨在解决传统方法依赖静态隐藏层表示的不足。提出DySem框架，通过动态语义维度提升相似度计算效果。**

- **链接: [https://arxiv.org/pdf/2605.29751](https://arxiv.org/pdf/2605.29751)**

> **作者:** Kaijie Zheng; Weiqin Wang; Yile Wang; Hui Huang
>
> **备注:** 18 pages, 23 figures, 5 tables
>
> **摘要:** Calculating semantic textual similarity is a foundational task in natural language processing. Current large language models (LLMs) based methods typically rely on extracting last-layer hidden states with fixed dimensions to compute similarity for every text pairs. We argue that this paradigm is suffer from two limitations: (i) The last hidden layer encodes more general knowledge rather than just semantic knowledge, making it suboptimal for semantic similarity computation; (ii) The hidden layer dimensions of LLMs are generally very large, which introduces some redundancy and noise for representing semantics. In this work, we propose DySem, a novel training-free framework that investigates more semantic-related internal components of LLMs via multilingual consensus, and shifts away from static representation spaces in favor of dynamic, sample-specific semantic dimensions by constructing text-dependent joint semantic set and computes similarity over this shared dimensional subset. Extensive experiments across various LLMs show that our method consistently outperforms recent baselines while maintaining lower dimensions for similarity calculation. The code is released at this https URL.
>
---
#### [replaced 095] EMCEE: Improving Multilingual Capability of LLMs via Bridging Knowledge and Reasoning with Extracted Synthetic Multilingual Context
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs在非英语语言上的性能下降问题。通过提取并融合语言特定知识，提升模型的多语言能力。**

- **链接: [https://arxiv.org/pdf/2503.05846](https://arxiv.org/pdf/2503.05846)**

> **作者:** Hamin Koo; Jaehyung Kim
>
> **备注:** ACL 2026 Main
>
> **摘要:** Large Language Models (LLMs) have achieved impressive progress across a wide range of tasks, yet their heavy reliance on English-centric training data leads to significant performance degradation in non-English languages. While existing multilingual prompting methods emphasize reformulating queries into English or enhancing reasoning capabilities, they often fail to incorporate the language- and culture-specific grounding that is essential for some queries. To address this limitation, we propose EMCEE (Extracting synthetic Multilingual Context and merging), a simple yet effective framework that enhances the multilingual capabilities of LLMs by explicitly extracting and utilizing query-relevant knowledge from the LLM itself. In particular, EMCEE first extracts synthetic context to uncover latent, language-specific knowledge encoded within the LLM, and then dynamically merges this contextual insight with reasoning-oriented outputs through a judgment-based selection mechanism. Extensive experiments on four multilingual benchmarks covering diverse languages and tasks demonstrate that EMCEE consistently outperforms prior approaches, achieving an average relative improvement of 16.4% overall and 31.7% in low-resource languages.
>
---
#### [replaced 096] ValueGround: Evaluating Culture-Conditioned Visual Value Grounding in MLLMs
- **分类: cs.CL**

- **简介: 该论文提出ValueGround，用于评估多模态大语言模型在视觉场景下的文化价值观判断能力。任务为跨模态文化价值对齐，解决视觉化响应选项下模型表现下降的问题。通过对比实验验证了模型在视觉选项下的性能下降。**

- **链接: [https://arxiv.org/pdf/2604.06484](https://arxiv.org/pdf/2604.06484)**

> **作者:** Zhipin Wang; Christoph Leiter; Christian Frey; Mohamed Hesham Ibrahim Abdalla; Josif Grabocka; Steffen Eger
>
> **备注:** Updated preprint
>
> **摘要:** Cultural values are expressed not only through language but also through visual scenes and everyday social practices. Yet existing evaluations of cultural values in language models are almost entirely text-only, leaving it unclear whether culture-conditioned judgments remain stable when response options are visualized. We introduce ValueGround, a benchmark for evaluating culture-conditioned visual value grounding in multimodal large language models (MLLMs). Built from World Values Survey questions, ValueGround uses minimally contrastive image pairs to represent opposing response options while controlling irrelevant variation. Given a country, a question, and an image pair, a model must choose the image that best matches the country's value tendency without access to the original response-option texts. Experiments across six MLLMs and 13 countries show that models perform substantially worse with visualized response options than with the original textual options, with average accuracy dropping from 72.8% to 62.6%. Our benchmark provides a controlled testbed for studying cross-modal transfer of culture-conditioned value judgments.
>
---
#### [replaced 097] Evidence for systematic semantic structure in individual phonemes
- **分类: cs.CL; q-bio.NC**

- **简介: 该论文属于语言学研究，旨在检验音素与意义的任意性假设。通过分析文本和跨语言实验，发现音素具有系统性语义结构，表明发音动作影响意义表达。**

- **链接: [https://arxiv.org/pdf/2603.17306](https://arxiv.org/pdf/2603.17306)**

> **作者:** Gexin Zhao
>
> **备注:** 31 pages, 4 figures
>
> **摘要:** A foundational assumption in linguistics holds that sound-meaning relations are largely arbitrary. Here we show that this assumption fails at the level of individual phonemes: each English phoneme carries a structured, multidimensional semantic profile that is recoverable from text, perceived across languages, and grounded in articulation. Three large language models independently detected consistent semantic structure across nine perceptual dimensions in 220 pairwise letter contrasts. Native English speakers (N = 93) confirmed these associations in a preregistered forced-choice task (85.3% agreement with model predictions), and listeners of five typologically diverse languages (N = 155) replicated the effect under audio presentation (73.2%-81.9% accuracy). Articulatory features predicted the structure with cross-validated R^2 of 0.56-0.98, indicating that the bodily act of producing a sound systematically shapes the meaning it conveys. These findings reframe phoneme-level iconicity as a pervasive, embodied property of the phonological system.
>
---
#### [replaced 098] Weight Decay Improves Language Model Plasticity
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究预训练模型的可塑性，解决如何提升模型在微调后的下游任务表现问题。通过调整权重衰减参数，增强模型适应能力。**

- **链接: [https://arxiv.org/pdf/2602.11137](https://arxiv.org/pdf/2602.11137)**

> **作者:** Tessa Han; Sebastian Bordt; Hanlin Zhang; Sham Kakade
>
> **摘要:** Large language models are typically trained in two broad phases: pretraining to produce a base model, followed by further training to improve downstream performance. However, hyperparameter optimization and scaling laws are studied primarily from the perspective of the base model's validation loss, overlooking a crucial model property: downstream adaptability. In this work, we study pretraining from the perspective of model plasticity, that is, the ability of the base model to successfully adapt to downstream tasks upon additional training. We focus on the role of weight decay, a key regularization parameter during pretraining, and show through systematic experiments that larger weight decay increases the plasticity of the pretrained model, resulting in greater performance gains downstream after fine-tuning. This effect can lead to counterintuitive trade-offs where base models that perform worse after pretraining can perform better after further training. Further investigation of weight decay's mechanistic effects on model behavior reveals that it encourages linearly separable representations, regularizes attention matrices, and reduces overfitting on the training data. Together, these findings highlight the importance of pretrained model plasticity, the limits of using cross-entropy loss as the sole metric for hyperparameter optimization, and the multifaceted role that a single optimization hyperparameter plays in shaping model behavior.
>
---
